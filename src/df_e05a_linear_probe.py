"""df_e05a_linear_probe: which frozen backbone's features separate real from fake best?

Backbones (frozen, P3 face crop input): CLIP ViT-L/14, SigLIP 2 SO400M/16-384, PE-Core bigG/14-448
(open_clip image embeddings, val/test reused from df_e02) and DINOv3 ViT-L/16 (CLS token
concatenated with the mean patch token, 2048-d).
Classifier: StandardScaler + LogisticRegression, C in {0.01, 0.1, 1, 10} chosen on val AUC,
trained on a stratified 15K train subset (a linear probe saturates well before 50K).
The winner (among ViT-L-size backbones, which are trainable on this Mac) becomes GenD's backbone.

Also df_e05c_rigid (training-free, Phase 3 deferral): RIGID on DINOv3 — real images keep a more
stable embedding under small Gaussian noise. score = 1 - cos(CLS(x), CLS(x + noise));
noise sigma in {0.02, 0.05, 0.1} (normalised pixel units) and threshold chosen on val.

Output: experiments/df_e05a_linear_probe/{emb/*.npy, search.csv, results.csv, scores_test.csv},
        experiments/df_e05c_rigid/{search.csv, results.csv}
"""
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

import df_e02_zero_shot as zs
from evaluation import DATA, best_threshold, eval_rows, load_index, report

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "experiments" / "df_e05a_linear_probe"
OUT_RIGID = PROJECT / "experiments" / "df_e05c_rigid"
EMB = OUT / "emb"
N_TRAIN = 15000
SEED = 0
DINO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
L_SIZE = {"clip_l14", "dinov3_l16"}  # trainable on the M6; SigLIP 2 SO400M and PE bigG are larger
SIGMAS = [0.02, 0.05, 0.1]


class Crops(Dataset):
    def __init__(self, rows: pd.DataFrame, tf):
        boxes = pd.read_csv(DATA / "face_boxes.csv").set_index("cache_path")
        self.paths = rows.cache_path.to_list()
        self.boxes = boxes.loc[rows.cache_path, ["x0", "y0", "x1", "y1"]].to_numpy()
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(DATA / self.paths[i]).convert("RGB").crop(tuple(int(v) for v in self.boxes[i]))
        return self.tf(im)


def dino_embed(model, rows: pd.DataFrame, name: str, sigma: float = 0.0) -> np.ndarray:
    path = EMB / f"dinov3_l16_crop_{name}{f'_noise{sigma}' if sigma else ''}.npy"
    if path.exists():
        return np.load(path)
    tf = v2.Compose([v2.ToImage(), v2.Resize((224, 224), antialias=True), v2.ToDtype(torch.float32, scale=True),
                     v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    gen = torch.Generator(device="mps").manual_seed(SEED)
    out = []
    with torch.no_grad(), torch.autocast("mps", dtype=torch.float16):
        for k, x in enumerate(DataLoader(Crops(rows, tf), batch_size=64, num_workers=4)):
            x = x.to("mps")
            if sigma:
                x = x + sigma * torch.randn(x.shape, device="mps", generator=gen)
            h = model(pixel_values=x).last_hidden_state.float()
            out.append(torch.cat([h[:, 0], h[:, 5:].mean(1)], 1).cpu().numpy())  # CLS + mean patch
            if k % 50 == 0:
                print(f"  dinov3 {name} sigma={sigma}: {k * 64}/{len(rows)}", flush=True)
    emb = np.concatenate(out)
    np.save(path, emb)
    return emb


def probe(X: dict, y: dict) -> tuple[dict, list]:
    best, search = None, []
    for c in [0.01, 0.1, 1, 10]:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)).fit(X["train"], y["train"])
        auc = roc_auc_score(y["val"], clf.decision_function(X["val"]))
        search.append({"C": c, "val_auc": auc})
        if best is None or auc > best[0]:
            best = (auc, c, clf)
    return {"val_auc": best[0], "C": best[1], "clf": best[2]}, search


def main():
    EMB.mkdir(parents=True, exist_ok=True)
    OUT_RIGID.mkdir(parents=True, exist_ok=True)
    df = load_index()
    train = df[df.split == "train"].groupby(["label", "prefix"]).sample(
        frac=N_TRAIN / (df.split == "train").sum(), random_state=SEED)
    rows = {"train": train, "val": eval_rows(df, "val"), "test": eval_rows(df, "test")}
    y = {s: (r.label == "fake").to_numpy().astype(int) for s, r in rows.items()}
    feats = {}

    # DINOv3 (+ RIGID noisy passes)
    from transformers import AutoModel
    dino = AutoModel.from_pretrained(DINO).to("mps").eval()
    feats["dinov3_l16"] = {s: dino_embed(dino, r, s) for s, r in rows.items()}
    rigid, rigid_search = {}, []
    for sigma in SIGMAS:
        for s in ["val", "test"]:
            noisy = dino_embed(dino, rows[s], s, sigma)
            clean = feats["dinov3_l16"][s]
            a, b = clean[:, :1024], noisy[:, :1024]  # CLS token
            rigid[(sigma, s)] = 1 - (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        rigid_search.append({"sigma": sigma, "val_auc": roc_auc_score(y["val"], rigid[(sigma, "val")])})
    del dino
    torch.mps.empty_cache()
    rs = pd.DataFrame(rigid_search)
    rs.round(4).to_csv(OUT_RIGID / "search.csv", index=False)
    sig = float(rs.loc[rs.val_auc.idxmax(), "sigma"])
    thr = best_threshold(y["val"], rigid[(sig, "val")])
    r = {"model": "rigid_dinov3", "sigma": sig, "val_auc": rs.val_auc.max(), "threshold": thr,
         **report(rows["test"], rigid[(sig, "test")], thr)}
    pd.DataFrame([r]).round(4).to_csv(OUT_RIGID / "results.csv", index=False)
    print("RIGID:", {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)

    # open_clip backbones: val/test from df_e02 cache, train subset embedded here
    for key in ["clip_l14", "siglip2_so400m", "pe_core_bigG"]:
        arch, pretrained, bs = zs.MODELS[key]
        tr_path = zs.EMB / f"{key}_crop_train15k.npy"
        if not tr_path.exists():
            model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device="mps")
            model.eval()
            if key == "pe_core_bigG":
                model.half()
            zs.embed_images(key, model, preprocess, train, "crop", "train15k", bs)
            del model
            torch.mps.empty_cache()
        full = {s: np.load(zs.EMB / f"{key}_crop_{s}.npy") for s in ["val", "test"]}
        keep = {s: ~df[df.split == s].near_dup.to_numpy() for s in ["val", "test"]}
        feats[key] = {"train": np.load(tr_path), **{s: full[s][keep[s]] for s in ["val", "test"]}}

    results, search, scores = [], [], {}
    for key, X in feats.items():
        best, srch = probe(X, y)
        search += [{"model": key, **s} for s in srch]
        s_val = best["clf"].decision_function(X["val"])
        s_test = best["clf"].decision_function(X["test"])
        thr = best_threshold(y["val"], s_val)
        results.append({"model": key, "dim": X["train"].shape[1], "l_size": key in L_SIZE, "C": best["C"],
                        "val_auc": best["val_auc"], "threshold": thr, **report(rows["test"], s_test, thr)})
        scores[key] = s_test
        print(results[-1], flush=True)

    pd.DataFrame(search).round(4).to_csv(OUT / "search.csv", index=False)
    res = pd.DataFrame(results).sort_values("val_auc", ascending=False)
    res.round(4).to_csv(OUT / "results.csv", index=False)
    pd.DataFrame({"cache_path": rows["test"].cache_path.to_numpy(), "label": rows["test"].label.to_numpy(),
                  **scores}).to_csv(OUT / "scores_test.csv", index=False)
    winner = res[res.l_size].iloc[0].model
    print(res.drop(columns=["threshold"]).round(4).to_string(index=False))
    print(f"\nGenD backbone (best ViT-L-size on val): {winner}")


if __name__ == "__main__":
    main()
