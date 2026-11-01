"""df_e02_zero_shot: can vision-language models spot HiDF fakes with no training at all?

Models (open_clip): CLIP ViT-L/14 (OpenAI), SigLIP 2 SO400M/16-384, PE-Core-bigG/14-448
(top of the OpenCLIP zero-shot leaderboard). Qwen2.5-VL-7B is in df_e02_zero_shot_vlm.py.
For each image we compare its embedding with the mean text embedding of a "real" and a "fake"
prompt group:  score = cos(img, fake) - cos(img, real).
Inputs: the full 512 px image and the P3 face crop.

Prompt set and decision threshold are chosen on val (no labels are used for anything else);
test is scored once. Image embeddings are cached per (model, input, split) and reused by the
Phase 4 linear probe.

Output: experiments/df_e02_zero_shot/{emb/*.npy, val_prompt_search.csv, results.csv, scores_test.csv}
"""
import argparse
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
OUT = PROJECT / "experiments" / "df_e02_zero_shot"
EMB = OUT / "emb"

MODELS = {
    "clip_l14": ("ViT-L-14", "openai", 64),
    "siglip2_so400m": ("ViT-SO400M-16-SigLIP2-384", "webli", 24),
    "pe_core_bigG": ("PE-Core-bigG-14-448", "meta", 8),
}

PROMPTS = {
    "basic": {
        "real": ["a photo of a real face", "a real photograph of a person"],
        "fake": ["a photo of a fake face", "a deepfake image of a person"],
    },
    "forensic": {
        "real": ["an authentic, unedited photo of a face", "a genuine photograph of a person's face",
                 "a natural photo of a human face"],
        "fake": ["a face-swapped image", "a digitally manipulated photo of a face",
                 "an AI-generated deepfake face"],
    },
    "detail": {
        "real": ["a face with natural skin texture and realistic lighting",
                 "a sharp photo of a real person with consistent skin and hair"],
        "fake": ["a face with unnaturally smooth skin and blending artifacts",
                 "a pasted face with mismatched skin tone and blurry edges"],
    },
}


class Images(Dataset):
    def __init__(self, rows: pd.DataFrame, boxes: pd.DataFrame | None, preprocess):
        self.paths = rows.cache_path.to_list()
        self.boxes = None if boxes is None else boxes.loc[rows.cache_path, ["x0", "y0", "x1", "y1"]].to_numpy()
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(DATA / self.paths[i]).convert("RGB")
        if self.boxes is not None:
            im = im.crop(tuple(int(v) for v in self.boxes[i]))
        return self.preprocess(im)


def embed_images(key: str, model, preprocess, rows: pd.DataFrame, view: str, split: str, bs: int) -> np.ndarray:
    path = EMB / f"{key}_{view}_{split}.npy"
    if path.exists():
        return np.load(path)
    boxes = pd.read_csv(DATA / "face_boxes.csv").set_index("cache_path") if view == "crop" else None
    loader = DataLoader(Images(rows, boxes, preprocess), batch_size=bs, num_workers=4)
    out = []
    half = next(model.parameters()).dtype == torch.float16
    with torch.no_grad(), torch.autocast("mps", dtype=torch.float16, enabled=not half):
        for k, x in enumerate(loader):
            x = x.to("mps", dtype=torch.float16 if half else torch.float32)
            out.append(model.encode_image(x).float().cpu().numpy())
            if k % 50 == 0:
                print(f"  {key} {view} {split}: {k * bs}/{len(rows)}", flush=True)
    emb = np.concatenate(out)
    np.save(path, emb)
    return emb


def embed_prompts(model, tokenizer) -> dict:
    out = {}
    with torch.no_grad():
        for name, groups in PROMPTS.items():
            out[name] = {}
            for cls, texts in groups.items():
                t = model.encode_text(tokenizer(texts).to("mps")).float()
                t = torch.nn.functional.normalize(t, dim=-1).mean(0)
                out[name][cls] = torch.nn.functional.normalize(t, dim=0).cpu().numpy()
    return out


def zero_shot_score(img: np.ndarray, text: dict) -> np.ndarray:
    img = img / np.linalg.norm(img, axis=1, keepdims=True)
    return img @ text["fake"] - img @ text["real"]


def metrics(y: np.ndarray, score: np.ndarray, thresh: float) -> dict:
    pred = (score >= thresh).astype(int)
    return {"auc": roc_auc_score(y, score), "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred)}


def best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold that maximises balanced accuracy on val."""
    cands = np.quantile(score, np.linspace(0.01, 0.99, 197))
    bal = [((score >= t)[y == 1].mean() + (score < t)[y == 0].mean()) / 2 for t in cands]
    return float(cands[int(np.argmax(bal))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    args = ap.parse_args()
    EMB.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "index.csv", low_memory=False)
    rows = {s: df[df.split == s] for s in ["val", "test"]}
    keep = {s: ~r.near_dup.to_numpy() for s, r in rows.items()}
    y = {s: (r.label == "fake").to_numpy()[keep[s]].astype(int) for s, r in rows.items()}

    search, results, test_scores = [], [], {}
    for key in args.models:
        arch, pretrained, bs = MODELS[key]
        print(f"== {key}", flush=True)
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device="mps")
        model.eval()
        if key == "pe_core_bigG":
            model.half()  # 2B params: fp16 weights halve memory (fp32 copy pushed the Mac into heavy swap)
        tokenizer = open_clip.get_tokenizer(arch)
        text = embed_prompts(model, tokenizer)
        for view in ["full", "crop"]:
            # embeddings are cached for the whole split; near-duplicates of train are dropped afterwards
            emb = {s: embed_images(key, model, preprocess, rows[s], view, s, bs)[keep[s]] for s in rows}
            val_auc = {p: roc_auc_score(y["val"], zero_shot_score(emb["val"], t)) for p, t in text.items()}
            for p, a in val_auc.items():
                search.append({"model": key, "view": view, "prompts": p, "val_auc": a})
            best_p = max(val_auc, key=val_auc.get)
            s_val = zero_shot_score(emb["val"], text[best_p])
            s_test = zero_shot_score(emb["test"], text[best_p])
            thr = best_threshold(y["val"], s_val)
            res = {"model": key, "view": view, "prompts": best_p, "threshold": thr,
                   "val_auc": val_auc[best_p], **{f"test_{k}": v for k, v in metrics(y["test"], s_test, thr).items()}}
            results.append(res)
            test_scores[f"{key}_{view}"] = s_test
            print(res, flush=True)
        del model
        torch.mps.empty_cache()

    pd.DataFrame(search).round(4).to_csv(OUT / "val_prompt_search.csv", index=False)
    res = pd.DataFrame(results).round(4)
    res.to_csv(OUT / "results.csv", index=False)
    t = rows["test"][keep["test"]]
    pd.DataFrame({"cache_path": t.cache_path.to_numpy(), "label": t.label.to_numpy(),
                  **test_scores}).to_csv(OUT / "scores_test.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
