"""df_e12 T4 (leakage A3): near-duplicate audit with DINOv3 embeddings.

The Phase 3 audit compared 16 x 16 thumbnails, which misses flipped, cropped or edited copies of
a photo. Here every image is embedded with the frozen, pre-trained DINOv3 ViT-L/16 (CLS token,
full image, 224 px); val/test images also as a horizontal flip. For each val/test image we take
its most similar train image (max over original and flip).

Calibration (no hand-set threshold):
- "same scene, different face": a fake vs its own base photo (same split by construction) —
  pairs that are NOT duplicates in the sense that matters but share everything except the face.
  Their similarity distribution sets the cut-off (its 50th percentile, like the Phase 3 audit).
- recall check: the 51 near-duplicates the thumbnail audit already found must score above it.
Candidates above the cut-off that the old audit missed are shown in a contact sheet for a look,
and test metrics are recomputed with them excluded.

Usage: python df_e12_neardup.py embed (GPU)  then  python df_e12_neardup.py analyse (CPU)
Output: experiments/df_e12_leakage/{dinov3_cls.npy, neardup.csv, neardup_summary.csv},
        documents/figures/phase5/e12_neardup.png
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import df_e10_evaluation as E
from evaluation import DATA, eval_rows, load_index

OUT = E.EXP / "df_e12_leakage"
EMB = OUT / "dinov3_cls.npy"
EMB_FLIP = OUT / "dinov3_cls_flip.npy"


class Ds(torch.utils.data.Dataset):
    """Module level so DataLoader workers (spawn on macOS) can pickle it."""
    def __init__(self, paths, flip, tf):
        self.paths, self.flip, self.tf = paths, flip, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        x = self.tf(Image.open(DATA / self.paths[i]).convert("RGB"))
        return x.flip(-1) if self.flip else x


def embed():
    from torch.utils.data import DataLoader
    from transformers import AutoModel

    import train
    from augment import IMAGENET, eval_transform

    df = load_index()
    df = df[df.split.notna()].reset_index(drop=True)

    enc = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m").to(train.DEVICE).eval()

    def run(paths, flip):
        out = []
        with torch.no_grad(), torch.autocast(train.DEVICE, dtype=torch.bfloat16):
            for k, x in enumerate(DataLoader(Ds(paths, flip, eval_transform(224, IMAGENET)), batch_size=64, num_workers=6)):
                z = enc(pixel_values=x.to(train.DEVICE)).last_hidden_state[:, 0].float()
                out.append(torch.nn.functional.normalize(z, dim=-1).cpu().numpy().astype(np.float16))
                if k % 100 == 0:
                    print(f"  {'flip' if flip else 'orig'} batch {k}", flush=True)
        return np.concatenate(out)

    np.save(EMB, run(df.cache_path.to_list(), False))
    ev = df[df.split != "train"]
    np.save(EMB_FLIP, run(ev.cache_path.to_list(), True))
    print("embedded", len(df), "+", len(ev), "flipped", flush=True)


def analyse():
    df = load_index()
    df = df[df.split.notna()].reset_index(drop=True)
    Z = torch.from_numpy(np.load(EMB).astype(np.float32))
    ev_idx = np.flatnonzero((df.split != "train").to_numpy())
    Zf = torch.from_numpy(np.load(EMB_FLIP).astype(np.float32))
    tr_idx = np.flatnonzero((df.split == "train").to_numpy())
    Zt = Z[tr_idx]

    best_sim, best_j, flipped = np.zeros(len(ev_idx)), np.zeros(len(ev_idx), int), np.zeros(len(ev_idx), bool)
    for i in range(0, len(ev_idx), 1024):
        so = Z[ev_idx[i:i + 1024]] @ Zt.T
        sf = Zf[i:i + 1024] @ Zt.T
        mo, jo = so.max(1)
        mf, jf = sf.max(1)
        use_f = (mf > mo).numpy()
        best_sim[i:i + 1024] = np.where(use_f, mf, mo)
        best_j[i:i + 1024] = np.where(use_f, jf, jo)
        flipped[i:i + 1024] = use_f

    # calibration: fake vs its own base (same scene, face replaced)
    pos = pd.Series(np.arange(len(df)), index=df.cache_path)
    real_by_id = df[df.label == "real"].drop_duplicates("base_id").set_index("base_id").cache_path
    fk = df[(df.label == "fake") & df.base_id.isin(real_by_id.index)].sample(5000, random_state=0)
    a = Z[pos[fk.cache_path].to_numpy()]
    b = Z[pos[real_by_id[fk.base_id]].to_numpy()]
    scene = (a * b).sum(1).numpy()
    cut = float(np.median(scene))

    res = df.iloc[ev_idx][["cache_path", "label", "split", "near_dup", "base_id"]].copy()
    res["max_sim"], res["flipped"] = best_sim, flipped
    res["train_match"] = df.cache_path.to_numpy()[tr_idx[best_j]]
    res["candidate"] = res.max_sim >= cut
    res.sort_values("max_sim", ascending=False).to_csv(OUT / "neardup.csv", index=False)

    old = res.near_dup
    summ = {"cut_off (median fake-vs-own-base sim)": cut,
            "fake-vs-own-base sim p5 / p95": f"{np.percentile(scene, 5):.3f} / {np.percentile(scene, 95):.3f}",
            "val/test max-sim to train median / p99": f"{np.median(best_sim):.3f} / {np.percentile(best_sim, 99):.3f}",
            "old near-dups found again": f"{int((old & res.candidate).sum())} / {int(old.sum())}",
            "candidates val": int((res.candidate & (res.split == 'val')).sum()),
            "candidates test": int((res.candidate & (res.split == 'test')).sum()),
            "new candidates (not flagged before) test": int((res.candidate & ~old & (res.split == 'test')).sum()),
            "of which matched via flip": int((res.candidate & ~old & res.flipped).sum())}
    for k, v in summ.items():
        print(f"{k}: {v}")
    pd.Series(summ).to_csv(OUT / "neardup_summary.csv")

    # does excluding the new candidates change test results?
    test = eval_rows(load_index(), "test").reset_index(drop=True)
    new_test = set(res.cache_path[res.candidate & ~old & (res.split == "test")])
    keep = ~test.cache_path.isin(new_test).to_numpy()
    y = (test.label == "fake").to_numpy().astype(int)
    models = E.load_models(test)
    for name in ["Classical", "M1 EfficientNet-B4", "M2 GenD (DINOv3)", "M3 DINOv3 + LoRA"]:
        s, p = models[name]
        on = (p[~keep] == y[~keep]).mean() if (~keep).any() else float("nan")
        print(f"{name:20s} acc all {(p == y).mean():.4f} | without candidates {(p[keep] == y[keep]).mean():.4f} "
              f"| on candidates {on:.4f} (n={int((~keep).sum())})")

    # contact sheet: top new candidates
    from PIL import Image
    top = res[res.candidate & ~old].sort_values("max_sim", ascending=False).head(12)
    if len(top):
        fig, axes = plt.subplots(len(top), 2, figsize=(4.4, 2.2 * len(top)))
        axes = np.atleast_2d(axes)
        for r, (_, t) in enumerate(top.iterrows()):
            for c, (p, title) in enumerate([(t.cache_path, f"{t.split} {t.label} sim {t.max_sim:.3f}{' (flip)' if t.flipped else ''}"),
                                            (t.train_match, "closest train image")]):
                axes[r, c].imshow(Image.open(DATA / p))
                axes[r, c].set_title(title, fontsize=7)
                axes[r, c].axis("off")
        fig.tight_layout()
        fig.savefig(E.FIG / "e12_neardup.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    {"embed": embed, "analyse": analyse}[sys.argv[1]]()
