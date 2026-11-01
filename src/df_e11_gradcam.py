"""df_e11: Grad-CAM of the three DL baselines checked against the real swap region.

HiDF fakes keep the base photo outside the face, so |fake - base| is a free ground-truth mask of
the swapped area. For each fake we ask how much of the Grad-CAM heat falls inside that mask.
- M1 EfficientNet-B4: classic Grad-CAM on the last conv feature map (7 x 7 at 224 px).
- M2 / M3 (DINOv3 ViT-L/16): Grad-CAM on the patch tokens entering the last block (14 x 14).
  The last block's own patch outputs can't be used: M2's head reads only the CLS token there.

Metrics per fake (CAM upsampled to 512 px and normalised to sum 1):
    energy_swap  share of heat inside the swap mask (uniform heat would give mask_area)
    energy_face  share of heat inside the P3 face mask
    point_hit    the hottest pixel lies inside the swap mask (dilated 15 px), "pointing game"

Usage: python df_e11_gradcam.py [--n 300]
Output: experiments/df_e11_gradcam/{gradcam.csv, summary.csv}, documents/figures/phase5/e11_gradcam.png
"""
import argparse
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import train
from augment import eval_transform
from evaluation import DATA, eval_rows, load_index
from step9_face_regions import face_mask

RUNS = {"M1": "df_e04_effnet_b4", "M2": "df_e05_gend", "M3": "df_e05b_dinov3_lora"}
OUT = train.PROJECT / "experiments" / "df_e11_gradcam"
FIG = train.PROJECT / "documents" / "figures" / "phase5"


def load_model(exp: str):
    cfg = json.loads((train.PROJECT / "experiments" / exp / "config.json").read_text())
    model = train.build(argparse.Namespace(**cfg)).to(train.DEVICE)
    model.load_state_dict(torch.load(train.PROJECT / "experiments" / exp / "best.pt", map_location=train.DEVICE))
    return model.eval(), cfg


def gradcam(model, x: torch.Tensor) -> np.ndarray:
    """Grad-CAM for the 'fake' logit, (B, h, w) in [0, 1]."""
    if isinstance(model, train.EffNet):
        a = model.net.forward_features(x)
        a.retain_grad()
        logit = model.net.forward_head(a).squeeze(1)
    else:
        out = model.enc(pixel_values=x, output_hidden_states=True)
        h = out.hidden_states[-2]
        h.retain_grad()
        if isinstance(model, train.GenD):
            logit = model.head(F.normalize(out.last_hidden_state[:, 0], dim=-1)).squeeze(1)
        else:
            logit = model.head(model.mac(out.last_hidden_state)).squeeze(1)
        a = h
    model.zero_grad()
    logit.sum().backward()
    g, a = a.grad, a.detach()
    if a.dim() == 4:  # conv: (B, C, h, w)
        cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1))
    else:  # ViT tokens: (B, 1 + registers + patches, C)
        n = int(round((a.shape[1] - 1) ** 0.5)) ** 2  # patch count; the rest are CLS + registers
        side = int(n ** 0.5)
        pa, pg = a[:, -n:], g[:, -n:]
        cam = F.relu((pg.mean(1, keepdim=True) * pa).sum(2)).reshape(-1, side, side)
    cam = cam / cam.flatten(1).max(1).values.clamp(min=1e-12)[:, None, None]
    return cam.cpu().numpy()


def swap_mask(fake: np.ndarray, base: np.ndarray) -> np.ndarray:
    d = cv2.GaussianBlur(np.abs(fake - base).mean(2), (0, 0), 2)
    m = (d > 0.05).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))).astype(bool)


def load(path: str) -> np.ndarray:
    return np.asarray(Image.open(DATA / path).convert("RGB"), dtype=np.float32) / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    df = load_index()
    test = eval_rows(df, "test")
    real_path = df[df.label == "real"].set_index("base_id").cache_path.to_dict()
    fakes = test[(test.label == "fake") & test.base_id.map(real_path).notna()]
    sample = fakes.sample(a.n, random_state=0)
    examples = pd.read_csv(train.PROJECT / "experiments" / "df_e10_evaluation" / "examples.csv")
    rows = pd.concat([sample.assign(set="random"), test[test.cache_path.isin(examples.cache_path)].assign(set="example")])
    regions = np.load(DATA / "face_regions.npz", allow_pickle=True)
    poly = dict(zip(regions["cache_path"], regions["polygons"]))

    cams = {}
    for name, exp in RUNS.items():
        model, cfg = load_model(exp)
        tf = eval_transform(cfg["size"], model.norm)
        out = []
        for i in range(0, len(rows), 8):
            x = torch.stack([tf(Image.open(DATA / p).convert("RGB")) for p in rows.cache_path.iloc[i:i + 8]]).to(train.DEVICE).requires_grad_(True)
            out.append(gradcam(model, x))
        cams[name] = np.concatenate(out)
        print(f"{name} done", flush=True)
        del model
        torch.mps.empty_cache()

    recs, keep = [], {}
    for k, (_, r) in enumerate(rows.iterrows()):
        img = load(r.cache_path)
        face = face_mask(poly[r.cache_path])
        swap = swap_mask(img, load(real_path[r.base_id])) if r.label == "fake" else None
        for name in RUNS:
            cam = cv2.resize(cams[name][k], (512, 512), interpolation=cv2.INTER_LINEAR)
            if r.set == "example":
                keep[(r.cache_path, name)] = cam
            if swap is None:
                continue
            w = cam / max(cam.sum(), 1e-12)
            yx = np.unravel_index(cam.argmax(), cam.shape)
            near = cv2.dilate(swap.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))).astype(bool)
            recs.append({"set": r.set, "cache_path": r.cache_path, "model": name, "mask_area": swap.mean(),
                         "energy_swap": w[swap].sum(), "energy_face": w[face].sum(), "face_area": face.mean(),
                         "point_hit": bool(near[yx])})
    res = pd.DataFrame(recs)
    res.to_csv(OUT / "gradcam.csv", index=False)
    rnd = res[res.set == "random"]
    summary = rnd.groupby("model")[["mask_area", "energy_swap", "face_area", "energy_face", "point_hit"]].mean()
    summary["focus_ratio"] = summary.energy_swap / summary.mask_area
    summary.round(3).to_csv(OUT / "summary.csv")
    print(summary.round(3).to_string())
    figure(examples, keep, real_path)


def figure(examples: pd.DataFrame, keep: dict, real_path: dict):
    fig, axes = plt.subplots(len(examples), 5, figsize=(12, 2.5 * len(examples)))
    for r, (_, e) in enumerate(examples.iterrows()):
        img = load(e.cache_path)
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"{e.kind} · {e.label}", fontsize=8, color="#1a7f37" if e.kind == "success" else "#c62828")
        if e.label == "fake":
            axes[r, 1].imshow(swap_mask(img, load(real_path[e.base_id])), cmap="gray")
            axes[r, 1].set_title("true swap region", fontsize=8)
        else:
            axes[r, 1].text(0.5, 0.5, "real photo:\nno swap", ha="center", va="center", fontsize=8)
        for c, name in enumerate(RUNS, start=2):
            axes[r, c].imshow(img)
            axes[r, c].imshow(keep[(e.cache_path, name)], cmap="jet", alpha=0.45, vmin=0, vmax=1)
            ok = e[f"{name}_ok"]
            axes[r, c].set_title(f"{name} Grad-CAM ({'right' if ok else 'wrong'})", fontsize=8)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Where the deep models look (Grad-CAM, red = most important)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "e11_gradcam.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
