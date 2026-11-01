"""Step 8: before/after figures for P2 (augmentation) and P4 (high-pass / FFT), plus the
P1-P4 pipeline diagram, for PREPROCESSING.md and slide 3.

Output: documents/figures/phase2/{p2_augmentations,p2_random,p4_filters,p4_spectrum,pipeline}.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
from torchvision.transforms import v2

from augment import AUGMENTATIONS, IMAGENET, train_transform
from features import fft_spectrum, residuals, to_gray

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
FIG = PROJECT / "documents" / "figures" / "phase2"
E01 = PROJECT / "experiments" / "df_e01_leakage_probe"

REAL, FAKE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def off(ax):
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def p2_figures(img: Image.Image):
    torch.manual_seed(0)
    x = v2.ToImage()(img)
    fig, axes = plt.subplots(1, len(AUGMENTATIONS) + 1, figsize=(2.3 * (len(AUGMENTATIONS) + 1), 2.8))
    axes[0].imshow(img), axes[0].set_title("original", fontsize=10)
    for ax, (name, t) in zip(axes[1:], AUGMENTATIONS.items()):
        ax.imshow(t(x).permute(1, 2, 0).numpy())
        ax.set_title(name, fontsize=10)
    for ax in axes:
        off(ax)
    fig.suptitle("P2: each augmentation applied alone (training only)", fontweight="bold")
    fig.savefig(FIG / "p2_augmentations.png", dpi=150, bbox_inches="tight")

    # Zoomed crops make blur / JPEG / noise visible at slide size
    crop = (176, 200, 336, 360)
    fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.8))
    for ax, name in zip(axes, ["original", "Gaussian blur", "JPEG q60-95", "Gaussian noise"]):
        y = x if name == "original" else AUGMENTATIONS[name](x)
        ax.imshow(Image.fromarray(y.permute(1, 2, 0).numpy()).crop(crop).resize((320, 320), Image.NEAREST))
        ax.set_title(f"{name} (zoom)", fontsize=10)
        off(ax)
    fig.savefig(FIG / "p2_zoom.png", dpi=150, bbox_inches="tight")

    t = train_transform(224, IMAGENET)
    mean, std = torch.tensor(IMAGENET[0])[:, None, None], torch.tensor(IMAGENET[1])[:, None, None]
    fig, axes = plt.subplots(1, 6, figsize=(13, 2.6))
    for ax in axes:
        y = (t(img) * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(y)
        off(ax)
    fig.suptitle("P2: six random draws of the full training pipeline (224 px model input)", fontweight="bold")
    fig.savefig(FIG / "p2_random.png", dpi=150, bbox_inches="tight")


def p4_filters(real: np.ndarray, fake: np.ndarray):
    rows = [("real", real), ("fake", fake)]
    titles = ["image", "Laplacian residual", "SRM 5×5 residual", "log |FFT|²"]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
    for i, (label, img) in enumerate(rows):
        gray = to_gray(img)
        res = residuals(gray)
        f = np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean()))) ** 2 + 1e-8)
        panels = [img, np.abs(res["lap"]), np.abs(res["srm3"]), f]
        for j, p in enumerate(panels):
            ax = axes[i, j]
            if j == 0:
                ax.imshow(p)
            elif j < 3:
                ax.imshow(p, cmap="gray", vmin=0, vmax=np.percentile(p, 99))
            else:
                ax.imshow(p, cmap="magma")
            off(ax)
        axes[i, 0].set_ylabel(label.upper(), fontsize=12, fontweight="bold", color=REAL if label == "real" else FAKE)
    for ax, t in zip(axes[0], titles):
        ax.set_title(t, fontsize=11)
    fig.suptitle("P4: high-pass filters remove face content and keep fine traces", fontweight="bold")
    fig.savefig(FIG / "p4_filters.png", dpi=150, bbox_inches="tight")


def p4_spectrum(df: pd.DataFrame):
    f = pd.read_parquet(E01 / "features_cache.parquet")
    rows = df.loc[f.index]
    cols = [c for c in f.columns if c.startswith("fft_")]
    freq = (np.arange(len(cols)) + 0.5) / len(cols) * 0.5
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for prefix, style in [("c", "-"), ("f", "--")]:
        for label, color in [("real", REAL), ("fake", FAKE)]:
            m = (rows.label == label) & (rows.prefix == prefix)
            axes[0].plot(freq, f.loc[m, cols].mean(), style, color=color, linewidth=2,
                         label=f"{label} · {prefix}*")
        d = (f.loc[(rows.label == "fake") & (rows.prefix == prefix), cols].mean()
             - f.loc[(rows.label == "real") & (rows.prefix == prefix), cols].mean())
        axes[1].plot(freq, d, style, color=INK2, linewidth=2, label=f"{prefix}* ids")
    axes[1].axhline(0, color=INK, linewidth=1)
    axes[0].set_title("Mean 1-D power spectrum (P1 cache)", fontweight="bold")
    axes[1].set_title("Fake − real", fontweight="bold")
    axes[0].set_ylabel("log power (rel. to DC)")
    axes[1].set_ylabel("Δ log power")
    for ax in axes:
        ax.set_xlabel("spatial frequency (cycles / px)")
        ax.grid(color=GRID)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.savefig(FIG / "p4_spectrum.png", dpi=150, bbox_inches="tight")


def e01_results():
    res = pd.read_csv(E01 / "results.csv")
    probes = [("meta", "file size"), ("global", "whole image\nFFT + noise"),
              ("background", "background only\n(leakage test)"), ("face", "face only\n(swap signal)")]
    sources = [("raw", "original files", "#c3c2b7"), ("cache_v1", "512 px JPEG q95", "#8fb8ea"),
               ("cache", "final P1 (q75 matched)", REAL)]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x, w = np.arange(len(probes)), 0.26
    for k, (src, label, color) in enumerate(sources):
        vals = [res[(res.source == src) & (res.features == p)].auc_all.iloc[0] for p, _ in probes]
        bars = ax.bar(x + (k - 1) * (w + 0.01), vals, w, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
    ax.axhline(0.5, color=INK, linewidth=1, linestyle="--")
    ax.annotate("chance", (x[-1] + 0.45, 0.5), xytext=(0, 3), textcoords="offset points", fontsize=9, ha="right")
    ax.set_xticks(x, [p[1] for p in probes])
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("val AUC (logistic regression)")
    ax.set_title("df_e01: format leakage before and after P1", fontweight="bold")
    ax.grid(axis="y", color=GRID)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.savefig(FIG / "e01_leakage.png", dpi=150, bbox_inches="tight")


def pipeline():
    steps = [
        ("HiDF image\n1024 px\nPNG / JPEG", "#f2f1ed"),
        ("P1 normalise\nJPEG q75 @1024\n512 px · JPEG q95", "#dbe8f8"),
        ("P3 face region\nMediaPipe oval +8 px\nmask · seam · crop", "#dbe8f8"),
        ("P2 augment\n(train only)\ncrop · flip · colour\nblur · JPEG · noise", "#fbe3d9"),
        ("deep models\nEfficientNet-B4\nGenD-CLIP", "#fbe3d9"),
        ("P4 filters\nLaplacian · SRM\nFFT spectrum", "#d6f0e5"),
        ("classical model\nSVM / RF / LogReg", "#d6f0e5"),
    ]
    pos = [(0.5, 2), (2.5, 2), (4.5, 2), (6.5, 3.1), (8.5, 3.1), (6.5, 0.9), (8.5, 0.9)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6)]
    fig, ax = plt.subplots(figsize=(13, 4.4))
    for (text, color), (x, y) in zip(steps, pos):
        ax.add_patch(FancyBboxPatch((x - 0.85, y - 0.6), 1.7, 1.2, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=color, edgecolor=INK2, linewidth=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=9.5, color=INK)
    for a, b in edges:
        (x0, y0), (x1, y1) = pos[a], pos[b]
        ax.add_patch(FancyArrowPatch((x0 + 0.85, y0), (x1 - 0.85, y1), arrowstyle="-|>", mutation_scale=14,
                                     color=INK2, linewidth=1.3, connectionstyle="arc3,rad=0"))
    ax.text(7.5, 4.05, "deep-learning branch", ha="center", fontsize=10, color=INK2, style="italic")
    ax.text(7.5, -0.05, "classical branch", ha="center", fontsize=10, color=INK2, style="italic")
    ax.set_xlim(-0.5, 9.5), ax.set_ylim(-0.3, 4.3)
    ax.axis("off")
    fig.savefig(FIG / "pipeline.png", dpi=150, bbox_inches="tight")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "index.csv", low_memory=False)
    real_path = df[df.label == "real"].set_index("base_id").cache_path
    fake_row = df[(df.label == "fake") & (df.split == "train") & (df.fake_check == "ok")
                  & df.base_id.isin(real_path.index)].sample(1, random_state=11).iloc[0]
    fake = np.asarray(Image.open(DATA / fake_row.cache_path).convert("RGB"))
    real = np.asarray(Image.open(DATA / real_path[fake_row.base_id]).convert("RGB"))
    p2_figures(Image.fromarray(fake))
    p4_filters(real, fake)
    p4_spectrum(df)
    e01_results()
    pipeline()
    print(f"figures written to {FIG}")


if __name__ == "__main__":
    main()
