"""Step 7 (preprocessing P3): pick the face segmenter with the free swap masks as reference.

HiDF has no face-segmentation labels, but for a fake with a known real base,
|fake - base| marks exactly the swapped pixels (face_regions.swap_mask). On 2,000
train fakes we run every segmenter on the *fake alone* (as at test time) and compare:
    coverage = |M ∩ S| / |S|      share of the swapped region inside the mask (recall)
    iou      = |M ∩ S| / |M ∪ S|  tightness
for several dilations of M and several swap-mask thresholds (sensitivity check).

Selection rule: highest mean IoU among (method, dilation) with mean coverage >= 0.95
at the default threshold.

Output: data/p3_benchmark.csv (per image), data/p3_benchmark_summary.csv,
        documents/figures/phase2/p3_*.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from face_regions import MediaPipeOval, SegFormerParsing, dilate, swap_mask

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
FIG = PROJECT / "documents" / "figures" / "phase2"
N_SAMPLES = 2000
THRESHOLDS = [6, 10, 15]
DEFAULT_THRESH = 10
DILATIONS = [0, 4, 8, 12, 16, 24, 32]  # px at 512
MIN_COVERAGE = 0.95
SEED = 0

COLORS = {"mediapipe": "#2a78d6", "segformer": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(rel: str) -> np.ndarray:
    return np.asarray(Image.open(DATA / rel).convert("RGB"))


def sample_fakes(df: pd.DataFrame) -> pd.DataFrame:
    real = df[df.label == "real"].set_index("base_id").cache_path
    fake = df[(df.label == "fake") & (df.split == "train") & (df.fake_check == "ok")
              & df.base_id.isin(real.index)]
    fake = fake.sample(N_SAMPLES, random_state=SEED).copy()
    fake["base_cache_path"] = real[fake.base_id].to_numpy()
    return fake


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "index.csv", low_memory=False)
    fakes = sample_fakes(df)
    mediapipe, segformer = MediaPipeOval(), SegFormerParsing()
    if (DATA / "p3_benchmark.csv").exists():
        res = pd.read_csv(DATA / "p3_benchmark.csv")
    else:
        res = run_benchmark(fakes, mediapipe, segformer)
        res.to_csv(DATA / "p3_benchmark.csv", index=False)
    summarize(res, fakes, mediapipe, segformer)


def run_benchmark(fakes: pd.DataFrame, mediapipe, segformer) -> pd.DataFrame:
    rows = []
    for k, r in enumerate(fakes.itertuples()):
        fake, base = load(r.cache_path), load(r.base_cache_path)
        swaps = {t: swap_mask(fake, base, thresh=t) for t in THRESHOLDS}
        preds = {"mediapipe": mediapipe(fake), "segformer": segformer(fake)}
        for method, m in preds.items():
            for px in DILATIONS:
                md = dilate(m, px) if m is not None else None
                for t, s in swaps.items():
                    if md is None:
                        cov = iou = np.nan
                    else:
                        inter = (md & s).sum()
                        cov, iou = inter / max(s.sum(), 1), inter / max((md | s).sum(), 1)
                    rows.append({"file_path": r.file_path, "method": method, "dilation": px,
                                 "thresh": t, "coverage": cov, "iou": iou,
                                 "detected": md is not None, "swap_frac": s.mean()})
        if k % 250 == 0:
            print(f"{k}/{len(fakes)}", flush=True)
    return pd.DataFrame(rows)


def summarize(res: pd.DataFrame, fakes: pd.DataFrame, mediapipe, segformer):
    summary = (res.groupby(["method", "thresh", "dilation"])
               .agg(coverage=("coverage", "mean"), coverage_p10=("coverage", lambda s: s.quantile(0.1)),
                    iou=("iou", "mean"), detect_rate=("detected", "mean"))
               .reset_index())
    summary.to_csv(DATA / "p3_benchmark_summary.csv", index=False)

    main_t = summary[summary.thresh == DEFAULT_THRESH]
    ok = main_t[main_t.coverage >= MIN_COVERAGE]
    best = ok.loc[ok.iou.idxmax()] if len(ok) else main_t.loc[main_t.coverage.idxmax()]
    print(main_t.round(3).to_string(index=False))
    print("\nselected:", best[["method", "dilation", "coverage", "iou"]].to_dict())
    print("\nsensitivity (best dilation per method, each threshold):")
    for t in THRESHOLDS:
        s = summary[(summary.thresh == t) & (summary.coverage >= MIN_COVERAGE)]
        if len(s):
            b = s.loc[s.iou.idxmax()]
            print(f"  thresh {t}: {b.method} dil {b.dilation} cov {b.coverage:.3f} iou {b.iou:.3f}")

    plot_tradeoff(main_t, best)
    plot_examples(fakes.head(4), mediapipe, segformer, best)


def plot_tradeoff(summary: pd.DataFrame, best: pd.Series):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for method, g in summary.groupby("method"):
        ax.plot(g.coverage, g.iou, marker="o", markersize=8, linewidth=2, color=COLORS[method], label=method)
        for c, i, d in zip(g.coverage, g.iou, g.dilation):
            if d > 16:  # points crowd together near coverage 1.0
                continue
            ax.annotate(f"{d}px", (c, i), xytext=(5, 4), textcoords="offset points", fontsize=8, color=INK2)
    ax.axvline(MIN_COVERAGE, color=INK, linestyle="--", linewidth=1)
    ax.annotate("coverage ≥ 95%", (MIN_COVERAGE, ax.get_ylim()[0]), xytext=(4, 6),
                textcoords="offset points", fontsize=9, color=INK)
    ax.scatter([best.coverage], [best.iou], s=220, facecolors="none", edgecolors=INK, linewidths=1.5, zorder=5)
    ax.set_xlabel("coverage of swapped region (mean)")
    ax.set_ylabel("IoU with swap mask (mean)")
    ax.set_title("P3 segmenter choice (labels = mask dilation; 24/32 px unlabeled)", fontweight="bold")
    ax.grid(color=GRID)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(FIG / "p3_tradeoff.png", dpi=150, bbox_inches="tight")


def plot_examples(rows: pd.DataFrame, mediapipe, segformer, best: pd.Series):
    titles = ["fake", "swap mask (reference)", "MediaPipe oval", "SegFormer parsing"]
    fig, axes = plt.subplots(len(rows), 4, figsize=(10, 2.6 * len(rows)))
    for i, r in enumerate(rows.itertuples()):
        fake, base = load(r.cache_path), load(r.base_cache_path)
        s = swap_mask(fake, base, thresh=DEFAULT_THRESH)
        for j, img in enumerate([fake, s, mediapipe(fake), segformer(fake)]):
            ax = axes[i, j]
            if j == 0:
                ax.imshow(img)
            else:
                m = img if img is not None else np.zeros_like(s)
                if j > 1:
                    m = dilate(m, int(best.dilation))
                overlay = fake.copy()
                overlay[m] = (0.5 * overlay[m] + 0.5 * np.array([235, 104, 52])).astype(np.uint8)
                ax.imshow(overlay)
                if j > 1:
                    inter = (m & s).sum()
                    ax.set_xlabel(f"cov {inter / s.sum():.2f} · IoU {inter / (m | s).sum():.2f}", fontsize=9)
            ax.set_xticks([]), ax.set_yticks([])
    for ax, t in zip(axes[0], titles):
        ax.set_title(t, fontsize=11)
    fig.suptitle(f"P3 examples (both masks dilated {int(best.dilation)} px; selected: {best.method})",
                 fontweight="bold")
    fig.savefig(FIG / "p3_examples.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
