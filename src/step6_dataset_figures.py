"""Step 6: dataset figures for DATA_CLEANING.md and slide 1 (dataset overview).

Output: documents/figures/phase1/*.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
FIG = PROJECT / "documents" / "figures" / "phase1"

REAL, FAKE = "#2a78d6", "#eb6834"  # categorical slots 1-2
JPEG, PNG = "#2a78d6", "#1baf7a"   # categorical slots 1, 3
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SPLITS = ["train", "val", "test"]

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "legend.frameon": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def bar_labels(ax, bars, fmt="{:,.0f}"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=9, color=INK2, xytext=(0, 2),
                    textcoords="offset points")


def class_by_split(df):
    counts = pd.crosstab(df.split, df.label).reindex(SPLITS)
    fig, ax = plt.subplots(figsize=(7, 4))
    x, w = np.arange(len(SPLITS)), 0.38
    for k, (label, color) in enumerate([("real", REAL), ("fake", FAKE)]):
        bars = ax.bar(x + (k - 0.5) * (w + 0.02), counts[label], w, color=color, label=label)
        bar_labels(ax, bars)
    ax.set_xticks(x, [f"{s}\n({counts.loc[s].sum():,} imgs, {counts.loc[s].sum() / len(df):.0%})"
                      for s in SPLITS])
    ax.set_ylabel("images")
    ax.set_title("Images per split (identity-safe)")
    ax.legend()
    fig.savefig(FIG / "class_by_split.png")


def format_by_source(df):
    raw = pd.read_csv(DATA / "raw_index.csv", low_memory=False)
    raw = raw[raw.format.notna()]
    groups = [(l, p) for l in ["real", "fake"] for p in ["c", "f"]]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.arange(len(groups))
    jpeg = [((raw.label == l) & (raw.prefix == p) & (raw.format == "JPEG")).sum() for l, p in groups]
    png = [((raw.label == l) & (raw.prefix == p) & (raw.format == "PNG")).sum() for l, p in groups]
    ax.barh(y, jpeg, color=JPEG, label="JPEG", height=0.6)
    ax.barh(y, png, left=jpeg, color=PNG, label="PNG", height=0.6, edgecolor="white", linewidth=2)
    ax.set_yticks(y, [f"{l} · {p}* ids" for l, p in groups])
    ax.invert_yaxis()
    ax.set_xlabel("images (original files)")
    ax.set_title("Original file format: real f* are PNG, fake f* are JPEG")
    ax.legend(loc="lower right")
    fig.savefig(FIG / "format_leakage.png")


def demographics(df):
    cols = [("base_race", "Race"), ("base_gender", "Gender"), ("base_age", "Age")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [5, 2, 3]})
    for ax, (col, title) in zip(axes, cols):
        share = pd.crosstab(df[col], df.label, normalize="columns")[["real", "fake"]]
        share = share.sort_values("real", ascending=False)
        x, w = np.arange(len(share)), 0.38
        for k, (label, color) in enumerate([("real", REAL), ("fake", FAKE)]):
            bars = ax.bar(x + (k - 0.5) * (w + 0.02), share[label] * 100, w, color=color, label=label)
            bar_labels(ax, bars, "{:.0f}%")
        ax.set_xticks(x, share.index)
        ax.set_title(title)
        ax.set_ylim(0, share.values.max() * 115)
    axes[0].set_ylabel("% of class (fakes: base person)")
    axes[0].legend()
    fig.savefig(FIG / "demographics.png")


def sample_grid(df, n=6):
    test = df[df.split == "test"]
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.4))
    for row, (label, color) in enumerate([("real", REAL), ("fake", FAKE)]):
        picks = test[test.label == label].sample(n, random_state=3)
        for ax, (_, r) in zip(axes[row], picks.iterrows()):
            ax.imshow(Image.open(DATA / r.cache_path).resize((256, 256)))
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(True), s.set_color(color), s.set_linewidth(3)
        axes[row, 0].set_ylabel(label.upper(), color=INK, fontsize=13, fontweight="bold")
    fig.suptitle("Sample test images (512 px cache)", fontweight="bold")
    fig.savefig(FIG / "samples.png")


def triplets(df, n=4):
    real = df[df.label == "real"].set_index("base_id").cache_path
    fake = df[(df.label == "fake") & (df.fake_check == "ok") & df.base_id.isin(real.index)
              & df.donor_id.isin(real.index)].sample(n, random_state=7)
    titles = ["base (real)", "donor (real)", "fake = base + donor face", "|fake − base|"]
    fig, axes = plt.subplots(n, 4, figsize=(10, 2.6 * n))
    for row, (_, r) in enumerate(fake.iterrows()):
        base = np.asarray(Image.open(DATA / real[r.base_id]).resize((256, 256)), dtype=np.float32)
        swap = np.asarray(Image.open(DATA / r.cache_path).resize((256, 256)), dtype=np.float32)
        donor = Image.open(DATA / real[r.donor_id]).resize((256, 256))
        diff = np.abs(swap - base).mean(2)
        for ax, img in zip(axes[row], [base / 255, donor, swap / 255, diff]):
            ax.imshow(img, cmap="magma" if img is diff else None, vmin=0, vmax=60 if img is diff else None)
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    for ax, t in zip(axes[0], titles):
        ax.set_title(t, fontsize=11)
    fig.suptitle("How HiDF fakes are built (difference map = free swap mask)", fontweight="bold")
    fig.savefig(FIG / "triplets.png")


def cleaning_funnel():
    log = pd.read_csv(DATA / "cleaning_log.csv")
    rules = log[log.rule.str.startswith("R")]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    y = np.arange(len(rules))
    bars = ax.barh(y, rules.rows, color=INK2, height=0.55)
    for b, v in zip(bars, rules.rows):
        ax.annotate(f"{v:,}", (b.get_width(), b.get_y() + b.get_height() / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, [f"{r} · {d}" for r, d in zip(rules.rule, rules.description)])
    ax.invert_yaxis()
    ax.set_xlabel("images")
    total, kept = log.rows.iloc[0], log[log.rule == "clean"].rows.sum()
    ax.set_title(f"Cleaning rules: {total:,} files → {kept:,} kept")
    fig.savefig(FIG / "cleaning_rules.png")


def fake_check_hist():
    chk = pd.read_csv(DATA / "fake_checks.csv").dropna(subset=["mad"])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(chk.mad, bins=np.logspace(-1, np.log10(130), 60), color=FAKE)
    ax.set_xscale("log"), ax.set_yscale("log")
    for x, t in [(1, "< 1: no visible swap"), (20, "> 20: wrong base id")]:
        ax.axvline(x, color=INK, linewidth=1, linestyle="--")
        ax.annotate(t, (x, ax.get_ylim()[1]), xytext=(4, -12), textcoords="offset points",
                    fontsize=9, color=INK)
    ax.set_xlabel("mean |fake − named base| (pixel value, 128 px)")
    ax.set_ylabel("fakes")
    ax.set_title("Fake vs. its named base image")
    fig.savefig(FIG / "fake_vs_base.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "index.csv", low_memory=False)
    class_by_split(df)
    format_by_source(df)
    demographics(df)
    sample_grid(df)
    triplets(df)
    cleaning_funnel()
    fake_check_hist()
    print(f"figures written to {FIG}")


if __name__ == "__main__":
    main()
