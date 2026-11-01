"""Step 11: Phase 3 comparison figures (zero-shot vs classical) for CLASSICAL_AND_ZERO_SHOT.md.

Output: documents/figures/phase3/{e02_zero_shot_bars,e02_roc,e02_scores,phase3_summary}.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

PROJECT = Path(__file__).resolve().parents[1]
E02 = PROJECT / "experiments" / "df_e02_zero_shot"
E03 = PROJECT / "experiments" / "df_e03_classical"
FIG = PROJECT / "documents" / "figures" / "phase3"

REAL, FAKE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NAMES = {"clip_l14": "CLIP ViT-L/14", "siglip2_so400m": "SigLIP 2 SO400M", "pe_core_bigG": "PE-Core bigG",
         "qwen2.5_vl_7b": "Qwen2.5-VL-7B"}
MODEL_COLORS = {"clip_l14": "#2a78d6", "siglip2_so400m": "#1baf7a", "pe_core_bigG": "#6250d6",
                "qwen2.5_vl_7b": "#eda100", "classical": "#eb6834"}


def style(ax):
    ax.grid(color=GRID)
    ax.spines[["top", "right"]].set_visible(False)


def zero_shot_bars(res: pd.DataFrame):
    clip_like = res[res.model != "qwen2.5_vl_7b"]
    models = list(dict.fromkeys(clip_like.model))
    fig, ax = plt.subplots(figsize=(8, 4))
    x, w = np.arange(len(models)), 0.38
    for k, (view, color) in enumerate([("full", "#8fb8ea"), ("crop", REAL)]):
        vals = [clip_like[(clip_like.model == m) & (clip_like.view == view)].test_auc.iloc[0] for m in models]
        bars = ax.bar(x + (k - 0.5) * (w + 0.02), vals, w, color=color,
                      label="full image" if view == "full" else "P3 face crop")
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=9, color=INK2)
    ax.axhline(0.5, color=INK, linestyle="--", linewidth=1)
    ax.set_xticks(x, [NAMES[m] for m in models])
    ax.set_ylim(0.4, 1.1)
    ax.set_ylabel("test AUC")
    ax.set_title("df_e02 zero-shot (no training): full image vs face crop", fontweight="bold")
    style(ax)
    ax.legend(frameon=False, loc="upper center", ncol=2)
    fig.savefig(FIG / "e02_zero_shot_bars.png", dpi=150, bbox_inches="tight")


def summary_roc(res: pd.DataFrame):
    scores = pd.read_csv(E02 / "scores_test.csv")
    y = (scores.label == "fake").astype(int)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    for m in dict.fromkeys(res[res.model != "qwen2.5_vl_7b"].model):
        best_view = res[res.model == m].sort_values("val_auc").iloc[-1].view
        s = scores[f"{m}_{best_view}"]
        fpr, tpr, _ = roc_curve(y, s)
        auc = res[(res.model == m) & (res.view == best_view)].test_auc.iloc[0]
        ax.plot(fpr, tpr, color=MODEL_COLORS[m], linewidth=2, label=f"{NAMES[m]} ({best_view}) {auc:.3f}")
    vlm = E02 / "vlm_test_scores.csv"
    if vlm.exists():
        v = pd.read_csv(vlm)
        fpr, tpr, _ = roc_curve((v.label == "fake").astype(int), v.score)
        auc = res[res.model == "qwen2.5_vl_7b"].test_auc.iloc[0]
        ax.plot(fpr, tpr, color=MODEL_COLORS["qwen2.5_vl_7b"], linewidth=2, label=f"Qwen2.5-VL-7B (1K subset) {auc:.3f}")
    c = pd.read_csv(E03 / "scores_test.csv")
    fpr, tpr, _ = roc_curve((c.label == "fake").astype(int), c.score)
    auc = pd.read_csv(E03 / "results.csv").iloc[0].test_auc
    ax.plot(fpr, tpr, color=MODEL_COLORS["classical"], linewidth=2.5, label=f"classical HGB {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color=INK2, linewidth=1)
    ax.set_xlabel("false positive rate"), ax.set_ylabel("true positive rate")
    ax.set_title("Phase 3 test ROC: zero-shot vs classical", fontweight="bold")
    style(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.savefig(FIG / "phase3_roc.png", dpi=150, bbox_inches="tight")


def score_distributions(res: pd.DataFrame):
    scores = pd.read_csv(E02 / "scores_test.csv")
    models = list(dict.fromkeys(res[res.model != "qwen2.5_vl_7b"].model))
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.4))
    for ax, m in zip(np.atleast_1d(axes), models):
        s = scores[f"{m}_crop"]
        bins = np.linspace(s.quantile(0.005), s.quantile(0.995), 60)
        for label, color in [("real", REAL), ("fake", FAKE)]:
            ax.hist(s[scores.label == label], bins=bins, color=color, alpha=0.6, label=label)
        thr = res[(res.model == m) & (res.view == "crop")].threshold.iloc[0]
        ax.axvline(thr, color=INK, linestyle="--", linewidth=1)
        ax.set_title(f"{NAMES[m]} (crop)", fontweight="bold", fontsize=11)
        ax.set_xlabel("cos(img, fake) − cos(img, real)")
        style(ax)
    np.atleast_1d(axes)[0].set_ylabel("test images")
    np.atleast_1d(axes)[0].legend(frameon=False)
    fig.savefig(FIG / "e02_scores.png", dpi=150, bbox_inches="tight")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(E02 / "results.csv")
    if (E02 / "vlm_results.csv").exists():
        res = pd.concat([res, pd.read_csv(E02 / "vlm_results.csv")], ignore_index=True)
    zero_shot_bars(res)
    summary_roc(res)
    score_distributions(res)
    print(res[["model", "view", "prompts", "val_auc", "test_auc", "test_accuracy", "test_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
