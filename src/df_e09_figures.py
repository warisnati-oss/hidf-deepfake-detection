"""df_e09 robustness figure: test AUC under each degradation, one line per model.

Reads every experiments/df_e09_robustness/results_*.csv, so models are added as they finish.
Output: documents/figures/phase4/e09_robustness.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from degrade import CONDITIONS

PROJECT = Path(__file__).resolve().parents[1]
E09 = PROJECT / "experiments" / "df_e09_robustness"
FIG = PROJECT / "documents" / "figures" / "phase4"
COLORS = ["#eb6834", "#2a78d6", "#1baf7a", "#6250d6", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NAMES = {"classical_hgb": "classical (HGB)", "df_e04_effnet_b4": "M1 EfficientNet-B4",
         "df_e05_gend": "M2 GenD (DINOv3)", "df_e05b_dinov3_lora": "M3 DINOv3 + LoRA",
         "df_e07_crop_effnet": "M1 + face crop", "df_e07_crop_gend": "M2 + face crop", "df_e07_crop_dinomac": "M3 + face crop", "df_e06_noaug_effnet": "M1, no augmentation", "df_e06_noaug_gend": "M2, no augmentation", "df_e06_noaug_dinomac": "M3, no augmentation", "df_e08_res384_effnet": "M1, 384 px", "df_e08_res384_gend": "M2, 384 px", "df_e08_res_dinomac": "M3, 320-448 px"}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    res = pd.concat([pd.read_csv(f) for f in sorted(E09.glob("results_*.csv"))])
    conds = [c for c in CONDITIONS if c in set(res.condition)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), sharex=True)
    for color, (model, g) in zip(COLORS, res.groupby("model", sort=False)):
        g = g.set_index("condition").reindex(conds)
        for ax, metric in zip(axes, ["test_auc", "test_accuracy"]):
            ax.plot(range(len(conds)), g[metric], marker="o", markersize=7, linewidth=2, color=color,
                    label=NAMES.get(model, model))
    for ax, title in zip(axes, ["Test AUC", "Test accuracy (clean-val threshold)"]):
        ax.axhline(0.5, color=INK, linestyle="--", linewidth=1)
        ax.set_xticks(range(len(conds)), conds, rotation=35, ha="right")
        ax.set_ylim(0.35, 1.02)
        ax.set_title(title, fontweight="bold")
        ax.grid(color=GRID)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("df_e09 robustness: models trained on clean images, tested on degraded ones", fontweight="bold")
    fig.savefig(FIG / "e09_robustness.png", dpi=150, bbox_inches="tight")
    print(res.pivot_table(index="condition", columns="model", values="test_auc", sort=False).round(3))


if __name__ == "__main__":
    main()
