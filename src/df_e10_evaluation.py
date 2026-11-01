"""df_e10: Phase 5 evaluation from saved test scores (CPU only, no model is re-run).

1. Comparison table: every model's test metrics with 95 % bootstrap CIs. The bootstrap resamples
   identity groups (group_id), not images, because images of one person are not independent.
   Paired CI of the accuracy / AUC difference to the M1 baseline on the same resamples.
2. Confusion matrices of the headline models.
3. Fairness: AUC, accuracy, false-positive rate (real called fake) and false-negative rate (fake
   missed) per base race / gender / age, plus same-race vs cross-race swaps (fakes only).

Output: experiments/df_e10_evaluation/{comparison.csv, fairness.csv, cross_race.csv}
        documents/figures/phase5/{e10_confusion.png, e10_fairness.png, e10_comparison.png}
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from evaluation import eval_rows, load_index

PROJECT = Path(__file__).resolve().parents[1]
EXP = PROJECT / "experiments"
OUT = EXP / "df_e10_evaluation"
FIG = PROJECT / "documents" / "figures" / "phase5"
N_BOOT, SEED = 1000, 0

DL = {"df_e04_effnet_b4": "M1 EfficientNet-B4", "df_e05_gend": "M2 GenD (DINOv3)",
      "df_e05b_dinov3_lora": "M3 DINOv3 + LoRA",
      "df_e07_crop_effnet": "M1, face crop", "df_e07_crop_gend": "M2, face crop", "df_e07_crop_dinomac": "M3, face crop",
      "df_e06_noaug_effnet": "M1, no aug", "df_e06_noaug_gend": "M2, no aug", "df_e06_noaug_dinomac": "M3, no aug",
      "df_e08_res384_effnet": "M1, 384 px", "df_e08_res384_gend": "M2, 384 px", "df_e08_res_dinomac": "M3, 320-448 px"}
HEADLINE = ["Zero-shot CLIP (crop)", "Classical", "M1 EfficientNet-B4", "M2 GenD (DINOv3)", "M3 DINOv3 + LoRA"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"Classical": "#eb6834", "M1 EfficientNet-B4": "#2a78d6", "M2 GenD (DINOv3)": "#1baf7a",
          "M3 DINOv3 + LoRA": "#6250d6"}


def load_models(test: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """name -> (score, 0/1 prediction), aligned to `test` rows."""
    def aligned(path, col):
        s = pd.read_csv(path).set_index("cache_path")[col]
        return s.loc[test.cache_path].to_numpy()

    zs_thr = pd.read_csv(EXP / "df_e02_zero_shot/results.csv").query("model == 'clip_l14' and view == 'crop'").threshold.iloc[0]
    s = aligned(EXP / "df_e02_zero_shot/scores_test.csv", "clip_l14_crop")
    models = {"Zero-shot CLIP (crop)": (s, (s >= zs_thr).astype(int))}
    s = aligned(EXP / "df_e03_classical/scores_test.csv", "score")
    models["Classical"] = (s, aligned(EXP / "df_e03_classical/scores_test.csv", "pred_fake").astype(int))
    for exp, name in DL.items():
        thr = json.load(open(EXP / exp / "results.json"))["threshold"]
        s = aligned(EXP / exp / "scores_test.csv", "score")
        models[name] = (s, (s >= thr).astype(int))
    return models


def metrics(y, s, p) -> dict:
    tp, fp = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum())
    fn, tn = int(((p == 0) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {"auc": roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan, "accuracy": (tp + tn) / len(y),
            "precision": prec, "recall": rec, "f1": 2 * prec * rec / max(1e-12, prec + rec),
            "fpr": fp / max(1, fp + tn), "fnr": fn / max(1, fn + tp), "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def boot_index(groups: np.ndarray, rng, n: int = N_BOOT) -> list[np.ndarray]:
    """n resamples of row indices, drawing whole identity groups with replacement."""
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    return [np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))]) for _ in range(n)]


def comparison(test, y, models, boots) -> pd.DataFrame:
    keys = ["auc", "accuracy", "precision", "recall", "f1"]
    ref_s, ref_p = models["M1 EfficientNet-B4"]
    rows = []
    for name, (s, p) in models.items():
        row = {"model": name, **metrics(y, s, p)}
        bs = [metrics(y[i], s[i], p[i]) for i in boots]
        for k in keys:
            row[f"{k}_lo"], row[f"{k}_hi"] = np.percentile([b[k] for b in bs], [2.5, 97.5])
        d_acc = [b["accuracy"] - ((ref_p[i] == y[i]).mean()) for b, i in zip(bs, boots)]
        d_auc = [b["auc"] - roc_auc_score(y[i], ref_s[i]) for b, i in zip(bs, boots)]
        row["d_acc_vs_m1_lo"], row["d_acc_vs_m1_hi"] = np.percentile(d_acc, [2.5, 97.5])
        row["d_auc_vs_m1_lo"], row["d_auc_vs_m1_hi"] = np.percentile(d_auc, [2.5, 97.5])
        rows.append(row)
        print(f"{name:24s} acc {row['accuracy']:.4f} [{row['accuracy_lo']:.4f}, {row['accuracy_hi']:.4f}]  "
              f"auc {row['auc']:.5f} [{row['auc_lo']:.5f}, {row['auc_hi']:.5f}]", flush=True)
    return pd.DataFrame(rows)


def fairness(test, y, models, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, cross = [], []
    for name in HEADLINE[1:]:
        s, p = models[name]
        for attr in ["base_race", "base_gender", "base_age"]:
            for g in sorted(test[attr].dropna().unique()):
                m = (test[attr] == g).to_numpy()
                idx = np.flatnonzero(m)
                row = {"model": name, "attribute": attr.replace("base_", ""), "group": g,
                       "n_real": int((y[m] == 0).sum()), "n_fake": int((y[m] == 1).sum()), **metrics(y[m], s[m], p[m])}
                bs = [metrics(y[idx[i]], s[idx[i]], p[idx[i]]) for i in boot_index(test.group_id.to_numpy()[idx], rng, 300)]
                for k in ["accuracy", "fpr", "fnr"]:
                    row[f"{k}_lo"], row[f"{k}_hi"] = np.percentile([b[k] for b in bs], [2.5, 97.5])
                rows.append(row)
        fake = test.label.eq("fake").to_numpy()
        for kind, m in [("same race", fake & (test.base_race == test.donor_race).to_numpy()),
                        ("cross race", fake & (test.base_race != test.donor_race).to_numpy()),
                        ("same gender", fake & (test.base_gender == test.donor_gender).to_numpy()),
                        ("cross gender", fake & (test.base_gender != test.donor_gender).to_numpy())]:
            cross.append({"model": name, "swap": kind, "n": int(m.sum()), "recall": p[m].mean()})
    return pd.DataFrame(rows), pd.DataFrame(cross)


def fig_confusion(models, y):
    fig, axes = plt.subplots(1, len(HEADLINE), figsize=(3.1 * len(HEADLINE), 3.3))
    for ax, name in zip(axes, HEADLINE):
        m = metrics(y, *models[name])
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cm):
            frac = v / cm[i].sum()
            ax.text(j, i, f"{v:,}\n{frac:.1%}", ha="center", va="center", fontsize=10,
                    color="white" if frac > 0.5 else INK)
        ax.set_xticks([0, 1], ["real", "fake"])
        ax.set_yticks([0, 1], ["real", "fake"])
        ax.set_xlabel("predicted")
        ax.set_title(f"{name}\nacc {m['accuracy']:.2%}", fontsize=10)
    axes[0].set_ylabel("true")
    fig.suptitle("Confusion matrices, clean test set (5,366 real / 4,234 fake)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "e10_confusion.png", dpi=150, bbox_inches="tight")


def fig_comparison(comp: pd.DataFrame):
    comp = comp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    yy = np.arange(len(comp))
    ax.errorbar(comp.accuracy * 100, yy, xerr=[(comp.accuracy - comp.accuracy_lo) * 100, (comp.accuracy_hi - comp.accuracy) * 100],
                fmt="o", color="#2a78d6", ecolor=INK2, capsize=3)
    ax.set_yticks(yy, comp.model)
    ax.set_xlabel("test accuracy % (95 % CI, identity-group bootstrap)")
    ax.set_xlim(97.5, 100)
    ax.grid(color=GRID)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "e10_comparison.png", dpi=150, bbox_inches="tight")


def fig_fairness(fair: pd.DataFrame):
    attrs = ["race", "gender", "age"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharey="row", gridspec_kw={"width_ratios": [5, 2, 3]})
    for c, attr in enumerate(attrs):
        sub = fair[fair.attribute == attr]
        groups = list(dict.fromkeys(sub.group))
        x, w = np.arange(len(groups)), 0.2
        for r, (k, label) in enumerate([("fpr", "false-positive rate % (real called fake)"),
                                        ("fnr", "false-negative rate % (fake missed)")]):
            ax = axes[r, c]
            for j, name in enumerate(HEADLINE[1:]):
                d = sub[sub.model == name].set_index("group").loc[groups]
                ax.bar(x + (j - 1.5) * w, d[k] * 100, w, color=COLORS[name], label=name,
                       yerr=[(d[k] - d[f"{k}_lo"]) * 100, (d[f"{k}_hi"] - d[k]) * 100], error_kw={"lw": 0.8, "ecolor": INK2})
            ax.set_xticks(x, [f"{g}\n(n={int(n)})" for g, n in zip(groups, sub[sub.model == HEADLINE[1]].set_index('group').loc[groups].eval('n_real + n_fake'))]
                          if r == 1 else [""] * len(groups), fontsize=8)
            ax.grid(color=GRID, axis="y")
            ax.spines[["top", "right"]].set_visible(False)
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
            if r == 0:
                ax.set_title(attr, fontsize=11)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Error rates by group of the base (target) person, clean test, 95 % CI", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "e10_fairness.png", dpi=150, bbox_inches="tight")


def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    test = eval_rows(load_index(), "test").reset_index(drop=True)
    y = (test.label == "fake").to_numpy().astype(int)
    models = load_models(test)
    rng = np.random.default_rng(SEED)
    boots = boot_index(test.group_id.to_numpy(), rng)
    print(f"{len(test)} test images, {test.group_id.nunique()} identity groups, {N_BOOT} resamples", flush=True)

    comp = comparison(test, y, models, boots)
    comp.to_csv(OUT / "comparison.csv", index=False)
    fair, cross = fairness(test, y, models, rng)
    fair.to_csv(OUT / "fairness.csv", index=False)
    cross.to_csv(OUT / "cross_race.csv", index=False)
    print(cross.to_string(index=False))

    fig_confusion(models, y)
    fig_comparison(comp[comp.model != "Zero-shot CLIP (crop)"])
    fig_fairness(fair)
    print("figures ok")


if __name__ == "__main__":
    main()
