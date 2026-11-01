"""df_e10: 3 success + 3 failure examples, classical vs deep learning (rubric criterion 6).

Selection is rule-based on saved test scores (no hand-picking):
- Success: images every DL baseline (M1, M2, M3) gets right and the classical model gets wrong.
  The 2 fakes with the lowest classical score + the 1 real with the highest classical score
  (the classical model's most confident mistakes).
- Failure: images every model gets wrong (classical, M1, M2, M3). There are exactly 3.

Each row: the test image, the real photo it was made from (fakes only), and |fake - base| as a
free swap mask that shows where the face was replaced.

Output: experiments/df_e10_evaluation/examples.csv, documents/figures/phase5/e10_examples.png
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import df_e10_evaluation as E
from evaluation import DATA, eval_rows, load_index

MODELS = ["Classical", "M1 EfficientNet-B4", "M2 GenD (DINOv3)", "M3 DINOv3 + LoRA"]
SHORT = {"Classical": "Classical", "M1 EfficientNet-B4": "M1", "M2 GenD (DINOv3)": "M2", "M3 DINOv3 + LoRA": "M3"}


def select(t: pd.DataFrame, y: np.ndarray, models: dict) -> pd.DataFrame:
    ok = {n: models[n][1] == y for n in MODELS}
    dl_ok = ok["M1 EfficientNet-B4"] & ok["M2 GenD (DINOv3)"] & ok["M3 DINOv3 + LoRA"]
    cl_s = models["Classical"][0]
    win = t[dl_ok & ~ok["Classical"]].assign(cl=cl_s[dl_ok & ~ok["Classical"]])
    fakes = win[win.label == "fake"].sort_values("cl").drop_duplicates("base_id").head(2)
    reals = win[win.label == "real"].sort_values("cl", ascending=False).head(1)
    fail = t[~ok["Classical"] & ~dl_ok & ~ok["M1 EfficientNet-B4"] & ~ok["M2 GenD (DINOv3)"] & ~ok["M3 DINOv3 + LoRA"]]
    out = pd.concat([fakes.assign(kind="success"), reals.assign(kind="success"), fail.assign(kind="failure")])
    for n in MODELS:
        out[SHORT[n]] = models[n][0][out.index]
        out[SHORT[n] + "_ok"] = ok[n][out.index]
    return out.drop(columns="cl")


def load(path: str) -> np.ndarray:
    return np.asarray(Image.open(DATA / path).convert("RGB"), dtype=np.float32) / 255


def figure(ex: pd.DataFrame, real_path: dict, thr: dict):
    fig, axes = plt.subplots(len(ex), 4, figsize=(11, 2.9 * len(ex)),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    for r, (_, row) in enumerate(ex.iterrows()):
        img = load(row.cache_path)
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"{row.kind.upper()} · {row.label}\n{row.cache_path.split('/')[-1]}", fontsize=8,
                             color="#1a7f37" if row.kind == "success" else "#c62828")
        base = real_path.get(row.base_id) if row.label == "fake" else None
        if base:
            b = load(base)
            axes[r, 1].imshow(b)
            axes[r, 1].set_title("real photo it was made from", fontsize=8)
            axes[r, 2].imshow(np.abs(img - b).mean(2), cmap="magma", vmin=0, vmax=0.3)
            axes[r, 2].set_title("|fake − real| = swapped area", fontsize=8)
        else:
            for a in axes[r, 1:3]:
                a.text(0.5, 0.5, "real photo:\nno source pair", ha="center", va="center", fontsize=8, color="#52514e")
        lines = [f"{row.base_race}, {row.base_gender}, {row.base_age}"]
        if row.label == "fake":
            lines.append(f"donor: {row.donor_race}, {row.donor_gender}")
        lines.append("")
        for n in MODELS:
            k = SHORT[n]
            pred = "fake" if (row[k] >= thr[n]) else "real"
            lines.append(f"{k:9s} {'✓' if row[k + '_ok'] else '✗'} says {pred}  ({row[k]:.3g}, thr {thr[n]:.3g})")
        axes[r, 3].text(0, 0.5, "\n".join(lines), va="center", fontsize=8, family="monospace")
        for a in axes[r]:
            a.set_xticks([])
            a.set_yticks([])
        axes[r, 3].axis("off")
    fig.suptitle("Classical vs deep learning: 3 successes (only classical wrong) and 3 failures (all wrong)", fontsize=11)
    fig.tight_layout()
    fig.savefig(E.FIG / "e10_examples.png", dpi=130, bbox_inches="tight")


def main():
    df = load_index()
    t = eval_rows(df, "test").reset_index(drop=True)
    y = (t.label == "fake").to_numpy().astype(int)
    models = E.load_models(t)
    thr = {"Classical": pd.read_csv(E.EXP / "df_e03_classical/results.csv").query("model == 'hgb'").threshold.iloc[0]}
    for exp, n in E.DL.items():
        if n in MODELS:
            thr[n] = json.load(open(E.EXP / exp / "results.json"))["threshold"]
    ex = select(t, y, models)
    real_path = df[df.label == "real"].set_index("base_id").cache_path.to_dict()
    cols = ["kind", "cache_path", "label", "base_id", "donor_id", "base_race", "base_gender", "base_age",
            "donor_race", "donor_gender"] + [c for n in MODELS for c in (SHORT[n], SHORT[n] + "_ok")]
    ex[cols].to_csv(E.OUT / "examples.csv", index=False)
    print(ex[cols].to_string(index=False))
    figure(ex, real_path, thr)


if __name__ == "__main__":
    main()
