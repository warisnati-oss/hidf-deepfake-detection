"""df_e03_classical: hand-crafted P4 features + classical classifiers.

Features: data/features_p4.parquet (247 per image, step10). Missing values -> train median.
Classifiers (hyper-parameters chosen on val AUC):
    logreg  StandardScaler + LogisticRegression, C in {0.01, 0.1, 1, 10}
    svm_rbf StandardScaler + SVC(RBF) on a 10K stratified train subsample (full set is too slow)
    rf      RandomForest, 500 trees, max_depth in {None, 20}, min_samples_leaf in {1, 5}
    hgb     HistGradientBoosting, learning_rate in {0.05, 0.1}, max_leaf_nodes in {31, 63}
The best classifier on val is scored once on test; its threshold maximises val balanced accuracy.

Analysis with the best classifier type:
    group ablation  retrain without each feature group, and with global features only
                    (no P3 face mask) -> how much each kind of trace and P3 contribute
    importance      permutation importance on val, per group and per feature

Output: experiments/df_e03_classical/{search.csv, results.csv, ablation.csv, importance_*.csv,
        scores_test.csv, *.png}
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
OUT = PROJECT / "experiments" / "df_e03_classical"
FIG = PROJECT / "documents" / "figures" / "phase3"
SEED = 0
N_SVM = 10000

GROUPS = {
    "fft": lambda c: c.startswith("fft_"),
    "noise (whole image)": lambda c: c.startswith("res_") and "_all_" in c,
    "noise (face vs outside)": lambda c: c.startswith("res_") and "_all_" not in c,
    "lbp texture": lambda c: c.startswith("lbp_"),
    "colour": lambda c: c.startswith("col_"),
    "seam": lambda c: c.startswith("seam_") or c == "face_area",
}
GLOBAL_ONLY = ["fft", "noise (whole image)"]  # need no P3 face mask

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"logreg": "#2a78d6", "svm_rbf": "#6250d6", "rf": "#1baf7a", "hgb": "#eb6834"}


def candidates():
    for c in [0.01, 0.1, 1, 10]:
        yield "logreg", {"C": c}, make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
    for c in [1, 10]:
        yield "svm_rbf", {"C": c}, make_pipeline(StandardScaler(), SVC(C=c, kernel="rbf", gamma="scale"))
    for depth in [None, 20]:
        for leaf in [1, 5]:
            yield "rf", {"max_depth": depth, "min_samples_leaf": leaf}, RandomForestClassifier(
                n_estimators=500, max_depth=depth, min_samples_leaf=leaf, n_jobs=6, random_state=SEED)
    for lr in [0.05, 0.1]:
        for leaves in [31, 63]:
            yield "hgb", {"learning_rate": lr, "max_leaf_nodes": leaves}, HistGradientBoostingClassifier(
                learning_rate=lr, max_leaf_nodes=leaves, max_iter=1000, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=30, random_state=SEED)


def scores(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def best_threshold(y, s) -> float:
    cands = np.quantile(s, np.linspace(0.01, 0.99, 197))
    bal = [((s >= t)[y == 1].mean() + (s < t)[y == 0].mean()) / 2 for t in cands]
    return float(cands[int(np.argmax(bal))])


def metrics(y, s, thr) -> dict:
    p = (s >= thr).astype(int)
    return {"auc": roc_auc_score(y, s), "accuracy": accuracy_score(y, p), "precision": precision_score(y, p),
            "recall": recall_score(y, p), "f1": f1_score(y, p)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "index.csv", low_memory=False)
    feats = pd.read_parquet(DATA / "features_p4.parquet")
    cols = list(feats.columns)
    med = feats[df.split == "train"].median()
    X = feats.fillna(med).to_numpy(np.float32)
    y = (df.label == "fake").to_numpy().astype(int)
    # val/test exclude photos that also appear in train under another id (leakage audit, test C)
    idx = {s: ((df.split == s) & ~(df.near_dup & (s != "train"))).to_numpy() for s in ["train", "val", "test"]}
    svm_idx = (df[idx["train"]].groupby(["label", "prefix"]).sample(frac=N_SVM / idx["train"].sum(),
                                                                   random_state=SEED).index.to_numpy())

    # 1. model search on val
    search, fitted = [], {}
    for name, params, model in candidates():
        tr = svm_idx if name == "svm_rbf" else np.flatnonzero(idx["train"])
        model.fit(X[tr], y[tr])
        auc = roc_auc_score(y[idx["val"]], scores(model, X[idx["val"]]))
        search.append({"model": name, **{k: str(v) for k, v in params.items()}, "val_auc": auc})
        print(search[-1], flush=True)
        if name not in fitted or auc > fitted[name][0]:
            fitted[name] = (auc, params, model)
    pd.DataFrame(search).round(4).to_csv(OUT / "search.csv", index=False)

    # 2. best of each family on test (threshold from val)
    results, roc = [], {}
    for name, (val_auc, params, model) in fitted.items():
        s_val, s_test = scores(model, X[idx["val"]]), scores(model, X[idx["test"]])
        thr = best_threshold(y[idx["val"]], s_val)
        m = metrics(y[idx["test"]], s_test, thr)
        pre = df.prefix[idx["test"]].to_numpy()
        results.append({"model": name, "params": str(params), "val_auc": val_auc, "threshold": thr,
                        **{f"test_{k}": v for k, v in m.items()},
                        "test_auc_c": roc_auc_score(y[idx["test"]][pre == "c"], s_test[pre == "c"]),
                        "test_auc_f": roc_auc_score(y[idx["test"]][pre == "f"], s_test[pre == "f"])})
        roc[name] = (s_test, thr)
    res = pd.DataFrame(results).sort_values("val_auc", ascending=False)
    res.round(4).to_csv(OUT / "results.csv", index=False)
    print(res.round(4).to_string(index=False), flush=True)

    best = res.iloc[0].model
    s_best, thr_best = roc[best]
    pd.DataFrame({"cache_path": df.cache_path[idx["test"]].to_numpy(), "label": df.label[idx["test"]].to_numpy(),
                  "score": s_best, "pred_fake": (s_best >= thr_best).astype(int)}).to_csv(OUT / "scores_test.csv", index=False)

    # 3. feature-group ablation with the best family's chosen settings
    base_model = fitted[best][2]
    ablation = []
    variants = {"all features": cols,
                "global only (no P3 mask)": [c for c in cols if any(GROUPS[g](c) for g in GLOBAL_ONLY)]}
    for g, f in GROUPS.items():
        variants[f"without {g}"] = [c for c in cols if not f(c)]
    for vname, vcols in variants.items():
        ci = [cols.index(c) for c in vcols]
        m = clone(base_model)
        tr = svm_idx if best == "svm_rbf" else np.flatnonzero(idx["train"])
        m.fit(X[np.ix_(tr, ci)], y[tr])
        ablation.append({"variant": vname, "n_features": len(ci),
                         "val_auc": roc_auc_score(y[idx["val"]], scores(m, X[np.ix_(idx["val"], ci)])),
                         "test_auc": roc_auc_score(y[idx["test"]], scores(m, X[np.ix_(idx["test"], ci)]))})
        print(ablation[-1], flush=True)
    abl = pd.DataFrame(ablation)
    abl.round(4).to_csv(OUT / "ablation.csv", index=False)

    # 4. permutation importance on val (groups and single features)
    rng = np.random.default_rng(SEED)
    Xv, yv = X[idx["val"]], y[idx["val"]]
    base_auc = roc_auc_score(yv, scores(base_model, Xv))

    def drop_when_permuted(ci: list[int], repeats: int) -> float:
        out = []
        for _ in range(repeats):
            Xp = Xv.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, ci] = Xp[perm][:, ci]
            out.append(base_auc - roc_auc_score(yv, scores(base_model, Xp)))
        return float(np.mean(out))

    gi = pd.DataFrame([{"group": g, "auc_drop": drop_when_permuted([i for i, c in enumerate(cols) if f(c)], 3)}
                       for g, f in GROUPS.items()]).sort_values("auc_drop", ascending=False)
    gi.round(4).to_csv(OUT / "importance_groups.csv", index=False)
    fi = pd.DataFrame([{"feature": c, "auc_drop": drop_when_permuted([i], 2)} for i, c in enumerate(cols)])
    fi = fi.sort_values("auc_drop", ascending=False)
    fi.round(5).to_csv(OUT / "importance_features.csv", index=False)

    plot(df, y, idx, res, roc, best, abl, gi, fi)


def plot(df, y, idx, res, roc, best, abl, gi, fi):
    yt = y[idx["test"]]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (s, _) in roc.items():
        fpr, tpr, _ = roc_curve(yt, s)
        auc = res.set_index("model").test_auc[name]
        ax.plot(fpr, tpr, color=COLORS[name], linewidth=2, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=INK2, linewidth=1)
    ax.set_xlabel("false positive rate"), ax.set_ylabel("true positive rate")
    ax.set_title("df_e03 classical models — test ROC", fontweight="bold")
    ax.grid(color=GRID), ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(FIG / "e03_roc.png", dpi=150, bbox_inches="tight")

    s, thr = roc[best]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(confusion_matrix(yt, (s >= thr).astype(int)), display_labels=["real", "fake"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format=",")
    ax.set_title(f"df_e03 {best} — test confusion matrix", fontweight="bold", fontsize=11)
    fig.savefig(FIG / "e03_confusion.png", dpi=150, bbox_inches="tight")
    plot_features(abl, fi, best)


def plot_features(abl, fi, best):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.75})
    a = abl.iloc[::-1]
    axes[0].barh(a.variant, a.test_auc, color=["#eb6834" if v == "all features" else "#8fb8ea" for v in a.variant],
                 height=0.6)
    for yv, v in enumerate(a.test_auc):
        axes[0].annotate(f"{v:.3f}", (v, yv), xytext=(-4, 0), textcoords="offset points", va="center",
                         ha="right", fontsize=9, color="white" if v > 0.9 else INK)
    axes[0].set_xlim(max(0.5, a.test_auc.min() - 0.05), 1.0)
    axes[0].set_xlabel("test AUC")
    axes[0].set_title("Feature-group ablation", fontweight="bold")
    top = fi.head(15).iloc[::-1]
    axes[1].barh(top.feature, top.auc_drop, color=INK2, height=0.6)
    axes[1].set_xlabel("val AUC drop when permuted")
    axes[1].set_title(f"Top-15 features ({best})", fontweight="bold")
    for ax in axes:
        ax.grid(axis="x", color=GRID), ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG / "e03_features.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
