"""df_e12 T2 (leakage B5 + B6): face geometry and the choice of base photos.

B5  Could face position / size alone separate real from fake (e.g. the swap tool re-aligns or
    rescales faces)? Gradient boosting on geometry only, trained on train, scored on test:
      position/size  centre, width, height, area of the MediaPipe face-oval box (6 numbers)
      + shape        the 36 face-oval points, centred and scaled (face shape comes from the donor,
                     so some signal here is the swap itself, not a leak)
    Paired check: how far a fake's face box moves from its own base photo's box.
B6  HiDF only made fakes from some reals. If bases differ from the other reals (pose, quality),
    "photo type" could separate classes. Real test photos are split into
      base (used as a base for >= 1 fake) · donor only · unused
    and each group is scored against the same test fakes.

Output: experiments/df_e12_leakage/{geometry.csv, base_selection.csv}
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import df_e10_evaluation as E
from evaluation import DATA, load_index

OUT = E.EXP / "df_e12_leakage"
MODELS = ["Classical", "M1 EfficientNet-B4", "M2 GenD (DINOv3)", "M3 DINOv3 + LoRA"]


def geometry(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    reg = np.load(DATA / "face_regions.npz", allow_pickle=True)
    poly = dict(zip(reg["cache_path"], reg["polygons"]))
    det = dict(zip(reg["cache_path"], reg["detected"]))
    pos, shape = [], []
    for p in df.cache_path:
        pts = np.asarray(poly[p], np.float32)
        lo, hi = pts.min(0), pts.max(0)
        c, wh = (lo + hi) / 2, hi - lo
        pos.append([c[0], c[1], wh[0], wh[1], wh[0] * wh[1], float(det[p])])
        shape.append(((pts - pts.mean(0)) / max(wh.max(), 1)).ravel())
    pos = pd.DataFrame(pos, columns=["cx", "cy", "w", "h", "area", "detected"], index=df.index)
    return pos, np.asarray(shape)


def main():
    OUT.mkdir(exist_ok=True)
    df = load_index()
    df = df[df.split.notna() & ~df.near_dup].reset_index(drop=True)
    pos, shape = geometry(df)
    y = (df.label == "fake").to_numpy().astype(int)
    tr, te = (df.split == "train").to_numpy(), (df.split == "test").to_numpy()

    rows = []
    for name, X in [("position/size (6)", pos.to_numpy()), ("position/size + oval shape (78)", np.hstack([pos, shape]))]:
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(X[tr], y[tr])
        rows.append({"features": name, "test_auc": roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])})
        print(rows[-1], flush=True)
    # paired: fake vs its own base photo (same scene, face replaced)
    real_pos = pos[df.label == "real"].set_index(df.base_id[df.label == "real"])
    f = df.label == "fake"
    b = real_pos.reindex(df.base_id[f]).to_numpy()
    d = pos[f].to_numpy() - b
    ok = ~np.isnan(d).any(1)
    for k, col in enumerate(["cx", "cy", "w", "h"]):
        rows.append({"features": f"paired |fake - base| {col} (px, median / p95)",
                     "median": float(np.median(np.abs(d[ok, k]))), "p95": float(np.percentile(np.abs(d[ok, k]), 95))})
    for col in ["cx", "cy", "w", "h"]:
        rows.append({"features": f"mean {col} real / fake (px)", "real": pos[col][~f].mean(), "fake": pos[col][f].mean()})
    geo = pd.DataFrame(rows)
    geo.to_csv(OUT / "geometry.csv", index=False)
    print(geo.round(3).to_string(index=False))

    # B6: base selection
    test = df[te].reset_index(drop=True)
    yt = (test.label == "fake").to_numpy().astype(int)
    used_base, used_donor = set(df.base_id[f]), set(df.donor_id[f].dropna())
    role = np.where(test.label == "fake", "fake",
                    np.where(test.base_id.isin(used_base), "base",
                             np.where(test.base_id.isin(used_donor), "donor only", "unused")))
    models = E.load_models(test)
    rows = []
    for name in MODELS:
        s, p = models[name]
        for r in ["base", "donor only", "unused"]:
            m = role == r
            both = m | (role == "fake")
            rows.append({"model": name, "real_group": r, "n_real": int(m.sum()),
                         "auc_vs_all_test_fakes": roc_auc_score(yt[both], s[both]), "fpr": p[m].mean(),
                         "median_score": float(np.median(s[m]))})
    base = pd.DataFrame(rows)
    base.to_csv(OUT / "base_selection.csv", index=False)
    print(base.assign(fpr=(base.fpr * 100).round(2), auc_vs_all_test_fakes=base.auc_vs_all_test_fakes.round(5)).to_string(index=False))

    # can geometry tell base reals from unused reals? (photo-type difference)
    real_te = test[role != "fake"]
    is_base = (role[role != "fake"] == "base").astype(int)
    real_tr = df[tr & (df.label == "real").to_numpy()]
    yb_tr = real_tr.base_id.isin(used_base).astype(int).to_numpy()
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(pos.loc[real_tr.index], yb_tr)
    te_idx = df.index[te][role != "fake"]
    print(f"geometry: base vs non-base reals, test AUC {roc_auc_score(is_base, clf.predict_proba(pos.loc[te_idx])[:, 1]):.3f}"
          f" (base share {is_base.mean():.2f})")


if __name__ == "__main__":
    main()
