"""df_e03 leakage audit: is the classical model's 0.998 test AUC real?

A  label-shuffle control: same HGB, train labels permuted -> must give AUC ~0.5
   (catches pipeline bugs that let the label reach the features).
B  background-only HGB: the strongest classical model on features computed only OUTSIDE the
   P3 face mask (noise-residual stats, LBP, colour). A fake's background is its real base's
   pixels re-saved, so anything above ~0.5 means compression/processing history leaks.
C  near-duplicate photos across splits: the identity split cannot see the same photo stored
   under two unrelated ids. Compare 16x16 thumbnails of every val/test image with every
   train image; the reference for "same photo" is the distance between a fake and its own
   real base (same scene, only the face differs).

Output: experiments/df_e03_classical/leakage_audit.csv, near_duplicates.csv
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.feature import local_binary_pattern
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from features import _moments, residuals, to_gray
from step9_face_regions import face_mask

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
OUT = PROJECT / "experiments" / "df_e03_classical"
WORKERS = 6  # run alone: next to PE-bigG (15 GB) macOS killed it twice
SEED = 0
N_TRAIN_BG = 20000

HGB = HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=31, max_iter=1000, early_stopping=True,
                                     validation_fraction=0.1, n_iter_no_change=30, random_state=SEED)


def background_features(job) -> dict:
    cv2.setNumThreads(1)
    path, polygon = job
    rgb = np.asarray(Image.open(DATA / path).convert("RGB"))
    face = face_mask(polygon)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    bg = ~cv2.dilate(face.astype(np.uint8), k).astype(bool)  # 20 px safety gap from the face
    gray = to_gray(rgb)
    out = {}
    for name, r in residuals(gray).items():
        for stat, v in zip(["std", "mabs", "skew", "kurt"], _moments(r[bg])):
            out[f"bg_res_{name}_{stat}"] = v
    for p, rad in [(8, 1), (16, 2)]:
        lbp = local_binary_pattern(gray.astype(np.uint8), p, rad, method="uniform")[bg]
        hist = np.bincount(lbp.astype(int), minlength=p + 2) / max(lbp.size, 1)
        out.update({f"bg_lbp_p{p}_{i:02d}": v for i, v in enumerate(hist)})
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    for c in range(3):
        out[f"bg_ycc{c}_mean"], out[f"bg_ycc{c}_std"] = float(ycc[..., c][bg].mean()), float(ycc[..., c][bg].std())
    return out


def thumb(path: str) -> np.ndarray:
    with Image.open(DATA / path) as im:
        return np.asarray(im.convert("L").resize((16, 16), Image.BILINEAR), dtype=np.float32).ravel()


def save(results: list[dict]):
    old = pd.read_csv(OUT / "leakage_audit.csv") if (OUT / "leakage_audit.csv").exists() else pd.DataFrame()
    new = pd.DataFrame(results)
    if len(old):
        old = old[~old.test.isin(new.test)]
    pd.concat([old, new]).round(4).to_csv(OUT / "leakage_audit.csv", index=False)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default="ABC")
    tests = ap.parse_args().tests
    df = pd.read_csv(DATA / "index.csv", low_memory=False)
    y = (df.label == "fake").to_numpy().astype(int)
    split = df.split.to_numpy()
    results = []

    # A. label shuffle
    if "A" in tests:
        run_a(df, y, split, results)
        save(results)
    if "B" in tests:
        run_b(df, split, results)
        save(results)
    if "C" in tests:
        run_c(df, split, results)
        save(results)
    print(pd.read_csv(OUT / "leakage_audit.csv").to_string(index=False))


def run_a(df, y, split, results):
    feats = pd.read_parquet(DATA / "features_p4.parquet")
    X = feats.fillna(feats[split == "train"].median()).to_numpy(np.float32)
    tr, te = split == "train", split == "test"
    y_perm = np.random.default_rng(SEED).permutation(y[tr])
    m = clone(HGB).fit(X[tr], y_perm)
    results.append({"test": "A label shuffle (all 247 features)", "test_auc": roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])})
    print(results[-1], flush=True)


def run_b(df, split, results):
    tr, te = split == "train", split == "test"
    # load once: every npz["key"] access re-reads the whole array, and an indexed view keeps it alive
    polygons = np.load(DATA / "face_regions.npz", allow_pickle=True)["polygons"]
    rows = pd.concat([df[tr].groupby(["label", "prefix"]).sample(frac=N_TRAIN_BG / tr.sum(), random_state=SEED),
                      df[te]])
    jobs = [(p, polygons[i].copy()) for i, p in zip(rows.index, rows.cache_path)]
    with ProcessPoolExecutor(WORKERS) as pool:
        bg = pd.DataFrame(list(pool.map(background_features, jobs, chunksize=64)), index=rows.index)
    bg_tr, bg_te = (rows.split == "train").to_numpy(), (rows.split == "test").to_numpy()
    Xb = bg.fillna(bg[bg_tr].median()).to_numpy(np.float32)
    yb = (rows.label == "fake").to_numpy().astype(int)
    pre = rows.prefix.to_numpy()
    m = clone(HGB).fit(Xb[bg_tr], yb[bg_tr])
    s = m.predict_proba(Xb[bg_te])[:, 1]
    results.append({"test": "B background only (HGB, noise + LBP + colour)", "test_auc": roc_auc_score(yb[bg_te], s),
                    "test_auc_c": roc_auc_score(yb[bg_te][pre[bg_te] == "c"], s[pre[bg_te] == "c"]),
                    "test_auc_f": roc_auc_score(yb[bg_te][pre[bg_te] == "f"], s[pre[bg_te] == "f"])})
    print(results[-1], flush=True)


def run_c(df, split, results):
    tr = split == "train"
    with ProcessPoolExecutor(WORKERS) as pool:
        T = np.stack(list(pool.map(thumb, df.cache_path, chunksize=256)))
    T = (T - T.mean(1, keepdims=True)) / (T.std(1, keepdims=True) + 1e-6)  # brightness/contrast invariant
    Tt = torch.from_numpy(T)

    real_idx = df[df.label == "real"].reset_index().set_index("base_id")["index"]
    fk = df[(df.label == "fake") & (df.fake_check == "ok") & df.base_id.isin(real_idx.index)].sample(3000, random_state=SEED)
    pair_d = torch.linalg.norm(Tt[fk.index.to_numpy()] - Tt[real_idx[fk.base_id].to_numpy()], dim=1).numpy() / 16
    ref = float(np.percentile(pair_d, 50))

    train_t = Tt[np.flatnonzero(tr)]
    dup_rows = []
    for s_name in ["val", "test"]:
        q_idx = np.flatnonzero(split == s_name)
        nn_d, nn_i = [], []
        for i in range(0, len(q_idx), 512):
            d = torch.cdist(Tt[q_idx[i:i + 512]], train_t) / 16
            v, j = d.min(1)
            nn_d.append(v.numpy())
            nn_i.append(j.numpy())
        nn_d, nn_i = np.concatenate(nn_d), np.concatenate(nn_i)
        close = nn_d <= ref
        results.append({"test": f"C near-duplicates {s_name}->train (dist <= median fake-vs-own-base {ref:.3f})",
                        "count": int(close.sum()), "share": float(close.mean()),
                        "nn_dist_p1": float(np.percentile(nn_d, 1)), "nn_dist_median": float(np.median(nn_d))})
        print(results[-1], flush=True)
        train_rows = np.flatnonzero(tr)[nn_i[close]]
        dup_rows.append(pd.DataFrame({"split": s_name, "query": df.cache_path.to_numpy()[q_idx[close]],
                                      "train_match": df.cache_path.to_numpy()[train_rows], "dist": nn_d[close]}))
    pd.concat(dup_rows).to_csv(OUT / "near_duplicates.csv", index=False)
    # Flag them so every evaluation (val selection and test) excludes them
    flagged = set(pd.concat(dup_rows)["query"])
    df["near_dup"] = df.cache_path.isin(flagged)
    df.to_csv(DATA / "index.csv", index=False)
    print(f"fake-vs-own-base distance: median {ref:.3f}, p5 {np.percentile(pair_d, 5):.3f}, p95 {np.percentile(pair_d, 95):.3f}")


if __name__ == "__main__":
    main()
