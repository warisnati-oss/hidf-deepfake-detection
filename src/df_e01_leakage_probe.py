"""df_e01_leakage_probe: does the original file format leak the label, and does P1 remove it?

Real f* images were PNG and fake f* images JPEG; c* images were JPEG on both sides.
So a c* score measures genuine swap signal, and an f* score far above it means the
model is reading the file format.

Probes (StandardScaler + LogisticRegression, fit on a train subset, AUC on val):
    meta      log file size (+ PNG flag for originals)
    global    P4 mask-free features: FFT spectrum + whole-image noise-residual stats
each on   raw      = original files (1024 px, original encoding)
          cache_v1 = 512 px, uniform JPEG q95 (first P1 version)
          cache    = final P1: JPEG q75 history matching at 1024 px, then 512 px JPEG q95

Region probes (P3 face mask): noise-residual stats outside the face only vs inside only.
A fake's background is its real base's pixels re-saved, so a background-only probe can
only work through compression history (leakage); genuine swap signal lives in the face.

Output: experiments/df_e01_leakage_probe/{features_*.parquet, results.csv}
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import face_features, global_features
from step9_face_regions import face_mask

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
RAW = PROJECT / "dataset" / "HiDF"
OUT = PROJECT / "experiments" / "df_e01_leakage_probe"
N_TRAIN = 16000
WORKERS = 6
SEED = 0


def feats(job: tuple[str, np.ndarray]) -> dict:
    path, polygon = job
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    face = face_mask(polygon * rgb.shape[0] / 512, dilation=8 * rgb.shape[0] // 512, size=rgb.shape[0])
    f = global_features(rgb)
    f.update({k: v for k, v in face_features(rgb, face).items() if k.startswith("res_")})
    return f


def extract(rows: pd.DataFrame, source: str) -> pd.DataFrame:
    out_file = OUT / f"features_{source}.parquet"
    if out_file.exists():
        return pd.read_parquet(out_file)
    if source == "raw":
        paths = [str(RAW / p) for p in rows.file_path]
    else:
        folder = "cache512_v1/" if source == "cache_v1" else "cache512/"
        paths = [str(DATA / p.replace("cache512/", folder)) for p in rows.cache_path]
    regions = np.load(DATA / "face_regions.npz", allow_pickle=True)
    poly = dict(zip(regions["cache_path"], regions["polygons"]))
    jobs = [(p, poly[c]) for p, c in zip(paths, rows.cache_path)]
    with ProcessPoolExecutor(WORKERS) as pool:
        f = pd.DataFrame(list(pool.map(feats, jobs, chunksize=32)), index=rows.index)
    f["log_bytes"] = np.log([Path(p).stat().st_size for p in paths])
    f["is_png"] = (rows.ext == "png").astype(float) if source == "raw" else 0.0
    f.to_parquet(out_file)
    return f


def probe(X_tr, y_tr, X_va, y_va, prefix_va) -> dict:
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_va)[:, 1]
    res = {"auc_all": roc_auc_score(y_va, p)}
    for pre in ["c", "f"]:
        m = prefix_va == pre
        res[f"auc_{pre}"] = roc_auc_score(y_va[m], p[m])
    res["gap_f_minus_c"] = res["auc_f"] - res["auc_c"]
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "index.csv", low_memory=False)
    train = (df[df.split == "train"]
             .groupby(["label", "prefix"], group_keys=False)
             .apply(lambda g: g.sample(int(N_TRAIN * len(g) / (df.split == "train").sum()), random_state=SEED)))
    # pandas 3 drops the grouping columns from apply(); take the full rows back by index
    rows = pd.concat([df.loc[train.index], df[df.split == "val"]])

    results = []
    for source in ["raw", "cache_v1", "cache"]:
        f = extract(rows, source)
        y = (rows.label == "fake").to_numpy().astype(int)
        tr = (rows.split == "train").to_numpy()
        groups = {
            "meta": ["log_bytes", "is_png"] if source == "raw" else ["log_bytes"],
            "global": [c for c in f.columns if c.startswith("fft_") or "_all_" in c],
            "background": [c for c in f.columns if "_out_" in c],
            "face": [c for c in f.columns if "_face_" in c],
        }
        for name, cols in groups.items():
            X = f[cols].fillna(0).to_numpy()
            r = probe(X[tr], y[tr], X[~tr], y[~tr], rows.prefix.to_numpy()[~tr])
            results.append({"source": source, "features": name, **r})
            print(results[-1], flush=True)

    res = pd.DataFrame(results).round(4)
    res.to_csv(OUT / "results.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
