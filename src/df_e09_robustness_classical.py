"""df_e09_robustness (classical part): does the classical model survive real-world degradations?

The df_e03 classical model (HistGradientBoosting, lr 0.05, 31 leaves, trained on clean train
features) is refit with the same settings and seed, saved, and NOT retrained on degraded data.
For every degradation in degrade.CONDITIONS, each test image is degraded, the P3 face mask is
recomputed on the degraded image (as a real deployment would; mean polygon if no face is found),
the 247 P4 features are extracted, and the model is scored with its clean-val threshold.

Output: experiments/df_e09_robustness/{classical_hgb.joblib, features/classical_<cond>.parquet,
        results_classical.csv}
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier

from degrade import CONDITIONS, degrade
from evaluation import DATA, best_threshold, eval_rows, load_index, report
from features import face_features, global_features
from step9_face_regions import face_mask

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "experiments" / "df_e09_robustness"
WORKERS = 5  # leave a core for the GPU job's data loader
SEED = 0

_seg = None
_mean_poly = None


def _init(mean_poly):
    global _seg, _mean_poly
    from face_regions import MediaPipeOval
    cv2.setNumThreads(1)
    _seg = MediaPipeOval()
    _mean_poly = mean_poly


def extract(job) -> dict:
    path, condition = job
    rgb = degrade(np.asarray(Image.open(DATA / path).convert("RGB")), condition, key=path)
    res = _seg.landmarker.detect(_seg._mp_image(image_format=_seg._format, data=np.ascontiguousarray(rgb)))
    if res.face_landmarks:
        lm = res.face_landmarks[0]
        poly = np.array([[lm[i].x * 512, lm[i].y * 512] for i in _seg.oval_idx], np.float32)
        found = True
    else:
        poly, found = _mean_poly, False
    f = global_features(rgb)
    f.update(face_features(rgb, face_mask(poly)))
    f["face_found"] = found
    return f


def main():
    (OUT / "features").mkdir(parents=True, exist_ok=True)
    df = load_index()
    feats = pd.read_parquet(DATA / "features_p4.parquet")
    cols = list(feats.columns)
    med = feats[df.split == "train"].median()
    tr = (df.split == "train").to_numpy()
    val, test = eval_rows(df, "val"), eval_rows(df, "test")

    model_file = OUT / "classical_hgb.joblib"
    if model_file.exists():
        model = joblib.load(model_file)
    else:
        model = HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=31, max_iter=1000, early_stopping=True,
                                               validation_fraction=0.1, n_iter_no_change=30, random_state=SEED)
        model.fit(feats[tr].fillna(med).to_numpy(np.float32), (df.label[tr] == "fake").to_numpy().astype(int))
        joblib.dump(model, model_file)
    y_val = (val.label == "fake").to_numpy().astype(int)
    thr = best_threshold(y_val, model.predict_proba(feats.loc[val.index].fillna(med).to_numpy(np.float32))[:, 1])

    regions = np.load(DATA / "face_regions.npz", allow_pickle=True)["polygons"]
    mean_poly = regions.mean(0)
    results = []
    for condition in CONDITIONS:
        f_file = OUT / "features" / f"classical_{condition}.parquet"
        if condition == "clean":
            X = feats.loc[test.index]
            found = 1.0
        else:
            if f_file.exists():
                X = pd.read_parquet(f_file)
            else:
                with ProcessPoolExecutor(WORKERS, initializer=_init, initargs=(mean_poly,)) as pool:
                    X = pd.DataFrame(list(pool.map(extract, [(p, condition) for p in test.cache_path], chunksize=32)),
                                     index=test.index)
                X.to_parquet(f_file)
            found = float(X.face_found.mean())
        s = model.predict_proba(X[cols].fillna(med).to_numpy(np.float32))[:, 1]
        results.append({"model": "classical_hgb", "condition": condition, "face_found": found, "threshold": thr,
                        **report(test, s, thr)})
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in results[-1].items()
               if k in ["condition", "face_found", "test_auc", "test_accuracy", "test_recall", "test_f1"]}, flush=True)
        pd.DataFrame(results).round(4).to_csv(OUT / "results_classical.csv", index=False)


if __name__ == "__main__":
    main()
