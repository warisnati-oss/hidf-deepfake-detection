"""Step 10 (preprocessing P4): extract the 247 classical features for every image.

Global features (FFT spectrum, whole-image residual stats) + face features (residual stats
inside/outside the P3 face mask, LBP, colour vs surrounding skin, seam gradients).

Output: data/features_p4.parquet (index = row of data/index.csv)
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per worker; avoids oversubscription

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from features import face_features, global_features
from step9_face_regions import face_mask

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
WORKERS = 6


def extract(job: tuple[str, np.ndarray]) -> dict:
    cv2.setNumThreads(1)
    path, polygon = job
    with Image.open(DATA / path) as im:
        rgb = np.asarray(im.convert("RGB"))
    f = global_features(rgb)
    f.update(face_features(rgb, face_mask(polygon)))
    return f


def main():
    df = pd.read_csv(DATA / "index.csv", low_memory=False)
    regions = np.load(DATA / "face_regions.npz", allow_pickle=True)
    assert (regions["cache_path"] == df.cache_path.to_numpy()).all()
    jobs = list(zip(df.cache_path, regions["polygons"]))
    with ProcessPoolExecutor(WORKERS) as pool:
        feats = pd.DataFrame(list(pool.map(extract, jobs, chunksize=64)), index=df.index)
    feats.to_parquet(DATA / "features_p4.parquet")
    print(f"{feats.shape[1]} features for {len(feats)} images; NaN cells: {int(feats.isna().sum().sum())}")


if __name__ == "__main__":
    main()
