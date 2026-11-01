"""Step 5 (preprocessing P1): format normalisation cache with compression-history matching.

The four groups were saved differently (df_e01):
    real c* JPEG q75 · fake c* JPEG q99 · real f* PNG (lossless) · fake f* JPEG q99
so file size / compression traces predict the label (AUC 0.99 from file size alone).

Every image -> RGB -> JPEG q75 4:2:0 at 1024 px (the lowest quality in the dataset, so every
image now carries one q75 compression) -> 512 px (Lanczos) -> JPEG q95 4:4:4.
The first version without the q75 step is kept as data/cache512_v1 for comparison.

Output: data/cache512/{real,fake}/{stem}.jpg ; data/index.csv gains `cache_path`
"""
import io
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
DATASET = PROJECT / "dataset" / "HiDF"
OUT_DIR = PROJECT / "data"
CACHE = OUT_DIR / "cache512"
SIZE = 512
QUALITY = 95
HISTORY_QUALITY = 75
WORKERS = 6


def convert(job: tuple[str, str]) -> None:
    src, dst = job
    dst = OUT_DIR / dst
    if dst.exists():
        return
    with Image.open(DATASET / src) as im:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=HISTORY_QUALITY, subsampling=2)
    with Image.open(buf) as im:
        im = im.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
        im.save(dst, "JPEG", quality=QUALITY, subsampling=0)


def main():
    df = pd.read_csv(OUT_DIR / "index.csv", low_memory=False)
    stems = df.file_path.str.split("/").str[-1].str.rsplit(".", n=1).str[0]
    df["cache_path"] = CACHE.name + "/" + df.label + "/" + stems + ".jpg"
    assert df.cache_path.is_unique
    for label in ["real", "fake"]:
        (CACHE / label).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(WORKERS) as pool:
        list(pool.map(convert, zip(df.file_path, df.cache_path), chunksize=64))
    df.to_csv(OUT_DIR / "index.csv", index=False)
    print(f"cached {len(df)} images to {CACHE}")


if __name__ == "__main__":
    main()
