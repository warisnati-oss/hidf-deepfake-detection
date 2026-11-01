"""Step 2: check every fake against the real image it claims to be built from.

A fake `{base_id}_{donor_id}` should equal its real base image everywhere except the
swapped face. On 128px grey/RGB thumbnails we measure:
    mad          mean absolute pixel difference fake vs named base (0-255 scale)
    changed_frac fraction of pixels whose difference > 12

Typical fakes: mad 2-10. Two failure modes:
    mad < 1    swap did (almost) nothing -> fake is effectively a real image
    mad > 20   named base is the wrong image -> try to recover the true base by
               matching the outer border of the image (background/hair/clothes),
               which a face swap leaves untouched.

Border match: 32x32 greyscale thumbnails, 6px outer ring, L1 distance. On normal fakes
the named base is always the nearest neighbour with distance <= 2.5, while the
second-nearest is >= 15, so distance < 5 counts as a confident match.

Output: data/fake_checks.csv
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
DATASET = PROJECT / "dataset" / "HiDF"
OUT_DIR = PROJECT / "data"
WORKERS = 6

MISMATCH_MAD = 20
MATCH_DIST = 5

_yy, _xx = np.mgrid[:32, :32]
RING = ((_yy < 6) | (_yy >= 26) | (_xx < 6) | (_xx >= 26)).ravel()


def load_rgb(rel: str, size: int) -> np.ndarray:
    with Image.open(DATASET / rel) as im:
        return np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32)


def pixel_diff(pair: tuple[str, str]) -> tuple[float, float]:
    d = np.abs(load_rgb(pair[0], 128) - load_rgb(pair[1], 128)).mean(2)
    return float(d.mean()), float((d > 12).mean())


def border(rel: str) -> np.ndarray:
    with Image.open(DATASET / rel) as im:
        g = np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float32)
    return g.ravel()[RING]


def main():
    df = pd.read_csv(OUT_DIR / "raw_index.csv", low_memory=False)
    # .jpeg files in Real-img are copies of fakes (see step 3), never a valid base
    real = df[(df.label == "real") & (df.ext != "jpeg")].drop_duplicates("base_id")
    real_path = real.set_index("base_id").file_path
    fake = df[(df.label == "fake") & (df.bytes > 0)].copy()

    has_base = fake.base_id.isin(real_path.index)
    fake["mad"] = np.nan
    fake["changed_frac"] = np.nan
    pairs = list(zip(fake.file_path[has_base], real_path[fake.base_id[has_base]]))
    with ProcessPoolExecutor(WORKERS) as pool:
        res = list(pool.map(pixel_diff, pairs, chunksize=64))
    fake.loc[has_base, ["mad", "changed_frac"]] = res

    # Recover the true base for fakes whose named base doesn't match
    bad = fake[(fake.mad > MISMATCH_MAD) | ~has_base]
    with ProcessPoolExecutor(WORKERS) as pool:
        real_b = np.stack(list(pool.map(border, real.file_path, chunksize=128)))
        bad_b = np.stack(list(pool.map(border, bad.file_path)))
    ids = real.base_id.to_numpy()
    nn_id, nn_dist = [], []
    for x in bad_b:
        d = np.abs(real_b - x).mean(1)
        i = d.argmin()
        nn_id.append(ids[i])
        nn_dist.append(d[i])
    fake["nn_base_id"] = pd.Series(nn_id, index=bad.index)
    fake["nn_dist"] = pd.Series(nn_dist, index=bad.index)

    fake["check"] = "ok"
    fake.loc[fake.mad < 1, "check"] = "near_identical"
    fake.loc[bad.index, "check"] = np.where(fake.loc[bad.index, "nn_dist"] < MATCH_DIST,
                                             "base_recovered", "base_unknown")

    cols = ["file_path", "base_id", "donor_id", "mad", "changed_frac", "nn_base_id", "nn_dist", "check"]
    fake[cols].to_csv(OUT_DIR / "fake_checks.csv", index=False)
    print(fake.check.value_counts())


if __name__ == "__main__":
    main()
