"""Step 1: build an index of all HiDF images (one row per image file).

Real images:  Real-img/{id}.{jpg,jpeg,png}
Fake images:  Fake-img/{base_id}_{donor_id}.{jpg,png}
              base_id  = image that supplies the scene/body
              donor_id = image whose face is swapped in

Per file we record format, size, resolution and an MD5 hash (for exact-duplicate
detection). Demographics are joined from metadata.csv (Img rows only).

Output: data/raw_index.csv (no rows dropped; cleaning happens in step 2)
"""
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
DATASET = PROJECT / "dataset" / "HiDF"
OUT_DIR = PROJECT / "data"
WORKERS = 6  # leave cores for other jobs running on this Mac


def file_info(path: Path) -> dict:
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    try:
        with Image.open(path) as im:
            fmt, (w, h), mode = im.format, im.size, im.mode
    except Exception:
        fmt, w, h, mode = None, None, None, None
    return {"md5": md5, "format": fmt, "width": w, "height": h, "mode": mode,
            "bytes": path.stat().st_size}


def list_files() -> pd.DataFrame:
    rows = []
    for f in sorted((DATASET / "Real-img").iterdir()):
        rows.append({"file_path": str(f.relative_to(DATASET)), "label": "real",
                     "base_id": f.stem, "donor_id": None})
    for f in sorted((DATASET / "Fake-img").iterdir()):
        base, donor = f.stem.split("_")
        rows.append({"file_path": str(f.relative_to(DATASET)), "label": "fake",
                     "base_id": base, "donor_id": donor})
    df = pd.DataFrame(rows)
    df["ext"] = df.file_path.str.rsplit(".", n=1).str[-1].str.lower()
    df["prefix"] = df.base_id.str[0]
    return df


def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(DATASET / "metadata.csv")
    meta = meta.apply(lambda c: c.str.strip())
    meta = meta[meta["Img/Vid"] == "Img"].drop_duplicates("ID")
    return meta.rename(columns={"ID": "id", "Base/Swap": "role", "Race": "race",
                                "Gender": "gender", "Age": "age"})[
        ["id", "role", "race", "gender", "age"]]


def main():
    df = list_files()
    with ProcessPoolExecutor(WORKERS) as pool:
        info = list(pool.map(file_info, [DATASET / p for p in df.file_path], chunksize=64))
    df = pd.concat([df, pd.DataFrame(info)], axis=1)

    meta = load_metadata().set_index("id")
    for who in ["base", "donor"]:
        joined = meta.reindex(df[f"{who}_id"])
        for col in ["race", "gender", "age"]:
            df[f"{who}_{col}"] = joined[col].to_numpy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "raw_index.csv", index=False)
    print(df.groupby(["label", "prefix", "ext"]).size())
    print("unreadable:", df.format.isna().sum())


if __name__ == "__main__":
    main()
