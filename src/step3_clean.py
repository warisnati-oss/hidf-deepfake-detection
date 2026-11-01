"""Step 3: apply cleaning rules to the raw index.

Rules (in order; each logged with its row count):
    R1 drop empty files (0 bytes, unreadable)
    R2 drop .jpeg files in Real-img: all 235 are byte-identical copies of fakes
    R3 drop fakes whose swap changed (almost) nothing (step 2: near_identical)
    R4 drop exact duplicate files (same MD5) - keep the first; record the dropped id as
       an alias so the split keeps both ids in the same identity group
    R5 fakes with a wrongly named base: replace base_id with the recovered one (step 2)

Fakes whose true base is not in the dataset (step 2: base_unknown) are kept: no real
counterpart exists, so they cannot leak across splits.

Output: data/index.csv, data/id_aliases.csv, data/cleaning_log.csv
"""
from pathlib import Path

import pandas as pd

from step1_build_index import load_metadata

PROJECT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT / "data"


def main():
    df = pd.read_csv(OUT_DIR / "raw_index.csv", low_memory=False)
    checks = pd.read_csv(OUT_DIR / "fake_checks.csv").set_index("file_path")
    log = [("raw", "all files", len(df))]

    def drop(mask, rule, why):
        nonlocal df
        log.append((rule, why, int(mask.sum())))
        df = df[~mask]

    drop(df.bytes == 0, "R1", "empty file")
    drop((df.label == "real") & (df.ext == "jpeg"), "R2", "real .jpeg = copy of a fake")
    fake_check = df.file_path.map(checks.check)
    drop(fake_check == "near_identical", "R3", "fake with no visible swap")

    dup = df.md5.duplicated(keep="first")
    keep_id = df.groupby("md5").base_id.transform("first")
    aliases = pd.DataFrame({"id": df.base_id[dup], "alias_of": keep_id[dup]})
    drop(dup, "R4", "exact duplicate file")

    fake_check = df.file_path.map(checks.check)
    fix = fake_check == "base_recovered"
    df.loc[fix, "named_base_id"] = df.loc[fix, "base_id"]
    df.loc[fix, "base_id"] = df.loc[fix, "file_path"].map(checks.nn_base_id)
    meta = load_metadata().set_index("id").reindex(df.loc[fix, "base_id"])
    for col in ["race", "gender", "age"]:
        df.loc[fix, f"base_{col}"] = meta[col].to_numpy()
    log.append(("R5", "fake base id corrected", int(fix.sum())))
    df["fake_check"] = fake_check.fillna("")

    log.append(("clean", "real", int((df.label == "real").sum())))
    log.append(("clean", "fake", int((df.label == "fake").sum())))

    df = df.drop(columns=["md5"]).reset_index(drop=True)
    df.to_csv(OUT_DIR / "index.csv", index=False)
    aliases.to_csv(OUT_DIR / "id_aliases.csv", index=False)
    log = pd.DataFrame(log, columns=["rule", "description", "rows"])
    log.to_csv(OUT_DIR / "cleaning_log.csv", index=False)
    print(log.to_string(index=False))


if __name__ == "__main__":
    main()
