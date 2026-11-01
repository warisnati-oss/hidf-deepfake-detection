"""Step 4: identity-safe train / val / test split.

The same person appears as a real image and inside fakes (as scene base or as face
donor), so a random split would put that person in train and test. We build a graph
whose nodes are image ids and whose edges are
    fake  base_id -- donor_id   (and the original named base id, if corrected)
    alias duplicate id -- kept id
and treat every connected component as one group. Whole groups go to one split.

StratifiedGroupKFold with 20 folds (stratified on label + c/f prefix):
14 folds train (70%), 3 val (15%), 3 test (15%).

Output: data/index.csv gains `group_id` and `split` columns
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PROJECT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT / "data"
SEED = 42


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def main():
    df = pd.read_csv(OUT_DIR / "index.csv", low_memory=False)
    aliases = pd.read_csv(OUT_DIR / "id_aliases.csv")

    uf = UnionFind()
    for i in df.base_id:
        uf.find(i)
    fake = df[df.label == "fake"]
    for b, d in zip(fake.base_id, fake.donor_id):
        uf.union(b, d)
    for b, n in fake[["base_id", "named_base_id"]].dropna().itertuples(index=False):
        uf.union(b, n)
    for a, k in aliases.itertuples(index=False):
        uf.union(a, k)

    roots = df.base_id.map(uf.find)
    df["group_id"] = pd.factorize(roots)[0]

    strata = df.label + "_" + df.prefix
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    fold = pd.Series(-1, index=df.index)
    for k, (_, idx) in enumerate(sgkf.split(df, strata, df.group_id)):
        fold.iloc[idx] = k
    df["split"] = pd.cut(fold, [-1, 13, 16, 19], labels=["train", "val", "test"]).astype(str)

    # Sanity: no id (base or donor) may appear in two splits
    ids = pd.concat([df[["base_id", "split"]].rename(columns={"base_id": "id"}),
                     fake.assign(split=df.split)[["donor_id", "split"]].rename(columns={"donor_id": "id"})])
    leaked = ids.groupby("id").split.nunique().gt(1).sum()
    assert leaked == 0, f"{leaked} ids appear in more than one split"

    df.to_csv(OUT_DIR / "index.csv", index=False)
    sizes = df.groupby("group_id").size()
    print(f"groups: {len(sizes)}, largest: {sizes.max()}, ids leaked across splits: {leaked}")
    print(pd.crosstab(df.split, [df.label, df.prefix], margins=True))
    print(pd.crosstab(df.split, df.label, normalize="index").round(3))


if __name__ == "__main__":
    main()
