"""Person-overlap impact: AUC on test images whose person is seen vs unseen in train, and
feasibility of a person-disjoint split (component sizes after adding ArcFace same-person edges)."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
import step12_person_check as s
from step4_split import UnionFind

THR = 0.5


def main():
    D = s.DATA
    df = pd.read_csv(D / "index.csv", low_memory=False)
    real = df[df.label == "real"].reset_index()
    E = np.load(D / "arcface_real.npy"); ok = ~np.isnan(E).any(1)
    Et = torch.from_numpy(np.nan_to_num(E))
    train = np.flatnonzero((real.split == "train").to_numpy() & ok)

    # test fakes carry the donor's face -> embed them too
    tf = df[(df.label == "fake") & (df.split == "test") & ~df.near_dup]
    f_file = s.OUT / "arcface_test_fakes.npy"
    if f_file.exists():
        Ef = np.load(f_file)
    else:
        with ProcessPoolExecutor(6, initializer=s._init) as pool:
            Ef = np.stack(list(pool.map(s.embed, tf.cache_path, chunksize=64)))
        np.save(f_file, Ef)

    def max_sim(X):
        X = torch.from_numpy(np.nan_to_num(X)); return torch.cat([(X[i:i+1024] @ Et[train].T).max(1).values for i in range(0, len(X), 1024)]).numpy()

    tr_real = real[(real.split == "test") & ~real.near_dup]
    seen = pd.Series(np.concatenate([max_sim(E[tr_real.index]) >= THR, max_sim(Ef) >= THR]),
                     index=pd.concat([tr_real.cache_path, tf.cache_path]).to_numpy())
    print(f"test images with a seen person (sim>={THR}): real {seen[tr_real.cache_path].mean():.3f}, fake {seen[tf.cache_path].mean():.3f}")
    # per-image flag, reused by every Phase 4/5 evaluation
    seen.rename("person_seen").rename_axis("cache_path").reset_index().to_csv(s.OUT / "test_person_seen.csv", index=False)

    rows = []
    c = pd.read_csv(s.PROJECT / "experiments/df_e03_classical/scores_test.csv")
    z = pd.read_csv(s.PROJECT / "experiments/df_e02_zero_shot/scores_test.csv")
    for name, frame, col in [("classical HGB", c, "score"), ("CLIP L/14 crop", z, "clip_l14_crop"),
                             ("PE bigG crop", z, "pe_core_bigG_crop"), ("SigLIP 2 crop", z, "siglip2_so400m_crop")]:
        y = (frame.label == "fake").astype(int).to_numpy(); sn = seen[frame.cache_path].to_numpy()
        rows.append({"model": name, "auc_seen_person": roc_auc_score(y[sn], frame[col][sn]),
                     "auc_unseen_person": roc_auc_score(y[~sn], frame[col][~sn]), "n_seen": int(sn.sum()), "n_unseen": int((~sn).sum())})
    res = pd.DataFrame(rows).round(4); res.to_csv(s.OUT / "seen_vs_unseen.csv", index=False); print(res.to_string(index=False))

    # feasibility of a person-disjoint split
    uf = UnionFind()
    for g, i in zip(df.group_id, df.base_id): uf.union(f"g{g}", i)
    idx = np.flatnonzero(ok)
    for i in range(0, len(idx), 2048):
        sim = Et[idx[i:i+2048]] @ Et[idx].T
        a, b = torch.nonzero(sim >= THR, as_tuple=True)
        for x, y_ in zip(a.tolist(), b.tolist()):
            if idx[i + x] < idx[y_]: uf.union(real.base_id[idx[i + x]], real.base_id[idx[y_]])
    comp = df.base_id.map(uf.find).value_counts()
    print(f"person-merged components: {len(comp)}, largest {comp.iloc[0]} ({comp.iloc[0]/len(df):.1%}), 2nd {comp.iloc[1]}, 3rd {comp.iloc[2]}")


if __name__ == "__main__":
    main()
