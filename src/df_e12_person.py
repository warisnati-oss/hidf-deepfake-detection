"""df_e12 T3 (leakage A2): AUC on test images whose person is clearly NOT in train.

The identity split can't see one person stored under two unrelated ids. Reuses the ArcFace
embeddings of step 12 (all reals + test fakes). A test image is "unseen" at threshold t when its
own face has max cosine similarity < t to every train real, and for a fake also its base and its
donor (the two people it is made of). Images without a usable face embedding are left out of the
unseen set. Lower t = stricter (0.5 was used in Phase 3-5). Below ~0.25 nothing is left: with ~27K train
faces, the nearest one is always >= 0.20 by chance.

If memorising people drove the scores, AUC / accuracy on the strict-unseen subset would drop.

Output: experiments/df_e12_leakage/person_strict.csv
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from evaluation import DATA, eval_rows, load_index
import df_e10_evaluation as E

OUT = E.EXP / "df_e12_leakage"
THRESHOLDS = [0.5, 0.4, 0.3, 0.25]  # 0.2 leaves nothing: the nearest of ~27K train faces is always >= 0.20
MODELS = ["Classical", "M1 EfficientNet-B4", "M2 GenD (DINOv3)", "M3 DINOv3 + LoRA"]


def main():
    OUT.mkdir(exist_ok=True)
    df = load_index()
    real = df[df.label == "real"].reset_index(drop=True)
    Er = np.load(DATA / "arcface_real.npy")
    ok_r = ~np.isnan(Er).any(1)
    Et = torch.from_numpy(np.nan_to_num(Er[(real.split == "train").to_numpy() & ok_r]))

    def max_sim(X):
        bad = np.isnan(X).any(1)
        X = torch.from_numpy(np.nan_to_num(X))
        s = torch.cat([(X[i:i + 1024] @ Et.T).max(1).values for i in range(0, len(X), 1024)]).numpy()
        s[bad] = np.nan
        return s

    sim_real = pd.Series(max_sim(Er), index=real.base_id)  # every real, by its id
    test = eval_rows(df, "test").reset_index(drop=True)
    tf = test[test.label == "fake"]
    Ef = np.load(E.EXP / "person_check" / "arcface_test_fakes.npy")
    assert len(Ef) == len(tf)
    own = pd.Series(np.nan, index=test.index)
    own[test.label == "real"] = sim_real.reindex(test.base_id[test.label == "real"]).to_numpy()
    own[tf.index] = max_sim(Ef)
    base = test.base_id.map(sim_real)  # for reals this is the image itself
    donor = test.donor_id.map(sim_real)  # NaN for reals
    worst = pd.concat([own, base, donor.where(test.label == "fake", -1)], axis=1).max(axis=1, skipna=False)
    print(f"test images with a usable embedding chain: {worst.notna().mean():.1%}")

    y = (test.label == "fake").to_numpy().astype(int)
    models = E.load_models(test)
    rows = []
    for t in THRESHOLDS:
        unseen = (worst < t).to_numpy()
        for name in MODELS:
            s, p = models[name]
            for subset, m in [("unseen", unseen), ("seen", worst.ge(t).to_numpy())]:
                if (y[m] == 0).sum() == 0 or (y[m] == 1).sum() == 0:
                    continue
                rows.append({"threshold": t, "model": name, "subset": subset, "n_real": int((y[m] == 0).sum()),
                             "n_fake": int((y[m] == 1).sum()), "auc": roc_auc_score(y[m], s[m]),
                             "accuracy": (p[m] == y[m]).mean(), "fnr": 1 - p[m][y[m] == 1].mean(),
                             "fpr": p[m][y[m] == 0].mean()})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "person_strict.csv", index=False)
    show = res.assign(auc=res.auc.round(5), accuracy=(res.accuracy * 100).round(2), fnr=(res.fnr * 100).round(2),
                      fpr=(res.fpr * 100).round(2))
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
