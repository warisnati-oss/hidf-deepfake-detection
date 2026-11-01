"""Shared evaluation for Phase 4+ models.

Protocol: val/test exclude `near_dup` images; the decision threshold maximises val balanced
accuracy; test is reported overall, per c*/f* prefix and per person seen/unseen in train
(ArcFace check, experiments/person_check/test_person_seen.csv).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"


def load_index() -> pd.DataFrame:
    return pd.read_csv(DATA / "index.csv", low_memory=False)


def eval_rows(df: pd.DataFrame, split: str) -> pd.DataFrame:
    return df[(df.split == split) & ~df.near_dup]


def best_threshold(y: np.ndarray, s: np.ndarray) -> float:
    cands = np.quantile(s, np.linspace(0.01, 0.99, 197))
    bal = [((s >= t)[y == 1].mean() + (s < t)[y == 0].mean()) / 2 for t in cands]
    return float(cands[int(np.argmax(bal))])


def report(test_rows: pd.DataFrame, s_test: np.ndarray, thr: float) -> dict:
    """All test metrics for one model. `test_rows` must be eval_rows(df, 'test') in the score order."""
    y = (test_rows.label == "fake").to_numpy().astype(int)
    p = (s_test >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p).ravel()
    out = {"test_auc": roc_auc_score(y, s_test), "test_accuracy": accuracy_score(y, p),
           "test_precision": precision_score(y, p), "test_recall": recall_score(y, p), "test_f1": f1_score(y, p),
           "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    pre = test_rows.prefix.to_numpy()
    for k in ["c", "f"]:
        out[f"test_auc_{k}"] = roc_auc_score(y[pre == k], s_test[pre == k])
    seen = pd.read_csv(PROJECT / "experiments" / "person_check" / "test_person_seen.csv").set_index("cache_path")
    sn = seen.person_seen.reindex(test_rows.cache_path).to_numpy().astype(bool)
    out["test_auc_person_seen"] = roc_auc_score(y[sn], s_test[sn])
    out["test_auc_person_unseen"] = roc_auc_score(y[~sn], s_test[~sn])
    return out
