# Deepfake Detection on HiDF

Course project for 2110651 Digital Image Processing: classify face photos from the
[HiDF](https://github.com/DSAIL-SKKU/HiDF) dataset (Kang et al., KDD 2025) as real or fake (face swap). We compare a
zero-shot model, a classical model (hand-made features + gradient boosting) and three deep
models (EfficientNet-B4, GenD on DINOv3, DINOv3 + LoRA). We also test robustness, fairness and
possible leakage.

**Start here:** `documents/PLAN.md` (plan and status), then one document per phase:

| Phase | Document |
|---|---|
| 1 Data cleaning and split | `documents/DATA_CLEANING.md` |
| 2 Preprocessing (P1–P4) | `documents/PREPROCESSING.md` |
| 3 Zero-shot and classical | `documents/CLASSICAL_AND_ZERO_SHOT.md` |
| 4 Deep learning (12 runs) | `documents/DEEP_LEARNING.md` |
| 5 Evaluation | `documents/EVALUATION.md` |
| 5b Leakage checks | `documents/LEAKAGE.md` |
| Literature | `documents/LITERATURE_REVIEW.md` |
| Full history | `documents/EXPERIMENT_LOG.md` |

## Results (clean test set, 9,600 images)

| Model | AUC | Accuracy |
|---|---|---|
| Zero-shot CLIP (face crop) | 0.953 | 90.6 % |
| Classical (P4 features + gradient boosting) | 0.998 | 98.1 % |
| M1 EfficientNet-B4 | 0.9999 | 99.7 % |
| M2 GenD (DINOv3) | 0.9998 | 99.7 % |
| M3 DINOv3 + LoRA | 1.0000 | 99.8 % |

## Repository layout

- `src/` — all code. `step*.py` build the data (index, cleaning, split, cache, face regions,
  features); `df_eXX_*.py` are the experiments; `train.py` trains the deep models.
- `documents/` — write-ups and figures (`documents/figures/<phase>/`).
- `experiments/` — configs, logs and result tables per experiment (`df_eXX_*`).
  Weights, embeddings and features are not in the repo (too large).
- `data/index.csv` — every image with its label, identity group and train/val/test split.

## Not included

The HiDF dataset (`dataset/`), the 512 px image cache (`data/cache512/`), trained weights and
downloaded model files (`models/`). Put the dataset at `dataset/HiDF/` and rebuild with the
`src/step*.py` scripts in order. Environment: Python 3.12, `pip install -r requirements.lock.txt`
(Apple Silicon / MPS).
