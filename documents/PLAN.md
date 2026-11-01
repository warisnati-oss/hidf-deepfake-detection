# Deepfake Detection on HiDF — Project Plan

Course: 2110651 Digital Image Processing. Task: binary real/fake classification of face images.
Deliverable: slide presentation following `2110651_Project_Template.pdf` (+ peer review / Q&A).
Hardware: single Mac (Apple M6, 32 GB, MPS). Experiments are named `df_eXX_<name>`;
each finished phase gets its own `.md` in `documents/`.

## Grading rubric → where we cover it

Each criterion is 5 points. Counts in brackets are for a 3-person / 4-person team; **our team has 4**,
so we need 5 papers, 3 preprocessing methods, baseline + 3 adjustments. Time is not a constraint.

| # | Criterion | Our coverage |
|---|---|---|
| 1 | Dataset + preparation, with visualisations | HiDF overview, sample images, class/demographic charts, identity-safe split (Phase 1) |
| 2 | Literature review [4 / 5 papers], gap linked to our method | 5 papers → `LITERATURE_REVIEW.md` |
| 3 | Preprocessing [2 / 3 methods], problem-specific, pipeline diagram + before/after | 4 methods below |
| 4 | Model understanding: pipeline + architecture + design choices | Classical, EfficientNet-B4, GenD-CLIP diagrams |
| 5 | Experimentation [baseline + 2 / 3 adjustments], Acc/Prec/Rec/F1/AUC + confusion matrices | Zero-shot vs classical vs DL M1 vs DL M2, plus adjustments |
| 6 | Analysis: 3 success + 3 failure examples, classical vs DL | Picked from the test set after Phase 4 |
| 7 | Peer review + Q&A | Prepare for leakage / split / generalisation questions |

## Dataset facts (from initial inspection, 2026-10-24)

- `dataset/HiDF/Real-img/` — 38,582 images, all 1024×1024 aligned faces.
  `c*` ids (~19.6K, JPG) and `f*` ids (~19K, PNG).
- `dataset/HiDF/Fake-img/` — 31,249 face swaps, 1024×1024, named `target_source.jpg`
  (face of `source` pasted into `target`). Mostly JPG, 850 PNG.
- `metadata.csv` — 42,901 rows: ID, Img/Vid, Base/Swap, Race, Gender, Age.
- `FINAL_DATASET.csv` — unrelated dataset (Unsplash URLs + StyleGAN3), not used.
- GitHub lists 30,250 real / 30,250 fake images; local folders have more — explain in Phase 1.

### Known risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| Format leakage | Real `f*` are PNG, fake `f*` are JPG → a model can learn the file format | Re-encode everything identically (preprocessing P1); leakage probe `df_e01` |
| Identity leakage | Every fake's target/source also appears as a real image | Group split by connected components of the target–source graph (7,411 groups, largest 1,922 images) |
| Dirty metadata | 42 duplicate real ids, 18 fakes with unknown ids, trailing spaces | Clean in Phase 1, log every rule |
| Class imbalance | 38.6K real vs 31.2K fake | Report balanced accuracy / AUC; class weights if needed |

## Literature review (criterion 2)

| Paper | Role in our story |
|---|---|
| Rössler et al., FaceForensics++ / XceptionNet (ICCV 2019) | Classic CNN detector baseline |
| Durall et al., "Unmasking DeepFakes with simple features" (2020) | Classical FFT-spectrum + SVM detector → basis of our classical method |
| Shiohara & Yamasaki, Self-Blended Images (CVPR 2022) | Blending-artifact view of face swaps |
| Kang et al., HiDF (KDD 2025) | Our dataset; existing detectors ~0.70 AUC (EfficientNet-B4) |
| Yan et al., Effort (ICML 2025) | Parameter-efficient CLIP adaptation |
| Yermakov et al., GenD (WACV 2026) | LayerNorm-only tuning → our DL M2 |

Gap to state: HiDF fakes are human-indistinguishable and existing detectors score poorly; recent
parameter-efficient foundation-model tuning has not been evaluated on HiDF, and published
HiDF results don't control for identity overlap or file-format leakage.

## Preprocessing (criterion 3) — each with before/after examples

| ID | Method | Problem it solves |
|---|---|---|
| P1 | Compression-history matching: JPEG q75 at 1024 px → 512 px resize → JPEG q95 | Classes were saved differently (real c* q75, fakes q99, real f* PNG); decode speed |
| P2 | Augmentation / noise addition: JPEG re-compression, Gaussian noise, Gaussian blur, colour jitter, flip | Over-fitting to fragile pixel traces; robustness |
| P3 | Face-region segmentation: best of MediaPipe landmarks / BiSeNet face parsing / SegFace, chosen by coverage + IoU against the free swap masks (fake − real base) → face mask, seam band, square crop | Swap artifacts live on the inner face; background/hair is noise |
| P4 | High-pass / noise-residual filtering (Laplacian, SRM filters) + FFT spectrum | Exposes blending and resampling traces for the classical model |

## Models (criterion 4) and experiments (criterion 5)

| ID | Model | Type |
|---|---|---|
| `df_e01_leakage_probe` | Classical features on raw files vs P1 cache | Sanity check (must drop to ~chance on cache, or the shortcut is still there) |
| `df_e02_zero_shot` | CLIP ViT-L/14, SigLIP 2 SO400M/16-384, PE-Core-L/14-336 with prompt ensembles ("a real photo of a face" vs "a deepfake face swap"); full image and P3 crop | Zero-shot |
| `df_e03_classical` | P4 features (FFT azimuthal spectrum, LBP texture, noise-residual stats, face-vs-surround colour, seam gradients) + SVM / Random Forest / LogReg. The classifier and features are classical; the face mask comes from P3 | Classical |
| `df_e04_effnet_b4` | EfficientNet-B4 fully fine-tuned, 224 px | **DL M1 = baseline** (matches HiDF paper) |
| `df_e05a_linear_probe` | Frozen CLIP L/14, SigLIP 2, PE-Core-L, **DINOv3 ViT-L/16** features + logistic regression | Picks the GenD backbone |
| `df_e05_gend` | GenD on the best linear-probe backbone: LayerNorm-only tuning + alignment/uniformity loss | DL M2 (state of the art) |
| `df_e05b_dinov3_lora` | DINOv3 ViT-L/16 + LoRA adapters on attention, DINO-MAC style (1st place NTIRE 2026 Robust Deepfake Detection) | DL M3 (robustness-oriented) |
| `df_e05c_rigid` | RIGID on DINOv3: embedding stability under small noise, no training | Training-free zero-shot (deferred from Phase 3) |

Adjustments run on all three deep models (M1, M2, M3) — user decision 2026-10-26:

| ID | Adjustment |
|---|---|
| `df_e06_augmentation` | Without vs with P2 |
| `df_e07_face_crop` | Full image vs P3 inner-face crop |
| `df_e08_resolution` | 224 vs 384 px |

Metrics for every model on the test split: accuracy, precision, recall, F1, AUC, confusion matrix,
plus AUC on person-seen vs person-unseen test images (ArcFace check, `experiments/person_check/`).
Val/test always exclude `near_dup` images.

### Extras (analysis depth — planned, since time allows)
- `df_e09_robustness` — test under JPEG q50/q75, downscale, blur.
- `df_e10_fairness` — per race / gender / age metrics.
- Grad-CAM on DL models; free swap masks (fake − its real target) to check whether
  heatmaps land on the swapped region.

## Phases → documents

| Phase | Work | Doc | Status |
|---|---|---|---|
| 1 | Cleaning, index, identity split, P1 cache, dataset charts | `DATA_CLEANING.md` | ✅ done |
| 2 | P2–P4 implementation + before/after figures, `df_e01` | `PREPROCESSING.md` | ✅ done |
| 3 | `df_e02` zero-shot, `df_e03` classical (incl. with vs without P3 face features) | `CLASSICAL_AND_ZERO_SHOT.md` | ✅ done |
| 4 | `df_e05a` linear probe → `df_e04`, `df_e05` GenD, `df_e06`–`df_e08` | `DEEP_LEARNING.md` (done 2026-11-01, all 12 runs) | done |
| 5 | Comparison table, confusion matrices, 3 success + 3 failure examples, extras | `EVALUATION.md` (done 2026-11-01: CIs, confusion, fairness, examples, Grad-CAM) | done |
| 5b | Leakage review: every possible leak, status, proposed tests T1–T6 | `LEAKAGE.md` (2026-11-01, tests T1–T5 done) | done |
| 6 | Literature review write-up + slide deck in the template's section order | `LITERATURE_REVIEW.md` (done 2026-10-31, 5 papers), slides | literature done |

Every step, decision, bug and rerun is also recorded chronologically in `EXPERIMENT_LOG.md`.

## Proposed layout

```
deepfake/
  dataset/HiDF/          raw (read-only)
  data/                  index.csv, splits, 512px cache
  src/                   scripts, one per step/experiment
  experiments/df_eXX_*/  config, logs, metrics, checkpoints, figures
  documents/             PLAN.md + one .md per phase
```

## Setup
- Project environment: `deepfake/.venv` (Python 3.12, separate from the shared `myenv-macos`);
  exact versions in `requirements.lock.txt`.
- DINOv3 (`facebook/dinov3-vitl16-pretrain-lvd1689m`) is gated: accept Meta's licence on its
  Hugging Face page and run `hf auth login` once before Phase 4.
- After verifying the extracted folders are complete, the two zips (38 GB) can be deleted.
