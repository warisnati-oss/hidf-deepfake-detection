# Literature review (criterion 2)

Five papers, chosen because each one is the source of one part of our pipeline: the dataset and
its problem (1), the robustness view of evaluation (2), the classical model (3), DL M2 (4) and DL M3 (5).
Citation details checked on 2026-10-31 against the publisher / arXiv pages (links at the end).

| # | Paper | Used in our project as |
|---|---|---|
| 1 | Kang et al., HiDF, KDD 2025 | Dataset and problem statement |
| 2 | Rössler et al., FaceForensics++, ICCV 2019 | Evaluation design: compression robustness, CNN baseline (M1) |
| 3 | Durall et al., Unmasking DeepFakes with simple Features, 2020 | Classical model (P4 features + classifier) |
| 4 | Yermakov et al., GenD, WACV 2026 | DL M2 |
| 5 | Hopf et al., Robust Deepfake Detection, NTIRE 2026 Challenge Report (winner DINO-MAC) | DL M3 |

---

## 1. HiDF: A Human-Indistinguishable Deepfake Dataset

Chaewon Kang, Seoyoon Jeong, Jonghyun Lee, Daejin Choi, Simon S. Woo, Jinyoung Han.
*KDD 2025*, pp. 5527–5538. DOI 10.1145/3711896.3737399.

- **Problem:** popular deepfake datasets (FF++, Celeb-DF, DFDC, …) contain fakes that people can
  easily spot, so detectors trained on them may not transfer to realistic fakes.
- **Contribution:** a dataset of ~62K images and 8K videos of high-quality face swaps
  (`base_donor.jpg`: the donor's face swapped into the base photo), checked in a human study to be
  indistinguishable from real photos.
- **Result:** existing detectors score poorly on HiDF; e.g. EfficientNet-B4 image AUC ≈ 0.70.
- **Limitations we found when using it:**
  - Those detectors were not trained on HiDF's own training data.
  - Real and fake images have different compression histories: real JPEG q75 / PNG, fake JPEG q99.
    A background-only probe reached AUC 0.65, and file size alone gave 0.99.
  - The same identities appear in many images, so a random split leaks people between train and test.
- **In our project:** the dataset. We fix the two shortcuts with P1 (one common compression history)
  and an identity-group split, and remove 51 near-duplicate images from val/test. Then we train on it.

## 2. FaceForensics++: Learning to Detect Manipulated Facial Images

Andreas Rössler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, Matthias Nießner.
*ICCV 2019*.

- **Problem:** there was no large benchmark to train and compare face-manipulation detectors.
- **Contribution:** 1,000 videos manipulated by four methods (Deepfakes, Face2Face, FaceSwap,
  NeuralTextures), released at three compression levels (raw, c23, c40), plus a human study and a
  benchmark of detectors: hand-crafted steganalysis features vs CNNs (XceptionNet best).
- **Results:**
  - Learned CNN detectors beat hand-crafted features and beat humans.
  - Every detector's accuracy drops markedly as compression gets stronger (c23 → c40).
  - Training on the face region beats training on the full frame.
- **In our project:**
  - **Evaluation design:** we test every model under degradations (df_e09: JPEG, blur, resize,
    noise), not only on clean images.
  - **The face-crop adjustment** (df_e07) follows its face-region result.
  - **M1** follows its supervised-CNN approach. M1 uses EfficientNet-B4, the HiDF paper's own
    baseline, in place of XceptionNet.
  - **Classical vs DL:** its hand-crafted-vs-CNN comparison is the same comparison our rubric asks for.

## 3. Unmasking DeepFakes with simple Features

Ricard Durall, Margret Keuper, Franz-Josef Pfreundt, Janis Keuper. arXiv:1911.00686 (v3, 2020).

- **Problem:** deep detectors need large labelled datasets and compute. Is a simple classical
  pipeline enough?
- **Method:**
  1. 2-D DFT of the image.
  2. Azimuthal average, giving a 1-D power spectrum.
  3. A simple classifier (logistic regression / SVM / k-means).

  The idea: generators and up-sampling leave abnormal high-frequency spectra.
- **Results:**
  - 100 % accuracy on high-resolution faces with as few as 20 training samples.
  - CelebA: 100 % supervised, 96 % unsupervised.
  - FF++ video: 91 %.
- **In our project:** the basis of the classical model.
  - Our P4 keeps its FFT azimuthal spectrum.
  - It adds noise residuals (Laplacian/SRM), LBP texture, colour statistics (YCbCr/HSV) and a seam
    feature: 247 features in total, classified with gradient boosting.
  - **Clean test images:** AUC 0.998.
  - **Degraded images:** under blur σ 2, AUC 0.55; under ×0.25 downscaling, 0.45. Hand-crafted
    frequency/texture cues are destroyed by the same degradations FF++ warns about.

## 4. Deepfake Detection that Generalizes Across Benchmarks (GenD)

Andrii Yermakov, Jan Cech, Jiri Matas, Mario Fritz. *WACV 2026*, pp. 773–783. arXiv:2508.06248.

- **Problem:** detectors trained on one dataset fail on others (poor cross-dataset generalisation).
- **Method:**
  - Start from a pre-trained foundation encoder (CLIP, DINOv2/v3, PE).
  - Fine-tune only its LayerNorm parameters (~0.03 % of all weights) plus a linear head on
    L2-normalised features.
  - Train with metric-learning terms (alignment + uniformity) that shape a hyperspherical feature
    space.
- **Result:** best average cross-dataset AUROC among recent methods over 14 deepfake datasets
  (2019–2025).
- **In our project:** **DL M2** = GenD on DINOv3 ViT-L/16. The backbone was chosen by a frozen
  linear probe (df_e05a).
  - **Clean test images:** AUC 0.9998 while training 101K parameters (0.6 % of M1's).
  - **Robustness:** depends entirely on augmentation. Without it, accuracy at JPEG q30 falls from
    96.6 % to 80.5 %.

## 5. Robust Deepfake Detection, NTIRE 2026 Challenge: Report (winning method DINO-MAC)

Benedikt Hopf, Radu Timofte, Chenfan Qu, et al. (57 authors). arXiv:2604.24163, April 2026.

- **Problem:** detectors must stay reliable on degraded images: noise, blur, compression, colour
  filters, overlays, including strengths not seen in training.
- **Challenge:** 337 participants, 57 final submissions. Top methods share three ingredients:
  large foundation models, ensembles and degradation training.
- **Winner (rank 1, team ShallowReal), DINO-MAC:**
  - DINOv3-Large fine-tuned with LoRA (r 32, α 64).
  - Multi-Aspect Classification head: [CLS] + 4 [REG] + mean-patch tokens = 6144-d → 2-layer MLP.
  - BCE + supervised contrastive loss.
  - Deep supervision on the last 4 layers (dropout 0.2), drop path 0.1.
  - Random training resolution 384–1152 px with random interpolation.
- **In our project:** **DL M3** follows this recipe, with one change: the training resolution is
  scaled down to 224–320 px (320–448 px in the df_e08 run) to fit our GPU.
  - Best robustness of all models: with the face crop, AUC ≥ 0.997 under all 9 degradations.
  - Its random resizing acts as partial augmentation. With P2 off, it still keeps 97.7 % accuracy
    at JPEG q30, where M2 drops to 80.5 %.

---

## Gap and how our project addresses it

1. **HiDF is claimed hard, but no one has trained on it with shortcuts controlled.** [1] reports
   ~0.70 AUC for detectors not trained on HiDF.
   - We find two shortcuts: compression history (file size alone gives AUC 0.99) and identity
     overlap.
   - After removing them, every trained model reaches test AUC ≥ 0.998. That includes the classical
     model.
   - Masking checks (face blanked → AUC ≈ 0.52) confirm that the models read the swapped face, not
     a background shortcut.
   - So in-dataset, HiDF fakes are "human-indistinguishable" but not machine-indistinguishable.
2. **The real difference between methods is robustness, as [2] and [5] argue.**
   - On clean images, classical ≈ DL.
   - Under blur, noise and downscaling, the classical model [3] collapses to chance. The DL models
     [4, 5] stay at AUC ≥ 0.99, but only when trained with degradation augmentation.
3. **Recent parameter-efficient foundation-model methods [4, 5] had not been tested on HiDF.** We
   compare them with a fully fine-tuned CNN (M1, the HiDF baseline) under one protocol: same split,
   same preprocessing and the same 9 degradations.

**Limitation to state:** all results are in-dataset (HiDF train → HiDF test). Cross-dataset
generalisation, the main goal of [4], is not tested.

## Sources

- HiDF: <https://pure.ewha.ac.kr/en/publications/hidf-a-human-indistinguishable-deepfake-dataset-2/> · code <https://github.com/DSAIL-SKKU/HiDF>
- FaceForensics++: <https://arxiv.org/abs/1901.08971>
- Durall et al.: <https://arxiv.org/abs/1911.00686>
- GenD: <https://openaccess.thecvf.com/content/WACV2026/html/Yermakov_Deepfake_Detection_that_Generalizes_Across_Benchmarks_WACV_2026_paper.html> · <https://arxiv.org/abs/2508.06248>
- NTIRE 2026 report: <https://arxiv.org/abs/2604.24163>
