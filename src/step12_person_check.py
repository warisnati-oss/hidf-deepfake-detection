"""Step 12: person-level leakage check with face recognition (before Phase 4).

The split keeps image ids apart, but celebrity datasets hold several different photos of the
same person under unrelated ids. Here we embed every real image with ArcFace (InsightFace
buffalo_l, ResNet-50 trained on WebFace600K) and ask: does a val/test real image show a person
who also appears in a train real image?

Pipeline per real image (512 px cache): MediaPipe landmarks -> 5 points (eye centres, nose tip,
mouth corners) -> similarity transform onto the ArcFace 112x112 template -> 512-d embedding.
Same person: cosine similarity >= threshold; reported at 0.3 / 0.4 / 0.5 (ArcFace same-person pairs
typically score > 0.4, different people < 0.2).

Output: data/arcface_real.npy (+ index rows), experiments/person_check/{summary.csv, pairs.csv},
        documents/figures/phase3/person_pairs.png
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.transform import SimilarityTransform

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
MODELS = PROJECT / "models"
OUT = PROJECT / "experiments" / "person_check"
FIG = PROJECT / "documents" / "figures" / "phase3"
WORKERS = 6
THRESHOLDS = [0.3, 0.4, 0.5]

ARCFACE_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                             [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)
IRIS, NOSE_TIP, MOUTH = (468, 473), 1, (61, 291)

_lm = _rec = None


def _init():
    global _lm, _rec
    import onnxruntime as ort
    from face_regions import MediaPipeOval
    cv2.setNumThreads(1)
    _lm = MediaPipeOval()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    _rec = ort.InferenceSession(str(MODELS / "arcface_w600k_r50.onnx"), opts, providers=["CPUExecutionProvider"])


def five_points(rgb: np.ndarray) -> np.ndarray | None:
    res = _lm.landmarker.detect(_lm._mp_image(image_format=_lm._format, data=rgb))
    if not res.face_landmarks:
        return None
    lm = res.face_landmarks[0]
    h, w = rgb.shape[:2]
    p = lambda i: [lm[i].x * w, lm[i].y * h]
    eyes = sorted([p(i) for i in IRIS])      # image-left first, as in the template
    mouth = sorted([p(i) for i in MOUTH])
    return np.array([eyes[0], eyes[1], p(NOSE_TIP), mouth[0], mouth[1]], np.float32)


def embed(path: str) -> np.ndarray:
    rgb = np.asarray(Image.open(DATA / path).convert("RGB"))
    pts = five_points(rgb)
    if pts is None:
        return np.full(512, np.nan, np.float32)
    tf = SimilarityTransform.from_estimate(pts, ARCFACE_TEMPLATE)
    face = cv2.warpAffine(rgb, tf.params[:2], (112, 112), borderValue=0)
    x = ((face.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
    e = _rec.run(None, {_rec.get_inputs()[0].name: x})[0][0]
    return e / np.linalg.norm(e)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "index.csv", low_memory=False)
    real = df[df.label == "real"]
    emb_file = DATA / "arcface_real.npy"
    if emb_file.exists():
        E = np.load(emb_file)
    else:
        with ProcessPoolExecutor(WORKERS, initializer=_init) as pool:
            E = np.stack(list(pool.map(embed, real.cache_path, chunksize=64)))
        np.save(emb_file, E)
    ok = ~np.isnan(E).any(1)
    print(f"embedded {ok.sum()}/{len(E)} real images", flush=True)

    split = real.split.to_numpy()
    Et = torch.from_numpy(np.nan_to_num(E))
    train = np.flatnonzero((split == "train") & ok)
    rows, summary = [], []
    for s in ["val", "test"]:
        q = np.flatnonzero((split == s) & ok & ~real.near_dup.to_numpy())
        best, arg = [], []
        for i in range(0, len(q), 1024):
            sim = Et[q[i:i + 1024]] @ Et[train].T
            v, j = sim.max(1)
            best.append(v.numpy())
            arg.append(j.numpy())
        best, arg = np.concatenate(best), np.concatenate(arg)
        for t in THRESHOLDS:
            hit = best >= t
            # share of the split's identity groups that contain at least one such image
            groups = real.group_id.to_numpy()[q]
            g_share = pd.Series(hit).groupby(groups).any().mean()
            summary.append({"split": s, "threshold": t, "real_images": len(q), "same_person_in_train": int(hit.sum()),
                            "share_images": hit.mean(), "share_groups": g_share})
        rows.append(pd.DataFrame({"split": s, "query": real.cache_path.to_numpy()[q],
                                  "train_match": real.cache_path.to_numpy()[train[arg]], "sim": best}))
        print(f"{s}: max-sim percentiles 5/25/50/75/95 =", np.percentile(best, [5, 25, 50, 75, 95]).round(3), flush=True)

    # reference: within-train nearest neighbour (excluding self) on a sample
    rng = np.random.default_rng(0)
    smp = rng.choice(train, 3000, replace=False)
    sim = Et[smp] @ Et[train].T
    sim[torch.arange(len(smp)), torch.from_numpy(np.searchsorted(train, smp))] = -1
    print("train->train NN sim percentiles 5/25/50/75/95 =",
          np.percentile(sim.max(1).values.numpy(), [5, 25, 50, 75, 95]).round(3))

    pairs = pd.concat(rows)
    pairs.to_csv(OUT / "pairs.csv", index=False)
    summary = pd.DataFrame(summary).round(4)
    summary.to_csv(OUT / "summary.csv", index=False)
    print(summary.to_string(index=False))
    plot_pairs(pairs[pairs.split == "test"])


def plot_pairs(p: pd.DataFrame):
    import matplotlib.pyplot as plt
    bins = [(0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.0)]
    fig, axes = plt.subplots(2, len(bins), figsize=(2.6 * len(bins), 5.6))
    for j, (lo, hi) in enumerate(bins):
        sel = p[(p.sim >= lo) & (p.sim < hi)]
        if not len(sel):
            continue
        r = sel.sample(1, random_state=j).iloc[0]
        for i, path in enumerate([r["query"], r.train_match]):
            axes[i, j].imshow(Image.open(DATA / path).resize((200, 200)))
            axes[i, j].set_xticks([]), axes[i, j].set_yticks([])
        axes[0, j].set_title(f"sim {r.sim:.2f}\n(bin {lo}-{hi})", fontsize=10)
    axes[0, 0].set_ylabel("test", fontsize=11)
    axes[1, 0].set_ylabel("nearest train", fontsize=11)
    fig.suptitle("ArcFace: test real image vs most similar train real image", fontweight="bold")
    fig.savefig(FIG / "person_pairs.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
