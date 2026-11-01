"""Step 9 (preprocessing P3): face regions for every image with the segmenter chosen in step 7
(MediaPipe face oval, dilated 8 px at 512).

Per image we store the face-oval polygon (convex hull points, 512 px coords), so masks can
be rebuilt instantly with `face_mask(...)`, plus a square crop box = bounding box of the
dilated mask + 15 % margin, clipped to the image. Images with no detected face fall back
to the mean polygon over all detections (HiDF faces are aligned, so this is a sane prior).

Output: data/face_regions.npz (polygons), data/face_boxes.csv (crop box, detected flag),
        documents/figures/phase2/p3_pipeline.png
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from face_regions import MediaPipeOval, dilate

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
FIG = PROJECT / "documents" / "figures" / "phase2"
SIZE = 512
DILATION = 8
MARGIN = 0.15
N_POINTS = 36  # face-oval landmarks
WORKERS = 6

_segmenter = None


def _init():
    global _segmenter
    _segmenter = MediaPipeOval()


def _oval(rel: str) -> np.ndarray | None:
    rgb = np.asarray(Image.open(DATA / rel).convert("RGB"))
    res = _segmenter.landmarker.detect(_segmenter._mp_image(image_format=_segmenter._format, data=rgb))
    if not res.face_landmarks:
        return None
    lm = res.face_landmarks[0]
    return np.array([[lm[i].x * SIZE, lm[i].y * SIZE] for i in _segmenter.oval_idx], np.float32)


def face_mask(polygon: np.ndarray, dilation: int = DILATION, size: int = SIZE) -> np.ndarray:
    mask = np.zeros((size, size), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.round(polygon).astype(np.int32)), 1)
    return dilate(mask.astype(bool), dilation)


def crop_box(mask: np.ndarray, margin: float = MARGIN) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) / 2 * (1 + margin)
    half = min(half, SIZE / 2)
    cy, cx = np.clip(cy, half, SIZE - half), np.clip(cx, half, SIZE - half)
    return int(cx - half), int(cy - half), int(cx + half), int(cy + half)


def main():
    df = pd.read_csv(DATA / "index.csv", low_memory=False)
    with ProcessPoolExecutor(WORKERS, initializer=_init) as pool:
        ovals = list(pool.map(_oval, df.cache_path, chunksize=64))

    detected = np.array([o is not None for o in ovals])
    mean_poly = np.mean([o for o in ovals if o is not None], axis=0)
    polys = np.stack([o if o is not None else mean_poly for o in ovals]).astype(np.float32)
    np.savez_compressed(DATA / "face_regions.npz", polygons=polys, detected=detected,
                        cache_path=df.cache_path.to_numpy())

    boxes = [crop_box(face_mask(p)) for p in polys]
    out = pd.DataFrame(boxes, columns=["x0", "y0", "x1", "y1"])
    out.insert(0, "cache_path", df.cache_path)
    out["detected"] = detected
    out["face_frac"] = [face_mask(p).mean() for p in polys]
    out.to_csv(DATA / "face_boxes.csv", index=False)

    print(f"detected {detected.sum()}/{len(df)} ({detected.mean():.4%})")
    print(out.assign(label=df.label).groupby("label")[["detected", "face_frac"]].mean())
    print("crop size px:", (out.x1 - out.x0).describe()[["min", "50%", "max"]].to_dict())
    plot_pipeline(df, polys)


def plot_pipeline(df: pd.DataFrame, polys: np.ndarray):
    seg = MediaPipeOval()
    picks = df[(df.split == "train")].sample(3, random_state=5).index
    titles = ["input (512 px)", "478 landmarks", "face oval + 8 px", "crop → 224 px"]
    fig, axes = plt.subplots(3, 4, figsize=(10, 7.6))
    for row, i in enumerate(picks):
        rgb = np.asarray(Image.open(DATA / df.cache_path[i]).convert("RGB"))
        res = seg.landmarker.detect(seg._mp_image(image_format=seg._format, data=rgb))
        lm = np.array([[p.x * SIZE, p.y * SIZE] for p in res.face_landmarks[0]]) if res.face_landmarks else None
        mask = face_mask(polys[i])
        x0, y0, x1, y1 = crop_box(mask)
        overlay = rgb.copy()
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([235, 104, 52])).astype(np.uint8)
        axes[row, 0].imshow(rgb)
        axes[row, 1].imshow(rgb)
        if lm is not None:
            axes[row, 1].scatter(lm[:, 0], lm[:, 1], s=0.6, c="#1baf7a")
        axes[row, 2].imshow(overlay)
        axes[row, 2].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="white", lw=1.5, ls="--"))
        axes[row, 3].imshow(Image.fromarray(rgb).crop((x0, y0, x1, y1)).resize((224, 224), Image.LANCZOS))
        for ax in axes[row]:
            ax.set_xticks([]), ax.set_yticks([])
    for ax, t in zip(axes[0], titles):
        ax.set_title(t, fontsize=11)
    fig.suptitle("P3: face-region segmentation with MediaPipe", fontweight="bold")
    fig.savefig(FIG / "p3_pipeline.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
