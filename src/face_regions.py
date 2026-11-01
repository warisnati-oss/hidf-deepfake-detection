"""Face-region segmentation (preprocessing P3) and the free swap mask used to evaluate it.

Segmenters (each returns a boolean face mask at the input image size):
    MediaPipeOval    MediaPipe Face Landmarker, 478 landmarks -> convex hull of the face-oval points
    SegFormerParsing SegFormer face parsing (CelebAMask-HQ, 19 classes) -> union of the face-part classes

Swap mask (fakes only): |fake - real base| averaged over RGB, thresholded, closed, largest
component kept, holes filled. It marks exactly the pixels the face swap changed.
"""
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

PROJECT = Path(__file__).resolve().parents[1]
MODELS = PROJECT / "models"

# SegFormer classes that belong to the swapped face (not hair, ears, neck, hat, cloth)
FACE_CLASSES = ["skin", "nose", "eye_g", "l_eye", "r_eye", "l_brow", "r_brow", "mouth", "u_lip", "l_lip"]


def swap_mask(fake: np.ndarray, base: np.ndarray, thresh: float = 10, close_px: int = 5) -> np.ndarray:
    diff = np.abs(fake.astype(np.float32) - base.astype(np.float32)).mean(2)
    m = diff > thresh
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    m = lab == (np.argmax(sizes) + 1)
    return ndimage.binary_fill_holes(m)


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


class MediaPipeOval:
    name = "mediapipe"

    def __init__(self):
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp_image = __import__("mediapipe").Image
        self._format = __import__("mediapipe").ImageFormat.SRGB
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task")),
            num_faces=1)
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        oval = vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL
        self.oval_idx = sorted({c.start for c in oval} | {c.end for c in oval})

    def __call__(self, rgb: np.ndarray) -> np.ndarray | None:
        h, w = rgb.shape[:2]
        res = self.landmarker.detect(self._mp_image(image_format=self._format, data=np.ascontiguousarray(rgb)))
        if not res.face_landmarks:
            return None
        lm = res.face_landmarks[0]
        pts = np.array([[lm[i].x * w, lm[i].y * h] for i in self.oval_idx], dtype=np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(pts), 1)
        return mask.astype(bool)


class SegFormerParsing:
    name = "segformer"

    def __init__(self, device: str = "mps"):
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        path = MODELS / "segformer-face-parsing"
        self.torch = torch
        self.device = device
        self.proc = SegformerImageProcessor.from_pretrained(path)
        self.model = SegformerForSemanticSegmentation.from_pretrained(path).to(device).eval()
        id2label = self.model.config.id2label
        self.face_ids = [int(i) for i, n in id2label.items() if n in FACE_CLASSES]

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        return self.batch([rgb])[0]

    def batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        torch = self.torch
        h, w = images[0].shape[:2]
        inputs = self.proc(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        logits = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        labels = logits.argmax(1).cpu().numpy()
        return [np.isin(l, self.face_ids) for l in labels]
