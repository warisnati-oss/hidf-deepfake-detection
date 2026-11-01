"""Test-time degradations for the robustness experiment (df_e09), shared by every model.

Applied to the 512 px P1 cache image (uint8 RGB), each at a fixed severity, deterministic per image:
    jpeg_q    re-encode as JPEG quality q (4:2:0), as social media uploads do
    blur_s    Gaussian blur, sigma s pixels
    resize_f  downscale by f (area) and upscale back to 512 (bicubic): lost resolution
    noise_s   additive Gaussian noise, sigma s on the 0-255 scale (seeded by the file name)
"""
import io
import zlib

import cv2
import numpy as np
from PIL import Image

CONDITIONS = {
    "clean": None,
    "jpeg_75": ("jpeg", 75), "jpeg_50": ("jpeg", 50), "jpeg_30": ("jpeg", 30),
    "blur_1": ("blur", 1.0), "blur_2": ("blur", 2.0),
    "resize_0.5": ("resize", 0.5), "resize_0.25": ("resize", 0.25),
    "noise_5": ("noise", 5.0), "noise_10": ("noise", 10.0),
}


def degrade(rgb: np.ndarray, condition: str, key: str = "") -> np.ndarray:
    spec = CONDITIONS[condition]
    if spec is None:
        return rgb
    kind, v = spec
    if kind == "jpeg":
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, "JPEG", quality=int(v), subsampling=2)
        return np.asarray(Image.open(buf).convert("RGB"))
    if kind == "blur":
        return cv2.GaussianBlur(rgb, (0, 0), sigmaX=v)
    if kind == "resize":
        h, w = rgb.shape[:2]
        small = cv2.resize(rgb, (int(w * v), int(h * v)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    if kind == "noise":
        rng = np.random.default_rng(zlib.crc32(key.encode()))
        return np.clip(rgb.astype(np.float32) + rng.normal(0, v, rgb.shape), 0, 255).astype(np.uint8)
    raise ValueError(condition)
