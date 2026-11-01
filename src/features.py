"""Preprocessing P4: high-pass / noise-residual / frequency features for the classical model.

Groups (names prefixed so they can be selected or ablated):
    fft_*    1-D azimuthally averaged log power spectrum (Durall et al. 2020), 128 bins over
             normalised frequency 0..0.5 cycles/px, so 512 and 1024 px inputs are comparable
    res_*    noise-residual statistics after Laplacian + 3 SRM high-pass filters, inside the
             face, outside it, and their ratio (a pasted face carries a different noise)
    lbp_*    uniform LBP histograms (P=8,R=1 and P=16,R=2) inside the face (skin texture)
    col_*    YCbCr + HSV mean/std inside the face vs the skin band around it (colour mismatch)
    seam_*   Sobel gradient magnitude on the face boundary band vs inside / outside (blend seam)

`fft_*` and the whole-image residual stats need no mask; the other groups need the P3 face mask.
"""
import cv2
import numpy as np
from scipy import stats
from skimage.feature import local_binary_pattern

N_FFT_BINS = 128

SRM_KERNELS = {
    "lap": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32),
    "srm1": np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], np.float32),           # 1st order
    "srm2": np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], np.float32) / 2,       # 2nd order
    "srm3": np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
                      [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], np.float32) / 12,  # KV 5x5
}


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


def fft_spectrum(gray: np.ndarray, n_bins: int = N_FFT_BINS) -> np.ndarray:
    """Azimuthal average of the log power spectrum, bins over normalised radius 0..0.5."""
    h, w = gray.shape
    f = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
    power = np.log(np.abs(f) ** 2 + 1e-8)
    yy, xx = np.indices((h, w))
    r = np.hypot((yy - h / 2) / h, (xx - w / 2) / w)  # cycles / px
    idx = np.minimum((r / 0.5 * n_bins).astype(int), n_bins)
    sums = np.bincount(idx.ravel(), power.ravel(), minlength=n_bins + 1)[:n_bins]
    counts = np.bincount(idx.ravel(), minlength=n_bins + 1)[:n_bins]
    spec = sums / np.maximum(counts, 1)
    return spec - spec[0]  # remove overall brightness/contrast scale


def residuals(gray: np.ndarray) -> dict[str, np.ndarray]:
    return {k: cv2.filter2D(gray, cv2.CV_32F, kern) for k, kern in SRM_KERNELS.items()}


def _moments(x: np.ndarray) -> list[float]:
    if x.size < 10:
        return [np.nan] * 4
    return [float(np.std(x)), float(np.mean(np.abs(x))), float(stats.skew(x)), float(stats.kurtosis(x))]


def global_features(rgb: np.ndarray) -> dict[str, float]:
    """Mask-free features (used by the leakage probe and as part of the classical model)."""
    gray = to_gray(rgb)
    out = {f"fft_{i:03d}": v for i, v in enumerate(fft_spectrum(gray))}
    for name, r in residuals(gray).items():
        for stat, v in zip(["std", "mabs", "skew", "kurt"], _moments(r.ravel())):
            out[f"res_{name}_all_{stat}"] = v
    return out


def face_features(rgb: np.ndarray, face: np.ndarray) -> dict[str, float]:
    """Features that compare the face region with its surroundings."""
    gray = to_gray(rgb)
    size = max(int(np.sqrt(face.sum())), 32)
    k = lambda px: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    f8 = face.astype(np.uint8)
    band = max(size // 20, 3)
    outer = cv2.dilate(f8, k(band * 4)).astype(bool) & ~face          # skin/hair ring around face
    seam = (cv2.dilate(f8, k(band)) - cv2.erode(f8, k(band))).astype(bool)
    inner = cv2.erode(f8, k(band)).astype(bool)
    outside = ~cv2.dilate(f8, k(band)).astype(bool)

    out = {}
    for name, r in residuals(gray).items():
        a, b = _moments(r[inner]), _moments(r[outside])
        for stat, va, vb in zip(["std", "mabs", "skew", "kurt"], a, b):
            out[f"res_{name}_face_{stat}"] = va
            out[f"res_{name}_out_{stat}"] = vb
        out[f"res_{name}_ratio_std"] = a[0] / (b[0] + 1e-6)

    g8 = gray.astype(np.uint8)
    for p, rad in [(8, 1), (16, 2)]:
        lbp = local_binary_pattern(g8, p, rad, method="uniform")[inner]
        hist = np.bincount(lbp.astype(int), minlength=p + 2) / max(lbp.size, 1)
        out.update({f"lbp_p{p}_{i:02d}": v for i, v in enumerate(hist)})

    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    for space, img in [("ycc", ycc), ("hsv", hsv)]:
        for c in range(3):
            fi, ou = img[..., c][inner], img[..., c][outer]
            out[f"col_{space}{c}_face_mean"] = float(fi.mean())
            out[f"col_{space}{c}_face_std"] = float(fi.std())
            out[f"col_{space}{c}_ring_mean"] = float(ou.mean()) if ou.size else np.nan
            out[f"col_{space}{c}_ring_std"] = float(ou.std()) if ou.size else np.nan
            out[f"col_{space}{c}_diff_mean"] = out[f"col_{space}{c}_face_mean"] - out[f"col_{space}{c}_ring_mean"]

    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    for region, m in [("seam", seam), ("inner", inner), ("out", outside)]:
        out[f"seam_grad_{region}_mean"] = float(mag[m].mean())
        out[f"seam_grad_{region}_std"] = float(mag[m].std())
    out["seam_grad_ratio_inner"] = out["seam_grad_seam_mean"] / (out["seam_grad_inner_mean"] + 1e-6)
    out["seam_grad_ratio_out"] = out["seam_grad_seam_mean"] / (out["seam_grad_out_mean"] + 1e-6)
    out["face_area"] = float(face.mean())
    return out
