"""df_e12 T1 (leakage B2): "compression-only fakes" — do the models detect the swap, or the
processing history of the face?

A HiDF fake's face is newly generated pixels saved once as JPEG q99 (4:2:0); a real `c*` face
carries the camera photo's earlier JPEG q75 compression. P1 re-encodes everything, but some
difference in history could survive inside the face, where the masking check can't see it.

Here we take REAL test photos, change nothing about who is in them, give them fake-like
processing, then run the normal P1 cache step and score them with the trained models:
    original        raw file -> P1 (sanity: must reproduce the cached image and its score)
    q99             raw -> JPEG q99 4:2:0 (the fakes' save step) -> P1
    face_resample   face region re-sampled (Lanczos +0.5 px / -0.5 px shift: wipes the old JPEG
                    grid, keeps the content), feathered back into the photo -> q99 -> P1
    face_noise      face region + Gaussian noise sigma 1 grey level (also wipes the old JPEG
                    quantisation, without smoothing) -> q99 -> P1
    whole_resample  control: the whole image re-sampled -> q99 -> P1 (fresh pixels everywhere,
                    no face/background mismatch)
    face_blur       smoothness probe (not compression): face region Gaussian blur sigma 1 -> q99 -> P1
If the models flag face_resample / face_noise, they rely on the face's processing history.

Images: 1,000 `c*` (camera JPEG) + 1,000 `f*` (PNG) real test photos, seed 0.
Usage: python df_e12_compression.py build   (CPU)   then   python df_e12_compression.py score   (GPU)
Output: data/df_e12_compression/<variant>/<stem>.jpg, experiments/df_e12_leakage/compression.csv
"""
import io
import json
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image

import df_e10_evaluation as E
from evaluation import DATA, best_threshold, eval_rows, load_index
from step5_build_cache import DATASET, HISTORY_QUALITY, QUALITY, SIZE
from step9_face_regions import face_mask

OUT = E.EXP / "df_e12_leakage"
IMG = DATA / "df_e12_compression"
VARIANTS = ["original", "q99", "face_resample", "face_noise", "whole_resample", "face_blur"]
N_PER_SOURCE = 1000


def p1(rgb: np.ndarray) -> Image.Image:
    """The P1 cache step of step5_build_cache.convert, on an RGB array."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=HISTORY_QUALITY, subsampling=2)
    with Image.open(buf) as im:
        return im.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)


def jpeg(rgb: np.ndarray, q: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=q, subsampling=2)
    return np.asarray(Image.open(buf).convert("RGB"))


def shift(x: np.ndarray, d: float) -> np.ndarray:
    m = np.float32([[1, 0, d], [0, 1, d]])
    return cv2.warpAffine(x, m, (x.shape[1], x.shape[0]), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)


def variant(rgb: np.ndarray, alpha: np.ndarray, name: str, rng) -> np.ndarray:
    x = rgb.astype(np.float32)
    if name == "original":
        return rgb
    if name == "q99":
        return jpeg(rgb, 99)
    if name == "whole_resample":
        return jpeg(np.clip(shift(shift(x, 0.5), -0.5), 0, 255).round().astype(np.uint8), 99)
    if name == "face_resample":
        new = shift(shift(x, 0.5), -0.5)
    elif name == "face_noise":
        new = x + rng.normal(0, 1.0, x.shape)
    elif name == "face_blur":
        new = cv2.GaussianBlur(x, (0, 0), 1.0)
    out = alpha[..., None] * new + (1 - alpha[..., None]) * x
    return jpeg(np.clip(out, 0, 255).round().astype(np.uint8), 99)


def build_one(job):
    cv2.setNumThreads(1)
    file_path, stem, polygon = job
    with Image.open(DATASET / file_path) as im:
        rgb = np.asarray(im.convert("RGB"))
    # face mask from the 512 px P3 polygon, scaled to 1024 px and feathered like a blend
    m = face_mask(polygon * 2, size=rgb.shape[0]).astype(np.float32)
    alpha = cv2.GaussianBlur(m, (0, 0), 6)
    rng = np.random.default_rng(zlib.crc32(stem.encode()))
    for name in VARIANTS:
        p1(variant(rgb, alpha, name, rng)).save(IMG / name / f"{stem}.jpg", "JPEG", quality=QUALITY, subsampling=0)


def sample() -> pd.DataFrame:
    test = eval_rows(load_index(), "test")
    real = test[test.label == "real"]
    return pd.concat([real[real.prefix == p].sample(N_PER_SOURCE, random_state=0) for p in ["c", "f"]])


def build():
    rows = sample()
    for name in VARIANTS:
        (IMG / name).mkdir(parents=True, exist_ok=True)
    reg = np.load(DATA / "face_regions.npz", allow_pickle=True)
    poly = dict(zip(reg["cache_path"], reg["polygons"]))
    stems = rows.cache_path.str.split("/").str[-1].str[:-4]
    jobs = [(f, s, np.asarray(poly[c], np.float32)) for f, s, c in zip(rows.file_path, stems, rows.cache_path)]
    with ProcessPoolExecutor(6) as pool:
        list(pool.map(build_one, jobs, chunksize=16))
    # sanity: "original" must reproduce the P1 cache
    diffs = [np.abs(np.asarray(Image.open(IMG / "original" / f"{s}.jpg"), np.int16)
                    - np.asarray(Image.open(DATA / c), np.int16)).max() for s, c in zip(stems[:50], rows.cache_path[:50])]
    print(f"built {len(rows)} x {len(VARIANTS)}; original vs cache max pixel diff over 50 images: {max(diffs)}")


def score():
    import torch
    from torch.utils.data import DataLoader

    import train
    from augment import eval_transform

    rows = sample()
    stems = rows.cache_path.str.split("/").str[-1].str[:-4].to_numpy()
    prefix = rows.prefix.to_numpy()
    test = eval_rows(load_index(), "test").reset_index(drop=True)
    ref = E.load_models(test)
    y = (test.label == "fake").to_numpy()
    out = []

    def record(model, name, s, thr, ref_s):
        r_med, f_med = np.median(ref_s[~y]), np.median(ref_s[y])
        for src in ["c", "f", "all"]:
            m = np.ones(len(s), bool) if src == "all" else prefix == src
            out.append({"model": model, "variant": name, "source": src, "n": int(m.sum()),
                        "called_fake_pct": 100 * (s[m] >= thr).mean(), "median_score": float(np.median(s[m])),
                        "position_real0_fake1": float((np.median(s[m]) - r_med) / (f_med - r_med))})

    # classical
    feats = pd.read_parquet(DATA / "features_p4.parquet")
    df = load_index()
    cols, med = list(feats.columns), feats[df.split == "train"].median()
    clf = joblib.load(E.EXP / "df_e09_robustness" / "classical_hgb.joblib")
    val = eval_rows(df, "val")
    thr = best_threshold((val.label == "fake").to_numpy().astype(int),
                         clf.predict_proba(feats.loc[val.index].fillna(med).to_numpy(np.float32))[:, 1])
    ref_cl = clf.predict_proba(feats.loc[eval_rows(df, "test").index].fillna(med).to_numpy(np.float32))[:, 1]
    reg = np.load(DATA / "face_regions.npz", allow_pickle=True)
    poly = dict(zip(reg["cache_path"], reg["polygons"]))
    for name in VARIANTS:
        f_file = OUT / f"compression_features_{name}.parquet"
        if not f_file.exists():
            paths = [f"df_e12_compression/{name}/{s}.jpg" for s in stems]
            with ProcessPoolExecutor(6) as pool:
                X = pd.DataFrame(list(pool.map(extract_with_poly, [(p, poly[c]) for p, c in zip(paths, rows.cache_path)], chunksize=16)))
            X.to_parquet(f_file)
        X = pd.read_parquet(f_file)
        record("Classical", name, clf.predict_proba(X[cols].fillna(med).to_numpy(np.float32))[:, 1], thr, ref_cl)
        print("Classical", name, flush=True)

    # deep models
    for exp, model_name in [("df_e04_effnet_b4", "M1 EfficientNet-B4"), ("df_e05_gend", "M2 GenD (DINOv3)"),
                            ("df_e05b_dinov3_lora", "M3 DINOv3 + LoRA")]:
        cfg = json.loads((E.EXP / exp / "config.json").read_text())
        model = train.build(train.argparse.Namespace(**cfg)).to(train.DEVICE)
        model.load_state_dict(torch.load(E.EXP / exp / "best.pt", map_location=train.DEVICE))
        thr = json.load(open(E.EXP / exp / "results.json"))["threshold"]
        tf = eval_transform(cfg["size"], model.norm)
        for name in VARIANTS:
            ds = train.Faces(pd.DataFrame({"cache_path": [f"df_e12_compression/{name}/{s}.jpg" for s in stems],
                                           "label": "real"}), "full", tf)
            s = train.predict(model, DataLoader(ds, batch_size=32, num_workers=4), cfg.get("precision", "bf16") == "bf16")
            record(model_name, name, s, thr, ref[model_name][0])
            print(model_name, name, flush=True)
        del model
        torch.mps.empty_cache()

    res = pd.DataFrame(out)
    res.to_csv(OUT / "compression.csv", index=False)
    show = res[res.source == "all"].pivot(index="variant", columns="model", values="called_fake_pct").loc[VARIANTS]
    print("\n% of real photos called fake (all 2,000):\n" + show.round(2).to_string())
    print("\nposition between real (0) and fake (1) medians:\n" +
          res[res.source == "all"].pivot(index="variant", columns="model", values="position_real0_fake1").loc[VARIANTS].round(3).to_string())


def extract_with_poly(job):
    """P4 features with the image's own P3 polygon (content is unchanged, so the face is too)."""
    from features import face_features, global_features
    cv2.setNumThreads(1)
    path, polygon = job
    with Image.open(DATA / path) as im:
        rgb = np.asarray(im.convert("RGB"))
    f = global_features(rgb)
    f.update(face_features(rgb, face_mask(polygon)))
    return f


if __name__ == "__main__":
    {"build": build, "score": score}[sys.argv[1]]()
