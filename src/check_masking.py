"""Masking check for a trained deep model: does it look at the swapped face or at something else?

The trained model (best.pt of an experiment) scores the test set three ways:
    original           as trained
    face blanked       P3 face mask (+8 px, plus a 24 px safety band) filled with mid-grey -> background only
    background blanked everything outside the P3 face mask filled with mid-grey -> face only
A model that reads the swap should drop to ~0.5 AUC with the face blanked. If it stays high,
it uses a cue outside the face (a shortcut). Blanking adds the same artificial edge to real and
fake images, so it cannot by itself separate the classes.

Usage: python check_masking.py --exp df_e04_effnet_b4
Output: experiments/<exp>/masking_check.csv
"""
import argparse
import json

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

import train
from augment import eval_transform
from evaluation import DATA, eval_rows, load_index
from step9_face_regions import face_mask


class Masked(Dataset):
    def __init__(self, rows, polys, mode, tf, boxes=None):
        self.paths, self.polys, self.mode, self.tf = rows.cache_path.to_list(), polys, mode, tf
        self.y = (rows.label == "fake").to_numpy().astype(np.float32)
        self.boxes = boxes  # crop-view models: P3 box applied after blanking, as in training

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        rgb = np.asarray(Image.open(DATA / self.paths[i]).convert("RGB")).copy()
        m = face_mask(self.polys[i])
        if self.mode == "face_blanked":
            band = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (49, 49))).astype(bool)
            rgb[band] = 128
        elif self.mode == "background_blanked":
            rgb[~m] = 128
        im = Image.fromarray(rgb)
        if self.boxes is not None:
            im = im.crop(tuple(int(v) for v in self.boxes[i]))
        return self.tf(im), self.y[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    a = ap.parse_args()
    out_dir = train.PROJECT / "experiments" / a.exp
    cfg = json.loads((out_dir / "config.json").read_text())
    args = train.argparse.Namespace(**cfg)
    model = train.build(args).to(train.DEVICE)
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=train.DEVICE))

    df = load_index()
    test = eval_rows(df, "test")
    regions = np.load(DATA / "face_regions.npz", allow_pickle=True)
    poly = dict(zip(regions["cache_path"], regions["polygons"]))
    polys = [poly[p] for p in test.cache_path]
    y = (test.label == "fake").to_numpy().astype(int)
    boxes = None
    if cfg.get("view") == "crop":
        b = pd.read_csv(DATA / "face_boxes.csv").set_index("cache_path")
        boxes = b.loc[test.cache_path, ["x0", "y0", "x1", "y1"]].to_numpy()
    rows = []
    for mode in ["original", "face_blanked", "background_blanked"]:
        ds = Masked(test, polys, mode, eval_transform(cfg["size"], model.norm), boxes)
        s = train.predict(model, DataLoader(ds, batch_size=64, num_workers=4), cfg.get("precision", "bf16") == "bf16")
        rows.append({"exp": a.exp, "mode": mode, "test_auc": roc_auc_score(y, s)})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).round(4).to_csv(out_dir / "masking_check.csv", index=False)


if __name__ == "__main__":
    main()
