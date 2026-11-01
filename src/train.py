"""Phase 4 trainer shared by all deep models, so data, schedule and evaluation are identical.

Models
    effnet_b4    DL M1 baseline: EfficientNet-B4 (Noisy Student, timm), fully fine-tuned, BCE.
    gend         DL M2: GenD (Yermakov et al., WACV 2026). Frozen ViT-L backbone, only LayerNorm
                 parameters + a linear head on the L2-normalised CLS token are trained.
                 Loss = BCE + 0.1 * alignment + 0.5 * uniformity (paper defaults for CLIP).
    dinov3_lora  DL M3: DINO-MAC style (NTIRE 2026 robust-detection winner). DINOv3 ViT-L/16 with
                 LoRA (r=32, alpha=64) on q/k/v/o projections, drop path 0.1, MAC head
                 (CLS + 4 register + mean patch tokens = 6144-d -> MLP), deep supervision on the
                 last 4 layers (dropout 0.2), supervised contrastive auxiliary loss.
                 Paper trains at random 384-1152 px; here random sizes in --size-range (M6 budget).

Data: P1 cache (512 px), optional P3 face crop (--view crop), P2 augmentation (--no-augment to
disable), train split only; val (minus near_dup) picks the best epoch and the threshold; test is
scored once at the end with evaluation.report.

Usage: python train.py --exp df_e04_effnet_b4 --model effnet_b4 [--view full] [--size 224] ...
Output: experiments/<exp>/{config.json, history.csv, best.pt, results.json, scores_val.csv, scores_test.csv}
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from augment import CLIP, IMAGENET, eval_transform, train_transform
from degrade import CONDITIONS, degrade
from evaluation import DATA, best_threshold, eval_rows, load_index, report

PROJECT = Path(__file__).resolve().parents[1]
DEVICE = "mps"


# ---------------------------------------------------------------- data
class Faces(Dataset):
    def __init__(self, rows: pd.DataFrame, view: str, tf, condition: str = "clean"):
        self.paths = rows.cache_path.to_list()
        self.y = (rows.label == "fake").to_numpy().astype(np.float32)
        self.tf = tf
        self.condition = condition  # df_e09 test-time degradation (degrade.CONDITIONS)
        self.boxes = None
        if view == "crop":
            boxes = pd.read_csv(DATA / "face_boxes.csv").set_index("cache_path")
            self.boxes = boxes.loc[rows.cache_path, ["x0", "y0", "x1", "y1"]].to_numpy()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(DATA / self.paths[i]).convert("RGB")
        if self.condition != "clean":
            im = Image.fromarray(degrade(np.asarray(im), self.condition, key=self.paths[i]))
        if self.boxes is not None:  # crop box from the clean image's P3 mask
            im = im.crop(tuple(int(v) for v in self.boxes[i]))
        return self.tf(im), self.y[i]


# ---------------------------------------------------------------- models
class EffNet(nn.Module):
    norm = IMAGENET

    def __init__(self):
        super().__init__()
        import timm
        self.net = timm.create_model("tf_efficientnet_b4.ns_jft_in1k", pretrained=True, num_classes=1, drop_rate=0.3)

    def forward(self, x):
        return {"logit": self.net(x).squeeze(1)}

    def loss(self, out, y):
        return F.binary_cross_entropy_with_logits(out["logit"], y)


def align_uniform(z: torch.Tensor, y: torch.Tensor, t: float = 2.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Wang & Isola (2020) on the unit hypersphere; alignment over same-class pairs."""
    d2 = torch.cdist(z, z).pow(2)
    same = (y[:, None] == y[None]) & ~torch.eye(len(y), dtype=torch.bool, device=z.device)
    align = d2[same].mean() if same.any() else z.new_zeros(())
    off = ~torch.eye(len(y), dtype=torch.bool, device=z.device)
    uniform = torch.log(torch.exp(-t * d2[off]).mean())
    return align, uniform


class GenD(nn.Module):
    def __init__(self, backbone: str):
        super().__init__()
        self.backbone = backbone
        if backbone == "clip_l14":
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
            self.enc = model.visual
            self.enc.proj = None  # use the CLS token before the text projection
            self.norm, dim = CLIP, 1024
        elif backbone == "dinov3_l16":
            from transformers import AutoModel
            self.enc = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
            self.norm, dim = IMAGENET, 1024
        else:
            raise ValueError(backbone)
        for name, p in self.enc.named_parameters():
            p.requires_grad = any(k in name.lower() for k in ["ln_", "norm", "layernorm"])
        self.head = nn.Linear(dim, 1)

    def features(self, x):
        if self.backbone == "clip_l14":
            return self.enc(x)
        return self.enc(pixel_values=x).last_hidden_state[:, 0]

    def forward(self, x):
        z = F.normalize(self.features(x), dim=-1)
        return {"logit": self.head(z).squeeze(1), "z": z}

    def loss(self, out, y):
        align, uniform = align_uniform(out["z"].float(), y)
        return F.binary_cross_entropy_with_logits(out["logit"], y) + 0.1 * align + 0.5 * uniform


def supcon(z: torch.Tensor, y: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al. 2020) on normalised features."""
    sim = z @ z.T / tau
    eye = torch.eye(len(y), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e4)
    pos = (y[:, None] == y[None]) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -(log_prob * pos).sum(1).div(pos.sum(1).clamp(min=1)).mean()


class DinoMAC(nn.Module):
    norm = IMAGENET

    def __init__(self, rank: int = 32, alpha: int = 64, hidden: int = 1024):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel
        enc = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m", drop_path_rate=0.1)
        cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        self.enc = get_peft_model(enc, cfg)
        self.n_reg = enc.config.num_register_tokens
        d = enc.config.hidden_size * (2 + self.n_reg)  # CLS + registers + mean patch
        self.head = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        # intermediate ViT activations are un-normalised (magnitudes in the hundreds) -> LayerNorm first
        self.aux = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.2), nn.Linear(d, 1)) for _ in range(4)])

    def mac(self, h):
        return torch.cat([h[:, 0], h[:, 1:1 + self.n_reg].flatten(1), h[:, 1 + self.n_reg:].mean(1)], 1)

    def forward(self, x):
        out = self.enc(pixel_values=x, output_hidden_states=True)
        hs = list(out.hidden_states[-4:-1]) + [out.last_hidden_state]  # final layer: after the output norm
        feats = [self.mac(h) for h in hs]
        return {"logit": self.head(feats[-1]).squeeze(1),
                "aux": [a(f).squeeze(1) for a, f in zip(self.aux, feats)],
                "z": F.normalize(feats[-1], dim=-1)}

    def loss(self, out, y):
        main = F.binary_cross_entropy_with_logits(out["logit"], y)
        aux = torch.stack([F.binary_cross_entropy_with_logits(a, y) for a in out["aux"]]).mean()
        return main + 0.5 * aux + 0.1 * supcon(out["z"].float(), y)


def build(args) -> nn.Module:
    if args.model == "effnet_b4":
        return EffNet()
    if args.model == "gend":
        return GenD(args.backbone)
    if args.model == "dinov3_lora":
        return DinoMAC()
    raise ValueError(args.model)


# ---------------------------------------------------------------- training
def predict(model, loader, bf16: bool = True) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=bf16):
        for x, _ in loader:
            out.append(model(x.to(DEVICE))["logit"].float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model", required=True, choices=["effnet_b4", "gend", "dinov3_lora"])
    ap.add_argument("--backbone", default="dinov3_l16", help="gend only: clip_l14 | dinov3_l16")
    ap.add_argument("--view", default="full", choices=["full", "crop"])
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--size-range", type=int, nargs=2, default=None, help="random train size per batch (multiples of 16)")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--epochs", type=int, default=10, help="maximum epochs")
    ap.add_argument("--patience", type=int, default=2, help="stop after this many epochs without val AUC gain")
    ap.add_argument("--no-robustness", action="store_true", help="skip the df_e09 degraded-test evaluation")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--warmup", type=float, default=1.0, help="warm-up epochs")
    ap.add_argument("--train-limit", type=int, default=None, help="debug: subsample train")
    ap.add_argument("--eval-limit", type=int, default=None, help="debug: subsample val and test")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training: evaluate best.pt + history.csv of a run stopped early by hand")
    ap.add_argument("--precision", default=None, choices=["bf16", "fp32"], help="default: fp32 for dinov3_lora, else bf16")
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="leakage control: permute the train labels; a leak-free pipeline must then give AUC ~0.5")
    args = ap.parse_args()

    defaults = {"effnet_b4": (2e-4, 1e-2), "gend": (3e-4, 0.0), "dinov3_lora": (1e-4, 1e-2)}
    lr, wd = defaults[args.model]
    args.lr = args.lr or lr
    args.weight_decay = wd if args.weight_decay is None else args.weight_decay
    # DINO-MAC hit scattered non-finite steps in bf16 even with fp32 losses (df_e05b, 2026-10-27) -> fp32;
    # same speed as bf16 on this GPU (bench_speed.py)
    args.precision = args.precision or ("fp32" if args.model == "dinov3_lora" else "bf16")

    out_dir = PROJECT / "experiments" / args.exp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)

    df = load_index()
    train_rows = df[df.split == "train"]
    if args.train_limit:
        train_rows = train_rows.sample(args.train_limit, random_state=args.seed)
    if args.shuffle_labels:  # df_e12 T5 leakage control: val / test keep their true labels
        train_rows = train_rows.assign(label=np.random.default_rng(args.seed).permutation(train_rows.label.to_numpy()))
    val_rows, test_rows = eval_rows(df, "val"), eval_rows(df, "test")
    if args.eval_limit:
        val_rows = val_rows.sample(args.eval_limit, random_state=args.seed)
        test_rows = test_rows.sample(args.eval_limit, random_state=args.seed)

    model = build(args).to(DEVICE)
    norm = model.norm
    bf16 = args.precision == "bf16"
    max_size = args.size_range[1] if args.size_range else args.size
    train_ds = Faces(train_rows, args.view, train_transform(max_size, norm, augment=not args.no_augment))
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        drop_last=True, persistent_workers=True)
    eval_loader = lambda rows: DataLoader(Faces(rows, args.view, eval_transform(args.size, norm)),
                                          batch_size=args.batch * 2, num_workers=args.workers)
    val_loader, test_loader = eval_loader(val_rows), eval_loader(test_rows)

    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"{args.exp}: trainable {n_train:,} / {n_all:,} params ({n_train / n_all:.2%})", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps, warm = args.epochs * len(loader), int(args.warmup * len(loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    history, best_auc = [], -1.0
    if args.eval_only:  # df_e08_res_dinomac: capped at 2 epochs (~13 h/epoch), 2026-10-30
        history = pd.read_csv(out_dir / "history.csv").to_dict("records")
        best_auc = max(h["val_auc"] for h in history)
    y_val = (val_rows.label == "fake").to_numpy().astype(int)
    for epoch in range(0 if args.eval_only else args.epochs):
        model.train()
        t0, losses, bad_steps = time.time(), [], 0
        for k, (x, y) in enumerate(loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            if args.size_range:
                s = 16 * random.randint(args.size_range[0] // 16, args.size_range[1] // 16)
                # random interpolation as in DINO-MAC; "area" is left out: MPS can't do it for uneven size ratios
                x = F.interpolate(x, size=(s, s), mode=random.choice(["bilinear", "bicubic"]), antialias=True)
            with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                out = model(x)
            # losses in fp32: bf16 similarity matrices / logits made DINO-MAC go NaN (df_e05b, 2026-10-27)
            out = {k: [t.float() for t in v] if isinstance(v, list) else v.float() for k, v in out.items()}
            loss = model.loss(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not (torch.isfinite(loss) and torch.isfinite(grad_norm)):
                bad_steps += 1  # skip the update; stop the run if it keeps happening
                finite = {n: bool(torch.isfinite(torch.stack(v) if isinstance(v, list) else v).all()) for n, v in out.items()}
                print(f"  non-finite step {k}: size {x.shape[-1]} loss {loss.item():.4g} grad {grad_norm.item():.4g} "
                      f"outputs finite {finite}", flush=True)
                if bad_steps > 20:
                    raise RuntimeError(f"more than 20 non-finite steps in epoch {epoch} (last at step {k})")
                sched.step()
                continue
            opt.step()
            sched.step()
            losses.append(loss.item())
            if k % 200 == 0:
                print(f"  epoch {epoch} step {k}/{len(loader)} loss {np.mean(losses[-200:]):.4f} "
                      f"({(k + 1) * args.batch / (time.time() - t0):.0f} img/s)", flush=True)
        s_val = predict(model, val_loader, bf16)
        auc = roc_auc_score(y_val, s_val)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_auc": auc,
                        "minutes": (time.time() - t0) / 60})
        print(f"epoch {epoch}: loss {history[-1]['train_loss']:.4f} val_auc {auc:.4f} "
              f"({history[-1]['minutes']:.1f} min)", flush=True)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        if auc > best_auc:
            best_auc = auc
            torch.save({k: v for k, v in model.state_dict().items()}, out_dir / "best.pt")
        elif epoch - int(np.argmax([h["val_auc"] for h in history])) >= args.patience:
            print(f"early stop: no val AUC gain for {args.patience} epochs", flush=True)
            break

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=DEVICE))
    s_val, s_test = predict(model, val_loader, bf16), predict(model, test_loader, bf16)
    thr = best_threshold(y_val, s_val)
    res = {"exp": args.exp, "model": args.model, "view": args.view, "size": args.size,
           "augment": not args.no_augment, "best_epoch": int(np.argmax([h["val_auc"] for h in history])),
           "val_auc": best_auc, "threshold": thr, "trainable_params": n_train, **report(test_rows, s_test, thr)}
    (out_dir / "results.json").write_text(json.dumps(res, indent=2))
    pd.DataFrame({"cache_path": val_rows.cache_path.to_numpy(), "label": val_rows.label.to_numpy(), "score": s_val}).to_csv(out_dir / "scores_val.csv", index=False)
    pd.DataFrame({"cache_path": test_rows.cache_path.to_numpy(), "label": test_rows.label.to_numpy(), "score": s_test}).to_csv(out_dir / "scores_test.csv", index=False)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}), flush=True)

    # df_e09: same model and threshold on degraded test images
    if not args.no_robustness:
        rob = []
        for cond in CONDITIONS:
            s = s_test if cond == "clean" else predict(model, DataLoader(
                Faces(test_rows, args.view, eval_transform(args.size, norm), condition=cond),
                batch_size=args.batch * 2, num_workers=args.workers), bf16)
            rob.append({"model": args.exp, "condition": cond, "threshold": thr, **report(test_rows, s, thr)})
            print(f"robustness {cond}: auc {rob[-1]['test_auc']:.4f} acc {rob[-1]['test_accuracy']:.4f}", flush=True)
        e09 = PROJECT / "experiments" / "df_e09_robustness"
        e09.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rob).round(4).to_csv(e09 / f"results_{args.exp}.csv", index=False)
    print("RUN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
