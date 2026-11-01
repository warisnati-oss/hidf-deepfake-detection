"""Debug DINO-MAC (M3) training on a small subset: each loss term, gradient norm and logit
spread per step, so a run that does not learn or goes NaN shows it within minutes.

Usage: python debug_dinomac.py [bf16|fp32] [lr] [steps] [size_lo size_hi]
"""
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import train
from augment import IMAGENET, train_transform
from evaluation import load_index


def main():
    dtype = sys.argv[1] if len(sys.argv) > 1 else "bf16"
    lr = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-4
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 80
    size_range = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else None
    torch.manual_seed(0)
    df = load_index()
    rows = df[df.split == "train"].sample(16 * (steps + 1), random_state=0)
    dl = DataLoader(train.Faces(rows, "full", train_transform(224, IMAGENET)), batch_size=16, shuffle=True,
                    num_workers=4, drop_last=True)
    m = train.build(train.argparse.Namespace(model="dinov3_lora", backbone="dinov3_l16")).to("mps")
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    for step, (x, y) in enumerate(dl):
        x, y = x.to("mps"), y.to("mps")
        if size_range:
            s = 16 * torch.randint(size_range[0] // 16, size_range[1] // 16 + 1, (1,)).item()
            x = F.interpolate(x, size=(s, s), mode="bicubic", antialias=True)
        with torch.autocast("mps", dtype=torch.bfloat16, enabled=dtype == "bf16"):
            out = m(x)
        main_l = F.binary_cross_entropy_with_logits(out["logit"].float(), y)
        aux = torch.stack([F.binary_cross_entropy_with_logits(a.float(), y) for a in out["aux"]]).mean()
        sc = train.supcon(out["z"].float(), y)
        loss = main_l + 0.5 * aux + 0.1 * sc
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        bad = not torch.isfinite(loss) or not torch.isfinite(gn)
        if step % 10 == 0 or bad:
            print(f"{dtype} lr {lr:g} step {step:3d} main {main_l.item():.3f} aux {aux.item():.3f} supcon {sc.item():.3f} "
                  f"grad {gn.item():.2f} logit_std {out['logit'].float().std().item():.3f}", flush=True)
        if bad or step >= steps:
            break


if __name__ == "__main__":
    main()
