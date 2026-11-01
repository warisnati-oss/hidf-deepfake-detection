"""Training-speed check on random data: is the GPU free, and how fast does a model train?

Usage: python bench_speed.py effnet_b4 [bf16|fp16|fp32] [size] [batch]
Reference on a free M6 GPU (2026-10-26, bf16, 224 px, batch 32): effnet_b4 ~42 img/s,
gend ~18.5 img/s, dinov3_lora ~16 img/s.
"""
import sys
import time

import torch

import train


def main():
    model = sys.argv[1]
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[sys.argv[2] if len(sys.argv) > 2 else "bf16"]
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 224
    bs = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    m = train.build(train.argparse.Namespace(model=model, backbone="dinov3_l16")).to("mps")
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, 1e-4)
    x = torch.randn(bs, 3, size, size, device="mps")
    y = (torch.rand(bs, device="mps") > 0.5).float()
    for i in range(6):
        if i == 2:
            torch.mps.synchronize()
            t = time.time()
        with torch.autocast("mps", dtype=dtype or torch.float32, enabled=dtype is not None):
            loss = m.loss(m(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.mps.synchronize()
    print(f"{model} {sys.argv[2] if len(sys.argv) > 2 else 'bf16'} {size}px bs{bs}: {4 * bs / (time.time() - t):.1f} img/s  "
          f"loss {loss.item():.3f}  mem {torch.mps.driver_allocated_memory() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
