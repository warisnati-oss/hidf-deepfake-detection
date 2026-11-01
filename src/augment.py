"""Preprocessing P2: training-time augmentation / noise addition.

Face-swap traces are subtle high-frequency signals that compression, blur or resizing
can wipe out; a model that relies on one exact pixel pattern breaks on re-uploaded
images. Random degradations force it to learn traces that survive them (as in GenD
and SBI training recipes). Applied to the training split only; val/test use
`eval_transform`.

Order follows a real image pipeline: geometry -> colour -> blur -> JPEG -> sensor noise.
"""
import torch
from torchvision.transforms import v2

IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
CLIP = ([0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758])

AUGMENTATIONS = {
    "random resized crop": v2.RandomResizedCrop(512, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
    "horizontal flip": v2.RandomHorizontalFlip(p=1.0),
    "colour jitter": v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
    "Gaussian blur": v2.GaussianBlur(kernel_size=7, sigma=(0.5, 1.5)),
    "JPEG q60-95": v2.JPEG(quality=(60, 95)),
    "Gaussian noise": v2.Compose([v2.ToDtype(torch.float32, scale=True),
                                  v2.GaussianNoise(sigma=0.03, clip=True),
                                  v2.ToDtype(torch.uint8, scale=True)]),
}


def train_transform(size: int = 224, norm=IMAGENET, augment: bool = True) -> v2.Compose:
    if not augment:
        return eval_transform(size, norm)
    return v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomApply([v2.ColorJitter(0.2, 0.2, 0.2, 0.03)], p=0.5),
        v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 1.5))], p=0.2),
        v2.RandomApply([v2.JPEG(quality=(60, 95))], p=0.3),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomApply([v2.GaussianNoise(sigma=0.02, clip=True)], p=0.2),
        v2.Normalize(*norm),
    ])


def eval_transform(size: int = 224, norm=IMAGENET) -> v2.Compose:
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((size, size), antialias=True),  # exact square: P3 crop boxes can be 1 px off square
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(*norm),
    ])
