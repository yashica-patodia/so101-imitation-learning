"""Frozen DINOv2 feature extraction (the baseline image encoder)."""

from __future__ import annotations

import numpy as np
import torch

from nano_vla import config as C
from nano_vla.data.video import resize_batch

_MEAN = torch.tensor(C.DINO_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(C.DINO_STD).view(1, 3, 1, 1)


def load_dino(device: torch.device):
    return torch.hub.load("facebookresearch/dinov2", C.DINO_MODEL, verbose=False).eval().to(device)


def preprocess(frames: np.ndarray) -> torch.Tensor:
    """uint8 [N,H,W,3] -> normalized float [N,3,224,224]."""
    return (resize_batch(frames, C.DINO_RES) / 255.0 - _MEAN) / _STD


@torch.no_grad()
def patch_features(model, frames: np.ndarray, device: torch.device, batch: int = 64) -> np.ndarray:
    """uint8 frames [N,H,W,3] -> float16 [N, 64, 384] pooled patch tokens."""
    outs = []
    for i in range(0, len(frames), batch):
        x = preprocess(frames[i : i + batch]).to(device)
        f = model.forward_features(x)["x_norm_patchtokens"]  # [B, 256, 384]
        b, p, d = f.shape
        g = int(round(p**0.5))
        f = f.reshape(b, g // C.DINO_POOL, C.DINO_POOL, g // C.DINO_POOL, C.DINO_POOL, d).mean(dim=(2, 4)).reshape(b, -1, d)
        outs.append(f.to(torch.float16).cpu())
    return torch.cat(outs).numpy()
