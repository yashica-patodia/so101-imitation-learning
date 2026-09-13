"""Video decoding (PyAV) and frame resizing."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F


def decode_at(path: str | Path, wanted: list[float], tol: float = 0.02) -> np.ndarray:
    """Decode `path` once, sequentially, and return uint8 frames [N, H, W, 3] nearest to
    each wanted timestamp (seconds in the file's own clock). Sequential decode is much
    faster than seeking for AV1."""
    out: list[np.ndarray | None] = [None] * len(wanted)
    order = np.argsort(wanted)
    targets = [wanted[i] for i in order]
    j = 0
    last = None
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        tb = float(stream.time_base)
        for frame in c.decode(stream):
            ts = frame.pts * tb
            while j < len(targets) and ts >= targets[j] - tol:
                cand = frame if last is None or abs(ts - targets[j]) <= abs(last[0] - targets[j]) else last[1]
                out[order[j]] = cand.to_ndarray(format="rgb24")
                j += 1
            last = (ts, frame)
            if j >= len(targets):
                break
    if last is None:
        raise RuntimeError(f"no frames decoded from {path}")
    for k in range(len(out)):
        if out[k] is None:  # wanted time past end of file
            out[k] = last[1].to_ndarray(format="rgb24")
    return np.stack(out)


def resize_batch(frames: np.ndarray, res: int) -> torch.Tensor:
    """uint8 [N, H, W, 3] -> float [N, 3, res, res] in 0..255."""
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    return F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False, antialias=True)


def to_uint8_hwc(x: torch.Tensor) -> np.ndarray:
    return x.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
