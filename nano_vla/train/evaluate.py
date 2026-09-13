"""Offline evaluation: per-joint mean absolute error at each horizon step in raw joint
units, next to two trivial baselines the model must beat: hold-last and linear
extrapolation from the last two states."""

from __future__ import annotations

import numpy as np
import torch

from nano_vla import config as C


@torch.no_grad()
def evaluate(enc, probe, dl, stats: dict, device, target: str = "state", key: str = "img") -> dict[str, np.ndarray]:
    """key: "img" for DINOv2-feature encoders, "frm" for pixel encoders."""
    pre = "state" if target == "state" else "action"
    t_std = torch.as_tensor(stats[pre + "_std"], device=device)
    t_mean = torch.as_tensor(stats[pre + "_mean"], device=device)
    s_std = torch.as_tensor(stats["state_std"], device=device)
    s_mean = torch.as_tensor(stats["state_mean"], device=device)
    err = {"probe": 0.0, "hold": 0.0, "linear": 0.0}
    n = 0
    steps = torch.arange(1, C.HORIZON + 1, device=device, dtype=torch.float32)[None, :, None]
    for batch in dl:
        x, state, tgt = batch[key].to(device), batch["state"].to(device), batch["target"].to(device)
        tokens, _, keep = enc(x, state)
        pred = probe(tokens, keep, batch["last"].to(device))
        raw_tgt, raw_pred = tgt * t_std + t_mean, pred * t_std + t_mean
        raw_state = state * s_std + s_mean
        hold = raw_state[:, -1:, :].expand(-1, C.HORIZON, -1)
        linear = raw_state[:, -1:, :] + (raw_state[:, -1] - raw_state[:, -2])[:, None, :] * steps
        err["probe"] = err["probe"] + (raw_pred - raw_tgt).abs().sum(0)
        err["hold"] = err["hold"] + (hold - raw_tgt).abs().sum(0)
        err["linear"] = err["linear"] + (linear - raw_tgt).abs().sum(0)
        n += x.shape[0]
    return {k: (v / n).cpu().numpy() for k, v in err.items()}  # each [HORIZON, 6]


def format_table(err: dict[str, np.ndarray]) -> str:
    lines = ["step | " + " | ".join(f"{k:>7s}" for k in err)]
    for h in range(C.HORIZON):
        lines.append(f"{h + 1:4d} | " + " | ".join(f"{err[k][h].mean():7.3f}" for k in err))
    lines.append("mean | " + " | ".join(f"{err[k].mean():7.3f}" for k in err))
    lines.append("per joint (probe, mean over horizon): " + ", ".join(f"{nm}={err['probe'][:, j].mean():.2f}" for j, nm in enumerate(C.JOINT_NAMES)))
    return "\n".join(lines)
