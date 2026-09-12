"""Windowed dataset over precomputed episode features (see extract_features.py).

One sample = a 1 s context window + the next HORIZON joint targets:

    img      float32 [C, T_IMG, P, D]   DINOv2 patch features per camera (pooled to P patches)
    state    float32 [T_JOINT, 6]        normalized follower joint positions in the window
    target   float32 [HORIZON, 6]        normalized future *actions* (leader commands)
    last     float32 [6]                 normalized last observed state (for residual prediction)

Windows are cut with a fixed stride so the same code serves MAE pretraining
(targets unused) and probe training.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

T_IMG = 5  # image frames per window (5 Hz x 1 s)
T_JOINT = 10  # joint rows per window (10 Hz x 1 s)
HORIZON = 10  # future action steps predicted (10 Hz x 1 s)
JOINT_DIM = 6


def pool_patches(x: np.ndarray, pool: int) -> np.ndarray:
    """[T, 256, D] (16x16 grid) -> [T, (16/pool)^2, D] by average pooling."""
    if pool == 1:
        return x
    t, p, d = x.shape
    g = int(round(p**0.5))
    x = x.reshape(t, g // pool, pool, g // pool, pool, d).mean(axis=(2, 4))
    return x.reshape(t, -1, d)


class EpisodeFeatures:
    """Loads one directory of episode_*.npz produced by extract_features.py."""

    def __init__(self, root: str | Path, episodes: list[int] | None = None, pool: int = 2):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.cams = self.meta["cams"]
        files = sorted(self.root.glob("episode_*.npz"))
        if episodes is not None:
            files = [files[i] for i in episodes]
        self.episodes = []
        for f in files:
            z = np.load(f, allow_pickle=True)
            imgs = np.stack([pool_patches(z["img_" + c.split(".")[-1]].astype(np.float32), pool) for c in self.cams])
            self.episodes.append(
                {
                    "img": imgs,  # [C, T5, P, D]
                    "img_t": z["img_t"],
                    "state": z["state"],
                    "action": z["action"],
                    "joint_t": z["joint_t"],
                }
            )
        self.feat_dim = self.episodes[0]["img"].shape[-1]
        self.n_patches = self.episodes[0]["img"].shape[2]

    def joint_stats(self) -> dict[str, np.ndarray]:
        s = np.concatenate([e["state"] for e in self.episodes])
        a = np.concatenate([e["action"] for e in self.episodes])
        return {
            "state_mean": s.mean(0),
            "state_std": s.std(0) + 1e-6,
            "action_mean": a.mean(0),
            "action_std": a.std(0) + 1e-6,
        }


class WindowDataset(Dataset):
    def __init__(self, feats: EpisodeFeatures, stats: dict[str, np.ndarray], stride_s: float = 0.3):
        self.feats = feats
        self.stats = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in stats.items()}
        img_hz, joint_hz = feats.meta["img_hz"], feats.meta["joint_hz"]
        self.index: list[tuple[int, int, int]] = []  # (episode, img_start, joint_start)
        for ei, e in enumerate(feats.episodes):
            n_j = len(e["joint_t"])
            # last joint window must leave HORIZON future rows
            max_joint_start = n_j - T_JOINT - HORIZON
            stride_j = max(1, int(round(stride_s * joint_hz)))
            for js in range(0, max_joint_start + 1, stride_j):
                t_end = e["joint_t"][js + T_JOINT - 1]
                # image window: the T_IMG frames at or before t_end
                ie = int(np.searchsorted(e["img_t"], t_end, side="right"))
                is_ = ie - T_IMG
                if is_ < 0:
                    continue
                self.index.append((ei, is_, js))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        ei, is_, js = self.index[i]
        e = self.feats.episodes[ei]
        img = torch.from_numpy(e["img"][:, is_ : is_ + T_IMG])  # [C, T_IMG, P, D]
        st = torch.from_numpy(e["state"][js : js + T_JOINT])
        fut = torch.from_numpy(e["action"][js + T_JOINT : js + T_JOINT + HORIZON])
        st_n = (st - self.stats["state_mean"]) / self.stats["state_std"]
        fut_n = (fut - self.stats["action_mean"]) / self.stats["action_std"]
        return {"img": img, "state": st_n, "target": fut_n, "last": st_n[-1], "episode": ei}


def split_episodes(n: int, val_frac: float = 0.1) -> tuple[list[int], list[int]]:
    n_val = max(1, int(round(n * val_frac)))
    return list(range(n - n_val)), list(range(n - n_val, n))
