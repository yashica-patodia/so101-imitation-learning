"""Windowed dataset over extracted episode files (see data/extract.py).

One sample = a WINDOW_S context + the next HORIZON joint targets:
    img      float32 [C, T_IMG, P, D]   DINOv2 features per camera   (mode="dino")
    frm      float32 [C, T_IMG, 3, R, R] raw frames in 0..1          (mode="pixel")
    state    float32 [T_JOINT, 6]        normalized follower joints in the window
    target   float32 [HORIZON, 6]        normalized future targets
    last     float32 [6]                 normalized last observed state
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from nano_vla import config as C


class EpisodeSet:
    """Loads a feature directory into memory, cut to a task level (A or B)."""

    def __init__(self, root: str | Path, episodes: list[int] | None = None, level: str = "A", mode: str = "dino"):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.cams = self.meta["cam_short"]
        self.mode = mode
        files = sorted(self.root.glob("episode_*.npz"))
        if episodes is not None:
            files = [files[i] for i in episodes]
        after = C.LEVEL_AFTER_CLOSE_S[level]
        self.episodes = []
        for f in files:
            z = np.load(f, allow_pickle=True)
            t_end = float(z["t_close"]) + after
            im = z["img_t"] <= t_end + 1e-6
            jm = z["joint_t"] <= t_end + 1e-6
            e = {"img_t": z["img_t"][im], "joint_t": z["joint_t"][jm], "state": z["state"][jm], "action": z["action"][jm]}
            if mode == "dino":
                e["img"] = np.stack([z["feat_" + c][im] for c in self.cams])  # [C, T, P, D] float16
            else:
                e["frm"] = np.stack([z["frm_" + c][im] for c in self.cams])  # [C, T, R, R, 3] uint8
            self.episodes.append(e)
        first = self.episodes[0]
        self.feat_dim = first["img"].shape[-1] if mode == "dino" else None
        self.n_patches = first["img"].shape[2] if mode == "dino" else None

    def joint_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.concatenate([e["state"] for e in self.episodes]), np.concatenate([e["action"] for e in self.episodes])


def joint_stats(sets: list[EpisodeSet]) -> dict[str, np.ndarray]:
    s = np.concatenate([x.joint_arrays()[0] for x in sets])
    a = np.concatenate([x.joint_arrays()[1] for x in sets])
    return {"state_mean": s.mean(0), "state_std": s.std(0) + 1e-6, "action_mean": a.mean(0), "action_std": a.std(0) + 1e-6}


class WindowDataset(Dataset):
    def __init__(self, eps: EpisodeSet, stats: dict[str, np.ndarray], stride_s: float = 0.3, target: str = "state"):
        """target: "state" predicts future follower positions, "action" predicts leader commands."""
        self.eps = eps
        self.target_key = target
        self.stats = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in stats.items()}
        stride_j = max(1, int(round(stride_s * C.JOINT_HZ)))
        self.index: list[tuple[int, int, int]] = []
        for ei, e in enumerate(eps.episodes):
            n_j = len(e["joint_t"])
            for js in range(0, n_j - C.T_JOINT - C.HORIZON + 1, stride_j):
                t_end = e["joint_t"][js + C.T_JOINT - 1]
                ie = int(np.searchsorted(e["img_t"], t_end, side="right"))
                if ie - C.T_IMG < 0:
                    continue
                self.index.append((ei, ie - C.T_IMG, js))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        ei, is_, js = self.index[i]
        e = self.eps.episodes[ei]
        st = torch.from_numpy(e["state"][js : js + C.T_JOINT])
        fut = torch.from_numpy(e[self.target_key][js + C.T_JOINT : js + C.T_JOINT + C.HORIZON])
        pre = "state" if self.target_key == "state" else "action"
        st_n = (st - self.stats["state_mean"]) / self.stats["state_std"]
        fut_n = (fut - self.stats[pre + "_mean"]) / self.stats[pre + "_std"]
        out = {"state": st_n, "target": fut_n, "last": st_n[-1], "episode": ei}
        if self.eps.mode == "dino":
            out["img"] = torch.from_numpy(e["img"][:, is_ : is_ + C.T_IMG].astype(np.float32))
        else:
            f = torch.from_numpy(e["frm"][:, is_ : is_ + C.T_IMG]).float() / 255.0  # [C,T,R,R,3]
            out["frm"] = f.permute(0, 1, 4, 2, 3)
        return out


def split_episodes(n: int, val_frac: float = 0.1) -> tuple[list[int], list[int]]:
    n_val = max(1, int(round(n * val_frac)))
    return list(range(n - n_val)), list(range(n - n_val, n))
