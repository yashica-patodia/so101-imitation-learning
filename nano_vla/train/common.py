"""Shared training utilities: device, seeding, dataset assembly, checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from nano_vla.data.windows import EpisodeSet, WindowDataset, joint_stats, split_episodes


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_datasets(feature_dirs: list[str], level: str, stats: dict | None, stride: float, val_frac: float, target: str, mode: str = "dino"):
    """Returns (train_ds, val_ds, stats, first_train_set). Stats are computed from the
    training episodes if not given (pass saved stats when fine-tuning)."""
    train_sets, val_sets = [], []
    for fdir in feature_dirs:
        n = len(list(Path(fdir).glob("episode_*.npz")))
        tr_eps, va_eps = split_episodes(n, val_frac)
        train_sets.append(EpisodeSet(fdir, tr_eps, level, mode))
        val_sets.append(EpisodeSet(fdir, va_eps, level, mode))
    if stats is None:
        stats = joint_stats(train_sets)
    train_ds = ConcatDataset([WindowDataset(s, stats, stride, target) for s in train_sets])
    val_ds = ConcatDataset([WindowDataset(s, stats, 0.5, target) for s in val_sets])
    return train_ds, val_ds, stats, train_sets[0]


def loaders(train_ds, val_ds, batch: int, workers: int = 0):
    return (
        DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers, drop_last=True),
        DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers),
    )


def save_stats(out: Path, stats: dict) -> None:
    np.savez(out / "stats.npz", **stats)


def load_stats(path: Path) -> dict:
    return dict(np.load(path))


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


class JsonlLog:
    def __init__(self, path: Path):
        self.f = open(path, "a")

    def write(self, rec: dict) -> None:
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
