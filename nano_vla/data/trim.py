"""Cutting an episode down to the reach-and-grasp segment using the gripper joint."""

from __future__ import annotations

import numpy as np


def find_gripper_close(t: np.ndarray, gripper: np.ndarray, min_travel: float = 5.0) -> float | None:
    """Time of the first gripper closure.

    The gripper starts open; we take its starting level from the first 5% of rows and its
    closed level as the episode minimum, and return the first time it crosses halfway
    between the two. None if it never travels more than `min_travel` units."""
    open_level = float(np.median(gripper[: max(3, len(gripper) // 20)]))
    closed_level = float(gripper.min())
    if open_level - closed_level < min_travel:
        return None
    thr = open_level - 0.5 * (open_level - closed_level)
    below = np.where(gripper < thr)[0]
    return float(t[below[0]]) if len(below) else None


def grid_indices(t: np.ndarray, hz: float, t_end: float | None = None) -> np.ndarray:
    """Row indices nearest to a uniform grid at `hz` from t[0] to t_end (or the end)."""
    stop = t[-1] if t_end is None else min(t_end, t[-1])
    grid = np.arange(t[0], stop + 1e-6, 1.0 / hz)
    hi = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    lo = hi - 1
    idx = np.where(np.abs(t[lo] - grid) <= np.abs(t[hi] - grid), lo, hi)  # nearest row, not next
    return np.unique(idx)
