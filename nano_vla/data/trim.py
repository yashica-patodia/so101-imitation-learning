"""Cutting an episode down to the reach-and-grasp segment using the gripper joint."""

from __future__ import annotations

import numpy as np


def find_gripper_close(t: np.ndarray, gripper: np.ndarray, min_travel: float = 5.0) -> float | None:
    """Time of the grasp: the first time the gripper crosses from open to closed.

    Works whether the episode starts with the gripper open (hbseong) or closed
    (5hadytru, which opens on the way to the object). Open/closed levels are the
    episode max/min; the threshold is halfway. We first find the first row where the
    gripper is open, then the first row after it where it is closed. None if the
    gripper never travels more than `min_travel` units or never closes after opening."""
    lo, hi = float(gripper.min()), float(gripper.max())
    if hi - lo < min_travel:
        return None
    thr = lo + 0.5 * (hi - lo)
    is_open = gripper > thr
    if not is_open.any():
        return None
    i_open = int(np.argmax(is_open))
    after = np.where(~is_open[i_open:])[0]
    return float(t[i_open + after[0]]) if len(after) else None


def grid_indices(t: np.ndarray, hz: float, t_end: float | None = None) -> np.ndarray:
    """Row indices nearest to a uniform grid at `hz` from t[0] to t_end (or the end)."""
    stop = t[-1] if t_end is None else min(t_end, t[-1])
    grid = np.arange(t[0], stop + 1e-6, 1.0 / hz)
    hi = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    lo = hi - 1
    idx = np.where(np.abs(t[lo] - grid) <= np.abs(t[hi] - grid), lo, hi)  # nearest row, not next
    return np.unique(idx)
