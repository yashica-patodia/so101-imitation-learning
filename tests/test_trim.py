import numpy as np

from nano_vla.data.trim import find_gripper_close, grid_indices


def test_close_detected_halfway():
    t = np.arange(0, 10, 1 / 30, dtype=np.float32)
    g = np.full_like(t, 95.0)
    g[t >= 6.0] = 30.0  # closes at 6 s
    tc = find_gripper_close(t, g)
    assert tc is not None and abs(tc - 6.0) < 0.05


def test_no_close_returns_none():
    t = np.arange(0, 5, 1 / 30, dtype=np.float32)
    assert find_gripper_close(t, np.full_like(t, 95.0)) is None


def test_grid_indices_rate():
    t = np.arange(0, 3, 1 / 30, dtype=np.float32)
    idx = grid_indices(t, 5.0, t_end=2.0)
    assert len(idx) == 11  # 0, .2, ..., 2.0
    assert np.allclose(t[idx], np.arange(0, 2.01, 0.2), atol=0.02)


def test_close_detected_when_episode_starts_closed():
    t = np.arange(0, 12, 1 / 30, dtype=np.float32)
    g = np.full_like(t, 1.0)
    g[(t >= 3) & (t < 5)] = 16.0  # opens on the way to the object
    g[(t >= 5) & (t < 8)] = 5.0  # closes on the object (jaws stopped by it)
    g[t >= 8] = 16.0  # releases
    tc = find_gripper_close(t, g)
    assert tc is not None and abs(tc - 5.0) < 0.05
