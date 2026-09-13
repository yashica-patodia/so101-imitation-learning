"""Replays a recorded episode through PolicyRunner and checks the online path gives the
same numbers as the offline dataset path. Skips if the Kaggle checkpoint / local test
features are not present."""

from pathlib import Path

import numpy as np
import pytest
import torch

from nano_vla import config as C
from nano_vla.data.windows import EpisodeSet, WindowDataset

CKPT = Path("outputs/kaggle_v1/dino_policy/best.pt")
FEAT = Path("/private/tmp/claude-501/-Users-yashicap-research-Robotics-Project-1/7ab20af7-87a3-41f3-a219-8435439e619e/scratchpad/feat_v2")


@pytest.mark.skipif(not CKPT.exists() or not FEAT.exists(), reason="needs Kaggle checkpoint and local features")
def test_online_matches_offline():
    from nano_vla.robot.policy_runner import PolicyRunner

    runner = PolicyRunner(str(CKPT), device=torch.device("cpu"))
    eps = EpisodeSet(FEAT, [0], "A", "dino")
    stats = {k: v.cpu().numpy() for k, v in runner.stats.items()}
    ds = WindowDataset(eps, stats, 0.1)
    e = eps.episodes[0]
    cams = eps.cams  # local test set cams (top, right) stand in for the policy's (front, overhead)
    # feed the episode online up to the first window's end, then predict
    ei, is_, js = ds.index[5]
    for k in range(js, js + C.T_JOINT):
        runner.push_state(e["state"][k])
    for c_pol, c_loc in zip(runner.cams, cams):
        for k in range(is_, is_ + C.T_IMG):
            runner.push_feature(c_pol, e["img"][cams.index(c_loc), k])
    assert runner.ready
    online = runner.predict()
    # offline: same window through the dataset + encoder/probe
    b = ds[5]
    with torch.no_grad():
        tokens, _, keep = runner.enc(b["img"][None], b["state"][None])
        off_n = runner.probe(tokens, keep, b["last"][None])[0]
    offline = (off_n * runner.stats["state_std"] + runner.stats["state_mean"]).numpy()
    assert online.shape == (C.HORIZON, C.JOINT_DIM)
    assert np.allclose(online, offline, atol=1e-3), np.abs(online - offline).max()
