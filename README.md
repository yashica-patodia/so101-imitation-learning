# SO-101 nano grasping policy

A small transformer policy that makes a LeRobot SO-101 arm reach and close its
gripper on an object placed in front of it, built as a comparison of two vision
encoders under one shared probe head: frozen DINOv2 (baseline) versus a masked
autoencoder trained from scratch on the robot's own pixels (ours), following the
evaluation protocol of the OctoSense paper (arXiv 2606.27317).

Start with [docs/proposal.md](docs/proposal.md), then [docs/plan.md](docs/plan.md).

## Repository map

```
docs/               proposal, 18-step plan, design decisions, calibration log
robot/              hardware helpers: find ports, teleop tracking check, gripper calibration
notebooks/          Kaggle notebooks, one per phase (01 = download + trim + extract)
nano_vla/           the Python package
  config.py         every constant: rates, window, horizon, resolutions, model sizes
  data/
    download.py     per-file Hugging Face download with retries
    lerobot_meta.py LeRobot v3.0 metadata + parquet rows, no lerobot dependency
    video.py        PyAV sequential decoding, resizing
    trim.py         gripper-close detection, time-grid sampling
    dino.py         frozen DINOv2 patch features
    extract.py      CLI: download -> trim -> one .npz per episode (features, frames, joints)
    windows.py      EpisodeSet / WindowDataset: 1 s context -> next 1 s of joints
  models/
    layers.py       transformer block, token embedder (time + patch + modality)
    mae.py          encoder, OctoSense-style masking, MAE with light decoder
    probe.py        cross-attention action probe (HORIZON queries -> joints)
  train/
    common.py       device, seeds, dataset assembly, checkpoints, logging
    evaluate.py     per-joint error vs hold-last and linear baselines
    train_mae.py    stage 1 pretraining on feature tokens
    train_probe.py  stage 2: frozen probe / fine-tune / from-scratch policy
tests/              small synthetic tests (trim logic, model shapes)
data/, outputs/     local scratch, git-ignored
```

## Pipeline

```bash
# Phase 1 (Kaggle, notebooks/kaggle_01_extract.ipynb, or locally):
python -m nano_vla.data.extract --repo-id 5hadytru/so101_grasp_1 --work data/hf/grasp_1 --out data/features/grasp_1

# Baseline arm, direct policy on frozen DINOv2 features:
python -m nano_vla.train.train_probe --scratch --features data/features/grasp_1 --out outputs/dino_policy

# Feature-token MAE pretraining + frozen probe (representation test):
python -m nano_vla.train.train_mae --features data/features/grasp_1 --out outputs/mae
python -m nano_vla.train.train_probe --mae outputs/mae/best.pt --features data/features/grasp_1 --out outputs/mae_probe

# Fine-tune a probe on our own episodes:
python -m nano_vla.train.train_probe --mae outputs/mae/best.pt --init-probe outputs/mae_probe/best.pt \
    --stats outputs/mae_probe/stats.npz --features data/features/ours --out outputs/probe_ours --epochs 10

# Tests:
python -m pytest tests -q
```

`train_probe.py` prints per-horizon-step mean absolute error in raw joint units
next to hold-last and linear-extrapolation baselines; a policy has to beat both
before it goes near the robot.

## Hardware

SO-100/SO-101 leader + follower, overhead + wrist cameras, M4 MacBook. Setup and
calibration notes: [docs/calibration-log.md](docs/calibration-log.md).
