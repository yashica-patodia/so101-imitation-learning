# nano_vla

A small OctoSense-style multimodal masked autoencoder + action probe for the
SO-101. Design rationale: `docs/nano-vla-design.md`.

Pipeline (run from the repo root with the project venv):

```bash
# 0. Download a LeRobot v3.0 dataset (retry on Hub timeouts)
HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=60 .venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('hbseong/record-pick-and-place-pos5-so101', repo_type='dataset', local_dir='data/hf/hbseong_pos5', max_workers=4)"

# 1. Frozen DINOv2 features at 5 Hz + joints at 10 Hz, one .npz per episode
.venv/bin/python -m nano_vla.extract_features --root data/hf/hbseong_pos5 \
    --repo-id hbseong/record-pick-and-place-pos5-so101 --out data/features/hbseong_pos5

# 2. Pretrain the MAE (stage 1)
.venv/bin/python -m nano_vla.train_mae --features data/features/hbseong_pos5 \
    --out outputs/nano_vla/mae --epochs 30

# 3. Train the probe on the frozen encoder and evaluate offline (stage 2)
.venv/bin/python -m nano_vla.train_probe --mae outputs/nano_vla/mae/best.pt \
    --features data/features/hbseong_pos5 --out outputs/nano_vla/probe --epochs 20

# 4. Later: fine-tune the probe on our own episodes before deploying
.venv/bin/python -m nano_vla.train_probe --mae outputs/nano_vla/mae/best.pt \
    --features data/features/ours --init-probe outputs/nano_vla/probe/best.pt \
    --out outputs/nano_vla/probe_ours --epochs 10
```

`train_probe.py` prints per-horizon-step mean absolute error in raw joint
units next to two trivial baselines (hold last state, linear extrapolation).
The probe has to beat both before it is worth putting on the robot.
