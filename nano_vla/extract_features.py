"""Stage 0: precompute frozen DINOv2 patch features + downsampled joints per episode.

For every episode of a LeRobot v3.0 dataset stored locally, this writes one
.npz file containing

    img_<cam>   float16 [T_img, 256, D]   DINOv2 patch tokens for frames at IMG_HZ
    img_t       float32 [T_img]           timestamps (s, relative to episode start)
    state       float32 [T_j, 6]          observation.state at JOINT_HZ
    action      float32 [T_j, 6]          action (leader arm target) at JOINT_HZ
    joint_t     float32 [T_j]             timestamps of the joint rows
    task        str

Image frames are decoded with lerobot's own video decoder so timestamps line up
exactly with the parquet rows. Videos are never kept in memory beyond one
episode.

Usage:
    python -m nano_vla.extract_features --root data/hf/hbseong_pos5 \
        --repo-id hbseong/record-pick-and-place-pos5-so101 --out data/features/hbseong_pos5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames

IMG_HZ = 5.0  # image frames per second kept (source is 30 fps -> every 6th frame)
JOINT_HZ = 10.0  # joint rows per second kept (every 3rd frame)
DINO_RES = 224  # 224 / 14 = 16 -> 256 patch tokens
DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DINO_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_dino(name: str, device: torch.device):
    model = torch.hub.load("facebookresearch/dinov2", name, verbose=False)
    return model.eval().to(device)


@torch.no_grad()
def dino_patches(model, frames: torch.Tensor, device: torch.device, batch: int = 32) -> np.ndarray:
    """frames: uint8/float [N, 3, H, W] in [0,1] -> float16 [N, 256, D]."""
    outs = []
    for i in range(0, frames.shape[0], batch):
        x = frames[i : i + batch].float()
        if x.max() > 1.5:
            x = x / 255.0
        x = F.interpolate(x, size=(DINO_RES, DINO_RES), mode="bilinear", align_corners=False)
        x = ((x - DINO_MEAN) / DINO_STD).to(device)
        feats = model.forward_features(x)["x_norm_patchtokens"]
        outs.append(feats.to(torch.float16).cpu())
    return torch.cat(outs).numpy()


def episode_rows(ds: LeRobotDataset, ep_idx: int) -> dict:
    """Return the parquet rows of one episode as numpy arrays (no video decode)."""
    ep = ds.meta.episodes[ep_idx]
    start, end = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
    hf = ds.hf_dataset
    sub = hf.select(range(start, end))
    cols = sub.with_format("numpy")
    return {
        "timestamp": np.asarray(cols["timestamp"], dtype=np.float32),
        "state": np.stack(cols["observation.state"]).astype(np.float32),
        "action": np.stack(cols["action"]).astype(np.float32),
        "task_index": int(cols["task_index"][0]),
    }


def subsample_indices(timestamps: np.ndarray, hz: float) -> np.ndarray:
    """Indices of rows closest to a uniform grid at `hz` starting at the first timestamp."""
    t0, t1 = float(timestamps[0]), float(timestamps[-1])
    grid = np.arange(t0, t1 + 1e-6, 1.0 / hz)
    idx = np.searchsorted(timestamps, grid)
    idx = np.clip(idx, 0, len(timestamps) - 1)
    return np.unique(idx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="local dataset dir (snapshot_download target)")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dino", default="dinov2_vits14")
    ap.add_argument("--episodes", type=int, default=None, help="limit for smoke tests")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"device={device}")

    ds = LeRobotDataset(args.repo_id, root=args.root)
    cams = list(ds.meta.camera_keys)
    print(f"{ds.meta.total_episodes} episodes, cams={cams}, fps={ds.meta.fps}")
    model = load_dino(args.dino, device)

    n = ds.meta.total_episodes if args.episodes is None else min(args.episodes, ds.meta.total_episodes)
    meta = {"repo_id": args.repo_id, "cams": cams, "img_hz": IMG_HZ, "joint_hz": JOINT_HZ, "dino": args.dino}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    t_start = time.time()
    for ep_idx in range(n):
        dst = out / f"episode_{ep_idx:06d}.npz"
        if dst.exists() and not args.overwrite:
            continue
        rows = episode_rows(ds, ep_idx)
        ts = rows["timestamp"]
        img_idx = subsample_indices(ts, IMG_HZ)
        joint_idx = subsample_indices(ts, JOINT_HZ)
        img_ts = ts[img_idx].tolist()

        ep = ds.meta.episodes[ep_idx]
        payload = {
            "img_t": ts[img_idx] - ts[0],
            "state": rows["state"][joint_idx],
            "action": rows["action"][joint_idx],
            "joint_t": ts[joint_idx] - ts[0],
            "task": str(ds.meta.tasks.iloc[rows["task_index"]].name),
        }
        for cam in cams:
            from_ts = float(ep[f"videos/{cam}/from_timestamp"])
            video_path = ds.root / ds.meta.get_video_file_path(ep_idx, cam)
            frames = decode_video_frames(video_path, [from_ts + t for t in img_ts], ds.tolerance_s, ds.video_backend)
            payload["img_" + cam.split(".")[-1]] = dino_patches(model, frames, device)
        np.savez(dst, **payload)
        el = time.time() - t_start
        print(f"ep {ep_idx + 1}/{n}  imgs={len(img_idx)} joints={len(joint_idx)}  {el / (ep_idx + 1):.1f}s/ep", flush=True)
    print("done", out)


if __name__ == "__main__":
    main()
