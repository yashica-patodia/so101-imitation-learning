"""Phase 1 CLI: download -> trim at gripper close -> per-episode feature files.

Each episode_XXXXXX.npz holds
    feat_<cam>  float16 [T, 64, 384]      DINOv2 pooled patch features (baseline arm)
    frm_<cam>   uint8   [T, 160, 160, 3]  raw frames (pixel-MAE arm)
    img_t       float32 [T]               frame times, s from episode start, IMG_HZ grid
    state       float32 [Tj, 6]           follower joints at JOINT_HZ
    action      float32 [Tj, 6]           leader commands at JOINT_HZ
    joint_t     float32 [Tj]
    t_close     float32                   gripper-close time, s from episode start
    task        str
and meta.json describes the set. Segment stored: [0, t_close + KEEP_AFTER_CLOSE_S].

Usage:
    python -m nano_vla.data.extract --repo-id 5hadytru/so101_grasp_1 \
        --work /kaggle/tmp/grasp_1 --out /kaggle/working/features/grasp_1 --delete-videos
    --episodes 3 for a smoke test; --skip-download if the raw files are present.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from nano_vla import config as C
from nano_vla.data.dino import load_dino, patch_features
from nano_vla.data.download import download
from nano_vla.data.lerobot_meta import DatasetMeta
from nano_vla.data.trim import find_gripper_close, grid_indices
from nano_vla.data.video import decode_at, resize_batch, to_uint8_hwc
from nano_vla.train.common import pick_device


def extract_episode(meta: DatasetMeta, ep, dino, device) -> dict | None:
    rows = meta.rows(ep)
    t, state, action = rows["timestamp"], rows["state"], rows["action"]
    t_close = find_gripper_close(t, state[:, C.GRIPPER_IDX])
    if t_close is None:
        return None
    t_end = t_close + C.KEEP_AFTER_CLOSE_S
    img_idx = grid_indices(t, C.IMG_HZ, t_end)
    j_idx = grid_indices(t, C.JOINT_HZ, t_end)
    payload = {
        "img_t": t[img_idx] - t[0],
        "state": state[j_idx],
        "action": action[j_idx],
        "joint_t": t[j_idx] - t[0],
        "t_close": np.float32(t_close - t[0]),
        "task": meta.task_string(rows["task_index"]),
    }
    for cam in meta.cams:
        off = meta.video_offset(ep, cam)
        frames = decode_at(meta.video_path(ep, cam), [off + float(x) for x in t[img_idx]])
        short = cam.split(".")[-1]
        payload["frm_" + short] = to_uint8_hwc(resize_batch(frames, C.FRAME_RES))
        if dino is not None:
            payload["feat_" + short] = patch_features(dino, frames, device)
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--work", required=True, help="scratch dir for the raw download")
    ap.add_argument("--out", required=True, help="feature output dir")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--no-dino", action="store_true")
    ap.add_argument("--delete-videos", action="store_true")
    args = ap.parse_args()

    work, out = Path(args.work), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        download(args.repo_id, work)
    meta = DatasetMeta.load(work)
    device = pick_device()
    dino = None if args.no_dino else load_dino(device)
    n = len(meta) if args.episodes is None else min(args.episodes, len(meta))
    print(f"{args.repo_id}: {len(meta)} episodes, cams={meta.cams}, fps={meta.fps}, device={device}, doing {n}", flush=True)

    summary, t0 = [], time.time()
    for i in range(n):
        ep = meta.episodes.iloc[i]
        ep_idx = int(ep["episode_index"])
        dst = out / f"episode_{ep_idx:06d}.npz"
        if dst.exists():
            continue
        payload = extract_episode(meta, ep, dino, device)
        if payload is None:
            print(f"ep {ep_idx}: gripper never closes, skipped", flush=True)
            summary.append({"episode": ep_idx, "skipped": "no_close"})
            continue
        np.savez(dst, **payload)
        summary.append({"episode": ep_idx, "t_close": float(payload["t_close"]), "n_img": int(len(payload["img_t"])), "n_joint": int(len(payload["joint_t"]))})
        print(f"ep {ep_idx} ({i + 1}/{n})  close={payload['t_close']:.2f}s imgs={len(payload['img_t'])}  {(time.time() - t0) / (i + 1):.1f}s/ep", flush=True)

    if args.delete_videos and any("t_close" in x for x in summary):
        for f in glob.glob(str(work / "videos" / "**" / "*.mp4"), recursive=True):
            os.remove(f)
    info = {
        "repo_id": args.repo_id, "cams": meta.cams, "cam_short": [c.split(".")[-1] for c in meta.cams],
        "img_hz": C.IMG_HZ, "joint_hz": C.JOINT_HZ, "frame_res": C.FRAME_RES, "feat_dim": C.DINO_FEAT_DIM,
        "n_patches": C.DINO_N_PATCHES, "keep_after_close_s": C.KEEP_AFTER_CLOSE_S, "episodes": summary,
    }
    (out / "meta.json").write_text(json.dumps(info, indent=1))
    kept = [s for s in summary if "t_close" in s]
    if kept:
        L = np.array([s["n_img"] / C.IMG_HZ for s in kept])
        print(f"done: {len(kept)} kept, {len(summary) - len(kept)} skipped; segment s: median {np.median(L):.1f}, min {L.min():.1f}, max {L.max():.1f}")


if __name__ == "__main__":
    main()
