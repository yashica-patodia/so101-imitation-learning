"""Phase 1 on Kaggle: download a LeRobot v3.0 SO-101 dataset, trim each episode
to the reach-and-grasp segment, and store per episode:

    feat_<cam>  float16 [T, 64, 384]      DINOv2-small patch features, 16x16 grid avg-pooled to 8x8
    frm_<cam>   uint8   [T, 160, 160, 3]  raw RGB frames for the pixel MAE
    img_t       float32 [T]               frame times (s from episode start), IMG_HZ grid
    state       float32 [Tj, 6]           observation.state at JOINT_HZ
    action      float32 [Tj, 6]           action at JOINT_HZ
    joint_t     float32 [Tj]
    t_close     float32                   time the gripper closed (s from episode start)
    task        str

The segment kept is [0, t_close + KEEP_AFTER_CLOSE_S]; the loader cuts shorter
for Level A. No lerobot dependency: parquet via pandas, video via PyAV, so it
runs on a bare Kaggle image.

Usage (Kaggle or anywhere):
    python -m nano_vla.kaggle_extract --repo-id 5hadytru/so101_grasp_1 \
        --work /kaggle/tmp/grasp_1 --out /kaggle/working/features/grasp_1
    add --episodes 2 for a smoke test, --skip-download if files are present.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

IMG_HZ = 5.0
JOINT_HZ = 10.0
FRAME_RES = 160
DINO_RES = 224
POOL = 2  # 16x16 -> 8x8 patches
KEEP_AFTER_CLOSE_S = 1.5  # covers the lift; Level A cuts at 0.5 in the loader
GRIPPER_IDX = 5
DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DINO_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------------------------------------------------------- download
def download(repo_id: str, local_dir: Path) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    files = [s.rfilename for s in HfApi().dataset_info(repo_id).siblings]
    missing = [f for f in files if not (local_dir / f).exists()]
    print(f"{repo_id}: {len(files)} files, {len(missing)} to fetch", flush=True)
    for round_ in range(20):
        if not missing:
            break
        still = []
        for f in missing:
            try:
                hf_hub_download(repo_id, f, repo_type="dataset", local_dir=str(local_dir))
            except Exception as e:  # noqa: BLE001
                still.append(f)
        print(f"  round {round_}: {len(files) - len(still)}/{len(files)} present", flush=True)
        missing = still
        if missing:
            time.sleep(5)
    if missing:
        raise RuntimeError(f"{len(missing)} files still missing after retries")


# ----------------------------------------------------------------------------- metadata
def load_meta(root: Path) -> tuple[dict, pd.DataFrame, list[str], pd.DataFrame]:
    info = json.loads((root / "meta" / "info.json").read_text())
    eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(root / "meta/episodes/chunk-*/*.parquet")))])
    eps = eps.sort_values("episode_index").reset_index(drop=True)
    cams = [k for k, v in info["features"].items() if v["dtype"] in ("video", "image")]
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    return info, eps, cams, tasks


def episode_rows(root: Path, info: dict, ep: pd.Series) -> pd.DataFrame:
    path = root / info["data_path"].format(chunk_index=int(ep["data/chunk_index"]), file_index=int(ep["data/file_index"]))
    df = pd.read_parquet(path, columns=["episode_index", "timestamp", "observation.state", "action", "task_index"])
    return df[df["episode_index"] == int(ep["episode_index"])].reset_index(drop=True)


# ----------------------------------------------------------------------------- trimming
def find_gripper_close(t: np.ndarray, gripper: np.ndarray) -> float | None:
    """First time the gripper drops halfway from its starting (open) level toward its
    episode minimum (closed). None if it never closes."""
    open_level = float(np.median(gripper[: max(3, len(gripper) // 20)]))
    closed_level = float(gripper.min())
    if open_level - closed_level < 5.0:  # no meaningful closure in this episode
        return None
    thr = open_level - 0.5 * (open_level - closed_level)
    below = np.where(gripper < thr)[0]
    return float(t[below[0]]) if len(below) else None


def grid_indices(t: np.ndarray, hz: float, t_end: float) -> np.ndarray:
    grid = np.arange(t[0], min(t_end, t[-1]) + 1e-6, 1.0 / hz)
    idx = np.searchsorted(t, grid)
    return np.unique(np.clip(idx, 0, len(t) - 1))


# ----------------------------------------------------------------------------- video
def decode_at(path: Path, wanted: list[float], tol: float = 0.02) -> np.ndarray:
    """Sequentially decode `path` and return uint8 frames [N, H, W, 3] nearest to each
    wanted timestamp (seconds in the file's own clock). Sequential decode is far faster
    than seeking for AV1."""
    out: list[np.ndarray | None] = [None] * len(wanted)
    order = np.argsort(wanted)
    targets = [wanted[i] for i in order]
    j = 0
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        tb = float(stream.time_base)
        last = None
        for frame in c.decode(stream):
            ts = frame.pts * tb
            while j < len(targets) and ts >= targets[j] - tol:
                # pick this frame or the previous one, whichever is closer
                cand = frame if last is None or abs(ts - targets[j]) <= abs(last[0] - targets[j]) else last[1]
                out[order[j]] = cand.to_ndarray(format="rgb24")
                j += 1
            last = (ts, frame)
            if j >= len(targets):
                break
    for k in range(len(out)):
        if out[k] is None:  # past end of file: reuse the last decoded frame
            out[k] = last[1].to_ndarray(format="rgb24")
    return np.stack(out)


def resize_batch(frames: np.ndarray, res: int) -> torch.Tensor:
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    return F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False, antialias=True)


@torch.no_grad()
def dino_features(model, frames: np.ndarray, device, batch: int = 64) -> np.ndarray:
    outs = []
    for i in range(0, len(frames), batch):
        x = resize_batch(frames[i : i + batch], DINO_RES) / 255.0
        x = ((x - DINO_MEAN) / DINO_STD).to(device)
        f = model.forward_features(x)["x_norm_patchtokens"]  # [B, 256, 384]
        b, p, d = f.shape
        g = int(round(p**0.5))
        f = f.reshape(b, g // POOL, POOL, g // POOL, POOL, d).mean(dim=(2, 4)).reshape(b, -1, d)
        outs.append(f.to(torch.float16).cpu())
    return torch.cat(outs).numpy()


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--work", required=True, help="where the raw dataset is downloaded (scratch)")
    ap.add_argument("--out", required=True, help="feature output dir")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--no-dino", action="store_true")
    ap.add_argument("--delete-videos", action="store_true", help="remove each mp4 once all its episodes are done")
    args = ap.parse_args()

    work, out = Path(args.work), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        download(args.repo_id, work)
    info, eps, cams, tasks = load_meta(work)
    n = len(eps) if args.episodes is None else min(args.episodes, len(eps))
    device = pick_device()
    model = None if args.no_dino else torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False).eval().to(device)
    print(f"{args.repo_id}: {len(eps)} episodes, cams={cams}, fps={info['fps']}, device={device}, doing {n}", flush=True)

    # Group episodes by video file so each file is decoded once.
    key = lambda e, cam: (int(e[f"videos/{cam}/chunk_index"]), int(e[f"videos/{cam}/file_index"]))  # noqa: E731
    summary = []
    t0 = time.time()
    done_files: dict[tuple, int] = {}
    for i in range(n):
        ep = eps.iloc[i]
        ep_idx = int(ep["episode_index"])
        dst = out / f"episode_{ep_idx:06d}.npz"
        if dst.exists():
            continue
        rows = episode_rows(work, info, ep)
        t = rows["timestamp"].to_numpy(np.float32)
        state = np.stack(rows["observation.state"].to_numpy()).astype(np.float32)
        action = np.stack(rows["action"].to_numpy()).astype(np.float32)
        t_close = find_gripper_close(t, state[:, GRIPPER_IDX])
        if t_close is None:
            print(f"ep {ep_idx}: gripper never closes, skipped", flush=True)
            summary.append({"episode": ep_idx, "skipped": "no_close"})
            continue
        t_end = t_close + KEEP_AFTER_CLOSE_S
        img_idx = grid_indices(t, IMG_HZ, t_end)
        j_idx = grid_indices(t, JOINT_HZ, t_end)
        payload = {
            "img_t": t[img_idx] - t[0],
            "state": state[j_idx],
            "action": action[j_idx],
            "joint_t": t[j_idx] - t[0],
            "t_close": np.float32(t_close - t[0]),
            "task": str(tasks.index[int(rows["task_index"].iloc[0])]) if len(tasks) else "",
        }
        for cam in cams:
            from_ts = float(ep[f"videos/{cam}/from_timestamp"])
            vpath = work / info["video_path"].format(video_key=cam, chunk_index=key(ep, cam)[0], file_index=key(ep, cam)[1])
            frames = decode_at(vpath, [from_ts + float(x) for x in t[img_idx]])
            short = cam.split(".")[-1]
            payload["frm_" + short] = resize_batch(frames, FRAME_RES).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
            if model is not None:
                payload["feat_" + short] = dino_features(model, frames, device)
            done_files[(cam, key(ep, cam))] = done_files.get((cam, key(ep, cam)), 0) + 1
        np.savez(dst, **payload)
        summary.append({"episode": ep_idx, "t_close": float(payload["t_close"]), "n_img": int(len(img_idx)), "n_joint": int(len(j_idx))})
        el = time.time() - t0
        print(f"ep {ep_idx} ({i + 1}/{n})  close={payload['t_close']:.2f}s imgs={len(img_idx)} joints={len(j_idx)}  {el / (i + 1):.1f}s/ep", flush=True)

    if args.delete_videos:
        for f in glob.glob(str(work / "videos" / "**" / "*.mp4"), recursive=True):
            os.remove(f)
    meta = {"repo_id": args.repo_id, "cams": cams, "cam_short": [c.split(".")[-1] for c in cams], "img_hz": IMG_HZ, "joint_hz": JOINT_HZ,
            "frame_res": FRAME_RES, "feat_pool": POOL, "feat_dim": 384, "n_patches": (16 // POOL) ** 2, "keep_after_close_s": KEEP_AFTER_CLOSE_S,
            "episodes": summary}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    kept = [s for s in summary if "t_close" in s]
    if kept:
        L = np.array([s["n_img"] / IMG_HZ for s in kept])
        print(f"done: {len(kept)} kept, {len(summary) - len(kept)} skipped; segment length s: median {np.median(L):.1f}, min {L.min():.1f}, max {L.max():.1f}")


if __name__ == "__main__":
    main()
