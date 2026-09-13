"""Run the grasp policy on the real SO-101 follower at 10 Hz.

Two modes:
  --dry-run   torque OFF. You move the arm by hand (or teleop with the leader in another
              terminal); the script only reads joints + cameras, predicts, and prints how
              far its 0.5 s-ahead prediction was from what actually happened, next to the
              hold-last baseline. Nothing is sent to the motors. Do this first.
  live        torque ON. Every tick sends the next target from the current chunk; a new
              chunk is predicted every --replan ticks. Per-tick motion is capped by
              --max-delta (degrees) through lerobot's max_relative_target. Ctrl+C stops and
              disables torque.

Cameras: the policy expects the cameras it was trained with, in order (see the
checkpoint's cfg["cams"], e.g. front, overhead). Map each to a device index:
    --cam front=1 --cam overhead=0
Which physical camera plays "front" is a choice; the fine-tune on your own episodes
must have used the same mapping (see docs/robot-day.md).

    python -m nano_vla.robot.run_policy --checkpoint outputs/probe_ours/best.pt \
        --port /dev/tty.usbmodemXXXX --cam front=1 --cam overhead=0 --dry-run
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from nano_vla import config as C
from nano_vla.robot.policy_runner import PolicyRunner

TICK = 1.0 / C.JOINT_HZ  # 0.1 s
FRAME_EVERY = int(round(C.JOINT_HZ / C.IMG_HZ))  # a new image every 2 ticks
JOINT_KEYS = [f"{n}.pos" for n in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]]


def parse_cams(items: list[str]) -> dict[str, int | str]:
    out = {}
    for it in items:
        name, idx = it.split("=", 1)
        out[name] = int(idx) if idx.isdigit() else idx
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--port", required=True, help="follower serial port")
    ap.add_argument("--id", default="my_follower", help="calibration id")
    ap.add_argument("--cam", action="append", default=[], help="name=index, one per policy camera")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--replan", type=int, default=3, help="ticks between chunk predictions (live)")
    ap.add_argument("--max-delta", type=float, default=8.0, help="max joint change per tick, degrees (live)")
    ap.add_argument("--duration", type=float, default=30.0, help="seconds to run")
    ap.add_argument("--log", default=None, help="jsonl of states/predictions for later analysis")
    args = ap.parse_args()

    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

    runner = PolicyRunner(args.checkpoint)
    cam_map = parse_cams(args.cam)
    missing = [c for c in runner.cams if c not in cam_map]
    if missing:
        raise SystemExit(f"policy needs cameras {runner.cams}; give --cam for {missing}")
    cameras = {c: OpenCVCameraConfig(index_or_path=cam_map[c], width=640, height=480, fps=30) for c in runner.cams}
    follower = SOFollower(SOFollowerRobotConfig(port=args.port, id=args.id, cameras=cameras, max_relative_target=None if args.dry_run else args.max_delta))
    print(f"policy kind={runner.kind} cams={runner.cams} device={runner.device}; {'DRY RUN (torque off)' if args.dry_run else 'LIVE'}")
    follower.connect()
    if args.dry_run:
        follower.bus.disable_torque()
        print("torque disabled. Move the arm by hand or with the leader. Ctrl+C to stop.")
    log = open(args.log, "a") if args.log else None

    # dry-run scoring: compare the prediction made 5 ticks ago (0.5 s ahead) with what happened
    pending: deque = deque()
    err_probe, err_hold, n_scored = np.zeros(6), np.zeros(6), 0
    chunk, chunk_i = None, 0
    tick, t_start = 0, time.perf_counter()
    try:
        while time.perf_counter() - t_start < args.duration:
            t0 = time.perf_counter()
            obs = follower.get_observation()
            joints = np.array([obs[k] for k in JOINT_KEYS], dtype=np.float32)
            runner.push_state(joints)
            if tick % FRAME_EVERY == 0:
                for c in runner.cams:
                    runner.push_frame(c, obs[c])
            # score old predictions
            while pending and pending[0][0] <= tick:
                _, pred5, hold = pending.popleft()
                err_probe += np.abs(pred5 - joints)
                err_hold += np.abs(hold - joints)
                n_scored += 1
            if runner.ready and (chunk is None or chunk_i >= args.replan or args.dry_run):
                chunk = runner.predict()
                chunk_i = 0
                pending.append((tick + 5, chunk[4].copy(), joints.copy()))
            line = {"t": round(time.perf_counter() - t_start, 3), "tick": tick, "state": joints.round(2).tolist()}
            if chunk is not None:
                target = chunk[min(chunk_i, C.HORIZON - 1)]
                line["target"] = target.round(2).tolist()
                if not args.dry_run:
                    sent = follower.send_action({k: float(v) for k, v in zip(JOINT_KEYS, target)})
                    line["sent"] = [round(sent[k], 2) for k in JOINT_KEYS]
                chunk_i += 1
            if log:
                log.write(json.dumps(line) + "\n")
            if tick % 10 == 0 and n_scored:
                print(f"t={line['t']:5.1f}s  0.5s-ahead MAE  policy {err_probe.sum() / n_scored / 6:5.2f}  hold-last {err_hold.sum() / n_scored / 6:5.2f}   gripper now {joints[5]:5.1f}" + (f"  next target {chunk[2].round(1).tolist()}" if chunk is not None else ""), flush=True)
            tick += 1
            time.sleep(max(0.0, TICK - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        follower.disconnect()  # disables torque by default
        if log:
            log.close()
        if n_scored:
            print(f"\nover {n_scored} predictions, 0.5 s ahead, per-joint MAE (degrees):")
            print("  policy   ", (err_probe / n_scored).round(2).tolist())
            print("  hold-last", (err_hold / n_scored).round(2).tolist())
        print(f"loop rate: {tick / max(1e-6, time.perf_counter() - t_start):.1f} Hz over {tick} ticks")


if __name__ == "__main__":
    main()
