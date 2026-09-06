"""Teleoperate for a fixed duration and measure follower tracking error.

Runs the standard leader->follower teleop loop while logging, at every cycle,
the difference between the commanded position (leader) and the follower's
actual position. Prints per-joint mean/p95/max error and the achieved loop
rate at the end.

SAFETY: on connect the follower moves toward the leader's pose. Hold the
leader roughly matching the follower before starting. Motion per cycle is
clamped by --max-relative-target as an extra guard.

Usage:
  python scripts/check_teleop_tracking.py \
      --leader-port /dev/tty.usbmodemXXX --follower-port /dev/tty.usbmodemYYY \
      --seconds 20
"""

import argparse
import time

from lerobot.robots.so_follower import SOFollower, SOFollowerConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leader-port", required=True)
    parser.add_argument("--follower-port", required=True)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--max-relative-target", type=float, default=20)
    args = parser.parse_args()

    leader = SOLeader(SOLeaderConfig(port=args.leader_port, id="my_leader"))
    follower = SOFollower(
        SOFollowerConfig(
            port=args.follower_port,
            id="my_follower",
            max_relative_target=args.max_relative_target,
        )
    )

    leader.connect()
    follower.connect()
    print(f"Connected. Teleoperating for {args.seconds:.0f}s — move the leader now.")

    errors: dict[str, list[float]] = {}
    cycle_times: list[float] = []
    t_end = time.time() + args.seconds
    period = 1.0 / args.fps

    try:
        while time.time() < t_end:
            t0 = time.perf_counter()
            action = leader.get_action()
            follower.send_action(action)
            obs = follower.get_observation()
            for key, target in action.items():
                if key.endswith(".pos") and key in obs:
                    errors.setdefault(key, []).append(abs(target - obs[key]))
            cycle_times.append(time.perf_counter() - t0)
            leftover = period - (time.perf_counter() - t0)
            if leftover > 0:
                time.sleep(leftover)
    finally:
        leader.disconnect()
        follower.disconnect()

    print(f"\n{'joint':<20}{'mean':>8}{'p95':>8}{'max':>8}   (degrees; gripper in 0-100 units)")
    worst_mean = 0.0
    for key, errs in errors.items():
        errs_sorted = sorted(errs)
        mean = sum(errs) / len(errs)
        p95 = errs_sorted[int(0.95 * (len(errs) - 1))]
        joint = key.removesuffix(".pos")
        print(f"{joint:<20}{mean:>8.2f}{p95:>8.2f}{max(errs):>8.2f}")
        if joint != "gripper":
            worst_mean = max(worst_mean, mean)

    hz = 1.0 / (sum(cycle_times) / len(cycle_times))
    print(f"\nloop rate: {hz:.1f} Hz over {len(cycle_times)} cycles")
    if worst_mean < 4.0:
        print("VERDICT: PASS — tracking error is within normal range for SO-101.")
    else:
        print("VERDICT: CHECK — a joint has high mean error; suspect calibration or load.")


if __name__ == "__main__":
    main()
