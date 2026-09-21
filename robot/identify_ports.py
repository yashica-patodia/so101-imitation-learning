"""Identify which USB port is the leader arm and which is the follower.

Connects to both SO-101 arms read-only (no torque, nothing moves), then asks
you to wiggle the LEADER arm. The port whose motor positions change is the
leader. Prints the ready-to-use teleoperate command at the end.

Usage: python robot/identify_ports.py
"""

import glob
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

SO101_MOTORS = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}


def read_positions(bus: FeetechMotorsBus) -> dict[str, int]:
    return bus.sync_read("Present_Position", normalize=False)


def main() -> None:
    ports = sorted(glob.glob("/dev/tty.usbmodem*"))
    if len(ports) != 2:
        raise SystemExit(f"Expected exactly 2 USB arms, found {len(ports)}: {ports}")

    buses = {}
    for port in ports:
        bus = FeetechMotorsBus(port=port, motors=dict(SO101_MOTORS))
        bus.connect(handshake=False)
        buses[port] = bus
        print(f"Connected (read-only) to {port}")

    baseline = {port: read_positions(bus) for port, bus in buses.items()}

    print("\n>>> Wiggle the LEADER arm (the one you hold). Watching for up to 60s... <<<\n")

    deltas = {port: 0 for port in buses}
    deadline = time.time() + 60
    while time.time() < deadline:
        time.sleep(0.3)
        for port, bus in buses.items():
            now = read_positions(bus)
            deltas[port] = sum(abs(now[m] - baseline[port][m]) for m in now)
        moved = [p for p, d in deltas.items() if d > 300]
        if len(moved) == 1 and min(deltas.values()) < 100:
            break

    for port, delta in deltas.items():
        print(f"{port}: total movement = {delta} ticks")

    leader = max(deltas, key=deltas.get)
    follower = min(deltas, key=deltas.get)

    for bus in buses.values():
        bus.disconnect()

    if deltas[leader] < 300:
        raise SystemExit("\nNeither arm moved enough — run again and wiggle harder.")
    if deltas[follower] > 100:
        raise SystemExit("\nBoth arms moved — hold the follower still and run again.")

    print(f"\nLEADER   = {leader}")
    print(f"FOLLOWER = {follower}")
    print("\nTeleoperate with:\n")
    print(
        "lerobot-teleoperate \\\n"
        f"  --robot.type=so101_follower --robot.port={follower} --robot.id=my_follower \\\n"
        f"  --teleop.type=so101_leader --teleop.port={leader} --teleop.id=my_leader"
    )


if __name__ == "__main__":
    main()
