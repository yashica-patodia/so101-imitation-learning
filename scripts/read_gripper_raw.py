"""Live-print the follower gripper's raw position for 20 seconds.

Read-only (torque stays off — move the jaws by hand). Compares each reading
against the calibration file's recorded range so you can see exactly where
the physical jaw-contact point falls relative to the calibrated "closed".

Usage: python scripts/read_gripper_raw.py
"""

import json
import time
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

FOLLOWER_PORT = "/dev/tty.usbmodem5C821086091"
CALIB = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json"

calib = json.loads(CALIB.read_text())["gripper"]
print(f"calibrated gripper range: closed={calib['range_min']}  open={calib['range_max']}\n")

bus = FeetechMotorsBus(
    port=FOLLOWER_PORT,
    motors={"gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100)},
)
bus.connect(handshake=False)
print("Reading for 20s — slowly close the jaws by hand until they JUST touch, hold,")
print("then open fully. Watch the numbers:\n")

readings = []
t_end = time.time() + 20
while time.time() < t_end:
    pos = bus.sync_read("Present_Position", normalize=False)["gripper"]
    readings.append(pos)
    vs_min = pos - calib["range_min"]
    print(f"  raw={pos:5d}   ({vs_min:+d} ticks vs calibrated closed)")
    time.sleep(0.5)

bus.disconnect()
print(f"\nlowest value you reached: {min(readings)}  (calibrated closed = {calib['range_min']})")
print(f"highest value you reached: {max(readings)}  (calibrated open = {calib['range_max']})")
