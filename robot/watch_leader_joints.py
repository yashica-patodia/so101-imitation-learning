"""Live-print the leader arm's joint positions. Twist one joint at a time and
watch which number moves. Ctrl+C to quit."""

import time

from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderConfig
from lerobot.teleoperators.so_leader.so_leader import SOLeader

cfg = SOLeaderConfig(port="/dev/tty.usbmodem5C821075291", id="my_leader", use_degrees=True)
leader = SOLeader(cfg)
leader.connect()
print("Connected. Twist the wrist ROLL joint slowly, full range, both directions.\n")

try:
    while True:
        action = leader.get_action()
        line = "  ".join(f"{k.removesuffix('.pos')}: {v:7.1f}" for k, v in action.items())
        print("\r" + line, end="", flush=True)
        time.sleep(0.05)
except KeyboardInterrupt:
    print()
    leader.disconnect()
