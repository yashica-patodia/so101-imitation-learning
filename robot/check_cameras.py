"""Grab one frame from the overhead and wrist cameras exactly as the recorder will see
them (640x480, optional rotation), save them side by side, and print quick checks:
brightness, color cast, and the image the policy will actually receive (160x160).

Usage:
  python robot/check_cameras.py --overhead 0 --front 1 --rotate-front 180
Opens nothing on the arm; cameras only.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

ROT = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, -90: cv2.ROTATE_90_COUNTERCLOCKWISE}


def grab(index: int, rotate: int) -> np.ndarray:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame = None
    for _ in range(20):  # let auto-exposure settle
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    if frame is None:
        raise SystemExit(f"camera {index}: no frame (wrong index, or in use by another app)")
    if ROT[rotate] is not None:
        frame = cv2.rotate(frame, ROT[rotate])
    return frame


def report(name: str, bgr: np.ndarray) -> None:
    b, g, r = [float(bgr[..., i].mean()) for i in range(3)]
    bright = (r + g + b) / 3
    cast = "neutral"
    if r > 1.25 * b:
        cast = "WARM/ORANGE cast: use a whiter light"
    elif b > 1.25 * r:
        cast = "blue cast"
    level = "ok" if 70 < bright < 190 else ("TOO DARK" if bright <= 70 else "TOO BRIGHT")
    print(f"{name:9s} {bgr.shape[1]}x{bgr.shape[0]}  brightness {bright:5.1f} ({level})  R/G/B {r:.0f}/{g:.0f}/{b:.0f} ({cast})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overhead", type=int, required=True)
    ap.add_argument("--front", type=int, required=True, help="wrist camera index")
    ap.add_argument("--rotate-front", type=int, default=180, choices=[0, 90, 180, -90])
    ap.add_argument("--rotate-overhead", type=int, default=0, choices=[0, 90, 180, -90])
    ap.add_argument("--out", default="outputs/camera_check.png")
    args = ap.parse_args()

    over = grab(args.overhead, args.rotate_overhead)
    front = grab(args.front, args.rotate_front)
    report("overhead", over)
    report("front", front)
    small = [cv2.resize(cv2.resize(x, (160, 160), interpolation=cv2.INTER_AREA), (480, 480), interpolation=cv2.INTER_NEAREST) for x in (over, front)]
    top = np.hstack([cv2.resize(over, (640, 480)), cv2.resize(front, (640, 480))])
    bottom = np.hstack([cv2.copyMakeBorder(s, 0, 0, 80, 80, cv2.BORDER_CONSTANT) for s in small])
    for img, txt in ((top, "as recorded: overhead | front (wrist)"), (bottom, "what the pixel model sees (160x160)")):
        cv2.putText(img, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, np.vstack([top, bottom]))
    print(f"saved {args.out}\ncheck: jaws at the BOTTOM of the front view; whole reachable area + rest pose in the overhead view; matte surface, no reflections")


if __name__ == "__main__":
    main()
