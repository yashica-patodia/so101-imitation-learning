# Calibration log

## 2026-09-05 — follower gripper range fix

**Symptom:** gripper servo (id 6) hit overload protection whenever the leader
trigger was fully squeezed; teleop tracking showed a constant ~18-unit gripper
error (0-100 scale).

**Diagnosis:** measured raw jaw-contact position by hand with torque off:
jaws touch at **2049-2052** (two independent runs). The Aug 20 calibration had
recorded `range_min=1514` — 535 ticks past the physical jaw-collision point —
so "fully closed" commanded an unreachable position and stalled the servo.
Full-open measured 3352 vs recorded 3266.

**Fix:** patched `my_follower.json` gripper `range_min` 1514→2049 and
`range_max` 3266→3352 (measured values), then wrote the calibration to the
servo registers via `bus.write_calibration()`.

**Verification:** 20 s teleop tracking test — gripper mean error 17.6 → 3.7
units; repeated full trigger squeezes produced no overload.

**Lesson:** when calibrating, sweep the gripper only to jaw contact, never
squeeze past it. The recorded min becomes the commanded "fully closed".

Backups of the pre-fix files: `my_follower.backup-aug20.json`,
`my_leader.backup-aug20.json` (in the lerobot calibration cache).
