# Robot day checklist

Everything below runs on the MacBook with the project venv, from the repo root.
Times are rough. Keep the kill switch (Ctrl+C) reachable whenever torque is on.

## 0. Before touching the arm (5 min)

```bash
.venv/bin/python -m pytest tests -q            # all green, incl. the online==offline replay test
.venv/bin/python robot/identify_ports.py       # follower port
.venv/bin/lerobot-find-cameras opencv          # camera indices; note which is overhead, which is wrist
```

Decide the camera mapping once and keep it for recording, fine-tuning and running:
the policy was trained with cameras named `front` and `overhead`. Use
**overhead → your overhead camera** and **front → your wrist camera**.

## 1. Record 60–100 grasp episodes (15–25 min)

Object at a random reachable spot, arm at rest, teleop the reach, close the gripper,
hold 1 s, stop. ~5 s each. Change the object every 10 episodes. Cover near / far /
left / right. Last third: add a small lift after the close (for Level B).

```bash
.venv/bin/lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodemFOLLOWER --robot.id=my_follower \
  --robot.cameras='{"overhead": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}, "front": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}}' \
  --teleop.type=so101_leader --teleop.port=/dev/tty.usbmodemLEADER --teleop.id=my_leader \
  --dataset.repo_id=yashica/so101_grasp_ours --dataset.root=data/ours \
  --dataset.num_episodes=80 --dataset.episode_time_s=8 --dataset.reset_time_s=5 \
  --dataset.single_task="grasp the object" --dataset.push_to_hub=false
```

Camera names in the dataset must be exactly `overhead` and `front` (that is what the
policy checkpoint expects). Camera indices: replace 0 / 1 with what step 0 printed.

## 2. Extract features from your episodes (3–5 min on the M4)

```bash
.venv/bin/python -m nano_vla.data.extract --repo-id ours --work data/ours \
    --out data/features/ours --skip-download
```

Check the printed "done: N kept" line and segment lengths (should be ~3–6 s).

## 3. Fine-tune the policy on your episodes (~10 min on the M4)

Start from the Kaggle checkpoint, reuse its normalization stats, hold out 10%:

```bash
.venv/bin/python -m nano_vla.train.train_probe --scratch \
    --features data/features/ours --out outputs/policy_ours --epochs 15 --batch 16 --lr 1e-4 \
    --init-probe outputs/kaggle_v1/dino_policy/best.pt --init-encoder outputs/kaggle_v1/dino_policy/best.pt \
    --stats outputs/kaggle_v1/dino_policy/stats.npz
```

Read the table: the policy must beat hold-last from step 2 on. If it does not, record
more episodes before going live.

## 4. Dry run, torque off (5 min)

```bash
.venv/bin/python -m nano_vla.robot.run_policy --checkpoint outputs/policy_ours/best.pt \
    --port /dev/tty.usbmodemFOLLOWER --cam front=1 --cam overhead=0 --dry-run --duration 60 --log outputs/dry_run.jsonl
```

Move the arm through a grasp by hand or with the leader (leader teleop in a second
terminal). The printed 0.5 s-ahead MAE for the policy should be at or below hold-last,
and the "next target" gripper value should drop when you approach the object.
Loop rate should read ~10 Hz at the end.

## 5. First live attempt (15 min)

Object in the easiest spot, arm at rest, hand on Ctrl+C.

```bash
.venv/bin/python -m nano_vla.robot.run_policy --checkpoint outputs/policy_ours/best.pt \
    --port /dev/tty.usbmodemFOLLOWER --cam front=1 --cam overhead=0 --duration 12 --max-delta 6 --log outputs/live_01.jsonl
```

Watch three attempts. Note the failure mode: reaching short, closing early, hesitating.
`--max-delta` is the per-tick joint cap in degrees; raise to 10 once the motion looks
sane, lower if it jerks. `--replan 3` re-predicts every 0.3 s.

## 6. Level A trials (1–2 h with debugging)

20 trials, random reachable position each, one attempt per trial, on video. Success =
gripper closed on the object and holding for 2 s. Record the 20 positions (a sheet with
x/y on the table) so the second encoder gets the same ones. Log file per trial.
