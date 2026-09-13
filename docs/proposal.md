# Proposal: a nano grasping policy for the SO-101

Draft 2026-09-12, for alignment before any building starts.

## 1. Problem

Build a "grasper": a policy that makes the real SO-101 follower arm reach out
and close its gripper on an object placed anywhere within reach, starting from
a fixed rest pose. No language input. Two levels:

- **Level A, grasp.** Success = gripper closed on the object and holding it.
- **Level B, grasp and lift.** Success = object lifted clear of the table.

The model takes the last second of camera frames and joint angles and outputs
the next second of follower joint positions. It is run in a loop on the robot.
Deadline: first working grasp on the real arm by tomorrow.

## 2. Approach (revised 2026-09-12: two vision encoders)

A small transformer policy head over tokens from two cameras and the joint
stream. Following the OctoSense evaluation protocol, two vision encoders are
compared with the *same* probe head, same data, same trial positions:

- **Arm 1, baseline: frozen DINOv2-small features.** Precomputed once.
- **Arm 2, ours: a masked autoencoder trained from scratch on the robot's own
  pixels** from both cameras plus the joint stream. ViT-small proportions cut
  in half (depth 8, width 384), patch 16 on 160x160 frames, OctoSense tube
  masking with a Dirichlet split across cameras and joints, per-patch
  normalized pixel loss. The paper's FSQ tokenizer stage is skipped.

Primary protocol for both: frozen encoder + probe (isolates the
representation, as in the paper). End-to-end fine-tuning of Arm 2 is an
optional extra row, pending the prof's answer.

```
inputs                                    output
  overhead cam, 5 frames at 5 Hz  ─┐
  wrist cam,    5 frames at 5 Hz  ─┼─►  frozen DINOv2 ─► tokens ─┐
  follower joints, 10 steps at 10 Hz ─────────────► tokens ─────┼─► 4-layer transformer ─► next 10 joint positions (1 s chunk)
```

Data flow, three stages:

1. **Feature extraction (once, on the MacBook).** Every episode is decoded,
   frames sampled at 5 Hz, run through a frozen DINOv2-small encoder, and
   stored as small feature files. Videos are then no longer needed.
2. **Training (Kaggle or Colab GPU, or the MacBook for small runs).** The
   transformer learns to map a 1 s window of features + joints to the next
   1 s of joints. Loss: L1 on normalized joint positions.
3. **Deployment (MacBook connected to the arm).** At 10 Hz: grab both camera
   frames, run DINOv2 on the newest frame only, run the transformer, send the
   first few predicted joint positions to the follower, repeat.

## 3. Data

The professor asked for at least 1 hour of recordings. Hugging Face has far
more than that for the SO-101, all in the same LeRobot format and the same six
joint names as our arm.

| Dataset | Episodes | Minutes | Cameras | Content |
|---|---|---|---|---|
| 5hadytru/so101_grasp_2 | 780 | 290 | front + overhead | 20+ objects, half the scenes cluttered, pick into bin |
| 5hadytru/so101_grasp_1 | 210 | 68 | front + overhead | same setup, earlier batch |
| dobri420/pick-cube-so101 | 540 | 115 | 3 cameras | cube, deliberately varied positions |
| TakuyaHiraoka/so101_pick_diverse_objects | 962 | 158 | wrist only | ~70 objects, pick-up only |
| hbseong/record-pick-and-place-pos5-so101 | 240 | 66 | top + right | blue cube into box (already downloaded) |

Plan:

- **Primary: 5hadytru/so101_grasp_2 (+ grasp_1).** Two cameras, most objects,
  most minutes. Download size 11.3 GB + 3.0 GB; the laptop has 28 GB free, and
  the videos can be deleted after feature extraction (features are ~1/30 the size).
- **Trim to the task.** Every episode is cut at the moment the gripper closes.
  Level A keeps start → close + 0.5 s. Level B keeps start → close + 1.5 s,
  which covers the lift. The cut point comes from the gripper joint, no
  labeling needed.
- **Our own recordings.** These are essential: no public dataset has our
  camera placement, table, or lighting. Target 60–100 grasp episodes
  (~5 s each, 10–15 min of arm time) with varied objects and positions,
  recorded with the existing `lerobot-record` setup. Used to fine-tune the
  policy after pretraining on the public data.
- **Open X-Embodiment: not now.** Different robots and action spaces; no
  benefit for a 4M-parameter model on a two-day timeline.

## 4. Major design decisions

| # | Decision | Chosen | Why |
|---|---|---|---|
| 1 | Model type | Two vision encoders under one shared probe head: frozen DINOv2 (baseline) vs. our pixel MAE pretrained from scratch (revised 2026-09-12) | Matches the OctoSense protocol of comparing a self-supervised multimodal encoder against internet-pretrained vision encoders |
| 2 | Output | Next 10 follower joint positions at 10 Hz (a 1 s chunk) | A chunk gives smooth motion and avoids the "copy the current pose" failure of single-step prediction; leader commands come later per prof |
| 3 | Inputs | Both cameras, 5 frames each at 5 Hz, plus 10 joint states at 10 Hz | Prof wants both cameras; 1 s of context is enough for a reactive reach |
| 4 | Image encoders | Baseline: frozen DINOv2-small, features precomputed. Ours: MAE on raw 160x160 frames, depth 8, width 384, patch 16, tube + Dirichlet masking | Baseline is cheap and reliable; ours learns from the robot's own sensors as in the paper |
| 5 | Architecture | Early fusion, 4 layers, d=256, ~4M params | Small enough to train on a laptop; scale up only if it underfits |
| 6 | Task data | Trimmed grasp segments of public pick-and-place data + own recordings | Turns 6+ hours of public data into grasp data for free |
| 7 | Levels | A (grasp) first, B (lift) second, same code with a longer trim | Get one real success before adding difficulty |
| 8 | Baseline | ACT via `lerobot-train` on the same trimmed data | Zero custom code, and a credible safety net if the nano policy fails on the robot |
| 9 | Evaluation | Offline: per-joint error vs. hold-last and linear baselines. On robot: 20 trials, object at random reachable positions, success rate per level | Offline number picks checkpoints; the robot number is the result |
| 10 | Compute | Feature extraction and inference on the M4 MacBook; training on Kaggle (30 free GPU-hours/week) or Colab; MacBook for quick runs | No lab GPU mentioned; Kaggle's quota is the most generous free option |
| 11 | Reporting | Google Doc (decisions + results table), repo README (code + numbers), short demo video | Prof asked for all three |

## 5. Timeline (see the 18-step phased plan: docs/nano-vla-plan-steps.md)

Day 1: phases 1–4 (data, DINOv2 baseline, pixel-MAE pretraining launched, own recordings). Day 2: phase 5 (robot loop, Level A trials) and the frozen-MAE probe. Day 3: comparison on the robot, Level B, report.

## 6. Risks

- **Domain gap.** Public data was recorded on other rigs. Mitigation: own
  recordings for fine-tuning, and both cameras roughly matching the 5hadytru
  front + overhead placement if possible.
- **Hub flakiness.** Downloads timed out repeatedly yesterday. Mitigation:
  the per-file retry downloader already written.
- **Video decoding.** torchcodec fails to load on the Mac. Mitigation: switch
  the decoder backend to pyav (one line).
- **1 hour of own recordings** is not achievable in one day alongside
  everything else; the public data covers the hour, own data covers the rig.
  To confirm with the prof.

## 7. Open questions

1. Is public data acceptable toward the "1 hour of recordings" ask, with our
   own recordings at 10–15 minutes?
2. For Level B, how high is "lifted": a few centimeters off the table?
3. Is ACT acceptable as the baseline, or does the prof have another in mind?
4. Which arm pose is the rest pose, and can the object be anywhere including
   near the arm's base?
