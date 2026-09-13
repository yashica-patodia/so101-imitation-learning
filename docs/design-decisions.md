# Nano VLA design decisions

Reference paper: OctoSense (arXiv 2606.27317). Goal: a small masked-autoencoder
model over camera frames + joint angles that predicts the next few joint angles
for a grasp task on the SO-101.

## Decisions so far (2026-09-11)

1. **Two-stage design (pretrain + probe).** Stage 1: masked autoencoder with
   random masks over image and joint tokens, no action labels needed. Stage 2:
   freeze the encoder, train a small probe that outputs the next K joint angles.
   Chosen because it lets Open X-Embodiment data (end-effector actions) help
   pretraining even though our policy outputs joint angles.

2. **Reconstruction targets.** Joints: regress raw angles (6 values per step).
   Images: regress frozen DINOv2 per-patch features, precomputed offline and
   stored as float16. No FSQ tokenizer stage (paper's Sec. 4.2 skipped).

5. **Confirmed with prof: "nano pi" = a small pi-zero style model.** Version 1
   has no language conditioning (single task); a frozen text embedding of the
   task string can be added as one extra token later.
6. **Fusion:** early fusion (one transformer over all image + joint tokens).
7. **Baseline:** ACT trained with lerobot-train on the same data; both models
   reported in one success-rate table.
8. **Evaluation:** offline per-joint MAE at each horizon step on held-out
   episodes (checkpoint selection only) + real-robot success rate over 20+
   grasp trials (headline number).
9. **Compute:** M4 MacBook (MPS) for feature extraction and small runs; Colab /
   Kaggle GPU for full training.

## Open questions (to confirm with prof)

- None blocking. Remaining prof check-ins: whether skipping the FSQ tokenizer
  stage is acceptable, and whether a longer pretraining window is wanted.

3. **Training window (first pass, chosen for ease of training).** Two cameras
   (top + right/wrist), each a separate image modality with its own sensor
   index in the positional encoding; 5 image frames over 1 s at 5 Hz (every 6th
   frame of 30 fps; 4 Hz did not divide evenly), joints at 10 Hz (10 tokens),
   horizon K = 10 steps (1 s). DINOv2-small patch grid 16x16 is average-pooled
   2x2 to 64 patches per frame, so one window is 2 x 5 x 64 = 640 image tokens
   + 10 joint tokens. Same window for MAE and probe. 4-layer, d=256 encoder +
   2-layer d=128 decoder, 3.8M params total. Masking: 75% of image patches as
   spatio-temporal tubes with a Dirichlet split across cameras; joints kept
   with Bernoulli(0.2 + 0.5 * Beta(2,5)). Loss: L1 on all tokens (paper Sec. 4.2).

4. **Data plan.** Stage 1 MAE pretraining on hbseong/record-pick-and-place-pos5-so101
   plus jackvial/so101_pickplace_success_120_v2 and the grasp sets below
   (dongyoonkim/so101-pi05-base-dataset is 185 GB; the laptop has 28 GB free,
   so it is out unless an external drive or cloud bucket is used).
   Stage 2 probe on hbseong alone.
   Fine-tune the probe on 20-50 of our own episodes before real-robot deployment.

## Candidate Hugging Face datasets (checked 2026-09-11)

All are LeRobot v3.0, 30 fps, and use the same 6-D action/state names as our
smoke test (`shoulder_pan.pos ... gripper.pos`), so they load with the same code.

| id | episodes | frames | cameras | task |
|---|---|---|---|---|
| hbseong/record-pick-and-place-pos5-so101 | 240 | 119k | top, right (480x640) | pick blue cube, place in white box |
| jackvial/so101_pickplace_success_120_v2 | 120 | 43k | top, side (600x800) | pick orange cube, place on X marker |
| lerobot/svla_so101_pickplace | 50 | 12k | up, side (480x640) | pink lego brick into transparent box |
| yangfengzzz/yellow_block_grasp_vary | 100 | 28k | (so101_follower) | grasp yellow block, varied positions |
| zz4321/so101_grasp_rubic | 102 | 34k | (so101_follower) | grasp Rubik's cube |
| dongyoonkim/so101-pi05-base-dataset | merged from 150 SO-100/101 repos | large | mixed | mixed; pretraining pool |

Aggregates: lerobot/community_dataset_v3 (791 datasets, 46 robot types, tagged
so100/so101) and allenai/MolmoAct2-SO100_101-Dataset (manifest of 1220 SO repos).

## Code

`nano_vla/` in this repo: `extract_features.py` (stage 0), `data.py`,
`model.py`, `train_mae.py` (stage 1), `train_probe.py` (stage 2 + offline eval
against hold-last and linear-extrapolation baselines). See `nano_vla/README.md`.
