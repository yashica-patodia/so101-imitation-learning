# 18-step plan in six phases (revised 2026-09-12)

Live version with owners, durations and done-criteria:
https://claude.ai/code/artifact/67a24f15-0e67-4c18-aca2-ad92e917fc2c

**Phase 1, data (can start now).** 1 download 5hadytru grasp_1 then grasp_2 ·
2 fix decoder (pyav), extract 2 episodes · 3 trim at gripper-close, store DINOv2
features + raw 160x160 frames + joints per episode.

**Phase 2, baseline encoder (now).** 4 train DINOv2 policy, beat hold-last and
linear baselines offline · 5 prepare ACT notebook for Kaggle (user launches).

**Phase 3, our encoder (now, background).** 6 write pixel MAE (depth 8, width
384, patch 16, tube + Dirichlet masking) · 7 pretrain on Kaggle, 4–8 h.

**Phase 4, own rig (needs arm).** 8 record 60–100 grasp episodes (+ lift in
last third) · 9 extract with same trim · 10 fine-tune DINOv2 policy on them.

**Phase 5, robot (needs user).** 11 write 10 Hz loop with speed limit + kill
switch, dry-run torque off · 12 first live attempt · 13 Level A 20 trials,
DINOv2 arm; record the 20 positions.

**Phase 6, comparison.** 14 probe on frozen pixel MAE, same head/data ·
15 optional end-to-end fine-tune (waits on prof) · 16 MAE arm on robot, same
20 positions · 17 Level B for the better arm (retrim close + 1.5 s) ·
18 report: table with DINOv2 / our MAE / ACT rows, demo clip, failure analysis.

Waits on the prof: step 15, and whether step 7 adds Open-X data.
Clock: day 1 steps 1–10; day 2 steps 11–14; day 3 steps 15–18.
