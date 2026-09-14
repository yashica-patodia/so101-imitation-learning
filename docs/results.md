# Results log

## 2026-09-13 — offline, grasp_2 (780 episodes, Level A), Kaggle T4, notebook so101-train v1

Held-out = last 10% of episodes (815 windows). Error = mean absolute error in raw
joint units (0–100 scale) over the 6 joints, per horizon step (0.1 s each).

| step (s ahead) | DINOv2 direct policy | frozen MAE probe | hold-last | linear |
|---|---|---|---|---|
| 0.1 | 1.28 | 1.48 | 1.69 | 0.86 |
| 0.3 | 3.23 | 3.56 | 5.02 | 4.01 |
| 0.5 | 4.84 | 5.24 | 8.35 | 8.39 |
| 1.0 | 8.49 | 8.95 | 15.97 | 21.83 |
| **mean** | **5.09** | **5.45** | 9.00 | 10.29 |

Per joint (direct policy, mean over horizon): pan 5.5, lift 5.8, elbow 6.8,
wrist_flex 5.0, wrist_roll 4.3, gripper 3.1.

Training: direct policy 20 epochs, 265 s/epoch, best at epoch 13, train loss
kept falling to 0.007 while val plateaued from epoch 5 (overfits). Feature MAE
30 epochs, val masked L1 1.34 → 0.71. Frozen probe 20 epochs, best at epoch 6.

Reading: both beat hold-last at every step and linear from 0.2 s on; linear
wins only at 0.1 s, as expected. The frozen-MAE probe is within 7% of the
end-to-end policy, so the masked pretraining keeps most of the control-relevant
information. Next: fine-tune on our own episodes, then the robot loop.

## 2026-09-14 — pixel MAE (ours) vs DINOv2, offline, grasp_2 Level A, notebook so101-pixel-mae v1

Pixel MAE pretraining: 60 epochs, 185 s/epoch on T4 (3.1 h), val masked L1 on
per-patch-normalized pixels 0.41 (joint 0.10). Frozen probe: 20 epochs, 474 s/epoch,
best at epoch 8.

| step (s ahead) | DINOv2 features, direct policy (end-to-end) | DINOv2 features, frozen feature-MAE + probe | **pixel MAE from scratch, frozen + probe (ours)** | hold-last | linear |
|---|---|---|---|---|---|
| 0.1 | 1.28 | 1.48 | 1.39 | 1.69 | 0.86 |
| 0.5 | 4.84 | 5.24 | 5.12 | 8.35 | 8.39 |
| 1.0 | 8.49 | 8.95 | 9.01 | 15.97 | 21.83 |
| **mean** | **5.09** | 5.45 | **5.38** | 9.00 | 10.29 |

Per joint (ours, mean over horizon): pan 5.7, lift 5.6, elbow 7.1, wrist_flex 5.5,
wrist_roll 4.7, gripper 3.6.

Reading: under the same frozen-encoder + probe protocol, the encoder learned from
six hours of the robot's own pixels (5.38) edges out the one built on DINOv2
features (5.45). The end-to-end DINOv2 policy (5.09) remains the best single
number because its encoder is trained on the task. Both trained rows beat hold-last
at every step and linear from 0.2 s on. Next: both arms fine-tuned on our own
episodes, then the same 20 positions on the robot.
