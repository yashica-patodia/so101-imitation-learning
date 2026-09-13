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
