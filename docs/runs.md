# Runs: what was trained, with which script, on which data

Kaggle notebooks are private to the account `yashicapatodia`; the GitHub links are public.

## Datasets

| Name | Source on Hugging Face | Size | Role |
|---|---|---|---|
| grasp_2 | https://huggingface.co/datasets/5hadytru/so101_grasp_2 | 780 episodes, ~5 h, front (wrist) + overhead cameras | training set for all three models; its last 10% of episodes is the validation split |
| grasp_1 | https://huggingface.co/datasets/5hadytru/so101_grasp_1 | 210 episodes, ~1 h, same rig | held-out test set (never trained on) |

Both are cut at the gripper-close moment and converted to per-episode feature files by
the extraction notebook:

| Step | Kaggle notebook | Notebook source | Script it runs |
|---|---|---|---|
| Download, trim, extract | https://www.kaggle.com/code/yashicapatodia/so101-extract (Version 1 output = grasp_1, 209 episodes; Version 2 output = grasp_2, 780 episodes) | https://github.com/yashica-patodia/so101-imitation-learning/blob/main/notebooks/kaggle_01_extract.ipynb | https://github.com/yashica-patodia/so101-imitation-learning/blob/main/nano_vla/data/extract.py |

## Models

| # | Model | Kaggle notebook (run) | Notebook source | Training scripts | Model code | Trained on | Checkpoint in the output |
|---|---|---|---|---|---|---|---|
| 1 | DINOv2 direct policy: frozen DINOv2 features, small transformer + action head trained end to end | https://www.kaggle.com/code/yashicapatodia/so101-train2 (section 1) | https://github.com/yashica-patodia/so101-imitation-learning/blob/main/notebooks/kaggle_02_train_baseline.ipynb | https://github.com/yashica-patodia/so101-imitation-learning/blob/main/nano_vla/train/train_probe.py with `--scratch` | https://github.com/yashica-patodia/so101-imitation-learning/blob/main/nano_vla/models/mae.py (Encoder), https://github.com/yashica-patodia/so101-imitation-learning/blob/main/nano_vla/models/probe.py | grasp_2 | outputs/dino_policy/best.pt |
| 2 | Frozen feature-MAE + probe: masked autoencoder over DINOv2 feature tokens + joints, then frozen, action head on top | https://www.kaggle.com/code/yashicapatodia/so101-train2 (sections 2 and 3) | https://github.com/yashica-patodia/so101-imitation-learning/blob/main/notebooks/kaggle_02_train_baseline.ipynb | https://github.com/yashica-patodia/so101-imitation-learning/blob/main/nano_vla/train/train_mae.py `--kind feature`, then https://github.com/yashica-patodia/so101-imitation-learning/blob/main/nano_vla/train/train_probe.py `--mae` | https://github.com/yashica-patodia/so101-imitation-learning/blob/main/nano_vla/models/mae.py, https://github.com/yashica-patodia/so101-imitation-learning/blob/main/nano_vla/models/probe.py | grasp_2 | outputs/mae/best.pt, outputs/mae_probe/best.pt |
| 3 | Pixel MAE from scratch + probe: masked autoencoder over raw 160x160 frames + joints, no pretrained vision, then frozen, same action head | https://www.kaggle.com/code/yashicapatodia/so101-pixel-mae | https://github.com/yashica-patodia/so101-imitation-learning/blob/main/notebooks/kaggle_03_pixel_mae.ipynb | https://github.com/yashica-patodia/so101-imitation-learning/blob/main/nano_vla/train/train_mae.py `--kind pixel`, then https://github.com/yashica-patodia/so101-imitation-learning/blob/main/nano_vla/train/train_probe.py `--mae` | https://github.com/yashica-patodia/so101-imitation-learning/blob/main/nano_vla/models/pixel_mae.py, https://github.com/yashica-patodia/so101-imitation-learning/blob/main/nano_vla/models/probe.py | grasp_2 | outputs/pixel_mae/best.pt, outputs/pixel_probe/best.pt |

Shared by all three: https://github.com/yashica-patodia/so101-imitation-learning/blob/main/nano_vla/data/windows.py (1 s window → next 1 s of joints),
https://github.com/yashica-patodia/so101-imitation-learning/blob/main/nano_vla/train/evaluate.py (error vs hold-last and linear), https://github.com/yashica-patodia/so101-imitation-learning/blob/main/nano_vla/config.py.

## Checks

| Check | Kaggle notebook | Notebook source | Script | Data |
|---|---|---|---|---|
| Do the models use the images? (correct / swapped / no images) | not yet run on Kaggle; run locally on 2026-09-19 for models 1 and 2 | https://github.com/yashica-patodia/so101-imitation-learning/blob/main/notebooks/kaggle_04_ablation.ipynb | https://github.com/yashica-patodia/so101-imitation-learning/blob/main/nano_vla/train/ablate.py | grasp_1, first 75 episodes |

Numbers for every run are in https://github.com/yashica-patodia/so101-imitation-learning/blob/main/docs/results.md.
