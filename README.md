# SO-101 Imitation Learning — Foundation Project

End-to-end imitation learning on a LeRobot SO-100/SO-101 leader-follower arm pair:
teleoperation → data collection → ACT policy training → real-robot evaluation.

**Hardware:** SO-100/SO-101 leader + follower pair, 2 cameras (overhead + wrist).

## Goal

Train an [ACT (Action Chunking Transformer)](https://arxiv.org/abs/2304.13705) policy
on ~50 teleoperated demonstrations of a pick-and-place task, deploy it on the real
arm, and report success rate over 20+ evaluation trials.

## Project checklist

- [ ] **1. Environment setup** — install `lerobot`, verify USB connections to both arms
- [ ] **2. Motor setup & calibration** — configure motor IDs, calibrate leader and follower
- [ ] **3. Camera setup** — find camera indices, mount overhead + wrist views, verify streams
- [ ] **4. Teleoperation test** — leader drives follower smoothly at full control rate
- [ ] **5. Task & scene design** — fixed workspace, chosen props, consistent lighting
- [ ] **6. Record dataset** — ~50 episodes, push to Hugging Face Hub
- [ ] **7. Sanity-check data** — visualize episodes, prune bad ones
- [ ] **8. Train ACT** — on free GPU (Kaggle/Colab), track with W&B
- [ ] **9. Evaluate on robot** — 20+ trials, success-rate table, record videos
- [ ] **10. Write up results** — README with videos, numbers, and failure analysis

## Repository layout (planned)

```
configs/        # teleop, recording, and training configs
scripts/        # setup, recording, and eval helper scripts
notebooks/      # training notebook for Kaggle/Colab
eval/           # evaluation protocol + results
docs/           # setup notes, calibration log, lessons learned
```

## Results

_To be filled in: dataset link, success-rate table, demo videos._
