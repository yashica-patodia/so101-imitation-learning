"""Stage 2: train the action probe on a frozen MAE encoder, and evaluate offline.

Reports per-joint mean absolute error (in raw joint units, 0-100 scale on the
SO-101) at each horizon step on held-out episodes, next to two trivial
baselines: "hold last state" and "linear extrapolation of the last two states".
The probe must beat both to be worth deploying.

Usage:
    python -m nano_vla.train_probe --mae outputs/nano_vla/mae/best.pt \
        --features data/features/hbseong_pos5 --out outputs/nano_vla/probe --epochs 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import HORIZON, EpisodeFeatures, WindowDataset, split_episodes
from .model import MAE, ActionProbe

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_mae(path: str, device) -> tuple[MAE, dict]:
    ck = torch.load(path, map_location="cpu")
    cfg = ck["cfg"]
    model = MAE(cfg["n_cams"], cfg["n_patches"], cfg["feat_dim"], d=cfg["d"], depth=cfg["depth"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg


@torch.no_grad()
def evaluate(enc, probe, dl, stats, device) -> dict:
    """Returns MAE in raw units: [HORIZON, 6] for probe, hold-last and linear baselines."""
    a_std = torch.as_tensor(stats["action_std"], device=device)
    a_mean = torch.as_tensor(stats["action_mean"], device=device)
    s_std = torch.as_tensor(stats["state_std"], device=device)
    s_mean = torch.as_tensor(stats["state_mean"], device=device)
    err = {"probe": 0.0, "hold": 0.0, "linear": 0.0}
    n = 0
    for batch in dl:
        img, state, tgt = batch["img"].to(device), batch["state"].to(device), batch["target"].to(device)
        tokens, _, keep = enc(img, state)
        pred = probe(tokens, keep, batch["last"].to(device))
        raw_tgt = tgt * a_std + a_mean
        raw_pred = pred * a_std + a_mean
        raw_state = state * s_std + s_mean
        hold = raw_state[:, -1:, :].expand(-1, HORIZON, -1)
        vel = raw_state[:, -1] - raw_state[:, -2]
        steps = torch.arange(1, HORIZON + 1, device=device, dtype=torch.float32)[None, :, None]
        linear = raw_state[:, -1:, :] + vel[:, None, :] * steps
        err["probe"] = err["probe"] + (raw_pred - raw_tgt).abs().sum(0)
        err["hold"] = err["hold"] + (hold - raw_tgt).abs().sum(0)
        err["linear"] = err["linear"] + (linear - raw_tgt).abs().sum(0)
        n += img.shape[0]
    return {k: (v / n).cpu().numpy() for k, v in err.items()}


def fmt_table(err: dict) -> str:
    lines = ["step | " + " | ".join(f"{k:>7s}" for k in err)]
    for h in range(HORIZON):
        lines.append(f"{h + 1:4d} | " + " | ".join(f"{err[k][h].mean():7.3f}" for k in err))
    lines.append("mean | " + " | ".join(f"{err[k].mean():7.3f}" for k in err))
    lines.append("per joint (probe, mean over horizon): " + ", ".join(f"{n}={err['probe'][:, j].mean():.2f}" for j, n in enumerate(JOINT_NAMES)))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mae", required=True)
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--stride", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--unfreeze", action="store_true", help="also fine-tune the encoder (lower lr)")
    ap.add_argument("--init-probe", default=None, help="probe checkpoint to start from (fine-tuning)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    torch.manual_seed(0)
    mae, cfg = load_mae(args.mae, device)
    enc = mae.enc
    stats = dict(np.load(Path(args.mae).parent / "stats.npz"))

    train_sets, val_sets = [], []
    for fdir in args.features:
        n = len(list(Path(fdir).glob("episode_*.npz")))
        tr_eps, va_eps = split_episodes(n, args.val_frac)
        train_sets.append(WindowDataset(EpisodeFeatures(fdir, tr_eps, cfg["pool"]), stats, args.stride))
        val_sets.append(WindowDataset(EpisodeFeatures(fdir, va_eps, cfg["pool"]), stats, 0.5))
    train_ds, val_ds = torch.utils.data.ConcatDataset(train_sets), torch.utils.data.ConcatDataset(val_sets)
    dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    vdl = DataLoader(val_ds, batch_size=args.batch)
    print(f"train windows={len(train_ds)} val windows={len(val_ds)} device={device}")

    probe = ActionProbe(cfg["d"]).to(device)
    if args.init_probe:
        probe.load_state_dict(torch.load(args.init_probe, map_location="cpu")["probe"])
    params = [{"params": probe.parameters(), "lr": args.lr}]
    if args.unfreeze:
        params.append({"params": enc.parameters(), "lr": args.lr * 0.1})
    else:
        for p in enc.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in params], total_steps=args.epochs * len(dl), pct_start=0.1)

    best = float("inf")
    log = open(out / "log.jsonl", "a")
    for epoch in range(args.epochs):
        probe.train()
        enc.train(args.unfreeze)
        t0, tot = time.time(), 0.0
        for batch in dl:
            img, state = batch["img"].to(device), batch["state"].to(device)
            if args.unfreeze:
                tokens, _, keep = enc(img, state)
            else:
                with torch.no_grad():
                    tokens, _, keep = enc(img, state)
            pred = probe(tokens, keep, batch["last"].to(device))
            loss = torch.nn.functional.smooth_l1_loss(pred, batch["target"].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        probe.eval()
        enc.eval()
        err = evaluate(enc, probe, vdl, stats, device)
        rec = {"epoch": epoch, "train_loss": tot / len(dl), "val_mae": {k: float(v.mean()) for k, v in err.items()}, "sec": time.time() - t0}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"ep {epoch:3d} train {rec['train_loss']:.4f}  val MAE probe {err['probe'].mean():.3f} | hold {err['hold'].mean():.3f} | linear {err['linear'].mean():.3f}  {rec['sec']:.0f}s")
        ck = {"probe": probe.state_dict(), "encoder": enc.state_dict(), "cfg": cfg, "stats": {k: v.tolist() for k, v in stats.items()}}
        torch.save(ck, out / "last.pt")
        if err["probe"].mean() < best:
            best = err["probe"].mean()
            torch.save(ck, out / "best.pt")
            (out / "eval.txt").write_text(fmt_table(err))
    print("\nbest checkpoint, held-out MAE in raw joint units:")
    print((out / "eval.txt").read_text())


if __name__ == "__main__":
    main()
