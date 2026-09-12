"""Stage 1: pretrain the nano multimodal MAE on precomputed features.

Usage:
    python -m nano_vla.train_mae --features data/features/hbseong_pos5 --out outputs/nano_vla/mae \
        --epochs 30 --batch 32
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import EpisodeFeatures, WindowDataset, split_episodes
from .model import MAE


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True, help="one or more feature dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--pool", type=int, default=2, help="pool DINO 16x16 grid by this factor")
    ap.add_argument("--stride", type=float, default=0.3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    torch.manual_seed(0)

    train_sets, val_sets, stats_src = [], [], []
    for fdir in args.features:
        n = len(list(Path(fdir).glob("episode_*.npz")))
        tr_eps, va_eps = split_episodes(n, args.val_frac)
        tr = EpisodeFeatures(fdir, tr_eps, args.pool)
        va = EpisodeFeatures(fdir, va_eps, args.pool)
        train_sets.append(tr)
        val_sets.append(va)
        stats_src.append(tr)
    # Joint normalization from the training episodes of all feature dirs.
    s = np.concatenate([e["state"] for f in stats_src for e in f.episodes])
    a = np.concatenate([e["action"] for f in stats_src for e in f.episodes])
    stats = {"state_mean": s.mean(0), "state_std": s.std(0) + 1e-6, "action_mean": a.mean(0), "action_std": a.std(0) + 1e-6}
    np.savez(out / "stats.npz", **stats)

    train_ds = torch.utils.data.ConcatDataset([WindowDataset(f, stats, args.stride) for f in train_sets])
    val_ds = torch.utils.data.ConcatDataset([WindowDataset(f, stats, 1.0) for f in val_sets])
    n_cams, n_patches, feat_dim = len(train_sets[0].cams), train_sets[0].n_patches, train_sets[0].feat_dim
    print(f"train windows={len(train_ds)} val windows={len(val_ds)} cams={n_cams} patches={n_patches} D={feat_dim} device={device}")

    dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True)
    vdl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model = MAE(n_cams, n_patches, feat_dim, d=args.d, depth=args.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MAE params: {n_params / 1e6:.2f}M (encoder {sum(p.numel() for p in model.enc.parameters()) / 1e6:.2f}M)")
    cfg = {**vars(args), "n_cams": n_cams, "n_patches": n_patches, "feat_dim": feat_dim, "cams": train_sets[0].cams}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    total = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.1)
    best = float("inf")
    log = open(out / "log.jsonl", "a")
    step = 0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        agg = {}
        for batch in dl:
            img = batch["img"].to(device)
            state = batch["state"].to(device)
            loss, m = model.loss(img, state)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
        tr = {k: v / len(dl) for k, v in agg.items()}

        model.eval()
        vagg = {}
        torch.manual_seed(1234)  # fixed masks for a comparable val number
        with torch.no_grad():
            for batch in vdl:
                _, m = model.loss(batch["img"].to(device), batch["state"].to(device))
                for k, v in m.items():
                    vagg[k] = vagg.get(k, 0.0) + v
        torch.manual_seed(epoch + 1)
        va = {k: v / max(1, len(vdl)) for k, v in vagg.items()}
        rec = {"epoch": epoch, "train": tr, "val": va, "sec": time.time() - t0}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"ep {epoch:3d}  train loss {tr['loss']:.4f} (masked {tr['masked']:.4f})  val masked {va['masked']:.4f}  img {va['img']:.4f} joint {va['joint']:.4f}  {rec['sec']:.0f}s")
        torch.save({"model": model.state_dict(), "cfg": cfg}, out / "last.pt")
        if va["masked"] < best:
            best = va["masked"]
            torch.save({"model": model.state_dict(), "cfg": cfg}, out / "best.pt")
    print("best val masked loss", best)


if __name__ == "__main__":
    main()
