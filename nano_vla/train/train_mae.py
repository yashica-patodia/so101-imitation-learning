"""Stage 1: pretrain the multimodal MAE on DINOv2-feature tokens + joints.

    python -m nano_vla.train.train_mae --features data/features/grasp_1 --out outputs/mae --epochs 30
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from nano_vla import config as C
from nano_vla.models.mae import MAE
from nano_vla.train.common import JsonlLog, build_datasets, loaders, pick_device, save_json, save_stats, seed_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", default="A", choices=["A", "B"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=C.D_MODEL)
    ap.add_argument("--depth", type=int, default=C.DEPTH)
    ap.add_argument("--stride", type=float, default=0.3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    seed_all(0)
    train_ds, val_ds, stats, first = build_datasets(args.features, args.level, None, args.stride, args.val_frac, "state")
    save_stats(out, stats)
    dl, vdl = loaders(train_ds, val_ds, args.batch, args.workers)
    n_cams, n_patches, feat_dim = len(first.cams), first.n_patches, first.feat_dim
    print(f"train windows={len(train_ds)} val windows={len(val_ds)} cams={n_cams} patches={n_patches} D={feat_dim} device={device}")

    model = MAE(n_cams, n_patches, feat_dim, d=args.d, depth=args.depth).to(device)
    cfg = {**vars(args), "n_cams": n_cams, "n_patches": n_patches, "feat_dim": feat_dim, "cams": first.cams}
    save_json(out / "config.json", cfg)
    print(f"MAE params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(dl), pct_start=0.1)
    log, best = JsonlLog(out / "log.jsonl"), float("inf")
    for epoch in range(args.epochs):
        model.train()
        t0, agg = time.time(), {}
        for batch in dl:
            loss, m = model.loss(batch["img"].to(device), batch["state"].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
        tr = {k: v / len(dl) for k, v in agg.items()}
        model.eval()
        vagg = {}
        torch.manual_seed(1234)  # fixed masks so val numbers are comparable across epochs
        with torch.no_grad():
            for batch in vdl:
                _, m = model.loss(batch["img"].to(device), batch["state"].to(device))
                for k, v in m.items():
                    vagg[k] = vagg.get(k, 0.0) + v
        torch.manual_seed(epoch + 1)
        va = {k: v / max(1, len(vdl)) for k, v in vagg.items()}
        log.write({"epoch": epoch, "train": tr, "val": va, "sec": time.time() - t0})
        print(f"ep {epoch:3d}  train {tr['loss']:.4f} (masked {tr['masked']:.4f})  val masked {va['masked']:.4f} img {va['img']:.4f} joint {va['joint']:.4f}  {time.time() - t0:.0f}s")
        ck = {"model": model.state_dict(), "cfg": cfg}
        torch.save(ck, out / "last.pt")
        if va["masked"] < best:
            best = va["masked"]
            torch.save(ck, out / "best.pt")
    print("best val masked loss", best)


if __name__ == "__main__":
    main()
