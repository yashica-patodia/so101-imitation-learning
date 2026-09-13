"""Stage 1: masked-autoencoder pretraining.

    --kind feature   MAE over frozen DINOv2 feature tokens + joints   (models/mae.py)
    --kind pixel     MAE over raw 160x160 frames + joints, no pretrained vision (models/pixel_mae.py)

    python -m nano_vla.train.train_mae --kind pixel --features /kaggle/input/.../grasp_2 --out outputs/pixel_mae --epochs 60
Supports --resume to continue from <out>/last.pt (Kaggle sessions cap at ~9 h).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from nano_vla import config as C
from nano_vla.models.mae import MAE
from nano_vla.models.pixel_mae import PATCH, PixelMAE
from nano_vla.train.common import JsonlLog, build_datasets, loaders, pick_device, save_json, save_stats, seed_all


def build_model(kind: str, first, args) -> tuple[torch.nn.Module, dict]:
    n_cams = len(first.cams)
    if kind == "feature":
        cfg = {"kind": "feature", "n_cams": n_cams, "n_patches": first.n_patches, "feat_dim": first.feat_dim, "d": args.d, "depth": args.depth}
        return MAE(n_cams, first.n_patches, first.feat_dim, d=args.d, depth=args.depth), cfg
    n_patches = (C.FRAME_RES // PATCH) ** 2
    cfg = {"kind": "pixel", "n_cams": n_cams, "n_patches": n_patches, "d": args.d, "depth": args.depth, "heads": args.heads, "dec_d": args.dec_d, "dec_depth": args.dec_depth, "dec_heads": args.heads}
    return PixelMAE(n_cams, n_patches, d=args.d, depth=args.depth, heads=args.heads, dec_d=args.dec_d, dec_depth=args.dec_depth, dec_heads=args.heads), cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="feature", choices=["feature", "pixel"])
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", default="A", choices=["A", "B"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="default 3e-4 feature, 1.5e-4 pixel")
    ap.add_argument("--d", type=int, default=None, help="default 256 feature, 384 pixel")
    ap.add_argument("--depth", type=int, default=None, help="default 4 feature, 8 pixel")
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--dec-d", type=int, default=192)
    ap.add_argument("--dec-depth", type=int, default=4)
    ap.add_argument("--stride", type=float, default=0.3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    pixel = args.kind == "pixel"
    args.lr = args.lr or (1.5e-4 if pixel else 3e-4)
    args.d = args.d or (384 if pixel else C.D_MODEL)
    args.depth = args.depth or (8 if pixel else C.DEPTH)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    seed_all(0)
    mode = "pixel" if pixel else "dino"
    key = "frm" if pixel else "img"
    train_ds, val_ds, stats, first = build_datasets(args.features, args.level, None, args.stride, args.val_frac, "state", mode)
    save_stats(out, stats)
    dl, vdl = loaders(train_ds, val_ds, args.batch, args.workers)
    model, cfg = build_model(args.kind, first, args)
    model = model.to(device)
    cfg = {**cfg, "cams": first.cams, "level": args.level, "mode": mode}
    save_json(out / "config.json", {**vars(args), **cfg})
    print(f"kind={args.kind} train windows={len(train_ds)} val windows={len(val_ds)} device={device} params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(dl), pct_start=0.1)
    start_epoch, best = 0, float("inf")
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best = ck["epoch"] + 1, ck.get("best", best)
        print(f"resumed at epoch {start_epoch}", flush=True)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    log = JsonlLog(out / "log.jsonl")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, agg = time.time(), {}
        for batch in dl:
            x, state = batch[key].to(device, non_blocking=True), batch["state"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                loss, m = model.loss(x, state)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
        tr = {k: v / len(dl) for k, v in agg.items()}
        model.eval()
        vagg = {}
        torch.manual_seed(1234)
        with torch.no_grad():
            for batch in vdl:
                _, m = model.loss(batch[key].to(device), batch["state"].to(device))
                for k, v in m.items():
                    vagg[k] = vagg.get(k, 0.0) + v
        torch.manual_seed(epoch + 1)
        va = {k: v / max(1, len(vdl)) for k, v in vagg.items()}
        log.write({"epoch": epoch, "train": tr, "val": va, "sec": time.time() - t0})
        print(f"ep {epoch:3d}  train {tr['loss']:.4f} (masked {tr['masked']:.4f})  val masked {va['masked']:.4f} img {va['img']:.4f} joint {va['joint']:.4f}  {time.time() - t0:.0f}s", flush=True)
        if va["masked"] < best:
            best = va["masked"]
            torch.save({"model": model.state_dict(), "cfg": cfg}, out / "best.pt")
        torch.save({"model": model.state_dict(), "cfg": cfg, "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch, "best": best}, out / "last.pt")
    print("best val masked loss", best)


if __name__ == "__main__":
    main()
