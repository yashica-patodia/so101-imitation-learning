"""Stage 2: train the action probe on an encoder and evaluate offline.

Three modes, all through the same script:
    --mae PATH               frozen pretrained encoder + probe (representation test)
    --mae PATH --unfreeze    pretrained init, fine-tune everything
    --scratch                random-init encoder, trained end to end (direct policy)

    python -m nano_vla.train.train_probe --mae outputs/mae/best.pt --features data/features/grasp_1 --out outputs/probe
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from nano_vla import config as C
from nano_vla.models.mae import MAE, Encoder
from nano_vla.models.probe import ActionProbe
from nano_vla.train.common import JsonlLog, build_datasets, load_stats, loaders, pick_device, save_json, save_stats, seed_all
from nano_vla.train.evaluate import evaluate, format_table


def load_encoder(path: str, device) -> tuple[Encoder, dict]:
    ck = torch.load(path, map_location="cpu")
    cfg = ck["cfg"]
    mae = MAE(cfg["n_cams"], cfg["n_patches"], cfg["feat_dim"], d=cfg["d"], depth=cfg["depth"])
    mae.load_state_dict(ck["model"])
    return mae.enc.to(device), cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mae", default=None, help="pretrained MAE checkpoint")
    ap.add_argument("--scratch", action="store_true", help="random-init encoder, train end to end")
    ap.add_argument("--unfreeze", action="store_true", help="fine-tune the encoder too (lr x0.1)")
    ap.add_argument("--init-probe", default=None, help="probe checkpoint to fine-tune from")
    ap.add_argument("--stats", default=None, help="stats.npz to reuse (fine-tuning); default: from --mae dir or recomputed")
    ap.add_argument("--target", default="state", choices=["state", "action"])
    ap.add_argument("--level", default="A", choices=["A", "B"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--stride", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--d", type=int, default=C.D_MODEL)
    ap.add_argument("--depth", type=int, default=C.DEPTH)
    args = ap.parse_args()
    if not args.mae and not args.scratch:
        ap.error("give --mae PATH or --scratch")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    seed_all(0)

    stats = None
    if args.stats:
        stats = load_stats(Path(args.stats))
    elif args.mae and (Path(args.mae).parent / "stats.npz").exists():
        stats = load_stats(Path(args.mae).parent / "stats.npz")
    train_ds, val_ds, stats, first = build_datasets(args.features, args.level, stats, args.stride, args.val_frac, args.target)
    save_stats(out, stats)
    dl, vdl = loaders(train_ds, val_ds, args.batch)
    print(f"train windows={len(train_ds)} val windows={len(val_ds)} device={device}")

    if args.mae:
        enc, cfg = load_encoder(args.mae, device)
        train_enc = args.unfreeze
    else:
        cfg = {"n_cams": len(first.cams), "n_patches": first.n_patches, "feat_dim": first.feat_dim, "d": args.d, "depth": args.depth, "cams": first.cams}
        enc = Encoder(cfg["n_cams"], cfg["n_patches"], cfg["feat_dim"], args.d, args.depth).to(device)
        train_enc = True
    probe = ActionProbe(cfg["d"]).to(device)
    if args.init_probe:
        probe.load_state_dict(torch.load(args.init_probe, map_location="cpu")["probe"])
    params = [{"params": probe.parameters(), "lr": args.lr}]
    if train_enc:
        params.append({"params": enc.parameters(), "lr": args.lr * (1.0 if args.scratch else 0.1)})
    else:
        for p in enc.parameters():
            p.requires_grad_(False)
    save_json(out / "config.json", {**vars(args), **cfg, "train_encoder": train_enc})
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in params], total_steps=args.epochs * len(dl), pct_start=0.1)

    log, best = JsonlLog(out / "log.jsonl"), float("inf")
    for epoch in range(args.epochs):
        probe.train()
        enc.train(train_enc)
        t0, tot = time.time(), 0.0
        for batch in dl:
            img, state = batch["img"].to(device), batch["state"].to(device)
            if train_enc:
                tokens, _, keep = enc(img, state)
            else:
                with torch.no_grad():
                    tokens, _, keep = enc(img, state)
            pred = probe(tokens, keep, batch["last"].to(device))
            loss = torch.nn.functional.smooth_l1_loss(pred, batch["target"].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        probe.eval()
        enc.eval()
        err = evaluate(enc, probe, vdl, stats, device, args.target)
        rec = {"epoch": epoch, "train_loss": tot / len(dl), "val_mae": {k: float(v.mean()) for k, v in err.items()}, "sec": time.time() - t0}
        log.write(rec)
        print(f"ep {epoch:3d} train {rec['train_loss']:.4f}  val MAE probe {err['probe'].mean():.3f} | hold {err['hold'].mean():.3f} | linear {err['linear'].mean():.3f}  {rec['sec']:.0f}s")
        ck = {"probe": probe.state_dict(), "encoder": enc.state_dict(), "cfg": cfg, "stats": {k: v.tolist() for k, v in stats.items()}, "target": args.target}
        torch.save(ck, out / "last.pt")
        if err["probe"].mean() < best:
            best = err["probe"].mean()
            torch.save(ck, out / "best.pt")
            (out / "eval.txt").write_text(format_table(err))
    print("\nbest checkpoint, held-out MAE in raw joint units:\n" + (out / "eval.txt").read_text())


if __name__ == "__main__":
    main()
