"""Does the policy actually use the images? Evaluate a trained probe checkpoint under
three conditions on the same held-out windows:

    normal     correct images + correct joints
    shuffled   images from a DIFFERENT episode + correct joints
    blind      all image tokens masked out, joints only

If 'shuffled' and 'blind' are about as good as 'normal', the model is predicting from
joint history alone and any claim about the vision encoder is empty.

    python -m nano_vla.train.ablate --checkpoint outputs/kaggle_v1/dino_policy/best.pt --features data/features/grasp_1_part
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from nano_vla import config as C
from nano_vla.data.windows import EpisodeSet, WindowDataset
from nano_vla.models.mae import Encoder
from nano_vla.models.pixel_mae import PixelEncoder
from nano_vla.models.probe import ActionProbe
from nano_vla.train.common import pick_device


def load(checkpoint: str, device):
    ck = torch.load(checkpoint, map_location="cpu")
    cfg = ck["cfg"]
    kind = cfg.get("kind") or "feature"
    if kind == "pixel":
        enc = PixelEncoder(cfg["n_cams"], cfg["n_patches"], cfg["d"], cfg["depth"], cfg.get("heads", 6))
    else:
        enc = Encoder(cfg["n_cams"], cfg["n_patches"], cfg["feat_dim"], cfg["d"], cfg["depth"])
    enc.load_state_dict(ck["encoder"])
    probe = ActionProbe(cfg["d"])
    probe.load_state_dict(ck["probe"])
    stats = {k: np.array(v, dtype=np.float32) for k, v in ck["stats"].items()}
    return enc.to(device).eval(), probe.to(device).eval(), stats, kind, ck.get("target", "state")


@torch.no_grad()
def run(enc, probe, dl, stats, device, key, target, condition: str) -> np.ndarray:
    pre = "state" if target == "state" else "action"
    t_std = torch.as_tensor(stats[pre + "_std"], device=device)
    err, n = 0.0, 0
    for batch in dl:
        x, state, tgt = batch[key].to(device), batch["state"].to(device), batch["target"].to(device)
        ep = batch["episode"]
        keep = None
        if condition == "shuffled":
            # pair every sample with the images of a sample from a different episode
            perm = torch.roll(torch.arange(len(ep)), len(ep) // 2)
            ok = ep[perm] != ep
            x = torch.where(ok.view(-1, *[1] * (x.dim() - 1)).to(device), x[perm], x)
            if not ok.all():
                x, state, tgt, last = x[ok], state[ok], tgt[ok], batch["last"][ok]
            else:
                last = batch["last"]
        else:
            last = batch["last"]
        if condition == "blind":
            n_tok_img = x.shape[1] * x.shape[2] * (x.shape[3] if key == "img" else (x.shape[-1] // 16) ** 2)
            keep = torch.zeros(x.shape[0], n_tok_img + state.shape[1], dtype=torch.bool, device=device)
            keep[:, n_tok_img:] = True
        tokens, _, keep = enc(x, state, keep)
        pred = probe(tokens, keep, last.to(device))
        err = err + ((pred - tgt) * t_std).abs().sum(0)
        n += x.shape[0]
    return (err / n).cpu().numpy()  # [HORIZON, 6]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--level", default="A")
    ap.add_argument("--episodes", type=int, default=None, help="use only the LAST n episodes of the dir")
    args = ap.parse_args()
    device = pick_device()
    for ckpt in args.checkpoint:
        enc, probe, stats, kind, target = load(ckpt, device)
        mode, key = ("pixel", "frm") if kind == "pixel" else ("dino", "img")
        import glob

        n = len(glob.glob(args.features + "/episode_*.npz"))
        idx = list(range(n))[-args.episodes :] if args.episodes else None
        ds = WindowDataset(EpisodeSet(args.features, idx, args.level, mode), stats, 0.3, target)
        dl = DataLoader(ds, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(0))
        print(f"\n{ckpt}  ({kind}, {len(ds)} windows from {len(ds.eps.episodes)} episodes)")
        res = {c: run(enc, probe, dl, stats, device, key, target, c) for c in ["normal", "shuffled", "blind"]}
        print("cond      |  mean | 0.1s | 0.5s | 1.0s | " + " ".join(f"{j[:6]:>6s}" for j in C.JOINT_NAMES))
        for c, e in res.items():
            print(f"{c:9s} | {e.mean():5.2f} | {e[0].mean():4.2f} | {e[4].mean():4.2f} | {e[9].mean():4.2f} | " + " ".join(f"{e[:, j].mean():6.2f}" for j in range(6)))


if __name__ == "__main__":
    main()
