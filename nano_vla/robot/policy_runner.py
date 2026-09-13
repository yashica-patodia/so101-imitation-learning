"""Hardware-free online policy: keeps the rolling window the model was trained on and
produces the next second of joint targets.

    runner = PolicyRunner("outputs/kaggle_v1/dino_policy/best.pt")
    every 0.1 s:            runner.push_state(joints)              # 6 floats, raw units
    every 0.2 s:            runner.push_frame("front", rgb_uint8)  # per camera, H x W x 3
    when runner.ready:      chunk = runner.predict()               # [HORIZON, 6] raw joint targets

Image features are computed only for the newest frame (DINOv2 for the baseline
arm, nothing extra for the pixel arm), which is the caching idea from OctoSense
Sec. 4.3 in miniature. The same class is used by the dry run, the live loop, and
the replay test, so the online path is exactly the offline path.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from nano_vla import config as C
from nano_vla.models.mae import Encoder
from nano_vla.models.pixel_mae import PixelEncoder
from nano_vla.models.probe import ActionProbe
from nano_vla.train.common import pick_device


class PolicyRunner:
    def __init__(self, checkpoint: str, device: torch.device | None = None):
        self.device = device or pick_device()
        ck = torch.load(checkpoint, map_location="cpu")
        cfg = ck["cfg"]
        self.cfg = cfg
        self.kind = cfg.get("kind") or "feature"
        self.cams: list[str] = list(cfg["cams"])
        if self.kind == "pixel":
            self.enc = PixelEncoder(cfg["n_cams"], cfg["n_patches"], cfg["d"], cfg["depth"], cfg.get("heads", 6))
        else:
            self.enc = Encoder(cfg["n_cams"], cfg["n_patches"], cfg["feat_dim"], cfg["d"], cfg["depth"])
        self.enc.load_state_dict(ck["encoder"])
        self.probe = ActionProbe(cfg["d"])
        self.probe.load_state_dict(ck["probe"])
        self.enc.to(self.device).eval()
        self.probe.to(self.device).eval()
        self.stats = {k: torch.as_tensor(np.array(v, dtype=np.float32), device=self.device) for k, v in ck["stats"].items()}
        self.target = ck.get("target", "state")
        self._dino = None
        self.frames: dict[str, deque] = {c: deque(maxlen=C.T_IMG) for c in self.cams}
        self.states: deque = deque(maxlen=C.T_JOINT)

    # ---------------------------------------------------------------- inputs
    def push_state(self, joints) -> None:
        self.states.append(np.asarray(joints, dtype=np.float32))

    def push_frame(self, cam: str, rgb: np.ndarray) -> None:
        """rgb: uint8 [H, W, 3]. For the feature arm this runs DINOv2 on this one frame."""
        if self.kind == "pixel":
            from nano_vla.data.video import resize_batch, to_uint8_hwc

            self.frames[cam].append(to_uint8_hwc(resize_batch(rgb[None], C.FRAME_RES))[0])
        else:
            from nano_vla.data.dino import load_dino, patch_features

            if self._dino is None:
                self._dino = load_dino(self.device)
            self.frames[cam].append(patch_features(self._dino, rgb[None], self.device)[0])

    def push_feature(self, cam: str, feat: np.ndarray) -> None:
        """Precomputed DINOv2 features [64, 384] (used by the replay test)."""
        self.frames[cam].append(np.asarray(feat, dtype=np.float32))

    @property
    def ready(self) -> bool:
        return len(self.states) == C.T_JOINT and all(len(q) == C.T_IMG for q in self.frames.values())

    def reset(self) -> None:
        self.states.clear()
        for q in self.frames.values():
            q.clear()

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def predict(self) -> np.ndarray:
        """Returns [HORIZON, 6] joint targets in raw units for the next 1 s at 10 Hz."""
        assert self.ready, "buffers not full yet"
        st = torch.from_numpy(np.stack(self.states)).to(self.device)
        st_n = (st - self.stats["state_mean"]) / self.stats["state_std"]
        if self.kind == "pixel":
            frm = torch.from_numpy(np.stack([np.stack(self.frames[c]) for c in self.cams])).float().to(self.device) / 255.0
            x = frm.permute(0, 1, 4, 2, 3)[None]  # [1, C, T, 3, R, R]
        else:
            x = torch.from_numpy(np.stack([np.stack(self.frames[c]) for c in self.cams]).astype(np.float32)).to(self.device)[None]  # [1, C, T, P, D]
        tokens, _, keep = self.enc(x, st_n[None])
        pred_n = self.probe(tokens, keep, st_n[-1][None])[0]
        pre = "state" if self.target == "state" else "action"
        return (pred_n * self.stats[pre + "_std"] + self.stats[pre + "_mean"]).cpu().numpy()
