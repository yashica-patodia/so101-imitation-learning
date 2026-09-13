"""Multimodal masked autoencoder over image-feature tokens and joint tokens.

Encoder runs over visible tokens only; a light decoder fills in the masked ones
and regresses the original features / joints. Masking follows OctoSense
Sec. 4.3: spatio-temporal tubes for cameras with a Dirichlet split across
modalities, Bernoulli keep for the 1-D joint stream.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nano_vla import config as C
from nano_vla.models.layers import Block, Tokenizer


class Encoder(nn.Module):
    def __init__(self, n_cams: int, n_patches: int, feat_dim: int, d: int = C.D_MODEL, depth: int = C.DEPTH, heads: int = C.HEADS):
        super().__init__()
        self.tok = Tokenizer(n_cams, n_patches, feat_dim, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(d)
        self.d = d

    def forward(self, img, state, keep: torch.Tensor | None = None):
        """keep: bool [B,N], True = visible (None = all). Masked rows are excluded from
        attention via key padding and zeroed in the output."""
        x, mod_id = self.tok(img, state)
        if keep is None:
            keep = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        pad = ~keep
        for blk in self.blocks:
            x = blk(x, key_padding_mask=pad)
        return self.norm(x) * keep[..., None], mod_id, keep


def sample_mask(b: int, n_cams: int, n_patches: int, n_joint: int, device, keep_mean: float = 0.25) -> torch.Tensor:
    """OctoSense-style mask. Cameras: tube masks (a patch position hidden at every time
    step), share per camera from a Dirichlet whose concentration is drawn from
    {0.5, 1, 2} so whole-camera dropout is common. Joints: Bernoulli keep with rate
    0.2 + 0.5 * Beta(2, 5)."""
    gamma = torch.tensor([0.5, 1.0, 2.0])[torch.randint(0, 3, (b,))]
    keeps = []
    for i in range(b):
        probs = torch.distributions.Dirichlet(torch.full((n_cams + 1,), float(gamma[i]))).sample()
        rows = []
        for cam in range(n_cams):
            keep_ratio = min(1.0, keep_mean * (n_cams + 1) * float(probs[cam]))
            tube = torch.zeros(n_patches, dtype=torch.bool)
            tube[torch.randperm(n_patches)[: int(round(keep_ratio * n_patches))]] = True
            rows.append(tube.repeat(C.T_IMG))
        r0 = torch.distributions.Beta(2.0, 5.0).sample()
        rows.append(torch.rand(n_joint) < (0.2 + 0.5 * r0))
        keeps.append(torch.cat(rows))
    return torch.stack(keeps).to(device)


class MAE(nn.Module):
    def __init__(self, n_cams, n_patches, feat_dim, d=C.D_MODEL, depth=C.DEPTH, heads=C.HEADS, dec_d=C.DEC_D, dec_depth=C.DEC_DEPTH):
        super().__init__()
        self.enc = Encoder(n_cams, n_patches, feat_dim, d, depth, heads)
        self.n_cams, self.n_patches = n_cams, n_patches
        self.to_dec = nn.Linear(d, dec_d)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_d))
        self.dec_pos = nn.Linear(d, dec_d)
        self.dec_blocks = nn.ModuleList(Block(dec_d, 4) for _ in range(dec_depth))
        self.dec_norm = nn.LayerNorm(dec_d)
        self.head_img = nn.Linear(dec_d, feat_dim)
        self.head_joint = nn.Linear(dec_d, C.JOINT_DIM)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, img, state, keep=None):
        b, c, t, p, d_in = img.shape
        if keep is None:
            keep = sample_mask(b, c, p, state.shape[1], img.device)
        z, _, keep = self.enc(img, state, keep)
        pos_only, _ = self.enc.tok(torch.zeros_like(img), torch.zeros_like(state))
        h = torch.where(keep[..., None], self.to_dec(z), self.mask_token + self.dec_pos(pos_only))
        for blk in self.dec_blocks:
            h = blk(h)
        h = self.dec_norm(h)
        n_img = c * t * p
        return self.head_img(h[:, :n_img]).reshape(b, c, t, p, d_in), self.head_joint(h[:, n_img:]), keep

    def loss(self, img, state, keep=None):
        pred_img, pred_joint, keep = self(img, state, keep)
        b, c, t, p, _ = img.shape
        l_img = (pred_img - img).abs().mean(-1).reshape(b, c * t * p)
        l_joint = (pred_joint - state).abs().mean(-1)
        per_tok = torch.cat([l_img, l_joint], dim=1)
        masked = ~keep
        loss = per_tok.mean()  # all tokens, as in the paper (V-JEPA 2.1 finding)
        loss_masked = (per_tok * masked).sum() / masked.sum().clamp(min=1)
        return loss, {"loss": loss.item(), "masked": loss_masked.item(), "img": l_img.mean().item(), "joint": l_joint.mean().item()}
