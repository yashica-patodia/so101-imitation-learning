"""Shared building blocks: transformer block and the token embedder."""

from __future__ import annotations

import torch
import torch.nn as nn

from nano_vla import config as C


class Block(nn.Module):
    def __init__(self, d: int, heads: int, mlp: int = 4, drop: float = 0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.n1(x)
        x = x + self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class Tokenizer(nn.Module):
    """Projects per-patch features and joint rows into d-dim tokens and adds summed
    (time, patch position, modality) embeddings: a learned stand-in for OctoSense's
    4-D RoPE over (t, u1, u2, s)."""

    def __init__(self, n_cams: int, n_patches: int, feat_dim: int, d: int):
        super().__init__()
        self.n_cams, self.n_patches = n_cams, n_patches
        self.img_proj = nn.Linear(feat_dim, d)
        self.joint_proj = nn.Linear(C.JOINT_DIM, d)
        self.time_img = nn.Embedding(C.T_IMG, d)
        self.time_joint = nn.Embedding(C.T_JOINT, d)
        self.pos = nn.Embedding(n_patches, d)
        self.modality = nn.Embedding(n_cams + 1, d)  # cameras 0..C-1, joints = C

    def forward(self, img: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """img [B,C,T,P,D], state [B,Tj,6] -> tokens [B,N,d], modality id per token [N]."""
        b, c, t, p, _ = img.shape
        x_img = (
            self.img_proj(img)
            + self.time_img.weight[None, None, :, None]
            + self.pos.weight[None, None, None, :]
            + self.modality.weight[:c][None, :, None, None]
        ).reshape(b, c * t * p, -1)
        x_j = self.joint_proj(state) + self.time_joint.weight[None] + self.modality.weight[c][None, None]
        mod_id = torch.cat([torch.arange(c, device=img.device).repeat_interleave(t * p), torch.full((state.shape[1],), c, device=img.device)])
        return torch.cat([x_img, x_j], dim=1), mod_id
