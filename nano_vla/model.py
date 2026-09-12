"""Nano multimodal MAE (stage 1) and action probe (stage 2).

Early fusion: one transformer over all tokens. Each token carries a summed
positional embedding for (time, patch position, modality), a nano version of
the 4-D RoPE over (t, u1, u2, s) in OctoSense Sec. 4.3.

Modalities: one per camera (image tokens = pooled DINOv2 patch features) plus
one for the joint-state time series.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import HORIZON, JOINT_DIM, T_IMG, T_JOINT


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
    """Projects raw features into d-dim tokens and adds (time, position, modality) embeddings."""

    def __init__(self, n_cams: int, n_patches: int, feat_dim: int, d: int):
        super().__init__()
        self.n_cams, self.n_patches = n_cams, n_patches
        self.img_proj = nn.Linear(feat_dim, d)
        self.joint_proj = nn.Linear(JOINT_DIM, d)
        self.time_img = nn.Embedding(T_IMG, d)
        self.time_joint = nn.Embedding(T_JOINT, d)
        self.pos = nn.Embedding(n_patches, d)
        self.modality = nn.Embedding(n_cams + 1, d)  # cams 0..C-1, joints = C

    def forward(self, img: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """img [B,C,T,P,D], state [B,Tj,6] -> tokens [B, N, d], modality id per token [N]."""
        b, c, t, p, _ = img.shape
        dev = img.device
        x_img = self.img_proj(img)  # [B,C,T,P,d]
        x_img = (
            x_img
            + self.time_img.weight[None, None, :, None]
            + self.pos.weight[None, None, None, :]
            + self.modality.weight[:c][None, :, None, None]
        )
        x_img = x_img.reshape(b, c * t * p, -1)
        x_j = self.joint_proj(state) + self.time_joint.weight[None] + self.modality.weight[c][None, None]
        mod_id = torch.cat(
            [torch.arange(c, device=dev).repeat_interleave(t * p), torch.full((state.shape[1],), c, device=dev)]
        )
        return torch.cat([x_img, x_j], dim=1), mod_id


class Encoder(nn.Module):
    def __init__(self, n_cams: int, n_patches: int, feat_dim: int, d: int = 256, depth: int = 4, heads: int = 4):
        super().__init__()
        self.tok = Tokenizer(n_cams, n_patches, feat_dim, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(d)
        self.d = d

    def forward(self, img, state, keep: torch.Tensor | None = None):
        """keep: bool [B, N] of visible tokens (None = all). Returns (tokens [B,N,d] with masked rows
        zeroed, mod_id [N], keep [B,N])."""
        x, mod_id = self.tok(img, state)
        if keep is None:
            keep = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        # Run attention only among visible tokens by passing masked ones as key padding.
        pad = ~keep
        for blk in self.blocks:
            x = blk(x, key_padding_mask=pad)
        x = self.norm(x) * keep[..., None]
        return x, mod_id, keep


class MAE(nn.Module):
    """Encoder over visible tokens + light decoder that fills in masked ones."""

    def __init__(self, n_cams, n_patches, feat_dim, d=256, depth=4, heads=4, dec_d=128, dec_depth=2):
        super().__init__()
        self.enc = Encoder(n_cams, n_patches, feat_dim, d, depth, heads)
        self.n_cams = n_cams
        self.to_dec = nn.Linear(d, dec_d)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_d))
        self.dec_pos = nn.Linear(d, dec_d)  # re-inject positional info for masked tokens
        self.dec_blocks = nn.ModuleList(Block(dec_d, 4) for _ in range(dec_depth))
        self.dec_norm = nn.LayerNorm(dec_d)
        self.head_img = nn.Linear(dec_d, feat_dim)
        self.head_joint = nn.Linear(dec_d, JOINT_DIM)
        nn.init.normal_(self.mask_token, std=0.02)

    def sample_mask(self, b: int, n_img_per_cam: int, n_joint: int, device) -> torch.Tensor:
        """OctoSense-style masking. Cameras: tube masks (a patch position hidden across all frames),
        ratio from a Dirichlet over modalities. Joints: Bernoulli keep with a Beta-distributed rate."""
        c = self.n_cams
        p_per = n_img_per_cam // T_IMG
        gamma = torch.tensor([0.5, 1.0, 2.0])[torch.randint(0, 3, (b,))]
        keeps = []
        for i in range(b):
            probs = torch.distributions.Dirichlet(torch.full((c + 1,), float(gamma[i]))).sample()
            keep_rows = []
            for cam in range(c):
                # Dirichlet share scaled so the mean keep ratio is 0.25 (75% masked, as in MAE);
                # gamma<1 draws push whole cameras toward fully masked -> robustness to a dropped sensor.
                keep_ratio = min(1.0, 0.25 * (c + 1) * float(probs[cam]))
                n_keep = int(round(keep_ratio * p_per))
                tube = torch.zeros(p_per, dtype=torch.bool)
                tube[torch.randperm(p_per)[:n_keep]] = True
                keep_rows.append(tube.repeat(T_IMG))  # same patch mask at every time step
            r0 = torch.distributions.Beta(2.0, 5.0).sample()
            keep_j = torch.rand(n_joint) < (0.2 + 0.5 * r0)
            keeps.append(torch.cat(keep_rows + [keep_j]))
        return torch.stack(keeps).to(device)

    def forward(self, img, state, keep=None):
        b, c, t, p, d_in = img.shape
        if keep is None:
            keep = self.sample_mask(b, t * p, state.shape[1], img.device)
        z, mod_id, keep = self.enc(img, state, keep)
        # Decoder input: encoded visible tokens, learned mask token + positional embedding elsewhere.
        pos_only, _ = self.enc.tok(torch.zeros_like(img), torch.zeros_like(state))
        h = torch.where(keep[..., None], self.to_dec(z), self.mask_token + self.dec_pos(pos_only))
        for blk in self.dec_blocks:
            h = blk(h)
        h = self.dec_norm(h)
        n_img = c * t * p
        pred_img = self.head_img(h[:, :n_img]).reshape(b, c, t, p, d_in)
        pred_joint = self.head_joint(h[:, n_img:])
        return pred_img, pred_joint, keep

    def loss(self, img, state, keep=None):
        pred_img, pred_joint, keep = self(img, state, keep)
        b, c, t, p, _ = img.shape
        n_img = c * t * p
        l_img = (pred_img - img).abs().mean(-1).reshape(b, n_img)
        l_joint = (pred_joint - state).abs().mean(-1)
        per_tok = torch.cat([l_img, l_joint], dim=1)
        masked = ~keep
        # Loss on all tokens (paper follows V-JEPA 2.1); masked-only reported for monitoring.
        loss = per_tok.mean()
        loss_masked = (per_tok * masked).sum() / masked.sum().clamp(min=1)
        return loss, {"loss": loss.item(), "masked": loss_masked.item(), "img": l_img.mean().item(), "joint": l_joint.mean().item()}


class ActionProbe(nn.Module):
    """Cross-attention probe: HORIZON learned queries attend to frozen encoder tokens, output
    residual over the last observed state (normalized units)."""

    def __init__(self, d: int, heads: int = 4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(HORIZON, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, JOINT_DIM))

    def forward(self, tokens: torch.Tensor, keep: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        q = self.queries[None].expand(tokens.shape[0], -1, -1)
        h, _ = self.attn(q, tokens, tokens, key_padding_mask=~keep, need_weights=False)
        return last[:, None, :] + self.mlp(self.norm(h))  # [B, HORIZON, 6]
