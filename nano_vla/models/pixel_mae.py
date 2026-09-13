"""Pixel-level multimodal MAE: our own image encoder, learned from the robot's raw
frames plus the joint stream, with no pretrained vision model anywhere.

Input per window: frames [B, C, T, 3, R, R] (R = 160) and joints [B, Tj, 6].
Each frame is cut into 16x16 patches (100 per frame), so a window is
C*T*100 image tokens + Tj joint tokens. Masking is the OctoSense scheme from
models/mae.py (tubes across time, Dirichlet split across cameras, Bernoulli for
joints). The encoder runs on visible tokens only (gathered, MAE-style); a
lighter decoder fills in mask tokens and regresses per-patch normalized pixels
(the original MAE's norm-pix target) and raw normalized joints.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nano_vla import config as C
from nano_vla.models.layers import Block
from nano_vla.models.mae import sample_mask

PATCH = 16


def patchify(frm: torch.Tensor) -> torch.Tensor:
    """[B, C, T, 3, R, R] -> [B, C, T, P, 3*PATCH*PATCH]."""
    b, c, t, ch, r, _ = frm.shape
    g = r // PATCH
    x = frm.reshape(b, c, t, ch, g, PATCH, g, PATCH).permute(0, 1, 2, 4, 6, 3, 5, 7)
    return x.reshape(b, c, t, g * g, ch * PATCH * PATCH)


def unpatchify(p: torch.Tensor, r: int = C.FRAME_RES) -> torch.Tensor:
    """[B, C, T, P, 3*PATCH*PATCH] -> [B, C, T, 3, R, R]."""
    b, c, t, n, d = p.shape
    g = r // PATCH
    x = p.reshape(b, c, t, g, g, 3, PATCH, PATCH).permute(0, 1, 2, 5, 3, 6, 4, 7)
    return x.reshape(b, c, t, 3, r, r)


def norm_pix(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-patch normalization of the target (MAE 'norm_pix_loss')."""
    m = p.mean(-1, keepdim=True)
    v = p.var(-1, keepdim=True)
    return (p - m) / (v + eps).sqrt()


class PixelTokenizer(nn.Module):
    def __init__(self, n_cams: int, n_patches: int, d: int):
        super().__init__()
        self.n_cams, self.n_patches = n_cams, n_patches
        self.patch_proj = nn.Linear(3 * PATCH * PATCH, d)
        self.joint_proj = nn.Linear(C.JOINT_DIM, d)
        self.time_img = nn.Embedding(C.T_IMG, d)
        self.time_joint = nn.Embedding(C.T_JOINT, d)
        self.pos = nn.Embedding(n_patches, d)
        self.modality = nn.Embedding(n_cams + 1, d)

    def forward(self, patches: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """patches [B,C,T,P,768], state [B,Tj,6] -> tokens [B, C*T*P + Tj, d]."""
        b, c, t, p, _ = patches.shape
        x = (
            self.patch_proj(patches)
            + self.time_img.weight[None, None, :, None]
            + self.pos.weight[None, None, None, :]
            + self.modality.weight[:c][None, :, None, None]
        ).reshape(b, c * t * p, -1)
        xj = self.joint_proj(state) + self.time_joint.weight[None] + self.modality.weight[c][None, None]
        return torch.cat([x, xj], dim=1)

    def positional_only(self, b: int, c: int, t: int, p: int, tj: int, device) -> torch.Tensor:
        """Position + time + modality embeddings with zero content, for decoder mask tokens."""
        zero_p = torch.zeros(b, c, t, p, 3 * PATCH * PATCH, device=device)
        zero_j = torch.zeros(b, tj, C.JOINT_DIM, device=device)
        return self.forward(zero_p, zero_j)


class PixelEncoder(nn.Module):
    """Same interface as models.mae.Encoder: forward(x, state, keep) -> (tokens, mod_id, keep).
    Here x is raw frames. Encoder attention runs over visible tokens only (gather + pad),
    output is scattered back to full length with masked rows zeroed."""

    def __init__(self, n_cams: int, n_patches: int, d: int = 384, depth: int = 8, heads: int = 6):
        super().__init__()
        self.tok = PixelTokenizer(n_cams, n_patches, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(d)
        self.d = d
        self.n_cams, self.n_patches = n_cams, n_patches

    def forward(self, frm: torch.Tensor, state: torch.Tensor, keep: torch.Tensor | None = None):
        patches = patchify(frm)
        x = self.tok(patches, state)  # [B, N, d]
        b, n, d = x.shape
        if keep is None:
            keep = torch.ones(b, n, dtype=torch.bool, device=x.device)
        # gather visible tokens, pad to the longest visible count in the batch
        counts = keep.sum(1)
        max_k = int(counts.max())
        order = torch.argsort((~keep).int(), dim=1, stable=True)[:, :max_k]  # visible first
        xv = torch.gather(x, 1, order[..., None].expand(-1, -1, d))
        pad = torch.arange(max_k, device=x.device)[None] >= counts[:, None]
        for blk in self.blocks:
            xv = blk(xv, key_padding_mask=pad)
        xv = self.norm(xv) * (~pad)[..., None]
        out = torch.zeros_like(x)
        out.scatter_(1, order[..., None].expand(-1, -1, d), xv)
        mod_id = torch.cat([torch.arange(self.n_cams, device=x.device).repeat_interleave(frm.shape[2] * self.n_patches), torch.full((state.shape[1],), self.n_cams, device=x.device)])
        return out, mod_id, keep


class PixelMAE(nn.Module):
    def __init__(self, n_cams: int, n_patches: int, d: int = 384, depth: int = 8, heads: int = 6, dec_d: int = 192, dec_depth: int = 4, dec_heads: int = 6):
        super().__init__()
        self.enc = PixelEncoder(n_cams, n_patches, d, depth, heads)
        self.n_cams, self.n_patches = n_cams, n_patches
        self.to_dec = nn.Linear(d, dec_d)
        self.dec_pos = nn.Linear(d, dec_d)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_d))
        self.dec_blocks = nn.ModuleList(Block(dec_d, dec_heads) for _ in range(dec_depth))
        self.dec_norm = nn.LayerNorm(dec_d)
        self.head_pix = nn.Linear(dec_d, 3 * PATCH * PATCH)
        self.head_joint = nn.Linear(dec_d, C.JOINT_DIM)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, frm, state, keep=None):
        b, c, t, _, r, _ = frm.shape
        p = self.n_patches
        if keep is None:
            keep = sample_mask(b, c, p, state.shape[1], frm.device)
        z, _, keep = self.enc(frm, state, keep)
        pos = self.enc.tok.positional_only(b, c, t, p, state.shape[1], frm.device)
        h = torch.where(keep[..., None], self.to_dec(z), self.mask_token + self.dec_pos(pos))
        for blk in self.dec_blocks:
            h = blk(h)
        h = self.dec_norm(h)
        n_img = c * t * p
        pred_pix = self.head_pix(h[:, :n_img]).reshape(b, c, t, p, -1)
        pred_joint = self.head_joint(h[:, n_img:])
        return pred_pix, pred_joint, keep

    def loss(self, frm, state, keep=None, visible_weight: float = 0.1):
        pred_pix, pred_joint, keep = self(frm, state, keep)
        b, c, t, _, _, _ = frm.shape
        target = norm_pix(patchify(frm))
        l_img = (pred_pix - target).abs().mean(-1).reshape(b, -1)
        l_joint = (pred_joint - state).abs().mean(-1)
        per_tok = torch.cat([l_img, l_joint], dim=1)
        masked = ~keep
        loss_masked = (per_tok * masked).sum() / masked.sum().clamp(min=1)
        loss_visible = (per_tok * keep).sum() / keep.sum().clamp(min=1)
        loss = loss_masked + visible_weight * loss_visible  # masked-first, as in the original MAE
        return loss, {"loss": loss.item(), "masked": loss_masked.item(), "img": l_img.mean().item(), "joint": l_joint.mean().item()}

    @torch.no_grad()
    def reconstruct(self, frm, state, keep=None):
        """For visualization: returns (masked input, reconstruction, original) as [B,C,T,3,R,R] in 0..1.
        Reconstruction un-normalizes each predicted patch with the original patch's mean/std."""
        pred_pix, _, keep = self(frm, state, keep)
        b, c, t, _, r, _ = frm.shape
        pt = patchify(frm)
        m, s = pt.mean(-1, keepdim=True), (pt.var(-1, keepdim=True) + 1e-6).sqrt()
        rec = unpatchify(pred_pix * s + m, r).clamp(0, 1)
        keep_img = keep[:, : c * t * self.n_patches].reshape(b, c, t, self.n_patches, 1).float()
        masked_in = unpatchify(pt * keep_img + 0.5 * (1 - keep_img), r)
        return masked_in, rec, frm
