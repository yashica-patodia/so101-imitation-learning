"""Action probe: HORIZON learned queries cross-attend to encoder tokens and output a
residual over the last observed state, in normalized joint units."""

from __future__ import annotations

import torch
import torch.nn as nn

from nano_vla import config as C


class ActionProbe(nn.Module):
    def __init__(self, d: int = C.D_MODEL, heads: int = C.HEADS):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(C.HORIZON, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, C.JOINT_DIM))

    def forward(self, tokens: torch.Tensor, keep: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        q = self.queries[None].expand(tokens.shape[0], -1, -1)
        h, _ = self.attn(q, tokens, tokens, key_padding_mask=~keep, need_weights=False)
        return last[:, None, :] + self.mlp(self.norm(h))  # [B, HORIZON, 6]
