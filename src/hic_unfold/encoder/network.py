"""Loop encoder q(z | x): infer the loop-extrusion configuration from a
distance matrix.

Architecture mirrors the conditional denoiser (Section 4.4) — pair featurization
on per-locus conditioning, a genomic-separation embedding, then a 2D conv
ResNet — but without FiLM time conditioning since this is a one-shot supervised
mapping x -> logits, not a diffusion step. Output is symmetrized and the head
is zero-initialised so untrained predictions are 0 (sigmoid -> 0.5 everywhere).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    def __init__(self, d_h: int, dilation: int, n_groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, d_h)
        self.conv1 = nn.Conv2d(d_h, d_h, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(n_groups, d_h)
        self.conv2 = nn.Conv2d(d_h, d_h, kernel_size=3, padding=dilation, dilation=dilation)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class LoopEncoder(nn.Module):
    def __init__(self, N: int, d_c: int = 16, d_pair: int = 32, d_sep: int = 16,
                 d_h: int = 96, dilations: tuple[int, ...] = (1, 2, 4, 8, 1)):
        super().__init__()
        self.N = N
        self.pair_proj = nn.Conv2d(3 * d_c, d_pair, kernel_size=1)
        self.sep_embed = nn.Embedding(N, d_sep)

        d_in = 1 + d_pair + d_sep
        self.in_proj = nn.Conv2d(d_in, d_h, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([_ResBlock(d_h, dilation=d) for d in dilations])
        self.out_norm = nn.GroupNorm(8, d_h)
        self.out_proj = nn.Conv2d(d_h, 1, kernel_size=1)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _pair_features(self, c: torch.Tensor) -> torch.Tensor:
        N = c.size(-1)
        Si = c.unsqueeze(-1).expand(-1, -1, -1, N)
        Sj = c.unsqueeze(-2).expand(-1, -1, N, -1)
        return torch.cat([Si, Sj, (Si - Sj).abs()], dim=1)

    def _sep_features(self, B: int, device: torch.device) -> torch.Tensor:
        N = self.N
        i = torch.arange(N, device=device)
        sep = (i[:, None] - i[None, :]).abs()
        emb = self.sep_embed(sep).permute(2, 0, 1).unsqueeze(0)
        return emb.expand(B, -1, -1, -1)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Returns raw logits with the same shape as x. Apply sigmoid to get
        per-pair loop probabilities.

        x: (B, 1, N, N) — standardized log1p(D).
        c: (B, d_c, N) — per-locus conditioning features.
        """
        B = x.size(0)
        pair = self.pair_proj(self._pair_features(c))
        sep = self._sep_features(B, x.device)
        h = torch.cat([x, pair, sep], dim=1)
        h = self.in_proj(h)
        for blk in self.blocks:
            h = blk(h)
        out = self.out_proj(self.out_norm(h))
        out = 0.5 * (out + out.transpose(-1, -2))
        return out
