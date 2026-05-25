"""Conditional denoiser network for p(x | z, c).

Architecture (Section 4.4 of the spec, "sanity-check scale" variant):
    - Per-locus features c (B, d_c, N) -> AlphaFold-style pair representation
      (B, d_pair, N, N) via outer concat + 1x1 conv.
    - Genomic-separation sinusoidal embedding added as additional channels.
    - Loop-latent z concatenated as one channel.
    - Noisy distance image x_t concatenated as one channel.
    - Body: 2D conv ResNet with dilations (1, 2, 4, 8) and FiLM timestep modulation.
    - Output is symmetrized so the predicted v stays in the symmetric subspace.

Predicts v in the v-prediction parameterisation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional embedding. t: integer or float tensor of any shape."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t.float().unsqueeze(-1) * freqs
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class FiLM(nn.Module):
    def __init__(self, d_t: int, d_h: int):
        super().__init__()
        self.proj = nn.Linear(d_t, 2 * d_h)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.proj(t_emb)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.view(-1, x.size(1), 1, 1)
        beta = beta.view(-1, x.size(1), 1, 1)
        return x * (1 + gamma) + beta


class ResBlock(nn.Module):
    def __init__(self, d_h: int, dilation: int, d_t: int, n_groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, d_h)
        self.conv1 = nn.Conv2d(d_h, d_h, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(n_groups, d_h)
        self.conv2 = nn.Conv2d(d_h, d_h, kernel_size=3, padding=dilation, dilation=dilation)
        self.film = FiLM(d_t, d_h)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.film(h, t_emb)
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class Denoiser(nn.Module):
    def __init__(self, N: int, d_c: int = 16, d_pair: int = 32, d_sep: int = 16,
                 d_h: int = 96, d_t: int = 128,
                 dilations: tuple[int, ...] = (1, 2, 4, 8, 1)):
        super().__init__()
        self.N = N
        self.d_t = d_t

        self.time_mlp = nn.Sequential(
            nn.Linear(d_t, d_t), nn.SiLU(), nn.Linear(d_t, d_t),
        )

        self.pair_proj = nn.Conv2d(3 * d_c, d_pair, kernel_size=1)
        self.sep_embed = nn.Embedding(N, d_sep)

        d_in = 1 + d_pair + d_sep + 1
        self.in_proj = nn.Conv2d(d_in, d_h, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResBlock(d_h, d, d_t) for d in dilations])
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

    def forward(self, x_t: torch.Tensor, z: torch.Tensor, c: torch.Tensor,
                t_idx: torch.Tensor) -> torch.Tensor:
        """Returns the predicted v with shape matching x_t.

        Shapes:
            x_t: (B, 1, N, N)
            z:   (B, 1, N, N)
            c:   (B, d_c, N)
            t_idx: (B,)  integer timestep indices
        """
        B = x_t.size(0)
        t_emb = self.time_mlp(sinusoidal_embedding(t_idx, self.d_t))

        pair = self.pair_proj(self._pair_features(c))
        sep = self._sep_features(B, x_t.device)
        h = torch.cat([x_t, pair, sep, z], dim=1)
        h = self.in_proj(h)
        for blk in self.blocks:
            h = blk(h, t_emb)
        out = self.out_proj(self.out_norm(h))
        out = 0.5 * (out + out.transpose(-1, -2))
        return out
