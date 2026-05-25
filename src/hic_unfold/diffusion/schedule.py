"""Cosine noise schedule and v-prediction utilities.

Notation matches Salimans & Ho (2022): with alpha_bar(t) the cumulative product
of (1 - beta_t),

    x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * eps
    v   = sqrt(alpha_bar) * eps - sqrt(1 - alpha_bar) * x_0
    x_0 = sqrt(alpha_bar) * x_t - sqrt(1 - alpha_bar) * v
"""

from __future__ import annotations

import math

import torch


def make_cosine_schedule(T: int = 1000, s: float = 0.008,
                         device: str | torch.device = "cpu") -> torch.Tensor:
    """Return alpha_bar of shape (T+1,) on a cosine schedule (Nichol & Dhariwal 2021).
    alpha_bars[0] == 1 (clean), alpha_bars[T] ~ 0 (pure noise)."""
    steps = torch.linspace(0, 1, T + 1, device=device)
    f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    return alpha_bar.clamp(min=1e-8, max=1.0)


def _broadcast(coef: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return coef.view(-1, *([1] * (like.dim() - 1)))


def q_sample(x0: torch.Tensor, t_idx: torch.Tensor, alpha_bars: torch.Tensor,
             noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward diffusion. Returns (x_t, v_target)."""
    if noise is None:
        noise = torch.randn_like(x0)
    a_bar = alpha_bars[t_idx]
    alpha = _broadcast(a_bar.sqrt(), x0)
    sigma = _broadcast((1 - a_bar).sqrt(), x0)
    x_t = alpha * x0 + sigma * noise
    v = alpha * noise - sigma * x0
    return x_t, v


def v_to_x0(x_t: torch.Tensor, v: torch.Tensor, t_idx: torch.Tensor,
            alpha_bars: torch.Tensor) -> torch.Tensor:
    a_bar = alpha_bars[t_idx]
    alpha = _broadcast(a_bar.sqrt(), x_t)
    sigma = _broadcast((1 - a_bar).sqrt(), x_t)
    return alpha * x_t - sigma * v


def v_to_eps(x_t: torch.Tensor, v: torch.Tensor, t_idx: torch.Tensor,
             alpha_bars: torch.Tensor) -> torch.Tensor:
    a_bar = alpha_bars[t_idx]
    alpha = _broadcast(a_bar.sqrt(), x_t)
    sigma = _broadcast((1 - a_bar).sqrt(), x_t)
    return sigma * x_t + alpha * v
