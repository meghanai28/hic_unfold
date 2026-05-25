"""Deterministic DDIM sampler for a v-prediction denoiser."""

from __future__ import annotations

import torch

from .schedule import v_to_eps, v_to_x0


@torch.no_grad()
def ddim_sample(denoiser, z: torch.Tensor, c: torch.Tensor,
                alpha_bars: torch.Tensor, n_steps: int = 50,
                shape: tuple[int, ...] | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """Run DDIM (eta=0) for n_steps from t=T to t=0, returning the denoised x_0.

    z: (B, 1, N, N); c: (B, d_c, N); alpha_bars: (T+1,).
    """
    device = z.device
    T = alpha_bars.numel() - 1
    B = z.size(0)
    N = z.size(-1)
    if shape is None:
        shape = (B, 1, N, N)

    timesteps = torch.linspace(T, 0, n_steps + 1, device=device).round().long()

    x = torch.randn(*shape, device=device, generator=generator)
    for i in range(n_steps):
        t_now = timesteps[i]
        t_next = timesteps[i + 1]
        t_batch = t_now.expand(B)
        v = denoiser(x, z, c, t_batch)
        x0 = v_to_x0(x, v, t_batch, alpha_bars)
        eps = v_to_eps(x, v, t_batch, alpha_bars)
        a_next = alpha_bars[t_next]
        alpha_next = a_next.sqrt()
        sigma_next = (1 - a_next).sqrt()
        x = alpha_next * x0 + sigma_next * eps

    x = 0.5 * (x + x.transpose(-1, -2))
    return x
