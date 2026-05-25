"""Ensemble-consistency-guided DDIM sampler (Section 5.2 of the spec).

Co-evolve M conformations through the reverse diffusion process. At every step
predict the clean conformations, apply the (differentiable) forward operator,
compare the ensemble-averaged contact map to the observed bulk H, and
backpropagate to push every x_t in the direction that reduces the bulk-fit MSE:

    v_hat   = denoiser(x_t, z, c, t)
    x0_hat  = sqrt(a_t) x_t - sqrt(1-a_t) v_hat
    D_hat   = expm1(x0_hat * sigma + mu)                # nm
    H_pred  = mean_m sigmoid((d0 - D_hat_m) / tau)
    loss    = MSE(H_pred, H_obs)
    g       = grad(loss, x_t)
    x_{t-1} = ddim_step(x_t, v_hat) - eta(t) * g

The forward operator gradient is the only signal that couples conformations to
the measured bulk — this is what makes it a *deconvolution* and distinguishes it
from unguided prior sampling (step 5) and from maxent reweighting (step 7).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from ..forward import soft_contact
from .schedule import _broadcast


def guided_ddim_sample(
    denoiser,
    z: torch.Tensor,
    c: torch.Tensor,
    alpha_bars: torch.Tensor,
    H_obs: torch.Tensor,
    *,
    d0: float,
    tau: float,
    mu: float,
    sigma: float,
    n_steps: int = 200,
    eta: float = 5.0,
    eta_schedule: Optional[Callable[[int, int], float]] = None,
    D_max: float = 5000.0,
    log_every: int = 0,
) -> dict:
    """Run guided DDIM. Returns dict with sampled distance matrices and trajectory.

    Args:
        denoiser    : trained Denoiser (set to eval mode beforehand).
        z, c        : conditioning per co-evolving sample.
        alpha_bars  : (T+1,) cosine schedule.
        H_obs       : (N, N) observed bulk contact map.
        d0, tau     : calibrated forward-operator parameters (nm).
        mu, sigma   : log1p(D) standardisation from training.
        n_steps     : number of DDIM steps.
        eta         : guidance scale (multiplier on the gradient step).
        eta_schedule: optional callable (t_now, T) -> float; multiplies eta.
        D_max       : clamp predicted D to this max (nm) for numerical stability.
        log_every   : print fit MSE every N steps (0 to disable).
    """
    device = z.device
    T = alpha_bars.numel() - 1
    M, _, N, _ = z.shape

    timesteps = torch.linspace(T, 0, n_steps + 1, device=device).round().long()
    x = torch.randn(M, 1, N, N, device=device)

    losses: list[float] = []
    grad_norms: list[float] = []

    for i in range(n_steps):
        t_now = timesteps[i]
        t_next = timesteps[i + 1]
        a_now = alpha_bars[t_now]
        a_next = alpha_bars[t_next]
        alpha_now = a_now.sqrt()
        sigma_now = (1.0 - a_now).sqrt()
        alpha_next = a_next.sqrt()
        sigma_next = (1.0 - a_next).sqrt()

        x = x.detach().requires_grad_(True)

        v = denoiser(x, z, c, t_now.expand(M))
        x0 = alpha_now * x - sigma_now * v

        D = torch.expm1(x0.squeeze(1) * sigma + mu)
        D = torch.clamp(D, min=0.0, max=D_max)
        D = 0.5 * (D + D.transpose(-1, -2))

        C_pred = soft_contact(D, d0=d0, tau=tau)         # (M, N, N)
        H_pred = C_pred.mean(dim=0)                       # (N, N)
        loss = ((H_pred - H_obs) ** 2).mean()
        losses.append(float(loss.item()))

        g = torch.autograd.grad(loss, x)[0]
        gn = float(g.flatten().norm().item())
        grad_norms.append(gn)

        eps_pred = sigma_now * x + alpha_now * v
        x_ddim = alpha_next * x0 + sigma_next * eps_pred

        eta_t = eta
        if eta_schedule is not None:
            eta_t = eta * eta_schedule(int(t_now.item()), T)

        x = x_ddim.detach() - eta_t * g.detach()

        if log_every and (i + 1) % log_every == 0:
            print(f"  step {i+1:4d}/{n_steps}: t={int(t_now.item()):4d}, "
                  f"loss={loss.item():.6f}, |g|={gn:.4f}")

    x = 0.5 * (x + x.transpose(-1, -2))
    D_final = torch.expm1(x.squeeze(1) * sigma + mu)
    D_final = torch.clamp(D_final, min=0.0, max=D_max)
    D_final = 0.5 * (D_final + D_final.transpose(-1, -2))
    # zero out diagonal so D_final has zero on diag
    diag_idx = torch.arange(N, device=device)
    D_final[:, diag_idx, diag_idx] = 0.0

    return {
        "x": x.detach(),
        "D": D_final.detach(),
        "losses": losses,
        "grad_norms": grad_norms,
    }
