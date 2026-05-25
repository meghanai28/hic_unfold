"""Maximum-entropy reweighting (Section 5.3 of the spec).

Given M conformations sampled from the prior, with per-sample contact maps
C_m (N, N), find per-pair Lagrange multipliers lambda (N, N) symmetric such
that the reweighted ensemble average matches a target bulk map H:

    w_m = exp(<lambda, C_m>) / Z(lambda)
    sum_m w_m * C_m = H

Dual formulation (convex in lambda; we minimise):

    L(lambda) = log Z(lambda) - <lambda, H>
              = logsumexp_m(<lambda, C_m>) - <lambda, H>
    dL/dlambda = E_q[C] - H

We solve with Adam, optionally adding an L2 ridge that discourages individual
multipliers from blowing up when the prior is far from H.
"""

from __future__ import annotations

import numpy as np
import torch


def effective_sample_size(w: np.ndarray) -> float:
    """Kish ESS: 1 / sum(w^2). Equals M when uniform, 1 when degenerate."""
    w = np.asarray(w, dtype=np.float64)
    s2 = float((w ** 2).sum())
    return 1.0 / max(s2, 1e-30)


def maxent_reweight(C_samples: np.ndarray, H_target: np.ndarray,
                    lr: float = 0.05, num_steps: int = 1500,
                    l2: float = 0.0, device: str | None = None,
                    log_every: int = 0) -> dict:
    """Solve maximum-entropy reweighting in the dual.

    Returns dict with:
        lambda     : (N, N) symmetric multipliers
        weights    : (M,) sample weights summing to 1
        eff_M      : effective sample size (Kish)
        H_pred     : (N, N) reweighted contact map (should match H_target)
        fit_mse    : MSE between H_pred and H_target on the upper triangle
        fit_pearson: Pearson correlation of H_pred and H_target on upper tri
        losses     : list of dual-loss values per step
    """
    C_samples = np.asarray(C_samples, dtype=np.float32)
    H_target = np.asarray(H_target, dtype=np.float32)
    M, N, _ = C_samples.shape
    assert H_target.shape == (N, N)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    iu = np.triu_indices(N, k=1)
    K = iu[0].size

    C_up = torch.tensor(C_samples[:, iu[0], iu[1]], device=device)      # (M, K)
    H_up = torch.tensor(H_target[iu], device=device)                    # (K,)
    lam = torch.zeros(K, device=device, requires_grad=True)

    opt = torch.optim.Adam([lam], lr=lr)
    losses: list[float] = []
    for step in range(num_steps):
        opt.zero_grad()
        logits = C_up @ lam
        loss = torch.logsumexp(logits, dim=0) - (lam * H_up).sum()
        if l2 > 0:
            loss = loss + l2 * (lam ** 2).sum()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if log_every and (step + 1) % log_every == 0:
            with torch.no_grad():
                logw = logits - torch.logsumexp(logits, dim=0)
                w_now = logw.exp()
                C_pred = (w_now[:, None] * C_up).sum(dim=0)
                fit = float(((C_pred - H_up) ** 2).mean().item())
                ess = float(1.0 / (w_now ** 2).sum().item())
            print(f"  step {step+1:5d}: loss={loss.item():.4f}, fit MSE={fit:.6f}, ESS={ess:.1f}")

    with torch.no_grad():
        logits = C_up @ lam
        logw = logits - torch.logsumexp(logits, dim=0)
        w = logw.exp().cpu().numpy().astype(np.float64)
        lam_up = lam.detach().cpu().numpy()

    lam_full = np.zeros((N, N), dtype=np.float32)
    lam_full[iu] = lam_up
    lam_full[iu[1], iu[0]] = lam_up

    H_pred = (w[:, None, None] * C_samples).sum(axis=0)
    fit_mse = float(((H_pred[iu] - H_target[iu]) ** 2).mean())
    fit_pcc = float(np.corrcoef(H_pred[iu], H_target[iu])[0, 1])

    return {
        "lambda": lam_full,
        "weights": w,
        "eff_M": effective_sample_size(w),
        "H_pred": H_pred,
        "fit_mse": fit_mse,
        "fit_pearson": fit_pcc,
        "losses": losses,
    }
