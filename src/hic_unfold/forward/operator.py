"""Stage-2 forward operator (Section 5.1 of the spec).

A single-cell distance matrix D contributes contact entries
    c(D)[i,j] = sigmoid((d0 - D[i,j]) / tau)
to the bulk map. The bulk prediction H_pred is the ensemble average of c over
a batch of M conformations. d0 and tau are nuisance parameters calibrated so
that applying the operator to real single-cell distance matrices reproduces a
real (or pseudo-) bulk contact map.

The operator is intentionally backend-polymorphic — it accepts numpy arrays
(for offline calibration) and torch tensors (so it can sit inside the guided-
sampling loop in step 8 with gradients flowing through d0/tau and through D).
"""

from __future__ import annotations

import numpy as np


def _is_torch_tensor(x) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return isinstance(x, torch.Tensor)


def soft_contact(D, d0: float = 500.0, tau: float = 100.0):
    """Differentiable contact: sigmoid((d0 - D) / tau).

    For torch tensors, returns a torch tensor on the same device with gradients
    flowing through D, d0, and tau. For numpy arrays, returns a numpy array.
    """
    if _is_torch_tensor(D):
        import torch
        return torch.sigmoid((d0 - D) / tau)
    arg = np.clip((np.asarray(D) - d0) / tau, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(arg))


def apply_forward(D_batch, d0: float = 500.0, tau: float = 100.0):
    """Per-cell soft contact + ensemble mean. D_batch shape (M, N, N)."""
    C = soft_contact(D_batch, d0=d0, tau=tau)
    if _is_torch_tensor(C):
        return C.mean(dim=0)
    return C.mean(axis=0)


def calibrate_d0_tau(D_population: np.ndarray, H_target: np.ndarray,
                     d0_grid: np.ndarray, tau_grid: np.ndarray,
                     mask: np.ndarray | None = None) -> dict:
    """Grid-search (d0, tau) to minimise MSE between apply_forward(D_population)
    and H_target.

    mask: optional boolean (N, N) mask; if given, only those entries contribute
          to the loss (useful for ignoring the diagonal or boundary).

    Returns a dict with keys: d0, tau, mse, mse_grid, pearson.
    """
    D = np.asarray(D_population, dtype=np.float32)
    Ht = np.asarray(H_target, dtype=np.float32)
    mse_grid = np.zeros((len(d0_grid), len(tau_grid)))
    pcc_grid = np.zeros_like(mse_grid)
    iu = np.triu_indices(Ht.shape[-1], k=1)
    Ht_flat = Ht[iu]
    for i, d0 in enumerate(d0_grid):
        for j, tau in enumerate(tau_grid):
            Hp = apply_forward(D, d0=float(d0), tau=float(tau))
            diff = Hp - Ht
            if mask is not None:
                diff = diff[mask]
            mse_grid[i, j] = float((diff ** 2).mean())
            pcc_grid[i, j] = float(np.corrcoef(Hp[iu], Ht_flat)[0, 1])
    bi, bj = np.unravel_index(int(mse_grid.argmin()), mse_grid.shape)
    return {
        "d0": float(d0_grid[bi]),
        "tau": float(tau_grid[bj]),
        "mse": float(mse_grid[bi, bj]),
        "pearson": float(pcc_grid[bi, bj]),
        "mse_grid": mse_grid,
        "pearson_grid": pcc_grid,
        "d0_grid": np.asarray(d0_grid),
        "tau_grid": np.asarray(tau_grid),
    }
