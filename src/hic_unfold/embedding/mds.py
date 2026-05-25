"""Classical (metric) MDS — Section 5.4 of the spec.

Given a pairwise distance matrix D (N x N), recover positions X (N x dim) by
double-centring B = -1/2 J D^2 J and taking the top-`dim` eigen-components.

If D is a valid Euclidean distance matrix from `dim`-dim positions, the top
`dim` eigenvalues of B are non-negative and the remaining eigenvalues are zero;
non-zero residual eigenvalues quantify how far D is from being realisable in
`dim` Euclidean dimensions.
"""

from __future__ import annotations

import numpy as np


def classical_mds(D: np.ndarray, dim: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, eigvals) where X is (N, dim) and eigvals are all N eigenvalues
    of the double-centred matrix sorted in descending order."""
    N = D.shape[0]
    if D.shape != (N, N):
        raise ValueError("D must be square")
    D2 = D ** 2
    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * (J @ D2 @ J)
    B = 0.5 * (B + B.T)  # numerical symmetry
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    top_vals = np.clip(eigvals[:dim], a_min=0.0, a_max=None)
    X = eigvecs[:, :dim] * np.sqrt(top_vals)[None, :]
    return X, eigvals


def mds_residual(D: np.ndarray, dim: int = 3) -> dict:
    """How close is D to being a valid dim-dimensional Euclidean distance matrix?

    Returns:
        rmse: root-mean-square deviation between D and the distance matrix of
              MDS-embedded coordinates.
        relative_rmse: rmse / mean(D[upper_off_diag]).
        eig_ratio: sum(top dim eigenvalues) / sum(|all eigenvalues|).
                   1.0 means perfectly dim-dimensional Euclidean.
        eigvals: all N eigenvalues of the double-centred matrix.
    """
    X, eigvals = classical_mds(D, dim=dim)
    diff = X[:, None, :] - X[None, :, :]
    D_recon = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices_from(D, k=1)
    err = D[iu] - D_recon[iu]
    rmse = float(np.sqrt((err ** 2).mean()))
    mean_d = float(D[iu].mean()) or 1.0
    top = float(np.clip(eigvals[:dim], 0.0, None).sum())
    total = float(np.abs(eigvals).sum()) or 1.0
    return {
        "rmse": rmse,
        "relative_rmse": rmse / mean_d,
        "eig_ratio": top / total,
        "eigvals": eigvals,
    }
