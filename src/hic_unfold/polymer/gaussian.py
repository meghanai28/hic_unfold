"""Gaussian (Rouse-like) polymer with loop-anchor constraints.

Given a loop configuration ``z`` (sparse symmetric N x N from the loop-extrusion
simulator), build a graph on N beads with:
    - backbone springs of strength ``backbone_k`` between every (i, i+1), and
    - loop springs of strength ``loop_k`` between every (i, j) with z[i, j] > 0.

Bead positions X in R^{N x 3} are jointly Gaussian with precision matrix equal
to the graph Laplacian K = D - A (per coordinate). K has a single zero eigenvalue
(global translation); we sample from the pseudoinverse by skipping that mode,
which is equivalent to fixing the centre of mass at the origin.

This polymer is the architecture's classical/HIPPS-style prior — used here as
the *data generator* (the source of (z, x) training pairs for Stage-1 Step 2),
not as the final method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PolymerConfig:
    backbone_k: float = 1.0
    loop_k: float = 10.0
    zero_eig_threshold: float = 1e-8

    def __post_init__(self) -> None:
        if self.backbone_k <= 0:
            raise ValueError("backbone_k must be > 0")
        if self.loop_k <= 0:
            raise ValueError("loop_k must be > 0")


def build_laplacian(z: np.ndarray, N: int, cfg: PolymerConfig) -> np.ndarray:
    """Build the graph Laplacian K = D - A for a chain with backbone + loop springs."""
    if z.shape != (N, N):
        raise ValueError(f"z must have shape ({N}, {N}), got {z.shape}")
    A = np.zeros((N, N), dtype=np.float64)
    idx = np.arange(N - 1)
    A[idx, idx + 1] = cfg.backbone_k
    A[idx + 1, idx] = cfg.backbone_k

    loop_mask = (z != 0).astype(np.float64)
    np.fill_diagonal(loop_mask, 0.0)
    A = A + cfg.loop_k * loop_mask

    K = np.diag(A.sum(axis=1)) - A
    return K


def sample_positions(K: np.ndarray, rng: np.random.Generator,
                     dim: int = 3, zero_eig_threshold: float = 1e-8) -> np.ndarray:
    """Sample X ~ N(0, K^+) per coordinate, where K^+ is the pseudoinverse.
    Skips the zero (translation) mode so the centre of mass is at the origin."""
    eigvals, eigvecs = np.linalg.eigh(K)
    nonzero = eigvals > zero_eig_threshold
    if not nonzero.any():
        raise RuntimeError("Laplacian has no positive eigenvalues; check graph connectivity")
    lam = eigvals[nonzero]
    V = eigvecs[:, nonzero]
    Z = rng.standard_normal((nonzero.sum(), dim))
    X = V @ (Z / np.sqrt(lam)[:, None])
    return X


def positions_to_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix from positions X (N, dim)."""
    diff = X[:, None, :] - X[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def sample_distance_matrix(z: np.ndarray, N: int, rng: np.random.Generator,
                           cfg: PolymerConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """One-shot: build K, sample positions, return (distance_matrix, positions)."""
    cfg = cfg or PolymerConfig()
    K = build_laplacian(z, N, cfg)
    X = sample_positions(K, rng, zero_eig_threshold=cfg.zero_eig_threshold)
    return positions_to_distance_matrix(X), X


def soft_contact(D: np.ndarray, d0: float = 4.0, tau: float = 1.0) -> np.ndarray:
    """Sigmoid soft contact: c(d) = sigmoid((d0 - d) / tau). The differentiable
    forward operator from Section 5.1 of the architecture spec; used here only
    to render distance matrices as Hi-C-style contact maps."""
    return 1.0 / (1.0 + np.exp((D - d0) / tau))
