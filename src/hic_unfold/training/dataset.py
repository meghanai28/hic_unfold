"""Simulated (z, x) dataset generator and PyTorch Dataset.

Each cell uses an independently randomized CTCF arrangement (number, position,
orientation, strength) and LEF parameters (count, processivity). Running the
loop-extrusion simulator once yields the cell's loop matrix z; running the
Gaussian polymer once given z yields the conformation's distance matrix D.

This gives a diverse training corpus where (z, D) varies meaningfully across
examples, so the denoiser must learn the conditional map z -> D rather than
memorize a single configuration.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

from hic_unfold.polymer.gaussian import (
    PolymerConfig,
    sample_distance_matrix,
)
from hic_unfold.simulator.loop_extrusion import (
    LoopExtrusionConfig,
    run_to_snapshot,
    snapshot_to_loop_matrix,
)


def generate_random_cell(N: int, rng: np.random.Generator,
                         le_steps: int = 300,
                         backbone_k: float = 1.0,
                         loop_k: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    """One randomized cell -> (z, D). z and D are float32 N x N matrices."""
    n_ctcf = int(rng.integers(1, 5))
    ctcf_left = np.zeros(N, dtype=np.float64)
    ctcf_right = np.zeros(N, dtype=np.float64)
    margin = max(2, N // 20)
    for _ in range(n_ctcf):
        pos = int(rng.integers(margin, N - margin))
        strength = float(rng.uniform(0.7, 1.0))
        if rng.random() < 0.5:
            ctcf_left[pos] = max(ctcf_left[pos], strength)
        else:
            ctcf_right[pos] = max(ctcf_right[pos], strength)

    n_lefs = int(rng.integers(1, 4))
    processivity = float(rng.uniform(200.0, 500.0))

    le_cfg = LoopExtrusionConfig(
        N=N, num_lefs=n_lefs, processivity=processivity,
        ctcf_left_stop=ctcf_left, ctcf_right_stop=ctcf_right,
    )
    poly_cfg = PolymerConfig(backbone_k=backbone_k, loop_k=loop_k)

    L, R = run_to_snapshot(le_cfg, le_steps, rng)
    z = snapshot_to_loop_matrix(L, R, N).astype(np.float32)
    D, _ = sample_distance_matrix(z, N, rng, poly_cfg)
    return z, D.astype(np.float32)


def generate_dataset(num_cells: int, N: int, save_path: str | Path,
                     seed: int = 0, le_steps: int = 300,
                     log_every: int = 500) -> dict:
    rng = np.random.default_rng(seed)
    Z = np.zeros((num_cells, N, N), dtype=np.float32)
    D = np.zeros((num_cells, N, N), dtype=np.float32)
    for k in range(num_cells):
        zk, dk = generate_random_cell(N, rng, le_steps=le_steps)
        Z[k] = zk
        D[k] = dk
        if log_every and (k + 1) % log_every == 0:
            print(f"  generated {k + 1}/{num_cells}")

    logD = np.log1p(D)
    mu = float(logD.mean())
    sigma = float(logD.std())

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        z=Z, D=D,
        mu=np.array(mu), sigma=np.array(sigma),
        N=np.array(N), num_cells=np.array(num_cells),
    )
    return {"path": save_path, "N": N, "num_cells": num_cells, "mu": mu, "sigma": sigma}


class SimulatedDataset:
    """Lazy wrapper around a saved .npz; behaves like a torch.utils.data.Dataset.

    Returns (z_tensor, x_tensor) per item, with x already standardized using the
    stored mu, sigma:
        x = (log1p(D) - mu) / sigma
    """

    def __init__(self, path: str | Path):
        import torch  # local import so the module is usable without torch installed
        self._torch = torch
        path = Path(path)
        with np.load(path) as f:
            self.z = f["z"].astype(np.float32)
            self.D = f["D"].astype(np.float32)
            self.mu = float(f["mu"])
            self.sigma = float(f["sigma"])
            self.N = int(f["N"])
        x_np = (np.log1p(self.D) - self.mu) / max(self.sigma, 1e-8)
        self.x = x_np.astype(np.float32)

    def __len__(self) -> int:
        return self.z.shape[0]

    def __getitem__(self, idx: int):
        torch = self._torch
        z = torch.from_numpy(self.z[idx])[None]   # (1, N, N)
        x = torch.from_numpy(self.x[idx])[None]   # (1, N, N)
        return z, x


def make_positional_c(N: int, d_c: int, device, dtype=None):
    """Sinusoidal per-locus positional features, shape (1, d_c, N).
    Stub for build step 5 where this will be replaced with sequence/CTCF/ATAC tracks."""
    import torch
    dtype = dtype or torch.float32
    half = d_c // 2
    pos = torch.arange(N, device=device, dtype=dtype)
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device, dtype=dtype) / half)
    args = pos[:, None] * freqs[None, :]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    if emb.size(-1) < d_c:
        pad = torch.zeros(N, d_c - emb.size(-1), device=device, dtype=dtype)
        emb = torch.cat([emb, pad], dim=-1)
    return emb.T.unsqueeze(0).contiguous()
