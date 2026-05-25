"""1D stochastic loop-extrusion simulator (Stage-1, Step-1 of the build plan).

A loop-extruding factor (LEF, cohesin) is a pair of legs that load together at
one lattice site and then translocate in opposite directions, extruding a loop.
Legs stall at oriented CTCF barriers and the LEF unloads after a finite lifetime,
re-loading elsewhere. A snapshot of all LEF anchor positions in one realisation
defines the cell's loop configuration `z`.

Orientation convention used here:
    ctcf_left_stop[i]  = probability that a *left-moving* leg sitting at site i
                         stalls when trying to move to i-1. Set by a forward (→)
                         CTCF at site i.
    ctcf_right_stop[i] = probability that a *right-moving* leg sitting at site i
                         stalls when trying to move to i+1. Set by a reverse (←)
                         CTCF at site i.

Convergent CTCFs (forward on the left, reverse on the right) trap a LEF between
them and produce a corner peak in the averaged loop matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class LoopExtrusionConfig:
    N: int
    num_lefs: int
    processivity: float
    ctcf_left_stop: np.ndarray
    ctcf_right_stop: np.ndarray
    load_prob: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.N <= 1:
            raise ValueError("N must be > 1")
        if self.num_lefs < 1:
            raise ValueError("num_lefs must be >= 1")
        if self.num_lefs > self.N:
            raise ValueError("num_lefs cannot exceed N")
        if self.processivity <= 0:
            raise ValueError("processivity must be > 0")
        self.ctcf_left_stop = np.asarray(self.ctcf_left_stop, dtype=np.float64)
        self.ctcf_right_stop = np.asarray(self.ctcf_right_stop, dtype=np.float64)
        if self.ctcf_left_stop.shape != (self.N,):
            raise ValueError(f"ctcf_left_stop must have shape ({self.N},)")
        if self.ctcf_right_stop.shape != (self.N,):
            raise ValueError(f"ctcf_right_stop must have shape ({self.N},)")
        if ((self.ctcf_left_stop < 0).any() or (self.ctcf_left_stop > 1).any()
                or (self.ctcf_right_stop < 0).any() or (self.ctcf_right_stop > 1).any()):
            raise ValueError("CTCF stop probabilities must lie in [0, 1]")
        if self.load_prob is None:
            self.load_prob = np.ones(self.N, dtype=np.float64) / self.N
        else:
            lp = np.asarray(self.load_prob, dtype=np.float64)
            if lp.shape != (self.N,):
                raise ValueError(f"load_prob must have shape ({self.N},)")
            if (lp < 0).any() or lp.sum() <= 0:
                raise ValueError("load_prob must be nonnegative with positive sum")
            self.load_prob = lp / lp.sum()


def _sample_free_site(occupied: np.ndarray, load_prob: np.ndarray,
                      rng: np.random.Generator) -> int:
    free = np.where(occupied == 0)[0]
    if free.size == 0:
        return -1
    w = load_prob[free]
    s = w.sum()
    if s <= 0:
        return int(rng.choice(free))
    return int(rng.choice(free, p=w / s))


def _initial_load(cfg: LoopExtrusionConfig,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = np.zeros(cfg.num_lefs, dtype=np.int32)
    right = np.zeros(cfg.num_lefs, dtype=np.int32)
    occupied = np.zeros(cfg.N, dtype=np.int32)
    for k in range(cfg.num_lefs):
        pos = _sample_free_site(occupied, cfg.load_prob, rng)
        if pos < 0:
            raise RuntimeError("ran out of free sites during initial load")
        left[k] = pos
        right[k] = pos
        occupied[pos] = 1
    return left, right, occupied


def _step(left: np.ndarray, right: np.ndarray, occupied: np.ndarray,
          cfg: LoopExtrusionConfig, rng: np.random.Generator) -> None:
    p_unload = 1.0 / cfg.processivity
    for k in range(cfg.num_lefs):
        L = int(left[k])
        R = int(right[k])

        if rng.random() < p_unload:
            occupied[L] -= 1
            if R != L:
                occupied[R] -= 1
            new_pos = _sample_free_site(occupied, cfg.load_prob, rng)
            if new_pos >= 0:
                left[k] = new_pos
                right[k] = new_pos
                occupied[new_pos] += 1
            continue

        # Left leg moves leftward; CTCF at current site can stall it.
        if L > 0 and occupied[L - 1] == 0:
            if rng.random() > cfg.ctcf_left_stop[L]:
                occupied[L] -= 1
                occupied[L - 1] += 1
                left[k] = L - 1

        # Right leg moves rightward; CTCF at current site can stall it.
        R = int(right[k])
        if R < cfg.N - 1 and occupied[R + 1] == 0:
            if rng.random() > cfg.ctcf_right_stop[R]:
                occupied[R] -= 1
                occupied[R + 1] += 1
                right[k] = R + 1


def run_to_snapshot(cfg: LoopExtrusionConfig, num_steps: int,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Equilibrate for `num_steps` and return one snapshot (left, right) of LEF positions."""
    left, right, occupied = _initial_load(cfg, rng)
    for _ in range(num_steps):
        _step(left, right, occupied, cfg, rng)
    return left.copy(), right.copy()


def snapshot_to_loop_matrix(left: np.ndarray, right: np.ndarray, N: int) -> np.ndarray:
    """Sparse symmetric N×N matrix with entry 1 at each LEF anchor pair (L,R), L≠R."""
    z = np.zeros((N, N), dtype=np.int8)
    for L, R in zip(left.tolist(), right.tolist()):
        if L != R:
            z[L, R] = 1
            z[R, L] = 1
    return z


def snapshot_to_extrusion_footprint(left: np.ndarray, right: np.ndarray, N: int) -> np.ndarray:
    """Sum of indicator squares {(i,j) : L≤i,j≤R} over LEFs — a proxy for the contact
    block a loop induces. Averaged over snapshots this reveals TAD-like structure."""
    M = np.zeros((N, N), dtype=np.float32)
    for L, R in zip(left.tolist(), right.tolist()):
        if L != R:
            M[L:R + 1, L:R + 1] += 1.0
    return M


@dataclass
class EnsembleResult:
    snapshots: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    avg_loop_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    avg_footprint: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


def simulate_ensemble(cfg: LoopExtrusionConfig, num_cells: int,
                      steps_per_cell: int, seed: int = 0) -> EnsembleResult:
    """Run `num_cells` independent realisations; return per-cell snapshots and
    the ensemble averages of the loop matrix and extrusion footprint."""
    rng = np.random.default_rng(seed)
    snapshots: list[tuple[np.ndarray, np.ndarray]] = []
    avg_loop = np.zeros((cfg.N, cfg.N), dtype=np.float64)
    avg_fp = np.zeros((cfg.N, cfg.N), dtype=np.float64)
    for _ in range(num_cells):
        L, R = run_to_snapshot(cfg, steps_per_cell, rng)
        snapshots.append((L, R))
        avg_loop += snapshot_to_loop_matrix(L, R, cfg.N)
        avg_fp += snapshot_to_extrusion_footprint(L, R, cfg.N)
    avg_loop /= num_cells
    avg_fp /= num_cells
    return EnsembleResult(snapshots=snapshots, avg_loop_matrix=avg_loop, avg_footprint=avg_fp)
