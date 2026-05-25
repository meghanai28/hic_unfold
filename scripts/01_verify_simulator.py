"""Verification for Step-1 of the build plan.

Place two convergent CTCFs on a lattice, run many independent realisations of
the loop-extrusion process, and confirm that averaging snapshots reveals:
    1. a corner peak in the averaged loop matrix at the CTCF anchor pair, and
    2. a TAD-like enriched block in the averaged extrusion footprint.

Run from the project root:
    python scripts/01_verify_simulator.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hic_unfold.simulator.loop_extrusion import (  # noqa: E402
    LoopExtrusionConfig,
    simulate_ensemble,
)


def main() -> None:
    N = 100
    a, b = 33, 66
    ctcf_left_stop = np.zeros(N)
    ctcf_right_stop = np.zeros(N)
    ctcf_left_stop[a] = 0.99
    ctcf_right_stop[b] = 0.99

    cfg = LoopExtrusionConfig(
        N=N,
        num_lefs=4,
        processivity=400.0,
        ctcf_left_stop=ctcf_left_stop,
        ctcf_right_stop=ctcf_right_stop,
    )

    print(f"running ensemble: N={N}, num_lefs={cfg.num_lefs}, "
          f"processivity={cfg.processivity}, CTCFs at ({a}, {b})")
    result = simulate_ensemble(cfg, num_cells=1500, steps_per_cell=400, seed=42)

    corner = result.avg_loop_matrix[a, b]
    diag_mean = float(np.mean(result.avg_loop_matrix))
    print(f"avg loop matrix: corner peak Z[{a},{b}]={corner:.4f}, "
          f"matrix mean={diag_mean:.5f}, ratio={corner / max(diag_mean, 1e-12):.1f}x")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    im0 = axes[0].imshow(result.avg_loop_matrix, origin="lower", cmap="viridis")
    axes[0].set_title("avg loop matrix (corner peak)")
    axes[0].axhline(a, color="white", lw=0.5, ls=":")
    axes[0].axvline(b, color="white", lw=0.5, ls=":")
    axes[0].set_xlabel("locus j")
    axes[0].set_ylabel("locus i")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(np.log1p(result.avg_footprint), origin="lower", cmap="magma")
    axes[1].set_title("log1p avg extrusion footprint (TAD block)")
    axes[1].axhline(a, color="white", lw=0.5, ls=":")
    axes[1].axhline(b, color="white", lw=0.5, ls=":")
    axes[1].axvline(a, color="white", lw=0.5, ls=":")
    axes[1].axvline(b, color="white", lw=0.5, ls=":")
    axes[1].set_xlabel("locus j")
    axes[1].set_ylabel("locus i")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.suptitle("Step-1 verification: averaged snapshots recover TAD-like structure")
    fig.tight_layout()
    out_path = out_dir / "01_simulator_verify.png"
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
