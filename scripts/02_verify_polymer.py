"""Step-1b verification: full (loop-extrusion -> polymer -> contact) pipeline.

For each simulated cell:
    1. Run the loop-extrusion simulator to get a snapshot (left, right) of LEF anchors
    2. Convert the snapshot to a loop matrix z
    3. Build a Gaussian polymer with backbone springs + extra springs at every z anchor
    4. Sample 3D positions and compute the pairwise distance matrix
    5. Apply the soft-contact forward operator (Section 5.1) to get a per-cell contact map

Averaging contact maps across cells should reproduce a Hi-C-like map:
    - decay with genomic separation
    - a TAD-like block between the two CTCFs
    - a corner peak at the CTCF-CTCF coordinate

Run:
    python scripts/02_verify_polymer.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hic_unfold.polymer.gaussian import (  # noqa: E402
    PolymerConfig,
    sample_distance_matrix,
    soft_contact,
)
from hic_unfold.simulator.loop_extrusion import (  # noqa: E402
    LoopExtrusionConfig,
    run_to_snapshot,
    snapshot_to_loop_matrix,
)


def observed_over_expected(C: np.ndarray) -> np.ndarray:
    """Divide each diagonal by its mean so the genomic-distance decay is removed.
    Standard Hi-C normalization that exposes corner peaks and TAD boundaries."""
    N = C.shape[0]
    out = np.zeros_like(C)
    for s in range(-(N - 1), N):
        diag = np.diag(C, k=s)
        m = diag.mean()
        if m > 0:
            i = np.arange(max(0, -s), min(N, N - s))
            out[i, i + s] = diag / m
    return out


def main() -> None:
    N = 100
    a, b = 33, 66
    num_cells = 800
    steps_per_cell = 500

    ctcf_left_stop = np.zeros(N)
    ctcf_right_stop = np.zeros(N)
    ctcf_left_stop[a] = 1.0
    ctcf_right_stop[b] = 1.0

    le_cfg = LoopExtrusionConfig(
        N=N, num_lefs=2, processivity=500.0,
        ctcf_left_stop=ctcf_left_stop, ctcf_right_stop=ctcf_right_stop,
    )
    poly_cfg = PolymerConfig(backbone_k=1.0, loop_k=15.0)

    d0, tau = 4.0, 1.0
    rng = np.random.default_rng(42)

    print(f"running pipeline: N={N}, num_cells={num_cells}, CTCFs at ({a}, {b})")
    avg_contact = np.zeros((N, N))
    avg_distance = np.zeros((N, N))
    first_D = None
    first_C = None
    first_X = None

    for c in range(num_cells):
        L, R = run_to_snapshot(le_cfg, steps_per_cell, rng)
        z = snapshot_to_loop_matrix(L, R, N)
        D, X = sample_distance_matrix(z, N, rng, cfg=poly_cfg)
        C = soft_contact(D, d0=d0, tau=tau)
        avg_distance += D
        avg_contact += C
        if c == 0:
            first_D, first_C, first_X = D, C, X
    avg_distance /= num_cells
    avg_contact /= num_cells

    obs_exp = observed_over_expected(avg_contact)
    sep = b - a
    corner = avg_contact[a, b]
    bg = np.diag(avg_contact, k=sep).mean()
    print(f"avg contact: C[{a},{b}]={corner:.3f}, "
          f"mean at same separation={bg:.3f}, enrichment={corner / max(bg, 1e-9):.2f}x")
    inside = avg_contact[a:b + 1, a:b + 1].mean()
    off = (avg_contact[:a, b + 1:].mean() + avg_contact[b + 1:, :a].mean()) / 2
    print(f"TAD block (inside) mean={inside:.3f}, off-block mean={off:.3f}, "
          f"ratio={inside / max(off, 1e-9):.2f}x")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 4.5))

    ax = fig.add_subplot(1, 4, 1)
    im = ax.imshow(first_C, origin="lower", cmap="Reds", vmin=0, vmax=1)
    ax.set_title("one cell: contact map\n(noisy, heterogeneous)")
    ax.set_xlabel("locus j"); ax.set_ylabel("locus i")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(1, 4, 2)
    im = ax.imshow(avg_contact, origin="lower", cmap="Reds")
    ax.set_title(f"averaged over {num_cells} cells\n(Hi-C-like)")
    for v in (a, b):
        ax.axhline(v, color="black", lw=0.5, ls=":")
        ax.axvline(v, color="black", lw=0.5, ls=":")
    ax.set_xlabel("locus j"); ax.set_ylabel("locus i")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(1, 4, 3)
    vmax = max(2.0, np.percentile(obs_exp, 99))
    im = ax.imshow(obs_exp, origin="lower", cmap="seismic", vmin=0, vmax=vmax)
    ax.set_title("observed / expected\n(corner peak + TAD block visible)")
    for v in (a, b):
        ax.axhline(v, color="black", lw=0.5, ls=":")
        ax.axvline(v, color="black", lw=0.5, ls=":")
    ax.set_xlabel("locus j"); ax.set_ylabel("locus i")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(1, 4, 4, projection="3d")
    ax.plot(first_X[:, 0], first_X[:, 1], first_X[:, 2], "-", color="lightgray", lw=0.7)
    sc = ax.scatter(first_X[:, 0], first_X[:, 1], first_X[:, 2],
                    c=np.arange(N), cmap="viridis", s=10)
    ax.scatter(*first_X[a], color="red", s=80, label=f"CTCF at {a}")
    ax.scatter(*first_X[b], color="blue", s=80, label=f"CTCF at {b}")
    ax.set_title("one cell: 3D conformation")
    ax.legend(loc="upper left", fontsize=7)

    fig.suptitle("Step-1b: loop-extrusion + polymer reproduces Hi-C-like contact map")
    fig.tight_layout()
    out_path = out_dir / "02_polymer_verify.png"
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")

    # P(s) contact-scaling check: average contact vs genomic separation.
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    seps = np.arange(1, N)
    p_s = [np.diag(avg_contact, k=s).mean() for s in seps]
    ax2.loglog(seps, p_s, "o-", ms=3)
    ax2.set_xlabel("genomic separation s")
    ax2.set_ylabel("P(contact | s)")
    ax2.set_title("P(s) contact scaling")
    ax2.grid(True, which="both", alpha=0.3)
    fig2.tight_layout()
    out2 = out_dir / "02_polymer_ps.png"
    fig2.savefig(out2, dpi=130)
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
