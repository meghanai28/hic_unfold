"""Quantify ensemble diversity: show the deconvolved ensemble is genuinely
heterogeneous, not collapsed to a near-mean blob.

Reviewers will ask: "Maybe your ensemble is just M copies of the bulk average?
Bulk-matching is trivial then." This figure answers it directly with:
    - per-pair distance variance across cells (high = diverse)
    - cell-cell pairwise distance heatmap (off-diagonal large = cells differ)
    - K-means clusters in conformation space (structured modes)
    - comparison of diversity in: real Bintu cells, step-8 guided, naive prior

Run:
    python scripts/31_ensemble_diversity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(line_buffering=True)


def per_pair_std(D_stack: np.ndarray) -> np.ndarray:
    """Std of D[i,j] across cells, returned as N x N matrix."""
    return D_stack.std(axis=0)


def cell_cell_rmsd(D_stack: np.ndarray) -> np.ndarray:
    """Pairwise RMSD between all cells (M x M)."""
    M = D_stack.shape[0]
    flat = D_stack.reshape(M, -1).astype(np.float32)
    sq = ((flat[:, None, :] - flat[None, :, :]) ** 2).mean(axis=-1)
    return np.sqrt(sq)


def diversity_score(D_stack: np.ndarray) -> float:
    """One number: mean cell-to-ensemble-mean RMSD."""
    m = D_stack.mean(axis=0)
    return float(np.sqrt(((D_stack - m[None]) ** 2).mean(axis=(1, 2))).mean())


def main() -> None:
    real_path = ROOT / "data" / "real" / "IMR90_chr21-28-30Mb_preprocessed.npz"
    f = np.load(real_path)
    D_real = f["D"]

    import torch
    diff_ckpt = torch.load(ROOT / "checkpoints" / "step05_diffusion_real.pt",
                           map_location="cpu", weights_only=False)
    val_idx = np.array(diff_ckpt["val_idx"])

    D_real_val = D_real[val_idx]

    gd8 = np.load(ROOT / "checkpoints" / "step08_guided.npz")
    D_guided = gd8["D_samples"]

    g10 = np.load(ROOT / "checkpoints" / "step10_realhic.npz")
    D_hic = g10["D_samples"]

    # Reference "collapsed" ensemble: M copies of the mean
    mean_real = D_real_val.mean(axis=0)
    D_collapsed = np.broadcast_to(mean_real, D_guided.shape).copy()
    # Add small per-cell noise so it's not literally identical (numerically)
    D_collapsed = D_collapsed + np.random.default_rng(0).normal(0, 1.0, size=D_collapsed.shape)

    sets = {
        "Real Bintu held-out": D_real_val[:128],
        "Step-8 guided (Bintu pseudo-bulk)": D_guided,
        "Step-10 guided (real Hi-C)": D_hic,
        "Collapsed (mean ± 1 nm noise)": D_collapsed[:128],
    }

    print(f"{'condition':<40s} {'diversity':>12s} {'std_med':>10s}")
    print("-" * 65)
    scores: dict[str, float] = {}
    stds: dict[str, np.ndarray] = {}
    for name, D in sets.items():
        d = diversity_score(D)
        s = per_pair_std(D)
        stds[name] = s
        scores[name] = d
        med_std = float(np.median(s[np.triu_indices(s.shape[0], k=1)]))
        print(f"{name:<40s} {d:>12.1f} {med_std:>10.1f}")

    # Cell-cell RMSD heatmaps
    print("computing cell-cell RMSD heatmaps...")
    rmsds = {name: cell_cell_rmsd(D[:96]) for name, D in sets.items()}

    # K-means cluster guided ensemble — does it have structured modes?
    flat = D_guided.reshape(D_guided.shape[0], -1)
    km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(flat)
    cluster_means = np.zeros((4, *D_guided.shape[1:]), dtype=np.float32)
    cluster_counts = np.zeros(4, dtype=int)
    for c in range(4):
        mask = km.labels_ == c
        cluster_counts[c] = mask.sum()
        if mask.any():
            cluster_means[c] = D_guided[mask].mean(axis=0)
    print(f"K-means cluster sizes in step-8 guided ensemble: {cluster_counts.tolist()}")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 4)

    # Row 1: per-pair std maps for each condition
    vmax = max(s.max() for s in stds.values())
    for col, (name, s) in enumerate(stds.items()):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(s, origin="lower", cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"per-pair distance std\n{name}\n(diversity={scores[name]:.1f})", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 2: cell-cell RMSD heatmaps
    vmax_r = max(r.max() for r in rmsds.values())
    for col, (name, r) in enumerate(rmsds.items()):
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(r, origin="lower", cmap="viridis", vmin=0, vmax=vmax_r)
        ax.set_title(f"cell-cell RMSD\n{name}", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 3: K-means cluster means
    for c in range(4):
        ax = fig.add_subplot(gs[2, c])
        im = ax.imshow(cluster_means[c], origin="lower", cmap="viridis")
        ax.set_title(f"cluster {c} (n={cluster_counts[c]})\nstep-8 guided", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Summary text panel (overlay)
    fig.text(0.99, 0.02,
             "Top row: per-pair distance std across cells; bright = pair varies a lot between cells\n"
             "Middle: cell-cell RMSD; bright off-diagonal = cells genuinely differ from each other\n"
             "Bottom: K-means clusters in guided ensemble; distinct means = structured single-cell modes",
             ha="right", va="bottom", fontsize=9, family="monospace")

    fig.suptitle("Ensemble diversity: deconvolution recovers heterogeneous single-cell structures")
    fig.tight_layout()
    out = out_dir / "31_ensemble_diversity.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")

    np.savez_compressed(ROOT / "checkpoints" / "step19_diversity.npz",
        diversity_scores=np.array(list(scores.values()), dtype=np.float32),
        diversity_names=np.array(list(scores.keys()), dtype=object),
        cluster_counts=cluster_counts,
        cluster_means=cluster_means,
    )


if __name__ == "__main__":
    main()
