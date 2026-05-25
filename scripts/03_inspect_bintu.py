"""Inspect the Bintu 2018 chromatin tracing data.

Renders:
    - the population-averaged contact map for one region
    - log of (1 - contact rate) so the TAD/decay structure is visible
    - one example single-cell distance matrix
    - one example 3D conformation

Run:
    python scripts/03_inspect_bintu.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hic_unfold.data import load_bintu_csv  # noqa: E402


def main() -> None:
    region = "IMR90_chr21-28-30Mb"
    ds = load_bintu_csv(ROOT / "data" / "raw_bintu2018" / f"{region}.csv")
    print(f"region {region}: {ds.num_cells} cells x {ds.num_segments} segments")

    C = ds.contact_map(threshold_nm=500.0)
    cell_idx = 7
    D_one = ds.distance_matrix(cell_idx)
    coords_one = ds.coords_nm[cell_idx]
    valid_one = np.isfinite(coords_one[:, 0])
    print(f"example cell {cell_idx}: {valid_one.sum()}/{ds.num_segments} loci detected")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13, 4.5))

    ax = fig.add_subplot(1, 3, 1)
    im = ax.imshow(C, origin="lower", cmap="Reds", vmin=0, vmax=1)
    ax.set_title(f"population contact map\n{ds.num_cells} cells, d<500nm")
    ax.set_xlabel("30kb segment j"); ax.set_ylabel("30kb segment i")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(1, 3, 2)
    with np.errstate(invalid="ignore"):
        log_C = np.log10(np.maximum(C, 1e-3))
    im = ax.imshow(log_C, origin="lower", cmap="Reds")
    ax.set_title("log10 contact rate\n(reveals TAD structure)")
    ax.set_xlabel("segment j"); ax.set_ylabel("segment i")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    pts = coords_one[valid_one]
    idx = np.where(valid_one)[0]
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-", color="lightgray", lw=0.7)
    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=idx, cmap="viridis", s=15)
    ax.set_title(f"cell {cell_idx}: 3D trace")
    plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.1, label="segment idx")

    fig.suptitle("Bintu et al. 2018 — real single-cell chromatin tracing (chr21:28-30Mb, IMR90)")
    fig.tight_layout()
    out_path = out_dir / "03_bintu_inspect.png"
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
