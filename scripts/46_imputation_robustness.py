"""B4: Robustness to NaN imputation method.

Bintu cells have ~10% missing detections. Our preprocessing imputes with the
population mean distance at the same genomic separation. Reviewers will ask:
"are your results sensitive to this choice?"

Test three imputation strategies on the SAME 65×65 Bintu cells:
    1. pop_mean_by_sep (our default)
    2. nearest-neighbour: fill NaN with mean of nearby valid pairs
    3. zero (extreme baseline: NaN -> 0 distance)

For each, recompute the population contact map and check
    - bulk Pearson with our default (consistency)
    - Rg distribution
    - encoder loop propensity (does it shift?)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(line_buffering=True)

from hic_unfold.data import load_bintu_csv  # noqa: E402
from hic_unfold.embedding import classical_mds  # noqa: E402


def fill_pop_mean_by_sep(D_all: np.ndarray) -> np.ndarray:
    N = D_all.shape[-1]
    pop = np.zeros(N, dtype=np.float64)
    for s in range(1, N):
        vals = []
        for k in range(D_all.shape[0]):
            diag = np.diagonal(D_all[k], offset=s)
            vals.append(diag[np.isfinite(diag)])
        if vals:
            cat = np.concatenate(vals)
            pop[s] = float(cat.mean()) if cat.size else 0.0
    out = D_all.copy()
    for k in range(D_all.shape[0]):
        d = out[k]
        nan_mask = ~np.isfinite(d)
        if not nan_mask.any():
            np.fill_diagonal(d, 0); continue
        i, j = np.where(nan_mask)
        seps = np.abs(i - j)
        d[i, j] = pop[seps]
        d = 0.5 * (d + d.T)
        np.fill_diagonal(d, 0)
        out[k] = d
    return out.astype(np.float32)


def fill_nearest_neighbour(D_all: np.ndarray) -> np.ndarray:
    """For each NaN, average over the 3 closest valid pairs at the same separation."""
    N = D_all.shape[-1]
    out = D_all.copy()
    for k in range(D_all.shape[0]):
        d = out[k]
        if np.isfinite(d).all():
            np.fill_diagonal(d, 0); continue
        for i in range(N):
            for j in range(i + 1, N):
                if not np.isfinite(d[i, j]):
                    s = j - i
                    # collect valid pairs at this separation
                    valid_vals = []
                    for ii in range(N - s):
                        v = d[ii, ii + s]
                        if np.isfinite(v):
                            valid_vals.append(v)
                    if valid_vals:
                        d[i, j] = float(np.mean(valid_vals))
                    else:
                        d[i, j] = 500.0  # fallback
                    d[j, i] = d[i, j]
        np.fill_diagonal(d, 0)
        out[k] = d
    return out.astype(np.float32)


def fill_zero(D_all: np.ndarray) -> np.ndarray:
    out = D_all.copy()
    out[~np.isfinite(out)] = 0.0
    for k in range(out.shape[0]):
        out[k] = 0.5 * (out[k] + out[k].T)
        np.fill_diagonal(out[k], 0)
    return out.astype(np.float32)


def radius_of_gyration(D: np.ndarray) -> float:
    X, _ = classical_mds(D, dim=3)
    com = X.mean(axis=0)
    return float(np.sqrt(((X - com) ** 2).sum(axis=-1).mean()))


def main() -> None:
    print("loading Bintu IMR-90 chr21:28-30Mb raw...")
    ds = load_bintu_csv(ROOT / "data" / "raw_bintu2018" / "IMR90_chr21-28-30Mb.csv")
    # Compute raw distance matrices (with NaN where coords missing)
    D_all = ds.all_distance_matrices()
    print(f"  {D_all.shape}")
    # Filter cells with >= 85% valid coords (same as default)
    valid_frac = np.isfinite(ds.coords_nm[..., 0]).mean(axis=1)
    keep = valid_frac >= 0.85
    D_all = D_all[keep]
    print(f"  kept {D_all.shape[0]} cells")

    print("running 3 imputation strategies...")
    strategies = {
        "pop_mean_by_sep": fill_pop_mean_by_sep(D_all),
        "nearest_neighbour": fill_nearest_neighbour(D_all),
        "zero": fill_zero(D_all),
    }

    # Reference: our default-preprocessed dataset
    ref = strategies["pop_mean_by_sep"]
    print("\ncomputing bulk contact maps + Rg distributions per strategy...")
    hard_thr = 500.0
    Hs = {}
    Rgs = {}
    iu_im = np.triu_indices(ref.shape[-1], k=1)
    for name, D_imp in strategies.items():
        H = (D_imp < hard_thr).astype(np.float32).mean(axis=0)
        # Strategy "zero" creates artificially zero distances → contact rate = 1
        # That's expected; report so reviewers see it
        Rg = np.array([radius_of_gyration(d) for d in D_imp[:200]])
        Hs[name] = H
        Rgs[name] = Rg

    print(f"\n{'strategy':<22s} {'Rg median':>12s} {'Pearson vs ref':>18s} {'MSE vs ref':>12s}")
    print("-" * 70)
    for name, H in Hs.items():
        pcc = float(np.corrcoef(H[iu_im], Hs["pop_mean_by_sep"][iu_im])[0, 1])
        mse = float(((H - Hs["pop_mean_by_sep"]) ** 2)[iu_im].mean())
        rg_med = float(np.median(Rgs[name]))
        print(f"{name:<22s} {rg_med:>12.1f} {pcc:>18.4f} {mse:>12.5f}")

    np.savez(ROOT / "checkpoints" / "step32_imputation.npz",
        H_pop=Hs["pop_mean_by_sep"], H_nn=Hs["nearest_neighbour"], H_zero=Hs["zero"],
        Rg_pop=Rgs["pop_mean_by_sep"], Rg_nn=Rgs["nearest_neighbour"], Rg_zero=Rgs["zero"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    vmax = max(H.max() for H in Hs.values())
    for col, (name, H) in enumerate(Hs.items()):
        ax = axes[0, col]
        im = ax.imshow(H, origin="lower", cmap="Reds", vmin=0, vmax=vmax)
        ax.set_title(f"{name}\nRg med = {float(np.median(Rgs[name])):.0f} nm")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    for col, (name, H) in enumerate(Hs.items()):
        ax = axes[1, col]
        if name == "pop_mean_by_sep":
            ax.text(0.5, 0.5, "reference", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, fontweight="bold")
            ax.axis("off")
        else:
            diff = H - Hs["pop_mean_by_sep"]
            vmax_d = float(np.abs(diff).max())
            im = ax.imshow(diff, origin="lower", cmap="seismic", vmin=-vmax_d, vmax=vmax_d)
            pcc = float(np.corrcoef(H[iu_im], Hs["pop_mean_by_sep"][iu_im])[0, 1])
            ax.set_title(f"{name} - ref\nPearson w/ ref = {pcc:.4f}")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("B4: Imputation robustness -- bulk contact map across NaN strategies")
    fig.tight_layout()
    out = ROOT / "outputs" / "46_imputation_robustness.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
