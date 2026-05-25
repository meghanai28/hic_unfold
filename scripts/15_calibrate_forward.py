"""Build step 6: calibrate the Stage-2 forward operator d0, tau.

The spec calls for verifying that A(real single cells) reproduces a real bulk
map. We use a hard 500 nm contact threshold pseudo-bulk over a calibration
subset of Bintu cells as the reference (the standard Bintu et al. contact
metric), then grid-search (d0, tau) so the soft-form A matches it. The
calibrated parameters are then validated on a held-out subset — same fit
quality on cells the calibration didn't see is what we want.

Run:
    python scripts/15_calibrate_forward.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hic_unfold.forward import apply_forward, calibrate_d0_tau  # noqa: E402


def main() -> None:
    region = "IMR90_chr21-28-30Mb"
    data_path = ROOT / "data" / "real" / f"{region}_preprocessed.npz"
    f = np.load(data_path)
    D_all = f["D"]
    N = int(f["N"])
    M = D_all.shape[0]
    print(f"loaded {M} preprocessed real cells at N={N}")

    rng = np.random.default_rng(123)
    perm = rng.permutation(M)
    n_cal = int(0.7 * M)
    cal_idx = perm[:n_cal]
    val_idx = perm[n_cal:]
    D_cal = D_all[cal_idx]
    D_val = D_all[val_idx]
    print(f"calibration set: {len(cal_idx)} cells, held-out: {len(val_idx)} cells")

    hard_threshold = 500.0  # nm — Bintu convention
    H_target_cal = (D_cal < hard_threshold).mean(axis=0).astype(np.float32)
    H_target_val = (D_val < hard_threshold).mean(axis=0).astype(np.float32)

    iu = np.triu_indices(N, k=1)
    off_diag_mask = np.zeros((N, N), dtype=bool)
    off_diag_mask[iu] = True
    off_diag_mask |= off_diag_mask.T

    d0_grid = np.linspace(300, 700, 21).astype(np.float32)
    tau_grid = np.linspace(30, 250, 12).astype(np.float32)
    print(f"grid search: {len(d0_grid)} d0 x {len(tau_grid)} tau = {len(d0_grid) * len(tau_grid)} cells")

    res = calibrate_d0_tau(D_cal, H_target_cal, d0_grid, tau_grid, mask=off_diag_mask)
    print(f"best: d0={res['d0']:.1f} nm, tau={res['tau']:.1f} nm")
    print(f"  calibration-set MSE={res['mse']:.6f}, Pearson={res['pearson']:.4f}")

    H_pred_val = apply_forward(D_val, d0=res["d0"], tau=res["tau"])
    val_mse = float(((H_pred_val - H_target_val)[off_diag_mask] ** 2).mean())
    val_pcc = float(np.corrcoef(H_pred_val[iu], H_target_val[iu])[0, 1])
    print(f"held-out:  MSE={val_mse:.6f}, Pearson={val_pcc:.4f}")

    out_path = ROOT / "checkpoints" / "step06_forward_params.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, d0=res["d0"], tau=res["tau"], hard_threshold=hard_threshold,
             cal_mse=res["mse"], cal_pearson=res["pearson"],
             val_mse=val_mse, val_pearson=val_pcc)
    print(f"saved {out_path}")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(res["mse_grid"], origin="lower", aspect="auto", cmap="viridis_r",
                   extent=[tau_grid[0], tau_grid[-1], d0_grid[0], d0_grid[-1]])
    ax.set_xlabel("tau (nm)"); ax.set_ylabel("d0 (nm)")
    ax.set_title("calibration MSE surface")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.scatter([res["tau"]], [res["d0"]], color="red", marker="x", s=80,
               label=f"d0={res['d0']:.0f}, tau={res['tau']:.0f}")
    ax.legend(loc="upper right", fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(res["pearson_grid"], origin="lower", aspect="auto", cmap="viridis",
                   extent=[tau_grid[0], tau_grid[-1], d0_grid[0], d0_grid[-1]],
                   vmin=0.9, vmax=1.0)
    ax.set_xlabel("tau (nm)"); ax.set_ylabel("d0 (nm)")
    ax.set_title("calibration Pearson surface")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.scatter([res["tau"]], [res["d0"]], color="red", marker="x", s=80)

    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(H_target_val[iu], H_pred_val[iu], s=2, alpha=0.4, color="C3")
    lim = max(H_target_val.max(), H_pred_val.max())
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
    ax.set_xlabel(f"target P(d<{hard_threshold:.0f}nm)  [held-out]")
    ax.set_ylabel("predicted soft contact")
    ax.set_title(f"per-pair agreement (held-out)\nPearson={val_pcc:.4f}, MSE={val_mse:.5f}")

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(H_target_val, origin="lower", cmap="Reds", vmin=0, vmax=H_target_val.max())
    ax.set_title("hard pseudo-bulk (held-out)\nP(d<500nm)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(H_pred_val, origin="lower", cmap="Reds", vmin=0, vmax=H_target_val.max())
    ax.set_title(f"soft A(D_val) with calibrated d0, tau")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, 2])
    diff = H_pred_val - H_target_val
    vmax = float(np.abs(diff).max())
    im = ax.imshow(diff, origin="lower", cmap="seismic", vmin=-vmax, vmax=vmax)
    ax.set_title("predicted - target (held-out)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"Step-6 forward-operator calibration  ({region})")
    fig.tight_layout()
    out = out_dir / "15_calibrate_forward.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
