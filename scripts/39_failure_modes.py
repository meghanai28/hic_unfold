"""A6: Failure-mode characterization.

For each Bintu cell we compute several quality and structural metrics:
    - valid_frac: fraction of segments with detected coordinates
    - Rg: radius of gyration (compactness)
    - mean log distance (proxy for overall extension)
    - encoder loop mass: how many loops the encoder confidently predicts
Then we measure per-cell how well our pipeline reproduces THAT cell:
    - prediction RMSD (Kabsch-aligned) when conditioned on its own z_hat
    - bulk fit Pearson against a small ensemble built from that one cell's z_hat

Output: scatter plots showing where the model works (most cells) and where
it fails (specific quality regimes). Honest "limitations" section material.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(line_buffering=True)

from hic_unfold.diffusion import Denoiser, ddim_sample, make_cosine_schedule  # noqa: E402
from hic_unfold.embedding import classical_mds  # noqa: E402
from hic_unfold.training import make_positional_c  # noqa: E402


def conformation_from_D(D: np.ndarray) -> np.ndarray:
    X, _ = classical_mds(D, dim=3)
    return X - X.mean(axis=0)


def kabsch(P, Q):
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return P @ R.T


def rmsd(A, B) -> float:
    return float(np.sqrt(((A - B) ** 2).sum(axis=-1).mean()))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_path = ROOT / "data" / "real" / "IMR90_chr21-28-30Mb_preprocessed.npz"
    f = np.load(real_path)
    D_real = f["D"]; z_hat_all = f["z_hat"]; valid_frac_all = f["valid_frac"]
    N = int(f["N"])

    diff_ckpt = torch.load(ROOT / "checkpoints" / "step05_diffusion_real.pt",
                           map_location=device, weights_only=False)
    val_idx = np.array(diff_ckpt["val_idx"])
    mu = float(diff_ckpt["mu"]); sigma = float(diff_ckpt["sigma"])

    net = Denoiser(N=N, d_c=int(diff_ckpt["d_c"]), d_pair=32, d_sep=16, d_h=96,
                   d_t=128, dilations=(1, 2, 4, 8, 1)).to(device)
    net.load_state_dict(diff_ckpt["state_dict"]); net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    alpha_bars = make_cosine_schedule(T=int(diff_ckpt["T"]), device=device)
    c_const = make_positional_c(N, int(diff_ckpt["d_c"]), device)

    rng = np.random.default_rng(2028)
    n_cells = 64  # subsample for speed
    pick = rng.choice(val_idx, size=n_cells, replace=False)

    # K predictions per cell, conditioned on that cell's z_hat
    K = 8
    print(f"sampling K={K} predictions for each of {n_cells} held-out cells...")
    z_per_cell = z_hat_all[pick]
    z_t = torch.from_numpy(z_per_cell)[:, None].repeat_interleave(K, dim=0).to(device)
    c_batch = c_const.expand(n_cells * K, -1, -1)
    t0 = time.time()
    with torch.no_grad():
        x = ddim_sample(net, z_t, c_batch, alpha_bars, n_steps=100)
    D_samp = np.expm1(x.squeeze(1).cpu().numpy() * sigma + mu)
    D_samp = np.maximum(D_samp, 0)
    for k in range(D_samp.shape[0]):
        D_samp[k] = 0.5 * (D_samp[k] + D_samp[k].T)
        np.fill_diagonal(D_samp[k], 0)
    D_samp = D_samp.reshape(n_cells, K, N, N)
    print(f"  {time.time()-t0:.1f}s")

    # Per-cell quality metrics + prediction error
    quality_rows = []
    print(f"\nper-cell metrics...")
    for k in range(n_cells):
        D_cell = D_real[pick[k]]
        valid_frac = float(valid_frac_all[pick[k]])
        X_meas = conformation_from_D(D_cell)
        Rg = float(np.sqrt((X_meas ** 2).sum(axis=-1).mean()))
        mean_logd = float(np.log1p(D_cell).mean())
        loop_mass = float(np.triu(z_per_cell[k], k=1).sum())
        # RMSD over the K samples (Kabsch-aligned)
        rs = []
        for j in range(K):
            X_pred = kabsch(conformation_from_D(D_samp[k, j]), X_meas)
            rs.append(rmsd(X_pred, X_meas))
        rmsd_med = float(np.median(rs))
        rmsd_min = float(min(rs))
        # Bulk fit: K predictions' average vs the single cell
        H_pred = (D_samp[k].mean(axis=0) < 500).astype(np.float32)
        H_meas = (D_cell < 500).astype(np.float32)
        iu = np.triu_indices(N, k=1)
        pcc = float(np.corrcoef(H_pred[iu], H_meas[iu])[0, 1]) \
            if H_pred[iu].std() > 0 and H_meas[iu].std() > 0 else float("nan")
        quality_rows.append((valid_frac, Rg, mean_logd, loop_mass,
                             rmsd_med, rmsd_min, pcc))
    q = np.array(quality_rows)

    print(f"\nsummary:")
    print(f"  per-cell prediction RMSD: median={np.median(q[:, 4]):.1f} nm  "
          f"IQR=[{np.percentile(q[:, 4], 25):.0f}, {np.percentile(q[:, 4], 75):.0f}]")
    print(f"  per-cell best-of-{K} RMSD: median={np.median(q[:, 5]):.1f} nm")
    print(f"  per-cell single-cell bulk Pearson: median={np.median(q[:, 6]):.3f}")

    # Identify low-performing cells
    bad_mask = q[:, 4] > np.percentile(q[:, 4], 75)
    good_mask = q[:, 4] < np.percentile(q[:, 4], 25)
    print(f"\ncells with WORST RMSD (top quartile):")
    print(f"  median valid_frac: {q[bad_mask, 0].mean():.3f}   "
          f"median Rg: {q[bad_mask, 1].mean():.1f} nm  "
          f"median loop_mass: {q[bad_mask, 3].mean():.2f}")
    print(f"cells with BEST RMSD (bottom quartile):")
    print(f"  median valid_frac: {q[good_mask, 0].mean():.3f}   "
          f"median Rg: {q[good_mask, 1].mean():.1f} nm  "
          f"median loop_mass: {q[good_mask, 3].mean():.2f}")

    # Correlations: does each quality metric predict failure?
    metrics = ["valid_frac", "Rg (nm)", "mean log distance", "encoder loop mass"]
    print(f"\nCorrelations of quality metrics with prediction RMSD:")
    for i, name in enumerate(metrics):
        r = float(np.corrcoef(q[:, i], q[:, 4])[0, 1])
        print(f"  {name:<28s} Pearson with RMSD = {r:+.3f}")

    np.savez(ROOT / "checkpoints" / "step25_failure_modes.npz",
        quality=q, columns=np.array(["valid_frac", "Rg", "mean_logd",
                                     "loop_mass", "rmsd_med", "rmsd_min", "pcc"]),
    )

    # Figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for k, (name, col) in enumerate(zip(metrics, [0, 1, 2, 3])):
        if k >= 4: break
        r, c_ = divmod(k, 3)
        ax = axes[r, c_]
        ax.scatter(q[:, col], q[:, 4], s=30, alpha=0.6, color="C3", edgecolor="black")
        pcc = float(np.corrcoef(q[:, col], q[:, 4])[0, 1])
        ax.set_xlabel(name); ax.set_ylabel("prediction RMSD (nm)")
        ax.set_title(f"{name} vs RMSD (Pearson {pcc:+.2f})")
        ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(q[:, 6], q[:, 4], s=30, alpha=0.6, color="C0", edgecolor="black")
    pcc = float(np.corrcoef(q[:, 6], q[:, 4])[0, 1])
    ax.set_xlabel("single-cell bulk Pearson")
    ax.set_ylabel("prediction RMSD (nm)")
    ax.set_title(f"single-cell bulk fit vs RMSD (Pearson {pcc:+.2f})")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.axis("off")
    summary = (
        "FAILURE MODE CHARACTERISATION\n\n"
        f"n_cells (held-out): {n_cells}\n"
        f"K samples per cell: {K}\n\n"
        f"Prediction RMSD (nm):\n"
        f"  median:       {np.median(q[:, 4]):.1f}\n"
        f"  IQR:          [{np.percentile(q[:, 4], 25):.0f}, {np.percentile(q[:, 4], 75):.0f}]\n"
        f"  best-of-{K}:    {np.median(q[:, 5]):.1f}\n\n"
        f"Quality-RMSD correlations:\n"
        f"  valid_frac:    {np.corrcoef(q[:, 0], q[:, 4])[0, 1]:+.3f}\n"
        f"  Rg:            {np.corrcoef(q[:, 1], q[:, 4])[0, 1]:+.3f}\n"
        f"  log distance:  {np.corrcoef(q[:, 2], q[:, 4])[0, 1]:+.3f}\n"
        f"  loop mass:     {np.corrcoef(q[:, 3], q[:, 4])[0, 1]:+.3f}\n\n"
        "Interpretation:\n"
        "  Strong correlation with a metric => model\n"
        "  fails specifically on cells with that property.\n"
        "  Weak correlations => no systematic failure mode\n"
        "  (errors are dominated by the irreducible\n"
        "  single-cell stochasticity)."
    )
    ax.text(0.0, 0.95, summary, fontsize=10, va="top", family="monospace")

    fig.suptitle("Per-cell failure-mode map: when does the prediction RMSD exceed the noise floor?")
    fig.tight_layout()
    out = ROOT / "outputs" / "39_failure_modes.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
