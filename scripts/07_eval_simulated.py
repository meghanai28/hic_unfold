"""Build step 3 evaluation: confirm sampled matrices embed to sane polymers
and reproduce P(s) scaling.

Loads the trained denoiser, samples conformations conditioned on validation
loop matrices z, and reports:
    1. P(s) — soft-contact probability vs genomic separation — for samples
       overlaid with the training-distribution P(s). Should overlap closely.
    2. MDS residual — for each sampled distance matrix, classical MDS to R^3,
       reconstruct distances, compute residual. Compares to MDS residual of
       training data (which by construction comes from valid 3D positions).
    3. A grid of sample/target images for qualitative inspection.

Run:
    python scripts/07_eval_simulated.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hic_unfold.diffusion import Denoiser, ddim_sample, make_cosine_schedule  # noqa: E402
from hic_unfold.embedding import mds_residual  # noqa: E402
from hic_unfold.polymer.gaussian import PolymerConfig, sample_distance_matrix, soft_contact  # noqa: E402
from hic_unfold.training import SimulatedDataset, make_positional_c  # noqa: E402


def p_of_s(C_stack: np.ndarray) -> np.ndarray:
    """Mean contact across cells as a function of genomic separation s."""
    N = C_stack.shape[-1]
    mean_C = C_stack.mean(axis=0)
    return np.array([np.diag(mean_C, k=s).mean() for s in range(1, N)])


def unstandardize(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.expm1(x * sigma + mu)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ROOT / "checkpoints" / "step03.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    N = int(ckpt["N"]); d_c = int(ckpt["d_c"]); T = int(ckpt["T"])
    mu, sigma = ckpt["mu"], ckpt["sigma"]
    print(f"loaded {ckpt_path}: N={N}, mu={mu:.3f}, sigma={sigma:.3f}")

    net = Denoiser(N=N, d_c=d_c, d_pair=32, d_sep=16, d_h=96, d_t=128,
                   dilations=(1, 2, 4, 8, 1)).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    alpha_bars = make_cosine_schedule(T=T, device=device)
    c_const = make_positional_c(N, d_c, device)

    data_path = ROOT / "data" / "sim" / f"step03_N{N}_M5000.npz"
    ds = SimulatedDataset(data_path)

    rng = np.random.default_rng(123)
    n_eval = 64
    eval_idx = rng.choice(len(ds), size=n_eval, replace=False)

    z_eval = torch.from_numpy(ds.z[eval_idx])[:, None].to(device)
    D_eval_true = ds.D[eval_idx]
    c_batch = c_const.expand(n_eval, -1, -1)

    print(f"sampling {n_eval} conformations via DDIM (100 steps)...")
    with torch.no_grad():
        x_samp = ddim_sample(net, z_eval, c_batch, alpha_bars, n_steps=100)
    x_samp_np = x_samp.squeeze(1).cpu().numpy()
    D_samp = np.stack([unstandardize(x, mu, sigma) for x in x_samp_np])
    D_samp = np.maximum(D_samp, 0)
    for s in D_samp:
        np.fill_diagonal(s, 0.0)

    # ---- P(s) check ----
    d0, tau = 4.0, 1.0
    C_train = soft_contact(ds.D, d0=d0, tau=tau)
    C_samp = soft_contact(D_samp, d0=d0, tau=tau)
    ps_train = p_of_s(C_train)
    ps_samp = p_of_s(C_samp)
    seps = np.arange(1, N)

    # ---- MDS residuals ----
    res_train = [mds_residual(d, dim=3) for d in ds.D[eval_idx]]
    res_samp = [mds_residual(d, dim=3) for d in D_samp]
    rel_train = np.array([r["relative_rmse"] for r in res_train])
    rel_samp = np.array([r["relative_rmse"] for r in res_samp])
    eig_train = np.array([r["eig_ratio"] for r in res_train])
    eig_samp = np.array([r["eig_ratio"] for r in res_samp])
    print(f"MDS relative RMSE (lower=better Euclidean fit):")
    print(f"  training:  median={np.median(rel_train):.4f}, IQR=[{np.percentile(rel_train,25):.4f}, {np.percentile(rel_train,75):.4f}]")
    print(f"  samples:   median={np.median(rel_samp):.4f}, IQR=[{np.percentile(rel_samp,25):.4f}, {np.percentile(rel_samp,75):.4f}]")
    print(f"MDS eigenvalue ratio (1.0 = fully 3D Euclidean):")
    print(f"  training:  median={np.median(eig_train):.4f}")
    print(f"  samples:   median={np.median(eig_samp):.4f}")

    # ---- per-cell sample-vs-target RMSE, compared with two baselines ----
    rmse_pair = np.sqrt(((D_samp - D_eval_true) ** 2).mean(axis=(1, 2)))
    # Baseline 1: random unrelated cell's D. Upper bound on what conditioning could buy.
    rand_idx = rng.choice(len(ds), size=n_eval, replace=False)
    rmse_rand = np.sqrt(((ds.D[rand_idx] - D_eval_true) ** 2).mean(axis=(1, 2)))
    # Baseline 2: natural polymer noise floor. Same z, fresh draw from the Gaussian
    # polymer. This is the irreducible RMSE — even a perfect model can't beat it.
    rmse_floor = []
    floor_rng = np.random.default_rng(91)
    for k in range(n_eval):
        D_alt, _ = sample_distance_matrix(ds.z[eval_idx[k]], N, floor_rng,
                                          PolymerConfig(backbone_k=1.0, loop_k=15.0))
        rmse_floor.append(float(np.sqrt(((D_alt - D_eval_true[k]) ** 2).mean())))
    rmse_floor = np.array(rmse_floor)
    print(f"random-to-target RMSE (upper bound):   median={np.median(rmse_rand):.3f}")
    print(f"sample-to-target RMSE (this model):    median={np.median(rmse_pair):.3f}")
    print(f"polymer noise floor (lower bound):     median={np.median(rmse_floor):.3f}")
    gap_total = np.median(rmse_rand) - np.median(rmse_floor)
    gap_closed = np.median(rmse_rand) - np.median(rmse_pair)
    if gap_total > 1e-6:
        print(f"  fraction of conditioning gap closed: {100 * gap_closed / gap_total:.1f}%")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    ax.loglog(seps, ps_train, "o-", ms=3, label="training data", color="black")
    ax.loglog(seps, ps_samp, "s-", ms=3, label="diffusion samples", color="C3")
    ax.set_xlabel("genomic separation s"); ax.set_ylabel("P(contact | s)")
    ax.set_title("P(s) scaling — samples vs training")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    bins = np.linspace(0, max(rel_train.max(), rel_samp.max()) * 1.05, 25)
    ax.hist(rel_train, bins=bins, alpha=0.6, label=f"training (med={np.median(rel_train):.3f})",
            color="black")
    ax.hist(rel_samp, bins=bins, alpha=0.6, label=f"samples (med={np.median(rel_samp):.3f})",
            color="C3")
    ax.set_xlabel("MDS relative RMSE"); ax.set_ylabel("# cells")
    ax.set_title("MDS Euclidean-fit residual")
    ax.legend()

    ax = axes[2]
    ax.boxplot([rmse_rand, rmse_pair, rmse_floor],
               tick_labels=["random\n(upper)", "diffusion\nsample", "polymer\nnoise floor"])
    ax.set_ylabel("RMSE to target D")
    ax.set_title("sample-to-target distance error")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(np.median(rmse_floor), color="C2", ls="--", lw=1, alpha=0.7,
               label=f"floor median ({np.median(rmse_floor):.2f})")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out = out_dir / "07_eval_metrics.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")

    # Qualitative grid
    n_show = 4
    fig2, axes2 = plt.subplots(3, n_show, figsize=(3 * n_show, 8))
    for k in range(n_show):
        axes2[0, k].imshow(ds.z[eval_idx[k]], origin="lower", cmap="gray_r")
        axes2[0, k].set_title(f"z (cond.) #{eval_idx[k]}"); axes2[0, k].axis("off")
        axes2[1, k].imshow(D_eval_true[k], origin="lower", cmap="viridis")
        axes2[1, k].set_title("target D"); axes2[1, k].axis("off")
        axes2[2, k].imshow(D_samp[k], origin="lower", cmap="viridis")
        axes2[2, k].set_title(f"sampled D\nRMSE={rmse_pair[k]:.2f}"); axes2[2, k].axis("off")
    fig2.tight_layout()
    out2 = out_dir / "07_eval_grid.png"
    fig2.savefig(out2, dpi=130)
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
