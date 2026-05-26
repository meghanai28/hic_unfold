"""Ablation comparison: run guided sampling with both the full and the
z-ablated prior against the same held-out Bintu pseudo-bulk target, and
compare bulk fit and single-cell statistics.

If z conditioning is doing the work the architecture claims, the ablated model
should be measurably worse — by Pearson, by Rg recovery, by P(s) shape, or by
the diversity of the sampled ensemble.

Run:
    python scripts/22_ablation_guided_sampling.py
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

from hic_unfold.diffusion import (  # noqa: E402
    Denoiser, guided_ddim_sample, make_cosine_schedule,
)
from hic_unfold.embedding import classical_mds  # noqa: E402
from hic_unfold.training import make_positional_c  # noqa: E402


def radius_of_gyration(D: np.ndarray) -> float:
    X, _ = classical_mds(D, dim=3)
    com = X.mean(axis=0)
    return float(np.sqrt(((X - com) ** 2).sum(axis=-1).mean()))


def p_of_s(C_stack: np.ndarray) -> np.ndarray:
    N = C_stack.shape[-1]
    M = C_stack.mean(axis=0)
    return np.array([np.diag(M, k=s).mean() for s in range(1, N)])


def load_denoiser(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    N = int(ckpt["N"]); d_c = int(ckpt["d_c"]); T = int(ckpt["T"])
    net = Denoiser(N=N, d_c=d_c, d_pair=32, d_sep=16, d_h=96, d_t=128,
                   dilations=(1, 2, 4, 8, 1)).to(device)
    net.load_state_dict(ckpt["state_dict"]); net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, ckpt


def run_guided(net, z, c, alpha_bars, H_obs, d0, tau, mu, sigma,
               n_steps=200, eta=30000.0):
    return guided_ddim_sample(
        net, z, c, alpha_bars, H_obs,
        d0=d0, tau=tau, mu=mu, sigma=sigma,
        n_steps=n_steps, eta=eta,
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    region = "IMR90_chr21-28-30Mb"
    real_path = ROOT / "data" / "real" / f"{region}_preprocessed.npz"
    fwd_path = ROOT / "checkpoints" / "step06_forward_params.npz"

    fwd = np.load(fwd_path)
    d0 = float(fwd["d0"]); tau = max(float(fwd["tau"]), 80.0)
    hard_thr = float(fwd["hard_threshold"])

    f = np.load(real_path)
    D_real = f["D"]; z_hat_all = f["z_hat"]
    N = int(f["N"]); mu = float(f["mu"]); sigma = float(f["sigma"])

    # Load both models.
    net_full, c_full = load_denoiser(ROOT / "checkpoints" / "step05_diffusion_real.pt", device)
    net_abl, c_abl = load_denoiser(ROOT / "checkpoints" / "step11_ablated_no_z.pt", device)
    val_idx = np.array(c_full["val_idx"])
    train_idx = np.setdiff1d(np.arange(D_real.shape[0]), val_idx)

    H_target = (D_real[val_idx] < hard_thr).mean(axis=0).astype(np.float32)
    H_obs = torch.tensor(H_target, device=device)

    alpha_bars = make_cosine_schedule(T=int(c_full["T"]), device=device)
    c_const = make_positional_c(N, int(c_full["d_c"]), device)

    M_samp = 128
    rng = np.random.default_rng(2026)
    z_idx = rng.choice(train_idx, size=M_samp, replace=True)
    z_full = torch.from_numpy(z_hat_all[z_idx])[:, None].to(device)
    z_zero = torch.zeros_like(z_full)
    c_batch = c_const.expand(M_samp, -1, -1)

    print(f"running guided sampling with FULL model + z_hat conditioning...")
    t0 = time.time()
    res_full = run_guided(net_full, z_full, c_batch, alpha_bars, H_obs,
                          d0, tau, mu, sigma)
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"running guided sampling with ABLATED model + z=0...")
    t0 = time.time()
    res_abl = run_guided(net_abl, z_zero, c_batch, alpha_bars, H_obs,
                         d0, tau, mu, sigma)
    print(f"  done in {time.time()-t0:.1f}s")

    D_full = res_full["D"].cpu().numpy()
    D_abl = res_abl["D"].cpu().numpy()

    C_full = (D_full < hard_thr).astype(np.float32)
    C_abl = (D_abl < hard_thr).astype(np.float32)
    H_full = C_full.mean(axis=0)
    H_abl = C_abl.mean(axis=0)

    iu = np.triu_indices(N, k=1)
    pcc_full = float(np.corrcoef(H_full[iu], H_target[iu])[0, 1])
    pcc_abl = float(np.corrcoef(H_abl[iu], H_target[iu])[0, 1])
    mse_full = float(((H_full - H_target)[iu] ** 2).mean())
    mse_abl = float(((H_abl - H_target)[iu] ** 2).mean())

    Rg_target = np.array([radius_of_gyration(d) for d in D_real[val_idx]])
    Rg_full = np.array([radius_of_gyration(d) for d in D_full])
    Rg_abl = np.array([radius_of_gyration(d) for d in D_abl])

    # Diversity: per-sample pairwise distance to ensemble mean
    def diversity(D_stack: np.ndarray) -> float:
        m = D_stack.mean(axis=0)
        return float(np.sqrt(((D_stack - m[None]) ** 2).mean(axis=(1, 2))).mean())

    div_target = diversity(D_real[val_idx])
    div_full = diversity(D_full)
    div_abl = diversity(D_abl)

    print("\n=== ABLATION RESULTS ===")
    print(f"{'Metric':<40s} {'FULL':>12s} {'ABLATED (z=0)':>15s} {'TARGET':>10s}")
    print(f"{'-'*80}")
    print(f"{'Bulk Pearson (vs target H)':<40s} {pcc_full:>12.4f} {pcc_abl:>15.4f}")
    print(f"{'Bulk MSE (vs target H)':<40s} {mse_full:>12.5f} {mse_abl:>15.5f}")
    print(f"{'Rg median (nm)':<40s} {np.median(Rg_full):>12.1f} {np.median(Rg_abl):>15.1f} {np.median(Rg_target):>10.1f}")
    print(f"{'Rg std (nm)':<40s} {Rg_full.std():>12.1f} {Rg_abl.std():>15.1f} {Rg_target.std():>10.1f}")
    print(f"{'ensemble diversity (RMSD to mean)':<40s} {div_full:>12.1f} {div_abl:>15.1f} {div_target:>10.1f}")

    final_loss_full = res_full["losses"][-1]
    final_loss_abl = res_abl["losses"][-1]
    print(f"\n{'final guidance loss':<40s} {final_loss_full:>12.6f} {final_loss_abl:>15.6f}")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROOT / "checkpoints" / "step11_ablation_results.npz",
        D_full=D_full, D_abl=D_abl, Rg_target=Rg_target,
        Rg_full=Rg_full, Rg_abl=Rg_abl,
        pcc_full=pcc_full, pcc_abl=pcc_abl,
        mse_full=mse_full, mse_abl=mse_abl,
        losses_full=np.array(res_full["losses"]),
        losses_abl=np.array(res_abl["losses"]),
    )

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(H_target, origin="lower", cmap="Reds", vmin=0,
                   vmax=max(H_target.max(), H_full.max(), H_abl.max()))
    ax.set_title(f"target pseudo-bulk\n{len(val_idx)} held-out cells")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(H_full, origin="lower", cmap="Reds", vmin=0,
                   vmax=max(H_target.max(), H_full.max(), H_abl.max()))
    ax.set_title(f"FULL model + z_hat\nPearson={pcc_full:.4f}")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(H_abl, origin="lower", cmap="Reds", vmin=0,
                   vmax=max(H_target.max(), H_full.max(), H_abl.max()))
    ax.set_title(f"ABLATED model + z=0\nPearson={pcc_abl:.4f}")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(res_full["losses"], color="C0", lw=1, label=f"full (final {final_loss_full:.5f})")
    ax.plot(res_abl["losses"], color="C3", lw=1, label=f"ablated (final {final_loss_abl:.5f})")
    ax.set_yscale("log"); ax.set_xlabel("DDIM step"); ax.set_ylabel("guidance MSE")
    ax.set_title("guidance loss trajectories")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    bins = np.linspace(min(Rg_target.min(), Rg_full.min(), Rg_abl.min()),
                       max(Rg_target.max(), Rg_full.max(), Rg_abl.max()), 35)
    ax.hist(Rg_target, bins=bins, density=True, alpha=0.5, color="black",
            label=f"target (med={np.median(Rg_target):.0f})")
    ax.hist(Rg_full, bins=bins, density=True, alpha=0.5, color="C0",
            label=f"full (med={np.median(Rg_full):.0f})")
    ax.hist(Rg_abl, bins=bins, density=True, alpha=0.5, color="C3",
            label=f"ablated (med={np.median(Rg_abl):.0f})")
    ax.set_xlabel("Rg (nm)"); ax.set_ylabel("density")
    ax.set_title("Rg distribution")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    ps_target = p_of_s((D_real[val_idx] < hard_thr).astype(np.float32))
    ps_full = p_of_s(C_full); ps_abl = p_of_s(C_abl)
    seps = np.arange(1, N)
    ax.loglog(seps, ps_target, "o-", ms=3, color="black", label="target")
    ax.loglog(seps, ps_full, "s-", ms=3, color="C0", label="full")
    ax.loglog(seps, ps_abl, "^-", ms=3, color="C3", label="ablated")
    ax.set_xlabel("genomic separation s"); ax.set_ylabel("P(contact | s)")
    ax.set_title("P(s) scaling"); ax.legend(); ax.grid(True, which="both", alpha=0.3)

    ax = fig.add_subplot(gs[2, 0])
    diff_full = H_full - H_target
    vmax = max(float(np.abs(diff_full).max()), 0.1)
    im = ax.imshow(diff_full, origin="lower", cmap="seismic", vmin=-vmax, vmax=vmax)
    ax.set_title("FULL - target residual")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[2, 1])
    diff_abl = H_abl - H_target
    im = ax.imshow(diff_abl, origin="lower", cmap="seismic", vmin=-vmax, vmax=vmax)
    ax.set_title("ABLATED - target residual")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    summary = (
        f"Ablation result\n\n"
        f"Bulk Pearson:\n"
        f"  FULL (z_hat):      {pcc_full:.4f}\n"
        f"  ABLATED (z=0):     {pcc_abl:.4f}\n"
        f"  delta:             {pcc_full - pcc_abl:+.4f}\n\n"
        f"Bulk MSE:\n"
        f"  FULL:              {mse_full:.5f}\n"
        f"  ABLATED:           {mse_abl:.5f}\n\n"
        f"Rg median (nm) vs target {np.median(Rg_target):.0f}:\n"
        f"  FULL:              {np.median(Rg_full):.1f}\n"
        f"  ABLATED:           {np.median(Rg_abl):.1f}\n\n"
        f"Ensemble diversity:\n"
        f"  FULL:              {div_full:.1f}\n"
        f"  ABLATED:           {div_abl:.1f}\n"
        f"  TARGET:            {div_target:.1f}\n\n"
        f"Final guidance MSE:\n"
        f"  FULL:              {final_loss_full:.6f}\n"
        f"  ABLATED:           {final_loss_abl:.6f}"
    )
    ax.text(0.0, 0.95, summary, fontsize=10, va="top", family="monospace")

    fig.suptitle("z-conditioning ablation: full vs z=0 prior on the same target")
    fig.tight_layout()
    out = out_dir / "22_ablation_comparison.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
