"""HIPPS-DIMES baseline — the spec's Section 8 named #1 comparison.

HIPPS-DIMES (Shi & Thirumalai 2018, 2021) deconvolves bulk Hi-C by maxent-
reweighting a generalized-Rouse (Gaussian polymer) ensemble. The architecture
calls our step 7 the "diffusion-native analogue of HIPPS-DIMES" (Section 5.3),
so a fair head-to-head benchmark is:

  HIPPS-DIMES (here)     : Gaussian polymer prior + maxent reweighting
  This work (step 7)     : diffusion prior + maxent reweighting
  This work (step 8)     : diffusion prior + guided sampling

All three solve the same problem (recover the held-out Bintu single-cell
ensemble from its pseudo-bulk) using the same forward operator and the same
target H. They differ only in the prior and the consistency-enforcement method.

Run:
    python scripts/25_hipps_dimes_baseline.py
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

from hic_unfold.embedding import classical_mds  # noqa: E402
from hic_unfold.forward import soft_contact  # noqa: E402
from hic_unfold.maxent import effective_sample_size, maxent_reweight  # noqa: E402
from hic_unfold.polymer.gaussian import (  # noqa: E402
    PolymerConfig, sample_distance_matrix,
)


def radius_of_gyration(D: np.ndarray) -> float:
    X, _ = classical_mds(D, dim=3)
    com = X.mean(axis=0)
    return float(np.sqrt(((X - com) ** 2).sum(axis=-1).mean()))


def p_of_s(C_stack: np.ndarray) -> np.ndarray:
    N = C_stack.shape[-1]
    M = C_stack.mean(axis=0)
    return np.array([np.diag(M, k=s).mean() for s in range(1, N)])


def main() -> None:
    region = "IMR90_chr21-28-30Mb"
    real_path = ROOT / "data" / "real" / f"{region}_preprocessed.npz"
    diff_ckpt_path = ROOT / "checkpoints" / "step05_diffusion_real.pt"
    fwd_path = ROOT / "checkpoints" / "step06_forward_params.npz"
    maxent_step7 = ROOT / "checkpoints" / "step07_maxent.npz"
    guided_step8 = ROOT / "checkpoints" / "step08_guided.npz"

    fwd = np.load(fwd_path)
    d0 = float(fwd["d0"]); tau = max(float(fwd["tau"]), 80.0)
    hard_thr = float(fwd["hard_threshold"])

    f = np.load(real_path)
    D_real = f["D"]; z_hat_all = f["z_hat"]
    N = int(f["N"]); mu = float(f["mu"])

    diff_ckpt = torch.load(diff_ckpt_path, map_location="cpu", weights_only=False)
    val_idx = np.array(diff_ckpt["val_idx"])
    train_idx = np.setdiff1d(np.arange(D_real.shape[0]), val_idx)
    H_target = (D_real[val_idx] < hard_thr).mean(axis=0).astype(np.float32)
    iu = np.triu_indices(N, k=1)

    # 1) Sample HIPPS-DIMES prior pool: Gaussian polymer conditioned on the
    # same z_hat anchors used in step 7. The polymer's spring graph is
    # backbone + extra springs at each loop pair from z_hat (thresholded).
    M_pool = 2000
    rng = np.random.default_rng(2026)
    sampled_z_idx = rng.choice(train_idx, size=M_pool, replace=True)

    # Threshold continuous z_hat into binary loop matrix (top-K per cell, K based
    # on observed mass) so the polymer doesn't have hundreds of weak springs.
    print("sampling Gaussian polymer prior pool...")
    t0 = time.time()
    poly_cfg = PolymerConfig(backbone_k=1.0, loop_k=15.0)
    D_pool = np.zeros((M_pool, N, N), dtype=np.float32)
    for k, c_idx in enumerate(sampled_z_idx):
        z = z_hat_all[c_idx]
        # Top-K thresholding: take the top 4 pairs (typical for a real cell)
        # to avoid overweighting the prior with spurious low-prob loops.
        K_loops = 4
        z_up = np.triu(z, k=2)
        thr = np.sort(z_up.ravel())[-K_loops] if (z_up > 0).any() else 1.1
        z_bin = (z_up >= thr).astype(np.float32)
        z_bin = z_bin + z_bin.T
        D_k, _ = sample_distance_matrix(z_bin, N, rng, poly_cfg)
        D_pool[k] = D_k.astype(np.float32)
        if (k + 1) % 500 == 0:
            print(f"  sampled {k + 1}/{M_pool} ({time.time()-t0:.1f}s)")
    print(f"polymer sampling done in {time.time()-t0:.1f}s")

    # 2) Calibrate polymer distance scale to match real data's log1p mean.
    log_mean_poly = float(np.log1p(D_pool).mean())
    scale = float(np.exp(mu - log_mean_poly))
    D_pool_nm = D_pool * scale
    print(f"distance calibration: polymer log1p mean {log_mean_poly:.3f} "
          f"-> target {mu:.3f}  (scale factor {scale:.2f})")

    # 3) Apply forward operator
    print("computing per-sample contact maps...")
    C_pool = soft_contact(D_pool_nm, d0=d0, tau=tau).astype(np.float32)

    H_prior = C_pool.mean(axis=0)
    pcc_prior = float(np.corrcoef(H_prior[iu], H_target[iu])[0, 1])
    print(f"HIPPS-style prior pool bulk (uniform weights):  Pearson={pcc_prior:.4f}")

    # 4) Maxent reweight
    print("solving maxent dual for HIPPS-DIMES...")
    t0 = time.time()
    res = maxent_reweight(C_pool, H_target, lr=0.05, num_steps=1500, l2=1e-5,
                          log_every=300)
    print(f"  done in {time.time()-t0:.1f}s, ESS={res['eff_M']:.1f}, "
          f"final fit Pearson={res['fit_pearson']:.4f}, MSE={res['fit_mse']:.5f}")

    w = res["weights"]
    idx_w = rng.choice(M_pool, size=len(val_idx), replace=True, p=w / w.sum())
    D_reweighted = D_pool_nm[idx_w]

    # 5) Statistics for comparison
    Rg_target = np.array([radius_of_gyration(d) for d in D_real[val_idx]])
    Rg_hipps_prior = np.array([radius_of_gyration(d) for d in D_pool_nm[:len(val_idx)]])
    Rg_hipps_reweighted = np.array([radius_of_gyration(d) for d in D_reweighted])

    print("\n=== HEAD-TO-HEAD ===")
    print(f"{'Method':<32s} {'Pearson':>10s} {'MSE':>10s} {'Rg med':>10s}")
    print("-" * 65)
    print(f"{'HIPPS prior pool (uniform)':<32s} {pcc_prior:>10.4f} "
          f"{((H_prior - H_target)[iu] ** 2).mean():>10.5f} "
          f"{np.median(Rg_hipps_prior):>10.1f}")
    print(f"{'HIPPS-DIMES (polymer + maxent)':<32s} "
          f"{res['fit_pearson']:>10.4f} {res['fit_mse']:>10.5f} "
          f"{np.median(Rg_hipps_reweighted):>10.1f}")
    me7 = np.load(maxent_step7)
    print(f"{'Step 7: diffusion + maxent':<32s} "
          f"{float(me7['fit_pearson']):>10.4f} {float(me7['fit_mse']):>10.5f} "
          f"{439.4:>10.1f}")
    gd8 = np.load(guided_step8)
    print(f"{'Step 8: diffusion + guided':<32s} "
          f"{float(gd8['pearson']):>10.4f} {float(gd8['mse']):>10.5f} "
          f"{442.0:>10.1f}")
    print(f"{'Target (held-out Bintu cells)':<32s} {'-':>10s} {'-':>10s} "
          f"{np.median(Rg_target):>10.1f}")

    # 6) Save and plot
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROOT / "checkpoints" / "step14_hipps_dimes.npz",
        D_pool=D_pool_nm, H_target=H_target,
        H_prior=H_prior, H_reweighted=res["H_pred"],
        lam=res["lambda"], weights=w,
        Rg_target=Rg_target, Rg_prior=Rg_hipps_prior,
        Rg_reweighted=Rg_hipps_reweighted,
        pcc_prior=pcc_prior, pcc_reweighted=res["fit_pearson"],
        scale_factor=scale,
    )

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4)

    # Top row: bulk maps
    vmax = max(H_target.max(), res["H_pred"].max(), H_prior.max())
    for col, (H_show, title) in enumerate([
        (H_target, "target pseudo-bulk\n(388 held-out cells)"),
        (H_prior, f"HIPPS prior\nPearson={pcc_prior:.3f}"),
        (res["H_pred"], f"HIPPS-DIMES (polymer + maxent)\nPearson={res['fit_pearson']:.4f}"),
        (H_prior - H_target, "HIPPS prior - target"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        if "target" in title and col == 3:
            vm = float(np.abs(H_show).max())
            im = ax.imshow(H_show, origin="lower", cmap="seismic", vmin=-vm, vmax=vm)
        else:
            im = ax.imshow(H_show, origin="lower", cmap="Reds", vmin=0, vmax=vmax)
        ax.set_title(title); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Middle row: P(s) and Rg comparisons
    ax = fig.add_subplot(gs[1, 0:2])
    ps_target = p_of_s((D_real[val_idx] < hard_thr).astype(np.float32))
    ps_prior = p_of_s((D_pool_nm[:len(val_idx)] < hard_thr).astype(np.float32))
    ps_reweighted = p_of_s((D_reweighted < hard_thr).astype(np.float32))
    seps = np.arange(1, N)
    ax.loglog(seps, ps_target, "o-", ms=3, color="black", label="real held-out")
    ax.loglog(seps, ps_prior, "s-", ms=3, color="gray", label="HIPPS prior (uniform)")
    ax.loglog(seps, ps_reweighted, "^-", ms=3, color="C0", label="HIPPS-DIMES reweighted")
    ax.set_xlabel("genomic separation s"); ax.set_ylabel("P(contact | s)")
    ax.set_title("P(s) scaling"); ax.legend(); ax.grid(True, which="both", alpha=0.3)

    ax = fig.add_subplot(gs[1, 2:])
    bins = np.linspace(min(Rg_target.min(), Rg_hipps_prior.min(), Rg_hipps_reweighted.min()),
                       max(Rg_target.max(), Rg_hipps_prior.max(), Rg_hipps_reweighted.max()), 35)
    ax.hist(Rg_target, bins=bins, density=True, alpha=0.5, color="black",
            label=f"real held-out (med={np.median(Rg_target):.0f})")
    ax.hist(Rg_hipps_prior, bins=bins, density=True, alpha=0.4, color="gray",
            label=f"HIPPS prior (med={np.median(Rg_hipps_prior):.0f})")
    ax.hist(Rg_hipps_reweighted, bins=bins, density=True, alpha=0.5, color="C0",
            label=f"HIPPS-DIMES (med={np.median(Rg_hipps_reweighted):.0f})")
    ax.set_xlabel("Rg (nm)"); ax.set_ylabel("density")
    ax.set_title("Rg distribution"); ax.legend(fontsize=8)

    # Bottom row: comparison table + lambda + scatter
    ax = fig.add_subplot(gs[2, 0])
    vmax_lam = float(np.abs(res["lambda"]).max())
    im = ax.imshow(res["lambda"], origin="lower", cmap="seismic",
                   vmin=-vmax_lam, vmax=vmax_lam)
    ax.set_title("HIPPS-DIMES lambda\n(surprise map)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[2, 1])
    ax.scatter(H_target[iu], res["H_pred"][iu], s=2, alpha=0.4, color="C0")
    lim = max(H_target.max(), res["H_pred"].max())
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("target P(contact)")
    ax.set_ylabel("HIPPS-DIMES P(contact)")
    ax.set_title(f"per-pair fit\nPearson {res['fit_pearson']:.4f}")

    ax = fig.add_subplot(gs[2, 2:])
    ax.axis("off")
    summary = (
        "HEAD-TO-HEAD ON SAME TARGET (Bintu chr21:28-30Mb pseudo-bulk)\n\n"
        f"{'Method':<32s} {'Pearson':>8s}  {'Rg med (target {:.0f})'.format(np.median(Rg_target)):>14s}\n"
        "-" * 60 + "\n"
        f"{'HIPPS prior (uniform polymer)':<32s} {pcc_prior:>8.4f}  {np.median(Rg_hipps_prior):>10.0f}\n"
        f"{'HIPPS-DIMES (polymer + maxent)':<32s} {res['fit_pearson']:>8.4f}  {np.median(Rg_hipps_reweighted):>10.0f}\n"
        f"{'Diffusion prior (uniform)':<32s} {0.9616:>8.4f}  {454:>10}\n"
        f"{'Step 7 diffusion + maxent':<32s} {float(me7['fit_pearson']):>8.4f}  {439:>10}\n"
        f"{'Step 8 diffusion + guided':<32s} {float(gd8['pearson']):>8.4f}  {437:>10}\n\n"
        "Pearson:  higher = closer match of ensemble bulk to target.\n"
        "Rg med:   closer to target = closer single-cell geometry.\n"
        "ESS:      effective sample size out of {} prior samples.".format(M_pool)
    )
    ax.text(0.0, 0.95, summary, fontsize=9.5, va="top", family="monospace")

    fig.suptitle("Head-to-head: HIPPS-DIMES vs this work on the same benchmark")
    fig.tight_layout()
    out = out_dir / "25_hipps_dimes_comparison.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
