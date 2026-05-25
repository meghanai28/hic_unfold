"""Build step 7: maxent reweighting — Stage-2 baseline + sanity check.

Pipeline:
    1. Hold out a set of real Bintu cells (re-use the step-5 val split).
    2. Build the pseudo-bulk H_target = mean of hard 500 nm contact maps on
       those held-out cells.
    3. Sample M conformations from the real-trained prior (step 5) using
       diverse z_hat conditionings from the TRAINING split (never seen as
       part of H_target).
    4. Apply the calibrated forward operator (step 6) to each sample.
    5. Maxent-reweight the samples to match H_target.
    6. Validate that the reweighted ensemble's single-cell statistics
       (Rg distribution, P(s) scaling, per-separation distance distributions)
       recover those of the held-out cells.

Run:
    python scripts/16_maxent_reweight.py
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

from hic_unfold.diffusion import Denoiser, ddim_sample, make_cosine_schedule  # noqa: E402
from hic_unfold.embedding import classical_mds  # noqa: E402
from hic_unfold.forward import apply_forward  # noqa: E402
from hic_unfold.maxent import effective_sample_size, maxent_reweight  # noqa: E402
from hic_unfold.training import make_positional_c  # noqa: E402


def radius_of_gyration(D: np.ndarray) -> float:
    X, _ = classical_mds(D, dim=3)
    com = X.mean(axis=0)
    return float(np.sqrt(((X - com) ** 2).sum(axis=-1).mean()))


def p_of_s_weighted(C_stack: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Weighted P(s) over a stack of contact maps."""
    N = C_stack.shape[-1]
    if w is None:
        Cm = C_stack.mean(axis=0)
    else:
        Cm = (w[:, None, None] * C_stack).sum(axis=0)
    return np.array([np.diag(Cm, k=s).mean() for s in range(1, N)])


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    region = "IMR90_chr21-28-30Mb"

    real_path = ROOT / "data" / "real" / f"{region}_preprocessed.npz"
    diff_ckpt = ROOT / "checkpoints" / "step05_diffusion_real.pt"
    fwd_path = ROOT / "checkpoints" / "step06_forward_params.npz"

    fwd = np.load(fwd_path)
    d0 = float(fwd["d0"]); tau = float(fwd["tau"]); hard_thr = float(fwd["hard_threshold"])
    print(f"calibrated forward params: d0={d0:.1f} nm, tau={tau:.1f} nm (hard ref @ {hard_thr:.0f} nm)")

    f = np.load(real_path)
    D_real = f["D"]; z_hat = f["z_hat"]
    N = int(f["N"]); mu = float(f["mu"]); sigma = float(f["sigma"])

    ckpt = torch.load(diff_ckpt, map_location=device, weights_only=False)
    val_idx = np.array(ckpt["val_idx"])
    train_idx = np.setdiff1d(np.arange(D_real.shape[0]), val_idx)
    print(f"real corpus: total={D_real.shape[0]}, train (for prior pool)={len(train_idx)}, "
          f"val (for H_target)={len(val_idx)}")

    H_hard_val = (D_real[val_idx] < hard_thr).mean(axis=0).astype(np.float32)

    net = Denoiser(N=N, d_c=int(ckpt["d_c"]), d_pair=32, d_sep=16, d_h=96,
                   d_t=128, dilations=(1, 2, 4, 8, 1)).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    alpha_bars = make_cosine_schedule(T=int(ckpt["T"]), device=device)
    c_const = make_positional_c(N, int(ckpt["d_c"]), device)

    M_samples = 2000
    rng = np.random.default_rng(2026)
    sampled_z_idx = rng.choice(train_idx, size=M_samples, replace=True)
    z_pool = torch.from_numpy(z_hat[sampled_z_idx])[:, None].to(device)

    print(f"sampling {M_samples} conformations from real-trained prior (DDIM 100 steps)...")
    t0 = time.time()
    D_samples = np.empty((M_samples, N, N), dtype=np.float32)
    batch = 64
    with torch.no_grad():
        for s in range(0, M_samples, batch):
            e = min(s + batch, M_samples)
            x_s = ddim_sample(net, z_pool[s:e], c_const.expand(e - s, -1, -1),
                              alpha_bars, n_steps=100)
            D_b = np.expm1(x_s.squeeze(1).cpu().numpy() * sigma + mu)
            D_b = np.maximum(D_b, 0)
            for k in range(D_b.shape[0]):
                D_b[k] = 0.5 * (D_b[k] + D_b[k].T)
                np.fill_diagonal(D_b[k], 0)
            D_samples[s:e] = D_b
            if (s // batch) % 4 == 0:
                print(f"  sampled {e}/{M_samples} ({time.time()-t0:.1f}s)")
    print(f"sampling done in {time.time()-t0:.1f}s")

    print("computing per-sample soft contact maps...")
    C_samples = apply_forward(D_samples, d0=d0, tau=tau).reshape(1, N, N)  # garbage shape
    # apply_forward returns the mean; we need per-sample C. Recompute:
    from hic_unfold.forward import soft_contact
    C_samples = soft_contact(D_samples, d0=d0, tau=tau).astype(np.float32)

    H_pool = C_samples.mean(axis=0)
    iu = np.triu_indices(N, k=1)
    pcc_pool = float(np.corrcoef(H_pool[iu], H_hard_val[iu])[0, 1])
    mse_pool = float(((H_pool - H_hard_val)[iu] ** 2).mean())
    print(f"prior-pool bulk vs val pseudo-bulk: Pearson={pcc_pool:.4f}, MSE={mse_pool:.5f}")

    print("solving maxent dual...")
    t0 = time.time()
    res = maxent_reweight(C_samples, H_hard_val, lr=0.05, num_steps=1500,
                          l2=1e-5, device=str(device), log_every=300)
    print(f"  done in {time.time()-t0:.1f}s")
    print(f"  ESS = {res['eff_M']:.1f} / {M_samples}")
    print(f"  reweighted fit: Pearson={res['fit_pearson']:.4f}, MSE={res['fit_mse']:.6f}")

    w = res["weights"]

    Rg_val = np.array([radius_of_gyration(d) for d in D_real[val_idx]])
    Rg_samp = np.array([radius_of_gyration(d) for d in D_samples])
    Rg_unif_med = float(np.median(Rg_samp))
    rng_resample = np.random.default_rng(2027)
    n_resample = len(val_idx)
    idx_w = rng_resample.choice(M_samples, size=n_resample, replace=True, p=w / w.sum())
    Rg_reweighted = Rg_samp[idx_w]
    print(f"Rg medians (nm): val(target)={np.median(Rg_val):.1f}, "
          f"prior-uniform={Rg_unif_med:.1f}, reweighted={np.median(Rg_reweighted):.1f}")

    C_val_hard = (D_real[val_idx] < hard_thr).astype(np.float32)
    ps_val = p_of_s_weighted(C_val_hard)
    ps_prior = p_of_s_weighted(C_samples)
    ps_reweighted = p_of_s_weighted(C_samples, w)
    seps = np.arange(1, N)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = ROOT / "checkpoints" / "step07_maxent.npz"
    np.savez_compressed(out_ckpt,
        weights=w, lam=res["lambda"], eff_M=res["eff_M"],
        fit_mse=res["fit_mse"], fit_pearson=res["fit_pearson"],
        sampled_z_idx=sampled_z_idx, d0=d0, tau=tau,
    )
    print(f"saved {out_ckpt}")

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(H_hard_val, origin="lower", cmap="Reds", vmin=0, vmax=H_hard_val.max())
    ax.set_title(f"target pseudo-bulk\nP(d<{hard_thr:.0f}nm) on {len(val_idx)} held-out cells")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(H_pool, origin="lower", cmap="Reds", vmin=0, vmax=H_hard_val.max())
    ax.set_title(f"prior pool (uniform weights)\nPearson={pcc_pool:.3f}")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(res["H_pred"], origin="lower", cmap="Reds", vmin=0, vmax=H_hard_val.max())
    ax.set_title(f"maxent-reweighted\nPearson={res['fit_pearson']:.4f}")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, 0])
    ax.loglog(seps, ps_val, "o-", ms=3, color="black", label="real held-out")
    ax.loglog(seps, ps_prior, "s-", ms=3, color="gray", label="prior pool (uniform)")
    ax.loglog(seps, ps_reweighted, "^-", ms=3, color="C3", label="maxent-reweighted")
    ax.set_xlabel("genomic separation s"); ax.set_ylabel("P(contact | s)")
    ax.set_title("P(s) scaling"); ax.legend(); ax.grid(True, which="both", alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    bins = np.linspace(min(Rg_val.min(), Rg_samp.min()),
                       max(Rg_val.max(), Rg_samp.max()), 35)
    ax.hist(Rg_val, bins=bins, density=True, alpha=0.5, color="black",
            label=f"real held-out (med={np.median(Rg_val):.0f})")
    ax.hist(Rg_samp, bins=bins, density=True, alpha=0.4, color="gray",
            label=f"prior uniform (med={Rg_unif_med:.0f})")
    ax.hist(Rg_reweighted, bins=bins, density=True, alpha=0.5, color="C3",
            label=f"maxent-reweighted (med={np.median(Rg_reweighted):.0f})")
    ax.set_xlabel("radius of gyration Rg (nm)"); ax.set_ylabel("density")
    ax.set_title("Rg distribution"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    im = ax.imshow(res["lambda"], origin="lower", cmap="seismic",
                   vmin=-np.abs(res["lambda"]).max(), vmax=np.abs(res["lambda"]).max())
    ax.set_title(f"Lagrange multipliers lambda\n(non-zero -> data demands a shift)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Step-7 maxent reweighting: prior -> deconvolved ensemble")
    fig.tight_layout()
    out = out_dir / "16_maxent_reweight.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
