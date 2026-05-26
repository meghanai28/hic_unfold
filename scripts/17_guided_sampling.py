"""Build step 8 — Stage-2 guided sampling, the architecture's headline method.

Setup is identical to step 7 so the methods are directly comparable:
    target H : pseudo-bulk from the step-5 val cells (388 cells, d < 500 nm)
    prior     : real-trained denoiser (step-5 checkpoint)
    forward   : calibrated d0, tau (step-6 checkpoint)
    baseline  : maxent reweighting (step 7)

Expectation per spec: guided sampling should match or beat maxent on bulk fit
AND on single-cell statistics (Rg, P(s)).

Run:
    python scripts/17_guided_sampling.py
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
    Denoiser,
    guided_ddim_sample,
    make_cosine_schedule,
)
from hic_unfold.embedding import classical_mds  # noqa: E402
from hic_unfold.forward import soft_contact  # noqa: E402
from hic_unfold.training import make_positional_c  # noqa: E402


def radius_of_gyration(D: np.ndarray) -> float:
    X, _ = classical_mds(D, dim=3)
    com = X.mean(axis=0)
    return float(np.sqrt(((X - com) ** 2).sum(axis=-1).mean()))


def p_of_s(C_stack: np.ndarray) -> np.ndarray:
    N = C_stack.shape[-1]
    M = C_stack.mean(axis=0)
    return np.array([np.diag(M, k=s).mean() for s in range(1, N)])


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    region = "IMR90_chr21-28-30Mb"

    real_path = ROOT / "data" / "real" / f"{region}_preprocessed.npz"
    diff_ckpt = ROOT / "checkpoints" / "step05_diffusion_real.pt"
    fwd_path = ROOT / "checkpoints" / "step06_forward_params.npz"
    maxent_path = ROOT / "checkpoints" / "step07_maxent.npz"

    fwd = np.load(fwd_path)
    # Use a slightly larger tau than the optimum (30 nm) so gradients flow
    # well during guided sampling. Pearson is still > 0.99 here.
    d0 = float(fwd["d0"])
    tau = max(float(fwd["tau"]), 80.0)
    print(f"forward params for guidance: d0={d0:.0f} nm, tau={tau:.0f} nm")

    f = np.load(real_path)
    D_real = f["D"]; z_hat = f["z_hat"]
    N = int(f["N"]); mu = float(f["mu"]); sigma = float(f["sigma"])
    hard_thr = float(fwd["hard_threshold"])

    ckpt = torch.load(diff_ckpt, map_location=device, weights_only=False)
    val_idx = np.array(ckpt["val_idx"])
    train_idx = np.setdiff1d(np.arange(D_real.shape[0]), val_idx)
    print(f"real corpus: train={len(train_idx)}, val (target ensemble)={len(val_idx)}")

    H_target = (D_real[val_idx] < hard_thr).mean(axis=0).astype(np.float32)
    H_obs = torch.tensor(H_target, device=device)

    net = Denoiser(N=N, d_c=int(ckpt["d_c"]), d_pair=32, d_sep=16, d_h=96,
                   d_t=128, dilations=(1, 2, 4, 8, 1)).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    alpha_bars = make_cosine_schedule(T=int(ckpt["T"]), device=device)
    c_const = make_positional_c(N, int(ckpt["d_c"]), device)

    M = 128
    rng = np.random.default_rng(2026)
    z_idx = rng.choice(train_idx, size=M, replace=True)
    z_init = torch.from_numpy(z_hat[z_idx])[:, None].to(device)
    c_batch = c_const.expand(M, -1, -1)

    eta = 30000.0
    n_steps = 200
    print(f"running guided DDIM: M={M}, n_steps={n_steps}, eta={eta}")
    t0 = time.time()
    res = guided_ddim_sample(
        net, z_init, c_batch, alpha_bars, H_obs,
        d0=d0, tau=tau, mu=mu, sigma=sigma,
        n_steps=n_steps, eta=eta, log_every=40,
    )
    print(f"guided sampling done in {time.time()-t0:.1f}s")
    D_samp = res["D"].cpu().numpy()

    C_samp_hard = (D_samp < hard_thr).astype(np.float32)
    H_pred_hard = C_samp_hard.mean(axis=0)
    iu = np.triu_indices(N, k=1)
    pcc_guided = float(np.corrcoef(H_pred_hard[iu], H_target[iu])[0, 1])
    mse_guided = float(((H_pred_hard - H_target)[iu] ** 2).mean())
    print(f"\nguided bulk fit (hard 500nm): Pearson={pcc_guided:.4f}, MSE={mse_guided:.5f}")

    me = np.load(maxent_path)
    print(f"  step-7 maxent reference:    Pearson={float(me['fit_pearson']):.4f}, "
          f"MSE={float(me['fit_mse']):.5f}")

    print("computing single-cell statistics...")
    Rg_target = np.array([radius_of_gyration(d) for d in D_real[val_idx]])
    Rg_guided = np.array([radius_of_gyration(d) for d in D_samp])
    print(f"Rg median (nm): target={np.median(Rg_target):.1f}, "
          f"guided={np.median(Rg_guided):.1f}")

    C_target_hard = (D_real[val_idx] < hard_thr).astype(np.float32)
    ps_target = p_of_s(C_target_hard)
    ps_guided = p_of_s(C_samp_hard)
    seps = np.arange(1, N)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_out = ROOT / "checkpoints" / "step08_guided.npz"
    np.savez_compressed(ckpt_out,
        D_samples=D_samp, z_idx=z_idx, d0=d0, tau=tau,
        losses=np.array(res["losses"]),
        grad_norms=np.array(res["grad_norms"]),
        pearson=pcc_guided, mse=mse_guided, eta=eta, n_steps=n_steps,
    )
    print(f"saved {ckpt_out}")

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 3)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(H_target, origin="lower", cmap="Reds", vmin=0, vmax=H_target.max())
    ax.set_title(f"target pseudo-bulk\n{len(val_idx)} held-out real cells")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(H_pred_hard, origin="lower", cmap="Reds", vmin=0, vmax=H_target.max())
    ax.set_title(f"guided ensemble bulk\nPearson={pcc_guided:.4f}")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 2])
    diff = H_pred_hard - H_target
    vmax = float(np.abs(diff).max())
    im = ax.imshow(diff, origin="lower", cmap="seismic", vmin=-vmax, vmax=vmax)
    ax.set_title("guided - target")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(res["losses"], lw=1, color="C3")
    ax.set_yscale("log"); ax.set_xlabel("DDIM step"); ax.set_ylabel("ensemble MSE(H_pred, H_obs)")
    ax.set_title("guidance-loss trajectory"); ax.grid(True, which="both", alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(res["grad_norms"], lw=1, color="C2")
    ax.set_yscale("log"); ax.set_xlabel("DDIM step"); ax.set_ylabel("||grad_x loss||")
    ax.set_title("guidance gradient norm"); ax.grid(True, which="both", alpha=0.3)

    ax = fig.add_subplot(gs[1, 2])
    ax.loglog(seps, ps_target, "o-", ms=3, color="black", label="real held-out")
    ax.loglog(seps, ps_guided, "^-", ms=3, color="C3", label="guided")
    ax.set_xlabel("genomic separation s"); ax.set_ylabel("P(contact | s)")
    ax.set_title("P(s) scaling"); ax.legend(); ax.grid(True, which="both", alpha=0.3)

    ax = fig.add_subplot(gs[2, 0])
    bins = np.linspace(min(Rg_target.min(), Rg_guided.min()),
                       max(Rg_target.max(), Rg_guided.max()), 35)
    ax.hist(Rg_target, bins=bins, density=True, alpha=0.5, color="black",
            label=f"real held-out (med={np.median(Rg_target):.0f})")
    ax.hist(Rg_guided, bins=bins, density=True, alpha=0.5, color="C3",
            label=f"guided (med={np.median(Rg_guided):.0f})")
    ax.set_xlabel("Rg (nm)"); ax.set_ylabel("density")
    ax.set_title("radius of gyration distribution"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, 1])
    n_show = 4
    sample_show = D_samp[:n_show]
    target_show = D_real[val_idx[:n_show]]
    grid_h = np.concatenate([np.concatenate(list(sample_show), axis=1),
                             np.concatenate(list(target_show), axis=1)], axis=0)
    im = ax.imshow(grid_h, origin="lower", cmap="viridis", aspect="auto")
    ax.set_title("top: 4 guided samples\nbottom: 4 real cells")
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[2, 2])
    # Per-pair scatter target vs guided
    ax.scatter(H_target[iu], H_pred_hard[iu], s=2, alpha=0.4, color="C3")
    lim = max(H_target.max(), H_pred_hard.max())
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("target P(contact)"); ax.set_ylabel("guided P(contact)")
    ax.set_title(f"per-pair fit (Pearson={pcc_guided:.4f})")

    fig.suptitle(f"Step-8 guided sampling — Stage-2 deconvolution ({region})")
    fig.tight_layout()
    out = out_dir / "17_guided_sampling.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")

    print("\n=== SUMMARY ===")
    print(f"  step 5 prior pool (uniform):  Pearson 0.962  (vs target H)")
    print(f"  step 7 maxent reweighted:     Pearson {float(me['fit_pearson']):.4f}")
    print(f"  step 8 guided sampling:       Pearson {pcc_guided:.4f}")
    print(f"  Rg median target:             {np.median(Rg_target):.1f} nm")
    print(f"  Rg median guided:             {np.median(Rg_guided):.1f} nm")


if __name__ == "__main__":
    main()
