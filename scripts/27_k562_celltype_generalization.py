"""Cross-cell-type generalization: deconvolve K562 (leukemia) chr21:28-30Mb
with the IMR90-trained prior, encoder, and forward operator.

Same region as the training data but a different cell type. Compares:
    - IMR90-trained pipeline applied to K562  -> K562-specific deconvolution
    - Measured K562 bulk + single-cell stats   -> ground truth
    - IMR90 deconvolution from step 8          -> what the IMR90 looks like

If the architecture captures general chromatin biology (rather than IMR90
idiosyncrasies), the K562 deconvolution should reproduce K562's specific
patterns, not just regress to the IMR90 distribution.

Run:
    python scripts/27_k562_celltype_generalization.py
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

from hic_unfold.data import load_bintu_csv, preprocess_bintu  # noqa: E402
from hic_unfold.diffusion import (  # noqa: E402
    Denoiser, guided_ddim_sample, make_cosine_schedule,
)
from hic_unfold.embedding import classical_mds  # noqa: E402
from hic_unfold.encoder import LoopEncoder  # noqa: E402
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
    enc_ckpt_path = ROOT / "checkpoints" / "step05_encoder_N65.pt"
    diff_ckpt_path = ROOT / "checkpoints" / "step05_diffusion_real.pt"
    fwd_path = ROOT / "checkpoints" / "step06_forward_params.npz"

    fwd = np.load(fwd_path)
    d0 = float(fwd["d0"]); tau = max(float(fwd["tau"]), 80.0)
    hard_thr = float(fwd["hard_threshold"])

    print("loading K562 chr21:28-30Mb...")
    ds = load_bintu_csv(ROOT / "data" / "raw_bintu2018" / "K562_chr21-28-30Mb.csv")
    real = preprocess_bintu(ds, min_valid_frac=0.85)
    N = real.D.shape[-1]
    print(f"  preprocessed: {real.D.shape[0]} K562 cells, N={N}")

    # Compare measured K562 vs IMR90 (load IMR90 preprocessed)
    f_imr = np.load(ROOT / "data" / "real" / "IMR90_chr21-28-30Mb_preprocessed.npz")
    D_imr = f_imr["D"]; z_imr = f_imr["z_hat"]
    diff_ckpt = torch.load(diff_ckpt_path, map_location=device, weights_only=False)
    val_imr_idx = np.array(diff_ckpt["val_idx"])

    # Side-by-side measured population contact maps
    H_imr = (D_imr[val_imr_idx] < hard_thr).mean(axis=0).astype(np.float32)
    H_k = (real.D < hard_thr).mean(axis=0).astype(np.float32)
    iu = np.triu_indices(N, k=1)
    pcc_k_vs_imr = float(np.corrcoef(H_k[iu], H_imr[iu])[0, 1])
    print(f"measured K562 vs measured IMR90 bulk Pearson: {pcc_k_vs_imr:.4f}")
    print("  (cell-type-specific patterns are real but partially overlapping)")

    Rg_k_meas = np.array([radius_of_gyration(d) for d in real.D])
    Rg_imr_meas = np.array([radius_of_gyration(d) for d in D_imr[val_imr_idx]])
    print(f"Rg medians (nm): K562={np.median(Rg_k_meas):.1f}, "
          f"IMR90={np.median(Rg_imr_meas):.1f}")

    # Encode K562 cells with IMR90-sim-trained encoder
    enc_ckpt = torch.load(enc_ckpt_path, map_location=device, weights_only=False)
    enc = LoopEncoder(N=N, d_c=int(enc_ckpt["d_c"]), d_pair=32, d_sep=16, d_h=96,
                      dilations=(1, 2, 4, 8, 1)).to(device)
    enc.load_state_dict(enc_ckpt["state_dict"]); enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    c_const = make_positional_c(N, int(enc_ckpt["d_c"]), device)

    mu_k = float(np.log1p(real.D).mean()); sigma_k = float(np.log1p(real.D).std())
    print(f"K562 log1p stats: mu={mu_k:.3f}, sigma={sigma_k:.3f}")
    x_k = ((np.log1p(real.D) - mu_k) / max(sigma_k, 1e-8)).astype(np.float32)
    print("encoding K562 cells -> z_hat...")
    z_hat_k = np.empty((x_k.shape[0], N, N), dtype=np.float32)
    bs = 64
    with torch.no_grad():
        for s in range(0, x_k.shape[0], bs):
            e = min(s + bs, x_k.shape[0])
            x_b = torch.from_numpy(x_k[s:e])[:, None].to(device)
            c_b = c_const.expand(e - s, -1, -1)
            z_hat_k[s:e] = torch.sigmoid(enc(x_b, c_b))[:, 0].cpu().numpy()

    # Split, target H from val
    rng = np.random.default_rng(2027)
    perm = rng.permutation(x_k.shape[0])
    n_val = int(0.1 * x_k.shape[0])
    val_idx_k = perm[:n_val]
    train_idx_k = perm[n_val:]
    print(f"K562 split: train={len(train_idx_k)}, val={n_val}")
    H_target = (real.D[val_idx_k] < hard_thr).mean(axis=0).astype(np.float32)

    # Load IMR90-trained diffusion model
    net = Denoiser(N=N, d_c=int(diff_ckpt["d_c"]), d_pair=32, d_sep=16, d_h=96,
                   d_t=128, dilations=(1, 2, 4, 8, 1)).to(device)
    net.load_state_dict(diff_ckpt["state_dict"]); net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    alpha_bars = make_cosine_schedule(T=int(diff_ckpt["T"]), device=device)
    mu_train = float(diff_ckpt["mu"]); sigma_train = float(diff_ckpt["sigma"])

    M_samp = 128
    z_idx = rng.choice(train_idx_k, size=M_samp, replace=True)
    z_pool = torch.from_numpy(z_hat_k[z_idx])[:, None].to(device)
    c_batch = c_const.expand(M_samp, -1, -1)
    H_obs = torch.tensor(H_target, device=device)

    print(f"\nrunning guided DDIM on K562 (M={M_samp}, 200 steps, eta=5000)...")
    t0 = time.time()
    res = guided_ddim_sample(
        net, z_pool, c_batch, alpha_bars, H_obs,
        d0=d0, tau=tau, mu=mu_train, sigma=sigma_train,
        n_steps=200, eta=5000.0, log_every=50,
    )
    print(f"done in {time.time()-t0:.1f}s")
    D_samp = res["D"].cpu().numpy()

    C_samp_hard = (D_samp < hard_thr).astype(np.float32)
    H_pred = C_samp_hard.mean(axis=0)
    pcc = float(np.corrcoef(H_pred[iu], H_target[iu])[0, 1])
    mse = float(((H_pred - H_target)[iu] ** 2).mean())
    print(f"\nbulk fit on K562:    Pearson={pcc:.4f}, MSE={mse:.5f}")
    print(f"  reference (IMR90 in-domain, step 8): Pearson=0.9869")
    print(f"  reference (chr21:18-20Mb generalization, step 23): Pearson=0.9748")

    Rg_pred = np.array([radius_of_gyration(d) for d in D_samp])
    print(f"Rg medians (nm): K562 target={np.median(Rg_k_meas):.1f}, "
          f"K562 guided={np.median(Rg_pred):.1f}")

    # Cell-type discrimination check: is the K562 prediction closer to K562 truth
    # than to the IMR90 truth?
    pcc_kpred_vs_k = pcc
    pcc_kpred_vs_imr = float(np.corrcoef(H_pred[iu], H_imr[iu])[0, 1])
    print(f"\nCell-type discrimination:")
    print(f"  K562 guided  vs  K562 measured  bulk: Pearson={pcc_kpred_vs_k:.4f}")
    print(f"  K562 guided  vs  IMR90 measured bulk: Pearson={pcc_kpred_vs_imr:.4f}")
    if pcc_kpred_vs_k > pcc_kpred_vs_imr:
        print("  -> K562 prediction is MORE LIKE K562 than IMR90: cell-type-specific recovery works.")
    else:
        print("  -> K562 prediction is more like IMR90: cell-type discrimination fails.")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROOT / "checkpoints" / "step16_k562_generalization.npz",
        D_samples=D_samp, H_target=H_target, H_pred=H_pred,
        H_k_meas=H_k, H_imr_meas=H_imr,
        Rg_target=Rg_k_meas, Rg_pred=Rg_pred,
        pearson_k=pcc_kpred_vs_k, pearson_vs_imr=pcc_kpred_vs_imr,
        losses=np.array(res["losses"]),
    )

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3)

    vmax = max(H_target.max(), H_pred.max(), H_imr.max())
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(H_imr, origin="lower", cmap="Reds", vmin=0, vmax=vmax)
    ax.set_title(f"MEASURED IMR90 (held-out)\n({len(val_imr_idx)} cells)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(H_target, origin="lower", cmap="Reds", vmin=0, vmax=vmax)
    ax.set_title(f"MEASURED K562 (held-out)\n({n_val} cells)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(H_pred, origin="lower", cmap="Reds", vmin=0, vmax=vmax)
    ax.set_title(f"PREDICTED K562 (guided)\nPearson w/ K562={pcc_kpred_vs_k:.4f}")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, 0])
    diff = H_pred - H_target
    vmax_r = float(np.abs(diff).max())
    im = ax.imshow(diff, origin="lower", cmap="seismic", vmin=-vmax_r, vmax=vmax_r)
    ax.set_title(f"K562 pred - K562 measured\n(residual)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, 1])
    diff = H_k - H_imr
    vmax_d = float(np.abs(diff).max())
    im = ax.imshow(diff, origin="lower", cmap="seismic", vmin=-vmax_d, vmax=vmax_d)
    ax.set_title(f"MEASURED K562 - MEASURED IMR90\n(cell-type-specific differences)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, 2])
    bins = np.linspace(min(Rg_k_meas.min(), Rg_pred.min(), Rg_imr_meas.min()),
                       max(Rg_k_meas.max(), Rg_pred.max(), Rg_imr_meas.max()), 40)
    ax.hist(Rg_k_meas, bins=bins, density=True, alpha=0.5, color="black",
            label=f"K562 measured (med={np.median(Rg_k_meas):.0f})")
    ax.hist(Rg_pred, bins=bins, density=True, alpha=0.5, color="C3",
            label=f"K562 guided (med={np.median(Rg_pred):.0f})")
    ax.hist(Rg_imr_meas, bins=bins, density=True, alpha=0.4, color="C0",
            label=f"IMR90 measured (med={np.median(Rg_imr_meas):.0f})")
    ax.set_xlabel("Rg (nm)"); ax.set_ylabel("density")
    ax.set_title("Rg distributions across cell types")
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[2, 0])
    seps = np.arange(1, N)
    ps_k = p_of_s((real.D[val_idx_k] < hard_thr).astype(np.float32))
    ps_pred = p_of_s(C_samp_hard)
    ps_imr = p_of_s((D_imr[val_imr_idx] < hard_thr).astype(np.float32))
    ax.loglog(seps, ps_k, "o-", ms=3, color="black", label="K562 measured")
    ax.loglog(seps, ps_pred, "^-", ms=3, color="C3", label="K562 guided")
    ax.loglog(seps, ps_imr, "s-", ms=3, color="C0", label="IMR90 measured")
    ax.set_xlabel("genomic separation s"); ax.set_ylabel("P(contact | s)")
    ax.set_title("P(s) scaling")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(res["losses"], lw=1, color="C3")
    ax.set_yscale("log")
    ax.set_xlabel("DDIM step"); ax.set_ylabel("guidance MSE")
    ax.set_title("guidance loss"); ax.grid(True, which="both", alpha=0.3)

    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    summary = (
        f"K562 generalization\n\n"
        f"Pipeline trained on: IMR90 chr21:28-30Mb\n"
        f"Applied unchanged to: K562 chr21:28-30Mb\n"
        f"K562 cells used: {real.D.shape[0]} (val={n_val})\n\n"
        f"Bulk Pearson:\n"
        f"  K562 guided vs K562 measured:   {pcc_kpred_vs_k:.4f}\n"
        f"  K562 guided vs IMR90 measured:  {pcc_kpred_vs_imr:.4f}\n"
        f"  delta (cell-type discrimination): {pcc_kpred_vs_k - pcc_kpred_vs_imr:+.4f}\n\n"
        f"Rg median (nm):\n"
        f"  K562 measured:  {np.median(Rg_k_meas):.0f}\n"
        f"  K562 guided:    {np.median(Rg_pred):.0f}\n"
        f"  IMR90 measured: {np.median(Rg_imr_meas):.0f}\n\n"
        f"Measured K562 vs IMR90 bulk Pearson: {pcc_k_vs_imr:.4f}\n"
        f"  (real cell-type differences are subtle\n"
        f"   at this region; both cell types share\n"
        f"   most TAD-level structure here)"
    )
    ax.text(0.0, 0.95, summary, fontsize=10, va="top", family="monospace")

    fig.suptitle("Cross-cell-type generalization: K562 with IMR90-trained pipeline")
    fig.tight_layout()
    out = out_dir / "27_k562_generalization.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
