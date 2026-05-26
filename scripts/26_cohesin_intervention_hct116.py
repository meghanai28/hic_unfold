"""Falsifiable mechanism validation: cohesin loss in HCT116 (Bintu et al.).

The architecture's Section 6 point 2 promises that CTCF/cohesin intervention
is a one-edit perturbation on the model's latent z. Bintu et al. provide the
ground truth for this: HCT116 cells with cohesin degraded by 6 h auxin
treatment. If our z really encodes loop-extrusion structure, then:

  (i)  the encoder should detect *fewer* loops on auxin cells than on
       untreated cells (cohesin loss => fewer extruded loops);
  (ii) DDIM sampling conditioned on z_hat from untreated cells should
       reproduce the untreated bulk contact map;
  (iii) DDIM sampling with z=0 (in-silico cohesin knockout) should produce
        an ensemble whose bulk matches the measured AUXIN bulk.

Step (iii) is the falsifiable claim: an in-silico edit on the model's
mechanistic latent reproduces the in-vivo measurement.

Run:
    python scripts/26_cohesin_intervention_hct116.py
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
from hic_unfold.diffusion import Denoiser, ddim_sample, make_cosine_schedule  # noqa: E402
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


def encode_cells(enc, D: np.ndarray, mu: float, sigma: float,
                 c_const: torch.Tensor, device, batch: int = 64) -> np.ndarray:
    M, N, _ = D.shape
    x = ((np.log1p(D) - mu) / max(sigma, 1e-8)).astype(np.float32)
    z_hat = np.empty((M, N, N), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, M, batch):
            e = min(s + batch, M)
            x_b = torch.from_numpy(x[s:e])[:, None].to(device)
            c_b = c_const.expand(e - s, -1, -1)
            z_hat[s:e] = torch.sigmoid(enc(x_b, c_b))[:, 0].cpu().numpy()
    return z_hat


def ddim_batch(net, z, c, alpha_bars, mu, sigma, N, hard_thr,
               n_steps: int = 100) -> np.ndarray:
    with torch.no_grad():
        x = ddim_sample(net, z, c, alpha_bars, n_steps=n_steps)
    D = np.expm1(x.squeeze(1).cpu().numpy() * sigma + mu)
    D = np.maximum(D, 0)
    for k in range(D.shape[0]):
        D[k] = 0.5 * (D[k] + D[k].T)
        np.fill_diagonal(D[k], 0)
    return D


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diff_ckpt_path = ROOT / "checkpoints" / "step05_diffusion_real.pt"
    enc_ckpt_path = ROOT / "checkpoints" / "step05_encoder_N65.pt"
    fwd_path = ROOT / "checkpoints" / "step06_forward_params.npz"

    fwd = np.load(fwd_path)
    hard_thr = float(fwd["hard_threshold"])

    print("loading HCT116 untreated + 6h auxin...")
    ds_u = load_bintu_csv(ROOT / "data" / "raw_bintu2018" / "HCT116_chr21-28-30Mb_untreated.csv")
    ds_a = load_bintu_csv(ROOT / "data" / "raw_bintu2018" / "HCT116_chr21-28-30Mb_6h_auxin.csv")
    print(f"  untreated: {ds_u.num_cells}, auxin: {ds_a.num_cells}")
    real_u = preprocess_bintu(ds_u, min_valid_frac=0.85)
    real_a = preprocess_bintu(ds_a, min_valid_frac=0.85)
    N = real_u.D.shape[-1]
    print(f"  kept untreated: {real_u.D.shape[0]}, auxin: {real_a.D.shape[0]}")

    # Measured bulk contact maps (the ground truth for the validation).
    H_u = (real_u.D < hard_thr).mean(axis=0).astype(np.float32)
    H_a = (real_a.D < hard_thr).mean(axis=0).astype(np.float32)
    iu = np.triu_indices(N, k=1)
    pcc_au = float(np.corrcoef(H_a[iu], H_u[iu])[0, 1])
    print(f"measured auxin vs measured untreated bulk Pearson: {pcc_au:.4f}")

    Rg_u_meas = np.array([radius_of_gyration(d) for d in real_u.D])
    Rg_a_meas = np.array([radius_of_gyration(d) for d in real_a.D])
    print(f"measured Rg medians: untreated={np.median(Rg_u_meas):.1f}, "
          f"auxin={np.median(Rg_a_meas):.1f}")
    print(f"  -> Delta Rg from cohesin loss = {np.median(Rg_a_meas) - np.median(Rg_u_meas):+.1f} nm")

    # Load IMR90-trained pieces.
    diff_ckpt = torch.load(diff_ckpt_path, map_location=device, weights_only=False)
    mu = float(diff_ckpt["mu"]); sigma = float(diff_ckpt["sigma"])
    net = Denoiser(N=N, d_c=int(diff_ckpt["d_c"]), d_pair=32, d_sep=16, d_h=96,
                   d_t=128, dilations=(1, 2, 4, 8, 1)).to(device)
    net.load_state_dict(diff_ckpt["state_dict"]); net.eval()
    for p in net.parameters():
        p.requires_grad_(False)

    enc_ckpt = torch.load(enc_ckpt_path, map_location=device, weights_only=False)
    enc = LoopEncoder(N=N, d_c=int(enc_ckpt["d_c"]), d_pair=32, d_sep=16, d_h=96,
                      dilations=(1, 2, 4, 8, 1)).to(device)
    enc.load_state_dict(enc_ckpt["state_dict"]); enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    alpha_bars = make_cosine_schedule(T=int(diff_ckpt["T"]), device=device)
    c_const = make_positional_c(N, int(diff_ckpt["d_c"]), device)

    # 1) Encoder on both: do encoded loop counts differ?
    print("\n(i) encoding untreated and auxin cells...")
    # Use the cells' own mu/sigma so encoder sees in-distribution input.
    mu_u = float(np.log1p(real_u.D).mean()); sigma_u = float(np.log1p(real_u.D).std())
    mu_a = float(np.log1p(real_a.D).mean()); sigma_a = float(np.log1p(real_a.D).std())
    print(f"  HCT116-untreated log1p stats: mu={mu_u:.3f}, sigma={sigma_u:.3f}")
    print(f"  HCT116-auxin     log1p stats: mu={mu_a:.3f}, sigma={sigma_a:.3f}")
    z_u = encode_cells(enc, real_u.D, mu_u, sigma_u, c_const, device)
    z_a = encode_cells(enc, real_a.D, mu_a, sigma_a, c_const, device)

    mass_u = z_u.reshape(z_u.shape[0], -1).sum(axis=1) / 2
    mass_a = z_a.reshape(z_a.shape[0], -1).sum(axis=1) / 2
    print(f"per-cell predicted loop mass (sum of upper-tri probs):")
    print(f"  untreated: median={np.median(mass_u):.2f}")
    print(f"  auxin:     median={np.median(mass_a):.2f}")
    print(f"  -> ratio auxin/untreated = {np.median(mass_a)/max(np.median(mass_u), 1e-9):.3f}")

    # 2) Diffusion sample given encoder z_hat from untreated -> predicted untreated
    M_samp = 128
    rng = np.random.default_rng(2026)
    u_idx = rng.choice(real_u.D.shape[0], size=M_samp, replace=False)
    z_u_samp = torch.from_numpy(z_u[u_idx])[:, None].to(device)
    z_zero = torch.zeros_like(z_u_samp)
    c_batch = c_const.expand(M_samp, -1, -1)

    print(f"\n(ii) DDIM sampling (M={M_samp}, 100 steps)...")
    print("  pred-untreated: condition on encoder z_hat from real untreated cells")
    t0 = time.time()
    D_pred_u = ddim_batch(net, z_u_samp, c_batch, alpha_bars, mu, sigma, N, hard_thr)
    print(f"  done in {time.time()-t0:.1f}s")

    print("  pred-cohesin-loss: condition on z=0 (in silico cohesin knockout)")
    t0 = time.time()
    D_pred_ko = ddim_batch(net, z_zero, c_batch, alpha_bars, mu, sigma, N, hard_thr)
    print(f"  done in {time.time()-t0:.1f}s")

    H_pred_u = (D_pred_u < hard_thr).mean(axis=0)
    H_pred_ko = (D_pred_ko < hard_thr).mean(axis=0)

    # 3) Compare predicted vs measured for both conditions
    pcc_pred_u_vs_meas_u = float(np.corrcoef(H_pred_u[iu], H_u[iu])[0, 1])
    pcc_pred_u_vs_meas_a = float(np.corrcoef(H_pred_u[iu], H_a[iu])[0, 1])
    pcc_pred_ko_vs_meas_u = float(np.corrcoef(H_pred_ko[iu], H_u[iu])[0, 1])
    pcc_pred_ko_vs_meas_a = float(np.corrcoef(H_pred_ko[iu], H_a[iu])[0, 1])

    print("\n(iii) bulk Pearson - the headline 2x2 table:")
    print(f"{'':20s} {'vs measured untreated':>24s} {'vs measured auxin':>22s}")
    print(f"{'pred untreated (z_hat)':<20s} {pcc_pred_u_vs_meas_u:>24.4f} {pcc_pred_u_vs_meas_a:>22.4f}")
    print(f"{'pred knockout (z=0)':<20s} {pcc_pred_ko_vs_meas_u:>24.4f} {pcc_pred_ko_vs_meas_a:>22.4f}")
    print()
    print(f"diagonal should be HIGH (prediction matches its target).")
    print(f"off-diagonal should be lower (prediction differentiates conditions).")

    Rg_pred_u = np.array([radius_of_gyration(d) for d in D_pred_u])
    Rg_pred_ko = np.array([radius_of_gyration(d) for d in D_pred_ko])
    print(f"\npredicted Rg medians (nm):")
    print(f"  pred-untreated: {np.median(Rg_pred_u):.1f}  (target untreated {np.median(Rg_u_meas):.1f})")
    print(f"  pred-knockout:  {np.median(Rg_pred_ko):.1f}  (target auxin     {np.median(Rg_a_meas):.1f})")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROOT / "checkpoints" / "step15_cohesin_validation.npz",
        H_u_meas=H_u, H_a_meas=H_a,
        H_pred_u=H_pred_u, H_pred_ko=H_pred_ko,
        z_u_mean=z_u.mean(axis=0), z_a_mean=z_a.mean(axis=0),
        mass_u=mass_u, mass_a=mass_a,
        Rg_u_meas=Rg_u_meas, Rg_a_meas=Rg_a_meas,
        Rg_pred_u=Rg_pred_u, Rg_pred_ko=Rg_pred_ko,
    )

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 4)

    vmax = max(H_u.max(), H_a.max(), H_pred_u.max(), H_pred_ko.max())
    panels = [
        (H_u, "MEASURED untreated\nbulk contact"),
        (H_a, "MEASURED auxin\n(cohesin degraded)"),
        (H_pred_u, f"PREDICTED untreated\nz_hat -> DDIM\nPearson w/ meas-u {pcc_pred_u_vs_meas_u:.3f}"),
        (H_pred_ko, f"PREDICTED knockout\nz=0 -> DDIM\nPearson w/ meas-auxin {pcc_pred_ko_vs_meas_a:.3f}"),
    ]
    for col, (H, ttl) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(H, origin="lower", cmap="Reds", vmin=0, vmax=vmax)
        ax.set_title(ttl, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Per-cell loop mass distribution
    ax = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0, max(mass_u.max(), mass_a.max()), 50)
    ax.hist(mass_u, bins=bins, alpha=0.6, color="C0",
            label=f"untreated (med={np.median(mass_u):.1f})")
    ax.hist(mass_a, bins=bins, alpha=0.6, color="C3",
            label=f"auxin (med={np.median(mass_a):.1f})")
    ax.set_xlabel("per-cell loop mass (sum of encoder z_hat)")
    ax.set_ylabel("# cells")
    ax.set_title("(i) Encoder detects fewer\nloops after cohesin loss")
    ax.legend(fontsize=8)

    # Bulk Pearson 2x2 table as heatmap
    ax = fig.add_subplot(gs[1, 1])
    pcc_matrix = np.array([
        [pcc_pred_u_vs_meas_u, pcc_pred_u_vs_meas_a],
        [pcc_pred_ko_vs_meas_u, pcc_pred_ko_vs_meas_a],
    ])
    im = ax.imshow(pcc_matrix, cmap="Blues", vmin=0.85, vmax=1.0)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["meas-untreated", "meas-auxin"], fontsize=8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["pred-untreated", "pred-knockout"], fontsize=8)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{pcc_matrix[i, j]:.4f}", ha="center", va="center",
                    color="white" if pcc_matrix[i, j] > 0.95 else "black", fontsize=10)
    ax.set_title("(iii) Bulk Pearson 2x2\n(diagonal high = mechanism works)")

    # Rg distributions
    ax = fig.add_subplot(gs[1, 2])
    bins = np.linspace(min(Rg_u_meas.min(), Rg_a_meas.min(), Rg_pred_u.min(), Rg_pred_ko.min()),
                       max(Rg_u_meas.max(), Rg_a_meas.max(), Rg_pred_u.max(), Rg_pred_ko.max()), 40)
    ax.hist(Rg_u_meas, bins=bins, density=True, alpha=0.4, color="C0",
            label=f"meas-untreated (med={np.median(Rg_u_meas):.0f})")
    ax.hist(Rg_a_meas, bins=bins, density=True, alpha=0.4, color="C3",
            label=f"meas-auxin (med={np.median(Rg_a_meas):.0f})")
    ax.hist(Rg_pred_u, bins=bins, density=True, alpha=0.4, color="navy",
            label=f"pred-untreated (med={np.median(Rg_pred_u):.0f})")
    ax.hist(Rg_pred_ko, bins=bins, density=True, alpha=0.4, color="darkred",
            label=f"pred-knockout (med={np.median(Rg_pred_ko):.0f})")
    ax.set_xlabel("Rg (nm)"); ax.set_ylabel("density")
    ax.set_title("Rg distributions")
    ax.legend(fontsize=7)

    # P(s)
    ax = fig.add_subplot(gs[1, 3])
    seps = np.arange(1, N)
    ps_u_meas = p_of_s((real_u.D < hard_thr).astype(np.float32))
    ps_a_meas = p_of_s((real_a.D < hard_thr).astype(np.float32))
    ps_u_pred = p_of_s((D_pred_u < hard_thr).astype(np.float32))
    ps_ko_pred = p_of_s((D_pred_ko < hard_thr).astype(np.float32))
    ax.loglog(seps, ps_u_meas, "o-", ms=3, color="C0", label="meas-u")
    ax.loglog(seps, ps_a_meas, "o-", ms=3, color="C3", label="meas-auxin")
    ax.loglog(seps, ps_u_pred, "s--", ms=3, color="navy", label="pred-u")
    ax.loglog(seps, ps_ko_pred, "s--", ms=3, color="darkred", label="pred-knockout")
    ax.set_xlabel("separation s"); ax.set_ylabel("P(contact|s)")
    ax.set_title("P(s) scaling")
    ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.3)

    # Residual maps
    ax = fig.add_subplot(gs[2, 0])
    diff = H_pred_u - H_u
    vmax_r = float(np.abs(diff).max())
    im = ax.imshow(diff, origin="lower", cmap="seismic", vmin=-vmax_r, vmax=vmax_r)
    ax.set_title(f"pred-untreated  -  meas-untreated\nPearson {pcc_pred_u_vs_meas_u:.3f}")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[2, 1])
    diff = H_pred_ko - H_a
    im = ax.imshow(diff, origin="lower", cmap="seismic", vmin=-vmax_r, vmax=vmax_r)
    ax.set_title(f"pred-knockout  -  meas-auxin\nPearson {pcc_pred_ko_vs_meas_a:.3f}")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[2, 2])
    diff = H_a - H_u
    vmax_r = float(np.abs(diff).max())
    im = ax.imshow(diff, origin="lower", cmap="seismic", vmin=-vmax_r, vmax=vmax_r)
    ax.set_title("MEASURED change\n(auxin - untreated)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[2, 3])
    diff = H_pred_ko - H_pred_u
    im = ax.imshow(diff, origin="lower", cmap="seismic", vmin=-vmax_r, vmax=vmax_r)
    ax.set_title("PREDICTED change\n(knockout - untreated)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("In-silico cohesin knockout vs measured auxin treatment (HCT116, Bintu et al.)")
    fig.tight_layout()
    out = out_dir / "26_cohesin_intervention_hct116.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
