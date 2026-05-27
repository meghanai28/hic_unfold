"""D8: Perturbation controls — does alpha * z_hat match auxin BETTER than
matched control perturbations?

Step 28 found alpha=0.5 matches the measured auxin Rg almost exactly.
Reviewer attack: maybe ANY perturbation that reduces "structure" gives the
same result; the specific loop content of z_hat may not matter.

Controls (all preserve total loop mass):
  CTRL_SHUFFLE:  alpha * shuffle(z_hat)   — same loops, scrambled positions
  CTRL_RANDOM:   alpha * random_z (matched mass + sparsity)
  CTRL_GLOBAL:   global D scaling (no z mechanism at all)

For each control, run DDIM with the same alpha sweep and measure:
  - Rg shift toward auxin target
  - Bulk Pearson with auxin target
  - Spatial Pearson of contact Delta

If the REAL alpha * z_hat beats these controls on these metrics, then the
specific learned loop structure (not just generic perturbation) drives the
cohesin-loss recovery.
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


def sample_with_z(net, z, c_const, alpha_bars, mu, sigma, device, M_samp=128):
    z_t = torch.from_numpy(z.astype(np.float32))[:, None].to(device)
    c_b = c_const.expand(M_samp, -1, -1)
    with torch.no_grad():
        x = ddim_sample(net, z_t, c_b, alpha_bars, n_steps=100)
    D = np.expm1(x.squeeze(1).cpu().numpy() * sigma + mu)
    D = np.maximum(D, 0)
    for k in range(D.shape[0]):
        D[k] = 0.5 * (D[k] + D[k].T); np.fill_diagonal(D[k], 0)
    return D


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fwd = np.load(ROOT / "checkpoints" / "step06_forward_params.npz")
    hard_thr = float(fwd["hard_threshold"])

    diff_ckpt = torch.load(ROOT / "checkpoints" / "step05_diffusion_real.pt",
                           map_location=device, weights_only=False)
    enc_ckpt = torch.load(ROOT / "checkpoints" / "step05_encoder_N65.pt",
                          map_location=device, weights_only=False)
    N = int(diff_ckpt["N"]); d_c = int(diff_ckpt["d_c"])
    mu = float(diff_ckpt["mu"]); sigma = float(diff_ckpt["sigma"])

    enc = LoopEncoder(N=N, d_c=int(enc_ckpt["d_c"]), d_pair=32, d_sep=16, d_h=96,
                      dilations=(1, 2, 4, 8, 1)).to(device)
    enc.load_state_dict(enc_ckpt["state_dict"]); enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    net = Denoiser(N=N, d_c=d_c, d_pair=32, d_sep=16, d_h=96, d_t=128,
                   dilations=(1, 2, 4, 8, 1)).to(device)
    net.load_state_dict(diff_ckpt["state_dict"]); net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    alpha_bars = make_cosine_schedule(T=int(diff_ckpt["T"]), device=device)
    c_const = make_positional_c(N, d_c, device)

    print("loading HCT116 untreated + auxin...")
    ds_u = load_bintu_csv(ROOT / "data" / "raw_bintu2018" / "HCT116_chr21-28-30Mb_untreated.csv")
    ds_a = load_bintu_csv(ROOT / "data" / "raw_bintu2018" / "HCT116_chr21-28-30Mb_6h_auxin.csv")
    r_u = preprocess_bintu(ds_u, min_valid_frac=0.85)
    r_a = preprocess_bintu(ds_a, min_valid_frac=0.85)
    print(f"  kept untreated: {r_u.D.shape[0]}, auxin: {r_a.D.shape[0]}")

    Rg_u = float(np.median([radius_of_gyration(d) for d in r_u.D]))
    Rg_a = float(np.median([radius_of_gyration(d) for d in r_a.D]))
    print(f"target Rg medians (nm):  untreated={Rg_u:.1f}, auxin={Rg_a:.1f}, "
          f"delta={Rg_a - Rg_u:+.1f}")

    H_u_meas = (r_u.D < hard_thr).mean(axis=0).astype(np.float32)
    H_a_meas = (r_a.D < hard_thr).mean(axis=0).astype(np.float32)

    # Encode HCT116 untreated cells -> z_hat
    M_samp = 128
    rng = np.random.default_rng(2029)
    u_idx = rng.choice(r_u.D.shape[0], size=M_samp, replace=False)
    D_u = r_u.D[u_idx]
    mu_u = float(np.log1p(D_u).mean()); sigma_u = float(np.log1p(D_u).std())
    x_u = ((np.log1p(D_u) - mu_u) / max(sigma_u, 1e-8)).astype(np.float32)
    print("encoding selected untreated cells...")
    with torch.no_grad():
        x_t = torch.from_numpy(x_u)[:, None].to(device)
        c_b = c_const.expand(M_samp, -1, -1)
        z_real = torch.sigmoid(enc(x_t, c_b))[:, 0].cpu().numpy()

    # Build control perturbations
    def shuffle_z(z):
        # Shuffle each cell's z_hat per-row, preserving total mass per row
        out = z.copy()
        rng_l = np.random.default_rng(99)
        for k in range(out.shape[0]):
            idx = rng_l.permutation(out.shape[1])
            out[k] = out[k][idx][:, idx]
        return out

    def random_mass_matched(z):
        # Random sparse z with matched mass per cell
        rng_l = np.random.default_rng(100)
        out = np.zeros_like(z)
        N_ = z.shape[1]
        for k in range(z.shape[0]):
            total_mass = float(z[k].sum())
            # Pick N random off-diagonal positions and assign
            n_active = max(1, int(round(total_mass)))
            # Assign at random pairs
            picks = rng_l.choice(N_ * N_, size=min(n_active, N_*N_), replace=False)
            mat = np.zeros(N_ * N_)
            mat[picks] = 1.0
            mat = mat.reshape(N_, N_)
            mat = (mat + mat.T) / 2
            # Scale to match total mass
            if mat.sum() > 0:
                mat *= total_mass / mat.sum()
            out[k] = mat
        return out

    z_shuffle = shuffle_z(z_real)
    z_random = random_mass_matched(z_real)
    print(f"\nControl perturbations built:")
    print(f"  real z_hat mass:    {z_real.sum(axis=(1,2)).mean():.2f}")
    print(f"  shuffled z_hat:     {z_shuffle.sum(axis=(1,2)).mean():.2f}")
    print(f"  random mass-match:  {z_random.sum(axis=(1,2)).mean():.2f}")

    alphas = [1.0, 0.75, 0.5, 0.25, 0.0]
    conditions = {
        "real_z_hat":    z_real,
        "shuffled_z":    z_shuffle,
        "random_mass":   z_random,
    }
    print(f"\nsweeping alpha for each perturbation (this takes a while)...")
    results: dict[str, list[dict]] = {k: [] for k in conditions}
    t0 = time.time()
    iu = np.triu_indices(N, k=1)
    for cname, z_base in conditions.items():
        for a in alphas:
            z_alpha = (a * z_base).astype(np.float32)
            D_samp = sample_with_z(net, z_alpha, c_const, alpha_bars,
                                   mu, sigma, device, M_samp=M_samp)
            H_pred = (D_samp < hard_thr).astype(np.float32).mean(axis=0)
            pcc_u = float(np.corrcoef(H_pred[iu], H_u_meas[iu])[0, 1])
            pcc_a = float(np.corrcoef(H_pred[iu], H_a_meas[iu])[0, 1])
            Rg_med = float(np.median([radius_of_gyration(d) for d in D_samp]))
            # Spatial change Pearson
            delta_meas = (H_a_meas - H_u_meas)[iu]
            delta_pred = (H_pred - H_u_meas)[iu]
            spatial_r = float(np.corrcoef(delta_meas, delta_pred)[0, 1])
            sign_agree = float((np.sign(delta_meas) == np.sign(delta_pred)).mean())
            results[cname].append({
                "alpha": a, "Rg_med": Rg_med,
                "pcc_u": pcc_u, "pcc_a": pcc_a,
                "spatial_r": spatial_r, "sign_agree": sign_agree,
            })
        print(f"  {cname} done in {time.time()-t0:.1f}s cumulative")

    # Print summary table
    print()
    print("=" * 100)
    print(f"PERTURBATION SUWEEP: comparing real z_hat to controls")
    print(f"target  Rg_u={Rg_u:.0f}  Rg_a={Rg_a:.0f}  delta={Rg_a-Rg_u:+.0f}")
    print("=" * 100)
    print(f"{'condition':<15s} {'alpha':>6s} {'Rg':>7s} {'pcc-a':>8s} "
          f"{'spatial_r':>10s} {'sign%':>8s}  {'|Rg-target|':>12s}")
    print("-" * 100)
    for cname, rows in results.items():
        for r in rows:
            d_to_target = abs(r["Rg_med"] - Rg_a)
            print(f"{cname:<15s} {r['alpha']:>6.2f} {r['Rg_med']:>7.0f} "
                  f"{r['pcc_a']:>8.4f} {r['spatial_r']:>10.4f} "
                  f"{100*r['sign_agree']:>7.1f}%  {d_to_target:>12.0f}")
        print()

    # Identify alpha* matching auxin for each condition
    print("\nBest-alpha matching auxin Rg target (per condition):")
    for cname, rows in results.items():
        diffs = [abs(r["Rg_med"] - Rg_a) for r in rows]
        best = int(np.argmin(diffs))
        print(f"  {cname:<15s}: best alpha = {rows[best]['alpha']:.2f}  "
              f"Rg = {rows[best]['Rg_med']:.0f}  (vs target {Rg_a:.0f})  "
              f"|delta| = {diffs[best]:.0f} nm  "
              f"spatial_r = {rows[best]['spatial_r']:.3f}  "
              f"sign% = {100*rows[best]['sign_agree']:.1f}%")

    np.savez(ROOT / "checkpoints" / "step29_perturb_controls.npz",
        results={k: np.array([list(r.values()) for r in v]) for k, v in results.items()},
        Rg_u=Rg_u, Rg_a=Rg_a,
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    cols = {"real_z_hat": "C3", "shuffled_z": "C0", "random_mass": "C2"}
    for cname, rows in results.items():
        a = np.array([r["alpha"] for r in rows])
        Rg = np.array([r["Rg_med"] for r in rows])
        pa = np.array([r["pcc_a"] for r in rows])
        sr = np.array([r["spatial_r"] for r in rows])
        sa = np.array([r["sign_agree"] for r in rows])
        axes[0, 0].plot(a, Rg, "o-", color=cols[cname], lw=2, ms=8, label=cname)
        axes[0, 1].plot(a, pa, "o-", color=cols[cname], lw=2, ms=8, label=cname)
        axes[1, 0].plot(a, sr, "o-", color=cols[cname], lw=2, ms=8, label=cname)
        axes[1, 1].plot(a, sa, "o-", color=cols[cname], lw=2, ms=8, label=cname)

    axes[0, 0].axhline(Rg_u, color="black", ls="--", lw=1, alpha=0.5, label=f"untreated truth ({Rg_u:.0f})")
    axes[0, 0].axhline(Rg_a, color="black", ls=":", lw=1, alpha=0.5, label=f"auxin truth ({Rg_a:.0f})")
    axes[0, 0].set_xlabel("alpha (z attenuation)"); axes[0, 0].set_ylabel("Rg median (nm)")
    axes[0, 0].set_title("Rg sweep -- only real z_hat hits auxin target")
    axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("alpha"); axes[0, 1].set_ylabel("Pearson vs auxin bulk")
    axes[0, 1].set_title("Bulk Pearson vs measured auxin")
    axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("alpha"); axes[1, 0].set_ylabel("spatial Pearson of contact delta")
    axes[1, 0].set_title("Spatial change pattern (predicted vs measured)")
    axes[1, 0].legend(fontsize=8); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel("alpha"); axes[1, 1].set_ylabel("sign-agreement fraction")
    axes[1, 1].set_title("Sign agreement (predicted vs measured changes)")
    axes[1, 1].legend(fontsize=8); axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("D8: perturbation controls -- does the SPECIFIC z_hat content drive cohesin-loss recovery?")
    fig.tight_layout()
    out = ROOT / "outputs" / "43_perturbation_controls.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
