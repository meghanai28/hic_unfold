"""Partial z attenuation sweep: model partial cohesin loss.

Step 26 found that full z=0 over-predicts the auxin ensemble by ~2.3x. The
natural hypothesis: 6 h of auxin degrades cohesin but doesn't eliminate it
completely, so z = alpha * z_hat with alpha < 1 should better match the
measured auxin ensemble than alpha = 0.

For alpha in [0, 1] we DDIM-sample with z = alpha * z_hat and compare the
predicted ensemble to:
    - measured HCT116 untreated  (alpha = 1 should match this)
    - measured HCT116 auxin      (some alpha* in (0, 1) should match this)

Run:
    python scripts/28_partial_z_attenuation.py
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


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diff_ckpt_path = ROOT / "checkpoints" / "step05_diffusion_real.pt"
    enc_ckpt_path = ROOT / "checkpoints" / "step05_encoder_N65.pt"
    fwd_path = ROOT / "checkpoints" / "step06_forward_params.npz"

    fwd = np.load(fwd_path)
    hard_thr = float(fwd["hard_threshold"])

    print("loading HCT116 untreated + auxin (Bintu)...")
    ds_u = load_bintu_csv(ROOT / "data" / "raw_bintu2018" / "HCT116_chr21-28-30Mb_untreated.csv")
    ds_a = load_bintu_csv(ROOT / "data" / "raw_bintu2018" / "HCT116_chr21-28-30Mb_6h_auxin.csv")
    real_u = preprocess_bintu(ds_u, min_valid_frac=0.85)
    real_a = preprocess_bintu(ds_a, min_valid_frac=0.85)
    N = real_u.D.shape[-1]

    H_u_meas = (real_u.D < hard_thr).mean(axis=0).astype(np.float32)
    H_a_meas = (real_a.D < hard_thr).mean(axis=0).astype(np.float32)
    iu = np.triu_indices(N, k=1)

    Rg_u_meas = np.array([radius_of_gyration(d) for d in real_u.D])
    Rg_a_meas = np.array([radius_of_gyration(d) for d in real_a.D])
    target_Rg_u = float(np.median(Rg_u_meas))
    target_Rg_a = float(np.median(Rg_a_meas))
    print(f"target Rg medians: untreated={target_Rg_u:.1f}, auxin={target_Rg_a:.1f}")

    # Encode untreated cells -> z_hat
    enc_ckpt = torch.load(enc_ckpt_path, map_location=device, weights_only=False)
    enc = LoopEncoder(N=N, d_c=int(enc_ckpt["d_c"]), d_pair=32, d_sep=16, d_h=96,
                      dilations=(1, 2, 4, 8, 1)).to(device)
    enc.load_state_dict(enc_ckpt["state_dict"]); enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    c_const = make_positional_c(N, int(enc_ckpt["d_c"]), device)

    mu_u = float(np.log1p(real_u.D).mean()); sigma_u = float(np.log1p(real_u.D).std())
    x_u = ((np.log1p(real_u.D) - mu_u) / max(sigma_u, 1e-8)).astype(np.float32)
    print("encoding untreated HCT116 cells -> z_hat...")
    z_hat_u = np.empty((x_u.shape[0], N, N), dtype=np.float32)
    bs = 64
    with torch.no_grad():
        for s in range(0, x_u.shape[0], bs):
            e = min(s + bs, x_u.shape[0])
            x_b = torch.from_numpy(x_u[s:e])[:, None].to(device)
            c_b = c_const.expand(e - s, -1, -1)
            z_hat_u[s:e] = torch.sigmoid(enc(x_b, c_b))[:, 0].cpu().numpy()

    diff_ckpt = torch.load(diff_ckpt_path, map_location=device, weights_only=False)
    mu_train = float(diff_ckpt["mu"]); sigma_train = float(diff_ckpt["sigma"])
    net = Denoiser(N=N, d_c=int(diff_ckpt["d_c"]), d_pair=32, d_sep=16, d_h=96,
                   d_t=128, dilations=(1, 2, 4, 8, 1)).to(device)
    net.load_state_dict(diff_ckpt["state_dict"]); net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    alpha_bars = make_cosine_schedule(T=int(diff_ckpt["T"]), device=device)

    M_samp = 128
    rng = np.random.default_rng(2027)
    u_idx = rng.choice(real_u.D.shape[0], size=M_samp, replace=False)
    z_hat_base = torch.from_numpy(z_hat_u[u_idx])[:, None].to(device)
    c_batch = c_const.expand(M_samp, -1, -1)

    alphas = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    print(f"\nsweep over alpha (z attenuation): {alphas}")

    results: list[dict] = []
    print(f"\n{'alpha':>6s} {'Rg med':>8s} {'pcc-u':>8s} {'pcc-a':>8s} "
          f"{'MSE-u':>10s} {'MSE-a':>10s}")
    for alpha in alphas:
        z_alpha = z_hat_base * float(alpha)
        t0 = time.time()
        with torch.no_grad():
            x = ddim_sample(net, z_alpha, c_batch, alpha_bars, n_steps=100)
        D = np.expm1(x.squeeze(1).cpu().numpy() * sigma_train + mu_train)
        D = np.maximum(D, 0)
        for k in range(D.shape[0]):
            D[k] = 0.5 * (D[k] + D[k].T); np.fill_diagonal(D[k], 0)
        H_pred = (D < hard_thr).astype(np.float32).mean(axis=0)
        pcc_u = float(np.corrcoef(H_pred[iu], H_u_meas[iu])[0, 1])
        pcc_a = float(np.corrcoef(H_pred[iu], H_a_meas[iu])[0, 1])
        mse_u = float(((H_pred - H_u_meas)[iu] ** 2).mean())
        mse_a = float(((H_pred - H_a_meas)[iu] ** 2).mean())
        Rg_pred = float(np.median([radius_of_gyration(d) for d in D]))
        results.append({
            "alpha": alpha, "Rg_pred": Rg_pred,
            "pcc_u": pcc_u, "pcc_a": pcc_a,
            "mse_u": mse_u, "mse_a": mse_a,
            "wall_s": time.time() - t0,
        })
        print(f"{alpha:>6.2f} {Rg_pred:>8.1f} {pcc_u:>8.4f} {pcc_a:>8.4f} "
              f"{mse_u:>10.5f} {mse_a:>10.5f}")

    arr = np.array([(r["alpha"], r["Rg_pred"], r["pcc_u"], r["pcc_a"],
                     r["mse_u"], r["mse_a"]) for r in results])

    # Find alpha that best matches auxin Rg and best matches auxin bulk
    rg_diff_a = np.abs(arr[:, 1] - target_Rg_a)
    rg_diff_u = np.abs(arr[:, 1] - target_Rg_u)
    best_a_rg = int(np.argmin(rg_diff_a))
    best_u_rg = int(np.argmin(rg_diff_u))
    best_a_pcc = int(np.argmax(arr[:, 3]))
    best_u_pcc = int(np.argmax(arr[:, 2]))
    print(f"\nBest match by Rg:  untreated <- alpha={arr[best_u_rg, 0]:.2f} "
          f"(Rg {arr[best_u_rg, 1]:.0f}, target {target_Rg_u:.0f})")
    print(f"                   auxin     <- alpha={arr[best_a_rg, 0]:.2f} "
          f"(Rg {arr[best_a_rg, 1]:.0f}, target {target_Rg_a:.0f})")
    print(f"Best match by bulk Pearson: untreated <- alpha={arr[best_u_pcc, 0]:.2f}")
    print(f"                            auxin     <- alpha={arr[best_a_pcc, 0]:.2f}")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROOT / "checkpoints" / "step17_partial_z_sweep.npz",
        alphas=arr[:, 0], Rg_pred=arr[:, 1],
        pcc_u=arr[:, 2], pcc_a=arr[:, 3],
        mse_u=arr[:, 4], mse_a=arr[:, 5],
        target_Rg_u=target_Rg_u, target_Rg_a=target_Rg_a,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(arr[:, 0], arr[:, 1], "o-", color="C3", lw=2, ms=8)
    ax.axhline(target_Rg_u, color="C0", ls="--", lw=1.5, label=f"target untreated ({target_Rg_u:.0f})")
    ax.axhline(target_Rg_a, color="C2", ls="--", lw=1.5, label=f"target auxin ({target_Rg_a:.0f})")
    ax.set_xlabel("alpha (z scaling)"); ax.set_ylabel("predicted Rg median (nm)")
    ax.set_title("Predicted Rg vs alpha (full z = active cohesin, alpha = 0 = no cohesin)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(arr[:, 0], arr[:, 2], "o-", color="C0", lw=2, ms=8, label="vs measured-untreated")
    ax.plot(arr[:, 0], arr[:, 3], "s-", color="C2", lw=2, ms=8, label="vs measured-auxin")
    ax.set_xlabel("alpha (z scaling)"); ax.set_ylabel("bulk Pearson")
    ax.set_title("Bulk fit vs alpha")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(arr[:, 0], arr[:, 4], "o-", color="C0", lw=2, ms=8, label="vs measured-untreated")
    ax.plot(arr[:, 0], arr[:, 5], "s-", color="C2", lw=2, ms=8, label="vs measured-auxin")
    ax.set_xlabel("alpha (z scaling)"); ax.set_ylabel("bulk MSE")
    ax.set_title("Bulk MSE vs alpha (lower = closer match)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")
    summary = (
        "Partial-cohesin-loss sweep\n\n"
        f"Target Rg medians (nm):\n"
        f"  untreated (full cohesin):    {target_Rg_u:.0f}\n"
        f"  auxin (cohesin degraded):    {target_Rg_a:.0f}\n"
        f"  delta from cohesin loss:     +{target_Rg_a - target_Rg_u:.0f}\n\n"
        "Sweep table:\n"
        f"{'alpha':>6s} {'Rg pred':>8s} {'pcc-u':>8s} {'pcc-a':>8s}\n"
        + "-" * 35 + "\n"
        + "\n".join([f"{r['alpha']:>6.2f} {r['Rg_pred']:>8.0f} "
                     f"{r['pcc_u']:>8.4f} {r['pcc_a']:>8.4f}"
                     for r in results])
        + "\n\n"
        f"Best alpha for matching auxin Rg: {arr[best_a_rg, 0]:.2f}\n"
        f"Best alpha for matching auxin bulk Pearson: {arr[best_a_pcc, 0]:.2f}\n\n"
        "Interpretation:\n"
        "If alpha* > 0, the auxin condition retains some\n"
        "loop activity (consistent with partial cohesin\n"
        "function after 6 h auxin)."
    )
    ax.text(0.0, 0.95, summary, fontsize=9.5, va="top", family="monospace")

    fig.suptitle("Partial z attenuation: in-silico graded cohesin loss")
    fig.tight_layout()
    out = out_dir / "28_partial_z_attenuation.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
