"""Robustness sweep: how sensitive is guided sampling to eta, M, tau?

Same setup as step 8 (target = chr21:28-30Mb val pseudo-bulk; step-5 prior).
For each sweep, vary one parameter while holding the others at the step-8
defaults (eta=30000, M=128, tau=80, n_steps=200). Report:
    - bulk Pearson against target
    - bulk MSE
    - Rg median (vs target 420 nm)
    - guidance loss decrease (initial / final)
    - wall-clock time

Run:
    python scripts/24_sensitivity_sweep.py
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


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    region = "IMR90_chr21-28-30Mb"
    real_path = ROOT / "data" / "real" / f"{region}_preprocessed.npz"
    diff_ckpt_path = ROOT / "checkpoints" / "step05_diffusion_real.pt"
    fwd_path = ROOT / "checkpoints" / "step06_forward_params.npz"

    fwd = np.load(fwd_path)
    d0 = float(fwd["d0"])
    hard_thr = float(fwd["hard_threshold"])

    f = np.load(real_path)
    D_real = f["D"]; z_hat_all = f["z_hat"]
    N = int(f["N"]); mu = float(f["mu"]); sigma = float(f["sigma"])

    diff_ckpt = torch.load(diff_ckpt_path, map_location=device, weights_only=False)
    val_idx = np.array(diff_ckpt["val_idx"])
    train_idx = np.setdiff1d(np.arange(D_real.shape[0]), val_idx)
    H_target = (D_real[val_idx] < hard_thr).mean(axis=0).astype(np.float32)
    H_obs = torch.tensor(H_target, device=device)

    Rg_target_median = float(np.median([radius_of_gyration(d) for d in D_real[val_idx]]))

    net = Denoiser(N=N, d_c=int(diff_ckpt["d_c"]), d_pair=32, d_sep=16, d_h=96,
                   d_t=128, dilations=(1, 2, 4, 8, 1)).to(device)
    net.load_state_dict(diff_ckpt["state_dict"]); net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    alpha_bars = make_cosine_schedule(T=int(diff_ckpt["T"]), device=device)
    c_const = make_positional_c(N, int(diff_ckpt["d_c"]), device)

    iu = np.triu_indices(N, k=1)

    def run_one(eta: float, M: int, tau: float, n_steps: int = 200,
                seed: int = 2026) -> dict:
        rng = np.random.default_rng(seed)
        z_idx = rng.choice(train_idx, size=M, replace=True)
        z_pool = torch.from_numpy(z_hat_all[z_idx])[:, None].to(device)
        c_batch = c_const.expand(M, -1, -1)
        t0 = time.time()
        res = guided_ddim_sample(
            net, z_pool, c_batch, alpha_bars, H_obs,
            d0=d0, tau=tau, mu=mu, sigma=sigma,
            n_steps=n_steps, eta=eta,
        )
        wall = time.time() - t0
        D_samp = res["D"].cpu().numpy()
        C_samp_hard = (D_samp < hard_thr).astype(np.float32)
        H_pred = C_samp_hard.mean(axis=0)
        pcc = float(np.corrcoef(H_pred[iu], H_target[iu])[0, 1])
        mse = float(((H_pred - H_target)[iu] ** 2).mean())
        Rg_samp_median = float(np.median([radius_of_gyration(d) for d in D_samp]))
        return {
            "eta": eta, "M": M, "tau": tau, "n_steps": n_steps,
            "pearson": pcc, "mse": mse,
            "Rg_med": Rg_samp_median,
            "loss_start": float(res["losses"][0]),
            "loss_end": float(res["losses"][-1]),
            "wall_s": wall,
        }

    sweeps = {
        "eta": ([100, 1000, 5000, 30000, 100000, 300000],
                "eta", {"M": 128, "tau": 80.0}),
        "M": ([32, 64, 128, 256], "M", {"eta": 30000.0, "tau": 80.0}),
        "tau": ([40, 60, 80, 120, 200], "tau", {"eta": 30000.0, "M": 128}),
    }

    all_results: list[dict] = []
    print(f"target Rg median: {Rg_target_median:.1f} nm")
    print(f"\n{'param':<8s} {'value':>10s} {'Pearson':>10s} {'MSE':>10s} "
          f"{'Rg_med':>10s} {'loss_end':>12s} {'wall_s':>8s}")
    print("-" * 75)
    for sweep_name, (values, var_key, fixed) in sweeps.items():
        for v in values:
            kwargs = {**fixed}
            kwargs[var_key] = v
            r = run_one(**kwargs)
            r["sweep"] = sweep_name
            r["value"] = v
            all_results.append(r)
            print(f"{sweep_name:<8s} {str(v):>10s} {r['pearson']:>10.4f} "
                  f"{r['mse']:>10.5f} {r['Rg_med']:>10.1f} "
                  f"{r['loss_end']:>12.6f} {r['wall_s']:>8.1f}")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROOT / "checkpoints" / "step13_sensitivity.npz",
        results=np.array(all_results, dtype=object),
        Rg_target=Rg_target_median,
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    sweep_colors = {"eta": "C0", "M": "C2", "tau": "C3"}

    for col, (sweep_name, (values, var_key, fixed)) in enumerate(sweeps.items()):
        sub = [r for r in all_results if r["sweep"] == sweep_name]
        xs = np.array([r["value"] for r in sub])
        pccs = np.array([r["pearson"] for r in sub])
        rgs = np.array([r["Rg_med"] for r in sub])

        ax = axes[0, col]
        ax.plot(xs, pccs, "o-", color=sweep_colors[sweep_name], lw=2, ms=8)
        if sweep_name in ("eta",):
            ax.set_xscale("log")
        ax.set_xlabel(sweep_name)
        ax.set_ylabel("bulk Pearson")
        ax.set_title(f"Pearson vs {sweep_name}\n(others fixed at step-8 defaults)")
        ax.grid(True, which="both", alpha=0.3)
        ax.axhline(0.987, color="black", ls="--", lw=0.7, alpha=0.5, label="step-8 default")
        ax.legend(fontsize=8)

        ax = axes[1, col]
        ax.plot(xs, rgs, "o-", color=sweep_colors[sweep_name], lw=2, ms=8)
        if sweep_name in ("eta",):
            ax.set_xscale("log")
        ax.set_xlabel(sweep_name)
        ax.set_ylabel("Rg median (nm)")
        ax.axhline(Rg_target_median, color="black", ls="--", lw=0.7, alpha=0.5,
                   label=f"target {Rg_target_median:.0f} nm")
        ax.set_title(f"Rg vs {sweep_name}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Sensitivity sweep: guided-sampling deconvolution robustness")
    fig.tight_layout()
    out = out_dir / "24_sensitivity_sweep.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
