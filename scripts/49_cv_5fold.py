"""B3: 5-fold cell-level cross-validation on Bintu IMR90 chr21:28-30Mb.

Split the ~3,881 Bintu cells into 5 stratified random folds. For each
fold:
    - hold out 20% of cells as the bulk-target ensemble
    - use the other 80%'s encoder z_hats as the z_pool for sampling
    - run guided DDIM, compute bulk Pearson against the held-out bulk
    - 95% bootstrap CI (n=2000)

Report mean +- std Pearson across folds plus per-fold CIs. A consistent,
high Pearson across folds (rather than one cherry-picked split) is what
a reviewer wants to see to rule out test-set leakage or overfit.
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
from hic_unfold.encoder import LoopEncoder  # noqa: E402
from hic_unfold.training import make_positional_c  # noqa: E402


def bootstrap_pearson(a: np.ndarray, b: np.ndarray, n: int = 2000,
                       seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    K = a.size
    vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, K, size=K)
        vals[i] = np.corrcoef(a[idx], b[idx])[0, 1]
    return (float(np.percentile(vals, 2.5)),
            float(np.median(vals)),
            float(np.percentile(vals, 97.5)))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    enc_state = torch.load(ROOT / "checkpoints" / "step05_encoder_N65.pt",
                            map_location=device, weights_only=False)
    diff_state = torch.load(ROOT / "checkpoints" / "step05_diffusion_real.pt",
                             map_location=device, weights_only=False)
    fwd = np.load(ROOT / "checkpoints" / "step06_forward_params.npz")
    hard_thr = float(fwd["hard_threshold"])
    d0 = float(fwd["d0"]); tau = max(float(fwd["tau"]), 80.0)

    ds = load_bintu_csv(ROOT / "data" / "raw_bintu2018" / "IMR90_chr21-28-30Mb.csv")
    real = preprocess_bintu(ds, min_valid_frac=0.85)
    n_cells, N, _ = real.D.shape
    print(f"loaded {n_cells} cells at N={N}")

    enc = LoopEncoder(N=N, d_c=int(enc_state["d_c"]), d_pair=32, d_sep=16, d_h=96,
                      dilations=(1, 2, 4, 8, 1)).to(device)
    enc.load_state_dict(enc_state["state_dict"]); enc.eval()
    c_enc = make_positional_c(N, int(enc_state["d_c"]), device)
    mu_loc = float(np.log1p(real.D).mean()); sigma_loc = float(np.log1p(real.D).std())
    x = ((np.log1p(real.D) - mu_loc) / max(sigma_loc, 1e-8)).astype(np.float32)
    z_hat = np.empty((n_cells, N, N), dtype=np.float32)
    bs = 64
    print("encoding all cells once...")
    with torch.no_grad():
        for s in range(0, n_cells, bs):
            e = min(s + bs, n_cells)
            x_b = torch.from_numpy(x[s:e])[:, None].to(device)
            c_b = c_enc.expand(e - s, -1, -1)
            z_hat[s:e] = torch.sigmoid(enc(x_b, c_b))[:, 0].cpu().numpy()

    net = Denoiser(N=N, d_c=int(diff_state["d_c"]), d_pair=32, d_sep=16, d_h=96,
                   d_t=128, dilations=(1, 2, 4, 8, 1)).to(device)
    net.load_state_dict(diff_state["state_dict"]); net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    alpha_bars = make_cosine_schedule(T=int(diff_state["T"]), device=device)
    c_diff = make_positional_c(N, int(diff_state["d_c"]), device)
    mu_train = float(diff_state["mu"]); sigma_train = float(diff_state["sigma"])

    rng = np.random.default_rng(2027)
    perm = rng.permutation(n_cells)
    folds = np.array_split(perm, 5)

    iu = np.triu_indices(N, k=1)
    fold_results = []
    for k, val_idx in enumerate(folds):
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != k])
        H_target = (real.D[val_idx] < hard_thr).mean(axis=0).astype(np.float32)
        M_samp = 128
        z_idx = rng.choice(train_idx, size=M_samp, replace=True)
        z_pool = torch.from_numpy(z_hat[z_idx])[:, None].to(device)
        c_batch = c_diff.expand(M_samp, -1, -1)
        H_obs = torch.tensor(H_target, device=device)

        print(f"\nfold {k+1}/5: train={len(train_idx)}, val={len(val_idx)}")
        t0 = time.time()
        res = guided_ddim_sample(
            net, z_pool, c_batch, alpha_bars, H_obs,
            d0=d0, tau=tau, mu=mu_train, sigma=sigma_train,
            n_steps=200, eta=30000.0, log_every=200,
        )
        D_samp = res["D"].cpu().numpy()
        H_pred = (D_samp < hard_thr).astype(np.float32).mean(axis=0)
        a, b = H_pred[iu], H_target[iu]
        pcc = float(np.corrcoef(a, b)[0, 1])
        mse = float(((a - b) ** 2).mean())
        lo, med, hi = bootstrap_pearson(a, b, n=2000, seed=2027 + k)
        print(f"  Pearson={pcc:.4f} [{lo:.4f}, {hi:.4f}]   MSE={mse:.5f}   "
              f"{time.time()-t0:.1f}s")
        fold_results.append({"fold": k+1, "n_val": int(len(val_idx)),
                             "pearson": pcc, "mse": mse,
                             "ci_lo": lo, "ci_med": med, "ci_hi": hi,
                             "H_target": H_target, "H_pred": H_pred})

    pearsons = np.array([r["pearson"] for r in fold_results])
    mses = np.array([r["mse"] for r in fold_results])
    print("\n" + "=" * 60)
    print(f"5-fold CV summary on IMR90 chr21:28-30Mb")
    print(f"  Pearson mean +- std:  {pearsons.mean():.4f} +- {pearsons.std():.4f}")
    print(f"  Pearson range:        [{pearsons.min():.4f}, {pearsons.max():.4f}]")
    print(f"  MSE mean +- std:      {mses.mean():.5f} +- {mses.std():.5f}")

    np.savez_compressed(
        ROOT / "checkpoints" / "step34_cv_5fold.npz",
        pearsons=pearsons, mses=mses,
        ci_lo=np.array([r["ci_lo"] for r in fold_results]),
        ci_hi=np.array([r["ci_hi"] for r in fold_results]),
    )

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for k, r in enumerate(fold_results):
        vmax = max(r["H_target"].max(), r["H_pred"].max())
        ax = axes[0, k]
        im = ax.imshow(r["H_target"], origin="lower", cmap="Reds", vmin=0, vmax=vmax)
        ax.set_title(f"fold {r['fold']} target\n({r['n_val']} held-out cells)", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax = axes[1, k]
        im = ax.imshow(r["H_pred"], origin="lower", cmap="Reds", vmin=0, vmax=vmax)
        ax.set_title(f"predicted\nPearson {r['pearson']:.4f}\n"
                     f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"B3: 5-fold cell-level CV  --  "
                 f"Pearson {pearsons.mean():.4f} +- {pearsons.std():.4f}")
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "49_cv_5fold.png", dpi=130)
    print(f"\nsaved outputs/49_cv_5fold.png")

    # Box / dot plot
    fig, ax = plt.subplots(figsize=(7, 5))
    x_pos = np.arange(1, 6)
    los = [r["ci_lo"] for r in fold_results]
    his = [r["ci_hi"] for r in fold_results]
    err = [[m - l for m, l in zip(pearsons, los)],
           [h - m for h, m in zip(his, pearsons)]]
    ax.errorbar(x_pos, pearsons, yerr=err, fmt="o", ms=12, capsize=8, lw=2,
                color="black", ecolor="gray")
    ax.axhline(pearsons.mean(), color="C3", ls="--", lw=1.5,
               label=f"mean = {pearsons.mean():.4f}")
    ax.fill_between([0.5, 5.5],
                    pearsons.mean() - pearsons.std(),
                    pearsons.mean() + pearsons.std(),
                    color="C3", alpha=0.15, label=f"+/- std ({pearsons.std():.4f})")
    ax.set_xticks(x_pos); ax.set_xlabel("fold")
    ax.set_ylabel("bulk Pearson (95% bootstrap CI)")
    ax.set_title("B3: 5-fold cell-level cross-validation on Bintu IMR90 chr21:28-30Mb")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 5.5)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "49_cv_5fold_summary.png", dpi=130)
    print(f"saved outputs/49_cv_5fold_summary.png")


if __name__ == "__main__":
    main()
