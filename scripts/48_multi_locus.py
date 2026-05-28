"""B2: Multi-locus / multi-condition bulk-fit generalisation.

Apply one fixed model (step-5 diffusion trained on IMR90 chr21:28-30Mb +
step-5 encoder) to every single-cell tracing dataset we have:

    1. IMR90 chr21:28-30Mb  (in-domain)
    2. IMR90 chr21:18-20Mb  (cross-locus, same cell type)
    3. K562  chr21:28-30Mb  (cross-cell-type, same locus)
    4. HCT116 chr21:28-30Mb untreated (cross-cell-type)

For each, encode cells, sample with guided DDIM, and report bulk Pearson
between the predicted ensemble contact map and the true held-out bulk,
with 95% bootstrap CIs (n=2000 over pair indices).

This is the multi-condition replacement for a true multi-chromosome
test: Bintu 2018 only imaged chr21, so we sweep cell types and loci
within their dataset instead.
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

DATASETS = [
    ("IMR90_chr21:28-30Mb", "IMR90_chr21-28-30Mb.csv"),
    ("IMR90_chr21:18-20Mb", "IMR90_chr21-18-20Mb.csv"),
    ("K562_chr21:28-30Mb",  "K562_chr21-28-30Mb.csv"),
    ("HCT116_chr21:28-30Mb_untreated", "HCT116_chr21-28-30Mb_untreated.csv"),
]


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


def run_locus(name: str, csv: str, device: torch.device,
              enc_state: dict, diff_state: dict, fwd: dict, rng_seed: int) -> dict:
    print(f"\n=== {name} ===")
    path = ROOT / "data" / "raw_bintu2018" / csv
    ds = load_bintu_csv(path)
    real = preprocess_bintu(ds, min_valid_frac=0.85)
    n_cells, N, _ = real.D.shape
    print(f"  loaded {n_cells} cells, N={N}")

    # Encoder
    enc = LoopEncoder(N=N, d_c=int(enc_state["d_c"]), d_pair=32, d_sep=16, d_h=96,
                      dilations=(1, 2, 4, 8, 1)).to(device)
    enc.load_state_dict(enc_state["state_dict"]); enc.eval()
    c_enc = make_positional_c(N, int(enc_state["d_c"]), device)

    mu_loc = float(np.log1p(real.D).mean()); sigma_loc = float(np.log1p(real.D).std())
    x = ((np.log1p(real.D) - mu_loc) / max(sigma_loc, 1e-8)).astype(np.float32)
    z_hat = np.empty((n_cells, N, N), dtype=np.float32)
    bs = 64
    with torch.no_grad():
        for s in range(0, n_cells, bs):
            e = min(s + bs, n_cells)
            x_b = torch.from_numpy(x[s:e])[:, None].to(device)
            c_b = c_enc.expand(e - s, -1, -1)
            z_hat[s:e] = torch.sigmoid(enc(x_b, c_b))[:, 0].cpu().numpy()

    # Held-out split
    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(n_cells)
    n_val = int(0.3 * n_cells)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    hard_thr = float(fwd["hard_threshold"])
    H_target = (real.D[val_idx] < hard_thr).mean(axis=0).astype(np.float32)

    # Diffusion
    net = Denoiser(N=N, d_c=int(diff_state["d_c"]), d_pair=32, d_sep=16, d_h=96,
                   d_t=128, dilations=(1, 2, 4, 8, 1)).to(device)
    net.load_state_dict(diff_state["state_dict"]); net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    alpha_bars = make_cosine_schedule(T=int(diff_state["T"]), device=device)
    c_diff = make_positional_c(N, int(diff_state["d_c"]), device)

    mu_train = float(diff_state["mu"]); sigma_train = float(diff_state["sigma"])
    d0 = float(fwd["d0"]); tau = max(float(fwd["tau"]), 80.0)

    M_samp = 128
    z_idx = rng.choice(train_idx, size=M_samp, replace=True)
    z_pool = torch.from_numpy(z_hat[z_idx])[:, None].to(device)
    c_batch = c_diff.expand(M_samp, -1, -1)
    H_obs = torch.tensor(H_target, device=device)

    print(f"  guided DDIM (M={M_samp})...")
    t0 = time.time()
    res = guided_ddim_sample(
        net, z_pool, c_batch, alpha_bars, H_obs,
        d0=d0, tau=tau, mu=mu_train, sigma=sigma_train,
        n_steps=200, eta=30000.0, log_every=200,
    )
    print(f"  done {time.time()-t0:.1f}s")
    D_samp = res["D"].cpu().numpy()
    H_pred = (D_samp < hard_thr).astype(np.float32).mean(axis=0)

    iu = np.triu_indices(N, k=1)
    a, b = H_pred[iu], H_target[iu]
    pcc = float(np.corrcoef(a, b)[0, 1])
    mse = float(((a - b) ** 2).mean())
    lo, med, hi = bootstrap_pearson(a, b, n=2000, seed=rng_seed)

    return {"name": name, "n_cells": n_cells, "N": N,
            "pearson": pcc, "mse": mse,
            "ci_lo": lo, "ci_med": med, "ci_hi": hi,
            "H_target": H_target, "H_pred": H_pred}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    enc_state = torch.load(ROOT / "checkpoints" / "step05_encoder_N65.pt",
                            map_location=device, weights_only=False)
    diff_state = torch.load(ROOT / "checkpoints" / "step05_diffusion_real.pt",
                             map_location=device, weights_only=False)
    fwd = np.load(ROOT / "checkpoints" / "step06_forward_params.npz")

    results = []
    for i, (name, csv) in enumerate(DATASETS):
        results.append(run_locus(name, csv, device,
                                  enc_state, diff_state, fwd,
                                  rng_seed=2027 + i))

    print("\n" + "=" * 80)
    print(f"{'dataset':<38s} {'cells':>6s} {'Pearson':>9s} {'95% CI':>22s}")
    print("-" * 80)
    for r in results:
        ci = f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
        print(f"{r['name']:<38s} {r['n_cells']:>6d} {r['pearson']:>9.4f} {ci:>22s}")

    np.savez_compressed(
        ROOT / "checkpoints" / "step33_multi_locus.npz",
        names=np.array([r["name"] for r in results]),
        pearson=np.array([r["pearson"] for r in results]),
        ci_lo=np.array([r["ci_lo"] for r in results]),
        ci_hi=np.array([r["ci_hi"] for r in results]),
        mse=np.array([r["mse"] for r in results]),
        n_cells=np.array([r["n_cells"] for r in results]),
    )

    fig, axes = plt.subplots(2, len(results), figsize=(4.5 * len(results), 8))
    for col, r in enumerate(results):
        ax = axes[0, col]
        vmax = max(r["H_target"].max(), r["H_pred"].max())
        im = ax.imshow(r["H_target"], origin="lower", cmap="Reds", vmin=0, vmax=vmax)
        ax.set_title(f"{r['name']}\ntarget ({r['n_cells']} cells)", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[1, col]
        im = ax.imshow(r["H_pred"], origin="lower", cmap="Reds", vmin=0, vmax=vmax)
        ax.set_title(f"predicted\nPearson {r['pearson']:.4f}\n"
                     f"95% CI [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("B2: bulk fit across loci and cell types -- one fixed model")
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "48_multi_locus.png", dpi=130)
    print(f"saved outputs/48_multi_locus.png")

    # Forest plot
    fig, ax = plt.subplots(figsize=(10, 4))
    y = np.arange(len(results))
    meds = [r["pearson"] for r in results]
    los = [r["ci_lo"] for r in results]
    his = [r["ci_hi"] for r in results]
    err = [[m - l for m, l in zip(meds, los)], [h - m for h, m in zip(his, meds)]]
    ax.errorbar(meds, y, xerr=err, fmt="o", ms=11, capsize=8, lw=2,
                color="black", ecolor="gray")
    for k, (m, l, h) in enumerate(zip(meds, los, his)):
        ax.text(h + 0.002, k, f"{m:.4f} [{l:.4f}, {h:.4f}]", va="center", fontsize=10)
    ax.set_yticks(y); ax.set_yticklabels([r["name"] for r in results], fontsize=10)
    ax.set_xlabel("bulk Pearson with held-out target (95% bootstrap CI, n=2000)")
    ax.set_title("B2: multi-locus / multi-cell-type bulk fit")
    ax.set_xlim(min(los) - 0.01, 1.0)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(ROOT / "outputs" / "48_multi_locus_forest.png", dpi=130)
    print(f"saved outputs/48_multi_locus_forest.png")


if __name__ == "__main__":
    main()
