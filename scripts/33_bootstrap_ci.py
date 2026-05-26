"""Bootstrap 95% CIs for the headline metrics so improvements can be called
statistically significant.

For each (method, target) pair, bootstrap-resample the upper-triangle pair
indices to get a distribution of Pearson correlations. Report median and 95%
CI for each method and check whether intervals overlap.

Run:
    python scripts/33_bootstrap_ci.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(line_buffering=True)


def boot_pearson(a: np.ndarray, b: np.ndarray, n_boot: int = 2000,
                 seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = a.shape[0]
    out = np.zeros(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        out[k] = np.corrcoef(a[idx], b[idx])[0, 1]
    return float(np.median(out)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main() -> None:
    fwd = np.load(ROOT / "checkpoints" / "step06_forward_params.npz")
    hard_thr = float(fwd["hard_threshold"])

    real_path = ROOT / "data" / "real" / "IMR90_chr21-28-30Mb_preprocessed.npz"
    f = np.load(real_path)
    D_real = f["D"]

    import torch
    diff_ckpt = torch.load(ROOT / "checkpoints" / "step05_diffusion_real.pt",
                           map_location="cpu", weights_only=False)
    val_idx = np.array(diff_ckpt["val_idx"])
    H_target = (D_real[val_idx] < hard_thr).astype(np.float32).mean(axis=0)

    gd8 = np.load(ROOT / "checkpoints" / "step08_guided.npz")
    H_g8 = (gd8["D_samples"] < hard_thr).astype(np.float32).mean(axis=0)

    me7 = np.load(ROOT / "checkpoints" / "step07_maxent.npz")
    H_me7 = me7["lam"] * 0 + 0  # placeholder
    # Reweighted prediction for step 7: recompute from saved weights and pool z_hat
    # Actually we saved H_pred there. Look at the file.
    if "H_pred" in me7.files:
        H_me7 = me7["H_pred"]
    else:
        # Reconstruct via stored pool — too expensive; use the val target check value
        H_me7 = None

    h14 = np.load(ROOT / "checkpoints" / "step14_hipps_dimes.npz")
    H_hipps = h14["H_reweighted"]

    N = H_target.shape[0]
    iu = np.triu_indices(N, k=1)
    target_v = H_target[iu]

    methods = {
        "Step 7: diffusion + maxent": H_me7,
        "Step 8: diffusion + guided": H_g8,
        "HIPPS-DIMES (polymer + maxent)": H_hipps,
    }

    print(f"{'method':<36s} {'Pearson':>10s} {'95% CI lo':>12s} {'95% CI hi':>12s}")
    print("-" * 75)
    cis: dict[str, tuple[float, float, float]] = {}
    for name, H in methods.items():
        if H is None:
            print(f"{name:<36s} (skipped — saved H not found)")
            continue
        pred_v = H[iu]
        med, lo, hi = boot_pearson(pred_v, target_v, n_boot=2000, seed=2026)
        cis[name] = (med, lo, hi)
        print(f"{name:<36s} {med:>10.4f} {lo:>12.4f} {hi:>12.4f}")

    print()
    # Pairwise comparisons: is the difference > 0 in >97.5% of bootstrap samples?
    print("pairwise differences (positive = first method better):")
    rng = np.random.default_rng(2027)
    keys = list(cis.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a_name = keys[i]; b_name = keys[j]
            a_pred = methods[a_name][iu]; b_pred = methods[b_name][iu]
            n_boot = 2000; out = np.zeros(n_boot)
            for k in range(n_boot):
                idx = rng.integers(0, target_v.shape[0], size=target_v.shape[0])
                pa = np.corrcoef(a_pred[idx], target_v[idx])[0, 1]
                pb = np.corrcoef(b_pred[idx], target_v[idx])[0, 1]
                out[k] = pa - pb
            med = float(np.median(out))
            lo = float(np.percentile(out, 2.5))
            hi = float(np.percentile(out, 97.5))
            pval_better = float((out > 0).mean())
            print(f"  {a_name[:25]:<25s} - {b_name[:25]:<25s}: "
                  f"delta={med:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
                  f"P(better)={pval_better:.3f}")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]
    names = list(cis.keys())
    meds = [cis[n][0] for n in names]
    los = [cis[n][1] for n in names]
    his = [cis[n][2] for n in names]
    pos = np.arange(len(names))
    err_lo = [m - l for m, l in zip(meds, los)]
    err_hi = [h - m for h, m in zip(his, meds)]
    ax.errorbar(pos, meds, yerr=[err_lo, err_hi], fmt="o", ms=12, capsize=10,
                lw=2, color="black", ecolor="gray")
    for k, c in enumerate(colors[:len(names)]):
        ax.scatter([pos[k]], [meds[k]], s=200, color=c, zorder=10, edgecolor="black", lw=1)
    ax.set_xticks(pos); ax.set_xticklabels(names, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("bulk Pearson vs target")
    ax.set_title("Headline comparison with 95% bootstrap CIs (n=2000)")
    for k, (m, l, h) in enumerate(zip(meds, los, his)):
        ax.text(pos[k], h + 0.001, f"{m:.4f}\n[{l:.4f}, {h:.4f}]",
                ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(min(los) - 0.005, max(his) + 0.01)
    fig.tight_layout()
    out = out_dir / "33_bootstrap_ci.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
