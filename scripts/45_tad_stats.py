"""D7: Statistical TAD boundary recovery — beyond raw recall/precision.

Step 32 reported recall=0.90, precision=0.90 for step-8 guided TAD boundaries
vs Bintu imaging truth. Now make it statistical:

  - Permutation null: shuffle predicted boundary positions, recompute recall/precision
  - Bootstrap CI on the recall/precision numbers
  - Nearest-boundary distance distribution: how far is the nearest predicted
    boundary from each true boundary?  Compared to random boundaries.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(line_buffering=True)


def insulation_score(C, w=5):
    N = C.shape[0]
    out = np.full(N, np.nan)
    for i in range(w, N - w):
        out[i] = C[i - w:i, i + 1:i + w + 1].mean()
    m, M = np.nanmin(out), np.nanmax(out)
    if M > m: out = (out - m) / (M - m)
    return out


def local_minima(s, min_distance=3):
    out = []
    N = s.shape[0]
    for i in range(1, N - 1):
        if not np.isfinite(s[i]):
            continue
        left = s[i - 1] if np.isfinite(s[i - 1]) else np.inf
        right = s[i + 1] if np.isfinite(s[i + 1]) else np.inf
        if s[i] < left and s[i] < right:
            if not out or i - out[-1] >= min_distance:
                out.append(i)
    return out


def overlap(b1, b2, tol=2):
    """recall1 = fraction of b1 with match in b2; precision = fraction of b2 with match in b1"""
    if len(b1) == 0 or len(b2) == 0:
        return 0.0, 0.0
    matched_1 = sum(any(abs(a - b) <= tol for b in b2) for a in b1)
    matched_2 = sum(any(abs(a - b) <= tol for b in b1) for a in b2)
    return matched_1 / len(b1), matched_2 / len(b2)


def main() -> None:
    fwd = np.load(ROOT / "checkpoints" / "step06_forward_params.npz")
    hard_thr = float(fwd["hard_threshold"])
    import torch
    real_path = ROOT / "data" / "real" / "IMR90_chr21-28-30Mb_preprocessed.npz"
    f = np.load(real_path)
    D_real = f["D"]
    diff_ckpt = torch.load(ROOT / "checkpoints" / "step05_diffusion_real.pt",
                           map_location="cpu", weights_only=False)
    val_idx = np.array(diff_ckpt["val_idx"])
    H_truth = (D_real[val_idx] < hard_thr).astype(np.float32).mean(axis=0)

    gd8 = np.load(ROOT / "checkpoints" / "step08_guided.npz")
    H_g8 = (gd8["D_samples"] < hard_thr).astype(np.float32).mean(axis=0)
    h14 = np.load(ROOT / "checkpoints" / "step14_hipps_dimes.npz")
    H_hipps = h14["H_reweighted"]

    s_truth = insulation_score(H_truth)
    s_g8 = insulation_score(H_g8)
    s_hipps = insulation_score(H_hipps)
    b_truth = local_minima(s_truth)
    b_g8 = local_minima(s_g8)
    b_hipps = local_minima(s_hipps)
    N = H_truth.shape[0]

    print(f"truth boundaries:    {b_truth}")
    print(f"step-8 boundaries:   {b_g8}")
    print(f"HIPPS boundaries:    {b_hipps}")

    # Bootstrap CIs on recall/precision
    def boot_overlap(b_pred, b_truth_, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        recalls = np.zeros(n); precs = np.zeros(n)
        for k in range(n):
            # Resample truth boundaries WITH replacement
            sb = list(rng.choice(b_truth_, size=len(b_truth_), replace=True)) if b_truth_ else []
            r, p = overlap(sb, b_pred, tol=2)
            recalls[k] = r; precs[k] = p
        return (float(np.median(recalls)), float(np.percentile(recalls, 2.5)),
                float(np.percentile(recalls, 97.5)),
                float(np.median(precs)), float(np.percentile(precs, 2.5)),
                float(np.percentile(precs, 97.5)))

    r8_m, r8_lo, r8_hi, p8_m, p8_lo, p8_hi = boot_overlap(b_g8, b_truth)
    rh_m, rh_lo, rh_hi, ph_m, ph_lo, ph_hi = boot_overlap(b_hipps, b_truth)
    print(f"\nbootstrap recall/precision (95% CI):")
    print(f"  step-8:      recall={r8_m:.2f} [{r8_lo:.2f}, {r8_hi:.2f}]  "
          f"precision={p8_m:.2f} [{p8_lo:.2f}, {p8_hi:.2f}]")
    print(f"  HIPPS-DIMES: recall={rh_m:.2f} [{rh_lo:.2f}, {rh_hi:.2f}]  "
          f"precision={ph_m:.2f} [{ph_lo:.2f}, {ph_hi:.2f}]")

    # Permutation null: random boundary sets
    rng = np.random.default_rng(2031)
    n_perm = 5000
    null_recalls_g8 = np.zeros(n_perm)
    null_precs_g8 = np.zeros(n_perm)
    null_recalls_hipps = np.zeros(n_perm)
    null_precs_hipps = np.zeros(n_perm)
    for k in range(n_perm):
        # Random boundaries: pick len(b_g8) bins uniformly
        rb_g8 = list(rng.choice(np.arange(5, N - 5), size=len(b_g8), replace=False))
        rb_hipps = list(rng.choice(np.arange(5, N - 5), size=len(b_hipps), replace=False))
        null_recalls_g8[k], null_precs_g8[k] = overlap(b_truth, rb_g8, tol=2)
        null_recalls_hipps[k], null_precs_hipps[k] = overlap(b_truth, rb_hipps, tol=2)

    # observed
    obs_r8, obs_p8 = overlap(b_truth, b_g8, tol=2)
    obs_rh, obs_ph = overlap(b_truth, b_hipps, tol=2)
    p_emp_r8 = float((null_recalls_g8 >= obs_r8).mean())
    p_emp_p8 = float((null_precs_g8 >= obs_p8).mean())
    p_emp_rh = float((null_recalls_hipps >= obs_rh).mean())
    p_emp_ph = float((null_precs_hipps >= obs_ph).mean())

    print(f"\npermutation null (random boundary sets matched for count):")
    print(f"  step-8 recall    {obs_r8:.2f}  null mean {null_recalls_g8.mean():.3f}  emp p={p_emp_r8:.4f}")
    print(f"  step-8 precision {obs_p8:.2f}  null mean {null_precs_g8.mean():.3f}  emp p={p_emp_p8:.4f}")
    print(f"  HIPPS  recall    {obs_rh:.2f}  null mean {null_recalls_hipps.mean():.3f}  emp p={p_emp_rh:.4f}")
    print(f"  HIPPS  precision {obs_ph:.2f}  null mean {null_precs_hipps.mean():.3f}  emp p={p_emp_ph:.4f}")

    # Nearest-boundary distance distribution
    def nearest_dist(true_bs, pred_bs):
        if len(pred_bs) == 0: return np.array([np.inf] * len(true_bs))
        return np.array([min(abs(t - p) for p in pred_bs) for t in true_bs])

    d_g8 = nearest_dist(b_truth, b_g8)
    d_hipps = nearest_dist(b_truth, b_hipps)
    # Null: random
    null_dists = []
    for _ in range(2000):
        rb = list(rng.choice(np.arange(5, N - 5), size=len(b_g8), replace=False))
        null_dists.extend(nearest_dist(b_truth, rb).tolist())

    print(f"\nnearest predicted boundary to each true boundary:")
    print(f"  step-8:  median dist = {float(np.median(d_g8)):.1f} bins  mean = {float(d_g8.mean()):.2f}")
    print(f"  HIPPS:   median dist = {float(np.median(d_hipps)):.1f} bins  mean = {float(d_hipps.mean()):.2f}")
    print(f"  null random:  median = {float(np.median(null_dists)):.1f}  mean = {float(np.mean(null_dists)):.2f}")

    np.savez(ROOT / "checkpoints" / "step31_tad_stats.npz",
        b_truth=np.array(b_truth), b_g8=np.array(b_g8), b_hipps=np.array(b_hipps),
        ci_g8=(r8_m, r8_lo, r8_hi, p8_m, p8_lo, p8_hi),
        ci_hipps=(rh_m, rh_lo, rh_hi, ph_m, ph_lo, ph_hi),
        p_emp=(p_emp_r8, p_emp_p8, p_emp_rh, p_emp_ph),
        d_g8=d_g8, d_hipps=d_hipps,
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax = axes[0, 0]
    ax.hist(null_recalls_g8, bins=20, alpha=0.6, color="gray", label=f"random null (n={n_perm})")
    ax.axvline(obs_r8, color="C3", lw=2.5, label=f"step-8 observed = {obs_r8:.2f}, p={p_emp_r8:.4f}")
    ax.axvline(obs_rh, color="C0", lw=2.5, label=f"HIPPS observed = {obs_rh:.2f}, p={p_emp_rh:.4f}")
    ax.set_xlabel("recall"); ax.set_ylabel("count")
    ax.set_title("Recall: observed vs random-boundary null")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.hist(null_precs_g8, bins=20, alpha=0.6, color="gray", label=f"random null")
    ax.axvline(obs_p8, color="C3", lw=2.5, label=f"step-8 observed = {obs_p8:.2f}, p={p_emp_p8:.4f}")
    ax.axvline(obs_ph, color="C0", lw=2.5, label=f"HIPPS observed = {obs_ph:.2f}, p={p_emp_ph:.4f}")
    ax.set_xlabel("precision"); ax.set_ylabel("count")
    ax.set_title("Precision: observed vs random-boundary null")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.hist(null_dists, bins=15, alpha=0.6, color="gray", density=True,
            label=f"random nulls")
    ax.hist(d_g8, bins=8, alpha=0.7, color="C3", density=True,
            label=f"step-8 (med {float(np.median(d_g8)):.1f})")
    ax.hist(d_hipps, bins=8, alpha=0.6, color="C0", density=True,
            label=f"HIPPS (med {float(np.median(d_hipps)):.1f})")
    ax.set_xlabel("nearest pred boundary to true boundary (bins)")
    ax.set_ylabel("density")
    ax.set_title("Distance to nearest predicted boundary")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")
    summary = (
        f"STATISTICAL TAD BOUNDARY RECOVERY\n\n"
        f"True boundaries: {len(b_truth)}\n"
        f"Step-8 boundaries: {len(b_g8)}\n"
        f"HIPPS boundaries:  {len(b_hipps)}\n\n"
        f"95% bootstrap CIs:\n"
        f"  step-8: recall {r8_m:.2f} [{r8_lo:.2f}, {r8_hi:.2f}]\n"
        f"           prec {p8_m:.2f} [{p8_lo:.2f}, {p8_hi:.2f}]\n"
        f"  HIPPS:  recall {rh_m:.2f} [{rh_lo:.2f}, {rh_hi:.2f}]\n"
        f"           prec {ph_m:.2f} [{ph_lo:.2f}, {ph_hi:.2f}]\n\n"
        f"Permutation p-values vs random boundary sets:\n"
        f"  step-8 recall      p={p_emp_r8:.4f}\n"
        f"  step-8 precision   p={p_emp_p8:.4f}\n"
        f"  HIPPS recall       p={p_emp_rh:.4f}\n"
        f"  HIPPS precision    p={p_emp_ph:.4f}\n\n"
        f"Nearest-boundary distance (median, bins):\n"
        f"  step-8: {float(np.median(d_g8)):.1f}\n"
        f"  HIPPS:  {float(np.median(d_hipps)):.1f}\n"
        f"  random: {float(np.median(null_dists)):.1f}\n"
    )
    ax.text(0.0, 0.95, summary, fontsize=10, va="top", family="monospace")

    fig.suptitle("D7: TAD boundary recovery -- bootstrap CIs + permutation tests")
    fig.tight_layout()
    out = ROOT / "outputs" / "45_tad_stats.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
