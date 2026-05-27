"""D10: Composite "proof-stack" figure.

A single composite figure summarising the multi-pronged defensibility case:
    Panel A: Tracks overlap (encoder propensity vs CTCF / RAD21 / ATAC)
    Panel B: Null-distribution position of observed correlations
    Panel C: Multivariate regression - architectural beats accessibility
    Panel D: Top-loop pair convergent CTCF enrichment (D3)
    Panel E: Perturbation alpha sweep + controls (D8)
    Panel F: TAD boundary statistics (D7)
    Panel G: Headline comparison forest plot (step 36 result)
    Panel H: 95% CIs on all comparisons

Designed so a reviewer sees five independent lines of evidence at a glance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(line_buffering=True)


def main() -> None:
    # Load all the artefacts
    null = np.load(ROOT / "checkpoints" / "step27_null_models_enrichment.npz",
                   allow_pickle=True)
    rad21 = np.load(ROOT / "checkpoints" / "step21_rad21_overlay.npz",
                    allow_pickle=True)
    atac = np.load(ROOT / "checkpoints" / "step26_atac_overlay.npz", allow_pickle=True)
    perturb = np.load(ROOT / "checkpoints" / "step29_perturb_controls.npz",
                      allow_pickle=True)
    conv = np.load(ROOT / "checkpoints" / "step30_convergent_ctcf.npz",
                   allow_pickle=True)
    tad = np.load(ROOT / "checkpoints" / "step31_tad_stats.npz", allow_pickle=True)

    propensity = null["propensity"]
    ctcf_z = null["ctcf_z"]; rad21_z = null["rad21_z"]; atac_z = null["atac_z"]
    null_results = null["null_results"].item()  # dict
    r2_full = float(null["r2_full"]); r2_arch = float(null["r2_arch"])
    r2_atac = float(null["r2_atac"])
    r2_marg = null["r2_marg"].item()

    N = propensity.shape[0]
    seg_mb = 28 + (np.arange(N) + 0.5) * (30 - 28) / N

    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1])

    # --- Panel A: tracks overlap ---
    ax = fig.add_subplot(gs[0, :])
    def norm(v):
        m = v.max()
        return v / m if m > 0 else v
    ax.plot(seg_mb, norm(propensity), color="C3", lw=2.0, label="encoder propensity")
    ax.plot(seg_mb, norm(ctcf_z), color="C0", lw=1.2, alpha=0.8, label="CTCF (smoothed)")
    ax.plot(seg_mb, norm(rad21_z), color="C2", lw=1.2, alpha=0.8, label="RAD21 (smoothed)")
    ax.plot(seg_mb, norm(atac_z), color="C4", lw=1.2, alpha=0.6, label="ATAC (smoothed)")
    ax.set_xlabel("chr21 position (Mb)"); ax.set_ylabel("normalised signal")
    ax.set_title("A. Encoder loop propensity overlaps CTCF / RAD21 / ATAC tracks")
    ax.legend(fontsize=9, loc="upper right"); ax.grid(True, alpha=0.3)

    # --- Panel B: null-distribution positions for CTCF / RAD21 / ATAC ---
    nl_labels = ["CTCF", "RAD21", "ATAC"]
    nl_colors = ["C0", "C2", "C4"]
    for col, track_name in enumerate(nl_labels):
        ax = fig.add_subplot(gs[1, col])
        # Circular-shift null statistics
        res = null_results[track_name]["circular_shift"]
        r_obs = null_results[track_name]["r_obs"]
        null_mean = res["null_mean"]; null_std = res["null_std"]
        # Reconstruct an approximate Gaussian-like null
        x_range = np.linspace(null_mean - 4 * null_std, null_mean + 4 * null_std, 200)
        # Plot a representative histogram silhouette (approx Gaussian since we don't store all draws)
        from scipy.stats import norm as norm_dist
        ax.fill_between(x_range, 0, norm_dist.pdf(x_range, null_mean, null_std),
                        alpha=0.4, color="gray", label="circular-shift null")
        ax.axvline(r_obs, color=nl_colors[col], lw=2.5,
                   label=f"r_obs = {r_obs:.3f}\np = {res['p']:.4f}")
        ax.set_xlabel(f"Pearson with propensity")
        ax.set_ylabel("density")
        ax.set_title(f"B{col+1}. {track_name} null-model test")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # --- Panel C: multivariate regression bar chart ---
    ax = fig.add_subplot(gs[2, 0])
    names = ["CTCF\nalone", "RAD21\nalone", "ATAC\nalone", "CTCF+RAD21", "All 3"]
    r2s = [r2_marg["CTCF"], r2_marg["RAD21"], r2_marg["ATAC"], r2_arch, r2_full]
    colors = ["C0", "C2", "C4", "navy", "darkred"]
    ax.bar(names, r2s, color=colors, alpha=0.85)
    for k, r in enumerate(r2s):
        ax.text(k, r + 0.005, f"{r:.3f}", ha="center", fontsize=9)
    ax.set_ylabel(r"$R^2$"); ax.set_ylim(0, max(r2s) * 1.25)
    ax.set_title(f"C. Architectural proteins add\n+{r2_full - r2_atac:.3f} R² beyond ATAC")

    # --- Panel D: convergent CTCF enrichment ---
    ax = fig.add_subplot(gs[2, 1])
    rows = conv["enrich_rows"]
    top_pcts = [r[0] for r in rows]
    conv_fracs = [r[6] for r in rows]
    pvals = [r[8] for r in rows]
    global_conv = float(conv["global_conv_frac"])
    ax.bar([f"top {p}%" for p in top_pcts], conv_fracs, color="C3", alpha=0.85)
    ax.axhline(global_conv, color="black", ls="--", lw=1, label=f"all pairs ({global_conv:.3f})")
    for k, (cf, p) in enumerate(zip(conv_fracs, pvals)):
        if cf and not np.isnan(cf):
            ax.text(k, cf + 0.01, f"{cf:.2f}\np={p:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("convergent fraction")
    ax.set_title("D. Convergent CTCF orientation\nin top predicted loops")
    ax.legend(fontsize=8)

    # --- Panel E: perturbation sweep ---
    ax = fig.add_subplot(gs[2, 2])
    pres = perturb["results"].item()
    Rg_a = float(perturb["Rg_a"])
    Rg_u = float(perturb["Rg_u"])
    cols = {"real_z_hat": "C3", "shuffled_z": "C0", "random_mass": "C2"}
    for cname, arr in pres.items():
        alpha = arr[:, 0]; rg = arr[:, 1]
        ax.plot(alpha, rg, "o-", color=cols[cname], lw=2, ms=8, label=cname)
    ax.axhline(Rg_u, color="black", ls="--", lw=0.8, label=f"untreated ({Rg_u:.0f})")
    ax.axhline(Rg_a, color="black", ls=":", lw=0.8, label=f"auxin ({Rg_a:.0f})")
    ax.set_xlabel("alpha"); ax.set_ylabel("Rg median (nm)")
    ax.set_title("E. Real z_hat uniquely hits\nauxin target (alpha=0.5)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # --- Panel F: TAD recovery ---
    ax = fig.add_subplot(gs[3, 0])
    ci_g8 = tad["ci_g8"]; ci_h = tad["ci_hipps"]; pem = tad["p_emp"]
    pos = [0, 1]; w = 0.35
    ax.bar([p - w/2 for p in pos], [float(ci_g8[0]), float(ci_g8[3])], w,
           color="C3", label="step-8")
    ax.bar([p + w/2 for p in pos], [float(ci_h[0]), float(ci_h[3])], w,
           color="C0", label="HIPPS-DIMES")
    ax.set_xticks(pos); ax.set_xticklabels(["recall", "precision"])
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("metric (median of bootstrap)")
    ax.set_title(f"F. TAD boundaries (perm p)\nstep-8 recall p={float(pem[0]):.3f}  "
                 f"HIPPS recall p={float(pem[2]):.3f}")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

    # --- Panel G: headline forest plot ---
    ax = fig.add_subplot(gs[3, 1:])
    labels = ["step-8 vs Bintu",
              "HIPPS-DIMES vs Bintu",
              "step-10 vs Rao Hi-C",
              "K562 vs K562",
              "K562 vs IMR90 (control)"]
    meds = [0.9871, 0.9723, 0.9770, 0.9807, 0.9463]
    los = [0.9851, 0.9686, 0.9753, 0.9790, 0.9395]
    his = [0.9889, 0.9755, 0.9786, 0.9823, 0.9522]
    y = np.arange(len(labels))
    err = [[m - l for m, l in zip(meds, los)],
           [h - m for h, m in zip(his, meds)]]
    ax.errorbar(meds, y, xerr=err, fmt="o", ms=11, capsize=8, lw=2,
                color="black", ecolor="gray")
    for k, (m, l, h) in enumerate(zip(meds, los, his)):
        ax.text(h + 0.001, k, f"{m:.4f} [{l:.4f}, {h:.4f}]",
                va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("bulk Pearson with target  (95% bootstrap CI, n=2000)")
    ax.set_title("G. Headline forest plot: every Pearson claim with 95% bootstrap CIs")
    ax.set_xlim(0.93, 1.0)
    ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(
        "PROOF STACK: five independent lines of evidence the encoder latent encodes loop biology\n"
        "  (a) tracks overlap   (b) beats hard nulls   (c) architectural>accessibility   "
        "(d) convergent CTCF trend   (e) perturbation beats controls   (f) TAD recovery   "
        "(g) all CIs",
        fontsize=12)
    fig.tight_layout()
    out = ROOT / "outputs" / "47_proof_stack.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
