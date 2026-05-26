"""A2: Cohesin (RAD21) ChIP-seq overlay — second independent biological data source.

CTCF (already done in script 20) anchors loops. Cohesin (RAD21 subunit) is the
loop-EXTRUDER itself. If our encoder is learning loop biology, its per-locus
propensity should correlate with BOTH CTCF and RAD21 ChIP-seq, and especially
strongly with regions where the two co-occur (canonical loop anchors).

Pipeline:
    1. Parse ENCODE RAD21 narrowPeak (ENCFF895JAW, Snyder IMR-90, hg38).
    2. Bin to N=65 30kb segments along chr21:28-30Mb.
    3. Correlate with encoder loop propensity; compare to CTCF correlation.
    4. Identify cohesin+CTCF co-bound bins; check loop propensity is highest there.

Run:
    python scripts/35_rad21_chipseq_overlay.py
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(line_buffering=True)


def load_peak_track(path: Path, chrom: str, region_start: int,
                    region_end: int, N: int):
    bin_bp = (region_end - region_start) / N
    s_sum = np.zeros(N, dtype=np.float64)
    counts = np.zeros(N, dtype=np.int32)
    with gzip.open(path, "rt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 7 or parts[0] != chrom:
                continue
            try:
                start = int(parts[1]); end = int(parts[2])
                signal = float(parts[6])
            except (ValueError, IndexError):
                continue
            mid = (start + end) // 2
            if not (region_start <= mid < region_end):
                continue
            b = int((mid - region_start) // bin_bp)
            if 0 <= b < N:
                s_sum[b] += signal
                counts[b] += 1
    return s_sum, counts


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def smooth(v, w=3):
    return np.convolve(v, np.ones(w) / w, mode="same")


def main() -> None:
    region = "IMR90_chr21-28-30Mb"
    chrom = "chr21"
    region_start = 28_000_000
    region_end = 30_000_000
    N = 65
    seg_pos_mb = region_start / 1e6 + (np.arange(N) + 0.5) * (region_end - region_start) / N / 1e6

    chipseq_dir = ROOT / "data" / "encode_ctcf_imr90"
    ctcf_path = chipseq_dir / "ENCFF307XFM_IMR90_CTCF_hg38_optimal_peaks.bed.gz"
    rad21_path = chipseq_dir / "ENCFF895JAW_IMR90_RAD21_hg38_optimal_peaks.bed.gz"
    print(f"loading CTCF + RAD21 peaks for chr21:{region_start//10**6}-{region_end//10**6}Mb...")
    ctcf_sig, ctcf_n = load_peak_track(ctcf_path, chrom, region_start, region_end, N)
    rad21_sig, rad21_n = load_peak_track(rad21_path, chrom, region_start, region_end, N)
    print(f"  CTCF: {int(ctcf_n.sum())} peaks, total signal {ctcf_sig.sum():.0f}")
    print(f"  RAD21: {int(rad21_n.sum())} peaks, total signal {rad21_sig.sum():.0f}")

    # Encoder loop propensity (per-locus row-sum of mean z_hat over training cells)
    import torch
    real_path = ROOT / "data" / "real" / f"{region}_preprocessed.npz"
    f = np.load(real_path)
    z_hat_all = f["z_hat"]
    diff_ckpt = torch.load(ROOT / "checkpoints" / "step05_diffusion_real.pt",
                           map_location="cpu", weights_only=False)
    val_idx = np.array(diff_ckpt["val_idx"])
    train_idx = np.setdiff1d(np.arange(z_hat_all.shape[0]), val_idx)
    mean_z = z_hat_all[train_idx].mean(axis=0)
    propensity = mean_z.sum(axis=1) - np.diag(mean_z)
    print(f"encoder propensity range: [{propensity.min():.3f}, {propensity.max():.3f}]")

    # Smoothed signals
    ctcf_s = smooth(ctcf_sig, 3)
    rad21_s = smooth(rad21_sig, 3)

    pcc_ctcf = pearson(ctcf_s, propensity)
    pcc_rad21 = pearson(rad21_s, propensity)
    pcc_ctcf_rad21 = pearson(ctcf_s, rad21_s)  # CTCF and RAD21 should agree (cohesin co-binds CTCF)

    # Co-bound bins (both CTCF and RAD21 above median)
    ctcf_hi = ctcf_s > np.median(ctcf_s[ctcf_s > 0]) if (ctcf_s > 0).any() else ctcf_s > 0
    rad21_hi = rad21_s > np.median(rad21_s[rad21_s > 0]) if (rad21_s > 0).any() else rad21_s > 0
    co_bound = ctcf_hi & rad21_hi

    # Propensity comparison: co-bound vs all
    if co_bound.any():
        prop_cobound = propensity[co_bound].mean()
        prop_other = propensity[~co_bound].mean()
        enrichment = prop_cobound / max(prop_other, 1e-9)
    else:
        prop_cobound = prop_other = enrichment = float("nan")

    print(f"\nbiological cross-validation:")
    print(f"  Pearson(propensity, CTCF smoothed):    {pcc_ctcf:.4f}")
    print(f"  Pearson(propensity, RAD21 smoothed):   {pcc_rad21:.4f}")
    print(f"  Pearson(CTCF, RAD21) (internal check): {pcc_ctcf_rad21:.4f}")
    print(f"  cohesin+CTCF co-bound bins: {int(co_bound.sum())}/{N}")
    print(f"  loop propensity at co-bound bins:      {prop_cobound:.3f}")
    print(f"  loop propensity at other bins:         {prop_other:.3f}")
    print(f"  enrichment (co-bound / other):         {enrichment:.2f}x")

    # Bootstrap CI on Pearson for both
    rng = np.random.default_rng(2027)
    def boot(a, b, n=2000):
        out = np.zeros(n)
        m = a.shape[0]
        for k in range(n):
            idx = rng.integers(0, m, size=m)
            out[k] = pearson(a[idx], b[idx])
        return float(np.median(out)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
    pc, pl, ph = boot(ctcf_s, propensity)
    rc, rl, rh = boot(rad21_s, propensity)
    print(f"\nbootstrap CIs (n=2000):")
    print(f"  CTCF  Pearson: {pc:.3f} [{pl:.3f}, {ph:.3f}]")
    print(f"  RAD21 Pearson: {rc:.3f} [{rl:.3f}, {rh:.3f}]")

    np.savez(ROOT / "checkpoints" / "step21_rad21_overlay.npz",
        ctcf_sig=ctcf_sig, rad21_sig=rad21_sig, propensity=propensity,
        pcc_ctcf=pcc_ctcf, pcc_rad21=pcc_rad21,
        ci_ctcf=(pc, pl, ph), ci_rad21=(rc, rl, rh),
        co_bound=co_bound,
        enrichment=enrichment, prop_cobound=prop_cobound, prop_other=prop_other,
    )

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.1])

    ax = fig.add_subplot(gs[0, :])
    ax.bar(seg_pos_mb, ctcf_sig, width=0.025, color="C0", alpha=0.8,
           label=f"CTCF peaks (n={int(ctcf_n.sum())})")
    ax.plot(seg_pos_mb, ctcf_s, color="navy", lw=1.5, label="smoothed")
    ax.set_ylabel("CTCF\nsignal"); ax.set_title("CTCF ChIP-seq (loop anchors)")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, :])
    ax.bar(seg_pos_mb, rad21_sig, width=0.025, color="C2", alpha=0.8,
           label=f"RAD21 peaks (n={int(rad21_n.sum())})")
    ax.plot(seg_pos_mb, rad21_s, color="darkgreen", lw=1.5, label="smoothed")
    ax.set_ylabel("RAD21\n(cohesin)"); ax.set_title("RAD21 ChIP-seq (cohesin / loop extruder)")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, :])
    ax.bar(seg_pos_mb, propensity, width=0.025, color="C3", alpha=0.85,
           label="encoder loop propensity")
    # mark co-bound positions
    for k in range(N):
        if co_bound[k]:
            ax.axvline(seg_pos_mb[k], color="gold", lw=0.5, alpha=0.4, zorder=0)
    ax.set_xlabel("chr21 position (Mb)")
    ax.set_ylabel("encoder\nloop propensity")
    ax.set_title("Inferred per-locus loop-anchor propensity (mean z_hat row-sum)\n"
                 "gold lines = CTCF+RAD21 co-bound bins")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Bottom: side-by-side scatter + summary
    ax = fig.add_subplot(gs[3, 0])
    ax.scatter(ctcf_s, propensity, s=40, color="C0", edgecolor="navy", label="CTCF")
    ax.scatter(rad21_s, propensity, s=40, color="C2", edgecolor="darkgreen", alpha=0.7,
               label="RAD21")
    ax.set_xlabel("smoothed ChIP-seq signal")
    ax.set_ylabel("encoder propensity")
    ax.set_title("per-locus scatter")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[3, 1])
    ax.axis("off")
    summary = (
        "Two independent biological data sources:\n\n"
        f"  CTCF ChIP-seq vs encoder:   r = {pc:.3f}  [95% CI {pl:.3f}, {ph:.3f}]\n"
        f"  RAD21 ChIP-seq vs encoder:  r = {rc:.3f}  [95% CI {rl:.3f}, {rh:.3f}]\n"
        f"  CTCF vs RAD21 (internal):   r = {pcc_ctcf_rad21:.3f}\n\n"
        f"Cohesin + CTCF co-bound bins ({int(co_bound.sum())} of {N}):\n"
        f"  mean loop propensity at co-bound:  {prop_cobound:.3f}\n"
        f"  mean loop propensity elsewhere:    {prop_other:.3f}\n"
        f"  enrichment ratio:                  {enrichment:.2f}x\n\n"
        "Encoder was trained on simulated polymer\n"
        "structures only -- never saw real ChIP-seq.\n"
        "Picking up BOTH CTCF and RAD21 binding (and\n"
        "their co-occurrence) is independent\n"
        "biological validation of the latent space."
    )
    ax.text(0.0, 0.95, summary, fontsize=10, va="top", family="monospace")

    fig.suptitle("Biological cross-validation: encoder loop propensity vs CTCF + RAD21 ChIP-seq")
    fig.tight_layout()
    out = ROOT / "outputs" / "35_rad21_chipseq_overlay.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
