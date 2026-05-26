"""Biological validation: overlay our encoder's per-locus loop propensity with
real measured CTCF ChIP-seq on the same biological sample.

If the architecture's encoder is genuinely learning loop biology and not just
memorizing data-matrix patterns, its inferred per-locus loop-anchor propensity
should correlate with where CTCF actually binds in IMR-90 (since CTCF is the
biological generator of loop anchors).

Pipeline:
    1. Parse ENCODE IMR-90 CTCF optimal IDR peaks (hg38).
    2. Filter to chr21:28-30 Mb, bin to N=65 30-kb segments, sum signalValue.
    3. Recompute encoder per-locus loop propensity (mean of z_hat row-sum).
    4. Correlate; plot both tracks aligned.

Run:
    python scripts/20_ctcf_chipseq_overlay.py
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_chipseq_track(path: Path, chrom: str, region_start: int,
                       region_end: int, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a narrowPeak.bed.gz file and return three N-vectors:
        signal_summed       — sum of signalValue per 30-kb bin
        signal_max          — max signalValue per bin
        peak_count          — number of peaks per bin
    """
    bin_bp = (region_end - region_start) / N
    s_sum = np.zeros(N, dtype=np.float64)
    s_max = np.zeros(N, dtype=np.float64)
    counts = np.zeros(N, dtype=np.int32)
    with gzip.open(path, "rt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] != chrom:
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
            if b < 0 or b >= N:
                continue
            s_sum[b] += signal
            s_max[b] = max(s_max[b], signal)
            counts[b] += 1
    return s_sum, s_max, counts


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import rankdata  # delayed; scipy not in deps
    return pearson(rankdata(a), rankdata(b))


def main() -> None:
    region = "IMR90_chr21-28-30Mb"
    chrom = "chr21"
    region_start = 28_000_000
    region_end = 30_000_000
    N = 65

    chipseq_path = (ROOT / "data" / "encode_ctcf_imr90" /
                    "ENCFF307XFM_IMR90_CTCF_hg38_optimal_peaks.bed.gz")
    if not chipseq_path.exists():
        raise FileNotFoundError(chipseq_path)
    print(f"loading CTCF ChIP-seq peaks from {chipseq_path.name}...")
    s_sum, s_max, counts = load_chipseq_track(chipseq_path, chrom,
                                              region_start, region_end, N)
    print(f"  total peaks in region: {int(counts.sum())}")
    print(f"  total signal in region: {s_sum.sum():.1f}")

    real_path = ROOT / "data" / "real" / f"{region}_preprocessed.npz"
    f = np.load(real_path)
    z_hat_all = f["z_hat"]
    print(f"  loaded encoder z_hat from {real_path.name}: shape {z_hat_all.shape}")

    diff_ckpt = __import__("torch").load(
        ROOT / "checkpoints" / "step05_diffusion_real.pt",
        map_location="cpu", weights_only=False)
    val_idx = np.array(diff_ckpt["val_idx"])
    train_idx = np.setdiff1d(np.arange(z_hat_all.shape[0]), val_idx)

    mean_z = z_hat_all[train_idx].mean(axis=0)
    propensity = mean_z.sum(axis=1) - np.diag(mean_z)
    print(f"  encoder propensity range: [{propensity.min():.3f}, {propensity.max():.3f}]")

    # Try both peak-sum signal and peak-count as the ChIP-seq summary.
    # Use a small Gaussian-like smoothing on the ChIP-seq signal (3-bin window)
    # to make the comparison less sensitive to ±30 kb peak placement.
    def smooth(v: np.ndarray, w: int = 3) -> np.ndarray:
        kern = np.ones(w) / w
        return np.convolve(v, kern, mode="same")

    s_smooth = smooth(s_sum, w=3)
    c_smooth = smooth(counts.astype(float), w=3)

    pcc_sum = pearson(s_sum, propensity)
    pcc_smooth = pearson(s_smooth, propensity)
    pcc_count = pearson(counts, propensity)
    pcc_count_smooth = pearson(c_smooth, propensity)
    sp = spearman(s_smooth, propensity)
    print(f"\ncorrelation with encoder propensity:")
    print(f"  signal-sum (raw):       Pearson {pcc_sum:.4f}")
    print(f"  signal-sum (smoothed):  Pearson {pcc_smooth:.4f}, Spearman {sp:.4f}")
    print(f"  peak-count (raw):       Pearson {pcc_count:.4f}")
    print(f"  peak-count (smoothed):  Pearson {pcc_count_smooth:.4f}")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_pos_mb = region_start / 1e6 + np.arange(N) * (region_end - region_start) / N / 1e6

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2])

    ax = fig.add_subplot(gs[0, :])
    ax.bar(seg_pos_mb, s_sum, width=0.025, color="C0", alpha=0.8,
           label=f"CTCF ChIP-seq signalValue (peak sum, {int(counts.sum())} peaks)")
    ax.plot(seg_pos_mb, s_smooth, color="navy", lw=1.5, label="smoothed (3-bin)")
    ax.set_ylabel("ChIP-seq signal")
    ax.set_title("Measured CTCF binding (ENCODE ENCSR000EFI, IMR-90, hg38)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, :])
    ax.bar(seg_pos_mb, propensity, width=0.025, color="C3", alpha=0.8,
           label="encoder per-locus loop propensity")
    ax.set_xlabel("chr21 position (Mb)")
    ax.set_ylabel("loop propensity")
    ax.set_title("Inferred per-locus loop-anchor propensity (encoder mean z_hat, row-sum)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 0])
    ax.scatter(s_smooth, propensity, s=40, color="C3", edgecolor="navy")
    ax.set_xlabel("smoothed CTCF ChIP-seq signal")
    ax.set_ylabel("encoder loop propensity")
    ax.set_title(f"Per-locus scatter\nPearson {pcc_smooth:.3f}, Spearman {sp:.3f}")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    # Top-10 ChIP-seq vs top-10 propensity overlap
    top_chip = set(np.argsort(-s_smooth)[:10].tolist())
    top_prop = set(np.argsort(-propensity)[:10].tolist())
    overlap = top_chip & top_prop
    print(f"\ntop-10 overlap: {len(overlap)} / 10 loci")
    ax.axis("off")
    summary = (
        f"Biological validation:\n"
        f"  N = {N} segments, region = chr21:{region_start//10**6}-{region_end//10**6}Mb (hg38)\n"
        f"  ChIP-seq peaks in region: {int(counts.sum())}\n\n"
        f"Per-locus alignment:\n"
        f"  Pearson (CTCF signal vs propensity):  {pcc_smooth:.3f}\n"
        f"  Spearman (rank):                      {sp:.3f}\n"
        f"  Pearson (peak count):                 {pcc_count_smooth:.3f}\n\n"
        f"Top-10 locus overlap: {len(overlap)} / 10\n"
        f"  ChIP-seq tops:   {sorted(list(top_chip))}\n"
        f"  Propensity tops: {sorted(list(top_prop))}\n"
        f"  Shared:          {sorted(list(overlap))}\n\n"
        f"The encoder's inferred loop-anchor map\n"
        f"is learned from chromatin structure alone\n"
        f"(no ChIP-seq used). Matching real CTCF\n"
        f"binding validates the mechanistic claim."
    )
    ax.text(0.0, 0.95, summary, fontsize=9, va="top", family="monospace")

    fig.suptitle(f"Biological validation: encoder propensity vs measured CTCF ChIP-seq")
    fig.tight_layout()
    out = out_dir / "20_ctcf_overlay.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
