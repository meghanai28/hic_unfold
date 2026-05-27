"""D6 + B5: distance-matched loop-pair enrichment + per-pair BH-FDR.

D6.  Top predicted loops should connect CTCF+RAD21 co-bound anchors more often
     than random pairs MATCHED FOR GENOMIC SEPARATION (so the trivially-high
     near-diagonal contact rate doesn't drive the result).

B5.  For each (i, j) pair, test if the encoder's mean predicted probability is
     significantly above a separation-matched null. BH-FDR correct.
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


def load_peak_track(path: Path, chrom: str, start: int, end: int, N: int):
    bin_bp = (end - start) / N
    s = np.zeros(N, dtype=np.float64)
    with gzip.open(path, "rt") as f:
        for line in f:
            p = line.strip().split("\t")
            if len(p) < 7 or p[0] != chrom: continue
            try:
                a = int(p[1]); b = int(p[2]); sig = float(p[6])
            except (ValueError, IndexError):
                continue
            mid = (a + b) // 2
            if start <= mid < end:
                k = int((mid - start) // bin_bp)
                if 0 <= k < N: s[k] += sig
    return s


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    n = pvals.shape[0]
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(n) + 1)
    for i in range(n - 2, -1, -1):
        q[i] = min(q[i], q[i + 1])
    out = np.empty_like(q)
    out[order] = q
    return np.clip(out, 0, 1)


def main() -> None:
    N = 65; start = 28_000_000; end = 30_000_000
    chip_dir = ROOT / "data" / "encode_ctcf_imr90"
    ctcf = load_peak_track(chip_dir / "ENCFF307XFM_IMR90_CTCF_hg38_optimal_peaks.bed.gz",
                           "chr21", start, end, N)
    rad21 = load_peak_track(chip_dir / "ENCFF895JAW_IMR90_RAD21_hg38_optimal_peaks.bed.gz",
                            "chr21", start, end, N)
    print(f"CTCF non-zero bins: {int((ctcf > 0).sum())}, RAD21: {int((rad21 > 0).sum())}")

    # Co-bound: both above their median
    ctcf_thr = float(np.median(ctcf[ctcf > 0])) if (ctcf > 0).any() else 0
    rad21_thr = float(np.median(rad21[rad21 > 0])) if (rad21 > 0).any() else 0
    co_bound = (ctcf > ctcf_thr) & (rad21 > rad21_thr)
    print(f"CTCF+RAD21 co-bound bins: {int(co_bound.sum())}/{N}")

    import torch
    real_path = ROOT / "data" / "real" / "IMR90_chr21-28-30Mb_preprocessed.npz"
    f = np.load(real_path)
    z_hat_all = f["z_hat"]
    diff_ckpt = torch.load(ROOT / "checkpoints" / "step05_diffusion_real.pt",
                           map_location="cpu", weights_only=False)
    val_idx = np.array(diff_ckpt["val_idx"])
    train_idx = np.setdiff1d(np.arange(z_hat_all.shape[0]), val_idx)
    mean_z = z_hat_all[train_idx].mean(axis=0)
    # Symmetrize
    mean_z = 0.5 * (mean_z + mean_z.T)

    # Build all pairs (i, j) with j - i >= 4 (skip near-diagonal)
    iu_i, iu_j = np.triu_indices(N, k=4)
    probs = mean_z[iu_i, iu_j]
    seps = iu_j - iu_i
    n_pairs = len(probs)
    print(f"\ntotal pairs (sep >= 4): {n_pairs}")

    # =================== D6: top predicted loops vs distance-matched null ===================
    print("\n" + "=" * 80)
    print("D6: TOP PREDICTED LOOP PAIRS - distance-matched enrichment for")
    print("    connecting CTCF+RAD21 co-bound anchors")
    print("=" * 80)

    co_bound_pair = co_bound[iu_i] & co_bound[iu_j]
    co_bound_frac = float(co_bound_pair.mean())
    print(f"global rate of co-bound pairs (sep>=4): {co_bound_frac:.3%}")

    print(f"\n{'top X%':>8s} {'n_pairs':>8s} {'co_bound':>10s} "
          f"{'observed rate':>16s} {'matched null':>16s} {'fold':>6s} {'perm p':>10s}")
    print("-" * 80)

    rng = np.random.default_rng(2027)
    n_perm = 5000
    d6_rows = []
    for top_pct in [1, 5, 10, 20]:
        top_n = max(1, int(n_pairs * top_pct / 100))
        top_idx = np.argsort(-probs)[:top_n]
        observed = float(co_bound_pair[top_idx].mean())
        observed_count = int(co_bound_pair[top_idx].sum())

        # Build separation-matched null: for each top pair, sample a random pair
        # with the SAME genomic separation
        sep_to_pairs: dict[int, np.ndarray] = {}
        for s in np.unique(seps):
            sep_to_pairs[int(s)] = np.where(seps == s)[0]
        null_rates = np.zeros(n_perm)
        for p in range(n_perm):
            sampled = np.array([
                int(rng.choice(sep_to_pairs[int(seps[ti])]))
                for ti in top_idx
            ])
            null_rates[p] = co_bound_pair[sampled].mean()
        null_mean = float(null_rates.mean())
        fold = float(observed / max(null_mean, 1e-9))
        emp_p = float((null_rates >= observed).mean())
        d6_rows.append((top_pct, top_n, observed_count, observed, null_mean, fold, emp_p))
        print(f"{top_pct:>7d}%  {top_n:>8d} {observed_count:>10d} "
              f"{observed:>16.3%} {null_mean:>16.3%} {fold:>6.2f} {emp_p:>10.4f}")

    # =================== B5: per-pair significance (BH-FDR) ===================
    print("\n" + "=" * 80)
    print("B5: PER-PAIR ENCODER PREDICTION  Benjamini-Hochberg FDR")
    print("=" * 80)
    # For each pair, p = fraction of pairs at the same separation with >= the
    # observed prediction (one-sided)
    print(f"computing per-pair empirical p-values from separation-matched null...")
    p_vals = np.zeros(n_pairs)
    for k in range(n_pairs):
        s = int(seps[k])
        same_sep_probs = probs[seps == s]
        # one-sided: probability of seeing this value by chance at this separation
        p_vals[k] = float((same_sep_probs >= probs[k]).mean())
    q_vals = benjamini_hochberg(p_vals)

    for q_thr in [0.05, 0.10, 0.25]:
        n_sig = int((q_vals < q_thr).sum())
        # of significant, what fraction connect CTCF+RAD21 co-bound anchors?
        sig_mask = q_vals < q_thr
        if sig_mask.any():
            cob = float(co_bound_pair[sig_mask].mean())
        else:
            cob = float("nan")
        print(f"  q < {q_thr:.2f}:  {n_sig:>4d} pairs;  fraction co-bound = {cob:.3f}  "
              f"(vs global {co_bound_frac:.3f})")

    # Top-K significant pairs
    sig_idx = np.argsort(p_vals)[:15]
    print(f"\ntop-15 most-significant predicted loop pairs (lowest empirical p, sep>=4):")
    print(f"  {'pair':>10s}  {'sep':>4s}  {'prob':>8s}  {'p':>8s}  {'q (BH)':>8s}  "
          f"{'co-bound?':>10s}")
    for k in sig_idx:
        i, j = iu_i[k], iu_j[k]
        cb = "YES" if co_bound_pair[k] else "."
        print(f"  ({int(i):2d},{int(j):2d})    {int(seps[k]):>4d}   "
              f"{probs[k]:>8.4f}  {p_vals[k]:>8.4f}  {q_vals[k]:>8.4f}     {cb:>10s}")

    np.savez(ROOT / "checkpoints" / "step28_distance_matched.npz",
        d6_rows=np.array(d6_rows, dtype=object),
        p_vals=p_vals, q_vals=q_vals,
        co_bound=co_bound, co_bound_pair=co_bound_pair,
        iu_i=iu_i, iu_j=iu_j, probs=probs,
    )

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, 0])
    top_pcts = [r[0] for r in d6_rows]
    obs_rates = [r[3] for r in d6_rows]
    null_rates_ = [r[4] for r in d6_rows]
    folds = [r[5] for r in d6_rows]
    x = np.arange(len(top_pcts))
    w = 0.35
    ax.bar(x - w/2, obs_rates, w, color="C3", label="observed")
    ax.bar(x + w/2, null_rates_, w, color="gray", label="distance-matched null")
    ax.set_xticks(x); ax.set_xticklabels([f"top {p}%" for p in top_pcts])
    ax.set_ylabel("fraction connecting CTCF+RAD21 co-bound anchors")
    ax.set_title("D6: Top predicted loops vs distance-matched null")
    ax.axhline(float(co_bound_pair.mean()), color="black", ls="--", lw=0.7,
               label=f"all pairs (sep>=4) = {float(co_bound_pair.mean()):.3f}")
    for k, (f, p) in enumerate(zip(folds, [r[6] for r in d6_rows])):
        ax.text(k, max(obs_rates[k], null_rates_[k]) + 0.005, f"{f:.2f}x\np={p:.3f}",
                ha="center", fontsize=9)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(p_vals, bins=40, color="C0", alpha=0.85)
    ax.set_xlabel("per-pair empirical p (sep-matched null)")
    ax.set_ylabel("pair count"); ax.set_title("B5: per-pair p-value distribution")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    qs = sorted(q_vals)
    ax.plot(qs, np.arange(len(qs)) + 1, color="C2", lw=2)
    ax.set_xlabel("BH q-value"); ax.set_ylabel("# pairs with q <= x")
    ax.axvline(0.05, color="red", ls=":", lw=1, label="q=0.05")
    ax.axvline(0.10, color="orange", ls=":", lw=1, label="q=0.10")
    ax.set_title("B5: cumulative count vs FDR threshold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Significant pair map
    ax = fig.add_subplot(gs[1, 0])
    sig_map = np.zeros((N, N))
    for k in range(n_pairs):
        if q_vals[k] < 0.10:
            sig_map[iu_i[k], iu_j[k]] = 1
            sig_map[iu_j[k], iu_i[k]] = 1
    ax.imshow(sig_map, origin="lower", cmap="Reds")
    # Overlay CTCF+RAD21 co-bound bins
    for k in range(N):
        if co_bound[k]:
            ax.axvline(k, color="cyan", lw=0.5, alpha=0.5)
            ax.axhline(k, color="cyan", lw=0.5, alpha=0.5)
    ax.set_title("BH-significant pairs (q<0.10)\ncyan = CTCF+RAD21 co-bound bins")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 1])
    bins = np.linspace(0, max(probs), 30)
    ax.hist(probs[co_bound_pair], bins=bins, density=True, alpha=0.6, color="C3",
            label=f"co-bound pairs (n={int(co_bound_pair.sum())})")
    ax.hist(probs[~co_bound_pair], bins=bins, density=True, alpha=0.6, color="gray",
            label=f"other pairs (n={int((~co_bound_pair).sum())})")
    ax.set_xlabel("encoder mean z_hat"); ax.set_ylabel("density")
    ax.set_title("encoder prob distribution: co-bound vs other pairs")
    ax.set_yscale("log"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    sig_05 = int((q_vals < 0.05).sum())
    sig_10 = int((q_vals < 0.10).sum())
    summary = (
        "DISTANCE-MATCHED + PER-PAIR FDR\n\n"
        f"Total pairs (sep>=4):  {n_pairs}\n"
        f"BH q<0.05 significant: {sig_05}\n"
        f"BH q<0.10 significant: {sig_10}\n\n"
        "D6 top-K enrichment vs sep-matched null:\n"
        + "\n".join([f"  top {r[0]}%:  obs={r[3]:.3f}  null={r[4]:.3f}  "
                     f"fold={r[5]:.2f}x  p={r[6]:.3f}" for r in d6_rows])
        + "\n\n"
        "Each (i,j) tested against null built from\n"
        "the empirical distribution of predictions\n"
        "at the SAME genomic separation. This rules\n"
        "out 'trivially-high near-diagonal' artefacts."
    )
    ax.text(0.0, 0.95, summary, fontsize=10, va="top", family="monospace")

    fig.suptitle("D6 (distance-matched loop enrichment) + B5 (per-pair BH FDR)")
    fig.tight_layout()
    out = out_dir / "42_distance_matched_loops.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
