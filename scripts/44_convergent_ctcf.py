"""D3: Convergent CTCF motif orientation enrichment.

Loop extrusion theory predicts that paired CTCF anchors in cohesin-mediated
loops are in CONVERGENT orientation (--> ... <--). This is the strongest
mechanistic signature available from sequence alone.

Pipeline:
  1. Scan chr21:28-30Mb (hg38) with JASPAR CTCF PWM MA0139.1 on both strands.
  2. Identify strong motif sites (score in top quantile).
  3. Bin into N=65 30kb segments; record orientation(s) per bin.
  4. For top predicted loop pairs from the encoder, classify the anchor-pair
     orientation: CONVERGENT (+/-), DIVERGENT (-/+), TANDEM (+/+ or -/-).
  5. Compare fractions in top predicted loops vs distance-matched controls.

If high-probability predicted loops are enriched for convergent orientation,
the latent reflects bona fide loop-extrusion biology.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(line_buffering=True)


# JASPAR MA0139.1 CTCF PFM (counts at each of 19 positions)
CTCF_PFM = {
    "A": [87, 167, 281, 56, 8, 744, 40, 107, 851, 5, 333, 54, 12, 56, 104, 372, 82, 117, 402],
    "C": [291, 145, 49, 800, 903, 13, 528, 433, 11, 0, 3, 12, 0, 8, 733, 13, 482, 322, 181],
    "G": [76, 414, 449, 21, 0, 65, 334, 48, 32, 903, 566, 504, 890, 775, 5, 507, 307, 73, 266],
    "T": [459, 187, 134, 36, 2, 91, 11, 324, 18, 3, 9, 341, 8, 71, 67, 17, 37, 396, 59],
}
PSEUDO = 0.5  # pseudo-count for log-odds


def build_pwm() -> np.ndarray:
    """Return log-odds PWM (4, L) where rows are A,C,G,T."""
    L = len(CTCF_PFM["A"])
    pfm = np.zeros((4, L), dtype=np.float64)
    for k, nt in enumerate("ACGT"):
        pfm[k] = CTCF_PFM[nt]
    pfm = pfm + PSEUDO
    pfm = pfm / pfm.sum(axis=0, keepdims=True)
    bg = 0.25
    log_odds = np.log2(pfm / bg)
    return log_odds


def scan_sequence(seq: str, pwm: np.ndarray, threshold: float = 8.0):
    """Return list of (position, strand, score). Position is 0-based start of motif."""
    L = pwm.shape[1]
    # Encode sequence: A=0 C=1 G=2 T=3 N=ignore
    enc_map = {"A": 0, "C": 1, "G": 2, "T": 3, "a": 0, "c": 1, "g": 2, "t": 3}
    nseq = np.full(len(seq), -1, dtype=np.int8)
    for k, b in enumerate(seq):
        v = enc_map.get(b)
        if v is not None:
            nseq[k] = v
    out = []
    n = len(seq) - L + 1
    # Forward strand
    print(f"scanning forward strand ({n} positions)...")
    score_fwd = np.zeros(n, dtype=np.float64)
    for j in range(L):
        col = pwm[:, j]
        for k in range(n):
            v = nseq[k + j]
            if v < 0:
                score_fwd[k] = -np.inf
                break
            score_fwd[k] += col[v]
    # Reverse complement: complement nucleotide is 3-v (A<->T, C<->G), and reverse position order
    print("scanning reverse strand...")
    score_rev = np.zeros(n, dtype=np.float64)
    rev_pwm = pwm[::-1, ::-1]  # complement + reverse
    for j in range(L):
        col = rev_pwm[:, j]
        for k in range(n):
            v = nseq[k + j]
            if v < 0:
                score_rev[k] = -np.inf
                break
            score_rev[k] += col[v]
    for k in range(n):
        if score_fwd[k] > threshold:
            out.append((k, "+", float(score_fwd[k])))
        if score_rev[k] > threshold:
            out.append((k, "-", float(score_rev[k])))
    return out, score_fwd, score_rev


def main() -> None:
    fa_path = ROOT / "data" / "encode_ctcf_imr90" / "chr21_28_30Mb_hg38.fa"
    print(f"loading {fa_path}...")
    seq = open(fa_path).read().split("\n", 1)[1].replace("\n", "")
    print(f"  sequence length: {len(seq)}")
    print(f"  first 60: {seq[:60]}")

    pwm = build_pwm()
    print(f"PWM shape: {pwm.shape}, score range: [{pwm.min():.2f}, {pwm.max():.2f}]")
    # max possible score
    max_score = float(pwm.max(axis=0).sum())
    print(f"max possible PWM score: {max_score:.2f}")

    # Use a strict absolute threshold in bits. CTCF JASPAR motif consensus
    # scores ~17 bits; 12 bits keeps high-confidence hits only.
    thr = 12.0
    motifs, score_fwd, score_rev = scan_sequence(seq, pwm, threshold=thr)
    print(f"\nabsolute PWM threshold: {thr:.2f} bits (top hits only)")
    strong = motifs
    print(f"strong motif hits (score >= {thr:.2f}): {len(strong)}")
    plus = sum(1 for _, s, _ in strong if s == "+")
    minus = len(strong) - plus
    print(f"  on + strand: {plus}  on - strand: {minus}")

    # Bin to N=65 30kb segments
    N = 65
    region_start = 28_000_000
    region_end = 30_000_000
    bin_bp = (region_end - region_start) / N

    fwd_bin = np.zeros(N, dtype=np.int32)
    rev_bin = np.zeros(N, dtype=np.int32)
    for pos, strand, sc in strong:
        b = int((pos + 10) // bin_bp)  # use motif midpoint
        if 0 <= b < N:
            if strand == "+":
                fwd_bin[b] += 1
            else:
                rev_bin[b] += 1
    print(f"\nper-bin motif counts:")
    print(f"  bins with + motif: {int((fwd_bin > 0).sum())}")
    print(f"  bins with - motif: {int((rev_bin > 0).sum())}")
    print(f"  bins with both:    {int(((fwd_bin > 0) & (rev_bin > 0)).sum())}")

    # Load encoder predictions
    import torch
    f = np.load(ROOT / "data" / "real" / "IMR90_chr21-28-30Mb_preprocessed.npz")
    z_hat_all = f["z_hat"]
    diff_ckpt = torch.load(ROOT / "checkpoints" / "step05_diffusion_real.pt",
                           map_location="cpu", weights_only=False)
    val_idx = np.array(diff_ckpt["val_idx"])
    train_idx = np.setdiff1d(np.arange(z_hat_all.shape[0]), val_idx)
    mean_z = z_hat_all[train_idx].mean(axis=0)
    mean_z = 0.5 * (mean_z + mean_z.T)

    iu_i, iu_j = np.triu_indices(N, k=4)
    probs = mean_z[iu_i, iu_j]
    seps = iu_j - iu_i
    n_pairs = len(probs)
    print(f"\ntotal pairs (sep>=4): {n_pairs}")

    # Classify each pair's orientation
    def pair_class(i: int, j: int) -> str:
        # CONVERGENT requires + at i AND - at j  (i < j; arrows face inward)
        has_plus_i = fwd_bin[i] > 0
        has_minus_i = rev_bin[i] > 0
        has_plus_j = fwd_bin[j] > 0
        has_minus_j = rev_bin[j] > 0
        if (not (has_plus_i or has_minus_i)) or (not (has_plus_j or has_minus_j)):
            return "no_motif"
        if has_plus_i and has_minus_j:
            return "convergent"
        if has_minus_i and has_plus_j:
            return "divergent"
        return "tandem"  # +/+ or -/- (or mixed at one bin)

    # Classify all sep>=4 pairs
    pair_classes = np.array([pair_class(int(iu_i[k]), int(iu_j[k])) for k in range(n_pairs)])
    n_have_motif = int((pair_classes != "no_motif").sum())
    print(f"\npairs with motif at BOTH anchors: {n_have_motif}")
    counts = {c: int((pair_classes == c).sum()) for c in ["convergent", "divergent", "tandem", "no_motif"]}
    print(f"global counts: {counts}")
    if n_have_motif > 0:
        global_conv_frac = counts["convergent"] / n_have_motif
        print(f"global convergent fraction (among motif pairs): {global_conv_frac:.3f}")
    else:
        global_conv_frac = 0.0
        print("warning: no pairs have motifs at both anchors")

    # Top predicted loops: orientation enrichment
    print(f"\n{'top %':>6s} {'n':>5s} {'w/motif':>8s} {'convergent':>12s} {'divergent':>11s} "
          f"{'tandem':>8s} {'conv frac':>10s} {'enrich vs all':>14s}")
    print("-" * 90)
    rng = np.random.default_rng(2030)
    n_perm = 5000
    enrich_rows = []
    for top_pct in [1, 5, 10, 20]:
        top_n = max(1, int(n_pairs * top_pct / 100))
        top_idx = np.argsort(-probs)[:top_n]
        top_cls = pair_classes[top_idx]
        with_motif = int((top_cls != "no_motif").sum())
        c_conv = int((top_cls == "convergent").sum())
        c_div = int((top_cls == "divergent").sum())
        c_tan = int((top_cls == "tandem").sum())
        if with_motif > 0:
            conv_frac = c_conv / with_motif
            enrich = conv_frac / max(global_conv_frac, 1e-9)
        else:
            conv_frac = float("nan"); enrich = float("nan")
        # Permutation p: same #top_n random pairs at matched separations
        sep_to_pairs: dict[int, np.ndarray] = {}
        for s in np.unique(seps):
            sep_to_pairs[int(s)] = np.where(seps == s)[0]
        null_conv_fracs = np.zeros(n_perm)
        for p in range(n_perm):
            sampled = np.array([
                int(rng.choice(sep_to_pairs[int(seps[ti])]))
                for ti in top_idx
            ])
            sc = pair_classes[sampled]
            wm = (sc != "no_motif").sum()
            null_conv_fracs[p] = ((sc == "convergent").sum() / wm) if wm > 0 else 0.0
        if not np.isnan(conv_frac):
            emp_p = float((null_conv_fracs >= conv_frac).mean())
        else:
            emp_p = float("nan")
        enrich_rows.append((top_pct, top_n, with_motif, c_conv, c_div, c_tan,
                            conv_frac, enrich, emp_p))
        print(f"{top_pct:>5d}% {top_n:>5d} {with_motif:>8d} {c_conv:>12d} "
              f"{c_div:>11d} {c_tan:>8d} {conv_frac:>10.3f} {enrich:>12.2f}x  "
              f"p={emp_p:.3f}")

    np.savez(ROOT / "checkpoints" / "step30_convergent_ctcf.npz",
        enrich_rows=np.array(enrich_rows, dtype=object),
        fwd_bin=fwd_bin, rev_bin=rev_bin,
        global_conv_frac=global_conv_frac, counts=counts,
    )

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    seg_mb = region_start / 1e6 + (np.arange(N) + 0.5) * (region_end - region_start) / N / 1e6
    ax = axes[0, 0]
    ax.bar(seg_mb - 0.005, fwd_bin, width=0.012, color="C0", label="+ motif")
    ax.bar(seg_mb + 0.005, rev_bin, width=0.012, color="C3", label="- motif")
    ax.set_xlabel("chr21 position (Mb)"); ax.set_ylabel("PWM motif count")
    ax.set_title("CTCF JASPAR MA0139.1 motif counts per 30kb bin (+/- strand)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    top_pcts = [r[0] for r in enrich_rows]
    conv_fracs = [r[6] for r in enrich_rows]
    pvals = [r[8] for r in enrich_rows]
    bars = ax.bar([f"top {p}%" for p in top_pcts], conv_fracs, color="C3", alpha=0.85)
    ax.axhline(global_conv_frac, color="black", ls="--", lw=1,
               label=f"all pairs ({global_conv_frac:.3f})")
    for k, (cf, p) in enumerate(zip(conv_fracs, pvals)):
        if np.isfinite(cf):
            ax.text(k, cf + 0.01, f"{cf:.3f}\np={p:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("convergent fraction (of motif pairs)")
    ax.set_title("Convergent orientation in top predicted loop pairs\nvs all motif pairs")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    labels = ["convergent", "divergent", "tandem", "no_motif"]
    sizes = [counts[l] for l in labels]
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=["C3", "C2", "C1", "lightgray"])
    ax.set_title("Global orientation breakdown (all pairs sep>=4)")

    ax = axes[1, 1]
    ax.axis("off")
    summary = (
        "CONVERGENT CTCF ENRICHMENT (D3)\n\n"
        f"PWM threshold: 99.7 percentile of scores\n"
        f"Total strong motifs in 2 Mb region: {len(strong)}\n"
        f"  + strand: {plus},   - strand: {minus}\n\n"
        f"Pairs (sep>=4) total:               {n_pairs}\n"
        f"  with motif at both anchors:       {n_have_motif}\n"
        f"  convergent of these:              {counts['convergent']}\n"
        f"  divergent of these:               {counts['divergent']}\n"
        f"  tandem of these:                  {counts['tandem']}\n"
        f"  global convergent fraction:        {global_conv_frac:.3f}\n\n"
        f"Top predicted loops convergent fraction:\n"
        + "\n".join([f"  top {r[0]}%: conv={r[6]:.3f}  enrich={r[7]:.2f}x  p={r[8]:.3f}"
                     for r in enrich_rows if np.isfinite(r[6])])
        + "\n\n"
        "Convention: + at i and - at j (i<j) faces\n"
        "inward = canonical CTCF loop anchor orientation."
    )
    ax.text(0.0, 0.95, summary, fontsize=10, va="top", family="monospace")

    fig.suptitle("D3: Convergent CTCF motif orientation in top predicted loops")
    fig.tight_layout()
    out = out_dir / "44_convergent_ctcf.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
