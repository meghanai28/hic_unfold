"""TAD boundaries via insulation score on the deconvolved ensemble.

Standard insulation-score TAD calling: at each locus, the diamond-shaped
window above the diagonal is summed; local minima are TAD boundaries.

Compares boundaries called from:
    - Bintu measured (held-out 388 cells), the ground truth
    - Step-8 guided deconvolution
    - Step-10 guided from real Hi-C
    - HIPPS-DIMES (polymer + maxent, step 14)

If the deconvolved ensemble has the same TAD organization as the imaging truth,
the boundary positions should overlap heavily.

Run:
    python scripts/32_tad_boundaries.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(line_buffering=True)


def insulation_score(C: np.ndarray, w: int = 5) -> np.ndarray:
    """For each locus i, sum the wxw diamond of contacts immediately upstream/
    downstream of i. Local minima = TAD boundaries."""
    N = C.shape[0]
    out = np.full(N, np.nan, dtype=np.float64)
    for i in range(w, N - w):
        block = C[i - w:i, i + 1:i + w + 1]
        out[i] = block.mean()
    # Min-max normalise for plotting
    m = np.nanmin(out); M = np.nanmax(out)
    if M > m:
        out_norm = (out - m) / (M - m)
    else:
        out_norm = out
    return out_norm


def find_local_minima(s: np.ndarray, min_distance: int = 3) -> list[int]:
    """Indices of local minima in insulation score (TAD boundaries)."""
    N = s.shape[0]
    out: list[int] = []
    for i in range(1, N - 1):
        if not np.isfinite(s[i]):
            continue
        left = s[i - 1] if np.isfinite(s[i - 1]) else np.inf
        right = s[i + 1] if np.isfinite(s[i + 1]) else np.inf
        if s[i] < left and s[i] < right:
            if not out or i - out[-1] >= min_distance:
                out.append(i)
    return out


def boundary_overlap(b1: list[int], b2: list[int], tol: int = 2) -> tuple[int, int, int]:
    """How many boundaries in b1 have a match in b2 within ±tol bins."""
    matched_1 = 0
    for b in b1:
        if any(abs(b - bb) <= tol for bb in b2):
            matched_1 += 1
    matched_2 = 0
    for b in b2:
        if any(abs(b - bb) <= tol for bb in b1):
            matched_2 += 1
    return matched_1, matched_2, max(len(b1), len(b2))


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
    D_real_val = D_real[val_idx]
    H_real = (D_real_val < hard_thr).astype(np.float32).mean(axis=0)

    gd8 = np.load(ROOT / "checkpoints" / "step08_guided.npz")
    H_g8 = (gd8["D_samples"] < hard_thr).astype(np.float32).mean(axis=0)

    g10 = np.load(ROOT / "checkpoints" / "step10_realhic.npz")
    H_g10 = (g10["D_samples"] < hard_thr).astype(np.float32).mean(axis=0)

    h14 = np.load(ROOT / "checkpoints" / "step14_hipps_dimes.npz")
    H_hipps = h14["H_reweighted"]

    series = {
        "Real Bintu (truth)": H_real,
        "Step-8 guided": H_g8,
        "Step-10 from Hi-C": H_g10,
        "HIPPS-DIMES": H_hipps,
    }

    print(f"computing insulation scores (window=5)...")
    insulation = {name: insulation_score(H, w=5) for name, H in series.items()}
    boundaries = {name: find_local_minima(s, min_distance=3) for name, s in insulation.items()}

    print(f"\nTAD boundaries found per condition:")
    for name, b in boundaries.items():
        print(f"  {name:<22s} {len(b)} boundaries at positions {b}")

    print(f"\nboundary overlap with real (truth), tolerance ±2 bins:")
    truth_bs = boundaries["Real Bintu (truth)"]
    for name, b in boundaries.items():
        if name == "Real Bintu (truth)":
            continue
        m1, m2, total = boundary_overlap(truth_bs, b, tol=2)
        precision = m2 / max(len(b), 1)
        recall = m1 / max(len(truth_bs), 1)
        print(f"  {name:<22s} recall={recall:.2f} ({m1}/{len(truth_bs)})  "
              f"precision={precision:.2f} ({m2}/{len(b)})")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(len(series) + 1, 1, height_ratios=[1.6, 1, 1, 1, 1])

    N = H_real.shape[0]
    x = np.arange(N)

    # Top: heatmap stack
    ax = fig.add_subplot(gs[0, 0])
    vmax = max(H.max() for H in series.values())
    for i, (name, H) in enumerate(series.items()):
        # Overlay heatmaps as colour-coded line plots is messy; instead show contact at fixed
        # separation = 5 across loci (a 1D proxy for "is this region in a TAD?")
        contact_at_s5 = np.diag(H, k=5)
        pad = np.full(N, np.nan); pad[:len(contact_at_s5)] = contact_at_s5
        ax.plot(x, pad, lw=1.5, label=name, alpha=0.85)
    ax.set_xlabel("30kb segment along chr21:28-30Mb")
    ax.set_ylabel("contact at s=5 (proxy for local compaction)")
    ax.set_title("local contact density along the region")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # One insulation track per condition
    colors = ["black", "C0", "C2", "C3"]
    for i, (name, s) in enumerate(insulation.items()):
        ax = fig.add_subplot(gs[i + 1, 0])
        ax.plot(x, s, color=colors[i], lw=1.5)
        ax.fill_between(x, 0, s, alpha=0.25, color=colors[i])
        for b in boundaries[name]:
            ax.axvline(b, color="red", ls=":", lw=1, alpha=0.7)
        ax.set_xlabel("segment")
        ax.set_ylabel("insulation\n(norm.)")
        ax.set_title(f"{name}  -  {len(boundaries[name])} boundaries", fontsize=10)
        ax.set_xlim(0, N - 1)
        ax.grid(True, alpha=0.3)

    fig.suptitle("TAD boundary calls (insulation score, w=5; minima = boundaries)")
    fig.tight_layout()
    out = out_dir / "32_tad_boundaries.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")

    np.savez_compressed(ROOT / "checkpoints" / "step20_tad.npz",
        insulation=np.array(list(insulation.values())),
        names=np.array(list(insulation.keys()), dtype=object),
        boundaries=np.array(list(boundaries.values()), dtype=object),
    )


if __name__ == "__main__":
    main()
