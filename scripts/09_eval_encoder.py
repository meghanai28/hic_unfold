"""Build step 4 evaluation: verify loop recovery on held-out simulated cells.

For each validation cell:
    1. Run the trained encoder to get per-pair loop logits.
    2. Compute AUROC over the upper triangle (ignores symmetry duplication).
    3. Compute top-k recall, where k = number of true loops in this cell:
       what fraction of the top-k predicted entries match a real loop?
    4. Compute precision/recall at threshold 0.5.

Renders an aggregate metrics figure and a grid of per-cell predictions.

Run:
    python scripts/09_eval_encoder.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hic_unfold.encoder import LoopEncoder  # noqa: E402
from hic_unfold.training import SimulatedDataset, make_positional_c  # noqa: E402


def upper_tri(M: np.ndarray) -> np.ndarray:
    N = M.shape[-1]
    iu = np.triu_indices(N, k=1)
    return M[..., iu[0], iu[1]]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """One-sample AUROC. Returns NaN if labels are single-class."""
    if labels.sum() == 0 or labels.sum() == labels.size:
        return float("nan")
    order = np.argsort(-scores)
    l_sorted = labels[order]
    tp = np.cumsum(l_sorted)
    fp = np.cumsum(1 - l_sorted)
    P = labels.sum(); F = (1 - labels).sum()
    tpr = np.concatenate([[0], tp / P])
    fpr = np.concatenate([[0], fp / F])
    return float(np.trapezoid(tpr, fpr))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = ROOT / "checkpoints" / "step04_encoder.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    N = int(ckpt["N"]); d_c = int(ckpt["d_c"])
    print(f"loaded {ckpt_path}: N={N}")

    net = LoopEncoder(N=N, d_c=d_c, d_pair=32, d_sep=16, d_h=96,
                      dilations=(1, 2, 4, 8, 1)).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    c_const = make_positional_c(N, d_c, device)

    data_path = ROOT / "data" / "sim" / f"step03_N{N}_M5000.npz"
    ds = SimulatedDataset(data_path)
    perm = np.random.default_rng(0).permutation(len(ds))
    val_idx = perm[:500].tolist()

    z_val = ds.z[val_idx]
    x_val = ds.x[val_idx]
    x_t = torch.from_numpy(x_val)[:, None].to(device)
    c_batch = c_const.expand(len(val_idx), -1, -1)
    with torch.no_grad():
        logits = net(x_t, c_batch).cpu().numpy()[:, 0]
    probs = 1.0 / (1.0 + np.exp(-logits))

    z_up = upper_tri(z_val)
    p_up = upper_tri(probs)

    aurocs = np.array([auroc(p_up[k], z_up[k]) for k in range(len(val_idx))])
    aurocs_valid = aurocs[~np.isnan(aurocs)]
    print(f"per-cell AUROC: median={np.median(aurocs_valid):.4f}, "
          f"mean={np.mean(aurocs_valid):.4f}, "
          f"#cells with at least one loop: {len(aurocs_valid)}/{len(val_idx)}")

    # Top-k recall
    topk_recall = []
    for k_idx in range(len(val_idx)):
        labels = z_up[k_idx]
        K = int(labels.sum())
        if K == 0:
            continue
        order = np.argsort(-p_up[k_idx])
        topk = order[:K]
        topk_recall.append(labels[topk].sum() / K)
    topk_recall = np.array(topk_recall)
    print(f"top-k recall (k = #true loops per cell): "
          f"median={np.median(topk_recall):.3f}, mean={np.mean(topk_recall):.3f}")

    # Precision/recall at threshold 0.5
    pred_bin = (p_up >= 0.5).astype(np.int8)
    tp = int((pred_bin & z_up.astype(np.int8)).sum())
    fp = int((pred_bin & (1 - z_up.astype(np.int8))).sum())
    fn = int(((1 - pred_bin) & z_up.astype(np.int8)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    print(f"threshold 0.5: precision={prec:.3f}, recall={rec:.3f}, "
          f"TP={tp}, FP={fp}, FN={fn}")

    # Baseline (random scores)
    rng = np.random.default_rng(7)
    baseline_aurocs = []
    baseline_topk = []
    for k_idx in range(len(val_idx)):
        labels = z_up[k_idx]
        K = int(labels.sum())
        if K == 0:
            continue
        rand_scores = rng.uniform(size=labels.shape)
        baseline_aurocs.append(auroc(rand_scores, labels))
        order = np.argsort(-rand_scores)[:K]
        baseline_topk.append(labels[order].sum() / K)
    baseline_aurocs = np.array(baseline_aurocs)
    baseline_topk = np.array(baseline_topk)
    print(f"random baseline: AUROC median={np.median(baseline_aurocs):.4f}, "
          f"top-k recall median={np.median(baseline_topk):.3f}")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    ax = axes[0]
    ax.hist(aurocs_valid, bins=np.linspace(0, 1, 30), alpha=0.7, color="C0",
            label=f"encoder (med={np.median(aurocs_valid):.3f})")
    ax.hist(baseline_aurocs, bins=np.linspace(0, 1, 30), alpha=0.5, color="gray",
            label=f"random (med={np.median(baseline_aurocs):.3f})")
    ax.set_xlabel("per-cell AUROC"); ax.set_ylabel("# val cells")
    ax.set_title("per-cell loop-recovery AUROC")
    ax.legend()

    ax = axes[1]
    bins = np.linspace(0, 1, 12)
    ax.hist(topk_recall, bins=bins, alpha=0.7, color="C0",
            label=f"encoder (med={np.median(topk_recall):.2f})")
    ax.hist(baseline_topk, bins=bins, alpha=0.5, color="gray",
            label=f"random (med={np.median(baseline_topk):.2f})")
    ax.set_xlabel("top-k recall (k = #true loops)"); ax.set_ylabel("# val cells")
    ax.set_title("top-k loop recall")
    ax.legend()

    ax = axes[2]
    ax.bar(["precision", "recall"], [prec, rec], color=["C0", "C1"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("score")
    ax.set_title(f"threshold 0.5\nTP={tp}, FP={fp}, FN={fn}")
    for i, v in enumerate([prec, rec]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")

    fig.tight_layout()
    out = out_dir / "09_encoder_metrics.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")

    # Qualitative grid: pick a mix of high/low AUROC cells
    valid_mask = ~np.isnan(aurocs)
    valid_indices = np.where(valid_mask)[0]
    order = np.argsort(-aurocs[valid_mask])
    high = valid_indices[order[: 3]]
    low = valid_indices[order[-3:]]
    picks = list(high) + list(low)

    fig2, axes2 = plt.subplots(3, len(picks), figsize=(2.7 * len(picks), 8))
    for col, i in enumerate(picks):
        axes2[0, col].imshow(z_val[i], origin="lower", cmap="gray_r", vmin=0, vmax=1)
        axes2[0, col].set_title(f"true z (cell {val_idx[i]})\nAUROC={aurocs[i]:.3f}")
        axes2[0, col].axis("off")
        axes2[1, col].imshow(probs[i], origin="lower", cmap="Reds", vmin=0, vmax=1)
        axes2[1, col].set_title("predicted P(loop)"); axes2[1, col].axis("off")
        axes2[2, col].imshow(ds.D[val_idx[i]], origin="lower", cmap="viridis")
        axes2[2, col].set_title("input distance D"); axes2[2, col].axis("off")

    fig2.suptitle("Encoder predictions: top three rows by AUROC (left), bottom three (right)")
    fig2.tight_layout()
    out2 = out_dir / "09_encoder_grid.png"
    fig2.savefig(out2, dpi=130)
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
