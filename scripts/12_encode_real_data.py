"""Pseudo-label real Bintu cells with the simulation-trained encoder.

Pipeline:
    1. Load the Bintu chr21:28-30Mb CSV and compute per-cell distance matrices.
    2. Filter cells with too many NaN; impute the rest with the population mean
       at each genomic separation.
    3. Standardize each cell's log1p(D) with real-data mu, sigma.
    4. Run the N=65 encoder on every cell to obtain per-pair loop probabilities.
    5. Save the preprocessed corpus (D_real, x_standardized, z_hat, mu, sigma)
       as a .npz ready for step-5 diffusion training.

Run:
    python scripts/12_encode_real_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hic_unfold.data import load_bintu_csv, preprocess_bintu  # noqa: E402
from hic_unfold.encoder import LoopEncoder  # noqa: E402
from hic_unfold.training import make_positional_c  # noqa: E402


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    region = "IMR90_chr21-28-30Mb"

    print(f"loading {region}...")
    ds = load_bintu_csv(ROOT / "data" / "raw_bintu2018" / f"{region}.csv")
    print(f"  {ds.num_cells} cells x {ds.num_segments} segments")

    print("preprocessing (filter + impute)...")
    real = preprocess_bintu(ds, min_valid_frac=0.85)
    N = real.D.shape[-1]
    M = real.D.shape[0]

    logD = np.log1p(real.D)
    mu = float(logD.mean()); sigma = float(logD.std())
    print(f"real-data standardisation: mu={mu:.4f}, sigma={sigma:.4f}")
    x = ((logD - mu) / max(sigma, 1e-8)).astype(np.float32)

    ckpt_path = ROOT / "checkpoints" / "step05_encoder_N65.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    assert int(ckpt["N"]) == N, f"encoder N={ckpt['N']} != real N={N}"
    d_c = int(ckpt["d_c"])
    enc = LoopEncoder(N=N, d_c=d_c, d_pair=32, d_sep=16, d_h=96,
                      dilations=(1, 2, 4, 8, 1)).to(device)
    enc.load_state_dict(ckpt["state_dict"])
    enc.eval()
    c_const = make_positional_c(N, d_c, device)

    z_hat_prob = np.zeros((M, N, N), dtype=np.float32)
    batch = 64
    print(f"encoding {M} real cells in batches of {batch}...")
    with torch.no_grad():
        for start in range(0, M, batch):
            end = min(start + batch, M)
            x_b = torch.from_numpy(x[start:end])[:, None].to(device)
            c_b = c_const.expand(end - start, -1, -1)
            logits = enc(x_b, c_b)
            z_hat_prob[start:end] = torch.sigmoid(logits)[:, 0].cpu().numpy()

    pair_mass = z_hat_prob.sum(axis=(1, 2)) / 2
    print(f"per-cell predicted loop count (sum of upper-triangle probs):")
    print(f"  median={np.median(pair_mass):.2f}, mean={pair_mass.mean():.2f}, "
          f"p90={np.percentile(pair_mass, 90):.2f}")
    print(f"per-pair median prob (across cells): {np.median(z_hat_prob):.4f}")
    print(f"per-pair max prob: {z_hat_prob.max():.4f}")

    out_path = ROOT / "data" / "real" / f"{region}_preprocessed.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        D=real.D, x=x, z_hat=z_hat_prob,
        valid_frac=real.valid_frac, cell_indices=real.cell_indices,
        pop_mean_by_sep=real.pop_mean_by_sep,
        mu=np.array(mu), sigma=np.array(sigma), N=np.array(N),
    )
    print(f"saved {out_path}")

    rng = np.random.default_rng(42)
    picks = rng.choice(M, size=4, replace=False)
    fig, axes = plt.subplots(3, len(picks), figsize=(3.2 * len(picks), 8))
    for col, k in enumerate(picks):
        axes[0, col].imshow(real.D[k], origin="lower", cmap="viridis")
        axes[0, col].set_title(f"real D (cell {real.cell_indices[k]})\n"
                               f"valid={100*real.valid_frac[k]:.0f}%")
        axes[0, col].axis("off")
        axes[1, col].imshow(x[k], origin="lower", cmap="coolwarm")
        axes[1, col].set_title("standardised log1p(D)"); axes[1, col].axis("off")
        axes[2, col].imshow(z_hat_prob[k], origin="lower", cmap="Reds", vmin=0, vmax=1)
        axes[2, col].set_title(f"encoder z_hat\nmass={pair_mass[k]:.2f}")
        axes[2, col].axis("off")
    fig.suptitle(f"Step-5: encoder pseudo-labels on real Bintu cells ({region})")
    fig.tight_layout()
    fig_path = ROOT / "outputs" / "12_encode_real_examples.png"
    fig.savefig(fig_path, dpi=130)
    print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
