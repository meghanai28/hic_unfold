"""Build step 5 validation: compare the real-trained prior's samples to
held-out real Bintu cells on single-cell statistics.

Statistics checked:
    1. Population contact map (mean of 500 nm soft contact across cells).
    2. P(s) scaling.
    3. Radius-of-gyration distribution (MDS embeds each D to R^3, then Rg).
    4. Pairwise distance distributions at a few characteristic separations.

Run:
    python scripts/14_eval_real.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hic_unfold.diffusion import Denoiser, ddim_sample, make_cosine_schedule  # noqa: E402
from hic_unfold.embedding import classical_mds  # noqa: E402
from hic_unfold.polymer.gaussian import soft_contact  # noqa: E402
from hic_unfold.training import make_positional_c  # noqa: E402


def radius_of_gyration(D: np.ndarray) -> float:
    X, _ = classical_mds(D, dim=3)
    com = X.mean(axis=0)
    return float(np.sqrt(((X - com) ** 2).sum(axis=-1).mean()))


def p_of_s(C_stack: np.ndarray) -> np.ndarray:
    N = C_stack.shape[-1]
    M = C_stack.mean(axis=0)
    return np.array([np.diag(M, k=s).mean() for s in range(1, N)])


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    region = "IMR90_chr21-28-30Mb"
    real_path = ROOT / "data" / "real" / f"{region}_preprocessed.npz"
    ckpt_path = ROOT / "checkpoints" / "step05_diffusion_real.pt"

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    N = int(ckpt["N"]); d_c = int(ckpt["d_c"]); T = int(ckpt["T"])
    mu = float(ckpt["mu"]); sigma = float(ckpt["sigma"])
    val_idx = ckpt["val_idx"]
    print(f"loaded {ckpt_path}: N={N}, mu={mu:.3f}, sigma={sigma:.3f}, val={len(val_idx)}")

    net = Denoiser(N=N, d_c=d_c, d_pair=32, d_sep=16, d_h=96, d_t=128,
                   dilations=(1, 2, 4, 8, 1)).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    alpha_bars = make_cosine_schedule(T=T, device=device)
    c_const = make_positional_c(N, d_c, device)

    f = np.load(real_path)
    D_real = f["D"]; z_hat = f["z_hat"]

    val_arr = np.array(val_idx)
    rng = np.random.default_rng(33)
    n_eval = min(128, len(val_arr))
    pick = rng.choice(val_arr, size=n_eval, replace=False)

    z_b = torch.from_numpy(z_hat[pick])[:, None].to(device)
    c_batch = c_const.expand(n_eval, -1, -1)
    batch = 32
    samples = []
    print(f"DDIM-sampling {n_eval} conformations...")
    with torch.no_grad():
        for s_start in range(0, n_eval, batch):
            s_end = min(s_start + batch, n_eval)
            x_s = ddim_sample(net, z_b[s_start:s_end], c_batch[:s_end - s_start],
                              alpha_bars, n_steps=100)
            samples.append(x_s.squeeze(1).cpu().numpy())
    x_samp = np.concatenate(samples, axis=0)
    D_samp = np.expm1(x_samp * sigma + mu)
    D_samp = np.maximum(D_samp, 0)
    for k in range(D_samp.shape[0]):
        D_samp[k] = 0.5 * (D_samp[k] + D_samp[k].T)
        np.fill_diagonal(D_samp[k], 0)
    print(f"sampled D: range [{D_samp.min():.1f}, {D_samp.max():.1f}] nm")

    D_real_val = D_real[pick]

    threshold = 500.0
    C_real = (D_real_val < threshold).astype(np.float32)
    C_samp = (D_samp < threshold).astype(np.float32)

    real_contact = C_real.mean(axis=0)
    samp_contact = C_samp.mean(axis=0)
    ps_real = p_of_s(C_real)
    ps_samp = p_of_s(C_samp)
    seps = np.arange(1, N)

    print("computing radius of gyration distributions...")
    Rg_real = np.array([radius_of_gyration(d) for d in D_real_val])
    Rg_samp = np.array([radius_of_gyration(d) for d in D_samp])
    print(f"Rg (nm): real median={np.median(Rg_real):.1f}, "
          f"samples median={np.median(Rg_samp):.1f}")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(real_contact, origin="lower", cmap="Reds", vmin=0, vmax=1)
    ax.set_title(f"real held-out contact map\n({n_eval} cells, d<{threshold:.0f}nm)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(samp_contact, origin="lower", cmap="Reds", vmin=0, vmax=1)
    ax.set_title("model samples contact map")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 2])
    ax.loglog(seps, ps_real, "o-", ms=3, color="black", label=f"real (n={n_eval})")
    ax.loglog(seps, ps_samp, "s-", ms=3, color="C3", label="model samples")
    ax.set_xlabel("genomic separation s"); ax.set_ylabel("P(contact | s)")
    ax.set_title("P(s) scaling")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    bins = np.linspace(min(Rg_real.min(), Rg_samp.min()),
                       max(Rg_real.max(), Rg_samp.max()), 30)
    ax.hist(Rg_real, bins=bins, alpha=0.6, color="black", label=f"real ({np.median(Rg_real):.0f})")
    ax.hist(Rg_samp, bins=bins, alpha=0.6, color="C3",
            label=f"samples ({np.median(Rg_samp):.0f})")
    ax.set_xlabel("radius of gyration Rg (nm)"); ax.set_ylabel("# cells")
    ax.set_title("Rg distribution (median in legend)")
    ax.legend()

    ax = fig.add_subplot(gs[1, 1])
    for sep, color in zip([5, 15, 30, 50], ["C0", "C1", "C2", "C3"]):
        d_r = np.concatenate([np.diag(d, k=sep) for d in D_real_val])
        d_s = np.concatenate([np.diag(d, k=sep) for d in D_samp])
        bins = np.linspace(0, max(d_r.max(), d_s.max()), 40)
        ax.hist(d_r, bins=bins, density=True, histtype="step", color=color,
                ls="-", lw=1.5, label=f"real s={sep}")
        ax.hist(d_s, bins=bins, density=True, histtype="step", color=color,
                ls="--", lw=1.5, label=f"sample s={sep}")
    ax.set_xlabel("pairwise distance (nm)"); ax.set_ylabel("density")
    ax.set_title("distance distributions by separation\n(solid=real, dashed=samples)")
    ax.legend(fontsize=7, ncol=2)

    ax = fig.add_subplot(gs[1, 2])
    diff = samp_contact - real_contact
    vmax = np.abs(diff).max()
    im = ax.imshow(diff, origin="lower", cmap="seismic", vmin=-vmax, vmax=vmax)
    ax.set_title("samples - real (residual contact)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"Step-5 validation: prior trained on real Bintu data ({region})")
    fig.tight_layout()
    out = out_dir / "14_eval_real_metrics.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")

    # Qualitative grid
    n_show = 4
    fig2, axes2 = plt.subplots(3, n_show, figsize=(3.2 * n_show, 8))
    for col in range(n_show):
        axes2[0, col].imshow(z_hat[pick[col]], origin="lower", cmap="Reds", vmin=0, vmax=1)
        axes2[0, col].set_title("z_hat (conditioning)"); axes2[0, col].axis("off")
        axes2[1, col].imshow(D_real_val[col], origin="lower", cmap="viridis")
        axes2[1, col].set_title("held-out real D"); axes2[1, col].axis("off")
        axes2[2, col].imshow(D_samp[col], origin="lower", cmap="viridis")
        axes2[2, col].set_title("model sample"); axes2[2, col].axis("off")
    fig2.tight_layout()
    out2 = out_dir / "14_eval_real_grid.png"
    fig2.savefig(out2, dpi=130)
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
