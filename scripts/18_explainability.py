"""Build step 9 — mechanistic explainability (Section 6 of the spec).

Five analyses, each turning the architecture's structural interpretability into
a concrete artifact:

  (1) Loop annotation — each generated cell arrives with its z_hat. Render
      cells alongside their conditioning to demonstrate "the latent IS the
      explanation".

  (2) CTCF / loop knockout — pick a strong predicted loop anchor pair, zero
      it out in z_hat, re-sample. The contact-map difference is the loop's
      structural contribution. Falsifiable analogue of CTCF-deletion Hi-C.

  (3) Per-locus loop propensity — column-sum of mean z_hat tells us how often
      each genomic locus is a loop anchor. Biological readout, comparable to
      CTCF ChIP-seq.

  (4) Prior-vs-constraint attribution in loop space — compare mean z_hat under
      the uniform prior to mean z_hat re-weighted by maxent. Difference labels
      each loop as "expected from biology" or "imposed by this measured Hi-C".

  (5) Lambda surprise map — the maxent Lagrange multipliers form an N x N
      heatmap of where the data demanded a shift from the prior. Localises
      what is sample-specific.

Run:
    python scripts/18_explainability.py
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
from hic_unfold.training import make_positional_c  # noqa: E402


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    region = "IMR90_chr21-28-30Mb"

    real_path = ROOT / "data" / "real" / f"{region}_preprocessed.npz"
    diff_ckpt_path = ROOT / "checkpoints" / "step05_diffusion_real.pt"
    fwd_path = ROOT / "checkpoints" / "step06_forward_params.npz"
    maxent_path = ROOT / "checkpoints" / "step07_maxent.npz"

    f = np.load(real_path)
    D_real = f["D"]; z_hat_all = f["z_hat"]
    N = int(f["N"]); mu = float(f["mu"]); sigma = float(f["sigma"])
    fwd = np.load(fwd_path)
    d0 = float(fwd["d0"]); tau = max(float(fwd["tau"]), 80.0)

    diff_ckpt = torch.load(diff_ckpt_path, map_location=device, weights_only=False)
    val_idx = np.array(diff_ckpt["val_idx"])
    train_idx = np.setdiff1d(np.arange(D_real.shape[0]), val_idx)

    me = np.load(maxent_path)
    w_maxent = me["weights"]              # (M=2000,)
    lam = me["lam"]                       # (N, N) symmetric Lagrange multipliers
    sampled_z_idx = me["sampled_z_idx"]   # (2000,) indices into z_hat_all

    net = Denoiser(N=N, d_c=int(diff_ckpt["d_c"]), d_pair=32, d_sep=16, d_h=96,
                   d_t=128, dilations=(1, 2, 4, 8, 1)).to(device)
    net.load_state_dict(diff_ckpt["state_dict"]); net.eval()
    alpha_bars = make_cosine_schedule(T=int(diff_ckpt["T"]), device=device)
    c_const = make_positional_c(N, int(diff_ckpt["d_c"]), device)

    iu = np.triu_indices(N, k=1)

    # ---------- (2) CTCF / loop knockout intervention --------------------
    # Find a val cell with a strong, isolated loop in its z_hat — one anchor
    # pair with much higher prob than the others.
    print("identifying a candidate cell for the loop-knockout intervention...")
    best_idx, best_pair, best_strength = None, None, -1.0
    for c_idx in val_idx[:200]:
        z = z_hat_all[c_idx]
        ranked_vals = np.sort(z[iu])[::-1]
        if ranked_vals[0] < 0.3:
            continue
        margin = ranked_vals[0] / max(ranked_vals[5], 1e-6)
        if margin < 1.5:
            continue
        flat = np.argmax(z * np.triu(np.ones_like(z), k=4))  # avoid near-diagonal
        i, j = flat // N, flat % N
        if abs(i - j) < 4:
            continue
        score = float(z[i, j]) * margin
        if score > best_strength:
            best_strength = score
            best_idx = int(c_idx); best_pair = (int(i), int(j))
    if best_idx is None:
        # Fallback: pick the cell whose max anchor probability is largest.
        scores = z_hat_all[val_idx].reshape(len(val_idx), -1).max(axis=1)
        bi = int(np.argmax(scores))
        best_idx = int(val_idx[bi])
        flat = int(np.argmax(z_hat_all[best_idx]))
        best_pair = (flat // N, flat % N)
    i_anchor, j_anchor = best_pair
    print(f"  picked cell {best_idx}, anchor ({i_anchor}, {j_anchor}), "
          f"z_hat = {z_hat_all[best_idx, i_anchor, j_anchor]:.3f}")

    n_ko = 8  # samples each (baseline + intervention) for noise reduction
    z_base = z_hat_all[best_idx].copy()
    z_ko = z_base.copy()
    radius = 1
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            ii = i_anchor + di; jj = j_anchor + dj
            if 0 <= ii < N and 0 <= jj < N:
                z_ko[ii, jj] = 0.0
                z_ko[jj, ii] = 0.0

    z_base_t = torch.from_numpy(z_base).to(device).expand(n_ko, 1, N, N).contiguous()
    z_ko_t = torch.from_numpy(z_ko).to(device).expand(n_ko, 1, N, N).contiguous()
    c_batch = c_const.expand(n_ko, -1, -1)

    print(f"sampling {n_ko} baseline + {n_ko} knockout conformations via DDIM...")
    with torch.no_grad():
        x_base = ddim_sample(net, z_base_t, c_batch, alpha_bars, n_steps=100)
        x_ko = ddim_sample(net, z_ko_t, c_batch, alpha_bars, n_steps=100)
    D_base = np.expm1(x_base.squeeze(1).cpu().numpy() * sigma + mu)
    D_ko = np.expm1(x_ko.squeeze(1).cpu().numpy() * sigma + mu)
    D_base = np.maximum(D_base, 0); D_ko = np.maximum(D_ko, 0)
    for arr in (D_base, D_ko):
        for k in range(arr.shape[0]):
            arr[k] = 0.5 * (arr[k] + arr[k].T)
            np.fill_diagonal(arr[k], 0)

    base_mean = D_base.mean(axis=0)
    ko_mean = D_ko.mean(axis=0)
    delta_D = ko_mean - base_mean  # positive = farther apart after KO
    delta_anchor = float(delta_D[i_anchor, j_anchor])
    delta_global_mean = float(np.abs(delta_D).mean())
    print(f"  Delta D at anchor ({i_anchor},{j_anchor}): {delta_anchor:+.1f} nm")
    print(f"  mean |Delta D| globally: {delta_global_mean:.2f} nm")

    # ---------- (3) Per-locus loop propensity ----------------------------
    mean_z_train = z_hat_all[train_idx].mean(axis=0)
    propensity = mean_z_train.sum(axis=1) - np.diag(mean_z_train)
    print(f"per-locus loop propensity: top-5 loci = {np.argsort(-propensity)[:5].tolist()}")

    # ---------- (4) Prior vs constraint loop distribution ----------------
    z_prior_pool = z_hat_all[sampled_z_idx]          # (M=2000, N, N)
    mean_z_prior = z_prior_pool.mean(axis=0)
    mean_z_constrained = (w_maxent[:, None, None] * z_prior_pool).sum(axis=0)
    z_delta = mean_z_constrained - mean_z_prior
    # Most up-weighted loops
    delta_flat = z_delta[iu]
    top_up = np.argsort(-delta_flat)[:5]
    print(f"top-5 loops 'imposed by the measured bulk':")
    for k in top_up:
        i, j = iu[0][k], iu[1][k]
        print(f"  ({i:2d},{j:2d}): prior={mean_z_prior[i, j]:.4f}, "
              f"constrained={mean_z_constrained[i, j]:.4f}, "
              f"delta={z_delta[i, j]:+.4f}")

    # ---------- assemble figure ------------------------------------------
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1])

    # (1) Loop annotation — 3 example cells
    rng = np.random.default_rng(7)
    show_idx = list(rng.choice(val_idx[:200], size=3, replace=False))
    for col, c_idx in enumerate(show_idx):
        z = z_hat_all[c_idx]
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(z, origin="lower", cmap="Reds", vmin=0, vmax=1)
        ax.set_title(f"(1) cell {c_idx}\nz_hat (loop annotation)")
        ax.axis("off")
    # Spare cell column 3 used as legend / explanation
    ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    ax.text(0.05, 0.5,
            "(1) The latent IS the explanation:\n"
            "every generated cell arrives\nlabeled with which loops\n"
            "were active (red = high\n loop probability per pair).",
            fontsize=11, va="center")

    # (2) CTCF / loop knockout
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(base_mean, origin="lower", cmap="viridis")
    ax.scatter(j_anchor, i_anchor, s=80, facecolor="none", edgecolor="cyan", lw=1.5)
    ax.set_title(f"(2) baseline\n(loop at ({i_anchor},{j_anchor}) active)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(ko_mean, origin="lower", cmap="viridis", vmin=base_mean.min(), vmax=base_mean.max())
    ax.scatter(j_anchor, i_anchor, s=80, facecolor="none", edgecolor="cyan", lw=1.5)
    ax.set_title("(2) loop knockout\n(anchor zeroed in z_hat)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, 2])
    vmax = max(50, float(np.abs(delta_D).max()))
    im = ax.imshow(delta_D, origin="lower", cmap="seismic", vmin=-vmax, vmax=vmax)
    ax.scatter(j_anchor, i_anchor, s=80, facecolor="none", edgecolor="black", lw=1.5)
    ax.set_title("(2) KO - baseline (Delta D, nm)\nred = farther apart after KO")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    ax.text(0.0, 0.5,
            "(2) CTCF intervention:\n"
            f"zero out z_hat anchor\n"
            f"({i_anchor}, {j_anchor}) for cell {best_idx}\n"
            f"and resample. Distance at\n"
            f"that anchor changes by\n"
            f"{delta_anchor:+.0f} nm — a true\n"
            "mechanism perturbation,\n"
            "not a black-box probe.",
            fontsize=11, va="center")

    # (3) Per-locus loop propensity
    ax = fig.add_subplot(gs[2, 0:2])
    ax.plot(np.arange(N), propensity, color="C3", lw=1.5)
    ax.fill_between(np.arange(N), 0, propensity, alpha=0.3, color="C3")
    top = np.argsort(-propensity)[:5]
    for k in top:
        ax.axvline(k, color="black", lw=0.5, ls=":", alpha=0.5)
        ax.text(k, propensity[k] * 1.05, str(k), ha="center", fontsize=8)
    ax.set_xlabel("30kb segment index (chr21:28-30Mb)")
    ax.set_ylabel("per-locus loop propensity")
    ax.set_title("(3) Per-locus loop propensity\n(mean of encoder z_hat over training cells, row-sum)")
    ax.grid(True, alpha=0.3)

    # (4) Prior vs constraint
    ax = fig.add_subplot(gs[2, 2])
    im = ax.imshow(mean_z_prior, origin="lower", cmap="Reds",
                   vmin=0, vmax=max(mean_z_prior.max(), mean_z_constrained.max()))
    ax.set_title("(4) prior loop distribution\n(uniform-weight mean z_hat)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[2, 3])
    im = ax.imshow(mean_z_constrained, origin="lower", cmap="Reds",
                   vmin=0, vmax=max(mean_z_prior.max(), mean_z_constrained.max()))
    ax.set_title("(4) Hi-C-constrained\n(maxent-weighted)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[3, 0])
    vmax = float(np.abs(z_delta).max())
    im = ax.imshow(z_delta, origin="lower", cmap="seismic", vmin=-vmax, vmax=vmax)
    ax.set_title("(4) constrained - prior\nred = imposed by data, blue = suppressed")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # (5) Lambda surprise map
    ax = fig.add_subplot(gs[3, 1])
    vmax = float(np.abs(lam).max())
    im = ax.imshow(lam, origin="lower", cmap="seismic", vmin=-vmax, vmax=vmax)
    ax.set_title("(5) Lambda surprise map\nwhere data demanded a shift")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[3, 2:])
    ax.axis("off")
    ax.text(0.0, 0.95,
            "(3) The model's mean predicted-loop pattern peaks at specific loci — "
            "these are inferred loop anchors,\n      comparable to measured CTCF-ChIP "
            "peaks. (Top-5 loci: " + ", ".join(map(str, top.tolist())) + ".)\n\n"
            "(4) Comparing the prior (unweighted) to the maxent-constrained ensemble "
            "labels each loop as\n      'expected from biology' (zero change), "
            "'imposed by this measured Hi-C' (red), or\n      'suppressed by the data' "
            "(blue).\n\n"
            "(5) The Lagrange multipliers from step 7 form a per-pair surprise map: "
            "the magnitude is\n      proportional to how much the prior had to be "
            "pushed at that contact. Bright entries\n      mark sample-specific "
            "contacts not generic to chromatin biology.",
            fontsize=10, va="top")

    fig.suptitle("Step-9 mechanistic explainability (Section 6 of the architecture)")
    fig.tight_layout()
    out = out_dir / "18_explainability.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
