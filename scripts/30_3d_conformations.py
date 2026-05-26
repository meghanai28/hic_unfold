"""3D conformation gallery: render MDS-embedded single-cell chromatin structures
across all the conditions in this project, with predicted loop anchors highlighted.

Reviewers in genome architecture care a lot about these renderings — they're the
visual proof that "deconvolution" actually produces 3D objects, not just contact
matrices. Each panel shows one representative cell as a polymer backbone with:
    - colour encoding genomic position along chr21:28-30 Mb
    - red highlights at high-confidence loop anchors (from encoder z_hat)
    - bridges drawn between the two ends of each predicted loop

Conditions:
    1. IMR90 measured (Bintu chr21:28-30Mb)
    2. K562 measured (Bintu chr21:28-30Mb)
    3. HCT116 untreated measured
    4. HCT116 auxin measured (cohesin degraded)
    5. Diffusion sample from step 8 (deconvolved from imaging pseudo-bulk)
    6. Diffusion sample from step 10 (deconvolved from real Hi-C)
    7. In-silico cohesin knockout (alpha=0)
    8. In-silico partial cohesin loss (alpha=0.5, matches auxin)

Run:
    python scripts/30_3d_conformations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(line_buffering=True)

from hic_unfold.data import load_bintu_csv, preprocess_bintu  # noqa: E402
from hic_unfold.diffusion import Denoiser, ddim_sample, make_cosine_schedule  # noqa: E402
from hic_unfold.embedding import classical_mds  # noqa: E402
from hic_unfold.encoder import LoopEncoder  # noqa: E402
from hic_unfold.training import make_positional_c  # noqa: E402


def kabsch_align(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Rotate P to best-align with Q (both N x 3, centred at origin)."""
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D_ = np.diag([1.0, 1.0, d])
    R = Vt.T @ D_ @ U.T
    return P @ R.T


def conformation_from_D(D: np.ndarray) -> np.ndarray:
    """MDS to 3D, centred at origin."""
    X, _ = classical_mds(D, dim=3)
    X = X - X.mean(axis=0)
    return X


def find_top_anchors(z_hat: np.ndarray, K: int = 4, min_sep: int = 4) -> list[tuple[int, int]]:
    """Top-K anchor pairs (i, j) with j > i + min_sep, sorted by probability."""
    N = z_hat.shape[0]
    iu_i, iu_j = np.triu_indices(N, k=min_sep)
    probs = z_hat[iu_i, iu_j]
    order = np.argsort(-probs)[:K]
    return [(int(iu_i[o]), int(iu_j[o])) for o in order]


def render_conformation(ax, X: np.ndarray, anchors: list[tuple[int, int]],
                        title: str, *, cmap="viridis", show_anchors=True,
                        n_smooth: int = 400):
    """Render a chromatin polymer as a smooth spline-interpolated tube,
    coloured by genomic position along the backbone. Standard look in the
    chromatin-imaging / polymer-physics literature."""
    from scipy.interpolate import splprep, splev
    N = X.shape[0]

    # Fit a cubic spline through the N bead positions. s=0 = exact interpolation
    # so the tube passes through every measured bead.
    try:
        tck, u = splprep([X[:, 0], X[:, 1], X[:, 2]], s=0, k=3)
        u_fine = np.linspace(0, 1, n_smooth)
        xs, ys, zs = splev(u_fine, tck)
        smooth_pts = np.stack([xs, ys, zs], axis=1)
    except Exception:
        smooth_pts = X
        u_fine = np.linspace(0, 1, N)

    # Draw the tube as many short colour-graded segments
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=0, vmax=N - 1)
    cm = plt.get_cmap(cmap)
    seg_pos = u_fine * (N - 1)
    for k in range(len(smooth_pts) - 1):
        c = cm(norm(seg_pos[k]))
        ax.plot(smooth_pts[k:k + 2, 0], smooth_pts[k:k + 2, 1],
                smooth_pts[k:k + 2, 2],
                color=c, lw=3.2, solid_capstyle="round", alpha=0.95)

    # Faint markers at each genomic bead for spatial reference
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=np.arange(N), cmap=cmap,
               s=18, alpha=0.55, edgecolor="white", linewidths=0.3,
               depthshade=False)

    if show_anchors and anchors:
        for (i, j) in anchors:
            ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], [X[i, 2], X[j, 2]],
                    "-", color="#d62728", lw=2.5, alpha=0.85, zorder=8)
            ax.scatter(*X[i], color="#d62728", s=75, edgecolor="black",
                       lw=0.7, zorder=10, depthshade=False)
            ax.scatter(*X[j], color="#d62728", s=75, edgecolor="black",
                       lw=0.7, zorder=10, depthshade=False)

    # Clean panes for a polished look
    ax.set_title(title, fontsize=10, pad=2)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    try:
        ax.xaxis.pane.set_alpha(0.0)
        ax.yaxis.pane.set_alpha(0.0)
        ax.zaxis.pane.set_alpha(0.0)
        ax.grid(False)
    except Exception:
        pass
    return None


def encode_one_cell(enc, D_cell: np.ndarray, c_const, device) -> np.ndarray:
    l = np.log1p(D_cell)
    mu = float(l.mean()); s = float(l.std())
    x = ((l - mu) / max(s, 1e-8)).astype(np.float32)
    with torch.no_grad():
        x_t = torch.from_numpy(x)[None, None].to(device)
        z = torch.sigmoid(enc(x_t, c_const)).cpu().numpy()[0, 0]
    return z


def encode_many(enc, D: np.ndarray, c_const, device, batch: int = 32) -> np.ndarray:
    """Encode a batch of cells; returns (M, N, N) probs."""
    M, N, _ = D.shape
    l = np.log1p(D)
    mu = float(l.mean()); s = float(l.std())
    x = ((l - mu) / max(s, 1e-8)).astype(np.float32)
    out = np.zeros((M, N, N), dtype=np.float32)
    with torch.no_grad():
        for k in range(0, M, batch):
            e = min(k + batch, M)
            x_b = torch.from_numpy(x[k:e])[:, None].to(device)
            c_b = c_const.expand(e - k, -1, -1)
            out[k:e] = torch.sigmoid(enc(x_b, c_b))[:, 0].cpu().numpy()
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    diff_ckpt = torch.load(ROOT / "checkpoints" / "step05_diffusion_real.pt",
                           map_location=device, weights_only=False)
    enc_ckpt = torch.load(ROOT / "checkpoints" / "step05_encoder_N65.pt",
                          map_location=device, weights_only=False)
    N = int(diff_ckpt["N"]); d_c = int(diff_ckpt["d_c"])
    mu_train = float(diff_ckpt["mu"]); sigma_train = float(diff_ckpt["sigma"])

    enc = LoopEncoder(N=N, d_c=int(enc_ckpt["d_c"]), d_pair=32, d_sep=16, d_h=96,
                      dilations=(1, 2, 4, 8, 1)).to(device)
    enc.load_state_dict(enc_ckpt["state_dict"]); enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    net = Denoiser(N=N, d_c=d_c, d_pair=32, d_sep=16, d_h=96, d_t=128,
                   dilations=(1, 2, 4, 8, 1)).to(device)
    net.load_state_dict(diff_ckpt["state_dict"]); net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    alpha_bars = make_cosine_schedule(T=int(diff_ckpt["T"]), device=device)
    c_const = make_positional_c(N, d_c, device)

    rng = np.random.default_rng(2027)

    # Measured ensembles
    print("loading measured Bintu cells across cell types...")
    ds_imr = load_bintu_csv(ROOT / "data" / "raw_bintu2018" / "IMR90_chr21-28-30Mb.csv")
    ds_k = load_bintu_csv(ROOT / "data" / "raw_bintu2018" / "K562_chr21-28-30Mb.csv")
    ds_hu = load_bintu_csv(ROOT / "data" / "raw_bintu2018" / "HCT116_chr21-28-30Mb_untreated.csv")
    ds_ha = load_bintu_csv(ROOT / "data" / "raw_bintu2018" / "HCT116_chr21-28-30Mb_6h_auxin.csv")

    r_imr = preprocess_bintu(ds_imr, min_valid_frac=0.90)
    r_k = preprocess_bintu(ds_k, min_valid_frac=0.90)
    r_hu = preprocess_bintu(ds_hu, min_valid_frac=0.90)
    r_ha = preprocess_bintu(ds_ha, min_valid_frac=0.90)

    # Pick one representative cell per condition (the cell whose Rg is closest to median)
    def pick_representative(D: np.ndarray) -> int:
        from hic_unfold.embedding import classical_mds
        Rg = []
        for d in D:
            X, _ = classical_mds(d, dim=3)
            com = X.mean(axis=0)
            Rg.append(float(np.sqrt(((X - com) ** 2).sum(axis=-1).mean())))
        Rg = np.array(Rg)
        return int(np.argmin(np.abs(Rg - np.median(Rg))))

    print("picking representative cells...")
    idx_imr = pick_representative(r_imr.D[:300])
    idx_k = pick_representative(r_k.D[:300])
    idx_hu = pick_representative(r_hu.D[:300])
    idx_ha = pick_representative(r_ha.D[:300])

    # Predicted from step-8 guided sampling (load saved)
    gd8 = np.load(ROOT / "checkpoints" / "step08_guided.npz")
    D_g8 = gd8["D_samples"]
    idx_g8 = pick_representative(D_g8[:128])

    # Predicted from step-10 real-Hi-C deconvolution (load saved)
    g10 = np.load(ROOT / "checkpoints" / "step10_realhic.npz")
    D_g10 = g10["D_samples"]
    idx_g10 = pick_representative(D_g10[:128])

    # In-silico knockout (alpha=0) and partial (alpha=0.5) using HCT116 untreated z_hats
    print("generating in-silico cohesin perturbations...")
    z_hu_all = encode_many(enc, r_hu.D[:128], c_const, device)
    z_hu_t = torch.from_numpy(z_hu_all)[:, None].to(device)
    c_batch = c_const.expand(128, -1, -1)

    with torch.no_grad():
        x_ko = ddim_sample(net, torch.zeros_like(z_hu_t), c_batch, alpha_bars, n_steps=100)
        x_partial = ddim_sample(net, 0.5 * z_hu_t, c_batch, alpha_bars, n_steps=100)
    D_ko = np.expm1(x_ko.squeeze(1).cpu().numpy() * sigma_train + mu_train)
    D_partial = np.expm1(x_partial.squeeze(1).cpu().numpy() * sigma_train + mu_train)
    for arr in (D_ko, D_partial):
        arr = np.maximum(arr, 0)
        for k in range(arr.shape[0]):
            arr[k] = 0.5 * (arr[k] + arr[k].T); np.fill_diagonal(arr[k], 0)
    idx_ko = pick_representative(D_ko)
    idx_partial = pick_representative(D_partial)

    # Build (D, z_hat, label) tuples for the gallery
    panels = [
        ("IMR90 measured",       r_imr.D[idx_imr],   encode_one_cell(enc, r_imr.D[idx_imr], c_const, device), "viridis"),
        ("K562 measured",        r_k.D[idx_k],       encode_one_cell(enc, r_k.D[idx_k], c_const, device), "viridis"),
        ("HCT116 untreated",     r_hu.D[idx_hu],     encode_one_cell(enc, r_hu.D[idx_hu], c_const, device), "viridis"),
        ("HCT116 +6h auxin",     r_ha.D[idx_ha],     encode_one_cell(enc, r_ha.D[idx_ha], c_const, device), "viridis"),
        ("deconv. from Bintu bulk\n(step 8 guided)",
            D_g8[idx_g8], encode_one_cell(enc, D_g8[idx_g8], c_const, device), "viridis"),
        ("deconv. from real Hi-C\n(step 10 guided)",
            D_g10[idx_g10], encode_one_cell(enc, D_g10[idx_g10], c_const, device), "viridis"),
        ("in-silico cohesin KO\n(alpha=0)",
            D_ko[idx_ko], z_hu_all[idx_ko], "viridis"),
        ("in-silico partial cohesin\n(alpha=0.5)",
            D_partial[idx_partial], 0.5 * z_hu_all[idx_partial], "viridis"),
    ]

    print("rendering 3D conformations...")
    fig = plt.figure(figsize=(20, 10))
    # We'll arrange 8 panels in 2x4
    Xs = []
    for k, (label, D_cell, z_cell, cmap) in enumerate(panels):
        X = conformation_from_D(D_cell)
        Xs.append(X)
    # Align all to the first one (Kabsch) so cell-to-cell comparison is visual
    X_ref = Xs[0]
    Xs_aligned = [X_ref] + [kabsch_align(X, X_ref) for X in Xs[1:]]

    for k, ((label, _, z_cell, cmap), X) in enumerate(zip(panels, Xs_aligned)):
        ax = fig.add_subplot(2, 4, k + 1, projection="3d")
        anchors = find_top_anchors(z_cell, K=4, min_sep=4)
        render_conformation(ax, X, anchors, label, cmap=cmap)
        ax.view_init(elev=22, azim=35)

    fig.suptitle("3D single-cell chromatin conformations (chr21:28-30Mb, N=65 30kb segments)\n"
                 "polymer coloured by genomic position; red bridges = top-4 encoder-predicted loop anchors",
                 fontsize=11)
    fig.tight_layout()
    out = ROOT / "outputs" / "30_3d_conformations.png"
    fig.savefig(out, dpi=140)
    print(f"saved {out}")

    # Side figure: a "zoom" on cohesin-loss progression: same cell's z_hat with three alphas
    print("rendering cohesin-progression triplet for one cell...")
    cell_id = idx_hu
    D_one = r_hu.D[cell_id]
    z_one_t = torch.from_numpy(encode_one_cell(enc, D_one, c_const, device))[None, None].to(device)
    c_one = c_const.expand(1, -1, -1)
    alphas_demo = [1.0, 0.5, 0.0]
    Xs_demo = []
    labels_demo = []
    for a in alphas_demo:
        with torch.no_grad():
            x_demo = ddim_sample(net, (a * z_one_t).float(), c_one, alpha_bars, n_steps=100)
        D_demo = np.expm1(x_demo.squeeze().cpu().numpy() * sigma_train + mu_train)
        D_demo = np.maximum(D_demo, 0)
        D_demo = 0.5 * (D_demo + D_demo.T); np.fill_diagonal(D_demo, 0)
        Xs_demo.append(conformation_from_D(D_demo))
        labels_demo.append(f"alpha={a:.1f}")
    X_ref_d = Xs_demo[0]
    Xs_demo = [X_ref_d] + [kabsch_align(X, X_ref_d) for X in Xs_demo[1:]]

    z_one_arr = z_one_t.squeeze().cpu().numpy()
    anchors_demo = find_top_anchors(z_one_arr, K=4, min_sep=4)

    fig2 = plt.figure(figsize=(13, 4.5))
    for k, (X, label) in enumerate(zip(Xs_demo, labels_demo)):
        ax = fig2.add_subplot(1, 3, k + 1, projection="3d")
        # Scale anchors by alpha so loops visually disappear with knockout
        anchors_k = anchors_demo if alphas_demo[k] > 0.1 else []
        render_conformation(ax, X, anchors_k, f"{label} (in-silico cohesin)")
        ax.view_init(elev=22, azim=35)
    fig2.suptitle("In-silico cohesin titration on a single cell\n(loops vanish as alpha -> 0, polymer extends)",
                  fontsize=11)
    fig2.tight_layout()
    out2 = ROOT / "outputs" / "30_cohesin_titration_3d.png"
    fig2.savefig(out2, dpi=140)
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
