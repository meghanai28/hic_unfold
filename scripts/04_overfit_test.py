"""Section 9 step 2: overfit-on-one-example test.

Generate one (z, x) pair from the loop-extrusion + polymer pipeline. Train the
conditional denoiser to memorize it. Sample with DDIM and check that the
generated distance matrix matches the training target much better than chance.

This is a *correctness* test for the diffusion implementation. If it fails, the
diffusion code is broken — debug here before training on real data.

Run:
    python scripts/04_overfit_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hic_unfold.diffusion import (  # noqa: E402
    Denoiser,
    ddim_sample,
    make_cosine_schedule,
    q_sample,
)
from hic_unfold.polymer.gaussian import (  # noqa: E402
    PolymerConfig,
    sample_distance_matrix,
)
from hic_unfold.simulator.loop_extrusion import (  # noqa: E402
    LoopExtrusionConfig,
    run_to_snapshot,
    snapshot_to_loop_matrix,
)


def generate_one_example(N: int, seed: int):
    """Run loop-extrusion + polymer once. Returns (D, z) where D is the
    pairwise distance matrix and z is the loop matrix."""
    rng = np.random.default_rng(seed)
    ctcf_left_stop = np.zeros(N)
    ctcf_right_stop = np.zeros(N)
    a, b = N // 3, 2 * N // 3
    ctcf_left_stop[a] = 1.0
    ctcf_right_stop[b] = 1.0

    le_cfg = LoopExtrusionConfig(
        N=N, num_lefs=2, processivity=400.0,
        ctcf_left_stop=ctcf_left_stop, ctcf_right_stop=ctcf_right_stop,
    )
    poly_cfg = PolymerConfig(backbone_k=1.0, loop_k=15.0)

    L, R = run_to_snapshot(le_cfg, 500, rng)
    z = snapshot_to_loop_matrix(L, R, N).astype(np.float32)
    D, _ = sample_distance_matrix(z, N, rng, poly_cfg)
    return D.astype(np.float32), z


def standardize(D: np.ndarray) -> tuple[np.ndarray, float, float]:
    logD = np.log1p(D)
    mu, sigma = logD.mean(), logD.std()
    return ((logD - mu) / (sigma + 1e-8)).astype(np.float32), float(mu), float(sigma)


def unstandardize(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.expm1(x * sigma + mu)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    N = 48
    n_steps_train = 4000
    T = 1000
    d_c = 16

    D_target, z = generate_one_example(N=N, seed=11)
    x_target_np, mu, sigma = standardize(D_target)
    print(f"target distance matrix: range [{D_target.min():.2f}, {D_target.max():.2f}], "
          f"log1p mean={mu:.3f}, std={sigma:.3f}")

    x_target = torch.from_numpy(x_target_np)[None, None].to(device)
    z_t = torch.from_numpy(z)[None, None].to(device)
    # Stub for c (per-locus features): sinusoidal positional embedding. Real
    # conditioning (sequence, CTCF, ATAC) will replace this in build step 5.
    import math as _math
    half = d_c // 2
    pos = torch.arange(N, device=device, dtype=torch.float32)
    freqs = torch.exp(-_math.log(10000.0) * torch.arange(half, device=device) / half)
    args_pe = pos[:, None] * freqs[None, :]
    c_emb = torch.cat([args_pe.sin(), args_pe.cos()], dim=-1)  # (N, d_c)
    c_t = c_emb.T.unsqueeze(0).contiguous()                    # (1, d_c, N)

    net = Denoiser(N=N, d_c=d_c, d_pair=32, d_sep=16, d_h=96, d_t=128,
                   dilations=(1, 2, 4, 8, 1)).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"denoiser params: {n_params:,}")

    alpha_bars = make_cosine_schedule(T=T, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)

    losses = []
    t0 = time.time()
    for step in range(1, n_steps_train + 1):
        t_idx = torch.randint(1, T + 1, (1,), device=device)
        x_t, v_target = q_sample(x_target, t_idx, alpha_bars)
        v_pred = net(x_t, z_t, c_t, t_idx)
        loss = F.mse_loss(v_pred, v_target)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step % 500 == 0:
            recent = np.mean(losses[-200:])
            print(f"step {step:5d} | recent loss {recent:.4f} | elapsed {time.time()-t0:.1f}s")

    net.eval()
    n_samples = 4
    z_batch = z_t.expand(n_samples, -1, -1, -1).contiguous()
    c_batch = c_t.expand(n_samples, -1, -1).contiguous()
    samples_x = ddim_sample(net, z_batch, c_batch, alpha_bars, n_steps=100)
    samples_x = samples_x.squeeze(1).cpu().numpy()

    samples_D = np.stack([unstandardize(s, mu, sigma) for s in samples_x])
    samples_D = np.maximum(samples_D, 0)
    samples_D = 0.5 * (samples_D + samples_D.transpose(0, 2, 1))
    np.fill_diagonal(samples_D[0], 0)  # diag check on at least one
    for s in samples_D:
        np.fill_diagonal(s, 0)

    target_norm = np.sqrt((D_target ** 2).mean())
    mse_to_target = np.mean((samples_D - D_target[None]) ** 2, axis=(1, 2))
    rmse_to_target = np.sqrt(mse_to_target)
    print(f"sample-to-target RMSE: {rmse_to_target}  (target RMS={target_norm:.3f})")

    rng = np.random.default_rng(99)
    other_D, _ = sample_distance_matrix(np.zeros((N, N), dtype=np.float32), N, rng,
                                        PolymerConfig(backbone_k=1.0, loop_k=15.0))
    other_rmse = float(np.sqrt(np.mean((other_D - D_target) ** 2)))
    print(f"random unrelated polymer RMSE to target: {other_rmse:.3f}")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 5)

    ax = fig.add_subplot(gs[0, 0]); ax.imshow(D_target, origin="lower", cmap="viridis")
    ax.set_title("target D"); ax.axis("off")
    ax = fig.add_subplot(gs[0, 1]); ax.imshow(z, origin="lower", cmap="gray_r")
    ax.set_title("conditioning z"); ax.axis("off")

    for i in range(n_samples):
        ax = fig.add_subplot(gs[0, 2 + i] if i < 3 else gs[1, i - 1])
        ax.imshow(samples_D[i], origin="lower", cmap="viridis")
        ax.set_title(f"sample {i+1}\nRMSE={rmse_to_target[i]:.2f}"); ax.axis("off")

    ax = fig.add_subplot(gs[1, 0]); ax.imshow(other_D, origin="lower", cmap="viridis")
    ax.set_title(f"random unrelated\nRMSE={other_rmse:.2f}"); ax.axis("off")

    ax = fig.add_subplot(gs[1, 4])
    ax.plot(np.arange(1, n_steps_train + 1), losses, lw=0.7)
    ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("MSE(v)")
    ax.set_title("training loss")

    fig.suptitle(f"Step-2 overfit test: denoiser memorises one (z, x) pair "
                 f"(N={N}, params={n_params:,})")
    fig.tight_layout()
    out_path = out_dir / "04_overfit_test.png"
    fig.savefig(out_path, dpi=130)
    print(f"saved {out_path}")

    pass_threshold = 0.5 * other_rmse
    best = rmse_to_target.min()
    print(f"\nverdict: best sample RMSE={best:.3f}, pass threshold (< 0.5 * random) "
          f"= {pass_threshold:.3f}")
    if best < pass_threshold:
        print("  OK — denoiser successfully memorised the example")
    else:
        print("  FAIL — denoiser did not memorise; investigate before continuing")


if __name__ == "__main__":
    main()
