"""Build step 3: small-scale training of p(x | z, c) on simulated pairs.

Trains the conditional denoiser on the dataset produced by 05_generate_dataset.py.
Saves a checkpoint to checkpoints/step03.pt and the training-curve plot to
outputs/06_training_curve.png.

Run:
    python scripts/06_train_simulated.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hic_unfold.diffusion import (  # noqa: E402
    Denoiser,
    make_cosine_schedule,
    q_sample,
)
from hic_unfold.training import SimulatedDataset, make_positional_c  # noqa: E402


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = 48
    d_c = 16
    T = 1000
    batch_size = 64
    num_epochs = 30
    lr = 2e-4
    val_frac = 0.1

    data_path = ROOT / "data" / "sim" / f"step03_N{N}_M5000.npz"
    ckpt_path = ROOT / "checkpoints" / "step03.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    ds = SimulatedDataset(data_path)
    M = len(ds)
    n_val = max(1, int(M * val_frac))
    n_train = M - n_val
    perm = np.random.default_rng(0).permutation(M)
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"dataset: N={ds.N}, train={n_train}, val={n_val}, mu={ds.mu:.3f}, sigma={ds.sigma:.3f}")
    print(f"device: {device}")

    net = Denoiser(N=N, d_c=d_c, d_pair=32, d_sep=16, d_h=96, d_t=128,
                   dilations=(1, 2, 4, 8, 1)).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"denoiser params: {n_params:,}")
    alpha_bars = make_cosine_schedule(T=T, device=device)
    c_const = make_positional_c(N, d_c, device)  # (1, d_c, N)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    train_losses: list[float] = []
    val_losses: list[float] = []

    def run_val() -> float:
        net.eval()
        total = 0.0; n = 0
        with torch.no_grad():
            for z_b, x_b in val_loader:
                z_b = z_b.to(device); x_b = x_b.to(device)
                B = z_b.size(0)
                t_idx = torch.randint(1, T + 1, (B,), device=device)
                x_t, v_target = q_sample(x_b, t_idx, alpha_bars)
                c_b = c_const.expand(B, -1, -1)
                v_pred = net(x_t, z_b, c_b, t_idx)
                total += F.mse_loss(v_pred, v_target, reduction="sum").item()
                n += v_target.numel()
        net.train()
        return total / n

    t0 = time.time()
    step = 0
    for epoch in range(1, num_epochs + 1):
        net.train()
        for z_b, x_b in train_loader:
            z_b = z_b.to(device, non_blocking=True)
            x_b = x_b.to(device, non_blocking=True)
            B = z_b.size(0)
            t_idx = torch.randint(1, T + 1, (B,), device=device)
            x_t, v_target = q_sample(x_b, t_idx, alpha_bars)
            c_b = c_const.expand(B, -1, -1)
            v_pred = net(x_t, z_b, c_b, t_idx)
            loss = F.mse_loss(v_pred, v_target)
            opt.zero_grad(); loss.backward(); opt.step()
            train_losses.append(loss.item())
            step += 1
        v = run_val()
        val_losses.append(v)
        recent = np.mean(train_losses[-len(train_loader):])
        elapsed = time.time() - t0
        print(f"epoch {epoch:2d}/{num_epochs} | step {step:5d} | "
              f"train(last-epoch) {recent:.4f} | val {v:.4f} | elapsed {elapsed:.1f}s")

    torch.save({
        "state_dict": net.state_dict(),
        "N": N, "d_c": d_c, "T": T,
        "mu": ds.mu, "sigma": ds.sigma,
        "train_losses": train_losses, "val_losses": val_losses,
    }, ckpt_path)
    print(f"saved checkpoint {ckpt_path}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(train_losses)), train_losses, lw=0.4, alpha=0.5, label="train")
    val_x = np.arange(1, num_epochs + 1) * len(train_loader)
    ax.plot(val_x, val_losses, "o-", color="C1", lw=2, label="val")
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("MSE(v)")
    ax.set_title(f"Step-3 training curve (N={N}, {n_train} train / {n_val} val)")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = ROOT / "outputs" / "06_training_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
