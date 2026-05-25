"""Build step 5: retrain p(x | z, c) on real Bintu (D_real, z_hat) pairs.

z_hat are soft probabilities from the simulation-trained encoder. The diffusion
model's z conditioning channel accepts continuous values, so we feed the
probabilities directly. mu/sigma come from the real corpus (different scale
from sim).

Run:
    python scripts/13_train_real.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hic_unfold.diffusion import Denoiser, make_cosine_schedule, q_sample  # noqa: E402
from hic_unfold.training import make_positional_c  # noqa: E402


class RealZHatDataset(Dataset):
    def __init__(self, npz_path: Path):
        f = np.load(npz_path)
        self.x = f["x"].astype(np.float32)         # standardized log1p(D_real)
        self.z = f["z_hat"].astype(np.float32)     # soft probabilities in [0, 1]
        self.D = f["D"].astype(np.float32)
        self.mu = float(f["mu"]); self.sigma = float(f["sigma"])
        self.N = int(f["N"])

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        z = torch.from_numpy(self.z[idx])[None]
        x = torch.from_numpy(self.x[idx])[None]
        return z, x


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    region = "IMR90_chr21-28-30Mb"

    d_c = 16
    T = 1000
    batch_size = 32
    num_epochs = 30
    lr = 2e-4
    val_frac = 0.1

    data_path = ROOT / "data" / "real" / f"{region}_preprocessed.npz"
    ckpt_path = ROOT / "checkpoints" / "step05_diffusion_real.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    ds = RealZHatDataset(data_path)
    N = ds.N
    M = len(ds)
    n_val = max(1, int(M * val_frac))
    perm = np.random.default_rng(0).permutation(M)
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"real dataset: N={N}, train={len(train_idx)}, val={n_val}, "
          f"mu={ds.mu:.3f}, sigma={ds.sigma:.3f}")
    print(f"device: {device}")

    net = Denoiser(N=N, d_c=d_c, d_pair=32, d_sep=16, d_h=96, d_t=128,
                   dilations=(1, 2, 4, 8, 1)).to(device)
    print(f"denoiser params: {sum(p.numel() for p in net.parameters()):,}")
    alpha_bars = make_cosine_schedule(T=T, device=device)
    c_const = make_positional_c(N, d_c, device)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf"); best_state = None

    def run_val():
        net.eval()
        tot = 0.0; n = 0
        with torch.no_grad():
            for z_b, x_b in val_loader:
                z_b = z_b.to(device); x_b = x_b.to(device)
                B = z_b.size(0)
                t_idx = torch.randint(1, T + 1, (B,), device=device)
                x_t, v_target = q_sample(x_b, t_idx, alpha_bars)
                v_pred = net(x_t, z_b, c_const.expand(B, -1, -1), t_idx)
                tot += F.mse_loss(v_pred, v_target, reduction="sum").item()
                n += v_target.numel()
        net.train()
        return tot / n

    t0 = time.time(); step = 0
    for epoch in range(1, num_epochs + 1):
        net.train()
        for z_b, x_b in train_loader:
            z_b = z_b.to(device, non_blocking=True)
            x_b = x_b.to(device, non_blocking=True)
            B = z_b.size(0)
            t_idx = torch.randint(1, T + 1, (B,), device=device)
            x_t, v_target = q_sample(x_b, t_idx, alpha_bars)
            v_pred = net(x_t, z_b, c_const.expand(B, -1, -1), t_idx)
            loss = F.mse_loss(v_pred, v_target)
            opt.zero_grad(); loss.backward(); opt.step()
            train_losses.append(loss.item())
            step += 1
        v = run_val()
        val_losses.append(v)
        if v < best_val:
            best_val = v
            best_state = {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
        recent = np.mean(train_losses[-len(train_loader):])
        print(f"epoch {epoch:2d}/{num_epochs} | train {recent:.4f} | val {v:.4f} | "
              f"elapsed {time.time()-t0:.1f}s")

    print(f"best val: {best_val:.4f}")
    torch.save({
        "state_dict": best_state if best_state is not None else net.state_dict(),
        "N": N, "d_c": d_c, "T": T,
        "mu": ds.mu, "sigma": ds.sigma,
        "train_losses": train_losses, "val_losses": val_losses,
        "val_idx": val_idx,
    }, ckpt_path)
    print(f"saved {ckpt_path}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, lw=0.4, alpha=0.5, label="train")
    val_x = np.arange(1, num_epochs + 1) * len(train_loader)
    ax.plot(val_x, val_losses, "o-", color="C1", lw=2, label="val")
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("MSE(v)")
    ax.set_title(f"Step-5 training: real Bintu data (N={N})")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = ROOT / "outputs" / "13_train_real_curve.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
