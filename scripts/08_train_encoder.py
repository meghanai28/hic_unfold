"""Build step 4: train the encoder q(z | x) on simulated (z, D) pairs.

z is observed in the simulated corpus, so this is plain supervised learning —
no posterior-collapse risk, no ELBO. The model maps standardized log1p(D) to
per-pair loop logits. BCE loss with a positive-class weight handles the heavy
class imbalance (only a handful of nonzero entries in z per cell).

Run:
    python scripts/08_train_encoder.py
"""

from __future__ import annotations

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

from hic_unfold.encoder import LoopEncoder  # noqa: E402
from hic_unfold.training import SimulatedDataset, make_positional_c  # noqa: E402


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = 48
    d_c = 16
    batch_size = 64
    num_epochs = 25
    lr = 2e-4
    val_frac = 0.1

    data_path = ROOT / "data" / "sim" / f"step03_N{N}_M5000.npz"
    ckpt_path = ROOT / "checkpoints" / "step04_encoder.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    ds = SimulatedDataset(data_path)
    M = len(ds)
    n_val = max(1, int(M * val_frac))
    perm = np.random.default_rng(0).permutation(M)
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    z_all = ds.z[train_idx]
    pos_frac = float(z_all.mean())
    pos_weight = float((1 - pos_frac) / max(pos_frac, 1e-8))
    print(f"dataset: N={ds.N}, train={len(train_idx)}, val={n_val}")
    print(f"positive fraction in z: {pos_frac:.5f} -> pos_weight={pos_weight:.1f}")
    print(f"device: {device}")

    net = LoopEncoder(N=N, d_c=d_c, d_pair=32, d_sep=16, d_h=96,
                      dilations=(1, 2, 4, 8, 1)).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"encoder params: {n_params:,}")

    c_const = make_positional_c(N, d_c, device)
    pw = torch.tensor(pos_weight, device=device)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_aurocs: list[float] = []

    def auroc_batch(logits: torch.Tensor, target: torch.Tensor) -> float:
        """Quick batched AUROC over the upper triangle of each (N, N) prediction.
        Returns the mean AUROC across the batch."""
        B, _, N_, _ = logits.shape
        iu = torch.triu_indices(N_, N_, offset=1, device=logits.device)
        scores = logits[:, 0, iu[0], iu[1]]
        labels = target[:, 0, iu[0], iu[1]]
        aurocs = []
        for b in range(B):
            s, l = scores[b], labels[b]
            if l.sum() == 0 or l.sum() == l.numel():
                continue
            order = torch.argsort(s, descending=True)
            l_sorted = l[order]
            tp = torch.cumsum(l_sorted, dim=0)
            fp = torch.cumsum(1 - l_sorted, dim=0)
            P = l.sum(); F_ = (1 - l).sum()
            tpr = tp / P; fpr = fp / F_
            tpr = torch.cat([torch.zeros(1, device=tpr.device), tpr])
            fpr = torch.cat([torch.zeros(1, device=fpr.device), fpr])
            auc = torch.trapz(tpr, fpr).item()
            aurocs.append(auc)
        return float(np.mean(aurocs)) if aurocs else float("nan")

    def run_val() -> tuple[float, float]:
        net.eval()
        tot_loss, tot_n, aurocs = 0.0, 0, []
        with torch.no_grad():
            for z_b, x_b in val_loader:
                z_b = z_b.to(device); x_b = x_b.to(device)
                B = z_b.size(0)
                c_b = c_const.expand(B, -1, -1)
                logits = net(x_b, c_b)
                loss = F.binary_cross_entropy_with_logits(logits, z_b, pos_weight=pw,
                                                          reduction="sum")
                tot_loss += loss.item(); tot_n += z_b.numel()
                aurocs.append(auroc_batch(logits, z_b))
        net.train()
        return tot_loss / tot_n, float(np.nanmean(aurocs))

    t0 = time.time(); step = 0
    for epoch in range(1, num_epochs + 1):
        net.train()
        for z_b, x_b in train_loader:
            z_b = z_b.to(device, non_blocking=True)
            x_b = x_b.to(device, non_blocking=True)
            B = z_b.size(0)
            c_b = c_const.expand(B, -1, -1)
            logits = net(x_b, c_b)
            loss = F.binary_cross_entropy_with_logits(logits, z_b, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
            train_losses.append(loss.item())
            step += 1
        v_loss, v_auc = run_val()
        val_losses.append(v_loss); val_aurocs.append(v_auc)
        recent = np.mean(train_losses[-len(train_loader):])
        elapsed = time.time() - t0
        print(f"epoch {epoch:2d}/{num_epochs} | step {step:5d} | "
              f"train {recent:.4f} | val_loss {v_loss:.4f} | val_AUROC {v_auc:.4f} | "
              f"elapsed {elapsed:.1f}s")

    torch.save({
        "state_dict": net.state_dict(),
        "N": N, "d_c": d_c,
        "mu": ds.mu, "sigma": ds.sigma,
        "pos_weight": pos_weight,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_aurocs": val_aurocs,
    }, ckpt_path)
    print(f"saved {ckpt_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(train_losses, lw=0.4, alpha=0.5, label="train")
    val_x = np.arange(1, num_epochs + 1) * len(train_loader)
    ax1.plot(val_x, val_losses, "o-", color="C1", lw=2, label="val")
    ax1.set_yscale("log"); ax1.set_xlabel("step"); ax1.set_ylabel("BCE / pixel")
    ax1.set_title("encoder training loss"); ax1.legend(); ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(np.arange(1, num_epochs + 1), val_aurocs, "o-", color="C2")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("val AUROC")
    ax2.set_title("validation AUROC (upper-tri pairs)")
    ax2.set_ylim(0.5, 1.0); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = ROOT / "outputs" / "08_encoder_training.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
