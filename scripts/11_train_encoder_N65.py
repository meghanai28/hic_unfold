"""Train the loop encoder q(z | x) at N=65 so it can be applied to real Bintu cells.
Same architecture and protocol as 08_train_encoder.py; only the dataset and
checkpoint paths change."""

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
sys.stdout.reconfigure(line_buffering=True)

from hic_unfold.encoder import LoopEncoder  # noqa: E402
from hic_unfold.training import SimulatedDataset, make_positional_c  # noqa: E402


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = 65
    d_c = 16
    batch_size = 48
    num_epochs = 18
    lr = 2e-4
    val_frac = 0.1

    data_path = ROOT / "data" / "sim" / f"step05_N{N}_M5000.npz"
    ckpt_path = ROOT / "checkpoints" / "step05_encoder_N65.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    ds = SimulatedDataset(data_path)
    M = len(ds)
    n_val = max(1, int(M * val_frac))
    perm = np.random.default_rng(0).permutation(M)
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False, num_workers=0)

    pos_frac = float(ds.z[train_idx].mean())
    pos_weight = float((1 - pos_frac) / max(pos_frac, 1e-8))
    print(f"N={ds.N}, train={len(train_idx)}, val={n_val}, pos_frac={pos_frac:.5f}, pos_weight={pos_weight:.1f}")
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

    def auroc_batch(logits, target):
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
            l_s = l[order]
            P = l.sum(); F_ = (1 - l).sum()
            tpr = torch.cat([torch.zeros(1, device=s.device), torch.cumsum(l_s, 0) / P])
            fpr = torch.cat([torch.zeros(1, device=s.device), torch.cumsum(1 - l_s, 0) / F_])
            aurocs.append(torch.trapz(tpr, fpr).item())
        return float(np.mean(aurocs)) if aurocs else float("nan")

    def run_val():
        net.eval()
        tot, n, aurocs = 0.0, 0, []
        with torch.no_grad():
            for z_b, x_b in val_loader:
                z_b = z_b.to(device); x_b = x_b.to(device)
                B = z_b.size(0)
                logits = net(x_b, c_const.expand(B, -1, -1))
                loss = F.binary_cross_entropy_with_logits(logits, z_b, pos_weight=pw, reduction="sum")
                tot += loss.item(); n += z_b.numel()
                aurocs.append(auroc_batch(logits, z_b))
        net.train()
        return tot / n, float(np.nanmean(aurocs))

    best_val = float("inf"); best_state = None
    t0 = time.time(); step = 0
    for epoch in range(1, num_epochs + 1):
        net.train()
        for z_b, x_b in train_loader:
            z_b = z_b.to(device, non_blocking=True)
            x_b = x_b.to(device, non_blocking=True)
            B = z_b.size(0)
            logits = net(x_b, c_const.expand(B, -1, -1))
            loss = F.binary_cross_entropy_with_logits(logits, z_b, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
            train_losses.append(loss.item())
            step += 1
        v_loss, v_auc = run_val()
        val_losses.append(v_loss); val_aurocs.append(v_auc)
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        recent = np.mean(train_losses[-len(train_loader):])
        print(f"epoch {epoch:2d}/{num_epochs} | train {recent:.4f} | val {v_loss:.4f} | "
              f"AUROC {v_auc:.4f} | elapsed {time.time()-t0:.1f}s")

    print(f"best val loss: {best_val:.4f} (saving best state)")
    torch.save({
        "state_dict": best_state if best_state is not None else net.state_dict(),
        "N": N, "d_c": d_c,
        "mu": ds.mu, "sigma": ds.sigma,
        "pos_weight": pos_weight,
        "train_losses": train_losses, "val_losses": val_losses, "val_aurocs": val_aurocs,
    }, ckpt_path)
    print(f"saved {ckpt_path}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(train_losses, lw=0.4, alpha=0.5, label="train")
    val_x = np.arange(1, num_epochs + 1) * len(train_loader)
    a1.plot(val_x, val_losses, "o-", color="C1", lw=2, label="val")
    a1.set_yscale("log"); a1.set_xlabel("step"); a1.set_ylabel("BCE / pixel")
    a1.set_title("encoder N=65 training"); a1.legend(); a1.grid(True, which="both", alpha=0.3)
    a2.plot(np.arange(1, num_epochs + 1), val_aurocs, "o-", color="C2")
    a2.set_xlabel("epoch"); a2.set_ylabel("val AUROC"); a2.set_ylim(0.5, 1.0)
    a2.set_title("validation AUROC"); a2.grid(True, alpha=0.3)
    fig.tight_layout()
    out = ROOT / "outputs" / "11_encoder_N65_training.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
