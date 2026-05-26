"""Encoder fine-tuning: close the sim-to-real gap using HCT116 auxin as a
supervisory signal for cohesin-aware encoding.

Step 26 showed the sim-trained encoder doesn't detect cohesin loss (predicted
loop mass is essentially identical for untreated vs auxin cells). Step 28
showed that the diffusion model + forward operator correctly reproduce the
auxin ensemble when given z = 0.5 * z_hat. So the encoder is over-predicting
loop mass on auxin cells by 2x.

Fix: fine-tune the encoder with a contrastive loss that enforces

    mean(loop_mass | auxin) <= 0.5 * mean(loop_mass | untreated)

with a magnitude-preservation regularizer that anchors the untreated mass
near its initial value (preserves the learned loop-localisation).

Run:
    python scripts/29_encoder_finetune.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(line_buffering=True)

from hic_unfold.data import load_bintu_csv, preprocess_bintu  # noqa: E402
from hic_unfold.encoder import LoopEncoder  # noqa: E402
from hic_unfold.training import make_positional_c  # noqa: E402


def mass_per_cell(z_logits: torch.Tensor) -> torch.Tensor:
    """Sum of sigmoid(z_logits) over upper triangle per cell."""
    B, _, N, _ = z_logits.shape
    p = torch.sigmoid(z_logits)
    iu = torch.triu_indices(N, N, offset=1, device=z_logits.device)
    return p[:, 0, iu[0], iu[1]].sum(dim=-1)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_ckpt_path = ROOT / "checkpoints" / "step05_encoder_N65.pt"
    out_ckpt_path = ROOT / "checkpoints" / "step18_encoder_finetuned.pt"

    print("loading HCT116 untreated + auxin...")
    ds_u = load_bintu_csv(ROOT / "data" / "raw_bintu2018" / "HCT116_chr21-28-30Mb_untreated.csv")
    ds_a = load_bintu_csv(ROOT / "data" / "raw_bintu2018" / "HCT116_chr21-28-30Mb_6h_auxin.csv")
    real_u = preprocess_bintu(ds_u, min_valid_frac=0.85)
    real_a = preprocess_bintu(ds_a, min_valid_frac=0.85)
    N = real_u.D.shape[-1]
    print(f"  untreated: {real_u.D.shape[0]}, auxin: {real_a.D.shape[0]}")

    # Split each set into train/test
    rng = np.random.default_rng(2027)
    n_test_u = int(0.2 * real_u.D.shape[0])
    n_test_a = int(0.2 * real_a.D.shape[0])
    perm_u = rng.permutation(real_u.D.shape[0])
    perm_a = rng.permutation(real_a.D.shape[0])
    test_u = perm_u[:n_test_u]; train_u = perm_u[n_test_u:]
    test_a = perm_a[:n_test_a]; train_a = perm_a[n_test_a:]
    print(f"  train untreated: {len(train_u)}, test untreated: {len(test_u)}")
    print(f"  train auxin:     {len(train_a)}, test auxin:     {len(test_a)}")

    # Standardise each set with its own mu/sigma so encoder input is in-dist
    def standardise(D: np.ndarray) -> np.ndarray:
        l = np.log1p(D)
        mu = float(l.mean()); s = float(l.std())
        return ((l - mu) / max(s, 1e-8)).astype(np.float32)
    x_u_full = standardise(real_u.D)
    x_a_full = standardise(real_a.D)

    # Load encoder
    enc_ckpt = torch.load(enc_ckpt_path, map_location=device, weights_only=False)
    d_c = int(enc_ckpt["d_c"])
    enc = LoopEncoder(N=N, d_c=d_c, d_pair=32, d_sep=16, d_h=96,
                      dilations=(1, 2, 4, 8, 1)).to(device)
    enc.load_state_dict(enc_ckpt["state_dict"])
    c_const = make_positional_c(N, d_c, device)

    def evaluate(enc_local):
        enc_local.eval()
        with torch.no_grad():
            def batch_mass(x_set, n_max=400):
                idx = np.arange(len(x_set))[:n_max]
                x_b = torch.from_numpy(x_set[idx])[:, None].to(device)
                c_b = c_const.expand(len(idx), -1, -1)
                z = enc_local(x_b, c_b)
                return mass_per_cell(z).cpu().numpy()
            m_u = batch_mass(x_u_full[test_u])
            m_a = batch_mass(x_a_full[test_a])
        enc_local.train()
        return m_u, m_a

    print("\nbefore fine-tuning:")
    m_u_before, m_a_before = evaluate(enc)
    ratio_before = float(np.median(m_a_before) / max(np.median(m_u_before), 1e-9))
    print(f"  test-set median loop mass: untreated={np.median(m_u_before):.2f}, "
          f"auxin={np.median(m_a_before):.2f}, ratio={ratio_before:.3f} (target ~0.5)")

    # Fine-tune
    target_ratio = 0.5
    target_u_mass = float(np.median(m_u_before))
    print(f"target: auxin <= {target_ratio} * untreated; anchor untreated near {target_u_mass:.2f}")

    opt = torch.optim.Adam(enc.parameters(), lr=5e-5)
    batch_size = 32
    n_epochs = 8
    losses_contrast: list[float] = []
    losses_anchor: list[float] = []

    print(f"fine-tuning for {n_epochs} epochs, batch {batch_size}, lr 5e-5...")
    t0 = time.time()
    enc.train()
    for epoch in range(n_epochs):
        # shuffle each set
        rng.shuffle(train_u)
        rng.shuffle(train_a)
        n_batches = min(len(train_u), len(train_a)) // batch_size
        for b in range(n_batches):
            u_idx = train_u[b * batch_size:(b + 1) * batch_size]
            a_idx = train_a[b * batch_size:(b + 1) * batch_size]
            x_u_b = torch.from_numpy(x_u_full[u_idx])[:, None].to(device)
            x_a_b = torch.from_numpy(x_a_full[a_idx])[:, None].to(device)
            c_b = c_const.expand(batch_size, -1, -1)

            z_u = enc(x_u_b, c_b)
            z_a = enc(x_a_b, c_b)
            mass_u = mass_per_cell(z_u).mean()
            mass_a = mass_per_cell(z_a).mean()

            l_contrast = F.relu(mass_a - target_ratio * mass_u)
            l_anchor = (mass_u - target_u_mass) ** 2 / max(target_u_mass ** 2, 1.0)
            loss = l_contrast + 0.5 * l_anchor
            opt.zero_grad(); loss.backward(); opt.step()
            losses_contrast.append(float(l_contrast.item()))
            losses_anchor.append(float(l_anchor.item()))

        m_u_now, m_a_now = evaluate(enc)
        r = np.median(m_a_now) / max(np.median(m_u_now), 1e-9)
        print(f"  epoch {epoch + 1}/{n_epochs}  u_med={np.median(m_u_now):.2f}  "
              f"a_med={np.median(m_a_now):.2f}  ratio={r:.3f}  "
              f"loss_contrast={np.mean(losses_contrast[-n_batches:]):.4f}  "
              f"elapsed {time.time() - t0:.1f}s")

    print("\nafter fine-tuning:")
    m_u_after, m_a_after = evaluate(enc)
    ratio_after = float(np.median(m_a_after) / max(np.median(m_u_after), 1e-9))
    print(f"  test-set median loop mass: untreated={np.median(m_u_after):.2f}, "
          f"auxin={np.median(m_a_after):.2f}, ratio={ratio_after:.3f}")
    print(f"\n  RATIO change:  before {ratio_before:.3f}  ->  after {ratio_after:.3f}  "
          f"(target {target_ratio:.2f})")

    torch.save({
        "state_dict": enc.state_dict(),
        "N": N, "d_c": d_c,
        "ratio_before": ratio_before,
        "ratio_after": ratio_after,
        "losses_contrast": losses_contrast,
        "losses_anchor": losses_anchor,
        "fine_tune_data": "HCT116 untreated+auxin contrastive",
    }, out_ckpt_path)
    print(f"saved {out_ckpt_path}")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(losses_contrast, lw=0.5, alpha=0.6, color="C3", label="contrastive (hinge)")
    ax.plot(losses_anchor, lw=0.5, alpha=0.6, color="C0", label="anchor (untreated mass)")
    ax.set_xlabel("step"); ax.set_ylabel("loss")
    ax.set_title("fine-tune loss"); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    ax = axes[1]
    bins = np.linspace(min(m_u_before.min(), m_a_before.min()),
                       max(m_u_before.max(), m_a_before.max()), 30)
    ax.hist(m_u_before, bins=bins, density=True, alpha=0.5, color="C0",
            label=f"untreated (med={np.median(m_u_before):.1f})")
    ax.hist(m_a_before, bins=bins, density=True, alpha=0.5, color="C3",
            label=f"auxin (med={np.median(m_a_before):.1f})")
    ax.set_xlabel("per-cell loop mass"); ax.set_ylabel("density")
    ax.set_title(f"BEFORE: ratio = {ratio_before:.3f}")
    ax.legend(fontsize=8)

    ax = axes[2]
    bins = np.linspace(min(m_u_after.min(), m_a_after.min()),
                       max(m_u_after.max(), m_a_after.max()), 30)
    ax.hist(m_u_after, bins=bins, density=True, alpha=0.5, color="C0",
            label=f"untreated (med={np.median(m_u_after):.1f})")
    ax.hist(m_a_after, bins=bins, density=True, alpha=0.5, color="C3",
            label=f"auxin (med={np.median(m_a_after):.1f})")
    ax.set_xlabel("per-cell loop mass"); ax.set_ylabel("density")
    ax.set_title(f"AFTER: ratio = {ratio_after:.3f}  (target {target_ratio:.2f})")
    ax.legend(fontsize=8)

    fig.suptitle("Encoder fine-tuning closes the cohesin-loss detection gap")
    fig.tight_layout()
    out = out_dir / "29_encoder_finetune.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
