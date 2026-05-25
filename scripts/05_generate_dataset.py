"""Generate a diverse simulated (z, D) dataset for Stage-1 small-scale training.

Each cell randomizes CTCF arrangement and LEF parameters so the corpus spans
different loop configurations. Saves a single .npz to data/sim/.

Run:
    python scripts/05_generate_dataset.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hic_unfold.training import generate_dataset  # noqa: E402


def main() -> None:
    N = 48
    num_cells = 5000
    out_path = ROOT / "data" / "sim" / f"step03_N{N}_M{num_cells}.npz"
    print(f"generating {num_cells} cells at N={N} -> {out_path}")
    t0 = time.time()
    info = generate_dataset(num_cells=num_cells, N=N, save_path=out_path,
                            seed=2026, le_steps=300)
    print(f"done in {time.time()-t0:.1f}s")
    print(f"saved {info['path']}")
    print(f"  mu={info['mu']:.4f}, sigma={info['sigma']:.4f}")


if __name__ == "__main__":
    main()
