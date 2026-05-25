"""Regenerate the simulated (z, D) corpus at N=65, matching the Bintu
chr21:28-30Mb chromatin-tracing dimension. Same logic as 05_generate_dataset.py."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hic_unfold.training import generate_dataset  # noqa: E402


def main() -> None:
    N = 65
    num_cells = 5000
    out_path = ROOT / "data" / "sim" / f"step05_N{N}_M{num_cells}.npz"
    print(f"generating {num_cells} cells at N={N} -> {out_path}")
    t0 = time.time()
    info = generate_dataset(num_cells=num_cells, N=N, save_path=out_path,
                            seed=2026, le_steps=400)
    print(f"done in {time.time()-t0:.1f}s")
    print(f"  mu={info['mu']:.4f}, sigma={info['sigma']:.4f}")


if __name__ == "__main__":
    main()
