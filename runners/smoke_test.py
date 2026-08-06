#!/usr/bin/env python
"""Minimal end-to-end smoke test of the shipped pipeline (CPU, ~1 min).

Runs the pool-based BO loop on Branin-Fav with a tiny budget for one
transfer-learning surrogate (Sequential), the baseline MFGP, and one GP variant
from baselines.py (NARGP), then checks that every module the figure scripts
import is importable. Exercises: grid construction, DNN pretrain/fine-tune,
BLR head, const-std LF EI, HF argmin step, and the GP fits.

  python runners/smoke_test.py
"""
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))

from benchmark import MFGP, Sequential, SyntheticBenchmark, run_bo_lf_blr
from synthetic_functions import branin_hf, branin_lf
import baselines

BUDGET = 5   # HF-equivalent units; enough for several LF + HF steps


def run(model_cls, name):
    t0 = time.time()
    bench = SyntheticBenchmark('Branin-Fav', branin_hf, branin_lf, dim=2,
                               alpha=0.8, cost_ratio=0.1, f_star=0.397887,
                               grid_size=50)
    out = run_bo_lf_blr(bench, model_cls, budget=BUDGET, seed=42, device='cpu')
    regret = out['final_regret']
    assert np.isfinite(regret), regret
    print(f"  {name:12s} final_regret={regret:.4f}  ({time.time() - t0:.1f}s)")


def main():
    print(f"pool R2 check: ", end='')
    b = SyntheticBenchmark('Branin-Fav', branin_hf, branin_lf, 2, 0.8, 0.1,
                           0.397887, 50)
    assert b.X.shape == (2500, 2), b.X.shape       # 50x50 candidate grid
    assert 0.9 < b.r2 <= 1.0, b.r2                 # paper: R2 = 0.97
    print(f"pool {b.X.shape}, R2 = {b.r2:.2f}  OK")

    for cls, name in [(Sequential, 'Sequential'), (MFGP, 'MFGP'),
                      (baselines.NARGP, 'NARGP')]:
        run(cls, name)

    print("figure-script imports: ", end='')
    sys.path.insert(0, str(REPO / 'figures'))
    import _common                                          # noqa: F401
    import grid_stats                                       # noqa: F401
    from effect_sizes_rank_biserial import rank_biserial    # noqa: F401
    r, n = rank_biserial(np.array([1.0, 2.0, -0.5]))
    assert n == 3 and -1 <= r <= 1
    print("OK")
    print("SMOKE TEST PASSED")


if __name__ == '__main__':
    main()
