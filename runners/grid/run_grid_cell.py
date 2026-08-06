#!/usr/bin/env python
"""Run ONE synthetic grid cell: 6 models x N seeds on a constructed-LF CSV, in
parallel (mp.Pool, spawn) over (model, seed). Reuses run_ext.run_one + model
factory (paired GP-vs-TL stays on one worker). Resumable. CPU-forced.

  PY run_grid_cell.py --csv data/grid_cells/cell_00.csv --cellname cell_00 \
     --models MFGP DKL SparseMFGP Progressive KnowledgeDistillation PseudoLabeling \
     --seeds 42 43 ... --budget 50 --init-budget 10 --cost-ratio 0.1 \
     --n-workers 60 --outdir <results>/cell_00
"""
import argparse
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path

HERE = Path(__file__).resolve().parent  # <repo>/runners/grid
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))   # benchmark.py + baselines.py both live in src/
sys.path.insert(0, str(HERE))           # for run_ext (same directory)
import numpy as np
import pandas as pd


def _run(task):
    csv, cellname, model, seed, cost_ratio, budget, init_budget, nthreads = task
    os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU (small-N GP many-worker => CPU)
    os.environ["INIT_BUDGET"] = str(init_budget)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = str(nthreads)
    import torch
    torch.set_num_threads(nthreads)
    from benchmark import ChemistryBenchmark
    import run_ext as R
    t0 = time.time()
    try:
        bench = ChemistryBenchmark(cellname, str(csv), cost_ratio, False, True, False)
        cfg = dict(budget=float(budget), cost_ratio=cost_ratio)
        r = R.run_one(bench, cfg, model, seed, torch.device("cpu"))
        return dict(benchmark=cellname, model=model, seed=int(seed),
                    final_regret=r["final_regret"], auc=r["auc"], n_hf=r["n_hf"],
                    n_lf=r["n_lf"], elapsed=round(time.time() - t0, 1),
                    budgets=r["budgets"], regrets=r["regrets"])
    except Exception as e:
        import traceback
        return dict(benchmark=cellname, model=model, seed=int(seed),
                    error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-400:],
                    elapsed=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--cellname", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--budget", type=float, default=50)
    ap.add_argument("--init-budget", type=float, default=10)
    ap.add_argument("--cost-ratio", type=float, default=0.1)
    ap.add_argument("--n-workers", type=int, default=min(os.cpu_count() or 8, 60))
    ap.add_argument("--nthreads", type=int, default=1)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    out = Path(a.outdir); (out / "cells").mkdir(parents=True, exist_ok=True)
    sf = out / "cells" / f"summary_{a.cellname}.csv"
    tf = out / "cells" / f"traj_{a.cellname}.csv"
    done = set()
    if sf.exists():
        d = pd.read_csv(sf)
        done = set(zip(d.model, d.seed))                 # resume

    tasks = [(a.csv, a.cellname, m, s, a.cost_ratio, a.budget, a.init_budget, a.nthreads)
             for m in a.models for s in a.seeds if (m, s) not in done]
    print(f"{a.cellname}: {len(tasks)} runs (skip {len(done)} done), "
          f"n_workers={a.n_workers}, budget={a.budget} rho={a.cost_ratio}", flush=True)
    if not tasks:
        print("nothing to do"); return

    ctx = mp.get_context("spawn")
    with ctx.Pool(a.n_workers) as pool:
        results = pool.map(_run, tasks)

    srows, trows, nerr = [], [], 0
    for r in results:
        if "error" in r:
            nerr += 1
            print(f"  ERR {r['model']} s{r['seed']}: {r['error']}", flush=True)
            continue
        srows.append({k: r[k] for k in ("benchmark", "model", "seed", "final_regret",
                                        "auc", "n_hf", "n_lf", "elapsed")})
        for b, rg in zip(r["budgets"], r["regrets"]):
            trows.append(dict(benchmark=r["benchmark"], model=r["model"], seed=r["seed"],
                              budget=b, regret=rg))

    def _append(path, rows):
        df = pd.DataFrame(rows)
        if path.exists():
            df = pd.concat([pd.read_csv(path), df], ignore_index=True)
        tmp = path.with_suffix(path.suffix + ".tmp"); df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    if srows:
        _append(sf, srows); _append(tf, trows)
    print(f"{a.cellname}: wrote {len(srows)} ok, {nerr} err -> {sf}", flush=True)


if __name__ == "__main__":
    main()
