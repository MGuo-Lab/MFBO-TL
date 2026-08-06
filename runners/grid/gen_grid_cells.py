#!/usr/bin/env python
"""Materialise the feasible grid LFs as drop-in benchmark CSVs.

For each unique (n_b, dem) in grid_cells.csv (feasible), build the synthetic LF
on the Polariz substrate (DETERMINISTIC seed=0), measure its actual (R^2, top10,
top50), and write data/grid_cells/cell_<id>.csv (f0..f9, HF, LF) -- HF/LF signed
(best=min), use_smiles=False, so run_ext/ChemistryBenchmark loads it unchanged.

Manifest grid_manifest.csv maps cell_id -> (n_b, dem, r2, top10, ...). Login-safe.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SUB = os.path.join(REPO, "data", "polarizability_pca10.csv")
OUTDIR = os.path.join(REPO, "data", "grid_cells")
os.makedirs(OUTDIR, exist_ok=True)
Q, KFOLDS, N_TREES, DEMOTE_Q, SEED = 0.05, 5, 80, 0.12, 0

df = pd.read_csv(SUB)
fcols = [f"f{i}" for i in range(10)]
X = df[fcols].to_numpy(float); y = df["HF"].to_numpy(float)
N = len(y); nO = max(int(round(Q * N)), 50)
basin = np.argsort(y)[:nO]; basin_set = set(basin.tolist())
bulk_idx = np.array([i for i in range(N) if i not in basin_set])


def build(n_b, dem, seed=SEED):
    rng = np.random.RandomState(seed)
    pred = np.full(N, np.nan)
    for tr, te in KFold(KFOLDS, shuffle=True, random_state=seed).split(np.arange(N)):
        s = set(tr.tolist())
        b = [i for i in bulk_idx if i in s]; o = [i for i in basin if i in s]
        sel = list(rng.permutation(b)[:min(n_b, len(b))]) + o
        rf = RandomForestRegressor(n_estimators=N_TREES, n_jobs=1, random_state=seed,
                                   max_features="sqrt").fit(X[sel], y[sel])
        pred[te] = rf.predict(X[te])
    n_dem = int(round(dem * len(basin)))
    if n_dem:
        hit = rng.permutation(basin)[:n_dem]
        pred[hit] = np.quantile(pred, DEMOTE_Q) + 1e-6 * rng.randn(n_dem)
    return pred


cells = pd.read_csv(os.path.join(HERE, "grid_cells.csv"))
uniq = cells[cells.feasible].drop_duplicates(["n_b", "dem"]).reset_index(drop=True)
man = []
for cid, r in uniq.iterrows():
    lf = build(int(r.n_b), float(r.dem))
    r2 = float(np.corrcoef(lf, y)[0, 1]) ** 2
    t10 = len(set(np.argsort(lf)[:10]) & set(np.argsort(y)[:10])) / 10
    t50 = len(set(np.argsort(lf)[:50]) & set(np.argsort(y)[:50])) / 50
    out = df[fcols].copy(); out["HF"] = y; out["LF"] = lf
    name = f"cell_{cid:02d}"
    out.to_csv(os.path.join(OUTDIR, f"{name}.csv"), index=False)
    man.append(dict(cell_id=name, n_b=int(r.n_b), dem=float(r.dem),
                    r2=round(r2, 4), top10=round(t10, 3), top50=round(t50, 3)))
mdf = pd.DataFrame(man)
mdf.to_csv(os.path.join(HERE, "grid_manifest.csv"), index=False)
print(f"wrote {len(mdf)} cell CSVs -> {OUTDIR}")
print(f"R2 range [{mdf.r2.min():.2f},{mdf.r2.max():.2f}]  top10 [{mdf.top10.min():.2f},{mdf.top10.max():.2f}]")
print(mdf.head(6).to_string(index=False))
