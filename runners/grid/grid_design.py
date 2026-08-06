#!/usr/bin/env python
"""Inverse-design the (R^2 x top10-agreement) grid on the Polariz substrate.

Dense sweep of the two knobs (n_b = bulk training size -> R^2; dem = gentle
demotion of the top-5% basin -> top10 agreement), measure (R^2, top10) for each
(avg over RF seeds), then for each TARGET cell on the user's 0.1-step grid assign
the nearest-reachable knob setting. Cells with no close knob = infeasible (empty).

Output: grid_cells.csv (target_r2,target_top10,n_b,dem,meas_r2,meas_top10,feasible)
+ feasibility heatmap. Login-safe (sklearn, ~5 min).
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SUB = os.path.join(REPO, "data", "polarizability_pca10.csv")
Q, KFOLDS, N_TREES, DEMOTE_Q = 0.05, 5, 80, 0.12

df = pd.read_csv(SUB)
X = df[[f"f{i}" for i in range(10)]].to_numpy(float)
y = df["HF"].to_numpy(float)
N = len(y)
nO = max(int(round(Q * N)), 50)
basin = np.argsort(y)[:nO]
basin_set = set(basin.tolist())
bulk_idx = np.array([i for i in range(N) if i not in basin_set])


def build(n_b, dem, seed):
    rng = np.random.RandomState(seed)
    pred = np.full(N, np.nan)
    for tr_pool, te in KFold(KFOLDS, shuffle=True, random_state=seed).split(np.arange(N)):
        s = set(tr_pool.tolist())
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


def metrics(pred):
    r2 = float(np.corrcoef(pred, y)[0, 1]) ** 2
    t10 = len(set(np.argsort(pred)[:10]) & set(np.argsort(y)[:10])) / 10
    return r2, t10


# ---- dense knob sweep (avg 2 RF seeds) ----
N_B = sorted(set(int(v) for v in np.unique(np.round(np.geomspace(8, 1050, 13)))))
DEM = list(np.round(np.linspace(0, 1, 12), 3))
sweep = []
for n_b in N_B:
    for dem in DEM:
        rs = [metrics(build(n_b, dem, sd)) for sd in (0, 1)]
        sweep.append(dict(n_b=n_b, dem=dem,
                          r2=np.mean([r[0] for r in rs]), top10=np.mean([r[1] for r in rs])))
sw = pd.DataFrame(sweep)
print(f"swept {len(sw)} (n_b,dem); reached R2 [{sw.r2.min():.2f},{sw.r2.max():.2f}] "
      f"top10 [{sw.top10.min():.2f},{sw.top10.max():.2f}]")

# ---- inverse-design to the target grid ----
R2_T = np.round(np.arange(0.1, 0.96, 0.1), 2)        # 0.1..0.9
T10_T = np.round(np.arange(0.0, 1.01, 0.1), 2)       # 0.0..1.0 (incl band-gap edge)
TOL = 0.11
rows = []
for r2t in R2_T:
    for t10t in T10_T:
        d = np.hypot(sw.r2 - r2t, sw.top10 - t10t)
        j = int(d.idxmin())
        feasible = bool(d[j] < TOL)
        rows.append(dict(target_r2=r2t, target_top10=t10t,
                         n_b=int(sw.n_b[j]), dem=float(sw.dem[j]),
                         meas_r2=round(float(sw.r2[j]), 3), meas_top10=round(float(sw.top10[j]), 3),
                         dist=round(float(d[j]), 3), feasible=feasible))
cells = pd.DataFrame(rows)
cells.to_csv(os.path.join(HERE, "grid_cells.csv"), index=False)
nfeas = int(cells.feasible.sum())
print(f"target grid {len(cells)} cells ({len(R2_T)}x{len(T10_T)}); feasible {nfeas}; "
      f"unique LFs {cells[cells.feasible].drop_duplicates(['n_b','dem']).shape[0]}")

# ---- feasibility heatmap ----
piv = cells.pivot(index="target_top10", columns="target_r2", values="feasible").astype(float)
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(piv.values, origin="lower", cmap="Greens", vmin=0, vmax=1, aspect="auto",
               extent=[R2_T.min()-.05, R2_T.max()+.05, T10_T.min()-.05, T10_T.max()+.05])
for _, r in cells.iterrows():
    ax.text(r.target_r2, r.target_top10, "o" if r.feasible else "x",
            ha="center", va="center", fontsize=8, color="k" if r.feasible else "r")
ax.set_xlabel("target global R$^2$"); ax.set_ylabel("target top-10 agreement")
ax.set_title(f"Grid feasibility on Polariz (o=reachable {nfeas}/{len(cells)}, x=infeasible)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "grid_feasibility.png"), dpi=150)
print(f"saved -> {os.path.join(HERE, 'grid_feasibility.png')}  and grid_cells.csv")
