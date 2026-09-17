"""Build the 126 controlled LF sources from the condition manifest."""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
SUB = os.path.join(REPO, 'data', 'polarizability_pca10.csv')
OUTDIR = os.path.join(REPO, 'data', 'grid_cells')
os.makedirs(OUTDIR, exist_ok=True)
(Q, KFOLDS, N_TREES, DEMOTE_Q, SEED) = (0.05, 5, 80, 0.12, 0)
df = pd.read_csv(SUB)
fcols = [f'f{i}' for i in range(10)]
X = df[fcols].to_numpy(float)
y = df['HF'].to_numpy(float)
N = len(y)
nO = max(int(round(Q * N)), 50)
basin = np.argsort(y)[:nO]
basin_set = set(basin.tolist())
bulk_idx = np.array([i for i in range(N) if i not in basin_set])

def build(n_b, dem, seed=SEED):
    rng = np.random.RandomState(seed)
    pred = np.full(N, np.nan)
    for (tr, te) in KFold(KFOLDS, shuffle=True, random_state=seed).split(np.arange(N)):
        s = set(tr.tolist())
        b = [i for i in bulk_idx if i in s]
        o = [i for i in basin if i in s]
        sel = list(rng.permutation(b)[:min(n_b, len(b))]) + o
        rf = RandomForestRegressor(n_estimators=N_TREES, n_jobs=1, random_state=seed, max_features='sqrt').fit(X[sel], y[sel])
        pred[te] = rf.predict(X[te])
    n_dem = int(round(dem * len(basin)))
    if n_dem:
        hit = rng.permutation(basin)[:n_dem]
        pred[hit] = np.quantile(pred, DEMOTE_Q) + 1e-06 * rng.randn(n_dem)
    return pred
def rf_noisy(n_b, seed):
    rng = np.random.RandomState(seed)
    pred = np.full(N, np.nan)
    for (tr, te) in KFold(KFOLDS, shuffle=True, random_state=seed).split(np.arange(N)):
        s = set(tr.tolist())
        b = [i for i in bulk_idx if i in s]
        o = [i for i in basin if i in s]
        sel = list(rng.permutation(b)[:min(n_b, len(b))]) + o
        rf = RandomForestRegressor(n_estimators=N_TREES, n_jobs=1, random_state=seed, max_features='sqrt').fit(X[sel], y[sel])
        pred[te] = rf.predict(X[te])
    return pred

def build_noisy(n_b, sigma, opt, seed=SEED):
    rng = np.random.RandomState(seed)
    pred = rf_noisy(n_b, seed)
    if sigma > 0:
        pred = pred + sigma * pred.std() * rng.randn(N)
    k = int(round(abs(opt) * len(basin)))
    if k:
        hit = rng.permutation(basin)[:k]
        if opt < 0:
            pred[hit] = np.quantile(pred, DEMOTE_Q) + 1e-06 * rng.randn(k)
        else:
            pred[hit] = y[hit]
    return pred

def rf_target(n_b, seed):
    basin = np.argsort(y)[:max(int(0.05 * N), 50)]
    bulk_for_rf = np.array([i for i in range(N) if i not in set(basin)])
    rng = np.random.RandomState(seed)
    pred = np.full(N, np.nan)
    for (tr, te) in KFold(KFOLDS, shuffle=True, random_state=seed).split(np.arange(N)):
        s = set(tr.tolist())
        b = [i for i in bulk_for_rf if i in s]
        o = [i for i in basin if i in s]
        sel = list(rng.permutation(b)[:min(n_b, len(b))]) + o
        rf = RandomForestRegressor(n_estimators=N_TREES, n_jobs=1, random_state=seed, max_features='sqrt').fit(X[sel], y[sel])
        pred[te] = rf.predict(X[te])
    return pred

def base_target(n_b, sigma, seed=SEED):
    rng = np.random.RandomState(seed)
    p = rf_target(n_b, seed)
    if sigma > 0:
        p = p + sigma * p.std() * rng.randn(N)
    return p

def target_top10(p, k):
    H10 = list(np.argsort(y)[:10])
    p = p.copy()
    rng = np.random.RandomState(SEED)
    keep = list(rng.permutation(H10)[:k])
    drop = [i for i in H10 if i not in set(keep)]
    lo = np.sort(p)[:max(15, k + 1)]
    p[keep] = p.min() - np.arange(1, k + 1) * (0.001 * (p.std() + 1e-09))
    if drop:
        p[drop] = np.quantile(p, 0.05) + 1e-06 * rng.randn(len(drop))
    return p

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cells', nargs='+')
    parser.add_argument('--outdir', default=OUTDIR)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    manifest = pd.read_csv(os.path.join(REPO, 'results/grid/grid_manifest.csv'))
    for row in manifest.itertuples():
        if args.cells and row.cell_id not in args.cells:
            continue
        values = df[fcols].copy()
        values['HF'] = y
        if pd.notna(row.dem):
            values['LF'] = build(int(row.n_b), float(row.dem))
        elif str(row.opt).startswith('target'):
            values['LF'] = target_top10(base_target(int(row.n_b), float(row.sigma)),
                                       int(str(row.opt)[6:]))
        else:
            values['LF'] = build_noisy(int(row.n_b), float(row.sigma), float(row.opt))
        values.to_csv(os.path.join(args.outdir, row.cell_id + '.csv'), index=False)
        print(row.cell_id)
