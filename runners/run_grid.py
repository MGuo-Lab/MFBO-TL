"""Run the controlled fidelity-quality grid."""
from pathlib import Path
import argparse, os, sys, time
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pools
import run_tl
GP_FAMILY = {'MFGP', 'DKL', 'SparseMFGP'}

def _cls(name):
    import benchmark, baselines
    return {'MFGP': benchmark.MFGP, 'DKL': baselines.DKLMultiFidelity, 'SparseMFGP': baselines.SparseMFGP}[name]

def fps(X, n, seed):
    from scipy.spatial.distance import cdist
    np.random.seed(seed)
    sel = [np.random.randint(len(X))]
    mind = cdist(X, X[sel]).min(axis=1)
    mind[sel] = -np.inf
    for _ in range(n - 1):
        nx = int(np.argmax(mind))
        sel.append(nx)
        mind = np.minimum(mind, np.linalg.norm(X - X[nx], axis=1))
        mind[nx] = -np.inf
    return sel

def run_one(bench, cfg, model_name, seed, device):
    import torch
    from benchmark import expected_improvement
    cls = _cls(model_name)
    is_gp = model_name in GP_FAMILY
    np.random.seed(seed)
    torch.manual_seed(seed)
    (X, y_hf, y_lf) = (bench.X, bench.y_hf, bench.y_lf)
    (rho, budget) = (bench.cost_ratio, cfg['budget'])
    pct = (np.argsort(np.argsort(y_hf)) + 1) / bench.n_candidates
    init_budget = float(os.environ.get('INIT_BUDGET', budget))
    n_init_lf = max(2, int(0.1 * init_budget * 0.5 / rho))
    n_init_hf = max(2, int(0.1 * init_budget * 0.5))
    init = fps(X, n_init_lf + n_init_hf, seed)
    lf_idx = set(init[:n_init_lf])
    hf_idx = set(init[n_init_lf:n_init_lf + n_init_hf])
    cur = len(lf_idx) * rho + len(hf_idx) * 1.0
    lf_per_hf = max(1, int(1.0 / rho))
    lfc = it = 0
    regrets = [float(max(0.0, y_hf[list(hf_idx)].min() - bench.f_star))]
    budgets = [cur]
    picks = []
    max_it = int(budget / rho) + 1000
    while cur < budget and it < max_it:
        it += 1
        rem = budget - cur
        if rem >= 1.0:
            eval_hf = not (rem >= rho and lfc < lf_per_hf)
        elif rem >= rho:
            eval_hf = False
        else:
            break
        model = cls(X.shape[1], device=device)
        model.fit(X[np.array(sorted(lf_idx))], y_lf[np.array(sorted(lf_idx))], X[np.array(sorted(hf_idx))], y_hf[np.array(sorted(hf_idx))])
        if eval_hf:
            mean = np.asarray(model.predict(X)[0], dtype=float)
            mm = mean.copy()
            mm[list(hf_idx)] = np.inf
            nxt = int(np.argmin(mm))
            hf_idx.add(nxt)
            cur += 1.0
            lfc = 0
            picks.append(float(pct[nxt]))
        else:
            (mean, std) = model.predict(X)
            ybest = y_hf[list(hf_idx)].min()
            ei = expected_improvement(np.asarray(mean, float), np.asarray(std, float), ybest)
            ei[list(lf_idx)] = -np.inf
            nxt = int(np.argmax(ei))
            lf_idx.add(nxt)
            cur += rho
            lfc += 1
        regrets.append(float(max(0.0, y_hf[list(hf_idx)].min() - bench.f_star)))
        budgets.append(cur)
    _trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    auc = float(_trapz(regrets, budgets) / max(budgets[-1] - budgets[0], 1e-09))
    return dict(final_regret=regrets[-1], auc=auc, n_hf=len(hf_idx), n_lf=len(lf_idx), pick_prec_top5=float(np.mean([p <= 0.05 + 1e-12 for p in picks])) if picks else np.nan, budgets=budgets, regrets=regrets)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cells', nargs='+', default=[f'cell_{i:02d}' for i in range(126)])
    parser.add_argument('--models', nargs='+', choices=sorted(GP_FAMILY) + run_tl.MODELS, default=sorted(GP_FAMILY) + run_tl.MODELS)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(42, 52)))
    parser.add_argument('--outdir', type=Path, default=ROOT / 'outputs/grid')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '1')))
    os.environ['INIT_BUDGET'] = '10'
    (args.outdir / 'cells').mkdir(parents=True, exist_ok=True)
    for cell in args.cells:
        (bench, cfg) = pools.make_bench(cell, ROOT / 'data')
        cfg['_name'] = cell
        summary = args.outdir / 'cells' / f'summary_{cell}.csv'
        done = set()
        if summary.exists():
            done = set(pd.read_csv(summary)[['model', 'seed']].itertuples(index=False, name=None))
        for model in args.models:
            for seed in args.seeds:
                if (model, seed) in done:
                    continue
                result = (run_one if model in GP_FAMILY else run_tl.run_one)(bench, cfg, model, seed, torch.device('cpu'))
                row = dict(benchmark=cell, model=model, seed=seed, final_regret=result['final_regret'], auc=result['auc'])
                run_tl._append(args.outdir / 'cells' / f'summary_{cell}.csv', [row])
                run_tl._append(args.outdir / 'cells' / f'traj_{cell}.csv', [dict(benchmark=cell, model=model, seed=seed, budget=b, regret=r) for (b, r) in zip(result['budgets'], result['regrets'])])
                print(cell, model, seed, result['final_regret'], flush=True)
if __name__ == '__main__':
    main()
