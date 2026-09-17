'Run the HF-only GP and random-sampling controls.'
from __future__ import annotations
import argparse, os, sys, time, traceback
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(HERE))
import pools as POOLS
from run_tl import fps, _append
SF_ARM = os.environ.get('SF_ARM', 'ei')
assert SF_ARM in ('ei', 'random')
SF_INIT = 'own'

class SFGP:

    def __init__(self, input_dim, device=None):
        import torch
        self.input_dim = input_dim
        self.device = device or torch.device('cpu')

    def fit(self, X_hf, y_hf):
        import torch
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood
        train_X = torch.tensor(np.asarray(X_hf, float), dtype=torch.float64).to(self.device)
        train_Y = torch.tensor(np.asarray(y_hf, float).ravel(), dtype=torch.float64).unsqueeze(-1).to(self.device)
        self.model = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1)).to(self.device)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

    def predict(self, X):
        import torch
        X_t = torch.tensor(np.asarray(X, float), dtype=torch.float64).to(self.device)
        self.model.eval()
        with torch.no_grad():
            post = self.model.posterior(X_t)
            m = post.mean.cpu().numpy().ravel()
            s = post.variance.sqrt().cpu().numpy().ravel()
        return (np.asarray(m, float), np.maximum(np.asarray(s, float), 1e-06))

def run_one(bench, cfg, seed, device):
    import torch
    from benchmark import expected_improvement
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed + 1000)
    (X, y_hf) = (bench.X, bench.y_hf)
    (rho, budget) = (bench.cost_ratio, cfg['budget'])
    pct = (np.argsort(np.argsort(y_hf)) + 1) / bench.n_candidates
    init_budget = float(cfg.get('init_budget', os.environ.get('INIT_BUDGET', budget)))
    n_init_lf = max(2, int(0.1 * init_budget * 0.5 / rho))
    n_init_hf = max(2, int(0.1 * init_budget * 0.5))
    init = fps(X, n_init_lf + n_init_hf, seed)
    hf_idx = set((int(i) for i in init[n_init_lf:n_init_lf + n_init_hf]))
    cur = float(len(hf_idx))
    it = 0
    regrets = [float(max(0.0, y_hf[list(hf_idx)].min() - bench.f_star))]
    budgets = [cur]
    picks = []
    nfail = 0
    while cur + 1.0 <= budget + 1e-09 and it < 500:
        it += 1
        Hh = np.array(sorted(hf_idx))
        if SF_ARM == 'random':
            cand = np.setdiff1d(np.arange(bench.n_candidates), Hh)
            nxt = int(rng.choice(cand))
        else:
            try:
                model = SFGP(X.shape[1], device=device)
                model.fit(X[Hh], y_hf[Hh])
                (mean, std) = model.predict(X)
            except Exception:
                nfail += 1
                cand = np.setdiff1d(np.arange(bench.n_candidates), Hh)
                nxt = int(rng.choice(cand))
            else:
                if SF_ARM == 'greedy':
                    mm = mean.copy()
                    mm[Hh] = np.inf
                    nxt = int(np.argmin(mm))
                else:
                    ei = np.asarray(expected_improvement(mean, std, float(y_hf[Hh].min())), float).copy()
                    ei[Hh] = -np.inf
                    nxt = int(np.argmax(ei))
        hf_idx.add(nxt)
        cur += 1.0
        picks.append(float(pct[nxt]))
        regrets.append(float(max(0.0, y_hf[list(hf_idx)].min() - bench.f_star)))
        budgets.append(cur)
    tr = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return dict(final_regret=regrets[-1], auc=float(tr(regrets, budgets) / max(budgets[-1] - budgets[0], 1e-09)), n_hf=len(hf_idx), n_lf=0, pick_prec_top5=float(np.mean([p <= 0.05 + 1e-12 for p in picks])) if picks else np.nan, budgets=budgets, regrets=regrets, nfail=nfail)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benches', nargs='*', default=list(POOLS.PAPER))
    ap.add_argument('--seeds', type=int, nargs='*', default=list(range(42, 82)))
    ap.add_argument('--budget', type=float)
    ap.add_argument('--outdir', default=str(REPO / 'outputs'))
    args = ap.parse_args()
    import torch
    dev = torch.device('cpu')
    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '1')))
    ml = f'SF{SF_ARM.upper()}' if SF_ARM != 'random' else 'RandomHF'
    out = Path(args.outdir) / f'sf_{SF_ARM}__{SF_INIT}'
    (out / 'cells').mkdir(parents=True, exist_ok=True)
    tag = f's{min(args.seeds)}-{max(args.seeds)}' if len(args.seeds) > 1 else f's{args.seeds[0]}'
    for bn in args.benches:
        (b, cfg) = POOLS.make_bench(bn, REPO / 'data')
        cfg = dict(cfg, budget=args.budget if args.budget is not None else cfg['budget'])
        print(f"{bn}: N={b.n_candidates} dim={b.dim} budget={cfg['budget']}", flush=True)
        sf = out / 'cells' / f'summary_{bn}_{ml}_{tag}.csv'
        tf = out / 'cells' / f'traj_{bn}_{ml}_{tag}.csv'
        done = set(pd.read_csv(sf)['seed'].tolist()) if sf.exists() else set()
        for seed in args.seeds:
            if seed in done:
                continue
            t0 = time.time()
            try:
                r = run_one(b, cfg, seed, dev)
            except Exception:
                print(f'[FAIL] {bn} {ml} s{seed}', flush=True)
                traceback.print_exc()
                continue
            _append(sf, [dict(benchmark=bn, model=ml, seed=seed, final_regret=r['final_regret'], auc=r['auc'], n_hf=r['n_hf'], n_lf=0, pick_prec_top5=r['pick_prec_top5'], nfail=r['nfail'], elapsed=round(time.time() - t0, 1), sf_arm=SF_ARM, sf_init=SF_INIT)])
            _append(tf, [dict(benchmark=bn, model=ml, seed=seed, budget=bb, regret=rr) for (bb, rr) in zip(r['budgets'], r['regrets'])])
            print(f"{bn} {ml}/{SF_INIT} s{seed}: final={r['final_regret']:.4f} auc={r['auc']:.4f} nhf={r['n_hf']} fail={r['nfail']} ({time.time() - t0:.0f}s)", flush=True)
if __name__ == '__main__':
    main()
