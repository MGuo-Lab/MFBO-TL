'Run GP-base, GP-SV and GP-DKL.'
from __future__ import annotations
import argparse, os, sys, time, traceback
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'experiments' / 'extra_baselines'))
sys.path.insert(0, str(HERE))
from _shared_calibration import ece_gaussian, nll_gaussian
import pools as POOLS
from run_tl import fps, _append, EXT_BUDGETS, _load_picks, _replay
GP_ACQ = os.environ.get('GP_ACQ', 'ei')
assert GP_ACQ in ('ei', 'greedy')
CKPT_DIR = os.environ.get('CKPT_DIR', '')

def _cls(name):
    import benchmark as B, baselines as X
    return {'MFGP': B.MFGP, 'DKL': X.DKLMultiFidelity, 'SparseMFGP': X.SparseMFGP}[name]

def _lf_posterior(model_name, model, X):
    try:
        if model_name == 'MFGP':
            import torch
            X_fid = np.hstack([X, np.zeros((len(X), 1))])
            X_t = torch.tensor(X_fid, dtype=torch.float64).to(model.device)
            model.model.eval()
            with torch.no_grad():
                post = model.model.posterior(X_t)
                m = post.mean.cpu().numpy().ravel()
                s = post.variance.sqrt().cpu().numpy().ravel()
            return (np.asarray(m, float), np.maximum(np.asarray(s, float), 1e-06))
        (m, s) = model.predict_lf(X)
        return (np.asarray(m, float), np.maximum(np.asarray(s, float), 1e-06))
    except Exception:
        return None

def run_one(bench, cfg, model_name, seed, device):
    import torch
    from benchmark import expected_improvement
    cls = _cls(model_name)
    np.random.seed(seed)
    torch.manual_seed(seed)
    (X, y_hf, y_lf) = (bench.X, bench.y_hf, bench.y_lf)
    rho = bench.cost_ratio
    budget = float(EXT_BUDGETS.get(cfg['_name'], cfg['budget']))
    pct = (np.argsort(np.argsort(y_hf)) + 1) / bench.n_candidates
    init_budget = float(cfg.get('init_budget', os.environ.get('INIT_BUDGET', cfg['budget'])))
    n_init_lf = max(2, int(0.1 * init_budget * 0.5 / rho))
    n_init_hf = max(2, int(0.1 * init_budget * 0.5))
    init = fps(X, n_init_lf + n_init_hf, seed)
    lf_idx = set((int(i) for i in init[:n_init_lf]))
    hf_idx = set((int(i) for i in init[n_init_lf:n_init_lf + n_init_hf]))
    cur = len(lf_idx) * rho + len(hf_idx) * 1.0
    lf_per_hf = max(1, int(1.0 / rho))
    lfc = it = 0
    regrets = [float(max(0.0, y_hf[list(hf_idx)].min() - bench.f_star))]
    budgets = [cur]
    picks = []
    cal = []
    plog = [dict(step=0, fid='L0', idx=int(i), budget=cur) for i in init[:n_init_lf]] + [dict(step=0, fid='H0', idx=int(i), budget=cur) for i in init[n_init_lf:n_init_lf + n_init_hf]]
    resumed_from = np.nan
    pk = _load_picks(cfg['_name'], model_name, seed, f'gp_{GP_ACQ}')
    if pk is not None:
        (cur, lfc, it, regrets, budgets, picks) = _replay(pk, y_hf, bench.f_star, pct, rho, lf_idx, hf_idx)
        plog = [dict(step=int(r.step), fid=r.fid, idx=int(r.idx), budget=float(r.budget)) for r in pk.itertuples()]
        resumed_from = cur
        print(f"  [resume] {cfg['_name']} {model_name} s{seed}: replayed {it} picks to budget {cur:.3f}", flush=True)
    max_it = int(budget / rho) + 1000
    fit_failed_at = np.nan
    while cur < budget and it < max_it:
        it += 1
        rem = budget - cur
        if rem >= 1.0:
            eval_hf = not (rem >= rho and lfc < lf_per_hf)
        elif rem >= rho:
            eval_hf = False
        else:
            break
        L = np.array(sorted(lf_idx))
        Hh = np.array(sorted(hf_idx))
        model = cls(X.shape[1], device=device)
        try:
            model.fit(X[L], y_lf[L], X[Hh], y_hf[Hh])
        except Exception as e:
            fit_failed_at = cur
            print(f'  [fit-fail] budget {cur:.3f} n_lf={len(L)} n_hf={len(Hh)}: {type(e).__name__}', flush=True)
            break
        (mean, std) = model.predict(X)
        mean = np.asarray(mean, float)
        std = np.asarray(std, float)
        lfp = _lf_posterior(model_name, model, X)
        if lfp is not None:
            held = np.setdiff1d(np.arange(bench.n_candidates), np.array(sorted(lf_idx | hf_idx)))
            if len(held) >= 5:
                (mh, sh) = (lfp[0][held], lfp[1][held])
                cal.append((float(ece_gaussian(y_lf[held], mh, sh)), float(nll_gaussian(y_lf[held], mh, sh)), float(np.mean(sh))))
        if eval_hf:
            mm = mean.copy()
            mm[list(hf_idx)] = np.inf
            nxt = int(np.argmin(mm))
            hf_idx.add(nxt)
            cur += 1.0
            lfc = 0
            picks.append(float(pct[nxt]))
            plog.append(dict(step=it, fid='H', idx=nxt, budget=cur))
        else:
            if GP_ACQ == 'greedy':
                mm = mean.copy()
                mm[list(lf_idx)] = np.inf
                nxt = int(np.argmin(mm))
            else:
                ei = expected_improvement(mean, std, float(y_hf[list(hf_idx)].min()))
                ei = np.asarray(ei, float).copy()
                ei[list(lf_idx)] = -np.inf
                nxt = int(np.argmax(ei))
            lf_idx.add(nxt)
            cur += rho
            lfc += 1
            plog.append(dict(step=it, fid='L', idx=nxt, budget=cur))
        regrets.append(float(max(0.0, y_hf[list(hf_idx)].min() - bench.f_star)))
        budgets.append(cur)
        if CKPT_DIR and it % 50 == 0:
            cd = Path(CKPT_DIR) / f'gp_{GP_ACQ}' / 'cells'
            cd.mkdir(parents=True, exist_ok=True)
            cf = cd / f"picks_{cfg['_name']}_{model_name}_s{seed}.csv"
            pd.DataFrame([dict(benchmark=cfg['_name'], model=model_name, seed=seed, **q) for q in plog]).to_csv(str(cf) + '.tmp', index=False)
            os.replace(str(cf) + '.tmp', cf)
    tr = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    if cal:
        (ece_m, nll_m, sh_m) = (float(np.mean([c[i] for c in cal])) for i in range(3))
        (ece_f, nll_f, sh_f) = cal[-1]
    else:
        ece_m = nll_m = sh_m = ece_f = nll_f = sh_f = np.nan
    return dict(final_regret=regrets[-1], auc=float(tr(regrets, budgets) / max(budgets[-1] - budgets[0], 1e-09)), n_hf=len(hf_idx), n_lf=len(lf_idx), pick_prec_top5=float(np.mean([p <= 0.05 + 1e-12 for p in picks])) if picks else np.nan, budgets=budgets, regrets=regrets, plog=plog, resumed_from=resumed_from, budget_used=budget, fit_failed_at=fit_failed_at, ece_lf_mean=ece_m, nll_lf_mean=nll_m, sharp_lf_mean=sh_m, ece_lf_final=ece_f, nll_lf_final=nll_f, sharp_lf_final=sh_f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benches', nargs='*', default=list(POOLS.PAPER))
    ap.add_argument('--models', nargs='*', default=['MFGP', 'SparseMFGP', 'DKL'])
    ap.add_argument('--seeds', type=int, nargs='*', default=list(range(42, 82)))
    ap.add_argument('--budget', type=float)
    ap.add_argument('--outdir', default=str(REPO / 'outputs'))
    args = ap.parse_args()
    if args.budget is not None:
        EXT_BUDGETS.update({bn: args.budget for bn in args.benches})
    import torch
    dev = torch.device(os.environ.get('GP_DEVICE', 'cpu'))
    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '4')))
    out = Path(args.outdir) / f'gp_{GP_ACQ}'
    (out / 'cells').mkdir(parents=True, exist_ok=True)
    tag = f's{min(args.seeds)}-{max(args.seeds)}' if len(args.seeds) > 1 else f's{args.seeds[0]}'
    for bn in args.benches:
        (b, cfg) = POOLS.make_bench(bn, REPO / 'data')
        cfg = dict(cfg)
        cfg['_name'] = bn
        print(f"{bn}: N={b.n_candidates} dim={b.dim} budget={cfg['budget']} ext_budget={EXT_BUDGETS.get(bn, cfg['budget'])}", flush=True)
        for ml in args.models:
            sf = out / 'cells' / f'summary_{bn}_{ml}_{tag}.csv'
            tf = out / 'cells' / f'traj_{bn}_{ml}_{tag}.csv'
            pf = out / 'cells' / f'picks_{bn}_{ml}_{tag}.csv'
            done = set(pd.read_csv(sf)['seed'].tolist()) if sf.exists() else set()
            for root in os.environ.get('SKIP_DIRS', '').split(':'):
                f2 = Path(root) / f'gp_{GP_ACQ}' / 'cells' / sf.name
                if root and f2.exists():
                    try:
                        done |= set(pd.read_csv(f2)['seed'].tolist())
                    except Exception:
                        pass
            for seed in args.seeds:
                if seed in done:
                    continue
                t0 = time.time()
                try:
                    r = run_one(b, cfg, ml, seed, dev)
                except Exception:
                    print(f'[FAIL] {bn} {ml} s{seed}', flush=True)
                    traceback.print_exc()
                    continue
                _append(sf, [dict(benchmark=bn, model=ml, seed=seed, final_regret=r['final_regret'], auc=r['auc'], n_hf=r['n_hf'], n_lf=r['n_lf'], pick_prec_top5=r['pick_prec_top5'], ece_lf_mean=r['ece_lf_mean'], nll_lf_mean=r['nll_lf_mean'], sharp_lf_mean=r['sharp_lf_mean'], ece_lf_final=r['ece_lf_final'], nll_lf_final=r['nll_lf_final'], sharp_lf_final=r['sharp_lf_final'], elapsed=round(time.time() - t0, 1), gp_acq=GP_ACQ, budget_used=r['budget_used'], resumed_from=r['resumed_from'], fit_failed_at=r['fit_failed_at'])])
                _append(pf, [dict(benchmark=bn, model=ml, seed=seed, **p) for p in r['plog']])
                _append(tf, [dict(benchmark=bn, model=ml, seed=seed, budget=bb, regret=rr) for (bb, rr) in zip(r['budgets'], r['regrets'])])
                print(f"{bn} {ml} s{seed}: final={r['final_regret']:.4f} auc={r['auc']:.4f} ece={r['ece_lf_mean']:.3f} ({time.time() - t0:.0f}s)", flush=True)
if __name__ == '__main__':
    main()
