"""Profile surrogate fits and pool-wide predictions in FLOPs."""
import os, sys, time, argparse
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(HERE))
os.environ.setdefault('HF_HEAD', 'blr_replace')
os.environ.setdefault('LF_TARGET', 'hf_argmin')
import pools as POOLS
import run_tl as T
SCHED = {'Branin-Fav': (24, 259), 'Branin-Unfav': (24, 52), 'Park-Fav': (24, 256), 'Park-Unfav': (24, 52), 'COFs': (15, 230), 'FreeSolv': (24, 259), 'Polarizability': (16, 83), 'HOPV15': (15, 149), 'Matbench-Gap': (10, 199)}
INIT = {'Branin-Fav': (2, 25), 'Branin-Unfav': (2, 5), 'Park-Fav': (2, 25), 'Park-Unfav': (2, 5), 'COFs': (2, 23), 'FreeSolv': (2, 25), 'Polarizability': (2, 8), 'HOPV15': (2, 15), 'Matbench-Gap': (2, 20)}
GP_MODELS = ['MFGP', 'SparseMFGP', 'DKL']
TL_MODELS = T.MODELS
PUB = {'DKL': 'DKL Multi-Fidelity', 'SparseMFGP': 'Sparse MFGP', 'DNGOGradient': 'DNGO-Gradient', 'TwoStageJoint': 'Two-Stage Joint', 'DNGOJoint': 'DNGO-Joint', 'DomainAdaptationMMD': 'Domain Adaptation (MMD)', 'SoftParameterSharing': 'Soft Parameter Sharing'}
FRACTIONS = [0.1, 0.3, 0.5, 0.7, 1.0]

def _linalg_formulas():
    import torch
    a = torch.ops.aten

    def chol(a_shape, *args, out_shape=None, **kw):
        N = a_shape[-1]
        return int(N) ** 3 // 3

    def solve(b_shape, a_shape, *args, out_shape=None, **kw):
        N = a_shape[-1]
        K = b_shape[-1] if len(b_shape) > 1 else 1
        return 2 * int(N) * int(N) * int(K)

    def solve_tri(a_shape, b_shape, *args, out_shape=None, **kw):
        N = a_shape[-1]
        K = b_shape[-1] if len(b_shape) > 1 else 1
        return 2 * int(N) * int(N) * int(K)
    m = {}
    for (name, fn) in [('linalg_cholesky_ex', chol), ('cholesky', chol), ('cholesky_solve', solve), ('triangular_solve', solve), ('linalg_solve_triangular', solve_tri), ('linalg_solve', solve_tri)]:
        op = getattr(a, name, None)
        if op is not None:
            m[op.default] = fn
    return m

def blr_fit_flops(N, D, grid=5):
    return 2 * N * D * D + 2 * N * D + grid * (8 * D ** 3 + 2 * D * D + 2 * N * D * D + 2 * N * N * D + 3 * N) + 8 * D ** 3 + 2 * D * D

def blr_predict_flops(N, D):
    return 2 * N * D + 2 * N * D * D + 2 * N * D + 2 * N

def gp_cls(name):
    import benchmark as B, baselines as X
    return {'MFGP': B.MFGP, 'DKL': X.DKLMultiFidelity, 'SparseMFGP': X.SparseMFGP}[name]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-seeds', type=int, default=2)
    ap.add_argument('--out', default=str(REPO / 'outputs/flop_profile_blr'))
    ap.add_argument('--benches', nargs='*', default=list(SCHED))
    ap.add_argument('--models', nargs='*', default=GP_MODELS + TL_MODELS)
    a = ap.parse_args()
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '4')))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    csv = out / 'flop_profile.csv'
    rows = pd.read_csv(csv).to_dict('records') if csv.exists() else []
    done = {(r['benchmark'], r['model'], r['seed'], r['fraction']) for r in rows}
    custom = _linalg_formulas()
    dev = torch.device('cpu')
    for bn in a.benches:
        (bench, cfg) = POOLS.make_bench(bn, REPO / 'data')
        X = bench.X
        (npool, dim) = (len(X), X.shape[1])
        (nh_f, nl_f) = SCHED[bn]
        for mn in a.models:
            for seed in range(42, 42 + a.n_seeds):
                rng = np.random.RandomState(seed)
                for frac in FRACTIONS:
                    if (bn, PUB.get(mn, mn), seed, frac) in done:
                        continue
                    n_hf = max(2, int(round(frac * nh_f)))
                    n_lf = max(2, int(round(frac * nl_f)))
                    perm = rng.permutation(npool)
                    lf_i = perm[:min(n_lf, npool)]
                    hf_i = rng.permutation(npool)[:n_hf]
                    rec = dict(benchmark=bn, model=PUB.get(mn, mn), seed=seed, fraction=frac, n_hf=n_hf, n_lf=n_lf, n_train=n_hf + n_lf, pool=npool, dim=dim, numpy_fit_flops=0, numpy_predict_flops=0)
                    t0 = time.time()
                    try:
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        if mn in GP_MODELS:
                            model = gp_cls(mn)(dim, device=dev)
                            fc = FlopCounterMode(display=False, custom_mapping=custom)
                            with fc:
                                model.fit(X[lf_i], bench.y_lf[lf_i], X[hf_i], bench.y_hf[hf_i])
                            fc2 = FlopCounterMode(display=False, custom_mapping=custom)
                            with fc2:
                                model.predict(X)
                            rec['fit_flops'] = int(fc.get_total_flops())
                            rec['predict_flops'] = int(fc2.get_total_flops())
                        else:
                            model = T._cls(mn)(dim, device=dev)
                            fc = FlopCounterMode(display=False, custom_mapping=custom)
                            with fc:
                                model.fit(X[lf_i], bench.y_lf[lf_i], X[hf_i], bench.y_hf[hf_i])
                                head = T.Head(model, X[hf_i], bench.y_hf[hf_i])
                            D = len(head.m)
                            rec['numpy_fit_flops'] = int(blr_fit_flops(n_hf, D))
                            fc2 = FlopCounterMode(display=False, custom_mapping=custom)
                            with fc2:
                                head.predict(X)
                            rec['numpy_predict_flops'] = int(blr_predict_flops(npool, D))
                            rec['fit_flops'] = int(fc.get_total_flops()) + rec['numpy_fit_flops']
                            rec['predict_flops'] = int(fc2.get_total_flops()) + rec['numpy_predict_flops']
                        rec['ok'] = 1
                    except Exception as e:
                        rec['fit_flops'] = -1
                        rec['predict_flops'] = -1
                        rec['ok'] = 0
                        rec['err'] = repr(e)[:140]
                    rec['elapsed'] = round(time.time() - t0, 2)
                    rows.append(rec)
                    pd.DataFrame(rows).to_csv(csv, index=False)
                print(f'{bn} {mn} done ({len(rows)} rows)', flush=True)
    print('DONE', csv)
if __name__ == '__main__':
    main()
