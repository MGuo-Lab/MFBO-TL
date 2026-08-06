"""FLOP PROFILING: hardware-independent computational cost per surrogate.

Counts the floating-point operations of one fit() and one predict()-over-the-pool
for each surrogate at 5 training-set fractions, using torch FlopCounterMode with
custom formulas added for the linear-algebra ops the GP family uses (Cholesky and
triangular/cholesky solves are NOT in FlopCounterMode's default table, so without
them the GP cost would be undercounted). matmul/addmm/bmm (NN layers, kernel
matmuls, variational ops) are covered by the default table.

The BO-loop FLOP cost is reconstructed offline as
    total = sum_{n=n_init..n_final} [ fit_FLOPs(n) + predict_FLOPs(n) ]
(one fit + one pool-prediction per iteration). FLOPs are hardware-independent, so
this is comparable across surrogates without the FP64/utilization/framework
confounds of wall-clock. Per-(model,bench) process isolation (DKL CUDA faults are
contained); the first fit of a few GP/DNN models dumps the per-op breakdown so we
can verify the linalg ops were actually captured.
"""
import sys, time, argparse, queue
from pathlib import Path
import pandas as pd
import multiprocessing as mp

REPO = Path(__file__).resolve().parents[1]

BENCHES = {
    'Branin-Fav':     dict(kind='synth', func='branin', dim=2, alpha=0.8, cost_ratio=0.1, f_star=0.397887, grid_size=50, n_hf=24, n_lf=259),
    'Branin-Unfav':   dict(kind='synth', func='branin', dim=2, alpha=0.1, cost_ratio=0.5, f_star=0.397887, grid_size=50, n_hf=24, n_lf=52),
    'Park-Fav':       dict(kind='synth', func='park',   dim=4, alpha=0.6, cost_ratio=0.1, f_star=0.0,      grid_size=10, n_hf=22, n_lf=234),
    'Park-Unfav':     dict(kind='synth', func='park',   dim=4, alpha=0.0, cost_ratio=0.5, f_star=0.0,      grid_size=10, n_hf=24, n_lf=52),
    'COFs':           dict(kind='chem', csv='cofs.csv',           cost_ratio=0.065, negate=True,  use_smiles=False, n_hf=15, n_lf=230),
    'FreeSolv':       dict(kind='chem', csv='freesolv.csv',       cost_ratio=0.1,   negate=False, use_smiles=True,  n_hf=24, n_lf=259),
    'Polarizability': dict(kind='chem', csv='polarizability.csv', cost_ratio=0.167, negate=True,  use_smiles=True,  n_hf=16, n_lf=83),
    'HOPV15':         dict(kind='chem', csv='hopv15.csv',         cost_ratio=0.1,   negate=True,  use_smiles=False, n_hf=15, n_lf=149),
    'Matbench-Gap':   dict(kind='chem', csv='matbench_gap.csv',   cost_ratio=0.05,  negate=False, use_smiles=False, n_hf=10, n_lf=199),
}
GP_MODELS  = ['MFGP', 'NARGP', 'DKL Multi-Fidelity', 'Sparse MFGP']
DNN_MODELS = ['Sequential', 'Progressive', 'Curriculum', 'Two-Stage Joint', 'DNGO-Joint',
              'DNGO-Gradient', 'Knowledge Distillation', 'Domain Adaptation (MMD)',
              'Soft Parameter Sharing', 'Pseudo-Labeling', 'Adapter']
MODEL_NAMES = GP_MODELS + DNN_MODELS
FRACTIONS = [0.1, 0.3, 0.5, 0.7, 1.0]
DUMP = {('Branin-Fav', m) for m in ['MFGP', 'Sparse MFGP', 'DKL Multi-Fidelity', 'Sequential']}


def _linalg_formulas():
    import torch
    a = torch.ops.aten

    def chol(a_shape, *args, out_shape=None, **kw):
        N = a_shape[-1]; return int(N) ** 3 // 3                       # Cholesky ~ N^3/3

    def solve(b_shape, a_shape, *args, out_shape=None, **kw):
        N = a_shape[-1]; K = b_shape[-1] if len(b_shape) > 1 else 1
        return 2 * int(N) * int(N) * int(K)                           # tri/chol solve ~ 2 N^2 K

    def solve_tri(a_shape, b_shape, *args, out_shape=None, **kw):
        N = a_shape[-1]; K = b_shape[-1] if len(b_shape) > 1 else 1
        return 2 * int(N) * int(N) * int(K)

    m = {}
    for name, fn in [('linalg_cholesky_ex', chol), ('cholesky', chol),
                     ('cholesky_solve', solve), ('triangular_solve', solve),
                     ('linalg_solve_triangular', solve_tri), ('linalg_solve', solve_tri)]:
        op = getattr(a, name, None)
        if op is not None:
            m[op.default] = fn
    return m


def _child(model_name, bench_name, seeds, q):
    sys.path.insert(0, str(REPO / 'src'))   # benchmark.py + baselines.py both live in src/
    import numpy as np, torch
    from torch.utils.flop_counter import FlopCounterMode
    from benchmark import (ChemistryBenchmark, SyntheticBenchmark, MFGP, Sequential, Progressive,
                           Curriculum, TwoStageJoint, DNGOJoint, DNGOGradient, KnowledgeDistillation,
                           DomainAdaptationMMD, SoftParameterSharing, PseudoLabeling, Adapter)
    from synthetic_functions import branin_hf, branin_lf, park_hf, park_lf
    from baselines import NARGP, DKLMultiFidelity, SparseMFGP
    REG = {'MFGP': MFGP, 'NARGP': NARGP, 'DKL Multi-Fidelity': DKLMultiFidelity, 'Sparse MFGP': SparseMFGP,
           'Sequential': Sequential, 'Progressive': Progressive, 'Curriculum': Curriculum,
           'Two-Stage Joint': TwoStageJoint, 'DNGO-Joint': DNGOJoint, 'DNGO-Gradient': DNGOGradient,
           'Knowledge Distillation': KnowledgeDistillation, 'Domain Adaptation (MMD)': DomainAdaptationMMD,
           'Soft Parameter Sharing': SoftParameterSharing, 'Pseudo-Labeling': PseudoLabeling, 'Adapter': Adapter}
    SYNTH = {'branin': (branin_hf, branin_lf), 'park': (park_hf, park_lf)}
    custom = _linalg_formulas()
    cfg = BENCHES[bench_name]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.init(); _ = torch.zeros(1, device=device)
    if cfg['kind'] == 'chem':
        bench = ChemistryBenchmark(bench_name, REPO / 'data' / cfg['csv'], cfg['cost_ratio'],
                                   cfg['use_smiles'], True, cfg['negate'])
    else:
        hf, lf = SYNTH[cfg['func']]
        bench = SyntheticBenchmark(bench_name, hf, lf, cfg['dim'], cfg['alpha'], cfg['cost_ratio'],
                                   cfg['f_star'], cfg['grid_size'])
    X, npool, dim = bench.X, len(bench.X), bench.X.shape[1]
    Cls = REG[model_name]
    first = True
    for seed in seeds:
        rng = np.random.RandomState(seed)
        for frac in FRACTIONS:
            n_hf = max(2, int(round(frac * cfg['n_hf']))); n_lf = max(2, int(round(frac * cfg['n_lf'])))
            perm = rng.permutation(npool); lf_i, hf_i = perm[:n_lf], perm[n_lf:n_lf + n_hf]
            rec = dict(benchmark=bench_name, model=model_name, seed=seed, fraction=frac,
                       n_hf=n_hf, n_lf=n_lf, n_train=n_hf + n_lf, pool=npool, dim=dim)
            try:
                model = Cls(dim, device=device)
                fc = FlopCounterMode(display=False, custom_mapping=custom)
                with fc:
                    model.fit(X[lf_i], bench.y_lf[lf_i], X[hf_i], bench.y_hf[hf_i])
                rec['fit_flops'] = int(fc.get_total_flops())
                fc2 = FlopCounterMode(display=False, custom_mapping=custom)
                with fc2:
                    model.predict(X)
                rec['predict_flops'] = int(fc2.get_total_flops())
                rec['ok'] = 1
                if first and (bench_name, model_name) in DUMP:
                    print(f'--- OP BREAKDOWN {bench_name}/{model_name} frac{frac} (fit) ---', flush=True)
                    for mod, ops in fc.flop_counts.items():
                        for op, fl in ops.items():
                            if fl > 0:
                                print(f'    {str(op):45s} {fl:,}', flush=True)
                    first = False
            except Exception as e:
                rec['fit_flops'] = -1; rec['predict_flops'] = -1; rec['ok'] = 0; rec['err'] = repr(e)[:140]
            q.put(rec)
    q.put(None)


def run_pair(model_name, bench_name, seeds, timeout):
    ctx = mp.get_context('spawn'); qq = ctx.Queue()
    p = ctx.Process(target=_child, args=(model_name, bench_name, seeds, qq))
    t0 = time.time(); p.start(); out = []
    while time.time() - t0 < timeout:
        try:
            r = qq.get(timeout=3.0)
        except queue.Empty:
            if not p.is_alive(): break
            continue
        if r is None: break
        out.append(r)
    p.join(10)
    if p.is_alive(): p.terminate(); p.join()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-seeds', type=int, default=2)
    ap.add_argument('--base-seed', type=int, default=42)
    ap.add_argument('--pair-timeout', type=int, default=2400)
    ap.add_argument('--output-dir', type=str, required=True)
    a = ap.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    csv = out / 'flop_profile.csv'
    seeds = [a.base_seed + i for i in range(a.n_seeds)]
    rows, done = [], set()
    if csv.exists():
        old = pd.read_csv(csv); rows = old.to_dict('records'); need = len(seeds) * len(FRACTIONS)
        for (m, b), g in old.groupby(['model', 'benchmark']):
            if (g['ok'] == 1).sum() >= need: done.add((m, b))
    pairs = [(m, b) for b in BENCHES for m in MODEL_NAMES if (m, b) not in done]
    print(f'flop-profile: {len(pairs)} pairs to run ({len(done)} done)', flush=True)
    for i, (m, b) in enumerate(pairs):
        res = run_pair(m, b, seeds, a.pair_timeout)
        rows = [r for r in rows if not (r['model'] == m and r['benchmark'] == b)] + res
        pd.DataFrame(rows).to_csv(csv, index=False)
        ff = [r['fit_flops'] for r in res if r.get('ok')]
        rng = f'{min(ff):.2e}-{max(ff):.2e}' if ff else 'FAILED'
        print(f'  {i+1}/{len(pairs)}  {b:14s} {m:22s}  fit_flops {rng}', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
