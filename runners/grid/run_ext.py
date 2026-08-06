#!/usr/bin/env python
"""Matbench-Gap BUDGET-EXTENSION variant of run_fix6.py (both bugs fixed).

Verbatim copy of experiments/fix6_promote/run_fix6.py with ONE addition: the
INIT_BUDGET env decouples the warmup size from the run budget (see run_one).
Everything else (models, acquisition, round-robin schedule, FPS init, masking,
promotion) is identical, so this reproduces fix6 when run at the native budget.

Why this exists: at the native budget=20, Matbench-Gap is truncated -- every
surrogate is still descending (none has reached the optimum) and the best TL
leads the best GP only narrowly. This runner extends the SAME protocol to
HF_BUDGET=50/100 with INIT_BUDGET=20 (fixed budget-20 warmup) to test whether
that early TL lead persists or whether the GP family catches up given budget.

Bugs fixed (inherited from fix6):
  (1) numpy-2.4.1 random-LF fallback -> REAL LF acquisition runs.
  (2) LF u HF masking -> PROMOTION ON: an LF-screened candidate can still be
      HF-evaluated. Each fidelity masks only ITSELF.

Driven by submit_matbench_ext.sh: a per-seed SLURM array (one seed = one task =
all models on one GPU, so paired TL-vs-GP comparisons stay on identical
hardware) writing per-seed outdirs (no concurrent-append race).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent               # <repo>/runners/grid -> <repo>
sys.path.insert(0, str(REPO / 'src'))   # benchmark.py + baselines.py both live in src/

SEEDS = list(range(42, 52))   # 10 seeds
BENCHES = {
    'Branin-Fav': dict(csv='synthetic_branin_fav.csv', cost_ratio=0.1, use_smiles=False, negate=False, budget=50),
    'Branin-Unfav': dict(csv='synthetic_branin_unfav.csv', cost_ratio=0.5, use_smiles=False, negate=False, budget=50),
    'Park-Fav': dict(csv='synthetic_park_fav.csv', cost_ratio=0.1, use_smiles=False, negate=False, budget=50),
    'Park-Unfav': dict(csv='synthetic_park_unfav.csv', cost_ratio=0.5, use_smiles=False, negate=False, budget=50),
    'COFs': dict(csv='cofs.csv', cost_ratio=0.065, use_smiles=False, negate=True, budget=30),
    'FreeSolv': dict(csv='freesolv.csv', cost_ratio=0.1, use_smiles=True, negate=False, budget=50),
    'Polarizability': dict(csv='polarizability.csv', cost_ratio=0.167, use_smiles=True, negate=True, budget=30),
    'HOPV15': dict(csv='hopv15.csv', cost_ratio=0.1, use_smiles=False, negate=True, budget=30),
    'Matbench-Gap': dict(csv='matbench_gap.csv', cost_ratio=0.05, use_smiles=False, negate=False, budget=20),
    # Budget-crossover (R2, N) study: two NEW high-N gap benchmarks that hold N
    # in the Matbench-Gap regime (~3.3-3.7k) and vary only the LF-HF agreement.
    # ExptGap-Hautier = LOW R2 (0.41): experiment HF vs MP-PBE LF, Hautier
    #   compilation; an independent replication of the low-R2 Matbench-Gap corner.
    # JARVIS-gap = HIGH R2 (0.86): OptB88vdW LF vs TBmBJ HF, both DFT -> high
    #   agreement; the control where the hypothesis predicts the GP keeps its edge.
    # Objective/featurization are identical to Matbench-Gap (|gap-1.4|, Magpie->
    # PCA10), so the comparison isolates R2. Built by benchmarks/{expt_gap_pbe,
    # jarvis_gap}.py. rho matches the fidelity cost (PBE-vs-expt 0.05; mBJ-vs-OPT 0.1).
    'ExptGap-Hautier': dict(csv='expt_gap_pbe.csv', cost_ratio=0.05, use_smiles=False, negate=False, budget=20),
    'JARVIS-gap': dict(csv='jarvis_gap.csv', cost_ratio=0.1, use_smiles=False, negate=False, budget=20),
    # Elastic shear-modulus family (matbench_log_gvrh HF; LF = an MLIP's predicted
    # shear modulus). A CONTROLLED R2 sweep: identical HF/optimum/N/structures,
    # only the LF MLIP changes -> R2 = 0.28/0.51/0.55/0.77/0.80. Objective =
    # maximize log10(G_VRH) (negate handled in the signed CSV). Built on the l40s
    # VM by mlip_elastic/gen_mlip_elastic.py + benchmarks/elastic_modulus.py.
    'Elastic-CHGNet':    dict(csv='elastic_chgnet_shear.csv',    cost_ratio=0.05, use_smiles=False, negate=False, budget=20),
    'Elastic-MACE':      dict(csv='elastic_mace_shear.csv',      cost_ratio=0.05, use_smiles=False, negate=False, budget=20),
    'Elastic-SevenNet':  dict(csv='elastic_sevennet_shear.csv',  cost_ratio=0.05, use_smiles=False, negate=False, budget=20),
    'Elastic-CHGMatPES': dict(csv='elastic_chgmatpes_shear.csv', cost_ratio=0.05, use_smiles=False, negate=False, budget=20),
    'Elastic-MatterSim': dict(csv='elastic_mattersim_shear.csv', cost_ratio=0.05, use_smiles=False, negate=False, budget=20),
}
GP_FAMILY = {'MFGP', 'NARGP', 'DKL', 'SparseMFGP', 'TLDeepKernelJoint',
             'TLDKJ_p0', 'TLDKJ_p200'}
TL_FAMILY = {'Sequential', 'Curriculum', 'DNGOGradient', 'Progressive',
             'TwoStageJoint', 'DNGOJoint', 'KnowledgeDistillation',
             'DomainAdaptationMMD', 'SoftParameterSharing', 'PseudoLabeling',
             'Adapter', 'TLDeepKernel', 'DV_HF', 'HFBLR_Curr', 'HFBLR_Seq',
             'CalibTunedSeq', 'Progressive_h32', 'Progressive_h128'}
MODELS = ['MFGP', 'NARGP', 'DKL', 'Sequential', 'Curriculum', 'DNGOGradient']
# HF acquisition: 'argmin' (fix6 default, argmin posterior mean) or 'ei'
# (expected improvement on the HF posterior; requires a model with calibrated
# HF std, i.e. GP family or TLDeepKernel). Set via env so a single runner serves
# both protocols; EI runs MUST use a separate --outdir to avoid clobbering.
ACQ_HF = os.environ.get('ACQ_HF', 'argmin')


def _cls(name):
    import benchmark as B
    import baselines as X
    return {'MFGP': B.MFGP, 'NARGP': X.NARGP, 'DKL': X.DKLMultiFidelity,
            'SparseMFGP': X.SparseMFGP,
            'Sequential': B.Sequential, 'Curriculum': B.Curriculum,
            'DNGOGradient': B.DNGOGradient, 'Progressive': B.Progressive,
            'Progressive_h32': (lambda d, device=None: B.Progressive(d, hidden_dim=32, device=device)),
            'Progressive_h128': (lambda d, device=None: B.Progressive(d, hidden_dim=128, device=device)),
            'TwoStageJoint': B.TwoStageJoint, 'DNGOJoint': B.DNGOJoint,
            'KnowledgeDistillation': B.KnowledgeDistillation,
            'DomainAdaptationMMD': B.DomainAdaptationMMD,
            'SoftParameterSharing': B.SoftParameterSharing,
            'PseudoLabeling': B.PseudoLabeling, 'Adapter': B.Adapter,
            'TLDeepKernel': X.TLDeepKernel,
            'TLDeepKernelJoint': X.TLDeepKernelJoint,
            # within-method transfer-init ablation: p0 == plain DKL from scratch
            # (0 pretrain epochs), p200 == transfer-initialised. Identical seeds
            # and code path -> isolates the LF-pretrain init.
            'TLDKJ_p0': (lambda d, device=None: X.TLDeepKernelJoint(d, device=device, pretrain_epochs=0)),
            'TLDKJ_p200': (lambda d, device=None: X.TLDeepKernelJoint(d, device=device, pretrain_epochs=200)),
            'DV_HF': X.DV_HF,
            'HFBLR_Curr': (lambda d, device=None: X.HFBLR_TL(d, device=device, base='Curriculum')),
            'HFBLR_Seq': (lambda d, device=None: X.HFBLR_TL(d, device=device, base='Sequential')),
            'CalibTunedSeq': X.CalibTunedSeq,
            }[name]


def fps(X, n, seed):
    from scipy.spatial.distance import cdist
    np.random.seed(seed)
    sel = [np.random.randint(len(X))]
    mind = cdist(X, X[sel]).min(axis=1); mind[sel] = -np.inf
    for _ in range(n - 1):
        nx = int(np.argmax(mind)); sel.append(nx)
        mind = np.minimum(mind, np.linalg.norm(X - X[nx], axis=1)); mind[nx] = -np.inf
    return sel


def run_one(bench, cfg, model_name, seed, device):
    import torch
    from benchmark import expected_improvement
    cls = _cls(model_name)
    is_gp = model_name in GP_FAMILY
    np.random.seed(seed); torch.manual_seed(seed)

    X, y_hf, y_lf = bench.X, bench.y_hf, bench.y_lf
    rho, budget = bench.cost_ratio, cfg['budget']
    pct = (np.argsort(np.argsort(y_hf)) + 1) / bench.n_candidates

    # INIT_BUDGET decouples the initial-design (warmup) size from the run
    # budget. Defaults to `budget` -> behaviour identical to run_fix6.py. Set
    # INIT_BUDGET=20 alongside HF_BUDGET=50/100 so an extended-budget run keeps
    # the SAME 20-LF/2-HF warmup as the native budget-20 run: the trajectory is
    # then a clean continuation whose budget<=20 segment must reproduce the
    # existing fix6 budget-20 means (built-in validation).
    init_budget = float(os.environ.get('INIT_BUDGET', budget))
    n_init_lf = max(2, int(0.1 * init_budget * 0.5 / rho))
    n_init_hf = max(2, int(0.1 * init_budget * 0.5))
    init = fps(X, n_init_lf + n_init_hf, seed)
    lf_idx = set(init[:n_init_lf]); hf_idx = set(init[n_init_lf:n_init_lf + n_init_hf])

    cur = len(lf_idx) * rho + len(hf_idx) * 1.0
    lf_per_hf = max(1, int(1.0 / rho)); lfc = it = 0
    regrets = [float(max(0.0, y_hf[list(hf_idx)].min() - bench.f_star))]
    budgets = [cur]; picks = []

    # Iteration cap must scale with the budget: at rho=0.05 a budget-100 run
    # needs ~1000 round-robin steps, so the original flat `it < 500` silently
    # truncated long runs at budget ~50. Bound by the all-LF worst case + slack.
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
        model.fit(X[np.array(sorted(lf_idx))], y_lf[np.array(sorted(lf_idx))],
                  X[np.array(sorted(hf_idx))], y_hf[np.array(sorted(hf_idx))])
        if eval_hf:
            if ACQ_HF == 'ei':
                m, s = model.predict(X)
                m = np.asarray(m, dtype=float); s = np.asarray(s, dtype=float)
                ybest = y_hf[list(hf_idx)].min()
                ei = expected_improvement(m, s, ybest)
                ei[list(hf_idx)] = -np.inf                    # PROMOTION: HF-only mask
                nxt = int(np.argmax(ei))
            else:
                mean = np.asarray(model.predict(X)[0], dtype=float)
                mm = mean.copy(); mm[list(hf_idx)] = np.inf   # PROMOTION: HF-only mask
                nxt = int(np.argmin(mm))
            hf_idx.add(nxt); cur += 1.0; lfc = 0; picks.append(float(pct[nxt]))
        else:
            if is_gp:
                mean, std = model.predict(X); ybest = y_hf[list(hf_idx)].min()
            else:
                mean, std = model.predict_lf(X); ybest = y_lf[list(lf_idx)].min()
            ei = expected_improvement(np.asarray(mean, float), np.asarray(std, float), ybest)
            ei[list(lf_idx)] = -np.inf
            nxt = int(np.argmax(ei))
            lf_idx.add(nxt); cur += rho; lfc += 1
        regrets.append(float(max(0.0, y_hf[list(hf_idx)].min() - bench.f_star)))
        budgets.append(cur)

    # np.trapz was removed in numpy 2.0 (renamed trapezoid); support both.
    _trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    auc = float(_trapz(regrets, budgets) / max(budgets[-1] - budgets[0], 1e-9))
    return dict(final_regret=regrets[-1], auc=auc, n_hf=len(hf_idx), n_lf=len(lf_idx),
                pick_prec_top5=float(np.mean([p <= 0.05 + 1e-12 for p in picks])) if picks else np.nan,
                budgets=budgets, regrets=regrets)


def _append(path, rows):
    import os
    df = pd.DataFrame(rows)
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    tmp = path.with_suffix(path.suffix + '.tmp'); df.to_csv(tmp, index=False); os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benches', nargs='*', default=list(BENCHES))
    ap.add_argument('--models', nargs='*', default=MODELS)
    ap.add_argument('--seeds', type=int, nargs='*', default=SEEDS)
    ap.add_argument('--outdir', default=str(HERE / 'results'))
    args = ap.parse_args()
    import torch
    from benchmark import ChemistryBenchmark
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    print(f'fix6 | benches={args.benches} models={args.models} seeds={args.seeds} dev={dev}', flush=True)
    out = Path(args.outdir); (out / 'cells').mkdir(parents=True, exist_ok=True)
    for bn in args.benches:
        cfg = dict(BENCHES[bn])
        # HF_BUDGET env overrides the native budget (tight-budget regime test,
        # H4: does the transfer representation help when HF data is very scarce?)
        if os.environ.get('HF_BUDGET'):
            cfg['budget'] = float(os.environ['HF_BUDGET'])
        b = ChemistryBenchmark(bn, REPO / 'data' / cfg['csv'], cfg['cost_ratio'],
                               cfg['use_smiles'], True, cfg['negate'])
        print(f'{bn}: N={b.n_candidates} R2={b.r2:.3f}', flush=True)
        for ml in args.models:
            sf = out / 'cells' / f'summary_{bn}_{ml}.csv'
            tf = out / 'cells' / f'traj_{bn}_{ml}.csv'
            done = set(pd.read_csv(sf)['seed'].tolist()) if sf.exists() else set()
            for seed in args.seeds:
                if seed in done:
                    continue
                t0 = time.time()
                try:
                    r = run_one(b, cfg, ml, seed, dev)
                except Exception:
                    print(f'[FAIL] {bn} {ml} s{seed}', flush=True); traceback.print_exc(); continue
                _append(sf, [dict(benchmark=bn, model=ml, seed=seed,
                                  final_regret=r['final_regret'], auc=r['auc'],
                                  n_hf=r['n_hf'], n_lf=r['n_lf'],
                                  pick_prec_top5=r['pick_prec_top5'], elapsed=round(time.time()-t0,1))])
                _append(tf, [dict(benchmark=bn, model=ml, seed=seed, budget=round(bg,3), regret=rg)
                             for bg, rg in zip(r['budgets'], r['regrets'])])
                print(f'  {bn}/{ml}/s{seed}: regret={r["final_regret"]:.4f} auc={r["auc"]:.3f} ({time.time()-t0:.0f}s)', flush=True)
        print(f'DONE {bn} ({args.models})', flush=True)


if __name__ == '__main__':
    main()
