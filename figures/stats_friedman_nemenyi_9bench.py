"""Friedman + Nemenyi for the two average-rank panels of the Overleaf/GitHub
manuscript (Jaewook-MFBO-TL-Paper): Fig 1k (final regret) and Supp Fig 1j
(compute rank), per the second supervisor's recommendation.

Matrix A (Fig 1k): 9 benchmarks x 15 surrogates, values = seed-mean final
regret at the designated budget, assembled EXACTLY like make_final_regret.py
(Park pools evaluated at budget 10 from the trajectory; main_9bench core 12
+ NARGP/DKL/Sparse MFGP from extra_baselines_results and gpfamily_newbench).

Matrix B (Supp Fig 1j): 9 x 15 deterministic loop FLOPs from
figures/out/computing_flops_values.csv (one value per cell, as in
plot_computing_flops.py panel j).

Nemenyi CD = q_0.05(k=15)/sqrt(2) * sqrt(k(k+1)/(6N)) = 7.15 rank units at
N = 9 (scipy studentized_range; no scikit-posthocs on the cluster).

Also reports the auxiliary best-vs-each Wilcoxon over the 9 pool means with
Holm correction; note that with N = 9 blocks the smallest attainable exact
two-sided p is 2/512 = 0.0039, so 14-way Holm (x14 = 0.055) can never reach
0.05: the auxiliary test is underpowered by construction at N = 9 and is NOT
quoted in the manuscript.

Login-node-safe (cached CSVs, 9x15 arithmetic). Run from _scripts/:
    python stats_friedman_nemenyi_9bench.py
Writes FRIEDMAN_NEMENYI_9bench.md next to this script and prints the report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_final_regret import (BENCHMARKS, BUDGET, EARLY, EARLY_B,
                               _regret_at_early, load_data)
from _common import RESULTS

OUT_MD = HERE / 'out' / 'FRIEDMAN_NEMENYI_9bench.md'
# written by plot_computing_flops.py — run that first
FLOPS_CSV = HERE / 'out' / 'computing_flops_values.csv'
ALPHA = 0.05


def nemenyi_cd(k: int, n: int, alpha: float = ALPHA) -> float:
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return q * np.sqrt(k * (k + 1) / (6.0 * n))


def nemenyi_pairwise_p(rank_diff: float, k: int, n: int) -> float:
    z = rank_diff / np.sqrt(k * (k + 1) / (6.0 * n))
    return float(stats.studentized_range.sf(z * np.sqrt(2), k, np.inf))


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[idx]))
        adj[idx] = running
    return adj.tolist()


def build_final_regret_matrix() -> pd.DataFrame:
    """Seed-mean final regret per (benchmark, model), same as the Fig 1k bars."""
    summary, traj, _core = load_data()
    rows = []
    for b in BENCHMARKS:
        present = sorted(summary[summary.benchmark == b]['model'].unique())
        d = _regret_at_early(traj, b, present) if b in EARLY else \
            summary[summary.benchmark == b][['model', 'seed', 'final_regret']]
        g = d.groupby('model').final_regret.mean().reset_index()
        g['benchmark'] = b
        rows.append(g)
    piv = (pd.concat(rows, ignore_index=True)
             .pivot(index='benchmark', columns='model', values='final_regret')
             .loc[BENCHMARKS])
    if piv.isna().any().any():
        raise ValueError(f'incomplete grid:\n{piv.isna().sum()}')
    return piv


def build_flops_matrix() -> pd.DataFrame:
    d = pd.read_csv(FLOPS_CSV)
    piv = d.pivot(index='benchmark', columns='model', values='flops').loc[BENCHMARKS]
    if piv.isna().any().any():
        raise ValueError('incomplete FLOPs grid')
    return piv


def analyse(piv: pd.DataFrame, tag: str, lines: list[str], wilcoxon: bool):
    n, k = piv.shape
    raw = piv.values
    R = np.vstack([stats.rankdata(row) for row in raw])
    mean_ranks = pd.Series(R.mean(axis=0), index=piv.columns).sort_values()

    chi2, p_chi2 = stats.friedmanchisquare(*[raw[:, j] for j in range(k)])
    ff = (n - 1) * chi2 / (n * (k - 1) - chi2)
    p_ff = stats.f.sf(ff, k - 1, (k - 1) * (n - 1))
    cd = nemenyi_cd(k, n)

    lines.append(f'\n## {tag}')
    lines.append(f'- N = {n} benchmarks (blocks), k = {k} surrogates')
    lines.append(f'- Friedman (tie-corrected): chi2({k - 1}) = {chi2:.2f}, p = {p_chi2:.3g}')
    lines.append(f'- Iman-Davenport: F = {ff:.2f}, p = {p_ff:.3g}')
    lines.append(f'- Nemenyi CD (alpha = {ALPHA}): {cd:.2f} rank units')
    lines.append('\n| model | mean rank |')
    lines.append('|---|---|')
    for m, r in mean_ranks.items():
        lines.append(f'| {m} | {r:.2f} |')

    best = mean_ranks.index[0]
    lines.append(f'\nPairwise Nemenyi vs the top-ranked model ({best}):')
    lines.append('\n| vs | rank diff | Nemenyi p | > CD |')
    lines.append('|---|---|---|---|')
    for m in mean_ranks.index[1:]:
        diff = mean_ranks[m] - mean_ranks[best]
        p_pair = nemenyi_pairwise_p(diff, k, n)
        lines.append(f'| {m} | {diff:.2f} | {p_pair:.3g} | '
                     f'{"yes" if diff > cd else "no"} |')

    sep = [(a, b, mean_ranks[b] - mean_ranks[a])
           for i, a in enumerate(mean_ranks.index)
           for b in mean_ranks.index[i + 1:]
           if mean_ranks[b] - mean_ranks[a] > cd]
    lines.append(f'\nAll pairs separated by more than CD ({len(sep)} of {k * (k - 1) // 2}):')
    for a, b, d in sep:
        lines.append(f'- {a} vs {b} (diff {d:.2f})')

    if wilcoxon:
        lines.append(f'\nAuxiliary (NOT in the manuscript; see module docstring): '
                     f'Wilcoxon best-vs-each over the {n} benchmark means, Holm over {k - 1}:')
        names, pvals = [], []
        for m in mean_ranks.index[1:]:
            d = piv[best].values - piv[m].values
            try:
                pvals.append(stats.wilcoxon(d, zero_method='wilcox').pvalue)
            except ValueError:
                pvals.append(1.0)
            names.append(m)
        adj = holm(pvals)
        lines.append('\n| vs | p | Holm p |')
        lines.append('|---|---|---|')
        for m, p_raw, p_holm in zip(names, pvals, adj):
            lines.append(f'| {m} | {p_raw:.4f} | {p_holm:.4f} |')
    return mean_ranks, cd


def main():
    lines = ['# Friedman + Nemenyi, Overleaf manuscript rank panels (9 benchmarks x 15 surrogates)',
             '',
             'Fig 1k matrix = seed-mean final regret at the designated budget '
             '(Park pools at budget 10), identical to the make_final_regret.py bars. '
             'Supp Fig 1j matrix = deterministic loop FLOPs '
             '(computing_flops_values.csv).']
    piv_fr = build_final_regret_matrix()
    analyse(piv_fr, 'Fig 1k: final regret', lines, wilcoxon=True)
    piv_fl = build_flops_matrix()
    analyse(piv_fl, 'Supp Fig 1j: compute (FLOPs)', lines, wilcoxon=False)
    report = '\n'.join(lines) + '\n'
    OUT_MD.write_text(report)
    print(report)
    print(f'[written] {OUT_MD}')


if __name__ == '__main__':
    main()
