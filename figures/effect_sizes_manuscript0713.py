"""Rank-biserial effect sizes for the 0713 (GitHub/Overleaf) manuscript.

The active manuscript (repo Jaewook-MFBO-TL-Paper, lineage main_0713.tex) quotes
(a) per-benchmark mean final regrets of the best transfer-learning surrogate vs
the baseline MFGP (results/main_9bench, 20 seeds), and (b) the controlled
polarizability fidelity-quality grid (126 cells x 6 models x 10 seeds,
experiments/matbench_budget_ext/results_grid_polariz) with three families
compared pairwise per cell (protocol of
experiments/regime_review_20260625/make_fig1c_family_split.py: best-of-family
by cell mean, paired Wilcoxon on shared seeds).

This script attaches the matched-pairs rank-biserial correlation r to both:
  r = (W+ - W-)/(W+ + W-) on the nonzero paired differences (zeros dropped,
  average ranks), implementation self-tested against scipy in
  effect_sizes_rank_biserial.py (imported).

Sign convention here matches the manuscript's advantage convention: for a
comparison "first family vs second family" on a lower-is-better metric,
d = second - first, so r > 0 means the FIRST-named side is better. For the
benchmark table the first side is the best transfer-learning surrogate and the
second the baseline MFGP.

Outputs (written next to this script):
  effect_sizes_0713_bench.csv       9 benchmarks: best-TL(by mean) vs MFGP
  effect_sizes_0713_grid_cells.csv  126 cells x 3 comparisons x 2 metrics
  effect_sizes_0713_grid_summary.csv medians/IQR/%sig per comparison x metric
  effect_sizes_0713_tables.tex      LaTeX tabulars to paste into the SI
Cross-checks printed: manuscript means (0.41 vs 6.55 etc.), the grid headline
fractions (126/126 positive, 97% / 83% significant), and the r medians.

Login-node-safe: reads cached CSVs only. Run:
    python effect_sizes_manuscript0713.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from effect_sizes_rank_biserial import rank_biserial, wilcox_p, wtl

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / 'out'
OUT.mkdir(exist_ok=True)
MAIN9 = REPO / 'results' / 'main_9bench' / 'results_summary.csv'
GRID_CELLS = REPO / 'results' / 'grid' / 'cells'
GRID_MANIFEST = REPO / 'results' / 'grid' / 'grid_manifest.csv'

BENCHMARKS = ['Branin-Fav', 'Branin-Unfav', 'Park-Fav', 'Park-Unfav', 'COFs',
              'FreeSolv', 'Polarizability', 'HOPV15', 'Matbench-Gap']
TL9 = ['Sequential', 'Progressive', 'Curriculum', 'Adapter',
       'Soft Parameter Sharing', 'Knowledge Distillation',
       'Domain Adaptation (MMD)', 'Pseudo-Labeling', 'Two-Stage Joint',
       'DNGO-Joint', 'DNGO-Gradient']

FAMILIES = {
    'TL': ['Progressive', 'KnowledgeDistillation', 'PseudoLabeling'],
    'MFGP baseline': ['MFGP'],
    'MFGP variants': ['DKL', 'SparseMFGP'],
}
COMPARISONS = [('TL', 'MFGP baseline'),
               ('TL', 'MFGP variants'),
               ('MFGP variants', 'MFGP baseline')]


def bench_table() -> pd.DataFrame:
    s = pd.read_csv(MAIN9)
    rows = []
    for b in BENCHMARKS:
        g = s[s['benchmark'] == b]
        pa = g.pivot_table(index='seed', columns='model', values='final_regret')
        means = pa.mean()
        best_tl = means[[m for m in TL9 if m in means.index]].idxmin()
        d = (pa['MFGP'] - pa[best_tl]).dropna()   # >0: TL better
        r, n_nz = rank_biserial(d.values)
        rows.append(dict(benchmark=b, best_tl=best_tl,
                         mean_tl=float(means[best_tl]),
                         mean_mfgp=float(means['MFGP']),
                         n_pairs=len(d), n_nonzero=n_nz, wtl=wtl(d.values),
                         rank_biserial_r=r, wilcoxon_p=wilcox_p(d.values)))
    return pd.DataFrame(rows)


def grid_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    man = pd.read_csv(GRID_MANIFEST).set_index('cell_id')
    rows = []
    for f in sorted(glob.glob(str(GRID_CELLS / 'summary_cell_*.csv'))):
        cid = Path(f).name.replace('summary_', '').replace('.csv', '')
        d = pd.read_csv(f)
        for metric in ('final_regret', 'auc'):
            pa = d.pivot_table(index='seed', columns='model', values=metric)
            best = {}
            for fam, members in FAMILIES.items():
                have = [m for m in members if m in pa.columns]
                best[fam] = pa[have].mean().idxmin()
            for first, second in COMPARISONS:
                dd = (pa[best[second]] - pa[best[first]]).dropna()
                r, n_nz = rank_biserial(dd.values)
                rows.append(dict(
                    cell=cid, r2=float(man.loc[cid, 'r2']),
                    top10=float(man.loc[cid, 'top10']), metric=metric,
                    comparison=f'{first} vs {second}',
                    adv=float(dd.mean()), n_pairs=len(dd), n_nonzero=n_nz,
                    wtl=wtl(dd.values), rank_biserial_r=r,
                    wilcoxon_p=wilcox_p(dd.values)))
    cells = pd.DataFrame(rows)
    summ = []
    for (metric, comp), g in cells.groupby(['metric', 'comparison']):
        rr = g['rank_biserial_r']
        sig = (g['wilcoxon_p'] < 0.05)
        summ.append(dict(
            metric=metric, comparison=comp, n_cells=len(g),
            n_adv_positive=int((g['adv'] > 0).sum()),
            n_sig=int(sig.sum()), frac_sig=float(sig.mean()),
            median_adv=float(g['adv'].median()),
            median_r=float(rr.median(skipna=True)),
            q1_r=float(rr.quantile(0.25)), q3_r=float(rr.quantile(0.75)),
            min_r=float(rr.min()), max_r=float(rr.max()),
            n_r_defined=int(rr.notna().sum())))
    return cells, pd.DataFrame(summ)


def latex_tables(bench: pd.DataFrame, gsum: pd.DataFrame) -> str:
    ln = []
    ln.append('% ---- per-benchmark best-TL vs baseline MFGP (final regret) ----')
    ln.append('\\begin{tabular}{llrrrcr}')
    ln.append('\\toprule')
    ln.append('Benchmark & Best TL surrogate & TL & MFGP & $r$ & w/t/l & $p$ \\\\')
    ln.append('\\midrule')
    for _, x in bench.iterrows():
        p = f"${x['wilcoxon_p']:.2g}$".replace('e-0', '\\times 10^{-').replace(
            'e-', '\\times 10^{-')
        if '10^{-' in p:
            p = p.rstrip('$') + '}$'
        ln.append(f"{x['benchmark']} & {x['best_tl']} & "
                  f"{x['mean_tl']:.2f} & {x['mean_mfgp']:.2f} & "
                  f"${x['rank_biserial_r']:+.2f}$ & {x['wtl']} & {p} \\\\")
    ln.append('\\bottomrule')
    ln.append('\\end{tabular}')
    ln.append('')
    ln.append('% ---- grid: per-cell r distribution per family comparison ----')
    ln.append('\\begin{tabular}{llrrrr}')
    ln.append('\\toprule')
    ln.append('Metric & Comparison & Median $r$ & IQR & Cells sig. & '
              'Cells adv.$>0$ \\\\')
    ln.append('\\midrule')
    for _, x in gsum.iterrows():
        met = ('Final regret' if x['metric'] == 'final_regret'
               else 'Anytime-AUC')
        ln.append(f"{met} & {x['comparison']} & ${x['median_r']:+.2f}$ & "
                  f"$[{x['q1_r']:+.2f}, {x['q3_r']:+.2f}]$ & "
                  f"{x['n_sig']}/{x['n_cells']} & "
                  f"{x['n_adv_positive']}/{x['n_cells']} \\\\")
    ln.append('\\bottomrule')
    ln.append('\\end{tabular}')
    return '\n'.join(ln) + '\n'


def main() -> None:
    bench = bench_table()
    bench.to_csv(OUT / 'effect_sizes_0713_bench.csv', index=False)
    print('=== benchmarks: best-TL(by mean) vs baseline MFGP, final regret '
          '(r > 0 favours TL) ===')
    print(bench.to_string(index=False, float_format=lambda v: f'{v:.4g}'))

    cells, gsum = grid_tables()
    cells.to_csv(OUT / 'effect_sizes_0713_grid_cells.csv', index=False)
    gsum.to_csv(OUT / 'effect_sizes_0713_grid_summary.csv', index=False)
    man = pd.read_csv(GRID_MANIFEST)
    print(f"\ngrid manifest: {len(man)} cells, R2 "
          f"{man['r2'].min():.2f}-{man['r2'].max():.2f}, top10 "
          f"{man['top10'].min():.2f}-{man['top10'].max():.2f}")
    print('\n=== grid summary (r > 0 favours first-named family) ===')
    print(gsum.to_string(index=False, float_format=lambda v: f'{v:.4g}'))

    tex = latex_tables(bench, gsum)
    (OUT / 'effect_sizes_0713_tables.tex').write_text(tex)
    print(f"\nwrote effect_sizes_0713_tables.tex")


if __name__ == '__main__':
    main()
