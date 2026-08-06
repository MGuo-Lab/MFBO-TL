"""Matched-pairs rank-biserial effect sizes for the paper's Wilcoxon comparisons.

Motivation: the Results text reports paired Wilcoxon p-values and mean +/- s.e.m.
only. This script adds the effect size matched to the paired Wilcoxon signed-rank
test, the matched-pairs rank-biserial correlation

    r = (W+ - W-) / (n(n+1)/2)

computed on the nonzero paired differences (matching zero_method='wilcox' used by
every Wilcoxon test in the paper; ties in |d| get average ranks). r is in [-1, 1];
the sign gives the direction and |r| the fraction of signed-rank mass on the
winning side. Sign convention throughout: d = (GP-side value) - (TL-side value)
on minimization metrics, so r > 0 favours the transfer-learning side.

The implementation is manual (rankdata on |d|), NOT derived from scipy's
`wilcoxon(...).statistic`, whose meaning depends on `alternative` and version;
a self-test cross-checks it against scipy's one-sided statistic (= W+) and the
identity W+ + W- = n(n+1)/2 before any real data are touched.

Data source (per-seed summaries, post-fix6 keystone Block A, 2026-06-29):
  experiments/keystone_sweep_20260629/results/blockA/<BENCH>_<seedrange>/cells/
      summary_<BENCH>_<MODEL>.csv
13 pools x 15 models x 20 seeds (3 seed rows absent upstream: SparseMFGP on
Branin-Fav s42/s53 and Branin-Unfav s59; best-of-family minima skip them).

Outputs (CSV, written next to this script):
  effect_sizes_maintext.csv    one row per main-text Wilcoxon comparison, with
                               the recomputed p cross-checked against the paper
                               (NB the COFs-vs-MFGP row here is the keystone EI
                               arm; the text's 19/20 + Holm 6.9e-5 sentence is
                               the matched-greedy arm, see
                               effect_sizes_matched_greedy.py)
  effect_sizes_family_grid.csv 13 pools x 4 family comparisons (best-TL vs
                               best-GP and best-TL vs plain MFGP, on final
                               regret and anytime-AUC): r, w/t/l, p
  effect_sizes_vs_mfgp_holm.csv per-TL-method vs MFGP final regret on the three
                               elastic pools, Holm-corrected within pool
LaTeX (input by the manuscript SI):
  paper/paper_document/paper_figures/effect_sizes_table.tex
                               tabular body of Supplementary Table
                               tab:effect_sizes (the family grid; Holm across
                               the 13 pools within each column)

Login-node-safe: reads cached CSVs only. Run:
    python effect_sizes_rank_biserial.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# NOTE (release repo): only the pure helpers (rank_biserial, wilcox_p, wtl) are
# used here, imported by effect_sizes_manuscript0713.py. The standalone main()
# analyses the keystone Block A sweep, whose raw runs are NOT shipped in this
# repo (they remain in the archived experiment repo, MFBO-TL-Paper).
BLOCKA = REPO / 'experiments' / 'keystone_sweep_20260629' / 'results' / 'blockA'

POOLS = ['Branin-Fav', 'Branin-Unfav', 'Park-Fav', 'Park-Unfav',
         'COFs', 'FreeSolv', 'Polarizability', 'Matbench-Gap',
         'ExptGap-Hautier', 'JARVIS-gap', 'Elastic-CHGNet',
         'Elastic-SevenNet', 'Elastic-MatterSim']
GP = {'MFGP', 'NARGP', 'DKL', 'SparseMFGP'}
TL = {'Sequential', 'Progressive', 'Curriculum', 'Adapter',
      'SoftParameterSharing', 'KnowledgeDistillation', 'DomainAdaptationMMD',
      'PseudoLabeling', 'TwoStageJoint', 'DNGOJoint', 'DNGOGradient'}
ELASTIC = ['Elastic-CHGNet', 'Elastic-SevenNet', 'Elastic-MatterSim']


def rank_biserial(d: np.ndarray) -> tuple[float, int]:
    """Matched-pairs rank-biserial correlation on the nonzero differences.

    Returns (r, n_nonzero). r > 0 means the d > 0 direction dominates."""
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return np.nan, 0
    ranks = rankdata(np.abs(d))
    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())
    total = n * (n + 1) / 2.0
    assert abs((w_plus + w_minus) - total) < 1e-9
    return (w_plus - w_minus) / total, n


def wilcox_p(d: np.ndarray) -> float:
    """Two-sided Wilcoxon p as used throughout the paper (zeros dropped)."""
    d = np.asarray(d, dtype=float)
    if not (d != 0).any():
        return np.nan
    return float(wilcoxon(d).pvalue)


def self_test() -> None:
    """Cross-check the manual W+ against scipy's one-sided statistic and the
    manual r against 2*W+/T - 1, on random paired data with and without ties."""
    rng = np.random.default_rng(12345)
    for trial in range(300):
        n = int(rng.integers(5, 30))
        d = rng.normal(0.3, 1.0, size=n)
        if trial % 3 == 0:              # inject |d| ties and zeros
            d = np.round(d, 1)
        dnz = d[d != 0]
        if len(dnz) == 0 or (dnz > 0).all() == (dnz < 0).all():
            continue
        r, n_eff = rank_biserial(d)
        ranks = rankdata(np.abs(dnz))
        w_plus = float(ranks[dnz > 0].sum())
        total = n_eff * (n_eff + 1) / 2.0
        # scipy with alternative='greater' reports W+ (sum of positive ranks)
        s = wilcoxon(dnz, alternative='greater').statistic
        assert abs(s - w_plus) < 1e-9, (trial, s, w_plus)
        assert abs(r - (2.0 * w_plus / total - 1.0)) < 1e-12
        assert -1.0 <= r <= 1.0
    # hand-checked example: d = [1, 2, 3, -4]; ranks 1,2,3,4; W+=6, W-=4, T=10
    r, n_eff = rank_biserial(np.array([1.0, 2.0, 3.0, -4.0]))
    assert n_eff == 4 and abs(r - 0.2) < 1e-12
    print('[self-test] manual rank-biserial matches scipy W+ (300 trials) OK')


def load_blocka() -> pd.DataFrame:
    files = sorted(glob.glob(str(BLOCKA / '*_s*' / 'cells' / 'summary_*.csv')))
    if not files:
        raise FileNotFoundError(f'no summary CSVs under {BLOCKA}')
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if df.duplicated(subset=['benchmark', 'model', 'seed']).any():
        raise ValueError('duplicate (benchmark, model, seed) rows')
    missing = set(POOLS) - set(df['benchmark'].unique())
    if missing:
        raise ValueError(f'pools missing from Block A: {missing}')
    return df


def pivot(df: pd.DataFrame, bench: str, metric: str) -> pd.DataFrame:
    g = df[df['benchmark'] == bench]
    return g.pivot_table(index='seed', columns='model', values=metric)


def wtl(d: pd.Series | np.ndarray) -> str:
    d = np.asarray(d, dtype=float)
    return f'{int((d > 0).sum())}/{int((d == 0).sum())}/{int((d < 0).sum())}'


def compare(d: pd.Series, label: dict) -> dict:
    """d oriented so d > 0 favours TL. Returns one result row."""
    d = d.dropna()
    r, n_nz = rank_biserial(d.values)
    return dict(**label, n_pairs=len(d), n_nonzero=n_nz, wtl=wtl(d.values),
                rank_biserial_r=r, wilcoxon_p=wilcox_p(d.values),
                median_diff=float(d.median()))


def holm_adjust(pvals: pd.Series) -> pd.Series:
    """Holm step-down adjusted p-values; NaN entries stay NaN."""
    p = pvals.dropna().sort_values()
    k = len(p)
    adj, running = {}, 0.0
    for i, (idx, v) in enumerate(p.items()):
        running = max(running, min(1.0, (k - i) * v))
        adj[idx] = running
    return pvals.index.to_series().map(adj).astype(float)


def write_si_table(grid: pd.DataFrame) -> None:
    """Emit the Supplementary Table tabular body (family grid with r, w/t/l;
    Holm across the 13 pools within each of the four comparison columns)."""
    g = grid.copy()
    for (metric, comp), sub in g.groupby(['metric', 'comparison']):
        g.loc[sub.index, 'holm_p'] = holm_adjust(sub['wilcoxon_p'])

    def cell(bench: str, metric: str, comp: str) -> str:
        row = g[(g['benchmark'] == bench) & (g['metric'] == metric)
                & (g['comparison'] == comp)].iloc[0]
        if not np.isfinite(row['rank_biserial_r']):
            return 'tie & 0/20/0'
        star = '^{*}' if row['holm_p'] < 0.05 else ''
        return (f"${row['rank_biserial_r']:+.2f}{star}$ & {row['wtl']}")

    lines = [
        '% AUTO-GENERATED by paper/auto_figures/effect_sizes_rank_biserial.py',
        '% (matched-pairs rank-biserial r for the family-level comparisons,',
        '%  keystone Block A per-seed data; do not edit by hand)',
        '\\begin{tabular}{lrcrcrcrc}',
        '\\toprule',
        '& \\multicolumn{4}{c}{Best TL vs best GP}'
        ' & \\multicolumn{4}{c}{Best TL vs MFGP} \\\\',
        '\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}',
        '& \\multicolumn{2}{c}{Final regret}'
        ' & \\multicolumn{2}{c}{Anytime-AUC}'
        ' & \\multicolumn{2}{c}{Final regret}'
        ' & \\multicolumn{2}{c}{Anytime-AUC} \\\\',
        'Pool & $r$ & w/t/l & $r$ & w/t/l & $r$ & w/t/l & $r$ & w/t/l \\\\',
        '\\midrule',
    ]
    for bench in POOLS:
        cells = [cell(bench, metric, comp)
                 for comp in ('bestTL vs bestGP', 'bestTL vs MFGP')
                 for metric in ('final_regret', 'auc')]
        lines.append(f'{bench} & ' + ' & '.join(cells) + ' \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    out = HERE / 'out' / 'effect_sizes_table.tex'
    out.parent.mkdir(exist_ok=True)
    out.write_text('\n'.join(lines) + '\n')
    print(f'\nwrote {out}')


def main() -> None:
    self_test()
    df = load_blocka()
    n = df.groupby(['benchmark', 'model'])['seed'].size()
    print(f'loaded {len(df)} seed rows, {len(n)} cells, '
          f'seeds per cell {sorted(n.unique())}')

    # ---- 1. main-text fixed-model and best-of-family comparisons ----------
    fixed = [
        # (bench, metric, tl_model, gp_model, paper-quoted p)
        ('Branin-Fav', 'final_regret', 'PseudoLabeling', 'DKL', 1.9e-6),
        ('Branin-Unfav', 'final_regret', 'DNGOGradient', 'MFGP', 4.3e-4),
        ('Elastic-MatterSim', 'auc', 'PseudoLabeling', 'SparseMFGP', 8.4e-4),
    ]
    bof = [
        # (bench, metric, comparator, paper-quoted p)  comparator: bestGP | MFGP
        ('Elastic-MatterSim', 'auc', 'bestGP', 1.9e-6),
        ('Elastic-SevenNet', 'auc', 'bestGP', 0.0094),
        ('ExptGap-Hautier', 'auc', 'bestGP', 0.0153),
        ('JARVIS-gap', 'auc', 'bestGP', 0.0486),
        ('COFs', 'auc', 'MFGP', None),   # Holm p = 6.9e-5 quoted; raw p here
    ]
    rows = []
    for bench, metric, tl_m, gp_m, p_paper in fixed:
        pa = pivot(df, bench, metric)
        rows.append(compare(pa[gp_m] - pa[tl_m],
                            dict(benchmark=bench, metric=metric,
                                 comparison=f'{tl_m} vs {gp_m}',
                                 kind='fixed-model', paper_p=p_paper)))
    for bench, metric, comp, p_paper in bof:
        pa = pivot(df, bench, metric)
        best_tl = pa[[m for m in pa.columns if m in TL]].min(axis=1)
        ref = (pa[[m for m in pa.columns if m in GP]].min(axis=1)
               if comp == 'bestGP' else pa['MFGP'])
        rows.append(compare(ref - best_tl,
                            dict(benchmark=bench, metric=metric,
                                 comparison=f'bestTL vs {comp}',
                                 kind='best-of-family', paper_p=p_paper)))
    main_tbl = pd.DataFrame(rows)
    out = HERE / 'effect_sizes_maintext.csv'
    main_tbl.to_csv(out, index=False)
    print(f'\n=== main-text comparisons (r > 0 favours TL) -> {out.name} ===')
    print(main_tbl.to_string(index=False,
                             float_format=lambda v: f'{v:.4g}'))
    for _, row in main_tbl.iterrows():
        if row['paper_p'] is not None and np.isfinite(row.get('paper_p') or np.nan):
            if not np.isclose(row['wilcoxon_p'], row['paper_p'],
                              rtol=0.12, atol=5e-7):
                print(f'  [WARN] {row["benchmark"]} {row["comparison"]}: '
                      f'recomputed p={row["wilcoxon_p"]:.3g} vs paper '
                      f'p={row["paper_p"]:.3g}')

    # COFs anytime factor quoted as 1.53x (mean MFGP AUC / mean best-TL AUC)
    pa = pivot(df, 'COFs', 'auc')
    best_tl = pa[[m for m in pa.columns if m in TL]].min(axis=1)
    print(f'\nCOFs AUC factor check: mean MFGP / mean bestTL = '
          f'{pa["MFGP"].mean() / best_tl.mean():.3f} (paper: 1.53)')

    # ---- 2. family-comparison grid over all 13 pools ----------------------
    grid_rows = []
    for bench in POOLS:
        for metric in ('final_regret', 'auc'):
            pa = pivot(df, bench, metric)
            best_tl = pa[[m for m in pa.columns if m in TL]].min(axis=1)
            best_gp = pa[[m for m in pa.columns if m in GP]].min(axis=1)
            for comp, ref in (('bestGP', best_gp), ('MFGP', pa['MFGP'])):
                grid_rows.append(compare(ref - best_tl,
                                         dict(benchmark=bench, metric=metric,
                                              comparison=f'bestTL vs {comp}')))
    grid = pd.DataFrame(grid_rows)
    out = HERE / 'effect_sizes_family_grid.csv'
    grid.to_csv(out, index=False)
    print(f'\n=== family grid (13 pools x 4 comparisons) -> {out.name} ===')
    print(grid.to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    print('\n--- median r across pools (decided pools only) ---')
    for (metric, comp), g in grid.groupby(['metric', 'comparison']):
        dec = g.dropna(subset=['rank_biserial_r'])
        print(f'  {metric:12s} {comp:15s} median r = '
              f'{dec["rank_biserial_r"].median():+.3f} '
              f'(n={len(dec)} decided pools of {len(g)})')

    write_si_table(grid)

    # ---- 3. per-TL-method vs MFGP, elastic final regret, Holm within pool --
    holm_rows = []
    for bench in ELASTIC:
        pa = pivot(df, bench, 'final_regret')
        sub = []
        for m in sorted(TL):
            row = compare(pa['MFGP'] - pa[m],
                          dict(benchmark=bench, metric='final_regret',
                               comparison=f'{m} vs MFGP'))
            sub.append(row)
        sub = pd.DataFrame(sub).sort_values('wilcoxon_p').reset_index(drop=True)
        k = len(sub)
        holm, running = [], 0.0
        for i, p in enumerate(sub['wilcoxon_p']):
            running = max(running, min(1.0, (k - i) * p))
            holm.append(running)
        sub['holm_p'] = holm
        holm_rows.append(sub)
    holm_tbl = pd.concat(holm_rows, ignore_index=True)
    out = HERE / 'effect_sizes_vs_mfgp_holm.csv'
    holm_tbl.to_csv(out, index=False)
    sig = holm_tbl[holm_tbl['holm_p'] < 0.05]
    print(f'\n=== TL vs MFGP, elastic final regret, Holm within pool '
          f'-> {out.name} ===')
    print(sig.to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    print(f'  ({len(sig)} Holm-significant of {len(holm_tbl)}; '
          f'max sig Holm p = {sig["holm_p"].max():.3g}, '
          f'min sig |r| = {sig["rank_biserial_r"].abs().min():.3f})')


if __name__ == '__main__':
    main()
