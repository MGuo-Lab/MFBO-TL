"""final_regret.pdf -- panel letters shifted (a-j) -> (b-k) for the composite
main-text figure; everything else (data, colours, layout, no legend) identical
to the current manuscript PDF.

Adapted from paper/auto_figures/plot_headline9.py::barplot(). The average-rank
panel (k) covers ALL FIFTEEN surrogates (as the manuscript caption states; the
NARGP/DKL/Sparse MFGP cells on HOPV15 and Matbench-Gap come from
results/gpfamily_newbench) and carries a dashed line at best rank + Nemenyi
critical difference (alpha = 0.05, k = 15, N = 9 -> CD = 7.15 rank units;
Demsar 2006). Companion test report: stats_friedman_nemenyi_9bench.py.

Data: results/main_9bench + results/extra_baselines
      + results/gpfamily_newbench. Cached CSVs only (login-node safe).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (NEWFIGS, RESULTS, RENAME_MAP, add_panel_letter, letter_pt,
                     save_dual, code, CODE_KEY_TL, CODE_KEY_TL2, CODE_KEY_GP)

MFGP_COLOR = '#f2aa84'      # salmon -- GP family (unchanged)
DNN_COLOR = '#4e95d9'       # blue -- transfer learning (unchanged)
GP_FAMILY = {'MFGP', 'NARGP', 'DKL Multi-Fidelity', 'Sparse MFGP'}
GP_EXTRA = {'NARGP', 'DKL Multi-Fidelity', 'Sparse MFGP'}

BENCHMARKS = ['Branin-Fav', 'Branin-Unfav', 'Park-Fav', 'Park-Unfav', 'COFs',
              'FreeSolv', 'Polarizability', 'HOPV15', 'Matbench-Gap']
BUDGET = {'Branin-Fav': 50, 'Branin-Unfav': 50, 'Park-Fav': 50, 'Park-Unfav': 50,
          'COFs': 30, 'FreeSolv': 50, 'Polarizability': 30, 'HOPV15': 30,
          'Matbench-Gap': 20}
EARLY = {'Park-Fav', 'Park-Unfav'}
EARLY_B = 10
SPELL_MAP = {'Pseudo-Labeling': 'Pseudo-Labelling'}   # UK spelling, matches the main text
NCOLS, NROWS = 5, 2
# 2026-08-18 print-size re-export: canvas halved (30x11 in -> 15x6.6 in, i.e.
# ~2.1x the 174 mm print width instead of 4.4x) and fonts rebalanced so body
# text prints at >= 5.2 pt; y ticks use the compact display names (_common).
TITLE, LABEL, TICK = 15, 12, 12
LETTERS = list('bcdefghijk')      # (a-j) -> (b-k); 'a' is the overview panel
# uniform 8 pt print size: tight PDF width ~385 mm, print width 174 mm
LETTER_PT = letter_pt(403.0, 174)


def load_data():
    summ = pd.read_csv(RESULTS / 'main_9bench' / 'results_summary.csv')
    traj = pd.read_csv(RESULTS / 'main_9bench' / 'results_trajectory.csv')
    summ['model'] = summ['model'].replace(RENAME_MAP).replace(SPELL_MAP)
    traj['model'] = traj['model'].replace(RENAME_MAP).replace(SPELL_MAP)
    core_models = sorted(summ['model'].unique())        # 12 core surrogates
    ex_s, ex_t = [], []
    for d in [RESULTS / 'extra_baselines',
              RESULTS / 'gpfamily_newbench']:
        if (d / 'results_summary.csv').exists():
            ex_s.append(pd.read_csv(d / 'results_summary.csv'))
            if (d / 'results_trajectory.csv').exists():
                ex_t.append(pd.read_csv(d / 'results_trajectory.csv'))
    ex_s = pd.concat(ex_s, ignore_index=True)
    ex_s = ex_s[ex_s['model'].isin(GP_EXTRA)]
    ex_t = pd.concat(ex_t, ignore_index=True)
    ex_t = ex_t[ex_t['model'].isin(GP_EXTRA)]
    return (pd.concat([summ, ex_s], ignore_index=True),
            pd.concat([traj, ex_t], ignore_index=True), core_models)


def _regret_at_early(traj, bench, models):
    bd = traj[traj.benchmark == bench]
    rows = []
    for m in models:
        sub = bd[bd.model == m]
        for s in sub.seed.unique():
            sd = sub[sub.seed == s].sort_values('budget')
            v = sd[sd.budget <= EARLY_B]
            r = (v.iloc[-1] if len(v) else sd.iloc[0])['regret'] if len(sd) else np.nan
            rows.append({'model': m, 'seed': s, 'final_regret': r})
    return pd.DataFrame(rows)


def main():
    summary, traj, core_models = load_data()
    cmap = {m: (MFGP_COLOR if m in GP_FAMILY else DNN_COLOR)
            for m in summary['model'].unique()}
    stats = {}
    for b in BENCHMARKS:
        present = sorted(summary[summary.benchmark == b]['model'].unique())
        d = _regret_at_early(traj, b, present) if b in EARLY else \
            summary[summary.benchmark == b][['model', 'seed', 'final_regret']]
        g = d.groupby('model').final_regret.agg(['mean', 'std', 'count']).reset_index()
        g['se'] = g['std'] / np.sqrt(g['count'])
        stats[b] = g.sort_values('mean')

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(3 * NCOLS, 7.5))
    for i, b in enumerate(BENCHMARKS):
        ax = axes[i // NCOLS, i % NCOLS]
        s = stats[b]
        y = np.arange(len(s))
        ax.barh(y, s['mean'], xerr=s['se'], color=[cmap[m] for m in s['model']],
                capsize=2, alpha=0.9, edgecolor='none', height=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([code(m) for m in s['model']], fontsize=TICK)
        bb = EARLY_B if b in EARLY else BUDGET[b]
        ax.set_title(f'{b} (B = {bb})', fontsize=TITLE, loc='left')
        # x=-0.34: above the model-name tick-label column, i.e. the top-left
        # corner of the full panel (title-line height, no title overlap)
        add_panel_letter(ax, LETTERS[i], x=-0.24, y=1.02, size=LETTER_PT)
        ax.set_xlabel('Final regret', fontsize=LABEL)
        ax.tick_params(labelsize=TICK)
        ax.grid(axis='x', alpha=0.3, lw=0.5)

    # average-rank panel over ALL FIFTEEN surrogates (matches the caption), with
    # the Nemenyi critical-difference line (Demsar 2006)
    rr = []
    for b in BENCHMARKS:
        g = stats[b].copy()
        g['rank'] = g['mean'].rank()
        rr.append(g[['model', 'rank']])
    avg = pd.concat(rr).groupby('model')['rank'].agg(['mean', 'std', 'count']).reset_index()
    if not (avg['count'] == len(BENCHMARKS)).all():
        raise ValueError('rank panel needs every surrogate on every benchmark:\n'
                         f'{avg[avg["count"] != len(BENCHMARKS)]}')
    avg['se'] = avg['std'] / np.sqrt(avg['count'])
    avg = avg.sort_values('mean')
    k_models, n_bench = len(avg), len(BENCHMARKS)
    q = sps.studentized_range.ppf(0.95, k_models, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k_models * (k_models + 1) / (6.0 * n_bench))
    ax = axes[1, 4]
    y = np.arange(len(avg))
    ax.barh(y, avg['mean'], xerr=avg['se'], color=[cmap[m] for m in avg['model']],
            capsize=2, alpha=0.9, edgecolor='none', height=0.7)
    thresh = avg['mean'].iloc[0] + cd
    ax.axvline(thresh, color='#444444', lw=1.0, ls=(0, (4, 2)), zorder=3)
    ax.text(thresh - 0.18, 0.02, 'best + CD', transform=ax.get_xaxis_transform(),
            rotation=90, va='bottom', ha='right', fontsize=TICK - 1,
            style='italic', color='#444444')
    ax.set_xlim(0, max(avg['mean'].max() * 1.15, thresh * 1.06))
    ax.set_yticks(y)
    ax.set_yticklabels([code(m) for m in avg['model']], fontsize=TICK)
    ax.set_title('Average Rank', fontsize=TITLE, loc='left')
    add_panel_letter(ax, LETTERS[9], x=-0.24, y=1.02, size=LETTER_PT)
    ax.set_xlabel('Average rank', fontsize=LABEL)
    ax.tick_params(labelsize=TICK)
    ax.grid(axis='x', alpha=0.3, lw=0.5)

    plt.tight_layout(w_pad=2.2, h_pad=1.2, rect=(0, 0.075, 1, 1))
    # abbreviation key strip (full names; the axes carry the codes)
    for yy, line in ((0.052, CODE_KEY_TL), (0.030, CODE_KEY_TL2), (0.008, CODE_KEY_GP)):
        fig.text(0.5, yy, line, ha='center', va='bottom', fontsize=10.5,
                 color='#333333')
    save_dual(fig, NEWFIGS / 'final_regret')
    plt.close(fig)


if __name__ == '__main__':
    main()
