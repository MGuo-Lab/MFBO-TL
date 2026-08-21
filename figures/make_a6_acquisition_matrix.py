"""A6_acquisition_matrix.pdf -- letters a-i KEPT but restyled "(a)" -> bold
"a" outside the axes; the suptitle ("Surrogate x Acquisition: MFGP loss
is...") REMOVED (caption replaces it). Data, colours, layout, legend, value
labels: unchanged.

Adapted from paper/auto_figures/plot_a6_acquisition_matrix.py.
Data: results/main_9bench (MFGP UQ-EI + DNN greedy), mfgp_greedy_7bench +
mfgp_greedy_newbench (MFGP greedy), main_corrected (DNN BLR-std EI).
Reproduction verified against results/post_submission/acquisition_2x2/
A6_values.csv (the numbers dumped when the manuscript figure was generated).
Login-node safe.
"""
import sys
import glob
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import NEWFIGS, RESULTS, add_panel_letter, letter_pt, save_dual

MFGP_COLOR, HIGHLIGHT_COLOR = '#f2aa84', '#4e95d9'
TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE = 14, 12, 11, 12

DNN = 'DNGO-Gradient'
BENCHES = ['Branin-Fav', 'Branin-Unfav', 'Park-Fav', 'Park-Unfav', 'COFs',
           'FreeSolv', 'Polarizability', 'HOPV15', 'Matbench-Gap']
NCOLS = 5
BUDGET = {'Branin-Fav': 50, 'Branin-Unfav': 50, 'Park-Fav': 50, 'Park-Unfav': 50,
          'COFs': 30, 'FreeSolv': 50, 'Polarizability': 30, 'HOPV15': 30,
          'Matbench-Gap': 20}
EARLY = {'Park-Fav', 'Park-Unfav'}
EARLY_B = 10
COND = [
    ('MFGP · EI',     MFGP_COLOR, ''),
    ('MFGP · greedy', '#c46c4e',  ''),
    ('TL · greedy',   HIGHLIGHT_COLOR, ''),
    ('TL · EI',       '#9bb8d4', ''),
]
LETTERS = list('abcdefghi')
# uniform 8 pt print size: tight PDF width ~332 mm, print width 160 mm
# (2026-08-18 print-size re-export: canvas 32.5x10 in -> 13x5.4 in, ~2.1x)
LETTER_PT = letter_pt(332.0, 160)


def _load(p):
    fs = glob.glob(p)
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def _mean_se(df, bench, model):
    v = df[(df['benchmark'] == bench) & (df['model'] == model)]['final_regret'].dropna()
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, 0
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0, len(v)


def _early_mean_se(traj, bench, model, B):
    sub = traj[(traj['benchmark'] == bench) & (traj['model'] == model)]
    vals = []
    for s in sub['seed'].unique():
        sd = sub[sub['seed'] == s].sort_values('budget')
        v = sd[sd['budget'] <= B]
        vals.append((v.iloc[-1] if len(v) else sd.iloc[0])['regret'])
    vals = np.array([x for x in vals if np.isfinite(x)])
    if len(vals) == 0:
        return np.nan, np.nan, 0
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0, len(vals)


def _fmt(m):
    if not np.isfinite(m):
        return ''
    if m == 0:
        return '0'
    if abs(m) < 0.01:
        return f'{m:.2g}'
    return f'{m:.2f}'


def main():
    mc = pd.read_csv(RESULTS / 'main_corrected' / 'results_summary.csv')
    m9 = pd.read_csv(RESULTS / 'main_9bench' / 'results_summary.csv')
    mg = _load(str(RESULTS / 'mfgp_greedy_7bench' / 'summary_*.csv'))
    mgn = pd.read_csv(RESULTS / 'mfgp_greedy_newbench' / 'results_summary.csv')

    ei_greedy_src = m9
    mfgp_greedy_src = pd.concat([mg, mgn], ignore_index=True)
    dnn_blr_src = mc[mc['model'] == DNN].copy()

    t_m9 = pd.read_csv(RESULTS / 'main_9bench' / 'results_trajectory.csv')
    t_mg = pd.read_csv(RESULTS / 'mfgp_greedy_7bench' / 'results_trajectory.csv')
    t_mc = pd.read_csv(RESULTS / 'main_corrected' / 'results_trajectory.csv')

    data = {}
    for b in BENCHES:
        if b in EARLY:
            data[b] = [
                _early_mean_se(t_m9, b, 'MFGP', EARLY_B),
                _early_mean_se(t_mg, b, 'MFGP', EARLY_B),
                _early_mean_se(t_m9, b, DNN, EARLY_B),
                _early_mean_se(t_mc, b, DNN, EARLY_B),
            ]
        else:
            data[b] = [
                _mean_se(ei_greedy_src, b, 'MFGP'),
                _mean_se(mfgp_greedy_src, b, 'MFGP'),
                _mean_se(ei_greedy_src, b, DNN),
                _mean_se(dnn_blr_src, b, DNN),
            ]

    nrows = math.ceil(len(BENCHES) / NCOLS)
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(2.6 * NCOLS, 2.7 * nrows))
    axes = np.atleast_2d(axes)
    labels = [c[0] for c in COND]
    colors = [c[1] for c in COND]
    for idx, b in enumerate(BENCHES):
        ax = axes[idx // NCOLS, idx % NCOLS]
        means = [d[0] for d in data[b]]
        ses = [d[1] for d in data[b]]
        y = np.arange(len(labels))[::-1]
        ax.barh(y, means, xerr=ses, color=colors, capsize=2, alpha=0.92,
                edgecolor='white', height=0.68, linewidth=0.7,
                error_kw=dict(elinewidth=1.0, capthick=1.0))
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=TICK_SIZE)
        bb = EARLY_B if b in EARLY else BUDGET[b]
        ax.set_title(f'{b} (B = {bb})', fontsize=TITLE_SIZE, loc='left')
        add_panel_letter(ax, LETTERS[idx], x=-0.45, y=1.02, size=LETTER_PT)
        ax.set_xlabel('Final Regret (Mean ± SE)', fontsize=LABEL_SIZE)
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
        ax.grid(axis='x', alpha=0.3, linewidth=0.5)
        xmax = max([m for m in means if np.isfinite(m)] + [1e-6])
        for yi, (m, se) in zip(y, zip(means, ses)):
            if np.isfinite(m):
                ax.text(m + se + xmax * 0.035, yi, _fmt(m), va='center',
                        fontsize=TICK_SIZE - 1, color='#333333')
        ax.set_xlim(0, xmax * 1.28)

    for j in range(len(BENCHES), nrows * NCOLS):
        axes[j // NCOLS, j % NCOLS].axis('off')

    handles = [Patch(facecolor=c[1], edgecolor='white', label=c[0]) for c in COND]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=LEGEND_SIZE,
               frameon=True, fancybox=False, edgecolor='gray',
               bbox_to_anchor=(0.5, -0.02))
    # suptitle removed per the npj re-lettering instruction (caption replaces it)
    plt.tight_layout(w_pad=1.2, h_pad=1.4, rect=(0, 0.05, 1, 1))
    save_dual(fig, NEWFIGS / 'A6_acquisition_matrix')
    plt.close(fig)


if __name__ == '__main__':
    main()
