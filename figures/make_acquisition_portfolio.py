"""acquisition_portfolio.pdf -- letters a-d KEPT but restyled "(a)" -> bold
"a" outside the axes; the suptitle ("Acquisition portfolio across...")
REMOVED (caption replaces it). Data, colours, layout unchanged.

Adapted from experiments/acquisition_portfolio/plot_paper_figure.py.
Data: results/acq_portfolio/surrogates_winrate.csv + surrogates_means.csv
(the same cached analysis CSVs the manuscript figure was built from).
Login-node safe.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import NEWFIGS, RESULTS, add_panel_letter, letter_pt

# uniform 8 pt print size: tight PDF width 281.1 mm, print width 153 mm
LETTER_PT = letter_pt(281.1, 153)

AP = RESULTS / 'acq_portfolio'
ACQS = ['greedy', 'ei', 'pi', 'ucb', 'mes', 'ts']
UQ = ['ei', 'pi', 'ucb', 'mes', 'ts']
LABEL = {'greedy': 'Greedy', 'ei': 'EI', 'pi': 'PI', 'ucb': 'GP-UCB',
         'mes': 'MES', 'ts': 'Thompson'}
COLOR = {'greedy': '#2c6fbb', 'ei': '#9bb8d4', 'pi': '#7bb07b',
         'ucb': '#e0a458', 'mes': '#b07bb0', 'ts': '#c46c4e'}
BENCHES = ['Branin-Fav', 'Branin-Unfav', 'Park-Fav', 'Park-Unfav', 'COFs',
           'FreeSolv', 'Polarizability', 'HOPV15', 'Matbench-Gap']
EXAMPLES = ['COFs', 'FreeSolv', 'Polarizability']

wr = pd.read_csv(AP / 'surrogates_winrate.csv', index_col=0)
means = pd.read_csv(AP / 'surrogates_means.csv')

fig = plt.figure(figsize=(12.5, 7.0))
gs = GridSpec(2, 3, height_ratios=[1.05, 1.0], hspace=0.42, wspace=0.28)

# ---- top: win-rate heatmap (panel a) ----
axh = fig.add_subplot(gs[0, :])
data = wr.reindex(UQ)[BENCHES].values.astype(float)
im = axh.imshow(data, cmap='RdBu', vmin=0, vmax=1, aspect='auto')
axh.set_xticks(range(len(BENCHES)))
axh.set_xticklabels(BENCHES, rotation=28, ha='right', fontsize=9)
axh.set_yticks(range(len(UQ)))
axh.set_yticklabels([LABEL[a] for a in UQ], fontsize=10)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        v = data[i, j]
        if np.isfinite(v):
            axh.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=8,
                     color='white' if (v < 0.22 or v > 0.78) else '#222')
axh.set_title('Fraction of 11 transfer-learning surrogates on which each uncertainty '
              'acquisition beats greedy\n(red: uncertainty rarely beats greedy; blue: '
              'uncertainty usually beats greedy)', fontsize=10.5)
add_panel_letter(axh, 'a', x=-0.06, y=1.02, size=LETTER_PT)
cb = fig.colorbar(im, ax=axh, fraction=0.018, pad=0.01)
cb.set_label('fraction beating greedy', fontsize=8)
cb.ax.tick_params(labelsize=7)

# ---- bottom: three representative benchmarks (panels b-d) ----
for k, b in enumerate(EXAMPLES):
    ax = fig.add_subplot(gs[1, k])
    mu, se = [], []
    for a in ACQS:
        vals = means[(means.benchmark == b) & (means.acq == a)]['mean'].dropna().values
        mu.append(np.mean(vals) if len(vals) else np.nan)
        se.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
    x = np.arange(len(ACQS))
    ax.bar(x, mu, yerr=se, color=[COLOR[a] for a in ACQS], alpha=0.92,
           edgecolor='white', width=0.74, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[a] for a in ACQS], rotation=35, ha='right', fontsize=8)
    ax.set_title(b, fontsize=11)
    add_panel_letter(ax, chr(98 + k), x=-0.16 if k == 0 else -0.12, y=1.02, size=LETTER_PT)
    if k == 0:
        ax.set_ylabel('final regret\n(mean over surrogates)', fontsize=9)
    ax.grid(axis='y', alpha=0.3, lw=0.5)
    ytop = max([m + s for m, s in zip(mu, se) if np.isfinite(m)] + [1e-6])
    for xi, m, s in zip(x, mu, se):
        if np.isfinite(m):
            ax.text(xi, m + s + ytop * 0.03, f'{m:.2f}', ha='center', va='bottom',
                    fontsize=8, color='#222')
    ax.set_ylim(0, ytop * 1.16)

# suptitle removed per the npj re-lettering instruction (caption replaces it)
for ext in ('pdf', 'png'):
    fig.savefig(NEWFIGS / f'acquisition_portfolio.{ext}', dpi=200, bbox_inches='tight')
    print('wrote', NEWFIGS / f'acquisition_portfolio.{ext}')
