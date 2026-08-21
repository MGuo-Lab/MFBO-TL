"""Two outputs from the controlled Polariz family-split grid analysis
(adapted from experiments/regime_review_20260625/make_fig1c_family_split.py;
identical data pipeline, colours and statistics):

1) fig1c_family_split_grid.pdf (SI, full 3x3 grid) -- NEW bold letters a-i,
   row-major: a-c = Final-regret advantage row, d-f = Anytime-AUC advantage
   row, g-i = Marginal row. Content unchanged.

2) fig1_anytime_row.pdf (NEW, main-text Fig 1 panel row) -- ONLY the middle
   "Anytime-AUC advantage" heatmap row, letters l, m, n (left to right:
   TL vs MFGP baseline, TL vs MFGP variants, MFGP variants vs MFGP baseline),
   with the row-shared colour scale, significance stars and per-panel
   colorbars. Total width <= 12 in; ticks designed >= 9 pt so the print at
   174 mm keeps the smallest text above 5 pt.

Star criterion (changed 2026-07-24): asterisks now mark bins significant under
one stratified signed-rank test per non-empty bin (the bin's cells as strata,
sign-flip randomization) with Benjamini-Hochberg FDR at q = 0.05 across each
map's bins, computed by experiments/regime_review_20260625/grid_stats.py. The
previous uncorrected majority-vote rule (>= half of the bin's cells with raw
p < 0.05) is kept only in the SI sensitivity figure.

Reads cached grid CSVs only (login-node safe).
"""
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(os.path.dirname(os.path.abspath(__file__))))
from _common import NEWFIGS, REPO, add_panel_letter, letter_pt

import grid_stats                       # lives in this directory

# uniform 8 pt print size, computed per output from its tight PDF width:
# 2026-08-18 print-size re-export of the SI grid: 16.2x13.2 in -> 12.3x10 in
# (~1.8x the 174 mm print width); fonts raised so body text prints >= 5 pt.
LETTER_PT_GRID = letter_pt(315.0, 174)   # fig1c_family_split_grid (SI, 174 mm)
LETTER_PT_ROW = letter_pt(302.1, 174)    # fig1_anytime_row (main text, 174 mm)

warnings.filterwarnings('ignore')
GRID = REPO / 'results' / 'grid'

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'legend.fontsize': 8.5, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.dpi': 130, 'savefig.dpi': 165, 'axes.linewidth': 0.8,
})

FAMILIES = {
    'TL': ['Progressive', 'KnowledgeDistillation', 'PseudoLabeling'],
    'MFGP baseline': ['MFGP'],
    'MFGP variants': ['DKL', 'SparseMFGP'],
}
COMPARISONS = [
    ('TL', 'MFGP baseline'),
    ('TL', 'MFGP variants'),
    ('MFGP variants', 'MFGP baseline'),
]
METRICS = [('final_regret', 'Final-regret advantage'),
           ('auc', 'Anytime-AUC advantage')]
RED = '#c0392b'
GREEN = '#1f7a4d'
R2B = np.round(np.arange(0.1, 0.91, 0.1), 2)
T10 = np.round(np.arange(0.0, 1.01, 0.1), 2)


def despine(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def load_grid():
    man = pd.read_csv(GRID / 'grid_manifest.csv').set_index('cell_id')
    cells = {}
    for f in glob.glob(str(GRID / 'cells' / 'summary_cell_*.csv')):
        cid = os.path.basename(f).replace('summary_', '').replace('.csv', '')
        if cid in man.index:
            cells[cid] = pd.read_csv(f)
    print(f'loaded {len(cells)} grid cells')
    rows = []
    for cid, d in cells.items():
        rec = dict(cell=cid, r2=man.loc[cid, 'r2'], top10=man.loc[cid, 'top10'])
        ok = True
        for metric in ('final_regret', 'auc'):
            vals = {m: dict(zip(g.seed, g[metric])) for m, g in d.groupby('model')}
            best = {}
            for fam, members in FAMILIES.items():
                have = [m for m in members if m in vals and len(vals[m])]
                if not have:
                    ok = False
                    break
                means = {m: np.mean(list(vals[m].values())) for m in have}
                b = min(means, key=means.get)
                best[fam] = (b, vals[b], means[b])
            if not ok:
                break
            for first, second in COMPARISONS:
                key = f'{first}|{second}'
                n1, v1, m1 = best[first]
                n2, v2, m2 = best[second]
                seeds = sorted(set(v1) & set(v2))
                a1 = np.array([v1[s] for s in seeds])
                a2 = np.array([v2[s] for s in seeds])
                p = np.nan
                if len(seeds) >= 6 and not np.allclose(a1, a2):
                    try:
                        p = float(wilcoxon(a1, a2).pvalue)
                    except Exception:
                        pass
                rec[f'adv_{metric}_{key}'] = m2 - m1
                rec[f'p_{metric}_{key}'] = p
        if ok:
            rows.append(rec)
    G = pd.DataFrame(rows).dropna(
        subset=[f'adv_{m}_{a}|{b}' for m, _ in METRICS for a, b in COMPARISONS])
    print(f'usable cells: {len(G)}')
    G['r2bin'] = R2B[np.argmin(np.abs(G['r2'].values[:, None] - R2B[None, :]), axis=1)]
    G['t10bin'] = np.round(G['top10'], 2)
    return G


def row_mats(G, perbin, metric):
    """Advantage matrices + corrected star matrices for one metric row, plus
    the row-shared colour scale. Stars: one stratified signed-rank test per
    non-empty bin, BH-FDR at q = 0.05 across each map's bins (grid_stats.py)."""
    mats, sigs = [], []
    for first, second in COMPARISONS:
        key = f'{first}|{second}'
        adv = (G.groupby(['t10bin', 'r2bin'])[f'adv_{metric}_{key}'].mean()
               .unstack('r2bin').reindex(index=T10, columns=R2B))
        pb = perbin[(perbin.metric == metric) & (perbin.comparison == key)]
        star = (pb.pivot(index='t10bin', columns='r2bin', values='star_new')
                .reindex(index=T10, columns=R2B))
        mats.append(adv)
        sigs.append(star)
    vmax = np.nanmax([np.nanmax(np.abs(m.values)) for m in mats])
    return mats, sigs, vmax


def draw_heatmap(fig, ax, adv, sig, vmax, first, second, mtitle, col0,
                 cb_fontsize=10):
    ax.set_facecolor('0.92')
    im = ax.imshow(adv.values, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   aspect='auto',
                   extent=[R2B[0] - .05, R2B[-1] + .05, T10[0] - .05, T10[-1] + .05])
    for i, t in enumerate(T10):
        for j, rr in enumerate(R2B):
            if sig.values[i, j] == True:  # noqa: E712 (NaN-safe: empty bins skip)
                ax.text(rr, t, '*', ha='center', va='center', fontsize=13,
                        fontweight='bold')
    ax.set_xlabel('global LF-HF $R^2$', fontsize=11)
    if col0:
        ax.set_ylabel(f'{mtitle}\n\ntop-10 optimum agreement', fontsize=11)
    else:
        ax.set_ylabel('top-10 optimum agreement', fontsize=11)
    ax.set_title(f'{first}  vs  {second}\n(red: {first} better, blue: {second} better)',
                 fontsize=11)
    ax.tick_params(labelsize=10.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('advantage', fontsize=cb_fontsize)
    cb.ax.tick_params(labelsize=cb_fontsize)


def full_grid(G, perbin):
    """The SI 3x3 grid with NEW letters a-i (row-major)."""
    fig, axes = plt.subplots(3, 3, figsize=(12.3, 10.0))
    letters = list('abcdefghi')
    for r, (metric, mtitle) in enumerate(METRICS):
        mats, sigs, vmax = row_mats(G, perbin, metric)
        for c, ((first, second), adv, sig) in enumerate(zip(COMPARISONS, mats, sigs)):
            ax = axes[r][c]
            draw_heatmap(fig, ax, adv, sig, vmax, first, second, mtitle, col0=(c == 0))
            add_panel_letter(ax, letters[r * 3 + c], x=-0.20, y=1.13, size=LETTER_PT_GRID)
    for c, (first, second) in enumerate(COMPARISONS):
        ax = axes[2][c]
        for metric, lab, col, mk in [('final_regret', 'final regret', RED, 'o'),
                                     ('auc', 'anytime AUC', GREEN, 's')]:
            key = f'{first}|{second}'
            m = G.groupby('t10bin')[f'adv_{metric}_{key}'].agg(['mean', 'sem']).reindex(T10).dropna()
            ax.errorbar(m.index, m['mean'], yerr=m['sem'], marker=mk, color=col, lw=2,
                        capsize=3, label=lab)
        ax.axhline(0, color='k', lw=0.8, ls=':')
        ax.set_xlabel('top-10 optimum agreement', fontsize=11)
        ax.set_ylabel(f'advantage  (>0: {first} better)', fontsize=11)
        ax.set_title(f'Marginal: {first} vs {second}\n(averaged over $R^2$)', fontsize=11)
        ax.tick_params(labelsize=10.5)
        ax.legend(fontsize=10)
        despine(ax)
        ax.grid(alpha=0.25)
        add_panel_letter(ax, letters[6 + c], x=-0.20, y=1.13, size=LETTER_PT_GRID)
    lo = min(axes[2][c].get_ylim()[0] for c in range(3))
    hi = max(axes[2][c].get_ylim()[1] for c in range(3))
    for c in range(3):
        axes[2][c].set_ylim(lo, hi)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(NEWFIGS / f'fig1c_family_split_grid.{ext}', bbox_inches='tight')
    print('wrote', NEWFIGS / 'fig1c_family_split_grid.pdf + .png')
    plt.close(fig)


def anytime_row(G, perbin):
    """NEW main-text export: the Anytime-AUC advantage heatmap row only,
    letters l, m, n."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1))
    metric, mtitle = 'auc', 'Anytime-AUC advantage'
    mats, sigs, vmax = row_mats(G, perbin, metric)
    for c, ((first, second), adv, sig) in enumerate(zip(COMPARISONS, mats, sigs)):
        ax = axes[c]
        # cb_fontsize 9: at the 174 mm print width (12 in figure) the smallest
        # text stays >= 5 pt (9 pt x 0.576 = 5.2 pt)
        draw_heatmap(fig, ax, adv, sig, vmax, first, second, mtitle,
                     col0=(c == 0), cb_fontsize=9)
        add_panel_letter(ax, 'lmn'[c], x=-0.20 if c == 0 else -0.16, y=1.10, size=LETTER_PT_ROW)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(NEWFIGS / f'fig1_anytime_row.{ext}', bbox_inches='tight')
    print('wrote', NEWFIGS / 'fig1_anytime_row.pdf + .png')
    plt.close(fig)


if __name__ == '__main__':
    G = load_grid()
    _, PERBIN, _ = grid_stats.compute('split')
    full_grid(G, PERBIN)
    anytime_row(G, PERBIN)
