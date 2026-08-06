"""A1_calibration_vs_regret.pdf -- panel letters shifted (a-i) -> (e-m) for
the composite figure. Everything else (colours, per-surrogate palette, legend
slot, "(B = 10)" Park tags, Pearson-r boxes) identical to the manuscript PDF.

Adapted from paper/auto_figures/plot_a1_calibration.py::scatter_figure().
Data: results/calibration_sweep + calibration_newbench +
calibration_synthetic + calibration_park_early (Park B=10 override) +
calibration_gpfamily. Reproduction verified: per-benchmark Pearson r matches
the manuscript figure exactly (-0.31, -0.55, -0.51, -0.46, -0.88, -0.43,
-0.64, -0.08, -0.20). Login-node safe.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import NEWFIGS, RESULTS, RENAME_MAP, add_panel_letter, letter_pt, save_dual

TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE = 16, 14, 12, 12
HIGHLIGHT_COLOR = '#4e95d9'
DEFAULT_PALETTE = ['#2ca02c', '#d62728', '#9467bd', '#8c564b',
                   '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                   '#4d4d4d', '#bdbdbd', '#1f77b4', '#ff7f0e',
                   '#0d8a8a', '#a04500']

BENCHES = ['Branin-Fav', 'Branin-Unfav', 'Park-Fav', 'Park-Unfav',
           'COFs', 'FreeSolv', 'Polarizability', 'HOPV15', 'Matbench-Gap']
EARLY_BUDGET_BENCHES = ['Park-Fav', 'Park-Unfav']
GP_FAMILY = ['MFGP', 'NARGP', 'DKL Multi-Fidelity', 'Sparse MFGP']
TL_MODELS = ['Sequential', 'Progressive', 'Curriculum',
             'Pretrain-then-Joint Training', 'Stop-Gradient Joint Training',
             'End-to-End Joint Training', 'Knowledge Distillation',
             'Domain Adaptation (MMD)', 'Soft Parameter Sharing',
             'Pseudo-Labelling', 'Adapter']
MODELS = GP_FAMILY + TL_MODELS
GP_WARM = ['#f2aa84', '#e0714a', '#9c3318', '#d4a373']
MODEL_COLOR = {m: GP_WARM[i % len(GP_WARM)] for i, m in enumerate(GP_FAMILY)}
for i, m in enumerate(TL_MODELS):
    MODEL_COLOR[m] = DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)]

LETTERS = list('efghijklm')     # (a-i) -> (e-m)
# uniform 8 pt print size: tight PDF width 657.8 mm, print width 174 mm
LETTER_PT = letter_pt(657.8, 174)


def load_summary():
    parts = [pd.read_csv(RESULTS / 'calibration_sweep' / 'results_summary.csv')]
    for extra in ('calibration_newbench', 'calibration_synthetic'):
        f = RESULTS / extra / 'results_summary.csv'
        if f.exists():
            parts.append(pd.read_csv(f))
    summary = pd.concat(parts, ignore_index=True)
    early = pd.read_csv(RESULTS / 'calibration_park_early' / 'results_summary.csv')
    early = early[early['benchmark'].isin(EARLY_BUDGET_BENCHES)]
    summary = summary[~summary['benchmark'].isin(EARLY_BUDGET_BENCHES)]
    summary = pd.concat([summary, early], ignore_index=True)
    gpf = pd.read_csv(RESULTS / 'calibration_gpfamily' / 'results_summary.csv')
    summary = pd.concat([summary, gpf], ignore_index=True)
    summary['model'] = summary['model'].replace(RENAME_MAP)
    summary['model'] = summary['model'].replace({'Pseudo-Labeling': 'Pseudo-Labelling'})
    return summary


def main():
    summary = load_summary()
    present = [m for m in MODELS if m in set(summary['model'])]
    fig, axes = plt.subplots(2, 5, figsize=(26, 10))
    axes = axes.ravel()
    for idx, b in enumerate(BENCHES):
        ax = axes[idx]
        tag = '  (B$=$10)' if b in EARLY_BUDGET_BENCHES else ''
        bd = summary[summary['benchmark'] == b]
        per_model = (bd.groupby('model')[['lf_ece', 'final_regret']]
                       .mean().reindex(present).reset_index())
        for _, row in per_model.iterrows():
            if pd.isna(row['lf_ece']):
                continue
            ax.scatter(row['lf_ece'], row['final_regret'],
                       s=130, alpha=0.9, edgecolor='black', linewidth=0.6,
                       color=MODEL_COLOR.get(row['model'], '#777777'), zorder=3)
        if len(per_model.dropna()) > 2:
            r = per_model[['lf_ece', 'final_regret']].dropna().corr().iloc[0, 1]
            ax.text(0.95, 0.95, f'Pearson $r = {r:+.2f}$',
                    transform=ax.transAxes, fontsize=LEGEND_SIZE,
                    va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='gray', alpha=0.9))
        ax.set_xlabel('LF ECE (held-out)', fontsize=LABEL_SIZE)
        if idx % 5 == 0:
            ax.set_ylabel('Final BO regret', fontsize=LABEL_SIZE)
        ax.set_title(f'{b}{tag}', fontsize=TITLE_SIZE)
        add_panel_letter(ax, LETTERS[idx], x=-0.10, y=1.02, size=LETTER_PT)
        ax.grid(True, alpha=0.3, lw=0.5)
        ax.tick_params(axis='both', labelsize=TICK_SIZE)
    axl = axes[9]
    axl.axis('off')
    handles = [Line2D([0], [0], marker='o', linestyle='', markersize=9,
                      markerfacecolor=MODEL_COLOR[m], markeredgecolor='black',
                      markeredgewidth=0.5, label=m) for m in present]
    axl.legend(handles=handles, loc='center', fontsize=LEGEND_SIZE - 1,
               frameon=True, edgecolor='gray', ncol=1, labelspacing=0.7,
               title='GP family (warm) / Transfer-learning (cool)',
               title_fontsize=LEGEND_SIZE - 1)
    plt.tight_layout(w_pad=2.0, h_pad=2.5)
    save_dual(fig, NEWFIGS / 'A1_calibration_vs_regret')
    plt.close(fig)


if __name__ == '__main__':
    main()
