"""B2_topk_overlap_and_screening.pdf -- letters a, b KEPT but restyled
"(a)" -> bold "a" outside the axes. Data, colours, layout, legends unchanged.

Adapted from paper/auto_figures/plot_b2_topk.py. The (y_hf, y_lf) arrays are
rebuilt exactly as src/benchmark.py does, but WITHOUT importing it (that
would pull torch/botorch/rdkit onto the login node):
  * synthetic: same grid construction as SyntheticBenchmark._create_grid
    (Branin dim=2: 50x50 meshgrid; Park dim=4: ceil(sqrt(10))=4 points/dim,
    indexing='ij') + the same hf/lf functions from src/synthetic_functions.py
    with the same alpha values as plot_b2_topk.load_benchmark_y.
  * chemistry: y_hf/y_lf are the CSV HF/LF columns with the same negate
    flags (COFs/Polarizability/HOPV15 negated) -- descriptors/PCA only affect
    X, never y, so the overlap curves are bit-identical.
Panel (b) reads results/ranking_analysis/ranking_metrics.csv
unchanged. Login-node safe.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import NEWFIGS, REPO, RESULTS, add_panel_letter, letter_pt, save_dual

# uniform 8 pt print size: tight PDF width 378.2 mm, print width 160 mm
# 2026-08-18 print-size re-export: 15x6 in -> 12.6x5.2 in (~2.0x print width)
LETTER_PT = letter_pt(320.0, 160)

sys.path.insert(0, str(REPO / 'src'))
from synthetic_functions import branin_hf, branin_lf, park_hf, park_lf  # noqa: E402

RANKING = RESULTS / 'ranking_analysis'
DATA = REPO / 'data'
BENCHMARKS9 = ['Branin-Fav', 'Branin-Unfav', 'Park-Fav', 'Park-Unfav',
               'COFs', 'FreeSolv', 'Polarizability', 'HOPV15', 'Matbench-Gap']
HIGHLIGHT_COLOR, MFGP_COLOR = '#4e95d9', '#f2aa84'
TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE = 16, 14, 12, 12


def synth_grid(dim, grid_size):
    """Replicates SyntheticBenchmark._create_grid (src/benchmark.py:1184)."""
    if dim == 2:
        axes = [np.linspace(0, 1, grid_size) for _ in range(2)]
        grids = np.meshgrid(*axes)
    else:
        n_per_dim = int(np.ceil(grid_size ** 0.5))
        axes = [np.linspace(0, 1, n_per_dim) for _ in range(dim)]
        grids = np.meshgrid(*axes, indexing='ij')
    return np.column_stack([g.ravel() for g in grids])


def chem_y(csv, negate):
    df = pd.read_csv(DATA / csv)
    y_hf = df['HF'].values.astype(float).flatten()
    y_lf = df['LF'].values.astype(float).flatten()
    if negate:
        y_hf, y_lf = -y_hf, -y_lf
    return y_hf, y_lf


def load_benchmark_y():
    ys = {}
    X2 = synth_grid(2, 50)
    X4 = synth_grid(4, 10)
    ys['Branin-Fav'] = (branin_hf(X2).flatten(), branin_lf(X2, 0.8).flatten())
    ys['Branin-Unfav'] = (branin_hf(X2).flatten(), branin_lf(X2, 0.1).flatten())
    ys['Park-Fav'] = (park_hf(X4).flatten(), park_lf(X4, 0.6).flatten())
    ys['Park-Unfav'] = (park_hf(X4).flatten(), park_lf(X4, 0.0).flatten())
    ys['COFs'] = chem_y('cofs.csv', True)
    ys['FreeSolv'] = chem_y('freesolv.csv', False)
    ys['Polarizability'] = chem_y('polarizability.csv', True)
    ys['HOPV15'] = chem_y('hopv15.csv', True)
    ys['Matbench-Gap'] = chem_y('matbench_gap.csv', False)
    return ys


def main():
    metrics = pd.read_csv(RANKING / 'ranking_metrics.csv')
    metrics = metrics.set_index('benchmark').loc[BENCHMARKS9].reset_index()
    ys = load_benchmark_y()

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2),
                             gridspec_kw={'width_ratios': [1.15, 1]})
    ax_curve, ax_bar = axes

    # ---- (a) top-k overlap curves ----
    k_fractions = np.linspace(0.005, 0.30, 60)
    palette = {b: c for b, c in zip(
        BENCHMARKS9,
        [HIGHLIGHT_COLOR, '#1f77b4', '#9467bd', '#8c564b',
         MFGP_COLOR, '#2ca02c', '#17becf', '#d62728', '#bcbd22'])}
    for bench in BENCHMARKS9:
        y_hf, y_lf = ys[bench]
        n = len(y_hf)
        overlaps = []
        for frac in k_fractions:
            k = max(1, int(frac * n))
            top_hf = set(np.argsort(y_hf)[:k])
            top_lf = set(np.argsort(y_lf)[:k])
            overlaps.append(len(top_hf & top_lf) / k)
        ax_curve.plot(k_fractions * 100, overlaps, color=palette[bench], lw=2.2,
                      label=f'{bench} (n={n})')
    ax_curve.plot(k_fractions * 100, k_fractions,
                  'k--', alpha=0.45, lw=1.2, label='Random (E[overlap]=k/n)')
    ax_curve.set_xlabel('Top-k fraction (%)', fontsize=LABEL_SIZE)
    ax_curve.set_ylabel('Overlap (|LF top-k ∩ HF top-k|)/k', fontsize=LABEL_SIZE)
    ax_curve.set_title('Fidelity ranking alignment', fontsize=TITLE_SIZE)
    add_panel_letter(ax_curve, 'a', x=-0.10, y=1.02, size=LETTER_PT)
    ax_curve.set_xlim(0.5, 30)
    ax_curve.set_ylim(0, 1.05)
    ax_curve.grid(True, alpha=0.3, lw=0.5)
    ax_curve.tick_params(axis='both', labelsize=TICK_SIZE)
    ax_curve.legend(fontsize=LEGEND_SIZE - 1, loc='lower right',
                    frameon=True, edgecolor='gray')

    # ---- (b) screening regret per benchmark ----
    SCREENING_TOL = 0.3
    sr_full = metrics.set_index('benchmark')['screening_regret']
    regime_a = [b for b in BENCHMARKS9 if sr_full[b] <= SCREENING_TOL]
    regime_b = [b for b in BENCHMARKS9 if sr_full[b] > SCREENING_TOL]
    order = (sorted(regime_a, key=lambda b: float(sr_full[b]))
             + sorted(regime_b, key=lambda b: float(sr_full[b])))
    sr = metrics.set_index('benchmark').loc[order, 'screening_regret']
    top1 = metrics.set_index('benchmark').loc[order, 'top1_hit']
    colors = [HIGHLIGHT_COLOR if b in regime_a else MFGP_COLOR for b in order]

    y_pos = np.arange(len(order))
    ax_bar.barh(y_pos, sr.values, color=colors, edgecolor='none',
                height=0.7, alpha=0.9)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(order, fontsize=TICK_SIZE)
    ax_bar.set_xlabel(r'Screening regret $f_\mathrm{HF}[\arg\min f_\mathrm{LF}] - f^*_\mathrm{HF}$',
                      fontsize=LABEL_SIZE)
    ax_bar.set_title('LF-only screening regret', fontsize=TITLE_SIZE)
    add_panel_letter(ax_bar, 'b', x=-0.21, y=1.02, size=LETTER_PT)
    ax_bar.grid(axis='x', alpha=0.3, lw=0.5)
    ax_bar.tick_params(axis='both', labelsize=TICK_SIZE)
    for i, (b, val) in enumerate(zip(order, sr.values)):
        marker = '✓ top-1 hit' if int(top1[b]) == 1 else '✗ top-1 miss'
        ax_bar.text(val + 0.02 * abs(sr.values).max() + 1e-6, i, marker,
                    va='center', fontsize=LEGEND_SIZE - 1, color='#333')
    handles = [Patch(facecolor=HIGHLIGHT_COLOR,
                     label='Regime A (regret ≤ 0.3)'),
               Patch(facecolor=MFGP_COLOR,
                     label='Regime B (regret > 0.3)')]
    ax_bar.legend(handles=handles, fontsize=LEGEND_SIZE - 1,
                  loc='lower right', frameon=True, edgecolor='gray')

    plt.tight_layout(w_pad=3.0)
    save_dual(fig, NEWFIGS / 'B2_topk_overlap_and_screening')
    plt.close(fig)


if __name__ == '__main__':
    main()
