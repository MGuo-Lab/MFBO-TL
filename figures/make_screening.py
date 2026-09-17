'Figure 4a,b: LF/HF ranking overlap and screening regret.'
import os, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import NEWFIGS, letter_pt, save_dual
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
PKG = HERE.parent / 'src'
sys.path.insert(0, str(PKG))
import pools as POOLS
REPO = HERE.parent
ORDER = ['Branin-Fav', 'Branin-Unfav', 'Park-Fav', 'Park-Unfav', 'COFs', 'FreeSolv', 'Polarizability', 'HOPV15', 'Matbench-Gap', 'ExptGap-PBE', 'Elastic-CHGNet', 'Elastic-SevenNet', 'Elastic-MatterSim']
(REGIME_A, REGIME_B) = ('#9fb8ad', '#5b6770')
LABEL = {'Matbench-Gap': 'Matbench-gap'}
PALETTE = {'Branin-Fav': '#1f4e89', 'Branin-Unfav': '#7fb3e6', 'Park-Fav': '#5b2c83', 'Park-Unfav': '#b48ad8', 'COFs': '#e07b4f', 'FreeSolv': '#2f8f4e', 'Polarizability': '#1aa6b7', 'HOPV15': '#c2302f', 'Matbench-Gap': '#a08a12', 'ExptGap-PBE': '#8c564b', 'Elastic-CHGNet': '#f0a6d2', 'Elastic-SevenNet': '#c9559e', 'Elastic-MatterSim': '#7d1f5f'}
STYLE = {b: '-' if not b.startswith('Elastic') else '--' for b in ORDER}
(TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE) = (16, 14, 12, 12)
LETTER_PT = letter_pt(320.0, 160)
SCREENING_TOL = 0.3

def panel_letter(ax, letter, x, y, size):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=size, fontfamily='sans-serif', ha='left', va='bottom')

def load_y():
    ys = {}
    for name in ORDER:
        (b, cfg) = POOLS.make_bench(name, REPO / 'data')
        ys[name] = (np.asarray(b.y_hf, float), np.asarray(b.y_lf, float))
    return ys

def main():
    ys = load_y()
    rows = []
    k_fractions = np.linspace(0.005, 0.3, 60)
    curves = {}
    for bench in ORDER:
        (y_hf, y_lf) = ys[bench]
        n = len(y_hf)
        (o_hf, o_lf) = (np.argsort(y_hf, kind='stable'), np.argsort(y_lf, kind='stable'))
        ov = []
        for frac in k_fractions:
            k = max(1, int(frac * n))
            ov.append(len(set(o_hf[:k]) & set(o_lf[:k])) / k)
        curves[bench] = np.array(ov)
        (lf_best, hf_best) = (int(np.argmin(y_lf)), int(np.argmin(y_hf)))
        sr = float(y_hf[lf_best] - y_hf.min())
        n_lf_ties = int(np.sum(y_lf == y_lf.min()))
        sr_ties = y_hf[y_lf == y_lf.min()] - y_hf.min()
        r2 = float(np.corrcoef(y_hf, y_lf)[0, 1]) ** 2
        top10 = len(set(o_hf[:10]) & set(o_lf[:10])) / 10
        k1 = max(1, int(0.01 * n))
        rows.append(dict(benchmark=bench, n=n, r2=r2, top10_overlap=top10, top1pct_overlap=len(set(o_hf[:k1]) & set(o_lf[:k1])) / k1, screening_regret=sr, screening_regret_over_range=sr / float(y_hf.max() - y_hf.min()), top1_hit=int(lf_best == hf_best), hf_best_idx=hf_best, lf_best_idx=lf_best, lf_rank_of_hf_best=int(np.where(o_lf == hf_best)[0][0]) + 1, n_lf_ties_at_min=n_lf_ties, screening_regret_ties_min=float(sr_ties.min()), screening_regret_ties_max=float(sr_ties.max()), regime='A' if sr <= SCREENING_TOL else 'B'))
    met = pd.DataFrame(rows)
    met.to_csv(NEWFIGS / 'B2_topk_13_values.csv', index=False)
    print(met.to_string())
    (fig, axes) = plt.subplots(1, 2, figsize=(12.6, 6.4), gridspec_kw={'width_ratios': [1.15, 1]})
    (ax_curve, ax_bar) = axes
    for bench in ORDER:
        n = len(ys[bench][0])
        ax_curve.plot(k_fractions * 100, curves[bench], color=PALETTE[bench], lw=2.0, ls=STYLE[bench], label=f'{LABEL.get(bench, bench)} (n={n:,})')
    ax_curve.plot(k_fractions * 100, k_fractions, 'k:', alpha=0.6, lw=1.2, label='Random (E[overlap]=k/n)')
    ax_curve.set_xlabel('Top-k fraction (%)', fontsize=LABEL_SIZE)
    ax_curve.set_ylabel('Overlap (|LF top-k ∩ HF top-k|)/k', fontsize=LABEL_SIZE)
    ax_curve.set_title('Fidelity ranking alignment', fontsize=TITLE_SIZE)
    panel_letter(ax_curve, 'a', x=-0.1, y=1.02, size=LETTER_PT)
    ax_curve.set_xlim(0.5, 30)
    ax_curve.set_ylim(0, 1.05)
    ax_curve.grid(True, alpha=0.3, lw=0.5)
    ax_curve.tick_params(axis='both', labelsize=TICK_SIZE)
    fig.legend(*ax_curve.get_legend_handles_labels(), fontsize=LEGEND_SIZE - 1, loc='lower center', bbox_to_anchor=(0.5, 0.0), frameon=True, edgecolor='gray', ncol=5, columnspacing=1.2, handlelength=1.8)
    sr_full = met.set_index('benchmark')['screening_regret']
    regime_a = [b for b in ORDER if sr_full[b] <= SCREENING_TOL]
    regime_b = [b for b in ORDER if sr_full[b] > SCREENING_TOL]
    order = sorted(regime_a, key=lambda b: float(sr_full[b])) + sorted(regime_b, key=lambda b: float(sr_full[b]))
    sr = sr_full.loc[order]
    top1 = met.set_index('benchmark').loc[order, 'top1_hit']
    colors = [REGIME_A if b in regime_a else REGIME_B for b in order]
    y_pos = np.arange(len(order))
    xmax = float(np.abs(sr.values).max())
    ax_bar.set_xscale('symlog', linthresh=0.1, linscale=0.5)
    ax_bar.barh(y_pos, sr.values, color=colors, edgecolor='none', height=0.7, alpha=0.9)
    ax_bar.axvline(SCREENING_TOL, color='0.35', ls=(0, (4, 3)), lw=1.1, zorder=0)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([LABEL.get(b, b) for b in order], fontsize=TICK_SIZE - 1)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel('Screening regret $f_\\mathrm{HF}[\\arg\\min f_\\mathrm{LF}] - f^*_\\mathrm{HF}$ (HF units)', fontsize=LABEL_SIZE)
    ax_bar.set_title('LF-only screening regret', fontsize=TITLE_SIZE)
    panel_letter(ax_bar, 'b', x=-0.27, y=1.02, size=LETTER_PT)
    ax_bar.grid(axis='x', alpha=0.3, lw=0.5)
    ax_bar.tick_params(axis='both', labelsize=TICK_SIZE)
    for (i, (b, val)) in enumerate(zip(order, sr.values)):
        marker = ('✓ top-1 hit' if int(top1[b]) == 1 else '✗ top-1 miss') + f'  ({val:.3g})'
        ax_bar.text(val * 1.25 + 0.02, i, marker, va='center', fontsize=LEGEND_SIZE - 3, color='#333')
    ax_bar.set_xlim(0, xmax * 30)
    ax_bar.set_xticks([0, 0.1, 1, 10, 100, 1000])
    ax_bar.set_xticklabels(['0', '0.1', '1', '10', '100', '1000'])
    handles = [Patch(facecolor=REGIME_A, label='Regime A (regret ≤ 0.3)'), Patch(facecolor=REGIME_B, label='Regime B (regret > 0.3)')]
    ax_bar.legend(handles=handles, fontsize=LEGEND_SIZE - 1, loc='upper right', frameon=True, edgecolor='gray')
    plt.tight_layout(w_pad=3.0, rect=(0, 0.14, 1, 1))
    save_dual(fig, NEWFIGS / 'B2_topk_13')
    plt.close(fig)
if __name__ == '__main__':
    main()
