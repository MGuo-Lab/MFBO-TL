'Supplementary Figure 3: held-out LF calibration and attainment.'
import glob, os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R = Path(__file__).resolve().parents[1]
from matplotlib.ticker import MaxNLocator
OUT = Path(__file__).resolve().parent / 'out'
OUT.mkdir(exist_ok=True)
XCOL = {'mean': 'ece_lf_mean', 'final': 'ece_lf_final'}[os.environ.get('ECE_X', 'mean')]
POOLS = ['Branin-Fav', 'Branin-Unfav', 'Park-Fav', 'Park-Unfav', 'COFs', 'FreeSolv', 'Polarizability', 'HOPV15', 'Matbench-Gap', 'ExptGap-PBE', 'Elastic-CHGNet', 'Elastic-SevenNet', 'Elastic-MatterSim']
BUDGET = {p: 50 for p in POOLS[:4]}
BUDGET.update({p: 30 for p in POOLS[4:9]})
BUDGET.update({p: 30 for p in POOLS[9:]})
NAMES = {'SparseMFGP': 'Sparse MFGP', 'DNGOGradient': 'End-to-End Joint', 'DNGOJoint': 'Frozen-representation transfer', 'TwoStageJoint': 'Pretrain-then-Joint', 'DomainAdaptationMMD': 'Domain Adaptation (MMD)', 'SoftParameterSharing': 'Soft Parameter Sharing'}
GP = ['MFGP', 'Sparse MFGP', 'DKL']
TL = ['Frozen-representation transfer', 'Pretrain-then-Joint', 'End-to-End Joint', 'Soft Parameter Sharing', 'Domain Adaptation (MMD)']
MODELS = GP + TL
summ = pd.read_csv(Path(__file__).resolve().parents[1] / 'results/main/calibration.csv')
summ['model'] = summ.model.replace(NAMES)
ece = summ.groupby(['benchmark', 'model'])[['ece_lf_mean', 'ece_lf_final']].mean().reset_index()
att = pd.read_csv(OUT / 'fig1_attainment_values.csv').rename(columns={'pool': 'benchmark', 'mean': 'attainment', 'n': 'n_att'})
assert set(att.model) == set(MODELS), set(att.model) ^ set(MODELS)
T = ece.merge(att[['benchmark', 'model', 'attainment', 'n_att', 'se']], on=['benchmark', 'model'])
LABEL = {'Matbench-Gap': 'Matbench-gap'}
ABBR = {'MFGP': 'GP-base', 'Sparse MFGP': 'GP-SV', 'DKL': 'GP-DKL', 'Frozen-representation transfer': 'TL-base', 'Pretrain-then-Joint': 'TL-PtJ', 'End-to-End Joint': 'TL-E2E', 'Domain Adaptation (MMD)': 'TL-MMD', 'Soft Parameter Sharing': 'TL-SPS'}
STYLE = {'MFGP': ('#b5532e', 'o'), 'Sparse MFGP': ('#f2aa84', 's'), 'DKL': ('#e07b4f', 'D'), 'Frozen-representation transfer': ('#4e95d9', 'P'), 'Pretrain-then-Joint': ('#0b3d7a', 'D'), 'End-to-End Joint': ('#2467b3', '^'), 'Domain Adaptation (MMD)': ('#8fc0ec', 'X'), 'Soft Parameter Sharing': ('#8fc0ec', 'h')}
MS = {'o': 13, 's': 11, 'D': 9.5, '^': 14, 'v': 14, '*': 24, 'P': 15, 'X': 15, 'h': 14}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6.5, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2, 'ytick.major.size': 2, 'pdf.fonttype': 42, 'ps.fonttype': 42})
(TITLE, TICK, LAB) = (6.8, 6, 6.5)
XLAB = 'LF calibration error\n(held-out, loop mean)' if XCOL == 'ece_lf_mean' else 'LF calibration error\n(held-out, final refit)'
LETTERS = list(os.environ.get('CAL_LETTERS', 'abcdefghijklm'))
(fig, axes) = plt.subplots(3, 5, figsize=(7.2, 5.0))
axes = axes.ravel()
rows = []
for (i, b) in enumerate(POOLS):
    ax = axes[i]
    t = T[T.benchmark == b].set_index('model').reindex(MODELS).dropna(subset=[XCOL, 'attainment'])
    for (m, r) in t.iterrows():
        (col, mk) = STYLE[m]
        ax.errorbar(r[XCOL], r.attainment, yerr=r.se, fmt='none', ecolor='#9a9a9a', elinewidth=0.45, capsize=0, zorder=2)
        ax.scatter(r[XCOL], r.attainment, s=MS[mk], marker=mk, color=col, edgecolor='#222222', linewidth=0.3, zorder=3)
    rr = t[[XCOL, 'attainment']].corr().iloc[0, 1] if t.attainment.std() > 0 else np.nan
    rs = t[[XCOL, 'attainment']].corr(method='spearman').iloc[0, 1] if t.attainment.std() > 0 else np.nan
    rows.append(dict(benchmark=b, pearson_r=rr, spearman_rho=rs, n_models=len(t)))
    ax.text(0.97, 0.05, f'Pearson $r$ = {rr:+.2f}', transform=ax.transAxes, fontsize=5.3, va='bottom', ha='right', color='#333333', bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#bbbbbb', linewidth=0.4, alpha=0.9))
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(['0', '0.5', '1'], fontsize=TICK)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.tick_params(axis='x', labelsize=TICK, pad=1.5)
    ax.tick_params(axis='y', labelsize=TICK, pad=1.5)
    ax.text(0.0, 1.13, f'{LETTERS[i]}  {LABEL.get(b, b)}', transform=ax.transAxes, ha='left', va='bottom', fontsize=TITLE)
    ax.text(0.0, 1.02, f'B = {BUDGET[b]}', transform=ax.transAxes, ha='left', va='bottom', fontsize=5.3, color='#555555')
    ax.grid(lw=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    if i >= 8:
        ax.set_xlabel(XLAB, fontsize=6.0, labelpad=1.5)
    if i % 5 == 0:
        ax.set_ylabel('Attainment', fontsize=LAB, labelpad=1.5)
for a in (axes[13], axes[14]):
    a.axis('off')

def _h(m):
    (col, mk) = STYLE[m]
    return Line2D([0], [0], marker=mk, ls='', markersize=np.sqrt(MS[mk]) * 1.05, markerfacecolor=col, markeredgecolor='#222222', markeredgewidth=0.3, label=ABBR[m])
axes[13].legend(handles=[_h(m) for m in GP], loc='upper left', fontsize=5.8, frameon=False, title='GP family', title_fontsize=6.0, labelspacing=0.45, borderaxespad=0, handletextpad=0.5)
axes[14].legend(handles=[_h(m) for m in TL], loc='upper left', fontsize=5.8, frameon=False, title='TL surrogates', title_fontsize=6.0, labelspacing=0.45, borderaxespad=0, handletextpad=0.5)
fig.subplots_adjust(left=0.065, right=0.982, top=0.94, bottom=0.13, wspace=0.55, hspace=0.62)
fig.text(0.5, 0.032, 'Each point is one surrogate: mean attainment ± s.e. over seeds (Fig. 1) against its held-out LF calibration error averaged over the refits of the loop.', ha='center', va='bottom', fontsize=5.0, color='#333333')
fig.text(0.5, 0.016, 'TL: TL-base = base transfer-learning surrogate · TL-PtJ = pretrain-then-joint · TL-E2E = end-to-end joint · TL-SPS = soft parameter sharing · TL-MMD = domain adaptation (MMD)', ha='center', va='bottom', fontsize=5.0, color='#333333')
fig.text(0.5, 0.0, 'GP: GP-base = autoregressive multi-fidelity GP (MFGP) · GP-SV = sparse variational MFGP · GP-DKL = deep-kernel GP', ha='center', va='bottom', fontsize=5.0, color='#333333')
tag = 'mean' if XCOL == 'ece_lf_mean' else 'final'
STEM = os.environ.get('CAL_STEM', 'fig4_calibration_attain')
fig.savefig(OUT / f'{STEM}.pdf')
fig.savefig(OUT / f'{STEM}.png', dpi=110)
T.to_csv(OUT / f'{STEM}_values.csv', index=False)
print(f'x = {XCOL}')
print(pd.DataFrame(rows).round(2).to_string(index=False))
