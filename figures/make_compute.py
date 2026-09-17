'Supplementary Figure 4: compute per optimization run.'
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
import sys
sys.path.insert(0, str(HERE))
from _common import short, code
from _common import NEWFIGS as _NF
OUT = _NF
OUT.mkdir(exist_ok=True)
PROF = REPO / 'results' / 'flop_profile_blr' / 'flop_profile.csv'
BENCHES = ['Branin-Fav', 'Branin-Unfav', 'Park-Fav', 'Park-Unfav', 'COFs', 'FreeSolv', 'Polarizability', 'HOPV15', 'Matbench-Gap']
SCHED = {'Branin-Fav': (27, 283), 'Branin-Unfav': (7, 76), 'Park-Fav': (27, 280), 'Park-Unfav': (7, 76), 'COFs': (25, 245), 'FreeSolv': (27, 283), 'Polarizability': (10, 99), 'HOPV15': (17, 164), 'Matbench-Gap': (22, 209)}
from _common import GP_PAPER as _GPP, EXCLUDED_PAPER as _EXC, TL5_PROFILE as _TL5, RENAME_5TL, CODE_KEY_TL_5
GP_FAMILY = set(_GPP)
KEEP = set(_GPP) | set(_TL5)
RENAME = dict(RENAME_5TL)
(SALMON, BLUE) = ('#f2aa84', '#4e95d9')
ABC = list('abcdefghijk')
BLABEL = {'Matbench-Gap': 'Matbench-gap'}
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
(TITLE, LABEL, TICK) = (15, 13, 12)
TFLOP = 1000000000000.0
d = pd.read_csv(PROF)
d = d[d.ok == 1]
d = d[~d['model'].isin(_EXC)]
_missing = sorted(KEEP - set(d.model.unique()))
assert not _missing, f'FLOP profile is missing {_missing}'
d = d[d.model.isin(KEEP)]
med = d.groupby(['benchmark', 'model', 'fraction']).agg(N=('n_train', 'median'), fit=('fit_flops', 'median'), pred=('predict_flops', 'median')).reset_index()
print('Empirical FLOP scaling exponent (fit_FLOPs ~ N^k); theory: exact GP O(N^3), DNN O(N):')
for m in list(_TL5) + list(_GPP):
    g = med[med.model == m]
    (k, b) = np.polyfit(np.log(g.N), np.log(np.maximum(g.fit, 1)), 1)
    print(f'   {m:24s} k = {k:.2f}')

def recon(model, bench, col):
    g = med[(med.benchmark == bench) & (med.model == model)].sort_values('N')
    if len(g) < 2:
        return np.nan
    (a, b) = SCHED[bench]
    grid = np.arange(a, b + 1)
    return float(np.interp(grid, g.N.values, g[col].values).sum())
models = sorted(med.model.unique())
rows = [{'benchmark': b, 'model': m, 'flops': recon(m, b, 'fit') + recon(m, b, 'pred')} for b in BENCHES for m in models]
allt = pd.DataFrame(rows)
allt['disp'] = allt.model.replace(RENAME)
allt.to_csv(OUT / 'computing_flops_values_blr.csv', index=False)
print('\nTotal BO-loop FLOPs (TFLOPs), Sparse vs cheapest TL surrogate:')
for b in BENCHES:
    g = allt[allt.benchmark == b]
    sp = g[g.model == 'Sparse MFGP'].flops.values[0]
    ch = g[~g.model.isin(GP_FAMILY)].flops.min()
    print(f"   {b:14s} cheapest {ch / TFLOP:5.2f}  MFGP {g[g.model == 'MFGP'].flops.values[0] / TFLOP:5.2f}  Sparse {sp / TFLOP:5.2f} TFLOP ({sp / ch:4.0f}x cheapest)")
(fig, axes) = plt.subplots(2, 5, figsize=(15, 7.0))
for (i, b) in enumerate(BENCHES):
    ax = axes[i // 5, i % 5]
    g = allt[allt.benchmark == b].sort_values('flops')
    y = np.arange(len(g))
    ax.barh(y, g.flops / TFLOP, color=[SALMON if m in GP_FAMILY else BLUE for m in g.model], alpha=0.9, height=0.72, edgecolor='none')
    for (yi, (_, r)) in zip(y, g.iterrows()):
        ax.text(r.flops / TFLOP, yi, f' {r.flops / TFLOP:.2f}', va='center', ha='left', fontsize=10, color='#333')
    ax.set_yticks(y)
    ax.set_yticklabels([code(m) for m in g.disp], fontsize=TICK)
    ax.set_title(f'{ABC[i]}  {BLABEL.get(b, b)}', fontsize=TITLE, loc='left')
    ax.set_xlabel('Total compute (TFLOPs)', fontsize=LABEL)
    ax.set_xlim(0, g.flops.max() / TFLOP * 1.45)
    ax.tick_params(labelsize=TICK)
    ax.grid(axis='x', alpha=0.3, lw=0.5)
rr = []
for b in BENCHES:
    g = allt[allt.benchmark == b][['model', 'flops']].copy()
    g['rank'] = g['flops'].rank()
    rr.append(g[['model', 'rank']])
avg = pd.concat(rr).groupby('model')['rank'].mean().reset_index().sort_values('rank')
avg['disp'] = avg.model.replace(RENAME)
ax = axes[1, 4]
y = np.arange(len(avg))
ax.barh(y, avg['rank'], color=[SALMON if m in GP_FAMILY else BLUE for m in avg.model], alpha=0.9, height=0.72, edgecolor='none')
for (yi, r) in zip(y, avg['rank']):
    ax.text(r, yi, f' {r:.1f}', va='center', ha='left', fontsize=10, color='#333')
ax.set_yticks(y)
ax.set_yticklabels([code(m) for m in avg.disp], fontsize=TICK)
ax.set_title(f'{ABC[len(BENCHES)]}  Average compute rank', fontsize=TITLE, loc='left')
unmapped = sorted(set(allt.disp) - set(__import__('_common').CODE_NAMES))
assert not unmapped, f'no abbreviation for {unmapped}'
ax.set_xlabel('Average compute rank', fontsize=LABEL)
ax.set_xlim(0, avg['rank'].max() * 1.45)
ax.tick_params(labelsize=TICK)
ax.grid(axis='x', alpha=0.3, lw=0.5)
fig.legend(handles=[Patch(fc=SALMON, label='Gaussian-process family'), Patch(fc=BLUE, label='Transfer-learning surrogates')], loc='lower center', ncol=2, fontsize=13, frameon=True, fancybox=False, edgecolor='gray', handlelength=2.4, labelspacing=1.0, bbox_to_anchor=(0.5, 0.04))
fig.text(0.5, 0.024, CODE_KEY_TL_5, ha='center', va='bottom', fontsize=10.5, color='#333333')
fig.text(0.5, 0.006, 'GP: GP-base = autoregressive multi-fidelity GP (MFGP) · GP-SV = sparse variational MFGP · GP-DKL = deep-kernel GP', ha='center', va='bottom', fontsize=10.5, color='#333333')
plt.tight_layout(w_pad=1.4, h_pad=1.4, rect=(0, 0.085, 1, 1))
for stem in (OUT / 'computing_time_blr',):
    fig.savefig(f'{stem}.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(f'{stem}.png', dpi=200, bbox_inches='tight', facecolor='white')
    print('wrote', f'{stem}.pdf + .png')
plt.close()
