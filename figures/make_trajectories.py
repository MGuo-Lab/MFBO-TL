'Figure 2: mean regret trajectories and standard errors.'
import json, os, sys, numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
OUT = Path(__file__).resolve().parent / 'out'
OUT.mkdir(exist_ok=True)
D = json.load(open(Path(__file__).resolve().parents[1] / 'results/main/trajectories.json'))
POOLS = ['Branin-Fav', 'Branin-Unfav', 'Park-Fav', 'Park-Unfav', 'COFs', 'FreeSolv', 'Polarizability', 'HOPV15', 'Matbench-Gap', 'ExptGap-PBE', 'Elastic-CHGNet', 'Elastic-SevenNet', 'Elastic-MatterSim']
XMAX = {'Branin-Fav': 50, 'Branin-Unfav': 50, 'Park-Fav': 50, 'Park-Unfav': 50, 'COFs': 30, 'FreeSolv': 30, 'Polarizability': 30, 'HOPV15': 30, 'Matbench-Gap': 30, 'ExptGap-PBE': 30, 'Elastic-CHGNet': 30, 'Elastic-SevenNet': 30, 'Elastic-MatterSim': 30}
BFIG1 = {'Branin-Fav': 50, 'Branin-Unfav': 50, 'Park-Fav': 50, 'Park-Unfav': 50, 'COFs': 30, 'FreeSolv': 30, 'Polarizability': 30, 'HOPV15': 30, 'Matbench-Gap': 30, 'ExptGap-PBE': 30, 'Elastic-CHGNet': 30, 'Elastic-SevenNet': 30, 'Elastic-MatterSim': 30}
LABEL = {'Matbench-Gap': 'Matbench-gap'}
MODELS = D['models']
GP = [m for m in MODELS if m in set(D['gp'])]
TL = [m for m in MODELS if m not in set(D['gp'])]
ABBR = D['abbr']
LARGE = POOLS[9:]
LINEAR = LARGE + ['HOPV15']
YLOG = {'Matbench-Gap': (0.03, 1.5)}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2, 'ytick.major.size': 2, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})
(TITLE, TICK, LAB, SUB, LEG, FOOT) = (7.5, 6.5, 7, 6.5, 7.5, 6.5)
STYLE = {'MFGP': ('#b5532e', '-', 1.35), 'Sparse MFGP': ('#f2aa84', (0, (1.2, 1.2)), 1.0), 'DKL': ('#e07b4f', (0, (4, 1.5)), 1.0), 'Frozen-representation transfer': ('#0b3d7a', '-', 1.35), 'Pretrain-then-Joint': ('#2467b3', (0, (4, 1.5)), 1.0), 'End-to-End Joint': ('#4e95d9', (0, (4, 1.5)), 1.0), 'Soft Parameter Sharing': ('#8fc0ec', (0, (1.2, 1.2)), 1.0), 'Domain Adaptation (MMD)': ('#8fc0ec', (0, (4, 1.5)), 1.0)}

def reg_at(pts, grid):
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    idx = np.searchsorted(xs, grid, side='right') - 1
    return np.where(idx >= 0, ys[np.clip(idx, 0, len(ys) - 1)], ys[0])
rows = []
(fig, axes) = plt.subplots(3, 5, figsize=(7.2, 5.6))
ax = axes.ravel()
letters = 'abcdefghijklm'
for (i, p) in enumerate(POOLS):
    a = ax[i]
    P = D['pools'][p]
    xmax = XMAX[p]
    grid = np.round(np.arange(0.0, xmax + 1e-09, 0.25), 3)
    for m in MODELS:
        ser = P['series'].get(m, {})
        curves = []
        for (s, r) in ser.items():
            c = reg_at(r['pts'], grid)
            c = np.where(grid > r['hz'] + 0.26, np.nan, c)
            curves.append(c)
        if not curves:
            continue
        A = np.array(curves)
        n_at = np.sum(~np.isnan(A), axis=0)
        with np.errstate(all='ignore'):
            mean = np.where(n_at > 0, np.nanmean(A, axis=0), np.nan)
            se = np.where(n_at > 1, np.nanstd(A, axis=0, ddof=1) / np.sqrt(np.maximum(n_at, 1)), np.nan)
        mean[n_at < 10] = np.nan
        se[n_at < 10] = np.nan
        (col, ls, lw) = STYLE[m]
        a.plot(grid, mean, color=col, ls=ls, lw=lw, label=ABBR[m], solid_capstyle='butt')
        a.fill_between(grid, np.maximum(mean - se, 0), mean + se, color=col, alpha=0.07, lw=0)
        rows.append(dict(pool=p, model=m, n_min=int(n_at[n_at > 0].min()) if (n_at > 0).any() else 0, n_max=int(n_at.max()), mean_end=float(mean[~np.isnan(mean)][-1]) if (~np.isnan(mean)).any() else None))
    if p in LINEAR:
        ys = [r['mean_end'] for r in rows if r['pool'] == p and r['mean_end']]
        ymin = min(ys)
        ymax = a.get_ylim()[1]
        a.set_ylim(ymin - 0.12 * (ymax - ymin), ymax)
    elif p in YLOG:
        a.set_yscale('log')
        a.set_ylim(*YLOG[p])
    elif p.startswith('Park'):
        a.set_yscale('symlog', linthresh=1e-09, linscale=0.35)
        a.set_ylim(bottom=0)
        a.set_yticks([0, 1e-08, 1e-06, 0.0001, 0.01, 1.0])
    else:
        a.set_yscale('symlog', linthresh=0.001, linscale=0.35)
        a.set_ylim(bottom=0)
    a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xlim(0, xmax)
    a.text(0.0, 1.15, f'{letters[i]}  {LABEL.get(p, p)}', transform=a.transAxes, ha='left', va='bottom', fontsize=TITLE)
    a.text(0.0, 1.02, f'B = {xmax}', transform=a.transAxes, ha='left', va='bottom', fontsize=SUB, color='#555555')
    a.grid(lw=0.4, alpha=0.3)
    a.set_axisbelow(True)
    a.tick_params(labelsize=TICK, pad=1.5)
    for sp in ('top', 'right'):
        a.spines[sp].set_visible(False)
    if i >= 10 or i == 8 or i == 9:
        a.set_xlabel('Budget', fontsize=LAB, labelpad=1.5)
    if i % 5 == 0:
        a.set_ylabel('Simple regret', fontsize=LAB, labelpad=1.5)
for k in (13, 14):
    ax[k].axis('off')
hg = [Line2D([0], [0], color=STYLE[m][0], ls=STYLE[m][1], lw=STYLE[m][2], label=ABBR[m]) for m in GP]
ht = [Line2D([0], [0], color=STYLE[m][0], ls=STYLE[m][1], lw=STYLE[m][2], label=ABBR[m]) for m in TL]
LEGKW = dict(fontsize=LEG, frameon=False, title_fontsize=LEG, labelspacing=0.4, borderaxespad=0, handlelength=2.2, handletextpad=0.6)
ax[13].legend(handles=hg, loc='upper left', title='GP family', **LEGKW)
ax[13].add_artist(ax[13].get_legend())
ax[13].legend(handles=ht, loc='upper left', bbox_to_anchor=(0.98, 1.0), title='TL surrogates', ncol=1, columnspacing=1.2, **LEGKW)
fig.subplots_adjust(left=0.065, right=0.982, top=0.935, bottom=0.15, wspace=0.55, hspace=0.55)
FOOTNOTE = ['Mean ± s.e. over seeds; budget in HF-equivalent cost. Solid lines: GP-base and TL-base; dashed or dotted: the other surrogates.', 'TL: TL-base = base transfer-learning surrogate · TL-PtJ = pretrain-then-joint · TL-E2E = end-to-end joint', 'TL-SPS = soft parameter sharing · TL-MMD = domain adaptation (MMD)', 'GP: GP-base = autoregressive multi-fidelity GP (MFGP) · GP-SV = sparse variational MFGP · GP-DKL = deep-kernel GP']
for (k, line) in enumerate(FOOTNOTE):
    fig.text(0.5, 0.069 - 0.0198 * k, line, ha='center', va='bottom', fontsize=FOOT, color='#333333')
for ext in ('pdf', 'png', 'svg'):
    fig.savefig(f'{OUT}/fig2_trajectories.{ext}', dpi=400 if ext == 'png' else None)
plt.close(fig)
pd.DataFrame(rows).to_csv(f'{OUT}/fig2_trajectories_values.csv', index=False)
print(pd.DataFrame(rows).groupby('pool')[['n_min', 'n_max']].agg(['min', 'max']).to_string(), file=sys.stderr)
print('DONE', file=sys.stderr)
