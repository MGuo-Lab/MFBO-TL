'Figure 3: final-regret advantage in the controlled grid.'
import glob, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
GRID = Path(os.environ.get('GRID_DIR', REPO / 'results' / 'grid'))
OUT = Path(os.environ.get('FIG_OUT_DIR', REPO / 'figures' / 'out'))
OUT.mkdir(parents=True, exist_ok=True)
TL5 = os.environ.get('TL5', 'DNGOJoint,TwoStageJoint,DNGOGradient,SoftParameterSharing,DomainAdaptationMMD').split(',')
FAMILIES = {'TL': TL5, 'GP-base': ['MFGP'], 'GP variants': ['DKL', 'SparseMFGP']}
COMPARISONS = [('TL', 'GP-base'), ('TL', 'GP variants'), ('GP variants', 'GP-base')]
METRIC = 'final_regret'
R2B = np.round(np.arange(0.1, 0.91, 0.1), 2)
T10 = np.round(np.arange(0.0, 1.01, 0.1), 2)
plt.rcParams.update({'savefig.dpi': 200, 'pdf.fonttype': 42, 'ps.fonttype': 42})
(C_AGR, C_R2) = ('#2a9d8f', '#4b5563')
CMAP = LinearSegmentedColormap.from_list('gp_tl', ['#b5532e', '#f7f7f5', '#0b3d7a'])
man = pd.read_csv(GRID / 'grid_manifest.csv').set_index('cell_id')
rows = []
for f in sorted(glob.glob(str(GRID / 'cells' / 'summary_cell_*.csv'))):
    d = pd.read_csv(f)
    cid = d.benchmark.iloc[0]
    vals = {m: dict(zip(g.seed, g[METRIC])) for (m, g) in d.groupby('model')}
    best = {}
    for (fam, mem) in FAMILIES.items():
        have = [m for m in mem if m in vals]
        means = {m: np.mean(list(vals[m].values())) for m in have}
        b = min(means, key=means.get)
        best[fam] = means[b]
    rec = dict(cell=cid, r2=man.loc[cid, 'r2'], top10=man.loc[cid, 'top10'])
    for (f1, f2) in COMPARISONS:
        rec[f'{f1}|{f2}'] = best[f2] - best[f1]
    rows.append(rec)
G = pd.DataFrame(rows)
G['ri'] = np.clip(np.round((G.r2 - 0.1) / 0.1).astype(int), 0, 8)
G['ti'] = np.round(G.top10 * 10).astype(int)
print('cells', len(G))
mats = []
for (f1, f2) in COMPARISONS:
    k = f'{f1}|{f2}'
    M = np.full((11, 9), np.nan)
    for ((ti, ri), g) in G.groupby(['ti', 'ri']):
        M[ti, ri] = g[k].mean()
    mats.append(M)
vmax = float(np.nanmax([np.nanmax(np.abs(M)) for M in mats]))

def prof(k, col, vals):
    out = []
    for (i, v) in enumerate(vals):
        g = G[G[col] == i][k]
        out.append((v, g.mean(), g.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0.0, len(g)))
    return pd.DataFrame(out, columns=['x', 'm', 'se', 'n'])
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6.5, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2, 'ytick.major.size': 2, 'hatch.linewidth': 0.5})
(TICK, LAB, TITLE, LETTER, LEG, NOTE) = (6.5, 7, 7.5, 8.5, 7.2, 6.5)
NODATA = dict(facecolor='#e4e4e1', edgecolor='#8c8c8c', hatch='////', linewidth=0)
(fig, axes) = plt.subplots(2, 3, figsize=(6.9, 4.1), gridspec_kw={'height_ratios': [2.6, 1.15], 'hspace': 0.62, 'wspace': 0.42})
letters = os.environ.get('FIG1LN_LETTERS', 'a,b,c').split(',')
for (c, ((f1, f2), M)) in enumerate(zip(COMPARISONS, mats)):
    ax = axes[0, c]
    k = f'{f1}|{f2}'
    im = ax.imshow(M, origin='lower', cmap=CMAP, vmin=-vmax, vmax=vmax, aspect='auto', extent=(0.05, 0.95, -0.05, 1.05), interpolation='nearest')
    for (ti, ri) in zip(*np.where(np.isnan(M))):
        ax.add_patch(Rectangle((0.05 + 0.1 * ri, -0.05 + 0.1 * ti), 0.1, 0.1, **NODATA))
    ax.set_xticks(R2B)
    ax.set_xticklabels([f'{v:.1f}' if i % 2 == 0 else '' for (i, v) in enumerate(R2B)], fontsize=TICK)
    ax.set_yticks(T10)
    ax.set_yticklabels([f'{v:.1f}' for v in T10] if c == 0 else [], fontsize=TICK)
    ax.tick_params(pad=1.5)
    ax.set_xlabel('global LF–HF $R^2$', fontsize=LAB, labelpad=2)
    if c == 0:
        ax.set_ylabel('top-10 optimum agreement', fontsize=LAB, labelpad=2)
    ax.set_title(f'{f1} vs {f2}', fontsize=TITLE, pad=13)
    ax.text(-0.26 if c == 0 else -0.3, 1.12, letters[c], transform=ax.transAxes, fontsize=LETTER, va='bottom')
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cb.ax.tick_params(labelsize=TICK, length=2, width=0.5, pad=1.5)
    cb.outline.set_linewidth(0.5)
    if c == 2:
        cb.set_label('final-regret advantage', fontsize=LAB, labelpad=3)
    pos = int((G[k] > 1e-12).sum())
    tie = int((G[k].abs() <= 1e-12).sum())
    ax.text(0.5, 1.015, f'{f1} better in {pos}/{len(G)} · ties {tie}', transform=ax.transAxes, ha='center', va='bottom', fontsize=NOTE, color='#444')
    ax2 = axes[1, c]
    for (col, vals, colr, lab) in [('ti', T10, C_AGR, 'vs top-10 agreement (mean over $R^2$)'), ('ri', R2B, C_R2, 'vs global $R^2$ (mean over agreement)')]:
        P = prof(k, col, vals)
        ax2.fill_between(P.x, P.m - P.se, P.m + P.se, color=colr, alpha=0.18, lw=0)
        ax2.plot(P.x, P.m, '-o', color=colr, lw=1.0, ms=2.2, label=lab)
    ax2.axhline(0, color='#999', lw=0.6)
    ax2.set_xlim(-0.03, 1.03)
    ax2.set_xlabel('axis value (agreement or $R^2$)', fontsize=LAB, labelpad=2)
    if c == 0:
        ax2.set_ylabel('advantage\n(mean ± s.e.)', fontsize=LAB, labelpad=2)
    ax2.tick_params(labelsize=TICK, pad=1.5)
    for s in ('top', 'right'):
        ax2.spines[s].set_visible(False)
    rho_t = G[k].corr(G.top10, method='spearman')
    rho_r = G[k].corr(G.r2, method='spearman')
    ax2.text(0.99, 0.95, f'ρ(agreement) {rho_t:+.2f}\nρ($R^2$) {rho_r:+.2f}', transform=ax2.transAxes, ha='right', va='top', fontsize=NOTE, color='#444')
(hl, ll) = axes[1, 0].get_legend_handles_labels()
fig.legend(hl, ll, fontsize=LEG, frameon=False, loc='center', bbox_to_anchor=(0.5, axes[1, 0].get_position().y1 + 0.047), ncol=2, handlelength=2.0, columnspacing=2.0, borderaxespad=0)
cbpos = cb.ax.get_position()
fig.legend(handles=[Patch(label='no data', **NODATA)], loc='upper left', bbox_to_anchor=(cbpos.x0 - 0.004, cbpos.y0 - 0.045), fontsize=LEG, frameon=False, handlelength=1.3, handleheight=1.0, handletextpad=0.5, borderaxespad=0)
for ext in ('pdf', 'png'):
    fig.savefig(OUT / f"{os.environ.get('FIG_STEM', 'fig3_grid_final_regret')}.{ext}", bbox_inches='tight')
STEM = os.environ.get('FIG_STEM', 'fig3_grid_final_regret')
print('wrote', OUT / f'{STEM}.pdf', OUT / f'{STEM}.png', 'vmax', round(vmax, 4))
for (f1, f2) in COMPARISONS:
    k = f'{f1}|{f2}'
    print(k, 'pos', int((G[k] > 1e-12).sum()), 'tie', int((G[k].abs() <= 1e-12).sum()), 'neg', int((G[k] < -1e-12).sum()), 'rho_agr %.2f rho_r2 %.2f' % (G[k].corr(G.top10, method='spearman'), G[k].corr(G.r2, method='spearman')), 'mean %.4f' % G[k].mean())
G.to_csv(OUT / 'fig1ln_final_cells.csv', index=False)
