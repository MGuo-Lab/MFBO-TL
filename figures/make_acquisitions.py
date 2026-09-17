"""Supplementary Figures 1 and 2: matched acquisitions and the TL acquisition portfolio."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / 'figures/out'
OUTDIR.mkdir(exist_ok=True)
EXPL = json.loads((ROOT / 'results/main/trajectories.json').read_text())
TRAJ = pd.read_csv(ROOT / 'results/controls/acquisition_trajectories.csv')
B9 = list(EXPL['pools'])[:9]
BUD9 = {b: EXPL['pools'][b]['B'] for b in B9}
BUD9['Matbench-Gap'] = 20
TARGETS = ['5', '2', '1', '0.5']
LABEL = {'Matbench-Gap': 'Matbench-gap'}
ABBR = EXPL['abbr']
GP = EXPL['gp']
TLM = EXPL['tl']
MODELS = GP + TLM
RAW = {'DNGOJoint': 'Frozen-representation transfer', 'TwoStageJoint': 'Pretrain-then-Joint', 'DNGOGradient': 'End-to-End Joint', 'SoftParameterSharing': 'Soft Parameter Sharing', 'DomainAdaptationMMD': 'Domain Adaptation (MMD)', 'MFGP': 'MFGP', 'SparseMFGP': 'Sparse MFGP', 'DKL': 'DKL'}
(TL_COLOR, GP_COLOR) = ('#4e95d9', '#f2aa84')
(TL_DARK, GP_DARK) = ('#2f6fb0', '#c9714a')
DIV = LinearSegmentedColormap.from_list('gp_tl', [GP_DARK, '#f7f7f5', TL_DARK])
(TITLE, TICK, LAB) = (6.8, 6, 6.5)
S20 = set(range(42, 62))
TL_ROOTS = []
SUMMARY = {}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6.5, 'axes.linewidth': 0.5, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'hatch.linewidth': 0.4})

def load_traj(roots, arm):
    acq = arm.split('__hf_')[-1]
    if acq == 'argmin':
        acq = 'greedy'
    return TRAJ[(TRAJ.acquisition == acq) & (TRAJ.model != 'MFGP')].rename(columns={'pool': 'benchmark'})

def footnote(fig, y1=0.04, y2=0.025):
    fig.text(0.5, y1, 'LF query: indicated acquisition | HF query: greedy', ha='center', fontsize=6)

def attain_pts(pts, b, thrs):
    t0 = pts[0][0]
    if b <= t0 + 1e-09:
        return None
    acc = 0.0
    for thr in thrs:
        c = None
        for (x, y) in pts:
            if y <= thr + 1e-12:
                c = x
                break
        if c is not None and c <= b + 1e-09:
            acc += (b - max(c, t0)) / (b - t0)
    return acc / len(thrs)

def to_pts(bu, re_):
    pts = [[float(bu[0]), float(re_[0])]]
    for (x, y) in zip(bu[1:], re_[1:]):
        if y < pts[-1][1] - 1e-12:
            pts.append([float(x), float(y)])
    return pts

def thr_of(pool, keys=TARGETS):
    return [EXPL['pools'][pool]['thr'][k] for k in keys]

def per_seed_attain(traj, pool, model_raw, b, keys=TARGETS):
    d = traj[(traj.benchmark == pool) & (traj.model == model_raw)]
    out = {}
    thrs = thr_of(pool, keys)
    for (s, g) in d.groupby('seed'):
        g = g.sort_values('budget')
        hz = float(g.window.iloc[0])
        if hz + 0.26 < b:
            continue
        v = attain_pts(to_pts(g.budget.values, g.regret.values), b, thrs)
        if v is not None:
            out[int(s)] = v
    return out

def mean_se(vals):
    v = np.asarray(list(vals), float)
    if len(v) == 0:
        return (np.nan, np.nan, 0)
    return (float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0, int(len(v)))

def merged_rows(traj, pool, b, seeds=None, tag=''):
    rows = {}

    def get(raw):
        ps = per_seed_attain(traj, pool, raw, b)
        return {s: v for (s, v) in ps.items() if seeds is None or s in seeds}
    for (raw, disp) in RAW.items():
        ps = get(raw)
        if ps:
            rows[disp] = ps
    return rows

def save(fig, stem):
    fig.savefig(f'{OUTDIR}/{stem}.png', dpi=300, facecolor='white')
    fig.savefig(f'{OUTDIR}/{stem}.pdf', facecolor='white')
    plt.close(fig)
    print('wrote', stem, file=sys.stderr)

def style(a, xlabel=None, xmax=1.3):
    a.grid(axis='x', lw=0.4, alpha=0.35)
    a.set_axisbelow(True)
    for sp in ('top', 'right'):
        a.spines[sp].set_visible(False)
    a.tick_params(axis='y', length=0, pad=1.5)
    a.tick_params(axis='x', labelsize=TICK, pad=1.5)
    a.set_xlim(0, xmax)
    a.set_xticks([0, 0.5, 1.0])
    a.set_xticklabels(['0', '0.5', '1'], fontsize=TICK)
    if xlabel:
        a.set_xlabel(xlabel, fontsize=LAB, labelpad=1.5)

def compact(v):
    s = f'{v:+.2f}'
    return s.replace('+0.', '+.').replace('-0.', '−.').replace('+.00', '0').replace('−.00', '0')

def fig_S2():
    ACQS = ['greedy', 'ei', 'pi', 'ucb', 'mes', 'ts']
    UQ = ACQS[1:]
    ARM = {'greedy': 'blr_replace__hf_argmin', 'ei': 'blr_replace__hf_ei', 'pi': 'blr_replace__hf_pi', 'ucb': 'blr_replace__hf_ucb', 'mes': 'blr_replace__hf_mes', 'ts': 'blr_replace__hf_ts'}
    AL = {'greedy': 'Greedy', 'ei': 'EI', 'pi': 'PI', 'ucb': 'GP-UCB', 'mes': 'MES', 'ts': 'Thompson'}
    T = {a: load_traj(TL_ROOTS, ARM[a]) for a in ACQS}
    M = {a: {} for a in ACQS}
    PS = {a: {} for a in ACQS}
    for a in ACQS:
        for p in B9:
            rows = merged_rows(T[a], p, BUD9[p], seeds=S20, tag=f'S2-{a}')
            M[a][p] = {d: mean_se(v.values()) for (d, v) in rows.items()}
            PS[a][p] = rows
    recs = [dict(acq=a, pool=p, model=ABBR[d], mean=v[0], se=v[1], n=v[2]) for a in ACQS for p in B9 for (d, v) in M[a][p].items()]
    pd.DataFrame(recs).to_csv(f'{OUTDIR}/supp_acq_portfolio_values.csv', index=False)
    W = np.full((5, 9), np.nan)
    D = np.full((5, 9), np.nan)
    N = np.zeros((5, 9), int)
    for (i, a) in enumerate(UQ):
        for (j, p) in enumerate(B9):
            diffs = []
            for d in TLM:
                if d in PS[a][p] and d in PS['greedy'][p]:
                    common = sorted(set(PS[a][p][d]) & set(PS['greedy'][p][d]))
                    if len(common) >= 10:
                        diffs.append(np.mean([PS[a][p][d][s] - PS['greedy'][p][d][s] for s in common]))
            if diffs:
                W[i, j] = np.mean([x > 1e-12 for x in diffs])
                D[i, j] = np.mean(diffs)
                N[i, j] = len(diffs)
    pd.DataFrame(W, index=[AL[a] for a in UQ], columns=B9).to_csv(f'{OUTDIR}/supp_acq_portfolio_winshare.csv')
    pd.DataFrame(D, index=[AL[a] for a in UQ], columns=B9).to_csv(f'{OUTDIR}/supp_acq_portfolio_meandelta.csv')
    pd.DataFrame(N, index=[AL[a] for a in UQ], columns=B9).to_csv(f'{OUTDIR}/supp_acq_portfolio_n_models.csv')
    (T2, K2, L2, S2, V2, C2, F2, CB2) = (7.5, 6.5, 7, 6.5, 6.5, 6.2, 6.5, 7.5)
    fig = plt.figure(figsize=(7.2, 6.3))
    gs_top = fig.add_gridspec(1, 2, left=0.085, right=0.985, top=0.94, bottom=0.768, wspace=0.21)
    gs = fig.add_gridspec(2, 5, left=0.085, right=0.985, top=0.585, bottom=0.179, wspace=0.55, hspace=0.55)
    a1 = fig.add_subplot(gs_top[0, 0])
    a2 = fig.add_subplot(gs_top[0, 1])
    cols = [LABEL.get(p, p) for p in B9]
    SEQ = LinearSegmentedColormap.from_list('w_tl', ['#f7f7f5', TL_DARK])

    def hbar(a, im, label):
        p = a.get_position()
        cax = fig.add_axes([p.x0, p.y0 - 0.104, 0.42 * p.width, 0.011])
        c = fig.colorbar(im, cax=cax, orientation='horizontal')
        c.ax.tick_params(labelsize=K2, length=2, width=0.4, pad=1.5)
        c.outline.set_linewidth(0.4)
        cax.text(1.04, 0.5, label, transform=cax.transAxes, ha='left', va='center', fontsize=CB2)
        return c
    im = a1.imshow(W, cmap=SEQ, vmin=0, vmax=1, aspect='auto')
    a1.set_xticks(range(9))
    a1.set_xticklabels(cols, rotation=40, ha='right', fontsize=K2, rotation_mode='anchor')
    a1.set_yticks(range(5))
    a1.set_yticklabels([AL[a] for a in UQ], fontsize=K2)
    a1.tick_params(length=0, pad=1.5)
    for sp in a1.spines.values():
        sp.set_visible(False)
    for i in range(5):
        for j in range(9):
            if np.isfinite(W[i, j]):
                a1.text(j, i, f'{W[i, j]:.2f}', ha='center', va='center', fontsize=C2, color='white' if W[i, j] > 0.6 else '#222222')
    hbar(a1, im, 'share of TL surrogates')
    a1.text(0.0, 1.15, 'a  Share of TL surrogates attaining more than greedy', transform=a1.transAxes, ha='left', va='bottom', fontsize=T2)
    a1.text(0.0, 1.03, '5 surrogates · 20 seeds each', transform=a1.transAxes, ha='left', va='bottom', fontsize=S2, color='#555555')
    vmaxD = max(np.nanmax(np.abs(D)) if np.isfinite(D).any() else 1, 1e-06)
    im2 = a2.imshow(D, cmap=DIV, norm=TwoSlopeNorm(0, -vmaxD, vmaxD), aspect='auto')
    a2.set_xticks(range(9))
    a2.set_xticklabels(cols, rotation=40, ha='right', fontsize=K2, rotation_mode='anchor')
    a2.set_yticks(range(5))
    a2.set_yticklabels([AL[a] for a in UQ], fontsize=K2)
    a2.tick_params(length=0, pad=1.5)
    for sp in a2.spines.values():
        sp.set_visible(False)
    for i in range(5):
        for j in range(9):
            if np.isfinite(D[i, j]):
                a2.text(j, i, f'{D[i, j]:+.2f}', ha='center', va='center', fontsize=C2, color='white' if abs(D[i, j]) > 0.6 * vmaxD else '#222222')
    hbar(a2, im2, 'acquisition − greedy')
    a2.text(0.0, 1.15, 'b  Mean attainment difference vs greedy', transform=a2.transAxes, ha='left', va='bottom', fontsize=T2)
    a2.text(0.0, 1.03, 'mean over the 5 surrogates', transform=a2.transAxes, ha='left', va='bottom', fontsize=S2, color='#555555')
    letters = 'cdefghijk'
    SH = {'greedy': TL_DARK, 'ei': TL_COLOR, 'pi': '#7fb3e6', 'ucb': '#a9cbee', 'mes': '#c9def3', 'ts': '#e3eef9'}
    for (i, p) in enumerate(B9):
        a = fig.add_subplot(gs[i // 5, i % 5])
        (mu, se) = ([], [])
        for acq in ACQS:
            v = [M[acq][p][d][0] for d in TLM if d in M[acq][p] and M[acq][p][d][2] >= 10]
            mu.append(np.mean(v) if v else np.nan)
            se.append(np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
        y = np.arange(6)[::-1]
        a.barh(y, mu, xerr=se, color=[SH[acq] for acq in ACQS], height=0.72, edgecolor='none', error_kw=dict(lw=0.5, capsize=1.2, capthick=0.5, ecolor='#333333'))
        for (yi, m, s) in zip(y, mu, se):
            if np.isfinite(m):
                a.text(m + (s if np.isfinite(s) else 0) + 0.025, yi, f'{m:.2f}', va='center', ha='left', fontsize=V2, color='#222222')
        a.set_yticks(y)
        a.set_yticklabels([AL[acq] for acq in ACQS], fontsize=K2)
        a.grid(axis='x', lw=0.4, alpha=0.35)
        a.set_axisbelow(True)
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
        a.tick_params(axis='y', length=0, pad=1.5)
        a.tick_params(axis='x', labelsize=K2, pad=1.5)
        a.set_xlim(0, 1.3)
        a.set_xticks([0, 0.5, 1.0])
        a.set_xticklabels(['0', '0.5', '1'], fontsize=K2)
        if i >= 4:
            a.set_xlabel('Attainment', fontsize=L2, labelpad=1.5)
        a.text(0.0, 1.15, f'{letters[i]}  {LABEL.get(p, p)}', transform=a.transAxes, ha='left', va='bottom', fontsize=T2)
        a.text(0.0, 1.03, f'B = {BUD9[p]}', transform=a.transAxes, ha='left', va='bottom', fontsize=S2, color='#555555')
    NOTE2 = ["c–k: mean ± s.e. over the five TL surrogates of each surrogate's 20-seed mean attainment (seeds 42–61).", 'a, b: paired per surrogate (acquisition − greedy), then the share of positive differences (a) or the mean difference (b).', 'TL: TL-base = base transfer-learning surrogate · TL-PtJ = pretrain-then-joint · TL-E2E = end-to-end joint', 'TL-SPS = soft parameter sharing · TL-MMD = domain adaptation (MMD)', 'GP: GP-base = autoregressive multi-fidelity GP (MFGP) · GP-SV = sparse variational MFGP · GP-DKL = deep-kernel GP']
    for (k, line) in enumerate(NOTE2):
        fig.text(0.085, 0.108 - 0.0185 * k - (0.006 if k >= 2 else 0), line, ha='left', va='bottom', fontsize=F2, color='#333333')
    save(fig, 'supp_acq_portfolio')
    SUMMARY['S2'] = {'acqs': [AL[a] for a in ACQS], 'winshare': [[None if not np.isfinite(x) else round(float(x), 3) for x in r] for r in W], 'meandelta': [[None if not np.isfinite(x) else round(float(x), 4) for x in r] for r in D], 'n_models': N.tolist(), 'pool_means': {p: {AL[acq]: round(float(np.nanmean([M[acq][p][d][0] for d in TLM if d in M[acq][p]])), 4) if any((d in M[acq][p] for d in TLM)) else None for acq in ACQS} for p in B9}, 'coverage': {AL[acq]: {p: {ABBR[d]: M[acq][p][d][2] for d in TLM if d in M[acq][p]} for p in B9} for acq in ACQS}}

def matched_acquisitions():
    rows = []
    for pool in B9:
        for (model, label) in [('MFGP', 'GP-base'), ('DNGOJoint', 'TL-base')]:
            for acq in ['ei', 'greedy']:
                t = TRAJ[(TRAJ.model == model) & (TRAJ.acquisition == acq)].rename(columns={'pool': 'benchmark'})
                values = per_seed_attain(t, pool, model, BUD9[pool])
                (mean, se, n) = mean_se(values.values())
                if n != 20:
                    raise ValueError(f'{pool}/{label}/{acq}: expected 20 seeds, got {n}')
                rows.append(dict(pool=pool, condition=f"{label} {(acq.upper() if acq == 'ei' else acq)}", mean=mean, se=se, n=n, budget=BUD9[pool]))
    values = pd.DataFrame(rows)
    values.to_csv(OUTDIR / 'supp_acq_matrix_values.csv', index=False)
    conditions = ['GP-base EI', 'GP-base greedy', 'TL-base EI', 'TL-base greedy']
    (fig, axes) = plt.subplots(3, 3, figsize=(7.2, 5.4))
    fig.subplots_adjust(left=0.13, right=0.985, top=0.93, bottom=0.12, wspace=0.92, hspace=0.8)
    for (i, (pool, ax)) in enumerate(zip(B9, axes.flat)):
        data = values[values.pool == pool].set_index('condition')
        for (y, condition) in zip(np.arange(4)[::-1], conditions):
            row = data.loc[condition]
            color = GP_COLOR if condition.startswith('GP') else TL_COLOR
            hatch = '////' if condition.endswith('greedy') else None
            ax.barh(y, row['mean'], xerr=row.se, color=color, height=0.7, hatch=hatch, edgecolor='white' if hatch else 'none', linewidth=0.4, error_kw=dict(lw=0.5, capsize=1.5, ecolor='#333333'))
            ax.text(row['mean'] + row.se + 0.025, y, f"{row['mean']:.2f}", va='center', fontsize=5.7)
        ax.set_yticks(np.arange(4)[::-1], conditions, fontsize=6)
        ax.set_xlim(0, 1.22)
        ax.set_xticks([0, 0.5, 1], ['0', '0.5', '1'])
        ax.tick_params(axis='y', length=0, pad=3)
        ax.tick_params(axis='x', labelsize=6, length=2, width=0.5)
        ax.set_axisbelow(True)
        ax.grid(axis='x', alpha=0.3, lw=0.4)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
        ax.text(0, 1.2, f"{'abcdefghi'[i]}  {LABEL.get(pool, pool)}", transform=ax.transAxes, fontsize=7)
        ax.text(0, 1.04, f'B = {BUD9[pool]} | n = 20', transform=ax.transAxes, fontsize=5.4, color='#555555')
        if i >= 6:
            ax.set_xlabel('Attainment', fontsize=7)
    footnote(fig, 0.028)
    save(fig, 'supp_acq_matrix')
if __name__ == '__main__':
    matched_acquisitions()
    fig_S2()
