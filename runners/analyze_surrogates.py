"""Cross-surrogate analysis of the acquisition portfolio: does the DNGO-Gradient
finding (TS beats greedy on some chemistry, EI is the worst UQ, COFs catastrophic
for ALL UQ) GENERALISE across the 11 DNN / transfer-learning surrogates?

Sources (current env, base_seed=42, 20 seeds):
  greedy, pi, ucb, mes, ts  <- results/acq_portfolio/<acq>/  (fresh, all 11 models)
  ei (BLR-std EI)           <- results/acq_portfolio/ei/ if complete, else
                               results/main_corrected (verified identical to fresh
                               ei on the 7 original benchmarks; <=0.09 on HOPV15/
                               Matbench). Auto-upgrades to fresh ei when that job lands.

Outputs (results/acq_portfolio/):
  surrogates_means.csv      per (benchmark, model, acq) mean +/- SE final regret
  surrogates_winrate.csv    per (acq, benchmark) fraction of 11 surrogates beating greedy
  surrogates_winrate.png    win-rate heatmap (UQ acq x benchmark)
  surrogates_aggbars.png    9-panel bars, mean-over-surrogates per acq
Login-node-safe (pandas/numpy/matplotlib only).
"""
from __future__ import annotations
import glob, os
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# NOTE (release repo): the per-acquisition raw run dirs (greedy/ei/pi/ucb/mes/ts)
# under results/acq_portfolio are NOT shipped here — only the two derived CSVs
# (surrogates_means.csv, surrogates_winrate.csv) that the paper figure reads.
# Re-running this analysis requires the raw runs from the archived MFBO-TL-Paper repo.
AP = ROOT / 'results' / 'acq_portfolio'
ACQS = ['greedy', 'ei', 'pi', 'ucb', 'mes', 'ts']
UQ = ['ei', 'pi', 'ucb', 'mes', 'ts']
LABEL = {'greedy': 'Greedy', 'ei': 'EI', 'pi': 'PI', 'ucb': 'GP-UCB', 'mes': 'MES', 'ts': 'Thompson'}
BENCHES = ['Branin-Fav', 'Branin-Unfav', 'Park-Fav', 'Park-Unfav', 'COFs',
           'FreeSolv', 'Polarizability', 'HOPV15', 'Matbench-Gap']
HIGHCORR_CHEM = ['COFs', 'FreeSolv', 'Polarizability']


def load(d):
    d = str(d)
    if os.path.isfile(f'{d}/results_summary.csv'):
        df = pd.read_csv(f'{d}/results_summary.csv')
    else:
        fs = glob.glob(f'{d}/summary_*.csv')
        df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
    return df[df['final_regret'].notna()] if not df.empty else df


def acq_frame(acq):
    """Seed-level frame for one acquisition across all DNN/TL surrogates."""
    if acq == 'ei':
        fresh = load(AP / 'ei')
        nmod = fresh['model'].nunique() if not fresh.empty else 0
        if nmod >= 11:
            src, df = 'acq_portfolio/ei (fresh, matched)', fresh
        else:
            # main_corrected covers all 9 benches incl HOPV15/Matbench for 12 models
            df = load(ROOT / 'results' / 'main_corrected')
            src = 'main_corrected (BLR-std EI; fresh ei still running)'
    else:
        df = load(AP / acq); src = f'acq_portfolio/{acq}'
    df = df[df['model'] != 'MFGP'].copy()
    return df, src


def main():
    frames, srcs = {}, {}
    for a in ACQS:
        frames[a], srcs[a] = acq_frame(a)
    models = sorted(set(frames['greedy']['model'].unique()))
    print("surrogates (n=%d): %s" % (len(models), models))
    for a in ACQS:
        print(f"  {a:7s} source: {srcs[a]}  ({frames[a]['model'].nunique()} models, {len(frames[a])} rows)")

    # per (bench, model, acq) mean
    def cell_mean(a, b, m):
        d = frames[a]
        v = d[(d.benchmark == b) & (d.model == m)]['final_regret']
        return float(v.mean()) if len(v) else np.nan

    rows = []
    for b in BENCHES:
        for m in models:
            for a in ACQS:
                rows.append(dict(benchmark=b, model=m, acq=a, mean=cell_mean(a, b, m)))
    means = pd.DataFrame(rows)
    means.to_csv(AP / 'surrogates_means.csv', index=False)

    # ---- win-rate matrix: for each UQ acq x benchmark, fraction of surrogates with acq < greedy ----
    def winrate(a, b):
        g = {m: cell_mean('greedy', b, m) for m in models}
        x = {m: cell_mean(a, b, m) for m in models}
        pairs = [(g[m], x[m]) for m in models if np.isfinite(g[m]) and np.isfinite(x[m])]
        if not pairs:
            return np.nan, 0
        wins = sum(1 for gm, xm in pairs if xm < gm - 1e-9)
        return wins / len(pairs), len(pairs)

    wr = pd.DataFrame(index=UQ, columns=BENCHES, dtype=float)
    for a in UQ:
        for b in BENCHES:
            wr.loc[a, b] = winrate(a, b)[0]
    wr.to_csv(AP / 'surrogates_winrate.csv')

    print("\n=========== WIN-RATE: fraction of %d surrogates where UQ acq BEATS greedy ===========" % len(models))
    print("(0.00 = greedy wins on every surrogate; 1.00 = UQ wins on every surrogate)\n")
    hdr = "  acq    " + "".join(f"{b[:9]:>10s}" for b in BENCHES)
    print(hdr)
    for a in UQ:
        print(f"  {LABEL[a]:7s}" + "".join(f"{wr.loc[a,b]:>10.2f}" for b in BENCHES))

    # ---- COFs: does greedy beat ALL UQ on every surrogate? ----
    print("\n=========== COFs (flagship): greedy vs best-UQ per surrogate ===========")
    cofs_greedy_wins = 0; cofs_n = 0
    for m in models:
        g = cell_mean('greedy', 'COFs', m)
        uq = {a: cell_mean(a, 'COFs', m) for a in UQ}
        uq = {a: v for a, v in uq.items() if np.isfinite(v)}
        if not np.isfinite(g) or not uq:
            continue
        cofs_n += 1
        best_uq = min(uq, key=uq.get)
        win = g < uq[best_uq] - 1e-9
        cofs_greedy_wins += int(win)
        print(f"  {m:24s} greedy={g:6.2f}  best UQ={LABEL[best_uq]}={uq[best_uq]:6.2f}  -> {'greedy' if win else best_uq+' beats greedy'}")
    print(f"  => greedy beats ALL 5 UQ on {cofs_greedy_wins}/{cofs_n} surrogates")

    # ---- TS vs greedy on the 3 high-correlation chemistry benchmarks ----
    print("\n=========== Thompson vs greedy on high-correlation chemistry (per surrogate) ===========")
    for b in HIGHCORR_CHEM:
        wins = 0; n = 0; deltas = []
        for m in models:
            g = cell_mean('greedy', b, m); t = cell_mean('ts', b, m)
            if np.isfinite(g) and np.isfinite(t):
                n += 1; wins += int(t < g - 1e-9); deltas.append(t - g)
        md = np.median(deltas) if deltas else np.nan
        print(f"  {b:15s} TS beats greedy on {wins}/{n} surrogates (median TS-greedy = {md:+.3f})")

    # ---- EI the worst UQ? average rank of each UQ acq across (surrogate x benchmark) ----
    print("\n=========== Is EI the worst UQ? mean rank among the 5 UQ acqs (1=best,5=worst) ===========")
    ranks = {a: [] for a in UQ}
    for b in BENCHES:
        for m in models:
            vals = {a: cell_mean(a, b, m) for a in UQ}
            vals = {a: v for a, v in vals.items() if np.isfinite(v)}
            if len(vals) < len(UQ):
                continue
            order = sorted(vals, key=vals.get)            # ascending regret -> best first
            for r, a in enumerate(order, 1):
                ranks[a].append(r)
    for a in sorted(UQ, key=lambda a: np.mean(ranks[a]) if ranks[a] else 9):
        rs = ranks[a]
        print(f"  {LABEL[a]:8s} mean rank = {np.mean(rs):.2f}  (best on {sum(1 for r in rs if r==1)}, worst on {sum(1 for r in rs if r==5)} of {len(rs)} cells)")

    # ---- figures ----
    _heatmap(wr, models)
    _aggbars(means, models)
    print("\nfigures -> surrogates_winrate.png, surrogates_aggbars.png ; tables -> surrogates_{means,winrate}.csv")


def _heatmap(wr, models):
    fig, ax = plt.subplots(figsize=(11, 3.4))
    data = wr.values.astype(float)
    im = ax.imshow(data, cmap='RdBu', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(wr.columns))); ax.set_xticklabels(wr.columns, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(wr.index))); ax.set_yticklabels([LABEL[a] for a in wr.index], fontsize=10)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha='center', va='center',
                        color='white' if (v < 0.25 or v > 0.75) else 'black', fontsize=8)
    ax.set_title(f'Win-rate vs greedy across {len(models)} TL surrogates '
                 '(red=UQ rarely beats greedy, blue=UQ usually beats greedy)', fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label='fraction of surrogates UQ < greedy')
    plt.tight_layout(); fig.savefig(AP / 'surrogates_winrate.png', dpi=150, bbox_inches='tight'); plt.close(fig)


def _aggbars(means, models):
    import math
    COLOR = {'greedy': '#2c6fbb', 'ei': '#9bb8d4', 'pi': '#7bb07b', 'ucb': '#e0a458', 'mes': '#b07bb0', 'ts': '#c46c4e'}
    ncols = 5; nrows = math.ceil(len(BENCHES) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows)); axes = np.atleast_2d(axes)
    for idx, b in enumerate(BENCHES):
        ax = axes[idx // ncols, idx % ncols]
        mu, se = [], []
        for a in ACQS:
            vals = means[(means.benchmark == b) & (means.acq == a)]['mean'].dropna().values
            mu.append(np.mean(vals) if len(vals) else np.nan)
            se.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
        y = np.arange(len(ACQS))[::-1]
        ax.barh(y, mu, xerr=se, color=[COLOR[a] for a in ACQS], alpha=0.92, edgecolor='white', height=0.7, capsize=3)
        ax.set_yticks(y); ax.set_yticklabels([LABEL[a] for a in ACQS], fontsize=8)
        ax.set_title(f'({chr(97+idx)}) {b}', fontsize=11)
        ax.set_xlabel('mean-over-surrogates regret', fontsize=8); ax.grid(axis='x', alpha=0.3, lw=0.5)
        xm = max([m for m in mu if np.isfinite(m)] + [1e-6])
        for yi, m in zip(y, mu):
            if np.isfinite(m): ax.text(m + xm*0.02, yi, f'{m:.2f}', va='center', fontsize=7)
        ax.set_xlim(0, xm*1.25)
    for j in range(len(BENCHES), nrows*ncols):
        axes[j//ncols, j%ncols].axis('off')
    fig.suptitle(f'Acquisition portfolio averaged over {len(models)} TL surrogates (lower=better)', fontsize=13, y=1.0)
    plt.tight_layout(rect=(0,0,1,0.99)); fig.savefig(AP / 'surrogates_aggbars.png', dpi=150, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    main()
