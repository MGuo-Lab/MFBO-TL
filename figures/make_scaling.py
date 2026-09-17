'Figure 4c: surrogate-fit FLOP scaling.'
import sys
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import os
from _common import NEWFIGS, RESULTS, letter_pt, save_dual
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
PROF = RESULTS / 'flop_profile_blr' / 'flop_profile.csv'
from _common import GP_PAPER as _GPP, EXCLUDED_PAPER as _EXC
GP_FAMILY = set(_GPP)
(SALMON, BLUE) = ('#f2aa84', '#4e95d9')
(TITLE, LABEL, TICK, LEG) = (15, 13, 11, 11)
FAM = {'GP': (SALMON, 'Gaussian-process family'), 'TL': (BLUE, 'Transfer-learning surrogates')}
IDEAL = {'GP': (3, 'ideal $\\mathcal{O}(N^3)$'), 'TL': (1, 'ideal $\\mathcal{O}(N)$')}
THEORY = {'GP': '', 'TL': '=\\mathcal{O}(N)'}

def family_slopes(df, feats):
    ly = np.log10(df.fit.values)
    dummies = pd.get_dummies(df.model).values.astype(float)
    A = np.column_stack([dummies] + list(feats))
    (coef, *_) = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return (coef[dummies.shape[1]:], r2)
from _common import TL5_PROFILE as _TL5
KEEP = set(_GPP) | set(_TL5)
d = pd.read_csv(PROF)
d = d[~d['model'].isin(_EXC)]
_missing = sorted(KEEP - set(d.model.unique()))
assert not _missing, f'FLOP profile is missing {_missing}'
d = d[d.model.isin(KEEP)]
d = d[d.ok == 1].copy()
d['family'] = np.where(d.model.isin(GP_FAMILY), 'GP', 'TL')
med = d.groupby(['benchmark', 'model', 'family', 'fraction']).agg(N=('n_train', 'median'), fit=('fit_flops', 'median')).reset_index()
med = med[med.fit > 0].copy()
(fig, ax) = plt.subplots(1, 1, figsize=(9.2, 5.6))
laws = {}
R2 = {}
(nlo, nhi) = (med.N.min(), med.N.max())
xline = np.array([nlo, nhi])
for (fam, (col, lab)) in FAM.items():
    g = med[med.family == fam]
    ax.scatter(g.N, g.fit, s=18, c=col, alpha=0.28, edgecolors='none', zorder=2)
    ((kN,), r2N) = family_slopes(g, [np.log10(g.N.values)])
    xc = 10 ** np.mean(np.log10(g.N.values))
    yc = 10 ** np.mean(np.log10(g.fit.values))
    laws[fam] = (kN, xc, yc)
    extra = f'  (${THEORY[fam]}$)' if THEORY[fam] else ''
    ax.plot(xline, yc * (xline / xc) ** kN, color=col, lw=2.8, zorder=4, label=f'{lab}:  $k_N={kN:.2f}$' + extra)
    (p, txt) = IDEAL[fam]
    xref = np.array([xc, nhi])
    ax.plot(xref, yc * (xref / xc) ** p, color='0.45', lw=1.2, ls=(0, (4, 3)), zorder=5)
    ax.text(nhi, yc * (nhi / xc) ** p, '  ' + txt, color='0.4', fontsize=10, va='center')
    print(f'{fam}: k_N={kN:.2f} (R2={r2N:.2f})')
    R2[fam] = r2N
((kG, xcG, ycG), (kT, xcT, ycT)) = (laws['GP'], laws['TL'])
logNx = (np.log10(ycT) - kT * np.log10(xcT) - (np.log10(ycG) - kG * np.log10(xcG))) / (kG - kT)
Nx = 10 ** logNx
yx = ycG * (Nx / xcG) ** kG
print(f'break-even N = {Nx:.1f}')
trans = ax.get_xaxis_transform()
ax.axvline(Nx, color='0.4', ls=(0, (5, 4)), lw=1.4, zorder=3)
ax.scatter([Nx], [yx], s=72, c='white', edgecolors='0.15', linewidths=1.6, zorder=6)
ax.text(Nx, 0.135, f'break-even  $N\\approx{Nx:.0f}$', transform=trans, ha='center', va='bottom', fontsize=10.5, color='0.15', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.6', alpha=0.95))
ax.text(Nx / 1.14, 0.045, 'GP cheaper', transform=trans, ha='right', va='bottom', fontsize=9.5, color=SALMON)
ax.text(Nx * 1.14, 0.045, 'TL cheaper', transform=trans, ha='left', va='bottom', fontsize=9.5, color=BLUE)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Training-set size  $N$  (high- + low-fidelity points)', fontsize=LABEL)
ax.set_ylabel('Surrogate-fit compute  (FLOPs)', fontsize=LABEL)
PANEL_LETTER = os.environ.get('PANEL_LETTER', 'c')
if PANEL_LETTER:
    ax.text(-0.075, 1.01, PANEL_LETTER, transform=ax.transAxes, fontsize=letter_pt(230.3, 148), fontfamily='sans-serif', ha='left', va='bottom')
ax.legend(fontsize=LEG, loc='upper left', frameon=True, edgecolor='0.8')
ax.grid(True, which='both', alpha=0.22, lw=0.5)
ax.tick_params(labelsize=TICK)
plt.tight_layout()
save_dual(fig, NEWFIGS / 'compute_scaling_law_blr')
import json as _json
_json.dump(dict(kG=float(kG), kT=float(kT), break_even=float(Nx), r2={f: float(v) for (f, v) in R2.items()}), open(NEWFIGS / 'scaling_law_blr.json', 'w'))
plt.close(fig)
