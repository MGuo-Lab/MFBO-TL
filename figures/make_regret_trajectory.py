"""regret_trajectory.pdf -- NEW panel letters a-i (reading order) + legend
code names replaced by the paper's model names. Colours, line styles
(GP family solid / TL dashed), symlog axis, top-k overlap boxes: unchanged.

Adapted from experiments/fix6_promote/plot_fix15.py (the script that produced
the manuscript's regret_trajectory.pdf, == figs/fix15_trajectories.pdf).

Code-name -> paper-name mapping VERIFIED in code (not assumed):
  * experiments/fix6_promote/run_fix6.py:32,63-69 -- the runner imports
    src/benchmark.py ('DNGOGradient': B.DNGOGradient, 'DNGOJoint': B.DNGOJoint,
    'TwoStageJoint': B.TwoStageJoint) and
    experiments/extra_baselines/baselines.py ('DKL': X.DKLMultiFidelity,
    'SparseMFGP': X.SparseMFGP).
  * src/benchmark.py class semantics:
      - DNGOJoint (l.609): joint loss with y_lf_pred computed under
        torch.no_grad() (l.625-627) -> the HF loss cannot backpropagate into
        the LF network = STOP-GRADIENT Joint Training.
      - DNGOGradient (l.675): identical joint loss WITHOUT no_grad
        (l.699-701) -> gradients flow through both networks = END-TO-END
        Joint Training.
      - TwoStageJoint (l.535): stage 1 pretrains the LF net alone, stage 2
        joint-fine-tunes both = PRETRAIN-THEN-JOINT Training.
  * src/plotting/plot_regret_v7.py:13-17 rename_map agrees.

Login-node safe (cached CSVs + matplotlib only).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import NEWFIGS, REPO, add_panel_letter, letter_pt, save_dual, short

CELLS = REPO / 'results' / 'traj_cells'
DATA = REPO / 'data'

BENCHES = ['Branin-Fav', 'Branin-Unfav', 'Park-Fav', 'Park-Unfav', 'COFs',
           'FreeSolv', 'Polarizability', 'HOPV15', 'Matbench-Gap']
DATA_CFG = {
    'Branin-Fav': ('synthetic_branin_fav', False),
    'Branin-Unfav': ('synthetic_branin_unfav', False),
    'Park-Fav': ('synthetic_park_fav', False),
    'Park-Unfav': ('synthetic_park_unfav', False),
    'COFs': ('cofs', True),
    'FreeSolv': ('freesolv', False),
    'Polarizability': ('polarizability', True),
    'HOPV15': ('hopv15', True),
    'Matbench-Gap': ('matbench_gap', False),
}

GP = ['MFGP', 'NARGP', 'DKL', 'SparseMFGP']
TL = ['Sequential', 'Curriculum', 'DNGOGradient', 'Progressive', 'TwoStageJoint',
      'DNGOJoint', 'KnowledgeDistillation', 'DomainAdaptationMMD',
      'SoftParameterSharing', 'PseudoLabeling', 'Adapter']
MODELS = GP + TL

# verified mapping (see module docstring); spellings exactly match the
# final_regret.pdf tick labels
PAPER_NAME = {
    'MFGP': 'MFGP',
    'NARGP': 'NARGP',
    'DKL': 'DKL Multi-Fidelity',
    'SparseMFGP': 'Sparse MFGP',
    'Sequential': 'Sequential',
    'Curriculum': 'Curriculum',
    'DNGOGradient': 'End-to-End Joint Training',
    'Progressive': 'Progressive',
    'TwoStageJoint': 'Pretrain-then-Joint Training',
    'DNGOJoint': 'Stop-Gradient Joint Training',
    'KnowledgeDistillation': 'Knowledge Distillation',
    'DomainAdaptationMMD': 'Domain Adaptation (MMD)',
    'SoftParameterSharing': 'Soft Parameter Sharing',
    'PseudoLabeling': 'Pseudo-Labelling',
    'Adapter': 'Adapter',
}

# colours / styles: identical to plot_fix15.py
GP_COLORS = {'MFGP': '#08306b', 'NARGP': '#2171b5', 'DKL': '#6baed6',
             'SparseMFGP': '#00897b'}
TL_CMAP = plt.cm.tab20(np.linspace(0, 1, 20))
_warm = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 6]
TL_COLORS = {m: TL_CMAP[_warm[i]] for i, m in enumerate(TL)}
COLORS = {**GP_COLORS, **TL_COLORS}
LS = {m: ('-' if m in GP else '--') for m in MODELS}
LW = {m: (2.4 if m in GP else 1.3) for m in MODELS}
LETTERS = list('abcdefghi')
# uniform 8 pt print size: tight PDF width ~356 mm, print width 174 mm
# (2026-08-18 print-size re-export: 16x12 in -> 14x10.5 in, ~2.0x print width,
# fonts raised so body text prints at >= 5 pt)
LETTER_PT = letter_pt(356.0, 174)


def topk_overlap(bn, ks=(10, 30)):
    stem, negate = DATA_CFG[bn]
    df = pd.read_csv(DATA / f'{stem}.csv')
    y_hf = df['HF'].values.astype(float)
    y_lf = df['LF'].values.astype(float)
    if negate:
        y_hf, y_lf = -y_hf, -y_lf
    out = []
    for k in ks:
        kk = min(k, len(y_hf))
        out.append(len(set(np.argsort(y_hf)[:kk]) & set(np.argsort(y_lf)[:kk])) / kk)
    return out


def main():
    fig, axes = plt.subplots(3, 3, figsize=(14, 10.5))
    for i, (ax, bn) in enumerate(zip(axes.flat, BENCHES)):
        for ml in MODELS:
            f = CELLS / f'traj_{bn}_{ml}.csv'
            if not f.exists():
                continue
            tr = pd.read_csv(f)
            if tr.empty:
                continue
            grid = np.unique(tr['budget'].values)
            curves = []
            for _, g in tr.groupby('seed'):
                g = g.sort_values('budget')
                curves.append(np.interp(grid, g['budget'], g['regret']))
            mu = np.vstack(curves).mean(0)
            ax.plot(grid, mu, color=COLORS[ml], ls=LS[ml], lw=LW[ml],
                    label=PAPER_NAME[ml], alpha=0.9)
        ax.set_title(bn, fontsize=13, fontweight='bold', loc='left')
        add_panel_letter(ax, LETTERS[i], x=-0.10, y=1.01, size=LETTER_PT)
        ax.set_xlabel('budget (HF-equivalent cost)', fontsize=12)
        ax.set_ylabel('simple regret', fontsize=12)
        ax.set_yscale('symlog', linthresh=1e-3)
        ax.tick_params(labelsize=11)
        ax.grid(alpha=0.3)
        o10, o30 = topk_overlap(bn)
        ax.text(0.97, 0.97,
                f'LF/HF top-k overlap\ntop-10: {o10:.2f}   top-30: {o30:.2f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.6',
                          alpha=0.85))
    handles = [plt.Line2D([0], [0], color=COLORS[ml], ls=LS[ml], lw=LW[ml])
               for ml in MODELS]
    labels = [short(PAPER_NAME[ml]) for ml in MODELS]
    fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False,
               fontsize=10.5, title='GP family (solid) | TL family (dashed)',
               title_fontsize=10.5)
    fig.tight_layout(rect=[0, 0.08, 1, 0.99])
    save_dual(fig, NEWFIGS / 'regret_trajectory', png_dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
