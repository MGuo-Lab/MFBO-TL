"""Shared helpers for the 2026-07-13 panel-letter rework (npj Computational
Materials style: lowercase bold letters, no parentheses, outside the axes at
the title line height). Data / colours / layout of every figure are unchanged;
only the panel letters (and, where instructed, the suptitle / legend wording)
differ from the current manuscript PDFs.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent     # <repo>/figures
REPO = HERE.parent                         # repo root
RESULTS = REPO / 'results'
NEWFIGS = HERE / 'out'                     # regenerated figures land here;
NEWFIGS.mkdir(exist_ok=True)               # compare against paper/paper_figures/

RENAME_MAP = {'DNGO-Joint': 'Stop-Gradient Joint Training',
              'DNGO-Gradient': 'End-to-End Joint Training',
              'Two-Stage Joint': 'Pretrain-then-Joint Training'}

# Compact display names for AXIS TICK LABELS only (2026-08-18 print-size
# re-export): the full paper names stay in captions/legends/text, but a
# 27-character name repeated on ten y-axes cannot reach a 5.5 pt print size
# inside a five-column grid. Data keys are never shortened.
SHORT_NAMES = {'Knowledge Distillation': 'Knowledge Distill.',
               'Domain Adaptation (MMD)': 'Domain Adapt. (MMD)',
               'Soft Parameter Sharing': 'Soft Param. Sharing',
               'Pretrain-then-Joint Training': 'Pretrain-then-Joint',
               'Stop-Gradient Joint Training': 'Stop-Gradient Joint',
               'End-to-End Joint Training': 'End-to-End Joint',
               'DKL Multi-Fidelity': 'Deep-Kernel GP',
               'MFGP': 'Baseline MFGP'}


def short(name):
    return SHORT_NAMES.get(name, name)


# Every figure is drawn wider than its final print width, and by a DIFFERENT
# factor (1.3x to 5.2x), so a "title + 1pt" letter prints at 3.3-10.3pt
# depending on the figure. To make all panel letters the SAME physical size
# on the printed page, each script computes its letter size as
#   letter_pt = TARGET_LETTER_PRINT_PT * (tight-cropped PDF width / print width)
# using the measured PDF width of the previous build.
TARGET_LETTER_PRINT_PT = 8.0     # Nature Portfolio panel-letter size in print


def letter_pt(pdf_width_mm, print_width_mm):
    """Design-space font size that prints at TARGET_LETTER_PRINT_PT once the
    figure is scaled from its tight-cropped PDF width to its print width."""
    return TARGET_LETTER_PRINT_PT * pdf_width_mm / print_width_mm


def add_panel_letter(ax, letter, x=-0.08, y=1.02, size=16):
    """Nature Portfolio panel letter: lowercase bold sans-serif, no
    parentheses, top-left outside the axes at title-line height."""
    ax.text(x, y, letter, transform=ax.transAxes, fontweight='bold',
            fontsize=size, fontfamily='sans-serif', ha='left', va='bottom')


def save_dual(fig, stem, png_dpi=300):
    fig.savefig(f'{stem}.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(f'{stem}.png', dpi=png_dpi, bbox_inches='tight', facecolor='white')
    print(f'  wrote {stem}.pdf + .png')
