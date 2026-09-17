'Paths and display names shared by the figure scripts.'
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / 'results'
NEWFIGS = HERE / 'out'
NEWFIGS.mkdir(parents=True, exist_ok=True)
GP_PAPER = ['MFGP', 'DKL Multi-Fidelity', 'Sparse MFGP']
GP_CODE = ['MFGP', 'DKL', 'SparseMFGP']
EXCLUDED_PAPER = []
TL5_PROFILE = ['DNGO-Joint', 'Two-Stage Joint', 'DNGO-Gradient', 'Soft Parameter Sharing', 'Domain Adaptation (MMD)']
RENAME_5TL = {'DNGO-Joint': 'Frozen-representation transfer', 'Two-Stage Joint': 'Pretrain-then-Joint', 'DNGO-Gradient': 'End-to-End Joint'}
CODE_NAMES = {'MFGP': 'GP-base', 'Sparse MFGP': 'GP-SV', 'DKL Multi-Fidelity': 'GP-DKL', 'Frozen-representation transfer': 'TL-base', 'Pretrain-then-Joint': 'TL-PtJ', 'End-to-End Joint': 'TL-E2E', 'Soft Parameter Sharing': 'TL-SPS', 'Domain Adaptation (MMD)': 'TL-MMD'}
CODE_KEY_TL_5 = 'TL-base · TL-PtJ · TL-E2E · TL-SPS · TL-MMD'
CODE_KEY_GP = 'GP-base · GP-DKL · GP-SV'

def code(name):
    return CODE_NAMES.get(name, name)

def short(name):
    return code(name)

def letter_pt(pdf_width_mm, print_width_mm):
    return 8.0 * pdf_width_mm / print_width_mm

def save_dual(fig, stem, png_dpi=300):
    fig.savefig(f'{stem}.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(f'{stem}.png', dpi=png_dpi, bbox_inches='tight', facecolor='white')
