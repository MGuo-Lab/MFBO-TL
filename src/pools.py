"""Candidate pools and initial-design budgets."""
from pathlib import Path
PAPER = {'COFs': {'csv': 'cofs.csv', 'cost_ratio': 0.065, 'use_smiles': False, 'negate': True, 'budget': 30, 'init_budget': 30}, 'FreeSolv': {'csv': 'freesolv.csv', 'cost_ratio': 0.1, 'use_smiles': True, 'negate': False, 'budget': 30, 'init_budget': 50}, 'Polarizability': {'csv': 'polarizability.csv', 'cost_ratio': 0.167, 'use_smiles': True, 'negate': True, 'budget': 30, 'init_budget': 30}, 'HOPV15': {'csv': 'hopv15.csv', 'cost_ratio': 0.1, 'use_smiles': False, 'negate': True, 'budget': 30, 'init_budget': 30}, 'Matbench-Gap': {'csv': 'matbench_gap.csv', 'cost_ratio': 0.05, 'use_smiles': False, 'negate': False, 'budget': 30, 'init_budget': 20}, 'Branin-Fav': {'csv': 'synthetic_branin_fav.csv', 'cost_ratio': 0.1, 'use_smiles': False, 'negate': False, 'budget': 50, 'init_budget': 50}, 'Branin-Unfav': {'csv': 'synthetic_branin_unfav.csv', 'cost_ratio': 0.5, 'use_smiles': False, 'negate': False, 'budget': 50, 'init_budget': 50}, 'Park-Fav': {'csv': 'synthetic_park_fav.csv', 'cost_ratio': 0.1, 'use_smiles': False, 'negate': False, 'budget': 50, 'init_budget': 50}, 'Park-Unfav': {'csv': 'synthetic_park_unfav.csv', 'cost_ratio': 0.5, 'use_smiles': False, 'negate': False, 'budget': 50, 'init_budget': 50}, 'ExptGap-PBE': {'csv': 'expt_gap_pbe.csv', 'cost_ratio': 0.05, 'use_smiles': False, 'negate': False, 'budget': 30, 'init_budget': 20}, 'Elastic-CHGNet': {'csv': 'elastic_chgnet_shear.csv', 'cost_ratio': 0.05, 'use_smiles': False, 'negate': False, 'budget': 30, 'init_budget': 20}, 'Elastic-SevenNet': {'csv': 'elastic_sevennet_shear.csv', 'cost_ratio': 0.05, 'use_smiles': False, 'negate': False, 'budget': 30, 'init_budget': 20}, 'Elastic-MatterSim': {'csv': 'elastic_mattersim_shear.csv', 'cost_ratio': 0.05, 'use_smiles': False, 'negate': False, 'budget': 30, 'init_budget': 20}}

def config(name):
    if name in PAPER:
        return dict(PAPER[name])
    if name.startswith('cell_') and name[5:].isdigit() and 0 <= int(name[5:]) < 126:
        return dict(csv=f'grid_cells/{name}.csv', cost_ratio=0.1, use_smiles=False, negate=False, budget=50, init_budget=10)
    raise KeyError(name)

def make_bench(name, data_dir):
    from benchmark import ChemistryBenchmark
    cfg = config(name)
    bench = ChemistryBenchmark(name, str(Path(data_dir) / cfg['csv']), cfg['cost_ratio'], cfg['use_smiles'], True, cfg['negate'])
    return bench, cfg
