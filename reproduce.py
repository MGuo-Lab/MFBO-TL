"""Generate the paper's numerical figures and supplementary tables."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    'make_attainment', 'make_trajectories', 'make_grid', 'make_screening',
    'make_scaling', 'make_acquisitions', 'make_calibration', 'make_compute', 'make_tables',
]

if __name__ == '__main__':
    for name in SCRIPTS:
        subprocess.run([sys.executable, str(ROOT / 'figures' / f'{name}.py')],
                       cwd=ROOT, check=True)
    print('Figures and table values: figures/out/')
