"""Construct the Branin and Park candidate grids."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from synthetic_functions import branin_hf, branin_lf, park_hf, park_lf


def prepare(outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, dim, size, hf, lf, alphas in [
        ('branin', 2, 50, branin_hf, branin_lf, (0.8, 0.1)),
        ('park', 4, 4, park_hf, park_lf, (0.6, 0.0)),
    ]:
        grids = np.meshgrid(*[np.linspace(0, 1, size)] * dim,
                            indexing='xy' if dim == 2 else 'ij')
        x = np.column_stack([g.ravel() for g in grids])
        for suffix, alpha in zip(['fav', 'unfav'], alphas):
            frame = pd.DataFrame(x, columns=[f'f{i}' for i in range(dim)])
            frame['HF'] = hf(x).ravel()
            frame['LF'] = lf(x, alpha).ravel()
            path = outdir / f'synthetic_{name}_{suffix}.csv'
            frame.to_csv(path, index=False)
            print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=Path, default=ROOT / 'data')
    prepare(parser.parse_args().outdir)
