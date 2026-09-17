"""Build the three shear-modulus pools in GPa."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from benchmarks import _common


def build(model):
    source = _common.DATA_DIR / 'sources' / f'{model}_elastic.csv'
    frame = pd.read_csv(source).dropna(subset=['mlip_G', 'dft_G'])
    lf = frame.mlip_G.to_numpy(float)
    hf = frame.dft_G.to_numpy(float)
    keep = np.isfinite(lf) & np.isfinite(hf) & (lf > 0) & (hf > 0)
    frame = frame[keep].reset_index(drop=True)
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition
    featurizer = ElementProperty.from_preset('magpie')
    featurizer.set_n_jobs(1)
    features = featurizer.featurize_many(
        [Composition(f) for f in frame.formula], ignore_errors=True, pbar=False)
    x = _common.matrix_to_pca(np.asarray(features, dtype=np.float64))
    name = f'elastic_{model}_shear'
    return _common.write_benchmark_csv(name, x, -hf[keep], -lf[keep])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', choices=['chgnet', 'sevennet', 'mattersim'],
                        default=['chgnet', 'sevennet', 'mattersim'])
    for model in parser.parse_args().models:
        print(build(model))
