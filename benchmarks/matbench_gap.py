"""Matched experimental and PBE band-gap candidate pool."""
import sys
import numpy as np
import pandas as pd
from benchmarks import _common
NAME = 'matbench_gap'
RHO = 0.05
NEGATE = False
TARGET_GAP = 1.4
OBJECTIVE = f'minimize |E_g - {TARGET_GAP}| eV (encoded as HF/LF distance); negate=False'
_MATCHED_CSV = _common.CACHE_DIR / 'matbench_matched.csv'
_INSTALL_HINT = 'matminer + pymatgen are required for the matbench_gap benchmark.\nInstall them into the active environment, e.g.:\n    pip install matminer pymatgen\n(pymatgen >= 2024 needs Python >= 3.10; on Python 3.9 pip resolves to an older compatible pymatgen/matminer automatically.)'

def _gap_col(df, kind):
    for c in df.columns:
        if c.lower().replace('_', ' ') == f'gap {kind}':
            return c
    cands = [c for c in df.columns if 'gap' in c.lower()]
    if len(cands) == 1:
        return cands[0]
    raise KeyError(f"Could not find the '{kind}' gap column in {list(df.columns)}")

def _load_and_match():
    if _MATCHED_CSV.exists():
        return pd.read_csv(_MATCHED_CSV)
    try:
        from matminer.datasets import load_dataset
        from pymatgen.core import Composition
    except ImportError as e:
        raise RuntimeError(_INSTALL_HINT) from e
    try:
        df_expt = load_dataset('matbench_expt_gap')
        df_mp = load_dataset('matbench_mp_gap')
    except Exception as e:
        raise RuntimeError(f"Failed to download/load the matbench datasets via matminer ({e!r}).\nIf the compute node has no internet, run this loader once on a host that does (it caches into matminer's dataset dir), then rerun.\nDatasets: matbench_expt_gap (Zhuo 2018), matbench_mp_gap (Materials Project PBE).") from e
    expt_gap = _gap_col(df_expt, 'expt')
    pbe_gap = _gap_col(df_mp, 'pbe')
    print(f'[{NAME}] reducing {len(df_expt)} expt compositions ...', flush=True)
    df_expt = df_expt.copy()
    df_expt['formula'] = df_expt['composition'].apply(lambda s: Composition(s).reduced_formula)
    print(f'[{NAME}] reducing {len(df_mp)} MP structures to compositions (this is the slow step) ...', flush=True)
    df_mp = df_mp.copy()
    df_mp['formula'] = df_mp['structure'].apply(lambda s: s.composition.reduced_formula)
    mp_by_formula = df_mp.groupby('formula')[pbe_gap].mean()
    expt_by_formula = df_expt.groupby('formula')[expt_gap].mean()
    matched = pd.concat([expt_by_formula.rename('gap_expt'), mp_by_formula.rename('gap_pbe')], axis=1, join='inner')
    matched = matched.reset_index().dropna(subset=['gap_expt', 'gap_pbe'])
    _common.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    matched.to_csv(_MATCHED_CSV, index=False)
    print(f'[{NAME}] matched pool: {len(matched)} compositions (expt {len(expt_by_formula)} INNER-JOIN mp {len(mp_by_formula)})', flush=True)
    return matched

def prepare(force=False):
    if not force:
        cached = _common.load_cache(NAME)
        if cached is not None:
            _common.write_benchmark_csv(NAME, cached['X_pca'], cached['y_hf'], cached['y_lf'])
            return cached
    matched = _load_and_match()
    y_hf = np.abs(matched['gap_expt'].to_numpy(np.float64) - TARGET_GAP)
    y_lf = np.abs(matched['gap_pbe'].to_numpy(np.float64) - TARGET_GAP)
    try:
        from matminer.featurizers.composition import ElementProperty
        from pymatgen.core import Composition
    except ImportError as e:
        raise RuntimeError(_INSTALL_HINT) from e
    print(f'[{NAME}] Magpie-featurizing {len(matched)} compositions ...', flush=True)
    ep = ElementProperty.from_preset('magpie')
    ep.set_n_jobs(1)
    comps = [Composition(f) for f in matched['formula']]
    X_magpie = np.array(ep.featurize_many(comps, ignore_errors=True, pbar=False), dtype=np.float64)
    X_pca = _common.matrix_to_pca(X_magpie)
    _common.save_cache(NAME, X_pca, y_hf, y_lf, meta={'rho': RHO, 'negate': NEGATE, 'objective': OBJECTIVE, 'target_gap': TARGET_GAP, 'formula': matched['formula'].tolist()})
    _common.write_benchmark_csv(NAME, X_pca, y_hf, y_lf)
    _common.print_diagnostics(NAME, X_pca, y_hf, y_lf, RHO, OBJECTIVE)
    return {'X_pca': X_pca, 'y_hf': y_hf, 'y_lf': y_lf}
if __name__ == '__main__':
    pool = prepare(force='--force' in sys.argv)
    _common.print_diagnostics(NAME, pool['X_pca'], pool['y_hf'], pool['y_lf'], RHO, OBJECTIVE)
