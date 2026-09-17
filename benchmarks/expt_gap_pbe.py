"""Experimental and PBE gaps matched by reduced formula."""
import sys
import numpy as np
import pandas as pd
from benchmarks import _common
NAME = 'expt_gap_pbe'
RHO = 0.05
NEGATE = False
TARGET_GAP = 1.4
OBJECTIVE = f'minimize |E_g - {TARGET_GAP}| eV (Hautier expt HF, MP-PBE LF); negate=False'
_MATCHED_CSV = _common.CACHE_DIR / 'expt_gap_pbe_matched.csv'
_INSTALL_HINT = "matminer + pymatgen are required, and the 'expt_gap' and 'mp_nostruct_20181018' datasets must be cached (run matminer.datasets.load_dataset on a host with internet once)."

def _reduced(s):
    from pymatgen.core import Composition
    try:
        return Composition(str(s)).reduced_formula
    except Exception:
        return None

def _load_and_match():
    if _MATCHED_CSV.exists():
        return pd.read_csv(_MATCHED_CSV)
    try:
        from matminer.datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(_INSTALL_HINT) from e
    df_e = load_dataset('expt_gap')
    df_mp = load_dataset('mp_nostruct_20181018')
    print(f'[{NAME}] reducing {len(df_e)} expt + {len(df_mp)} MP formulae ...', flush=True)
    df_e = df_e.copy()
    df_e['rf'] = df_e['formula'].map(_reduced)
    df_mp = df_mp.copy()
    df_mp['rf'] = df_mp['formula'].map(_reduced)
    eg = df_e.dropna(subset=['rf']).groupby('rf')['gap expt'].mean()
    pg = df_mp.dropna(subset=['rf']).groupby('rf')['gap pbe'].mean()
    matched = pd.concat([eg.rename('gap_expt'), pg.rename('gap_pbe')], axis=1, join='inner').reset_index().dropna()
    matched = matched.rename(columns={'index': 'formula', 'rf': 'formula'})
    _common.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    matched.to_csv(_MATCHED_CSV, index=False)
    print(f'[{NAME}] matched pool: {len(matched)} compositions (expt {len(eg)} INNER-JOIN mp {len(pg)})', flush=True)
    return matched

def prepare(force=False):
    if not force:
        cached = _common.load_cache(NAME)
        if cached is not None:
            _common.write_benchmark_csv(NAME, cached['X_pca'], cached['y_hf'], cached['y_lf'])
            return cached
    matched = _load_and_match()
    fcol = 'formula' if 'formula' in matched.columns else matched.columns[0]
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
    comps = [Composition(f) for f in matched[fcol]]
    X_magpie = np.array(ep.featurize_many(comps, ignore_errors=True, pbar=False), dtype=np.float64)
    X_pca = _common.matrix_to_pca(X_magpie)
    _common.save_cache(NAME, X_pca, y_hf, y_lf, meta={'rho': RHO, 'negate': NEGATE, 'objective': OBJECTIVE, 'target_gap': TARGET_GAP, 'formula': matched[fcol].tolist()})
    _common.write_benchmark_csv(NAME, X_pca, y_hf, y_lf)
    _common.print_diagnostics(NAME, X_pca, y_hf, y_lf, RHO, OBJECTIVE)
    return {'X_pca': X_pca, 'y_hf': y_hf, 'y_lf': y_lf}
if __name__ == '__main__':
    pool = prepare(force='--force' in sys.argv)
    _common.print_diagnostics(NAME, pool['X_pca'], pool['y_hf'], pool['y_lf'], RHO, OBJECTIVE)
