"""HOPV15 experimental PCE and Scharber proxy."""
import sys
import urllib.request
import tarfile
import io
import numpy as np
import pandas as pd
from benchmarks import _common
NAME = 'hopv15'
RHO = 0.1
NEGATE = True
OBJECTIVE = 'maximize HF (experimental PCE); negate=True'
HOPV_URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/hopv.tar.gz'
_RAW_CSV = _common.CACHE_DIR / 'hopv_raw.csv'
ACCEPTOR_LUMO = 4.3
VOC_EMPIRICAL_LOSS = 0.3
EQE = 0.65
FILL_FACTOR = 0.65
P_IN = 100.0
_EG_GRID = np.array([0.5, 0.75, 1.0, 1.1, 1.2, 1.3, 1.34, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0])
_JMAX_GRID = np.array([65.0, 58.0, 49.6, 44.0, 41.0, 36.5, 35.0, 32.0, 28.0, 24.5, 21.0, 18.0, 15.5, 13.0, 9.5, 6.8, 4.6, 3.0, 1.8])

def _scharber_pce(homo, e_gap):
    v_oc = np.maximum(0.0, -homo - ACCEPTOR_LUMO - VOC_EMPIRICAL_LOSS)
    j_max = np.interp(e_gap, _EG_GRID, _JMAX_GRID)
    j_sc = EQE * j_max
    return v_oc * j_sc * FILL_FACTOR / P_IN * 100.0

def _download_raw():
    if _RAW_CSV.exists():
        return pd.read_csv(_RAW_CSV)
    _common.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(HOPV_URL, timeout=120) as resp:
            blob = resp.read()
        with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as tar:
            member = next((m for m in tar.getmembers() if m.name.endswith('hopv.csv')))
            df = pd.read_csv(tar.extractfile(member))
    except Exception as e:
        raise RuntimeError(f'Failed to download the HOPV dataset from the DeepChem mirror ({HOPV_URL}): {e!r}\nDownload it manually and place a CSV with columns [smiles, HOMO, LUMO, optical_gap, PCE] at:\n    {_RAW_CSV}\nCanonical source: Harvard Dataverse handle 1/29408375 (https://dash.harvard.edu/handle/1/29408375), file HOPV_15_revised_2.data; Lopez et al., Sci. Data 3, 160086 (2016).') from e
    df.to_csv(_RAW_CSV, index=False)
    return df

def prepare(force=False):
    if not force:
        cached = _common.load_cache(NAME)
        if cached is not None:
            _common.write_benchmark_csv(NAME, cached['X_pca'], cached['y_hf'], cached['y_lf'])
            return cached
    df = _download_raw()
    gap = df['optical_gap'].copy()
    if 'electrochemical_gap' in df.columns:
        gap = gap.fillna(df['electrochemical_gap'])
    gap = gap.fillna(df['LUMO'] - df['HOMO'])
    work = pd.DataFrame({'smiles': df['smiles'].astype(str), 'homo': pd.to_numeric(df['HOMO'], errors='coerce'), 'gap': pd.to_numeric(gap, errors='coerce'), 'pce': pd.to_numeric(df['PCE'], errors='coerce')})
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    valid_smiles = work['smiles'].apply(lambda s: Chem.MolFromSmiles(s) is not None)
    keep = valid_smiles & work['homo'].notna() & work['gap'].notna() & work['pce'].notna()
    n_dropped = int((~keep).sum())
    work = work[keep].reset_index(drop=True)
    print(f'[{NAME}] kept {len(work)} / {len(df)} molecules ({n_dropped} dropped: invalid SMILES or missing HOMO/gap/PCE)')
    y_hf = work['pce'].to_numpy(dtype=np.float64)
    y_lf = _scharber_pce(work['homo'].to_numpy(np.float64), work['gap'].to_numpy(np.float64))
    X_pca = _common.smiles_to_rdkit_pca(work['smiles'].tolist())
    _common.save_cache(NAME, X_pca, y_hf, y_lf, meta={'rho': RHO, 'negate': NEGATE, 'objective': OBJECTIVE, 'smiles': work['smiles'].tolist()})
    _common.write_benchmark_csv(NAME, X_pca, y_hf, y_lf)
    _common.print_diagnostics(NAME, X_pca, y_hf, y_lf, RHO, OBJECTIVE)
    return {'X_pca': X_pca, 'y_hf': y_hf, 'y_lf': y_lf}
if __name__ == '__main__':
    pool = prepare(force='--force' in sys.argv)
    _common.print_diagnostics(NAME, pool['X_pca'], pool['y_hf'], pool['y_lf'], RHO, OBJECTIVE)
