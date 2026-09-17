"""Descriptor preprocessing and candidate-pool output."""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / 'data'
CACHE_DIR = DATA_DIR / 'cache'
PCA_DIM = 10

def smiles_to_rdkit_pca(smiles_list, pca_dim=PCA_DIM):
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.ML.Descriptors import MoleculeDescriptors
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            features.append(calc.CalcDescriptors(mol))
        else:
            features.append([0.0] * len(descriptor_names))
    features = np.array(features, dtype=np.float64)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features_scaled = StandardScaler().fit_transform(features)
    return PCA(n_components=pca_dim).fit_transform(features_scaled)

def matrix_to_pca(X, pca_dim=PCA_DIM):
    X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = StandardScaler().fit_transform(X)
    n_comp = min(pca_dim, X_scaled.shape[1])
    return PCA(n_components=n_comp).fit_transform(X_scaled)

def cache_path(name):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f'{name}_featurized.pkl'

def load_cache(name):
    p = cache_path(name)
    if p.exists():
        with open(p, 'rb') as f:
            return pickle.load(f)
    return None

def save_cache(name, X_pca, y_hf, y_lf, meta=None):
    p = cache_path(name)
    with open(p, 'wb') as f:
        pickle.dump({'X_pca': np.asarray(X_pca, dtype=np.float64), 'y_hf': np.asarray(y_hf, dtype=np.float64), 'y_lf': np.asarray(y_lf, dtype=np.float64), 'meta': meta or {}}, f)
    return p

def write_benchmark_csv(name, X_pca, y_hf, y_lf):
    X_pca = np.asarray(X_pca, dtype=np.float64)
    cols = {f'f{i}': X_pca[:, i] for i in range(X_pca.shape[1])}
    cols['HF'] = np.asarray(y_hf, dtype=np.float64)
    cols['LF'] = np.asarray(y_lf, dtype=np.float64)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f'{name}.csv'
    pd.DataFrame(cols).to_csv(out, index=False)
    return out

def print_diagnostics(name, X_pca, y_hf, y_lf, rho, objective):
    X_pca = np.asarray(X_pca)
    y_hf = np.asarray(y_hf, dtype=np.float64)
    y_lf = np.asarray(y_lf, dtype=np.float64)
    r = float(np.corrcoef(y_lf, y_hf)[0, 1])
    print(f'[{name}] pool size           : {len(X_pca)}')
    print(f'[{name}] feature dim (LF=HF) : {X_pca.shape[1]}')
    print(f'[{name}] rho (cost ratio)    : {rho}')
    print(f'[{name}] Pearson r(LF, HF)   : {r:.4f}  (R^2 = {r ** 2:.4f})')
    print(f'[{name}] objective           : {objective}')
    print(f'[{name}] HF range            : [{y_hf.min():.4f}, {y_hf.max():.4f}]')
    print(f'[{name}] LF range            : [{y_lf.min():.4f}, {y_lf.max():.4f}]')
    return r
