"""Shared helpers for the new benchmark loaders.

The featurization here is deliberately a faithful copy of the logic already in
``src/benchmark.py`` (``ChemistryBenchmark._smiles_to_rdkit_features`` and the
``StandardScaler`` it applies in ``__init__``). The loaders write the PCA(10)
output as plain ``f0..f9`` columns; ``ChemistryBenchmark(use_smiles=False)`` then
applies its own ``StandardScaler`` on top — so the end-to-end transform is

    StandardScaler( PCA_10( StandardScaler( raw_descriptors ) ) )

which is *bit-for-bit identical* to the FreeSolv / Polarizability SMILES path.
Composition / Magpie featurizers (perovskite, matbench) reuse the same
``StandardScaler -> PCA(10)`` reduction so every chemistry benchmark shares one
featurization contract.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# rdkit is only needed by the SMILES loaders; import lazily inside the helper so
# the composition/Magpie loaders don't pay the (slow, on cephfs) rdkit import.

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

PCA_DIM = 10  # every chemistry benchmark embeds its pool in 10 dims (matches FreeSolv/COFs)


# ---------------------------------------------------------------------------
# Featurization (mirrors src/benchmark.py exactly)
# ---------------------------------------------------------------------------

def smiles_to_rdkit_pca(smiles_list, pca_dim=PCA_DIM):
    """SMILES -> RDKit 2D descriptors (~210) -> StandardScaler -> PCA(pca_dim).

    Byte-for-byte the same recipe as
    ``ChemistryBenchmark._smiles_to_rdkit_features`` in src/benchmark.py:
    invalid SMILES become an all-zero descriptor row (kept, not dropped) and
    NaN/inf descriptors are zeroed. Callers that want to drop invalid SMILES
    must do so *before* calling this (and drop the matching y rows).
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.ML.Descriptors import MoleculeDescriptors
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

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
    """Generic dense features -> StandardScaler -> PCA(pca_dim).

    Used for the composition one-hot (perovskite) and Magpie (matbench) feature
    matrices so they share the FreeSolv featurization contract.
    """
    X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = StandardScaler().fit_transform(X)
    n_comp = min(pca_dim, X_scaled.shape[1])
    return PCA(n_components=n_comp).fit_transform(X_scaled)


# ---------------------------------------------------------------------------
# Caching + CSV output
# ---------------------------------------------------------------------------

def cache_path(name):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}_featurized.pkl"


def load_cache(name):
    """Return the cached featurized pool dict, or None if absent."""
    p = cache_path(name)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def save_cache(name, X_pca, y_hf, y_lf, meta=None):
    p = cache_path(name)
    with open(p, "wb") as f:
        pickle.dump(
            {"X_pca": np.asarray(X_pca, dtype=np.float64),
             "y_hf": np.asarray(y_hf, dtype=np.float64),
             "y_lf": np.asarray(y_lf, dtype=np.float64),
             "meta": meta or {}},
            f,
        )
    return p


def write_benchmark_csv(name, X_pca, y_hf, y_lf):
    """Write the numeric ``f0..f{d-1}, HF, LF`` CSV read by ChemistryBenchmark.

    The CSV intentionally contains ONLY the PCA feature columns plus HF/LF, so
    ``ChemistryBenchmark(use_smiles=False)`` treats f0..f{d-1} as the features.
    """
    X_pca = np.asarray(X_pca, dtype=np.float64)
    cols = {f"f{i}": X_pca[:, i] for i in range(X_pca.shape[1])}
    cols["HF"] = np.asarray(y_hf, dtype=np.float64)
    cols["LF"] = np.asarray(y_lf, dtype=np.float64)
    out = DATA_DIR / f"{name}.csv"
    pd.DataFrame(cols).to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# Diagnostics (printed on first load, per task spec)
# ---------------------------------------------------------------------------

def print_diagnostics(name, X_pca, y_hf, y_lf, rho, objective):
    """Print pool size, feature dim, rho, Pearson r(LF, HF), objective direction.

    Pearson r is computed on the LF/HF targets as written to the CSV. It is
    invariant to the ``negate=True`` sign flip the pipeline later applies (both
    series flip together), so this is the same correlation the BO loop sees.
    """
    X_pca = np.asarray(X_pca)
    y_hf = np.asarray(y_hf, dtype=np.float64)
    y_lf = np.asarray(y_lf, dtype=np.float64)
    r = float(np.corrcoef(y_lf, y_hf)[0, 1])
    print(f"[{name}] pool size           : {len(X_pca)}")
    print(f"[{name}] feature dim (LF=HF) : {X_pca.shape[1]}")
    print(f"[{name}] rho (cost ratio)    : {rho}")
    print(f"[{name}] Pearson r(LF, HF)   : {r:.4f}  (R^2 = {r**2:.4f})")
    print(f"[{name}] objective           : {objective}")
    print(f"[{name}] HF range            : [{y_hf.min():.4f}, {y_hf.max():.4f}]")
    print(f"[{name}] LF range            : [{y_lf.min():.4f}, {y_lf.max():.4f}]")
    return r
