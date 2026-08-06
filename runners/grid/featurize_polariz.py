#!/usr/bin/env python
"""Reproduce ChemistryBenchmark's SMILES featurization for Polarizability WITHOUT
importing torch/botorch (rdkit + sklearn only -> login-safe). Writes a drop-in
CSV (f0..f9, HF, LF) where HF/LF are the SIGNED objective (negate=True applied,
best = min), so step0_feasibility.py can use it unchanged.
    PY=/users/k23070952/.conda/envs/paper_v1_py39/bin/python; $PY featurize_polariz.py
"""
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import os
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SRC = f"{REPO}/data/polarizability.csv"
OUT = f"{REPO}/data/polarizability_pca10.csv"

df = pd.read_csv(SRC)
names = [d[0] for d in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)
feats = []
for smi in df["smiles"].values:
    mol = Chem.MolFromSmiles(smi)
    feats.append(calc.CalcDescriptors(mol) if mol is not None else [0.0] * len(names))
feats = np.nan_to_num(np.array(feats, float), nan=0.0, posinf=0.0, neginf=0.0)

# featurization: StandardScaler -> PCA(10, rs=42)  (matches src/benchmark.py)
pca_x = PCA(n_components=10, random_state=42).fit_transform(StandardScaler().fit_transform(feats))
# __init__ then applies one more StandardScaler to self.X
X = StandardScaler().fit_transform(pca_x)

# negate=True for Polarizability -> signed objective, best = min
y_hf = -df["HF"].to_numpy(float)
y_lf = -df["LF"].to_numpy(float)

out = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
out["HF"] = y_hf
out["LF"] = y_lf
out.to_csv(OUT, index=False)
r2 = float(np.corrcoef(y_hf, y_lf)[0, 1]) ** 2
print(f"wrote {OUT}  N={len(out)}  d=10  realLF R2={r2:.3f}")
print(f"HF(signed) min={y_hf.min():.3f} max={y_hf.max():.3f}  best=min")
