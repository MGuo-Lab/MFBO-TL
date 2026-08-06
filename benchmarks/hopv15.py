#!/usr/bin/env python
"""HOPV15 (Harvard Organic Photovoltaic) benchmark loader.

Source
------
S. A. Lopez et al., "The Harvard organic photovoltaic dataset", Scientific Data
3, 160086 (2016). DOI: 10.1038/sdata.2016.86.
Canonical record: Harvard Dataverse, handle 1/29408375
(https://dash.harvard.edu/handle/1/29408375), file ``HOPV_15_revised_2.data``.

This loader downloads the widely-mirrored CSV-ified HOPV table (350 donor
molecules with SMILES, electronic levels and experimental photovoltaic metrics)
from the DeepChem/MoleculeNet S3 mirror, because the Harvard Dataverse host was
unreachable from the compute cluster at build time. The CSV exposes exactly the
fields the task expects (HOMO / LUMO / optical_gap / PCE).

Fidelities
----------
* HF (high fidelity)  = experimental power conversion efficiency (PCE, %).
* LF (low fidelity)   = Scharber-model PCE estimated from the donor HOMO and
  optical gap (Scharber et al., Adv. Mater. 18, 789 (2006)). This is the
  cheap, physics-based proxy for the measured PCE.

Objective : MAXIMIZE experimental PCE  -> registered with negate=True so the
            frozen min-oriented pipeline optimizes it unchanged.
rho       : 0.1  (same cost ratio as FreeSolv).
Featurize : SMILES -> RDKit 2D descriptors -> StandardScaler -> PCA(10),
            identical to the FreeSolv pipeline.

Run directly to download + featurize + cache + print diagnostics::

    python -m benchmarks.hopv15
"""

import sys
import urllib.request
import tarfile
import io

import numpy as np
import pandas as pd

from benchmarks import _common

NAME = "hopv15"
RHO = 0.1
NEGATE = True  # maximize experimental PCE
OBJECTIVE = "maximize HF (experimental PCE); negate=True"

# DeepChem / MoleculeNet mirror of the HOPV table (clean CSV with the fields we need).
HOPV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/hopv.tar.gz"
_RAW_CSV = _common.CACHE_DIR / "hopv_raw.csv"

# --- Scharber 2006 model constants -----------------------------------------
# PCE = V_OC * J_SC * FF / P_in, with the donor HOMO setting V_OC and the donor
# optical gap setting the max achievable photocurrent.
ACCEPTOR_LUMO = 4.3   # |E_LUMO(PC61BM)| in eV (Scharber 2006)
VOC_EMPIRICAL_LOSS = 0.3   # V, empirical loss term in V_OC
EQE = 0.65            # assumed external quantum efficiency (Scharber 2006)
FILL_FACTOR = 0.65    # assumed fill factor (Scharber 2006)
P_IN = 100.0          # mW/cm^2, AM1.5G one-sun input power

# Maximum AM1.5G short-circuit current density vs band gap (mA/cm^2), assuming
# step-function absorption + 100% internal collection (the Shockley-Queisser
# integrated photocurrent). Tabulated from the standard AM1.5G / SQ values
# (cf. Ruehle, Solar Energy 130 (2016) 139). Scharber's J_SC = EQE * J_max(E_g).
_EG_GRID = np.array([0.50, 0.75, 1.00, 1.10, 1.20, 1.30, 1.34, 1.40, 1.50, 1.60,
                     1.70, 1.80, 1.90, 2.00, 2.20, 2.40, 2.60, 2.80, 3.00])
_JMAX_GRID = np.array([65.0, 58.0, 49.6, 44.0, 41.0, 36.5, 35.0, 32.0, 28.0, 24.5,
                       21.0, 18.0, 15.5, 13.0, 9.5, 6.8, 4.6, 3.0, 1.8])


def _scharber_pce(homo, e_gap):
    """Scharber-model PCE (%) from donor HOMO (eV, negative) and gap (eV)."""
    v_oc = np.maximum(0.0, (-homo) - ACCEPTOR_LUMO - VOC_EMPIRICAL_LOSS)
    j_max = np.interp(e_gap, _EG_GRID, _JMAX_GRID)  # clamps outside the grid
    j_sc = EQE * j_max
    return v_oc * j_sc * FILL_FACTOR / P_IN * 100.0


def _download_raw():
    """Fetch the HOPV CSV (cached). Raise with clear instructions on failure."""
    if _RAW_CSV.exists():
        return pd.read_csv(_RAW_CSV)
    _common.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(HOPV_URL, timeout=120) as resp:
            blob = resp.read()
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith("hopv.csv"))
            df = pd.read_csv(tar.extractfile(member))
    except Exception as e:
        raise RuntimeError(
            "Failed to download the HOPV dataset from the DeepChem mirror "
            f"({HOPV_URL}): {e!r}\n"
            "Download it manually and place a CSV with columns "
            "[smiles, HOMO, LUMO, optical_gap, PCE] at:\n"
            f"    {_RAW_CSV}\n"
            "Canonical source: Harvard Dataverse handle 1/29408375 "
            "(https://dash.harvard.edu/handle/1/29408375), file "
            "HOPV_15_revised_2.data; Lopez et al., Sci. Data 3, 160086 (2016)."
        ) from e
    df.to_csv(_RAW_CSV, index=False)
    return df


def prepare(force=False):
    """Download + featurize HOPV15, cache the pool, write data/hopv15.csv.

    Returns the featurized pool dict. Prints the required diagnostics on first
    (uncached) build.
    """
    if not force:
        cached = _common.load_cache(NAME)
        if cached is not None:
            _common.write_benchmark_csv(NAME, cached["X_pca"], cached["y_hf"], cached["y_lf"])
            return cached

    df = _download_raw()

    # Coalesce a usable band gap: optical gap -> electrochemical gap -> (LUMO-HOMO).
    gap = df["optical_gap"].copy()
    if "electrochemical_gap" in df.columns:
        gap = gap.fillna(df["electrochemical_gap"])
    gap = gap.fillna(df["LUMO"] - df["HOMO"])

    work = pd.DataFrame({
        "smiles": df["smiles"].astype(str),
        "homo": pd.to_numeric(df["HOMO"], errors="coerce"),
        "gap": pd.to_numeric(gap, errors="coerce"),
        "pce": pd.to_numeric(df["PCE"], errors="coerce"),
    })

    # Drop invalid SMILES (RDKit-unparseable) and rows missing HF or LF inputs.
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    valid_smiles = work["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
    keep = valid_smiles & work["homo"].notna() & work["gap"].notna() & work["pce"].notna()
    n_dropped = int((~keep).sum())
    work = work[keep].reset_index(drop=True)
    print(f"[{NAME}] kept {len(work)} / {len(df)} molecules "
          f"({n_dropped} dropped: invalid SMILES or missing HOMO/gap/PCE)")

    y_hf = work["pce"].to_numpy(dtype=np.float64)                         # experimental PCE
    y_lf = _scharber_pce(work["homo"].to_numpy(np.float64),
                         work["gap"].to_numpy(np.float64))                # Scharber PCE

    X_pca = _common.smiles_to_rdkit_pca(work["smiles"].tolist())

    _common.save_cache(NAME, X_pca, y_hf, y_lf,
                       meta={"rho": RHO, "negate": NEGATE, "objective": OBJECTIVE,
                             "smiles": work["smiles"].tolist()})
    _common.write_benchmark_csv(NAME, X_pca, y_hf, y_lf)
    _common.print_diagnostics(NAME, X_pca, y_hf, y_lf, RHO, OBJECTIVE)
    return {"X_pca": X_pca, "y_hf": y_hf, "y_lf": y_lf}


if __name__ == "__main__":
    pool = prepare(force="--force" in sys.argv)
    _common.print_diagnostics(NAME, pool["X_pca"], pool["y_hf"], pool["y_lf"], RHO, OBJECTIVE)
