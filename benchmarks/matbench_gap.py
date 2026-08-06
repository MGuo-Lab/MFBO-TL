#!/usr/bin/env python
"""Matbench experimental band-gap benchmark (Zhuo expt HF / Materials Project PBE LF).

Source
------
* HF (high fidelity) : ``matbench_expt_gap`` -- experimentally measured band gaps
  compiled by Y. Zhuo, A. Mansouri Tehrani, J. Brgoch,
  J. Phys. Chem. Lett. 9, 1668 (2018). Composition + experimental gap (eV).
* LF (low fidelity)  : ``matbench_mp_gap`` -- Materials Project GGA-PBE computed
  band gaps. Structure + PBE gap (eV).
Both are loaded through ``matminer.datasets.load_dataset`` (Dunn et al.,
npj Comput. Mater. 6, 138 (2020)).

Matching
--------
MP structures are reduced to their composition (reduced formula); the PBE gap is
averaged over polymorphs sharing a composition, then joined to the experimental
compositions. The matched pool is the set of compositions present in both
datasets (~few thousand).

Objective : MINIMIZE |E_g - 1.4| eV (photovoltaic target gap). We therefore write
            HF = |gap_expt - 1.4| and LF = |gap_pbe - 1.4| so the frozen
            min-oriented pipeline optimizes the right thing unchanged
            (negate=False).
rho       : 0.005  (PBE is very cheap relative to an experiment).
Featurize : composition -> Magpie ElementProperty -> StandardScaler -> PCA(10).

Run directly to load + match + featurize + cache + print diagnostics::

    python -m benchmarks.matbench_gap
"""

import sys

import numpy as np
import pandas as pd

from benchmarks import _common

NAME = "matbench_gap"
# rho raised from the original 0.005 -> 0.05: at rho=0.005 the deterministic
# round-robin does floor(1/rho)=200 LF picks per HF pick, so the frozen loop hit
# its max_iter=500 cap after only ~2-4 HF evaluations (HF-starved). rho=0.05
# gives ~20 LF per HF -> a balanced number of HF evals within the same loop.
RHO_PREV = 0.005
RHO = 0.05
RHO_ADJUST_MSG = "ρ adjusted from 0.005 to 0.05 for MFBO loop balance"
NEGATE = False  # objective transform already encoded in HF/LF below
TARGET_GAP = 1.4  # eV, photovoltaic sweet spot
OBJECTIVE = f"minimize |E_g - {TARGET_GAP}| eV (encoded as HF/LF distance); negate=False"

_MATCHED_CSV = _common.CACHE_DIR / "matbench_matched.csv"

_INSTALL_HINT = (
    "matminer + pymatgen are required for the matbench_gap benchmark.\n"
    "Install them into the active environment, e.g.:\n"
    "    pip install matminer pymatgen\n"
    "(pymatgen >= 2024 needs Python >= 3.10; on Python 3.9 pip resolves to an "
    "older compatible pymatgen/matminer automatically.)"
)


def _gap_col(df, kind):
    """Find the gap column ('gap expt' / 'gap pbe') robustly."""
    for c in df.columns:
        if c.lower().replace("_", " ") == f"gap {kind}":
            return c
    cands = [c for c in df.columns if "gap" in c.lower()]
    if len(cands) == 1:
        return cands[0]
    raise KeyError(f"Could not find the '{kind}' gap column in {list(df.columns)}")


def _load_and_match():
    """Load both matbench datasets and join on composition. Cached to CSV."""
    if _MATCHED_CSV.exists():
        return pd.read_csv(_MATCHED_CSV)

    try:
        from matminer.datasets import load_dataset
        from pymatgen.core import Composition
    except ImportError as e:
        raise RuntimeError(_INSTALL_HINT) from e

    try:
        df_expt = load_dataset("matbench_expt_gap")   # composition (str) + gap expt
        df_mp = load_dataset("matbench_mp_gap")        # structure + gap pbe
    except Exception as e:
        raise RuntimeError(
            "Failed to download/load the matbench datasets via matminer "
            f"({e!r}).\nIf the compute node has no internet, run this loader once "
            "on a host that does (it caches into matminer's dataset dir), then "
            "rerun.\nDatasets: matbench_expt_gap (Zhuo 2018), matbench_mp_gap "
            "(Materials Project PBE)."
        ) from e

    expt_gap = _gap_col(df_expt, "expt")
    pbe_gap = _gap_col(df_mp, "pbe")

    # Reduce experimental compositions and MP structures to a common formula key.
    print(f"[{NAME}] reducing {len(df_expt)} expt compositions ...", flush=True)
    df_expt = df_expt.copy()
    df_expt["formula"] = df_expt["composition"].apply(lambda s: Composition(s).reduced_formula)

    print(f"[{NAME}] reducing {len(df_mp)} MP structures to compositions "
          f"(this is the slow step) ...", flush=True)
    df_mp = df_mp.copy()
    df_mp["formula"] = df_mp["structure"].apply(lambda s: s.composition.reduced_formula)

    # Average PBE gap over polymorphs sharing a composition.
    mp_by_formula = df_mp.groupby("formula")[pbe_gap].mean()

    # Average duplicate experimental gaps too, then inner-join on formula.
    expt_by_formula = df_expt.groupby("formula")[expt_gap].mean()
    matched = pd.concat([expt_by_formula.rename("gap_expt"),
                         mp_by_formula.rename("gap_pbe")], axis=1, join="inner")
    matched = matched.reset_index().dropna(subset=["gap_expt", "gap_pbe"])

    _common.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    matched.to_csv(_MATCHED_CSV, index=False)
    print(f"[{NAME}] matched pool: {len(matched)} compositions "
          f"(expt {len(expt_by_formula)} INNER-JOIN mp {len(mp_by_formula)})", flush=True)
    return matched


def prepare(force=False):
    """Load + match + Magpie-featurize matbench gaps; cache pool; write CSV."""
    print(f"[{NAME}] {RHO_ADJUST_MSG}", flush=True)
    if not force:
        cached = _common.load_cache(NAME)
        if cached is not None:
            _common.write_benchmark_csv(NAME, cached["X_pca"], cached["y_hf"], cached["y_lf"])
            return cached

    matched = _load_and_match()

    # Objective transform: distance from the PV target gap (minimization).
    y_hf = np.abs(matched["gap_expt"].to_numpy(np.float64) - TARGET_GAP)
    y_lf = np.abs(matched["gap_pbe"].to_numpy(np.float64) - TARGET_GAP)

    # Magpie composition featurization.
    try:
        from matminer.featurizers.composition import ElementProperty
        from pymatgen.core import Composition
    except ImportError as e:
        raise RuntimeError(_INSTALL_HINT) from e

    print(f"[{NAME}] Magpie-featurizing {len(matched)} compositions ...", flush=True)
    ep = ElementProperty.from_preset("magpie")
    ep.set_n_jobs(1)  # deterministic, single-process (a few thousand comps is fast)
    comps = [Composition(f) for f in matched["formula"]]
    X_magpie = np.array(ep.featurize_many(comps, ignore_errors=True, pbar=False),
                        dtype=np.float64)
    X_pca = _common.matrix_to_pca(X_magpie)

    _common.save_cache(NAME, X_pca, y_hf, y_lf,
                       meta={"rho": RHO, "negate": NEGATE, "objective": OBJECTIVE,
                             "target_gap": TARGET_GAP,
                             "formula": matched["formula"].tolist()})
    _common.write_benchmark_csv(NAME, X_pca, y_hf, y_lf)
    _common.print_diagnostics(NAME, X_pca, y_hf, y_lf, RHO, OBJECTIVE)
    return {"X_pca": X_pca, "y_hf": y_hf, "y_lf": y_lf}


if __name__ == "__main__":
    pool = prepare(force="--force" in sys.argv)
    _common.print_diagnostics(NAME, pool["X_pca"], pool["y_hf"], pool["y_lf"], RHO, OBJECTIVE)
