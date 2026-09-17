# Candidate pools

All CSV files include `HF` and `LF`. Numeric feature columns are named `f0`,
`f1`, …, except COFs, which retains its fourteen source descriptor names.
FreeSolv and Polarizability contain `smiles`; their RDKit descriptors are
standardized and reduced to ten principal components at loading. HOPV15 and
the five inorganic pools store ten principal components. Model inputs are
standardized over the candidate pool. Each row is one candidate; row order
also defines the indices in the query logs.

| Pool | File | Candidates | Objective minimized | `rho` |
|---|---|---:|---|---:|
| Branin-Fav | `synthetic_branin_fav.csv` | 2,500 | Branin value | 0.1 |
| Branin-Unfav | `synthetic_branin_unfav.csv` | 2,500 | Branin value | 0.5 |
| Park-Fav | `synthetic_park_fav.csv` | 256 | Park value | 0.1 |
| Park-Unfav | `synthetic_park_unfav.csv` | 256 | Park value | 0.5 |
| COFs | `cofs.csv` | 608 | Negative Xe/Kr selectivity | 0.065 |
| FreeSolv | `freesolv.csv` | 640 | Negative hydration free energy | 0.1 |
| Polarizability | `polarizability.csv` | 1,134 | Negative normalized polarizability | 0.167 |
| HOPV15 | `hopv15.csv` | 240 | Negative experimental PCE (%) | 0.1 |
| Matbench-Gap | `matbench_gap.csv` | 3,347 | Absolute gap distance from 1.4 eV | 0.05 |
| ExptGap-PBE | `expt_gap_pbe.csv` | 3,508 | Absolute gap distance from 1.4 eV | 0.05 |
| Elastic-CHGNet | `elastic_chgnet_shear.csv` | 10,819 | Negative shear modulus (GPa) | 0.05 |
| Elastic-SevenNet | `elastic_sevennet_shear.csv` | 10,460 | Negative shear modulus (GPa) | 0.05 |
| Elastic-MatterSim | `elastic_mattersim_shear.csv` | 10,946 | Negative shear modulus (GPa) | 0.05 |

COFs, Polarizability and HOPV15 store positive objectives and are negated by the
loader. The other files already encode the minimized objective. Regret is the
best observed HF objective minus the candidate-pool minimum, in that pool's
objective units.

`grid_cells/cell_00.csv` through `cell_125.csv` contain the controlled LF sources
on the 1,134-candidate Polarizability pool. They use signed objectives and
`rho = 0.1`. `polarizability_pca10.csv` contains their common features and HF
values. The condition parameters are in `results/grid/grid_manifest.csv`.

## Sources

- COFs, FreeSolv and Polarizability: [Sabanza-Gil et al., versioned code and data](https://doi.org/10.5281/zenodo.15269206), commit `1c24c08c99df12fc598e43493b309631bef9d1a2`. These pair GCMC Xe/Kr selectivity with Henry's-law selectivity, experimental hydration free energies with calculated values, and experimental polarizabilities with Hartree–Fock values, respectively. Original sources: [Gantzler et al.](https://doi.org/10.1039/D3DD00117B), [Mobley and Guthrie](https://doi.org/10.1007/s10822-014-9747-x), and [Ghahremanpour et al.](https://doi.org/10.1038/sdata.2018.62).
- HOPV15: [Lopez et al.](https://doi.org/10.1038/sdata.2016.86), obtained from the [DeepChem dataset mirror](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/hopv.tar.gz). HF is measured PCE; LF is the Scharber proxy calculated from HOMO and the available optical or electrochemical gap. Rows require valid SMILES and complete PCE, HOMO and gap values.
- Matbench-Gap: `matbench_expt_gap` and `matbench_mp_gap` from [Matbench](https://doi.org/10.1038/s41524-020-00406-3), loaded through [matminer](https://hackingmaterials.lbl.gov/matminer/datasets.html).
- ExptGap-PBE: matminer `expt_gap` ([Zhuo et al.](https://doi.org/10.1021/acs.jpclett.8b00124)) and `mp_nostruct_20181018` ([Materials Project](https://doi.org/10.1063/1.4812323)). Both gap pools average entries sharing a reduced formula and retain formulas present at both fidelities.
- Elasticity: Matbench `matbench_log_gvrh` ([de Jong et al.](https://doi.org/10.1038/sdata.2015.9)) supplies DFT shear moduli. LF moduli are calculated with [matcalc](https://github.com/materialsvirtuallab/matcalc) using [CHGNet](https://doi.org/10.1038/s42256-023-00716-3), [SevenNet](https://doi.org/10.1021/acs.jctc.4c00190), or [MatterSim](https://github.com/microsoft/mattersim). Finite, positive LF and HF moduli are retained separately for each potential.

Source datasets retain their original terms and attribution. The source
archive's MIT notice is reproduced in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Reconstruction

The supplied tables are ready for the experiment runners. The synthetic grids,
source chemistry tables and controlled LF sources can be prepared with:

```bash
python -m benchmarks.synthetic
python -m benchmarks.chemistry
python -m benchmarks.grid
```

For the molecular and composition preprocessing environment, use Python 3.9
and `requirements-data.txt` in a separate virtual environment:

```bash
python3.9 -m venv .venv-data
source .venv-data/bin/activate
python -m pip install -r requirements-data.txt
python -m benchmarks.hopv15
python -m benchmarks.matbench_gap
python -m benchmarks.expt_gap_pbe
python -m benchmarks.elastic_modulus
```

The elasticity builder reads the three `sources/*_elastic.csv` files. They
contain Matbench row `idx`, reduced `formula`, `mlip_G`, `mlip_K` and `dft_G`
(moduli in GPa). The candidate-pool builder uses the shear columns only.

To recalculate LF elastic moduli, `benchmarks/mlip_elastic.py` uses Python 3.12,
matcalc 0.5.1 and matminer 0.10.1. The source calculations used CHGNet 0.4.2,
sevenn 0.12.1 and MatterSim 1.2.5 in separate environments, with CPU calculators,
structure relaxation and `fmax = 0.1`:

```bash
python -m benchmarks.mlip_elastic --model CHGNet --workers 16 --out outputs/chgnet_elastic.csv
python -m benchmarks.mlip_elastic --model SevenNet --workers 16 --out outputs/sevennet_elastic.csv
python -m benchmarks.mlip_elastic --model MatterSim --workers 16 --out outputs/mattersim_elastic.csv
```

The recorded SevenNet calculations cover 10,560 of the 10,987 Matbench
structures. Their supplied row indices define the evaluated candidate set.
