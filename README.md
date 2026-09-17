# MFBO-TL

Code and data for **Transfer-Learning Surrogates for Scalable Multi-Fidelity
Bayesian Optimization in Molecular and Materials Discovery**, by Jaewook Lee,
Ethan Errington, Christian D. Lorenz and Miao Guo.

Version: **1.0.0**.

## Installation

The reference environment uses Python 3.9 on Linux. Installation and the example
were checked on Ubuntu 22.04. A CPU is sufficient; no special hardware is required.
Allow approximately 5–10 minutes for installation, depending on download speed.

```bash
git clone https://github.com/MGuo-Lab/MFBO-TL.git
cd MFBO-TL
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_THREADS=1
```

`requirements.txt` fixes NumPy, pandas, SciPy, scikit-learn, Matplotlib, PyTorch,
BoTorch, GPyTorch and RDKit versions. Commands below run from the repository root.

## Example

Recreate the attainment figure from the supplied per-seed results:

```bash
python figures/make_attainment.py
```

Expected output is `figures/out/fig1_attainment.pdf` and its numerical CSV files.
Reference values are in `results/expected/fig1_attainment_values.csv`.
This example takes approximately 10 seconds on a CPU.

To run one of the optimization trajectories used in the paper:

```bash
python runners/run_tl.py --benches Branin-Fav --models DNGOJoint --seeds 42
```

This uses the supplied 2,500-point Branin pool and a budget of 50 HF-equivalent
units. It writes the trajectory, query indices and summary under
`outputs/blr_replace__hf_argmin/cells/`. The recorded best-so-far trajectory is
provided in `examples/branin_fav_seed42.csv`. Allow approximately 2–5 minutes on
a CPU for this single-seed run.

## Reproduce figures and tables

```bash
python reproduce.py
```

Outputs are written to `figures/out/`. Numerical references are in
`results/expected/`; figure inputs and table mappings are described in
[results/README.md](results/README.md).

| Paper item | Script in `figures/` |
|---|---|
| Fig. 1b–p: attainment and deficit | `make_attainment.py` |
| Fig. 2: regret trajectories | `make_trajectories.py` |
| Fig. 3: controlled fidelity-quality grid | `make_grid.py` |
| Fig. 4a,b: LF screening | `make_screening.py` |
| Fig. 4c: compute scaling | `make_scaling.py` |
| Supplementary Figs. 1, 2: acquisition policies | `make_acquisitions.py` |
| Supplementary Fig. 3: LF calibration | `make_calibration.py` |
| Supplementary Fig. 4: computational cost | `make_compute.py` |
| Supplementary numerical tables | `make_tables.py` |

The Fig. 1a schematic has a separate TikZ source. With TeX Live, TikZ,
`standalone` and Helvetica installed:

```bash
mkdir -p figures/out
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=figures/out figures/overview.tex
```

## Run the experiments

```bash
python runners/run_tl.py
python runners/run_gp.py
python runners/run_grid.py
```

The first two commands run the five TL and three GP surrogates on all thirteen
pools, with seeds 42–81. The grid command runs all eight surrogates on 126
conditions, with seeds 42–51. Use `--benches`, `--models`, `--seeds` and `--outdir`
to select a main experiment; the grid runner uses `--cells` instead of `--benches`.

| Paper label | Command-line model name |
|---|---|
| GP-base | `MFGP` |
| GP-SV | `SparseMFGP` |
| GP-DKL | `DKL` |
| TL-base | `DNGOJoint` |
| TL-PtJ | `TwoStageJoint` |
| TL-E2E | `DNGOGradient` |
| TL-SPS | `SoftParameterSharing` |
| TL-MMD | `DomainAdaptationMMD` |

[docs/experiments.md](docs/experiments.md) gives the supplementary experiment
commands, budgets and initial designs.

## Data and citation

`data/` contains the thirteen candidate pools and 126 controlled LF sources.
Dataset definitions, reconstruction commands and source references are in
[data/README.md](data/README.md). Per-seed results are under `results/`.

Code is released under the [MIT License](LICENSE). Third-party materials retain
their source terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Use [CITATION.cff](CITATION.cff) to cite this software. Questions about the code
can be posted to the repository's [issue tracker](https://github.com/MGuo-Lab/MFBO-TL/issues).
