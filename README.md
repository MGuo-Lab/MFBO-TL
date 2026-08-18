# Transfer-learning surrogates for multi-fidelity Bayesian optimization

Publication repository for the manuscript

> **Transfer Learning Architectures for Scalable Multi-Fidelity Bayesian
> Optimization**
> (Springer Nature `sn-jnl` class)

It contains the manuscript, the canonical result files behind every figure and
quoted statistic, the scripts that regenerate all of them, and the frozen
benchmark code that produced the results. It was distilled from the full
experiment repository (`MFBO-TL-Paper`) — see `PROVENANCE.md` for exactly what
was copied, renamed, and left behind.

## Layout

| Path | Contents |
|---|---|
| `paper/` | `main.tex`, `references.bib`, `main.bbl`, `sn-jnl.cls`, `sn-nature.bst` and the 12 canonical figures in `paper/paper_figures/` (the arXiv submission package plus the backmatter statements merged upstream in PR #1, `e1628a8`) |
| `src/` | Frozen pipeline: `benchmark.py` (BO loop, 12 TL/DNN surrogates, benchmark loaders), `baselines.py` (NARGP, deep-kernel GP, sparse variational MFGP), `synthetic_functions.py` (Branin/Park LF–HF pairs) |
| `benchmarks/` | Dataset build scripts (`python -m benchmarks.hopv15`, `python -m benchmarks.matbench_gap`) — regenerate `data/*.csv` from public sources; `_common.py` does featurize → standardize → PCA(10) |
| `data/` | The nine candidate-pool CSVs used by every run (already built; 1.3 MB) |
| `results/` | Canonical result CSVs consumed by the figure/statistics scripts (see the figure map below) |
| `figures/` | Regeneration scripts. Outputs land in `figures/out/` (gitignored) for comparison against the shipped `paper/paper_figures/` |
| `runners/` | The scripts that produced `results/` (provenance; re-running is optional and compute-heavy) |
| `docs/` | `ROOT_CAUSE_INVESTIGATION.md` — the const-std LF-EI decision record referenced from `src/benchmark.py` |

## Rebuild the paper

```bash
cd paper && latexmk -pdf main.tex
```

## Regenerate every figure

Needs only `numpy pandas scipy matplotlib` (no torch). Run from anywhere:

```bash
cd figures
python make_final_regret.py          # Fig 1 b–k
python make_fig1c_and_anytime_row.py # Fig 1 l–n + Supp family-split grid
python make_regret_trajectory.py     # Fig 2
python make_a6_acquisition_matrix.py # Fig 3
python make_acquisition_portfolio.py # Fig 4 a–d
python make_a1_calibration.py        # Fig 4 e–m
python make_b2_topk.py               # Fig 5 a,b
python make_scaling_law.py           # Fig 5 c
python plot_computing_flops.py       # Supp computing_time (+ FLOPs table CSV)
python make_si_star_sensitivity.py   # Supp star-sensitivity figure + SI table
```

Outputs appear in `figures/out/`; compare against the canonical
`paper/paper_figures/`. `fig1_overview.png` is the one hand-made asset (no
generating script; the orphaned TikZ source is kept as
`figures/fig1_overview.tex` for provenance — do not regenerate from it).

## Reproduce the quoted statistics

```bash
cd figures
python grid_stats.py                     # 126-cell grid self-test + headline fractions
python plot_computing_flops.py           # writes out/computing_flops_values.csv (needed next)
python stats_friedman_nemenyi_9bench.py  # Friedman/Nemenyi rank statistics (SI)
python effect_sizes_manuscript0713.py    # rank-biserial effect-size tables (SI)
```

### Environment sensitivity note

Every figure and all corrected statistics (BH-corrected bin stars, map-level
Wilcoxon) reproduce across scipy versions. The *uncorrected* per-cell Wilcoxon
counts and a few borderline SI p-values follow scipy's exact-vs-approximation
switching on 10–20 paired seeds, so matching the manuscript digit-for-digit
requires the pinned `scipy==1.13.1` (`requirements.txt`): e.g. the anytime-AUC
map's 122/126 raw significant cells appear as 112/126 under scipy 1.15. The
`figures/grid_stats.py` self-test (`python grid_stats.py`) passes in full under
the pinned environment and flags exactly this drift otherwise.

## Re-run the experiments (optional, compute-heavy)

`requirements.txt` pins the reference environment (Python 3.9, torch 2.4.1,
botorch 0.10.0, gpytorch 1.11, rdkit). Main entry points:

- `python src/benchmark.py --n-seeds 20 --n-workers 48` — 12-surrogate suite
- `python runners/run_fix6.py` — 15-surrogate trajectory cells (`results/traj_cells/`)
- `python runners/run_flop_profile.py` — FLOP profile (`results/flop_profile/`)
- `python runners/ranking_analysis.py` — LF/HF ranking metrics
- `runners/grid/`: `featurize_polariz.py` → `grid_design.py` → `gen_grid_cells.py`
  → `run_grid_cell.py` per cell — the 126-cell controlled fidelity-quality grid
  (`results/grid/`)
- `runners/analyze_surrogates.py` — acquisition-portfolio aggregation (needs the
  raw per-acquisition runs, which are **not** shipped; see `PROVENANCE.md`)

## Figure → script → data map

| Figure (paper/paper_figures/) | Script (figures/) | Reads (results/…) |
|---|---|---|
| `final_regret.pdf` | `make_final_regret.py` | `main_9bench`, `extra_baselines`, `gpfamily_newbench` |
| `fig1_anytime_row.pdf`, `fig1c_family_split_grid.pdf` | `make_fig1c_and_anytime_row.py` | `grid/` (+ `grid_stats.py`) |
| `regret_trajectory.pdf` | `make_regret_trajectory.py` | `traj_cells/`, `data/*.csv` |
| `A6_acquisition_matrix.pdf` | `make_a6_acquisition_matrix.py` | `main_corrected`, `main_9bench`, `mfgp_greedy_7bench`, `mfgp_greedy_newbench` |
| `acquisition_portfolio.pdf` | `make_acquisition_portfolio.py` | `acq_portfolio/surrogates_{winrate,means}.csv` |
| `A1_calibration_vs_regret.pdf` | `make_a1_calibration.py` | `calibration_sweep`, `calibration_{newbench,synthetic,park_early,gpfamily}` |
| `B2_topk_overlap_and_screening.pdf` | `make_b2_topk.py` | `ranking_analysis/ranking_metrics.csv`, `data/*.csv`, `src/synthetic_functions.py` |
| `compute_scaling_law.pdf` | `make_scaling_law.py` | `flop_profile/flop_profile.csv` |
| `computing_time.pdf` | `plot_computing_flops.py` | `flop_profile/flop_profile.csv` |
| `si_star_sensitivity.pdf` | `make_si_star_sensitivity.py` | `grid/` (+ `grid_stats.py`) |
| `fig1_overview.png` | — (hand-made) | — |

Code ↔ paper naming: `TwoStageJoint` = Pretrain-then-Joint Training,
`DNGOJoint` = Stop-Gradient Joint Training, `DNGOGradient` = End-to-End Joint
Training (`RENAME_MAP` in `figures/_common.py`).
