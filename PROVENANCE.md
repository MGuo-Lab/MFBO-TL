# Provenance

This repository was distilled on **2026-08-06** from two source repositories on
the KCL CREATE filesystem, keeping only what is needed to publish and verify the
manuscript. Nothing here was newly computed; every result CSV and figure is a
byte-for-byte copy of the canonical artifact, and scripts were modified **only**
in their path-resolution lines (plus output redirection into `figures/out/`).

## Sources

| Source | State at copy time |
|---|---|
| `Jaewook-MFBO-TL-Paper` (github.com/jaewook-lee4014/Jaewook-MFBO-TL-Paper) | commit `63d2262` ("Add a submission package that carries main.bbl") — supplied everything under `paper/` |
| `MFBO-TL-Paper` (local only, never pushed) | branch `fix/lf-ei-const-std-restore-original` @ `7b9d476`; the figure scripts and most result CSVs were **untracked** in that repo — this repo is their first version control |

## Path map (old → new)

| Old (MFBO-TL-Paper) | New (this repo) |
|---|---|
| `src/{benchmark,synthetic_functions}.py` | `src/` (unchanged) |
| `experiments/extra_baselines/baselines.py` | `src/baselines.py` |
| `benchmarks/{__init__,_common,hopv15,matbench_gap}.py` | `benchmarks/` (unchanged) |
| `paper/paper_document/Jaewook_MFBO_TL_Paper_0713/new_figures/_scripts/*` | `figures/` |
| `experiments/regime_review_20260625/{grid_stats,make_si_star_sensitivity}.py` | `figures/` |
| `paper/auto_figures/{plot_computing_flops,effect_sizes_manuscript0713,effect_sizes_rank_biserial}.py` | `figures/` |
| `paper/paper_document/paper_figures/fig1_overview.tex` | `figures/fig1_overview.tex` |
| `paper/auto_figures/ROOT_CAUSE_INVESTIGATION.md` | `docs/` |
| `results/main_9bench/results_{summary,trajectory}.csv` | `results/main_9bench/` |
| `results/main_corrected/results_{summary,trajectory}.csv` | `results/main_corrected/` (MERGED variants, per-cell summaries, logs left behind) |
| `results/run_20260527_030709/*.csv` | `results/mfgp_greedy_7bench/` (renamed for meaning: MFGP greedy policy, 7 original benchmarks) |
| `results/mfgp_greedy_newbench/*` | `results/mfgp_greedy_newbench/` |
| `results/gpfamily_newbench/results_{summary,trajectory}.csv` | `results/gpfamily_newbench/` |
| `results/post_submission/extra_baselines_results/results_{summary,trajectory}.csv` | `results/extra_baselines/` (618 per-seed CSVs left behind — they aggregate into the two shipped files) |
| `results/post_submission/calibration_sweep/results_summary.csv` | `results/calibration_sweep/` |
| `results/calibration_{newbench,synthetic,park_early,gpfamily}/results_summary.csv` | same names |
| `results/post_submission/ranking_analysis/ranking_metrics.csv` | `results/ranking_analysis/` |
| `results/flop_profile_20260530/flop_profile.csv` | `results/flop_profile/` |
| `results/acq_portfolio/surrogates_{winrate,means}.csv` | `results/acq_portfolio/` |
| `experiments/fix6_promote/results/cells/traj_*.csv` (133 files) | `results/traj_cells/` (the 133 matching `summary_*.csv` left behind; figures read only the trajectories) |
| `experiments/matbench_budget_ext/mlip_elastic/grid_manifest.csv` | `results/grid/grid_manifest.csv` |
| `experiments/matbench_budget_ext/results_grid_polariz/cells/summary_cell_*.csv` (126) | `results/grid/cells/` (the 100 MB of `traj_cell_*.csv` left behind — not read by any figure) |
| `experiments/fix6_promote/run_fix6.py` | `runners/` |
| `experiments/extra_baselines/run_flop_profile.py` | `runners/` (hard-coded absolute repo path made relative) |
| `experiments/ranking_analysis/ranking_analysis.py` | `runners/` |
| `experiments/acquisition_portfolio/analyze_surrogates.py` | `runners/` |
| `experiments/matbench_budget_ext/{run_ext.py, mlip_elastic/{featurize_polariz,grid_design,gen_grid_cells,run_grid_cell}.py}` | `runners/grid/` |

## Script modifications

Path-resolution only, verified by regenerating every figure and statistic:

- `figures/_common.py`: `REPO` depth 4 → 1; `NEWFIGS` (output dir) → `figures/out/`
- `figures/{grid_stats,make_fig1c_and_anytime_row,effect_sizes_manuscript0713}.py`:
  grid inputs re-pointed from `experiments/matbench_budget_ext/…` to `results/grid/`
- `figures/make_regret_trajectory.py`: cells re-pointed to `results/traj_cells/`
- `figures/{make_a1_calibration,make_final_regret,make_b2_topk}.py`:
  `post_submission/` nesting flattened
- `figures/{make_scaling_law,plot_computing_flops}.py`: `flop_profile_20260530` → `flop_profile`
- `figures/make_a6_acquisition_matrix.py`: `run_20260527_030709` → `mfgp_greedy_7bench`
- `figures/{plot_computing_flops,make_si_star_sensitivity,stats_friedman_nemenyi_9bench,effect_sizes_*}.py`:
  outputs redirected into `figures/out/`
- `runners/*`: `sys.path` inserts collapsed to `src/` (which now also holds
  `baselines.py`); `run_ext` imported from its own directory
- The `mfgp_ei`→`mfgp_greedy_7bench` naming reflects what the data are: the
  MFGP greedy-policy control on the seven original benchmarks
  (`make_a6_acquisition_matrix.py` merges them with `mfgp_greedy_newbench`)

## Deliberately left behind (in `MFBO-TL-Paper`)

- ~1.0 GB SLURM logs, 27 `__pycache__` dirs, LaTeX build artifacts
- ~2.3 GB abandoned/superseded experiment campaigns (`uq_calib_rerun_20260702`,
  `claim_audit_20260607`, `lf_coverage_design`, `keystone_sweep_20260629`,
  `tldk_lab`, …) — many back claims recorded as DEAD/REFUTED in that repo's wiki
- ~575 MB result dirs no current figure reads, including `results/paper_v1`
  (consumed only by the retired ICML pipeline `src/plotting/`) and the raw
  per-acquisition portfolio runs (297 MB backing the two shipped 29 KB CSVs)
- `data/grid_cells/` (32 MB, 126 semi-synthetic pools): inputs for *re-running*
  the grid; regenerable via `runners/grid/featurize_polariz.py` →
  `grid_design.py` → `gen_grid_cells.py`
- `data/cache/` (21 MB): featurization caches + raw HOPV15 download; regenerable
  via `python -m benchmarks.hopv15` / `benchmarks.matbench_gap`
- Old ICML pipeline (`src/plotting/`, `paper/figures/`), 10 `main*.tex` backup
  variants, `wiki/`, dated audit narratives in `docs/`

## Known paper ↔ code gaps (inherited, documented)

- The BLR uncertainty head is fitted on the LF network only; the LF EI uses the
  BLR mean with a **constant** predictive std (0.1) — the original method that
  produced the published results. Full decision record:
  `docs/ROOT_CAUSE_INVESTIGATION.md`.
- HF selection is greedy argmin of the corrected mean (no HF EI).
