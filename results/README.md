# Recorded results

All result files refer to the candidate pools in `data/`. Main experiments use
seeds 42–81. GP-SV has 36 completed seeds on Branin-Fav and 39 on Branin-Unfav.
Grid conditions use seeds 42–51; acquisition controls use seeds 42–61.

| Input | Contents | Paper output |
|---|---|---|
| `main/trajectories.json` | Per-seed best-so-far trajectories and target thresholds | Figs. 1, 2; TL-base comparisons and summary deficits |
| `main/calibration.csv` | Mean LF expected calibration error by pool, model and seed | Supplementary Fig. 3 |
| `grid/grid_manifest.csv` | Parameters and LF quality measures for 126 conditions | Fig. 3 |
| `grid/cells/summary_cell_*.csv` | Per-model, per-seed grid outcomes | Fig. 3 |
| `flop_profile_blr/flop_profile.csv` | Fit and prediction FLOPs at five training fractions | Fig. 4c; Supplementary Fig. 4 |
| `controls/acquisition_trajectories.csv` | Per-seed trajectories for the LF acquisition policies | Supplementary Figs. 1, 2 |
| `controls/feature_ablation.csv` | Attainment and final regret for the feature ablations | Supplementary Table 6 |
| `controls/head_path.csv` | Attainment and final regret for six TL-base heads | Supplementary Table 7 |
| `controls/hf_only.csv` | HF-only and multi-fidelity outcomes at the table windows | Supplementary Table 8 |
| `controls/hopv15_picks.csv` | Initial observations and individual queries | Supplementary Table 11 |

`make_tables.py` also writes TL-base comparisons (Supplementary Table 9).
`make_attainment.py` writes mean attainment and deficits (Supplementary Table 10).
Surrogate configuration, architecture, benchmark and synthetic-scenario tables
are specified by `src/`, `runners/`, `benchmarks/` and `data/README.md`.

## Fields

In `trajectories.json`, `pools[pool]` contains the scoring budget `B`, objective
`range`, candidate count `n_pool`, and regret thresholds `thr` for the top 5%,
2%, 1% and 0.5% of candidates. `series[model][seed]` stores the initial observation
and subsequent improvement points as `pts = [[budget, regret], ...]`; `hz` is
the observed trajectory horizon. Regret is constant between improvement points.
`abbr`, `gp` and `tl` provide the model labels and families.

Acquisition CSVs use the same change-point representation. `window` is the
scoring budget, `budget` is the cumulative query cost, and `regret` is the
best-so-far HF regret. `acquisition` identifies the LF selection policy.

In the ablation tables, `t0` is the cost of the initial design. `attainment`
is the mean, over four targets, of the fraction of the remaining budget after
first reaching a target. `final_regret` is evaluated at budget 30.
`head_phi` selects learned or random features; `head_scalar` and `head_offset`
indicate the scalar LF input and additive LF offset. In the feature table,
`1_learned`, `2_randphi`, `4_nophi` and `3_hfonly` denote learned features,
random features, the scalar head and the HF-only network.

The HF-only table uses `attain` for attainment and `final` for final regret.
`hz` is the observed horizon and `t_init` is the initial cost. Its scoring budgets
are 50 for Branin and FreeSolv, 10 for Park, and 30 elsewhere.

Grid `n_b` is the bulk training count and `dem` is the demoted fraction of the
near-optimal basin. `r2` is squared LF–HF correlation; `top10` and `top50` are
ranking overlap fractions. `sigma` scales added global noise; `opt` sets a
basin promotion/demotion fraction or an explicit `targetK` top-ten overlap.
Grid `auc` is the budget-averaged regret.

HOPV15 query `idx` values are zero-based rows of `data/hopv15.csv`. `fid` is
`L0`/`H0` for the initial design and `L`/`H` for later queries. `step` identifies
the query iteration and `budget` records cumulative cost.

`expected/` contains numerical outputs for the paper figures. Running
`python reproduce.py` writes corresponding tables to `figures/out/`.
