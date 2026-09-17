# Experiment settings

The main comparison uses seeds 42–81, budget 50 on the four synthetic pools and
budget 30 on the nine molecular and materials pools. Initial designs use
farthest-point sampling. For the initial-design budget `B_init`, the counts are
`n_L = max(2, int(0.05 * B_init / rho))` and
`n_H = max(2, int(0.05 * B_init))`. LF and HF initial indices are disjoint.

| Pools | `B_init` |
|---|---:|
| Branin, Park, FreeSolv | 50 |
| COFs, Polarizability, HOPV15 | 30 |
| Matbench-Gap, ExptGap-PBE, three elasticity pools | 20 |
| Controlled grid | 10 |

Each HF observation costs one unit and each LF observation costs `rho`.
The schedule makes `floor(1/rho)` LF queries per HF query. Observations are
masked separately at each fidelity. Every iteration fits a new surrogate.
HF selection is greedy throughout. LF selection uses the HF mean for TL and
HF expected improvement for GP. TL predictions use the BLR residual head in
`runners/run_tl.py`.

## Acquisition controls

Supplementary Figs. 1 and 2 use seeds 42–61 on Branin-Fav, Branin-Unfav,
Park-Fav, Park-Unfav, COFs, FreeSolv, Polarizability, HOPV15 and Matbench-Gap.
Budgets are 50 on synthetic pools, 30 on COFs, FreeSolv, Polarizability and
HOPV15, and 20 on Matbench-Gap. Set the Matbench budget with
`EXT_BUDGETS=Matbench-Gap=20`.

For each selected pool and seed set, the LF policy is configured as follows:

| Policy | TL environment setting | GP-base environment setting |
|---|---|---|
| Greedy | `LF_TARGET=hf_argmin` | `GP_ACQ=greedy` |
| Expected improvement | `LF_TARGET=hf_ei` | `GP_ACQ=ei` |
| Probability of improvement | `LF_TARGET=hf_pi` | — |
| Lower confidence bound | `LF_TARGET=hf_ucb` | — |
| Max-value entropy search | `LF_TARGET=hf_mes` | — |
| Thompson sampling | `LF_TARGET=hf_ts` | — |

For example:

```bash
LF_TARGET=hf_ei python runners/run_tl.py --benches HOPV15 --seeds {42..61}
GP_ACQ=greedy python runners/run_gp.py --benches HOPV15 --models MFGP --seeds {42..61}
```

## Feature and head-path ablations

These use all nine molecular and materials pools, budget 30 and seeds 42–81.
Run `runners/run_tl.py` with the chosen `--benches` and the following settings.

| Feature ablation | Setting |
|---|---|
| Learned representation | Default |
| Random representation | `HEAD_PHI=random` |
| Scalar head | `HEAD_PHI=none` |
| HF-only network | `HF_ONLY=1` |

For the TL-base head-path ablation, add `--models DNGOJoint`:

| Head configuration | Setting |
|---|---|
| Learned features, LF offset and input | Default |
| Learned features, LF offset only | `HEAD_SCALAR=0` |
| Learned features, LF input only | `HEAD_OFFSET=0` |
| Learned features, LF removed | `HEAD_REMOVE_GL=1` |
| Random features, LF offset and input | `HEAD_PHI=random` |
| Random features, LF removed | `HEAD_PHI=random HEAD_REMOVE_GL=1` |

## HF-only controls

```bash
SF_ARM=ei python runners/run_hf.py
SF_ARM=random python runners/run_hf.py
```

The supplementary table uses budget 50 for Branin and FreeSolv, 10 for Park,
and 30 for the remaining pools. Pass `--budget 50` for FreeSolv and `--budget 10`
for Park. The initial HF indices match the multi-fidelity runs. HF-only
controls spend their budget entirely on HF observations.

## Computational cost

```bash
python runners/profile_compute.py
```

Profiles use seeds 42–43 and training fractions 0.1, 0.3, 0.5, 0.7 and 1.0 on
the nine original pools. `figures/make_compute.py` interpolates these profiles
and sums fit and prediction FLOPs over the fidelity schedule. These profiles
use FreeSolv budget 50 and Matbench-Gap budget 20; the other budgets match the
main comparison.

## Output files

Runners write `summary_*.csv` and `traj_*.csv` under the selected output directory.
Trajectory columns include `benchmark`, `model`, `seed`, `budget` and `regret`.
The TL and GP runners also write `picks_*.csv`, with query `step`, `fid`, `idx`
and cumulative `budget`. `L0` and `H0` identify initial observations; `L` and
`H` identify subsequent queries. Indices refer to zero-based candidate-table
rows. Checkpoints store query and random-number-generator state. Reusing the
same output directory skips completed seeds.
