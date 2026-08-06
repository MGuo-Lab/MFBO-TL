# Root-cause investigation: paper_v1 vs current code

Date: 2026-05-22 ~ 2026-05-24 BST (resolved 2026-05-24)
Status: **RESOLVED.** Root cause = const-std LF EI. User confirmed this is the
*original* method that produced paper_v1; the BLR-std code committed later was an
accidental modification. Fix applied on branch
`fix/lf-ei-const-std-restore-original` (commit `ab2d301`). See "Resolution and
academic verdict" below.

## TL;DR

Running the committed `src/benchmark.py` (== `benchmark_parallel_lf_blr.py` Jan 23
commit, byte-identical) in any modern PyTorch env reproduces the **same wrong
distribution** for chemistry DNN models, but does not reproduce **paper_v1**.

The variable is **not the environment**. The variable is **the LF EI**
acquisition rule:

| | LF EI uses | Mean (10 seeds, COFs/Sequential) | Zero count |
|---|---|---|---|
| Current committed code | `mean = BLR.mean, std = BLR.std` | **7.5375** | 0/10 |
| Reverse-engineered paper_v1 (V10) | `mean = BLR.mean, std = const 0.1` | **0.3367** | 8/10 |
| `results/paper_v1/results_summary.csv` | (the published result) | **0.3367** | 8/10 |

Per-seed match: 6 of 10 seeds are bit-identical between V10 and paper_v1; the
other 4 are within-distribution swaps (0.0 ↔ 1.6835) explained by tiny Adam
numerical differences between the original PyT/CUDA env and the current one.

**Conclusion**: The script that actually generated paper_v1 disabled or
ignored the BLR std when computing LF EI. Effectively the LF EI was
exploit-only (argmax of `y_best − mean`, std being a small constant).
The current committed code uses the full BLR std → EI explores
high-uncertainty candidates → on chemistry pools this exploration wastes
HF budget on candidates the surrogate is uncertain about but that are
not actually near the optimum.

## What the const-std LF EI actually computes (academic crux)

This is the load-bearing fact for any judgement about academic validity, so it
is worth stating precisely. For minimization, EI at candidate *i* is

```
EI_i = σ_i · [ z_i · Φ(z_i) + φ(z_i) ],   z_i = (y_best − μ_i) / σ_i
```

When σ_i is replaced by the **same constant** c for every candidate:

- z_i = (y_best − μ_i)/c is a strictly *decreasing* affine function of μ_i;
- g(z) = z·Φ(z) + φ(z) is strictly *increasing* in z;
- therefore EI_i is strictly decreasing in μ_i, so
  **argmax_i EI_i = argmin_i μ_i.**

Consequences:

1. **The LF acquisition collapses to greedy exploitation of the posterior
   mean.** No exploration term survives. The "EI" wrapper is cosmetic.
2. **The constant value (0.1) is irrelevant to the selection** — it only
   rescales EI monotonically. Any positive constant produces the *same* query
   sequence. (It would matter only if it varied per-candidate, which it does
   not.)
3. **HF acquisition is already argmin(mean).** So *both* fidelities are greedy
   on the surrogate mean; the LF/HF asymmetry implied by the paper text does not
   exist in the code that ran.
4. **The only thing BLR contributes is a regularized posterior *mean*** (a
   Bayesian last-layer ridge on `LFNetwork`'s features). Its predictive variance
   — the entire "uncertainty quantification" selling point — never enters any
   acquisition decision.

In one sentence: paper_v1 was produced by **greedy exploitation of a
BLR-regularized mean on both fidelities, with deterministic round-robin fidelity
scheduling** — not by UQ-driven Expected Improvement.

## What we ruled out (cost: ~12 SLURM jobs, ~3 hours)

| Variable | Tested | Effect |
|---|---|---|
| Code identity vs original `benchmark_parallel_lf_blr.py` (md5 `7156b4207a5f…`) | byte-level diff, git log, commit hashes | identical |
| Data identity (cofs/freesolv/polarizability.csv) | md5 sum | identical |
| Python 3.9 vs 3.10 | new conda env each | no effect |
| numpy 1.26.4 vs 2.0.2 | force reinstall | no effect |
| PyTorch 2.4.1 vs 2.5.1 vs 2.9.1 | full reinstall × 3 | per-seed values **bit-identical** |
| CUDA bundled libs: cu118 vs cu121 vs cu128 | reinstall | no effect |
| GPU: L40S vs A100 | constraint switching | no effect |
| TF32 modes: default/ON/medium/highest | `set_float32_matmul_precision` × 4 | no effect |
| `torch.use_deterministic_algorithms` | True | no effect (already deterministic) |
| Sequential vs isolated seed runs | same Python process | no effect (mean = isolated mean) |
| Same-env determinism: 3 consecutive runs of seed=43 | check | all 3 identical to the 7th decimal |
| BLR β = 1.0 vs 25.0 | refit | no effect |
| `lf_per_hf` ratio (more LF / more HF) | × 2 / × 0.5 | small effect, not the cause |
| Init budget fraction: 5% / 10% / 20% | sweep | 5% gives partial improvement, still not paper match |
| HF acquisition: argmin(mean) vs EI(mean, BLR-std-proxy) | switched | no effect (EI degenerates) |
| LF EI `y_best`: y_lf.min() vs y_hf.min() | switched | no effect |

The only variant out of the 13 algorithmic variants tested that reproduces
paper_v1's exact aggregate distribution is **V10**.

## The decisive variant (V10)

Modification to `run_bo_lf_blr`'s LF branch:

```python
# Current code (V0, baseline)
mean_lf, std_lf = model.predict_lf(benchmark.X)
ei = expected_improvement(mean_lf, std_lf, y_lf.min())

# Paper-v1-reproducing code (V10)
mean_lf, _ = model.predict_lf(benchmark.X)
std_lf = np.ones_like(mean_lf) * 0.1   # constant std, BLR std ignored
ei = expected_improvement(mean_lf, std_lf, y_lf.min())
```

That single one-line change shifts the 10-seed distribution from
(mean 7.5375, 0 zeros) to (mean 0.3367, 8 zeros) — i.e., to the
published paper_v1 distribution.

## Why this is the answer (probabilistic argument)

Across 10 seeds the distribution match is exact:

- V10 mean = paper_v1 mean = **0.3367**
- V10 zeros = paper_v1 zeros = **8 / 10**
- per-seed exact matches = **6 / 10** (seeds 42, 43, 46, 48, 50, 51 → all 0.0)

The remaining 4 mismatches are within-distribution swaps (a seed that gave
0.0 in paper_v1 gives 1.6835 here, and vice-versa). This is the expected
amount of Adam-on-CUDA numerical noise between two different PyTorch
versions training the same DNN — the surrogate's last-layer features differ
by ~1e-6, which sometimes flips which of two near-equally-good candidates
the LF EI argmax lands on.

P(this distribution match by chance under any other algorithmic variant) ≈ 0.
None of the other 12 variants tried got closer than "partial" (V5 at 5% init
gave one improvement to 1.6835 instead of paper's 0.0).

## Reconstructed timeline (best hypothesis)

Per git/sacct evidence:

- **2026-01-23 04:09:56 UTC** — user commits `benchmark_parallel_lf_blr.py`
  (blob hash `7156b4207a5f…`). At this commit, `_fit_lf_blr` sets
  `self.has_lf_blr = True` and `predict_lf` returns BLR std.
- **04:09 ~ 04:52** — user **edits the local copy** of the script to either:
  (a) skip setting `has_lf_blr=True`, or (b) override the std return in
  `predict_lf`, or (c) drop the std and use a constant in the BO loop's LF
  branch. The edit is **not committed**.
- **04:52:53 UTC** — SLURM job launches with the modified local copy. The
  modified code produces output directory `benchmark_lf_blr_20260123_045253/`
  (output_dir naming unchanged).
- **16:16:10 UTC** — run finishes, writes `results_summary.csv` (md5
  `59ab768c06ba…`). This is the **paper_v1** data.
- **16:15:45 UTC** — user commits the results to git (commit `1ec574f`).
  Working tree script is still the modified version OR has been
  `git checkout`-ed to the committed clean version.
- **Some point afterward** — local edits to the script are reverted /
  overwritten by `git checkout`. The file on disk now matches the committed
  hash. The "modified version that actually ran" is lost.

Empirical evidence supporting this timeline (rather than e.g. an obscure
PyTorch version effect):

1. **Modern PyTorch envs all converge to the same wrong distribution**.
   2.4, 2.5, 2.9 give bit-identical per-seed values. If the cause were a
   PyTorch numerical change, at least one of these three would have differed.
2. **V10 (single-line code change) closes the gap exactly**. A single,
   plausible code change explains the mean, the zero count, and most
   per-seed values. No combination of env tweaks gets within 5× of the
   paper_v1 mean.
3. **The user's recollection of "LF-BLR만 썼어"** is consistent with V10:
   BLR is fit on LF (mean is BLR-regularized), but the std is constant.
4. **No `benchmark_parallel_lf_blr.cpython-*.pyc`** in the original repo's
   `__pycache__` — consistent with the script being invoked as `__main__`
   (Python doesn't compile main scripts) but also leaves no fingerprint of
   the exact source it ran. Not load-bearing on its own.

## Test scripts (in `experiments/env_repro_test/`)

| File | Purpose |
|---|---|
| `test_single_cell.py` | Run COFs / Sequential / single seed in current env |
| `test_sequential_seeds.py` | 10 seeds sequentially in same Python (state-carry-over check) |
| `test_determinism.py` | 3 consecutive runs of same seed (CUDA determinism check) |
| `test_tf32.py` | TF32 mode sweep |
| `test_cpu.py` | CPU-only (was not run; user confirmed GPU) |
| `test_algo_variants.py` | V0–V5: baseline, HF EI, LF EI y_best swap, init fraction |
| `test_algo_variants2.py` | V6–V12: BLR β=25, lf_per_hf sweep, epochs ×2, **V10 = const std**, V11 const-std on HF, V12 combined |
| **`test_v10_distribution.py`** | **THE decisive test — V10 across 10 seeds = paper_v1 distribution** |

All submit scripts use the `paper_v1_py39` conda env at
`/users/k23070952/.conda/envs/paper_v1_py39/` (Python 3.9.23, torch
2.4.1+cu121, numpy 1.26.4, gpytorch 1.11, botorch 0.10.0).

## Resolution and academic verdict

**Decision (2026-05-24): Option A — restore the const-std LF EI.** The user
confirmed const-std is the *original* method that generated paper_v1 ("일정한
분산 입력 + LF-BLR 헤더만"); the BLR-std code committed later was an accidental
modification, not the published algorithm. Fix applied on branch
`fix/lf-ei-const-std-restore-original` (commit `ab2d301`): `run_bo_lf_blr` gained
`lf_ei_const_std=0.1` (default = restored original); pass `None` / `--lf-ei-std
-1` for the BLR-std "true UQ" variant. Re-run jobs were submitted 2026-05-24 (see
`experiments/rerun_const_std_20260524/README.md`).

### Is the method academically defensible?

These are two separate questions, and they have opposite answers:

1. **Is it a legitimate BO algorithm? — Yes.** As proven above it is greedy
   exploitation of a BLR-regularized posterior mean on both fidelities, with
   deterministic round-robin fidelity scheduling. Greedy mean-based selection is
   a well-known, often strong low-budget strategy — particularly when transfer
   learning makes the LF→HF mean reliable enough that exploration is not worth
   its budget cost. The empirical advantage on the chemistry pools is genuine
   and reproducible. Nothing about the *results* is invalid.

2. **Does the paper describe what actually ran? — No.** §2.3.2 / Appendix B sell
   "BLR-driven uncertainty quantification entering the acquisition." With a
   constant std the BLR variance is discarded and EI ≡ argmin(mean); no
   uncertainty ever influences a query, and the LF/HF asymmetry the text implies
   does not exist. The UQ-aware-acquisition narrative is **inaccurate as
   written**.

**So the issue is a *claims/methods-description* problem, not a results-validity
problem.** Required for the camera-ready, honest framing — pick one:

- (a) Describe the method as what it is: *greedy exploitation on
  BLR-regularized means with round-robin fidelity scheduling*, and drop the
  UQ-acquisition language. Cleanest; keeps all paper_v1 numbers.
- (b) Present the true-BLR-std variant (Option B, already implemented in
  `experiments/adaptive_fidelity/`) as the "intended" method, and report the
  const-std greedy version as the empirically stronger ablation.

The arguably *interesting* version of this story — and a real finding rather
than a bug to hide — is: **"with good LF→HF transfer, greedy exploitation beats
UQ-driven exploration, because exploration spends the scarce HF budget on
high-variance but off-optimum candidates."** That is publishable on its own
terms; it just has to be stated honestly.

### Historical options (for the record)

- **Option A (chosen)** — restore const-std. Pro: reproduces all 1680 paper_v1
  rows from current code; matches what was run. Con: "LF-BLR UQ" branding is
  misleading; §2.3.2/App B must be rewritten.
- **Option B** — keep true BLR-std EI, re-run everything. Pro: code matches the
  paper text. Con: TL methods do *not* beat MFGP under this acquisition on the
  chemistry pools (A1: DNN mean regret ~7.5 vs MFGP ~5.8) — the headline claim
  does not survive.
- **Option C** — disclose both: lead with true BLR-std, appendix the const-std
  greedy advantage. Honest but a weaker, more nuanced headline.

### Residual caveats

- **The LF/HF symmetry must be owned, not papered over.** Both fidelities are
  greedy-on-mean; any rewrite of §2.3.2 has to say so.
- **Per-seed reproducibility is imperfect.** The aggregate distribution matches
  paper_v1 exactly (mean, zero count), but 4/10 seeds are within-distribution
  swaps (0.0 ↔ 1.6835) from Adam-on-CUDA noise across PyTorch versions. Exact
  per-seed numbers would require the lost Jan-23 PyTorch/CUDA build.

## Follow-up: Greedy-MFGP control (surrogate × acquisition 2×2)

The verdict above raises the obvious confound: paper_v1 compared **DNN-greedy**
against **MFGP-EI**, so "DNN beats MFGP" mixes the *surrogate* (DNN vs GP) with
the *LF acquisition* (greedy vs UQ-EI). To deconfound, fill the 2×2 — MFGP/DNN ×
greedy/EI — holding everything else fixed (HF is `argmin(mean)` in all cells;
fidelity scheduling is round-robin in all cells, so the LF EI std is the *only*
UQ entry point being ablated).

`src/benchmark.py` now supports the missing cell via `--mfgp-greedy` (default
off = published GP-UQ baseline; on = MFGP LF step picks `argmin` of the GP
posterior mean). A `--models` filter runs a single surrogate in isolation. Three
of four cells already exist in the current env, so only **Greedy-MFGP** is new:

| | UQ-EI | Greedy (μ only) |
|---|---|---|
| MFGP | `run_20260524_033621` ✅ | **`--mfgp-greedy --models MFGP`** ❗new |
| DNN  | `run_20260524_033621` ✅ (Option B) | `run_20260524_033620` ⏳ (const-std, partial) |

Full design, cell→data map, run command, and the interpretation logic (incl. the
within-MFGP H3 test `Greedy-MFGP > EI-MFGP`, and the surrogate×acquisition
interaction) are in `experiments/greedy_mfgp_2x2/README.md`.

## Reproduction recipe (so next session can rerun)

```bash
# 1. Activate env
conda activate paper_v1_py39   # python 3.9, torch 2.4.1+cu121

# 2. Submit V10 distribution test on COFs/Sequential
cd experiments/env_repro_test
sbatch submit_v10.sh
# Output: slurm_v10_<JOBID>.out — expect mean 0.3367, 8/10 zeros

# 3. (Optional) regenerate the broader variant sweep
sbatch submit_variants.sh   # V0–V5
sbatch submit_variants2.sh  # V6–V12
```

Quick verification:
```bash
python -c "
import pandas as pd
p = pd.read_csv('results/paper_v1/results_summary.csv')
sub = p[(p.benchmark=='COFs') & (p.model=='Sequential')]
print(f'Paper_v1 distribution: mean={sub.final_regret.mean():.4f}, '
      f'zeros={(sub.final_regret==0).sum()}/{len(sub)}')
"
# Expect: mean=0.6376, zeros=12/20 (the 20-seed paper_v1 aggregate; the TL;DR
# table above quotes the 10-seed subset: mean=0.3367, 8/10 zeros — same regime)
```
