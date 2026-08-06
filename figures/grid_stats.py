#!/usr/bin/env python
"""Shared statistics for the 126-cell controlled Polariz grid (FDR re-analysis, 2026-07-24).

Replaces the multiplicity conventions used by make_fig1c_family_split.py and
make_figures.py (fig1_controlled_grid) up to 2026-07-13:

  (old) per-cell paired Wilcoxon with no multiple-testing correction; a display
        bin was starred when at least half of its cells had p < 0.05 (majority
        vote). 74 of the 97 non-empty bins hold a single cell, 17 hold two and
        6 hold three, so the "majority" was usually a single cell's raw test.

  (new) three levels, each corrected within one map, where a map is one
        (metric, comparison) pair:
    1. Condition level: per-cell paired two-sided Wilcoxon signed-rank on the
       10 common seeds, then Benjamini-Hochberg FDR across the map's 126 cells.
    2. Bin level (display stars): ONE test per non-empty display bin, a
       stratified signed-rank test with the bin's cells as strata and the
       paired per-seed differences within each cell as units (a paired-data
       analogue of van Elteren's stratified rank test; the two-sample original
       would break the seed pairing). The p-value comes from sign-flip
       randomization of the combined signed-rank statistic (exact enumeration
       for up to 2^16 sign patterns, otherwise 200k Monte-Carlo draws with a
       per-bin deterministic seed), followed by BH across the map's non-empty
       bins. In a single-cell bin the test reduces to that cell's Wilcoxon.
    3. Map level: one global two-sided Wilcoxon (plus an exact sign test)
       across the per-cell advantages, cells as units, so the map-wide
       direction is tested once without seed pseudo-replication.

  Untestable comparisons (identical per-seed values, mostly cells where both
  families sit on the regret floor) enter their family as p = 1, keeping the
  denominator fixed at the family size.

Conventions matched to the manuscript: two-sided tests, alpha = 0.05, zero
differences dropped (scipy `wilcoxon` default), advantage = mean(second) -
mean(first); both metrics are lower-is-better, so advantage > 0 favours the
first-named family. Cells share the seed set (42-51), so cell-level units are
exchangeable rather than fully independent; this is the same convention as
treating benchmarks as units in rank-based model comparisons.

Login-node safe: reads cached CSVs only. `python grid_stats.py` runs the
self-test (BH vs statsmodels and scipy, stratified test vs scipy exact
Wilcoxon on singleton strata, reproduction of the pre-correction headline
fractions) and prints the headline numbers.
"""
import glob
import os
import warnings
import zlib

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, wilcoxon

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
GRID = os.path.join(REPO, "results", "grid")

ALPHA = 0.05
N_MC = 200_000
ENUM_MAX_N = 16

# fig1c_family_split_grid: three families, three pairwise comparisons
FAMILIES_SPLIT = {
    "TL": ["Progressive", "KnowledgeDistillation", "PseudoLabeling"],
    "MFGP baseline": ["MFGP"],
    "MFGP variants": ["DKL", "SparseMFGP"],
}
COMPARISONS_SPLIT = [
    ("TL", "MFGP baseline"),
    ("TL", "MFGP variants"),
    ("MFGP variants", "MFGP baseline"),
]

# fig1_controlled_grid: pooled best-of-3 TL vs best-of-3 GP
FAMILIES_POOLED = {
    "TL": ["Progressive", "KnowledgeDistillation", "PseudoLabeling"],
    "GP": ["MFGP", "DKL", "SparseMFGP"],
}
COMPARISONS_POOLED = [("TL", "GP")]

METRICS = ["final_regret", "auc"]
R2B = np.round(np.arange(0.1, 0.91, 0.1), 2)
T10 = np.round(np.arange(0.0, 1.01, 0.1), 2)


# ----------------------------------------------------------------------
# loading and per-cell tests (selection logic verbatim from the figure scripts)
# ----------------------------------------------------------------------
def load_cells():
    man = pd.read_csv(os.path.join(GRID, "grid_manifest.csv")).set_index("cell_id")
    cells = {}
    for f in sorted(glob.glob(os.path.join(GRID, "cells", "summary_cell_*.csv"))):
        cid = os.path.basename(f).replace("summary_", "").replace(".csv", "")
        if cid in man.index:
            cells[cid] = pd.read_csv(f)
    if len(cells) != len(man):
        raise RuntimeError(f"expected {len(man)} cells, found {len(cells)}")
    return man, cells


def _cell_pvalue(a1, a2):
    """Per-cell paired Wilcoxon exactly as in make_fig1c_family_split.py."""
    if len(a1) >= 6 and not np.allclose(a1, a2):
        try:
            return float(wilcoxon(a1, a2).pvalue)
        except Exception:
            pass
    return np.nan


def per_cell(man, cells, families, comparisons):
    """One row per (cell, metric, comparison): advantage, raw p, per-seed diffs."""
    recs = []
    for cid in sorted(cells):
        d = cells[cid]
        for metric in METRICS:
            vals = {m: dict(zip(g.seed, g[metric])) for m, g in d.groupby("model")}
            best = {}
            ok = True
            for fam, members in families.items():
                have = [m for m in members if m in vals and len(vals[m])]
                if not have:
                    ok = False
                    break
                means = {m: np.mean(list(vals[m].values())) for m in have}
                b = min(means, key=means.get)
                best[fam] = (b, vals[b], means[b])
            if not ok:
                continue
            for first, second in comparisons:
                _, v1, m1 = best[first]
                _, v2, m2 = best[second]
                seeds = sorted(set(v1) & set(v2))
                a1 = np.array([v1[s] for s in seeds])
                a2 = np.array([v2[s] for s in seeds])
                recs.append(dict(
                    cell=cid, r2=man.loc[cid, "r2"], top10=man.loc[cid, "top10"],
                    metric=metric, comparison=f"{first}|{second}",
                    n_seeds=len(seeds), adv=m2 - m1, p=_cell_pvalue(a1, a2),
                    diffs=a2 - a1))
    df = pd.DataFrame(recs)
    df["r2bin"] = R2B[np.argmin(np.abs(df["r2"].values[:, None] - R2B[None, :]), axis=1)]
    df["t10bin"] = np.round(df["top10"], 2)
    n_maps = df.groupby(["metric", "comparison"]).size()
    if not (n_maps == len(man)).all():
        raise RuntimeError(f"incomplete maps:\n{n_maps}")
    return df


# ----------------------------------------------------------------------
# Benjamini-Hochberg
# ----------------------------------------------------------------------
def bh_adjust(p):
    """BH-adjusted p-values (step-up). Self-tested against statsmodels/scipy."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def add_condition_bh(percell):
    """BH across the cells of each map; untestable cells enter as p = 1."""
    percell = percell.copy()
    percell["p_filled"] = percell["p"].fillna(1.0)
    percell["p_bh"] = np.nan
    for _, sub in percell.groupby(["metric", "comparison"]):
        percell.loc[sub.index, "p_bh"] = bh_adjust(sub["p_filled"].values)
    percell["sig_raw"] = percell["p_filled"] < ALPHA
    percell["sig_bh"] = percell["p_bh"] <= ALPHA
    return percell


# ----------------------------------------------------------------------
# stratified signed-rank test per display bin
# ----------------------------------------------------------------------
def _bin_rng(metric, comparison, t10bin, r2bin):
    key = f"{metric}|{comparison}|{t10bin:.2f}|{r2bin:.2f}"
    return np.random.default_rng(zlib.crc32(key.encode()))


def stratified_signflip_p(diff_list, rng, enum_max_n=ENUM_MAX_N, n_mc=N_MC):
    """Two-sided p for the combined signed-rank statistic across strata.

    diff_list: one array of paired differences per stratum (cell). Zeros are
    dropped within each stratum; |d| are ranked within the stratum (average
    ranks on ties) and the statistic is U = sum_c (W+_c - E[W+_c]). Under H0
    each remaining pair's sign is an independent fair coin, so U is
    randomized by sign flips: exact enumeration when the total number of
    nonzero pairs is <= enum_max_n, else n_mc Monte-Carlo draws (add-one
    estimator). Returns NaN when no nonzero difference exists.
    """
    ranks = []
    signs = []
    for d in diff_list:
        d = np.asarray(d, dtype=float)
        d = d[d != 0]
        if len(d):
            ranks.append(rankdata(np.abs(d)))
            signs.append((d > 0).astype(float))
    if not ranks:
        return np.nan
    all_r = np.concatenate(ranks)
    all_s = np.concatenate(signs)
    e = all_r.sum() / 2.0
    u_obs = float(all_r @ all_s - e)
    n = len(all_r)
    if n <= enum_max_n:
        pats = np.arange(2 ** n, dtype=np.uint32)
        bits = ((pats[:, None] >> np.arange(n)) & 1).astype(float)
        u = bits @ all_r - e
        return float((np.abs(u) >= abs(u_obs) - 1e-12).mean())
    bits = rng.integers(0, 2, size=(n_mc, n)).astype(float)
    u = bits @ all_r - e
    return float((1 + (np.abs(u) >= abs(u_obs) - 1e-12).sum()) / (n_mc + 1))


def per_bin(percell):
    """One stratified test per non-empty bin, BH across each map's bins.

    star_old reproduces the pre-2026-07-24 majority rule (>= half of the bin's
    cells with raw p < 0.05, NaN counting as not significant); star_new is the
    BH-corrected stratified test.
    """
    recs = []
    for (metric, comp), sub in percell.groupby(["metric", "comparison"], sort=True):
        for (t, rb), bg in sub.groupby(["t10bin", "r2bin"], sort=True):
            rng = _bin_rng(metric, comp, t, rb)
            p = stratified_signflip_p(list(bg["diffs"]), rng)
            recs.append(dict(
                metric=metric, comparison=comp, t10bin=t, r2bin=rb,
                n_cells=len(bg), adv_mean=bg["adv"].mean(),
                frac_raw_sig=(bg["p"] < ALPHA).mean(), p_strat=p))
    pb = pd.DataFrame(recs)
    pb["p_strat_bh"] = np.nan
    for _, sub in pb.groupby(["metric", "comparison"]):
        pb.loc[sub.index, "p_strat_bh"] = bh_adjust(sub["p_strat"].fillna(1.0).values)
    pb["star_old"] = pb["frac_raw_sig"] >= 0.5
    pb["star_new"] = pb["p_strat_bh"] <= ALPHA
    return pb


# ----------------------------------------------------------------------
# one global test per map (cells as units)
# ----------------------------------------------------------------------
def map_global(percell):
    recs = []
    for (metric, comp), sub in percell.groupby(["metric", "comparison"], sort=True):
        adv = sub["adv"].values
        nz = adv[adv != 0]
        npos = int((adv > 0).sum())
        nneg = int((adv < 0).sum())
        wp = float(wilcoxon(nz).pvalue) if len(nz) >= 6 else np.nan
        sp = float(binomtest(npos, npos + nneg, 0.5).pvalue) if npos + nneg else np.nan
        recs.append(dict(
            metric=metric, comparison=comp, n_cells=len(adv), n_pos=npos,
            n_zero=int((adv == 0).sum()), n_neg=nneg,
            median_adv=float(np.median(adv)), wilcoxon_p=wp, sign_p=sp))
    return pd.DataFrame(recs)


# ----------------------------------------------------------------------
# convenience: full pipeline for one family spec
# ----------------------------------------------------------------------
def compute(spec="split"):
    """spec in {'split', 'pooled'} -> (percell, perbin, global_) DataFrames."""
    families, comparisons = {
        "split": (FAMILIES_SPLIT, COMPARISONS_SPLIT),
        "pooled": (FAMILIES_POOLED, COMPARISONS_POOLED),
    }[spec]
    man, cells = load_cells()
    percell = add_condition_bh(per_cell(man, cells, families, comparisons))
    perbin = per_bin(percell)
    global_ = map_global(percell)
    return percell, perbin, global_


def star_lookup(perbin):
    """{(metric, comparison, t10bin, r2bin): star_new} for the figure scripts."""
    return {(r.metric, r.comparison, r.t10bin, r.r2bin): bool(r.star_new)
            for r in perbin.itertuples()}


# ----------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------
def _selftest():
    rng = np.random.default_rng(0)

    # 1. BH vs statsmodels and scipy
    from statsmodels.stats.multitest import multipletests
    from scipy.stats import false_discovery_control
    for trial in range(5):
        p = rng.uniform(0, 1, size=rng.integers(5, 200))
        mine = bh_adjust(p)
        rej_sm, adj_sm, _, _ = multipletests(p, alpha=ALPHA, method="fdr_bh")
        adj_sp = false_discovery_control(p, method="bh")
        assert np.allclose(mine, adj_sm) and np.allclose(mine, adj_sp)
        assert ((mine <= ALPHA) == rej_sm).all()
    print("selftest 1  BH matches statsmodels + scipy          OK")

    # 2. stratified test on a single stratum == scipy exact two-sided Wilcoxon
    for trial in range(20):
        d = rng.normal(0.3, 1.0, size=10)
        p_mine = stratified_signflip_p([d], rng)
        p_scipy = float(wilcoxon(d, method="exact").pvalue)
        assert abs(p_mine - p_scipy) < 1e-12, (p_mine, p_scipy)
    print("selftest 2  singleton stratum == scipy exact        OK")

    # 3. MC branch consistent with enumeration on the same data
    for trial in range(3):
        ds = [rng.normal(0.5, 1.0, size=8), rng.normal(0.5, 1.0, size=8)]
        p_en = stratified_signflip_p(ds, rng, enum_max_n=16, n_mc=N_MC)
        p_mc = stratified_signflip_p(ds, np.random.default_rng(1), enum_max_n=0, n_mc=N_MC)
        assert abs(p_en - p_mc) < 0.01, (p_en, p_mc)
    print("selftest 3  MC agrees with enumeration (<0.01)      OK")

    # 4. reproduce the pre-correction headline fractions from raw data
    percell, perbin, global_ = compute("split")
    key = (percell.metric == "auc")
    f1 = percell[key & (percell.comparison == "TL|MFGP baseline")]
    f2 = percell[key & (percell.comparison == "TL|MFGP variants")]
    assert f1.sig_raw.sum() == 122 and len(f1) == 126
    assert f2.sig_raw.sum() == 105 and len(f2) == 126
    assert (f1.adv > 0).all() and (f2.adv > 0).all()
    print("selftest 4  reproduces raw 122/126 and 105/126      OK")
    return percell, perbin, global_


if __name__ == "__main__":
    percell, perbin, global_ = _selftest()
    print("\nheadline (split spec):")
    for (metric, comp), sub in percell.groupby(["metric", "comparison"]):
        pb = perbin[(perbin.metric == metric) & (perbin.comparison == comp)]
        g = global_[(global_.metric == metric) & (global_.comparison == comp)].iloc[0]
        print(f"  {metric:12s} {comp:30s} pos {int((sub.adv > 0).sum()):3d}/126  "
              f"raw {100 * sub.sig_raw.mean():5.1f}%  BH {100 * sub.sig_bh.mean():5.1f}%  "
              f"stars {int(pb.star_old.sum()):3d} -> {int(pb.star_new.sum()):3d}  "
              f"global W p={g.wilcoxon_p:.2e} sign p={g.sign_p:.2e}")
