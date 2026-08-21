#!/usr/bin/env python
"""SI sensitivity check for the starring criterion of the 126-cell grid maps.

Fills SI stub (iv) of the manuscript's supplementary note on the family-split
grid: for each of the six (metric, comparison) maps it compares the OLD
uncorrected majority rule (a bin starred when at least half of its cells reach
raw p < 0.05) with the NEW corrected criterion (one stratified signed-rank
test per non-empty bin, BH-FDR at q = 0.05 across the map's bins; see
grid_stats.py for the full protocol).

Outputs (written next to this script):
  si_star_sensitivity.pdf/.png    3 comparison rows x 4 columns
                                  (final-regret: majority | corrected,
                                   anytime-AUC: majority | corrected),
                                  panel letters a-l; colours = binned mean
                                  advantage, scale shared per metric
  si_star_sensitivity_table.tex   supptab:star_sensitivity (booktabs source)

Login-node safe: reads cached CSVs only. Run: python3 make_si_star_sensitivity.py
"""
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grid_stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

R2B, T10 = grid_stats.R2B, grid_stats.T10
COMPARISONS = grid_stats.COMPARISONS_SPLIT
METRICS = [("final_regret", "Final-regret advantage"),
           ("auc", "Anytime-AUC advantage")]
CRITERIA = [("star_old", "majority rule (uncorrected)"),
            ("star_new", "stratified test, BH-FDR $q\\leq0.05$")]

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 10.5,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 130, "savefig.dpi": 300, "axes.linewidth": 0.8,
})


def heatmap(fig, ax, adv, star, vmax, small_cb=True):
    ax.set_facecolor("0.92")
    im = ax.imshow(adv.values, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto",
                   extent=[R2B[0] - .05, R2B[-1] + .05, T10[0] - .05, T10[-1] + .05])
    for i, t in enumerate(T10):
        for j, rr in enumerate(R2B):
            if star.values[i, j] == True:  # noqa: E712 (empty bins are NaN)
                ax.text(rr, t, "*", ha="center", va="center", fontsize=12,
                        fontweight="bold")
    if small_cb:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("advantage", fontsize=9.5)
        cb.ax.tick_params(labelsize=9.5)


def pivots(percell, perbin, metric, comp_key):
    sub = percell[(percell.metric == metric) & (percell.comparison == comp_key)]
    adv = (sub.groupby(["t10bin", "r2bin"])["adv"].mean()
           .unstack("r2bin").reindex(index=T10, columns=R2B))
    pb = perbin[(perbin.metric == metric) & (perbin.comparison == comp_key)]
    stars = {crit: (pb.pivot(index="t10bin", columns="r2bin", values=crit)
                    .reindex(index=T10, columns=R2B)) for crit, _ in CRITERIA}
    return adv, stars


def make_figure(percell, perbin):
    # 2026-08-18 print-size re-export: 17x10.8 in -> 13.7x8.7 in (~2.0x print width)
    fig, axes = plt.subplots(3, 4, figsize=(13.7, 8.7))
    letters = "abcdefghijkl"
    vmax = {metric: np.nanmax([np.abs(
        pivots(percell, perbin, metric, f"{a}|{b}")[0].values)
        for a, b in COMPARISONS]) for metric, _ in METRICS}
    for r, (first, second) in enumerate(COMPARISONS):
        key = f"{first}|{second}"
        col = 0
        for metric, mtitle in METRICS:
            adv, stars = pivots(percell, perbin, metric, key)
            for crit, clabel in CRITERIA:
                ax = axes[r][col]
                heatmap(fig, ax, adv, stars[crit], vmax[metric])
                ax.set_title(f"{mtitle}\n{clabel}", fontsize=11)
                if col == 0:
                    ax.set_ylabel(f"{first} vs {second}\n\ntop-10 optimum agreement",
                                  fontsize=10.5)
                else:
                    ax.set_ylabel("top-10 optimum agreement", fontsize=10.5)
                if r == 2:
                    ax.set_xlabel("global LF-HF $R^2$")
                ax.text(-0.14, 1.08, letters[r * 4 + col], transform=ax.transAxes,
                        fontweight="bold", fontsize=16, fontfamily="sans-serif",
                        ha="left", va="bottom")
                col += 1
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"si_star_sensitivity.{ext}"),
                    bbox_inches="tight", facecolor="white")
    print("wrote si_star_sensitivity.pdf / .png ->", OUT)
    plt.close(fig)


def _sci(p):
    if not np.isfinite(p):
        return "--"
    e = int(np.floor(np.log10(p)))
    return f"${p / 10 ** e:.1f}\\times10^{{{e}}}$"


def make_table(percell, perbin, global_):
    label = {"final_regret": "Final regret", "auc": "Anytime AUC"}
    clabel = {"TL|MFGP baseline": "TL vs baseline MFGP",
              "TL|MFGP variants": "TL vs MFGP variants",
              "MFGP variants|MFGP baseline": "MFGP variants vs baseline MFGP"}
    lines = [
        "\\begin{table}[h]",
        "\\caption{Sensitivity of the controlled-grid maps to the bin-starring",
        "criterion. For each comparison--metric map of the family-split grid (126",
        "cells, 10 seeds per cell): cells with a positive advantage of the",
        "first-named family; cells individually significant under the uncorrected",
        "per-cell Wilcoxon test ($p<0.05$) and under Benjamini--Hochberg (BH)",
        "false-discovery-rate correction across the map's 126 cells ($q\\le0.05$);",
        "display bins starred under the previous majority rule versus the corrected",
        "criterion (one stratified signed-rank test per bin, BH across the map's 97",
        "non-empty bins); and the map-level two-sided Wilcoxon $p$ across the 126",
        "per-cell advantages (conditions as units).}",
        "\\label{supptab:star_sensitivity}%",
        "\\begin{tabular}{@{}llccccc@{}}",
        "\\toprule",
        "Metric & Comparison & Positive & Raw $p<0.05$ & BH $q\\le0.05$ & "
        "Bins starred & Map-level $p$ \\\\",
        "\\midrule",
    ]
    for metric, _ in METRICS:
        for first, second in COMPARISONS:
            key = f"{first}|{second}"
            sub = percell[(percell.metric == metric) & (percell.comparison == key)]
            pb = perbin[(perbin.metric == metric) & (perbin.comparison == key)]
            gl = global_[(global_.metric == metric)
                         & (global_.comparison == key)].iloc[0]
            lines.append(
                f"{label[metric]} & {clabel[key]} & {int((sub.adv > 0).sum())}/126 & "
                f"{int(sub.sig_raw.sum())} ({100 * sub.sig_raw.mean():.0f}\\%) & "
                f"{int(sub.sig_bh.sum())} ({100 * sub.sig_bh.mean():.0f}\\%) & "
                f"{int(pb.star_old.sum())} $\\to$ {int(pb.star_new.sum())} & "
                f"{_sci(gl.wilcoxon_p)} \\\\")
        if metric == "final_regret":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    out = os.path.join(OUT, "si_star_sensitivity_table.tex")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    percell, perbin, global_ = grid_stats.compute("split")
    make_figure(percell, perbin)
    make_table(percell, perbin, global_)
