'Gaussian calibration metrics.'
from __future__ import annotations
import numpy as np

def ece_gaussian(y_true, mean, std, n_bins=10):
    from scipy.stats import norm
    levels = np.linspace(0.05, 0.95, n_bins)
    abs_err = 0.0
    for lvl in levels:
        z = norm.ppf(0.5 + lvl / 2.0)
        in_band = np.abs(y_true - mean) <= z * std
        abs_err += abs(in_band.mean() - lvl)
    return abs_err / len(levels)

def nll_gaussian(y_true, mean, std):
    std = np.maximum(std, 1e-06)
    return float(np.mean(0.5 * np.log(2 * np.pi * std ** 2) + 0.5 * ((y_true - mean) / std) ** 2))

def calibration_metrics(model, benchmark, result):
    all_idx = np.arange(benchmark.n_candidates)
    sampled = np.array(list(result['lf_indices'] | result['hf_indices']))
    held = np.setdiff1d(all_idx, sampled, assume_unique=False)
    if len(held) < 5:
        return {}
    X_held = benchmark.X[held]
    y_lf_true = benchmark.y_lf[held]
    y_hf_true = benchmark.y_hf[held]
    (mean_lf, std_lf) = model.predict_lf(X_held)
    rmse_lf = float(np.sqrt(np.mean((y_lf_true - mean_lf) ** 2)))
    nll_lf = nll_gaussian(y_lf_true, mean_lf, std_lf)
    ece_lf = ece_gaussian(y_lf_true, mean_lf, std_lf)
    sharp_lf = float(np.mean(std_lf))
    (mean_hf, _) = model.predict(X_held)
    rmse_hf = float(np.sqrt(np.mean((y_hf_true - mean_hf) ** 2)))
    return dict(lf_rmse=rmse_lf, lf_nll=nll_lf, lf_ece=ece_lf, lf_sharpness=sharp_lf, hf_rmse=rmse_hf, n_held=int(len(held)))
