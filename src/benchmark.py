#!/usr/bin/env python
"""
Parallel Multi-Fidelity Benchmark with LF-BLR (Bayesian Linear Regression on LF Network)

Based on benchmark_parallel.py but with LF-BLR applied to all DNN models (except MFGP).

LF-BLR Strategy:
- BLR on LF network's last layer for uncertainty quantification
- LF selection: LF-BLR prediction + EI (exploration)
- HF selection: HF prediction + argmin (exploitation)

Usage:
    python benchmark_parallel_lf_blr.py --n-seeds 20 --n-workers 48
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube
from scipy.spatial.distance import cdist

# RDKit for molecular descriptors
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import argparse
import multiprocessing as mp
from multiprocessing import Pool, Manager
import time
import os
import signal
import warnings
warnings.filterwarnings('ignore')


# Per-seed wall: each (benchmark, model, seed) call to run_bo_lf_blr is bounded
# by SIGALRM(_SEED_WALL_SEC). If it fires, the existing except below records the
# seed as FAILED (n_exceptions = -1) and the next seed proceeds. Pure safety net;
# does not touch the BO loop, EI, fidelity selection, or surrogate code.
_SEED_WALL_SEC = 1800  # 30 minutes


class _SeedTimeoutError(BaseException):
    # Subclasses BaseException (NOT Exception) on purpose: the per-iteration
    # `except Exception` inside run_bo_lf_blr must NOT catch the wall-clock timeout
    # and convert it into a random-pick fallback. As a BaseException it propagates
    # past that handler to the per-seed handler in run_combination, which records the
    # seed as FAILED (n_exceptions = -1) — the behavior the docstring already promises.
    pass


def _seed_timeout_handler(signum, frame):
    raise _SeedTimeoutError(f"seed exceeded {_SEED_WALL_SEC}s wall (SIGALRM)")

# Set multiprocessing start method for CUDA compatibility
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

# BoTorch
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

# Local imports
from synthetic_functions import (
    branin_hf, branin_lf, park_hf, park_lf,
    SCENARIOS, FUNCTIONS
)


# =============================================================================
# Network Architectures
# =============================================================================

class LFNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        self.feature_net = nn.Sequential(*layers)
        self.out_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.out_layer(self.feature_net(x))

    def extract_features(self, x):
        return self.feature_net(x)


class HFNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        layers = []
        in_dim = input_dim + 1
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        self.feature_net = nn.Sequential(*layers)
        self.out_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x, y_lf):
        combined = torch.cat([x, y_lf], dim=-1)
        delta = self.out_layer(self.feature_net(combined))
        return y_lf + delta

    def extract_features(self, x, y_lf):
        combined = torch.cat([x, y_lf], dim=-1)
        return self.feature_net(combined)


class AdapterLayer(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=16):
        super().__init__()
        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, input_dim)

    def forward(self, x):
        return x + self.up(F.relu(self.down(x)))


# =============================================================================
# Bayesian Linear Regression for Last Layer
# =============================================================================

class BayesianLinearRegression:
    """Bayesian Linear Regression for uncertainty quantification on last layer."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha  # Prior precision
        self.beta = beta    # Noise precision
        self.m_N = None     # Posterior mean
        self.S_N = None     # Posterior covariance
        self.fitted = False

    def fit(self, Phi: np.ndarray, y: np.ndarray):
        """
        Fit BLR to features Phi and targets y.

        Args:
            Phi: Feature matrix (N, D) - typically last layer features
            y: Target values (N,)
        """
        Phi = np.asarray(Phi)
        y = np.asarray(y).flatten()

        N, D = Phi.shape

        # Add bias term
        Phi_bias = np.hstack([Phi, np.ones((N, 1))])
        D_bias = D + 1

        # Prior
        S_0_inv = self.alpha * np.eye(D_bias)

        # Posterior covariance
        S_N_inv = S_0_inv + self.beta * (Phi_bias.T @ Phi_bias)

        # Regularization for numerical stability
        S_N_inv += 1e-6 * np.eye(D_bias)

        try:
            self.S_N = np.linalg.inv(S_N_inv)
        except np.linalg.LinAlgError:
            self.S_N = np.linalg.pinv(S_N_inv)

        # Posterior mean
        self.m_N = self.beta * (self.S_N @ Phi_bias.T @ y)
        self.fitted = True

    def predict(self, Phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty.

        Args:
            Phi: Feature matrix (N, D)

        Returns:
            mean: Predicted mean (N,)
            std: Predicted standard deviation (N,)
        """
        if not self.fitted:
            raise RuntimeError("BLR not fitted yet")

        Phi = np.asarray(Phi)
        N = Phi.shape[0]

        # Add bias term
        Phi_bias = np.hstack([Phi, np.ones((N, 1))])

        # Predictive mean
        mean = Phi_bias @ self.m_N

        # Predictive variance
        var = np.zeros(N)
        for i in range(N):
            phi_i = Phi_bias[i:i+1].T
            # .item() extracts the (1,1) result as a Python scalar. On numpy>=2.0
            # BOTH a bare ndarray->scalar assignment (the paper_v1 random-LF root
            # cause) AND float() on a (1,1) array raise; .item() works on every
            # numpy version and is value-identical on numpy 1.26.
            var[i] = 1.0 / self.beta + (phi_i.T @ self.S_N @ phi_i).item()

        std = np.sqrt(np.maximum(var, 1e-10))

        return mean.flatten(), std.flatten()


# =============================================================================
# Model Classes with LF-BLR Support
# =============================================================================

class BaseModel:
    def __init__(self, input_dim: int, hidden_dim: int = 64, device=None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # LF-BLR components
        self.lf_blr = None
        self.has_lf_blr = False

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        raise NotImplementedError

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """Predict HF values (mean only, std=0.1 for non-BLR)"""
        raise NotImplementedError

    def predict_lf(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """Predict LF values with BLR uncertainty"""
        raise NotImplementedError

    def _fit_lf_blr(self, X_lf_t, y_lf):
        """Fit BLR on LF network's last layer features"""
        if not hasattr(self, 'lf_net'):
            return

        self.lf_net.eval()
        with torch.no_grad():
            features = self.lf_net.extract_features(X_lf_t).cpu().numpy()

        self.lf_blr = BayesianLinearRegression(alpha=1.0, beta=1.0)
        self.lf_blr.fit(features, y_lf)
        self.has_lf_blr = True


class MFGP(BaseModel):
    """MFGP already has proper UQ via GP posterior - no changes needed"""

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        n_lf, n_hf = len(X_lf), len(X_hf)
        X_lf_fid = np.hstack([X_lf, np.zeros((n_lf, 1))])
        X_hf_fid = np.hstack([X_hf, np.ones((n_hf, 1))])
        X_all = np.vstack([X_lf_fid, X_hf_fid])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        train_X = torch.tensor(X_all, dtype=torch.float64).to(self.device)
        train_Y = torch.tensor(y_all, dtype=torch.float64).unsqueeze(-1).to(self.device)
        self.model = SingleTaskMultiFidelityGP(
            train_X, train_Y,
            data_fidelities=[self.input_dim],
            outcome_transform=Standardize(m=1)
        ).to(self.device)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        self.is_fitted = True

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X_fid = np.hstack([X, np.ones((len(X), 1))])
        X_tensor = torch.tensor(X_fid, dtype=torch.float64).to(self.device)
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(X_tensor)
            mean = posterior.mean.cpu().numpy().flatten()
            std = posterior.variance.sqrt().cpu().numpy().flatten()
        return mean, np.maximum(std, 1e-6)

    def predict_lf(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """MFGP uses the same GP for both - just use HF prediction"""
        return self.predict(X)


class Sequential(BaseModel):
    def __init__(self, input_dim, hidden_dim=64, lf_epochs=200, hf_epochs=100, device=None):
        super().__init__(input_dim, hidden_dim, device)
        self.lf_epochs = lf_epochs
        self.hf_epochs = hf_epochs

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()
        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]
        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)
        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        # Fit LF-BLR after LF training
        self._fit_lf_blr(X_lf_t, y_lf_s)

        for p in self.lf_net.parameters():
            p.requires_grad = False
        opt = torch.optim.Adam(self.hf_net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.hf_epochs):
            opt.zero_grad()
            with torch.no_grad():
                y_lf_pred = self.lf_net(X_hf_t)
            y_hf_pred = self.hf_net(X_hf_t, y_lf_pred)
            F.mse_loss(y_hf_pred, y_hf_t).backward()
            opt.step()
        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1

    def predict_lf(self, X):
        """Predict LF with BLR uncertainty"""
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()

        with torch.no_grad():
            features = self.lf_net.extract_features(X_t).cpu().numpy()

        if self.has_lf_blr:
            mean_s, std_s = self.lf_blr.predict(features)
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            std = std_s * self.scaler_y.scale_[0]
            return mean, np.maximum(std, 1e-6)
        else:
            with torch.no_grad():
                mean_s = self.lf_net(X_t).cpu().numpy().flatten()
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            return mean, np.ones_like(mean) * 0.1


class Progressive(BaseModel):
    def __init__(self, input_dim, hidden_dim=64, epochs_per_stage=50, device=None):
        super().__init__(input_dim, hidden_dim, device)
        self.epochs_per_stage = epochs_per_stage

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()
        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]
        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)
        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3)
        for _ in range(self.epochs_per_stage * 2):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        # Fit LF-BLR
        self._fit_lf_blr(X_lf_t, y_lf_s)

        for p in self.lf_net.parameters():
            p.requires_grad = False
        for p in self.hf_net.feature_net.parameters():
            p.requires_grad = False
        opt = torch.optim.Adam(self.hf_net.out_layer.parameters(), lr=1e-3)
        for _ in range(self.epochs_per_stage):
            opt.zero_grad()
            with torch.no_grad():
                y_lf_pred = self.lf_net(X_hf_t)
            y_hf_pred = self.hf_net(X_hf_t, y_lf_pred)
            F.mse_loss(y_hf_pred, y_hf_t).backward()
            opt.step()
        for p in list(self.hf_net.feature_net.parameters())[-2:]:
            p.requires_grad = True
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.hf_net.parameters()), lr=1e-4)
        for _ in range(self.epochs_per_stage):
            opt.zero_grad()
            with torch.no_grad():
                y_lf_pred = self.lf_net(X_hf_t)
            y_hf_pred = self.hf_net(X_hf_t, y_lf_pred)
            F.mse_loss(y_hf_pred, y_hf_t).backward()
            opt.step()
        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1

    def predict_lf(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()

        with torch.no_grad():
            features = self.lf_net.extract_features(X_t).cpu().numpy()

        if self.has_lf_blr:
            mean_s, std_s = self.lf_blr.predict(features)
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            std = std_s * self.scaler_y.scale_[0]
            return mean, np.maximum(std, 1e-6)
        else:
            with torch.no_grad():
                mean_s = self.lf_net(X_t).cpu().numpy().flatten()
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            return mean, np.ones_like(mean) * 0.1


class Curriculum(BaseModel):
    def __init__(self, input_dim, hidden_dim=64, epochs=200, device=None):
        super().__init__(input_dim, hidden_dim, device)
        self.epochs = epochs

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()
        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]
        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)
        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3)
        for _ in range(self.epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        # Fit LF-BLR
        self._fit_lf_blr(X_lf_t, y_lf_s)

        for p in self.lf_net.parameters():
            p.requires_grad = False
        with torch.no_grad():
            y_lf_pred = self.lf_net(X_hf_t)
            residuals = torch.abs(y_hf_t - y_lf_pred).squeeze()
        sorted_idx = torch.argsort(residuals)
        opt = torch.optim.Adam(self.hf_net.parameters(), lr=1e-3)
        n_hf = len(X_hf_t)
        for epoch in range(self.epochs):
            n_use = min(n_hf, max(2, int((epoch + 1) / self.epochs * n_hf)))
            idx = sorted_idx[:n_use]
            opt.zero_grad()
            with torch.no_grad():
                y_lf_sub = self.lf_net(X_hf_t[idx])
            y_hf_pred = self.hf_net(X_hf_t[idx], y_lf_sub)
            F.mse_loss(y_hf_pred, y_hf_t[idx]).backward()
            opt.step()
        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1

    def predict_lf(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()

        with torch.no_grad():
            features = self.lf_net.extract_features(X_t).cpu().numpy()

        if self.has_lf_blr:
            mean_s, std_s = self.lf_blr.predict(features)
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            std = std_s * self.scaler_y.scale_[0]
            return mean, np.maximum(std, 1e-6)
        else:
            with torch.no_grad():
                mean_s = self.lf_net(X_t).cpu().numpy().flatten()
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            return mean, np.ones_like(mean) * 0.1


class TwoStageJoint(BaseModel):
    def __init__(self, input_dim, hidden_dim=64, stage1_epochs=100, stage2_epochs=100, device=None):
        super().__init__(input_dim, hidden_dim, device)
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()
        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]
        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)
        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3)
        for _ in range(self.stage1_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        # Fit LF-BLR after stage 1
        self._fit_lf_blr(X_lf_t, y_lf_s)

        opt = torch.optim.Adam(list(self.lf_net.parameters()) + list(self.hf_net.parameters()), lr=1e-4)
        for _ in range(self.stage2_epochs):
            opt.zero_grad()
            lf_loss = F.mse_loss(self.lf_net(X_lf_t), y_lf_t)
            y_lf_pred = self.lf_net(X_hf_t)
            hf_loss = F.mse_loss(self.hf_net(X_hf_t, y_lf_pred), y_hf_t)
            (0.3 * lf_loss + 0.7 * hf_loss).backward()
            opt.step()

        # Re-fit LF-BLR after joint training
        self._fit_lf_blr(X_lf_t, y_lf_s)

        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1

    def predict_lf(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()

        with torch.no_grad():
            features = self.lf_net.extract_features(X_t).cpu().numpy()

        if self.has_lf_blr:
            mean_s, std_s = self.lf_blr.predict(features)
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            std = std_s * self.scaler_y.scale_[0]
            return mean, np.maximum(std, 1e-6)
        else:
            with torch.no_grad():
                mean_s = self.lf_net(X_t).cpu().numpy().flatten()
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            return mean, np.ones_like(mean) * 0.1


class DNGOJoint(BaseModel):
    def __init__(self, input_dim, hidden_dim=64, epochs=300, alpha=0.5, device=None):
        super().__init__(input_dim, hidden_dim, device)
        self.epochs = epochs
        self.alpha = alpha

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()
        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]
        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)
        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(list(self.lf_net.parameters()) + list(self.hf_net.parameters()), lr=1e-3, weight_decay=1e-4)
        for _ in range(self.epochs):
            opt.zero_grad()
            lf_loss = F.mse_loss(self.lf_net(X_lf_t), y_lf_t)
            with torch.no_grad():
                y_lf_pred = self.lf_net(X_hf_t)
            hf_loss = F.mse_loss(self.hf_net(X_hf_t, y_lf_pred), y_hf_t)
            ((1 - self.alpha) * lf_loss + self.alpha * hf_loss).backward()
            opt.step()

        # Fit LF-BLR
        self._fit_lf_blr(X_lf_t, y_lf_s)

        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1

    def predict_lf(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()

        with torch.no_grad():
            features = self.lf_net.extract_features(X_t).cpu().numpy()

        if self.has_lf_blr:
            mean_s, std_s = self.lf_blr.predict(features)
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            std = std_s * self.scaler_y.scale_[0]
            return mean, np.maximum(std, 1e-6)
        else:
            with torch.no_grad():
                mean_s = self.lf_net(X_t).cpu().numpy().flatten()
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            return mean, np.ones_like(mean) * 0.1


class DNGOGradient(BaseModel):
    def __init__(self, input_dim, hidden_dim=64, epochs=300, device=None):
        super().__init__(input_dim, hidden_dim, device)
        self.epochs = epochs

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()
        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]
        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)
        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam([
            {'params': self.lf_net.parameters(), 'lr': 1e-3},
            {'params': self.hf_net.parameters(), 'lr': 5e-4}
        ], weight_decay=1e-4)
        for _ in range(self.epochs):
            opt.zero_grad()
            lf_loss = F.mse_loss(self.lf_net(X_lf_t), y_lf_t)
            y_lf_pred = self.lf_net(X_hf_t)
            hf_loss = F.mse_loss(self.hf_net(X_hf_t, y_lf_pred), y_hf_t)
            (lf_loss + hf_loss).backward()
            opt.step()

        # Fit LF-BLR
        self._fit_lf_blr(X_lf_t, y_lf_s)

        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1

    def predict_lf(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()

        with torch.no_grad():
            features = self.lf_net.extract_features(X_t).cpu().numpy()

        if self.has_lf_blr:
            mean_s, std_s = self.lf_blr.predict(features)
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            std = std_s * self.scaler_y.scale_[0]
            return mean, np.maximum(std, 1e-6)
        else:
            with torch.no_grad():
                mean_s = self.lf_net(X_t).cpu().numpy().flatten()
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            return mean, np.ones_like(mean) * 0.1


class KnowledgeDistillation(BaseModel):
    def __init__(self, input_dim, hidden_dim=64, lf_epochs=200, hf_epochs=100, alpha_kd=0.3, temperature=3.0, device=None):
        super().__init__(input_dim, hidden_dim, device)
        self.lf_epochs = lf_epochs
        self.hf_epochs = hf_epochs
        self.alpha_kd = alpha_kd
        self.temperature = temperature

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()
        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]
        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)
        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3)
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        # Fit LF-BLR
        self._fit_lf_blr(X_lf_t, y_lf_s)

        for p in self.lf_net.parameters():
            p.requires_grad = False
        opt = torch.optim.Adam(self.hf_net.parameters(), lr=1e-4)
        for _ in range(self.hf_epochs):
            opt.zero_grad()
            with torch.no_grad():
                teacher_pred = self.lf_net(X_hf_t)
            student_pred = self.hf_net(X_hf_t, teacher_pred)
            hard_loss = F.mse_loss(student_pred, y_hf_t)
            soft_student = student_pred / self.temperature
            soft_teacher = teacher_pred / self.temperature
            kd_loss = F.mse_loss(soft_student, soft_teacher) * (self.temperature ** 2)
            ((1 - self.alpha_kd) * hard_loss + self.alpha_kd * kd_loss).backward()
            opt.step()
        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1

    def predict_lf(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()

        with torch.no_grad():
            features = self.lf_net.extract_features(X_t).cpu().numpy()

        if self.has_lf_blr:
            mean_s, std_s = self.lf_blr.predict(features)
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            std = std_s * self.scaler_y.scale_[0]
            return mean, np.maximum(std, 1e-6)
        else:
            with torch.no_grad():
                mean_s = self.lf_net(X_t).cpu().numpy().flatten()
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            return mean, np.ones_like(mean) * 0.1


class DomainAdaptationMMD(BaseModel):
    def __init__(self, input_dim, hidden_dim=64, lf_epochs=200, hf_epochs=100, lambda_mmd=0.1, device=None):
        super().__init__(input_dim, hidden_dim, device)
        self.lf_epochs = lf_epochs
        self.hf_epochs = hf_epochs
        self.lambda_mmd = lambda_mmd

    def _mmd_loss(self, source, target, bandwidth=1.0):
        def rbf_kernel(x, y):
            diff = x.unsqueeze(1) - y.unsqueeze(0)
            dist_sq = torch.sum(diff ** 2, dim=-1)
            return torch.exp(-dist_sq / (2 * bandwidth ** 2))
        k_ss = rbf_kernel(source, source)
        k_tt = rbf_kernel(target, target)
        k_st = rbf_kernel(source, target)
        return k_ss.mean() + k_tt.mean() - 2 * k_st.mean()

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()
        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]
        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)
        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3)
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        # Fit LF-BLR after initial LF training
        self._fit_lf_blr(X_lf_t, y_lf_s)

        opt = torch.optim.Adam([
            {'params': self.lf_net.parameters(), 'lr': 1e-4},
            {'params': self.hf_net.parameters(), 'lr': 1e-4}
        ])
        for _ in range(self.hf_epochs):
            opt.zero_grad()
            lf_features = self.lf_net.extract_features(X_lf_t)
            with torch.no_grad():
                y_lf_pred = self.lf_net(X_hf_t)
            hf_features = self.hf_net.extract_features(X_hf_t, y_lf_pred)
            task_loss = F.mse_loss(self.hf_net(X_hf_t, y_lf_pred), y_hf_t)
            min_n = min(len(lf_features), len(hf_features))
            mmd = self._mmd_loss(lf_features[:min_n], hf_features[:min_n])
            (task_loss + self.lambda_mmd * mmd).backward()
            opt.step()

        # Re-fit LF-BLR after joint training
        self._fit_lf_blr(X_lf_t, y_lf_s)

        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1

    def predict_lf(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()

        with torch.no_grad():
            features = self.lf_net.extract_features(X_t).cpu().numpy()

        if self.has_lf_blr:
            mean_s, std_s = self.lf_blr.predict(features)
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            std = std_s * self.scaler_y.scale_[0]
            return mean, np.maximum(std, 1e-6)
        else:
            with torch.no_grad():
                mean_s = self.lf_net(X_t).cpu().numpy().flatten()
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            return mean, np.ones_like(mean) * 0.1


class SoftParameterSharing(BaseModel):
    def __init__(self, input_dim, hidden_dim=64, epochs=200, lambda_soft=0.01, device=None):
        super().__init__(input_dim, hidden_dim, device)
        self.epochs = epochs
        self.lambda_soft = lambda_soft

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()
        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]
        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)
        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(list(self.lf_net.parameters()) + list(self.hf_net.parameters()), lr=1e-3)
        for _ in range(self.epochs):
            opt.zero_grad()
            lf_loss = F.mse_loss(self.lf_net(X_lf_t), y_lf_t)
            y_lf_pred = self.lf_net(X_hf_t)
            hf_loss = F.mse_loss(self.hf_net(X_hf_t, y_lf_pred), y_hf_t)
            lf_params = list(self.lf_net.feature_net.parameters())
            hf_params = list(self.hf_net.feature_net.parameters())
            param_diff = 0.0
            if len(lf_params) > 0 and len(hf_params) > 0:
                lf_w = lf_params[0]
                hf_w = hf_params[0]
                min_in = min(lf_w.shape[1], hf_w.shape[1])
                param_diff = torch.sum((lf_w[:, :min_in] - hf_w[:, :min_in]) ** 2)
            (0.5 * lf_loss + 0.5 * hf_loss + self.lambda_soft * param_diff).backward()
            opt.step()

        # Fit LF-BLR
        self._fit_lf_blr(X_lf_t, y_lf_s)

        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1

    def predict_lf(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()

        with torch.no_grad():
            features = self.lf_net.extract_features(X_t).cpu().numpy()

        if self.has_lf_blr:
            mean_s, std_s = self.lf_blr.predict(features)
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            std = std_s * self.scaler_y.scale_[0]
            return mean, np.maximum(std, 1e-6)
        else:
            with torch.no_grad():
                mean_s = self.lf_net(X_t).cpu().numpy().flatten()
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            return mean, np.ones_like(mean) * 0.1


class PseudoLabeling(BaseModel):
    def __init__(self, input_dim, hidden_dim=64, lf_epochs=200, hf_epochs=100, pseudo_weight=0.5, device=None):
        super().__init__(input_dim, hidden_dim, device)
        self.lf_epochs = lf_epochs
        self.hf_epochs = hf_epochs
        self.pseudo_weight = pseudo_weight

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()
        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]
        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)
        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3)
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()

        # Fit LF-BLR
        self._fit_lf_blr(X_lf_t, y_lf_s)

        self.lf_net.eval()
        with torch.no_grad():
            lf_pred_on_hf = self.lf_net(X_hf_t)
            offset = (y_hf_t - lf_pred_on_hf).mean()
            lf_pred_all = self.lf_net(X_lf_t)
            pseudo_labels = lf_pred_all + offset
        for p in self.lf_net.parameters():
            p.requires_grad = False
        opt = torch.optim.Adam(self.hf_net.parameters(), lr=1e-4)
        for _ in range(self.hf_epochs):
            opt.zero_grad()
            with torch.no_grad():
                y_lf_for_hf = self.lf_net(X_hf_t)
            real_loss = F.mse_loss(self.hf_net(X_hf_t, y_lf_for_hf), y_hf_t)
            with torch.no_grad():
                y_lf_pseudo = self.lf_net(X_lf_t)
            pseudo_loss = F.mse_loss(self.hf_net(X_lf_t, y_lf_pseudo), pseudo_labels)
            (real_loss + self.pseudo_weight * pseudo_loss).backward()
            opt.step()
        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            y_hf = self.hf_net(X_t, y_lf)
            mean_s = y_hf.cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1

    def predict_lf(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()

        with torch.no_grad():
            features = self.lf_net.extract_features(X_t).cpu().numpy()

        if self.has_lf_blr:
            mean_s, std_s = self.lf_blr.predict(features)
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            std = std_s * self.scaler_y.scale_[0]
            return mean, np.maximum(std, 1e-6)
        else:
            with torch.no_grad():
                mean_s = self.lf_net(X_t).cpu().numpy().flatten()
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            return mean, np.ones_like(mean) * 0.1


class Adapter(BaseModel):
    """Adapter model - uses backbone instead of lf_net, needs special handling"""

    def __init__(self, input_dim, hidden_dim=64, lf_epochs=200, adapter_epochs=100, bottleneck_dim=16, device=None):
        super().__init__(input_dim, hidden_dim, device)
        self.lf_epochs = lf_epochs
        self.adapter_epochs = adapter_epochs
        self.bottleneck_dim = bottleneck_dim

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()
        X_lf_s, X_hf_s = X_scaled[:len(X_lf)], X_scaled[len(X_lf):]
        y_lf_s, y_hf_s = y_scaled[:len(y_lf)], y_scaled[len(y_lf):]
        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        ).to(self.device)
        self.out_layer = nn.Linear(self.hidden_dim, 1).to(self.device)
        self.adapters = nn.ModuleList([AdapterLayer(self.hidden_dim, self.bottleneck_dim) for _ in range(2)]).to(self.device)
        self.hf_out = nn.Linear(self.hidden_dim, 1).to(self.device)
        opt = torch.optim.Adam(list(self.backbone.parameters()) + list(self.out_layer.parameters()), lr=1e-3)
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            h = self.backbone(X_lf_t)
            F.mse_loss(self.out_layer(h), y_lf_t).backward()
            opt.step()

        # Fit LF-BLR on backbone features
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(X_lf_t).cpu().numpy()
        self.lf_blr = BayesianLinearRegression(alpha=1.0, beta=1.0)
        self.lf_blr.fit(features, y_lf_s)
        self.has_lf_blr = True

        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.out_layer.parameters():
            p.requires_grad = False
        opt = torch.optim.Adam(list(self.adapters.parameters()) + list(self.hf_out.parameters()), lr=1e-3)
        for _ in range(self.adapter_epochs):
            opt.zero_grad()
            h = X_hf_t
            adapter_idx = 0
            for i, module in enumerate(self.backbone):
                h = module(h)
                if isinstance(module, nn.Tanh) and adapter_idx < len(self.adapters):
                    h = self.adapters[adapter_idx](h)
                    adapter_idx += 1
            F.mse_loss(self.hf_out(h), y_hf_t).backward()
            opt.step()
        self.is_fitted = True

    def predict(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.backbone.eval()
        with torch.no_grad():
            h = X_t
            adapter_idx = 0
            for module in self.backbone:
                h = module(h)
                if isinstance(module, nn.Tanh) and adapter_idx < len(self.adapters):
                    h = self.adapters[adapter_idx](h)
                    adapter_idx += 1
            mean_s = self.hf_out(h).cpu().numpy().flatten()
        mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        return mean, np.ones_like(mean) * 0.1

    def predict_lf(self, X):
        """Predict LF using backbone + BLR"""
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.backbone.eval()

        with torch.no_grad():
            features = self.backbone(X_t).cpu().numpy()

        if self.has_lf_blr:
            mean_s, std_s = self.lf_blr.predict(features)
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            std = std_s * self.scaler_y.scale_[0]
            return mean, np.maximum(std, 1e-6)
        else:
            with torch.no_grad():
                mean_s = self.out_layer(self.backbone(X_t)).cpu().numpy().flatten()
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            return mean, np.ones_like(mean) * 0.1


# =============================================================================
# Benchmark Classes (same as original)
# =============================================================================

class SyntheticBenchmark:
    def __init__(self, name, hf_func, lf_func, dim, alpha, cost_ratio, f_star, grid_size=50):
        self.name = name
        self.hf_func = hf_func
        self.lf_func = lf_func
        self.dim = dim
        self.alpha = alpha
        self.cost_ratio = cost_ratio
        self.f_star = f_star
        self.grid_size = grid_size
        self._create_grid()
        corr = np.corrcoef(self.y_hf, self.y_lf)[0, 1]
        self.r2 = corr ** 2

    def _create_grid(self):
        if self.dim == 2:
            axes = [np.linspace(0, 1, self.grid_size) for _ in range(2)]
            grids = np.meshgrid(*axes)
            self.X = np.column_stack([g.ravel() for g in grids])
        else:
            n_per_dim = int(np.ceil(self.grid_size ** 0.5))
            axes = [np.linspace(0, 1, n_per_dim) for _ in range(self.dim)]
            grids = np.meshgrid(*axes, indexing='ij')
            self.X = np.column_stack([g.ravel() for g in grids])
        self.n_candidates = len(self.X)
        self.y_hf = self.hf_func(self.X).flatten()
        self.y_lf = self.lf_func(self.X, self.alpha).flatten()

    def evaluate_hf(self, indices):
        return self.y_hf[indices.astype(int).flatten()]

    def evaluate_lf(self, indices):
        return self.y_lf[indices.astype(int).flatten()]


class ChemistryBenchmark:
    def __init__(self, name, csv_path, cost_ratio, use_smiles=False, minimize=True, negate=False, pca_dim=10):
        self.name = name
        self.cost_ratio = cost_ratio
        self.minimize = minimize
        self.negate = negate
        self.pca_dim = pca_dim
        df = pd.read_csv(csv_path)
        if use_smiles:
            smiles_col = [c for c in df.columns if 'smiles' in c.lower()]
            if smiles_col:
                self.X = self._smiles_to_rdkit_features(df[smiles_col[0]].values)
            else:
                self.X = self._smiles_to_rdkit_features(df.iloc[:, 0].values)
        else:
            feature_cols = [c for c in df.columns if c not in ['HF', 'LF']]
            self.X = df[feature_cols].values
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
        self.y_hf = df['HF'].values.flatten()
        self.y_lf = df['LF'].values.flatten()

        if negate:
            self.y_hf = -self.y_hf
            self.y_lf = -self.y_lf

        self.f_star = self.y_hf.min()
        self.n_candidates = len(self.X)
        self.dim = self.X.shape[1]
        corr = np.corrcoef(self.y_hf, self.y_lf)[0, 1]
        self.r2 = corr ** 2

    def _smiles_to_rdkit_features(self, smiles_list):
        descriptor_names = [desc[0] for desc in Descriptors._descList]
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

        features = []
        valid_indices = []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                desc = calc.CalcDescriptors(mol)
                features.append(desc)
                valid_indices.append(i)
            else:
                features.append([0.0] * len(descriptor_names))
                valid_indices.append(i)

        features = np.array(features, dtype=np.float64)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # random_state is REQUIRED: for these matrix shapes sklearn's svd_solver
        # 'auto' selects randomized SVD, which is non-deterministic without a seed.
        # Without it the SMILES-pool (FreeSolv, polarizability) features differ every
        # time the benchmark is built -- even across models in one run -- which made
        # those two benchmarks irreproducible (COFs/HOPV15/Matbench use precomputed
        # descriptors and are unaffected). Fixed seed -> identical features always.
        pca = PCA(n_components=self.pca_dim, random_state=42)
        features_pca = pca.fit_transform(features_scaled)

        return features_pca

    def evaluate_hf(self, indices):
        return self.y_hf[indices.astype(int).flatten()]

    def evaluate_lf(self, indices):
        return self.y_lf[indices.astype(int).flatten()]


# =============================================================================
# Initial Sampling Methods (same as original)
# =============================================================================

def furthest_point_sampling(X: np.ndarray, n_samples: int, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    n_candidates = len(X)
    n_samples = min(n_samples, n_candidates)
    selected = [np.random.randint(n_candidates)]
    for _ in range(n_samples - 1):
        selected_X = X[selected]
        distances = cdist(X, selected_X, metric='euclidean')
        min_distances = distances.min(axis=1)
        min_distances[selected] = -np.inf
        next_idx = np.argmax(min_distances)
        selected.append(next_idx)
    return np.array(selected)


def latin_hypercube_sampling(bounds: np.ndarray, n_samples: int, seed: int = 42) -> np.ndarray:
    n_dims = len(bounds)
    sampler = LatinHypercube(d=n_dims, seed=seed)
    samples = sampler.random(n=n_samples)
    for i in range(n_dims):
        samples[:, i] = bounds[i, 0] + samples[:, i] * (bounds[i, 1] - bounds[i, 0])
    return samples


def find_nearest_candidates(X_candidates: np.ndarray, X_samples: np.ndarray) -> np.ndarray:
    distances = cdist(X_samples, X_candidates, metric='euclidean')
    return np.argmin(distances, axis=1)


# =============================================================================
# Acquisition & BO Loop with LF-BLR Strategy
# =============================================================================

def expected_improvement(mean, std, y_best, xi=0.01):
    with np.errstate(divide='ignore', invalid='ignore'):
        imp = y_best - mean - xi
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std < 1e-10] = 0.0
    return ei


# =============================================================================
# Acquisition-function portfolio (LF step, DNN surrogates).
#
# These extend the single EI-vs-greedy comparison behind paper Fig. A6
# (sec:res-uq) to the full standard taxonomy of myopic acquisition functions,
# so the claim "uncertainty-driven exploration does not help" is tested against
# a representative set rather than EI alone. All are written for MINIMISATION on
# a discrete candidate pool, take the surrogate's marginal predictive mean+std,
# and return a per-candidate SCORE whose argmax selects the next point (so the
# caller's `acq[sampled] = -inf; argmax` masking is uniform across all of them).
# Families (Shahriari et al. 2016; Garnett 2023):
#   improvement-based : expected_improvement (above), probability_of_improvement
#   optimistic        : lower_confidence_bound      (GP-UCB; Srinivas et al. 2010)
#   information-based  : max_value_entropy_search    (MES-Gumbel; Wang & Jegelka 2017)
#   sampling-based     : thompson_sample            (Thompson 1933; Russo et al. 2018)
# =============================================================================

def probability_of_improvement(mean, std, y_best, xi=0.01):
    """PI for minimisation: P(f(x) < y_best - xi) = Phi((y_best - mean - xi)/std)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = (y_best - mean - xi) / std
        pi = norm.cdf(Z)
        pi[std < 1e-10] = 0.0
    return pi


def lower_confidence_bound(mean, std, beta=2.0):
    """GP-LCB for minimisation. The optimistic value of a candidate is its lower
    confidence bound mean - beta*std; we want argmin of that, so return the
    negated score (beta*std - mean) for the caller's argmax. Larger beta = more
    exploration. beta=2.0 matches the GP-UCB default used in experiments/mfgp_variants."""
    return beta * std - mean


def thompson_sample(mean, std, rng):
    """Thompson sampling with independent marginals: draw one posterior sample per
    candidate and (for minimisation) prefer the smallest, i.e. return -sample so
    argmax picks argmin(sample). Only marginal (mean, std) are available from the
    LF-BLR head, matching the marginal TS used in experiments/mfgp_variants. `rng`
    is a seeded numpy Generator so a given BO seed is reproducible."""
    sample = mean + std * rng.standard_normal(size=mean.shape)
    return -sample


def _sample_min_values_gumbel(mean, std, n_samples, rng, n_grid=200):
    """Sample candidate MINIMUM values f* of the LF function over the pool via the
    Gumbel approximation of Wang & Jegelka (2017, sec. 3.1), adapted to
    minimisation by negation (g = -f, so the min of f is -(max of g)). Uses only
    the per-candidate marginals (the BLR head gives marginal variance), which is
    exactly the independent-Gaussian "MES-G" approximation. Returns f* of shape
    (n_samples,)."""
    m = -mean                              # posterior mean of g = -f
    s = np.maximum(std, 1e-9)
    lo = float(m.max())                    # max of g is at least the largest g-mean
    hi = float((m + 5.0 * s).max())        # ... and very rarely beyond mean + 5 sigma
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        # Degenerate (e.g. all std ~ 0): no UQ signal, fall back to the empirical min.
        return np.full(n_samples, float(mean.min()))
    grid = np.linspace(lo, hi, n_grid)
    # F(y) = Pr[max g <= y] = prod_i Phi((y - m_i)/s_i), evaluated on the grid in
    # log-space for numerical stability, then exponentiated. Monotone in [0, 1].
    Z = (grid[:, None] - m[None, :]) / s[None, :]
    F = np.exp(norm.logcdf(Z).sum(axis=1))

    def _quantile(r):
        idx = int(np.clip(np.searchsorted(F, r), 1, n_grid - 1))
        f0, f1 = F[idx - 1], F[idx]
        if f1 <= f0:
            return grid[idx]
        w = (r - f0) / (f1 - f0)
        return grid[idx - 1] + w * (grid[idx] - grid[idx - 1])

    # Percentile matching at r1=0.25, r2=0.75 (Wang & Jegelka): solve the 2x2
    # system y = a - b*log(-log(r)) for the Gumbel location a and scale b.
    y1, y2 = _quantile(0.25), _quantile(0.75)
    L1, L2 = np.log(-np.log(0.25)), np.log(-np.log(0.75))
    b = (y2 - y1) / (L1 - L2)              # > 0 since y2 > y1 and L1 > L2
    u = np.clip(rng.random(n_samples), 1e-6, 1 - 1e-6)
    if not np.isfinite(b) or b <= 0:
        # Gumbel fit degenerate: sample directly from the empirical grid CDF instead.
        gstar = np.interp(u, F, grid)
    else:
        a = y1 + b * L1
        gstar = a - b * np.log(-np.log(u))
    return -gstar                          # f* = -(g*)


def max_value_entropy_search(mean, std, rng, n_samples=10):
    """MES for minimisation (Wang & Jegelka 2017, Gumbel approximation), on the
    pool's marginal (mean, std). For each sampled min-value f*,
        gamma = (mean - f*) / std,
        info  = gamma * phi(gamma) / (2 Phi(gamma)) - log Phi(gamma),
    averaged over the f* samples. argmax selects (higher mutual information with
    the optimum = better). This is the negation-transformed form of the paper's
    Eq. (6): minimising f equals maximising g = -f with g* = -f*."""
    s = np.maximum(std, 1e-9)
    fstar = _sample_min_values_gumbel(mean, std, n_samples, rng)
    acq = np.zeros_like(mean, dtype=float)
    for ys in fstar:
        gamma = (mean - ys) / s
        Psi = np.clip(norm.cdf(gamma), 1e-10, 1.0)
        acq += gamma * norm.pdf(gamma) / (2.0 * Psi) - np.log(Psi)
    return acq / len(fstar)


def run_bo_lf_blr(benchmark, model_class, budget, seed=42, device=None,
                  sampling_method='fps', promotion_masking=False,
                  lf_ei_const_std=0.1, mfgp_lf_greedy=False,
                  calib_checkpoint_budget=None, lf_acq='ei'):
    """
    Run Bayesian Optimization with LF-BLR strategy.

    Key difference from original:
    - LF selection: LF-BLR prediction + EI (exploration)
    - HF selection: HF prediction + argmin (exploitation)
    - MFGP uses its own GP-based UQ (unchanged)

    LF EI uncertainty (`lf_ei_const_std`):
    - This is the ORIGINAL / published method that produced `results/paper_v1`
      (see paper/auto_figures/ROOT_CAUSE_INVESTIGATION.md). The LF-BLR head
      supplies the EI MEAN, but the EI std is a CONSTANT (default 0.1). The
      committed code had briefly used the BLR *predictive* std here, which
      over-explores on chemistry pools and does NOT reproduce paper_v1
      (DNN surrogates blow up from ~0.3-0.8 to ~6-7.5 regret on COFs).
    - `lf_ei_const_std=0.1` (default) restores the published behavior and
      reproduces paper_v1's aggregate distribution.
    - `lf_ei_const_std=None` uses the BLR predictive std instead (the
      "true UQ" variant — kept runnable for the camera-ready appendix
      comparison). Only affects DNN models; MFGP's acquisition is controlled
      separately by `mfgp_lf_greedy`.

    MFGP LF acquisition (`mfgp_lf_greedy`):
    - `False` (default) = published baseline: MFGP's LF step uses its GP
      posterior std in EI (true UQ-driven exploration). Reproduces paper_v1.
    - `True` = "Greedy-MFGP" control: MFGP's LF step instead picks argmin of
      the GP posterior mean (μ only), matching the DNN const-std (greedy)
      acquisition. This is the missing cell of the surrogate × acquisition
      2x2 (MFGP/DNN × EI/greedy) used to deconfound "is the DNN advantage
      from the surrogate or from the greedy acquisition?" — see
      paper/auto_figures/ROOT_CAUSE_INVESTIGATION.md. HF selection is argmin
      (greedy) for ALL models regardless, so this flag — like
      `lf_ei_const_std` — only changes the LF acquisition step.

    LF acquisition portfolio (`lf_acq`, DNN models only):
    - `'ei'` (default) = the paper behaviour described above (greedy when
      lf_ei_const_std is a constant, UQ-driven EI when it is None). Default keeps
      Figs 1-2 / A6 reproducing bit-for-bit.
    - `'pi'`, `'ucb'`, `'mes'`, `'ts'` = the other four members of the standard
      acquisition-function taxonomy (probability of improvement; GP lower
      confidence bound; max-value entropy search; Thompson sampling). They ALWAYS
      use the BLR predictive std, so together with UQ-driven EI they form the
      five-acquisition portfolio compared against the greedy baseline in the
      acquisition-portfolio ablation (extends sec:res-uq beyond EI-vs-greedy).
      Like `lf_ei_const_std`, this only changes the DNN LF acquisition step; MFGP
      is governed by `mfgp_lf_greedy`. 'ts'/'mes' are stochastic but seeded.

    Backward-compatible additions:
    - `promotion_masking=False` (default = frozen behavior, HF mask excludes
      LF ∪ HF indices). When True, HF mask excludes only HF indices, so an
      LF-evaluated candidate can be "promoted" to HF. Used by the A4
      ablation. Default False keeps Figs 1-2 reproducing bit-for-bit.
    - The returned dict now also includes `lf_indices`, `hf_indices`,
      `X_lf`, `y_lf`, `X_hf`, `y_hf`, and `n_promotions` (only meaningful
      when promotion_masking=True). Existing callers reading the original
      keys are unaffected.
    - `calib_checkpoint_budget=None` (default = frozen behavior). When set to
      a budget value, the returned dict's `calib_snapshot` holds a copy of the
      collected data (indices + X/y for both fidelities) at the last iteration
      whose cumulative budget <= the checkpoint. This is PURELY OBSERVATIONAL —
      it records state but never changes a sampling decision — so the default
      (None) path is byte-identical and Figs 1-2 reproduce bit-for-bit. Used by
      the A1 calibration figure to evaluate held-out calibration at an early
      "decision budget" on the saturating Park benchmarks.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Dedicated RNG for the stochastic LF acquisitions (Thompson sampling, MES
    # min-value draws). Kept separate from the global np.random stream so the
    # default lf_acq='ei' path never touches it and stays byte-identical to
    # paper_v1; only the 'ts'/'mes' branches below consume it.
    acq_rng = np.random.default_rng(seed)

    rho = benchmark.cost_ratio
    n_candidates = benchmark.n_candidates
    is_mfgp = (model_class == MFGP)

    init_budget = 0.1 * budget
    n_init_hf = max(2, int(init_budget * 0.5 / 1.0))
    n_init_lf = max(2, int(init_budget * 0.5 / rho))
    n_init_total = n_init_lf + n_init_hf

    # Initial sampling
    if sampling_method == 'lhs':
        bounds = np.array([[0, 1]] * benchmark.dim)
        lhs_samples = latin_hypercube_sampling(bounds, n_init_total, seed)
        X_min, X_max = benchmark.X.min(axis=0), benchmark.X.max(axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1
        lhs_samples_scaled = X_min + lhs_samples * X_range
        init_indices = find_nearest_candidates(benchmark.X, lhs_samples_scaled)
        init_indices = list(dict.fromkeys(init_indices))
        if len(init_indices) < n_init_total:
            remaining = n_init_total - len(init_indices)
            available = set(range(n_candidates)) - set(init_indices)
            if available:
                extra = furthest_point_sampling(
                    benchmark.X[list(available)],
                    remaining,
                    seed + 1000
                )
                extra_indices = [list(available)[i] for i in extra]
                init_indices.extend(extra_indices)
    else:
        init_indices = furthest_point_sampling(benchmark.X, n_init_total, seed).tolist()

    lf_indices = set(init_indices[:n_init_lf])
    hf_indices = set(init_indices[n_init_lf:n_init_lf + n_init_hf])

    X_lf = benchmark.X[list(lf_indices)]
    y_lf = benchmark.evaluate_lf(np.array(list(lf_indices)))
    X_hf = benchmark.X[list(hf_indices)]
    y_hf = benchmark.evaluate_hf(np.array(list(hf_indices)))

    current_budget = n_init_lf * rho + n_init_hf * 1.0
    lf_per_hf = max(1, int(1.0 / rho))
    lf_counter = 0

    regrets = [max(0, y_hf.min() - benchmark.f_star)]
    budgets = [current_budget]

    # Purely observational snapshot (does NOT affect any sampling decision):
    # when calib_checkpoint_budget is set, keep a copy of the collected data at
    # the last iteration whose cumulative budget <= the checkpoint, so a held-out
    # calibration metric can be evaluated at an EARLY decision budget. Used by the
    # A1 calibration figure for the saturating Park benchmarks (which exhaust
    # their candidate pool / saturate to ~0 regret by the full budget). Default
    # None => no snapshot is taken and behavior is byte-identical to before, so
    # Figs 1-2 reproduce bit-for-bit.
    calib_snapshot = None

    def _snapshot_if_due():
        nonlocal calib_snapshot
        if calib_checkpoint_budget is not None and current_budget <= calib_checkpoint_budget:
            calib_snapshot = {
                'lf_indices': set(lf_indices), 'hf_indices': set(hf_indices),
                'X_lf': X_lf.copy(), 'y_lf': y_lf.copy(),
                'X_hf': X_hf.copy(), 'y_hf': y_hf.copy(),
                'budget': current_budget,
            }
    _snapshot_if_due()

    n_promotions = 0  # HF picks that revisit an LF-evaluated candidate
    n_exceptions = 0  # iterations that fell back to a random pick (see except below)
    # n_attempts counts iterations that actually entered the BO try below; it EXCLUDES
    # the final budget-exhaustion `break` iteration (which increments `iteration` but
    # does no work and cannot raise), so the all-failed guard at the end of the loop is
    # exact rather than off-by-one on benchmarks whose budget tail triggers `break`.
    n_attempts = 0

    iteration = 0
    max_iter = 500

    while current_budget < budget and iteration < max_iter:
        iteration += 1
        remaining = budget - current_budget

        if remaining >= 1.0:
            if remaining >= rho and lf_counter < lf_per_hf:
                eval_hf = False
                cost = rho
                lf_counter += 1
            else:
                eval_hf = True
                cost = 1.0
                lf_counter = 0
        elif remaining >= rho:
            eval_hf = False
            cost = rho
        else:
            break

        n_attempts += 1
        try:
            model = model_class(benchmark.X.shape[1], device=device)
            model.fit(X_lf, y_lf, X_hf, y_hf)

            sampled = lf_indices | hf_indices

            if eval_hf:
                # HF selection: use HF prediction + argmin (exploitation)
                mean_hf, _ = model.predict(benchmark.X)
                mean_masked = mean_hf.copy()
                # Default (frozen): mask LF ∪ HF (no revisits at all).
                # promotion_masking=True: mask only HF (allow LF→HF promotion).
                hf_mask = hf_indices if promotion_masking else sampled
                mean_masked[list(hf_mask)] = np.inf
                next_idx = np.argmin(mean_masked)
                if promotion_masking and next_idx in lf_indices:
                    n_promotions += 1
            else:
                # LF selection: use LF-BLR prediction + EI (exploration)
                if is_mfgp:
                    # MFGP baseline: GP-based UQ via EI (true exploration).
                    # mfgp_lf_greedy=True: argmin of the GP posterior mean
                    # (μ only) — the Greedy-MFGP control that mirrors the DNN
                    # const-std greedy acquisition (see docstring / 2x2).
                    mean, std = model.predict(benchmark.X)
                    if mfgp_lf_greedy:
                        mean_masked = mean.copy()
                        mean_masked[list(sampled)] = np.inf
                        next_idx = np.argmin(mean_masked)
                    else:
                        y_best = y_hf.min()
                        ei = expected_improvement(mean, std, y_best)
                        ei[list(sampled)] = -np.inf
                        next_idx = np.argmax(ei)
                else:
                    # DNN models: the LF-BLR head supplies the predictive mean and
                    # std; `lf_acq` selects the LF acquisition rule (default 'ei' =
                    # paper behaviour). For 'ei' the EI std is a CONSTANT
                    # (lf_ei_const_std, default 0.1, reproduces paper_v1) unless it
                    # is None, in which case the BLR predictive std is used (the
                    # UQ-driven EI variant). The portfolio choices (pi/ucb/mes/ts)
                    # are UQ acquisitions that ALWAYS use the BLR predictive std
                    # (they ignore lf_ei_const_std). See the acquisition-portfolio
                    # block above for the formulas / references.
                    mean_lf, blr_std_lf = model.predict_lf(benchmark.X)
                    y_best_lf = y_lf.min()
                    if lf_acq == 'ei':
                        if lf_ei_const_std is not None:
                            std_lf = np.full_like(mean_lf, float(lf_ei_const_std))
                        else:
                            std_lf = blr_std_lf
                        acq = expected_improvement(mean_lf, std_lf, y_best_lf)
                    elif lf_acq == 'pi':
                        acq = probability_of_improvement(mean_lf, blr_std_lf, y_best_lf)
                    elif lf_acq == 'ucb':
                        acq = lower_confidence_bound(mean_lf, blr_std_lf)
                    elif lf_acq == 'mes':
                        acq = max_value_entropy_search(mean_lf, blr_std_lf, acq_rng)
                    elif lf_acq == 'ts':
                        acq = thompson_sample(mean_lf, blr_std_lf, acq_rng)
                    else:
                        raise ValueError(f"unknown lf_acq: {lf_acq!r}")
                    acq[list(sampled)] = -np.inf
                    next_idx = np.argmax(acq)

            if eval_hf:
                hf_indices.add(next_idx)
                X_hf = benchmark.X[list(hf_indices)]
                y_hf = benchmark.evaluate_hf(np.array(list(hf_indices)))
            else:
                lf_indices.add(next_idx)
                X_lf = benchmark.X[list(lf_indices)]
                y_lf = benchmark.evaluate_lf(np.array(list(lf_indices)))

            current_budget += cost
        except Exception as e:
            # An iteration failed (commonly model.fit / GP fit throwing). The
            # seeded random-pick fallback is KEPT so historical results stay
            # bit-identical, but it is now COUNTED and LOGGED. Previously a run
            # whose every iteration failed still looked like a valid result,
            # because np.random is seeded at loop start -> all models produced an
            # identical random sweep in ~0.2s. That fabricated the bogus
            # run_20260524_033621 ("blr-std") grid; see
            # paper/auto_figures/ROOT_CAUSE_INVESTIGATION.md.
            n_exceptions += 1
            if n_exceptions == 1:
                import traceback
                print(f"[run_bo_lf_blr] EXCEPTION at iter {iteration} "
                      f"({getattr(benchmark, 'name', '?')}/{model_class.__name__}/"
                      f"seed{seed}) -> random-pick fallback: {e!r}", flush=True)
                traceback.print_exc()
            available = set(range(n_candidates)) - (lf_indices | hf_indices)
            if available:
                next_idx = np.random.choice(list(available))
                if eval_hf:
                    hf_indices.add(next_idx)
                    X_hf = benchmark.X[list(hf_indices)]
                    y_hf = benchmark.evaluate_hf(np.array(list(hf_indices)))
                else:
                    lf_indices.add(next_idx)
                    X_lf = benchmark.X[list(lf_indices)]
                    y_lf = benchmark.evaluate_lf(np.array(list(lf_indices)))
            current_budget += cost

        regrets.append(max(0, y_hf.min() - benchmark.f_star))
        budgets.append(current_budget)
        _snapshot_if_due()

    # Silent-failure guard: if EVERY iteration fell back to a random pick, the
    # model never drove selection — the "result" is a seeded random sweep, not BO.
    # Raise so the caller records this seed as FAILED (NaN) and resume retries it,
    # rather than writing a fake-but-plausible regret. Real runs never hit this.
    if n_attempts > 0 and n_exceptions == n_attempts:
        raise RuntimeError(
            f"all {n_attempts} BO iterations failed and fell back to random picks "
            f"({getattr(benchmark, 'name', '?')}/{model_class.__name__}/seed{seed}); "
            f"no real BO happened — treating the seed as failed")
    if n_exceptions:
        print(f"[run_bo_lf_blr] {getattr(benchmark, 'name', '?')}/"
              f"{model_class.__name__}/seed{seed}: {n_exceptions}/{n_attempts} "
              f"iterations used the random fallback (partial failure)", flush=True)

    return {
        'regrets': regrets,
        'budgets': budgets,
        'final_regret': regrets[-1],
        'n_hf': len(hf_indices),
        'n_lf': len(lf_indices),
        'best_y': y_hf.min(),
        # Extra fields (ignored by paper_v1 callers; used by A1/A2/A4):
        'lf_indices': lf_indices,
        'hf_indices': hf_indices,
        'X_lf': X_lf,
        'y_lf': y_lf,
        'X_hf': X_hf,
        'y_hf': y_hf,
        'n_promotions': n_promotions,
        'n_exceptions': n_exceptions,
        'calib_snapshot': calib_snapshot,  # None unless calib_checkpoint_budget set
    }


# =============================================================================
# Parallel Worker Function
# =============================================================================

def _atomic_write_csv(df, path):
    """Write *df* to *path* atomically: write a temp file then os.replace it in.
    A job killed mid-write can never corrupt an existing checkpoint this way."""
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def run_combination(args):
    """Worker function for parallel execution.

    Resumable: completed seeds are read back from the per-task CSV checkpoint and
    skipped; only missing seeds run. A checkpoint is rewritten after every seed,
    so an interrupted job continues from the last finished seed, not from scratch.
    Seeds with a NaN final_regret (a prior failure) are treated as not-done and retried.
    """
    (bench_name, bench_config, model_name, model_class, budget, seeds,
     output_dir, worker_id, lf_ei_const_std, mfgp_lf_greedy, lf_acq) = args

    model_safe = model_name.replace(" ", "_")
    summary_file = output_dir / f'summary_{bench_name}_{model_safe}.csv'
    trajectory_file = output_dir / f'trajectory_{bench_name}_{model_safe}.csv'
    sampling_method = 'lhs' if bench_config['type'] == 'synthetic' else 'fps'

    # --- Resume: reuse any already-completed seeds for this (benchmark, model) ---
    results_summary = []
    results_trajectory = []
    done_seeds = set()
    if summary_file.exists():
        try:
            prev = pd.read_csv(summary_file)
            prev = prev[prev['seed'].isin(seeds)]
            ok = prev[prev['final_regret'].notna()]
            done_seeds = {int(s) for s in ok['seed'].tolist()}
            results_summary = ok.to_dict('records')
            if done_seeds and trajectory_file.exists():
                tprev = pd.read_csv(trajectory_file)
                tprev = tprev[tprev['seed'].isin(done_seeds)]
                results_trajectory = tprev.to_dict('records')
        except Exception as e:
            print(f"[Worker {worker_id}] {bench_name} + {model_name}: checkpoint "
                  f"unreadable ({e}); recomputing from scratch", flush=True)
            results_summary, results_trajectory, done_seeds = [], [], set()

    remaining = [s for s in seeds if int(s) not in done_seeds]

    if not remaining:
        # Everything already done — skip CUDA init and benchmark build entirely so
        # a resume-to-combine pass over a finished grid costs seconds, not hours.
        print(f"[Worker {worker_id}] {bench_name} + {model_name}: "
              f"{len(done_seeds)}/{len(seeds)} seeds cached — skipping", flush=True)
        return {
            'bench_name': bench_name, 'model_name': model_name,
            'n_seeds': len(done_seeds), 'elapsed': 0.0,
            'results_summary': results_summary,
            'results_trajectory': results_trajectory,
        }

    if torch.cuda.is_available():
        torch.cuda.init()
        device = torch.device('cuda:0')
        _ = torch.zeros(1).to(device)
    else:
        device = torch.device('cpu')

    print(f"[Worker {worker_id}] {bench_name} + {model_name}: Using {device} "
          f"({len(done_seeds)} cached, {len(remaining)} to run)", flush=True)

    # Create benchmark
    if bench_config['type'] == 'synthetic':
        benchmark = SyntheticBenchmark(
            bench_name,
            bench_config['hf_func'],
            bench_config['lf_func'],
            bench_config['dim'],
            bench_config['alpha'],
            bench_config['cost_ratio'],
            bench_config['f_star'],
            bench_config['grid_size']
        )
    else:
        benchmark = ChemistryBenchmark(
            bench_name,
            bench_config['csv_path'],
            bench_config['cost_ratio'],
            bench_config['use_smiles'],
            bench_config['minimize'],
            bench_config.get('negate', False)
        )

    start_time = time.time()

    for seed in remaining:
        seed_start = time.time()
        # Arm a 30-min per-seed alarm so a hung MFGP/MLP fit can't stall the grid.
        # Worker processes run with a single main thread, so signal.alarm is safe.
        prev_handler = signal.signal(signal.SIGALRM, _seed_timeout_handler)
        signal.alarm(_SEED_WALL_SEC)
        try:
            # Use LF-BLR BO loop
            result = run_bo_lf_blr(benchmark, model_class, budget, seed, device,
                                   sampling_method, lf_ei_const_std=lf_ei_const_std,
                                   mfgp_lf_greedy=mfgp_lf_greedy, lf_acq=lf_acq)
            seed_elapsed = time.time() - seed_start

            results_summary.append({
                'benchmark': bench_name,
                'model': model_name,
                'seed': seed,
                'final_regret': result['final_regret'],
                'n_hf': result['n_hf'],
                'n_lf': result['n_lf'],
                'best_y': result['best_y'],
                'elapsed_sec': round(seed_elapsed, 3),
                'n_exceptions': result['n_exceptions'],
            })

            for b, r in zip(result['budgets'], result['regrets']):
                results_trajectory.append({
                    'benchmark': bench_name,
                    'model': model_name,
                    'seed': seed,
                    'budget': round(b, 2),
                    'regret': r,
                })

        except (_SeedTimeoutError, Exception) as e:
            seed_elapsed = time.time() - seed_start
            print(f"[Worker {worker_id}] {bench_name} + {model_name} seed {seed} "
                  f"FAILED ({seed_elapsed:.1f}s): {e!r}", flush=True)
            results_summary.append({
                'benchmark': bench_name,
                'model': model_name,
                'seed': seed,
                'final_regret': np.nan,
                'n_hf': 0,
                'n_lf': 0,
                'best_y': np.nan,
                'elapsed_sec': round(seed_elapsed, 3),
                'n_exceptions': -1,  # -1 = the whole seed raised (no real BO)
            })
        finally:
            # Disarm the per-seed alarm and restore the prior handler — in a finally
            # so the alarm is ALWAYS cleared for this seed, and a timeout firing in the
            # narrow window after the try-body cannot escape this iteration uncaught.
            # (_SeedTimeoutError is a BaseException, so it is caught explicitly above
            # rather than swallowed by the per-iteration handler inside run_bo_lf_blr.)
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev_handler)

        # Checkpoint after every seed: an interrupted job resumes from here.
        # Write trajectory FIRST, then summary — summary is the resume completeness
        # marker, so a kill between the two never leaves a "done" seed without its curve.
        _atomic_write_csv(pd.DataFrame(results_trajectory), trajectory_file)
        _atomic_write_csv(pd.DataFrame(results_summary), summary_file)

    elapsed = time.time() - start_time

    return {
        'bench_name': bench_name,
        'model_name': model_name,
        'n_seeds': len(seeds),
        'elapsed': elapsed,
        'results_summary': results_summary,
        'results_trajectory': results_trajectory
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Parallel MF Benchmark with LF-BLR')
    parser.add_argument('--n-seeds', type=int, default=20, help='Number of seeds')
    parser.add_argument('--base-seed', type=int, default=42, help='Base seed')
    parser.add_argument('--n-workers', type=int, default=48, help='Number of parallel workers')
    parser.add_argument('--lf-ei-std', type=float, default=0.1,
                        help='Constant std for the LF EI (original published method, '
                             'default 0.1, reproduces paper_v1). Pass a NEGATIVE value '
                             '(e.g. -1) to use the BLR predictive std instead.')
    parser.add_argument('--mfgp-greedy', action='store_true',
                        help='Make MFGP greedy: its LF step picks argmin of the GP '
                             'posterior mean (mu only) instead of EI with the GP std. '
                             'Default off = published baseline (MFGP uses its GP UQ). '
                             'This is the Greedy-MFGP cell of the surrogate x acquisition '
                             '2x2; does NOT affect paper_v1 reproduction when off.')
    parser.add_argument('--lf-acq', type=str, default='ei',
                        choices=['ei', 'pi', 'ucb', 'mes', 'ts'],
                        help="LF acquisition for DNN surrogates (default 'ei' = paper "
                             'behaviour: greedy with --lf-ei-std 0.1, UQ-driven EI with '
                             '--lf-ei-std -1). pi=probability of improvement, ucb=GP '
                             'lower confidence bound, mes=max-value entropy search, '
                             'ts=Thompson sampling: these are UQ acquisitions that ALWAYS '
                             'use the BLR predictive std and form the 5-acquisition '
                             'portfolio (with UQ-EI) compared vs greedy. Only affects DNN '
                             'models; use one results dir per acquisition (VARIANT guard '
                             'enforces it). Does NOT affect paper_v1 reproduction when ei.')
    parser.add_argument('--models', nargs='+', default=None, metavar='NAME',
                        help='Subset of model names to run (default: all 12). Names must '
                             'match the keys in the models dict, e.g. --models MFGP. Lets '
                             'you fill a single cell (e.g. the Greedy-MFGP run) without '
                             'recomputing the rest of the grid.')
    parser.add_argument('--benchmarks', nargs='+', default=None, metavar='NAME',
                        help='Subset of benchmark names to run (default: all). Symmetric '
                             'with --models; names must match the keys in bench_configs, '
                             'e.g. --benchmarks HOPV15 Matbench-Gap. Lets you run only the '
                             'new chemistry benchmarks for a targeted verification without '
                             'recomputing the frozen 7. Pure task-filter: does not touch '
                             'the BO loop, acquisition, or any per-seed behavior.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to an existing results/run_* dir to resume into. '
                             'Completed (benchmark, model, seed) cells are loaded from '
                             'its per-task CSVs and skipped; only missing work runs. '
                             'If omitted, a new timestamped dir is created.')
    args = parser.parse_args()

    # Negative sentinel -> use BLR predictive std (the "true UQ" variant).
    lf_ei_const_std = None if args.lf_ei_std < 0 else args.lf_ei_std

    if args.resume:
        output_dir = Path(args.resume)
        if not output_dir.is_absolute():
            output_dir = Path(__file__).resolve().parent.parent / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"RESUMING into existing dir: {output_dir}")
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(__file__).resolve().parent.parent / 'results' / f'run_{timestamp}'
        output_dir.mkdir(parents=True, exist_ok=True)

    # Provenance guard (resume only): refuse to blend incompatible acquisition
    # variants into one results dir. The combined results_summary.csv has no per-row
    # variant tag, so resuming a const-std dir as BLR-std (or MFGP-EI as MFGP-greedy)
    # would silently mix cells of different methods. Compare the requested variant to
    # the existing VARIANT.txt; set ALLOW_VARIANT_MISMATCH=1 to override on purpose.
    if args.resume and (output_dir / 'VARIANT.txt').exists():
        _old = (output_dir / 'VARIANT.txt').read_text()
        _cur_std = 'const-std' if lf_ei_const_std is not None else 'BLR-std'
        _old_std = ('const-std' if 'const-std' in _old
                    else 'BLR-std' if 'BLR-std' in _old else None)
        _cur_greedy = bool(args.mfgp_greedy)
        # Older VARIANT.txt files predate the 'MFGP-LF=' token; skip that axis then.
        _old_greedy = ('greedy' in _old) if 'MFGP-LF=' in _old else None
        # Acquisition axis (LF-ACQ token). Older dirs predate it; skip then.
        _old_acq = (_old.split('LF-ACQ=', 1)[1].split()[0] if 'LF-ACQ=' in _old else None)
        _cur_acq = args.lf_acq
        _mismatch = ((_old_std is not None and _old_std != _cur_std) or
                     (_old_greedy is not None and _old_greedy != _cur_greedy) or
                     (_old_acq is not None and _old_acq != _cur_acq))
        if _mismatch and os.environ.get('ALLOW_VARIANT_MISMATCH') != '1':
            raise SystemExit(
                f"REFUSING to resume {output_dir}:\n"
                f"  existing VARIANT.txt : {_old.strip()}\n"
                f"  this run             : {_cur_std} | "
                f"MFGP-LF={'greedy' if _cur_greedy else 'EI'} | LF-ACQ={_cur_acq}\n"
                f"Resuming would blend incompatible acquisition variants in one dir "
                f"(the combined CSV has no per-row variant tag). Use a fresh results "
                f"dir, or set ALLOW_VARIANT_MISMATCH=1 to override intentionally.")

    # Tag the run NOW (not at the end) so a killed job is still identifiable.
    _variant = (f'const-std (lf_ei_const_std={lf_ei_const_std:g}) | reproduces paper_v1'
                if lf_ei_const_std is not None
                else 'BLR-std (lf_ei_const_std=None) | does NOT reproduce paper_v1')
    _mfgp = 'greedy (argmin mu)' if args.mfgp_greedy else 'EI (GP UQ, paper_v1)'
    (output_dir / 'VARIANT.txt').write_text(
        f"{_variant} | MFGP-LF={_mfgp} | LF-ACQ={args.lf_acq} | n_seeds={args.n_seeds} "
        f"base_seed={args.base_seed} "
        f"| {'resumed' if args.resume else 'started'} {datetime.now()}\n")

    data_dir = Path(__file__).resolve().parent.parent / 'data'

    print("=" * 80)
    print("Parallel Multi-Fidelity Benchmark with LF-BLR")
    print("=" * 80)
    print(f"Workers: {args.n_workers}")
    print(f"Seeds: {args.n_seeds}")
    print(f"Output: {output_dir}")
    print()
    print("LF-BLR Strategy:")
    print("  - LF selection: LF-BLR (EI) for exploration")
    print("  - HF selection: HF prediction (argmin) for exploitation")
    if args.mfgp_greedy:
        print("  - MFGP LF: GREEDY (argmin mu) — Greedy-MFGP control, NOT paper_v1 baseline")
    else:
        print("  - MFGP LF: Uses its own GP-based UQ via EI (paper_v1 baseline)")
    if lf_ei_const_std is not None:
        print(f"  - LF EI std: CONSTANT {lf_ei_const_std} (original published method, reproduces paper_v1)")
    else:
        print("  - LF EI std: BLR predictive std ('true UQ' variant — does NOT reproduce paper_v1)")
    if args.lf_acq == 'ei':
        print("  - LF acquisition: EI (paper baseline; greedy/UQ-EI set by --lf-ei-std)")
    else:
        print(f"  - LF acquisition: {args.lf_acq.upper()} (UQ portfolio variant, uses BLR std; "
              "extends the EI-vs-greedy comparison in sec:res-uq)")

    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Count: {torch.cuda.device_count()}")

    bench_configs = {
        'Branin-Fav': {
            'type': 'synthetic', 'hf_func': branin_hf, 'lf_func': branin_lf,
            'dim': 2, 'alpha': 0.8, 'cost_ratio': 0.1, 'f_star': 0.397887, 'grid_size': 50
        },
        'Branin-Unfav': {
            'type': 'synthetic', 'hf_func': branin_hf, 'lf_func': branin_lf,
            'dim': 2, 'alpha': 0.1, 'cost_ratio': 0.5, 'f_star': 0.397887, 'grid_size': 50
        },
        'Park-Fav': {
            'type': 'synthetic', 'hf_func': park_hf, 'lf_func': park_lf,
            'dim': 4, 'alpha': 0.6, 'cost_ratio': 0.1, 'f_star': 0.0, 'grid_size': 10
        },
        'Park-Unfav': {
            'type': 'synthetic', 'hf_func': park_hf, 'lf_func': park_lf,
            'dim': 4, 'alpha': 0.0, 'cost_ratio': 0.5, 'f_star': 0.0, 'grid_size': 10
        },
        'COFs': {
            'type': 'chemistry', 'csv_path': data_dir / 'cofs.csv',
            'cost_ratio': 0.065, 'use_smiles': False, 'minimize': True, 'negate': True
        },
        'FreeSolv': {
            'type': 'chemistry', 'csv_path': data_dir / 'freesolv.csv',
            'cost_ratio': 0.1, 'use_smiles': True, 'minimize': True, 'negate': False
        },
        'Polarizability': {
            'type': 'chemistry', 'csv_path': data_dir / 'polarizability.csv',
            'cost_ratio': 0.167, 'use_smiles': True, 'minimize': True, 'negate': True
        },
        # --- Additional chemistry benchmarks (same pipeline; see benchmarks/) ---
        # Featurized CSVs are produced by the loaders in benchmarks/ (run them
        # first: `python -m benchmarks.hopv15`, `python -m benchmarks.matbench_gap`).
        # They are plain numeric f0..f9 + HF + LF tables, so use_smiles=False and
        # ChemistryBenchmark consumes them exactly like COFs.
        'HOPV15': {  # experimental PCE (HF) vs Scharber PCE (LF); maximize -> negate
            'type': 'chemistry', 'csv_path': data_dir / 'hopv15.csv',
            'cost_ratio': 0.1, 'use_smiles': False, 'minimize': True, 'negate': True
        },
        'Matbench-Gap': {  # |Eg_expt-1.4| (HF) vs |Eg_pbe-1.4| (LF); minimize, no negate
            'type': 'chemistry', 'csv_path': data_dir / 'matbench_gap.csv',
            # cost_ratio raised 0.005 -> 0.05: at 0.005 the round-robin (200 LF/HF)
            # hit the frozen max_iter=500 cap after only ~2-4 HF evals; 0.05 (~20
            # LF/HF) gives a balanced ~10 HF evals. Mirrors benchmarks/matbench_gap.py.
            'cost_ratio': 0.05, 'use_smiles': False, 'minimize': True, 'negate': False
        },
        # 'Halide-Perovskite' is deferred (no clean PBE+expt CSV); see
        # benchmarks/halide_perovskite.py for how to wire it up once data exists.
    }

    budgets = {
        'Branin-Fav': 50, 'Branin-Unfav': 50,
        'Park-Fav': 50, 'Park-Unfav': 50,
        'COFs': 30, 'FreeSolv': 50, 'Polarizability': 30,
        'HOPV15': 30, 'Matbench-Gap': 20,
    }

    models = {
        'MFGP': MFGP,
        'Sequential': Sequential,
        'Progressive': Progressive,
        'Curriculum': Curriculum,
        'Two-Stage Joint': TwoStageJoint,
        'DNGO-Joint': DNGOJoint,
        'DNGO-Gradient': DNGOGradient,
        'Knowledge Distillation': KnowledgeDistillation,
        'Domain Adaptation (MMD)': DomainAdaptationMMD,
        'Soft Parameter Sharing': SoftParameterSharing,
        'Pseudo-Labeling': PseudoLabeling,
        'Adapter': Adapter,
    }

    if args.models:
        unknown = [m for m in args.models if m not in models]
        if unknown:
            raise SystemExit(f"Unknown model(s): {unknown}. Choose from: {list(models)}")
        models = {k: v for k, v in models.items() if k in args.models}
        print(f"Model filter active -> running only: {list(models)}")

    if args.benchmarks:
        unknown = [b for b in args.benchmarks if b not in bench_configs]
        if unknown:
            raise SystemExit(f"Unknown benchmark(s): {unknown}. Choose from: {list(bench_configs)}")
        bench_configs = {k: v for k, v in bench_configs.items() if k in args.benchmarks}
        print(f"Benchmark filter active -> running only: {list(bench_configs)}")

    seeds = [args.base_seed + i for i in range(args.n_seeds)]

    tasks = []
    worker_id = 0
    for bench_name, bench_config in bench_configs.items():
        for model_name, model_class in models.items():
            tasks.append((
                bench_name, bench_config, model_name, model_class,
                budgets[bench_name], seeds, output_dir, worker_id,
                lf_ei_const_std, args.mfgp_greedy, args.lf_acq
            ))
            worker_id += 1

    total_combinations = len(tasks)
    total_runs = total_combinations * args.n_seeds

    print(f"\nCombinations: {total_combinations} ({len(bench_configs)} benchmarks × {len(models)} models)")
    print(f"Total runs: {total_runs}")
    print("=" * 80)

    start_time = time.time()

    with Pool(processes=args.n_workers) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(run_combination, tasks)):
            results.append(result)
            elapsed = time.time() - start_time
            completed = i + 1
            eta = (elapsed / completed) * (total_combinations - completed) if completed > 0 else 0
            print(f"[{completed}/{total_combinations}] {result['bench_name']} + {result['model_name']}: "
                  f"{result['elapsed']:.1f}s ({result['n_seeds']} seeds) | ETA: {eta/60:.1f}min")

    total_time = time.time() - start_time

    # Rebuild the combined CSVs from the per-task checkpoints on disk so the result
    # reflects everything present — both resumed tasks and freshly-run ones. (Note:
    # 'results_summary.csv' does not match the 'summary_*.csv' glob, so no self-pickup.)
    def _read_nonempty(f):
        # A (benchmark, model) task whose seeds ALL failed writes a column-less
        # trajectory CSV; skip those so one all-failed task can't crash the whole
        # combine with EmptyDataError. Non-empty files read exactly as before.
        try:
            return pd.read_csv(f)
        except pd.errors.EmptyDataError:
            return None
    summ_frames = [d for d in map(_read_nonempty, sorted(output_dir.glob('summary_*.csv'))) if d is not None]
    traj_frames = [d for d in map(_read_nonempty, sorted(output_dir.glob('trajectory_*.csv'))) if d is not None]
    df_summary = (pd.concat(summ_frames, ignore_index=True)
                  if summ_frames else pd.DataFrame())
    df_trajectory = (pd.concat(traj_frames, ignore_index=True)
                     if traj_frames else pd.DataFrame())
    df_summary.to_csv(output_dir / 'results_summary.csv', index=False)
    df_trajectory.to_csv(output_dir / 'results_trajectory.csv', index=False)

    print("\n" + "=" * 80)
    print(f"Completed in {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Results saved to: {output_dir}")
    print(f"  - results_summary.csv: {len(df_summary)} rows (final metrics)")
    print(f"  - results_trajectory.csv: {len(df_trajectory)} rows (budget vs regret)")
    print("=" * 80)

    print("\nSUMMARY (Mean Final Regret)")
    print("-" * 60)
    for bench in (df_summary['benchmark'].unique() if 'benchmark' in df_summary.columns else []):
        print(f"\n{bench}:")
        summary = df_summary[df_summary['benchmark'] == bench].groupby('model')['final_regret'].agg(['mean', 'std'])
        summary = summary.sort_values('mean')
        for model, row in summary.head(3).iterrows():
            print(f"  {model:<30}: {row['mean']:.4f} ± {row['std']:.4f}")


if __name__ == '__main__':
    main()
