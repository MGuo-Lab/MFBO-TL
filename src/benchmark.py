'Surrogate models, pool loading and acquisition functions.'
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
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from synthetic_functions import branin_hf, branin_lf, park_hf, park_lf, SCENARIOS, FUNCTIONS

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

class BayesianLinearRegression:

    def __init__(self, alpha: float=1.0, beta: float=1.0):
        self.alpha = alpha
        self.beta = beta
        self.m_N = None
        self.S_N = None
        self.fitted = False

    def fit(self, Phi: np.ndarray, y: np.ndarray):
        Phi = np.asarray(Phi)
        y = np.asarray(y).flatten()
        (N, D) = Phi.shape
        Phi_bias = np.hstack([Phi, np.ones((N, 1))])
        D_bias = D + 1
        S_0_inv = self.alpha * np.eye(D_bias)
        S_N_inv = S_0_inv + self.beta * (Phi_bias.T @ Phi_bias)
        S_N_inv += 1e-06 * np.eye(D_bias)
        try:
            self.S_N = np.linalg.inv(S_N_inv)
        except np.linalg.LinAlgError:
            self.S_N = np.linalg.pinv(S_N_inv)
        self.m_N = self.beta * (self.S_N @ Phi_bias.T @ y)
        self.fitted = True

    def predict(self, Phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError('BLR not fitted yet')
        Phi = np.asarray(Phi)
        N = Phi.shape[0]
        Phi_bias = np.hstack([Phi, np.ones((N, 1))])
        mean = Phi_bias @ self.m_N
        var = np.zeros(N)
        for i in range(N):
            phi_i = Phi_bias[i:i + 1].T
            var[i] = 1.0 / self.beta + (phi_i.T @ self.S_N @ phi_i).item()
        std = np.sqrt(np.maximum(var, 1e-10))
        return (mean.flatten(), std.flatten())

class BaseModel:

    def __init__(self, input_dim: int, hidden_dim: int=64, device=None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lf_blr = None
        self.has_lf_blr = False

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        raise NotImplementedError

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def predict_lf(self, X) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _fit_lf_blr(self, X_lf_t, y_lf):
        if not hasattr(self, 'lf_net'):
            return
        self.lf_net.eval()
        with torch.no_grad():
            features = self.lf_net.extract_features(X_lf_t).cpu().numpy()
        self.lf_blr = BayesianLinearRegression(alpha=1.0, beta=1.0)
        self.lf_blr.fit(features, y_lf)
        self.has_lf_blr = True

class MFGP(BaseModel):

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        (n_lf, n_hf) = (len(X_lf), len(X_hf))
        X_lf_fid = np.hstack([X_lf, np.zeros((n_lf, 1))])
        X_hf_fid = np.hstack([X_hf, np.ones((n_hf, 1))])
        X_all = np.vstack([X_lf_fid, X_hf_fid])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        train_X = torch.tensor(X_all, dtype=torch.float64).to(self.device)
        train_Y = torch.tensor(y_all, dtype=torch.float64).unsqueeze(-1).to(self.device)
        self.model = SingleTaskMultiFidelityGP(train_X, train_Y, data_fidelities=[self.input_dim], outcome_transform=Standardize(m=1)).to(self.device)
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
        return (mean, np.maximum(std, 1e-06))

    def predict_lf(self, X) -> Tuple[np.ndarray, np.ndarray]:
        return self.predict(X)

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
        (X_lf_s, X_hf_s) = (X_scaled[:len(X_lf)], X_scaled[len(X_lf):])
        (y_lf_s, y_hf_s) = (y_scaled[:len(y_lf)], y_scaled[len(y_lf):])
        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)
        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=0.001)
        for _ in range(self.stage1_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()
        self._fit_lf_blr(X_lf_t, y_lf_s)
        opt = torch.optim.Adam(list(self.lf_net.parameters()) + list(self.hf_net.parameters()), lr=0.0001)
        for _ in range(self.stage2_epochs):
            opt.zero_grad()
            lf_loss = F.mse_loss(self.lf_net(X_lf_t), y_lf_t)
            y_lf_pred = self.lf_net(X_hf_t)
            hf_loss = F.mse_loss(self.hf_net(X_hf_t, y_lf_pred), y_hf_t)
            (0.3 * lf_loss + 0.7 * hf_loss).backward()
            opt.step()
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
        return (mean, np.ones_like(mean) * 0.1)

    def predict_lf(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()
        with torch.no_grad():
            features = self.lf_net.extract_features(X_t).cpu().numpy()
        if self.has_lf_blr:
            (mean_s, std_s) = self.lf_blr.predict(features)
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            std = std_s * self.scaler_y.scale_[0]
            return (mean, np.maximum(std, 1e-06))
        else:
            with torch.no_grad():
                mean_s = self.lf_net(X_t).cpu().numpy().flatten()
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            return (mean, np.ones_like(mean) * 0.1)

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
        (X_lf_s, X_hf_s) = (X_scaled[:len(X_lf)], X_scaled[len(X_lf):])
        (y_lf_s, y_hf_s) = (y_scaled[:len(y_lf)], y_scaled[len(y_lf):])
        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)
        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(list(self.lf_net.parameters()) + list(self.hf_net.parameters()), lr=0.001, weight_decay=0.0001)
        for _ in range(self.epochs):
            opt.zero_grad()
            lf_loss = F.mse_loss(self.lf_net(X_lf_t), y_lf_t)
            with torch.no_grad():
                y_lf_pred = self.lf_net(X_hf_t)
            hf_loss = F.mse_loss(self.hf_net(X_hf_t, y_lf_pred), y_hf_t)
            ((1 - self.alpha) * lf_loss + self.alpha * hf_loss).backward()
            opt.step()
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
        return (mean, np.ones_like(mean) * 0.1)

    def predict_lf(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()
        with torch.no_grad():
            features = self.lf_net.extract_features(X_t).cpu().numpy()
        if self.has_lf_blr:
            (mean_s, std_s) = self.lf_blr.predict(features)
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            std = std_s * self.scaler_y.scale_[0]
            return (mean, np.maximum(std, 1e-06))
        else:
            with torch.no_grad():
                mean_s = self.lf_net(X_t).cpu().numpy().flatten()
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            return (mean, np.ones_like(mean) * 0.1)

class DNGOGradient(BaseModel):

    def __init__(self, input_dim, hidden_dim=64, epochs=300, device=None):
        super().__init__(input_dim, hidden_dim, device)
        self.epochs = epochs

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        X_all = np.vstack([X_lf, X_hf])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_scaled = self.scaler_x.fit_transform(X_all)
        y_scaled = self.scaler_y.fit_transform(y_all.reshape(-1, 1)).flatten()
        (X_lf_s, X_hf_s) = (X_scaled[:len(X_lf)], X_scaled[len(X_lf):])
        (y_lf_s, y_hf_s) = (y_scaled[:len(y_lf)], y_scaled[len(y_lf):])
        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)
        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam([{'params': self.lf_net.parameters(), 'lr': 0.001}, {'params': self.hf_net.parameters(), 'lr': 0.0005}], weight_decay=0.0001)
        for _ in range(self.epochs):
            opt.zero_grad()
            lf_loss = F.mse_loss(self.lf_net(X_lf_t), y_lf_t)
            y_lf_pred = self.lf_net(X_hf_t)
            hf_loss = F.mse_loss(self.hf_net(X_hf_t, y_lf_pred), y_hf_t)
            (lf_loss + hf_loss).backward()
            opt.step()
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
        return (mean, np.ones_like(mean) * 0.1)

    def predict_lf(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()
        with torch.no_grad():
            features = self.lf_net.extract_features(X_t).cpu().numpy()
        if self.has_lf_blr:
            (mean_s, std_s) = self.lf_blr.predict(features)
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            std = std_s * self.scaler_y.scale_[0]
            return (mean, np.maximum(std, 1e-06))
        else:
            with torch.no_grad():
                mean_s = self.lf_net(X_t).cpu().numpy().flatten()
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            return (mean, np.ones_like(mean) * 0.1)

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
        (X_lf_s, X_hf_s) = (X_scaled[:len(X_lf)], X_scaled[len(X_lf):])
        (y_lf_s, y_hf_s) = (y_scaled[:len(y_lf)], y_scaled[len(y_lf):])
        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)
        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=0.001)
        for _ in range(self.lf_epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(X_lf_t), y_lf_t).backward()
            opt.step()
        self._fit_lf_blr(X_lf_t, y_lf_s)
        opt = torch.optim.Adam([{'params': self.lf_net.parameters(), 'lr': 0.0001}, {'params': self.hf_net.parameters(), 'lr': 0.0001}])
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
        return (mean, np.ones_like(mean) * 0.1)

    def predict_lf(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()
        with torch.no_grad():
            features = self.lf_net.extract_features(X_t).cpu().numpy()
        if self.has_lf_blr:
            (mean_s, std_s) = self.lf_blr.predict(features)
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            std = std_s * self.scaler_y.scale_[0]
            return (mean, np.maximum(std, 1e-06))
        else:
            with torch.no_grad():
                mean_s = self.lf_net(X_t).cpu().numpy().flatten()
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            return (mean, np.ones_like(mean) * 0.1)

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
        (X_lf_s, X_hf_s) = (X_scaled[:len(X_lf)], X_scaled[len(X_lf):])
        (y_lf_s, y_hf_s) = (y_scaled[:len(y_lf)], y_scaled[len(y_lf):])
        X_lf_t = torch.FloatTensor(X_lf_s).to(self.device)
        y_lf_t = torch.FloatTensor(y_lf_s).view(-1, 1).to(self.device)
        X_hf_t = torch.FloatTensor(X_hf_s).to(self.device)
        y_hf_t = torch.FloatTensor(y_hf_s).view(-1, 1).to(self.device)
        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(list(self.lf_net.parameters()) + list(self.hf_net.parameters()), lr=0.001)
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
        return (mean, np.ones_like(mean) * 0.1)

    def predict_lf(self, X):
        X_s = self.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(self.device)
        self.lf_net.eval()
        with torch.no_grad():
            features = self.lf_net.extract_features(X_t).cpu().numpy()
        if self.has_lf_blr:
            (mean_s, std_s) = self.lf_blr.predict(features)
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            std = std_s * self.scaler_y.scale_[0]
            return (mean, np.maximum(std, 1e-06))
        else:
            with torch.no_grad():
                mean_s = self.lf_net(X_t).cpu().numpy().flatten()
            mean = self.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
            return (mean, np.ones_like(mean) * 0.1)

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
        for (i, smi) in enumerate(smiles_list):
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
        pca = PCA(n_components=self.pca_dim, random_state=42)
        features_pca = pca.fit_transform(features_scaled)
        return features_pca

    def evaluate_hf(self, indices):
        return self.y_hf[indices.astype(int).flatten()]

    def evaluate_lf(self, indices):
        return self.y_lf[indices.astype(int).flatten()]

def furthest_point_sampling(X: np.ndarray, n_samples: int, seed: int=42) -> np.ndarray:
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

def expected_improvement(mean, std, y_best, xi=0.01):
    with np.errstate(divide='ignore', invalid='ignore'):
        imp = y_best - mean - xi
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std < 1e-10] = 0.0
    return ei

def probability_of_improvement(mean, std, y_best, xi=0.01):
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = (y_best - mean - xi) / std
        pi = norm.cdf(Z)
        pi[std < 1e-10] = 0.0
    return pi

def lower_confidence_bound(mean, std, beta=2.0):
    return beta * std - mean

def thompson_sample(mean, std, rng):
    sample = mean + std * rng.standard_normal(size=mean.shape)
    return -sample

def _sample_min_values_gumbel(mean, std, n_samples, rng, n_grid=200):
    m = -mean
    s = np.maximum(std, 1e-09)
    lo = float(m.max())
    hi = float((m + 5.0 * s).max())
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return np.full(n_samples, float(mean.min()))
    grid = np.linspace(lo, hi, n_grid)
    Z = (grid[:, None] - m[None, :]) / s[None, :]
    F = np.exp(norm.logcdf(Z).sum(axis=1))

    def _quantile(r):
        idx = int(np.clip(np.searchsorted(F, r), 1, n_grid - 1))
        (f0, f1) = (F[idx - 1], F[idx])
        if f1 <= f0:
            return grid[idx]
        w = (r - f0) / (f1 - f0)
        return grid[idx - 1] + w * (grid[idx] - grid[idx - 1])
    (y1, y2) = (_quantile(0.25), _quantile(0.75))
    (L1, L2) = (np.log(-np.log(0.25)), np.log(-np.log(0.75)))
    b = (y2 - y1) / (L1 - L2)
    u = np.clip(rng.random(n_samples), 1e-06, 1 - 1e-06)
    if not np.isfinite(b) or b <= 0:
        gstar = np.interp(u, F, grid)
    else:
        a = y1 + b * L1
        gstar = a - b * np.log(-np.log(u))
    return -gstar

def max_value_entropy_search(mean, std, rng, n_samples=10):
    s = np.maximum(std, 1e-09)
    fstar = _sample_min_values_gumbel(mean, std, n_samples, rng)
    acq = np.zeros_like(mean, dtype=float)
    for ys in fstar:
        gamma = (mean - ys) / s
        Psi = np.clip(norm.cdf(gamma), 1e-10, 1.0)
        acq += gamma * norm.pdf(gamma) / (2.0 * Psi) - np.log(Psi)
    return acq / len(fstar)
