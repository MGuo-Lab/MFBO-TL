'Sparse variational and deep-kernel multi-fidelity Gaussian processes.'
import numpy as np
import torch
import torch.nn as nn
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.preprocessing import StandardScaler
from typing import Tuple

class _SparseMFGPModel(ApproximateGP):

    def __init__(self, inducing_points, input_dim):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=input_dim, active_dims=list(range(input_dim))))
        self.fidelity_kernel = gpytorch.kernels.IndexKernel(num_tasks=2, rank=1)

    def forward(self, x):
        features = x[..., :-1]
        fidelity_idx = x[..., -1].long()
        mean = self.mean_module(features)
        covar = self.covar_module(features) * self.fidelity_kernel(fidelity_idx)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class SparseMFGP:

    def __init__(self, input_dim: int, hidden_dim: int=64, device=None, n_inducing: int=100, n_epochs: int=500, lr: float=0.01):
        self.input_dim = input_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_inducing = n_inducing
        self.n_epochs = n_epochs
        self.lr = lr
        self.is_fitted = False

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        (n_lf, n_hf) = (len(X_lf), len(X_hf))
        X_lf_fid = np.hstack([X_lf, np.zeros((n_lf, 1))])
        X_hf_fid = np.hstack([X_hf, np.ones((n_hf, 1))])
        X_all = np.vstack([X_lf_fid, X_hf_fid])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_t = torch.tensor(X_all, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_all, dtype=torch.float32).to(self.device)
        n_ind = min(self.n_inducing, X_t.shape[0])
        idx = torch.randperm(X_t.shape[0])[:n_ind]
        inducing_points = X_t[idx].clone()
        self.model = _SparseMFGPModel(inducing_points, self.input_dim).to(self.device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam([{'params': self.model.parameters()}, {'params': self.likelihood.parameters()}], lr=self.lr)
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=X_t.shape[0])
        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            output = self.model(X_t)
            loss = -mll(output, y_t)
            loss.backward()
            optimizer.step()
        self.is_fitted = True

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X_fid = np.hstack([X, np.ones((len(X), 1))])
        X_t = torch.tensor(X_fid, dtype=torch.float32).to(self.device)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad():
            posterior = self.likelihood(self.model(X_t))
            mean = posterior.mean.cpu().numpy().flatten()
            std = posterior.variance.sqrt().cpu().numpy().flatten()
        return (mean, np.maximum(std, 1e-06))

    def predict_lf(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X_fid = np.hstack([X, np.zeros((len(X), 1))])
        X_t = torch.tensor(X_fid, dtype=torch.float32).to(self.device)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad():
            posterior = self.likelihood(self.model(X_t))
            mean = posterior.mean.cpu().numpy().flatten()
            std = posterior.variance.sqrt().cpu().numpy().flatten()
        return (mean, np.maximum(std, 1e-06))

class _FeatureExtractor(nn.Module):

    def __init__(self, input_dim, bottleneck_dim=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, bottleneck_dim))

    def forward(self, x):
        return self.net(x)

class _DKLMultiFidelityGP(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, input_dim, bottleneck_dim=16):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = _FeatureExtractor(input_dim, bottleneck_dim)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=bottleneck_dim))
        self.fidelity_kernel = gpytorch.kernels.IndexKernel(num_tasks=2, rank=1)
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

    def forward(self, x):
        features = x[..., :-1]
        fidelity_idx = x[..., -1].long()
        projected = self.scale_to_bounds(self.feature_extractor(features))
        mean = self.mean_module(projected)
        covar = self.covar_module(projected) * self.fidelity_kernel(fidelity_idx)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class DKLMultiFidelity:

    def __init__(self, input_dim: int, hidden_dim: int=64, device=None, n_epochs: int=500, bottleneck_dim: int=16):
        self.input_dim = input_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_epochs = n_epochs
        self.bottleneck_dim = bottleneck_dim
        self.is_fitted = False

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        (n_lf, n_hf) = (len(X_lf), len(X_hf))
        X_lf_fid = np.hstack([X_lf, np.zeros((n_lf, 1))])
        X_hf_fid = np.hstack([X_hf, np.ones((n_hf, 1))])
        X_all = np.vstack([X_lf_fid, X_hf_fid])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])
        X_t = torch.tensor(X_all, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_all, dtype=torch.float32).to(self.device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = _DKLMultiFidelityGP(X_t, y_t, self.likelihood, self.input_dim, self.bottleneck_dim).to(self.device)
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam([{'params': self.model.feature_extractor.parameters(), 'lr': 0.001}, {'params': self.model.covar_module.parameters(), 'lr': 0.01}, {'params': self.model.mean_module.parameters(), 'lr': 0.01}, {'params': self.model.fidelity_kernel.parameters(), 'lr': 0.01}, {'params': self.likelihood.parameters(), 'lr': 0.01}])
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            output = self.model(X_t)
            loss = -mll(output, y_t)
            loss.backward()
            optimizer.step()
        self.is_fitted = True

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X_fid = np.hstack([X, np.ones((len(X), 1))])
        X_t = torch.tensor(X_fid, dtype=torch.float32).to(self.device)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad():
            posterior = self.likelihood(self.model(X_t))
            mean = posterior.mean.cpu().numpy().flatten()
            std = posterior.variance.sqrt().cpu().numpy().flatten()
        return (mean, np.maximum(std, 1e-06))

    def predict_lf(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X_fid = np.hstack([X, np.zeros((len(X), 1))])
        X_t = torch.tensor(X_fid, dtype=torch.float32).to(self.device)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad():
            posterior = self.likelihood(self.model(X_t))
            mean = posterior.mean.cpu().numpy().flatten()
            std = posterior.variance.sqrt().cpu().numpy().flatten()
        return (mean, np.maximum(std, 1e-06))
