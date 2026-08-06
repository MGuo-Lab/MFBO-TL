"""
Baselines for ICML 2026 rebuttal:
1. SparseMFGP (SVGP with IndexKernel)
2. NARGP (Nonlinear Autoregressive GP)
3. DKLMultiFidelity (Deep Kernel Learning + Multi-Fidelity)
4. SuccessiveHalving (surrogate-free baseline)
5. HF-Only Random Search (non-learning baseline)
6. LF-Screening (non-learning, surrogate-free MF baseline)
"""

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


# =============================================================================
# Baseline 1: Sparse MFGP (SVGP)
# =============================================================================

class _SparseMFGPModel(ApproximateGP):
    def __init__(self, inducing_points, input_dim):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim, active_dims=list(range(input_dim)))
        )
        self.fidelity_kernel = gpytorch.kernels.IndexKernel(num_tasks=2, rank=1)

    def forward(self, x):
        features = x[..., :-1]
        fidelity_idx = x[..., -1].long()
        mean = self.mean_module(features)
        covar = self.covar_module(features) * self.fidelity_kernel(fidelity_idx)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class SparseMFGP:
    """Sparse Multi-Fidelity GP using SVGP with IndexKernel."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, device=None,
                 n_inducing: int = 100, n_epochs: int = 500, lr: float = 0.01):
        self.input_dim = input_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_inducing = n_inducing
        self.n_epochs = n_epochs
        self.lr = lr
        self.is_fitted = False

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        n_lf, n_hf = len(X_lf), len(X_hf)
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

        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=self.lr)
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=X_t.shape[0])

        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            output = self.model(X_t)
            loss = -mll(output, y_t)
            loss.backward()
            optimizer.step()

        self.is_fitted = True

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X_fid = np.hstack([X, np.ones((len(X), 1))])  # HF fidelity=1
        X_t = torch.tensor(X_fid, dtype=torch.float32).to(self.device)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad():
            posterior = self.likelihood(self.model(X_t))
            mean = posterior.mean.cpu().numpy().flatten()
            std = posterior.variance.sqrt().cpu().numpy().flatten()
        return mean, np.maximum(std, 1e-6)

    def predict_lf(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X_fid = np.hstack([X, np.zeros((len(X), 1))])  # LF fidelity=0
        X_t = torch.tensor(X_fid, dtype=torch.float32).to(self.device)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad():
            posterior = self.likelihood(self.model(X_t))
            mean = posterior.mean.cpu().numpy().flatten()
            std = posterior.variance.sqrt().cpu().numpy().flatten()
        return mean, np.maximum(std, 1e-6)


# =============================================================================
# Baseline 2: NARGP (Nonlinear Autoregressive GP)
# =============================================================================

class NARGP:
    """Nonlinear Autoregressive GP (Perdikaris et al., 2017).
    Two-stage: LF GP -> augment HF input with LF predictions -> HF GP.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, device=None):
        self.input_dim = input_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_fitted = False

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        dtype = torch.float64
        X_lf_t = torch.tensor(X_lf, dtype=dtype).to(self.device)
        y_lf_t = torch.tensor(y_lf.flatten(), dtype=dtype).unsqueeze(-1).to(self.device)
        X_hf_t = torch.tensor(X_hf, dtype=dtype).to(self.device)
        y_hf_t = torch.tensor(y_hf.flatten(), dtype=dtype).unsqueeze(-1).to(self.device)

        # Stage 1: LF GP
        self.gp_lf = SingleTaskGP(X_lf_t, y_lf_t).to(self.device)
        mll_lf = ExactMarginalLogLikelihood(self.gp_lf.likelihood, self.gp_lf)
        fit_gpytorch_mll(mll_lf)

        # Stage 2: HF GP with augmented input [X_hf, mu_lf(X_hf)]
        self.gp_lf.eval()
        with torch.no_grad():
            lf_mean = self.gp_lf.posterior(X_hf_t).mean  # (N_hf, 1)
        X_hf_aug = torch.cat([X_hf_t, lf_mean], dim=-1)

        self.gp_hf = SingleTaskGP(X_hf_aug, y_hf_t).to(self.device)
        mll_hf = ExactMarginalLogLikelihood(self.gp_hf.likelihood, self.gp_hf)
        fit_gpytorch_mll(mll_hf)

        self.is_fitted = True

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        dtype = torch.float64
        X_t = torch.tensor(X, dtype=dtype).to(self.device)
        self.gp_lf.eval()
        self.gp_hf.eval()
        with torch.no_grad():
            lf_mean = self.gp_lf.posterior(X_t).mean
            X_aug = torch.cat([X_t, lf_mean], dim=-1)
            hf_post = self.gp_hf.posterior(X_aug)
            mean = hf_post.mean.cpu().numpy().flatten()
            std = hf_post.variance.sqrt().cpu().numpy().flatten()
        return mean, np.maximum(std, 1e-6)

    def predict_lf(self, X) -> Tuple[np.ndarray, np.ndarray]:
        dtype = torch.float64
        X_t = torch.tensor(X, dtype=dtype).to(self.device)
        self.gp_lf.eval()
        with torch.no_grad():
            lf_post = self.gp_lf.posterior(X_t)
            mean = lf_post.mean.cpu().numpy().flatten()
            std = lf_post.variance.sqrt().cpu().numpy().flatten()
        return mean, np.maximum(std, 1e-6)


# =============================================================================
# Baseline 3: DKL Multi-Fidelity
# =============================================================================

class _FeatureExtractor(nn.Module):
    """2-layer MLP matching existing DNN surrogate architecture."""
    def __init__(self, input_dim, bottleneck_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, bottleneck_dim),
        )

    def forward(self, x):
        return self.net(x)


class _DKLMultiFidelityGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, input_dim, bottleneck_dim=16):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = _FeatureExtractor(input_dim, bottleneck_dim)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=bottleneck_dim)
        )
        self.fidelity_kernel = gpytorch.kernels.IndexKernel(num_tasks=2, rank=1)
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        features = x[..., :-1]
        fidelity_idx = x[..., -1].long()
        projected = self.scale_to_bounds(self.feature_extractor(features))
        mean = self.mean_module(projected)
        covar = self.covar_module(projected) * self.fidelity_kernel(fidelity_idx)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class DKLMultiFidelity:
    """Deep Kernel Learning with Multi-Fidelity via IndexKernel."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, device=None,
                 n_epochs: int = 500, bottleneck_dim: int = 16):
        self.input_dim = input_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_epochs = n_epochs
        self.bottleneck_dim = bottleneck_dim
        self.is_fitted = False

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        n_lf, n_hf = len(X_lf), len(X_hf)
        X_lf_fid = np.hstack([X_lf, np.zeros((n_lf, 1))])
        X_hf_fid = np.hstack([X_hf, np.ones((n_hf, 1))])
        X_all = np.vstack([X_lf_fid, X_hf_fid])
        y_all = np.concatenate([y_lf.flatten(), y_hf.flatten()])

        X_t = torch.tensor(X_all, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_all, dtype=torch.float32).to(self.device)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = _DKLMultiFidelityGP(
            X_t, y_t, self.likelihood, self.input_dim, self.bottleneck_dim
        ).to(self.device)
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': self.model.feature_extractor.parameters(), 'lr': 1e-3},
            {'params': self.model.covar_module.parameters(), 'lr': 1e-2},
            {'params': self.model.mean_module.parameters(), 'lr': 1e-2},
            {'params': self.model.fidelity_kernel.parameters(), 'lr': 1e-2},
            {'params': self.likelihood.parameters(), 'lr': 1e-2},
        ])
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            output = self.model(X_t)
            loss = -mll(output, y_t)
            loss.backward()
            optimizer.step()

        self.is_fitted = True

    def predict(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X_fid = np.hstack([X, np.ones((len(X), 1))])  # HF fidelity=1
        X_t = torch.tensor(X_fid, dtype=torch.float32).to(self.device)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad():
            posterior = self.likelihood(self.model(X_t))
            mean = posterior.mean.cpu().numpy().flatten()
            std = posterior.variance.sqrt().cpu().numpy().flatten()
        return mean, np.maximum(std, 1e-6)

    def predict_lf(self, X) -> Tuple[np.ndarray, np.ndarray]:
        X_fid = np.hstack([X, np.zeros((len(X), 1))])  # LF fidelity=0
        X_t = torch.tensor(X_fid, dtype=torch.float32).to(self.device)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad():
            posterior = self.likelihood(self.model(X_t))
            mean = posterior.mean.cpu().numpy().flatten()
            std = posterior.variance.sqrt().cpu().numpy().flatten()
        return mean, np.maximum(std, 1e-6)


# =============================================================================
# New method: Transfer-Learned Deep Kernel (TLDK)
# =============================================================================

class TLDeepKernel:
    """Transfer-Learned Deep Kernel (TLDK).

    The transfer-learning surrogate (LF network -> HF residual network) supplies
    a transferable representation; a Gaussian-process head fitted on the
    transfer-learned HF features supplies a calibrated posterior. This fuses the
    TL transfer architecture (the representation engine) with DKL's GP readout,
    completing the dual-fidelity UQ the released TL models lacked (their HF head
    returned a constant dummy std of 0.1). LF acquisition uses the TL model's
    LF-BLR head; HF mean and uncertainty come from the GP head, enabling
    principled EI at the HF fidelity.
    """

    def __init__(self, input_dim, hidden_dim=64, device=None):
        from benchmark import Sequential
        self._tl = Sequential(input_dim, hidden_dim, device=device)
        self.device = self._tl.device
        self.input_dim = input_dim
        self._gp_ok = False
        self.is_fitted = False

    def _hf_features(self, X):
        tl = self._tl
        X_s = tl.scaler_x.transform(X)
        X_t = torch.FloatTensor(X_s).to(tl.device)
        tl.lf_net.eval(); tl.hf_net.eval()
        with torch.no_grad():
            y_lf = tl.lf_net(X_t)
            feats = tl.hf_net.extract_features(X_t, y_lf)
        return feats.double().cpu()

    def _tl_hf_pred_scaled(self, X):
        """Transfer-net HF prediction in scaled space (the GP head's mean fn)."""
        tl = self._tl
        X_t = torch.FloatTensor(tl.scaler_x.transform(X)).to(tl.device)
        tl.lf_net.eval(); tl.hf_net.eval()
        with torch.no_grad():
            y_lf = tl.lf_net(X_t)
            y_hf = tl.hf_net(X_t, y_lf)
        return y_hf.cpu().numpy().flatten()

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        from gpytorch.kernels import ScaleKernel, RBFKernel
        self._tl.fit(X_lf, y_lf, X_hf, y_hf)
        feats = self._hf_features(X_hf)
        y_hf_s = self._tl.scaler_y.transform(
            np.asarray(y_hf, dtype=float).reshape(-1, 1)).flatten()
        # The GP head models the RESIDUAL around the transfer-net HF prediction.
        # With few HF points the residual GP -> 0, so the prediction degrades
        # gracefully to the (informative) transfer prediction instead of being
        # replaced by an underdetermined GP mean (the V1 failure on COFs). The
        # residual std still supplies a calibrated HF posterior for EI.
        tl_pred_s = self._tl_hf_pred_scaled(X_hf)
        resid = y_hf_s - tl_pred_s
        train_y = torch.tensor(resid, dtype=torch.double).unsqueeze(-1)
        self._gp_ok = False
        try:
            gp = SingleTaskGP(feats, train_y,
                              covar_module=ScaleKernel(RBFKernel()))
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
            gp.eval()
            self._gp = gp
            self._gp_ok = True
        except Exception as e:
            print(f"[TLDK] GP head fit failed, falling back to TL HF mean: {e}",
                  flush=True)
        self.is_fitted = True
        return self

    def predict(self, X):
        tl_pred_s = self._tl_hf_pred_scaled(X)
        if not self._gp_ok:
            return self._tl.predict(X)
        feats = self._hf_features(X)
        with torch.no_grad():
            post = self._gp.posterior(feats)
            resid_mean = post.mean.squeeze(-1).cpu().numpy()
            std_s = post.variance.clamp_min(1e-12).sqrt().squeeze(-1).cpu().numpy()
        mean_s = tl_pred_s + resid_mean
        mean = self._tl.scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
        std = std_s * self._tl.scaler_y.scale_[0]
        return mean, np.maximum(std, 1e-6)

    def predict_lf(self, X):
        return self._tl.predict_lf(X)


class TLDeepKernelJoint:
    """Transfer-initialised deep-kernel multi-fidelity GP (TLDK-Joint).

    The deep-kernel feature extractor is PRETRAINED on low-fidelity data (the
    transfer step), then the whole deep kernel (features + multi-fidelity GP) is
    jointly trained on LF+HF via the marginal likelihood. This keeps DKL's
    GP-friendly jointly-learned features (its win on smooth problems) while using
    LF transfer as a better representation init (data efficiency). Transfer
    learning is the engine: the LF-pretrained representation initialises the
    kernel.
    """

    def __init__(self, input_dim, hidden_dim=64, device=None, n_epochs=500,
                 bottleneck_dim=16, pretrain_epochs=200):
        self.input_dim = input_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_epochs = n_epochs
        self.bottleneck_dim = bottleneck_dim
        self.pretrain_epochs = pretrain_epochs
        self.is_fitted = False

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        # 1) TRANSFER: pretrain the feature extractor on LF (predict y_lf)
        fe = _FeatureExtractor(self.input_dim, self.bottleneck_dim).to(self.device)
        head = nn.Linear(self.bottleneck_dim, 1).to(self.device)
        Xlf_t = torch.tensor(np.asarray(X_lf), dtype=torch.float32).to(self.device)
        ylf = np.asarray(y_lf, dtype=float)
        ylf_s = (ylf - ylf.mean()) / (ylf.std() + 1e-12)
        ylf_t = torch.tensor(ylf_s, dtype=torch.float32).unsqueeze(-1).to(self.device)
        opt = torch.optim.Adam(list(fe.parameters()) + list(head.parameters()), lr=1e-3)
        for _ in range(self.pretrain_epochs):
            opt.zero_grad()
            torch.nn.functional.mse_loss(head(fe(Xlf_t)), ylf_t).backward()
            opt.step()
        pretrained = {k: v.detach().clone() for k, v in fe.state_dict().items()}

        # 2) build the MF deep-kernel GP, warm-start its extractor, joint-train
        n_lf, n_hf = len(X_lf), len(X_hf)
        X_lf_fid = np.hstack([X_lf, np.zeros((n_lf, 1))])
        X_hf_fid = np.hstack([X_hf, np.ones((n_hf, 1))])
        X_all = np.vstack([X_lf_fid, X_hf_fid])
        y_all = np.concatenate([np.asarray(y_lf).flatten(), np.asarray(y_hf).flatten()])
        X_t = torch.tensor(X_all, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_all, dtype=torch.float32).to(self.device)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = _DKLMultiFidelityGP(
            X_t, y_t, self.likelihood, self.input_dim, self.bottleneck_dim).to(self.device)
        self.model.feature_extractor.load_state_dict(pretrained)   # transfer init
        self.model.train(); self.likelihood.train()
        optimizer = torch.optim.Adam([
            {'params': self.model.feature_extractor.parameters(), 'lr': 1e-3},
            {'params': self.model.covar_module.parameters(), 'lr': 1e-2},
            {'params': self.model.mean_module.parameters(), 'lr': 1e-2},
            {'params': self.model.fidelity_kernel.parameters(), 'lr': 1e-2},
            {'params': self.likelihood.parameters(), 'lr': 1e-2},
        ])
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            loss = -mll(self.model(X_t), y_t)
            loss.backward()
            optimizer.step()
        self.is_fitted = True
        return self

    def _post(self, X, fidelity):
        X_fid = np.hstack([X, np.full((len(X), 1), float(fidelity))])
        X_t = torch.tensor(X_fid, dtype=torch.float32).to(self.device)
        self.model.eval(); self.likelihood.eval()
        with torch.no_grad():
            post = self.likelihood(self.model(X_t))
            mean = post.mean.cpu().numpy().flatten()
            std = post.variance.sqrt().cpu().numpy().flatten()
        return mean, np.maximum(std, 1e-6)

    def predict(self, X):
        return self._post(X, 1.0)     # HF

    def predict_lf(self, X):
        return self._post(X, 0.0)     # LF


class DV_HF:
    """Discrepancy-guided, confidence-weighted Virtual-HF Transfer (DV-HF).

    Inspired by PseudoLabeling (regularise the HF net with virtual HF labels
    generated over the abundant LF pool), with three fixes to PseudoLabeling's
    weaknesses:
      (1) global mean offset  ->  input-dependent discrepancy delta(x), a
          BayesianRidge fitted on the HF residual y_hf - mu_LF(x) in LF-feature
          space (Kennedy-O'Hagan additive form). Virtual label y~(x)=mu_LF+delta.
      (2) uniform pseudo weight -> per-point confidence weight
          w(x) = 1/(sigma_LF^2 + sigma_delta^2) from the LF-BLR + discrepancy
          posteriors (down-weights virtual labels where LF extrapolates).
      (3) fixed lambda -> budget-adaptive lambda = lambda0 * n0/(n0 + n_HF)
          (lean on virtual data when HF is scarce, decay as HF accrues).
    sigma_delta(x) also gives a non-dummy HF posterior std -> EI usable at HF.
    The DNN HF net remains the engine (distinct from the GP-head TLDK variants).
    """

    def __init__(self, input_dim, hidden_dim=64, device=None, lf_epochs=200,
                 hf_epochs=100, lambda0=1.0, n0=3.0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lf_epochs = lf_epochs
        self.hf_epochs = hf_epochs
        self.lambda0 = lambda0
        self.n0 = n0
        self.sx = StandardScaler()
        self.sy = StandardScaler()
        self.is_fitted = False

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        import torch.nn.functional as F
        from benchmark import LFNetwork, HFNetwork, BayesianLinearRegression
        from sklearn.linear_model import BayesianRidge
        Xall = np.vstack([X_lf, X_hf])
        yall = np.concatenate([np.asarray(y_lf).ravel(), np.asarray(y_hf).ravel()])
        Xs = self.sx.fit_transform(Xall); ys = self.sy.fit_transform(yall.reshape(-1, 1)).ravel()
        nlf = len(X_lf)
        Xlf_t = torch.FloatTensor(Xs[:nlf]).to(self.device)
        ylf_s = ys[:nlf]; yhf_s = ys[nlf:]
        ylf_t = torch.FloatTensor(ylf_s).view(-1, 1).to(self.device)
        Xhf_t = torch.FloatTensor(Xs[nlf:]).to(self.device)
        yhf_t = torch.FloatTensor(yhf_s).view(-1, 1).to(self.device)
        # --- LF net ---
        self.lf_net = LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=1e-3)
        for _ in range(self.lf_epochs):
            opt.zero_grad(); F.mse_loss(self.lf_net(Xlf_t), ylf_t).backward(); opt.step()
        self.lf_net.eval()
        with torch.no_grad():
            feat_lf = self.lf_net.extract_features(Xlf_t).cpu().numpy()
            feat_hf = self.lf_net.extract_features(Xhf_t).cpu().numpy()
            mu_lf_hf = self.lf_net(Xhf_t).cpu().numpy().ravel()
            mu_lf_lf = self.lf_net(Xlf_t).cpu().numpy().ravel()
        # --- LF-BLR (mu_LF, sigma_LF) ---
        self.lf_blr = BayesianLinearRegression(); self.lf_blr.fit(feat_lf, ylf_s)
        # --- (1) input-dependent discrepancy delta(x) on LF features ---
        self.delta = BayesianRidge()
        self.delta.fit(feat_hf, yhf_s - mu_lf_hf)
        dhat_lf, dstd_lf = self.delta.predict(feat_lf, return_std=True)
        ytilde = torch.FloatTensor(mu_lf_lf + dhat_lf).view(-1, 1).to(self.device)
        # --- (2) confidence weight w(x) ---
        _, slf = self.lf_blr.predict(feat_lf)
        w = 1.0 / (slf ** 2 + dstd_lf ** 2 + 1e-6); w = w / w.mean()
        w_t = torch.FloatTensor(w).view(-1, 1).to(self.device)
        # --- (3) budget-adaptive lambda ---
        lam = self.lambda0 * self.n0 / (self.n0 + len(X_hf))
        # --- HF net: real + lam * weighted virtual ---
        self.hf_net = HFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        for p in self.lf_net.parameters():
            p.requires_grad = False
        opt = torch.optim.Adam(self.hf_net.parameters(), lr=1e-4)
        for _ in range(self.hf_epochs):
            opt.zero_grad()
            with torch.no_grad():
                ylf_for_hf = self.lf_net(Xhf_t); ylf_for_lf = self.lf_net(Xlf_t)
            real = F.mse_loss(self.hf_net(Xhf_t, ylf_for_hf), yhf_t)
            pseudo = (w_t * (self.hf_net(Xlf_t, ylf_for_lf) - ytilde) ** 2).mean()
            (real + lam * pseudo).backward(); opt.step()
        self.is_fitted = True
        return self

    def predict(self, X):
        Xs = self.sx.transform(X); Xt = torch.FloatTensor(Xs).to(self.device)
        self.lf_net.eval(); self.hf_net.eval()
        with torch.no_grad():
            ylf = self.lf_net(Xt); mean_s = self.hf_net(Xt, ylf).cpu().numpy().ravel()
            feat = self.lf_net.extract_features(Xt).cpu().numpy()
        _, dstd = self.delta.predict(feat, return_std=True)
        mean = self.sy.inverse_transform(mean_s.reshape(-1, 1)).ravel()
        return mean, np.maximum(dstd * self.sy.scale_[0], 1e-6)

    def predict_lf(self, X):
        Xs = self.sx.transform(X); Xt = torch.FloatTensor(Xs).to(self.device)
        self.lf_net.eval()
        with torch.no_grad():
            feat = self.lf_net.extract_features(Xt).cpu().numpy()
        m, s = self.lf_blr.predict(feat)
        mean = self.sy.inverse_transform(m.reshape(-1, 1)).ravel()
        return mean, np.maximum(s * self.sy.scale_[0], 1e-6)


class HFBLR_TL:
    """Transfer-learning surrogate with a REAL calibrated HF posterior.

    The released TL models have LF-mean, an LF-BLR head (calibrated LF posterior),
    but only an HF-MEAN with a dummy constant std (0.1) -> EI at HF collapses to
    argmin. This adds the missing HF-BLR head: a Bayesian linear regression on the
    HF network's last-layer features, giving a genuine HF posterior (mean, std).
    This is the dual-BLR the paper intended (Methods s2.3.2 / Appendix B) and the
    prerequisite the skeptic flagged for matched HF acquisition: now HF EI is a
    real manipulation on the TL arm. LF acquisition still uses the LF-BLR head.
    """

    def __init__(self, input_dim, hidden_dim=64, device=None, base='Curriculum'):
        from benchmark import Sequential, Curriculum
        cls = {'Sequential': Sequential, 'Curriculum': Curriculum}[base]
        self._tl = cls(input_dim, hidden_dim, device=device)
        self.device = self._tl.device
        self.input_dim = input_dim
        self.is_fitted = False

    def _hf_features(self, X):
        import torch
        tl = self._tl
        X_t = torch.FloatTensor(tl.scaler_x.transform(X)).to(tl.device)
        tl.lf_net.eval(); tl.hf_net.eval()
        with torch.no_grad():
            y_lf = tl.lf_net(X_t)
            feat = tl.hf_net.extract_features(X_t, y_lf).cpu().numpy()
        return feat

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        from benchmark import BayesianLinearRegression
        self._tl.fit(X_lf, y_lf, X_hf, y_hf)
        feat_hf = self._hf_features(X_hf)
        y_hf_s = self._tl.scaler_y.transform(np.asarray(y_hf, dtype=float).reshape(-1, 1)).ravel()
        self.hf_blr = BayesianLinearRegression()
        self.hf_blr.fit(feat_hf, y_hf_s)
        self.is_fitted = True
        return self

    def predict(self, X):
        m, s = self.hf_blr.predict(self._hf_features(X))
        mean = self._tl.scaler_y.inverse_transform(m.reshape(-1, 1)).ravel()
        std = s * self._tl.scaler_y.scale_[0]
        return mean, np.maximum(std, 1e-6)

    def predict_lf(self, X):
        return self._tl.predict_lf(X)


class CalibratedBLR:
    """Bayesian linear regression with alpha (prior precision) and beta (noise
    precision) estimated by EVIDENCE MAXIMIZATION (Bishop PRML 3.5.2), instead of
    the frozen BLR's fixed alpha=beta=1 (which floors predictive std at
    sqrt(1/beta)=1 and is miscalibrated by 6-47x). Iterates
      gamma = sum_i (beta*lambda_i)/(alpha + beta*lambda_i)   [lambda_i = eig(Phi^T Phi)]
      alpha <- gamma / (m_N^T m_N),   beta <- (N - gamma)/||y - Phi m_N||^2
    with clipping for stability. Same predictive form as the frozen BLR.
    """

    def __init__(self, n_iter=300, tol=1e-4, beta_max=1e8, beta_min=1e-4):
        self.n_iter = n_iter; self.tol = tol
        self.beta_max = beta_max; self.beta_min = beta_min
        self.fitted = False

    def fit(self, Phi, y):
        Phi = np.asarray(Phi, dtype=float); y = np.asarray(y, dtype=float).ravel()
        N, D = Phi.shape
        Pb = np.hstack([Phi, np.ones((N, 1))]); Db = D + 1
        PtP = Pb.T @ Pb; Pty = Pb.T @ y
        eig = np.clip(np.linalg.eigvalsh(PtP), 0.0, None)
        I = np.eye(Db); alpha, beta = 1.0, 1.0
        for _ in range(self.n_iter):
            S = np.linalg.inv(alpha * I + beta * PtP + 1e-10 * I)
            m = beta * (S @ Pty)
            lam = beta * eig
            gamma = float(np.sum(lam / (alpha + lam)))
            mm = float(m @ m)
            rss = float(np.sum((y - Pb @ m) ** 2))
            a_new = gamma / mm if mm > 1e-12 else alpha
            b_new = (N - gamma) / rss if (rss > 1e-12 and N > gamma) else beta
            a_new = float(np.clip(a_new, 1e-6, 1e8))
            b_new = float(np.clip(b_new, self.beta_min, self.beta_max))
            if abs(a_new - alpha) < self.tol * alpha and abs(b_new - beta) < self.tol * beta:
                alpha, beta = a_new, b_new; break
            alpha, beta = a_new, b_new
        self.alpha, self.beta = alpha, beta
        self.S_N = np.linalg.inv(alpha * I + beta * PtP + 1e-10 * I)
        self.m_N = beta * (self.S_N @ Pty)
        self.fitted = True
        return self

    def predict(self, Phi):
        Pb = np.hstack([np.asarray(Phi, dtype=float), np.ones((len(Phi), 1))])
        mean = Pb @ self.m_N
        var = 1.0 / self.beta + np.einsum('ij,jk,ik->i', Pb, self.S_N, Pb)
        return mean.ravel(), np.sqrt(np.maximum(var, 1e-12))


class CalibTunedSeq:
    """HPO-tuned Sequential TL surrogate + EVIDENCE-CALIBRATED dual BLR.

    Uses the COFs-optimised Sequential hyperparameters (best_params.json:
    lf_hidden_dim=64/lf_epochs=82, hf_hidden_dim=8/hf_epochs=31, tuned lr +
    weight decays) and replaces the fixed-alpha/beta=1 BLR with CalibratedBLR on
    BOTH the LF features (LF head) and the HF-net last-layer features (HF head).
    The tuned hf_hidden_dim=8 makes the HF-BLR well-posed (n_HF > 8 features), and
    evidence-calibrated beta removes the std=1 floor, so the HF posterior is
    genuinely calibrated -> a real test of whether HF-EI helps once the posterior
    is sound. predict()=HF-BLR (mean,std); predict_lf()=LF-BLR (mean,std).
    """

    def __init__(self, input_dim, device=None,
                 lf_hidden_dim=64, lf_epochs=82, lf_wd=2.6868261685971602e-05,
                 hf_hidden_dim=8, hf_epochs=31, hf_wd=0.0014559413456187186,
                 lr=0.0025704450915175502):
        self.input_dim = input_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lf_hidden_dim = lf_hidden_dim; self.lf_epochs = lf_epochs; self.lf_wd = lf_wd
        self.hf_hidden_dim = hf_hidden_dim; self.hf_epochs = hf_epochs; self.hf_wd = hf_wd
        self.lr = lr
        self.sx = StandardScaler(); self.sy = StandardScaler()
        self.is_fitted = False

    def fit(self, X_lf, y_lf, X_hf, y_hf):
        import torch.nn.functional as F
        from benchmark import LFNetwork, HFNetwork
        Xall = np.vstack([X_lf, X_hf])
        yall = np.concatenate([np.asarray(y_lf).ravel(), np.asarray(y_hf).ravel()])
        Xs = self.sx.fit_transform(Xall); ys = self.sy.fit_transform(yall.reshape(-1, 1)).ravel()
        nlf = len(X_lf)
        Xlf_t = torch.FloatTensor(Xs[:nlf]).to(self.device)
        ylf_t = torch.FloatTensor(ys[:nlf]).view(-1, 1).to(self.device)
        Xhf_t = torch.FloatTensor(Xs[nlf:]).to(self.device)
        yhf_t = torch.FloatTensor(ys[nlf:]).view(-1, 1).to(self.device)
        ylf_s = ys[:nlf]; yhf_s = ys[nlf:]
        # LF net (tuned)
        self.lf_net = LFNetwork(self.input_dim, self.lf_hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=self.lr, weight_decay=self.lf_wd)
        for _ in range(self.lf_epochs):
            opt.zero_grad(); F.mse_loss(self.lf_net(Xlf_t), ylf_t).backward(); opt.step()
        self.lf_net.eval()
        with torch.no_grad():
            feat_lf = self.lf_net.extract_features(Xlf_t).cpu().numpy()
        self.lf_blr = CalibratedBLR().fit(feat_lf, ylf_s)
        # HF net (tuned, small hidden=8 -> HF-BLR well-posed)
        for p in self.lf_net.parameters():
            p.requires_grad = False
        self.hf_net = HFNetwork(self.input_dim, self.hf_hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.hf_net.parameters(), lr=self.lr, weight_decay=self.hf_wd)
        for _ in range(self.hf_epochs):
            opt.zero_grad()
            with torch.no_grad():
                y_lf_pred = self.lf_net(Xhf_t)
            F.mse_loss(self.hf_net(Xhf_t, y_lf_pred), yhf_t).backward(); opt.step()
        self.hf_net.eval()
        with torch.no_grad():
            y_lf_pred = self.lf_net(Xhf_t)
            feat_hf = self.hf_net.extract_features(Xhf_t, y_lf_pred).cpu().numpy()
        self.hf_blr = CalibratedBLR().fit(feat_hf, yhf_s)
        self.is_fitted = True
        return self

    def _feat(self, X, which):
        X_t = torch.FloatTensor(self.sx.transform(X)).to(self.device)
        self.lf_net.eval(); self.hf_net.eval()
        with torch.no_grad():
            if which == 'lf':
                return self.lf_net.extract_features(X_t).cpu().numpy()
            y_lf = self.lf_net(X_t)
            return self.hf_net.extract_features(X_t, y_lf).cpu().numpy()

    def predict(self, X):
        # HF MEAN from the trained HF net (better than a BLR re-fit), HF STD from
        # the calibrated HF-BLR -> isolates "does calibrated HF-EI help" without
        # the BLR-mean degradation confound.
        X_t = torch.FloatTensor(self.sx.transform(X)).to(self.device)
        self.lf_net.eval(); self.hf_net.eval()
        with torch.no_grad():
            y_lf = self.lf_net(X_t)
            mean_s = self.hf_net(X_t, y_lf).cpu().numpy().ravel()
            feat = self.hf_net.extract_features(X_t, y_lf).cpu().numpy()
        _, s = self.hf_blr.predict(feat)
        mean = self.sy.inverse_transform(mean_s.reshape(-1, 1)).ravel()
        return mean, np.maximum(s * self.sy.scale_[0], 1e-6)

    def predict_lf(self, X):
        m, s = self.lf_blr.predict(self._feat(X, 'lf'))
        mean = self.sy.inverse_transform(m.reshape(-1, 1)).ravel()
        return mean, np.maximum(s * self.sy.scale_[0], 1e-6)


# =============================================================================
# Baseline 4: Successive Halving (surrogate-free)
# =============================================================================

def run_successive_halving(benchmark, budget, seed=42):
    """
    Surrogate-free successive halving using the SAME fidelity schedule as run_bo().

    Fidelity schedule: lf_per_hf = max(1, int(1.0 / rho)) LF evals per 1 HF eval.
    Initial sampling: same 10% budget with FPS/LHS as run_bo().
    LF turn: random from unevaluated pool.
    HF turn: pick candidate with best (lowest) LF score among HF-unevaluated.

    Returns dict matching run_bo() output format.
    """
    from benchmark import (
        furthest_point_sampling, latin_hypercube_sampling,
        find_nearest_candidates,
    )

    np.random.seed(seed)
    rho = benchmark.cost_ratio
    n_candidates = benchmark.n_candidates

    # --- Initial sampling (same as run_bo) ---
    init_budget = 0.1 * budget
    n_init_hf = max(2, int(init_budget * 0.5 / 1.0))
    n_init_lf = max(2, int(init_budget * 0.5 / rho))
    n_init_total = n_init_lf + n_init_hf

    is_synthetic = hasattr(benchmark, 'dim')
    if is_synthetic:
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
                    benchmark.X[list(available)], remaining, seed + 1000
                )
                extra_indices = [list(available)[i] for i in extra]
                init_indices.extend(extra_indices)
    else:
        init_indices = furthest_point_sampling(benchmark.X, n_init_total, seed).tolist()

    lf_evaluated = {}
    hf_evaluated = {}
    all_sampled = set()

    for idx in init_indices[:n_init_lf]:
        lf_evaluated[idx] = benchmark.evaluate_lf(np.array([idx]))[0]
        all_sampled.add(idx)
    for idx in init_indices[n_init_lf:n_init_lf + n_init_hf]:
        hf_evaluated[idx] = benchmark.evaluate_hf(np.array([idx]))[0]
        all_sampled.add(idx)

    current_budget = n_init_lf * rho + n_init_hf * 1.0
    best_hf = min(hf_evaluated.values()) if hf_evaluated else np.inf

    regrets = [max(0, best_hf - benchmark.f_star)]
    budgets_list = [current_budget]
    step_records = []

    # --- Same fidelity schedule as run_bo ---
    lf_per_hf = max(1, int(1.0 / rho))
    lf_counter = 0
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

        if eval_hf:
            # HF turn: pick best LF-scored candidate not yet HF-evaluated
            candidates = [
                (idx, score) for idx, score in lf_evaluated.items()
                if idx not in hf_evaluated
            ]
            if candidates:
                candidates.sort(key=lambda x: x[1])  # minimization
                idx = candidates[0][0]
            else:
                available = set(range(n_candidates)) - all_sampled
                if not available:
                    break
                idx = np.random.choice(list(available))
            hf_score = benchmark.evaluate_hf(np.array([idx]))[0]
            hf_evaluated[idx] = hf_score
            all_sampled.add(idx)
            best_hf = min(best_hf, hf_score)
            fidelity = 1
            observed = hf_score
        else:
            # LF turn: random from unevaluated
            unevaluated = set(range(n_candidates)) - all_sampled
            if not unevaluated:
                break
            idx = np.random.choice(list(unevaluated))
            lf_score = benchmark.evaluate_lf(np.array([idx]))[0]
            lf_evaluated[idx] = lf_score
            all_sampled.add(idx)
            fidelity = 0
            observed = lf_score

        current_budget += cost
        regrets.append(max(0, best_hf - benchmark.f_star))
        budgets_list.append(current_budget)
        step_records.append({
            'step': iteration,
            'fidelity': fidelity,
            'candidate_idx': int(idx),
            'observed_value': observed,
            'best_hf_so_far': best_hf,
            'wall_time_sec': 0.0,
        })

    return {
        'regrets': regrets if regrets else [float('inf')],
        'budgets': budgets_list if budgets_list else [0],
        'final_regret': regrets[-1] if regrets else float('inf'),
        'n_hf': len(hf_evaluated),
        'n_lf': len(lf_evaluated),
        'best_y': best_hf if hf_evaluated else np.inf,
        'step_records': step_records,
    }


# =============================================================================
# Baseline 5: HF-Only Random Search (non-learning)
# =============================================================================

def run_hf_random_search(benchmark, budget, seed=42):
    """
    Pure random search using only HF evaluations (no LF, no surrogate).

    Provides the absolute performance floor — what you get without any
    multi-fidelity or surrogate modelling.

    Initial sampling: same strategy as run_bo (LHS for synthetic, FPS for chemistry).
    Remaining budget: random HF evaluations from unevaluated candidates.

    Returns dict matching run_bo() output format.
    """
    from benchmark import (
        furthest_point_sampling, latin_hypercube_sampling,
        find_nearest_candidates,
    )

    np.random.seed(seed)
    n_candidates = benchmark.n_candidates

    # Total HF evaluations possible
    n_hf_total = int(budget / 1.0)

    # Initial sampling: 10% of budget (same as run_bo)
    n_init = max(2, int(0.1 * budget))
    n_init = min(n_init, n_hf_total)

    is_synthetic = hasattr(benchmark, 'grid_size')
    if is_synthetic:
        bounds = np.array([[0, 1]] * benchmark.dim)
        lhs_samples = latin_hypercube_sampling(bounds, n_init, seed)
        X_min, X_max = benchmark.X.min(axis=0), benchmark.X.max(axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1
        lhs_samples_scaled = X_min + lhs_samples * X_range
        init_indices = find_nearest_candidates(benchmark.X, lhs_samples_scaled)
        init_indices = list(dict.fromkeys(init_indices))
        if len(init_indices) < n_init:
            remaining = n_init - len(init_indices)
            available = set(range(n_candidates)) - set(init_indices)
            if available:
                extra = furthest_point_sampling(
                    benchmark.X[list(available)], remaining, seed + 1000
                )
                extra_indices = [list(available)[i] for i in extra]
                init_indices.extend(extra_indices)
    else:
        init_indices = furthest_point_sampling(benchmark.X, n_init, seed).tolist()

    hf_evaluated = {}
    all_sampled = set()

    # Phase 1: Initial HF evaluations
    for idx in init_indices[:n_init]:
        hf_evaluated[idx] = benchmark.evaluate_hf(np.array([idx]))[0]
        all_sampled.add(idx)

    current_budget = len(hf_evaluated) * 1.0
    best_hf = min(hf_evaluated.values()) if hf_evaluated else np.inf

    regrets = [max(0, best_hf - benchmark.f_star)]
    budgets_list = [current_budget]
    step_records = []
    iteration = 0

    # Phase 2: Random HF evaluations
    while current_budget + 1.0 <= budget:
        iteration += 1
        available = set(range(n_candidates)) - all_sampled
        if not available:
            break

        idx = np.random.choice(list(available))
        hf_score = benchmark.evaluate_hf(np.array([idx]))[0]
        hf_evaluated[idx] = hf_score
        all_sampled.add(idx)

        current_budget += 1.0
        best_hf = min(best_hf, hf_score)
        regrets.append(max(0, best_hf - benchmark.f_star))
        budgets_list.append(current_budget)
        step_records.append({
            'step': iteration,
            'fidelity': 1,
            'candidate_idx': int(idx),
            'observed_value': hf_score,
            'best_hf_so_far': best_hf,
            'wall_time_sec': 0.0,
        })

    return {
        'regrets': regrets if regrets else [float('inf')],
        'budgets': budgets_list if budgets_list else [0],
        'final_regret': regrets[-1] if regrets else float('inf'),
        'n_hf': len(hf_evaluated),
        'n_lf': 0,
        'best_y': best_hf if hf_evaluated else np.inf,
        'step_records': step_records,
    }


# =============================================================================
# Baseline 6: LF-Screening (non-learning, surrogate-free MF)
# =============================================================================

def run_lf_screening(benchmark, budget, seed=42):
    """
    LF-Screening: evaluate as many candidates as possible at LF,
    then spend remaining budget on HF evaluations of top-ranked LF candidates.

    This is the strongest possible non-learning MF baseline — it uses the
    LF fidelity directly for ranking without any surrogate model.

    Budget allocation (with dynamic reallocation when pool < n_lf_raw):
      1. Reserve minimum HF: n_reserve_hf = max(5, int(0.1 * budget))
      2. Plan LF: n_lf_raw = int((budget - n_reserve_hf) / rho)
      3. Cap by pool: n_lf = min(n_candidates, n_lf_raw)
      4. Reallocate: remaining_budget = budget - n_lf * rho → n_hf = int(remaining / 1.0)

    Returns dict matching run_bo() output format.
    """
    from benchmark import (
        furthest_point_sampling, latin_hypercube_sampling,
        find_nearest_candidates,
    )

    np.random.seed(seed)
    rho = benchmark.cost_ratio
    n_candidates = benchmark.n_candidates

    # --- Budget allocation with dynamic reallocation ---
    n_reserve_hf = max(5, int(0.1 * budget))
    lf_budget = budget - n_reserve_hf * 1.0
    n_lf_raw = int(lf_budget / rho)

    # Cap by pool size
    n_lf = min(n_candidates, n_lf_raw)

    # Dynamic reallocation: excess budget goes to HF
    actual_lf_cost = n_lf * rho
    remaining_budget = budget - actual_lf_cost
    n_hf = int(remaining_budget / 1.0)
    n_hf = max(n_hf, n_reserve_hf)  # at least the reserved amount

    # --- Initial sampling (same as run_bo) ---
    init_budget = 0.1 * budget
    n_init_hf = max(2, int(init_budget * 0.5 / 1.0))
    n_init_lf = max(2, int(init_budget * 0.5 / rho))
    n_init_total = n_init_lf + n_init_hf

    is_synthetic = hasattr(benchmark, 'grid_size')
    if is_synthetic:
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
                    benchmark.X[list(available)], remaining, seed + 1000
                )
                extra_indices = [list(available)[i] for i in extra]
                init_indices.extend(extra_indices)
    else:
        init_indices = furthest_point_sampling(benchmark.X, n_init_total, seed).tolist()

    lf_evaluated = {}
    hf_evaluated = {}
    all_sampled = set()

    # Initial LF evaluations
    for idx in init_indices[:n_init_lf]:
        lf_evaluated[idx] = benchmark.evaluate_lf(np.array([idx]))[0]
        all_sampled.add(idx)

    # Initial HF evaluations
    for idx in init_indices[n_init_lf:n_init_lf + n_init_hf]:
        hf_evaluated[idx] = benchmark.evaluate_hf(np.array([idx]))[0]
        all_sampled.add(idx)
        # Also get LF for these (free info for ranking, doesn't cost budget)
        if idx not in lf_evaluated:
            lf_evaluated[idx] = benchmark.evaluate_lf(np.array([idx]))[0]

    current_budget = n_init_lf * rho + n_init_hf * 1.0
    best_hf = min(hf_evaluated.values()) if hf_evaluated else np.inf

    regrets = [max(0, best_hf - benchmark.f_star)]
    budgets_list = [current_budget]
    step_records = []
    iteration = 0

    # --- Phase 1: LF Sweep ---
    # Evaluate remaining LF candidates (up to n_lf total, minus already evaluated)
    n_lf_remaining = n_lf - len(lf_evaluated)
    unevaluated_lf = list(set(range(n_candidates)) - set(lf_evaluated.keys()))
    np.random.shuffle(unevaluated_lf)
    n_lf_remaining = min(n_lf_remaining, len(unevaluated_lf))

    for i in range(n_lf_remaining):
        if current_budget + rho > budget:
            break
        idx = unevaluated_lf[i]
        iteration += 1

        lf_score = benchmark.evaluate_lf(np.array([idx]))[0]
        lf_evaluated[idx] = lf_score
        all_sampled.add(idx)

        current_budget += rho
        # Regret unchanged during LF phase (no new HF info)
        regrets.append(max(0, best_hf - benchmark.f_star))
        budgets_list.append(current_budget)
        step_records.append({
            'step': iteration,
            'fidelity': 0,
            'candidate_idx': int(idx),
            'observed_value': lf_score,
            'best_hf_so_far': best_hf,
            'wall_time_sec': 0.0,
        })

    # --- Phase 2: HF Top-k ---
    # Sort all LF-evaluated candidates by LF score (ascending = best first for minimization)
    lf_ranked = sorted(lf_evaluated.items(), key=lambda x: x[1])

    # Select top candidates not yet HF-evaluated
    hf_candidates = [idx for idx, _ in lf_ranked if idx not in hf_evaluated]

    for idx in hf_candidates:
        if current_budget + 1.0 > budget:
            break
        iteration += 1

        hf_score = benchmark.evaluate_hf(np.array([idx]))[0]
        hf_evaluated[idx] = hf_score

        current_budget += 1.0
        best_hf = min(best_hf, hf_score)
        regrets.append(max(0, best_hf - benchmark.f_star))
        budgets_list.append(current_budget)
        step_records.append({
            'step': iteration,
            'fidelity': 1,
            'candidate_idx': int(idx),
            'observed_value': hf_score,
            'best_hf_so_far': best_hf,
            'wall_time_sec': 0.0,
        })

    return {
        'regrets': regrets if regrets else [float('inf')],
        'budgets': budgets_list if budgets_list else [0],
        'final_regret': regrets[-1] if regrets else float('inf'),
        'n_hf': len(hf_evaluated),
        'n_lf': len(lf_evaluated),
        'best_y': best_hf if hf_evaluated else np.inf,
        'step_records': step_records,
    }
