'Run the five transfer-learning surrogates and the supplementary head ablations.'
from __future__ import annotations
import argparse, os, sys, time, traceback
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(HERE))
from _shared_calibration import ece_gaussian, nll_gaussian
import pools as POOLS

def _ext_budgets():
    d = {}
    for kv in os.environ.get('EXT_BUDGETS', '').split(','):
        if '=' in kv:
            (k, v) = kv.split('=')
            d[k.strip()] = float(v)
    return d
EXT_BUDGETS = _ext_budgets()

def _load_picks(bn, ml, seed, sub):
    import glob
    for root in os.environ.get('RESUME_DIRS', '').split(':'):
        if not root:
            continue
        for f in sorted(glob.glob(str(Path(root) / sub / 'cells' / f'picks_{bn}_{ml}_*.csv'))):
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            df = df[df.seed == seed]
            if len(df):
                return df.sort_values('step', kind='stable')
    return None

def _replay(picks_df, y_hf, f_star, pct, rho, lf_idx, hf_idx):
    lf_idx.clear()
    hf_idx.clear()
    init = picks_df[picks_df.step == 0]
    loop = picks_df[picks_df.step > 0]
    for r in init.itertuples():
        (lf_idx if r.fid == 'L0' else hf_idx).add(int(r.idx))
    cur = len(lf_idx) * rho + len(hf_idx) * 1.0
    lfc = it = 0
    regrets = [float(max(0.0, y_hf[list(hf_idx)].min() - f_star))]
    budgets = [cur]
    picks = []
    for r in loop.itertuples():
        it += 1
        if r.fid == 'H':
            hf_idx.add(int(r.idx))
            cur += 1.0
            lfc = 0
            picks.append(float(pct[int(r.idx)]))
        else:
            lf_idx.add(int(r.idx))
            cur += rho
            lfc += 1
        regrets.append(float(max(0.0, y_hf[list(hf_idx)].min() - f_star)))
        budgets.append(cur)
    return (cur, lfc, it, regrets, budgets, picks)
SEEDS = list(range(42, 82))
MODELS = ['DNGOJoint', 'TwoStageJoint', 'DNGOGradient', 'SoftParameterSharing', 'DomainAdaptationMMD']
HF_HEAD = 'blr_replace'
LF_TARGET = os.environ.get('LF_TARGET', 'hf_argmin')
WARM = 0
ACT = 'tanh'
SMALL = False
assert LF_TARGET in ('hf_argmin', 'hf_ei', 'hf_pi', 'hf_ucb', 'hf_mes', 'hf_ts')

def _cls(name):
    import benchmark as B
    return {m: getattr(B, m) for m in MODELS}[name]
HEAD_PHI = os.environ.get('HEAD_PHI', 'learned').lower()
HF_ONLY = int(os.environ.get('HF_ONLY', '0'))
HEAD_REMOVE_GL = int(os.environ.get('HEAD_REMOVE_GL', '0'))
assert HEAD_REMOVE_GL in (0, 1), HEAD_REMOVE_GL
if HEAD_REMOVE_GL:
    assert HEAD_PHI in ('random', 'learned') and (not HF_ONLY) and (HF_HEAD == 'blr_replace')
assert HEAD_PHI in ('learned', 'random', 'none'), HEAD_PHI
assert HF_ONLY in (0, 1), HF_ONLY
if HEAD_PHI in ('random', 'none') or HF_ONLY:
    assert HF_HEAD == 'blr_replace', 'HEAD_PHI=random / HEAD_PHI=none / HF_ONLY=1 are defined for the BLR head only'
HEAD_SCALAR = int(os.environ.get('HEAD_SCALAR', '1'))
HEAD_OFFSET = int(os.environ.get('HEAD_OFFSET', '1'))
assert HEAD_SCALAR in (0, 1), HEAD_SCALAR
assert HEAD_OFFSET in (0, 1), HEAD_OFFSET
if not (HEAD_SCALAR and HEAD_OFFSET):
    assert HEAD_PHI in ('random', 'learned') and (not HF_ONLY) and (HF_HEAD == 'blr_replace')
    assert not HEAD_REMOVE_GL, 'HEAD_REMOVE_GL=1 already means HEAD_SCALAR=0 HEAD_OFFSET=0; do not combine'

def blr_fit(Z, y, alpha_vec, beta, ev_dims=None, grid=(10.0, 30.0, 100.0, 300.0, 1000.0)):
    Z = np.asarray(Z, float)
    y = np.asarray(y, float).ravel()
    (N, D) = Z.shape
    alpha = np.asarray(alpha_vec, float).copy()
    ZtZ = Z.T @ Z
    Zty = Z.T @ y

    def solve(alpha, beta):
        A = np.diag(alpha) + beta * ZtZ + 1e-09 * np.eye(D)
        S = np.linalg.pinv(A)
        m = beta * S @ Zty
        return (m, S)
    if ev_dims is not None and N >= 4:
        ev = np.asarray(ev_dims)
        best = None
        for lam in grid:
            a = alpha.copy()
            a[ev] = lam
            (m, S) = solve(a, 1.0)
            Hm = Z @ S @ Z.T
            loo = (y - Z @ m) / np.maximum(1.0 - np.diag(Hm), 0.001)
            mse = float(np.mean(loo ** 2))
            if best is None or mse < best[0]:
                best = (mse, lam)
        (mse, lam) = best
        beta = float(np.clip(1.0 / max(mse, 1e-06), 1.0, 100.0))
        alpha[ev] = lam * beta
        alpha[[i for i in range(D) if i not in set(ev.tolist())]] *= beta
    (m, S) = solve(alpha, beta)
    return (m, S, alpha, beta)

def blr_predict(Z, m, S, beta):
    mu = Z @ m
    var = 1.0 / beta + np.einsum('ij,jk,ik->i', Z, S, Z)
    return (mu, np.sqrt(np.maximum(var, 1e-12)))

def _lf_out_and_phi(model, Xt):
    import torch
    model.lf_net.eval()
    with torch.no_grad():
        return (model.lf_net(Xt).cpu().numpy(), model.lf_net.extract_features(Xt).cpu().numpy())

def _head_feats(model, Xt, phi_net=None):
    import torch
    (lo, phi) = _lf_out_and_phi(model, Xt)
    if phi_net is not None:
        phi_net.eval()
        with torch.no_grad():
            phi = phi_net.extract_features(Xt).cpu().numpy()
    if HEAD_REMOVE_GL:
        lo = np.zeros_like(lo)
    return (lo, phi)

class Head:

    def __init__(self, model, X_hf, y_hf, phi_net=None):
        import torch
        self.model = model
        self.kind = HF_HEAD
        self.phi_net = phi_net
        Xs = model.scaler_x.transform(X_hf)
        ys = model.scaler_y.transform(np.asarray(y_hf).reshape(-1, 1)).ravel()
        Xt = torch.FloatTensor(Xs).to(model.device)
        (lo, phi) = _head_feats(model, Xt, phi_net)
        lo_z = lo if HEAD_SCALAR else np.zeros_like(lo)
        lo_o = lo if HEAD_OFFSET else np.zeros_like(lo)
        lo = lo_z
        if HEAD_PHI == 'none':
            Z = np.hstack([lo, np.ones((len(Xs), 1))])
            D = Z.shape[1]
            alpha = np.array([10.0, 1.0])
        else:
            Z = np.hstack([lo, phi, np.ones((len(Xs), 1))])
            D = Z.shape[1]
            alpha = np.full(D, 100.0)
            alpha[0] = 10.0
            alpha[-1] = 1.0
        (self.m, self.S, self.alpha, self.beta) = blr_fit(Z, ys - lo_o.ravel(), alpha, 10.0, ev_dims=np.arange(1, D - 1))

    def predict(self, X):
        import torch
        model = self.model
        Xs = model.scaler_x.transform(X)
        Xt = torch.FloatTensor(Xs).to(model.device)
        (lo, phi) = _head_feats(model, Xt, self.phi_net)
        lo_z = lo if HEAD_SCALAR else np.zeros_like(lo)
        lo_o = lo if HEAD_OFFSET else np.zeros_like(lo)
        lo = lo_z
        if HEAD_PHI == 'none':
            Z = np.hstack([lo, np.ones((len(Xs), 1))])
        else:
            Z = np.hstack([lo, phi, np.ones((len(Xs), 1))])
        (d_s, sd_s) = blr_predict(Z, self.m, self.S, self.beta)
        mu_s = lo_o.ravel() + d_s
        scale = float(model.scaler_y.scale_[0])
        mu = model.scaler_y.inverse_transform(mu_s.reshape(-1, 1)).ravel()
        return (mu, np.maximum(sd_s * scale, 1e-09))

class HFOnlyNet:

    def __init__(self, input_dim, hidden_dim=64, epochs=300, device=None):
        import torch
        from sklearn.preprocessing import StandardScaler
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.device = device or torch.device('cpu')
        self.is_fitted = False

    def fit(self, X_hf, y_hf):
        import torch, torch.nn.functional as F
        import benchmark as B
        Xs = self.scaler_x.fit_transform(np.asarray(X_hf, float))
        ys = self.scaler_y.fit_transform(np.asarray(y_hf, float).reshape(-1, 1)).flatten()
        Xt = torch.FloatTensor(Xs).to(self.device)
        yt = torch.FloatTensor(ys).view(-1, 1).to(self.device)
        self.lf_net = B.LFNetwork(self.input_dim, self.hidden_dim).to(self.device)
        opt = torch.optim.Adam(self.lf_net.parameters(), lr=0.001, weight_decay=0.0001)
        for _ in range(self.epochs):
            opt.zero_grad()
            F.mse_loss(self.lf_net(Xt), yt).backward()
            opt.step()
        self.is_fitted = True

def fps(X, n, seed):
    from scipy.spatial.distance import cdist
    rng = np.random.RandomState(seed)
    sel = [int(rng.randint(len(X)))]
    mind = cdist(X, X[sel]).min(axis=1)
    mind[sel] = -np.inf
    for _ in range(n - 1):
        nx = int(np.argmax(mind))
        sel.append(nx)
        mind = np.minimum(mind, np.linalg.norm(X - X[nx], axis=1))
        mind[nx] = -np.inf
    return np.array(sel)

def run_one(bench, cfg, model_name, seed, device):
    import torch
    import benchmark as B
    from benchmark import expected_improvement, probability_of_improvement, lower_confidence_bound, max_value_entropy_search, thompson_sample
    cls = _cls(model_name) if not HF_ONLY else None
    np.random.seed(seed)
    torch.manual_seed(seed)
    phi_net = None
    if HEAD_PHI == 'random':
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(10000 + seed)
            phi_net = B.LFNetwork(bench.X.shape[1], 64).to(device)
        phi_net.eval()
        for p in phi_net.parameters():
            p.requires_grad = False
    acq_rng = np.random.default_rng(seed)
    (X, y_hf, y_lf) = (bench.X, bench.y_hf, bench.y_lf)
    rho = bench.cost_ratio
    budget = float(EXT_BUDGETS.get(cfg['_name'], cfg['budget']))
    pct = (np.argsort(np.argsort(y_hf)) + 1) / bench.n_candidates
    init_budget = float(cfg.get('init_budget', os.environ.get('INIT_BUDGET', cfg['budget'])))
    n_init_lf = max(2, int(0.1 * init_budget * 0.5 / rho))
    n_init_hf = max(2, int(0.1 * init_budget * 0.5))
    init = fps(X, n_init_lf + n_init_hf, seed)
    if HF_ONLY:
        return run_one_hfonly(bench, cfg, seed, device, init, n_init_lf, n_init_hf, budget, pct, phi_net)
    lf_idx = set((int(i) for i in init[:n_init_lf]))
    hf_idx = set((int(i) for i in init[n_init_lf:n_init_lf + n_init_hf]))
    cur = len(lf_idx) * rho + len(hf_idx) * 1.0
    lf_per_hf = max(1, int(1.0 / rho))
    lfc = it = 0
    regrets = [float(max(0.0, y_hf[list(hf_idx)].min() - bench.f_star))]
    budgets = [cur]
    picks = []
    cal = []
    model = None
    plog = [dict(step=0, fid='L0', idx=int(i), budget=cur) for i in init[:n_init_lf]] + [dict(step=0, fid='H0', idx=int(i), budget=cur) for i in init[n_init_lf:n_init_lf + n_init_hf]]
    resumed_from = np.nan
    pk = _load_picks(cfg['_name'], model_name, seed, arm_name())
    if pk is not None:
        (cur, lfc, it, regrets, budgets, picks) = _replay(pk, y_hf, bench.f_star, pct, rho, lf_idx, hf_idx)
        plog = [dict(step=int(r.step), fid=r.fid, idx=int(r.idx), budget=float(r.budget)) for r in pk.itertuples()]
        resumed_from = cur
        print(f"  [resume] {cfg['_name']} {model_name} s{seed}: replayed {it} picks to budget {cur:.3f}", flush=True)
        import pickle
        for root in os.environ.get('RESUME_DIRS', '').split(':'):
            rf = Path(root) / arm_name() / 'cells' / f"rng_{cfg['_name']}_{model_name}_s{seed}.pkl"
            if root and rf.exists():
                st = pickle.load(open(rf, 'rb'))
                np.random.set_state(st['np'])
                torch.set_rng_state(st['torch'])
                acq_rng.bit_generator.state = st['acq']
                print(f'  [resume] RNG state restored from {rf.name}', flush=True)
                break
    max_it = int(budget / rho) + 1000
    fit_failed_at = np.nan
    while cur < budget and it < max_it:
        it += 1
        rem = budget - cur
        if rem >= 1.0:
            eval_hf = not (rem >= rho and lfc < lf_per_hf)
        elif rem >= rho:
            eval_hf = False
        else:
            break
        L = np.array(sorted(lf_idx))
        Hh = np.array(sorted(hf_idx))
        model = cls(X.shape[1], device=device)
        try:
            model.fit(X[L], y_lf[L], X[Hh], y_hf[Hh])
        except Exception as e:
            fit_failed_at = cur
            print(f'  [fit-fail] budget {cur:.3f}: {type(e).__name__}', flush=True)
            break
        head = Head(model, X[Hh], y_hf[Hh], phi_net)
        (mu_H, sd_H) = head.predict(X)
        (lf_mean, lf_std) = model.predict_lf(X)
        lf_mean = np.asarray(lf_mean, float)
        lf_std = np.asarray(lf_std, float)
        held = np.setdiff1d(np.arange(bench.n_candidates), np.array(sorted(lf_idx | hf_idx)))
        if len(held) >= 5:
            (mh, sh) = (lf_mean[held], lf_std[held])
            cal.append((float(ece_gaussian(y_lf[held], mh, sh)), float(nll_gaussian(y_lf[held], mh, sh)), float(np.mean(sh))))
        if eval_hf:
            mm = mu_H.copy()
            mm[list(hf_idx)] = np.inf
            nxt = int(np.argmin(mm))
            hf_idx.add(nxt)
            cur += 1.0
            lfc = 0
            picks.append(float(pct[nxt]))
            plog.append(dict(step=it, fid='H', idx=nxt, budget=cur))
        else:
            ybest = float(y_hf[list(hf_idx)].min())
            if LF_TARGET == 'lf':
                acq = -lf_mean
            elif LF_TARGET == 'hf_argmin':
                acq = -mu_H
            elif LF_TARGET == 'hf_ei':
                acq = expected_improvement(mu_H, sd_H, ybest)
            elif LF_TARGET == 'hf_pi':
                acq = probability_of_improvement(mu_H, sd_H, ybest)
            elif LF_TARGET == 'hf_ucb':
                acq = lower_confidence_bound(mu_H, sd_H)
            elif LF_TARGET == 'hf_mes':
                acq = max_value_entropy_search(mu_H, sd_H, acq_rng)
            elif LF_TARGET == 'hf_ts':
                acq = thompson_sample(mu_H, sd_H, acq_rng)
            acq = np.asarray(acq, float).copy()
            acq[list(lf_idx)] = -np.inf
            nxt = int(np.argmax(acq))
            lf_idx.add(nxt)
            cur += rho
            lfc += 1
            plog.append(dict(step=it, fid='L', idx=nxt, budget=cur))
        regrets.append(float(max(0.0, y_hf[list(hf_idx)].min() - bench.f_star)))
        budgets.append(cur)
    tr = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    auc = float(tr(regrets, budgets) / max(budgets[-1] - budgets[0], 1e-09))
    if cal:
        (ece_m, nll_m, sh_m) = (float(np.mean([c[i] for c in cal])) for i in range(3))
        (ece_f, nll_f, sh_f) = cal[-1]
    else:
        ece_m = nll_m = sh_m = ece_f = nll_f = sh_f = np.nan
    rng_state = dict(np=np.random.get_state(), torch=torch.get_rng_state(), acq=acq_rng.bit_generator.state)
    return dict(final_regret=regrets[-1], auc=auc, n_hf=len(hf_idx), n_lf=len(lf_idx), rng_state=rng_state, pick_prec_top5=float(np.mean([p <= 0.05 + 1e-12 for p in picks])) if picks else np.nan, budgets=budgets, regrets=regrets, plog=plog, resumed_from=resumed_from, budget_used=budget, fit_failed_at=fit_failed_at, ece_lf_mean=ece_m, nll_lf_mean=nll_m, sharp_lf_mean=sh_m, ece_lf_final=ece_f, nll_lf_final=nll_f, sharp_lf_final=sh_f)
HFONLY_NAME = 'HFOnlyNet'

def run_one_hfonly(bench, cfg, seed, device, init, n_init_lf, n_init_hf, budget, pct, phi_net):
    import torch
    (X, y_hf) = (bench.X, bench.y_hf)
    rho = bench.cost_ratio
    lf_idx = set()
    hf_idx = set((int(i) for i in init[n_init_lf:n_init_lf + n_init_hf]))
    cur = len(hf_idx) * 1.0
    it = 0
    regrets = [float(max(0.0, y_hf[list(hf_idx)].min() - bench.f_star))]
    budgets = [cur]
    picks = []
    plog = [dict(step=0, fid='H0', idx=int(i), budget=cur) for i in init[n_init_lf:n_init_lf + n_init_hf]]
    resumed_from = np.nan
    pk = _load_picks(cfg['_name'], HFONLY_NAME, seed, arm_name())
    if pk is not None:
        (cur, _lfc, it, regrets, budgets, picks) = _replay(pk, y_hf, bench.f_star, pct, rho, lf_idx, hf_idx)
        plog = [dict(step=int(r.step), fid=r.fid, idx=int(r.idx), budget=float(r.budget)) for r in pk.itertuples()]
        resumed_from = cur
        print(f"  [resume] {cfg['_name']} {HFONLY_NAME} s{seed}: replayed {it} picks to budget {cur:.3f}", flush=True)
        import pickle
        for root in os.environ.get('RESUME_DIRS', '').split(':'):
            rf = Path(root) / arm_name() / 'cells' / f"rng_{cfg['_name']}_{HFONLY_NAME}_s{seed}.pkl"
            if root and rf.exists():
                st = pickle.load(open(rf, 'rb'))
                np.random.set_state(st['np'])
                torch.set_rng_state(st['torch'])
                print(f'  [resume] RNG state restored from {rf.name}', flush=True)
                break
    fit_failed_at = np.nan
    while cur + 1.0 <= budget + 1e-09:
        it += 1
        Hh = np.array(sorted(hf_idx))
        model = HFOnlyNet(X.shape[1], device=device)
        try:
            model.fit(X[Hh], y_hf[Hh])
        except Exception as e:
            fit_failed_at = cur
            print(f'  [fit-fail] budget {cur:.3f}: {type(e).__name__}', flush=True)
            break
        head = Head(model, X[Hh], y_hf[Hh], phi_net)
        (mu_H, _sd_H) = head.predict(X)
        mm = np.asarray(mu_H, float).copy()
        mm[list(hf_idx)] = np.inf
        nxt = int(np.argmin(mm))
        hf_idx.add(nxt)
        cur += 1.0
        picks.append(float(pct[nxt]))
        plog.append(dict(step=it, fid='H', idx=nxt, budget=cur))
        regrets.append(float(max(0.0, y_hf[list(hf_idx)].min() - bench.f_star)))
        budgets.append(cur)
    tr = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    auc = float(tr(regrets, budgets) / max(budgets[-1] - budgets[0], 1e-09))
    rng_state = dict(np=np.random.get_state(), torch=torch.get_rng_state(), acq=None)
    return dict(final_regret=regrets[-1], auc=auc, n_hf=len(hf_idx), n_lf=0, rng_state=rng_state, pick_prec_top5=float(np.mean([p <= 0.05 + 1e-12 for p in picks])) if picks else np.nan, budgets=budgets, regrets=regrets, plog=plog, resumed_from=resumed_from, budget_used=budget, fit_failed_at=fit_failed_at, ece_lf_mean=np.nan, nll_lf_mean=np.nan, sharp_lf_mean=np.nan, ece_lf_final=np.nan, nll_lf_final=np.nan, sharp_lf_final=np.nan)

def _append(path, rows):
    df = pd.DataFrame(rows)
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def arm_name():
    return f'{HF_HEAD}__{LF_TARGET}' + ('__randphi' if HEAD_PHI == 'random' else '') + ('__nophi' if HEAD_PHI == 'none' else '') + ('__hfonly' if HF_ONLY else '') + ('__nogl' if HEAD_REMOVE_GL else '') + (f'__s{HEAD_SCALAR}o{HEAD_OFFSET}' if not (HEAD_SCALAR and HEAD_OFFSET) else '')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benches', nargs='*', default=list(POOLS.PAPER))
    ap.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    ap.add_argument('--seeds', type=int, nargs='*', default=SEEDS)
    ap.add_argument('--budget', type=float)
    ap.add_argument('--outdir', default=str(REPO / 'outputs'))
    args = ap.parse_args()
    if args.budget is not None:
        EXT_BUDGETS.update({bn: args.budget for bn in args.benches})
    import torch
    dev = torch.device(os.environ.get('TL_DEVICE', 'cpu'))
    torch.set_num_threads(int(os.environ.get('TORCH_THREADS', '1')))
    arm = arm_name()
    out = Path(args.outdir) / arm
    (out / 'cells').mkdir(parents=True, exist_ok=True)
    print(f'arm={arm} head_phi={HEAD_PHI} hf_only={HF_ONLY} remove_gl={HEAD_REMOVE_GL} head_scalar={HEAD_SCALAR} head_offset={HEAD_OFFSET} benches={args.benches} models={args.models} seeds={args.seeds}', flush=True)
    tag = f's{min(args.seeds)}-{max(args.seeds)}' if len(args.seeds) > 1 else f's{args.seeds[0]}'
    for bn in args.benches:
        (b, cfg) = POOLS.make_bench(bn, REPO / 'data')
        cfg = dict(cfg)
        cfg['_name'] = bn
        print(f"{bn}: N={b.n_candidates} dim={b.dim} budget={cfg['budget']} ext_budget={EXT_BUDGETS.get(bn, cfg['budget'])} rho={cfg['cost_ratio']}", flush=True)
        for ml in [HFONLY_NAME] if HF_ONLY else args.models:
            sf = out / 'cells' / f'summary_{bn}_{ml}_{tag}.csv'
            tf = out / 'cells' / f'traj_{bn}_{ml}_{tag}.csv'
            pf = out / 'cells' / f'picks_{bn}_{ml}_{tag}.csv'
            done = set(pd.read_csv(sf)['seed'].tolist()) if sf.exists() else set()
            for root in os.environ.get('SKIP_DIRS', '').split(':'):
                f2 = Path(root) / arm / 'cells' / sf.name
                if root and f2.exists():
                    try:
                        done |= set(pd.read_csv(f2)['seed'].tolist())
                    except Exception:
                        pass
            for seed in args.seeds:
                if seed in done:
                    continue
                t0 = time.time()
                try:
                    r = run_one(b, cfg, ml, seed, dev)
                except Exception:
                    print(f'[FAIL] {bn} {ml} s{seed}', flush=True)
                    traceback.print_exc()
                    continue
                _append(sf, [dict(benchmark=bn, model=ml, seed=seed, final_regret=r['final_regret'], auc=r['auc'], n_hf=r['n_hf'], n_lf=r['n_lf'], pick_prec_top5=r['pick_prec_top5'], ece_lf_mean=r['ece_lf_mean'], nll_lf_mean=r['nll_lf_mean'], sharp_lf_mean=r['sharp_lf_mean'], ece_lf_final=r['ece_lf_final'], nll_lf_final=r['nll_lf_final'], sharp_lf_final=r['sharp_lf_final'], elapsed=round(time.time() - t0, 1), hf_head=HF_HEAD, lf_target=LF_TARGET, warm=WARM, act=ACT, head_phi=HEAD_PHI, hf_only=HF_ONLY, budget_used=r['budget_used'], resumed_from=r['resumed_from'], fit_failed_at=r['fit_failed_at'], head_remove_gl=HEAD_REMOVE_GL, head_scalar=HEAD_SCALAR, head_offset=HEAD_OFFSET)])
                _append(pf, [dict(benchmark=bn, model=ml, seed=seed, **p) for p in r['plog']])
                import pickle
                pickle.dump(r['rng_state'], open(out / 'cells' / f'rng_{bn}_{ml}_s{seed}.pkl', 'wb'))
                _append(tf, [dict(benchmark=bn, model=ml, seed=seed, budget=bb, regret=rr) for (bb, rr) in zip(r['budgets'], r['regrets'])])
                print(f"{bn} {ml} s{seed}: final={r['final_regret']:.4f} auc={r['auc']:.4f} ece={r['ece_lf_mean']:.3f} ({time.time() - t0:.0f}s)", flush=True)
if __name__ == '__main__':
    main()
