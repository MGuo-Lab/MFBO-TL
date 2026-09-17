"""Calculate MLIP shear moduli for the Matbench structures."""
from __future__ import annotations
import argparse
import os
import time
import numpy as np
import pandas as pd

def _load_calc(model):
    m = model.lower()
    if m == 'chgnet':
        from chgnet.model.dynamics import CHGNetCalculator
        return CHGNetCalculator(use_device='cpu')
    if m in ('sevennet', '7net', 'sevenn'):
        from sevenn.calculator import SevenNetCalculator
        return SevenNetCalculator(model='7net-0')
    if m == 'mattersim':
        from mattersim.forcefield import MatterSimCalculator
        return MatterSimCalculator(device='cpu')
    raise ValueError(model)

def _mod(r, kind):
    want_shear = kind == 'shear'
    for (k, v) in r.items():
        kl = k.lower()
        if want_shear and 'shear' in kl and ('vrh' in kl):
            return float(v)
        if not want_shear and 'bulk' in kl and ('vrh' in kl):
            return float(v)
    for (k, v) in r.items():
        kl = k.lower()
        if want_shear and ('g_vrh' in kl or kl == 'g_vrh'):
            return float(v)
        if not want_shear and ('k_vrh' in kl or kl == 'k_vrh'):
            return float(v)
    return np.nan

def _worker(payload):
    (idxs, structs, model, fmax) = payload
    os.environ['OMP_NUM_THREADS'] = '1'
    import torch
    torch.set_num_threads(1)
    from matcalc import ElasticityCalc
    calc = _load_calc(model)
    ec = ElasticityCalc(calc, fmax=fmax, relax_structure=True, units_GPa=True)
    out = []
    for (i, s) in zip(idxs, structs):
        try:
            r = ec.calc(s)
            out.append((i, _mod(r, 'shear'), _mod(r, 'bulk')))
        except Exception:
            out.append((i, np.nan, np.nan))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['CHGNet', 'SevenNet', 'MatterSim'], default='CHGNet')
    ap.add_argument('--limit', type=int, default=0, help='0 = all rows')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--fmax', type=float, default=0.1)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from matminer.datasets import load_dataset
    print(f'[gen] loading matbench_log_gvrh (DFT HF) ...', flush=True)
    df = load_dataset('matbench_log_gvrh')
    if a.limit:
        df = df.iloc[:a.limit].reset_index(drop=True)
    structs = df['structure'].tolist()
    dft_g = 10.0 ** df['log10(G_VRH)'].to_numpy(float)
    n = len(structs)
    out = a.out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    done = set()
    if os.path.exists(out):
        try:
            done = set(pd.read_csv(out)['idx'].tolist())
        except Exception:
            done = set()
    todo = [i for i in range(n) if i not in done]
    print(f'[gen] N={n} done={len(done)} todo={len(todo)} model={a.model} workers={a.workers}', flush=True)
    if not todo:
        print('[gen] already complete', flush=True)
        return
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    nw = max(1, a.workers)
    SHARD = max(nw * 8, 64)
    t0 = time.time()
    for s0 in range(0, len(todo), SHARD):
        sub = todo[s0:s0 + SHARD]
        payloads = [(sub[i::nw], [structs[j] for j in sub[i::nw]], a.model, a.fmax) for i in range(nw) if sub[i::nw]]
        rows = None
        for attempt in (1, 2):
            try:
                with ctx.Pool(nw, maxtasksperchild=20) as pool:
                    chunks = pool.map(_worker, payloads)
                rows = [r for ch in chunks for r in ch]
                break
            except Exception as e:
                print(f"[gen] shard@{s0} attempt{attempt} FAILED ({type(e).__name__}); {('retry' if attempt == 1 else 'skip->NaN')}", flush=True)
        if rows is None:
            rows = [(j, float('nan'), float('nan')) for j in sub]
        d = pd.DataFrame(rows, columns=['idx', 'mlip_G', 'mlip_K'])
        d['dft_G'] = dft_g[d['idx'].to_numpy()]
        d['formula'] = [structs[i].composition.reduced_formula for i in d['idx']]
        d.to_csv(out, mode='a', header=not os.path.exists(out), index=False)
        print(f'[gen] {min(s0 + SHARD, len(todo))}/{len(todo)} done ({time.time() - t0:.0f}s, {(time.time() - t0) / max(s0 + len(sub), 1):.2f}s/struct)', flush=True)
    full = pd.read_csv(out)
    ok = int(full['mlip_G'].notna().sum())
    print(f'[gen] wrote {out}: {len(full)} rows, ok={ok}/{n}', flush=True)
if __name__ == '__main__':
    main()
