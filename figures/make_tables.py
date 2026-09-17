"""Reproduce the numerical supplementary tables from the per-seed results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures/out'
OUT.mkdir(exist_ok=True)
LABEL = {'MFGP':'GP-base','SparseMFGP':'GP-SV','Sparse MFGP':'GP-SV','DKL':'GP-DKL','DKL Multi-Fidelity':'GP-DKL',
         'DNGOJoint':'TL-base','Frozen-representation transfer':'TL-base','TwoStageJoint':'TL-PtJ','Pretrain-then-Joint':'TL-PtJ',
         'DNGOGradient':'TL-E2E','End-to-End Joint':'TL-E2E','SoftParameterSharing':'TL-SPS','Soft Parameter Sharing':'TL-SPS',
         'DomainAdaptationMMD':'TL-MMD','Domain Adaptation (MMD)':'TL-MMD','SF-GP EI (own init)':'SFGP EI','Random HF':'Random HF','HFOnlyNet':'HF-only network'}


def summarize(frame, keys, values):
    grouped = frame.groupby(keys, sort=False)
    out = grouped.size().rename('n').to_frame()
    for column in values:
        out[column + '_mean'] = grouped[column].mean()
        out[column + '_se'] = grouped[column].sem()
    return out.reset_index()


def ablations():
    f = pd.read_csv(ROOT/'results/controls/feature_ablation.csv')
    f['model'] = f.model.map(LABEL)
    f['arm'] = f.arm.replace({'1_learned':'learned','2_randphi':'random features','4_nophi':'scalar head','3_hfonly':'HF-only network'})
    summarize(f, ['pool','model','arm'], ['attainment','final_regret']).to_csv(OUT/'feature_ablation_summary.csv',index=False)
    h = pd.read_csv(ROOT/'results/controls/head_path.csv')
    summarize(h, ['pool','head_phi','head_scalar','head_offset'], ['attainment','final_regret']).to_csv(OUT/'head_path_summary.csv',index=False)


def hf_only():
    f = pd.read_csv(ROOT/'results/controls/hf_only.csv')
    f['model'] = f.model.map(LABEL)
    f['window'] = f.pool.map(lambda p: 50 if p.startswith('Branin') or p == 'FreeSolv' else 10 if p.startswith('Park') else 30)
    f['exact_optimum'] = f.final == 0
    summary = summarize(f, ['pool','model','window'], ['attain','final'])
    zeros = f.groupby(['pool','model','window'],sort=False).exact_optimum.sum().reset_index(name='n_exact_optimum')
    summary.merge(zeros,on=['pool','model','window']).to_csv(OUT/'hf_only_summary.csv',index=False)


def tlbase_comparison():
    f = pd.read_csv(OUT/'fig1_attainment_values.csv')
    rows = []
    for pool,g in f.groupby('pool',sort=False):
        base = g[g.abbr=='TL-base'].iloc[0]
        gp = g[g.abbr=='GP-base'].iloc[0]
        best = g[g.family=='GP'].sort_values('mean',ascending=False,kind='stable').iloc[0]
        rows.append(dict(pool=pool,tlbase_mean=base['mean'],tlbase_se=base.se,gpbase_mean=gp['mean'],gpbase_se=gp.se,
                         delta_gpbase=base['mean']-gp['mean'],best_gp=best.abbr,best_gp_mean=best['mean'],best_gp_se=best.se,
                         delta_best_gp=base['mean']-best['mean']))
    pd.DataFrame(rows).to_csv(OUT/'tlbase_vs_gp.csv',index=False)


def pick_logs():
    pool = pd.read_csv(ROOT/'data/hopv15.csv')
    hf,lf = -pool.HF.to_numpy(), -pool.LF.to_numpy()
    optimum = hf.min(); screen = hf[np.argmin(lf)]-optimum
    picks = pd.read_csv(ROOT/'results/controls/hopv15_picks.csv')
    seed_rows=[]; query_rows=[]
    for (model,seed),g in picks.groupby(['model','seed'],sort=False):
        init=g[g.step==0]; low=set(init.loc[init.fid=='L0','idx'].astype(int)); high=set(init.loc[init.fid=='H0','idx'].astype(int))
        best=hf[sorted(high)].min(); improved_after=False
        for row in g[g.step>0].sort_values('step',kind='stable').itertuples():
            idx=int(row.idx)
            if row.fid=='H':
                promoted=idx in low; improved=hf[idx]<best-1e-12
                high.add(idx); budget=round(len(low)*.1+len(high),9); best=min(best,hf[idx])
                if budget>15+1e-9:improved_after |= improved
                query_rows.append(dict(model=LABEL[model],seed=seed,budget=budget,promoted=int(promoted),regret=float(best-optimum)))
            else:low.add(idx)
        regret=float(best-optimum)
        seed_rows.append(dict(model=LABEL[model],seed=seed,final_regret=regret,final_at_lf_top=float(abs(regret-screen)<=1e-12),
                              improved_after_15=float(improved_after),reached_zero=float(regret==0)))
    seeds=pd.DataFrame(seed_rows);queries=pd.DataFrame(query_rows)
    summary=summarize(seeds,['model'],['final_regret','final_at_lf_top','improved_after_15'])
    summary['n_reached_zero']=summary.model.map(seeds.groupby('model').reached_zero.sum())
    summary['promoted_second_half']=summary.model.map(queries[queries.budget>15+1e-9].groupby('model').promoted.mean())
    summary.to_csv(OUT/'hopv15_picklog_summary.csv',index=False)
    seeds.to_csv(OUT/'hopv15_picklog_seeds.csv',index=False)


if __name__ == '__main__':
    ablations()
    hf_only()
    tlbase_comparison()
    pick_logs()
    print('Wrote supplementary table values to', OUT)
