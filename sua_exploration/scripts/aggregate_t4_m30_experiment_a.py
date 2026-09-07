#!/usr/bin/env python3
"""Fail-closed aggregate for the frozen 18-cell Experiment-A matrix (no launch logic)."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
ARMS=('z4','ph4','ac4','mb4','b4','ls4'); SEEDS=(42,43,44); SESSIONS=6
def require(x,m):
 if not x: raise ValueError(m)
def load(root,arm,seed):
 p=root/f'{arm}_s{seed}.json'; require(p.is_file(),f'missing {p}'); d=json.loads(p.read_text()); q=d.get('protocol',{}); require((d.get('seed'),d.get('signal_view'),q.get('calibration_n'),q.get('pool_size'),d.get('epoch_list'))==(seed,'sua',30,30,list(range(5,13))),f'contract {p}')
 vals=[]
 for epoch in map(str,range(5,13)):
  row=d['per_epoch'][epoch]['per_session_r2']; require(len(row)==SESSIONS,f'sessions {p}'); vals.append(row)
 names=sorted(vals[0]); return np.asarray([[np.mean([e[n] for e in vals]) for n in names]]),names
def decision(delta):
 session=delta.mean(0); seed=delta.mean(1); mean=float(delta.mean()); se=float(seed.std(ddof=1)/math.sqrt(3)); lower=mean-2*se
 return {'mean_delta':mean,'seed_mean_se':se,'paired_two_se_lower':lower,'positive_session_means':int((session>0).sum()),'positive_seed_means':int((seed>0).sum()),'state':'effective' if mean>=.03 and (session>0).sum()>=5 and (seed>0).sum()==3 and lower>0 else ('ineffective_for_practical_0.03' if mean+2*se<.03 else 'indeterminate')}
def main():
 p=argparse.ArgumentParser();p.add_argument('--result-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();require(not a.out.exists(),'write once')
 data={}; names=None
 for arm in ARMS:
  rows=[]
  for seed in SEEDS:
   x,n=load(a.result_dir,arm,seed); rows.append(x[0]); names=n if names is None else names
  data[arm]=np.asarray(rows)
 contrasts={arm:{'vs_z4':decision(data[arm]-data['z4']),'vs_t4_reference':'requires qualified reused T4 aggregate'} for arm in ARMS if arm!='z4'}
 out={'schema_version':1,'arms':list(ARMS),'seeds':list(SEEDS),'sessions':names,'per_arm_session_seed_epoch_window_r2':{k:v.tolist() for k,v in data.items()},'contrasts':contrasts,'component_sufficiency':'requires separately qualified T4 reference and hierarchical bootstrap/exact Wilcoxon; no row-shuffle extension launched here','conditional_row_shuffle_launched':False}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()
