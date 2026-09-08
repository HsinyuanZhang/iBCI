#!/usr/bin/env python3
"""Validate and plot the fixed-e3 M1 carrier intervention ledger; never scores."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--input',type=Path,required=True);a.add_argument('--outdir',type=Path,required=True);x=a.parse_args()
 if x.outdir.exists():raise FileExistsError(x.outdir)
 d=json.loads(x.input.read_text()); R=d['results']; S=('20121004','20121017','20121024');keys={'M10','ZERO','SHUF_101','SHUF_102','SHUF_103'}|{f'M{k}_{z}' for k in (5,8) for z in range(101,111)}
 if d.get('status')!='COMPLETED' or set(R)!=keys:raise RuntimeError(f'exact 25-condition ledger required; got {len(R)}')
 if sha(d['checkpoint'])!=d['checkpoint_sha256']:raise RuntimeError('checkpoint SHA mismatch')
 score=Path(d['checkpoint']).parent/'score_receipt.json'
 if sha(score)!=d['score_receipt_sha256']:raise RuntimeError('score receipt SHA mismatch')
 for k,row in R.items():
  if row.get('partial') or row.get('n_windows')!=3881 or set(row.get('per_session',{}))!=set(S) or set(row.get('carrier_sha256',{}))!=set(S):raise RuntimeError(f'{k}: incomplete ledger')
  vals=[]
  for s in S:
   q=row['per_session'][s]
   if q.get('window_count') not in (1305,1295,1281) or not np.isfinite(q.get('r2',np.nan)) or len(q.get('prediction_sha256',''))!=64 or len(row['carrier_sha256'][s])!=64:raise RuntimeError(f'{k}/{s}: invalid')
   vals.append(q['r2'])
  if not np.isfinite(row.get('equal_session_mean',np.nan)) or abs(np.mean(vals)-row['equal_session_mean'])>1e-8:raise RuntimeError(f'{k}: mean mismatch')
 def stat(names):
  v=np.array([[R[n]['per_session'][s]['r2'] for s in S] for n in names]);return {'conditions':names,'per_session':v.tolist(),'mean':float(v.mean()),'condition_mean_range':[float(v.mean(1).min()),float(v.mean(1).max())]}
 direct={'REAL':stat(['M10']),'ZERO':stat(['ZERO']),'SHUFFLE':stat([f'SHUF_{z}' for z in (101,102,103)])}
 budget={f'M{k}':stat([f'M{k}_{z}' for z in range(101,111)]) for k in (5,8)};budget['M10']=stat(['M10'])
 # Bootstrap samples sessions, not models; descriptive display only, no inference claim.
 rng=np.random.default_rng(0);boot={name:np.mean(np.asarray(v['per_session'])[:,rng.integers(0,3,3)],axis=(0,1)).tolist() for name,v in {**direct,**budget}.items()}
 out={'schema':'m1_rift_e3_inference_figure_data_v1','input_sha256':sha(x.input),'checkpoint_sha256':d['checkpoint_sha256'],'score_receipt_sha256':d['score_receipt_sha256'],'conditions':sorted(keys),'fixed_E0':'M10 B3 identity for all conditions','budget_label':'functional direct-carrier support only: M5=50%, M8=80%, M10=100%; not total calibration reduction or retraining','direct_intervention':direct,'support_budget':budget,'session_bootstrap_descriptive':{'resampling_unit':'3 sessions; not 10 model seeds','draws':1000,'values':boot,'not_significance_test':True}}
 x.outdir.mkdir();(x.outdir/'figure_data.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(10,4))
 for label,v,color in [('REAL',direct['REAL'],'#222222'),('ZERO',direct['ZERO'],'#d55e00'),('SHUFFLE',direct['SHUFFLE'],'#0072b2')]:
  y=np.mean(v['per_session'],axis=1);ax[0].scatter(np.full(len(y),label),y,color=color);ax[0].plot([label]*len(y),y,'_',color=color);ax[0].set_title('Fixed-e3 direct carrier intervention');ax[0].set_ylabel('equal-session R²')
 for label in ('M5','M8','M10'):
  v=budget[label];y=np.mean(v['per_session'],axis=1);ax[1].scatter(np.full(len(y),label),y);ax[1].plot([label]*len(y),y,'_')
 ax[1].set_title('Direct-carrier support only; E0 fixed M10');ax[1].set_ylabel('equal-session R²')
 fig.tight_layout();fig.savefig(x.outdir/'m1_inference_evidence.png',dpi=300);fig.savefig(x.outdir/'m1_inference_evidence.pdf')
if __name__=='__main__':main()
