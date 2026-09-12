#!/usr/bin/env python3
"""CPU-only source-day Wiener and neural-only target-adaptation baselines."""
from __future__ import annotations
import argparse, json, pickle, time, sys, hashlib
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.metrics import r2_score
WS=Path(__file__).resolve().parents[2]
if str(WS/'btransform_unified_v1/src') not in sys.path: sys.path.insert(0,str(WS/'btransform_unified_v1/src'))
from . import data
from .wiener import WienerRidge
from btransform_unified_v1.r2 import variance_weighted_r2

def _atom(p:Path,x:Any):
 p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix('.tmp'); q.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)); q.replace(p)
def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _activity(item): return np.asarray(item['activity'],np.float32).reshape(-1,np.asarray(item['activity']).shape[-1])
def _r2(y,p): return float(r2_score(np.asarray(y),np.asarray(p),multioutput='variance_weighted'))
def _fit_features(item,ids,context,transform=None,history=10):
 x=data.causal_features(item,ids,context=context,history=history)
 if transform is None:return x
 n=x.shape[0]; c=np.asarray(item['X']).shape[1]; bins=x.reshape(n,history,c)
 # Transform only 2-D neural rows.  Restore literal zero padding afterward:
 # centering transforms must not make an invalid causal bin appear observed.
 valid=np.asarray(item['starts'],np.int64)[ids,None]+context-history+np.arange(history)[None,:]>=int(item.get('pad',0))
 out=transform(bins.reshape(-1,c)).reshape(n,history,-1); out[~valid]=0.
 return out.reshape(n,-1)
def _fit(item,context,history,ridge,transform=None):
 ids=np.arange(len(item['starts'])); x=_fit_features(item,ids,context,transform,history); return WienerRidge(ridge).fit(x,np.asarray(item['Y'],np.float32))
def _score(model,item,context,history,transform=None,max_batches=None):
 ps=[];ts=[]
 for st in range(0,len(item['starts']),32):
  if max_batches is not None and st//32>=max_batches:break
  ids=np.arange(st,min(st+32,len(item['starts'])));ps.append(model.predict(_fit_features(item,ids,context,transform,history)));ts.append(np.asarray(item['Y'])[ids])
 p=np.concatenate(ps);t=np.concatenate(ts);return p,t,_r2(t,p)
def run(a):
 t0=time.monotonic(); d=data.load(a.task,evaluation=True); ctx=int(d['metadata']['context']); train=d['train']; ref=data.reference_source_day(train); src=train[ref]
 # activity is neural-only: source is used solely to fit adapters, target activity solely to calibrate them.
 from btransform_unified_v2.external_baselines_v1.core import CoralAligner,AlignedFA
 source_a=_activity(src); raw=_fit(src,ctx,a.history,a.ridge)
 coral=CoralAligner(source_a,ridge=a.coral_ridge,shrinkage=a.coral_shrinkage)
 fa=AlignedFA(source_a,latent_dim=a.fa_dim,seed=42,max_iter=a.fa_max_iter,n_init=a.fa_n_init,stable_fraction=a.fa_stable_fraction)
 fa_raw=_fit(src,ctx,a.history,a.ridge,fa.source_transform)
 methods={'wf_raw':(raw,None),'coral_wf':(raw,'coral'),'fa_wf':(fa_raw,fa.source_transform),'aligned_fa_wf':(fa_raw,'aligned_fa')}
 reports={}; preds={}; adapters={}
 for name,(model,kind) in methods.items():
  method_started=time.monotonic()
  rows={}; allp=[];allt=[];preds[name]={}; grouped_preds={};grouped_targets={};grouped_masks={}
  for ses,item in d['evaluation'].items():
   target_a=_activity(item)
   transform=None; diag={'target_labels_used':False,'target_backprop_used':False}
   if kind=='coral': ad=coral.calibrate(target_a); transform=ad.transform; diag=ad.diagnostics; adapters[(name,ses)]=ad
   elif kind=='aligned_fa': ad=fa.calibrate(target_a); transform=ad.transform; diag=ad.diagnostics; adapters[(name,ses)]=ad
   elif kind is not None: transform=kind
   p,y,r=_score(model,item,ctx,a.history,transform,a.max_batches); preds[name][ses]=p;rows[ses]={'standard_variance_weighted_r2':r,'full_compatible_flattened_r2':float(variance_weighted_r2(y,p)),'n':len(y),'adapter':diag};allp.append(p);allt.append(y); grouped_preds[ses]=p;grouped_targets[ses]=y;grouped_masks[ses]=np.ones(len(y),bool)
  reports[name]={'per_session':rows,'standard_variance_weighted_equal_session_mean':float(np.mean([r['standard_variance_weighted_r2'] for r in rows.values()])),'full_compatible_flattened_equal_session_mean':float(np.mean([r['full_compatible_flattened_r2'] for r in rows.values()])),'standard_variance_weighted_pooled_r2':_r2(np.concatenate(allt),np.concatenate(allp)),'full_compatible_flattened_pooled_r2':float(variance_weighted_r2(np.concatenate(allt),np.concatenate(allp))),'n_windows':int(sum(r['n'] for r in rows.values())),'calibration_predict_seconds':time.monotonic()-method_started}
  if a.task=='h1':
   from btransform_unified_v1.c2_protocol import HELDOUT_SESSION_TO_FALCON_KEY, grouped_session_metrics
   def pick(d,session): return d[session] if session in d else d[session.removeprefix('ses-')]
   mp={key:pick(grouped_preds,session) for session,key in HELDOUT_SESSION_TO_FALCON_KEY}; mt={key:pick(grouped_targets,session) for session,key in HELDOUT_SESSION_TO_FALCON_KEY}; mm={key:np.ones(len(mp[key]),bool) for key in mp}
   reports[name]['grouped_seven']=grouped_session_metrics(mp,mt,mm,HELDOUT_SESSION_TO_FALCON_KEY)
 if a.dest.exists() and any(a.dest.iterdir()): raise FileExistsError('fresh --dest must be empty')
 out={'schema':'external_gf_baselines_v1','task':a.task,'status':'SMOKE' if a.max_batches else 'COMPLETED','reference_source_day':ref,'reference_source_train_windows':len(src['starts']),'config':vars(a),'target_labels_used_for_fit':False,'target_backprop_used':False,'metric_scope':'public_calibration_local_development','official_test_used':False,'channel_coordinate_assumption':'positional rows; physical correspondence unverified' if a.task=='h1' else 'fixed unit-row/electrode correspondence','reports':reports,'runtime_seconds':time.monotonic()-t0,'source_hashes':{'source_activity':data.sha_array(source_a),'source_Y':data.sha_array(src['Y']),'source_X':data.sha_array(src['X']),'source_starts':data.sha_array(src['starts']),'run.py':_sha(__file__),'data.py':_sha(Path(__file__).with_name('data.py')),'wiener.py':_sha(Path(__file__).with_name('wiener.py'))},'target_support_hashes':{s:data.sha_array(_activity(v)) for s,v in d['evaluation'].items()},'target_input_hashes':{s:v.get('hashes',{}) for s,v in d['evaluation'].items()},'target_support_provenance':{s:v.get('support_provenance',{}) for s,v in d['evaluation'].items()},'surface_note':'support activity may be trialized/cubic while query X is raw neural; this paired control does not claim identical preprocessing'}
 a.dest.mkdir(parents=True,exist_ok=True)
 with (a.dest/'models.pkl').open('wb') as f:pickle.dump({'wf_raw':raw,'fa_wf':fa_raw,'coral':coral,'fa':fa,'target_adapters':adapters},f)
 for name,by in preds.items():
  for ses,p in by.items():np.save(a.dest/f'pred_{name}_{ses}.npy',p)
 _atom(a.dest/'receipt.json',out)
 return out
def rescore(a):
 """Re-score saved predictions; never re-fit or touch target labels as inputs."""
 prior=a.rescore_run.resolve(); old=json.loads((prior/'receipt.json').read_text()); d=data.load(a.task,evaluation=True); reports={}
 for method in old['reports']:
  rows={}; ps=[];ts=[]
  for ses,item in d['evaluation'].items():
   p=np.load(prior/f'pred_{method}_{ses}.npy'); y=np.asarray(item['Y'])[:len(p)]
   rows[ses]={'standard_variance_weighted_r2':_r2(y,p),'full_compatible_flattened_r2':float(variance_weighted_r2(y,p)),'n':len(y)};ps.append(p);ts.append(y)
  reports[method]={'per_session':rows,'standard_variance_weighted_equal_session_mean':float(np.mean([x['standard_variance_weighted_r2'] for x in rows.values()])),'full_compatible_flattened_equal_session_mean':float(np.mean([x['full_compatible_flattened_r2'] for x in rows.values()])),'standard_variance_weighted_pooled_r2':_r2(np.concatenate(ts),np.concatenate(ps)),'full_compatible_flattened_pooled_r2':float(variance_weighted_r2(np.concatenate(ts),np.concatenate(ps))),'n_windows':sum(x['n'] for x in rows.values())}
 out={'schema':'external_gf_baselines_v1_corrected_score_v1','metric_contract':'both full_compatible_flattened and standard_variance_weighted are reported','original_run':str(prior),'original_receipt_sha256':_sha(prior/'receipt.json'),'task':a.task,'reports':reports,'target_labels_used_for_fit':False,'target_backprop_used':False,'official_test_used':False,'code_hashes':{n:_sha(Path(__file__).with_name(n)) for n in ('run.py','data.py','wiener.py','core.py')},'activity_data_sha256':_sha(Path(__file__).parents[1]/'learnable_recency_v1/scripts/activity_data.py')}
 if a.dest.exists() and any(a.dest.iterdir()):raise FileExistsError('fresh --dest must be empty')
 a.dest.mkdir(parents=True);_atom(a.dest/'receipt.json',out);return out
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',choices=('m1','m2','h1'),default='m2');p.add_argument('--dest',type=Path,required=True);p.add_argument('--history',type=int,default=10);p.add_argument('--ridge',type=float,default=1.);p.add_argument('--coral-ridge',type=float,default=1e-3);p.add_argument('--coral-shrinkage',type=float,default=.1);p.add_argument('--fa-dim',type=int,default=10);p.add_argument('--fa-max-iter',type=int,default=1000);p.add_argument('--fa-n-init',type=int,default=3);p.add_argument('--fa-stable-fraction',type=float,default=.5);p.add_argument('--max-batches',type=int);p.add_argument('--rescore-run',type=Path);a=p.parse_args();print(json.dumps(rescore(a) if a.rescore_run else run(a),indent=2,default=str))
if __name__=='__main__':main()
