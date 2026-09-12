#!/usr/bin/env python3
"""Frozen static-RIFT causal input controls (identity, diagonal z-score, CORAL).

The controls estimate session statistics from pooled held-in raw support only;
target labels are never read by a transform.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys
from pathlib import Path
from typing import Any
import numpy as np
import torch
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; WS=ROOT.parent
for p in (Path('/home/xinyuan/miniconda3/envs/spint/lib/python3.10/site-packages'), HERE, ROOT/'learnable_recency_v1/src', ROOT/'learnable_recency_v1/scripts', ROOT/'btransform_unified_v1/src', WS):
 if str(p) not in sys.path: sys.path.insert(0,str(p))
from data import load_task
from metrics import prediction_report
from learnable_recency_v1.config import config_from_run_meta
from learnable_recency_v1.static_model import StaticLearnableRiftDecoder, StaticRiftStreamDecoder
from btransform_unified_v1.r2 import variance_weighted_r2
CKPTS={'m2':ROOT/'learnable_recency_v1/results/m2_static_learned_slope_s42/epoch_024.pt','m1':ROOT/'learnable_recency_v1/results/m1_static_s42/epoch_024.pt','h1':ROOT/'learnable_recency_v1/results/h1_static_s42/epoch_032.pt'}
EXPECTED={'m2':'8313e7bb202d43def2885b7e7c4c74dc8466af81aaccb18b437333ae86801612','m1':'562e0febc2818e2fc6952e383a5dea7fadbc6493034cee3557a315e1538cf4df','h1':'b79adeda51955c14bb60e4643835a5d8d6862cdb2480134fd9fd42a04a765b51'}
def sha_file(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def sha_arr(a):
 a=np.ascontiguousarray(a);h=hashlib.sha256();h.update(a.dtype.str.encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()
def fit(source, target, arm):
 s=np.concatenate([v['support'] for v in source.values()]).astype(np.float64); t=target['support'].astype(np.float64)
 if arm=='identity': return {'arm':arm}
 ms=s.mean(0); mt=t.mean(0); ss=np.maximum(s.std(0),1e-6); st=np.maximum(t.std(0),1e-6)
 if arm=='diag_z': return {'arm':arm,'ms':ms,'mt':mt,'ss':ss,'st':st}
 # covariance direction is target -> pooled held-in support
 cs=np.cov(s,rowvar=False);ct=np.cov(t,rowvar=False); n=cs.shape[0]; ridge=.001
 cs=(1-.1)*cs+.1*np.diag(np.diag(cs))+ridge*np.eye(n);ct=(1-.1)*ct+.1*np.diag(np.diag(ct))+ridge*np.eye(n)
 w,v=np.linalg.eigh(ct); inv=(v*(1/np.sqrt(np.maximum(w,1e-12))))@v.T
 w,v=np.linalg.eigh(cs); root=(v*np.sqrt(np.maximum(w,1e-12)))@v.T
 return {'arm':arm,'ms':ms,'mt':mt,'A':inv@root,'shrinkage':.1,'ridge':ridge}
def transform(raw, f):
 if f['arm']=='identity': return raw.copy()
 if f['arm']=='diag_z': return ((raw-f['mt'])/f['st']*f['ss']+f['ms']).astype(np.float32)
 return ((raw-f['mt'])@f['A']+f['ms']).astype(np.float32)
def contexts(item, raw):
 # `starts` address padded X; all existing task loaders use start: start+context.
 p=int(item['pad']); c=len(raw)-p # unused; retain literal zero left pad
 x=np.zeros_like(item['X'],dtype=np.float32); x[p:]=raw
 starts=np.asarray(item['starts'],np.int64); width=None
 # derives context from the fixed window/label layout externally
 return x,starts
def model_for(task):
 ck=CKPTS[task]
 if sha_file(ck)!=EXPECTED[task]: raise RuntimeError(f'{task}: frozen checkpoint hash drift')
 meta=json.loads((ck.parent/'run_meta.json').read_text()); state=torch.load(ck,map_location='cpu',weights_only=False)
 m=StaticLearnableRiftDecoder(task,config_from_run_meta(meta,task),context_bins=int(meta['context_bins']),seed=int(meta.get('seed',42))).cpu();m.temporal.set_attention_backend('local');m.load_state_dict(state['raw_state_dict'],strict=True)
 shadow=state['ema']['shadow']; named=dict(m.named_parameters())
 if set(shadow)!=set(named):raise RuntimeError('EMA schema drift')
 with torch.no_grad():
  for n,p in named.items():p.copy_(shadow[n].to(p.dtype))
 m.eval();return m,meta
def predict(task,m,item,raw,batch=32):
 x,starts=contexts(item,raw); context={'m1':100,'m2':50,'h1':300}[task]; out=[]
 for off in range(0,len(starts),batch):
  ss=starts[off:off+batch]; z=np.stack([x[i:i+context] for i in ss]); valid=np.stack([np.arange(i,i+context)>=item['pad'] for i in ss])
  with torch.inference_mode(): q=m(torch.from_numpy(z),input_valid_mask=torch.from_numpy(valid)).numpy()
  out.append(q)
 p=np.concatenate(out); 
 if task=='m2':
  from tfpd_exploration.src.m2_dual_track_v1 import plan as old_plan
  return p/float(old_plan.BEHAVIOR_SCALE)
 if task=='h1':
  from btransform_unified_v1 import h1_config
  return p/float(h1_config.TARGET_MULTIPLIER)
 return p
def metric(y,p):
 try:
  from sklearn.metrics import r2_score
  sk=float(r2_score(y,p,multioutput='variance_weighted'))
 except Exception: sk=float(variance_weighted_r2(y,p))
 return {'sklearn_variance_weighted_r2':sk,'legacy_flattened_r2':float(variance_weighted_r2(y,p))}
def stream_parity(task,m,item,raw):
 # Test contiguous real rows only; offline contexts start at actual padded indexes.
 p=item['pad']; n=min(12,len(raw)); start=p+min(10,max(0,len(raw)-n)); xx=np.zeros_like(item['X']);xx[p:]=raw
 with torch.inference_mode():
  off=m(torch.from_numpy(np.stack([xx[i-({'m1':99,'m2':49,'h1':299}[task]):i+1] for i in range(start,start+n)])),input_valid_mask=torch.ones((n, {'m1':100,'m2':50,'h1':300}[task]),dtype=torch.bool)).numpy()
 st=StaticRiftStreamDecoder(m); got=[]
 for i in range(start-({'m1':99,'m2':49,'h1':299}[task]),start+n): got.append(st.stream_step(torch.from_numpy(xx[i:i+1]),['s']).numpy()[0])
 got=np.asarray(got[-n:]); err=float(np.max(np.abs(off-got)))
 if err>2e-5: raise RuntimeError(f'stream/offline parity drift {err}')
 return err
def run(task,dest):
 if dest.exists():raise FileExistsError(f'destination must be fresh: {dest}')
 third_party=str(WS/'SPINT-main/third_party')
 if third_party in sys.path: sys.path.remove(third_party) # prefer complete installed FALCON package over the sample-only copy
 data=load_task(task,include_evaluation=True);m,meta=model_for(task); dest.mkdir(parents=True)
 stream_err={}; arms={}
 for arm in ('identity','diag_z','coral'):
  predictions={}; provenance={}
  for session,item in data['evaluation'].items():
   f=fit(data['train'],item,arm); raw=transform(np.asarray(item['X'])[item['pad']:],f); pred=predict(task,m,item,raw); y=np.asarray(item['Y']);
   if arm=='identity':stream_err[session]=stream_parity(task,m,item,raw)
   np.save(dest/f'{arm}_{session}_pred.npy',pred)
   predictions[session]=pred
   provenance[session]={'support_n':len(item['support']),'support_provenance':item['support_provenance']}
  summary=prediction_report(task,data['evaluation'],predictions)
  for session,row in summary['per_session'].items(): row.update(provenance[session])
  arms[arm]=summary
 report={'schema':'fair_v2_static_rift_controls_v1','task':task,'checkpoint':str(CKPTS[task]),'checkpoint_sha256':EXPECTED[task],'view':'EMA frozen','device':'cpu','transform_contract':'pooled ALL held-in raw support; target raw support only; no Y; input prior to frontend.local_conv; literal zero padded rows','coral':{'shrinkage':.1,'ridge':.001,'formula':'(x-mu_target) Ct^-1/2 Cs^1/2 + mu_source'},'offline_stream_real_row_max_abs_error':stream_err,'arms':arms}
 (dest/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,default=str)+'\n');return report
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--task',choices=['m1','m2','h1'],required=True);ap.add_argument('--dest',type=Path,required=True);a=ap.parse_args(); print(json.dumps(run(a.task,a.dest),indent=2,default=str))
if __name__=='__main__':main()
