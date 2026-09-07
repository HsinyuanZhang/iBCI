"""Archive-only startup partition for four frozen finalizer outputs."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]; F=ROOT/'tfpd_exploration/results/m2/family_v1/finalize_pair_v1'; R=F/'receipt.json'; H=ROOT/'tfpd_exploration/results/m2/family_v1/source_minival_e8_spint_m30_replay_v1.npz'; O=ROOT/'tfpd_exploration/results/m2/family_v1/finalized_startup_diagnostic_v1.json'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def m(y,p):
 d=y-y.mean(0); return {'count':len(y),'r2':float(1-((y-p)**2).sum()/(d*d).sum()),'mse':float(((y-p)**2).mean()),'prediction_std':p.std(0).tolist(),'correlation_per_output':[float(np.corrcoef(y[:,i],p[:,i])[0,1]) if len(y)>1 and y[:,i].std()>0 and p[:,i].std()>0 else None for i in range(2)]}
def rows(y,p,s,start):
 out=[]
 for n in list(dict.fromkeys(s.tolist()))+['POOLED']:
  base=np.ones(len(s),bool) if n=='POOLED' else s==n; q={'session':n,'all1011_primary_descriptive':m(y[base],p[base]),'partitions':{}}
  for k,z in [('startup_start_lt49',base&(start<49)),('full_w50_start_ge49',base&(start>=49))]: q['partitions'][k]=m(y[z],p[z])
  out.append(q)
 return out
def run():
 r=json.loads(R.read_text());
 with np.load(H,allow_pickle=False) as z: y=z['target']; s=z['session']; start=z['start']; hist={'e8':z['e8_prediction'],'spint':z['spint_prediction']}
 result={'schema':'m2_family_v1_finalized_startup_diagnostic_v1','status':'DESCRIPTIVE_ARCHIVE_ONLY_ALL1011_PRIMARY_NOT_SELECTION','finalizer_receipt_sha256':sha(R),'historical_npz_sha256':sha(H),'partition_rule':'start<49 versus start>=49','interpretation':'Partitions test only the predeclared zero-history hypothesis. All-1011 equal-session diagnostics remain primary; no endpoint or model is promoted.','historical':{k:rows(y,v,s,start) for k,v in hist.items()},'finalized':{}}
 for key,score in r['scores'].items():
  p=F/f'{key}_source_minival.npz'
  if not p.is_file() or sha(p)!=score['npz_sha256']: raise RuntimeError('finalizer NPZ hash drift')
  with np.load(p,allow_pickle=False) as z:
   if not(np.array_equal(y,z['target']) and np.array_equal(s,z['session']) and np.array_equal(start,z['start'])): raise RuntimeError('archive session/start/target mismatch')
   pred=z['prediction']
  result['finalized'][key]={'npz_sha256':sha(p),'recorded_equal_session_r2':score['equal_session_r2'],'recorded_pooled_r2':score['pooled_r2'],'rows':rows(y,pred,s,start)}
 O.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); return result
if __name__=='__main__': print(sha(O) if O.exists() else (run() and sha(O)))
