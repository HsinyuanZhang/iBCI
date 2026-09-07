"""Descriptive startup-vs-full-window partition of the sealed 1011 replay archive."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]; IN=ROOT/'tfpd_exploration/results/m2/family_v1/source_minival_e8_spint_m30_replay_v1.npz'; OUT=ROOT/'tfpd_exploration/results/m2/family_v1/e8_source_minival_startup_diagnostic_v1.json'
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def met(y,p):
 d=y-y.mean(0); return {'count':len(y),'r2':float(1-((y-p)**2).sum()/(d*d).sum()),'mse':float(((y-p)**2).mean()),'prediction_std':p.std(0).tolist(),'target_std':y.std(0).tolist(),'correlation_per_output':[float(np.corrcoef(y[:,i],p[:,i])[0,1]) if len(y)>1 and y[:,i].std()>0 and p[:,i].std()>0 else None for i in range(2)]}
def run():
 with np.load(IN,allow_pickle=False) as z: e=z['e8_prediction'].astype('float64');s=z['spint_prediction'].astype('float64');y=z['target'].astype('float64');start=z['start'];session=z['session']
 rows=[]
 for name in list(dict.fromkeys(session.tolist()))+['POOLED']:
  base=np.ones(len(y),bool) if name=='POOLED' else session==name
  part={}
  for label,mask in [('startup_start_lt49',base&(start<49)),('full_w50_start_ge49',base&(start>=49))]:
   part[label]={'start_min':None if not mask.any() else int(start[mask].min()),'start_max':None if not mask.any() else int(start[mask].max()),'e8':met(y[mask],e[mask]),'spint':met(y[mask],s[mask])}
  rows.append({'session':name,'partitions':part})
 return {'schema':'m2_family_v1_archive_startup_diagnostic_v1','status':'DESCRIPTIVE_ARCHIVE_ONLY_NOT_PRIMARY_OR_SELECTION','input_npz_sha256':h(IN),'partition_rule':'startup start<49 versus full W50 start>=49','interpretation':'The startup partition contains zero-history windows by construction. Differences can reflect partition size, target distribution, and target variance; this diagnostic neither overrides all-1011 primary metrics nor proves a causal mechanism.','rows':rows}
if __name__=='__main__': OUT.write_text(json.dumps(run(),indent=2,sort_keys=True)+'\n'); print(h(OUT))
