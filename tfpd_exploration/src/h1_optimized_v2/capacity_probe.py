"""Fixed, source-train-only 208-window capacity/learnability probe."""
from __future__ import annotations
import json, time, hashlib
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from .cache import ROOT as CACHE_ROOT, build_or_load, validate_authority
from .model import make_matched_pair
from .paired_train import groups
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank

OUT=CACHE_ROOT/'capacity_probe_208_source_v2'; W=700; LR=2e-4; MAX=260; DEADLINE=600
def sha(p):
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def r2(p,y):
 p,y=p.astype('float64'),y.astype('float64');return float(1-((p-y)**2).sum()/((y-y.mean(0))**2).sum())
def ids(cache):
 out={}
 for n,row in sorted(cache['train'].items()):
  s=np.asarray(row['query_starts']);out[n]=s[np.linspace(0,len(s)-1,16,dtype=int)].astype(int)
 return out
def batch(row, starts,dev):
 x=np.stack([row['neural'][s:s+W] for s in starts]).astype('float32');y=np.stack([row['velocity'][s+W-1] for s in starts]).astype('float32')
 return torch.as_tensor(x,device=dev),torch.as_tensor(y*20,device=dev),y
def score(m,cache,fixed,dev):
 pp=[];yy=[];m.eval()
 with torch.no_grad():
  for n,s in fixed.items():
   row=cache['train'][n];x,_,y=batch(row,s,dev);b=H1Bank(row['bank']['E0'].to(dev),row['bank']['T'].to(dev),row['bank']['unit_mask'].to(dev));pp.extend([(m.forward_last(x[i:i+4],b)/20).cpu().numpy() for i in range(0,len(x),4)]);yy.extend([y[i:i+4] for i in range(0,len(y),4)])
 p,y=np.concatenate(pp),np.concatenate(yy);return {'r2_concat':r2(p,y),'prediction_std':float(p.std()),'target_std':float(y.std()),'prediction_mean':float(p.mean()),'target_mean':float(y.mean())}
def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir(parents=True);cache=build_or_load();auth=json.loads((CACHE_ROOT/'source_cache_authority.json').read_text());validate_authority(cache,auth);fixed=ids(cache)
 (OUT/'frozen_ids.json').write_text(json.dumps({'ids':{k:v.tolist() for k,v in fixed.items()},'source_authority':auth,'recipe':{'windows':208,'per_session':16,'batch':16,'lr':LR,'updates':MAX,'dropout':'disabled','target':'20x native','selection':'none'}},indent=2,sort_keys=True)+'\n')
 dev=torch.device('cuda:0');full,t=make_matched_pair(activity_scale=32.);arms={'full':full.to(dev),'t':t.to(dev)};opt={k:torch.optim.AdamW(groups(m),lr=LR) for k,m in arms.items()};report={'schema':'h1_capacity_probe_208_source_v1','status':'RUNNING','input_authority':auth,'frozen_ids_sha256':sha(OUT/'frozen_ids.json'),'before':{k:score(m,cache,fixed,dev) for k,m in arms.items()},'updates_requested':MAX,'deadline_s':DEADLINE};start=time.time();done=0
 try:
  names=list(fixed)
  for step in range(MAX):
   if time.time()-start>DEADLINE:break
   n=names[step%len(names)];row=cache['train'][n];x,y,_=batch(row,fixed[n],dev);b=H1Bank(row['bank']['E0'].to(dev),row['bank']['T'].to(dev),row['bank']['unit_mask'].to(dev));keep=torch.ones((len(x),176),device=dev,dtype=torch.bool)
   for k,m in arms.items():
    opt[k].zero_grad(set_to_none=True)
    for i in range(0,len(x),4): (F.mse_loss(m.forward_last(x[i:i+4],b,dropout_keep=keep[i:i+4]),y[i:i+4])*.25).backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt[k].step()
   done=step+1
 finally:
  report.update({'status':'COMPLETE' if done==MAX else 'TIMEOUT_OR_INTERRUPTED','elapsed_s':time.time()-start,'updates_completed':done,'after':{k:score(m,cache,fixed,dev) for k,m in arms.items()}})
  for k,m in arms.items():torch.save({'model':m.state_dict(),'optimizer':opt[k].state_dict(),'updates':done},OUT/f'{k}_latest.pt')
  (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps(report,sort_keys=True))
if __name__=='__main__':main()
