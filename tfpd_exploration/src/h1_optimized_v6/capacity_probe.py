"""Fixed source-only V6 capacity gate; never imports an evaluation scorer."""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from tfpd_exploration.src.h1_optimized_v2.cache import ROOT,build_or_load,validate_authority
from tfpd_exploration.src.h1_optimized_v2.capacity_probe import ids,batch,r2
from tfpd_exploration.src.h1_optimized_v2.paired_train import groups
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
from .model import make_matched_pair
OUT=ROOT/'capacity_probe_208_source_v6_recency1040';LR=2e-4;UPDATES=1040;MICRO=4;EFFECTIVE=16;DEADLINE=600
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state_sha(m):
 h=hashlib.sha256()
 for n,t in sorted(m.state_dict().items()):h.update(n.encode());h.update(t.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def score(m,cache,fixed,d):
 ps=[];ys=[];m.eval()
 with torch.no_grad():
  for n,s in fixed.items():
   row=cache['train'][n];x,_,y=batch(row,s,d);b=H1Bank(*[row['bank'][q].to(d) for q in ('E0','T','unit_mask')]);ps.extend((m.forward_last(x[i:i+MICRO],b)/20).cpu().numpy() for i in range(0,len(x),MICRO));ys.extend(y[i:i+MICRO] for i in range(0,len(y),MICRO))
 p,y=np.concatenate(ps),np.concatenate(ys);return {'r2_concat':r2(p,y),'prediction_std':float(p.std()),'target_std':float(y.std())}
def main():
 if OUT.exists():raise FileExistsError(OUT)
 cache=build_or_load();auth=json.loads((ROOT/'source_cache_authority.json').read_text());validate_authority(cache,auth);fixed=ids(cache)
 recipe={'schema':'h1_v6_recency_source_gate_v1','disclosure':'known-source capacity gate for fixed multi-timescale query-read recency hypothesis; not formal/minival/hidden evidence','variable':'V6 query-only fixed per-head tau [2,4,8,16,32,64,128,infinity] prior added at query read to zero-initialized learned log16 bias; V4 signed frontend/FULL unchanged','frontend_contract_version':4,'temporal_contract_version':4,'source':'fixed 208 train endpoints; no minival evaluator/selection','fresh_init':'fresh V4 seed42 matched FULL then V4 recency query initialized from FULL; no V4/V5/probe warmstart','updates':UPDATES,'lr':LR,'effective_batch':EFFECTIVE,'microbatch':MICRO,'dropout':'disabled','target':'20x native then divide prediction by20','acceptance':'PASS iff r2>=.5 and pred_std>=.5*target_std; .1<=r2<.5 LEARNING_HOLD else FAIL'}
 OUT.mkdir(parents=True);(OUT/'frozen_ids.json').write_text(json.dumps({'ids':{k:v.tolist() for k,v in fixed.items()},'source_authority':auth,'recipe':recipe},indent=2,sort_keys=True)+'\n');d=torch.device('cuda:0');full,query=make_matched_pair();arms={'full_v4':full.to(d),'t_v6_recency':query.to(d)};code={x:sha(Path(__file__).with_name(x)) for x in ('model.py','capacity_probe.py')};code['current_query_v4/core.py']=sha(Path(__file__).parents[1]/'two_mainlines_long_v1/current_query_v4/core.py');opt={k:torch.optim.AdamW(groups(m),lr=LR) for k,m in arms.items()};before={k:score(m,cache,fixed,d) for k,m in arms.items()};initial={k:state_sha(m) for k,m in arms.items()};start=time.time();done=0;names=list(fixed)
 try:
  for step in range(UPDATES):
   if time.time()-start>DEADLINE:break
   n=names[step%len(names)];row=cache['train'][n];x,y,_=batch(row,fixed[n],d);b=H1Bank(*[row['bank'][q].to(d) for q in ('E0','T','unit_mask')])
   for k,m in arms.items():
    m.train();opt[k].zero_grad(set_to_none=True)
    for i in range(0,len(x),MICRO):(F.mse_loss(m.forward_last(x[i:i+MICRO],b),y[i:i+MICRO])*(MICRO/len(x))).backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt[k].step()
   done=step+1
 finally:
  after={k:score(m,cache,fixed,d) for k,m in arms.items()};decision={k:'PASS' if v['r2_concat']>=.5 and v['prediction_std']>=.5*v['target_std'] else 'LEARNING_HOLD' if v['r2_concat']>=.1 else 'FAIL' for k,v in after.items()}
  for k,m in arms.items():torch.save({'model':m.state_dict(),'optimizer':opt[k].state_dict(),'updates':done,'recipe':recipe},OUT/f'{k}_latest.pt')
  report={'schema':recipe['schema'],'status':'COMPLETE' if done==UPDATES else 'TIMEOUT_OR_INTERRUPTED','input_authority':auth,'code_sha256':code,'frozen_ids_sha256':sha(OUT/'frozen_ids.json'),'initial_state_sha256':initial,'trained_state_sha256':{k:state_sha(m) for k,m in arms.items()},'recipe':recipe,'before':before,'after':after,'updates_completed':done,'elapsed_s':time.time()-start,'decision':decision};(OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({'after':after,'decision':decision,'updates':done},sort_keys=True))
if __name__=='__main__':main()
