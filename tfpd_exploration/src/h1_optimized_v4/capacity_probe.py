"""Frozen V4 source-only 208-window gate; no minival access or selection."""
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
OUT=ROOT/'capacity_probe_208_source_v4'; LR=2e-4; UPDATES=260; DEADLINE=600; MICRO=4; EFFECTIVE=16
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state_sha(m):
 h=hashlib.sha256()
 for n,t in sorted(m.state_dict().items()):h.update(n.encode());h.update(t.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def score(m,cache,fixed,dev):
 p=[];y=[];m.eval()
 with torch.no_grad():
  for n,s in fixed.items():
   row=cache['train'][n];x,_,yy=batch(row,s,dev);b=H1Bank(*[row['bank'][q].to(dev) for q in ('E0','T','unit_mask')])
   p.extend([(m.forward_last(x[i:i+MICRO],b)/20).cpu().numpy() for i in range(0,len(x),MICRO)]);y.extend([yy[i:i+MICRO] for i in range(0,len(x),MICRO)])
 p,y=np.concatenate(p),np.concatenate(y);return {'r2_concat':r2(p,y),'prediction_std':float(p.std()),'target_std':float(y.std())}
def main():
 if OUT.exists():raise FileExistsError(OUT)
 cache=build_or_load();auth=json.loads((ROOT/'source_cache_authority.json').read_text());validate_authority(cache,auth);fixed=ids(cache)
 OUT.mkdir(parents=True); recipe={'revision':'v4','frontend_contract_version':4,'frontend_formula':'carrier=LN([E0,T]); raw=tanh(phi(carrier)); w=mask*(raw-mask_mean(raw))/sqrt(sum_units(mask*(raw-mask_mean(raw))^2)); mixed=sum_units(x*w)+pop_projection(mask_mean(x)); z=LN(GELU(pointwise(causal_depthwise_conv5(mixed))))','activity_scale':1,'window':700,'units':176,'output':7,'target':'20x native velocity then divide predictions by 20','updates':UPDATES,'lr':LR,'effective_batch':EFFECTIVE,'microbatch':MICRO,'dropout':'disabled','selection':'none'}
 (OUT/'frozen_ids.json').write_text(json.dumps({'ids':{k:v.tolist() for k,v in fixed.items()},'source_authority':auth,'recipe':recipe},indent=2,sort_keys=True)+'\n')
 dev=torch.device('cuda:0');full,t=make_matched_pair();arms={'full':full.to(dev),'t':t.to(dev)};init_sha={k:state_sha(m) for k,m in arms.items()};code_sha={name:sha(Path(__file__).with_name(name)) for name in ('model.py','capacity_probe.py')};opt={k:torch.optim.AdamW(groups(m),lr=LR) for k,m in arms.items()};before={k:score(m,cache,fixed,dev) for k,m in arms.items()};start=time.time();done=0
 names=list(fixed)
 try:
  for step in range(UPDATES):
   if time.time()-start>DEADLINE:break
   n=names[step%len(names)];row=cache['train'][n];x,y,_=batch(row,fixed[n],dev);b=H1Bank(*[row['bank'][q].to(dev) for q in ('E0','T','unit_mask')])
   for k,m in arms.items():
    m.train();opt[k].zero_grad(set_to_none=True)
    for i in range(0,len(x),MICRO):(F.mse_loss(m.forward_last(x[i:i+MICRO],b),y[i:i+MICRO])*(MICRO/len(x))).backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt[k].step()
   done=step+1
 finally:
  after={k:score(m,cache,fixed,dev) for k,m in arms.items()};status='COMPLETE' if done==UPDATES else 'TIMEOUT_OR_INTERRUPTED';accept={k:('PASS' if after[k]['r2_concat']>=.5 and after[k]['prediction_std']>=.5*after[k]['target_std'] else ('LEARNING_HOLD' if after[k]['r2_concat']>=.1 else 'FAIL')) for k in arms}
  for k,m in arms.items():torch.save({'model':m.state_dict(),'optimizer':opt[k].state_dict(),'updates':done,'recipe':recipe},OUT/f'{k}_latest.pt')
  report={'schema':'h1_capacity_probe_208_source_v4','status':status,'input_authority':auth,'code_sha256':code_sha,'frozen_ids_sha256':sha(OUT/'frozen_ids.json'),'initial_state_sha256':init_sha,'trained_state_sha256':{k:state_sha(m) for k,m in arms.items()},'recipe':recipe,'before':before,'after':after,'updates_completed':done,'elapsed_s':time.time()-start,'meaningful_acceptance':{'full':accept['full'],'t':accept['t'],'rule':'PASS iff r2>=0.5 AND prediction_std>=0.5*target_std; 0.1<=r2<0.5 is LEARNING_HOLD; r2<0.1 FAIL'}};(OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report['after'],sort_keys=True))
if __name__=='__main__':main()
