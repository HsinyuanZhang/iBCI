"""Fresh query-only V6 formal; V4 FULL12 is immutable external comparator."""
from __future__ import annotations
import hashlib,json,random,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from tfpd_exploration.src.h1_optimized_v2.cache import ROOT,build_or_load,validate_authority
from tfpd_exploration.src.h1_optimized_v2.paired_train import groups
from tfpd_exploration.src.h1_optimized_v2.score import evaluate
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
from tfpd_exploration.src.h1_optimized_v4.paired_train import batches,collate,keep,state_sha
from .model import make_matched_pair
from .ema import V6EMA
OUT=ROOT/'paired_v6_recency_queryonly_12ep_v1';V4=ROOT/'paired_v4_signed_12ep_v1';EPOCHS=12;LR=1e-4;MICRO=8;EMA=.9995
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 if OUT.exists():raise FileExistsError(OUT)
 cache=build_or_load();auth=json.loads((ROOT/'source_cache_authority.json').read_text());validate_authority(cache,auth);OUT.mkdir(parents=True)
 recipe={'seed':42,'design':'fixed V6 read-time priors only; known-source formal metric not hidden/test','arm':'fresh V6 query only; sealed V4 FULL12 external matched-budget comparator','init':'fresh V4 seed42 FULL then V6 temporal initialized from FULL; no probe warmstart','frontend_contract_version':4,'temporal_contract_version':4,'tau_bins':[2,4,8,16,32,64,128,'global'],'target':'20x native, prediction /20','epochs':12,'lr':LR,'optimizer':'V4 AdamW groups decay .01','schedule':'V4 epoch1 linear warmup then constant','dropout':'exact V4 deterministic whole-unit p=.10 mask generator','effective_batch':32,'microbatch':8,'ema':EMA,'selection':'all 12 predeclared EMA pooled frozen minival 2908, earliest tie','complete':'endpoint12 all 20325 bins'}
 protocol={'schema':'h1_v6_selection_protocol_freeze_v1','frozen_before_training':True,'epochs':list(range(1,13)),'governing':'EMA r2_concat frozen 2908','tie_break':'earliest','complete':'endpoint12 20325'};(OUT/'selection_protocol_freeze.json').write_text(json.dumps(protocol,indent=2,sort_keys=True)+'\n');(OUT/'input_authority.json').write_text(json.dumps({'source_cache_authority':auth,'recipe':recipe,'v4_full12_complete':.5360636632162086,'v4_full12_selection':.6021138713226868,'code_sha256':{x:sha(Path(__file__).with_name(x)) for x in ('model.py','ema.py','paired_train.py')}},indent=2,sort_keys=True)+'\n')
 torch.manual_seed(42);np.random.seed(42);random.seed(42);d=torch.device('cuda:0');full,m=make_matched_pair();full,m=full.to(d),m.to(d)
 for k in ('frontend','final_norm','readout'):
  if any(not torch.equal(a,b) for a,b in zip(getattr(full,k).state_dict().values(),getattr(m,k).state_dict().values())):raise RuntimeError('matched init drift')
 opt=torch.optim.AdamW(groups(m),lr=LR);ema=V6EMA(m,EMA);step=0;updates=len(batches(cache,1));rep={'schema':'h1_v6_recency_queryonly_formal_v1','status':'RUNNING','recipe':recipe,'initial_state_sha256':state_sha(m),'epochs':[]};start=time.time()
 for epoch in range(1,13):
  for bi,(n,s) in enumerate(batches(cache,epoch)):
   row=cache['train'][n];x,y=collate(row,s,d);b=H1Bank(*[row['bank'][q].to(d) for q in ('E0','T','unit_mask')]);mask=keep(len(s),epoch,bi,d);m.train();opt.zero_grad(set_to_none=True);step+=1
   for g in opt.param_groups:g['lr']=LR*min(1.,step/updates)
   for i in range(0,len(s),MICRO):(F.mse_loss(m.forward_last(x[i:i+MICRO],b,mask[i:i+MICRO]),y[i:i+MICRO])*(len(x[i:i+MICRO])/len(x))).backward()
   torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();ema.update_after_step(m)
  p={'model':m.state_dict(),'optimizer':opt.state_dict(),'ema':ema.checkpoint_state(),'epoch':epoch,'step':step,'frontend_contract_version':4,'temporal_contract_version':4};torch.save(p,OUT/f't_v6_epoch_{epoch:03d}.pt');torch.save(p,OUT/'t_v6_latest.pt');raw=evaluate(m,device=d,mode='selection');sc=ema.score_with_ema(m,lambda z:evaluate(z,device=d,mode='selection'));rep['epochs'].append({'epoch':epoch,'elapsed_s':time.time()-start,'raw_selection_diagnostic':raw,'ema_selection_governing':sc});(OUT/'progress.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n')
 win=max(rep['epochs'],key=lambda x:x['ema_selection_governing']['r2_concat']);freeze={'schema':'h1_v6_selection_freeze_v1','protocol_freeze_sha256':sha(OUT/'selection_protocol_freeze.json'),'selected':{'epoch':win['epoch'],'checkpoint':str(OUT/f"t_v6_epoch_{win['epoch']:03d}.pt"),'ema_r2_concat':win['ema_selection_governing']['r2_concat'],'tie_break':'earliest'}};(OUT/'selection_freeze.json').write_text(json.dumps(freeze,indent=2,sort_keys=True)+'\n');rep['selection_freeze']=freeze
 for key,path in (('complete_selected',Path(freeze['selected']['checkpoint'])),('complete_endpoint12',OUT/'t_v6_epoch_012.pt')):
  p=torch.load(path,map_location=d,weights_only=False);m.load_state_dict(p['model'],strict=True);e=V6EMA(m,EMA);e.load_checkpoint_state(p['ema']);rep[key]=e.score_with_ema(m,lambda z:evaluate(z,device=d,mode='complete'))
 rep['status']='COMPLETE';(OUT/'final.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
