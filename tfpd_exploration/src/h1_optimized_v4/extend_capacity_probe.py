"""One predeclared V4 continuation: 260 -> cumulative 1040 source updates."""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from tfpd_exploration.src.h1_optimized_v2.cache import ROOT,build_or_load,validate_authority
from tfpd_exploration.src.h1_optimized_v2.capacity_probe import batch,r2
from tfpd_exploration.src.h1_optimized_v2.paired_train import groups
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
from .model import make_matched_pair
PARENT=ROOT/'capacity_probe_208_source_v4';OUT=ROOT/'capacity_probe_208_source_v4_extend1040';LR=2e-4;START=260;TOTAL=1040;MICRO=4;EFFECTIVE=16;DEADLINE=600
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state_sha(m):
 h=hashlib.sha256()
 for n,t in sorted(m.state_dict().items()):h.update(n.encode());h.update(t.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def score(m,cache,ids,dev):
 p=[];y=[];m.eval()
 with torch.no_grad():
  for n,s in ids.items():
   row=cache['train'][n];x,_,yy=batch(row,np.asarray(s),dev);b=H1Bank(*[row['bank'][q].to(dev) for q in ('E0','T','unit_mask')])
   p.extend([(m.forward_last(x[i:i+MICRO],b)/20).cpu().numpy() for i in range(0,len(x),MICRO)]);y.extend([yy[i:i+MICRO] for i in range(0,len(yy),MICRO)])
 p,y=np.concatenate(p),np.concatenate(y);return {'r2_concat':r2(p,y),'prediction_std':float(p.std()),'target_std':float(y.std())}
def main():
 if OUT.exists():raise FileExistsError(OUT)
 frozen=json.loads((PARENT/'frozen_ids.json').read_text());cache=build_or_load();validate_authority(cache,frozen['source_authority']);ids=frozen['ids']
 recipe={'parent':'capacity_probe_208_source_v4','parent_updates':START,'absolute_updates':TOTAL,'extension_updates':TOTAL-START,'lr':LR,'effective_batch':EFFECTIVE,'microbatch':MICRO,'dropout':'disabled','sampler':'continue deterministic cyclical session order names[global_step % 13], beginning global_step=260','source_ids':'exact parent frozen IDs','selection':'none','acceptance':'PASS iff r2>=.5 and prediction_std>=.5*target_std; .1<=r2<.5 LEARNING_HOLD; else FAIL'}
 OUT.mkdir(parents=True);code={'model.py':sha(Path(__file__).with_name('model.py')),'extend_capacity_probe.py':sha(Path(__file__))};freeze={'schema':'h1_v4_capacity_extension_freeze_v1','status':'FROZEN_BEFORE_RUN','parent_report_sha256':sha(PARENT/'report.json'),'parent_checkpoint_sha256':{k:sha(PARENT/f'{k}_latest.pt') for k in ('full','t')},'code_sha256':code,'recipe':recipe,'parent_frozen_ids_sha256':sha(PARENT/'frozen_ids.json')};(OUT/'extension_freeze.json').write_text(json.dumps(freeze,indent=2,sort_keys=True)+'\n')
 dev=torch.device('cuda:0');full,t=make_matched_pair();arms={'full':full.to(dev),'t':t.to(dev)};opt={k:torch.optim.AdamW(groups(m),lr=LR) for k,m in arms.items()}
 for k,m in arms.items():
  payload=torch.load(PARENT/f'{k}_latest.pt',map_location=dev,weights_only=True)
  if payload['updates']!=START:raise RuntimeError('parent step drift')
  m.load_state_dict(payload['model'],strict=True);opt[k].load_state_dict(payload['optimizer'])
 before={k:score(m,cache,ids,dev) for k,m in arms.items()};start=time.time();done=START;names=list(ids)
 try:
  for global_step in range(START,TOTAL):
   if time.time()-start>DEADLINE:break
   n=names[global_step%len(names)];row=cache['train'][n];x,y,_=batch(row,np.asarray(ids[n]),dev);b=H1Bank(*[row['bank'][q].to(dev) for q in ('E0','T','unit_mask')])
   for k,m in arms.items():
    m.train();opt[k].zero_grad(set_to_none=True)
    for i in range(0,len(x),MICRO):(F.mse_loss(m.forward_last(x[i:i+MICRO],b),y[i:i+MICRO])*(MICRO/len(x))).backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt[k].step()
   done=global_step+1
 finally:
  after={k:score(m,cache,ids,dev) for k,m in arms.items()};decision={k:('PASS' if after[k]['r2_concat']>=.5 and after[k]['prediction_std']>=.5*after[k]['target_std'] else ('LEARNING_HOLD' if after[k]['r2_concat']>=.1 else 'FAIL')) for k in arms}
  for k,m in arms.items():torch.save({'model':m.state_dict(),'optimizer':opt[k].state_dict(),'updates':done,'parent_checkpoint_sha256':freeze['parent_checkpoint_sha256'][k],'recipe':recipe},OUT/f'{k}_latest.pt')
  report={'schema':'h1_capacity_probe_208_source_v4_extend1040','status':'COMPLETE' if done==TOTAL else 'TIMEOUT_OR_INTERRUPTED','extension_freeze_sha256':sha(OUT/'extension_freeze.json'),'parent':freeze,'before':before,'after':after,'absolute_updates_completed':done,'extension_updates_completed':done-START,'elapsed_s':time.time()-start,'trained_state_sha256':{k:state_sha(m) for k,m in arms.items()},'decision':decision};(OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({'after':after,'decision':decision,'updates':done},sort_keys=True))
if __name__=='__main__':main()
