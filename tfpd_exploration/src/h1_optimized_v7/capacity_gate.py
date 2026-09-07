"""Frozen V7 p=.30 source-only capacity gate; never touches minival."""
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

OUT=ROOT/'v7_dropout30_source208_gate_v1';SEED=42;P=0.30;UPDATES=1040;LR=2e-4;MICRO=4;EFFECTIVE=16;DEADLINE=600
REFERENCE_IDS=ROOT/'capacity_probe_208_source_v6_recency1040/frozen_ids.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state_sha(m):
 h=hashlib.sha256()
 for n,t in sorted(m.state_dict().items()):h.update(n.encode());h.update(t.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def keep(n,epoch,batch_index,d,p=P):
 """V4/V6 random stream exactly; V7 changes only its threshold."""
 g=torch.Generator(device='cpu').manual_seed(int.from_bytes(hashlib.sha256(f'{SEED}|keep|{epoch}|{batch_index}'.encode()).digest()[:8],'little'))
 k=torch.rand((n,176),generator=g)>=p;k[k.sum(-1)==0,0]=True;return k.to(d)
def score(m,cache,fixed,d):
 ps=[];ys=[];m.eval()
 with torch.no_grad():
  for n,s in sorted(fixed.items()):
   row=cache['train'][n];x,_,y=batch(row,s,d);b=H1Bank(*[row['bank'][q].to(d) for q in ('E0','T','unit_mask')])
   ps.extend((m.forward_last(x[i:i+MICRO],b)/20).cpu().numpy() for i in range(0,len(x),MICRO));ys.extend(y[i:i+MICRO] for i in range(0,len(y),MICRO))
 p,y=np.concatenate(ps),np.concatenate(ys);return {'r2_concat':r2(p,y),'prediction_std':float(p.std()),'target_std':float(y.std())}
def main():
 if OUT.exists():raise FileExistsError(OUT)
 cache=build_or_load();auth=json.loads((ROOT/'source_cache_authority.json').read_text());validate_authority(cache,auth);fixed=ids(cache)
 if sum(len(x) for x in fixed.values())!=208:raise RuntimeError('V7 frozen source gate cardinality drift')
 reference=json.loads(REFERENCE_IDS.read_text())['ids']
 if set(reference)!=set(fixed) or any(list(map(int,reference[k]))!=list(map(int,fixed[k])) for k in fixed):raise RuntimeError('V7 source IDs drift from frozen V6 gate')
 recipe={'schema':'h1_v7_dropout30_source_gate_v1','variable':'p=.30 deterministic whole-unit dropout only; V6 p=.10 baseline','architecture':'exact V4 FULL and V6 recency-query, contracts frontend4 temporal4','seed':SEED,'updates':UPDATES,'lr':LR,'effective_batch':EFFECTIVE,'microbatch':MICRO,'evaluation_dropout':'off','acceptance':'both arms R2>=.5 AND prediction_std>=.5*target_std','selection':'none; source-only gate'}
 OUT.mkdir(parents=True);(OUT/'frozen_ids.json').write_text(json.dumps({'ids':{k:v.tolist() for k,v in fixed.items()},'source_authority':auth,'recipe':recipe},indent=2,sort_keys=True)+'\n')
 d=torch.device('cuda:0');torch.manual_seed(SEED);np.random.seed(SEED);full,t=make_matched_pair();arms={'full_v4':full.to(d),'t_v6':t.to(d)}
 for key in ('frontend','final_norm','readout'):
  a,b=getattr(arms['full_v4'],key).state_dict(),getattr(arms['t_v6'],key).state_dict()
  if any(not torch.equal(a[n],b[n]) for n in a):raise RuntimeError('matched init drift '+key)
 opt={k:torch.optim.AdamW(groups(m),lr=LR) for k,m in arms.items()};before={k:score(m,cache,fixed,d) for k,m in arms.items()};initial={k:state_sha(m) for k,m in arms.items()};names=sorted(fixed);done=0;start=time.time()
 try:
  for step in range(UPDATES):
   if time.time()-start>DEADLINE:break
   n=names[step%len(names)];row=cache['train'][n];x,y,_=batch(row,fixed[n],d);b=H1Bank(*[row['bank'][q].to(d) for q in ('E0','T','unit_mask')]);k=keep(len(x),1,step,d)
   for name,m in arms.items():
    m.train();opt[name].zero_grad(set_to_none=True)
    for i in range(0,len(x),MICRO):(F.mse_loss(m.forward_last(x[i:i+MICRO],b,k[i:i+MICRO]),y[i:i+MICRO])*(len(x[i:i+MICRO])/len(x))).backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt[name].step()
   done=step+1
 finally:
  after={k:score(m,cache,fixed,d) for k,m in arms.items()};decision={k:'PASS' if x['r2_concat']>=.5 and x['prediction_std']>=.5*x['target_std'] else 'FAIL' for k,x in after.items()}
  for k,m in arms.items():torch.save({'model':m.state_dict(),'optimizer':opt[k].state_dict(),'updates':done,'recipe':recipe},OUT/f'{k}_latest.pt')
  report={'schema':recipe['schema'],'status':'COMPLETE' if done==UPDATES else 'TIMEOUT_OR_INTERRUPTED','recipe':recipe,'input_authority':auth,'frozen_ids_sha256':sha(OUT/'frozen_ids.json'),'reference_v6_frozen_ids':str(REFERENCE_IDS),'reference_v6_frozen_ids_sha256':sha(REFERENCE_IDS),'initial_state_sha256':initial,'trained_state_sha256':{k:state_sha(m) for k,m in arms.items()},'before':before,'after':after,'decision':decision,'updates_completed':done,'elapsed_s':time.time()-start,'formal_eligible':bool(done==UPDATES and all(v=='PASS' for v in decision.values()))};(OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({'after':after,'decision':decision},sort_keys=True))
if __name__=='__main__':main()
