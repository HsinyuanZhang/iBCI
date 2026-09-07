"""V7 formal: matched V4 FULL / exact-V6 query, with p=.30 dropout only."""
from __future__ import annotations
import hashlib,json,random,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from tfpd_exploration.src.h1_optimized_v2.cache import ROOT,build_or_load,validate_authority
from tfpd_exploration.src.h1_optimized_v2.paired_train import groups
from tfpd_exploration.src.h1_optimized_v2.score import evaluate
from tfpd_exploration.src.h1_optimized_v4.paired_train import batches,collate,state_sha
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
from .ema import V7EMA
from .model import make_matched_pair

OUT=ROOT/'v7_dropout30_paired12_v1';SEED=42;P=.30;EPOCHS=12;LR=1e-4;MICRO=8;EMA=.9995
GATE_REPORT=ROOT/'v7_dropout30_source208_gate_v1/report.json';AUTHORIZATION=ROOT/'v7_dropout30_formal_authorization.json';STATIC_PROTOCOL=Path(__file__).with_name('protocol_freeze.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def keep(n,e,b,d,p=P):
 g=torch.Generator(device='cpu').manual_seed(int.from_bytes(hashlib.sha256(f'{SEED}|keep|{e}|{b}'.encode()).digest()[:8],'little'))
 k=torch.rand((n,176),generator=g)>=p;k[k.sum(-1)==0,0]=True;return k.to(d)
def ckpt(m,o,eobj,e,step,arm):
 p={'model':m.state_dict(),'optimizer':o.state_dict(),'ema':eobj.checkpoint_state(),'epoch':e,'step':step,'rng':{'torch':torch.get_rng_state(),'numpy':np.random.get_state(),'python':random.getstate(),'cuda':torch.cuda.get_rng_state_all()},'frontend_contract_version':4,'recipe':'v7_dropout30_paired12_v1'}
 if arm=='t_v6':p.update({'operator_contract':'v6_recency_query_temporal_contract4','temporal_contract_version':4})
 else:p.update({'operator_contract':'v4_causal_full_window_no_query_temporal_contract'})
 return p
def preflight():
 if OUT.exists():raise FileExistsError(OUT)
 if not GATE_REPORT.is_file() or not AUTHORIZATION.is_file():raise RuntimeError('V7 formal requires completed gate and root authorization')
 gate=json.loads(GATE_REPORT.read_text());auth=json.loads(AUTHORIZATION.read_text());protocol_sha=sha(STATIC_PROTOCOL)
 if gate.get('status')!='COMPLETE' or gate.get('updates_completed')!=1040 or set(gate.get('decision',{}))!={'full_v4','t_v6'} or any(x!='PASS' for x in gate['decision'].values()):raise RuntimeError('V7 gate is not a completed two-arm PASS')
 if auth.get('gate_report_sha256')!=sha(GATE_REPORT) or auth.get('protocol_sha256')!=protocol_sha:raise RuntimeError('root authorization does not bind gate/protocol')
 return gate,auth,protocol_sha
def main():
 gate,authorization,static_protocol_sha=preflight()
 cache=build_or_load();auth=json.loads((ROOT/'source_cache_authority.json').read_text());validate_authority(cache,auth);OUT.mkdir(parents=True)
 recipe={'schema':'h1_v7_dropout30_paired_formal_v1','variable':'whole-unit dropout p=.30 only, replacing V6 p=.10','architecture':'exact V4 FULL + exact V6 query recency temporal; frontend4 temporal4 unchanged','seed':SEED,'init':'fresh seed42 matched pair; no probe/formal warmstart','epochs':EPOCHS,'lr':LR,'weight_decay':.01,'optimizer':'AdamW V4 groups','schedule':'linear warmup epoch1 then constant','effective_batch':32,'microbatch':MICRO,'ema':EMA,'dropout':'deterministic per-example whole-unit p=.30, shared masks FULL/T','selection':'all 12 predeclared EMA frozen-minival 2908 endpoint pooled R2; earliest tie','primary_complete':'EMA epoch12 frozen-minival all 20325 eval bins','comparators':{'v6_p10_t_selection':.49624704188123847,'v6_p10_t_complete':.4540323959160403,'c2_complete':.8884989023208618}}
 protocol=json.loads(STATIC_PROTOCOL.read_text());(OUT/'selection_protocol_freeze.json').write_text(json.dumps(protocol,indent=2,sort_keys=True)+'\n');(OUT/'input_authority.json').write_text(json.dumps({'source_cache_authority':auth,'recipe':recipe,'protocol_sha256':static_protocol_sha,'gate_report_sha256':sha(GATE_REPORT),'root_authorization_sha256':sha(AUTHORIZATION),'code_sha256':{**{x:sha(Path(__file__).with_name(x)) for x in ('model.py','ema.py','paired_train.py')},'v6_model.py':sha(Path(__file__).parents[1]/'h1_optimized_v6/model.py'),'v6_ema.py':sha(Path(__file__).parents[1]/'h1_optimized_v6/ema.py'),'v4_model.py':sha(Path(__file__).parents[1]/'h1_optimized_v4/model.py'),'current_query_v4_core.py':sha(Path(__file__).parents[1]/'two_mainlines_long_v1/current_query_v4/core.py')}},indent=2,sort_keys=True)+'\n')
 torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED);d=torch.device('cuda:0');full,t=make_matched_pair();arms={'full_v4':full.to(d),'t_v6':t.to(d)}
 for key in ('frontend','final_norm','readout'):
  a,b=getattr(arms['full_v4'],key).state_dict(),getattr(arms['t_v6'],key).state_dict()
  if any(not torch.equal(a[n],b[n]) for n in a):raise RuntimeError('matched init drift '+key)
 opt={k:torch.optim.AdamW(groups(m),lr=LR) for k,m in arms.items()};emas={'full_v4':DecoderEMA(arms['full_v4'],EMA),'t_v6':V7EMA(arms['t_v6'],EMA)};steps={k:0 for k in arms};updates=len(batches(cache,1));report={'schema':recipe['schema'],'status':'RUNNING','recipe':recipe,'initial_state_sha256':{k:state_sha(m) for k,m in arms.items()},'epochs':[]};start=time.time()
 for epoch in range(1,EPOCHS+1):
  for bi,(session,s) in enumerate(batches(cache,epoch)):
   row=cache['train'][session];x,y=collate(row,s,d);bank=H1Bank(*[row['bank'][q].to(d) for q in ('E0','T','unit_mask')]);mask=keep(len(s),epoch,bi,d)
   for arm,m in arms.items():
    m.train();opt[arm].zero_grad(set_to_none=True);steps[arm]+=1
    for pg in opt[arm].param_groups:pg['lr']=LR*min(1.,steps[arm]/updates)
    for i in range(0,len(s),MICRO):(F.mse_loss(m.forward_last(x[i:i+MICRO],bank,mask[i:i+MICRO]),y[i:i+MICRO])*(len(x[i:i+MICRO])/len(x))).backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt[arm].step();emas[arm].update_after_step(m)
  item={'epoch':epoch,'elapsed_s':time.time()-start,'arms':{}}
  for arm,m in arms.items():
   p=ckpt(m,opt[arm],emas[arm],epoch,steps[arm],arm);torch.save(p,OUT/f'{arm}_epoch_{epoch:03d}.pt');torch.save(p,OUT/f'{arm}_latest.pt');item['arms'][arm]={'raw_selection_diagnostic':evaluate(m,device=d,mode='selection'),'ema_selection_governing':emas[arm].score_with_ema(m,lambda z:evaluate(z,device=d,mode='selection'))}
  report['epochs'].append(item);(OUT/'progress.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 selected={}
 for arm in arms:
  win=max(report['epochs'],key=lambda x:x['arms'][arm]['ema_selection_governing']['r2_concat']);selected[arm]={'epoch':win['epoch'],'ema_r2_concat':win['arms'][arm]['ema_selection_governing']['r2_concat'],'checkpoint':str(OUT/f'{arm}_epoch_{win["epoch"]:03d}.pt'),'tie_break':'earliest'}
 freeze={'schema':'h1_v7_dropout30_selection_freeze_v1','protocol_sha256':sha(OUT/'selection_protocol_freeze.json'),'selected':selected};(OUT/'selection_freeze.json').write_text(json.dumps(freeze,indent=2,sort_keys=True)+'\n');report['selection_freeze']=freeze;report['complete_selected']={};report['complete_epoch12']={}
 for arm,m in arms.items():
  for label,path in [('complete_selected',Path(selected[arm]['checkpoint'])),('complete_epoch12',OUT/f'{arm}_epoch_012.pt')]:
   p=torch.load(path,map_location=d,weights_only=False);m.load_state_dict(p['model'],strict=True);e=DecoderEMA(m,EMA) if arm=='full_v4' else V7EMA(m,EMA);e.load_checkpoint_state(p['ema']);report[label][arm]=e.score_with_ema(m,lambda z:evaluate(z,device=d,mode='complete'))
 report['status']='COMPLETE';report['trained_state_sha256']={k:state_sha(m) for k,m in arms.items()};(OUT/'final.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
