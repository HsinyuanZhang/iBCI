"""Isolated exact resume of sealed V4 FULL epoch 12 through endpoint 24."""
from __future__ import annotations
import argparse, hashlib, json, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tfpd_exploration.src.h1_optimized_v2.cache import ROOT, build_or_load, validate_authority
from tfpd_exploration.src.h1_optimized_v2.paired_train import groups
from tfpd_exploration.src.h1_optimized_v2.score import evaluate
from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
from tfpd_exploration.src.h1_optimized_v4.model import make_matched_pair
from tfpd_exploration.src.h1_optimized_v4.paired_train import batches, collate, keep

SOURCE=ROOT/'paired_v4_signed_12ep_v1';OUT=ROOT/'v4_full_continue24_deterministic_v1';START,END,LR,MICRO,EMA=12,24,1e-4,8,.9995
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state_sha(m):
 h=hashlib.sha256()
 for n,t in sorted(m.state_dict().items()):h.update(n.encode());h.update(t.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def restore(payload):
 state=payload['rng'];torch.set_rng_state(state['torch'].cpu());np.random.set_state(state['numpy']);random.setstate(state['python']);torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])
def build(payload,device):
 full,_=make_matched_pair();full=full.to(device);full.load_state_dict(payload['model'],strict=True)
 if int(full.frontend_contract_version.item())!=4:raise RuntimeError('V4 frontend contract drift')
 opt=torch.optim.AdamW(groups(full),lr=LR);opt.load_state_dict(payload['optimizer']);ema=DecoderEMA(full,decay=EMA);ema.load_checkpoint_state(payload['ema'])
 # EMA checkpoints are intentionally portable CPU tensors, but resumed
 # update_after_step is an in-place GPU operation just as in the live V4 run.
 ema.shadow={name:value.to(device=device,dtype=torch.float32).detach().clone() for name,value in ema.shadow.items()}
 return full,opt,ema
def equal_tree(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
 return a==b
def one_step(model,opt,ema,cache,device):
 name,starts=batches(cache,START+1)[0];row=cache['train'][name];x,y=collate(row,starts,device);bank=H1Bank(*[row['bank'][q].to(device) for q in ('E0','T','unit_mask')]);mask=keep(len(starts),START+1,0,device);model.train();opt.zero_grad(set_to_none=True)
 for i in range(0,len(starts),MICRO):(F.mse_loss(model.forward_last(x[i:i+MICRO],bank,mask[i:i+MICRO]),y[i:i+MICRO])*(len(x[i:i+MICRO])/len(x))).backward()
 torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();ema.update_after_step(model);return {'session':name,'starts_sha256':hashlib.sha256(np.asarray(starts,dtype=np.int64).tobytes()).hexdigest(),'keep_sha256':hashlib.sha256(mask.detach().cpu().numpy().tobytes()).hexdigest()}
def preflight(cache,device):
 expected_step=START*len(batches(cache,1)); path=SOURCE/'full_epoch_012.pt'
 # Each branch independently deserializes; no CPU optimizer-moment storage is
 # shared between the two candidate resume tests.
 first=torch.load(path,map_location='cpu',weights_only=False);second=torch.load(path,map_location='cpu',weights_only=False)
 if first['epoch']!=START or second['epoch']!=START or first['step']!=expected_step or second['step']!=expected_step:raise RuntimeError('source checkpoint epoch/step does not match exact e12 resume boundary')
 a,ao,ae=build(first,device);b,bo,be=build(second,device)
 if state_sha(a)!=state_sha(b) or not equal_tree(ao.state_dict(),bo.state_dict()) or not equal_tree(ae.checkpoint_state(),be.checkpoint_state()):raise RuntimeError('independent loads diverge before the tested update')
 pre={'model_state_sha256':state_sha(a),'optimizer_state_exact_equal':True,'ema_state_exact_equal':True}
 restore(first);one=one_step(a,ao,ae,cache,device);restore(second);two=one_step(b,bo,be,cache,device)
 if one!=two or state_sha(a)!=state_sha(b) or not equal_tree(ao.state_dict(),bo.state_dict()) or not equal_tree(ae.checkpoint_state(),be.checkpoint_state()):raise RuntimeError('repeatable exact resume one-update test failed')
 return {'status':'PASS','claim':'repeatable exact restore plus one identical e13 update from two independent checkpoint deserializations; not an unobserved uninterrupted-epoch claim','source_checkpoint_sha256':sha(path),'source_checkpoint_epoch':START,'source_checkpoint_step':expected_step,'same_epoch13_batch':one,'preupdate':pre,'post_update_model_state_sha256':state_sha(a),'post_update_optimizer_state_exact_equal':True,'post_update_ema_state_exact_equal':True}
def main():
 p=argparse.ArgumentParser();p.add_argument('--preflight-only',action='store_true');args=p.parse_args()
 if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':raise RuntimeError('deterministic continuation requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA use')
 torch.use_deterministic_algorithms(True);cache=build_or_load();auth=json.loads((ROOT/'source_cache_authority.json').read_text());validate_authority(cache,auth);device=torch.device('cuda:0')
 if args.preflight_only:print(json.dumps(preflight(cache,device),sort_keys=True));return
 if OUT.exists():raise FileExistsError(OUT)
 v4final=json.loads((SOURCE/'final.json').read_text());protocol={'schema':'v4_full_continue24_deterministic_protocol_freeze_v1','frozen_before_execution':True,'source':'sealed V4 FULL epoch12 checkpoint only','primary':'endpoint epoch24 EMA complete stream 20325 bins','secondary_selection':{'candidate_epochs':list(range(13,25)),'surface':'frozen minival 2908 endpoints','governing':'EMA r2_concat','tie_break':'earliest'},'recipe_equivalence':'architecture, data, sampler, drop masks, AdamW groups, LR/warmup history, EMA, model state, optimizer, and RNG are preserved from V4','numerical_runtime_disclosure':'CUBLAS_WORKSPACE_CONFIG=:4096:8 plus torch.use_deterministic_algorithms(True) is intentionally added. This differs from the recorded V4 launch runtime; the run does not claim bit-identical continuation of an unobserved original-runtime GPU trajectory.','original_runtime_preflight_failure':{'status':'NOT_BIT_EXACT','first_parameter_difference':'frontend.phi.weight','max_abs_difference':3.725290298461914e-09,'interpretation':'GPU numerical reduction nondeterminism under original runtime, not a loader/data mismatch'}}
 OUT.mkdir(parents=True);(OUT/'protocol_freeze.json').write_text(json.dumps(protocol,indent=2,sort_keys=True)+'\n');receipt=preflight(cache,device);code={'continue24.py':sha(Path(__file__)),'v4_model.py':sha(Path(__file__).parents[1]/'h1_optimized_v4/model.py'),'v4_paired_train.py':sha(Path(__file__).parents[1]/'h1_optimized_v4/paired_train.py'),'ema.py':sha(Path(__file__).parents[1]/'h1_temporal_decoder_quick_product_v1/ema.py')};(OUT/'input_authority.json').write_text(json.dumps({'source_cache_authority':auth,'source_cache_authority_sha256':sha(ROOT/'source_cache_authority.json'),'source_checkpoint_sha256':receipt['source_checkpoint_sha256'],'source_final_sha256':sha(SOURCE/'final.json'),'source_input_authority_sha256':sha(SOURCE/'input_authority.json'),'code_sha256':code,'protocol':protocol,'resume_equivalence':receipt},indent=2,sort_keys=True)+'\n')
 payload=torch.load(SOURCE/'full_epoch_012.pt',map_location=device,weights_only=False);model,opt,ema=build(payload,device);restore(payload);step=int(payload['step']);updates=len(batches(cache,1));report={'schema':'v4_full_continue24_v1','status':'RUNNING','protocol':protocol,'resume_equivalence':receipt,'epochs':[]};started=time.time()
 for epoch in range(13,25):
  for bi,(name,starts) in enumerate(batches(cache,epoch)):
   row=cache['train'][name];x,y=collate(row,starts,device);bank=H1Bank(*[row['bank'][q].to(device) for q in ('E0','T','unit_mask')]);mask=keep(len(starts),epoch,bi,device);model.train();opt.zero_grad(set_to_none=True);step+=1
   for group in opt.param_groups:group['lr']=LR*min(1.,step/updates)
   for i in range(0,len(starts),MICRO):(F.mse_loss(model.forward_last(x[i:i+MICRO],bank,mask[i:i+MICRO]),y[i:i+MICRO])*(len(x[i:i+MICRO])/len(x))).backward()
   torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();ema.update_after_step(model)
  state={'model':model.state_dict(),'optimizer':opt.state_dict(),'ema':ema.checkpoint_state(),'epoch':epoch,'step':step,'rng':{'torch':torch.get_rng_state(),'numpy':np.random.get_state(),'python':random.getstate(),'cuda':torch.cuda.get_rng_state_all()},'frontend_contract_version':4,'recipe':'v4_full_continue24_v1'};torch.save(state,OUT/f'full_epoch_{epoch:03d}.pt');torch.save(state,OUT/'full_latest.pt');raw=evaluate(model,device=device,mode='selection');score=ema.score_with_ema(model,lambda z:evaluate(z,device=device,mode='selection'));report['epochs'].append({'epoch':epoch,'elapsed_s':time.time()-started,'raw_selection_diagnostic':raw,'ema_selection_governing':score});(OUT/'progress.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 winner=max(report['epochs'],key=lambda x:x['ema_selection_governing']['r2_concat']);freeze={'schema':'v4_full_continue24_selection_freeze_v1','protocol_freeze_sha256':sha(OUT/'protocol_freeze.json'),'selected':{'epoch':winner['epoch'],'ema_r2_concat':winner['ema_selection_governing']['r2_concat'],'checkpoint':str(OUT/f"full_epoch_{winner['epoch']:03d}.pt"),'tie_break':'earliest'}};(OUT/'selection_freeze.json').write_text(json.dumps(freeze,indent=2,sort_keys=True)+'\n');report['selection_freeze']=freeze
 for label,path in (('complete_selected',Path(freeze['selected']['checkpoint'])),('complete_endpoint24',OUT/'full_epoch_024.pt')):
  candidate=torch.load(path,map_location=device,weights_only=False);model.load_state_dict(candidate['model'],strict=True);endema=DecoderEMA(model,decay=EMA);endema.load_checkpoint_state(candidate['ema']);report[label]=endema.score_with_ema(model,lambda z:evaluate(z,device=device,mode='complete'))
 report['status']='COMPLETE';report['trained_state_sha256']=state_sha(model);(OUT/'final.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
