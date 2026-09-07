#!/usr/bin/env python3
"""Formal matched M2 joint-FiLM/RIFT B,D runner, including resume and ext4 scan."""
from __future__ import annotations
import argparse, contextlib, hashlib, json, math, random, sys
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'src',ROOT.parent/'btransform_unified_v1'/'src',ROOT.parent):
 if str(p) not in sys.path: sys.path.insert(0,str(p))
from scripts.rift_v1 import m2_concat_train as base
from btransform_unified_v1 import plan as v1_plan
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.model import whole_unit_dropout
from btransform_unified_v1.schedule import warmup_cosine_lr
from tfpd_exploration.src.m2_dual_track_v1 import training as old_training, sampler as old_sampler
from btransform_unified_v2.joint_m2_model import JointM2RiftDecoder, ARM_B, ARM_D
CELL='M2-RIFT-R50-D4-JOINT-FILM-M33-V1'; EPOCHS=24; SEED=42; CACHE=ROOT.parent/'tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache'
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p:Path,x:Any):p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');q.replace(p)
def hashes():
 paths=(Path(__file__),ROOT/'src/btransform_unified_v2/joint_m2_model.py',ROOT/'src/btransform_unified_v2/model.py',ROOT/'src/btransform_unified_v2/temporal.py',ROOT/'src/btransform_unified_v2/config.py',ROOT/'src/btransform_unified_v2/streaming.py',ROOT/'scripts/rift_v1/m2_concat_train.py',ROOT.parent/'tfpd_exploration/src/m2_hold_film_probe_v1/encoder.py',ROOT.parent/'tfpd_exploration/src/m2_dual_track_v1/champion.py',ROOT.parent/'tfpd_exploration/results/m2_hold_film_probe_v1/film_states.pt',ROOT.parent/'tfpd_exploration/results/m2_movement_t4_empty_epoch_pick_v1/selected_head.pt',base.MANIFEST)
 return {str(p):sha(p) for p in paths}
def chash(surface,banks):
 return {s:{name:sha(CACHE/surface/s/name) for name in ('calib_activity.npy','X_store.npy','target_store.npy','T.npy','eligible_starts.npy','e0_u.pt')} for s in banks}
def load_all(device):
 d,b=base._load_surface('source_train',device);md,mb=base._load_surface('source_minival',device);ed,eb=base._load_surface('ext4',device);return d,b,md,mb,ed,eb
def model_for(arm,seed,device,b,mb,eb):
 m=JointM2RiftDecoder(arm,seed=seed).to(device)
 for surf,banks in [('source_train',b),('source_minival',mb),('ext4',eb)]:m.install_session_memory(banks,CACHE/surf)
 return m.to(device)
def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
def restore(x):
 random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch'].cpu())
 if torch.cuda.is_available() and x.get('cuda') is not None:torch.cuda.set_rng_state_all([v.cpu() for v in x['cuda']])
def meta_for(arm,seed,epochs,manifest,b,mb,eb,smoke):return {'schema':'m2_rift_joint_train_v2','status':'SMOKE' if smoke else 'FORMAL','cell':CELL,'arm':arm,'seed':seed,'sampler_seed':SEED,'epochs':epochs,'context_bins':50,'depth':4,'attention_backend':'local','identity_provider':'HoldContrastFiLMEarlyPoolEncoder(100,50,64,side8,rank8,num_post3,t4_plus_contrast)+canonical_p0_empty_head','source_train_only_for_gradients':True,'official_test_used':False,'optimizer':{'name':'AdamW','batch':32,'lr_peak':v1_plan.LR_PEAK,'warmup_updates':3165,'total_updates':75960,'ema_decay':v1_plan.EMA_DECAY,'encoder_precision':'fp32','decoder_cuda_precision':'bf16'},'selection_disclosure':'ext4 is a declared visible development selection surface; it is not hidden-test leakage','manifest_digest':manifest['digest'],'source_hashes':hashes(),'cache_hashes':{'source_train':chash('source_train',b),'source_minival':chash('source_minival',mb),'ext4':chash('ext4',eb)}}
def required(state,meta,epoch=None):
 fixed={'schema':'m2_rift_joint_checkpoint_v2','cell':CELL,'arm':meta['arm'],'seed':meta['seed'],'smoke':False,'epochs':EPOCHS,'context_bins':50,'depth':4,'attention_backend':'local'}
 if epoch is not None:fixed['epoch']=epoch
 if any(state.get(k)!=v for k,v in fixed.items()) or state.get('source_hashes')!=meta['source_hashes'] or state.get('cache_hashes')!=meta['cache_hashes'] or int(state.get('global_step',-1)) != int(state.get('epoch',-1))*3165:raise RuntimeError('checkpoint contract mismatch')
def train(a):
 smoke=a.max_updates_smoke is not None
 if not smoke and a.epochs!=EPOCHS:raise ValueError('formal requires exactly 24 epochs')
 dev=torch.device(a.device);torch.set_num_threads(a.cpu_threads);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
 manifest=old_sampler.load_manifest(base.MANIFEST)
 if manifest['digest']!=base.MANIFEST_DIGEST:raise RuntimeError('frozen manifest drift')
 d,b,md,mb,ed,eb=load_all(dev);updates=old_training.count_updates(d)
 if updates!=3165:raise RuntimeError('update contract drift')
 m=model_for(a.arm,a.seed,dev,b,mb,eb);opt=old_training.build_optimizer(m.named_parameters(),lr=v1_plan.LR_PEAK,weight_decay=v1_plan.WEIGHT_DECAY);ema=DecoderEMA(m,decay=v1_plan.EMA_DECAY);dest=a.dest.resolve();meta=meta_for(a.arm,a.seed,a.epochs,manifest,b,mb,eb,smoke)
 step=0;start=1
 if a.resume:
  if smoke or not (dest/'run_meta.json').is_file() or a.resume.resolve().parent!=dest:raise RuntimeError('resume requires matching formal destination/checkpoint')
  old=json.loads((dest/'run_meta.json').read_text())
  if old!=meta:raise RuntimeError('resume run metadata mismatch')
  st=torch.load(a.resume,map_location=dev,weights_only=False);required(st,meta);m.load_state_dict(st['model']);opt.load_state_dict(st['optimizer']);ema.load_state_dict(st['ema']);restore(st['rng']);step=int(st['global_step']);start=int(st['epoch'])+1
  if start>EPOCHS:raise RuntimeError('cannot resume completed run')
 else:
  if dest.exists() and any(dest.iterdir()):raise FileExistsError('new destination must be empty')
  atomic(dest/'run_meta.json',meta)
 padding=base._surface_padding('source_train',b);last_losses=[]
 for epoch in range(start,a.epochs+1):
  m.train();losses=[]
  for bid,batch in enumerate(old_sampler.iter_manifest_batches(d,manifest,epoch,device=dev,target_space=base.old_plan.TRAINING_TARGET_SPACE)):
   step+=1;lr=warmup_cosine_lr(step,total_steps=EPOCHS*updates,warmup_steps=updates,peak=v1_plan.LR_PEAK,min_factor=v1_plan.LR_MIN_FACTOR)
   for group in opt.param_groups: group['lr']=lr
   g=torch.Generator(device='cpu');g.manual_seed(base.unit_dropout_seed(a.seed,epoch,bid));keep=whole_unit_dropout(batch.unit_mask,p=v1_plan.UNIT_DROPOUT,generator=g);valid=base._batch_valid(batch,surface='source_train',padding=padding,device=dev);opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type=dev.type,dtype=torch.bfloat16,enabled=dev.type=='cuda'):loss=nn.functional.mse_loss(m(batch.X,b[batch.session_id],dropout_keep=keep,input_valid_mask=valid).float(),batch.last_target)
   if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
   loss.backward();nn.utils.clip_grad_norm_(m.parameters(),v1_plan.GRAD_CLIP,error_if_nonfinite=True);opt.step();ema.update_after_step(m);losses.append(float(loss))
   if step==1 or step%100==0:base._heartbeat(dest,{'event':'step','epoch':epoch,'global_step':step,'loss':losses[-1],'lr':lr},status='SMOKE' if smoke else 'TRAINING')
   if smoke and step>=a.max_updates_smoke:break
  base._heartbeat(dest,{'event':'epoch','epoch':epoch,'global_step':step,'train_mse':float(np.mean(losses))},status='SMOKE' if smoke else 'TRAINING')
  last_losses=losses;st={'schema':'m2_rift_joint_checkpoint_v2','cell':CELL,'arm':a.arm,'seed':a.seed,'epoch':epoch,'global_step':step,'smoke':smoke,'epochs':a.epochs,'context_bins':50,'depth':4,'attention_backend':'local','source_hashes':meta['source_hashes'],'cache_hashes':meta['cache_hashes'],'model':m.state_dict(),'optimizer':opt.state_dict(),'ema':ema.state_dict(),'rng':rng()};base._atomic_checkpoint(dest/f'epoch_{epoch:03d}.pt',st)
  if smoke:break
  if not smoke and step != epoch*updates:raise RuntimeError('epoch update count drift')
 if not smoke and step != EPOCHS*updates:raise RuntimeError('formal global step drift')
 receipt={'schema':'m2_rift_joint_smoke_v2' if smoke else 'm2_rift_joint_train_receipt_v2','status':'COMPLETED','cell':CELL,'arm':a.arm,'seed':a.seed,'sampler_seed':SEED,'epochs':a.epochs,'global_step':step,'finite_loss':math.isfinite(float(np.mean(last_losses))),'encoder_all_grads':all(p.grad is not None for p in m.encoder.parameters()),'source_hashes':meta['source_hashes'],'cache_hashes':meta['cache_hashes']};atomic(dest/('smoke_receipt.json' if smoke else 'train_receipt.json'),receipt);return receipt
def validate_report(r):
 ex=dict(base.old_plan.EXT4_EXPECTED_WINDOWS)
 if r.get('partial') is not False or int(r.get('n_windows',-1))!=sum(ex.values()) or set(r.get('per_session',{}))!=set(ex) or not all(math.isfinite(float(r.get(k,float('nan')))) for k in ('equal_session_mean','pooled_r2')) or any(int(r['per_session'][s].get('window_count',-1))!=n or not math.isfinite(float(r['per_session'][s].get('r2',float('nan')))) for s,n in ex.items()):raise RuntimeError('malformed ext4 report')
def score(a):
 if a.max_updates_smoke is not None:raise RuntimeError('smoke cannot score')
 dest=a.dest.resolve();meta=json.loads((dest/'run_meta.json').read_text());receipt=json.loads((dest/'train_receipt.json').read_text())
 if meta.get('status')!='FORMAL' or meta.get('cell')!=CELL or meta.get('arm')!=a.arm or meta.get('seed')!=a.seed or meta.get('sampler_seed')!=SEED or meta.get('epochs')!=EPOCHS or meta.get('source_hashes')!=hashes() or receipt.get('epochs')!=EPOCHS or receipt.get('arm')!=a.arm or receipt.get('seed')!=a.seed or receipt.get('sampler_seed')!=SEED or receipt.get('cell')!=CELL or receipt.get('source_hashes')!=meta.get('source_hashes') or receipt.get('cache_hashes')!=meta.get('cache_hashes') or receipt.get('global_step')!=EPOCHS*3165 or receipt.get('status')!='COMPLETED':raise RuntimeError('formal train receipt contract mismatch')
 dev=torch.device(a.device);torch.set_num_threads(a.cpu_threads);d,b,md,mb,ed,eb=load_all(dev)
 if meta['cache_hashes']!={'source_train':chash('source_train',b),'source_minival':chash('source_minival',mb),'ext4':chash('ext4',eb)}:raise RuntimeError('cache drift')
 m=model_for(a.arm,a.seed,dev,b,mb,eb);progress={'schema':'m2_rift_joint_ext4_progress_v1','cell':CELL,'arm':a.arm,'seed':a.seed,'source_hashes':meta['source_hashes'],'cache_hashes':meta['cache_hashes'],'completed':{}};pp=dest/'ext4_score_progress.json'
 if pp.exists():progress=json.loads(pp.read_text())
 if progress.get('cell')!=CELL or progress.get('arm')!=a.arm or progress.get('seed')!=a.seed or progress.get('source_hashes')!=meta['source_hashes'] or progress.get('cache_hashes')!=meta['cache_hashes']:raise RuntimeError('score progress mismatch')
 pad=base._surface_padding('ext4',eb)
 for epoch in range(1,EPOCHS+1):
  cp=dest/f'epoch_{epoch:03d}.pt';cs=sha(cp);prior=progress['completed'].get(str(epoch))
  if prior is not None:
   if prior.get('checkpoint_sha256')!=cs:raise RuntimeError('prior checkpoint hash drift')
   validate_report(prior);continue
  st=torch.load(cp,map_location=dev,weights_only=False);required(st,meta,epoch);m.load_state_dict(st['model']);ema=DecoderEMA(m,decay=v1_plan.EMA_DECAY);ema.load_state_dict(st['ema']);r=base._with_ema(m,ema,lambda:base._score(m,ed,eb,pad,'ext4',dev));validate_report(r);base._heartbeat(dest,{'event':'ext4_score','epoch':epoch},status='SCORING');progress['completed'][str(epoch)]={**r,'checkpoint_sha256':cs};atomic(pp,progress)
 vals={int(k):float(v['equal_session_mean']) for k,v in progress['completed'].items()}
 if set(vals)!=set(range(1,EPOCHS+1)):raise RuntimeError('incomplete scan')
 best=min(((-v,e) for e,v in vals.items()))[1];scan={'schema':'m2_rift_joint_ext4_epoch_scan_v1','status':'COMPLETED','cell':CELL,'arm':a.arm,'seed':a.seed,'sampler_seed':SEED,'source_hashes':meta['source_hashes'],'cache_hashes':meta['cache_hashes'],'official_test_used':False,'ema_by_epoch':progress['completed'],'selection':{'rule':'earliest best equal_session_mean','epoch':best,'equal_session_mean':vals[best]},'endpoint24':progress['completed']['24']};atomic(dest/'ext4_epoch_scan.json',scan);atomic(dest/'score_receipt.json',scan);return scan
def main():
 p=argparse.ArgumentParser();p.add_argument('--dest',type=Path,required=True);p.add_argument('--seed',type=int,choices=(42,43,44),default=42);p.add_argument('--arm',choices=(ARM_B,ARM_D),required=True);p.add_argument('--stage',choices=('train','score','all'),default='train');p.add_argument('--device',default='cuda:0');p.add_argument('--epochs',type=int,default=EPOCHS);p.add_argument('--resume',type=Path);p.add_argument('--max-updates-smoke',type=int);p.add_argument('--cpu-threads',type=int,default=2);a=p.parse_args();
 if not 1<=a.epochs<=EPOCHS or a.max_updates_smoke is not None and a.max_updates_smoke<1:p.error('invalid epochs/smoke update count')
 if a.stage in ('score','all') and a.max_updates_smoke is not None:p.error('smoke cannot score')
 if a.stage=='score' and a.resume is not None:p.error('score cannot resume')
 result=train(a) if a.stage in ('train','all') else score(a)
 if a.stage=='all':result=score(a)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
