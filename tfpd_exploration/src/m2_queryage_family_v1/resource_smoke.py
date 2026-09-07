"""Disposable, source-train-only 100-update paired M2 QueryAge resource smoke.

This is intentionally not a trainer: it never opens minival, scores, saves a
checkpoint, selects a model, or promotes an artifact.  Imports that can touch
torch/data are delayed until the complete hash/gate preflight succeeds.
"""
from __future__ import annotations
import argparse, hashlib, json, os, tempfile, time
from pathlib import Path
import numpy as np

GO='M2_QUERYAGE_RESOURCE_SMOKE_GO'; UPDATES,WARM,TIMED=100,20,80
W,UNITS,OUTPUTS=50,96,2; LIMIT,MEMORY=900,22<<30
ROOT=Path(__file__).resolve().parents[3]
MANIFEST=ROOT/'tfpd_exploration/results/m2_dual_track_v1/20260905_101500/sampler/shuffled_batch_manifest_24.json'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(value,p):
 p.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(dir=p.parent,mode='w',delete=False) as h:
  t=Path(h.name);json.dump(value,h,sort_keys=True,indent=2,allow_nan=False);h.write('\n');h.flush();os.fsync(h.fileno())
 os.replace(t,p)
def prefix(x,*,seed,epoch,batch_id,probability=.5):
 """Deterministic p.5 left-history deletion; current bin is invariant."""
 import torch
 if x.ndim!=3 or x.shape[1:]!=(W,UNITS): raise RuntimeError('M2 [B,50,96] geometry required')
 generator=torch.Generator(device='cpu');generator.manual_seed(seed+1000003*epoch+9176*batch_id)
 chosen=torch.rand((len(x),),generator=generator)<probability; lengths=torch.full((len(x),),W,dtype=torch.int64)
 lengths[chosen]=torch.randint(1,W,(int(chosen.sum()),),generator=generator)
 out=x.clone()
 for i,n in enumerate(lengths.tolist()): out[i,:W-n]=0
 if not torch.equal(out[:,-1],x[:,-1]): raise RuntimeError('prefix altered current bin')
 return out,lengths
def preflight(out,physical_gpu):
 if out.exists(): raise FileExistsError(out)
 if os.environ.get(GO)!='1' or not out.is_absolute() or physical_gpu not in (0,1) or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(physical_gpu): raise RuntimeError('explicit GO/absolute output/one physical GPU gate required')
 return {'self':sha(Path(__file__)),'manifest':sha(MANIFEST)}
def resource_check(started,torch):
 if time.monotonic()-started>LIMIT or torch.cuda.max_memory_allocated()>MEMORY: raise RuntimeError('resource smoke time/memory limit')
def existing_source_folders(plan):
 root=plan.active_run_root()/'cache'/'source_train'
 folders=[root/session for session in plan.HELDIN_SESSIONS]
 if not root.is_dir() or any(not p.is_dir() for p in folders): raise RuntimeError('existing source cache directories required; no creation permitted')
 return folders
def run(out:Path,*,physical_gpu:int,threads:int=1):
 """Run exactly 100 source batches; callers must explicitly lease a GPU."""
 started=time.monotonic();out=Path(out); bound=preflight(out,physical_gpu)
 if threads!=1: raise RuntimeError('one CPU thread required')
 import torch
 if not torch.cuda.is_available(): raise RuntimeError('visible cuda:0 unavailable')
 from tfpd_exploration.src.m2_b_small_stability_v1 import training
 from tfpd_exploration.src.m2_b_small_stability_v1.ema import DecoderEMA
 from tfpd_exploration.src.m2_dual_track_v1 import data, plan, sampler, training as source_training
 from tfpd_exploration.src.m2_family_v1 import finalize_pair as finalizer
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(42);np.random.seed(42)
 folders=existing_source_folders(plan)
 closure=[Path(__file__),Path(__file__).with_name('__init__.py'),Path(__file__).with_name('model.py'),MANIFEST,finalizer.PAIR_ROOT/'pretrain_manifest.json',Path(training.__file__),ROOT/(DecoderEMA.__module__.replace('.','/')+'.py'),Path(data.__file__),Path(plan.__file__),Path(sampler.__file__),Path(source_training.__file__),Path(finalizer.__file__),ROOT/'tfpd_exploration/src/m2_family_v1/decoder.py',ROOT/'tfpd_exploration/src/m2_family_v1/routing.py',ROOT/'tfpd_exploration/src/m2_family_v1/config.py',ROOT/'tfpd_exploration/src/m2_family_v1/launch.py',ROOT/'tfpd_exploration/src/m2_b_small_stability_v1/config.py',ROOT/'tfpd_exploration/src/m2_b_small_stability_v1/decoder.py',ROOT/'tfpd_exploration/src/m2_dual_track_v1/contracts.py',ROOT/'tfpd_exploration/src/two_mainlines_long_v1/current_query_v2/__init__.py',ROOT/'tfpd_exploration/src/two_mainlines_long_v1/current_query_v2/core.py']
 for folder in folders:
  required=('X_store.npy','target_store.npy','eligible_starts.npy','T.npy','e0_u.pt','mapping.json','provenance.json','extra.json','calib_activity.npy')
  closure.extend(folder/name for name in required)
 if any(not p.is_file() for p in closure): raise RuntimeError('required source/code closure file missing')
 bound['closure']={str(p):sha(p) for p in closure}
 pretrain=json.loads((finalizer.PAIR_ROOT/'pretrain_manifest.json').read_text())
 bound['existing_training_source']=finalizer._verify_pretrain_recipe(pretrain)
 resource_check(started,torch)
 from .model import make_paired_queryage_decoders, shared_parameter_max_abs_diff
 torch.cuda.reset_peak_memory_stats();device=torch.device('cuda:0'); flat,route=make_paired_queryage_decoders(seed=42)
 if shared_parameter_max_abs_diff(flat,route)!=0.: raise RuntimeError('paired shared initialization drift')
 cells={'FLAT':flat.to(device),'ROUTE':route.to(device)}
 resource_check(started,torch)
 # This is a compound QueryAge + p=.5 prefix recipe diagnostic; it cannot
 # attribute any timing/result difference to prefix versus the prior Causal decoder.
 opts={k:training.make_optimizer(v.trainable_parameters().items()) for k,v in cells.items()};emas={k:DecoderEMA(v,decay=.9995) for k,v in cells.items()}
 banks={s:data.load_session_bank('source_train',s,device=device) for s in plan.HELDIN_SESSIONS}
 resource_check(started,torch)
 manifest=sampler.load_manifest(MANIFEST)
 if source_training.count_updates(banks)!=3165: raise RuntimeError('original epoch1 manifest must have 3165 batches')
 warm=[];steady=[];rows=[];mask_sha=hashlib.sha256();prefix_sha=hashlib.sha256();order_sha=hashlib.sha256();target_sha=hashlib.sha256()
 for batch_id,batch in enumerate(source_training.epoch_batches_shuffled(banks,manifest,1,device=device,target_space='decoder_raw')):
  if batch_id>=UPDATES: break
  resource_check(started,torch)
  if (not isinstance(batch.session_id,str) or not batch.window_ids or batch.X.dtype!=torch.float32 or batch.X.shape!=(len(batch.window_ids),W,UNITS) or batch.last_target.dtype!=torch.float32 or batch.last_target.shape!=(len(batch.window_ids),OUTPUTS) or not bool(torch.isfinite(batch.X).all()) or not bool(torch.isfinite(batch.last_target).all())): raise RuntimeError('source batch decoder_raw geometry/finite/identity drift')
  base=batch.unit_mask if batch.unit_mask is not None else batch.bank.unit_mask
  if base.ndim==1: base=base.unsqueeze(0).expand(batch.X.size(0),-1)
  keep=training.unit_dropout_mask(base,seed=42,epoch=1,batch_id=batch_id).to(device); shortened,lengths=prefix(batch.X,seed=42,epoch=1,batch_id=batch_id)
  order_sha.update(str(batch.session_id).encode());order_sha.update(np.asarray(batch.window_ids,dtype=np.int64).tobytes());target_sha.update(batch.last_target.detach().cpu().numpy().tobytes());mask_sha.update(keep.detach().cpu().numpy().tobytes());prefix_sha.update(lengths.numpy().tobytes());torch.cuda.synchronize();t0=time.perf_counter();loss={}
  for key,model in cells.items():
   opt=opts[key];model.train();opt.zero_grad(set_to_none=True);training.apply_cell_lr(opt,'S1-SMALL-COS',batch_id+1);value=shortened
   value_pred=model.forward_last(value,batch.bank,dropout_keep=keep); item=torch.nn.functional.mse_loss(value_pred.float(),batch.last_target.float())
   if not bool(torch.isfinite(item)): raise RuntimeError('nonfinite paired smoke loss')
   item.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();emas[key].update_after_step(model);loss[key]=float(item.detach())
  torch.cuda.synchronize();elapsed=time.perf_counter()-t0;(warm if batch_id<WARM else steady).append(elapsed);rows.append({'batch_id':batch_id,'rows':int(batch.X.size(0)),'cold_rows':int((lengths<W).sum()),'loss':loss})
  if (batch_id+1)%20==0: atomic({'status':'RUNNING','calls':batch_id+1,'elapsed_seconds':time.monotonic()-started},out/'live.json')
 if len(rows)!=UPDATES or any(e.n_updates!=UPDATES for e in emas.values()): raise RuntimeError('100 real paired updates/EMA required')
 mean=float(np.mean(steady));forecast=(mean*24*3165*1.5)+1800
 if forecast>=21600: raise RuntimeError('conservative 24epoch forecast exceeds hard budget')
 resource_check(started,torch)
 post={k:sha(k) for k in bound['closure']}
 post_authority={'self':sha(Path(__file__)),'manifest':sha(MANIFEST),'closure':post,'existing_training_source':finalizer._verify_pretrain_recipe(json.loads((finalizer.PAIR_ROOT/'pretrain_manifest.json').read_text()))}
 if post_authority!=bound: raise RuntimeError('fresh post closure/source mutation')
 resource_check(started,torch)
 result={'schema':'m2_queryage_pair_resource_smoke_v1','status':'PASS_100_SOURCE_TRAIN_PAIRED_UPDATES_NO_SCORE','no_minival_loaded_or_scored':True,'no_checkpoint_selection_or_promotion':True,'compound_recipe_disclosure':'QueryAge plus prefix diagnostic; no causal-prefix attribution claim','target':'unchanged decoder_raw native-times5 final-bin targets','updates':rows,'warm20_seconds':float(sum(warm)),'steady80_mean_two_model_paired_batch_seconds':mean,'max_memory_bytes':torch.cuda.max_memory_allocated(),'manifest_order_session_start_sha256':order_sha.hexdigest(),'decoder_raw_target_sha256':target_sha.hexdigest(),'dropout_sha256':mask_sha.hexdigest(),'prefix_lengths_sha256':prefix_sha.hexdigest(),'forecast_24x3165_pair_updates_seconds_with_50pct_margin_plus_1800_eval_checkpoint':forecast,'forecast_is_estimate_not_actual':True,'authority_pre':bound,'authority_post':{**bound,'closure':post}}
 result['authority_post']=post_authority;result['elapsed_seconds']=time.monotonic()-started
 atomic(result,out/'receipt.json');return result
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--physical-gpu',type=int,choices=(0,1),required=True);p.add_argument('--threads',type=int,choices=(1,),required=True);a=p.parse_args(argv);return run(a.output,physical_gpu=a.physical_gpu,threads=a.threads)
if __name__=='__main__':main()
