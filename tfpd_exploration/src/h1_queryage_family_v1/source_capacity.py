"""Prospective, source208-only paired QueryAge capacity gate; never auto-launches."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,random,tempfile,time
from pathlib import Path
import numpy as np
ARMS=('flat','route'); W,UNITS,OUT,MICRO,EFFECTIVE=700,176,7,4,16
IDS_SHA='da4bf975a5c1023bbffe26da87d0d4977f437d7db1db012283511ef140d96211'
CONFIG={'schema':'h1_queryage_source_capacity_v1','source':'exact fixed source208/13x16/W700/train only','seed':42,'optimizer':'fresh AdamW trusted groups lr=2e-4 wd=.01','microbatch':4,'effective_batch':16,'loss':'native velocity x20, four equal .25 micro means','dropout':False,'prefix':False,'ema':False,'modes':{'smoke20':20,'capacity260':260,'extend1040':1040},'limits':{'smoke20':900,'capacity260':3600,'extend1040':3600,'memory':22<<30}}
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dig(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def atomic(x,p):
 p.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(dir=p.parent,mode='w',delete=False) as h:t=Path(h.name);json.dump(x,h,sort_keys=True,indent=2,allow_nan=False);h.flush();os.fsync(h.fileno())
 os.replace(t,p)
def atomic_torch(value,path,torch):
 path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.pt',delete=False) as h:t=Path(h.name)
 try:
  torch.save(value,t);os.replace(t,path)
 finally:t.unlink(missing_ok=True)
def bindings(output,*,mode=None,physical_gpu=None,smoke_receipt=None,resume_checkpoint=None,capacity_receipt=None):
 """Pure pre-import closure for root authorization."""
 ids=ROOT/'results/decoder_validation_v2/20260905_190000/h1/capacity_probe_208_source_v2/frozen_ids.json'
 cache=ROOT/'results/decoder_validation_v2/20260905_190000/h1/source_cache.pt';authority=cache.with_name('source_cache_authority.json')
 # This is the deliberately explicit executable
 # closure.  Inputs live separately so a stage receipt does not obscure a
 # code-closure change with a mechanically necessary predecessor hash.
 paths=[Path(__file__),Path(__file__).with_name('__init__.py'),Path(__file__).with_name('model.py'),ROOT/'src/h1_family_v1/model.py',ROOT/'src/h1_optimized_v2/cache.py',ROOT/'src/h1_optimized_v2/data.py',ROOT/'src/h1_optimized_v2/paired_train.py',ROOT/'src/h1_optimized_v2/model.py',ROOT/'src/h1_optimized_v2/score.py',ROOT/'src/h1_temporal_decoder_quick_product_v1/ema.py',ROOT/'src/family_runtime_v1/diagnose_h1_selected_source208.py',ROOT/'src/family_runtime_v1/diagnose_h1_endpoint_raw.py',ROOT/'src/two_mainlines_long_v1/current_query_v2/__init__.py',ROOT/'src/two_mainlines_long_v1/current_query_v2/core.py',ROOT/'src/two_mainlines_long_v1/decoder/h1_temporal.py',ROOT/'src/two_mainlines_long_v1/decoder/h1_config.py']
 files={str(p):sha(p) for p in paths}
 capacity_ids=ROOT/'src/h1_optimized_v2/capacity_probe.py'
 files[str(capacity_ids)]=sha(capacity_ids)
 if len(files)!=17:raise RuntimeError('exact 17-file executable closure drift')
 inputs={str(p):sha(p) for p in (ROOT/'docs/PROTOCOL_H1_QUERYAGE_SOURCE_CAPACITY_V1_20260906.md',ids,cache,authority)}
 if inputs[str(ids)]!=IDS_SHA:raise RuntimeError('fixed IDs SHA drift')
 base_files={**files,**inputs}
 if resume_checkpoint is not None:inputs[str(resume_checkpoint.resolve())]=sha(resume_checkpoint)
 if capacity_receipt is not None:inputs[str(capacity_receipt.resolve())]=sha(capacity_receipt)
 if smoke_receipt is not None:inputs[str(smoke_receipt.resolve())]=sha(smoke_receipt)
 return {'protocol':CONFIG,'protocol_sha256':dig(CONFIG),'base_files':base_files,'files':files,'inputs':inputs,'output':str(output),'mode':mode,'physical_gpu':physical_gpu,'threads':1,'minival_rows_used':False}
def preflight(output,auth,auth_sha,mode,*,smoke_receipt=None,resume_checkpoint=None,capacity_receipt=None,allow_existing=False):
 if any(not Path(p).is_absolute() for p in (output,auth,*[p for p in (smoke_receipt,resume_checkpoint,capacity_receipt) if p is not None])):raise RuntimeError('absolute paths required')
 if any(Path(p).resolve()!=Path(p) for p in (output,auth,*[p for p in (smoke_receipt,resume_checkpoint,capacity_receipt) if p is not None])):raise RuntimeError('canonical absolute paths required')
 if output.exists() and not allow_existing:raise FileExistsError(output)
 physical=os.environ.get('CUDA_VISIBLE_DEVICES')
 if mode not in CONFIG['modes'] or os.environ.get('H1_QUERYAGE_CAPACITY_GO')!='1' or physical not in ('0','1') or os.environ.get('OMP_NUM_THREADS') not in ('1',None) or sha(auth)!=auth_sha:raise RuntimeError('explicit mode GO physical-GPU authorization gate')
 if mode in ('capacity260','extend1040') and smoke_receipt is None:raise RuntimeError('capacity requires bound smoke receipt')
 if mode=='extend1040' and (resume_checkpoint is None or capacity_receipt is None):raise RuntimeError('extend requires bound capacity receipt/checkpoint')
 if mode=='smoke20' and any(p is not None for p in (smoke_receipt,resume_checkpoint,capacity_receipt)):raise RuntimeError('fresh smoke cannot take predecessor artifacts')
 if mode=='capacity260' and any(p is not None for p in (resume_checkpoint,capacity_receipt)):raise RuntimeError('capacity260 must start fresh')
 if any(Path(p).parent==output or Path(p).parent in output.parents for p in (smoke_receipt,capacity_receipt) if p is not None):raise RuntimeError('output must be separate from predecessor artifacts')
 b=bindings(output,mode=mode,physical_gpu=int(physical),smoke_receipt=smoke_receipt,resume_checkpoint=resume_checkpoint,capacity_receipt=capacity_receipt)
 body=json.loads(Path(auth).read_text())
 if body.get('status')!='ROOT_REVIEW_GO' or body.get('bindings')!=b:raise RuntimeError('external authorization drift')
 return b
def sd(s):
 h=hashlib.sha256()
 for n,t in sorted(s.items()):a=t.detach().cpu().contiguous().numpy();h.update(n.encode());h.update(str(a.dtype).encode());h.update(np.asarray(a.shape,dtype=np.int64).tobytes());h.update(a.tobytes())
 return h.hexdigest()
def array_sha(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+np.asarray(a.shape,dtype=np.int64).tobytes()+a.tobytes()).hexdigest()
def manifest(root):
 """Fresh owned-output manifest; only files produced below this new output."""
 return {str(p):sha(p) for p in sorted(Path(root).rglob('*')) if p.is_file() and p.name!='receipt.json'}
def same(a,b):
 """Strict recursive disk-payload equality including tensors/RNG containers."""
 if type(a)!=type(b):return False
 if hasattr(a,'detach'):
  return str(a.dtype)==str(b.dtype) and tuple(a.shape)==tuple(b.shape) and bool((a.detach().cpu()==b.detach().cpu()).all())
 if isinstance(a,np.ndarray):return a.dtype==b.dtype and a.shape==b.shape and bool(np.array_equal(a,b))
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b
def _same_closure(left,right):
 if any(left.get(key)!=right.get(key) for key in ('protocol','protocol_sha256')):return False
 # A later stage necessarily adds its predecessor receipt/checkpoint hashes.
 # Every source/input already bound by the predecessor must still be present
 # byte-for-byte in the fresh current closure.
 if not isinstance(left.get('base_files'),dict) or left.get('base_files')!=right.get('base_files'):return False
 for key in ('files','inputs'):
  old,new=left.get(key),right.get(key)
  if not (isinstance(old,dict) and isinstance(new,dict) and all(new.get(path)==digest for path,digest in old.items())):return False
 return True
def _owned_receipt(body,path):
 recorded=body.get('owned_artifact_sha256')
 if not isinstance(recorded,dict) or not recorded or recorded!=manifest(Path(path).parent):raise RuntimeError('predecessor owned artifact mutation or missing manifest')
 if body.get('pre',{}).get('output')!=str(Path(path).parent):raise RuntimeError('predecessor output identity drift')
def smoke_forecasts(times):
 steady=np.asarray(times[4:],float)
 if len(times)!=20 or steady.shape!=(16,) or not np.isfinite(steady).all() or np.any(steady<=0):raise RuntimeError('exact20 smoke timing contract')
 return {'capacity260':float(steady.mean())*260*1.5+300,'total1040':float(steady.mean())*1040*1.5+600}
def capacity_eligible(mode,scores,losses):
 if mode not in ('capacity260','extend1040'):raise ValueError('capacity mode required')
 if set(scores)!=set(ARMS) or not losses or any(set(row)!=set(ARMS) or not np.isfinite(list(row.values())).all() for row in losses):raise RuntimeError('finite paired capacity losses required')
 passes=[]
 for arm in ARMS:
  pooled=scores[arm]['pooled'];r2,std,target=(float(pooled[k]) for k in ('r2_concat_float64','prediction_std_float64','target_std_float64'))
  if not np.isfinite((r2,std,target)).all() or std<0 or target<=0:raise RuntimeError('finite positive-variance capacity score required')
  if mode=='capacity260':passes.append(r2>=.1 and std>=.25*target and float(np.mean([x[arm] for x in losses[-32:]]))<float(np.mean([x[arm] for x in losses[:32]])))
  else:passes.append(r2>=.5 and std>=.5*target)
 return any(passes) if mode=='capacity260' else all(passes)
def validate_smoke(path,current):
 p=Path(path); body=json.loads(p.read_text())
 if sha(p)!=current['inputs'].get(str(p.resolve())):raise RuntimeError('smoke receipt SHA binding drift')
 if (body.get('mode')!='smoke20' or body.get('status')!='PASS_SMOKE_NO_CAPACITY_CLAIM' or body.get('pre')!=body.get('post') or body.get('updates')!=20 or len(body.get('timing',[]))!=20 or body.get('scores')!={} or body.get('checkpoint') is not None or not _same_closure(body['pre'],current)):raise RuntimeError('smoke contract/source/code drift')
 forecasts=smoke_forecasts(body['timing'])
 if any(value>3600 for value in forecasts.values()):raise RuntimeError('smoke forecast drift')
 if len(body.get('losses',[]))!=20 or any(set(row)!=set(ARMS) or not np.isfinite(list(row.values())).all() for row in body['losses']):raise RuntimeError('finite paired smoke losses required')
 _owned_receipt(body,p)
 return body
def validate_prior(path,checkpoint,current):
 p=Path(path);prior=json.loads(p.read_text())
 if sha(p)!=current['inputs'].get(str(p.resolve())):raise RuntimeError('prior receipt SHA binding drift')
 if (prior.get('mode')!='capacity260' or prior.get('status')!='COMPLETE_SOURCE_CAPACITY_NO_FORMAL' or prior.get('pre')!=prior.get('post') or prior.get('updates')!=260 or len(prior.get('history_losses',[]))!=260 or prior.get('checkpoint',{}).get('path')!=str(Path(checkpoint)) or prior['checkpoint'].get('sha256')!=sha(checkpoint) or not _same_closure(prior['pre'],current)):raise RuntimeError('prior capacity provenance drift')
 _owned_receipt(prior,p)
 if prior.get('losses')!=prior['history_losses']:raise RuntimeError('prior loss history mismatch')
 eligible=capacity_eligible('capacity260',prior['scores'],prior['history_losses'])
 if eligible is not True or prior.get('eligible_for_next_stage') is not True:raise RuntimeError('prior recomputed any-arm capacity gate failed')
 return prior
def checkpoint(torch,models,opts,step,binding):
 return {'schema':'h1_queryage_source_checkpoint_v1','step':step,'protocol_sha256':dig(CONFIG),'binding_sha256':dig(binding),'models':{a:copy.deepcopy(m.state_dict()) for a,m in models.items()},'optimizers':{a:copy.deepcopy(o.state_dict()) for a,o in opts.items()},'rng':{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'numpy':np.random.get_state(),'python':random.getstate()}}
def restore(torch,p,models,opts,step,binding):
 if p.get('schema')!='h1_queryage_source_checkpoint_v1' or p.get('step')!=step or p.get('protocol_sha256')!=dig(CONFIG) or p.get('binding_sha256')!=dig(binding):raise RuntimeError('resume checkpoint contract drift')
 for a in ARMS:models[a].load_state_dict(p['models'][a],strict=True);opts[a].load_state_dict(p['optimizers'][a])
 torch.set_rng_state(p['rng']['torch']);torch.cuda.set_rng_state_all(p['rng']['cuda']);np.random.set_state(p['rng']['numpy']);random.setstate(p['rng']['python'])
def batch(torch,row,starts,dev):
 neural,velocity,mask=np.asarray(row['neural']),np.asarray(row['velocity']),np.asarray(row['eval_mask']);starts=np.asarray(starts)
 if starts.dtype!=np.int64 or starts.shape!=(EFFECTIVE,) or np.any(starts<0) or np.any(starts+W>len(neural)) or neural.dtype!=np.float32 or velocity.dtype!=np.float32 or neural.shape[1:]!=(UNITS,) or velocity.shape!=(len(neural),OUT) or mask.dtype!=np.bool_ or mask.shape!=(len(neural),) or not np.all(mask[starts+W-1]) or not np.isfinite(neural).all() or not np.isfinite(velocity).all():raise RuntimeError('fixed source native batch contract')
 x=np.stack([neural[s:s+W] for s in starts]);y=np.stack([velocity[s+W-1] for s in starts])*20
 if x.shape!=(EFFECTIVE,W,UNITS) or y.shape!=(EFFECTIVE,OUT):raise RuntimeError('fixed source batch geometry')
 return torch.as_tensor(x,device=dev),torch.as_tensor(y,device=dev)
def update(torch,models,opts,x,y,bank,guard,*,require_gate=False):
 losses={}; grads={}
 for arm in ARMS:
  m,o=models[arm],opts[arm];m.train();o.zero_grad(set_to_none=True);total=0.
  for i in range(0,EFFECTIVE,MICRO):
   guard();loss=torch.nn.functional.mse_loss(m.forward_last(x[i:i+MICRO],bank,dropout_keep=None),y[i:i+MICRO]);(loss*.25).backward();total+=float(loss.detach())*.25;guard()
  grads[arm]=float(sum((p.grad.detach().abs().sum() for p in m.parameters() if p.grad is not None)))
  torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);o.step();losses[arm]=total;guard()
 gate=getattr(getattr(models['route'].frontend,'attn',None),'g',None)
 if require_gate and (grads['route']<=0 or gate is None or gate.grad is None or not bool(torch.isfinite(gate.grad).all()) or float(gate.grad.detach().abs().sum())<=0):raise RuntimeError('route actual g gradient absent')
 return losses
def run(output,auth,auth_sha,*,mode,smoke_receipt=None,resume_checkpoint=None,capacity_receipt=None,allow_cpu_for_test=False,source_loader=None):
 started=time.monotonic();b=preflight(output,auth,auth_sha,mode,smoke_receipt=smoke_receipt,resume_checkpoint=resume_checkpoint,capacity_receipt=capacity_receipt)
 if source_loader is not None and not allow_cpu_for_test:raise RuntimeError('source injection is CPU-fixture-only')
 # Receipt/hash/provenance rejection is deliberately before all torch and
 # executable helper imports, so a bad predecessor cannot construct a model.
 if smoke_receipt is not None:validate_smoke(smoke_receipt,b)
 if mode=='extend1040':prior=validate_prior(capacity_receipt,resume_checkpoint,b)
 import torch
 if not allow_cpu_for_test and (not torch.cuda.is_available() or torch.cuda.current_device()!=0):raise RuntimeError('visible cuda:0 required')
 torch.set_num_threads(1)
 # PyTorch permits this process-global limit only before prior parallel work.
 # A caller/test may already have fixed it at the required value.
 try:torch.set_num_interop_threads(1)
 except RuntimeError:
  if torch.get_num_interop_threads()!=1:raise
 torch.manual_seed(42);np.random.seed(42);random.seed(42);limit=CONFIG['limits'][mode]
 if not allow_cpu_for_test:torch.cuda.reset_peak_memory_stats()
 def guard():
  if not allow_cpu_for_test:torch.cuda.synchronize()
  if time.monotonic()-started>limit or (not allow_cpu_for_test and torch.cuda.max_memory_allocated()>CONFIG['limits']['memory']):raise RuntimeError('resource bound')
 guard()
 from tfpd_exploration.src.h1_optimized_v2.cache import CACHE,ROOT as CR,validate_authority
 from tfpd_exploration.src.h1_optimized_v2.paired_train import groups
 from tfpd_exploration.src.h1_queryage_family_v1.model import make_queryage_localbalanced_pair
 from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank,shared_parameter_max_abs_diff
 from tfpd_exploration.src.family_runtime_v1.diagnose_h1_selected_source208 import _require_ids,validate_manifest_against_cache,evaluate_source208
 cache,authority=(source_loader() if source_loader is not None else (torch.load(CACHE,map_location='cpu',weights_only=False),json.loads((CR/'source_cache_authority.json').read_text())));validate_authority(cache,authority);fixed=_require_ids();validate_manifest_against_cache(cache,authority,fixed)
 dev=torch.device('cpu' if allow_cpu_for_test else 'cuda:0');flat,route=make_queryage_localbalanced_pair(seed=42);models={'flat':flat.to(dev),'route':route.to(dev)}
 if shared_parameter_max_abs_diff(models['flat'],models['route'])!=0 or float(models['route'].frontend.attn.g.abs().max())!=0:raise RuntimeError('paired initialization/routing gate drift')
 opts={a:torch.optim.AdamW(groups(m),lr=2e-4,weight_decay=.01) for a,m in models.items()};guard();output.mkdir(parents=True);atomic({'bindings':b,'mode':mode},output/'input_authority.json');input_sha=sha(output/'input_authority.json')
 target=CONFIG['modes'][mode];begin=0;history=[]
 if mode=='extend1040':
  if sha(resume_checkpoint)!=prior['checkpoint']['sha256']:raise RuntimeError('resume checkpoint changed before load')
  restore(torch,torch.load(resume_checkpoint,map_location='cpu',weights_only=False),models,opts,260,prior['pre']);guard();begin=260;history=list(prior['history_losses'])
 times=[];losses=[];ident=hashlib.sha256();names=sorted(fixed);initial_parity=None
 for step in range(begin,target):
  name=names[step%13];starts=fixed[name];row=cache['train'][name];x,y=batch(torch,row,starts,dev);bank=H1Bank(*[row['bank'][k].to(dev) for k in ('E0','T','unit_mask')]);ident.update(name.encode());ident.update(starts.tobytes());ident.update(array_sha(x.detach().cpu().numpy()).encode());ident.update(array_sha(y.detach().cpu().numpy()).encode());ident.update(array_sha(row['bank']['unit_mask'].detach().cpu().numpy()).encode())
  if step==begin and mode!='extend1040':
   with torch.no_grad():
    guard();flat_zero=models['flat'].forward_last(x[:MICRO],bank);guard();route_zero=models['route'].forward_last(x[:MICRO],bank);guard();initial_parity=float((flat_zero-route_zero).abs().max())
   if initial_parity!=0.:raise RuntimeError('actual paired g0 source parity drift')
  guard();t=time.monotonic();losses.append(update(torch,models,opts,x,y,bank,guard,require_gate=(step==begin and mode!='extend1040')));guard();times.append(time.monotonic()-t)
  if step%20==0:atomic({'status':'TRAINING','step':step+1,'elapsed_seconds':time.monotonic()-started},output/'live.json')
 score={};ck=None
 if mode!='smoke20':
  from tfpd_exploration.src.family_runtime_v1.diagnose_h1_endpoint_raw import GuardedModel
  for a,m in models.items():
   before=sd(m.state_dict());proxy=GuardedModel(m,lambda calls,rows:guard());score[a],_=evaluate_source208(proxy,cache,fixed,dev,lambda r,d:H1Bank(*[r['bank'][k].to(d) for k in ('E0','T','unit_mask')]))
   if sd(m.state_dict())!=before:raise RuntimeError('source score mutated RAW state')
  ck=output/'capacity_checkpoint.pt';payload=checkpoint(torch,models,opts,target,b);atomic_torch(payload,ck,torch);loaded=torch.load(ck,map_location='cpu',weights_only=False)
  if not same(payload,loaded):raise RuntimeError('atomic disk checkpoint recursive equality drift')
  restore(torch,loaded,models,opts,target,b);guard()
 forecasts=smoke_forecasts(times) if mode=='smoke20' else None
 forecast=forecasts['capacity260'] if forecasts is not None else None
 if forecasts is not None and any(value>3600 for value in forecasts.values()):raise RuntimeError('smoke forecast gate')
 eligible=None
 if mode!='smoke20':eligible=capacity_eligible(mode,score,history+losses)
 if any(not np.isfinite(list(row.values())).all() for row in history+losses):raise RuntimeError('nonfinite final loss history')
 result={'schema':'h1_queryage_source_capacity_v1','status':'PASS_SMOKE_NO_CAPACITY_CLAIM' if mode=='smoke20' else 'COMPLETE_SOURCE_CAPACITY_NO_FORMAL','pre':b,'post':bindings(output,mode=mode,physical_gpu=int(os.environ.get('CUDA_VISIBLE_DEVICES','0')),smoke_receipt=smoke_receipt,resume_checkpoint=resume_checkpoint,capacity_receipt=capacity_receipt),'mode':mode,'updates':target,'losses':losses,'history_losses':history+losses,'source_order_target_mask_sha256':ident.hexdigest(),'initial_actual_g0_parity_max_abs':initial_parity,'parity_probe_batch':MICRO if initial_parity is not None else None,'scores':score,'eligible_for_next_stage':eligible,'checkpoint':None if ck is None else {'path':str(ck),'sha256':sha(ck)},'timing':times,'forecast_seconds':forecast,'minival_rows_used':False,'elapsed_seconds':time.monotonic()-started,'peak_memory_bytes':torch.cuda.max_memory_allocated() if not allow_cpu_for_test else 0}
 if result['post']!=b:raise RuntimeError('fresh post closure drift')
 result['forecasts_seconds']=forecasts
 result['owned_artifact_sha256']=manifest(output)
 final_auth=json.loads(Path(auth).read_text())
 if sha(auth)!=auth_sha or final_auth.get('status')!='ROOT_REVIEW_GO' or final_auth.get('bindings')!=b:raise RuntimeError('fresh external authorization drift')
 if sha(output/'input_authority.json')!=input_sha:raise RuntimeError('own input authority mutation')
 guard();result['elapsed_seconds']=time.monotonic()-started
 atomic(result,output/'receipt.json')
 if manifest(output)!=result['owned_artifact_sha256']:raise RuntimeError('fresh owned artifact mutation')
 return result
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--authorization',type=Path,required=True);p.add_argument('--authorization-sha256',required=True);p.add_argument('--mode',choices=CONFIG['modes'],required=True);p.add_argument('--smoke-receipt',type=Path);p.add_argument('--resume-checkpoint',type=Path);p.add_argument('--capacity-receipt',type=Path);a=p.parse_args(argv);return run(a.output.resolve(),a.authorization.resolve(),a.authorization_sha256,mode=a.mode,smoke_receipt=a.smoke_receipt.resolve() if a.smoke_receipt else None,resume_checkpoint=a.resume_checkpoint.resolve() if a.resume_checkpoint else None,capacity_receipt=a.capacity_receipt.resolve() if a.capacity_receipt else None)
if __name__=='__main__':main()
