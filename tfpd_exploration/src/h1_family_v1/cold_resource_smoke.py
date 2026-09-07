"""Gated 32-update, source-only resource smoke for the prospective H1 2x2 phase.

It is deliberately not a phase launcher: no minival/source208 forward, scoring,
selection, phase checkpoint, authorization, or formal output is written.
"""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np

UPDATES, MICRO, EFFECTIVE, LIMIT_SECONDS, MEMORY_LIMIT = 32, 8, 32, 300, 22 << 30
DIAGNOSTIC=Path('tfpd_exploration/results/family_runtime_v1/h1_selected_source208_diagnostic_v1/receipt.json')
COLD_ANALYSIS=Path('tfpd_exploration/results/family_runtime_v1/h1_selected_cold_segments_v1.json')
EXPECTED_RECEIPTS={DIAGNOSTIC:'f3ac9a999fef786ca435e0ff6eefcf679b82e3b3439a535cd66ae6e65c80e614',COLD_ANALYSIS:'379094dab4623444d402a159b625382bc61c320bf399a29d3e07065cfcd7e460'}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def atomic_json(value,path):
 import tempfile
 path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(dir=path.parent,mode='w',delete=False) as h:
  t=Path(h.name);json.dump(value,h,sort_keys=True,indent=2,allow_nan=False);h.write('\n');h.flush();os.fsync(h.fileno())
 os.replace(t,path)

def preflight(formal: Path, output: Path, *, arm: str, device: str, physical_gpu: int) -> dict:
 if output.exists(): raise FileExistsError(output)
 if arm not in ('flat','route') or device!='cuda:0' or physical_gpu not in (0,1): raise RuntimeError('exact arm/cuda:0/physical GPU required')
 if os.environ.get('H1_COLD_RESOURCE_SMOKE_GO')!='1' or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(physical_gpu): raise RuntimeError('explicit smoke GO and exact visible physical GPU required')
 exported=formal/'exports'/f'{arm}_selected_plain_ema.pt'
 if not exported.is_file(): raise RuntimeError('selected plain EMA missing')
 helper=Path(__file__).parents[1]/'family_runtime_v1/complete_h1_family_source.py'
 files={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('cold_history.py'),Path(__file__).with_name('cold_compare.py'),helper,exported,*EXPECTED_RECEIPTS)}
 if any(files[str(path)]!=expected for path,expected in EXPECTED_RECEIPTS.items()):raise RuntimeError('exact diagnostic/cold-analysis receipt SHA drift')
 # Snapshot own/helper bytes before the authoritative helper imports Torch.
 from tfpd_exploration.src.family_runtime_v1.complete_h1_family_source import artifact_audit,code_source_audit
 artifact=artifact_audit(formal); source=code_source_audit(formal)
 return {'artifact':artifact,'source':source,'files':files,'arm':arm,'physical_gpu':physical_gpu}

def _fresh_ema(model, decay=.9995):
 """Minimal EMA with M2 rule: first successful update copies RAW exactly."""
 class EMA:
  def __init__(self): self.shadow={};self.n_updates=0
  def update_after_step(self,m):
   self.n_updates+=1
   current=m.state_dict()
   if self.n_updates==1:self.shadow={k:v.detach().clone() for k,v in current.items()};return
   for k,v in current.items():self.shadow[k].mul_(decay).add_(v.detach(),alpha=1-decay)
 return EMA()

def state_digest(state):
 h=hashlib.sha256()
 for name,t in sorted(state.items()):
  a=t.detach().cpu().contiguous().numpy();h.update(name.encode());h.update(str(a.dtype).encode());h.update(np.asarray(a.shape,dtype=np.int64).tobytes());h.update(a.tobytes())
 return h.hexdigest()

def _run_update(torch, cells, opts, emas, batch, *, epoch, batch_id):
 from .cold_history import apply_cold_history
 x,target,bank,keep=batch
 if not 1<=len(x)<=EFFECTIVE or target.shape!=(len(x),7) or keep.shape!=(len(x),176):raise RuntimeError('effective batch geometry drift')
 full,_=apply_cold_history(x,seed=42,epoch=epoch,batch_id=batch_id,probability=0.)
 prefix,lengths=apply_cold_history(x,seed=42,epoch=epoch,batch_id=batch_id,probability=.5)
 if not torch.equal(prefix[:,-1],x[:,-1]): raise RuntimeError('cold transform changed current bin')
 losses={}
 for name,value in (('CONTROL',full),('PREFIX',prefix)):
  model,opt=cells[name],opts[name];model.train();opt.zero_grad(set_to_none=True)
  weighted=[]
  for offset in range(0,len(value),MICRO):
   loss=torch.nn.functional.mse_loss(model.forward_last(value[offset:offset+MICRO],bank,dropout_keep=keep[offset:offset+MICRO]).float(),target[offset:offset+MICRO].float())
   if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite smoke loss')
   weight=len(value[offset:offset+MICRO])/len(value)
   (loss*weight).backward();weighted.append(float(loss.detach())*weight)
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();emas[name].update_after_step(model);losses[name]=float(sum(weighted))
 return losses,lengths

def project_budget(samples):
 values=np.asarray(samples,dtype=np.float64)
 if values.shape!=(28,) or not np.isfinite(values).all() or np.any(values<=0):raise RuntimeError('28 finite positive paired update times required')
 # One sample executes TWO conditions. Two GPUs each own one arm's pair.
 per_pair=float(np.percentile(values,95)); two_gpu_train=per_pair*731*2*1.5
 return {'measured_pair_update_p95':per_pair,'training_2gpu_2epochs_4cells_50pct_margin':two_gpu_train,
         'serial_training_with_50pct_margin':two_gpu_train*2,'full_scoring_budget':3600.,'overhead_budget':900.,
         'total_conservative':two_gpu_train+4500.,'hard_budget':14400.,'two_gpu_parallel_arms_assumed':True}

def run(formal: Path, output: Path, *, arm: str, device='cuda:0', physical_gpu: int, threads=1):
 if threads!=1: raise RuntimeError('resource smoke requires one CPU thread')
 pre=preflight(formal,output,arm=arm,device=device,physical_gpu=physical_gpu)
 import torch
 if not torch.cuda.is_available() or torch.cuda.current_device()!=0: raise RuntimeError('requested visible cuda:0 unavailable')
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(42);np.random.seed(42);torch.cuda.reset_peak_memory_stats()
 from tfpd_exploration.src.h1_optimized_v2.cache import CACHE,ROOT as cache_root,validate_authority
 from tfpd_exploration.src.h1_optimized_v2.paired_train import groups
 from tfpd_exploration.src.h1_optimized_v4.paired_train import batches
 from tfpd_exploration.src.h1_family_v1.model import make_v2_unscaled_dot_localbalanced_pair
 from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
 cache=torch.load(CACHE,map_location='cpu',weights_only=False);validate_authority(cache,json.loads((cache_root/'source_cache_authority.json').read_text()))
 ordered=batches(cache,1)[:UPDATES]
 if len(ordered)!=UPDATES or any(not 1<=len(starts)<=EFFECTIVE for _,starts in ordered): raise RuntimeError('exact first32 original effective batches required')
 pair=make_v2_unscaled_dot_localbalanced_pair(seed=42);base=pair[0 if arm=='flat' else 1];del pair
 frozen=torch.load(formal/'exports'/f'{arm}_selected_plain_ema.pt',map_location='cpu',weights_only=True);frozen_sha=state_digest(frozen);base.load_state_dict(frozen,strict=True)
 import copy
 cells={n:copy.deepcopy(base).cuda() for n in ('CONTROL','PREFIX')};initial={n:state_digest(m.state_dict()) for n,m in cells.items()}
 if initial['CONTROL']!=initial['PREFIX'] or initial['CONTROL']!=frozen_sha:raise RuntimeError('control/prefix/frozen initial clone mismatch')
 opts={n:torch.optim.AdamW(groups(m),lr=1e-5,weight_decay=.01) for n,m in cells.items()};emas={n:_fresh_ema(m) for n,m in cells.items()}
 started=time.monotonic(); rows=[]; ident=hashlib.sha256()
 for i,(session,starts) in enumerate(ordered):
  if time.monotonic()-started>LIMIT_SECONDS:raise TimeoutError('300 second smoke wall')
  row=cache['train'][session]; starts=np.asarray(starts)
  count=len(starts)
  if starts.dtype!=np.int64 or starts.shape!=(count,) or np.any(starts<0) or np.any(starts+700>len(row['neural'])):raise RuntimeError('raw sampler starts contract')
  if row['neural'].dtype!=np.float32 or row['velocity'].dtype!=np.float32 or row['neural'].shape[1:]!=(176,) or row['velocity'].shape!=(len(row['neural']),7) or not np.isfinite(row['neural']).all() or not np.isfinite(row['velocity']).all():raise RuntimeError('raw source array contract')
  x=np.stack([row['neural'][s:s+700] for s in starts]);y=np.stack([row['velocity'][s+699] for s in starts])*20
  bank=H1Bank(*[row['bank'][k].cuda() for k in ('E0','T','unit_mask')])
  # retain original epoch-1 deterministic whole-unit law when present in sampler module.
  from tfpd_exploration.src.h1_family_v1.familyformal_split_train import dropout_keep
  keep=dropout_keep(n=count,epoch=1,batch_index=i,bank_mask=row['bank']['unit_mask'],device=torch.device('cuda:0'))
  torch.cuda.synchronize();before=time.monotonic();loss,lengths=_run_update(torch,cells,opts,emas,(torch.as_tensor(x,device='cuda:0'),torch.as_tensor(y,device='cuda:0'),bank,keep),epoch=1,batch_id=i);torch.cuda.synchronize();elapsed=time.monotonic()-before
  ident.update(session.encode());ident.update(starts.tobytes());ident.update(keep.detach().cpu().numpy().tobytes());ident.update(lengths.detach().cpu().numpy().tobytes());rows.append({'update':i+1,'rows':count,'elapsed_s':elapsed,'loss':loss,'cold_rows':int((lengths<700).sum())})
  if torch.cuda.max_memory_allocated()>MEMORY_LIMIT:raise MemoryError('22GiB smoke memory stop threshold')
  if time.monotonic()-started>LIMIT_SECONDS:raise TimeoutError('300 second smoke wall')
 if state_digest(torch.load(formal/'exports'/f'{arm}_selected_plain_ema.pt',map_location='cpu',weights_only=True))!=frozen_sha:raise RuntimeError('frozen selected export changed')
 post=preflight(formal,output,arm=arm,device=device,physical_gpu=physical_gpu)
 if post!=pre:raise RuntimeError('post smoke authority drift')
 forecast=project_budget([r['elapsed_s'] for r in rows[4:]])
 result={'schema':'h1_cold_history_resource_smoke_v1','status':'PASS_32_UPDATE_RESOURCE_SMOKE',
         'pre':pre,'post':post,'updates_per_condition':UPDATES,'microbatch':MICRO,'effective_batch':EFFECTIVE,
         'ema_updates':{k:v.n_updates for k,v in emas.items()},'initial_state':initial,
         'frozen_selected_state_digest':frozen_sha,'identity_sha256':ident.hexdigest(),'updates':rows,
         'first4_update_seconds':[r['elapsed_s'] for r in rows[:4]],'remaining28_update_seconds':[r['elapsed_s'] for r in rows[4:]],
         'peak_memory_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.monotonic()-started,
         'forecast_seconds':forecast,'no_minival_or_source208_forward':True,'no_selection_or_checkpoint':True,
         'source_targets_used_only_for_32_training_steps':True}
 atomic_json(result,output/'receipt.json');return result
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--formal',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--arm',choices=('flat','route'),required=True);p.add_argument('--device',choices=('cuda:0',),required=True);p.add_argument('--physical-gpu',type=int,choices=(0,1),required=True);p.add_argument('--threads',type=int,choices=(1,),required=True);a=p.parse_args(argv);return run(a.formal,a.output,arm=a.arm,device=a.device,physical_gpu=a.physical_gpu,threads=a.threads)
if __name__=='__main__':main()
