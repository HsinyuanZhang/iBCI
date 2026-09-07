"""Prospective fixed-two dense-supervision continuation; no automatic launch."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,random,tempfile,time
from pathlib import Path
import numpy as np
POSITIONS=(349,466,582,699);EPOCHS,UPDATES,MICRO,SOURCE_WINDOWS=2,731,8,23212
PROTOCOL={'schema':'h1_dense_phase2_paired_selectedema_v2','positions':list(POSITIONS),'loss':'equal .25 point weights, renormalized by total valid weight over each full effective batch; endpoint required','initial':'actual selected e12 plain EMA export; fresh AdamW and fresh EMA; same initialization as cold CONTROL','epochs':2,'updates_per_epoch':731,'microbatch':8,'lr':1e-5,'ema':.9995,'dropout':'original deterministic p=.1 and original epoch1/2 sampler identities','training':'source-train gradients only; existing combined cache is read and validated, minival is never used for updates or scored','primary':'fixed post-stage EMA complete20325; RAW secondary; no selection/promotion','runtime':'native W700 frontend/temporal/runtime unchanged','hard_seconds':21600,'memory_limit_bytes':22<<30}
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(v,p):
 p.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile(dir=p.parent,mode='w',delete=False) as h:t=Path(h.name);json.dump(v,h,sort_keys=True,indent=2,allow_nan=False);h.flush();os.fsync(h.fileno())
 os.replace(t,p)
def collect_bindings(formal,output,*,smoke_receipt=None):
 """Read-only authority body for an external authorizer; no model/cache load."""
 if not formal.is_absolute() or not output.is_absolute():raise RuntimeError('absolute formal/output required')
 paths=[Path(__file__),Path(__file__).with_name('dense_supervision.py'),
        Path(__file__).parents[1]/'family_runtime_v1/complete_h1_family_source.py',
        Path(__file__).parents[1]/'h1_temporal_decoder_quick_product_v1/ema.py',
        formal/'exports'/'flat_selected_plain_ema.pt',formal/'exports'/'route_selected_plain_ema.pt',
        formal/'workers'/'flat_complete.json',formal/'workers'/'route_complete.json']
 if smoke_receipt is not None: paths.append(smoke_receipt)
 files={str(p):sha(p) for p in paths}
 # This is the actual complete-H1 artifact, source and 83-file code closure,
 # rather than a hand-maintained subset of its inputs.
 from tfpd_exploration.src.family_runtime_v1.complete_h1_family_source import artifact_audit,code_source_audit
 artifact,source=artifact_audit(formal),code_source_audit(formal)
 identities={}
 for arm in ('flat','route'):
  worker=json.loads((formal/'workers'/f'{arm}_complete.json').read_text())
  identities[arm]={str(e):worker['identities'][str(e)] for e in (1,2)}
 if identities['flat']!=identities['route']:raise RuntimeError('formal arms source identity disagreement')
 bound={'protocol':PROTOCOL,'protocol_sha256':digest(PROTOCOL),'formal':str(formal),'output':str(output),'files':files,
        'formal_artifact':artifact,'formal_source':source,'formal_epoch_identities':identities['flat']}
 if smoke_receipt is not None:
  smoke=json.loads(smoke_receipt.read_text())
  validate_smoke(smoke,bound)
  bound['smoke_receipt']={'path':str(smoke_receipt),'sha256':files[str(smoke_receipt)],'forecast_seconds':smoke['forecast_seconds']}
 return bound
def validate_smoke(smoke,bound):
 if (smoke.get('schema')!='h1_dense_phase2_resource_smoke_v1' or smoke.get('status')!='PASS_RESOURCE_SMOKE_NO_SCORE' or smoke.get('pre')!=smoke.get('post') or smoke.get('updates_per_condition')!=20 or len(smoke.get('updates',[]))!=20 or smoke.get('ema_updates')!={'flat':20,'route':20} or smoke.get('microbatch')!=MICRO or smoke.get('no_scoring_selection_or_checkpoint') is not True):raise RuntimeError('passed paired dense smoke receipt required')
 for key in ('formal_artifact','formal_source','formal_epoch_identities','protocol','protocol_sha256','formal'):
  if smoke['pre'].get(key)!=bound[key]:raise RuntimeError('dense smoke source/protocol differs from formal candidate')
 if any(sha(path)!=value or bound['files'].get(path)!=value for path,value in smoke['pre']['files'].items()):raise RuntimeError('dense smoke executable/input closure drift')
 steady=np.asarray(smoke.get('steady16_seconds'),dtype=float);forecast=smoke.get('forecast_seconds',{}).get('formal_with_50pct_margin_plus_allowance',float('nan'))
 if steady.shape!=(16,) or not np.isfinite(steady).all() or np.any(steady<=0) or not np.isfinite(forecast) or not 0<forecast<21600 or abs(forecast-(float(np.percentile(steady,95))*1462*1.5+1800))>1e-7 or not 0<smoke.get('peak_memory_bytes',0)<22<<30:raise RuntimeError('dense smoke resource/forecast drift')
protocol_generator=collect_bindings
def preflight(formal,output,authorization,authorization_sha,*,smoke_receipt=None,allow_existing=False):
 if output.exists() and not allow_existing:raise FileExistsError(output)
 if os.environ.get('H1_DENSE_PHASE2_GO')!='1' or os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or len(authorization_sha)!=64 or sha(authorization)!=authorization_sha:raise RuntimeError('explicit GO/GPU0/new-output/authorization gate')
 bound=collect_bindings(formal,output,smoke_receipt=smoke_receipt)
 auth=json.loads(Path(authorization).read_text())
 if auth.get('schema')!='h1_dense_phase2_root_authorization_v2' or auth.get('bindings')!=bound:raise RuntimeError('external dense authorization drift')
 return bound
def dense_step(torch,models,opts,emas,x,record,starts,bank,keep,*,guard=lambda:None,live=lambda _:None):
 from .dense_supervision import forward_positions,dense_targets,weighted_terms
 target,valid=dense_targets(record,starts);losses={}
 full_denom=(valid.to(x.device)*.25).sum()
 if float(full_denom)==0:raise RuntimeError('no full effective dense denominator')
 for arm in ('flat','route'):
  m,o=models[arm],opts[arm];m.train();o.zero_grad(set_to_none=True);total=0.
  for i in range(0,len(x),MICRO):
   guard();p=forward_positions(m,x[i:i+MICRO],bank,keep[i:i+MICRO]);num,_=weighted_terms(p,target[i:i+MICRO].to(x),valid[i:i+MICRO]);loss=num/full_denom;(loss).backward();total+=float(loss.detach());guard();live({'arm':arm,'micro_start':i,'loss':float(loss.detach())})
  if not np.isfinite(total):raise RuntimeError('nonfinite dense loss')
  torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);o.step();emas[arm].update_after_step(m);guard();losses[arm]=total
 return {'loss':losses,'valid_fraction':float(valid.float().mean()),'endpoint_valid':bool(valid[:,-1].all())}
def _state_digest(state):
 h=hashlib.sha256()
 for n,t in sorted(state.items()):
  a=t.detach().cpu().contiguous().numpy();h.update(n.encode());h.update(a.tobytes())
 return h.hexdigest()
def _checkpoint(torch,models,opts,emas,epoch,identities,authority_sha256=None):
 # State dicts alias live parameters; clone before the atomic disk write so a
 # later optimizer step cannot change what this checkpoint claims to contain.
 if any(e.n_updates!=epoch*UPDATES for e in emas.values()):raise RuntimeError('checkpoint EMA/update count drift')
 frozen=lambda state:{k:v.detach().clone() for k,v in state.items()}
 return {'schema':'h1_dense_phase2_end_epoch_v1','protocol_sha256':digest(PROTOCOL),'authority_sha256':authority_sha256,'epoch':epoch,'global_step':epoch*UPDATES,'identities':copy.deepcopy(identities),'models':{a:frozen(m.state_dict()) for a,m in models.items()},'optimizers':copy.deepcopy({a:o.state_dict() for a,o in opts.items()}),'emas':copy.deepcopy({a:e.checkpoint_state() for a,e in emas.items()}),'rng':{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'numpy':np.random.get_state(),'python':random.getstate()}}
def _restore_checkpoint(torch,payload,models,opts,emas,epoch,identities,authority_sha256=None):
 if (payload.get('schema')!='h1_dense_phase2_end_epoch_v1' or payload.get('protocol_sha256')!=digest(PROTOCOL) or payload.get('authority_sha256')!=authority_sha256 or payload.get('epoch')!=epoch or payload.get('global_step')!=epoch*UPDATES or payload.get('identities')!=identities or set(payload.get('models',{}))!={'flat','route'} or set(payload.get('optimizers',{}))!={'flat','route'} or set(payload.get('emas',{}))!={'flat','route'}):raise RuntimeError('dense disk checkpoint topology/epoch/identity drift')
 for arm in ('flat','route'):
  if payload['emas'][arm].get('n_updates')!=epoch*UPDATES or payload['emas'][arm].get('decay')!=.9995 or set(payload['emas'][arm].get('shadow',{}))!={n for n,p in models[arm].named_parameters() if p.requires_grad}:raise RuntimeError('dense checkpoint EMA topology/count drift')
  models[arm].load_state_dict(payload['models'][arm],strict=True);opts[arm].load_state_dict(payload['optimizers'][arm]);emas[arm].load_checkpoint_state(payload['emas'][arm]);emas[arm].shadow={k:v.to(next(models[arm].parameters()).device) for k,v in emas[arm].shadow.items()}
  if _state_digest(models[arm].state_dict())!=_state_digest(payload['models'][arm]) or _state_digest(emas[arm].shadow)!=_state_digest(payload['emas'][arm]['shadow']):raise RuntimeError('dense disk restore state drift')
 torch.set_rng_state(payload['rng']['torch']);torch.cuda.set_rng_state_all(payload['rng']['cuda']);np.random.set_state(payload['rng']['numpy']);random.setstate(payload['rng']['python'])
def _load_selected(torch,formal,models):
 """Exact selected plain EMA clones; fresh optimizer/EMA deliberately follow."""
 initial={}
 for arm in ('flat','route'):
  state=torch.load(formal/'exports'/f'{arm}_selected_plain_ema.pt',map_location='cpu',weights_only=True)
  if not state or any(t.dtype!=torch.float32 or not bool(torch.isfinite(t).all()) for t in state.values()):raise RuntimeError('selected EMA FP32 finite drift')
  models[arm].load_state_dict(state,strict=True);initial[arm]=_state_digest(state)
  if _state_digest(models[arm].state_dict())!=initial[arm]:raise RuntimeError('selected EMA strict load drift')
 return initial
def _loop(torch,cache,models,opts,emas,*,epochs,limit_updates,device,start_epoch=1,expected=None,guard=lambda:None,live=lambda _:None,step_fn=None):
 from tfpd_exploration.src.h1_optimized_v4.paired_train import batches
 from tfpd_exploration.src.h1_family_v1.familyformal_split_train import dropout_keep,sampler_identity_digest
 from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
 rows=[];identities={}
 for epoch in range(start_epoch,epochs+1):
  ordered=batches(cache,epoch);sampler=sampler_identity_digest(ordered)
  if len(ordered)!=UPDATES or sum(len(s) for _,s in ordered)!=SOURCE_WINDOWS:raise RuntimeError('original sampler 731/23212 cardinality drift')
  if expected is not None and sampler!=expected[str(epoch)]['sampler_sha256']:raise RuntimeError('original exact sampler identity drift')
  keep_digest=hashlib.sha256();identities[str(epoch)]={'sampler_sha256':sampler,'batches':len(ordered),'windows':sum(len(s) for _,s in ordered)}
  for bi,(session,starts) in enumerate(ordered):
   if len(rows)>=limit_updates:return rows,identities
   guard();row=cache['train'][session]; starts=np.asarray(starts,dtype=np.int64)
   if starts.ndim!=1 or not 1<=len(starts)<=32 or len(np.unique(starts))!=len(starts) or np.any(starts<0) or np.any(starts+700>len(row['neural'])):raise RuntimeError('source batch geometry drift')
   x=np.stack([row['neural'][s:s+700] for s in starts]).astype('float32')
   if x.shape!=(len(starts),700,176) or not np.isfinite(x).all():raise RuntimeError('source W700 finite geometry drift')
   bank=H1Bank(*[row['bank'][k].to(device) for k in ('E0','T','unit_mask')]);keep=dropout_keep(n=len(x),epoch=epoch,batch_index=bi,bank_mask=row['bank']['unit_mask'],device=device)
   keep_digest.update(session.encode());keep_digest.update(starts.tobytes());keep_digest.update(keep.detach().cpu().numpy().tobytes())
   item=(step_fn or dense_step)(torch,models,opts,emas,torch.as_tensor(x,device=device),row,starts,bank,keep,guard=guard,live=live);rows.append({'epoch':epoch,'batch':bi,'session':session,'rows':len(x),**item});guard();live({'epoch':epoch,'batch':bi+1,'updates_per_arm':len(rows),'last':rows[-1]})
  identities[str(epoch)]['keep_sha256']=keep_digest.hexdigest()
  if expected is not None and keep_digest.hexdigest()!=expected[str(epoch)]['keep_sha256']:raise RuntimeError('original exact dropout identity drift')
 return rows,identities
def run(formal,output,authorization,authorization_sha,*,mode='formal2',smoke_receipt=None):
 started=time.monotonic()
 if mode not in ('smoke20','formal2'):raise ValueError('mode')
 if mode=='formal2' and smoke_receipt is None:raise RuntimeError('formal2 requires passed dense smoke receipt')
 bound=preflight(formal,output,authorization,authorization_sha,smoke_receipt=smoke_receipt)
 import torch
 if not torch.cuda.is_available() or torch.cuda.current_device()!=0:raise RuntimeError('GPU0 required')
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(42);np.random.seed(42);random.seed(42);torch.cuda.reset_peak_memory_stats()
 def guard():
  if time.monotonic()-started>(900 if mode=='smoke20' else 21600):raise TimeoutError('dense hard wall')
  if torch.cuda.max_memory_allocated()>22<<30:raise MemoryError('22GiB dense threshold')
 guard()
 from tfpd_exploration.src.h1_optimized_v2.cache import CACHE,ROOT,validate_authority
 from tfpd_exploration.src.h1_optimized_v2.paired_train import groups
 from tfpd_exploration.src.h1_family_v1.model import make_v2_unscaled_dot_localbalanced_pair
 from tfpd_exploration.src.h1_temporal_decoder_quick_product_v1.ema import DecoderEMA
 from tfpd_exploration.src.h1_family_v1.familyformal_split_train import atomic_torch_save
 cache=torch.load(CACHE,map_location='cpu',weights_only=False);validate_authority(cache,json.loads((ROOT/'source_cache_authority.json').read_text()));pair=make_v2_unscaled_dot_localbalanced_pair(seed=42);models={'flat':pair[0].cuda(),'route':pair[1].cuda()};initial=_load_selected(torch,formal,models);opts={a:torch.optim.AdamW(groups(m),lr=1e-5,weight_decay=.01) for a,m in models.items()};emas={a:DecoderEMA(m,decay=.9995) for a,m in models.items()}
 guard()
 if smoke_receipt is not None and initial!=json.loads(smoke_receipt.read_text())['initial']:raise RuntimeError('dense smoke initial state drift')
 output.mkdir(parents=True);atomic({'bindings':bound,'authorization_sha256':authorization_sha,'mode':mode,'initial':initial},output/'input_authority.json')
 authority_sha256=sha(output/'input_authority.json');last_live=[0.]
 def live(value):
  now=time.monotonic()
  if now-last_live[0]>=1.:
   atomic({'status':'TRAINING','elapsed_seconds':now-started,'last':value},output/'live.json');last_live[0]=now
 if mode=='smoke20':
  timings=[]
  # Per-update timing is deliberately paired (both arms), with warm 4 / steady 16.
  original=dense_step
  def timed(*args,**kwargs):
   torch.cuda.synchronize();t=time.monotonic();value=original(*args,**kwargs);torch.cuda.synchronize();timings.append(time.monotonic()-t);return value
  rows,ids=_loop(torch,cache,models,opts,emas,epochs=1,limit_updates=20,device=torch.device('cuda:0'),expected=bound['formal_epoch_identities'],guard=guard,live=live,step_fn=timed)
  if len(rows)!=20 or len(timings)!=20 or any(e.n_updates!=20 for e in emas.values()):raise RuntimeError('exact20 smoke update/EMA count')
  steady=np.asarray(timings[4:],dtype=float);forecast=float(np.percentile(steady,95))*1462*1.5+1800.
  if not forecast<21600:raise RuntimeError('smoke forecast violates formal hard bound')
  post=preflight(formal,output,authorization,authorization_sha,allow_existing=True)
  if post!=bound:raise RuntimeError('fresh smoke authority drift')
  guard()
  result={'schema':'h1_dense_phase2_resource_smoke_v1','status':'PASS_RESOURCE_SMOKE_NO_SCORE','pre':bound,'post':post,'initial':initial,'updates':rows,'identities':ids,'updates_per_condition':20,'microbatch':MICRO,'ema_updates':{a:e.n_updates for a,e in emas.items()},'warm4_seconds':timings[:4],'steady16_seconds':timings[4:],'forecast_seconds':{'paired_update_p95':float(np.percentile(steady,95)),'formal_with_50pct_margin_plus_allowance':forecast},'elapsed_seconds':time.monotonic()-started,'peak_memory_bytes':torch.cuda.max_memory_allocated(),'no_scoring_selection_or_checkpoint':True}
  atomic(result,output/'receipt.json');return result
 # Formal execution commits a disk-authoritative, strict-reloaded checkpoint at EACH epoch.
 rows=[];ids={};checkpoints=[]
 for epoch in (1,2):
  epoch_rows,identity=_loop(torch,cache,models,opts,emas,epochs=epoch,start_epoch=epoch,limit_updates=UPDATES,device=torch.device('cuda:0'),expected=bound['formal_epoch_identities'],guard=guard,live=live)
  rows.extend(epoch_rows);ids[str(epoch)]=identity[str(epoch)]
  if len(epoch_rows)!=UPDATES or any(emas[a].n_updates!=epoch*UPDATES for a in emas):raise RuntimeError('dense exact per-epoch update/EMA drift')
  payload=_checkpoint(torch,models,opts,emas,epoch,ids[str(epoch)],authority_sha256);path=output/f'dense_stage_epoch{epoch:02d}.pt';atomic_torch_save(payload,path);recorded=sha(path)
  loaded=torch.load(path,map_location='cpu',weights_only=False)
  if sha(path)!=recorded:raise RuntimeError('dense checkpoint changed before disk reload')
  _restore_checkpoint(torch,loaded,models,opts,emas,epoch,ids[str(epoch)],authority_sha256);guard();checkpoints.append({'epoch':epoch,'path':str(path),'sha256':recorded,'identities':ids[str(epoch)]})
 post=preflight(formal,output,authorization,authorization_sha,smoke_receipt=smoke_receipt,allow_existing=True)
 if post!=bound:raise RuntimeError('fresh formal authority drift')
 if any(sha(Path(x['path']))!=x['sha256'] for x in checkpoints):raise RuntimeError('checkpoint immutability drift')
 if sha(output/'input_authority.json')!=authority_sha256:raise RuntimeError('dense authority mutation')
 guard()
 result={'schema':'h1_dense_phase2_formal_train_v1','status':'TRAINING_COMPLETE_NO_SCORE_OR_SELECTION','pre':bound,'post':post,'initial':initial,'updates':rows,'identities':ids,'checkpoints':checkpoints,'ema_updates':{a:e.n_updates for a,e in emas.items()},'elapsed_seconds':time.monotonic()-started,'peak_memory_bytes':torch.cuda.max_memory_allocated(),'no_scoring_selection_or_promotion':True}
 atomic(result,output/'receipt.json');return result
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--formal',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--authorization',type=Path,required=True);p.add_argument('--authorization-sha256',required=True);p.add_argument('--mode',choices=('smoke20','formal2'),required=True);p.add_argument('--smoke-receipt',type=Path);a=p.parse_args(argv);return run(a.formal,a.output,a.authorization,a.authorization_sha256,mode=a.mode,smoke_receipt=a.smoke_receipt)
if __name__=='__main__':main()
