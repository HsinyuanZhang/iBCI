#!/usr/bin/env python3
"""Static builder and independent host verifier for M2 RIFT-R50 ablations.

This is a packer only: it never trains, opens NWB, calls EvalAI, or runs Docker.
The build accepts only the new file-per-array bank cache ABI; ``bank_payload``
objects are intentionally never accepted.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pickle, shutil, sys
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch

ROOT=Path('/home/xinyuan/Work_host/SPINT'); HERE=Path(__file__).resolve().parent
V1=ROOT/'btransform_unified_v1/src'; V2=ROOT/'btransform_unified_v2/src'
ARMS=('ACTIVITY_ONLY','NONE'); CHANNELS=96; CONTEXT=50; EPOCHS=24; STEPS=75960; GATE=1e-5
SIX_QUERY={'ses-2020-10-30-Run1','ses-2020-10-30-Run2','ses-2020-11-18-Run1','ses-2020-11-19-Run1','ses-2020-11-24-Run1','ses-2020-11-24-Run2'}
SEVEN_SOURCE={'ses-2020-10-19-Run1','ses-2020-10-19-Run2','ses-2020-10-20-Run1','ses-2020-10-20-Run2','ses-2020-10-27-Run1','ses-2020-10-27-Run2','ses-2020-10-28-Run1'}

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def arrsha(a:np.ndarray)->str:return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def jload(p:Path)->dict:
 x=json.loads(p.read_text());
 if not isinstance(x,dict):raise RuntimeError(f'object JSON required: {p}')
 return x
def json_digest(x:Mapping[str,Any])->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def need(root:Path,name:str)->Path:
 p=root/name
 if not p.is_file():raise FileNotFoundError(p)
 return p
def tag(session:str)->str:
 x=session.split('-')
 if len(x)!=5 or x[0]!='ses' or not x[4].startswith('Run'):raise ValueError(f'noncanonical session {session}')
 return f'{x[4]}_{x[1]}{x[2]}{x[3]}'
def session(value:str)->str:
 run,date=value.split('_')
 if not run.startswith('Run') or len(date)!=8 or not date.isdigit():raise ValueError(f'noncanonical tag {value}')
 return f'ses-{date[:4]}-{date[4:6]}-{date[6:]}-{run}'

def _e0(p:Path)->np.ndarray:
 x=torch.load(p,map_location='cpu',weights_only=False)
 if not isinstance(x,Mapping) or set(('E0','frozen_u'))-set(x):raise RuntimeError(f'{p}: requires E0 and frozen_u')
 return np.ascontiguousarray(torch.as_tensor(x['E0']).cpu().numpy(),dtype=np.float32)
def static(tag_value:str,row:Mapping[str,Any],arm:str)->None:
 if set(row)-{'session','E0','T','unit_mask','e0_sha256','t_sha256','cache_files'}:raise RuntimeError(f'{tag_value}: forbidden FULL/unknown static field')
 e,t,m=np.asarray(row['E0']),np.asarray(row['T']),np.asarray(row['unit_mask'])
 if e.shape!=(96,50) or t.shape!=(96,4) or m.shape!=(96,):raise RuntimeError(f'{tag_value}: E0/T/mask geometry drift')
 if e.dtype!=np.float32 or t.dtype!=np.float32 or m.dtype!=np.bool_ or not np.isfinite(e).all() or not np.isfinite(t).all() or not m.any():raise RuntimeError(f'{tag_value}: invalid finite static ABI')
 if arm=='ACTIVITY_ONLY' and t.any():raise RuntimeError(f'{tag_value}: ACTIVITY_ONLY requires P0+EMPTY zero-side T')
 if arm=='NONE' and (e.any() or t.any()):raise RuntimeError(f'{tag_value}: NONE requires E0/T zeros')

def banks(root:Path,arm:str)->tuple[dict[str,dict],dict]:
 manifest=jload(need(root,'manifest.json'))
 if manifest.get('schema')!='m2_rift_r50_ablation_bank_v1' or manifest.get('status')!='COMPLETED' or manifest.get('arm')!=arm:raise RuntimeError('bank manifest schema/status/arm mismatch')
 for key in ('base_cache_meta_sha256','builder_sha256','ext6_query_receipt','ext6_query_receipt_sha256','source_manifest_contract'):
  if not manifest.get(key):raise RuntimeError(f'bank manifest missing explicit binding {key}')
 # The builder records source bindings as ``rows`` keyed by surface/session;
 # directory topology itself is the authoritative ABI.  Do not accept an old
 # serialised bank object in place of these linked arrays.
 required_surfaces={'source_train','source_minival','ext4','ext6_query'}
 if any(not (root/s).is_dir() for s in required_surfaces):raise RuntimeError(f'missing cache root(s) under {root}')
 rows=manifest.get('rows')
 if not isinstance(rows,Mapping):raise RuntimeError('manifest must declare source-bank rows')
 row_hashes=('E0_sha256','T_sha256','frozen_u_sha256','X_store_sha256','target_store_sha256','eligible_starts_sha256','calib_activity_sha256')
 for key,meta in rows.items():
  if not (root/str(key)).is_dir():raise RuntimeError(f'manifest row missing its linked directory: {key}')
  if not isinstance(meta,Mapping) or any(not meta.get(h) for h in row_hashes):raise RuntimeError(f'{key}: missing explicit manifest hash')
 out={}
 # Official payload consists of the seven source-train sessions and the six
 # separately sealed public held-out query sessions.  Minival/ext4 rows are
 # validated above as cache provenance, but are not duplicate payload banks.
 chosen=[(str(k).split('/',1)[1],root/str(k),dict(v)) for k,v in rows.items() if str(k).startswith('source_train/')]
 if {s for s,_,_ in chosen}!=SEVEN_SOURCE:raise RuntimeError('source_train roster is not the canonical seven M2 sessions')
 qreceipt=jload(need(root/'ext6_query','official_heldout_query_banks.json'))
 if qreceipt.get('schema')!='m2_rift_r50_ablation_ext6_query_v1' or qreceipt.get('status')!='COMPLETED' or qreceipt.get('arm')!=arm or qreceipt.get('hidden_or_test_opened') is not False:raise RuntimeError('unsealed EXT6 query receipt')
 if sha(need(root/'ext6_query','official_heldout_query_banks.json'))!=manifest['ext6_query_receipt_sha256']:raise RuntimeError('EXT6 query receipt SHA drift')
 if set(qreceipt.get('sessions',{}))!=SIX_QUERY:raise RuntimeError('EXT6 query roster is not canonical six')
 for ses,qmeta in qreceipt['sessions'].items():
  if not isinstance(qmeta,Mapping) or any(not qmeta.get(key) for key in ('E0_sha256','T_sha256','frozen_u_sha256','X_link_sha256','target_link_sha256')):raise RuntimeError(f'{ses}: missing explicit EXT6 hash binding')
 chosen += [(str(s),root/'ext6_query'/str(s),dict(qreceipt.get('sessions',{}).get(str(s),{}))) for s in SIX_QUERY]
 for ses,base,meta in chosen:
  if not base.is_dir():raise RuntimeError(f'{ses}: absent selected bank directory')
  mapping=jload(need(base,'mapping.json'))
  if str(mapping.get('session_id',ses))!=ses:raise RuntimeError(f'{ses}: mapping session mismatch')
  e,t=_e0(need(base,'e0_u.pt')),np.ascontiguousarray(np.load(need(base,'T.npy')),dtype=np.float32)
  mp=base/'unit_mask.npy'; m=np.ascontiguousarray(np.load(mp),dtype=np.bool_) if mp.exists() else np.ones(96,dtype=np.bool_)
  for n in ('X_store.npy','target_store.npy','eligible_starts.npy'):need(base,n)
  k=tag(ses); row={'session':ses,'E0':e,'T':t,'unit_mask':m,'e0_sha256':arrsha(e),'t_sha256':arrsha(t),'cache_files':{n:sha(base/n) for n in ('mapping.json','e0_u.pt','T.npy','X_store.npy','target_store.npy','eligible_starts.npy')}}
  static(k,row,arm)
  ehash=meta.get('e0_sha256',meta.get('E0_sha256')); thash=meta.get('t_sha256',meta.get('T_sha256'))
  if not ehash or not thash or ehash!=row['e0_sha256'] or thash!=row['t_sha256']:raise RuntimeError(f'{ses}: manifest array hash drift')
  if k in out:raise RuntimeError(f'duplicate canonical tag {k}')
  out[k]=row
 if set(map(session,out))!=SEVEN_SOURCE|SIX_QUERY:raise RuntimeError('requires exact canonical 13-tag M2 roster')
 return out,manifest

def train(run:Path,arm:str,bank_manifest_sha:str)->tuple[Path,dict,Path,Path,dict]:
 p=need(run,'train_receipt.json');x=jload(p)
 if x.get('status')!='COMPLETED' or x.get('arm')!=arm:raise RuntimeError('completed matching train receipt required')
 meta=jload(need(run,'run_meta.json'))
 if int(x.get('epochs',-1))!=24 or int(x.get('global_step',-1))!=75960:raise RuntimeError('requires 24x3165 full source-seven train')
 if meta.get('schema')!='m2_rift_r50_ablation_train_v1' or meta.get('status')!='FORMAL' or meta.get('arm')!=arm:raise RuntimeError('formal arm-specific training metadata required')
 if int(meta.get('seed',-1))!=42 or int(meta.get('context_bins',-1))!=50 or meta.get('identity_interface')!='concat' or meta.get('variant')!='recency':raise RuntimeError('not original frozen RIFT R50 concat recipe')
 if x.get('source_hashes')!=meta.get('source_hashes') or x.get('frozen_cache_hashes')!=meta.get('frozen_cache_hashes'):raise RuntimeError('training receipt source/cache hash mismatch')
 if meta.get('bank_manifest_sha256')!=bank_manifest_sha:raise RuntimeError('training run is not bound to supplied bank manifest')
 for path,digest in meta['source_hashes'].items():
  if not isinstance(digest,str) or not digest or sha(Path(path))!=digest:raise RuntimeError(f'training source hash drift: {path}')
 return p,x,need(run,'epoch_024.pt'),need(run,'run_meta.json'),meta
def select(d:Path,arm:str)->tuple[Path,dict,Path]:
 p=need(d,'score_receipt.json');x=jload(p)
 if x.get('status')!='COMPLETED' or x.get('arm')!=arm:raise RuntimeError('completed matching score receipt required')
 s=x.get('selection');curve=x.get('ema_by_epoch')
 if not isinstance(s,Mapping) or 'equal_session_mean' not in str(s.get('rule','')):raise RuntimeError('requires ext6 equal_session_mean selection')
 if not isinstance(curve,Mapping) or {int(k) for k in curve}!={*range(1,25)}:raise RuntimeError('requires all 24 EMA score entries')
 scores={}
 for ep in range(1,25):
  r=curve[str(ep)]
  if not isinstance(r,Mapping) or not r.get('checkpoint_sha256') or 'equal_session_mean' not in r:raise RuntimeError(f'epoch {ep}: missing checkpoint SHA/report')
  scores[ep]=float(r['equal_session_mean'])
  if not np.isfinite(scores[ep]):raise RuntimeError('nonfinite selection score')
 chosen=int(s['epoch']); earliest=max(scores,key=lambda ep:(scores[ep],-ep))
 if chosen!=earliest:raise RuntimeError(f'selection must rederive earliest maximum {earliest}, found {chosen}')
 state=x.get('selected_ema_state',s.get('ema_state'))
 if not isinstance(state,Mapping):raise RuntimeError('selected_ema_state binding missing')
 ema=Path(str(state.get('path','')));ema=ema if ema.is_absolute() else d/ema
 if not ema.is_file() or sha(ema)!=state.get('sha256'):raise RuntimeError('selected EMA exact path/hash drift')
 if 'checkpoint_sha256' in state and state['checkpoint_sha256']!=curve[str(chosen)]['checkpoint_sha256']:raise RuntimeError('selected EMA checkpoint binding drift')
 return p,x,ema

def copy_runtime(dest:Path)->None:
 pkg=dest/'artifacts/pkg'
 for srcroot in (V1,V2):
  for f in srcroot.rglob('*.py'):
   if f.name in {'joint_m2_model.py','m2_mechanism_model.py','joint_m1_model.py'}:continue
   out=pkg/f.relative_to(srcroot);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,out)
 (pkg/'btransform_unified_v2/__init__.py').write_text('"""Container-safe RIFT package."""\nfrom .config import RiftTemporalConfig\n')
def source_hashes()->dict:
 d=ROOT/'btransform_unified_v2/scripts/m2_rift_r50_ablation_v1'
 return {str((d/n).relative_to(ROOT)):sha(need(d,n)) for n in ('build_banks.py','m2_ablation.py','score_ext6.py')}

def validate_all_checkpoints(run:Path,meta:Mapping[str,Any],score:Mapping[str,Any],selected_ema:Path)->dict[str,str]:
 """Seal every formal checkpoint and reconstruct the selected EMA exactly."""
 curve=score['ema_by_epoch']; hashes={}
 for epoch in range(1,25):
  cp=need(run,f'epoch_{epoch:03d}.pt'); digest=sha(cp); hashes[str(epoch)]=digest
  if digest!=curve[str(epoch)]['checkpoint_sha256']:raise RuntimeError(f'epoch {epoch}: score receipt checkpoint SHA drift')
  state=torch.load(cp,map_location='cpu',weights_only=False)
  expected={'schema':'m2_rift_concat_epoch_checkpoint_v1','epoch':epoch,'global_step':epoch*3165,'seed':42,'context_bins':50,'bias_mode':'recency','attention_backend':'local','identity_interface':'concat','epochs':24,'smoke':False}
  if any(state.get(k)!=v for k,v in expected.items()):raise RuntimeError(f'epoch {epoch}: formal checkpoint metadata drift')
  if state.get('source_hashes')!=meta['source_hashes'] or state.get('frozen_cache_hashes')!=meta['frozen_cache_hashes']:raise RuntimeError(f'epoch {epoch}: source/cache binding drift')
 chosen=int(score['selection']['epoch']); cp=torch.load(need(run,f'epoch_{chosen:03d}.pt'),map_location='cpu',weights_only=False)
 sys.path[:0]=[str(V1),str(V2)]
 from btransform_unified_v2.concat_model import RiftConcatDecoder
 model=RiftConcatDecoder('m2',context_bins=50,bias_mode='recency',seed=42)
 model.load_state_dict(cp['raw_state_dict'],strict=True)
 shadow=cp.get('ema',{}).get('shadow',{}); named=dict(model.named_parameters())
 if set(shadow)!=set(named):raise RuntimeError('selected checkpoint EMA parameter keys drift')
 with torch.no_grad():
  for name,value in named.items():value.copy_(shadow[name].to(value.dtype))
 rebuilt={k:v.detach().cpu().float() for k,v in model.state_dict().items()}
 sealed=torch.load(selected_ema,map_location='cpu',weights_only=False)
 if not isinstance(sealed,Mapping) or set(sealed)!=set(rebuilt):raise RuntimeError('selected EMA state keys differ from reconstructed EMA')
 for name,value in rebuilt.items():
  if not torch.equal(value,torch.as_tensor(sealed[name]).cpu().float()):raise RuntimeError(f'selected EMA tensor drift: {name}')
 return hashes

def build(a:argparse.Namespace)->None:
 if a.dest.exists():raise FileExistsError(f'fresh --dest required: {a.dest}')
 b,_=banks(a.bank_cache_root,a.arm); bank_sha=sha(need(a.bank_cache_root,'manifest.json'));tp,tr,final,meta_path,meta=train(a.run_dir,a.arm,bank_sha);sp,sc,ema=select(a.selection_dir,a.arm)
 sm_path=need(a.selection_dir,'manifest.json');sm=jload(sm_path)
 if sm.get('schema')!='m2_rift_r50_ablation_ext6_selection_v1' or sm.get('arm')!=a.arm:raise RuntimeError('selection manifest schema/arm drift')
 if Path(str(sm.get('run',''))).resolve()!=a.run_dir.resolve() or sm.get('run_meta_sha256')!=sha(meta_path) or sm.get('train_receipt_sha256')!=sha(tp):raise RuntimeError('selection manifest run/run_meta/train receipt binding drift')
 if sc.get('manifest_sha256')!=json_digest(sm):raise RuntimeError('score receipt selection manifest digest drift')
 checkpoint_hashes=validate_all_checkpoints(a.run_dir,meta,sc,ema)
 state=torch.load(ema,map_location='cpu',weights_only=False)
 if not isinstance(state,Mapping) or not state:raise RuntimeError('selected EMA must be actual nonempty state dict')
 a.dest.mkdir(parents=True);(a.dest/'artifacts').mkdir()
 for n in ('m2_rift_falcon_decoder.py','decode.py','Dockerfile','.dockerignore','README.md'):shutil.copy2(HERE/n,a.dest/n)
 copy_runtime(a.dest);sys.path[:0]=[str(HERE),str(V1),str(V2)]
 from m2_rift_falcon_decoder import PAYLOAD_SCHEMA
 # Cache file hashes are pack provenance, not runtime bank ABI.  Keeping them
 # out of every row lets the decoder reject any unrecognised/FULL fields.
 bank_cache_files={tag_value:dict(row['cache_files']) for tag_value,row in b.items()}
 runtime_banks={tag_value:{key:row[key] for key in ('session','E0','T','unit_mask','e0_sha256','t_sha256')} for tag_value,row in b.items()}
 bind={'bank_manifest_sha256':bank_sha,'bank_cache_files':bank_cache_files,'train_receipt_sha256':sha(tp),'run_meta_sha256':sha(meta_path),'run_path':str(a.run_dir.resolve()),'selection_receipt_sha256':sha(sp),'selection_manifest_sha256':sha(sm_path),'selection_path':str(a.selection_dir.resolve()),'epoch024_checkpoint_sha256':sha(final),'checkpoint_sha256_by_epoch':checkpoint_hashes,'selected_ema_sha256':sha(ema),'source_hashes':source_hashes(),'training_source_hashes':meta['source_hashes'],'training_frozen_cache_hashes':meta['frozen_cache_hashes']}
 payload={'schema':PAYLOAD_SCHEMA,'task':'m2','arm':a.arm,'context_bins':50,'bias_mode':'recency','identity_interface':'concat','behavior_scaling_factor':5.0,'bank_by_dataset_tag':runtime_banks,'ema_state_dict':{k:torch.as_tensor(v).cpu().float() for k,v in state.items()},'selection':{'surface':'ext6','epoch':int(sc['selection']['epoch']),'equal_session_mean':float(sc['selection']['equal_session_mean'])},'binding':bind}
 out=a.dest/'artifacts/m2_rift_r50_ablation.pkl'
 with out.open('wb') as f:pickle.dump(payload,f,protocol=4)
 rec={'status':'BUILT_NOT_HOST_VERIFIED','task':'m2','arm':a.arm,'method':'RIFT R50 concat cached CPU static ablation','payload_sha256':sha(out),'roster':sorted(b),'roster_count':13,'selection':payload['selection'],'binding':bind,'external_actions':False,'official_submission_id':None}
 (a.dest/'artifacts/local_pack_receipt.json').write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n');print(json.dumps(rec,indent=2,sort_keys=True))

def cache_dir(root:Path,k:str,payload:Mapping)->Path:
 ses=payload['bank_by_dataset_tag'][k]['session'];hits=[root/ses,root/'source_train'/ses,root/'ext6_query'/ses]
 hits=[p for p in hits if (p/'X_store.npy').is_file()]
 if len(hits)!=1:raise FileNotFoundError(f'{k}: expected one public X_store, found {hits}')
 return hits[0]
def trace(cache:Path,k:str)->np.ndarray:
 """Read public observations without treating zeroes as padding.

 New caches may expose a direct [time,96] trace or immutable overlapping
 [start,50,96] windows.  In the latter case reconstruct the trace from exact
 overlap, then use mapping's declared 49-bin query prefix instead of guessing
 from neural values.
 """
 x=np.asarray(np.load(cache/'X_store.npy',mmap_mode='r'),dtype=np.float32)
 mapping=jload(need(cache,'mapping.json'))
 starts=np.asarray(np.load(need(cache,'eligible_starts.npy')),dtype=np.int64)
 if starts.ndim!=1 or len(starts)<80 or np.any(np.diff(starts)<0):raise RuntimeError(f'{cache}: malformed eligible starts')
 declared=mapping.get('eligible_starts_padded')
 if declared is not None and not np.array_equal(starts,np.asarray(declared,dtype=np.int64)):raise RuntimeError(f'{cache}: mapping/eligible starts drift')
 # ``eligible_starts`` indexes legal *windows*, not raw neural coordinates.
 # The mapping, rather than its first window start, owns the padded timeline.
 padded=mapping.get('query_is_padded_timeline') is True
 pad=int(mapping.get('query_pad_bins',-1)) if padded else 0
 if padded and pad!=49:raise RuntimeError(f'{cache}: padded-timeline contract drift')
 if session(k) in SIX_QUERY and not padded:raise RuntimeError(f'{cache}: held-out query lacks padded-timeline contract')
 if x.ndim==2:
  if x.shape[1]!=96:raise RuntimeError(f'{cache}: malformed public trace')
  # Direct X_store is the session timeline.  For a declared padded timeline,
  # its first 49 rows are query padding even though legal window starts begin
  # at zero.  Never use those starts as raw-neural offsets.
  begin=pad if padded else 0
  if len(x)<begin+80:raise RuntimeError(f'{cache}: 2D timeline cannot cover first 80 raw bins')
  raw=x[begin:begin+80]
 elif x.ndim==3:
  if x.shape[1:]!=(50,96) or len(x)<80 or np.any(np.diff(starts[:80])!=1):raise RuntimeError(f'{cache}: cannot reconstruct chronological public trace')
  # row zero supplies its exact stored prehistory; following windows supply
  # one new final bin.  Equality checks make reconstruction auditable.
  for i in range(1,80):
   if not np.array_equal(x[i-1,1:],x[i,:-1]):raise RuntimeError(f'{cache}: overlapping X_store history drift')
  timeline=np.concatenate((x[0],x[1:80,-1]),axis=0)
  # ``timeline[0]`` has the exact global coordinate ``starts[0]``.  Select
  # the declared first raw coordinate (49 for a padded query) and reject a
  # window collection that cannot cover the required 80-bin session prefix.
  begin=pad if padded else 0; offset=begin-int(starts[0])
  if offset<0 or offset+80>len(timeline):raise RuntimeError(f'{cache}: 3D windows cannot cover first 80 raw bins')
  raw=timeline[offset:offset+80]
 else:raise RuntimeError(f'{cache}: X_store rank must be 2 or 3')
 if raw.shape!=(80,96) or not np.isfinite(raw).all():raise RuntimeError(f'{cache}: malformed reconstructed trace')
 return np.ascontiguousarray(raw)
def ref_forward(model,banks_,streams,valid):
 # State backend executes full R50 temporal attention and is independent from
 # the packed cached-KV backend under test.
 from btransform_unified_v2.cpu_runtime import CpuRiftRuntime
 if valid.ndim!=2 or valid.shape[1]!=len(streams) or valid.shape[0]>min(len(x) for x in streams):raise RuntimeError('reference valid-mask timeline/stream shape mismatch')
 r=CpuRiftRuntime(model,banks_,[x.session_id for x in banks_],temporal_backend='state');out=[]
 for t in range(valid.shape[0]):out.append(r.advance(torch.from_numpy(np.stack([x[t] for x in streams])),valid_mask=torch.from_numpy(valid[t])).cpu().numpy()/5.0)
 return np.asarray(out,dtype=np.float32)

def host(a:argparse.Namespace)->None:
 p=a.dest/'artifacts/m2_rift_r50_ablation.pkl';rp=a.dest/'artifacts/local_pack_receipt.json'
 if not p.is_file() or not rp.is_file():raise FileNotFoundError('build stage must complete first')
 if a.public_cache_root is None:raise ValueError('--public-cache-root is required')
 rec=jload(rp)
 if rec.get('status')!='BUILT_NOT_HOST_VERIFIED' or rec.get('payload_sha256')!=sha(p):raise RuntimeError('build receipt/payload drift')
 sys.path[:0]=[str(a.dest),str(V1),str(V2)]
 from falcon_challenge.config import FalconConfig,FalconTask
 from btransform_unified_v2.concat_model import RiftConcatDecoder
 from m2_rift_falcon_decoder import M2RiftCachedFalconDecoder,_task_bank,load_payload
 q=load_payload(p);tags=sorted(q['bank_by_dataset_tag']);hi=[k for k in tags if session(k) not in SIX_QUERY];ho=[k for k in tags if session(k) in SIX_QUERY]
 plans={'B1':[hi[0]],'B7_native':hi,'mixed_partial':[hi[0],ho[0],hi[1],ho[1]]};cfg=FalconConfig(task=FalconTask.m2);reports={};smoke=False
 for name,roster in plans.items():
  streams=[trace(cache_dir(a.public_cache_root,k,q),k) for k in roster]
  if min(map(len,streams))<80:raise RuntimeError(f'{name}: need >=80 real bins')
  packed=M2RiftCachedFalconDecoder(cfg,str(p),batch_size=len(roster));packed.reset(roster)
  m=RiftConcatDecoder('m2',context_bins=50,bias_mode='recency',seed=42);m.load_state_dict({k:torch.as_tensor(v) for k,v in q['ema_state_dict'].items()},strict=True);m.eval()
  want=ref_forward(m,[_task_bank(k,q['bank_by_dataset_tag'][k]) for k in roster],streams,np.ones((80,len(roster)),dtype=np.bool_))
  got=np.stack([packed.predict(np.stack([x[t] for x in streams])) for t in range(80)]);err=float(np.max(np.abs(got-want)))
  if err>GATE:raise RuntimeError(f'{name}: cached vs independent full-R50 gate {err}')
  reports[name]={'sessions':roster,'steps':80,'max_abs':err}
  if not smoke:
   raw=np.ascontiguousarray(streams[0][:40]);d=M2RiftCachedFalconDecoder(cfg,str(p),batch_size=1);d.reset([roster[0]]);expected=np.stack([d.predict(x[None])[0] for x in raw]);np.savez(a.dest/'artifacts/smoke_window.npz',tag_stem=np.asarray(roster[0]),window=raw,expected=expected);smoke=True
 # Partial evaluator batches leave inactive raw4/KV/history untouched; resume
 # the complete roster.  A genuine all-zero spike row remains valid evidence.
 roster=plans['mixed_partial'];streams=[trace(cache_dir(a.public_cache_root,k,q),k) for k in roster];d=M2RiftCachedFalconDecoder(cfg,str(p),batch_size=4);d.reset(roster)
 for t in range(40):d.predict(np.stack([x[t] for x in streams]))
 for t in range(20):d.predict(np.stack([x[40+t] for x in streams[:2]]))
 resumed=d.predict(np.stack([x[60] for x in streams]))
 m=RiftConcatDecoder('m2',context_bins=50,bias_mode='recency',seed=42);m.load_state_dict({k:torch.as_tensor(v) for k,v in q['ema_state_dict'].items()},strict=True);m.eval()
 valid=np.ones((61,4),dtype=np.bool_);valid[40:60,2:]=False
 expected_resume=ref_forward(m,[_task_bank(k,q['bank_by_dataset_tag'][k]) for k in roster],streams[:],valid)[60]
 resume_error=float(np.max(np.abs(resumed-expected_resume)))
 if resume_error>GATE:raise RuntimeError(f'inactive-row raw4/KV history resume gate {resume_error}')
 z=M2RiftCachedFalconDecoder(cfg,str(p),batch_size=1);z.reset([roster[0]]);zero=z.predict(np.zeros((1,96),dtype=np.float32))
 if not np.isfinite(resumed).all() or not np.isfinite(zero).all():raise RuntimeError('partial/all-zero valid probe nonfinite')
 out={'status':'HOST_REAL_PUBLIC_PARITY_PASS','arm':a.arm,'payload_sha256':sha(p),'parity':reports,'query_padding_bins':49,'partial_rows_raw4_kv_history_resume_checked':True,'inactive_resume_max_abs':resume_error,'all_zero_valid_probe':zero.tolist(),'external_actions':False}
 (a.dest/'artifacts/host_verify.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,indent=2,sort_keys=True))

def main()->int:
 x=argparse.ArgumentParser();x.add_argument('--stage',required=True,choices=('build','host'));x.add_argument('--arm',required=True,choices=ARMS);x.add_argument('--bank-cache-root',required=True,type=Path);x.add_argument('--run-dir',required=True,type=Path);x.add_argument('--selection-dir',required=True,type=Path);x.add_argument('--dest',required=True,type=Path);x.add_argument('--public-cache-root',type=Path);a=x.parse_args();os.environ.setdefault('CUDA_VISIBLE_DEVICES','');os.environ.setdefault('PYTHONNOUSERSITE','1');build(a) if a.stage=='build' else host(a);return 0
if __name__=='__main__':raise SystemExit(main())
