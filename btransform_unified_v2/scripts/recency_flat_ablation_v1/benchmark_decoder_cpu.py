#!/usr/bin/env python3
"""Selected-EMA, CPU-only end-to-end online RIFT decoder benchmark.

One task per process.  The timed operation is CpuRiftRuntime.advance(): static
selected-E0/T bank frontend (k5 + fusion), cached temporal attention, final
normalization and readout.  Checkpoint I/O, bank construction, M1 live-B3S
identity materialization and H1 NWB preparation are recorded separately.
"""
from __future__ import annotations
import argparse, copy, hashlib, importlib.util, json, math, os, platform, resource, statistics, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; WS=ROOT.parent; V1=WS/'btransform_unified_v1'
for p in (ROOT/'src',V1/'src',V1/'scripts',WS):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from btransform_unified_v1.bank import TaskBank
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v2.cpu_runtime import CpuRiftRuntime
from btransform_unified_v2.streaming import RiftStreamDecoder
from btransform_unified_v2.concat_model import RiftConcatDecoder
from btransform_unified_v2.model import RiftDecoder

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(x:Any)->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def assert_source_map(name:str,source:Any)->dict[str,str]:
 if not isinstance(source,dict) or not source:raise RuntimeError(f'{name}: missing source hash map')
 actual={}
 for raw,digest in source.items():
  p=Path(raw)
  if not p.is_file() or not isinstance(digest,str) or sha(p)!=digest:raise RuntimeError(f'{name}: source hash drift {p}')
  actual[str(p)]=digest
 return actual
def runtime_sources()->dict[str,str]:
 # Explicit paths avoid ambiguity from the deliberately task-specific imports.
 fixed=(Path(__file__),ROOT/'src/btransform_unified_v2/cpu_runtime.py',ROOT/'src/btransform_unified_v2/cpu_temporal.py',ROOT/'src/btransform_unified_v2/streaming.py',ROOT/'src/btransform_unified_v2/model.py',ROOT/'src/btransform_unified_v2/concat_model.py',ROOT/'src/btransform_unified_v2/temporal.py',ROOT/'src/btransform_unified_v2/config.py',V1/'src/btransform_unified_v1/model.py',V1/'src/btransform_unified_v1/identity_variant.py')
 return {str(p.resolve()):sha(p) for p in fixed}
def ah(x:Any)->str:
 a=np.ascontiguousarray(np.asarray(x));return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def js(p:Path)->dict[str,Any]:
 x=json.loads(p.read_text());
 if not isinstance(x,dict):raise RuntimeError(f'JSON object required: {p}')
 return x
def atom(p:Path,x:Any)->None:
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');t.replace(p)
def slopes(m:torch.nn.Module,mode:str)->dict[str,Any]:
 x=m.temporal.recency_slopes.detach().cpu()
 if x.shape!=(8,) or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()):raise RuntimeError('invalid recency_slopes')
 zero=int(torch.count_nonzero(x))==0;c=m.temporal.config
 expected=torch.tensor([0.0 if h is None or mode=='flat' else math.log(2.0)*c.bin_seconds/h for h in c.half_life_seconds],dtype=torch.float32)
 if mode=='flat' and not zero:raise RuntimeError('flat slopes must be exact bitwise zeros')
 if not torch.allclose(x,expected,rtol=0,atol=0):raise RuntimeError(f'actual slopes contradict declared {mode}')
 return {'shape':[8],'dtype':'float32','sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest(),'nonzero_count':int(torch.count_nonzero(x)),'all_zero':zero}
def summary(x:list[float])->dict[str,Any]:
 a=np.asarray(x,dtype=np.float64)*1e3
 return {'n':len(a),'samples_ms':a.tolist(),'mean_ms':float(a.mean()),'median_ms':float(np.median(a)),'p95_ms':float(np.percentile(a,95))}
def rss()->dict[str,int]:
 q={r.split(':')[0]:int(r.split()[1])*1024 for r in Path('/proc/self/status').read_text().splitlines() if r.startswith(('VmRSS:','VmHWM:'))}
 return {'rss_bytes':q.get('VmRSS',0),'vmhwm_bytes':max(q.get('VmHWM',0),resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)}
def footprint(m:torch.nn.Module)->dict[str,int]:
 p=sum(x.numel() for x in m.parameters());b=sum(x.numel() for x in m.buffers())
 return {'trainable_parameter_elements':p,'buffer_elements':b,'state_dict_elements':sum(x.numel() for x in m.state_dict().values()),'parameter_bytes':sum(x.numel()*x.element_size() for x in m.parameters()),'buffer_bytes':sum(x.numel()*x.element_size() for x in m.buffers())}
def cached_storage_bytes(m:torch.nn.Module,banks:list[TaskBank])->dict[str,int]:
 rt=CpuRiftRuntime(m,banks,[f'alloc{i}' for i in range(len(banks))],temporal_backend='cached');seen={}
 for group in (rt.cached_temporal.keys,rt.cached_temporal.values,rt.cached_temporal._scratch_k,rt.cached_temporal._scratch_v,rt.cached_temporal._scratch_valid,rt.cached_temporal.lengths,rt.cached_temporal._ages,rt.cached_temporal._positions,(rt.raw4,rt.e0,rt.carrier,rt.keep)):
  for x in group:seen[(str(x.device),x.untyped_storage().data_ptr())]=x.untyped_storage().nbytes()
 return {'unique_preallocated_runtime_storage_bytes':sum(seen.values()),'unique_storage_count':len(seen)}
def load_private(name:str,path:Path,add:Path|None=None):
 if add and str(add) not in sys.path:sys.path.insert(0,str(add))
 s=importlib.util.spec_from_file_location(name,path)
 if not s or not s.loader:raise RuntimeError(f'cannot load {path}')
 m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
@dataclass
class Spec:
 task:str;mode:str;model:torch.nn.Module;banks:list[TaskBank];raw:np.ndarray;valid:np.ndarray;context:int;meta:dict[str,Any];offline:dict[str,Any]

def validate_curve(score:dict[str,Any],epochs:int,metric:str)->int:
 if score.get('status')!='COMPLETED':raise RuntimeError('selected score is not completed')
 curve=score.get('ema_by_epoch',{})
 if set(curve)!={str(i) for i in range(1,epochs+1)}:raise RuntimeError('incomplete selected EMA curve')
 if any(not math.isfinite(float(curve[str(e)].get(metric,float('nan')))) for e in range(1,epochs+1)):raise RuntimeError('nonfinite score curve')
 best=min(range(1,epochs+1),key=lambda e:(-float(curve[str(e)][metric]),e))
 sel=score.get('selection',{})
 if int(sel.get('epoch',-1))!=best:raise RuntimeError('selection is not earliest maximum of complete curve')
 return best

def contiguous_rows(source:np.ndarray,start:int,n:int,width:int,label:str)->np.ndarray:
 x=np.asarray(source[start:start+n],np.float32)
 if x.shape!=(n,width):raise RuntimeError(f'{label}: need one contiguous [{n},{width}] raw segment at {start}, got {x.shape}')
 return np.ascontiguousarray(x)

def h1_selected(run:Path)->tuple[dict[str,Any],dict[str,Any],int,dict[str,Any]]:
 """Validate both the old pinned recency receipt and the flat binding format."""
 meta,receipt,selection=js(run/'run_meta.json'),js(run/'train_receipt.json'),js(run/'ho_m3_selection.json')
 if receipt.get('status')!='COMPLETED' or int(receipt.get('epochs',-1))!=32 or int(receipt.get('updates',-1))!=23392 or selection.get('status')!='HO_M3_DEVELOPMENT_SELECTION':raise RuntimeError('H1 receipt/selection is not completed signed-state14 evidence')
 curve=selection.get('curve')
 if not isinstance(curve,list) or len(curve)!=32:raise RuntimeError('H1 selection curve is not complete all32')
 rows=[]
 for e,row in enumerate(curve,1):
  if not isinstance(row,dict) or int(row.get('epoch',-1))!=e or int(row.get('epoch_zero_based',-2))!=e-1:raise RuntimeError('H1 curve epoch map drift')
  mean=float(row.get('val_ho_m3_grouped/r2_mean',float('nan')));per=row.get('per_session_r2')
  if not math.isfinite(mean) or not isinstance(per,dict) or len(per)!=7 or any(not math.isfinite(float(v)) for v in per.values()):raise RuntimeError('H1 curve lacks finite seven-session values')
  rows.append(mean)
 epoch=min(range(1,33),key=lambda e:(-rows[e-1],e));chosen=curve[epoch-1]
 if int(receipt.get('selected_epoch',-1))!=epoch or selection.get('selected')!=chosen:raise RuntimeError('H1 selected epoch is not earliest maximum selected row')
 return meta,receipt,epoch,chosen

def m1(args)->Spec:
 fullflat=load_private('_m1_full_flat',HERE/'m1_full_flat_train.py',HERE);legacy=fullflat.frozen
 run=args.run.resolve();meta=js(run/'run_meta.json');score=js(run/'score_receipt.json');train=js(run/'train_receipt.json')
 source_actual=assert_source_map('M1 run_meta',meta.get('source_hashes'))
 if train.get('source_hashes')!=meta['source_hashes'] or score.get('source_hashes')!=meta['source_hashes'] or train.get('source_contract')!=meta.get('source_contract') or score.get('source_contract')!=meta.get('source_contract'):raise RuntimeError('M1 metadata/train/score source contract drift')
 mode=str(meta.get('variant')); epoch=validate_curve(score,24,'equal_session_mean_channel_variance_weighted_r2')
 for report in score['ema_by_epoch'].values():legacy._validate_scored_report(report)
 ck=run/f'epoch_{epoch:03d}.pt'; expected=score['checkpoint_sha256_by_epoch'][str(epoch)]
 if sha(ck)!=expected or train.get('status')!='COMPLETED' or int(train.get('steps',-1))!=24*6665:raise RuntimeError('M1 selected checkpoint/train binding drift')
 state=torch.load(ck,map_location='cpu',weights_only=False)
 if state.get('raw_state_dict') is None or state.get('ema') is None or int(state['ema'].get('n_updates',-1))!=epoch*6665:raise RuntimeError('M1 checkpoint lacks selected EMA update count')
 if mode=='flat':live=fullflat._decoder(torch.device('cpu'),meta['arm'],42).eval()
 else:live=legacy._decoder(torch.device('cpu'),meta['arm'],42).eval()
 live.load_state_dict(state['raw_state_dict'],strict=True);e=DecoderEMA(live,decay=.9995);e.load_state_dict(state['ema']);e.apply_to(live);sl=slopes(live,mode)
 # The existing runner builds the representative HO3 M10/banks.  Identity
 # materialization and raw-stream preparation are separately reported offline.
 t=time.perf_counter();carriers,binding=fullflat._carrier_binding(args.carrier_pack);input_binding=binding if mode=='flat' else legacy._carrier_binding(args.carrier_pack)[1];mat=legacy._ho_material(carriers);live.install_session_memory({s:mat[s]['bank'] for s in legacy.HO},{s:mat[s]['calib10'] for s in legacy.HO})
 if input_binding.get('carrier_pack_npz_sha256') and sha(args.carrier_pack)!=input_binding['carrier_pack_npz_sha256']:raise RuntimeError('M1 actual carrier pack SHA drift')
 recorded=meta.get('carrier_binding',{})
 factual=('carrier_variant','carrier_pack_npz_sha256','carrier_pack_receipt_sha256','fit_sha256','carrier_pack_receipt_body')
 if input_binding.get('fit_sha256')!=meta.get('fit_sha256') or any(input_binding.get(k)!=recorded.get(k) for k in factual):raise RuntimeError('M1 actual carrier input/fit binding drift')
 from btransform_unified_v1.m1_b3s_joint import encode_b3s
 mats=[];live.eval();identity_start=time.perf_counter()
 with torch.inference_mode():
  for s in legacy.HO:
   b=mat[s]['bank'];raw=torch.from_numpy(np.ascontiguousarray(mat[s]['calib10'],np.float32));direct=np.asarray(b.carrier,np.float32) if meta['arm']=='D_JOINT' else np.zeros_like(b.carrier,np.float32)
   e0=encode_b3s(live.encoder,raw,torch.from_numpy(direct)).cpu().numpy();mats.append(TaskBank(b.session_id,np.ascontiguousarray(e0,np.float32),np.ascontiguousarray(direct,np.float32),b.unit_mask,b.X_store,b.target_store,b.window_ids,{**b.calibration_meta,'array_sha256':ah(e0),'carrier_sha256':ah(direct)}))
 identity_seconds=time.perf_counter()-identity_start;data_bank_identity_total_seconds=time.perf_counter()-t
 static=RiftConcatDecoder('m1',context_bins=100,bias_mode=mode,seed=42).eval();ls=live.state_dict();ss=static.state_dict();static.load_state_dict({k:ls[k] for k in ss},strict=True)
 if any(not torch.equal(static.state_dict()[k].cpu(),ls[k].cpu()) for k in ss):raise RuntimeError('M1 static EMA state mismatch')
 # Verify actual live B3S identity versus the materialized static E0/T for all
 # three held-out sessions before timing either implementation.
 live_checks=[]
 with torch.inference_mode():
  for s,b in zip(legacy.HO,mats):
   x=torch.from_numpy(contiguous_rows(mat[s]['dataset'].neural_data[s],99,100,64,f'M1 live parity {s}')[None].copy())
   d=float((live(x,[mat[s]['bank']])-static(x,[b])).abs().max())
   if not math.isfinite(d) or d>2e-5:raise RuntimeError(f'M1 live/static identity parity {s}: {d}')
   # The actual dataset's first query is left-padded; use its own valid mask.
   first=torch.from_numpy(np.ascontiguousarray(mat[s]['dataset'][0][0],np.float32)[None]);mask=legacy.valid_mask_from_padded_starts((mat[s]['starts'][0],),device=torch.device('cpu'))
   q=float((live(first,[mat[s]['bank']],input_valid_mask=mask)-static(first,[b],input_valid_mask=mask)).abs().max())
   if not math.isfinite(q) or q>2e-5:raise RuntimeError(f'M1 live/static padded-first parity {s}: {q}')
   live_checks.append({'session':s,'context_bins':100,'full_valid_max_abs_error':d,'first_query_valid_mask_sha256':ah(mask.numpy()),'first_query_padded_max_abs_error':q})
 # True contiguous online streams.  Ensure raw length covers both full-window
 # endpoints and all requested warmup/timed advances.
 t=time.perf_counter();n=max(2*100+1,args.warmup+args.timed);rows=[];banks=[];starts=[]
 for i in range(8):
  s=legacy.HO[i%3];d=mat[s]['dataset'];off=(i//3)*n;x=contiguous_rows(d.neural_data[s],99+off,n,64,f'M1 {s}')
  rows.append(x);banks.append(mats[list(legacy.HO).index(s)]);starts.append({'session':s,'start':99+off})
 raw=np.stack(rows,1);valid=np.ones(raw.shape[:2],bool);raw_seconds=time.perf_counter()-t
 contract=legacy._ho_contract(mat)
 if score.get('ho_contract')!=contract:raise RuntimeError('M1 actual held-out bank/fit contract contradicts selected score')
 offline={'kind':'M1 selected-EMA live B3S M10 identity materialization and contiguous raw preparation; excluded from online cached advance','data_bank_and_identity_total_seconds':data_bank_identity_total_seconds,'identity_only_seconds':identity_seconds,'raw_preparation_seconds':raw_seconds,'carrier_binding_sha256':input_binding['binding_sha256'],'fit_sha256':input_binding.get('fit_sha256'),'ho_contract':contract,'live_static_identity_parity':{'passed':True,'per_session':live_checks},'banks':[{'session':b.session_id,'e0_sha256':ah(b.E0),'carrier_sha256':ah(b.carrier)} for b in mats]}
 return Spec('m1',mode,static,banks,raw,valid,100,{'run':str(run),'selected_epoch':epoch,'checkpoint':str(ck),'checkpoint_sha256':expected,'score_receipt_sha256':sha(run/'score_receipt.json'),'source_hashes':source_actual,'score_status':score['status'],'selected_ema_update_count':int(state['ema']['n_updates']),'starts':starts,'raw_sha256':ah(raw),'slopes':sl,'live_joint_footprint':footprint(live),'static_online_footprint':footprint(static)},offline)

def m2(args)->Spec:
 t=time.perf_counter();sel=args.run.resolve();manifest=js(sel/'manifest.json');score=js(sel/'score_receipt.json');pkg_path=sel/'selected_ema.pt';run=Path(manifest['run']);meta=js(run/'run_meta.json');epoch=validate_curve(score,24,'equal_session_mean')
 source_actual=assert_source_map('M2 run_meta',meta.get('source_hashes'))
 if canonical(manifest)!=score.get('manifest_sha256') or epoch!=int(score['selection']['epoch']) or sha(pkg_path)!=score['selected_ema_state']['sha256']:raise RuntimeError('M2 selected EMA package/manifest binding drift')
 ck=Path(manifest['checkpoint_bytes'][str(epoch)]['path']);cksha=sha(ck);row=score['ema_by_epoch'][str(epoch)]
 if cksha!=manifest['checkpoint_bytes'][str(epoch)]['sha256'] or cksha!=row.get('checkpoint_sha256'):raise RuntimeError('M2 selected checkpoint manifest/score SHA drift')
 st=torch.load(ck,map_location='cpu',weights_only=False);pkg=torch.load(pkg_path,map_location='cpu',weights_only=True)
 if int(st.get('ema',{}).get('n_updates',-1))!=epoch*3165:raise RuntimeError('M2 selected EMA update count drift')
 mode=str(meta.get('variant','recency'));model=RiftConcatDecoder('m2',context_bins=50,bias_mode=mode,seed=int(meta['seed'])).eval();model.load_state_dict(pkg,strict=True)
 rawm=RiftConcatDecoder('m2',context_bins=50,bias_mode=mode,seed=int(meta['seed'])).eval();rawm.load_state_dict(st['raw_state_dict'],strict=True);e=DecoderEMA(rawm,decay=.9995);e.load_state_dict(st['ema']);e.apply_to(rawm)
 if set(pkg)!=set(rawm.state_dict()) or any(not torch.equal(pkg[k].cpu(),rawm.state_dict()[k].cpu()) for k in pkg):raise RuntimeError('M2 EMA package/raw+EMA mismatch')
 bmod=load_private('_m2_cpu',ROOT/'scripts/rift_v1/benchmark_cpu_m2.py',ROOT/'scripts/rift_v1');frozen=load_private('_m2_frozen',ROOT/'scripts/rift_v1/m2_ext6_epoch_pick.py',ROOT/'scripts/rift_v1');dual,bmap=bmod.base._load_surface('ext4',torch.device('cpu'))
 actual_cache=bmod.base._cache_hashes('ext4',dual,bmap)
 if actual_cache!=meta.get('frozen_cache_hashes',{}).get('ext4'):raise RuntimeError('M2 actual ext4 frozen cache drift')
 query_cache=Path(manifest['query_cache']);actual_query=frozen.query_asset_hashes(query_cache)
 if actual_query!=manifest.get('query_assets'):raise RuntimeError('M2 actual query assets drift')
 frozen.validate_complete_curve(score['ema_by_epoch'],manifest['query_assets']['sessions'])
 sessions=list(bmod.plan.EXT4_SESSIONS);n=max(2*50+1,args.warmup+args.timed);streams=[];rows=[]
 for j in range(8):
  s=sessions[j%4];start=int(np.asarray(dual[s].eligible_starts)[0 if j<4 else 1]);x=contiguous_rows(dual[s].X_store,start,n,96,f'M2 {s}');streams.append((s,start));rows.append(x)
 raw=np.stack(rows,1);banks=[bmap[s] for s,_start in streams];prep=time.perf_counter()-t
 cache=meta.get('cache_hashes',meta.get('frozen_cache_hashes',{}));query=manifest.get('query_assets',{})
 return Spec('m2',mode,model,banks,raw,np.ones(raw.shape[:2],bool),50,{'selection_dir':str(sel),'run':str(run),'selected_epoch':epoch,'checkpoint':str(ck),'checkpoint_sha256':cksha,'selected_ema_sha256':sha(pkg_path),'score_receipt_sha256':sha(sel/'score_receipt.json'),'score_receipt_manifest_sha256':score.get('manifest_sha256'),'manifest_sha256':sha(sel/'manifest.json'),'source_hashes':source_actual,'cache_hashes':actual_cache,'query_assets':actual_query,'score_status':score['status'],'selected_ema_update_count':int(st['ema']['n_updates']),'raw_sha256':ah(raw),'starts':[{'session':s,'start':int(q)} for s,q in streams],'slopes':slopes(model,mode),'asset_surface':'ext4 contiguous raw stream; ext6 is the all-24 selection surface'}, {'kind':'M2 offline selected-checkpoint load, selected-EMA package load, ext4 bank/data load and contiguous raw preparation; excluded from online cached advance','seconds':prep,'selected_checkpoint_manifest_sha256':manifest['checkpoint_bytes'][str(epoch)]['sha256'],'ext4_cache_metadata':actual_cache,'query_asset_receipt_sha256':actual_query.get('receipt_sha256')})

def h1(args)->Spec:
 # H1 uses a true contiguous neural stream.  Creating it from NWB is an explicit offline data-prep phase.
 flat=args.run.resolve();meta,receipt,epoch,selected=h1_selected(flat);mode=str(meta['variant']);ck=flat/f'epoch_{epoch:03d}.pt'
 source_actual=assert_source_map('H1 run_meta',meta.get('source_manifest_sha256'))
 # Historical pinned recency evidence predates per-checkpoint binding JSON;
 # use its selected curve row SHA and completed receipt.  Flat has the newer
 # per-checkpoint binding, which is additionally required when present.
 pinned_legacy={'recency_s42_formal_20260909':{16:'4a25f1bf6ac31c60302a4bc60d37538ef334799c70abe73a710323ff3918f8ba'}}
 binding_path=ck.with_suffix('.pt.binding.json');binding=js(binding_path) if binding_path.is_file() else None
 if mode=='flat':
  if binding is None:raise RuntimeError('H1 flat selected checkpoint lacks binding')
  expected_sha=binding.get('checkpoint_sha256')
  if binding.get('run_meta_sha256')!=sha(flat/'run_meta.json'):raise RuntimeError('H1 flat checkpoint run-meta binding drift')
 else:expected_sha=pinned_legacy.get(flat.name,{}).get(epoch)
 if not isinstance(expected_sha,str) or sha(ck)!=expected_sha:raise RuntimeError('H1 selected curve/pinned checkpoint SHA drift')
 # The signed-state H1 loader requires SPINT-main's ``src`` package ahead of
 # this workspace's package.  Reuse its sealed bootstrap before importing the
 # frozen runner.  The benchmark's earlier imports can cache this workspace's
 # top-level ``src`` package, so path order alone would still be ineffective.
 sys.modules.pop('common',None)
 for name in tuple(sys.modules):
  if name=='src' or name.startswith('src.'):sys.modules.pop(name,None)
 h1_common=load_private('common',ROOT/'scripts/h1_signed_state_r300_v1/common.py',ROOT/'scripts/h1_signed_state_r300_v1')
 h1_common.setup_imports()
 current=load_private('_h1_current',ROOT/'scripts/h1_signed_state_r300_v1/train.py',ROOT/'scripts/h1_signed_state_r300_v1')
 model=current.ht._decoder(mode,torch.device('cpu'),300,'dense').eval();st=torch.load(ck,map_location='cpu',weights_only=False)
 if int(st.get('ema',{}).get('n_updates',-1))!=epoch*731:raise RuntimeError('H1 selected EMA update count drift')
 model.load_state_dict(st['raw_state_dict'],strict=True);e=current.DecoderEMA(model,decay=.9995);e.load_state_dict(st['ema']);e.apply_to(model)
 t=time.perf_counter();_plan,_receipt,carriers=current._load_banks(args.h1_banks);ho=current.build_ho_signed(300,carriers)
 # build_ho_signed owns the matching static signed-state bank.  Load one NWB once for contiguous raw bins.
 from falcon_challenge.config import FalconTask
 from falcon_challenge.dataloaders import load_nwb
 key=ho['keys'][0];session=ho['banks'][key].session_id;path=current.b2.HO_DIR/f'sub-HumanPitt-held-out-calib_{session}.nwb';neural,_v,_c,_mask=load_nwb(path,FalconTask.h1);start=300;n=max(2*300+1,args.warmup+args.timed);raw=contiguous_rows(neural,start,n,176,f'H1 {session}')
 raw=np.stack([raw]*8,1);banks=[ho['banks'][key] for _ in range(8)]
 return Spec('h1',mode,model,banks,raw,np.ones(raw.shape[:2],bool),300,{'run':str(flat),'selected_epoch':epoch,'checkpoint':str(ck),'checkpoint_sha256':sha(ck),'selection_row_checkpoint_sha256':expected_sha,'selection_sha256':sha(flat/'ho_m3_selection.json'),'train_receipt_sha256':sha(flat/'train_receipt.json'),'checkpoint_binding_sha256':sha(binding_path) if binding is not None else None,'source_manifest_sha256':source_actual,'recipe_binding':meta.get('recipe_binding',meta.get('pairing_digests')),'score_status':'HO_M3_DEVELOPMENT_SELECTION','selected_ema_update_count':int(st['ema']['n_updates']),'raw_sha256':ah(raw),'starts':[{'session':session,'start':start}],'slopes':slopes(model,mode),'training_attention_backend':'dense','online_cached_backend':'finite local KV cache'}, {'kind':'H1 signed-state bank construction plus one NWB contiguous raw stream; data/bank preparation excluded from online cached advance','seconds':time.perf_counter()-t,'banks_receipt_sha256':sha(args.h1_banks/'receipt.json')})

def parity(s:Spec,b:int)->dict[str,Any]:
 raw=s.raw[:,:b];banks=s.banks[:b];ids=[f'i{i}' for i in range(b)];rt=CpuRiftRuntime(s.model,banks,ids,temporal_backend='cached');ref=RiftStreamDecoder(s.model);mx=0.;full=[]
 with torch.inference_mode():
  for t in range(len(raw)):
   x=torch.from_numpy(raw[t].copy());a=rt.advance(x);want=ref.stream_step(x,banks,ids)
   if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(want).all()):raise RuntimeError('nonfinite cached/reference output')
   mx=max(mx,float((a-want).abs().max()))
   if t in (s.context-1,2*s.context-1,len(raw)-1):
    y=s.model(torch.from_numpy(raw[t-s.context+1:t+1].transpose(1,0,2).copy()),banks,input_valid_mask=torch.ones(b,s.context,dtype=torch.bool))
    if not bool(torch.isfinite(y).all()):raise RuntimeError('nonfinite full-window output')
    d=float((a-y).abs().max());full.append({'end_bin':t,'valid_tokens':s.context,'max_abs_error':d});mx=max(mx,d)
  order=ids[::-1];x=torch.from_numpy(raw[-1][::-1].copy());a=rt.advance(x,order);want=ref.stream_step(x,list(reversed(banks)),order)
  if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(want).all()):raise RuntimeError('nonfinite reorder output')
  mx=max(mx,float((a-want).abs().max()));rt.reset_rows([order[-1]]);ref.reset(order[-1]);a=rt.advance(x);want=ref.stream_step(x,list(reversed(banks)),order)
  if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(want).all()):raise RuntimeError('nonfinite reset output')
  mx=max(mx,float((a-want).abs().max()))
 if not np.isfinite(mx) or mx>2e-5:raise RuntimeError(f'cached/reference/full parity failed: {mx}')
 return {'passed':True,'max_abs_error':mx,'full_window_endpoints':full,'reorder_and_reset':True}

def measure_round(s:Spec,b:int,warm:int,timed:int,round_index:int)->dict[str,Any]:
 raw=s.raw[:,:b];banks=s.banks[:b];observed=[torch.from_numpy(x.copy()) for x in raw[:warm+timed]]
 if len(observed)!=warm+timed:raise RuntimeError('timed input list length drift')
 ids=[f'i{i}' for i in range(b)];t=time.perf_counter();rt=CpuRiftRuntime(s.model,banks,ids,temporal_backend='cached');construct=time.perf_counter()-t;t=time.perf_counter();out=rt.advance(observed[0]);first=time.perf_counter()-t
 if not bool(torch.isfinite(out).all()):raise RuntimeError('nonfinite first advance output')
 with torch.inference_mode():
  for x in observed[1:warm]:
   out=rt.advance(x)
   if not bool(torch.isfinite(out).all()):raise RuntimeError('nonfinite warmup output')
  vals=[]
  for x in observed[warm:warm+timed]:
   t=time.perf_counter();out=rt.advance(x);vals.append(time.perf_counter()-t)
   if not bool(torch.isfinite(out).all()):raise RuntimeError('nonfinite timed output')
 return {'round':round_index,'construct_register_seconds':construct,'startup_first_advance_seconds':first,'input_tensor_preconstruction':'excluded; observed CPU tensors allocated before runtime construction','steady':summary(vals),**rss()}
def measure_pair(specs:list[Spec],b:int,warm:int,timed:int,rounds:int)->dict[str,Any]:
 out={s.mode:[] for s in specs}
 # Five alternating rounds reduce one-sided thermal/cache ordering; each mode
 # still receives a fresh registered runtime on every round.
 for r in range(rounds):
  order=specs if r%2==0 else list(reversed(specs))
  for s in order:out[s.mode].append(measure_round(s,b,warm,timed,r+1))
 return {mode:{'per_round':rows,'steady':summary([v/1e3 for row in rows for v in row['steady']['samples_ms']]),**rss()} for mode,rows in out.items()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--task',choices=('m1','m2','h1'),required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--paired-run',type=Path,required=True);p.add_argument('--dest',type=Path,required=True);p.add_argument('--carrier-pack',type=Path,default=ROOT/'results/m1_muscle_r100_v1/carrier_official4/carrier_pack.npz');p.add_argument('--h1-banks',type=Path,default=ROOT/'results/h1_signed_state_r300_v1/banks_official13_20260909');p.add_argument('--warmup',type=int,default=300);p.add_argument('--timed',type=int,default=128);p.add_argument('--rounds',type=int,default=5);p.add_argument('--cpu-threads',type=int,default=2);p.add_argument('--affinity',default='14,15');a=p.parse_args()
 if a.dest.exists():raise FileExistsError('destination must be fresh')
 if a.cpu_threads!=2 or a.warmup<1 or a.timed<128 or a.rounds<5:raise ValueError('require threads=2, timed>=128, rounds>=5')
 os.sched_setaffinity(0,{int(x) for x in a.affinity.split(',')});torch.set_num_threads(2);torch.set_num_interop_threads(1)
 runtime_before=runtime_sources()
 loader={'m1':m1,'m2':m2,'h1':h1}[a.task];s=loader(a);other=copy.copy(a);other.run=a.paired_run;paired=loader(other);specs=sorted([s,paired],key=lambda x:0 if x.mode=='recency' else 1)
 if {s.mode,paired.mode}!={'recency','flat'}:raise RuntimeError('paired runs must provide exactly recency and flat')
 if a.warmup<s.context:raise ValueError(f'warmup must be >= context ({s.context})')
 if s.context!=paired.context or ah(s.raw)!=ah(paired.raw) or s.meta.get('starts')!=paired.meta.get('starts') or not np.array_equal(s.valid,paired.valid):raise RuntimeError('paired modes do not bind identical raw stream/start/valid inputs')
 if len(s.raw)<a.warmup+a.timed:raise RuntimeError('representative stream too short')
 cases={f'B{b}':{'raw_sha256':ah(s.raw[:,:b]),'model_footprint_by_mode':{x.mode:footprint(x.model) for x in specs},'cached_preallocated_storage_by_mode':{x.mode:cached_storage_bytes(x.model,x.banks[:b]) for x in specs},'parity':{x.mode:parity(x,b) for x in specs},'cached_online':measure_pair(specs,b,a.warmup,a.timed,a.rounds)} for b in (1,4,8)}
 env={'cpu':next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),platform.processor()),'affinity':sorted(os.sched_getaffinity(0)),'torch':torch.__version__,'device':'cpu','dtype':'float32','inference_mode':True,'threads':2,'interop_threads':1}
 runtime_after=runtime_sources()
 if runtime_after!=runtime_before:raise RuntimeError('benchmark runtime import/source changed during measurement')
 rep={'schema':'selected_ema_decoder_cpu_cached_v1','task':s.task,'scope':'CpuRiftRuntime.advance only: static selected bank frontend k5/fusion + cached temporal + final norm/readout; excludes checkpoint load, calibration/bank building and input preparation','selected_by_mode':{x.mode:x.meta for x in specs},'offline_excluded_by_mode':{x.mode:x.offline for x in specs},'benchmark_runtime_source_hashes_before_after':{'before':runtime_before,'after':runtime_after,'unchanged':True},'protocol':{'batches':[1,4,8],'rounds':a.rounds,'mode_order':'recency-flat on rounds 1,3,5; flat-recency on rounds 2,4','warmup_actual_advances':a.warmup,'timed_actual_advances_per_round':a.timed,'fresh_runtime_each_round':True,'cached_backend':'CpuRiftRuntime temporal_backend=cached'},'environment':env,'cases':cases,'script_sha256':sha(Path(__file__))};atom(a.dest/'benchmark.json',rep);print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
