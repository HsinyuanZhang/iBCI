#!/usr/bin/env python3
"""Receipt-bound CPU parity and runtime benchmark for a formal M1 joint B/D run.

The benchmark uses HO3's real M10 activity and query streams.  It is a new
joint-model surface; it does not claim that the frozen-concat runtime applies.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, platform, resource, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]; WS=ROOT.parent; V1=WS/'btransform_unified_v1'
for p in (ROOT/'src',V1/'src',V1/'scripts',WS,ROOT/'scripts/rift_v1'):
 if str(p) not in sys.path: sys.path.insert(0,str(p))
from btransform_unified_v1.bank import TaskBank
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.m1_b3s_joint import encode_b3s
from btransform_unified_v2.concat_model import RiftConcatDecoder
from btransform_unified_v2.cpu_runtime import CpuRiftRuntime
from btransform_unified_v2.streaming import RiftStreamDecoder
from btransform_unified_v2.joint_m1_model import JointM1ConcatDecoder, ARM_B, ARM_D
import m1_joint_train as train

HO=("20121004","20121017","20121024"); COUNTS={"20121004":1305,"20121017":1295,"20121024":1281}; EPOCHS={str(i) for i in range(1,25)}; T=401
def sha(p):
 p=Path(p)
 if not p.is_file(): raise FileNotFoundError(p)
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def ft(label,x):
 if not bool(torch.isfinite(x).all()):raise RuntimeError(f'{label}: non-finite')
def ff(label,x):
 if not math.isfinite(float(x)):raise RuntimeError(f'{label}: non-finite')
def audit_sources():
 out={}
 for mod in tuple(sys.modules.values()):
  spec=getattr(mod,'__spec__',None); origin=getattr(spec,'origin',None); text=origin if isinstance(origin,str) and Path(origin).is_absolute() else getattr(mod,'__file__',None)
  if not isinstance(text,str) or not Path(text).is_absolute():continue
  p=Path(text).resolve()
  if p.is_file() and p.suffix=='.py' and (p.is_relative_to(ROOT) or p.is_relative_to(WS)):out[str(p)]=sha(p)
 required=(Path(__file__).resolve(),ROOT/'scripts/rift_v1/m1_joint_train.py',ROOT/'src/btransform_unified_v2/joint_m1_model.py',ROOT/'src/btransform_unified_v2/concat_model.py',ROOT/'src/btransform_unified_v2/cpu_runtime.py',ROOT/'src/btransform_unified_v2/streaming.py')
 if any(str(p.resolve()) not in out for p in required):raise RuntimeError('runtime source audit lacks a required actual import')
 return dict(sorted(out.items()))
def stable_sources(before,after):
 common=set(before)&set(after)
 if not common or any(before[key]!=after[key] for key in common):raise RuntimeError('already-imported runtime source hash drift')
 return {'common_imports_checked':len(common),'dynamic_new_imports_allowed':len(set(after)-set(before))}
def cpu():
 for row in Path('/proc/cpuinfo').read_text().splitlines():
  if row.startswith('model name'):return row.split(':',1)[1].strip()
 return platform.processor()
def rss():
 d={r.split(':')[0]:int(r.split()[1])*1024 for r in Path('/proc/self/status').read_text().splitlines() if r.startswith(('VmRSS:','VmHWM:'))}
 return {'rss_bytes':d.get('VmRSS',0),'vmhwm_bytes':max(d.get('VmHWM',0),resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)}
def summary(v):
 a=np.asarray(v)*1e3
 return {'n':len(a),'median_ms':float(np.median(a)),'p95_ms':float(np.percentile(a,95))}

def load(run):
 meta=json.loads((run/'run_meta.json').read_text()); tr=json.loads((run/'train_receipt.json').read_text()); sc=json.loads((run/'score_receipt.json').read_text())
 if any(not Path(p).is_file() or sha(p)!=h for p,h in meta.get('source_hashes',{}).items()):raise RuntimeError('formal joint source SHA drift')
 if not(meta.get('schema')=='m1_rift_joint_train_v1' and meta.get('status')=='FORMAL' and meta.get('cell')==train.CELL and meta.get('task')=='m1' and meta.get('arm') in (ARM_B,ARM_D) and int(meta.get('seed',-1))==42 and meta.get('source_train_only_for_gradients') is True and meta.get('official_test_used') is False and meta.get('epochs')==24 and meta.get('updates_per_epoch')==6665 and meta.get('total_updates')==159960 and isinstance(meta.get('source_contract'),dict) and isinstance(meta.get('source_hashes'),dict) and bool(meta['source_hashes']) and tr.get('schema')=='m1_rift_joint_train_receipt_v1' and tr.get('status')=='COMPLETED' and tr.get('cell')==meta['cell'] and tr.get('arm')==meta['arm'] and int(tr.get('seed',-1))==42 and int(tr.get('epochs',-1))==24 and int(tr.get('steps',-1))==159960 and tr.get('source_hashes')==meta['source_hashes'] and tr.get('source_contract')==meta['source_contract'] and sc.get('schema')=='m1_rift_joint_ho_calib_epoch_scan_v1' and sc.get('status')=='COMPLETED' and sc.get('cell')==meta['cell'] and sc.get('arm')==meta['arm'] and int(sc.get('seed',-1))==42 and sc.get('official_test_used') is False and sc.get('source_hashes')==meta['source_hashes'] and sc.get('source_contract')==meta['source_contract']):raise RuntimeError('requires completed formal M1 joint receipt triplet')
 if set(sc.get('ema_by_epoch',{}))!=EPOCHS or set(sc.get('checkpoint_sha256_by_epoch',{}))!=EPOCHS:raise RuntimeError('M1 joint receipt lacks complete 24 epoch EMA/checkpoint map')
 means={}
 for e in sorted(EPOCHS,key=int):
  row=sc['ema_by_epoch'][e]; per=row.get('per_session',{})
  if row.get('partial') is not False or row.get('n_windows')!=3881 or set(per)!=set(HO):raise RuntimeError(f'e{e}: incomplete HO3 score')
  v=[]
  for s,n in COUNTS.items():
   q=per[s]
   if q.get('window_count')!=n:raise RuntimeError(f'e{e}/{s}: window count drift')
   ff(f'e{e}/{s}',q.get('r2',float('nan')));v.append(float(q['r2']))
  means[int(e)]=sum(v)/3;ff(f'e{e} mean',row.get('equal_session_mean',float('nan')))
  if abs(means[int(e)]-float(row['equal_session_mean']))>1e-12:raise RuntimeError(f'e{e}: mean drift')
  ck=run/f'epoch_{int(e):03d}.pt'
  if not ck.is_file() or sha(ck)!=sc['checkpoint_sha256_by_epoch'][e]:raise RuntimeError(f'e{e}: checkpoint SHA drift')
 epoch=min(means,key=lambda e:(-means[e],e))
 if sc.get('selection')!={'epoch':epoch,'rule':'earliest maximum equal-session mean EMA'}:raise RuntimeError('selected EMA rule/epoch is not exact')
 ck=run/f'epoch_{epoch:03d}.pt'; st=train._validate_checkpoint(ck,meta,expected_epoch=epoch)
 raw,ema=st.get('raw_state_dict'),st.get('ema')
 if not isinstance(raw,dict) or not raw or not isinstance(ema,dict) or ema.get('n_updates')!=epoch*6665 or not isinstance(ema.get('shadow'),dict) or set(ema['shadow'])-set(raw):raise RuntimeError('selected checkpoint EMA/raw contract failed')
 for label,table in (('raw',raw),('EMA',ema['shadow'])):
  if any(not torch.is_tensor(v) or not bool(torch.isfinite(v).all()) for v in table.values()):raise RuntimeError(f'selected checkpoint {label} tensor non-finite')
 live=JointM1ConcatDecoder(meta['arm'],seed=42).eval(); live.load_state_dict(st['raw_state_dict']); ema=DecoderEMA(live,decay=.9995);ema.load_state_dict(st['ema']);ema.apply_to(live)
 return meta,sc,ck,epoch,live

def materialize(live,meta):
 material=train._ho_material(); banks={s:material[s]['bank'] for s in HO}; cal={s:material[s]['calib10'] for s in HO}; live.install_session_memory(banks,cal);live.eval()
 static=RiftConcatDecoder('m1',context_bins=100,bias_mode='recency',seed=42).eval(); ls=live.state_dict(); ss=static.state_dict()
 for key in ss:
  if key not in ls or ls[key].shape!=ss[key].shape or ls[key].dtype!=ss[key].dtype:raise RuntimeError(f'EMA base decoder key/shape/dtype mismatch: {key}')
 static.load_state_dict({k:ls[k] for k in ss},strict=True)
 for key,value in static.state_dict().items():
  if not torch.equal(value.cpu(),ls[key].cpu()):raise RuntimeError(f'EMA base decoder byte mismatch after load: {key}')
 out={}
 for s in HO:
  old=banks[s]; raw=torch.from_numpy(np.ascontiguousarray(cal[s],np.float32)); direct=np.asarray(old.carrier,np.float32) if meta['arm']==ARM_D else np.zeros_like(old.carrier,dtype=np.float32); side=torch.from_numpy(direct)
  with torch.inference_mode(): e=encode_b3s(live.encoder,raw,side).detach().cpu().numpy()
  if not np.isfinite(e).all():raise RuntimeError(f'{s}: non-finite live B3S E0')
  out[s]=TaskBank(session_id=old.session_id,E0=np.ascontiguousarray(e,np.float32),carrier=np.ascontiguousarray(direct,np.float32),unit_mask=old.unit_mask,X_store=old.X_store,target_store=old.target_store,window_ids=old.window_ids,calibration_meta={**old.calibration_meta,'array_sha256':ah(e),'carrier_sha256':ah(direct),'joint_static_side':'direct_carrier' if meta['arm']==ARM_D else 'zero_carrier'})
 return material,banks,out,static

def close(label,want,got):
 ft(label+' live',want);ft(label+' static',got);d=float((want-got).abs().max());ff(label,d)
 if d>1e-5:raise RuntimeError(f'{label}: max_abs {d} > 1e-5')
 return d
def live_static(live,static,material,banks,mats):
 checks=[]
 for s in HO:
  item=material[s]; ds=item['dataset']; starts=item['starts']; full=next((i for i,start in enumerate(starts) if start>=99),None)
  if full is None:raise RuntimeError(f'{s}: HO3 has no fully valid W100 window')
  picks=(0,full,len(ds)-1)
  for i in picks:
   x=np.ascontiguousarray(ds[i][0],np.float32)[None]; valid=train.valid_mask_from_padded_starts((starts[i],),device=torch.device('cpu'))
   with torch.inference_mode():d=close(f'live/static {s}/{i}',live(torch.from_numpy(x),banks[s],input_valid_mask=valid),static(torch.from_numpy(x),mats[s],input_valid_mask=valid))
   checks.append({'session':s,'dataset_index':i,'start':int(starts[i]),'kind':'first' if i==0 else 'last' if i==len(ds)-1 else 'full_valid_midpoint','max_abs_error':d})
 return {'passed':True,'all_three_ho3_sessions':True,'checks':checks,'finite_live_and_static_each_check':True,'B_side_and_static_bank_zero':bool(live.arm!=ARM_B or all(np.array_equal(mats[s].carrier,np.zeros_like(mats[s].carrier)) for s in HO))}
def flows(material):
 rows=[]
 for j in range(8):
  s=HO[j%len(HO)]; d=material[s]['dataset']; offset=(j//len(HO))*401; raw=np.asarray(d.neural_data[s][99+offset:99+offset+T],np.float32)
  if raw.shape!=(T,64) or not np.isfinite(raw).all():raise RuntimeError(f'{s}: real HO3 stream invalid')
  rows.append((s,offset,raw))
 return rows,np.stack([x[2] for x in rows],axis=1)
def parity(model,banks,raw):
 ids=[f'i{i}' for i in range(len(banks))]; canon=list(ids); pos={k:i for i,k in enumerate(canon)}; by=dict(zip(canon,banks)); ref=RiftStreamDecoder(model); cached=CpuRiftRuntime(model,banks,ids,temporal_backend='cached'); state=CpuRiftRuntime(model,banks,ids,temporal_backend='state'); ds=[]
 with torch.inference_mode():
  for t in range(T):
   order=ids[::-1] if t==11 else ids; x=torch.from_numpy(raw[t,[pos[k] for k in order]].copy()); want=ref.stream_step(x,[by[k] for k in order],order); a=cached.advance(x,order if t==11 else None); b=state.advance(x,order if t==11 else None);ds += [close(f'cached {t}',want,a),close(f'state {t}',want,b)];ids=list(order)
  reset=ids[-1];cached.reset_rows([reset]);state.reset_rows([reset]);ref.reset(reset);x=torch.from_numpy(raw[-1,[pos[k] for k in ids]].copy());want=ref.stream_step(x,[by[k] for k in ids],ids);ds += [close('cached reset',want,cached.advance(x)),close('state reset',want,state.advance(x))]
 return {'passed':True,'checked_advances':402,'max_abs_error':max(ds),'cached_and_state_reference':True,'canonical_reorder_t11':True,'asynchronous_reset':True,'finite_each_step':True}
def oracle(model,bank,x):
 rt=CpuRiftRuntime(model,[bank],['i0'],temporal_backend='cached');rows=[]
 with torch.inference_mode():
  for t in range(T):
   got=rt.advance(torch.from_numpy(x[t:t+1].copy()));ft(f'oracle runtime {t}',got)
   if t in (99,200,400):
    want=model(torch.from_numpy(x[t-99:t+1][None].copy()),bank,input_valid_mask=torch.ones(1,100,dtype=torch.bool));rows.append({'end_bin':t,'max_abs_error':close(f'oracle {t}',want,got)})
 return {'passed':True,'checks':rows,'finite_each_step':True}
def measure(model,banks,raw):
 vals=[]; rounds=[]
 with torch.inference_mode():
  for r in range(3):
   rt=CpuRiftRuntime(model,banks,[f'i{i}' for i in range(len(banks))],temporal_backend='cached');t=time.perf_counter();o=rt.advance(torch.from_numpy(raw[0].copy()));startup=time.perf_counter()-t;ft('startup',o)
   for i in range(1,300):ft('warmup',rt.advance(torch.from_numpy(raw[i].copy())))
   z=[]
   for i in range(300,400):
    t=time.perf_counter();o=rt.advance(torch.from_numpy(raw[i].copy()));d=time.perf_counter()-t;ft('timed',o);ff('timed seconds',d);z.append(d)
   vals+=z;rounds.append({'round':r+1,'startup_first_predict_seconds':startup,'warmup_calls':300,'timed_calls':100,'steady':summary(z)})
 return {'per_round':rounds,'decode_call':summary(vals),**rss()}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--dest',type=Path,required=True);p.add_argument('--parity-only',action='store_true');a=p.parse_args()
 if a.dest.exists():raise FileExistsError(f'destination must be fresh: {a.dest}')
 torch.set_num_threads(2);torch.set_num_interop_threads(1); before=audit_sources();meta,score,ck,epoch,live=load(a.run_dir.resolve()); material,banks,mats,static=materialize(live,meta)
 if train._ho_contract(material)!=score.get('ho_contract'):raise RuntimeError('actual HO3 raw/query/bank/calibration contract drift from score receipt')
 static_check=live_static(live,static,material,banks,mats);streams,raw8=flows(material);cases={}
 for B in (1,8):
  raw=raw8[:,:B]; selected=[mats[streams[i][0]] for i in range(B)]; row={'shape':list(raw.shape),'sessions':[streams[i][0] for i in range(B)],'offsets':[streams[i][1] for i in range(B)],'raw_sha256':ah(raw),'true_zero_bin_count':int(np.sum(np.all(raw==0,axis=-1))),'banks':[{"e0_sha256":b.calibration_meta['array_sha256'],"carrier_sha256":b.calibration_meta['carrier_sha256']} for b in selected],'parity':parity(static,selected,raw),'fullwindow_oracle':oracle(static,selected[0],raw[:,0])}
  if not a.parity_only:row['cached_timing']=measure(static,selected,raw)
  cases[f'B{B}']=row
 after=audit_sources(); rep={'schema':'m1_joint_cpu_benchmark_v1','status':'PREFLIGHT_COMPLETED' if a.parity_only else 'BENCHMARK_COMPLETED','run_dir':str(a.run_dir.resolve()),'arm':meta['arm'],'selected_ema_epoch':epoch,'checkpoint':str(ck),'checkpoint_sha256':sha(ck),'run_meta_sha256':sha(a.run_dir/'run_meta.json'),'train_receipt_sha256':sha(a.run_dir/'train_receipt.json'),'score_receipt_sha256':sha(a.run_dir/'score_receipt.json'),'source_hashes':meta['source_hashes'],'weights':'selected checkpoint EMA, including live B3S encoder; static decoder receives materialized HO3 E0 and zero carrier for B, or direct carrier for D','input_surface':{'name':'HO3 real M10 and query streams','sessions':list(HO),'windows':COUNTS,'different_from_frozen_concat_cpu_runtime':True},'live_vs_materialized_static':static_check,'protocol':{'cpu_threads':2,'interop_threads':1,'advances':402,'cached_and_state_reference_parity':True,'full_window_oracle_end_bins':[99,200,400],'warmup_calls':300,'rounds':3,'timed_calls_per_round':100,'timing_skipped':a.parity_only,'no_target_bp':True},'runtime_import_source_preflight_sha256':before,'runtime_import_source_sha256':after,'runtime_import_source_stability':stable_sources(before,after),'script_sha256':sha(Path(__file__)),'environment':{'cpu':cpu(),'torch':torch.__version__,'affinity':sorted(os.sched_getaffinity(0)),'inference_mode':True},'cases':cases}
 a.dest.mkdir(parents=True);(a.dest/'benchmark.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n');print(json.dumps(rep,indent=2,sort_keys=True))
if __name__=='__main__':main()
