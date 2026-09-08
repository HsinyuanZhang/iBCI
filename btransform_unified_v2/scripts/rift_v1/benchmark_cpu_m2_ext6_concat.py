#!/usr/bin/env python3
"""Sealed CPU benchmark: ext6 selects EMA e9; frozen ext4 supplies runtime streams."""
from __future__ import annotations
import argparse, hashlib, json, math, os, platform, resource, sys, time
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]; WS=ROOT.parent; V1=WS/'btransform_unified_v1'
for p in (ROOT/'src',V1/'src',WS,ROOT/'scripts/rift_v1'):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v2.concat_model import RiftConcatDecoder
from btransform_unified_v2.cpu_runtime import CpuRiftRuntime
from btransform_unified_v2.streaming import RiftStreamDecoder
import m2_ext6_epoch_pick as pick
SIX=tuple(pick.SIX); EPOCHS=tuple(range(1,25)); E9=9; W=50; T=401
DEFAULT=ROOT/'results/rift_v1/m2_r50_concat_s42_ext6_pick_v1'
DEFAULT_D42=ROOT/'results/rift_v1/m2_r50_joint_d_s42_formal_v1'
def sha(p):
 p=Path(p)
 if not p.is_file():raise FileNotFoundError(p)
 return hashlib.sha256(p.read_bytes()).hexdigest()
def cd(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def ff(label,x):
 if not math.isfinite(float(x)):raise RuntimeError(f'{label}: non-finite')
def ft(label,x):
 if not bool(torch.isfinite(x).all()):raise RuntimeError(f'{label}: non-finite')
def hashes(d,label):
 if not d:raise RuntimeError(f'{label}: no source hashes')
 for p,h in d.items():
  if not isinstance(h,str) or len(h)!=64 or sha(p)!=h:raise RuntimeError(f'{label}: hash drift {p}')
def runtime_sources():
 """Hash every workspace Python module actually imported by this execution."""
 out={}
 for mod in tuple(sys.modules.values()):
  spec=getattr(mod,'__spec__',None);origin=getattr(spec,'origin',None)
  text=origin if isinstance(origin,str) and Path(origin).is_absolute() else getattr(mod,'__file__',None)
  if not isinstance(text,str) or not Path(text).is_absolute():continue
  path=Path(text).resolve()
  if path.is_file() and path.suffix=='.py' and (path.is_relative_to(ROOT) or path.is_relative_to(WS)):
   out[str(path)]=sha(path)
 required=(Path(__file__).resolve(),ROOT/'scripts/rift_v1/benchmark_cpu_m2.py',ROOT/'src/btransform_unified_v2/cpu_runtime.py',ROOT/'src/btransform_unified_v2/streaming.py',ROOT/'src/btransform_unified_v2/concat_model.py',ROOT/'scripts/rift_v1/m2_ext6_epoch_pick.py')
 if any(str(path.resolve()) not in out for path in required):raise RuntimeError('runtime source audit lacks a required actual import')
 return dict(sorted(out.items()))
def eq(a,b,label):
 if a!=b:raise RuntimeError(f'{label}: binding mismatch')
def validate(root):
 root=root.resolve(); ps={n:root/n for n in ('manifest.json','score_receipt.json','selected_ema_receipt.json','validation_audit.json','selected_ema.pt')}
 if any(not p.is_file() for p in ps.values()):raise RuntimeError('missing ext6 selection artifact')
 m=json.loads(ps['manifest.json'].read_text());s=json.loads(ps['score_receipt.json'].read_text());er=json.loads(ps['selected_ema_receipt.json'].read_text());a=json.loads(ps['validation_audit.json'].read_text())
 if m.get('schema')!='m2_rift_ext6_epoch_pick_v2_manifest' or m.get('view')!='EMA' or m.get('epochs')!=list(EPOCHS) or m.get('official_test_used') is not False or m.get('evalai_opened') is not False:raise RuntimeError('manifest requires formal 24 epoch EMA ext6 scope')
 if s.get('schema')!='m2_rift_ext6_epoch_pick_v2_selection' or s.get('status')!='COMPLETED' or s.get('view')!='EMA' or s.get('official_test_used') is not False or s.get('evalai_opened') is not False:raise RuntimeError('completed EMA receipt required')
 eq(s,er,'selected EMA receipt')
 if s.get('manifest_sha256')!=cd(m):raise RuntimeError('receipt manifest binding failed')
 if a.get('schema')!='m2_concat_ext6_full_curve_validation_v1' or a.get('status')!='PASSED' or a.get('failures')!=[] or a.get('scope',{}).get('read_only') is not True or a.get('scope',{}).get('official_test_opened') is not False:raise RuntimeError('full curve validation audit failed')
 for n in ('manifest.json','score_receipt.json','selected_ema_receipt.json','selected_ema.pt'):
  if a.get('artifact_sha256',{}).get(n)!=sha(ps[n]):raise RuntimeError(f'audit artifact hash drift {n}')
 if a.get('checks',{}).get('all_24_checkpoint_curve_rows',{}).get('passed') is not True or len(a['checks']['all_24_checkpoint_curve_rows'].get('detail',[]))!=24:raise RuntimeError('audit lacks 24 curve rows')
 hashes(m.get('source_sha256',{}),'selection source'); assets=pick.query_asset_hashes(Path(m['query_cache']));eq(assets,m.get('query_assets',{}),'ext6 query cache')
 curve=s.get('ema_by_epoch',{}); cps=m.get('checkpoint_bytes',{})
 if set(curve)!={str(e) for e in EPOCHS}:raise RuntimeError('incomplete 24 epoch curve')
 for e in EPOCHS:
  r=curve[str(e)]; per=r.get('per_session',{})
  if r.get('view')!='EMA' or r.get('partial') is not False or r.get('checkpoint_sha256')!=cps.get(str(e),{}).get('sha256') or int(r.get('n_windows',-1))!=15403 or set(per)!=set(SIX):raise RuntimeError(f'e{e}: not complete 6 x 15403 EMA curve')
  xs=[]
  for session in SIX:
   row=per[session]
   if int(row.get('window_count',-1))!=int(assets['sessions'][session]['window_count']) or not isinstance(row.get('prediction_sha256'),str):raise RuntimeError(f'e{e}/{session}: window or prediction seal drift')
   ff(f'e{e}/{session} r2',row.get('r2',float('nan')));xs.append(float(row['r2']))
  ff(f'e{e} mean',r.get('equal_session_mean',float('nan')));ff(f'e{e} pooled',r.get('pooled_r2',float('nan')))
  if abs(float(r['equal_session_mean'])-float(np.mean(xs)))>1e-12:raise RuntimeError(f'e{e}: equal mean drift')
  if sha(cps[str(e)]['path'])!=r['checkpoint_sha256']:raise RuntimeError(f'e{e}: checkpoint drift')
 values={e:float(curve[str(e)]['equal_session_mean']) for e in EPOCHS};best=max(EPOCHS,key=lambda e:(values[e],-e))
 if best!=E9 or s.get('selection')!={'rule':'earliest maximum finite unweighted equal_session_mean','epoch':best,'equal_session_mean':values[best]} or s.get('selected')!=curve[str(best)]:raise RuntimeError('recomputed selected e9 binding failed')
 st=s.get('selected_ema_state',{})
 if st.get('path')!=str(ps['selected_ema.pt']) or st.get('sha256')!=sha(ps['selected_ema.pt']) or st.get('raw_state_serialized') is not False:raise RuntimeError('EMA-only tensor package binding failed')
 run=Path(m['run']); meta=json.loads((run/'run_meta.json').read_text());tr=json.loads((run/'train_receipt.json').read_text())
 if sha(run/'run_meta.json')!=m.get('run_meta_sha256') or sha(run/'train_receipt.json')!=m.get('train_receipt_sha256') or int(m.get('seed',-1))!=42 or int(meta.get('seed',-1))!=42 or meta.get('schema')!='m2_rift_concat_train_v1' or meta.get('status')!='FORMAL' or meta.get('cell')!=m.get('cell') or tr.get('schema')!='m2_rift_concat_train_receipt_v1' or tr.get('cell')!=meta.get('cell') or tr.get('status')!='COMPLETED' or int(tr.get('epochs',-1))!=24 or int(tr.get('global_step',-1))!=24*3165 or tr.get('source_hashes')!=meta.get('source_hashes') or tr.get('frozen_cache_hashes')!=meta.get('frozen_cache_hashes'):raise RuntimeError('formal run/meta/cache binding failed')
 hashes(meta.get('source_hashes',{}),'formal source')
 cp=Path(cps[str(E9)]['path']);state=torch.load(cp,map_location='cpu',weights_only=False)
 if state.get('schema')!='m2_rift_concat_epoch_checkpoint_v1' or state.get('cell')!=meta['cell'] or int(state.get('epoch',-1))!=E9 or int(state.get('global_step',-1))!=E9*3165 or state.get('smoke') is not False or state.get('source_hashes')!=meta['source_hashes'] or state.get('frozen_cache_hashes')!=meta['frozen_cache_hashes']:raise RuntimeError('selected e9 checkpoint contract failed')
 pkg=torch.load(ps['selected_ema.pt'],map_location='cpu',weights_only=True)
 if not isinstance(pkg,dict) or not pkg or not all(isinstance(x,torch.Tensor) and bool(torch.isfinite(x).all()) for x in pkg.values()):raise RuntimeError('EMA package values invalid')
 return m,s,meta,pkg,ps,assets,state
def model(meta,pkg,state):
 x=RiftConcatDecoder('m2',context_bins=W,bias_mode='recency',seed=int(meta['seed'])).eval();x.load_state_dict(pkg,strict=True)
 y=RiftConcatDecoder('m2',context_bins=W,bias_mode='recency',seed=int(meta['seed'])).eval();y.load_state_dict(state['raw_state_dict'],strict=True);e=DecoderEMA(y,decay=.9995);e.load_state_dict(state['ema']);e.apply_to(y)
 z=y.state_dict()
 if set(z)!=set(pkg) or any(not torch.equal(z[k].cpu(),pkg[k].cpu()) for k in pkg):raise RuntimeError('EMA package does not equal selected e9 checkpoint EMA')
 return x
def ext4_streams(meta):
 # The ext6 curve chooses EMA e9. Runtime input remains the frozen oldbenchmark
 # ext4 surface, whose real 401-bin starts deliberately avoid query padding.
 import benchmark_cpu_m2 as old
 dual,bmap=old.base._load_surface('ext4',torch.device('cpu'))
 if old.base._cache_hashes('ext4',dual,bmap)!=meta['frozen_cache_hashes']['ext4']:raise RuntimeError('actual ext4 cache drift from formal concat metadata')
 streams,raw=old.flows(dual,bmap)
 if raw.shape!=(T,8,96) or not np.isfinite(raw).all():raise RuntimeError('oldbenchmark ext4 B8 raw surface drift/nonfinite')
 return old,streams,raw,bmap
def check(label,want,got):
 ft(label+' want',want);ft(label+' got',got);d=float((want-got).abs().max());ff(label+' abs',d)
 if d>1e-5:raise RuntimeError(f'{label}: parity {d}')
 return d
def parity(m,banks,raw):
 ids=[f'i{i}' for i in range(len(banks))]; canon=list(ids);by=dict(zip(canon,banks));pos={k:i for i,k in enumerate(canon)};ref=RiftStreamDecoder(m);rt=CpuRiftRuntime(m,banks,ids,temporal_backend='cached');ds=[]
 with torch.inference_mode():
  for t in range(T):
   order=ids[::-1] if t==11 else ids;inp=torch.from_numpy(raw[t,[pos[k] for k in order]].copy());want=ref.stream_step(inp,[by[k] for k in order],order);got=rt.advance(inp,order if t==11 else None);ds.append(check(f'advance {t}',want,got));ids=list(order)
  reset=ids[-1];rt.reset_rows([reset]);ref.reset(reset);inp=torch.from_numpy(raw[-1,[pos[k] for k in ids]].copy());want=ref.stream_step(inp,[by[k] for k in ids],ids);got=rt.advance(inp);ds.append(check('reset advance',want,got))
 return {'passed':True,'checked_advances':402,'max_abs_error':max(ds),'canonical_reorder_t11':True,'asynchronous_reset':True,'finite_want_and_got_each_step':True}
def oracle(m,b,x):
 rt=CpuRiftRuntime(m,[b],['i0'],temporal_backend='cached');out=[]
 with torch.inference_mode():
  for t in range(T):
   got=rt.advance(torch.from_numpy(x[t:t+1].copy()));ft(f'oracle runtime {t}',got)
   if t in (49,200,400):
    want=m(torch.from_numpy(x[t-49:t+1][None].copy()),b,input_valid_mask=torch.ones(1,W,dtype=torch.bool));out.append({'end_bin':t,'max_abs_error':check(f'oracle {t}',want,got)})
 return {'passed':True,'checks':out,'finite_want_and_got_each_step':True}
def rss():
 x={a.split(':')[0]:int(a.split()[1])*1024 for a in Path('/proc/self/status').read_text().splitlines() if a.startswith(('VmRSS:','VmHWM:'))};return {'rss_bytes':x.get('VmRSS',0),'vmhwm_bytes':max(x.get('VmHWM',0),resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)}
def timing(m,banks,raw):
 vals=[]; rounds=[]
 with torch.inference_mode():
  for r in range(3):
   rt=CpuRiftRuntime(m,banks,[f'i{i}' for i in range(len(banks))],temporal_backend='cached');t=time.perf_counter();o=rt.advance(torch.from_numpy(raw[0].copy()));startup=time.perf_counter()-t;ft(f'timing startup {r}',o);ff(f'timing startup {r}',startup)
   for i in range(1,300):ft(f'warmup {r}/{i}',rt.advance(torch.from_numpy(raw[i].copy())))
   for i in range(300,400):
    t=time.perf_counter();o=rt.advance(torch.from_numpy(raw[i].copy()));d=time.perf_counter()-t;ft(f'timing {r}/{i}',o);ff(f'timing {r}/{i}',d);vals.append(d)
   rounds.append({'round':r+1,'startup_first_predict_seconds':startup,'warmup_calls':300,'timed_calls':100})
 a=np.asarray(vals)*1e3
 if not np.isfinite(a).all():raise RuntimeError('non-finite timings')
 return {'rounds':rounds,'decode_call':{'n':len(a),'median_ms':float(np.median(a)),'p95_ms':float(np.percentile(a,95))},**rss()}
def cpu():
 for x in Path('/proc/cpuinfo').read_text().splitlines():
  if x.startswith('model name'):return x.split(':',1)[1].strip()
 return platform.processor()
def original_d42_parity(run):
 """Compatibility proof for the earlier M2 joint-D seed-42 e17 measurement.

 It deliberately invokes no timing path and uses the actual ext4 B1/B8 inputs
 used by the historical CPU surface.  The generic parity routine above adds
 the required finite checks missing from that old benchmark.
 """
 import benchmark_cpu_m2 as old
 meta,score,ck,state,epoch=old.selected(Path(run).resolve());cert_path=ROOT/'results/rift_v1/m2_joint_d_s42_trained_cpu_benchmark_v1/benchmark.json';cert=json.loads(cert_path.read_text())
 if meta.get('schema')!='m2_rift_joint_train_v2' or meta.get('arm')!='D_JOINT' or int(meta.get('seed',-1))!=42 or epoch!=17:raise RuntimeError('original D42 compatibility mode requires formal joint-D seed42 e17')
 if cert.get('schema')!='m2_cpu_benchmark_v2' or cert.get('kind')!='joint' or cert.get('run_dir')!=str(Path(run).resolve()) or cert.get('selected_ema_epoch')!=17 or cert.get('checkpoint')!=str(ck) or cert.get('checkpoint_sha256')!=sha(ck) or cert.get('run_meta_sha256')!=sha(Path(run)/'run_meta.json') or cert.get('score_receipt_sha256')!=sha(Path(run)/'score_receipt.json'):raise RuntimeError('old D42 benchmark certificate checkpoint/meta binding failed')
 dual,bmap=old.base._load_surface('ext4',torch.device('cpu'));orig=[bmap[s] for s in old.plan.EXT4_SESSIONS]
 live,cached,mats,_=old.joint_materialized(meta,state,orig);streams,raw8=old.flows(dual,{b.session_id:b for b in mats});material=[]
 for bank,static in zip(orig,mats):
  start=int(dual[bank.session_id].eligible_starts[0]);x=np.asarray(dual[bank.session_id].X_store[start:start+W],np.float32)[None]
  with torch.inference_mode():want=live(torch.from_numpy(x.copy()),bank);got=cached(torch.from_numpy(x.copy()),static)
  material.append({'session':bank.session_id,'max_abs_error':check('D42 live/static '+bank.session_id,want,got)})
 cases={}
 for B in (1,8):
  raw=raw8[:,:B];banks=[mats[list(old.plan.EXT4_SESSIONS).index(streams[i][0])] for i in range(B)];prior=cert.get('cases',{}).get(f'B{B}',{})
  actual_banks=[old.ah(b.E0)+old.ah(b.carrier) for b in banks]
  if prior.get('shape')!=list(raw.shape) or prior.get('sessions')!=[streams[i][0] for i in range(B)] or prior.get('offsets')!=[streams[i][1] for i in range(B)] or prior.get('raw_sha256')!=old.ah(raw) or prior.get('bank_sha256')!=actual_banks:raise RuntimeError(f'old D42 B{B} exact-input certificate drift')
  cases[f'B{B}']={'shape':list(raw.shape),'sessions':prior['sessions'],'offsets':prior['offsets'],'raw_sha256':prior['raw_sha256'],'bank_sha256':actual_banks,'parity':parity(cached,banks,raw),'fullwindow_oracle':oracle(cached,banks[0],raw[:,0])}
 return {'schema':'m2_original_d42_e17_parity_v1','status':'PREFLIGHT_COMPLETED','mode':'parity_only_original_d42_e17','timing_skipped':True,'run_dir':str(Path(run).resolve()),'checkpoint':str(ck),'checkpoint_sha256':sha(ck),'selected_epoch':epoch,'source_hashes':meta['source_hashes'],'cache_hashes':meta.get('cache_hashes',meta.get('frozen_cache_hashes')),'old_benchmark_certificate':str(cert_path),'old_benchmark_certificate_sha256':sha(cert_path),'live_vs_static_all4_fullwindow':{'passed':True,'checks':material,'finite_and_close':True},'cases':cases}
def main():
 p=argparse.ArgumentParser(description='sealed M2 concat ext6 CPU benchmark');p.add_argument('--selection-dir',type=Path,default=DEFAULT);p.add_argument('--dest',type=Path,required=True);p.add_argument('--cpu-threads',type=int,default=2);p.add_argument('--preflight-only',action='store_true');p.add_argument('--parity-only-original-d42',action='store_true',help='strict finite 402-step parity-only proof for the prior joint-D seed42 e17 surface');p.add_argument('--original-d42-run',type=Path,default=DEFAULT_D42)
 a=p.parse_args()
 if a.cpu_threads!=2:raise ValueError('CPU benchmark contract requires --cpu-threads 2')
 if a.dest.exists():raise FileExistsError(f'fresh destination required: {a.dest}')
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 if a.parity_only_original_d42:
  import benchmark_cpu_m2
  audit=runtime_sources();rep=original_d42_parity(a.original_d42_run);rep['script_sha256']=sha(Path(__file__));rep['runtime_import_source_preflight_sha256']=audit;rep['runtime_import_source_sha256']=runtime_sources();rep['environment']={'cpu':cpu(),'torch':torch.__version__,'device':'cpu','threads':2,'affinity':sorted(os.sched_getaffinity(0)),'inference_mode':True};a.dest.mkdir(parents=True);(a.dest/'benchmark.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n');print(json.dumps(rep,indent=2,sort_keys=True));return
 manifest,score,meta,pkg,paths,assets,state=validate(a.selection_dir);import benchmark_cpu_m2;audit=runtime_sources();m=model(meta,pkg,state);old,ss,raw8,bmap=ext4_streams(meta);cases={}
 for B in (1,8):
  raw=raw8[:,:B];banks=[bmap[ss[i][0]] for i in range(B)];row={'shape':list(raw.shape),'sessions':[ss[i][0] for i in range(B)],'offsets':[ss[i][1] for i in range(B)],'raw_sha256':ah(raw),'bank_sha256':[old.ah(b.E0)+old.ah(b.carrier) for b in banks],'true_zero_bin_count':int(np.sum(np.all(raw==0,axis=-1))),'parity':parity(m,banks,raw),'fullwindow_oracle':oracle(m,banks[0],raw[:,0])}
  if not a.preflight_only and not a.parity_only_original_d42:row['cached_timing']=timing(m,banks,raw)
  cases[f'B{B}']=row
 rep={'schema':'m2_concat_ext6_cpu_benchmark_v1','status':'PREFLIGHT_COMPLETED' if a.preflight_only or a.parity_only_original_d42 else 'BENCHMARK_COMPLETED','selection_surface':{'name':'ext6_full_curve','sessions':list(SIX),'epochs':24,'windows_per_epoch':15403,'selected_ema_epoch':E9,'selection_rule':score['selection']['rule']},'stream_surface':{'name':'frozen_oldbenchmark_ext4','observed_bins_per_stream':T,'batches':[1,8],'sessions':list(old.plan.EXT4_SESSIONS),'selection_surface_is_distinct':True},'selection_dir':str(a.selection_dir.resolve()),'manifest_sha256':sha(paths['manifest.json']),'score_receipt_sha256':sha(paths['score_receipt.json']),'validation_audit_sha256':sha(paths['validation_audit.json']),'ema_package':str(paths['selected_ema.pt']),'ema_package_sha256':sha(paths['selected_ema.pt']),'selected_checkpoint':manifest['checkpoint_bytes'][str(E9)]['path'],'selected_checkpoint_sha256':manifest['checkpoint_bytes'][str(E9)]['sha256'],'formal_run_meta_sha256':sha(Path(manifest['run'])/'run_meta.json'),'source_hashes':meta['source_hashes'],'frozen_cache_hashes':meta['frozen_cache_hashes'],'query_assets':assets,'script_sha256':sha(Path(__file__)),'runtime_import_source_preflight_sha256':audit,'runtime_import_source_sha256':runtime_sources(),'environment':{'cpu':cpu(),'torch':torch.__version__,'device':'cpu','threads':2,'affinity':sorted(os.sched_getaffinity(0)),'inference_mode':True},'protocol':{'parity_advances':402,'warmup_calls_per_round':300,'rounds':3,'timed_calls_per_round':100,'full_window_oracle_bins':[49,200,400],'timing_skipped':bool(a.preflight_only or a.parity_only_original_d42)},'cases':cases}
 a.dest.mkdir(parents=True);(a.dest/'benchmark.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n');print(json.dumps(rep,indent=2,sort_keys=True))
if __name__=='__main__':main()
