#!/usr/bin/env python3
"""CPU-only parity preflight for completed M2 RIFT concat/joint runs; no long timing."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,platform,time,resource
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2]; WS=ROOT.parent; V1=WS/'btransform_unified_v1'
sys.path[:0]=[str(ROOT/'src'),str(V1/'src'),str(WS)]
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.bank import TaskBank
from btransform_unified_v2.cpu_runtime import CpuRiftRuntime
from btransform_unified_v2.streaming import RiftStreamDecoder
from btransform_unified_v2.concat_model import RiftConcatDecoder
from btransform_unified_v2 import RiftDecoder
from tfpd_exploration.src.m2_dual_track_v1 import data,plan,champion
import m2_concat_train as base

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def finite(x):return bool(np.isfinite(float(x)))
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def cpu():
 for l in Path('/proc/cpuinfo').read_text().splitlines():
  if l.startswith('model name'):return l.split(':',1)[1].strip()
 return platform.processor()
def ema_apply(m,state):
 e=DecoderEMA(m,decay=.9995);e.load_state_dict(state['ema']);e.apply_to(m);return e
def selected(run):
 meta=json.loads((run/'run_meta.json').read_text()); score=json.loads((run/'score_receipt.json').read_text()); train=json.loads((run/'train_receipt.json').read_text())
 joint=meta.get('schema')=='m2_rift_joint_train_v2'; expected_schema='m2_rift_joint_train_receipt_v2' if joint else 'm2_rift_concat_train_receipt_v1'
 if any(not Path(path).is_file() or sha(Path(path))!=digest for path,digest in meta.get('source_hashes',{}).items()):raise RuntimeError('current source file hash drift')
 if meta.get('status')!='FORMAL' or score.get('status')!='COMPLETED' or train.get('schema')!=expected_schema or train.get('status')!='COMPLETED' or score.get('cell')!=meta.get('cell') or train.get('cell')!=meta.get('cell') or score.get('official_test_used') is not False or ('source_hashes' in score and score.get('source_hashes')!=meta.get('source_hashes')) or (('cache_hashes' in score or 'frozen_cache_hashes' in score) and score.get('cache_hashes',score.get('frozen_cache_hashes'))!=meta.get('cache_hashes',meta.get('frozen_cache_hashes'))):raise RuntimeError('formal train/score schema binding failed')
 if int(train.get('epochs',-1))!=24 or int(train.get('global_step',-1))!=24*3165 or train.get('source_hashes')!=meta.get('source_hashes') or (train.get('cache_hashes',train.get('frozen_cache_hashes'))!=meta.get('cache_hashes',meta.get('frozen_cache_hashes'))):raise RuntimeError('formal train/cache/source receipt binding failed')
 if joint and train.get('arm')!=meta.get('arm'):raise RuntimeError('joint arm mismatch')
 epoch=int(score['selection']['epoch']); ck=run/f'epoch_{epoch:03d}.pt'; st=torch.load(ck,map_location='cpu',weights_only=False); expected=score.get('ema_by_epoch',{}).get(str(epoch),{}).get('checkpoint_sha256')
 if not expected or sha(ck)!=expected or st.get('smoke') or st.get('cell')!=meta.get('cell') or int(st.get('epoch',-1))!=epoch or int(st.get('global_step',-1))!=epoch*3165 or st.get('seed')!=meta.get('seed') or (joint and st.get('arm')!=meta.get('arm')) or st.get('source_hashes')!=meta['source_hashes'] or st.get('cache_hashes',st.get('frozen_cache_hashes'))!=meta.get('cache_hashes',meta.get('frozen_cache_hashes')):raise RuntimeError('selected checkpoint binding failed')
 return meta,score,ck,st,epoch
def concat(meta,st):
 m=RiftConcatDecoder('m2',context_bins=50,bias_mode='recency',seed=int(meta['seed'])).eval();m.load_state_dict(st['raw_state_dict']);ema_apply(m,st);return m
def joint_materialized(meta,st,banks):
 from btransform_unified_v2.joint_m2_model import JointM2RiftDecoder,ARM_D
 live=JointM2RiftDecoder(meta['arm'],seed=int(meta['seed'])).eval();live.load_state_dict(st['model']);ema_apply(live,st); live.install_session_memory({b.session_id:b for b in banks},Path(banks[0].calibration_meta['cache_root'])/'ext4')
 cached=RiftDecoder('m2',context_bins=50,bias_mode='recency',seed=int(meta['seed']),proj_dim=16).eval(); ld=live.state_dict(); cd=cached.state_dict()
 for k in cd:
  if k not in ld or cd[k].shape!=ld[k].shape:raise RuntimeError(f'joint shared state missing/shape drift: {k}')
  cd[k]=ld[k]
 cached.load_state_dict(cd,strict=True); mats=[]; t0=time.perf_counter()
 for bank in banks:
  raw=np.load(Path(bank.calibration_meta['cache_root'])/'ext4'/bank.session_id/'calib_activity.npy').astype(np.float32); direct=np.asarray(bank.carrier,np.float32) if meta['arm']==ARM_D else np.zeros_like(bank.carrier,np.float32); side=champion.empty_contrast_side(direct if meta['arm']==ARM_D else np.zeros_like(direct)); e0,_=champion.native_e0_and_u(live.encoder,torch.from_numpy(raw),side)
  mats.append(TaskBank(session_id=bank.session_id,E0=np.ascontiguousarray(e0.numpy()),carrier=np.ascontiguousarray(direct),unit_mask=bank.unit_mask,X_store=bank.X_store,target_store=bank.target_store,window_ids=bank.window_ids,calibration_meta={**bank.calibration_meta,'array_sha256':ah(e0.numpy()),'carrier_sha256':ah(direct)}))
 return live,cached,mats,time.perf_counter()-t0
def flows(dual,banks):
 sessions=list(plan.EXT4_SESSIONS);out=[]
 for j in range(8):
  s=sessions[j%4]; d=dual[s]; starts=np.asarray(d.eligible_starts); start=int(starts[0 if j<4 else 1]); store=np.asarray(d.X_store)
  x=np.asarray(store[start:start+401],np.float32)
  if x.shape!=(401,96):raise RuntimeError(f'{s} insufficient 401 bins from {start}: {x.shape}')
  out.append((s,start,x,banks[s]))
 raw=np.stack([x[2] for x in out],1);return out,raw
def parity(m,banks,raw):
 canonical=[f'i{i}' for i in range(len(banks))]; ids=list(canonical);pos={i:n for n,i in enumerate(canonical)};by=dict(zip(canonical,banks));ref=RiftStreamDecoder(m);rt=CpuRiftRuntime(m,banks,ids,temporal_backend='cached');mx=0.
 with torch.inference_mode():
  for t in range(len(raw)):
   order=ids[::-1] if t==11 else ids; inp=torch.from_numpy(raw[t,[pos[i] for i in order]].copy());want=ref.stream_step(inp,[by[i] for i in order],order);got=rt.advance(inp,order if t==11 else None);mx=max(mx,float((want-got).abs().max()));ids=order
  rt.reset_rows([ids[-1]]);ref.reset(ids[-1]);inp=torch.from_numpy(raw[-1,[pos[i] for i in ids]].copy());want=ref.stream_step(inp,[by[i] for i in ids],ids);got=rt.advance(inp);mx=max(mx,float((want-got).abs().max()))
 if not finite(mx) or mx>1e-5:raise RuntimeError(f'cached stream parity {mx}')
 return {'passed':True,'checked_advances':len(raw)+1,'max_abs_error':mx,'canonical_reorder_t11':True,'async_reset':True}
def oracle(m,bank,x):
 rt=CpuRiftRuntime(m,[bank],['i0'],temporal_backend='cached');rows=[]
 with torch.inference_mode():
  for t in range(401):
   got=rt.advance(torch.from_numpy(x[t:t+1].copy()))
   if t in (49,200,400):
    want=m(torch.from_numpy(x[t-49:t+1][None].copy()),bank,input_valid_mask=torch.ones(1,50,dtype=torch.bool));d=float((got-want).abs().max());
    if not finite(d) or d>1e-5:raise RuntimeError(f'oracle {t} {d}')
    rows.append({'end_bin':t,'max_abs_error':d})
 return {'passed':True,'checks':rows}
def rss():
 d={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/self/status').read_text().splitlines() if l.startswith(('VmRSS:','VmHWM:'))};return {'rss_bytes':d.get('VmRSS',0),'vmhwm_bytes':max(d.get('VmHWM',0),resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)}
def measure(m,banks,raw):
 all=[]; rounds=[]
 with torch.inference_mode():
  for r in range(3):
   rt=CpuRiftRuntime(m,banks,[f'i{i}' for i in range(len(banks))],temporal_backend='cached');t=time.perf_counter();rt.advance(torch.from_numpy(raw[0]));startup=time.perf_counter()-t
   for x in raw[1:300]:rt.advance(torch.from_numpy(x))
   vals=[]
   for x in raw[300:400]:t=time.perf_counter();rt.advance(torch.from_numpy(x));vals.append(time.perf_counter()-t)
   all+=vals; rounds.append({'startup_first_predict_seconds':startup,'calls':len(vals)})
 a=np.asarray(all)*1e3;return {'rounds':rounds,'decode_call':{'n':len(a),'median_ms':float(np.median(a)),'p95_ms':float(np.percentile(a,95))},**rss()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--dest',type=Path,required=True);p.add_argument('--cpu-threads',type=int,default=2);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
 if a.cpu_threads!=2:raise ValueError('CPU benchmark contract requires --cpu-threads 2')
 if a.dest.exists():raise FileExistsError(a.dest)
 torch.set_num_threads(a.cpu_threads);torch.set_num_interop_threads(1);meta,score,ck,st,epoch=selected(a.run_dir.resolve());dual,bmap=base._load_surface('ext4',torch.device('cpu'));orig=[bmap[s] for s in plan.EXT4_SESSIONS];kind='joint' if meta.get('schema')=='m2_rift_joint_train_v2' else 'concat'
 if kind=='concat':
  if base._cache_hashes('ext4',dual,bmap)!=meta['frozen_cache_hashes']['ext4']:raise RuntimeError('actual ext4 cache drift')
  cached=concat(meta,st);live=None;mats=orig;matwall=None
 else:
  for session,row in meta['cache_hashes']['ext4'].items():
   for name,digest in row.items():
    if name.endswith(('.npy','.pt')) and sha(WS/'tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache/ext4'/session/name)!=digest:raise RuntimeError(f'actual joint cache drift {session}/{name}')
  live,cached,mats,matwall=joint_materialized(meta,st,orig)
 streams,raw8=flows(dual,{b.session_id:b for b in mats}); cases={}; jointpar=None
 if live is not None:
  checks=[]
  for b,ob in zip(mats,orig):
   startbin=int(dual[b.session_id].eligible_starts[0]);x=np.array(dual[b.session_id].X_store[startbin:startbin+50],np.float32,copy=True)[None]
   with torch.inference_mode():d=float((live(torch.from_numpy(x),ob)-cached(torch.from_numpy(x),b)).abs().max())
   if not finite(d) or d>1e-5:raise RuntimeError(f'joint materialization parity {b.session_id} {d}')
   checks.append({'session':b.session_id,'max_abs_error':d})
  jointpar={'passed':True,'per_ext4_bank_fullwindow':checks,'materialization_total_wall_seconds':matwall}
 for B in (1,8):
  raw=raw8[:,:B];bs=[mats[list(plan.EXT4_SESSIONS).index(streams[i][0])] for i in range(B)];cases[f'B{B}']={'shape':list(raw.shape),'sessions':[streams[i][0] for i in range(B)],'offsets':[streams[i][1] for i in range(B)],'true_zero_bin_count':int(np.sum(np.all(raw==0,axis=-1))),'raw_sha256':ah(raw),'bank_sha256':[ah(x.E0)+ah(x.carrier) for x in bs],'parity':parity(cached,bs,raw),'fullwindow_oracle':oracle(cached,bs[0],raw[:,0])}
  if not a.preflight_only:cases[f'B{B}']['cached_timing']=measure(cached,bs,raw)
 rep={'schema':'m2_cpu_benchmark_v2','status':'PREFLIGHT_COMPLETED' if a.preflight_only else 'BENCHMARK_COMPLETED','kind':kind,'run_dir':str(a.run_dir.resolve()),'selected_ema_epoch':epoch,'checkpoint':str(ck),'checkpoint_sha256':sha(ck),'run_meta_sha256':sha(a.run_dir/'run_meta.json'),'score_receipt_sha256':sha(a.run_dir/'score_receipt.json'),'source_hashes':meta['source_hashes'],'cache_hashes':meta.get('cache_hashes',meta.get('frozen_cache_hashes')),'script_sha256':sha(Path(__file__)),'environment':{'cpu':cpu(),'torch':torch.__version__,'device':'cpu','threads':2,'affinity':sorted(os.sched_getaffinity(0)),'inference_mode':True},'joint_materialization':jointpar,'protocol':{'advances':402,'warmup':300,'rounds':3,'timed_calls_per_round':100,'preflight_only':a.preflight_only,'no_target_bp':True},'cases':cases};a.dest.mkdir(parents=True);(a.dest/'benchmark.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
