#!/usr/bin/env python3
"""Read-only M2 joint-D42/e17 M33 calibration-to-static-bank cost audit."""
from __future__ import annotations
import argparse,hashlib,json,os,resource,sys,time
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES','');os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];WS=ROOT.parent;V1=WS/'btransform_unified_v1'
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/rift_v1'),str(ROOT/'scripts/diagnostics_v1'),str(V1/'src'),str(V1/'scripts'),str(WS)]
from btransform_unified_v1.bank import TaskBank
from tfpd_exploration.src.m2_dual_track_v1 import champion,data,plan
import m2_concat_train as base
import benchmark_cpu_m2 as runtime
import m2_carrier_support_stability as stability
RUN=ROOT/'results/rift_v1/m2_r50_joint_d_s42_formal_v1'; CACHE=WS/'tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache'
def sha(p):
 h=hashlib.sha256();f=Path(p).open('rb')
 for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 f.close();return h.hexdigest()
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def source_hashes():
 paths=(Path(__file__),Path(stability.__file__),Path(champion.__file__),Path(data.__file__),Path(runtime.__file__),Path(base.__file__),ROOT/'src/btransform_unified_v2/joint_m2_model.py',WS/'streaming_calibration_exp/src/data/falcon_t4_features.py',Path(TaskBank.__module__.replace('.','/')))
 # TaskBank's source is resolved from its imported module, rather than a guessed path.
 import importlib
 bank_mod=importlib.import_module(TaskBank.__module__)
 paths=paths[:-1]+(Path(bank_mod.__file__),)
 return {str(q.resolve()):sha(q) for q in paths}
def opened_nwb_hashes(access):
    rows=access.get('opened_nwbs', {}) if isinstance(access, dict) else {}
    candidates=rows.get('opened', []) if isinstance(rows, dict) else []
    return {str(Path(q)):sha(Path(q)) for q in candidates if Path(q).is_file()}
def main():
 a=argparse.ArgumentParser();a.add_argument('--session',choices=plan.EXT4_SESSIONS,default=plan.EXT4_SESSIONS[0]);a.add_argument('--rounds',type=int,default=3);a.add_argument('--output',type=Path,required=True);a.add_argument('--cpus',default='12,13');a.add_argument('--preflight',action='store_true');x=a.parse_args()
 if x.output.exists():raise FileExistsError(x.output)
 if x.rounds < 1: raise ValueError('--rounds must be positive')
 cp={int(i) for i in x.cpus.split(',')};
 if not cp or not cp<={12,13}:raise RuntimeError('only CPUs 12,13 permitted')
 os.sched_setaffinity(0,cp);torch.set_num_threads(2);torch.set_num_interop_threads(1)
 meta,score,ck,st,epoch=runtime.selected(RUN); 
 if meta.get('arm')!='D_JOINT' or epoch!=17:raise RuntimeError('requires trained joint D42 EMA e17')
 dual,bmap=base._load_surface('ext4',torch.device('cpu')); bank=bmap[x.session]
 mean,std,norm=stability.normalizer(); bundles,access=stability.ext4_bundles(); bundle=bundles[x.session]
 # Source objects are loaded before all walls. Raw M33 is intentionally real source bundle, not cache activity.
 live,_,mats,_=runtime.joint_materialized(meta,st,[bank]); expected_e0=np.asarray(mats[0].E0); raw=np.load(CACHE/'ext4'/x.session/'calib_activity.npy').astype(np.float32)
 T0,_,_=stability.selected_carrier(bundle,np.arange(33),mean=mean,std=std,session=x.session)
 if not np.array_equal(T0,bank.carrier):raise RuntimeError('raw M33 T differs production cached bank')
 rows=[]
 for i in range(x.rounds):
  wall0=time.perf_counter(); t=time.perf_counter(); q,_,_=stability.selected_carrier(bundle,np.arange(33),mean=mean,std=std,session=x.session); c=time.perf_counter()-t
  t=time.perf_counter(); side=champion.empty_contrast_side(q);e0,_=champion.native_e0_and_u(live.encoder,torch.from_numpy(raw),side); e=time.perf_counter()-t
  t=time.perf_counter(); e0a=np.ascontiguousarray(e0.numpy()); qa=np.ascontiguousarray(q); correct_meta={**bank.calibration_meta,'array_sha256':ah(e0a),'carrier_sha256':ah(qa)}; out=TaskBank(session_id=bank.session_id,E0=e0a,carrier=qa,unit_mask=bank.unit_mask,X_store=bank.X_store,target_store=bank.target_store,window_ids=bank.window_ids,calibration_meta=correct_meta); bw=time.perf_counter()-t; wall=time.perf_counter()-wall0
  if not(np.isfinite(q).all() and np.isfinite(e0a).all() and np.isfinite(wall)): raise RuntimeError('non-finite calibration output or timing')
  if out.calibration_meta['array_sha256'] != ah(out.E0) or out.calibration_meta['carrier_sha256'] != ah(out.carrier): raise RuntimeError('TaskBank calibration metadata hash mismatch')
  if not(np.array_equal(out.carrier,bank.carrier) and np.array_equal(out.E0,expected_e0)):raise RuntimeError('trained runtime materialized bank mismatch')
  rows.append({'round':i+1,'T_solve_seconds':c,'trained_D42_E17_E0_seconds':e,'static_bank_seconds':bw,'warm_static_bank_wall_seconds':wall,'T_sha256':ah(out.carrier),'E0_sha256':ah(out.E0)})
 rep={'schema':'m2_joint_calibration_cost_v1','status':'PREFLIGHT_COMPLETED' if x.preflight else 'COMPLETED','scope':{'no_query_score':True,'target_bp':0,'actual_family':'trained joint D42 EMA e17, not support-stability baseline encoder','native_e0_and_u_scope':'native_e0_and_u materializes E0 and extra per-trial u together; this is actual production materializer scope, not an optimized E0-only cost'},'runtime':{'device':'cpu','affinity':sorted(os.sched_getaffinity(0)),'threads':2,'nice':os.nice(0),'max_rss_kib':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),'dtype':'float32 bank/interface'},'binding':{'run':str(RUN),'run_meta_sha256':sha(RUN/'run_meta.json'),'checkpoint':str(ck),'checkpoint_sha256':sha(ck),'epoch':epoch,'script_sha256':sha(Path(__file__)),'normalizer_sha256':sha(CACHE/'move_t4_normalizer.json'),'raw_activity_sha256':sha(CACHE/'ext4'/x.session/'calib_activity.npy'),'source_access':access,'source_nwb_sha256':opened_nwb_hashes(access),'source_hashes':source_hashes(),'profile':'native T4 M33 + trained joint encoder materialized through benchmark_cpu_m2'},'session':x.session,'parity':{'T_raw_source_equals_cached':True,'T_sha256':ah(T0),'trained_runtime_materialized_E0_sha256':ah(expected_e0), 'E0_byte_equal_trained_runtime_materialization':True},'timing_definition':'wall starts after raw M33/support objects/normalizer/trained encoder are in memory; ends immediately after T+trained E0+TaskBank. Includes static-bank metadata hashing performed during TaskBank construction; excludes post-timer parity verification, provenance file hashing, and report serialization. Component timings are diagnostic; wall is one perf_counter boundary.','rounds':rows}
 x.output.parent.mkdir(parents=True,exist_ok=True);x.output.write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n');print(x.output)
if __name__=='__main__':main()
