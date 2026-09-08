#!/usr/bin/env python3
"""Warm all-27 H1 582073 calibration cost after functional plan replay proof."""
from __future__ import annotations
import argparse, hashlib, io, json, os, pickle, sys, time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import numpy as np

WS=Path(__file__).resolve().parents[3]; ROOT=WS/'btransform_unified_v2'
HIST=Path('/home/xinyuan/Work_host/ibci_c3_film/SPINT-main')
AUTH=Path('/home/xinyuan/Work_host/ibci_c3_film/tfpd_exploration/h1_series_20260830/results/h1_cal_aug_all_source_m3_deployment_v1/source_authority')
DEFAULT_PLAN=ROOT/'results/diagnostics_v1/h1_source_plan_reconstruct_probe_v1/replayed_plan_arrays.npz'
DEFAULT_GATE=ROOT/'results/diagnostics_v1/h1_source_plan_replay_verify27_v1/report.json'
SOURCE=WS/'SPINT-main/local_data/h1_epfilm_evalai_v1/decoder.pt'
RIFT=WS/'tfpd_exploration/submissions/evalai_h1_rift_r300_cached_v1/artifacts/h1_rift_r300_recency_e22.pkl'

def need(x:bool,m:str)->None:
 if not x: raise RuntimeError(m)
def fsha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def asha(a:np.ndarray)->str:return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def utc()->str:return datetime.now(timezone.utc).isoformat()
def cpu_model()->str:
 for line in Path('/proc/cpuinfo').read_text().splitlines():
  if line.startswith('model name'):return line.split(':',1)[1].strip()
 return 'unknown'

def historical():
 sys.path.insert(0,str(HIST))
 from src.data import h1_m4_eb_pilot as pilot
 from src.data.h1_cal_aug_all_source_heldout_v1 import index_heldout_calib
 need(Path(pilot.__file__).resolve().is_relative_to(HIST),'historical pilot import escaped')
 return pilot,index_heldout_calib

def heldout_m3(path:Path,pilot:Any)->Any:
 """Literal public-M3 semantics of h1_m3_readout_calibration_v1/package.py::_load_heldout."""
 from falcon_challenge.config import FalconTask
 from falcon_challenge.dataloaders import load_nwb
 from pynwb import NWBHDF5IO
 neural,velocity,change,mask=load_nwb(path,FalconTask.h1)
 with NWBHDF5IO(str(path),'r',load_namespaces=True) as x: nums=np.asarray(x.read().acquisition['TrialNum'].data[:],np.float64)
 spikes64=np.asarray(neural,np.float64); targets64=np.asarray(velocity,np.float64); mask=np.asarray(mask,bool).reshape(-1)
 ordered=nums[mask&np.isfinite(nums)]; need(ordered.size>0 and not np.any(np.diff(ordered)<0),'heldout TrialNum order')
 vals=[]
 for v in ordered.tolist():
  if not vals or float(v)!=vals[-1]:vals.append(float(v))
 need(len(vals)==3,'heldout calibration is not exact M3')
 return pilot.H1PilotRecord(pilot.session_from_path(path),pilot.session_date(pilot.session_from_path(path)),path.resolve(),fsha(path),spikes64.astype(np.float32),targets64.astype(np.float32),np.asarray(change,bool).reshape(-1),mask,nums,tuple(vals),tuple(pilot._trial_blocks(v,spikes64,targets64,mask,nums) for v in vals))

def rift_payload(path:Path)->dict:
 import torch
 class U(pickle.Unpickler):
  def find_class(self,m,n):
   if m.startswith('numpy._core'):m=m.replace('numpy._core','numpy.core',1)
   if m=='torch.storage' and n=='_load_from_bytes':return lambda x:torch.load(io.BytesIO(x),map_location='cpu',weights_only=False)
   return super().find_class(m,n)
 with path.open('rb') as f:return U(f).load()

def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--replayed-plan',type=Path,default=DEFAULT_PLAN);ap.add_argument('--verify27',type=Path,default=DEFAULT_GATE);ap.add_argument('--dest',type=Path,required=True);ap.add_argument('--rounds',type=int,default=3);a=ap.parse_args()
 need(not a.dest.exists(),f'destination must be fresh: {a.dest}');need(a.rounds==3,'require exactly 3 warm rounds')
 need(os.environ.get('CUDA_VISIBLE_DEVICES') in (None,''),'CPU-only requires CUDA_VISIBLE_DEVICES empty');need(sorted(os.sched_getaffinity(0))==[14,15],'launch with taskset -c 14,15');need(all(os.environ.get(k)=='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'set OMP/MKL/OPENBLAS threads to 2')
 import torch
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 gate=json.loads(a.verify27.read_text());need(gate.get('status')=='PASS_ALL_27_FUNCTIONAL_REPLAY_T_BYTE_EQUAL','verified all-27 functional replay is required')
 need(fsha(a.replayed_plan)==gate['inputs']['replayed_plan']['container_sha256'],'replayed plan differs from verify27 gate')
 need(fsha(SOURCE)==gate['inputs']['source_payload']['sha256'],'source payload differs from verify27 gate')
 need(fsha(RIFT)==gate['inputs']['frozen_rift_payload']['sha256'],'frozen RIFT payload differs from verify27 gate')
 tags=gate['tags'];need(len(tags)==27 and all(v['replayed_vs_source']['byte_equal'] and v['replayed_vs_frozen']['byte_equal'] for v in tags.values()),'verify27 gate rows incomplete')
 with np.load(a.replayed_plan,allow_pickle=False) as z:
  need(set(z.files)=={'mean','scale','pcs','U','mu','tau2','q','lambda'},'replayed plan schema')
  plan=SimpleNamespace(mean=np.asarray(z['mean'],np.float64),scale=np.asarray(z['scale'],np.float64),pcs=np.asarray(z['pcs'],np.float64),U=np.asarray(z['U'],np.float64),mu=np.asarray(z['mu'],np.float64),tau2=float(z['tau2']),q=int(z['q']),ridge_lambda=float(z['lambda']))
 need(plan.q==12 and plan.ridge_lambda==10.,'plan is not q12/lambda10')
 norm=json.loads((AUTH/'normalizer.json').read_text());s_src=float(norm['s_src'])
 need(s_src==float(gate['inputs']['source_rms']),'source RMS differs from verify27 gate')
 pilot,index_heldout=historical(); source=torch.load(SOURCE,map_location='cpu',weights_only=False); frozen=rift_payload(RIFT)['bank_by_dataset_tag']
 heldin=pilot.index_heldin_calib(HIST/'data/000954');heldout=index_heldout(HIST/'data/000954')
 # Everything below is preload, outside all warm timing scopes.
 prepared={}
 for tag,row in source['sessions'].items():
  session=str(row['session']); trials=tuple(float(v) for v in row['calibration_trials']);need(len(trials)==3,'source payload trial budget')
  path=heldin.get(session) or heldout.get(session);need(path is not None,f'{tag}: calibration path absent')
  path=Path(path).resolve(); gate_row=tags[tag]
  need(str(path)==str(Path(gate_row['public_input']['path']).resolve()),f'{tag}: public path differs from verify27 gate')
  need(fsha(path)==gate_row['public_input']['sha256'],f'{tag}: public NWB differs from verify27 gate')
  need(list(trials)==list(gate_row['trials']),f'{tag}: public M3 trials differ from verify27 gate')
  rec=pilot.load_record(path) if session in heldin else heldout_m3(path,pilot)
  need(tuple(float(v) for v in rec.trial_values[:3])==trials,f'{tag}: trial contract drift')
  activity=np.ascontiguousarray(np.stack([pilot.interpolate_trial_identity(rec,v) for v in trials]),np.float32)
  prepared[tag]={'record':rec,'activity':activity,'path':path,'session':session,'trials':trials,'expected_t':np.ascontiguousarray(frozen[tag]['T'],np.float32),'expected_e0':np.ascontiguousarray(frozen[tag]['E0'],np.float32)}
 need(set(prepared)==set(tags)==set(frozen),'27-tag preload roster drift')
 sys.path.insert(0,str(WS));sys.path.insert(0,str(WS/'btransform_unified_v1/src'))
 from btransform_unified_v1.bank import TaskBank
 from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_calibration import load_frozen_c2_materializer
 mat=load_frozen_c2_materializer().eval().to('cpu')
 # One explicitly untimed materialization warms the frozen C2 module.  It is
 # neither a cold-start claim nor part of any of the formal 81 observations.
 warm_tag=sorted(prepared)[0]; warm=prepared[warm_tag]
 warm_raw=pilot.fit_deployment_carrier(warm['record'],plan,warm['trials'])['carrier']
 _warm_e0,_warm_direct=mat.materialize_bank(torch.from_numpy(warm['activity']),torch.from_numpy(np.ascontiguousarray(warm_raw/max(s_src,1e-12),np.float32)))
 del _warm_e0,_warm_direct,warm_raw
 per={tag:[] for tag in sorted(prepared)}
 for repeat in range(1,4):
  for tag in sorted(prepared):
   p=prepared[tag]; total0=time.perf_counter()
   t0=time.perf_counter();raw=pilot.fit_deployment_carrier(p['record'],plan,p['trials'])['carrier'];carrier=np.ascontiguousarray(raw/max(s_src,1e-12),np.float32);solve=time.perf_counter()-t0
   t0=time.perf_counter();e0,direct=mat.materialize_bank(torch.from_numpy(p['activity']),torch.from_numpy(carrier));e0=np.ascontiguousarray(e0.numpy(),np.float32);direct=np.ascontiguousarray(direct.numpy(),np.float32);embed=time.perf_counter()-t0
   t0=time.perf_counter();bank=TaskBank(session_id=tag,E0=e0,carrier=direct,unit_mask=np.ones(176,bool),X_store=np.zeros((0,300,176),np.float32),target_store=np.zeros((0,7),np.float32),window_ids=np.zeros(0,np.int64),calibration_meta={'shape':(176,700),'trial_count':3,'budget':3,'estimator':'replayed q12/lambda10 source carrier + frozen C2','array_sha256':'post_timer_identity'});bank_seconds=time.perf_counter()-t0
   total=time.perf_counter()-total0
   # Hashes/parity are deliberately after total's end point.
   timings=(solve,embed,bank_seconds,total); need(all(np.isfinite(x) and x>=0. for x in timings),'nonfinite or negative timing')
   need(total+1e-9>=solve+embed+bank_seconds,'total wall shorter than component sum')
   need(np.isfinite(carrier).all() and np.isfinite(e0).all(),'nonfinite warm bank')
   need(np.array_equal(direct,carrier),'C2 direct carrier differs from solve carrier')
   need(np.array_equal(np.asarray(bank.E0),e0) and np.array_equal(np.asarray(bank.carrier),carrier),'TaskBank differs from materialized arrays')
   per[tag].append({'round':repeat,'carrier_solve_seconds':solve,'e0_materialize_seconds':embed,'static_bank_seconds':bank_seconds,'total_warm_seconds':total,'timing_invariants_passed':True,'T_sha256':asha(carrier),'E0_sha256':asha(e0),'T_byte_equal_frozen':bool(np.array_equal(carrier,p['expected_t'])),'E0_byte_equal_frozen':bool(np.array_equal(e0,p['expected_e0'])),'direct_equals_carrier':True,'bank_array_parity':True,'bank_unit_mask_true':int(np.asarray(bank.unit_mask).sum())})
 all_parity=all(x['T_byte_equal_frozen'] and x['E0_byte_equal_frozen'] for v in per.values() for x in v)
 plan_receipt=json.loads((AUTH/'plan.json').read_text());plan_hash={n:pilot.array_sha256(getattr(plan,n)) for n in ('mean','scale','pcs','U','mu')}
 c2config=WS/'tfpd_exploration/src/two_mainlines_long_v1/decoder/h1_config.py'
 report={'schema':'h1_frozen_calibration_cost_v2','status':'COMPLETED' if all_parity else 'PARITY_FAILURE','utc':utc(),'script_sha256':fsha(Path(__file__).resolve()),'scope':{'cpu_only':True,'warm_inputs_preloaded':True,'prepared_inputs':'public calibration records and interpolated M3 [3,1024,176] activity are built before timing; raw-record loading and interpolation are excluded','target_support_labels_used':True,'query_scoring':False,'hidden_test_opened':False,'training':False,'optimizer_steps':0,'rounds':3,'tags':27},'cpu':{'model':cpu_model(),'affinity':sorted(os.sched_getaffinity(0)),'nice':os.nice(0),'torch_version':torch.__version__,'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),'omp':os.environ['OMP_NUM_THREADS'],'mkl':os.environ['MKL_NUM_THREADS'],'openblas':os.environ['OPENBLAS_NUM_THREADS'],'rss_bytes':None},'functional_replay_gate':{'path':str(a.verify27),'sha256':fsha(a.verify27),'status':gate['status'],'all_27_T_byte_equal':True,'note':'replayed FP64 plan arrays differ for pcs/U/mu from sealed authority but verified normalized FP32 T is byte-equal for all 27 tags.'},'plan':{'replayed_npz':str(a.replayed_plan),'replayed_npz_sha256':fsha(a.replayed_plan),'q':plan.q,'lambda':plan.ridge_lambda,'source_rms':s_src,'normalizer_sha256':fsha(AUTH/'normalizer.json'),'sealed_plan_receipt_sha256':fsha(AUTH/'plan.json'),'replayed_array_sha256':plan_hash,'sealed_expected_array_sha256':plan_receipt['array_sha256'],'array_hash_equal':{n:plan_hash[n]==plan_receipt['array_sha256'][n] for n in plan_hash}},'offline_not_target_cost':{'source_plan_probe_receipt':str(a.replayed_plan.parent/'report.json'),'source_plan_probe_receipt_sha256':fsha(a.replayed_plan.parent/'report.json'),'statement':'13-source plan reconstruction and raw/source-payload cold loads are pre-existing/offline and excluded from target calibration timing.'},'sources':{'source_payload':{'path':str(SOURCE),'sha256':fsha(SOURCE)},'frozen_rift_payload':{'path':str(RIFT),'sha256':fsha(RIFT)},'carrier_operator':{'path':str(Path(pilot.__file__).resolve()),'sha256':fsha(Path(pilot.__file__).resolve())},'c2_materializer':{'path':str(WS/'tfpd_exploration/src/two_mainlines_long_v1/decoder/h1_calibration.py'),'sha256':fsha(WS/'tfpd_exploration/src/two_mainlines_long_v1/decoder/h1_calibration.py'),'config_sha256':fsha(c2config),'checkpoint_sha256':mat.checkpoint_sha256}},'timer_contract':'Each total_warm_seconds uses one perf_counter boundary around carrier solve -> frozen-C2 E0 materialization -> TaskBank construction. M3 activities [3,1024,176], records, plan and C2 weights are preloaded. File/report hashes and parity checks are outside. TaskBank required array_sha256 metadata uses literal post_timer_identity; no metadata hash is computed inside timer. An untimed one-session C2 materialization occurs before all 81 formal observations.','sessions':{tag:{'session':p['session'],'public_calibration_path':str(p['path']),'public_input_sha256':fsha(p['path']),'trials':list(p['trials']),'activity_sha256':asha(p['activity']),'rounds':per[tag]} for tag,p in prepared.items()}}
 # RSS belongs after timing and never changes cost observations.
 try: report['cpu']['rss_bytes']=int(__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss)*1024
 except Exception: pass
 a.dest.mkdir(parents=True);(a.dest/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(a.dest/'report.json');return 0 if all_parity else 1
if __name__=='__main__':raise SystemExit(main())
