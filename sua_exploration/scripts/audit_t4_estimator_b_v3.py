#!/usr/bin/env python3
"""v3 no-data receipt/verifier and explicitly guarded source-only CPU execution."""
from __future__ import annotations
import argparse,hashlib,json,math,os,sys,tempfile
from pathlib import Path
from typing import Any
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[2];SUA=ROOT/'sua_exploration';sys.path.insert(0,str(SUA))
from mc_maze.t4_cross_budget_audit import load_frozen_source_development_manifest,pool_trial_count_matrix_from_receipt
from mc_maze.t4_cross_budget_protocol import canonical_direction_indices
from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
from mc_maze.t4_estimator_b_v3 import CANDIDATES,N_SOURCE,Session,run_candidate,winner,aggregate_cost
AUDIT=SUA/'results/sua_t4_cross_budget_source_audit_v1_20260802/source_audit.json';AUDIT_SHA='42a48b978c4c3a27f35239adcc1c196dc8f3aaadfa7d4234955b96ee0b7c119d';MANIFEST=SUA/'configs/subc_co_27_6_strict_train_val_manifest.json'
MAP=('sua_exploration/mc_maze/t4_estimator_b_v3.py','sua_exploration/mc_maze/t4_estimator_b_v2.py','sua_exploration/scripts/audit_t4_estimator_b_v3.py','sua_exploration/scripts/write_t4_estimator_b_v3_prelaunch.py','sua_exploration/tests/test_t4_estimator_b_v3.py','sua_exploration/mc_maze/t4_cross_budget_audit.py','sua_exploration/mc_maze/t4_cross_budget_protocol.py','sua_exploration/mc_maze/unit_side_features.py','sua_exploration/mc_maze/multisession_datamodule.py','sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json','sua_exploration/results/sua_t4_cross_budget_source_audit_v1_20260802/source_audit.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def strict(x:Any):
 if x is None or isinstance(x,(str,bool,int)):return x
 if isinstance(x,float):return x if math.isfinite(x) else None
 if hasattr(x,'tolist'):return strict(x.tolist())
 if isinstance(x,dict):return {str(k):strict(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [strict(v) for v in x]
 raise TypeError(type(x))
def source_map():return {p:sha(ROOT/p) for p in MAP}
def prelaunch():
 if sha(AUDIT)!=AUDIT_SHA:raise ValueError('audit hash drift')
 m=load_frozen_source_development_manifest(MANIFEST); train=list(m['nested_source_session_names'])
 if len(train)!=N_SOURCE:raise ValueError('not 27')
 return {'schema':'t4_estimator_b_v3_prelaunch_v1','status':'prelaunch_only_no_gpu','supersedes':{'v2_status':'NO_GO','reasons':['top-level winner/tie semantics absent','cost/state proxies absent from aggregate tie-break','EB covariance centred at empirical directional mean','fixed candidates performed unnecessary inner selection','actual trial_index ordinals not verified','top-level atomic output contract incomplete']},'no_gpu':True,'no_dataset_opened':True,'no_development_opened':True,'no_formal_opened':True,'source_sessions':train,'sealed_development_names':list(m['development_validation_names']),'sealed_formal_names':list(m['sealed_formal_test_names']),'chronology':{'trials':50,'support':[0,30],'score':[30,50],'actual_trial_index_required':True},'winner':{'eligible':'gate.pass_only','order':['lower_mean_ratio','lower_calibration_ops_proxy','smaller_persistent_state_bytes'],'exact_tie':'no_winner','zero_pass':'no_winner','success_status':'unique_cpu_winner_requires_new_gpu_prelaunch','automatic_gpu':False},'cost':{'descriptor':'4 float32/unit','actual_poisson_iterations_determine_ops':True,'wall_clock_components':['ordinary','candidate','reliability','inner_selection']},'runtime_work_receipt':{'source_reads':27,'EB_inner_prospective_fits':27*26*3,'EB_inner_reliability_fits':0,'outer_fits_per_candidate':27,'outer_reliability_candidate_fits_per_candidate':27*8*2,'estimate_note':'wall-clock is dataset/unit-count dependent and is measured in final aggregate; this is a fixed fit-count budget only'},'source_audit':{'path':str(AUDIT.resolve()),'sha256':AUDIT_SHA,'q_unit_plus_M':'not reused'},'source_map':source_map()}
def verify(r):
 if r.get('source_map')!=source_map():raise ValueError('source-map tamper/drift')
 if r.get('source_audit',{}).get('sha256')!=sha(AUDIT):raise ValueError('audit drift')
 if not(r.get('no_development_opened') and r.get('no_formal_opened') and r.get('no_gpu')):raise ValueError('scope drift')
 return True
def resolve_sources(r,data):
 base=(Path(data)/'sub-C').resolve();out={}
 for n in r['source_sessions']:
  p=(base/f'{n}_behavior+ecephys.nwb').resolve()
  if p.parent!=base or not p.is_file():raise FileNotFoundError(p)
  out[n]=p
 return out
def load_checked(name,path,expected):
 import numpy as np
 if sha(path)!=expected['source_sha256']:raise ValueError(f'{name}: source SHA mismatch')
 trials=list_datamodule_rewarded_trials(path,bin_size_ms=20,window_size=50,trial_result_filter='R')[:50]
 if len(trials)!=50 or any('trial_index' not in x for x in trials):raise ValueError(f'{name}: actual trial_index missing')
 starts=np.asarray([float(x['start_time']) for x in trials]);stops=np.asarray([float(x['stop_time']) for x in trials]);dirs=canonical_direction_indices([x.get('target_dir') for x in trials]);ordinals=np.asarray([int(x['trial_index']) for x in trials])
 old=expected['chronology_receipt']
 if not(np.array_equal(ordinals,np.asarray(old['ordinals'])) and np.array_equal(starts,np.asarray(old['start_times'])) and np.array_equal(stops,np.asarray(old['stop_times'])) and np.array_equal(dirs,np.asarray(old['snapped_direction_indices']))):raise ValueError(f'{name}: actual trial receipt mismatch')
 counts=pool_trial_count_matrix_from_receipt(path,trial_start_times=starts,trial_stop_times=stops,signal_view='sua');s=Session(name,counts,stops-starts,dirs,ordinals,sha(path));return s,{'source_sha256':sha(path),'actual_trial_ordinals':ordinals,'actual_start_times':starts,'actual_stop_times':stops,'actual_direction_indices':dirs,'verified_against_step2a':True}
def execute(data,out):
 if out.exists():raise FileExistsError(out)
 r=prelaunch();verify(r);old=json.loads(AUDIT.read_text())['source_train_sessions'];paths=resolve_sources(r,data);loaded=[load_checked(n,paths[n],old[n]) for n in r['source_sessions']];sessions=[x[0] for x in loaded];results={c:run_candidate(sessions,c) for c in CANDIDATES}; payload={'schema':'t4_estimator_b_v3_source_only_cpu_audit_v1','status':'completed_cpu_only','no_gpu':True,'no_development_opened':True,'no_formal_opened':True,'prelaunch':r,'source_trial_receipts':{x[0].name:x[1] for x in loaded},'candidates':{k:{**v,'cost_aggregate':aggregate_cost(v['rows'])} for k,v in results.items()},'winner':winner(results),'raw_counts_persisted':False}
 out.mkdir(parents=True); tmp=out/'aggregate.tmp';tmp.write_text(json.dumps(strict(payload),indent=2,sort_keys=True));tmp.replace(out/'aggregate.json');return payload
def main():
 p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true');p.add_argument('--verify-receipt',type=Path);p.add_argument('--execute-source-only',action='store_true');p.add_argument('--data-dir',type=Path);p.add_argument('--output-dir',type=Path);a=p.parse_args()
 if sum([a.dry_run,a.verify_receipt is not None,a.execute_source_only])!=1:p.error('choose one')
 if a.verify_receipt:verify(json.loads(a.verify_receipt.read_text()));print('verified');return
 if a.dry_run:print(json.dumps(strict(prelaunch()),indent=2,sort_keys=True));return
 if os.getenv('T4_ESTIMATOR_B_V3_REVIEWED_SOURCE_ONLY')!='YES' or not a.data_dir or not a.output_dir:p.error('explicit guard/data/output required')
 execute(a.data_dir,a.output_dir)
if __name__=='__main__':main()
