#!/usr/bin/env python3
"""v5 prelaunch retains v3 contract verbatim and adds v4 dual ordinal semantics."""
from __future__ import annotations
import argparse,hashlib,json,math,os,sys
from pathlib import Path
from typing import Any
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[2];SUA=ROOT/'sua_exploration';sys.path.insert(0,str(SUA))
from scripts.audit_t4_estimator_b_v3 import prelaunch as v3_prelaunch
from mc_maze.t4_cross_budget_audit import pool_trial_count_matrix_from_receipt
from mc_maze.t4_cross_budget_protocol import canonical_direction_indices
from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
from mc_maze.t4_estimator_b_v5 import CANDIDATES,Session,run_candidate,winner,aggregate_cost,ORDINAL_SEMANTICS_VERSION
AUDIT=SUA/'results/sua_t4_cross_budget_source_audit_v1_20260802/source_audit.json';AUDIT_SHA='42a48b978c4c3a27f35239adcc1c196dc8f3aaadfa7d4234955b96ee0b7c119d'
MAP=('sua_exploration/mc_maze/t4_estimator_b_v5.py','sua_exploration/mc_maze/t4_estimator_b_v4.py','sua_exploration/mc_maze/t4_estimator_b_v3.py','sua_exploration/mc_maze/t4_estimator_b_v2.py','sua_exploration/scripts/audit_t4_estimator_b_v5.py','sua_exploration/scripts/write_t4_estimator_b_v5_prelaunch.py','sua_exploration/tests/test_t4_estimator_b_v5.py','sua_exploration/mc_maze/t4_cross_budget_audit.py','sua_exploration/mc_maze/t4_cross_budget_protocol.py','sua_exploration/mc_maze/t4_cross_budget_features.py','sua_exploration/mc_maze/unit_side_features.py','sua_exploration/mc_maze/multisession_datamodule.py','sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json','sua_exploration/results/sua_t4_cross_budget_source_audit_v1_20260802/source_audit.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def strict(x:Any):
 if x is None or isinstance(x,(str,bool,int)):return x
 if isinstance(x,float):return x if math.isfinite(x) else None
 if hasattr(x,'tolist'):return strict(x.tolist())
 if isinstance(x,dict):return {str(k):strict(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [strict(v) for v in x]
 raise TypeError(type(x))
def smap():return {p:sha(ROOT/p) for p in MAP}
def prelaunch():
 # Exact v3 contract first: chronology, winner, cost, runtime and N_SOURCE validation survive.
 base=v3_prelaunch()
 if len(base['source_sessions'])!=27:raise ValueError('N_SOURCE source manifest drift')
 base.update({'schema':'t4_estimator_b_v5_prelaunch_v1','status':'prelaunch_only_no_gpu','supersedes':{'v4_status':'NO_GO','reason':'v4 dual ordinal reconciliation dropped material v3 frozen chronology/winner/cost/runtime contract fields'},'ordinal_provenance':{'version':ORDINAL_SEMANTICS_VERSION,'historical_field':'selected_rewarded_pool_ordinal (Step-2A; exactly 0..49)','current_field':'raw_NWB_trial_table_trial_index','identity_binding':['source_sha256','selected_rewarded_ordinal','start_time','stop_time','snapped_direction_index'],'raw_trial_index':'required, strictly increasing, stored, and deliberately not equality-compared to selected-pool ordinal'},'source_map':smap()})
 return base
def verify(r):
 if r.get('source_map')!=smap():raise ValueError('source-map tamper/drift')
 if len(r.get('source_sessions',[]))!=27:raise ValueError('N_SOURCE contract drift')
 for key in ('chronology','winner','cost','runtime_work_receipt'):
  if key not in r:raise ValueError(f'missing retained v3 contract: {key}')
 if r.get('source_audit',{}).get('sha256')!=sha(AUDIT):raise ValueError('source audit drift')
 if not(r.get('no_gpu') and r.get('no_development_opened') and r.get('no_formal_opened')):raise ValueError('scope drift')
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
 if len(trials)!=50 or any('trial_index' not in t for t in trials):raise ValueError(f'{name}: trial_index missing')
 starts=np.asarray([float(t['start_time']) for t in trials]);stops=np.asarray([float(t['stop_time']) for t in trials]);dirs=canonical_direction_indices([t.get('target_dir') for t in trials]);raw=np.asarray([int(t['trial_index']) for t in trials]);selected=np.arange(50,dtype=np.int64);old=expected['chronology_receipt']
 if not np.array_equal(np.asarray(old['ordinals']),selected):raise ValueError(f'{name}: historical ordinal semantics drift')
 if np.any(np.diff(raw)<=0):raise ValueError(f'{name}: raw trial_index is not strictly increasing')
 if not np.array_equal(starts,np.asarray(old['start_times'])):raise ValueError(f'{name}: selected start mismatch')
 if not np.array_equal(stops,np.asarray(old['stop_times'])):raise ValueError(f'{name}: selected stop mismatch')
 if not np.array_equal(dirs,np.asarray(old['snapped_direction_indices'])):raise ValueError(f'{name}: selected direction mismatch')
 counts=pool_trial_count_matrix_from_receipt(path,trial_start_times=starts,trial_stop_times=stops,signal_view='sua');s=Session(name,counts,stops-starts,dirs,selected,sha(path));return s,{'source_sha256':sha(path),'historical_selected_rewarded_ordinals':selected,'actual_raw_nwb_trial_indices':raw,'actual_start_times':starts,'actual_stop_times':stops,'actual_direction_indices':dirs,'ordinal_semantics':ORDINAL_SEMANTICS_VERSION,'verified_against_step2a_identity_fields':True}
def execute(data,out):
 if out.exists():raise FileExistsError(out)
 r=prelaunch();verify(r);old=json.loads(AUDIT.read_text())['source_train_sessions'];paths=resolve_sources(r,data);loaded=[load_checked(n,paths[n],old[n]) for n in r['source_sessions']];ss=[x[0] for x in loaded];results={c:run_candidate(ss,c) for c in CANDIDATES};payload={'schema':'t4_estimator_b_v5_source_only_cpu_audit_v1','status':'completed_cpu_only','no_gpu':True,'no_development_opened':True,'no_formal_opened':True,'prelaunch':r,'source_trial_receipts':{x[0].name:x[1] for x in loaded},'candidates':{k:{**v,'cost_aggregate':aggregate_cost(v['rows'])} for k,v in results.items()},'winner':winner(results),'raw_counts_persisted':False}
 out.mkdir(parents=True);tmp=out/'aggregate.tmp';tmp.write_text(json.dumps(strict(payload),indent=2,sort_keys=True));tmp.replace(out/'aggregate.json')
def main():
 p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true');p.add_argument('--verify-receipt',type=Path);p.add_argument('--execute-source-only',action='store_true');p.add_argument('--data-dir',type=Path);p.add_argument('--output-dir',type=Path);a=p.parse_args()
 if sum([a.dry_run,a.verify_receipt is not None,a.execute_source_only])!=1:p.error('choose one')
 if a.verify_receipt:verify(json.loads(a.verify_receipt.read_text()));print('verified');return
 if a.dry_run:print(json.dumps(strict(prelaunch()),indent=2,sort_keys=True));return
 if os.getenv('T4_ESTIMATOR_B_V5_REVIEWED_SOURCE_ONLY')!='YES' or not a.data_dir or not a.output_dir:p.error('explicit guard/data/output required')
 execute(a.data_dir,a.output_dir)
if __name__=='__main__':main()
