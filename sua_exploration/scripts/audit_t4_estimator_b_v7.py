#!/usr/bin/env python3
"""Final guarded v7 source-only CPU runner; never resolves development/formal sessions."""
from __future__ import annotations
import argparse,hashlib,json,math,os,sys
from pathlib import Path
from typing import Any
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[2];SUA=ROOT/'sua_exploration';sys.path.insert(0,str(SUA))
from scripts.audit_t4_estimator_b_v6 import prelaunch as v6_prelaunch
from scripts.audit_t4_estimator_b_v5 import load_checked
from mc_maze.t4_cross_budget_audit import load_frozen_source_development_manifest
from mc_maze.t4_estimator_b_v7 import CANDIDATES,run_candidate,winner,aggregate_cost
AUDIT=SUA/'results/sua_t4_cross_budget_source_audit_v1_20260802/source_audit.json';AUDIT_SHA='42a48b978c4c3a27f35239adcc1c196dc8f3aaadfa7d4234955b96ee0b7c119d'
MAP=('sua_exploration/mc_maze/t4_estimator_b_v7.py','sua_exploration/mc_maze/t4_estimator_b_v6.py','sua_exploration/mc_maze/t4_estimator_b_v5.py','sua_exploration/mc_maze/t4_estimator_b_v4.py','sua_exploration/mc_maze/t4_estimator_b_v3.py','sua_exploration/mc_maze/t4_estimator_b_v2.py','sua_exploration/scripts/audit_t4_estimator_b_v7.py','sua_exploration/scripts/audit_t4_estimator_b_v6.py','sua_exploration/scripts/audit_t4_estimator_b_v5.py','sua_exploration/scripts/audit_t4_estimator_b_v3.py','sua_exploration/scripts/write_t4_estimator_b_v7_prelaunch.py','sua_exploration/tests/test_t4_estimator_b_v7.py','sua_exploration/mc_maze/t4_cross_budget_audit.py','sua_exploration/mc_maze/t4_cross_budget_protocol.py','sua_exploration/mc_maze/t4_cross_budget_features.py','sua_exploration/mc_maze/unit_side_features.py','sua_exploration/mc_maze/multisession_datamodule.py','sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json','sua_exploration/results/sua_t4_cross_budget_source_audit_v1_20260802/source_audit.json')
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
 r=v6_prelaunch();r.update({'schema':'t4_estimator_b_v7_prelaunch_v1','status':'prelaunch_only_no_gpu','supersedes':{'v6_status':'NO_GO','reason':'v6 lacked a v6-sealed source-only execution path'},'execution':{'guard_env':'T4_ESTIMATOR_B_V7_REVIEWED_SOURCE_ONLY=YES','scope':'exactly_27_source_sessions_only','aggregate_schema':'t4_estimator_b_v7_source_only_cpu_audit_v1','atomic_fresh_output':True,'automatic_gpu':False},'source_map':smap()});return r
def verify(receipt):
 actual=dict(receipt);actual.pop('writer_sha256',None)
 if actual!=prelaunch():raise ValueError('canonical prelaunch receipt drift/tamper')
 return True
def resolve_sources(receipt,data):
 base=(Path(data)/'sub-C').resolve();out={}
 for n in receipt['source_sessions']:
  p=(base/f'{n}_behavior+ecephys.nwb').resolve()
  if p.parent!=base or not p.is_file():raise FileNotFoundError(p)
  out[n]=p
 return out
def execute_source_only(data_dir:Path,output_dir:Path):
 if output_dir.exists():raise FileExistsError(output_dir)
 receipt=prelaunch();verify(receipt)
 old=json.loads(AUDIT.read_text())['source_train_sessions']; paths=resolve_sources(receipt,data_dir);loaded=[load_checked(n,paths[n],old[n]) for n in receipt['source_sessions']];sessions=[x[0] for x in loaded]
 results={c:run_candidate(sessions,c) for c in CANDIDATES}
 payload={'schema':'t4_estimator_b_v7_source_only_cpu_audit_v1','status':'completed_cpu_only','no_gpu':True,'no_development_opened':True,'no_formal_opened':True,'prelaunch':receipt,'source_trial_receipts':{x[0].name:x[1] for x in loaded},'candidates':{k:{**v,'cost_aggregate':aggregate_cost(v['rows'])} for k,v in results.items()},'winner':winner(results),'raw_counts_persisted':False}
 output_dir.mkdir(parents=True);tmp=output_dir/'aggregate.tmp';tmp.write_text(json.dumps(strict(payload),indent=2,sort_keys=True));tmp.replace(output_dir/'aggregate.json');return payload
def main():
 p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true');p.add_argument('--verify-receipt',type=Path);p.add_argument('--execute-source-only',action='store_true');p.add_argument('--data-dir',type=Path);p.add_argument('--output-dir',type=Path);a=p.parse_args()
 if sum([a.dry_run,a.verify_receipt is not None,a.execute_source_only])!=1:p.error('choose one')
 if a.verify_receipt:verify(json.loads(a.verify_receipt.read_text()));print('verified');return
 if a.dry_run:print(json.dumps(strict(prelaunch()),indent=2,sort_keys=True));return
 if os.getenv('T4_ESTIMATOR_B_V7_REVIEWED_SOURCE_ONLY')!='YES' or not a.data_dir or not a.output_dir:p.error('explicit guard/data/output required')
 execute_source_only(a.data_dir,a.output_dir)
if __name__=='__main__':main()
