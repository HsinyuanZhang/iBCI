#!/usr/bin/env python3
"""v6 canonical prelaunch verifier: any frozen-value drift fails closed."""
from __future__ import annotations
import argparse,hashlib,json,math,os,sys
from pathlib import Path
from typing import Any
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[2];SUA=ROOT/'sua_exploration';sys.path.insert(0,str(SUA))
from scripts.audit_t4_estimator_b_v5 import prelaunch as v5_prelaunch
MAP=('sua_exploration/mc_maze/t4_estimator_b_v6.py','sua_exploration/mc_maze/t4_estimator_b_v5.py','sua_exploration/mc_maze/t4_estimator_b_v4.py','sua_exploration/mc_maze/t4_estimator_b_v3.py','sua_exploration/mc_maze/t4_estimator_b_v2.py','sua_exploration/scripts/audit_t4_estimator_b_v6.py','sua_exploration/scripts/audit_t4_estimator_b_v5.py','sua_exploration/scripts/audit_t4_estimator_b_v3.py','sua_exploration/scripts/write_t4_estimator_b_v6_prelaunch.py','sua_exploration/tests/test_t4_estimator_b_v6.py','sua_exploration/mc_maze/t4_cross_budget_audit.py','sua_exploration/mc_maze/t4_cross_budget_protocol.py','sua_exploration/mc_maze/t4_cross_budget_features.py','sua_exploration/mc_maze/unit_side_features.py','sua_exploration/mc_maze/multisession_datamodule.py','sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json','sua_exploration/results/sua_t4_cross_budget_source_audit_v1_20260802/source_audit.json')
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
 r=v5_prelaunch();r.update({'schema':'t4_estimator_b_v6_prelaunch_v1','status':'prelaunch_only_no_gpu','supersedes':{'v5_status':'NO_GO','reasons':['directly imported audit_t4_estimator_b_v3.py omitted from source map','verifier checked presence but not frozen contract values']},'source_map':smap()});return r
def canonical_receipt(): return prelaunch()
def verify(receipt):
 actual=dict(receipt);actual.pop('writer_sha256',None)
 expected=canonical_receipt()
 if actual!=expected:raise ValueError('canonical prelaunch receipt drift/tamper')
 return True
def main():
 p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true');p.add_argument('--verify-receipt',type=Path);a=p.parse_args()
 if int(a.dry_run)+int(a.verify_receipt is not None)!=1:p.error('choose one')
 if a.verify_receipt:verify(json.loads(a.verify_receipt.read_text()));print('verified');return
 print(json.dumps(strict(prelaunch()),indent=2,sort_keys=True))
if __name__=='__main__':main()
