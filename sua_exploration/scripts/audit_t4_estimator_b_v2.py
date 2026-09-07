#!/usr/bin/env python3
"""v2 prelaunch/verifier and guarded source-only CPU entrypoint; never resolves dev/formal."""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys
from pathlib import Path
from typing import Any
os.environ["CUDA_VISIBLE_DEVICES"]=""
ROOT=Path(__file__).resolve().parents[2]; SUA=ROOT/"sua_exploration"; sys.path.insert(0,str(SUA))
from mc_maze.t4_cross_budget_audit import load_frozen_source_development_manifest, pool_trial_count_matrix_from_receipt
from mc_maze.t4_cross_budget_protocol import canonical_direction_indices
from mc_maze.multisession_datamodule import list_datamodule_rewarded_trials
from mc_maze.t4_estimator_b_v2 import CANDIDATES, N_SOURCE, SCORE_STOP, Session, run_candidate

AUDIT=SUA/"results/sua_t4_cross_budget_source_audit_v1_20260802/source_audit.json"; AUDIT_SHA="42a48b978c4c3a27f35239adcc1c196dc8f3aaadfa7d4234955b96ee0b7c119d"
MANIFEST=SUA/"configs/subc_co_27_6_strict_train_val_manifest.json"
SOURCE_MAP=("sua_exploration/mc_maze/t4_estimator_b_v2.py","sua_exploration/scripts/audit_t4_estimator_b_v2.py","sua_exploration/scripts/write_t4_estimator_b_v2_prelaunch.py","sua_exploration/tests/test_t4_estimator_b_v2.py","sua_exploration/mc_maze/t4_cross_budget_audit.py","sua_exploration/mc_maze/t4_cross_budget_protocol.py","sua_exploration/mc_maze/unit_side_features.py","sua_exploration/mc_maze/multisession_datamodule.py","sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json","sua_exploration/results/sua_t4_cross_budget_source_audit_v1_20260802/source_audit.json")

def sha(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def strict(x:Any):
    if x is None or isinstance(x,(str,bool,int)): return x
    if isinstance(x,(float,)): return x if math.isfinite(x) else None
    if hasattr(x,"tolist"): return strict(x.tolist())
    if isinstance(x,dict): return {str(k):strict(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [strict(v) for v in x]
    raise TypeError(type(x))
def source_map(): return {name:sha(ROOT/name) for name in SOURCE_MAP}
def prelaunch(manifest:Path=MANIFEST):
    if sha(AUDIT)!=AUDIT_SHA: raise ValueError("source-audit hash drift")
    m=load_frozen_source_development_manifest(manifest); train=list(m["nested_source_session_names"])
    if len(train)!=N_SOURCE: raise ValueError("not exact 27 sources")
    return {"schema":"t4_estimator_b_v2_prelaunch_v1","mode":"no_data_prelaunch","supersedes":{"v1_status":"NO_GO","reasons":["candidate likelihood rows were not uniformly equal-per-direction","EB directional prior mean was unconstrained","per-fold IRLS/undefined/provenance receipts incomplete"]},"no_gpu":True,"no_dataset_opened":True,"no_development_opened":True,"no_formal_opened":True,"cuda_visible_devices":"","source_sessions":train,"sealed_development_names":list(m["development_validation_names"]),"sealed_formal_names":list(m["sealed_formal_test_names"]),"chronology":{"total_rewarded_trials":50,"support":[0,30],"score":[30,50],"real_trial_ordinals_required":True,"support_score_disjoint":True},"fairness":{"ordinary":"equal_per_direction_mean","eb":"equal_unique_direction_rate_rows","second_harmonic":"equal_unique_direction_rate_rows","poisson":"direction_aggregated_counts_exposure_then_equal_direction_normalized"},"eb":{"mean":"[mu_b,0,0]","covariance":"full_outer26_only","grid":[.25,1.,4.],"inner_receipt":"all 26 per-lambda ratios plus deterministic tie-break"},"poisson":{"max_iter":32,"tol":1e-7,"eta_clip":[-12.,12.],"unit_iteration_and_failure_receipt":True},"reliability":{"seeds":[1201,1207,1213,1217,1223,1229,1231,1237],"undefined":"fail_closed"},"gate":{"complete_folds_required":"27/27 ratio and reliability defined","mean_ratio_lte":.98,"mean_reliability_delta_gte":.02,"joint_nonworse_gte":"20/27","no_rank_nonconvergence_invalid_increase":True,"failfast":"only when joint failures >7; never winner"},"source_audit":{"path":str(AUDIT.resolve()),"sha256":AUDIT_SHA,"q_unit_plus_M":"not reused"},"source_map":source_map()}
def verify(receipt:dict):
    observed=source_map()
    if receipt.get("source_map")!=observed: raise ValueError("source-map tamper/drift")
    if receipt.get("source_audit",{}).get("sha256")!=sha(AUDIT): raise ValueError("source audit mismatch")
    if receipt.get("no_development_opened") is not True or receipt.get("no_formal_opened") is not True: raise ValueError("scope drift")
    return True
def resolve_sources(receipt:dict,data:Path):
    base=(data/"sub-C").resolve(); out={}
    for n in receipt["source_sessions"]:
        p=(base/f"{n}_behavior+ecephys.nwb").resolve()
        if p.parent!=base or not p.is_file(): raise FileNotFoundError(p)
        out[n]=p
    return out
def load_checked(name:str,path:Path,expected:dict):
    if sha(path)!=expected["source_sha256"]: raise ValueError(f"{name}: source SHA mismatch")
    trials=list_datamodule_rewarded_trials(path,bin_size_ms=20,window_size=50,trial_result_filter="R")[:50]
    starts=__import__('numpy').asarray([float(x['start_time']) for x in trials]); stops=__import__('numpy').asarray([float(x['stop_time']) for x in trials]); dirs=canonical_direction_indices([x.get('target_dir') for x in trials])
    old=expected["chronology_receipt"]
    if len(trials)!=50 or not __import__('numpy').array_equal(starts,__import__('numpy').asarray(old['start_times'])) or not __import__('numpy').array_equal(stops,__import__('numpy').asarray(old['stop_times'])) or not __import__('numpy').array_equal(dirs,__import__('numpy').asarray(old['snapped_direction_indices'])): raise ValueError(f"{name}: first50 chronology mismatch")
    counts=pool_trial_count_matrix_from_receipt(path,trial_start_times=starts,trial_stop_times=stops,signal_view="sua")
    return Session(name,counts,stops-starts,dirs,__import__('numpy').asarray(old['ordinals']),sha(path)),{"source_sha256":sha(path),"trial_ordinals":old['ordinals'],"start_times":starts,"stop_times":stops,"snapped_direction_indices":dirs,"verified_against_step2a_source_audit":True}
def main():
    p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true');p.add_argument('--verify-receipt',type=Path);p.add_argument('--execute-source-only',action='store_true');p.add_argument('--data-dir',type=Path);p.add_argument('--output-dir',type=Path);a=p.parse_args()
    if int(a.dry_run)+int(a.verify_receipt is not None)+int(a.execute_source_only)!=1:p.error('one mode')
    if a.verify_receipt: verify(json.loads(a.verify_receipt.read_text())); print('verified');return
    r=prelaunch()
    if a.dry_run: print(json.dumps(strict(r),indent=2,sort_keys=True));return
    if os.getenv('T4_ESTIMATOR_B_V2_REVIEWED_SOURCE_ONLY')!='YES' or not a.data_dir or not a.output_dir:p.error('guard/data/output required')
    verify(r); old=json.loads(AUDIT.read_text())['source_train_sessions']; paths=resolve_sources(r,a.data_dir); loaded=[load_checked(n,paths[n],old[n]) for n in r['source_sessions']]; sessions=[x[0] for x in loaded]; trial_receipts={s.name:receipt for s,receipt in loaded}
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    result={c:run_candidate(sessions,c) for c in CANDIDATES}; a.output_dir.mkdir(parents=True); (a.output_dir/'aggregate.json').write_text(json.dumps(strict({'prelaunch':r,'verified_actual_source_trial_receipts':trial_receipts,'result':result,'raw_counts_persisted':False}),indent=2,sort_keys=True))
if __name__=='__main__':main()
