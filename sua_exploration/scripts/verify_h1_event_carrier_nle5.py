#!/usr/bin/env python3
"""Independently verify immutable H1 NLE5 receipts and recompute all aggregates."""
from __future__ import annotations
import argparse, hashlib, json, math, stat
from pathlib import Path
from typing import Any, Mapping
import numpy as np

SCHEMA="h1_event_carrier_nle5_source_screen_v1"; PROTOCOL="h1_event_carrier_nonlinear_label_embedding_20260812_v1"
DATES=("19250101","19250108","19250113","19250115","19250119","19250120")

def need(x: bool, m: str)->None:
    if not x: raise ValueError(m)
def sha(p: Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for x in iter(lambda:f.read(4<<20),b""):h.update(x)
    return h.hexdigest()
def close(a:Any,b:Any)->None: need(math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-12),f"numeric mismatch {a} {b}")
def summary(rows:list[tuple[str,float]])->dict[str,Any]:
    x=np.asarray([v for _,v in rows],np.float64); need(x.size==13 and np.isfinite(x).all(),"aggregate requires 13 finite rows"); i=int(np.argmax(abs(x)))
    return {"defined_sessions":13,"mean":float(x.mean()),"median":float(np.median(x)),"positive":int((x>0).sum()),"zero":int((x==0).sum()),"negative":int((x<0).sum()),"leave_largest_absolute_out_mean":float(np.delete(x,i).mean()),"removed_session":rows[i][0]}
def verify_summary(a:Mapping[str,Any],b:Mapping[str,Any])->None:
    for k in ("defined_sessions","positive","zero","negative","removed_session"):need(a.get(k)==b[k],f"summary drift {k}")
    for k in ("mean","median","leave_largest_absolute_out_mean"):close(a.get(k),b[k])
def positive(x:Mapping[str,Any])->bool:return bool(x["defined_sessions"]==13 and x["mean"]>0 and x["median"]>0 and x["positive"]>=10 and x["leave_largest_absolute_out_mean"]>0)

def verify(receipt:Path)->dict[str,Any]:
    p=receipt.resolve(); need(p.is_file() and stat.S_IMODE(p.stat().st_mode)==0o444,"receipt must be 0444")
    side=p.with_suffix(p.suffix+".sha256"); need(side.is_file() and stat.S_IMODE(side.stat().st_mode)==0o444,"receipt SHA must be 0444")
    digest=sha(p); need(side.read_text().strip()==f"{digest}  {p.name}","receipt SHA mismatch")
    x=json.loads(p.read_text()); need(x.get("schema")==SCHEMA and x.get("protocol")==PROTOCOL,"schema/protocol")
    pre=p.parent/"predeclaration_v1.json"; ps=pre.with_suffix(pre.suffix+".sha256"); need(pre.is_file() and ps.is_file() and stat.S_IMODE(pre.stat().st_mode)==stat.S_IMODE(ps.stat().st_mode)==0o444,"missing immutable predeclaration")
    pd=sha(pre); need(ps.read_text().strip()==f"{pd}  {pre.name}" and x["predeclaration_sha256"]==pd,"predeclaration SHA")
    decl=json.loads(pre.read_text()); need(decl["written_before_nwb_open"] is True and decl["frozen_model"]["carrier_dim"]==5 and decl["selection"].startswith("no candidate"),"predeclaration contract")
    c=x["frozen_constants"]; need(c["carrier_dim"]==5 and c["embedding_dim"]==4 and tuple(c["support_budgets"])==(3,4) and c["rff_width"]==16 and c["rff_seed"]==190271 and c["single_model_no_outer_expanded_grid"] is True,"frozen model drift")
    scope=x["scope"]
    expected={"public_held_in_calibration_nwbs_opened":13,"minival_nwbs_opened":0,"held_out_nwbs_opened":0,"formal_test_labels_opened":0,"dense_velocity_opened":False,"within_event_position_trajectory_opened":False,"target_session_optimizer_steps":0,"target_session_backward_steps":0,"decoder_constructed":False,"trainer_constructed":False,"cuda_used":False}
    for k,v in expected.items():need(scope.get(k)==v,f"scope drift {k}")
    for prefix in ("module","runner","event_parser","hse5"):
        need(sha(Path(x["implementation_binding"][prefix+"_path"]))==x["implementation_binding"][prefix+"_sha256"],f"implementation hash {prefix}")
    src=x["source_binding"]; names=tuple(src["sessions"]); need(len(names)==len(set(names))==13 and tuple(src["dates"])==DATES,"source count/date")
    forbidden=("minival","held-out","heldout","formal","evalai")
    for r in src["files"]: need(sha(Path(r["path"]))==r["sha256"] and not any(s in r["path"].lower() for s in forbidden),"source path/hash")
    br=x["baseline_reproduction"]; need(br["passed"] is True and br["comparisons"]==78 and float(br["maximum_absolute_difference"])<=float(br["absolute_tolerance"]) and sha(Path(br["reference_path"]))==br["reference_sha256"],"HSE5 reproduction")
    if x["status"]=="STOP_CPU_NLE5_UNIDENTIFIED_SOURCE_TAG_SUPPORT":
        failure=x["failure"]; need("source tag support deficient" in failure["reason"] and x["gpu_authorized_by_this_screen"] is False and x["budgets"]=={},"fail-closed status")
        coverage=failure["source_tag_counts_by_outer_date"]; need(set(coverage)==set(DATES) and any(0 in row.values() for row in coverage.values()),"unidentified tag evidence")
        return {"status":"PASS_FAIL_CLOSED","receipt":str(p),"receipt_sha256":digest,"terminal_status":x["status"],"failure":failure,"baseline_reproduction":br}
    reports={}; passed=[]
    for budget in (3,4):
        item=x["budgets"][f"M{budget}"]; need(item["budget_trials"]==budget,"budget")
        maps=item["nonlinear_map_by_outer_date"]; bases=item["hse5_basis_by_outer_date"]; rows=item["nle5_sessions"]; base=item["hse5_baseline_sessions"]
        need(set(maps)==set(bases)==set(DATES) and tuple(rows)==tuple(base)==names,"row/date set")
        for date in DATES:
            source=tuple(maps[date]["source_sessions"]); expected_source=tuple(n for n in names if not n.startswith("ses-"+date))
            need(source==expected_source and maps[date]["rff_width"]==16 and maps[date]["output_rank"]==4,"outer source exclusion")
        for name in names:
            r=rows[name]; need(r["date"]==name[4:].split("T")[0] and r["carrier_dim"]==5 and r["target_design_rank"]==5 and r["outer_future_used_only_for_scoring"] is True,"target dimension/scope")
            need(r["correct_fit"]["target_optimizer_steps"]==r["correct_fit"]["target_backward_steps"]==0 and r["rotation_invariance_max_abs_error"]<=1e-11,"closed form/geometry")
            for control in ("label_shuffle","tag_shuffle","row_shuffle"): need(r[control]["fixed_points"]==0,"fixed point control")
        fields={"correct_r2":[(n,rows[n]["median_r2_correct"]) for n in names],"correct_minus_hse5":[(n,rows[n]["median_r2_correct"]-base[n]["median_r2_correct"]) for n in names],"correct_minus_label_shuffle":[(n,rows[n]["median_delta_label_shuffle"]) for n in names],"correct_minus_tag_shuffle":[(n,rows[n]["median_delta_tag_shuffle"]) for n in names],"correct_minus_row_shuffle":[(n,rows[n]["median_delta_row_shuffle"]) for n in names],"correct_minus_intercept":[(n,rows[n]["median_delta_intercept"]) for n in names]}
        for k,v in fields.items():verify_summary(item["aggregate"][k],summary(v))
        a=item["aggregate"]; material=positive(a["correct_minus_hse5"]) and a["correct_minus_hse5"]["mean"]>=.02 and a["correct_minus_hse5"]["median"]>=.01
        controls={k:positive(a[k]) for k in ("correct_minus_label_shuffle","correct_minus_tag_shuffle","correct_minus_row_shuffle","correct_minus_intercept")}; expected_gate=bool(material and all(controls.values()))
        need(item["gate"]["passed"] is expected_gate and item["gate"]["material_gain_vs_hse5"] is bool(material),"gate arithmetic")
        if expected_gate:passed.append(f"M{budget}")
        reports[f"M{budget}"]={k:a[k] for k in fields}
    status="PASS_CPU_NLE5_MATERIAL" if len(passed)==2 else "STOP_CPU_NLE5_NOT_MATERIAL"; need(x["status"]==status and x["gpu_authorized_by_this_screen"] is False,"terminal status")
    return {"status":"PASS","receipt":str(p),"receipt_sha256":digest,"terminal_status":status,"passing_budgets":passed,"budget_report":reports}

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("receipt",type=Path);z=a.parse_args();print(json.dumps(verify(z.receipt),indent=2,sort_keys=True))
