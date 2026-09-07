#!/usr/bin/env python3
"""Append-only terminal decision for one completed RT L-D fold0 run root.

This CPU-only reader never watches, starts a GPU process, or opens a new data
scope.  It accepts only the fixed v7 arm directories and fails closed when the
current outer-eval contract lacks the per-session evidence needed for paired
gates.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os
from pathlib import Path
from typing import Any, Mapping
import yaml

ROOT=Path(__file__).resolve().parents[2]
PREFLIGHT=ROOT/"sua_exploration/results/rt_ld_fold0_preflight_v2/RT_LD_CPU_PREFLIGHT_RECEIPT_v2.json"
ROOT_REVIEW=ROOT/"sua_exploration/results/rt_ld_device_handoff_v7/RT_LD_DEVICE_HANDOFF_ROOT_REVIEW_v7.json"
V7_PLAN=ROOT/"sua_exploration/results/rt_ld_device_handoff_v7/RT_LD_DEVICE_HANDOFF_SUPPLEMENTAL_PLAN_v7.json"
RESULT_DIR=ROOT/"sua_exploration/results/rt_ld_fold0_terminalize_v1"
PREPARED=RESULT_DIR/"RT_LD_FOLD0_TERMINALIZE_PREPARED_PLAN_v1.json"
TEST=ROOT/"sua_exploration/tests/test_rt_ld_fold0_terminalize_v1.py"
ARMS={"01_a0":("rt_ld_a0_full_m24_fold0_seed42","a0","full"),"02_g_full":("rt_ld_g_full_m24_fold0_seed42","g_full","full"),"03_g_xls":("rt_ld_g_xls_m24_fold0_seed42","g_xls","xls_v2")}
GATE=0.03
def _sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def _json(p:Path)->dict[str,Any]:
 v=json.loads(p.read_text(encoding="utf-8"));
 if not isinstance(v,dict):raise ValueError(f"not JSON object: {p}")
 return v
def _yaml(p:Path)->dict[str,Any]:
 v=yaml.safe_load(p.read_text(encoding="utf-8"));
 if not isinstance(v,dict):raise ValueError(f"not YAML object: {p}")
 return v
def _finite(x:Any,label:str)->float:
 if not isinstance(x,(float,int)) or not math.isfinite(float(x)):raise ValueError(f"non-finite {label}")
 return float(x)
def _immutable(p:Path)->None:
 if not p.is_file() or p.is_symlink() or (p.stat().st_mode&0o777)!=0o444:raise ValueError(f"immutable binding invalid: {p}")
def prepare()->None:
 if RESULT_DIR.exists():raise FileExistsError("prepared terminalization is append-only")
 for p in (PREFLIGHT,ROOT_REVIEW,V7_PLAN):_immutable(p)
 pre,review,plan=_json(PREFLIGHT),_json(ROOT_REVIEW),_json(V7_PLAN)
 if pre.get("status")!="CPU_PASS_QUEUE_READY" or review.get("status")!="PASS_ROOT_REVIEW__WATCHER_MAY_BE_ARMED" or plan.get("status")!="REVIEW_REQUIRED_NOT_ARMED":raise ValueError("upstream approval status drift")
 payload={"schema":"rt_ld_fold0_terminalize_prepared_plan_v1","status":"PREPARED_CPU_ONLY_NOT_TERMINAL","script_path":str(Path(__file__).resolve().relative_to(ROOT)),"script_sha256":_sha(Path(__file__).resolve()),"test_path":str(TEST.relative_to(ROOT)),"test_sha256":_sha(TEST),"bindings":{str(p.relative_to(ROOT)):_sha(p) for p in (PREFLIGHT,ROOT_REVIEW,V7_PLAN)},"exact_arm_directories":list(ARMS),"required_outer_eval_extension":"per_session_r2 mapping with finite r2/samples; current v1 outer evaluator lacks it and terminalization must fail closed until a separately authorized evidence-producing evaluator exists","frozen_gates":{"g_full_minus_a0":GATE,"g_full_minus_g_xls":GATE},"gpu_launched":False,"formal_heldout_opened":False}
 RESULT_DIR.mkdir(parents=True);PREPARED.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");os.chmod(PREPARED,0o444)
def _check_prepared()->dict[str,Any]:
 p=_json(PREPARED)
 if p.get("script_sha256")!=_sha(Path(__file__).resolve()) or p.get("test_sha256")!=_sha(TEST):raise ValueError("terminalizer code/test hash drift")
 for rel,sha in p["bindings"].items():
  if _sha(ROOT/rel)!=sha:raise ValueError(f"upstream binding drift: {rel}")
 return p
def _arm(run_root:Path,directory:str,spec:tuple[str,str,str])->dict[str,Any]:
 rid,model_arm,gain=spec; arm=(run_root/directory).resolve()
 if arm.parent!=run_root.resolve():raise ValueError("arm path escapes run root")
 paths={"config":arm/".hydra/config.yaml","selection":arm/"rt_nested_selection_receipt.json","split":arm/"split_manifest.json","eval":arm/"rt_ld_outer_eval.json"}
 if not all(x.is_file() for x in paths.values()):raise FileNotFoundError(f"missing exact arm artifact: {directory}")
 cfg,sel,split,ev=_yaml(paths["config"]),_json(paths["selection"]),_json(paths["split"]),_json(paths["eval"])
 if cfg.get("run_id")!=rid or cfg.get("model",{}).get("rt_ld_arm")!=model_arm or cfg.get("data",{}).get("rt_ld_gain_source")!=gain or cfg.get("seed")!=42 or cfg.get("data",{}).get("outer_loso_fold")!=0:raise ValueError(f"config arm identity drift: {directory}")
 cb=cfg.get("callbacks",{}).get("rt_nested_selection_receipt",{})
 if cb.get("monitor")!="val_heldin/r2_mean":raise ValueError("selection callback drift")
 if sel.get("schema")!="rt_clean_nested_loso_selection_receipt_v1" or sel.get("status")!="PASS_FIT_INNER_SELECTION_ONLY" or sel.get("run_id")!=rid or sel.get("arm")!="afc4_vel" or sel.get("outer_loso_fold")!=0 or sel.get("seed")!=42:raise ValueError(f"selection identity drift: {directory}")
 ckpt=Path(str(sel.get("best_model_path","")))
 if not ckpt.is_absolute() or not ckpt.is_file() or ckpt.resolve().parent.parent!=(arm/"checkpoints").resolve():raise ValueError("selected checkpoint is not exact current-arm checkpoint")
 if sel.get("best_model_sha256")!=_sha(ckpt) or sel.get("config_sha256")!=_sha(paths["config"]):raise ValueError("selection hash drift")
 if split.get("validation_protocol")!="nested_loso" or split.get("requested_side_feature_group")!="afc4_vel" or split.get("outer_loso_fold")!=0 or split.get("nested_selection",{}).get("checkpoint_metric")!="val_heldin/r2_mean":raise ValueError("split contract drift")
 rt=split.get("rt_ld",{});expected_gain="aligned_full_afc4" if gain=="full" else "strong_xls_v2"
 if rt.get("identity_carrier")!="aligned_full_afc4" or rt.get("gain_carrier")!=expected_gain or rt.get("identity_never_receives_xls_v2") is not True:raise ValueError("RT-LD split contract drift")
 if ev.get("schema")!="rt_clean_nested_loso_outer_eval_v1" or ev.get("status")!="PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP" or ev.get("run_id")!=rid or ev.get("arm")!="afc4_vel" or ev.get("seed")!=42 or ev.get("outer_loso_fold")!=0:raise ValueError("outer eval identity drift")
 for k in ("formal_heldout_opened","target_backpropagation","target_query_labels_used_for_calibration","target_query_labels_used_for_checkpoint_selection"):
  if ev.get(k) is not False:raise ValueError(f"outer eval scope/update drift: {k}")
 if ev.get("model_state_unchanged") is not True or ev.get("model_state_sha256_before")!=ev.get("model_state_sha256_after"):raise ValueError("outer model mutation evidence")
 sessions=ev.get("per_session_r2")
 if not isinstance(sessions,Mapping) or not sessions:raise ValueError("EVIDENCE_GAP: outer eval lacks required per_session_r2; cannot compute paired per-session gate")
 rendered={}
 for name,row in sessions.items():
  if not isinstance(row,Mapping) or not isinstance(row.get("samples"),int) or row["samples"]<=0:raise ValueError("per-session samples invalid")
  rendered[str(name)]={"r2":_finite(row.get("r2"),"per-session R2"),"samples":int(row["samples"])}
 if set(rendered)!={str(ev.get("outer_target_session"))}:raise ValueError("outer session identity/per-session evidence mismatch")
 if abs(_finite(ev.get("r2_variance_weighted"),"pooled R2")-rendered[str(ev["outer_target_session"])]["r2"])>1e-12:raise ValueError("single-session pooled/per-session R2 mismatch")
 return {"directory":directory,"paths":{k:str(v.resolve()) for k,v in paths.items()},"sha256":{k:_sha(v) for k,v in paths.items()},"checkpoint_path":str(ckpt),"checkpoint_sha256":_sha(ckpt),"outer_target_session":ev["outer_target_session"],"query_contract":{"query_start_trial":ev.get("query_start_trial"),"window_size":ev.get("window_size"),"query_windows_evaluated":ev.get("query_windows_evaluated")},"pooled_r2":_finite(ev.get("r2_variance_weighted"),"pooled R2"),"per_session":rendered}
def terminalize(run_root:Path)->dict[str,Any]:
 _check_prepared();root=run_root.resolve()
 if not set(x.name for x in root.iterdir())>=set(ARMS):raise ValueError("missing exact L-D arm directory")
 rows={d:_arm(root,d,s) for d,s in ARMS.items()}
 base=rows["01_a0"];gf=rows["02_g_full"];gx=rows["03_g_xls"]
 for x in (gf,gx):
  if x["outer_target_session"]!=base["outer_target_session"] or x["query_contract"]!=base["query_contract"] or set(x["per_session"])!=set(base["per_session"]):raise ValueError("outer session/query/split pairing contract mismatch")
 def delta(left,right):return {s:left["per_session"][s]["r2"]-right["per_session"][s]["r2"] for s in left["per_session"]}
 ga,gxdelta=delta(gf,base),delta(gf,gx)
 def summary(values):
  a=sorted(values.values());return {"pooled_delta":None,"mean":sum(a)/len(a),"median":a[len(a)//2],"per_session":values,"positive_count":sum(v>0 for v in a),"session_count":len(a)}
 one,two=summary(ga),summary(gxdelta);one["pooled_delta"]=gf["pooled_r2"]-base["pooled_r2"];two["pooled_delta"]=gf["pooled_r2"]-gx["pooled_r2"]
 passed=one["pooled_delta"]>=GATE and two["pooled_delta"]>=GATE
 return {"schema":"rt_ld_fold0_terminal_receipt_v1","status":"PASS_BOTH_GATES" if passed else "STOP_GATE_FAILED","development_only_single_fold_seed":True,"formal_heldout_opened":False,"deeper_film_attention_forbidden":not passed,"frozen_gate_threshold":GATE,"arms":rows,"g_full_minus_a0":one,"g_full_minus_g_xls":two}
def write(run_root:Path,output:Path)->dict[str,Any]:
 if output.exists():raise FileExistsError("terminal receipt append-only")
 result=terminalize(run_root);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");os.chmod(output,0o444);return result
def main()->None:
 p=argparse.ArgumentParser();p.add_argument("--prepare",action="store_true");p.add_argument("--run-root",type=Path);p.add_argument("--validate",action="store_true");p.add_argument("--write",type=Path);a=p.parse_args()
 if a.prepare:prepare();return
 if not a.run_root or (a.validate==bool(a.write)):p.error("use --prepare or exactly one of --validate/--write with --run-root")
 result=terminalize(a.run_root) if a.validate else write(a.run_root,a.write)
 print(json.dumps({"status":result["status"]}))
if __name__=="__main__":main()
