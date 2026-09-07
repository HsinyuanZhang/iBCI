#!/usr/bin/env python3
"""Append-only terminal decision for one RT L-D outer fold / one session."""
from __future__ import annotations
import argparse,hashlib,json,math,os
from pathlib import Path
from typing import Any,Mapping
import yaml
ROOT=Path(__file__).resolve().parents[2];PRE=ROOT/"sua_exploration/results/rt_ld_fold0_preflight_v2/RT_LD_CPU_PREFLIGHT_RECEIPT_v2.json";V8=ROOT/"sua_exploration/results/rt_ld_device_handoff_v8/RT_LD_DEVICE_HANDOFF_SUPPLEMENTAL_PLAN_v8.json";SUPER=ROOT/"sua_exploration/results/rt_ld_device_handoff_v7/RT_LD_HANDOFF_V7_SUPERSEDED_BY_V8.json";RETIRE=ROOT/"sua_exploration/results/rt_ld_device_handoff_v7/RT_LD_V7_RUNTIME_SCHEMA_DEFECT_RETIREMENT_v1.json";EVAL=ROOT/"streaming_calibration_exp/src/rt_clean_nested_loso_eval.py";OUTDIR=ROOT/"sua_exploration/results/rt_ld_fold0_terminalize_v2";PLAN=OUTDIR/"RT_LD_FOLD0_TERMINALIZE_PREPARED_PLAN_v2.json";TEST=ROOT/"sua_exploration/tests/test_rt_ld_fold0_terminalize_v2.py";ARMS={"01_a0":("rt_ld_a0_full_m24_fold0_seed42","a0","full"),"02_g_full":("rt_ld_g_full_m24_fold0_seed42","g_full","full"),"03_g_xls":("rt_ld_g_xls_m24_fold0_seed42","g_xls","xls_v2")};GATE=.03
def _sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def _json(p:Path)->dict[str,Any]:
 x=json.loads(p.read_text());
 if not isinstance(x,dict):raise ValueError("JSON object required")
 return x
def _yaml(p:Path)->dict[str,Any]:
 x=yaml.safe_load(p.read_text());
 if not isinstance(x,dict):raise ValueError("YAML object required")
 return x
def _num(x:Any)->float:
 if not isinstance(x,(int,float)) or not math.isfinite(float(x)):raise ValueError("non-finite R2")
 return float(x)
def _imm(p:Path)->None:
 if not p.is_file() or p.is_symlink() or (p.stat().st_mode&0o777)!=0o444:raise ValueError(f"immutable prerequisite: {p}")
def prepare()->None:
 if OUTDIR.exists():raise FileExistsError("v2 prepared plan append-only")
 for p in (PRE,V8,SUPER,RETIRE):_imm(p)
 if _json(PRE).get("status")!="CPU_PASS_QUEUE_READY" or _json(V8).get("status")!="REVIEW_REQUIRED_NOT_ARMED" or _json(SUPER).get("status")!="SUPERSEDED_DO_NOT_ARM" or _json(RETIRE).get("retirement",{}).get("watcher_pid_absent_after_stop") is not True:raise ValueError("precondition status")
 b={"schema":"rt_ld_fold0_terminalize_prepared_plan_v2","status":"PREPARED_CPU_ONLY_NOT_TERMINAL","script_path":str(Path(__file__).resolve().relative_to(ROOT)),"script_sha256":_sha(Path(__file__).resolve()),"test_path":str(TEST.relative_to(ROOT)),"test_sha256":_sha(TEST),"bindings":{str(p.relative_to(ROOT)):_sha(p) for p in (PRE,V8,SUPER,RETIRE)},"outer_evaluator_path":str(EVAL.relative_to(ROOT)),"outer_evaluator_sha256":_sha(EVAL),"exact_arm_dirs":list(ARMS),"single_outer_fold_seed":True,"frozen_gates":{"g_full_minus_a0":GATE,"g_full_minus_g_xls":GATE},"gpu_launched":False,"formal_heldout_opened":False}
 OUTDIR.mkdir(parents=True);PLAN.write_text(json.dumps(b,indent=2,sort_keys=True)+"\n");os.chmod(PLAN,0o444)
def _check()->None:
 p=_json(PLAN)
 if p.get("script_sha256")!=_sha(Path(__file__).resolve()) or p.get("test_sha256")!=_sha(TEST) or p.get("outer_evaluator_sha256")!=_sha(EVAL):raise ValueError("prepared hash drift")
 for rel,h in p["bindings"].items():
  if _sha(ROOT/rel)!=h:raise ValueError("upstream hash drift")
def _read(root:Path,dn:str,spec:tuple[str,str,str])->dict[str,Any]:
 rid,ma,gain=spec;d=(root/dn).resolve();cfgp=d/".hydra/config.yaml";selp=d/"rt_nested_selection_receipt.json";splitp=d/"split_manifest.json";evp=d/"rt_ld_outer_eval.json"
 if not all(p.is_file() for p in (cfgp,selp,splitp,evp)):raise FileNotFoundError(f"missing exact artifact {dn}")
 cfg,sel,split,ev=_yaml(cfgp),_json(selp),_json(splitp),_json(evp)
 if cfg.get("run_id")!=rid or cfg.get("model",{}).get("rt_ld_arm")!=ma or cfg.get("data",{}).get("rt_ld_gain_source")!=gain or cfg.get("seed")!=42 or cfg.get("data",{}).get("outer_loso_fold")!=0:raise ValueError("config arm")
 ck=Path(str(sel.get("best_model_path","")))
 if not ck.is_absolute() or not ck.is_file() or ck.resolve().parent.parent!=(d/"checkpoints").resolve() or sel.get("best_model_sha256")!=_sha(ck) or sel.get("config_sha256")!=_sha(cfgp):raise ValueError("selection checkpoint/hash")
 if sel.get("schema")!="rt_clean_nested_loso_selection_receipt_v1" or sel.get("status")!="PASS_FIT_INNER_SELECTION_ONLY" or sel.get("run_id")!=rid or sel.get("arm")!="afc4_vel" or sel.get("outer_loso_fold")!=0 or sel.get("seed")!=42 or sel.get("formal_heldout_opened") is not False or sel.get("outer_target_loaded_during_fit") is not False or sel.get("outer_target_query_labels_read_during_fit") is not False:raise ValueError("selection scope")
 if split.get("validation_protocol")!="nested_loso" or split.get("requested_side_feature_group")!="afc4_vel" or split.get("outer_loso_fold")!=0 or split.get("formal_heldout_opened") is not False or split.get("nested_selection",{}).get("outer_target_loaded_during_fit") is not False or split.get("nested_selection",{}).get("outer_target_query_labels_read_during_fit") is not False:raise ValueError("split scope")
 if ev.get("schema")!="rt_clean_nested_loso_outer_eval_v1" or ev.get("status")!="PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP" or ev.get("run_id")!=rid or ev.get("arm")!="afc4_vel" or ev.get("seed")!=42 or ev.get("outer_loso_fold")!=0 or "formal_heldout_opened" in ev:raise ValueError("actual evaluator identity/schema")
 for k in ("target_backpropagation","target_query_labels_used_for_calibration","target_query_labels_used_for_normalization","target_query_labels_used_for_checkpoint_selection","optimizer_present","model_training_mode"):
  if ev.get(k) is not False:raise ValueError("outer scope")
 if ev.get("target_query_labels_used_for_scoring_only") is not True or ev.get("model_state_unchanged") is not True or ev.get("model_state_sha256_before")!=ev.get("model_state_sha256_after"):raise ValueError("outer immutable")
 return {"r2":_num(ev.get("r2_variance_weighted")),"join":{k:ev.get(k) for k in ("outer_target_session","outer_target_path","query_start_trial","window_size","query_windows_evaluated","data_dir","inner_train_sessions","inner_validation_session")},"sha256":{"config":_sha(cfgp),"selection":_sha(selp),"split":_sha(splitp),"eval":_sha(evp),"checkpoint":_sha(ck)},"paths":{"config":str(cfgp.resolve()),"selection":str(selp.resolve()),"split":str(splitp.resolve()),"eval":str(evp.resolve()),"checkpoint":str(ck)}}
def terminalize(root:Path)->dict[str,Any]:
 _check();root=root.resolve();rows={d:_read(root,d,s) for d,s in ARMS.items()};base=rows["01_a0"]["join"]
 if any(x["join"]!=base for x in rows.values()):raise ValueError("paired outer session/query/split contract mismatch")
 ga=rows["02_g_full"]["r2"]-rows["01_a0"]["r2"];gx=rows["02_g_full"]["r2"]-rows["03_g_xls"]["r2"];passed=ga>=GATE and gx>=GATE
 return {"schema":"rt_ld_fold0_terminal_receipt_v2","status":"PASS_BOTH_GATES" if passed else "STOP_GATE_FAILED","single_outer_fold_seed":True,"outer_session_count":1,"formal_heldout_opened":False,"deeper_film_attention_forbidden":not passed,"frozen_gate_threshold":GATE,"arms":rows,"g_full_minus_a0":{"pooled_delta":ga,"mean":ga,"median":ga,"per_session_delta":{str(base["outer_target_session"]):ga},"positive_count":int(ga>0),"session_count":1},"g_full_minus_g_xls":{"pooled_delta":gx,"mean":gx,"median":gx,"per_session_delta":{str(base["outer_target_session"]):gx},"positive_count":int(gx>0),"session_count":1}}
def write(root:Path,out:Path)->dict[str,Any]:
 if out.exists():raise FileExistsError("terminal receipt append-only")
 r=terminalize(root);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n");os.chmod(out,0o444);return r
def main()->None:
 p=argparse.ArgumentParser();p.add_argument("--prepare",action="store_true");p.add_argument("--run-root",type=Path);p.add_argument("--validate",action="store_true");p.add_argument("--write",type=Path);a=p.parse_args()
 if a.prepare:prepare();return
 if not a.run_root or (a.validate==bool(a.write)):p.error("use --prepare or exactly one validate/write with run-root")
 r=terminalize(a.run_root) if a.validate else write(a.run_root,a.write);print(json.dumps({"status":r["status"]}))
if __name__=="__main__":main()
