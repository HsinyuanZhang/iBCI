#!/usr/bin/env python3
"""Fail-closed RT L-D v8: real outer-evaluator scope contract and v7 retirement.

Inert by default.  The legacy v7 watcher must be absent from /proc before v8
can release a device; v8 never kills or signals it.  The real evaluator has no
``formal_heldout_opened`` output field, so formal closure is proved from the
immutable fit selection/split receipts while its actual outer fields are
validated exactly.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path
from typing import Any, Mapping
ROOT=Path(__file__).resolve().parents[2];STREAMING=ROOT/"streaming_calibration_exp";V7_RUNNER=ROOT/"sua_exploration/scripts/rt_ld_device_handoff_v7.py";V7_PLAN=ROOT/"sua_exploration/results/rt_ld_device_handoff_v7/RT_LD_DEVICE_HANDOFF_SUPPLEMENTAL_PLAN_v7.json";RESULT_DIR=ROOT/"sua_exploration/results/rt_ld_device_handoff_v8";PLAN=RESULT_DIR/"RT_LD_DEVICE_HANDOFF_SUPPLEMENTAL_PLAN_v8.json";TEST=ROOT/"sua_exploration/tests/test_rt_ld_device_handoff_v8.py";EVALUATOR=ROOT/"streaming_calibration_exp/src/rt_clean_nested_loso_eval.py"
def _load(p:Path,n:str)->Any:
 s=importlib.util.spec_from_file_location(n,p);assert s and s.loader;m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
V7=_load(V7_RUNNER,"rt_ld_v7_sealed");V4=V7.V4;ARM_SPECS,ARM_ORDER,PARTITIONS=V7.ARM_SPECS,V7.ARM_ORDER,V7.PARTITIONS
def _sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def _json(p:Path)->dict[str,Any]:
 v=json.loads(p.read_text(encoding="utf-8"));
 if not isinstance(v,dict):raise ValueError(f"object required: {p}")
 return v
def _v7_process_present()->bool:
 for p in Path("/proc").glob("[0-9]*"):
  try:
   if "rt_ld_device_handoff_v7.py" in (p/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace"):return True
  except OSError:continue
 return False
def _outer_scope_v8(arm:Path,experiment:str)->None:
 spec=ARM_SPECS[experiment];ev=_json(arm/"rt_ld_outer_eval.json");sel=_json(arm/"rt_nested_selection_receipt.json");split=_json(arm/"split_manifest.json")
 if ev.get("schema")!="rt_clean_nested_loso_outer_eval_v1" or ev.get("status")!="PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP" or ev.get("run_id")!=spec["run_id"] or ev.get("arm")!="afc4_vel" or ev.get("seed")!=42 or ev.get("outer_loso_fold")!=0:raise ValueError("real outer evaluator identity contract")
 # This field is deliberately absent from the actual evaluator schema.  Do
 # not invent it; formal closure remains bound by the two fit receipts below.
 if "formal_heldout_opened" in ev:raise ValueError("unexpected invented outer formal field")
 for k in ("target_backpropagation","target_query_labels_used_for_calibration","target_query_labels_used_for_normalization","target_query_labels_used_for_checkpoint_selection","optimizer_present","model_training_mode"):
  if ev.get(k) is not False:raise ValueError(f"real outer scope/update field: {k}")
 if ev.get("target_query_labels_used_for_scoring_only") is not True or ev.get("model_state_unchanged") is not True or ev.get("model_state_sha256_before")!=ev.get("model_state_sha256_after"):raise ValueError("real outer state/scoring contract")
 if sel.get("formal_heldout_opened") is not False or sel.get("outer_target_loaded_during_fit") is not False or sel.get("outer_target_query_labels_read_during_fit") is not False:raise ValueError("selection receipt formal boundary")
 if split.get("formal_heldout_opened") is not False or split.get("nested_selection",{}).get("outer_target_loaded_during_fit") is not False or split.get("nested_selection",{}).get("outer_target_query_labels_read_during_fit") is not False:raise ValueError("split receipt formal boundary")
 expected="aligned_full_afc4" if spec["data_rt_ld_gain_source"]=="full" else "strong_xls_v2";rt=ev.get("rt_ld",{})
 if rt.get("identity_carrier")!="aligned_full_afc4" or rt.get("gain_carrier")!=expected or rt.get("identity_never_receives_xls_v2") is not True:raise ValueError("real outer RT-LD contract")
def _plan_self(p:Mapping[str,Any])->None:
 exp={"schema":"rt_ld_device_handoff_supplemental_plan_v8","status":"REVIEW_REQUIRED_NOT_ARMED","runner_path":str(Path(__file__).resolve().relative_to(ROOT)),"runner_sha256":_sha(Path(__file__).resolve()),"test_path":str(TEST.relative_to(ROOT)),"test_sha256":_sha(TEST),"v7_runner_path":str(V7_RUNNER.relative_to(ROOT)),"v7_runner_sha256":_sha(V7_RUNNER),"v7_plan_path":str(V7_PLAN.relative_to(ROOT)),"v7_plan_sha256":_sha(V7_PLAN),"outer_evaluator_path":str(EVALUATOR.relative_to(ROOT)),"outer_evaluator_sha256":_sha(EVALUATOR)}
 for k,v in exp.items():
  if p.get(k)!=v:raise ValueError(f"v8 self bind: {k}")
 if p.get("gpu_launched") is not False or p.get("formal_heldout_opened") is not False:raise ValueError("inert drift")
def prepare()->None:
 if RESULT_DIR.exists():raise FileExistsError("v8 append-only")
 V7._plan_self_check(_json(V7_PLAN));V4._plan_self_check(_json(V4.PLAN));r=V4._validate_v2_anchor();V4._validate_drift(r,_json(V4.DRIFT))
 body={"schema":"rt_ld_device_handoff_supplemental_plan_v8","status":"REVIEW_REQUIRED_NOT_ARMED","runner_path":str(Path(__file__).resolve().relative_to(ROOT)),"runner_sha256":_sha(Path(__file__).resolve()),"test_path":str(TEST.relative_to(ROOT)),"test_sha256":_sha(TEST),"v7_runner_path":str(V7_RUNNER.relative_to(ROOT)),"v7_runner_sha256":_sha(V7_RUNNER),"v7_plan_path":str(V7_PLAN.relative_to(ROOT)),"v7_plan_sha256":_sha(V7_PLAN),"outer_evaluator_path":str(EVALUATOR.relative_to(ROOT)),"outer_evaluator_sha256":_sha(EVALUATOR),"outer_scope_contract":"formal closure comes from selection/split receipts; real evaluator fields prove no target calibration/normalization/selection/backprop/optimizer and immutable model state","v7_retirement_required":"any /proc command line containing rt_ld_device_handoff_v7.py makes eligible=false; root must retire v7 explicitly","partitions":PARTITIONS,"ci_five_arm_terminal_receipts_absolute":{name:{date:str(path) for date,path in paths.items()} for name,paths in V7.TERMINALS.items()},"fixed_arm_mapping":ARM_SPECS,"fixed_training_order":ARM_ORDER,"gpu_launched":False,"formal_heldout_opened":False,"fail_closed":True}
 RESULT_DIR.mkdir(parents=True);PLAN.write_text(json.dumps(body,indent=2,sort_keys=True)+"\n");os.chmod(PLAN,0o444)
def _eligible_from_probes(retired:bool,ci:bool,a:tuple[str,bool,list[str]|None],b:tuple[str,bool,list[str]|None])->bool:return bool(retired and ci) and a[0]==b[0]=="exited" and a[1] is b[1] is False and a[2]==b[2]==[]
def eligible(name:str,stability_seconds:float=1.0)->bool:
 p=_json(PLAN);_plan_self(p);r=V4._validate_v2_anchor();V4._validate_drift(r,_json(V4.DRIFT));part=p["partitions"][name]
 def probe():
  d=int(part["device_index"]);return V4._runner_state(part),V4._replacement_static_runner_present(d,int(part["runner_pid"])),V4._compute_apps(d)
 a=probe();time.sleep(stability_seconds);b=probe();return _eligible_from_probes(not _v7_process_present(),V7._ci_receipts_ready(name),a,b)
def execute(name:str,run_root:Path)->None:
 root=V4._absolute_run_root(run_root)
 if root.exists() and any(root.iterdir()):raise FileExistsError("v8 refuses nonempty/reused run root")
 V4.validate_all_composed();root.mkdir(parents=True,exist_ok=True);part=_json(PLAN)["partitions"][name];env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(part["device_index"]))
 for exp,spec in ARM_SPECS.items():
  arm=(root/spec["directory"]).resolve()
  if arm.parent!=root:raise ValueError("arm escape")
  if not eligible(name):raise RuntimeError("v8 full predicate refused spawn")
  if subprocess.run([sys.executable,"src/train.py",f"experiment={exp}","seed=42",f"hydra.run.dir={arm}"],cwd=STREAMING,env=env,check=False).returncode:raise RuntimeError(f"train failed: {exp}")
  ck=V4._validate_fit_artifacts(arm,exp);out=arm/"rt_ld_outer_eval.json";ev=subprocess.run([sys.executable,"src/rt_clean_nested_loso_eval.py","--config",str(arm/".hydra/config.yaml"),"--checkpoint",str(ck),"--split-manifest",str(arm/"split_manifest.json"),"--selection-receipt",str(arm/"rt_nested_selection_receipt.json"),"--output",str(out),"--device","cuda:0"],cwd=STREAMING,env=env,check=False)
  if ev.returncode or not out.is_file():raise RuntimeError(f"outer eval failed: {exp}")
  _outer_scope_v8(arm,exp)
def main()->None:
 p=argparse.ArgumentParser();p.add_argument("--prepare",action="store_true");p.add_argument("--partition",choices=sorted(PARTITIONS));p.add_argument("--validate",action="store_true");p.add_argument("--watch",action="store_true");p.add_argument("--execute",action="store_true");p.add_argument("--run-root",type=Path);p.add_argument("--poll-seconds",type=float,default=30.0);a=p.parse_args()
 if a.prepare:prepare();return
 if not a.partition or not(a.validate or a.watch):p.error("use --prepare or --partition with --validate/--watch")
 if a.execute and not a.watch:p.error("--execute requires --watch")
 if a.watch and(not a.execute or a.run_root is None):p.error("--watch requires explicit --execute and --run-root")
 if a.watch:
  while not eligible(a.partition):time.sleep(a.poll_seconds)
  execute(a.partition,a.run_root);return
 print(json.dumps({"partition":a.partition,"eligible":eligible(a.partition)}))
if __name__=="__main__":main()
