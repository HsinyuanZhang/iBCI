#!/usr/bin/env python3
"""Append-only RT L-D v7: order-independent current-CI receipt latch.

Inert unless the explicit paired ``--watch --execute --run-root`` path is
requested.  This corrects only v6's JSON-key-order bug; all CI path, scope,
static-runner, replacement-runner, and two-empty-compute predicates remain.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, stat, subprocess, sys, time
from pathlib import Path
from typing import Any, Mapping

ROOT=Path(__file__).resolve().parents[2]; STREAMING=ROOT/"streaming_calibration_exp"
V6_RUNNER=ROOT/"sua_exploration/scripts/rt_ld_device_handoff_v6.py"; V6_PLAN=ROOT/"sua_exploration/results/rt_ld_device_handoff_v6/RT_LD_DEVICE_HANDOFF_SUPPLEMENTAL_PLAN_v6.json"
V4_RUNNER=ROOT/"sua_exploration/scripts/rt_ld_device_handoff_v4.py"; V4_PLAN=ROOT/"sua_exploration/results/rt_ld_device_handoff_v4/RT_LD_DEVICE_HANDOFF_SUPPLEMENTAL_PLAN_v4.json"
RESULT_DIR=ROOT/"sua_exploration/results/rt_ld_device_handoff_v7"; PLAN=RESULT_DIR/"RT_LD_DEVICE_HANDOFF_SUPPLEMENTAL_PLAN_v7.json"; TEST=ROOT/"sua_exploration/tests/test_rt_ld_device_handoff_v7.py"
def _load(path:Path,name:str)->Any:
 s=importlib.util.spec_from_file_location(name,path); assert s and s.loader; m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
V6=_load(V6_RUNNER,"rt_ld_v6_sealed"); V4=V6.V4; ARM_SPECS,ARM_ORDER,PARTITIONS=V6.ARM_SPECS,V6.ARM_ORDER,V6.PARTITIONS; TERMINALS,CI_SCHEMA,CI_ARMS=V6.TERMINALS,V6.CI_SCHEMA,V6.CI_ARMS
def _sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def _json(p:Path)->dict[str,Any]:
 v=json.loads(p.read_text(encoding="utf-8"));
 if not isinstance(v,dict):raise ValueError(f"expected object: {p}")
 return v
def _validate_ci_terminal(path:Path,date:str)->None:
 if not path.is_file() or path.is_symlink():raise ValueError("CI terminal missing")
 if stat.S_IMODE(path.stat().st_mode)!=0o444:raise ValueError("CI terminal mode")
 b=_json(path)
 if b.get("schema")!=CI_SCHEMA or b.get("status")!=V6._status(date) or b.get("outer_date")!=date:raise ValueError("CI schema/status/date")
 metrics=b.get("metrics")
 if not isinstance(metrics,Mapping) or len(metrics)!=5 or set(metrics)!=set(CI_ARMS):raise ValueError("CI exact five-arm metrics set")
 scope,updates=b.get("scope"),b.get("deployment_updates")
 if not isinstance(scope,Mapping) or any(scope.get(k) is not False for k in ("formal_heldout_opened","minival_opened","evalai_opened")):raise ValueError("CI scope")
 if not isinstance(updates,Mapping) or updates.get("optimizer_steps")!=0 or updates.get("backward_steps")!=0 or updates.get("model_state_unchanged") is not True:raise ValueError("CI updates")
def _ci_receipts_ready(name:str,paths:Mapping[str,Path]|None=None)->bool:
 chosen=TERMINALS[name] if paths is None else paths
 try:
  if tuple(chosen)!=tuple(TERMINALS[name]):return False
  for d,p in chosen.items():_validate_ci_terminal(Path(p),d)
 except (OSError,ValueError,json.JSONDecodeError):return False
 return True
def _plan_self_check(p:Mapping[str,Any])->None:
 exp={"schema":"rt_ld_device_handoff_supplemental_plan_v7","status":"REVIEW_REQUIRED_NOT_ARMED","runner_path":str(Path(__file__).resolve().relative_to(ROOT)),"runner_sha256":_sha(Path(__file__).resolve()),"test_path":str(TEST.relative_to(ROOT)),"test_sha256":_sha(TEST),"v6_runner_path":str(V6_RUNNER.relative_to(ROOT)),"v6_runner_sha256":_sha(V6_RUNNER),"v6_plan_path":str(V6_PLAN.relative_to(ROOT)),"v6_plan_sha256":_sha(V6_PLAN),"v4_runner_path":str(V4_RUNNER.relative_to(ROOT)),"v4_runner_sha256":_sha(V4_RUNNER),"v4_plan_path":str(V4_PLAN.relative_to(ROOT)),"v4_plan_sha256":_sha(V4_PLAN)}
 for k,v in exp.items():
  if p.get(k)!=v:raise ValueError(f"v7 plan bind: {k}")
 if p.get("gpu_launched") is not False or p.get("formal_heldout_opened") is not False:raise ValueError("inert drift")
def prepare()->None:
 if RESULT_DIR.exists():raise FileExistsError("v7 append-only")
 V6._plan_self_check(_json(V6_PLAN)); V4._plan_self_check(_json(V4_PLAN)); r=V4._validate_v2_anchor();V4._validate_drift(r,_json(V4.DRIFT))
 terminals={n:{d:{"path":str(p),"schema":CI_SCHEMA,"status":V6._status(d),"mode":"0444","metrics_contract":"len=5 and set={CI32-FULL,CI64-FULL,CI64-C0,CI64-LS,CI64-RS}","scope_all_false":True,"deployment_optimizer_steps":0,"deployment_backward_steps":0}for d,p in x.items()}for n,x in TERMINALS.items()}
 body={"schema":"rt_ld_device_handoff_supplemental_plan_v7","status":"REVIEW_REQUIRED_NOT_ARMED","runner_path":str(Path(__file__).resolve().relative_to(ROOT)),"runner_sha256":_sha(Path(__file__).resolve()),"test_path":str(TEST.relative_to(ROOT)),"test_sha256":_sha(TEST),"v6_runner_path":str(V6_RUNNER.relative_to(ROOT)),"v6_runner_sha256":_sha(V6_RUNNER),"v6_plan_path":str(V6_PLAN.relative_to(ROOT)),"v6_plan_sha256":_sha(V6_PLAN),"v4_runner_path":str(V4_RUNNER.relative_to(ROOT)),"v4_runner_sha256":_sha(V4_RUNNER),"v4_plan_path":str(V4_PLAN.relative_to(ROOT)),"v4_plan_sha256":_sha(V4_PLAN),"partitions":PARTITIONS,"ci_five_arm_terminal_receipts_absolute":terminals,"release_rule":"v6 full predicate, with order-independent exact five-arm CI metrics set","other_partition_or_five_date_aggregate_required":False,"fixed_arm_mapping":ARM_SPECS,"fixed_training_order":ARM_ORDER,"gpu_launched":False,"formal_heldout_opened":False,"fail_closed":True}
 RESULT_DIR.mkdir(parents=True);PLAN.write_text(json.dumps(body,indent=2,sort_keys=True)+"\n");os.chmod(PLAN,0o444)
def _eligible_from_probes(ci:bool,a:tuple[str,bool,list[str]|None],b:tuple[str,bool,list[str]|None])->bool:return bool(ci) and a[0]==b[0]=="exited" and a[1] is b[1] is False and a[2]==b[2]==[]
def eligible(name:str,stability_seconds:float=1.0)->bool:
 p=_json(PLAN);_plan_self_check(p);r=V4._validate_v2_anchor();V4._validate_drift(r,_json(V4.DRIFT));part=p["partitions"][name]
 def probe():
  device=int(part["device_index"]);return V4._runner_state(part),V4._replacement_static_runner_present(device,int(part["runner_pid"])),V4._compute_apps(device)
 a=probe();time.sleep(stability_seconds);b=probe();return _eligible_from_probes(_ci_receipts_ready(name),a,b)
def execute(name:str,run_root:Path)->None:
 root=V4._absolute_run_root(run_root)
 if root.exists() and any(root.iterdir()):raise FileExistsError("run root must be empty")
 V4.validate_all_composed();root.mkdir(parents=True,exist_ok=True);part=_json(PLAN)["partitions"][name];env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(part["device_index"]))
 for exp,spec in ARM_SPECS.items():
  arm=(root/spec["directory"]).resolve()
  if arm.parent!=root:raise ValueError("arm escape")
  if not eligible(name):raise RuntimeError("v7 TOCTOU refused spawn")
  train=subprocess.run([sys.executable,"src/train.py",f"experiment={exp}","seed=42",f"hydra.run.dir={arm}"],cwd=STREAMING,env=env,check=False)
  if train.returncode:raise RuntimeError(f"training failed: {exp}")
  ckpt=V4._validate_fit_artifacts(arm,exp);out=arm/"rt_ld_outer_eval.json";ev=subprocess.run([sys.executable,"src/rt_clean_nested_loso_eval.py","--config",str(arm/".hydra/config.yaml"),"--checkpoint",str(ckpt),"--split-manifest",str(arm/"split_manifest.json"),"--selection-receipt",str(arm/"rt_nested_selection_receipt.json"),"--output",str(out),"--device","cuda:0"],cwd=STREAMING,env=env,check=False)
  if ev.returncode or not out.is_file():raise RuntimeError(f"outer eval failed: {exp}")
  V4._validate_outer_artifact(out,exp)
def main()->None:
 a=argparse.ArgumentParser();a.add_argument("--prepare",action="store_true");a.add_argument("--partition",choices=sorted(PARTITIONS));a.add_argument("--validate",action="store_true");a.add_argument("--watch",action="store_true");a.add_argument("--execute",action="store_true");a.add_argument("--run-root",type=Path);a.add_argument("--poll-seconds",type=float,default=30.0);x=a.parse_args()
 if x.prepare:prepare();return
 if not x.partition or not(x.validate or x.watch):a.error("use --prepare or --partition with --validate/--watch")
 if x.execute and not x.watch:a.error("--execute requires --watch")
 if x.watch and(not x.execute or x.run_root is None):a.error("--watch requires explicit --execute and --run-root")
 if x.watch:
  while not eligible(x.partition):time.sleep(x.poll_seconds)
  execute(x.partition,x.run_root);return
 print(json.dumps({"partition":x.partition,"eligible":eligible(x.partition)}))
if __name__=="__main__":main()
