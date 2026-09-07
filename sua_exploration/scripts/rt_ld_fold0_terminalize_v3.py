#!/usr/bin/env python3
"""Fail-closed append-only terminalizer for exactly one RT L-D fold/session."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,os,tempfile
from pathlib import Path
from typing import Any,Mapping
ROOT=Path(__file__).resolve().parents[2];V2=ROOT/"sua_exploration/scripts/rt_ld_fold0_terminalize_v2.py";V8REVIEW=ROOT/"sua_exploration/results/rt_ld_device_handoff_v8/RT_LD_DEVICE_HANDOFF_ROOT_REVIEW_v8.json";OUT=ROOT/"sua_exploration/results/rt_ld_fold0_terminalize_v3";PLAN=OUT/"RT_LD_FOLD0_TERMINALIZE_PREPARED_PLAN_v3.json";TEST=ROOT/"sua_exploration/tests/test_rt_ld_fold0_terminalize_v3.py";REVIEW_SHA="390773d2fe1c486b3d29f53a641640f1f91d9bb4071ede8c5353d2316e115767"
def _load():
 s=importlib.util.spec_from_file_location("tv2",V2);assert s and s.loader;m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
B=_load();ARMS,GATE=B.ARMS,B.GATE
def _sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def _json(p:Path)->dict[str,Any]:
 x=json.loads(p.read_text());
 if not isinstance(x,dict):raise ValueError("object")
 return x
def _under(x:Path,p:Path)->bool:
 try:x.resolve().relative_to(p.resolve());return not x.is_symlink()
 except ValueError:return False
def prepare():
 if OUT.exists():raise FileExistsError("v3 append-only")
 B._imm(B.PRE);B._imm(B.V8);B._imm(B.SUPER);B._imm(B.RETIRE);B._imm(V8REVIEW)
 if _sha(V8REVIEW)!=REVIEW_SHA:raise ValueError("root v8 review SHA drift")
 body={"schema":"rt_ld_fold0_terminalize_prepared_plan_v3","status":"PREPARED_CPU_ONLY_NOT_TERMINAL","script_path":str(Path(__file__).resolve().relative_to(ROOT)),"script_sha256":_sha(Path(__file__).resolve()),"test_path":str(TEST.relative_to(ROOT)),"test_sha256":_sha(TEST),"supersedes":str(B.PLAN.relative_to(ROOT)),"bindings":{str(p.relative_to(ROOT)):_sha(p) for p in (B.PRE,B.V8,B.SUPER,B.RETIRE,V8REVIEW)},"root_v8_review_sha256":REVIEW_SHA,"single_outer_fold_seed":True,"frozen_gate":GATE,"gpu_launched":False,"formal_heldout_opened":False}
 OUT.mkdir(parents=True);PLAN.write_text(json.dumps(body,indent=2,sort_keys=True)+"\n");os.chmod(PLAN,0o444)
def _check():
 p=_json(PLAN)
 if p.get("script_sha256")!=_sha(Path(__file__).resolve()) or p.get("test_sha256")!=_sha(TEST):raise ValueError("plan hash drift")
 for r,h in p["bindings"].items():
  if _sha(ROOT/r)!=h:raise ValueError("upstream hash drift")
def _read(root:Path,dn:str,spec:tuple[str,str,str])->dict[str,Any]:
 d=(root/dn).resolve();cfgp=d/".hydra/config.yaml";selp=d/"rt_nested_selection_receipt.json";splitp=d/"split_manifest.json";evp=d/"rt_ld_outer_eval.json";cfg=B._yaml(cfgp);sel=_json(selp);split=_json(splitp);ev=_json(evp);rid,ma,gain=spec;ck=Path(sel["best_model_path"])
 if cfg.get("run_id")!=rid or cfg.get("model",{}).get("rt_ld_arm")!=ma or cfg.get("data",{}).get("rt_ld_gain_source")!=gain or cfg.get("seed")!=42 or cfg.get("data",{}).get("outer_loso_fold")!=0:raise ValueError("config identity")
 if sel.get("best_model_sha256")!=_sha(ck) or sel.get("config_sha256")!=_sha(cfgp) or ev.get("run_id")!=rid or ev.get("arm")!="afc4_vel" or ev.get("seed")!=42 or ev.get("outer_loso_fold")!=0:raise ValueError("base receipt/eval")
 if not _under(ck,d/"checkpoints") or not ck.is_file():raise ValueError("checkpoint containment/symlink")
 exact={"selection_receipt_path":d/"rt_nested_selection_receipt.json","config_path":d/".hydra/config.yaml","split_manifest_path":d/"split_manifest.json","run_dir":d}
 for k,p in exact.items():
  if Path(str(sel.get(k,""))).resolve()!=p.resolve():raise ValueError("selection exact path")
 if sel.get("split_manifest_sha256")!=_sha(d/"split_manifest.json") or sel.get("selected_by_metric")!="val_heldin/r2_mean" or sel.get("selected_metric_scope")!="inner_validation_session_only":raise ValueError("selection split/metric")
 rt=split.get("rt_ld",{});expect="aligned_full_afc4" if gain=="full" else "strong_xls_v2"
 if split.get("nested_selection",{}).get("clean") is not True or split.get("nested_selection",{}).get("inner_validation_only_for_checkpoint_selection") is not True or rt.get("identity_carrier")!="aligned_full_afc4" or rt.get("gain_carrier")!=expect:raise ValueError("split nested/rtld")
 ert=ev.get("rt_ld",{})
 if ert.get("identity_carrier")!="aligned_full_afc4" or ert.get("gain_carrier")!=expect:raise ValueError("outer rtld")
 return {"r2":B._num(ev.get("r2_variance_weighted")),"join":{k:ev.get(k) for k in ("outer_target_session","outer_target_path","query_start_trial","window_size","query_windows_evaluated","data_dir","inner_train_sessions","inner_validation_session")},"sha256":{"config":_sha(cfgp),"selection":_sha(selp),"split":_sha(splitp),"eval":_sha(evp),"checkpoint":_sha(ck)},"paths":{"config":str(cfgp.resolve()),"selection":str(selp.resolve()),"split":str(splitp.resolve()),"eval":str(evp.resolve()),"checkpoint":str(ck)}}
def terminalize(root:Path)->dict[str,Any]:
 _check();root=root.resolve()
 if set(x.name for x in root.iterdir())!=set(ARMS) or any(x.is_symlink() for x in root.iterdir()):raise ValueError("run-root exact children")
 rows={d:_read(root,d,s) for d,s in ARMS.items()};base=rows["01_a0"]["join"]
 for x in rows.values():
  for k in ("outer_target_session","outer_target_path","data_dir","inner_validation_session"):
   if not isinstance(x["join"].get(k),str) or not x["join"][k]:raise ValueError("missing join field")
  for k in ("query_start_trial","window_size","query_windows_evaluated"):
   if not isinstance(x["join"].get(k),int) or x["join"][k]<=0:raise ValueError("invalid join number")
  if not isinstance(x["join"].get("inner_train_sessions"),list) or not x["join"]["inner_train_sessions"]:raise ValueError("missing inner train")
  if x["join"]!=base:raise ValueError("paired join mismatch")
 a=rows["02_g_full"]["r2"]-rows["01_a0"]["r2"];b=rows["02_g_full"]["r2"]-rows["03_g_xls"]["r2"];ok=a>=GATE and b>=GATE
 return {"schema":"rt_ld_fold0_terminal_receipt_v3","status":"PASS_BOTH_GATES" if ok else "STOP_GATE_FAILED","prepared_plan_path":str(PLAN.resolve()),"prepared_plan_sha256":_sha(PLAN),"upstream_bindings":_json(PLAN)["bindings"],"single_outer_fold_seed_one_session_only":True,"formal_heldout_opened":False,"deeper_film_attention_forbidden":not ok,"arms":rows,"g_full_minus_a0":{"pooled_delta":a,"mean":a,"median":a,"per_session_delta":{base["outer_target_session"]:a},"positive_count":int(a>0),"session_count":1},"g_full_minus_g_xls":{"pooled_delta":b,"mean":b,"median":b,"per_session_delta":{base["outer_target_session"]:b},"positive_count":int(b>0),"session_count":1}}
def write(root:Path,out:Path):
 if out.exists() or out.is_symlink():raise FileExistsError("append-only/symlink output")
 r=terminalize(root);out.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix=".terminal-",dir=out.parent)
 try:
  with os.fdopen(fd,"w") as h:json.dump(r,h,indent=2,sort_keys=True,allow_nan=False);h.write("\n");h.flush();os.fsync(h.fileno())
  os.replace(tmp,out);os.chmod(out,0o444)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
 return r
def main():
 p=argparse.ArgumentParser();p.add_argument("--prepare",action="store_true");p.add_argument("--run-root",type=Path);p.add_argument("--validate",action="store_true");p.add_argument("--write",type=Path);a=p.parse_args()
 if a.prepare:prepare();return
 if not a.run_root or(a.validate==bool(a.write)):p.error("prepare or validate/write")
 print(json.dumps({"status":(terminalize(a.run_root) if a.validate else write(a.run_root,a.write))["status"]}))
if __name__=="__main__":main()
