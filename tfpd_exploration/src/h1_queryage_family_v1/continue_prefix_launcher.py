"""Root-only admission/launcher for fixed H1 continuation epochs 13 through 24."""
from __future__ import annotations
import argparse, hashlib, json, math, os, subprocess, sys, tempfile, time
from pathlib import Path
ARMS={"flat":0,"route":1}; CPU={"flat":"12-15","route":"8-11"}
GO="H1_QUERYAGE_CONTINUE_24_GO"; SCHEMA="h1_queryage_continue_24_root_authorization_v1"
PARENT_STATUS="COMPLETE_FIXED_FORMAL_NO_PROMOTION"; SELECTION_BINS=2908; COMPLETE_BINS=20325; LIMIT=21600
PROTOCOL=Path(__file__).resolve().parents[2]/"docs/PROTOCOL_H1_QUERYAGE_CONTINUE_24_V1_20260906.md"

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for x in iter(lambda:f.read(1048576),b""):h.update(x)
 return h.hexdigest()
def read(p):
 x=json.loads(Path(p).read_text())
 if not isinstance(x,dict):raise RuntimeError("JSON object required")
 return x
def canon(p):
 p=Path(p)
 if not p.is_absolute() or p.resolve()!=p:raise RuntimeError("canonical absolute path required")
 return p
def inside(a,b):return os.path.commonpath((str(a),str(b)))==str(a)
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile("w",dir=p.parent,delete=False) as f:
  q=Path(f.name);json.dump(x,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def manifest(root):
 return {str(p):sha(p) for p in sorted(Path(root).rglob("*")) if p.is_file() and p.name not in ("receipt.json","FAILED_OR_INCOMPLETE.json")}
def ent(r):return {"epoch":r["epoch"],"ema_r2_float64":r["selection"]["r2_concat_float64"],"checkpoint":r["checkpoint"],"checkpoint_sha256":r["checkpoint_sha256"]}
def earliest_ema(rows):
 if [r.get("epoch") for r in rows]!=list(range(1,25)):raise RuntimeError("selection must cover exactly epochs 1..24")
 if any(not isinstance(r.get("selection",{}).get("r2_concat_float64"),(int,float)) or not math.isfinite(r["selection"]["r2_concat_float64"]) for r in rows):raise RuntimeError("selection EMA values must be finite")
 return max(enumerate(rows),key=lambda q:(q[1]["selection"]["r2_concat_float64"],-q[0]))[1]

def audit_parent(parent,*,output_for_runtime_audit=None):
 parent=canon(parent);receipt=read(parent/"receipt.json");freeze=read(parent/"selection_freeze.json");inp=read(parent/"input_authority.json")
 if receipt.get("status")!=PARENT_STATUS or receipt.get("selection_freeze")!=freeze or receipt.get("selection_freeze_sha256")!=sha(parent/"selection_freeze.json") or receipt.get("authority")!=receipt.get("authority_post") or inp.get("authority")!=receipt.get("authority") or receipt.get("owned_artifact_sha256")!=manifest(parent):raise RuntimeError("parent formal incomplete or manifest/authority drift")
 arms={}
 for arm in ARMS:
  rows=read(parent/"workers"/(arm+"_complete.json")).get("epochs")
  final=read(parent/"workers"/(arm+"_final.json")
  )
  if not isinstance(rows,list) or [r.get("epoch") for r in rows]!=list(range(1,13)) or final.get("status")!="COMPLETE_POST_FREEZE" or set(final.get("reports",{}))!={"selected","epoch12"}:raise RuntimeError("parent all-epoch/final record drift")
  for r in rows:
   ck=parent/"checkpoints"/(arm+"_epoch_%03d.pt"%r["epoch"])
   if r.get("checkpoint")!=str(ck) or r.get("checkpoint_sha256")!=sha(ck) or read(parent/"workers"/(arm+"_epoch_%03d.json"%r["epoch"]))!=r or r.get("selection",{}).get("n_bins")!=SELECTION_BINS:raise RuntimeError("parent checkpoint record drift")
  last=ent(rows[-1])
  if freeze.get("epoch12",{}).get(arm)!=last or last["checkpoint"]!=str(parent/"checkpoints"/(arm+"_epoch_012.pt")):raise RuntimeError("resume must use raw parent epoch12 checkpoint, never plain EMA")
  for k in ("selected","epoch12"):
   z=final["reports"][k]
   if z.get("epoch")!=freeze[k][arm]["epoch"] or z.get("complete",{}).get("n_bins")!=COMPLETE_BINS or sha(z["plain_ema_path"])!=z.get("plain_ema_sha256") or sha(z["complete_archive"])!=z.get("complete_archive_sha256"):raise RuntimeError("parent final exports drift")
  arms[arm]={"epoch12":last,"complete_sha256":sha(parent/"workers"/(arm+"_complete.json"))}
 # Reuse the completed proof's formal audit as a read-only independent closure
 # check.  It imports validators only; it does not construct a model or cache.
 from tfpd_exploration.src.family_runtime_v1 import complete_h1_queryage_source as complete
 audit_output=canon(output_for_runtime_audit) if output_for_runtime_audit is not None else parent.parent / (parent.name+"_continuation_audit_sentinel")
 bindings=complete.collect_bindings(parent,audit_output)
 formal,_=complete._audit_formal(parent,bindings,guard=lambda:None)
 if formal!=receipt:raise RuntimeError("independent formal audit receipt drift")
 elapsed=receipt.get("elapsed_seconds")
 if not isinstance(elapsed,(int,float)) or not math.isfinite(elapsed) or elapsed<=0:raise RuntimeError("parent elapsed evidence must be finite positive")
 return {"receipt_sha256":sha(parent/"receipt.json"),"freeze_sha256":sha(parent/"selection_freeze.json"),"input_sha256":sha(parent/"input_authority.json"),"runtime_audit_bindings_sha256":hashlib.sha256(json.dumps(bindings,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest(),"parent_elapsed_seconds":float(elapsed),"arms":arms}

def collect_authority(parent_formal,output):
 parent_formal,output=canon(parent_formal),canon(output)
 if inside(parent_formal,output):raise RuntimeError("disjoint output required")
 parent=audit_parent(parent_formal,output_for_runtime_audit=output)
 if not PROTOCOL.is_file():raise FileNotFoundError("fixed protocol missing")
 from . import continue_prefix_train as train
 bindings=train.collect_bindings(output,parent_formal=parent_formal,parent_receipt=parent_formal/"receipt.json",parent_authorization=parent_formal/"input_authority.json")
 forecast=parent["parent_elapsed_seconds"]*1.5+1800.
 if not math.isfinite(forecast) or forecast<=0 or forecast>=LIMIT:raise RuntimeError("continuation forecast is not admissible under six-hour budget")
 return {"schema":SCHEMA,"status":"ROOT_REVIEW_GO","bindings":bindings,"parent_audit":parent,"protocol":{"path":str(PROTOCOL),"sha256":sha(PROTOCOL)},"launcher_sha256":sha(Path(__file__).resolve()),"train_sha256":sha(Path(train.__file__).resolve()),"output":str(output),"limits":{"wall_seconds":LIMIT,"peak_rss_bytes":22<<30,"torch_threads":1,"torch_interop_threads":1,"physical_gpu":ARMS,"cpu_affinity":CPU},"resource_forecast":{"formula":"parent_actual_elapsed_seconds * 1.5 + 1800","parent_actual_elapsed_seconds":parent["parent_elapsed_seconds"],"forecast_seconds":forecast},"rules":{"epochs":list(range(13,25)),"selection":"EMA R2 on fixed 2908 selection across 1..24; earliest tie","endpoint":24,"disclosure":"authorized after inspecting epoch1..12 trajectory; not initially preregistered 24"}}

def validate_ready(ready,parent,parent_formal):
 if set(ready)!=set(ARMS):raise RuntimeError("both arms required")
 common=None
 for arm,x in ready.items():
  old=read(parent_formal/"barrier"/(arm+".ready.json"));restore=x.get("strict_restore_evidence")
  if (x.get("arm")!=arm or x.get("physical_gpu")!=ARMS[arm] or x.get("parent_epoch12_checkpoint_sha256")!=parent["arms"][arm]["epoch12"]["checkpoint_sha256"]
      or x.get("parent_ready_sha256")!=sha(parent_formal/"barrier"/(arm+".ready.json")) or x.get("parent_shared_init_sha256")!=old.get("shared_init_sha256")
      or not isinstance(x.get("shared_init_sha256"),str) or not x["shared_init_sha256"] or x["shared_init_sha256"]!=old.get("shared_init_sha256")
      or not isinstance(restore,dict) or set(restore)!={"raw_state_sha256","ema_state_sha256","optimizer_state_sha256","rng_state_sha256","global_step","ema_updates","lr","exact_restore"}
      or any(not isinstance(restore[k],str) or not restore[k] for k in ("raw_state_sha256","ema_state_sha256","optimizer_state_sha256","rng_state_sha256"))
      or restore.get("global_step")!=8772 or restore.get("ema_updates")!=8772 or restore.get("lr")!=1e-4 or restore.get("exact_restore") is not True
      or set(x.get("identities",{}))!={str(i) for i in range(13,25)}):raise RuntimeError("restore/barrier drift")
  now=(x.get("shared_init_sha256"),x["identities"])
  if common is None:common=now
  elif common!=now:raise RuntimeError("paired schedule identity drift")

def freeze_selection(parent,output):
 pa=audit_parent(parent,output_for_runtime_audit=output);selected={};endpoint={};old={};new={}
 ready={arm:read(output/"barrier"/(arm+".ready.json")) for arm in ARMS};start=read(output/"barrier"/"START")
 if start.get("status")!="ROOT_RELEASED_MATCHED_CONTINUATION" or start.get("ready_sha256")!={arm:sha(output/"barrier"/(arm+".ready.json")) for arm in ARMS}:raise RuntimeError("start/ready binding drift")
 for arm in ARMS:
  before=read(parent/"workers"/(arm+"_complete.json"))["epochs"]; cp=output/"workers"/(arm+"_continue_complete.json");after=read(cp).get("epochs")
  if not isinstance(after,list) or [r.get("epoch") for r in after]!=list(range(13,25)):raise RuntimeError("continuation coverage drift")
  for r in after:
   ck=output/"checkpoints"/(arm+"_epoch_%03d.pt"%r["epoch"])
   score=r.get("selection",{}).get("r2_concat_float64")
   if r.get("checkpoint")!=str(ck) or r.get("checkpoint_sha256")!=sha(ck) or read(output/"workers"/(arm+"_epoch_%03d.json"%r["epoch"]))!=r or r.get("selection",{}).get("n_bins")!=SELECTION_BINS or not isinstance(score,(int,float)) or not math.isfinite(score) or r.get("identities")!=ready[arm]["identities"].get(str(r["epoch"])) or r.get("parent_checkpoint_sha256")!=pa["arms"][arm]["epoch12"]["checkpoint_sha256"]:raise RuntimeError("continuation checkpoint drift")
  allrows=before+after; win=earliest_ema(allrows);selected[arm]=ent(win);endpoint[arm]=ent(after[-1]);old[arm]=pa["arms"][arm]["complete_sha256"];new[arm]=sha(cp)
 z={"schema":"h1_queryage_continue_selection_freeze_v1","selected":selected,"epoch24":endpoint,"parent_records_sha256":old,"continuation_records_sha256":new};atomic(output/"continuation_selection_freeze.json",z);return z

def stop(children):
 for p in children.values():
  if p.poll() is None:p.terminate()
 for p in children.values():
  try:p.wait(timeout=15)
  except subprocess.TimeoutExpired:p.kill();p.wait(timeout=15)
def close_handles(handles):
 for h in handles:
  if not h.closed:h.close()
 handles.clear()
def spawn(fun,arm,output,extra,handles):
 env={**os.environ,"CUDA_VISIBLE_DEVICES":str(ARMS[arm]),GO:"1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1","PYTHONNOUSERSITE":"1"}
 a={"arm":arm,"output":str(output),"physical_gpu":ARMS[arm],**extra}
 s="import json,sys;from pathlib import Path;from tfpd_exploration.src.h1_queryage_family_v1.continue_prefix_train import "+fun+";a=json.loads(sys.argv[1]);a={k:Path(v) if k in ('output','start_marker') else v for k,v in a.items()};"+fun+"(**a)"
 h=(output/(arm+"_"+fun+".log")).open("x");handles.append(h)
 return subprocess.Popen(["taskset","-c",CPU[arm],sys.executable,"-c",s,json.dumps(a)],cwd=Path(__file__).resolve().parents[3],env=env,stdout=h,stderr=subprocess.STDOUT)
def reject_overlap(*,parent_formal,output,authorization,fresh):
 from tfpd_exploration.src.family_runtime_v1 import complete_h1_queryage_source as complete
 roots={parent_formal,Path(__file__).resolve().parent,PROTOCOL.parent,Path(complete.__file__).resolve().parent}
 # Both the parent formal's frozen authority and the independent formal audit
 # name every input whose parent is an immutable source root for this run.
 for source in list(fresh["bindings"].get("inputs",{}))+list(read(parent_formal/"receipt.json").get("authority",{}).get("bindings",{}).get("inputs",{})):
  # Inputs normally name immutable files.  Treat a named directory as a root,
  # but do not promote each file's broad results-directory parent into a root.
  source=canon(Path(source));roots.add(source if source.is_dir() else source)
 for root in roots:
  if inside(root,output) or inside(root,authorization):raise RuntimeError("output/authorization overlaps immutable parent/source/code/protocol root")

def run(*,parent_formal,output,authorization,authorization_sha):
 parent_formal,output,authorization=canon(parent_formal),canon(output),canon(authorization)
 if os.environ.get(GO)!="1" or output.exists() or inside(parent_formal,output) or inside(output,authorization):raise RuntimeError("explicit GO/fresh output/external authority required")
 fresh=collect_authority(parent_formal,output)
 reject_overlap(parent_formal=parent_formal,output=output,authorization=authorization,fresh=fresh)
 if sha(authorization)!=authorization_sha or read(authorization)!=fresh:raise RuntimeError("external root authority drift")
 if subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip():raise RuntimeError("both GPUs must be free")
 started=time.monotonic();output.mkdir();atomic(output/"input_authority.json",{"authorization_path":str(authorization),"authorization_sha256":authorization_sha,"authority":fresh});sidecar_sha=sha(output/"input_authority.json")
 from . import continue_prefix_train as train
 train.require_authority(output)
 for d in ("barrier","workers","checkpoints","exports"):(output/d).mkdir()
 children={};handles=[]
 try:
  for arm in ARMS:children[arm]=spawn("worker_run",arm,output,{"start_marker":str(output/"barrier"/"START")},handles)
  paths={a:output/"barrier"/(a+".ready.json") for a in ARMS}
  while not all(p.is_file() for p in paths.values()):
   if any(p.poll() is not None for p in children.values()):raise RuntimeError("owned child exited before barrier; no retry")
   if time.monotonic()-started>LIMIT:raise TimeoutError("six-hour budget")
   time.sleep(.1)
  ready={a:read(p) for a,p in paths.items()};validate_ready(ready,fresh["parent_audit"],parent_formal)
  # Re-read the externally bound authority immediately before releasing the
  # common barrier; no worker may train after an admission drift.
  train.require_authority(output)
  atomic(output/"barrier"/"START",{"status":"ROOT_RELEASED_MATCHED_CONTINUATION","ready_sha256":{a:sha(p) for a,p in paths.items()}})
  while any(p.poll() is None for p in children.values()):
   if any(p.poll() not in (None,0) for p in children.values()):raise RuntimeError("worker failed; no retry")
   if time.monotonic()-started>LIMIT:raise TimeoutError("six-hour budget")
   time.sleep(.2)
  if any(p.returncode for p in children.values()):raise RuntimeError("worker failed")
  freeze=freeze_selection(parent_formal,output);freeze_sha=sha(output/"continuation_selection_freeze.json");children={a:spawn("finalizer_run",a,output,{},handles) for a in ARMS}
  while any(p.poll() is None for p in children.values()):
   if any(p.poll() not in (None,0) for p in children.values()):raise RuntimeError("finalizer failed; no retry")
   if time.monotonic()-started>LIMIT:raise TimeoutError("six-hour budget")
   time.sleep(.2)
  if any(p.returncode for p in children.values()):raise RuntimeError("finalizer failed")
  close_handles(handles)
  finals={a:read(output/"workers"/(a+"_final.json")) for a in ARMS}
  for a,x in finals.items():
   if x.get("status")!="COMPLETE_POST_FREEZE" or set(x.get("reports",{}))!={"selected","epoch24"}:raise RuntimeError("finalizer topology drift")
   for label,r in x["reports"].items():
    archive=output/"exports"/(a+"_"+label+"_complete_native_float64.npz");plain=output/"exports"/(a+"_"+label+"_plain_ema.pt")
    reproduced=r.get("selection_reproduced",{});score=reproduced.get("r2_concat_float64")
    if (r.get("epoch")!=freeze[label][a]["epoch"] or r.get("checkpoint_sha256")!=freeze[label][a]["checkpoint_sha256"]
        or reproduced.get("n_bins")!=SELECTION_BINS or not isinstance(score,(int,float)) or not math.isfinite(score) or abs(score-freeze[label][a]["ema_r2_float64"])>1e-5
        or r.get("complete",{}).get("n_bins")!=COMPLETE_BINS or r.get("complete_archive")!=str(archive) or r.get("plain_ema_path")!=str(plain) or sha(archive)!=r.get("complete_archive_sha256") or sha(plain)!=r.get("plain_ema_sha256")):raise RuntimeError("complete four-export drift")
  # The parent and external root approval must still be byte-identical after
  # both children have exited; our sidecar is frozen before any child exists.
  sidecar=output/"input_authority.json"
  if read(sidecar)!={"authorization_path":str(authorization),"authorization_sha256":authorization_sha,"authority":fresh} or sha(authorization)!=authorization_sha or read(authorization)!=fresh or collect_authority(parent_formal,output)!=fresh or sha(sidecar)!=sidecar_sha or sha(output/"continuation_selection_freeze.json")!=freeze_sha:raise RuntimeError("post closure drift")
  z={"schema":"h1_queryage_continue_24_complete_pair_v1","status":"COMPLETE_FIXED_CONTINUATION_NO_PROMOTION","parent_formal":str(parent_formal),"selection_freeze":freeze,"selection_freeze_sha256":freeze_sha,"finals":finals,"authority":fresh,"authority_post":fresh,"authorization_path":str(authorization),"authorization_sha256":authorization_sha,"elapsed_seconds":time.monotonic()-started,"owned_artifact_sha256":manifest(output)};atomic(output/"receipt.json",z);return z
 except BaseException as e:
  stop(children);atomic(output/"FAILED_OR_INCOMPLETE.json",{"status":"FAILED_OR_INCOMPLETE_NO_RETRY","reason":repr(e),"children":{a:p.returncode for a,p in children.items()}});raise
 finally:
  close_handles(handles)
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--parent-formal",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--authorization",type=Path,required=True);p.add_argument("--authorization-sha",required=True);a=p.parse_args(argv);return run(parent_formal=a.parent_formal,output=a.output,authorization=a.authorization,authorization_sha=a.authorization_sha)
if __name__=="__main__":main()
