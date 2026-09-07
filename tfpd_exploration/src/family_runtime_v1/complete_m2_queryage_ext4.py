"""Externally authorized CPU-T1 complete chronological QueryAge proof."""
from __future__ import annotations
import argparse, hashlib, json, os, resource, tempfile, time
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from tfpd_exploration.src.m2_dual_track_v1 import data, plan
from tfpd_exploration.src.m2_queryage_family_v1 import evaluate_selected_ext4 as native
from tfpd_exploration.src.m2_same_query_comparator_v1 import core
from . import complete_m2_family_source as source_complete
from .m2_family_queryage import M2FamilyQueryAgeRuntime, M2RuntimeBank
from .m2_family_queryage_cached import M2FamilyQueryAgeCachedRuntime

GO_ENV="M2_QUERYAGE_COMPLETE_EXT4_GO"; EVALUATOR_SHA256="c1d0410a8ef8e642ea50585bec816172fa51afcb0ced2beb4ceedda3436b291d"
SCHEMA="family_runtime_v1_m2_queryage_complete_ext4_v2"; HARD_SECONDS=3600; HARD_RSS_BYTES=4<<30
EXPECTED_ENDPOINTS=2069; EXPECTED_PUBLIC_RAW_BINS=10839
RUNTIMES={"uncached":M2FamilyQueryAgeRuntime,"cached":M2FamilyQueryAgeCachedRuntime}
def sha(p:Path)->str:
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def _json(p):
 try:v=json.loads(Path(p).read_text())
 except (OSError,json.JSONDecodeError) as e:raise RuntimeError(f"invalid JSON: {p}") from e
 if not isinstance(v,dict):raise RuntimeError("JSON object required")
 return v
def _atomic_json(p,v):
 with tempfile.NamedTemporaryFile("w",dir=p.parent,delete=False) as f:
  t=Path(f.name);json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def _atomic_npz(p,a):
 with tempfile.NamedTemporaryFile(dir=p.parent,suffix=".npz",delete=False) as f:t=Path(f.name)
 np.savez_compressed(t,**a);os.replace(t,p);return sha(p)
def _inside(parent,child):return os.path.commonpath((str(parent),str(child)))==str(parent)
def _sha(v,label):
 if not isinstance(v,str) or len(v)!=64 or any(c not in "0123456789abcdef" for c in v):raise RuntimeError(f"{label}: lowercase SHA-256 required")
def _guard(started):
 if torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") not in ("","-1"):raise RuntimeError("fresh CPU-only process required")
 if torch.get_num_threads()!=1 or torch.get_num_interop_threads()!=1:raise RuntimeError("exactly one torch and interop CPU thread required")
 if time.monotonic()-started>HARD_SECONDS:raise TimeoutError("3600-second complete proof budget exceeded")
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>HARD_RSS_BYTES:raise RuntimeError("4-GiB RSS budget exceeded")
def _digest(model):return {k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()}
def runtime_code():
 out=dict(source_complete.runtime_code())
 for p in (Path(__file__).with_name("m2_family_queryage.py"),Path(__file__).with_name("m2_family_queryage_cached.py"),Path(__file__)):
  if not p.is_file():raise RuntimeError("QueryAge runtime code missing")
  out[p.name]=sha(p)
 return out

def _native_authority(run_root,finalizer_root,finalizer_sha,native_root,native_sha):
 _sha(finalizer_sha,"finalizer receipt");_sha(native_sha,"native receipt")
 if sha(Path(native.__file__))!=EVALUATOR_SHA256:raise RuntimeError("stable QueryAge evaluator code drift")
 native_root=Path(native_root);rp=native_root/"receipt.json"
 if not native_root.is_absolute() or not rp.is_file() or sha(rp)!=native_sha:raise RuntimeError("hash-bound completed native receipt required")
 fresh=native.authority(Path(run_root),Path(finalizer_root),finalizer_sha);fresh["output"]=str(native_root)
 receipt=_json(rp)
 if receipt.get("schema")!=native.SCHEMA or receipt.get("status")!="COMPLETE_FIXED_SELECTED_EXT4_DEVELOPMENT_ONLY" or receipt.get("pre")!=fresh or receipt.get("post")!=fresh:raise RuntimeError("native receipt fresh pre/post authority drift")
 arms=receipt.get("arms")
 if not isinstance(arms,Mapping) or set(arms)!={"FLAT","ROUTE"}:raise RuntimeError("exact native paired arms required")
 for arm in ("FLAT","ROUTE"):
  r,s=arms[arm],fresh["selected"][arm]
  if r.get("selected")!=s:raise RuntimeError(f"{arm}: native selected/fresh authority mismatch")
  p,h=r.get("native_archive_path"),r.get("native_archive_sha256")
  if not isinstance(p,str) or not Path(p).is_absolute() or not isinstance(h,str) or not Path(p).is_file() or sha(Path(p))!=h:raise RuntimeError(f"{arm}: native gold archive drift")
 return receipt,fresh
def collect_bindings(run_root,finalizer_root,finalizer_sha,native_root,native_sha,output,runtime_kind="uncached"):
 if runtime_kind not in RUNTIMES:raise ValueError("runtime must be uncached or cached")
 roots=tuple(map(Path,(run_root,finalizer_root,native_root)));output=Path(output)
 if not output.is_absolute() or output.exists() or any(not p.is_absolute() for p in roots) or any(_inside(p,output) for p in roots):raise RuntimeError("fresh absolute output separate from all input roots required")
 receipt,authority=_native_authority(*roots[:2],finalizer_sha,roots[2],native_sha)
 return {"schema":SCHEMA,"output":str(output),"runtime_kind":runtime_kind,"native_receipt_sha256":native_sha,"native_authority":authority,"native_receipt_status":receipt["status"],"runtime_code_sha256":runtime_code()}
def _authorize(binding,authorization,authorization_sha):
 _sha(authorization_sha,"external root authorization");authorization=Path(authorization)
 if not authorization.is_absolute() or not authorization.is_file() or sha(authorization)!=authorization_sha:raise RuntimeError("hash-bound external root authorization required")
 if _json(authorization).get("status")!="ROOT_REVIEW_GO" or _json(authorization).get("authority")!=binding:raise RuntimeError("external authorization does not bind full fresh authority/output")
def _load_model(arm,selected):
 p=Path(selected["path"])
 if sha(p)!=selected["sha256"]:raise RuntimeError("selected export mutated immediately before deserialize")
 m=native.make_paired_queryage_decoders(42)[0 if arm=="FLAT" else 1];state=torch.load(p,map_location="cpu",weights_only=True)
 if not isinstance(state,Mapping) or not state or any(not torch.is_tensor(v) or v.dtype!=torch.float32 or not bool(torch.isfinite(v).all()) for v in state.values()):raise RuntimeError("finite FP32 plain EMA required")
 m.load_state_dict(state,strict=True);return m.eval()
def _load_gold(path,expected_sha,rows):
 if sha(path)!=expected_sha:raise RuntimeError("native gold archive changed immediately before NPZ read")
 with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 native.validate_archive(a,{"rows":rows});return a

def _stream(arm,selected,gold,rows,started,heartbeat,runtime_kind="uncached"):
 if runtime_kind not in RUNTIMES:raise ValueError("runtime must be uncached or cached")
 _guard(started);model=_load_model(arm,selected);_guard(started);before=_digest(model);parts=[];reports=[];offset=public=0
 for sealed in rows:
  session,count=sealed["session"],int(sealed["window_count"]);folder=data.cache_root()/"ext4"/session
  if not folder.is_dir() or any(not(folder/n).is_file() for n in native.REQUIRED_CACHE):raise RuntimeError(f"{session}: all nine existing cache files required")
  if {n:sha(folder/n) for n in native.REQUIRED_CACHE}!=sealed["files"]:raise RuntimeError(f"{session}: cache bytes drift before load")
  raw,starts,target=(np.load(folder/n,mmap_mode="r") for n in ("X_store.npy","eligible_starts.npy","target_store.npy"));sl=slice(offset,offset+count)
  if raw.dtype!=np.float32 or starts.dtype!=np.int64 or target.dtype!=np.float32 or starts.shape!=(count,) or target.shape!=(count,plan.OUT_DIM) or not np.array_equal(gold["target"][sl],target.astype(np.float64)) or not np.array_equal(gold["start"][sl],starts) or not np.array_equal(gold["session"][sl],np.asarray([session]*count)) or core.array_sha256(starts)!=sealed["ordered_window_starts_sha256"] or core.array_sha256(target)!=sealed["target_sha256"]:raise RuntimeError(f"{session}: sealed raw/start/target identity drift")
  _guard(started);source=data.load_session_bank("ext4",session,device="cpu");_guard(started);runtime=RUNTIMES[runtime_kind](model,M2RuntimeBank(source.E0,source.T,source.unit_mask));_guard(started);endpoint={int(s+plan.WINDOW-1):i for i,s in enumerate(starts)};obs=[];seen=[];eg=ed=0.
  for tick in range(plan.WINDOW-1,len(raw)):
   _guard(started);got=runtime.predict(np.array(raw[tick:tick+1],dtype=np.float32,copy=True));_guard(started);public+=1
   if got.shape!=(1,plan.OUT_DIM) or got.dtype!=np.float32 or not got.flags.owndata or not got.flags.c_contiguous or not np.isfinite(got).all():raise RuntimeError("public B1 output contract drift")
   if public%128==0:heartbeat(arm,session,public)
   i=endpoint.get(tick)
   if i is None:continue
   w=np.array(raw[int(starts[i]):int(starts[i])+plan.WINDOW],dtype=np.float32,copy=True)
   if not np.array_equal(runtime.raw[0].numpy(),w):raise RuntimeError(f"{session}: W50 public history drift")
   _guard(started)
   with torch.inference_mode():direct=(model.forward_last(torch.from_numpy(w[None]),runtime.bank,runtime.bank.unit_mask)/plan.BEHAVIOR_SCALE).numpy()
   _guard(started);np.testing.assert_allclose(got,direct,atol=1e-5,rtol=1e-5);np.testing.assert_allclose(got.astype(np.float64),gold["prediction"][offset+i:offset+i+1],atol=1e-5,rtol=1e-5)
   eg=max(eg,float(np.abs(got.astype(np.float64)-gold["prediction"][offset+i:offset+i+1]).max()));ed=max(ed,float(np.abs(got-direct).max()));obs.append(got[0]);seen.append(i)
  if seen!=list(range(count)):raise RuntimeError(f"{session}: endpoint order/cardinality drift")
  pred=np.asarray(obs,dtype=np.float64);session_r2=float(core.variance_weighted_r2(target,pred));gold_r2=float(core.variance_weighted_r2(gold["target"][sl],gold["prediction"][sl]))
  if abs(session_r2-gold_r2)>1e-5:raise RuntimeError(f"{session}: public/native-gold session R2 drift")
  reports.append({"session":session,"endpoint_count":count,"public_raw_bins":len(raw)-plan.WINDOW+1,"native_r2":session_r2,"native_gold_r2":gold_r2,"max_public_native_gold_abs_error":eg,"max_public_direct_abs_error":ed});parts.append(pred);offset+=count
 if offset!=EXPECTED_ENDPOINTS or public!=EXPECTED_PUBLIC_RAW_BINS:raise RuntimeError("global 2069/10839 cardinality drift")
 if _digest(model)!=before:raise RuntimeError("selected model state mutated during inference")
 prediction=np.concatenate(parts);arrays={"prediction":prediction,"target":gold["target"],"start":gold["start"],"session":gold["session"]}
 return {"rows":reports,"endpoint_count":offset,"public_raw_bins":public,"equal_session_r2":float(np.mean([r["native_r2"] for r in reports])),"pooled_r2":float(core.variance_weighted_r2(arrays["target"],prediction)),"max_public_native_gold_abs_error":max(r["max_public_native_gold_abs_error"] for r in reports),"max_public_direct_abs_error":max(r["max_public_direct_abs_error"] for r in reports)},arrays

def run(run_root,finalizer_root,finalizer_sha,native_root,native_sha,output,authorization,authorization_sha,runtime_kind="uncached"):
 started=time.monotonic()
 if torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") not in ("","-1"):raise RuntimeError("fresh CPU-only process required")
 if os.environ.get(GO_ENV)!="1":raise RuntimeError("explicit reviewed GO required")
 torch.set_num_threads(1)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:
  if torch.get_num_interop_threads()!=1:raise
 _guard(started)
 binding=collect_bindings(run_root,finalizer_root,finalizer_sha,native_root,native_sha,output,runtime_kind);_authorize(binding,authorization,authorization_sha);_guard(started);output=Path(output);output.mkdir(parents=True);_atomic_json(output/"authority_pre.json",{"authority":binding,"authorization_sha256":authorization_sha});pre_file_sha=sha(output/"authority_pre.json")
 receipt=_json(Path(native_root)/"receipt.json");rows=receipt["pre"]["rows"];results={}
 def heart(a,s,n):_guard(started);_atomic_json(output/"heartbeat.json",{"status":"SCORING","arm":a,"session":s,"public_calls":n,"elapsed_seconds":time.monotonic()-started})
 for arm in ("FLAT","ROUTE"):
  r=receipt["arms"][arm];gold=_load_gold(Path(r["native_archive_path"]),r["native_archive_sha256"],rows);item,arrays=_stream(arm,binding["native_authority"]["selected"][arm],gold,rows,started,heart,runtime_kind)
  if abs(item["equal_session_r2"]-r["equal_session_r2"])>1e-5 or abs(item["pooled_r2"]-r["pooled_r2"])>1e-5:raise RuntimeError(f"{arm}: R2 reproduction drift")
  sealed_rows=r.get("rows")
  if not isinstance(sealed_rows,list) or len(sealed_rows)!=len(item["rows"]):raise RuntimeError(f"{arm}: native per-session receipt rows drift")
  for live,sealed in zip(item["rows"],sealed_rows,strict=True):
   if live["session"]!=sealed.get("session") or abs(live["native_r2"]-float(sealed.get("native_r2")))>1e-5:raise RuntimeError(f"{arm}: per-session native R2 drift")
  p=output/f"{arm}_public_native_float64.npz";item["archive_path"],item["archive_sha256"]=str(p),_atomic_npz(p,arrays)
  if sha(p)!=item["archive_sha256"]:raise RuntimeError("fresh own NPZ hash drift")
  results[arm]=item
 _guard(started);post_receipt,post_native=_native_authority(run_root,finalizer_root,finalizer_sha,native_root,native_sha);post={**binding,"native_authority":post_native,"runtime_code_sha256":runtime_code()}
 if post!=binding or post_receipt!=receipt:raise RuntimeError("fresh authority changed during complete proof")
 _authorize(binding,authorization,authorization_sha)
 if sha(output/"authority_pre.json")!=pre_file_sha or any(sha(Path(item["archive_path"]))!=item["archive_sha256"] for item in results.values()):raise RuntimeError("own authority or output archive mutated before final receipt")
 _guard(started)
 out={"schema":SCHEMA,"status":"PASS_COMPLETE_QUERYAGE_STREAM_EQUIVALENCE_ONLY","runtime_kind":runtime_kind,"authority_pre":binding,"authority_post":post,"authority_pre_sha256":pre_file_sha,"external_authorization_sha256":authorization_sha,"results":results,"elapsed_seconds":time.monotonic()-started,"threads":torch.get_num_threads(),"interop_threads":torch.get_num_interop_threads(),"peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"hard_rss_bytes":HARD_RSS_BYTES,"parameter_updates":0,"new_selection_or_calibration":False,"not_official_latency":True};_atomic_json(output/"receipt.json",out);return out
def main(argv=None):
 p=argparse.ArgumentParser()
 for n in ("run-root","finalizer-root","native-root","output","authorization"):p.add_argument("--"+n,required=True,type=Path)
 for n in ("finalizer-sha","native-sha","authorization-sha"):p.add_argument("--"+n,required=True)
 p.add_argument("--runtime-kind",choices=tuple(RUNTIMES),default="uncached")
 a=p.parse_args(argv);return run(a.run_root,a.finalizer_root,a.finalizer_sha,a.native_root,a.native_sha,a.output,a.authorization,a.authorization_sha,a.runtime_kind)
if __name__=="__main__":main()
