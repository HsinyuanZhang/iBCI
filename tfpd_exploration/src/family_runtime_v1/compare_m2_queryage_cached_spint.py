"""Externally authorized same-image B7 timing plan for selected QueryAge runtimes.

This is deliberately prospective: it never runs automatically and refuses
until finalization and both cached/uncached complete-stream proofs are bound.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, inspect, json, os, resource, tempfile, time
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch

from tfpd_exploration.src.m2_dual_track_v1 import data, plan
from tfpd_exploration.src.m2_queryage_family_v1 import finalize_pair as finalizer
from tfpd_exploration.src.m2_queryage_family_v1 import evaluate_selected_ext4 as native
from tfpd_exploration.src.m2_queryage_family_v1.model import make_paired_queryage_decoders
from .benchmark import stats
from .m2_family_causal import M2RuntimeBank
from .m2_family_queryage_cached import M2FamilyQueryAgeCachedRuntime
from . import complete_m2_queryage_ext4 as complete
from .m2_spint_comparison import AS_SHIPPED, EXPECTED, WRAPPER

ROOT=Path(__file__).resolve().parents[3]
GO="M2_QUERYAGE_CACHED_SPINT_GO"; IMAGE="sha256:8de56c58939ebd8306954ea7d180aceb7269fd3df28192f11df0dcaa7b60dc7f"; SPINT_SHA="855b9d06e9c22ca1c1ded78df4ec15af875707bace78b5528499f05f7a212519"
SCHEMA="m2_queryage_selected_cached_vs_original_image_public_b7_latency_v1"; W=50
HARD_SECONDS=3600; HARD_RSS_BYTES=8<<30

def sha(p:Path)->str:
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def _json(p):
 x=json.loads(Path(p).read_text());
 if not isinstance(x,dict):raise RuntimeError("JSON object required")
 return x
def _canonical(value):
 """Use the JSON form that an external authorization/receipt actually binds."""
 return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))
def _hash(v,label):
 if not isinstance(v,str) or len(v)!=64 or any(c not in "0123456789abcdef" for c in v):raise RuntimeError(label+" SHA required")
def _public(name,x):
 if x.shape!=(7,2) or x.dtype!=np.float32 or not x.flags.owndata or not x.flags.c_contiguous or not np.isfinite(x).all():raise RuntimeError(name+": public B7 output contract")
def _tag(s):
 p=s.removeprefix("ses-").split("-");return f"{p[-1]}_{''.join(p[:3])}"

def _inside(parent,child):return os.path.commonpath((str(Path(parent)),str(Path(child))))==str(Path(parent))
def _guard(started,threads=None):
 if torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") not in ("","-1"):raise RuntimeError("fresh CPU-only process required")
 if torch.get_num_threads()!=(threads if threads is not None else 1) or torch.get_num_interop_threads()!=1:raise RuntimeError("CPU thread/interop authority drift")
 if time.monotonic()-started>HARD_SECONDS:raise TimeoutError("3600-second benchmark budget exceeded")
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>HARD_RSS_BYTES:raise RuntimeError("8-GiB benchmark RSS budget exceeded")

def _atomic_json(path,value):
 with tempfile.NamedTemporaryFile("w",dir=Path(path).parent,delete=False) as f:
  temp=Path(f.name);json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
 os.replace(temp,path)

def _proof(kind,root,expected):
 """Require the completed proof's own pre/post immutable closure."""
 p=Path(root)/"receipt.json"
 if not p.is_file() or sha(p)!=expected:raise RuntimeError(kind+" proof receipt drift")
 body=_json(p)
 if body.get("status")!="PASS_COMPLETE_QUERYAGE_STREAM_EQUIVALENCE_ONLY" or body.get("runtime_kind")!=kind:raise RuntimeError(kind+" complete proof missing")
 pre,post=body.get("authority_pre"),body.get("authority_post")
 if not isinstance(pre,Mapping) or pre!=post or not isinstance(pre.get("runtime_code_sha256"),Mapping):raise RuntimeError(kind+" proof lacks stable pre/post runtime closure")
 if dict(pre["runtime_code_sha256"])!=complete.runtime_code():raise RuntimeError(kind+" completed runtime code closure drift")
 na=pre.get("native_authority");native_sha=pre.get("native_receipt_sha256")
 if not isinstance(na,Mapping) or not isinstance(native_sha,str):raise RuntimeError(kind+" native authority missing")
 _,fresh=complete._native_authority(Path(na["run_root"]),Path(na["finalizer_root"]),na["finalizer_receipt_sha256"],Path(na["output"]),native_sha)
 if fresh!=na:raise RuntimeError(kind+" current native authority drift")
 if set(body.get("results",{}))!={"FLAT","ROUTE"}:raise RuntimeError(kind+" proof requires exact FLAT/ROUTE results")
 # The complete-proof schema stores its fixed sidecar SHA, not a path field.
 authority_path,authority_sha=str(Path(root)/"authority_pre.json"),body.get("authority_pre_sha256")
 if body.get("authority_pre_path",authority_path)!=authority_path:raise RuntimeError(kind+" unexpected proof sidecar location")
 if not isinstance(authority_path,str) or not isinstance(authority_sha,str) or not Path(authority_path).is_file() or sha(Path(authority_path))!=authority_sha:raise RuntimeError(kind+" proof pre-authority sidecar drift")
 if _json(authority_path).get("authority")!=pre:raise RuntimeError(kind+" proof sidecar authority drift")
 for arm,item in body["results"].items():
  archive=item.get("archive_path");digest=item.get("archive_sha256")
  if arm not in ("FLAT","ROUTE") or not isinstance(archive,str) or not Path(archive).is_file() or sha(Path(archive))!=digest:raise RuntimeError(kind+" proof archive drift")
 return {"path":str(p),"sha256":expected,"runtime_code_sha256":dict(pre["runtime_code_sha256"]),"native_authority":dict(na),"result_archives":{a:{"path":v["archive_path"],"sha256":v["archive_sha256"]} for a,v in body["results"].items()}}

def _revalidate(binding:Mapping[str,Any],source_hashes:Mapping[str,Any]|None=None):
 """Cheap, no-model revalidation used immediately before and after timings."""
 if binding.get("schema")!=SCHEMA or binding.get("image")!=IMAGE:raise RuntimeError("benchmark schema/image authority drift")
 for p,h in binding.get("code",{}).items():
  if not Path(p).is_file() or sha(Path(p))!=h:raise RuntimeError("benchmark code closure drift")
 for arm in ("FLAT","ROUTE"):
  item=binding.get("selected",{}).get(arm,{})
  if not isinstance(item,Mapping) or not Path(item.get("path","")).is_file() or sha(Path(item["path"]))!=item.get("sha256"):raise RuntimeError(arm+" selected export drift")
 for kind,item in binding.get("proofs",{}).items():
  if _proof(kind,Path(item["path"]).parent,item["sha256"])!=item:raise RuntimeError(kind+" proof authority drift")
  native_authority=item["native_authority"]
  if (native_authority.get("run_root")!=binding.get("run_root") or native_authority.get("finalizer_root")!=binding.get("finalizer_root")
      or native_authority.get("selected")!=binding.get("selected")):raise RuntimeError(kind+" proof is from a different selected experiment")
 if "run_root" in binding:
  audit,receipt,selected=native._selected_exports(Path(binding["run_root"]),Path(binding["finalizer_root"]),binding["finalizer_receipt_sha256"])
  protocol=_json(Path(binding["run_root"])/"protocol_preconstruction.json")
  trainer=finalizer._current_trainer_authority(protocol)
  if _canonical(audit)!=binding.get("finalizer_audit") or selected!=binding.get("selected") or receipt.get("authority_pre_model")!=binding.get("finalizer_source_minival_authority") or trainer!=binding.get("trainer_source_train_authority"):raise RuntimeError("current trainer/finalizer/source authority drift")
 for p,h in binding.get("original",{}).items():
  if not Path(p).is_file() or sha(Path(p))!=h:raise RuntimeError("frozen original wrapper/payload drift")
 if binding.get("calls")==2048:
  smokes=binding.get("required_current_smoke_receipts",{})
  if set(smokes)!={"T1","T2"}:raise RuntimeError("formal timing requires T1/T2 smoke receipts")
  for name,item in smokes.items():
   if not Path(item["path"]).is_file() or sha(Path(item["path"]))!=item["sha256"]:raise RuntimeError(name+" smoke receipt drift")
   body=_json(item["path"]);a=body.get("authority",{})
   if body.get("status")!="PASS_PUBLIC_STREAM_TIMING_ONLY" or body.get("calls")!=32 or a.get("calls")!=32 or a.get("threads")!=(1 if name=="T1" else 2):raise RuntimeError(name+" smoke mode drift")
   for key in ("selected","source_raw_sha256","proofs","code","original","image","spint_module_sha256"):
    if a.get(key)!=binding.get(key):raise RuntimeError(name+" smoke authority drift")
   ap,ah=body.get("authority_pre_path"),body.get("authority_pre_sha256")
   if not isinstance(ap,str) or not isinstance(ah,str) or not Path(ap).is_file() or sha(Path(ap))!=ah or _json(ap).get("authority")!=a:raise RuntimeError(name+" smoke sidecar authority drift")
 if source_hashes is not None and source_hashes!=binding.get("source_raw_sha256"):raise RuntimeError("seven-lane source authority drift")

def collect_bindings(run_root:Path,finalizer_root:Path,finalizer_sha:str,proof_roots:Mapping[str,Path],proof_shas:Mapping[str,str],output:Path,*,calls:int,warmup:int,threads:int,smoke_receipts=None)->dict[str,Any]:
 if calls not in (32,2048) or warmup!=128 or threads not in (1,2):raise RuntimeError("only 32/2048 calls, warmup128, T1/T2")
 if not all(Path(p).is_absolute() for p in (run_root,finalizer_root,output)) or output.exists() or _inside(run_root,output) or _inside(finalizer_root,output):raise RuntimeError("fresh absolute output separate from inputs required")
 _hash(finalizer_sha,"finalizer receipt");rp=finalizer_root/"receipt.json";fp=finalizer_root/"selection_freeze.json"
 if sha(rp)!=finalizer_sha:raise RuntimeError("finalizer receipt drift")
 audit,receipt,selected=native._selected_exports(run_root,finalizer_root,finalizer_sha);audit=_canonical(audit)
 freeze=_json(fp)
 if _canonical(freeze.get("audit"))!=audit:raise RuntimeError("finalizer freeze drift")
 proofs={}
 for kind in ("uncached","cached"):
  root,expected=Path(proof_roots[kind]),proof_shas[kind];_hash(expected,kind+" proof")
  if not root.is_absolute() or _inside(root,output):raise RuntimeError("proof roots must be absolute and separate from output")
  proofs[kind]=_proof(kind,root,expected)
 cached_path=Path(inspect.getsourcefile(M2FamilyQueryAgeCachedRuntime) or "")
 code_paths=(Path(__file__),Path(finalizer.__file__),cached_path,Path(make_paired_queryage_decoders.__code__.co_filename))
 if any(not p.is_file() for p in code_paths):raise RuntimeError("benchmark/runtime/model source closure missing")
 protocol=_json(run_root/"protocol_preconstruction.json")
 trainer_source_train_authority=finalizer._current_trainer_authority(protocol)
 _,_,_,source_hashes=_source(1+warmup+calls,trainer_source_train_authority)
 binding={"schema":SCHEMA,"run_root":str(run_root),"finalizer_root":str(finalizer_root),"finalizer_receipt_sha256":finalizer_sha,"finalizer_audit":audit,"output":str(output),"calls":calls,"warmup":warmup,"threads":threads,"interop_threads":1,"image":IMAGE,"spint_module_sha256":SPINT_SHA,"selected":selected,"finalizer_source_minival_authority":receipt["authority_pre_model"],"trainer_source_train_authority":trainer_source_train_authority,"proofs":proofs,"code":{str(p):sha(p) for p in code_paths},"source_raw_sha256":source_hashes,"original":{str(p):v for p,v in EXPECTED.items()}}
 if calls==2048:
  if not isinstance(smoke_receipts,Mapping) or set(smoke_receipts)!={"T1","T2"}:raise RuntimeError("formal 2048 timing requires T1/T2 smoke receipts")
  binding["required_current_smoke_receipts"]={}
  for name,item in smoke_receipts.items():
   p,h=Path(item["path"]),item["sha256"];_hash(h,name+" smoke")
   binding["required_current_smoke_receipts"][name]={"path":str(p),"sha256":h}
  _revalidate(binding)
 return binding

def _source(count,authority=None):
 rows=[];e0=[];t=[];mask=[];hashes={}
 if not isinstance(authority,Mapping):raise RuntimeError("required completed trainer source-train authority missing")
 for s in plan.HELDIN_SESSIONS:
  d=plan.active_run_root()/"cache"/"source_train"/s
  paths=[d/n for n in ("X_store.npy","target_store.npy","eligible_starts.npy","T.npy","e0_u.pt","mapping.json","provenance.json","extra.json","calib_activity.npy")]
  if any(not p.is_file() for p in paths):raise RuntimeError("existing source bank missing")
  hashes[s]={p.name:sha(p) for p in paths}
  expected={str(p):authority.get(str(p)) for p in paths} if isinstance(authority,Mapping) else None
  if expected is not None and (any(v is None for v in expected.values()) or expected!={str(p):sha(p) for p in paths}):raise RuntimeError("source-train authority hash drift before load")
  raw=np.load(d/"X_store.npy",mmap_mode="r")
  if raw.dtype!=np.float32 or raw.ndim!=2 or raw.shape[1]!=96 or len(raw)<49+count or not np.all(raw[:49]==0) or not np.isfinite(raw[49:49+count]).all():raise RuntimeError("source raw FP32 W50 geometry drift")
  rows.append(np.array(raw[49:49+count],dtype=np.float32,copy=True)); payload=torch.load(d/"e0_u.pt",map_location="cpu",weights_only=False);E=payload.get("E0");T=np.load(d/"T.npy")
  if not isinstance(E,torch.Tensor) or E.shape!=(96,50) or E.dtype!=torch.float32 or T.shape!=(96,4) or T.dtype!=np.float32 or not bool(torch.isfinite(E).all()) or not np.isfinite(T).all():raise RuntimeError("source bank E0/T FP32 geometry drift")
  e0.append(E.contiguous());t.append(torch.from_numpy(T).contiguous());mask.append(torch.ones(96,dtype=torch.bool))
 if any(x.shape!=(count,96) for x in rows):raise RuntimeError("no repeated/insufficient B7 bins")
 return [_tag(s) for s in plan.HELDIN_SESSIONS],M2RuntimeBank(torch.stack(e0),torch.stack(t),torch.stack(mask)),np.stack(rows,1),hashes

def _build(wrapper,config,binding,tags,bank):
 constructor={};reset={};begun=time.perf_counter_ns();original=wrapper.T4CachedIdentityDecoder(config,str(AS_SHIPPED),batch_size=7);constructor["ORIGINAL"]=(time.perf_counter_ns()-begun)/1e6;begun=time.perf_counter_ns();original.reset([Path(x) for x in tags]);reset["ORIGINAL"]=(time.perf_counter_ns()-begun)/1e6;engines={"ORIGINAL":original}
 if (getattr(original,"device",None) is None or original.device.type!="cpu" or getattr(original,"smooth_observations",None) is not False
     or not all(hasattr(original,name) for name in ("observation_buffer","local_identities","_decode_with_identity","behavior_scaling_factor"))):raise RuntimeError("frozen Original CPU/unsmoothed/oracle contract drift")
 for arm,index in (("FLAT",0),("ROUTE",1)):
  item=binding["selected"][arm]
  if sha(Path(item["path"]))!=item["sha256"]:raise RuntimeError("export changed before load")
  begun=time.perf_counter_ns();m=make_paired_queryage_decoders(42)[index];m.load_state_dict(torch.load(item["path"],map_location="cpu",weights_only=True),strict=True);m.eval();engines[arm]=M2FamilyQueryAgeCachedRuntime(m,bank,batch_size=7);constructor[arm]=(time.perf_counter_ns()-begun)/1e6;begun=time.perf_counter_ns();engines[arm].reset(bank,batch=7);reset[arm]=(time.perf_counter_ns()-begun)/1e6
 return engines,constructor,reset

def _model_digest(model):
 return {k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()}

def _spint_oracle(engine):
 """Independent full decode, intentionally outside timed public calls."""
 neural=torch.as_tensor(engine.observation_buffer.copy().transpose(1,0,2),dtype=torch.float32,device=engine.device)
 with torch.inference_mode():rows=[engine._decode_with_identity(neural[i:i+1],identity)[:,-1,:] for i,identity in enumerate(engine.local_identities)]
 return (torch.cat(rows).cpu().numpy()/engine.behavior_scaling_factor).astype(np.float32,copy=True)

def _production_contract(binding,authorization):
 """A hand-written authorization cannot silently omit real proof inputs."""
 if set(binding.get("selected",{}))!={"FLAT","ROUTE"} or set(binding.get("proofs",{}))!={"uncached","cached"}:raise RuntimeError("production requires exact selected arms and both complete proofs")
 for key in ("code","trainer_source_train_authority","finalizer_source_minival_authority","finalizer_audit","source_raw_sha256","original"):
  if not isinstance(binding.get(key),Mapping) or not binding[key]:raise RuntimeError("production requires nonempty "+key)
 out=Path(binding.get("output","") or "");auth=Path(authorization)
 if not out.is_absolute() or out.exists() or not out.parent.is_dir() or out.with_name(out.name+".authority_pre.json").exists():raise RuntimeError("production requires fresh absolute output and sidecar")
 roots=[Path(binding.get(k,"") or "") for k in ("run_root","finalizer_root")]
 for kind,item in binding["proofs"].items():
  roots.append(Path(item["path"]).parent)
  na=item.get("native_authority",{})
  if na.get("finalizer_receipt_sha256")!=binding.get("finalizer_receipt_sha256"):raise RuntimeError(kind+" production finalizer receipt authority mismatch")
  roots.append(Path(na.get("output","") or ""))
 if not auth.is_absolute() or not auth.is_file() or auth in (out,out.with_name(out.name+".authority_pre.json")) or any(not p.is_absolute() or _inside(p,out) or _inside(p,auth) for p in roots):raise RuntimeError("production output/authorization must be separate from all input roots")

def run(binding:Mapping[str,Any],*,authorization:Path,authorization_sha:str,wrapper=None,config=None,source=None,allow_cpu_fixture=False)->dict[str,Any]:
 started=time.monotonic();actual_original=wrapper is None
 if not allow_cpu_fixture and (wrapper is not None or source is not None):raise RuntimeError("injected wrapper/source is CPU-fixture-only")
 if os.environ.get(GO)!="1" or torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") not in ("","-1"):raise RuntimeError("explicit GO and fresh CPU-only process required")
 if _json(authorization).get("status")!="ROOT_REVIEW_GO" or sha(authorization)!=authorization_sha or _json(authorization).get("authority")!=binding:raise RuntimeError("external authorization drift")
 if not allow_cpu_fixture:_production_contract(binding,authorization)
 if not allow_cpu_fixture:
  auth=Path(authorization);out=Path(binding.get("output","") or "")
  if not auth.is_absolute() or not auth.is_file() or auth==out or any(_inside(Path(item["path"]).parent,auth) for item in binding.get("proofs",{}).values()):raise RuntimeError("external authorization must be absolute and separate from proof/output artifacts")
 if binding.get("calls") not in (32,2048) or binding.get("warmup")!=128 or binding.get("threads") not in (1,2) or binding.get("interop_threads")!=1:raise RuntimeError("frozen timing mode authority drift")
 torch.set_num_threads(binding["threads"])
 try:torch.set_num_interop_threads(1)
 except RuntimeError:
  if torch.get_num_interop_threads()!=1:raise
 _guard(started,binding["threads"]);_revalidate(binding)
 if wrapper is None:
  import src.models.components.spint as spint
  if binding.get("spint_module_sha256")!=SPINT_SHA or sha(Path(spint.__file__))!=SPINT_SHA:raise RuntimeError("actual frozen SPINT module drift")
  for p,h in EXPECTED.items():
   if sha(p)!=h:raise RuntimeError("original wrapper/payload drift")
  spec=importlib.util.spec_from_file_location("_frozen_m2",WRAPPER);wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)
  from falcon_challenge.config import FalconConfig,FalconTask
  config=FalconConfig(task=FalconTask.m2)
 tags,bank,values,source_hash=source or _source(1+binding["warmup"]+binding["calls"],binding.get("trainer_source_train_authority"))
 if source is None:_revalidate(binding,source_hash)
 if values.shape!=(1+binding["warmup"]+binding["calls"],7,96) or values.dtype!=np.float32 or not values.flags.c_contiguous:raise RuntimeError("continuous native FP32 B7 source contract drift")
 authority_path=None;authority_sha=None
 if binding.get("output"):
  output=Path(binding["output"]);authority_path=output.with_name(output.name+".authority_pre.json")
  if output.exists() or authority_path.exists():raise RuntimeError("benchmark output no longer fresh")
  _atomic_json(authority_path,{"authority":binding,"authorization_sha256":authorization_sha});authority_sha=sha(authority_path)
 engines,constructor_ms,reset_ms=_build(wrapper,config,binding,tags,bank)
 state_pre={n:_model_digest(engines[n].model) for n in ("FLAT","ROUTE")}
 names=("ORIGINAL","FLAT","ROUTE");times={n:[] for n in names};first={};ind=np.zeros((7,W,96),dtype=np.float32);oracle={n:0. for n in names}
 for i,value in enumerate(values):
  _guard(started,binding["threads"])
  ind[:,:-1]=ind[:,1:];ind[:,-1]=value;out={};shift=i%3
  for n in names[shift:]+names[:shift]:
   start=time.perf_counter_ns();out[n]=engines[n].predict(value);elapsed=(time.perf_counter_ns()-start)/1e6;_public(n,out[n])
   _guard(started,binding["threads"])
   if i==0:first[n]=elapsed
   elif i>=1+binding["warmup"]:times[n].append(elapsed)
  if i in {0,49,50,len(values)-1}:
   observed=np.ascontiguousarray(engines["ORIGINAL"].observation_buffer.transpose(1,0,2))
   if not np.array_equal(observed,ind):raise RuntimeError("Original public buffer differs from independent W50 raw history")
   want=_spint_oracle(engines["ORIGINAL"]);np.testing.assert_allclose(out["ORIGINAL"],want,atol=1e-5,rtol=1e-5);oracle["ORIGINAL"]=max(oracle["ORIGINAL"],float(np.abs(out["ORIGINAL"]-want).max()));_guard(started,binding["threads"])
   for n in ("FLAT","ROUTE"):
    if not np.array_equal(engines[n].raw.numpy(),ind):raise RuntimeError("candidate W50 history drift")
    with torch.no_grad():want=(engines[n].model.forward_last(engines[n].raw,bank,bank.unit_mask)/5).numpy()
    np.testing.assert_allclose(out[n],want,atol=1e-5,rtol=1e-5);oracle[n]=max(oracle[n],float(np.abs(out[n]-want).max()))
    _guard(started,binding["threads"])
 if any(len(v)!=binding["calls"] for v in times.values()):raise RuntimeError("timing cardinality drift")
 if any(_model_digest(engines[n].model)!=state_pre[n] for n in state_pre):raise RuntimeError("candidate parameter state mutated")
 if source is None:_revalidate(binding,_source(1+binding["warmup"]+binding["calls"],binding.get("trainer_source_train_authority"))[3])
 if _json(authorization).get("status")!="ROOT_REVIEW_GO" or sha(authorization)!=authorization_sha or _json(authorization).get("authority")!=binding:raise RuntimeError("external authorization changed during timing")
 if actual_original:
  import src.models.components.spint as spint
  if sha(Path(spint.__file__))!=binding.get("spint_module_sha256"):raise RuntimeError("actual imported SPINT module changed during timing")
 _guard(started,binding["threads"])
 result={"schema":SCHEMA,"status":"PASS_PUBLIC_STREAM_TIMING_ONLY","authority":binding,"authority_pre_path":str(authority_path) if authority_path else None,"authority_pre_sha256":authority_sha,"batch":7,"source_shape":[int(v) for v in values.shape],"public_output_shape":[7,2],"public_output_dtype":"float32","calls":binding["calls"],"warmup":binding["warmup"],"threads":binding["threads"],"interop_threads":1,"constructor_ms":constructor_ms,"reset_ms":reset_ms,"rotating_three_engine_order":True,"first_public_call_ms":first,"per_call_public_ms":times,"steady_public_call_ms":{n:stats(v) for n,v in times.items()},"whole_process_oracle_max_abs_error":oracle,"candidate_oracle_max_abs_error":{n:oracle[n] for n in ("FLAT","ROUTE")},"source_raw_sha256":source_hash,"elapsed_seconds":time.monotonic()-started,"not_official_latency":True,"quality_selection_or_generalization_claim":False,"parameter_updates":0}
 if authority_path:
  if sha(authority_path)!=authority_sha:raise RuntimeError("own pre-authority receipt drift")
  _atomic_json(binding["output"],result)
  result["receipt_sha256"]=sha(Path(binding["output"]))
 return result

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument("--run-root",type=Path,required=True);p.add_argument("--finalizer-root",type=Path,required=True);p.add_argument("--finalizer-sha",required=True);p.add_argument("--proof-uncached",type=Path,required=True);p.add_argument("--proof-cached",type=Path,required=True);p.add_argument("--proof-uncached-sha",required=True);p.add_argument("--proof-cached-sha",required=True);p.add_argument("--smoke-t1",type=Path);p.add_argument("--smoke-t2",type=Path);p.add_argument("--smoke-t1-sha");p.add_argument("--smoke-t2-sha");p.add_argument("--output",type=Path,required=True);p.add_argument("--authorization",type=Path,required=True);p.add_argument("--authorization-sha",required=True);p.add_argument("--threads",type=int,choices=(1,2),required=True);p.add_argument("--calls",type=int,choices=(32,2048),required=True);a=p.parse_args(argv)
 smokes={"T1":{"path":str(a.smoke_t1),"sha256":a.smoke_t1_sha},"T2":{"path":str(a.smoke_t2),"sha256":a.smoke_t2_sha}} if a.calls==2048 and all((a.smoke_t1,a.smoke_t2,a.smoke_t1_sha,a.smoke_t2_sha)) else None
 binding=collect_bindings(a.run_root,a.finalizer_root,a.finalizer_sha,{"uncached":a.proof_uncached,"cached":a.proof_cached},{"uncached":a.proof_uncached_sha,"cached":a.proof_cached_sha},a.output,calls=a.calls,warmup=128,threads=a.threads,smoke_receipts=smokes);run(binding,authorization=a.authorization,authorization_sha=a.authorization_sha)
if __name__=="__main__":main()
