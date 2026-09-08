#!/usr/bin/env python3
"""Read-only validation of a completed multi-seed joint-D ext6 EMA curve."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, sys
from pathlib import Path
from typing import Any, Mapping

ROOT=Path(__file__).resolve().parents[2]
SIX=("ses-2020-10-30-Run1","ses-2020-10-30-Run2","ses-2020-11-18-Run1","ses-2020-11-19-Run1","ses-2020-11-24-Run1","ses-2020-11-24-Run2")
EPOCHS=tuple(range(1,25)); CELL="M2-RIFT-R50-D4-JOINT-FILM-M33-V1"
PICK_SCHEMA="m2_rift_ext6_epoch_pick_multiseed_v1"
QUERY_FILES=("mapping.json","e0_u.pt","T.npy","eligible_starts.npy","X_store.npy","target_store.npy")
ARTIFACTS=("manifest.json","score_progress.json","score_receipt.json","selected_ema.pt","selected_ema_receipt.json")
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def load(p:Path)->dict[str,Any]:
 x=json.loads(p.read_text());
 if not isinstance(x,dict):raise RuntimeError(f"JSON object required: {p}")
 return x
def canonical(x:Mapping[str,Any])->str:return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def finite(x:Any)->float:
 x=float(x)
 if not math.isfinite(x):raise ValueError("non-finite value")
 return x

def main()->None:
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=ROOT);p.add_argument("--seed",type=int,choices=(43,44),required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 root,out=a.root.resolve(),a.output.resolve()
 for entry in (root,root/"src",root.parent):
  if str(entry) not in sys.path:sys.path.insert(0,str(entry))
 if out.exists():raise FileExistsError(f"refusing existing output {out}")
 pick=root/f"results/rift_v1/m2_r50_joint_d_s{a.seed}_ext6_pick_v1";run=root/f"results/rift_v1/m2_r50_joint_d_s{a.seed}_formal_v1";paths={n:pick/n for n in ARTIFACTS};checks:dict[str,dict[str,bool]]={};failures:list[str]=[]
 def req(name:str,ok:bool)->bool:
  checks[name]={"passed":bool(ok)}
  if not ok:failures.append(name)
  return bool(ok)
 def guard(name:str,fn:Any)->Any:
  try:
   x=fn();req(name,bool(x));return x
  except Exception as exc:
   req(name,False);failures.append(f"{name}: {exc}");return None
 try:
  manifest=load(paths["manifest.json"]);progress=load(paths["score_progress.json"]);receipt=load(paths["score_receipt.json"]);selected_receipt=load(paths["selected_ema_receipt.json"]);meta=load(run/"run_meta.json");train=load(run/"train_receipt.json");formal_score=load(run/"score_receipt.json");formal_scan=load(run/"ext4_epoch_scan.json")
 except Exception as exc:
  report={"schema":f"m2_joint_d_s{a.seed}_ext6_full_curve_validation_v1","status":"FAILED","failures":[str(exc)],"checks":{"inputs":{"passed":False}}}
 else:
  artifact_sha256={n:guard(f"artifact:{n}",lambda q=q:q.is_file() and sha(q)) for n,q in paths.items()};artifact_sha256={n:x for n,x in artifact_sha256.items() if isinstance(x,str)}
  em={"schema":"m2_rift_joint_train_v2","status":"FORMAL","cell":CELL,"arm":"D_JOINT","seed":a.seed,"sampler_seed":42,"epochs":24,"context_bins":50,"depth":4,"attention_backend":"local","official_test_used":False,"source_train_only_for_gradients":True}
  req("formal:meta",all(meta.get(k)==v for k,v in em.items()) and meta.get("optimizer",{}).get("total_updates")==75960 and meta.get("optimizer",{}).get("warmup_updates")==3165 and isinstance(meta.get("source_hashes"),dict) and bool(meta["source_hashes"]) and isinstance(meta.get("cache_hashes"),dict))
  req("formal:train",all(train.get(k)==v for k,v in {"schema":"m2_rift_joint_train_receipt_v2","status":"COMPLETED","cell":CELL,"arm":"D_JOINT","seed":a.seed,"sampler_seed":42,"epochs":24,"global_step":75960,"source_hashes":meta.get("source_hashes"),"cache_hashes":meta.get("cache_hashes")}.items()))
  req("formal:ext4_receipt_equals_scan",formal_score==formal_scan);req("formal:ext4_identity",all(formal_score.get(k)==v for k,v in {"schema":"m2_rift_joint_ext4_epoch_scan_v1","status":"COMPLETED","cell":CELL,"arm":"D_JOINT","seed":a.seed,"sampler_seed":42,"source_hashes":meta.get("source_hashes"),"cache_hashes":meta.get("cache_hashes"),"official_test_used":False}.items()))
  for src,digest in meta.get("source_hashes",{}).items():guard(f"formal:source:{src}",lambda src=src,digest=digest:Path(src).is_file() and sha(Path(src))==digest)
  for surface,sessions0 in meta.get("cache_hashes",{}).items():
   if not isinstance(sessions0,Mapping):req(f"formal:cache:{surface}",False);continue
   for session,files in sessions0.items():
    if not isinstance(files,Mapping):req(f"formal:cache:{surface}/{session}",False);continue
    for name,digest in files.items():guard(f"formal:cache:{surface}/{session}/{name}",lambda surface=surface,session=session,name=name,digest=digest:(root.parent/"tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache"/surface/session/name).is_file() and sha(root.parent/"tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache"/surface/session/name)==digest)
  expected={"schema":PICK_SCHEMA+"_manifest","run":str(run),"run_meta_sha256":sha(run/"run_meta.json"),"train_receipt_sha256":sha(run/"train_receipt.json"),"cell":CELL,"run_schema":"m2_rift_joint_train_v2","seed":a.seed,"view":"EMA","epochs":list(EPOCHS),"official_test_used":False,"evalai_opened":False}
  req("manifest:identity",all(manifest.get(k)==v for k,v in expected.items()));sources=manifest.get("source_sha256",{})
  picker=root/"scripts/rift_v1/m2_ext6_epoch_pick_multiseed.py";req("manifest:picker_source",isinstance(sources,Mapping) and sources.get(str(picker))==sha(picker));req("manifest:no_688_source",isinstance(sources,Mapping) and all("688" not in str(x) for x in sources))
  for src,digest in sources.items() if isinstance(sources,Mapping) else ():guard(f"manifest:source:{src}",lambda src=src,digest=digest:Path(src).is_file() and sha(Path(src))==digest)
  cps=manifest.get("checkpoint_bytes",{});req("manifest:24_checkpoint_rows",isinstance(cps,Mapping) and set(cps)=={str(e) for e in EPOCHS})
  for e in EPOCHS:
   item=cps.get(str(e),{}) if isinstance(cps,Mapping) else {};cp=run/f"epoch_{e:03d}.pt";guard(f"manifest:checkpoint:{e}",lambda item=item,cp=cp:isinstance(item,Mapping) and item.get("path")==str(cp) and item.get("sha256")==sha(cp))
   def checkpoint_identity(cp=cp,e=e)->bool:
    import torch
    state=torch.load(cp,map_location="cpu",weights_only=False);ema=state.get("ema",{}) if isinstance(state,Mapping) else {}
    def valid_tensors(values:Any)->bool:return isinstance(values,Mapping) and bool(values) and all(torch.is_tensor(v) and v.numel()>0 and (not(v.is_floating_point() or v.is_complex()) or bool(torch.isfinite(v).all())) for v in values.values())
    return isinstance(state,Mapping) and all(state.get(k)==v for k,v in {"schema":"m2_rift_joint_checkpoint_v2","cell":CELL,"arm":"D_JOINT","seed":a.seed,"epoch":e,"global_step":e*3165,"smoke":False,"epochs":24,"context_bins":50,"depth":4,"attention_backend":"local","source_hashes":meta["source_hashes"],"cache_hashes":meta["cache_hashes"]}.items()) and valid_tensors(state.get("model")) and isinstance(ema,Mapping) and ema.get("decay")==0.9995 and ema.get("n_updates")==e*3165 and valid_tensors(ema.get("shadow")) and set(ema["shadow"]).issubset(state["model"])
   guard(f"formal:checkpoint_identity:{e}",checkpoint_identity)
  assets=manifest.get("query_assets",{});cache=Path(str(manifest.get("query_cache","")));sessions=assets.get("sessions",{}) if isinstance(assets,Mapping) else {}
  guard("query:receipt",lambda:cache.is_dir() and (cache/"official_heldout_query_banks.json").is_file() and assets.get("receipt_sha256")==sha(cache/"official_heldout_query_banks.json"));req("query:six_sessions",isinstance(sessions,Mapping) and set(sessions)==set(SIX))
  for s in SIX:
   item=sessions.get(s,{}) if isinstance(sessions,Mapping) else {};req(f"query:{s}:identity",isinstance(item,Mapping) and item.get("query_start_trial")==0 and isinstance(item.get("window_count"),int) and item["window_count"]>0 and set(item.get("files",{}))==set(QUERY_FILES))
   for name in QUERY_FILES:
    digest=item.get("files",{}).get(name) if isinstance(item,Mapping) else None;guard(f"query:{s}:{name}",lambda s=s,name=name,digest=digest:(cache/s/name).is_file() and sha(cache/s/name)==digest)
  req("query:total_windows",isinstance(sessions,Mapping) and sum(int(sessions[s].get("window_count",-1)) for s in SIX if isinstance(sessions.get(s),Mapping))==15403)
  m33root=Path(str(manifest.get("joint_m33_root","")));m33=manifest.get("joint_m33_sha256",{});req("m33:six_sessions",isinstance(m33,Mapping) and set(m33)==set(SIX))
  for s in SIX:
   def m33_ok(s=s)->bool:
    import numpy as np
    path=m33root/s/"calib_activity.npy"
    return path.is_file() and m33.get(s)==sha(path) and np.load(path,mmap_mode="r").shape==(33,100,96)
   guard(f"m33:{s}",m33_ok)
  def picker_manifest_ok()->bool:
   picker_path=root/"scripts/rift_v1/m2_ext6_epoch_pick_multiseed.py";spec=importlib.util.spec_from_file_location("frozen_m2_ext6_epoch_pick_multiseed",picker_path)
   if spec is None or spec.loader is None:raise RuntimeError("cannot load frozen multiseed picker")
   module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.verify_manifest(manifest,run,cache,m33root);return True
  guard("manifest:frozen_picker_verify",picker_manifest_ok)
  oracle=manifest.get("sealed_bt_oracle",{});guard("oracle:binding",lambda:isinstance(oracle,Mapping) and oracle.get("receipt_sha256")==sha(Path(oracle["receipt"])) and oracle.get("oracle_source_sha256")==sha(Path(oracle["oracle_source"])) and oracle.get("payload_sha256")==sha(Path(oracle["payload"])))
  req("progress:identity",progress.get("schema")==PICK_SCHEMA+"_progress" and progress.get("manifest_sha256")==canonical(manifest) and isinstance(progress.get("completed"),Mapping));rows=progress.get("completed",{});req("curve:24_epochs",isinstance(rows,Mapping) and set(rows)=={str(e) for e in EPOCHS})
  for e in EPOCHS:
   row=rows.get(str(e),{}) if isinstance(rows,Mapping) else {};per=row.get("per_session",{}) if isinstance(row,Mapping) else {}
   def row_ok(row=row,per=per,e=e)->bool:
    vals=[finite(per[s]["r2"]) for s in SIX]
    return row.get("view")=="EMA" and row.get("partial") is False and set(per)==set(SIX) and row.get("n_windows")==15403 and all(per[s].get("window_count")==sessions[s].get("window_count") and isinstance(per[s].get("prediction_sha256"),str) and len(per[s]["prediction_sha256"])==64 for s in SIX) and abs(finite(row["equal_session_mean"])-sum(vals)/6)<=1e-12 and math.isfinite(finite(row["pooled_r2"])) and row.get("checkpoint_sha256")==cps[str(e)]["sha256"]
   guard(f"curve:epoch:{e}",row_ok)
  best=guard("selection:computable",lambda:min(EPOCHS,key=lambda e:(-finite(rows[str(e)]["equal_session_mean"]),e)))
  package_alias_statistics:dict[str,Any]={}
  if isinstance(best,int):
   sel={"rule":"earliest maximum finite unweighted equal_session_mean","epoch":best,"equal_session_mean":finite(rows[str(best)]["equal_session_mean"])}
   req("receipt:identity",receipt.get("schema")==PICK_SCHEMA+"_selection" and receipt.get("status")=="COMPLETED" and receipt.get("manifest_sha256")==canonical(manifest) and receipt.get("view")=="EMA" and receipt.get("selection")==sel and receipt.get("selected")==rows[str(best)] and receipt.get("ema_by_epoch")==rows and receipt.get("evalai_opened") is False and receipt.get("official_test_used") is False);req("receipt:duplicate",receipt==selected_receipt);state=receipt.get("selected_ema_state",{});req("package:receipt",isinstance(state,Mapping) and state.get("path")==str(paths["selected_ema.pt"]) and state.get("sha256")==sha(paths["selected_ema.pt"]) and state.get("raw_state_serialized") is False)
   def package_ok()->bool:
    nonlocal package_alias_statistics
    import torch
    from btransform_unified_v2.joint_m2_model import JointM2RiftDecoder
    cp=torch.load(run/f"epoch_{best:03d}.pt",map_location="cpu",weights_only=False);pkg=torch.load(paths["selected_ema.pt"],map_location="cpu",weights_only=True);raw,shadow=cp.get("model"),cp.get("ema",{}).get("shadow")
    if not(isinstance(raw,Mapping) and isinstance(shadow,Mapping) and isinstance(pkg,Mapping)):return False
    model=JointM2RiftDecoder("D_JOINT",seed=a.seed).cpu();model.load_state_dict(raw,strict=True)
    named=dict(model.named_parameters())
    if set(named)!=set(shadow):return False
    with torch.no_grad():
     for name,param in named.items():param.copy_(shadow[name].to(param.device,param.dtype))
    live_state=model.state_dict();named_ptrs={value.data_ptr() for value in named.values()};alias_keys=[key for key,value in live_state.items() if torch.is_tensor(value) and value.data_ptr() in named_ptrs and key not in named]
    package_alias_statistics={"full_state_key_count":len(live_state),"named_parameter_count":len(named),"ema_shadow_key_count":len(shadow),"ema_parameter_alias_state_key_count":len(alias_keys),"ema_parameter_alias_state_keys":alias_keys}
    expected={key:value.detach().clone() for key,value in live_state.items()}
    if not(set(pkg)==set(expected)==set(raw) and all(torch.is_tensor(pkg[k]) and torch.equal(pkg[k],expected[k]) for k in expected) and all(not(torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all())) for value in pkg.values())):return False
    loaded=model.load_state_dict(pkg,strict=True)
    return not loaded.missing_keys and not loaded.unexpected_keys and all(torch.equal(value,pkg[key]) for key,value in model.state_dict().items())
   guard("package:strict_model_ema_bytes",package_ok)
  else:
   for n in ("receipt:identity","receipt:duplicate","package:receipt","package:strict_model_ema_bytes"):req(n,False)
  report={"schema":f"m2_joint_d_s{a.seed}_ext6_full_curve_validation_v1","status":"PASSED" if not failures else "FAILED","failures":failures,"scope":{"read_only":True,"evalai_opened":False,"official_test_opened":False},"artifact_sha256":artifact_sha256,"auditor_source_sha256":sha(Path(__file__)),"picker_source_sha256":sha(picker),"manifest_file_sha256":sha(paths["manifest.json"]),"formal_three_receipt_sha256":{"run_meta.json":sha(run/"run_meta.json"),"train_receipt.json":sha(run/"train_receipt.json"),"score_receipt.json":sha(run/"score_receipt.json")},"package_alias_statistics":package_alias_statistics,"checks":checks,"selected_epoch":best if isinstance(best,int) else None}
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps({"status":report["status"],"output":str(out)}))
if __name__=="__main__":main()
