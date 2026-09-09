"""Score both chronological held-out M1 dates from one frozen source-selected run."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]; sys.path[:0]=[str(ROOT/"src"),str(ROOT.parent/"btransform_unified_v1/src"),str(ROOT)]
import m1_train as train
import m1_data as data
from m1_data import SPLIT_ID,SOURCE_SESSIONS,TARGET_SESSIONS,contract,assert_contract,materialize_sources,materialize_target
from btransform_unified_v2.cross_session_m1_model import ARMS,CrossSessionM1Decoder,Z_NONE

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def atomic_json(p:Path,v:object):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(".tmp");t.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n");t.replace(p)
def pack_metadata(path:Path,label:str)->dict:
 if not path.is_file():raise FileNotFoundError(f"missing {label} carrier pack: {path}")
 with np.load(path,allow_pickle=False) as z:
  if "metadata" not in z.files:raise RuntimeError(f"{label} carrier-pack metadata missing")
  try: meta=json.loads(str(z["metadata"].item()))
  except (ValueError,json.JSONDecodeError) as e:raise RuntimeError(f"invalid {label} carrier-pack metadata") from e
 if not isinstance(meta,dict):raise RuntimeError(f"{label} carrier-pack metadata must be an object")
 return meta
def projection_code_seal(gate:Path)->dict:
 p=gate.parents[2]/"profile_projection_code_seal.json"
 if not p.is_file():raise RuntimeError(f"missing projection code seal: {p}")
 x=json.loads(p.read_text());bindings=x.get("source_code_sha256",{})
 if x.get("schema")!="carrier_last1_v2_profile_projection_code_seal" or not isinstance(bindings,dict) or not bindings:raise RuntimeError("invalid projection code seal")
 for name,digest in bindings.items():
  q=Path(name)
  if not isinstance(digest,str) or not q.is_file() or sha(q)!=digest:raise RuntimeError(f"projection code seal drift: {q}")
 return {"path":str(p.resolve()),"sha256":sha(p),"source_code_sha256":bindings}
def canonical_target_carriers(a,gate:Path)->tuple[dict[str,np.ndarray],dict]|None:
 """Read a canonical projected target pack without using m1_data._carrier_pack.

 The frozen target loader incorrectly treats the source-pack ancestry in a
 canonical target pack as a target roster.  This verifier owns that boundary;
 materialize_target is subsequently called with B so it performs the shared
 query/support/calibration materialization without reaching that loader.
 """
 if a.arm!="D_JOINT" or a.carrier_pack is None:return None
 if a.source_carrier_pack is None:raise RuntimeError("external D target pack requires --source-carrier-pack")
 target_path=a.carrier_pack.resolve();source_path=a.source_carrier_pack.resolve();target_meta=pack_metadata(target_path,"target");source_meta=pack_metadata(source_path,"source")
 common={"schema":"carrier_last1_v2_pack","dataset":"m1","stage":a.temporal_stage}
 if any(target_meta.get(k)!=v for k,v in common.items()) or target_meta.get("surface")!="target":raise RuntimeError("canonical target-pack identity drift")
 if any(source_meta.get(k)!=v for k,v in common.items()) or source_meta.get("surface")!="source":raise RuntimeError("canonical source-pack identity drift")
 if tuple(target_meta.get("sources",()))!=SOURCE_SESSIONS or tuple(target_meta.get("targets",()))!=TARGET_SESSIONS:raise RuntimeError("canonical target-pack roster/ancestry drift")
 if tuple(source_meta.get("sources",()))!=SOURCE_SESSIONS or tuple(source_meta.get("targets",()))!=():raise RuntimeError("source carrier-pack roster drift")
 source_sha=sha(source_path);gate_sha=sha(gate)
 if target_meta.get("source_pack_sha256")!=source_sha or target_meta.get("source_gate_sha256")!=gate_sha:raise RuntimeError("canonical target-pack source binding drift")
 for key in ("profile_id","candidate","source_seal","frozen_fit_sha256"):
  if target_meta.get(key)!=source_meta.get(key):raise RuntimeError(f"canonical source/target pack metadata drift: {key}")
 frozen=Path(source_meta.get("frozen_fit",""));frozen=frozen if frozen.is_file() or frozen.is_absolute() else source_path.parent/frozen;frozen_sha=source_meta.get("frozen_fit_sha256")
 if not frozen.is_file() or not isinstance(frozen_sha,str) or sha(frozen)!=frozen_sha:raise RuntimeError("canonical source-pack frozen-fit binding drift")
 carriers={}
 with np.load(target_path,allow_pickle=False) as z:
  for target in TARGET_SESSIONS:
   key=f"carrier/{target}"
   if key not in z.files:raise RuntimeError(f"canonical target-pack missing {key}")
   x=np.asarray(z[key])
   if x.dtype!=np.float32 or x.shape!=(64,4) or not np.isfinite(x).all():raise RuntimeError(f"invalid canonical target carrier {key}")
   carriers[target]=np.ascontiguousarray(x)
 return carriers,{"target_pack":{"path":str(target_path),"sha256":sha(target_path),"metadata":target_meta},"source_pack":{"path":str(source_path),"sha256":source_sha,"metadata":source_meta},"source_gate":{"path":str(gate),"sha256":gate_sha},"projection_code_seal":projection_code_seal(gate)}
def write_npz(p:Path,row:dict)->str:
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise FileExistsError(f"refuse to overwrite completed target artifact: {p}")
 t=p.with_suffix(".tmp.npz");np.savez_compressed(t,**row);t.replace(p);return sha(p)
def same_tensor_tree(a,b)->bool:
 if isinstance(a,dict) or isinstance(b,dict):
  return isinstance(a,dict) and isinstance(b,dict) and set(a)==set(b) and all(same_tensor_tree(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)) or isinstance(b,(list,tuple)):
  return type(a) is type(b) and len(a)==len(b) and all(same_tensor_tree(x,y) for x,y in zip(a,b))
 if isinstance(a,torch.Tensor) or isinstance(b,torch.Tensor):
  return isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 return type(a) is type(b) and a==b
def require_train(a):
 rec=json.loads((a.dest/"train_receipt.json").read_text());pf=json.loads((a.dest/"preflight.json").read_text());expected=train.metadata(a)
 if rec.get("status")!="COMPLETED" or pf.get("status")!="PASSED":raise RuntimeError("completed chronological train/preflight required")
 for value in (rec,pf):
  for k in ("split_id","sources","targets","arm","seed","data_contract"):
   if value.get(k)!=expected[k]:raise RuntimeError(f"receipt split mismatch: {k}")
 for name,digest in rec.get("source_code_sha256",{}).items():
  if not Path(name).is_file() or sha(Path(name))!=digest:raise RuntimeError(f"training source-code binding drift: {name}")
 cp=a.dest/"resume_latest.pt";pick=a.dest/"selected_ema.pt";actual={p.name:sha(p) for p in (cp,pick)}
 if rec.get("checkpoint_sha256")!=actual:raise RuntimeError("checkpoint checksum differs from chronological receipt")
 final=torch.load(cp,map_location=a.device,weights_only=False);selected=torch.load(pick,map_location=a.device,weights_only=False)
 for q in (final,selected):
  for k in ("split_id","sources","targets","arm","seed","data_contract"):
   if q.get("meta",{}).get(k)!=expected[k]:raise RuntimeError(f"checkpoint split mismatch: {k}")
 if int(final.get("epoch",0))!=train.EPOCHS:raise RuntimeError("resume checkpoint is not fixed epoch 24")
 if final.get("best")!=rec.get("selected") or int(final.get("step",-1))!=int(rec.get("steps",-2)) or int(selected.get("epoch",-1))!=int(rec.get("selected",{}).get("epoch",-2)):raise RuntimeError("selected/fixed24 checkpoint provenance mismatch")
 curve=[];curve_hash=hashlib.sha256()
 for epoch in range(1,train.EPOCHS+1):
  jp=a.dest/f"ema_epoch_{epoch:03d}.json";pp=a.dest/f"ema_epoch_{epoch:03d}.pt"
  if not jp.is_file() or not pp.is_file():raise RuntimeError(f"missing source EMA epoch artifact {epoch}")
  curve_hash.update(sha(jp).encode());curve_hash.update(sha(pp).encode());row=json.loads(jp.read_text());state=torch.load(pp,map_location=a.device,weights_only=False)
  values=row.get("source_val",{}).get("per_session",{})
  if row.get("epoch")!=epoch or state.get("epoch")!=epoch or row.get("step")!=state.get("step") or state.get("meta")!=expected or set(values)!=set(SOURCE_SESSIONS) or not np.isfinite(list(values.values())).all() or abs(float(np.mean(list(values.values())))-float(row.get("source_val",{}).get("equal_session_mean",np.nan)))>1e-12:raise RuntimeError(f"invalid source EMA curve epoch {epoch}")
  curve.append(row)
 if any(curve[i]["step"]<=curve[i-1]["step"] for i in range(1,len(curve))) or curve[-1]["step"]!=rec.get("steps"):raise RuntimeError("source EMA step curve mismatch")
 best=curve[0]
 for row in curve[1:]:
  if row["source_val"]["equal_session_mean"]>best["source_val"]["equal_session_mean"]:best=row
 if best!=rec.get("selected") or best!=final.get("best"):raise RuntimeError("receipt selected epoch is not earliest source maximum")
 chosen_epoch=int(best["epoch"]);chosen_state=torch.load(a.dest/f"ema_epoch_{chosen_epoch:03d}.pt",map_location=a.device,weights_only=False)
 if chosen_state.get("meta")!=expected or chosen_state.get("step")!=best.get("step") or not same_tensor_tree(selected.get("ema",{}),chosen_state.get("ema",{})):raise RuntimeError("selected EMA tensor state differs from selected epoch artifact")
 rec["source_curve_sha256"]=curve_hash.hexdigest()
 return rec,final,selected
def rows(model,ema,ds,bank,device,target):
 raw={n:p.detach().clone() for n,p in model.named_parameters()};model.eval();train._ema_apply(model,ema);ys=[];ps=[];starts=[];po=[];qo=[];prefix=int(np.asarray(ds.trial_start_indices[target])[0])
 with torch.inference_mode():
  for x,y,s,v,st in train.loader(ds,42,shuffle=False):
   if set(s)!={target}:raise RuntimeError("target score batch drift")
   p=model(x.to(device),bank,input_valid_mask=v.to(device)).cpu().numpy();z=st.numpy().astype(np.int64);ys.append(y[:,-1].numpy());ps.append(p);starts.append(z);po.append(z+ds.window_size-1);qo.append(z+ds.window_size-1-prefix)
 with torch.no_grad():
  for n,p in model.named_parameters():p.copy_(raw[n])
 return {"target":np.concatenate(ys),"prediction":np.concatenate(ps),"window_start_padded":np.concatenate(starts),"output_index_padded":np.concatenate(po),"output_index_query_relative":np.concatenate(qo),"prefix_bins":np.asarray([prefix],dtype=np.int64)}
def metric(row,target):
 from btransform_unified_v1.r2 import variance_weighted_r2
 r=float(variance_weighted_r2(row["target"],row["prediction"]));return {"per_session":{target:r},"equal_session_mean":r}
def main():
 p=argparse.ArgumentParser();p.add_argument("--dest",type=Path,required=True);p.add_argument("--arm",choices=ARMS,required=True);p.add_argument("--seed",type=int,default=42);p.add_argument("--device",default="cuda:0");p.add_argument("--carrier-pack",type=Path);p.add_argument("--source-carrier-pack",type=Path);p.add_argument("--source-gate",type=Path,required=True);p.add_argument("--temporal-stage",choices=("inner","outer"),default="outer");a=p.parse_args();
 if a.carrier_pack: __import__("os").environ["TARGET_CARRIER_PACK"]=str(a.carrier_pack.resolve())
 if a.source_carrier_pack: __import__("os").environ["SOURCE_CARRIER_PACK"]=str(a.source_carrier_pack.resolve())
 __import__("os").environ["CARRIER_STAGE"]=a.temporal_stage
 if a.seed!=42:raise ValueError("chronological protocol seed is 42")
 a.dest=a.dest.resolve();
 gate=a.source_gate.resolve()
 if not gate.is_file():raise RuntimeError("explicit paired source gate missing")
 g=json.loads(gate.read_text())
 if g.get("schema")!="carrier_last1_v2_paired_audit" or g.get("status")!="PASSED" or g.get("phase")!="source" or g.get("stage")!=a.temporal_stage or g.get("dataset")!="m1":raise RuntimeError("paired source gate identity/status mismatch")
 if a.temporal_stage=="outer" and not (a.dest/"source_seal.json").is_file():raise RuntimeError("outer target scoring requires own source seal")
 canonical=canonical_target_carriers(a,gate)
 rec,final,selected=require_train(a);cache=a.dest.parent.parent/"source_prepare_cache";src=materialize_sources(cache=cache)
 if tuple(src["sources"])!=SOURCE_SESSIONS or rec.get("actual_source_arrays")!=train.source_evidence(src):raise RuntimeError("chronological source evidence differs from receipt")
 for target in TARGET_SESSIONS:
  out=a.dest/f"score_{target}"
  if (out/"score_receipt.json").exists():raise FileExistsError(f"refuse to overwrite completed target score: {out}")
  # For an externally projected D target pack, the frozen B path materializes
  # the same target query/support/calibration data without entering its broken
  # target-pack roster loader.  Only the carrier bank is then replaced.
  tar=materialize_target(target,src,"B_ACTIVITY_ONLY" if canonical is not None else a.arm)
  if canonical is not None:tar["target_bank"]=data._bank(target,canonical[0][target])
  model=CrossSessionM1Decoder(a.arm,seed=a.seed).to(a.device);train._configure(model);banks={**src["banks"],target:tar["target_bank"]};calib=src["calib"] if a.arm==Z_NONE else {**src["calib"],target:tar["target_calib"]};train._install(model,a.arm,banks,calib)
  from btransform_unified_v1.ema import DecoderEMA
  ema=DecoderEMA(model,train.EMA_DECAY);model.load_state_dict(final["model"],strict=True);ema.load_state_dict(final["ema"]);fixed=rows(model,ema,tar["target"],tar["target_bank"],torch.device(a.device),target);fixed_h=write_npz(a.dest/f"score_{target}"/"target_epoch24_predictions.npz",fixed)
  ema.load_state_dict(selected["ema"]);chosen=rows(model,ema,tar["target"],tar["target_bank"],torch.device(a.device),target);chosen_h=write_npz(a.dest/f"score_{target}"/"target_selected_predictions.npz",chosen)
  score_receipt={**train.metadata(a),"status":"COMPLETED","target":target,"source_selected":rec["selected"],"source_curve_sha256":rec["source_curve_sha256"],"target_selected_ema":metric(chosen,target),"target_epoch24_ema":metric(fixed,target),"target_labels_used_for_selection":False,"target_optimizer_steps":0,"checkpoint_sha256":rec["checkpoint_sha256"],"canonical_target_pack_binding":None if canonical is None else canonical[1],"source_code_sha256":{str(Path(__file__)):sha(Path(__file__)),str(Path(__file__).with_name("m1_train.py")):sha(Path(__file__).with_name("m1_train.py")),str(Path(__file__).with_name("m1_data.py")):sha(Path(__file__).with_name("m1_data.py")),str(ROOT/"src/btransform_unified_v2/cross_session_m1_model.py"):sha(ROOT/"src/btransform_unified_v2/cross_session_m1_model.py"),str(ROOT.parent/"streaming_calibration_exp/src/data/falcon_datamodule.py"):sha(ROOT.parent/"streaming_calibration_exp/src/data/falcon_datamodule.py"),str(ROOT.parent/"tfpd_exploration/src/m1_emg_syn3_fcm_v1/data.py"):sha(ROOT.parent/"tfpd_exploration/src/m1_emg_syn3_fcm_v1/data.py"),str(ROOT.parent/"tfpd_exploration/src/m1_emg_syn3_fcm_v1/syn3.py"):sha(ROOT.parent/"tfpd_exploration/src/m1_emg_syn3_fcm_v1/syn3.py")},"target_audit_arrays":{"selected":{"file":"target_selected_predictions.npz","sha256":chosen_h},"epoch24":{"file":"target_epoch24_predictions.npz","sha256":fixed_h}}}
  atomic_json(a.dest/f"score_{target}"/"score_receipt.json",score_receipt)
if __name__=="__main__":main()
