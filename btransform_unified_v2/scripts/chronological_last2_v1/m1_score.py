"""Score both chronological held-out M1 dates from one frozen source-selected run."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]; sys.path[:0]=[str(ROOT/"src"),str(ROOT.parent/"btransform_unified_v1/src"),str(ROOT)]
import m1_train as train
from m1_data import SPLIT_ID,SOURCE_SESSIONS,TARGET_SESSIONS,contract,assert_contract,materialize_sources,materialize_target
from btransform_unified_v2.cross_session_m1_model import ARMS,CrossSessionM1Decoder,Z_NONE

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def atomic_json(p:Path,v:object):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(".tmp");t.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n");t.replace(p)
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
 p=argparse.ArgumentParser();p.add_argument("--dest",type=Path,required=True);p.add_argument("--arm",choices=ARMS,required=True);p.add_argument("--seed",type=int,default=42);p.add_argument("--device",default="cuda:0");a=p.parse_args()
 if a.seed!=42:raise ValueError("chronological protocol seed is 42")
 a.dest=a.dest.resolve();rec,final,selected=require_train(a);cache=a.dest.parent.parent/"source_prepare_cache";src=materialize_sources(cache=cache)
 if tuple(src["sources"])!=SOURCE_SESSIONS or rec.get("actual_source_arrays")!=train.source_evidence(src):raise RuntimeError("chronological source evidence differs from receipt")
 for target in TARGET_SESSIONS:
  out=a.dest/f"score_{target}"
  if (out/"score_receipt.json").exists():raise FileExistsError(f"refuse to overwrite completed target score: {out}")
  tar=materialize_target(target,src,a.arm);model=CrossSessionM1Decoder(a.arm,seed=a.seed).to(a.device);train._configure(model);banks={**src["banks"],target:tar["target_bank"]};calib=src["calib"] if a.arm==Z_NONE else {**src["calib"],target:tar["target_calib"]};train._install(model,a.arm,banks,calib)
  from btransform_unified_v1.ema import DecoderEMA
  ema=DecoderEMA(model,train.EMA_DECAY);model.load_state_dict(final["model"],strict=True);ema.load_state_dict(final["ema"]);fixed=rows(model,ema,tar["target"],tar["target_bank"],torch.device(a.device),target);fixed_h=write_npz(a.dest/f"score_{target}"/"target_epoch24_predictions.npz",fixed)
  ema.load_state_dict(selected["ema"]);chosen=rows(model,ema,tar["target"],tar["target_bank"],torch.device(a.device),target);chosen_h=write_npz(a.dest/f"score_{target}"/"target_selected_predictions.npz",chosen)
  atomic_json(a.dest/f"score_{target}"/"score_receipt.json",{**train.metadata(a),"status":"COMPLETED","target":target,"source_selected":rec["selected"],"source_curve_sha256":rec["source_curve_sha256"],"target_selected_ema":metric(chosen,target),"target_epoch24_ema":metric(fixed,target),"target_labels_used_for_selection":False,"target_optimizer_steps":0,"checkpoint_sha256":rec["checkpoint_sha256"],"source_code_sha256":{str(Path(__file__)):sha(Path(__file__)),str(Path(__file__).with_name("m1_train.py")):sha(Path(__file__).with_name("m1_train.py")),str(Path(__file__).with_name("m1_data.py")):sha(Path(__file__).with_name("m1_data.py")),str(ROOT/"src/btransform_unified_v2/cross_session_m1_model.py"):sha(ROOT/"src/btransform_unified_v2/cross_session_m1_model.py"),str(ROOT.parent/"streaming_calibration_exp/src/data/falcon_datamodule.py"):sha(ROOT.parent/"streaming_calibration_exp/src/data/falcon_datamodule.py"),str(ROOT.parent/"tfpd_exploration/src/m1_emg_syn3_fcm_v1/data.py"):sha(ROOT.parent/"tfpd_exploration/src/m1_emg_syn3_fcm_v1/data.py"),str(ROOT.parent/"tfpd_exploration/src/m1_emg_syn3_fcm_v1/syn3.py"):sha(ROOT.parent/"tfpd_exploration/src/m1_emg_syn3_fcm_v1/syn3.py")},"target_audit_arrays":{"selected":{"file":"target_selected_predictions.npz","sha256":chosen_h},"epoch24":{"file":"target_epoch24_predictions.npz","sha256":fixed_h}}})
if __name__=="__main__":main()
