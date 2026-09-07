"""Frozen post-train scorer/exporter for formal M1 source-development checkpoints.

No optimizer is built. Epoch selection is EMA equal-session R², earliest tie;
all choices are made only on the known-source development split, before any
outer-24 access. Exports include every scored target/prediction and exact
session/window identifiers for independent CPU recomputation.
"""
from __future__ import annotations
import copy, glob, hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from . import bank,plan
from .data import build_source_only_datamodule
from .source_dev import _all_dev_batches,_banks,_clone,_metrics,_model,_name,_shape,_split,_sha as _window_sha

def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _eval(model,dev_ds,batches,banks,device,train_y):
 model.eval(); ps=[]; ys=[]; sessions=[]; starts=[]
 with torch.inference_mode():
  for ids,item in zip(batches,DataLoader(dev_ds,batch_sampler=batches),strict=True):
   x,y,_c,s,_t=item[:5]; name=_name(s[0])
   if any(_name(v)!=name for v in s): raise RuntimeError("non-pure dev batch")
   x,y=_shape(x,y); ps.append(model.forward_last(x.to(device),banks[name]).cpu().numpy()); ys.append(y.numpy()); sessions.extend([name]*len(y)); starts.extend([dev_ds.base.window_indices[i][1] for i in ids])
 p=np.concatenate(ps); y=np.concatenate(ys)
 if len(p)!=len(dev_ds): raise RuntimeError("incomplete scoring")
 per={}
 session_arr=np.asarray(sessions)
 for name in plan.SOURCE_SESSIONS: per[name]=_metrics(p[session_arr==name],y[session_arr==name],train_y)
 return p,y,session_arr,np.asarray(starts,dtype=np.int64),per,float(np.mean([per[n]["r2"]["model"] for n in plan.SOURCE_SESSIONS]))
def _ema_into(model,shadow):
 expected={name for name,param in model.named_parameters() if param.requires_grad}
 if set(shadow)!=expected: raise RuntimeError(f"EMA shadow keys incomplete/unexpected: missing={sorted(expected-set(shadow))[:5]} extra={sorted(set(shadow)-expected)[:5]}")
 state=model.state_dict()
 for name,param in model.named_parameters():
  if name in shadow: state[name]=shadow[name].to(dtype=param.dtype)
 model.load_state_dict(state,strict=True)
def run(tag,kind,device="cuda:0"):
 loaded=bank.load(); dm=build_source_only_datamodule(loaded); tr,dv,_rows=_split(dm); train_ds=_clone(dm.train_dataset,tr,loaded); dev_ds=_clone(dm.train_dataset,dv,loaded); batches=_all_dev_batches(dev_ds)
 train_y=[]
 for item in DataLoader(train_ds,batch_size=32,shuffle=False):
  _,y,*_=item
  if y.dim()==3:y=y[:,-1,:]
  train_y.append(y.numpy())
 train_y=np.concatenate(train_y); dev=torch.device(device); banks=_banks(dev)
 paths=sorted(Path(plan.RESULT_ROOT).glob(f"source_dev_{tag}_{kind}_epoch_*.pt"))
 epochs=[int(p.stem.rsplit("_",1)[1]) for p in paths]
 if epochs != list(range(1,13)): raise RuntimeError(f"formal postscore requires exact epochs 1..12, got {epochs}")
 rows=[]; exports=[]
 for path in paths:
  ckpt=torch.load(path,map_location="cpu",weights_only=False)
  if ckpt["split_train_sha256"] != _window_sha(tr) or ckpt["split_dev_sha256"] != _window_sha(dv): raise RuntimeError("checkpoint split hash drift")
  if ckpt.get("recipe",{}).get("ema") != .9995: raise RuntimeError("checkpoint authority/recipe drift")
  model=_model(kind).to(dev); model.load_state_dict(ckpt["model"],strict=True)
  for variant in ("raw","ema"):
   candidate=copy.deepcopy(model) if variant=="ema" else model
   if variant=="ema": _ema_into(candidate,ckpt["ema"]["shadow"])
   p,y,sessions,starts,per,equal=_eval(candidate,dev_ds,batches,banks,dev,train_y)
   out=Path(plan.RESULT_ROOT)/f"formal_{tag}_{kind}_epoch_{int(ckpt['epoch']):03d}_{variant}_predictions.npz"
   if out.exists(): raise FileExistsError(out)
   np.savez_compressed(out,prediction=p,target=y,session=sessions,window_start=starts,window_end_exclusive=starts+plan.WINDOW,bin_timestep=starts+plan.WINDOW-1,train_target_mean=train_y.mean(0),split_train_sha256=np.asarray([ckpt["split_train_sha256"]]),split_dev_sha256=np.asarray([ckpt["split_dev_sha256"]]),checkpoint_sha256=np.asarray([_sha(path)]),carrier_npz_sha256=np.asarray([loaded["receipt"]["digests"]["npz"]]))
   rows.append({"epoch":int(ckpt["epoch"]),"variant":variant,"equal_session_mean_r2":equal,"per_session":per,"prediction_export":str(out),"prediction_export_sha256":_sha(out),"checkpoint":str(path),"checkpoint_sha256":_sha(path)})
   exports.append(out)
 ema_rows=[row for row in rows if row["variant"]=="ema"]; selected=min(ema_rows,key=lambda r:(-r["equal_session_mean_r2"],r["epoch"]))
 endpoint=max(ema_rows,key=lambda r:r["epoch"])
 for label,row in (("selected",selected),("epoch12",endpoint)):
  ckpt=torch.load(row["checkpoint"],map_location="cpu",weights_only=False); model=_model(kind); model.load_state_dict(ckpt["model"],strict=True); _ema_into(model,ckpt["ema"]["shadow"])
  out=Path(plan.RESULT_ROOT)/f"formal_{tag}_{kind}_{label}_ema_model_state.pt"
  if out.exists(): raise FileExistsError(out)
  torch.save(model.state_dict(),out); row["plain_ema_model_state"] = str(out); row["plain_ema_model_state_sha256"]=_sha(out)
 report={"schema":"m1_optimized_v2_formal_postscore_v1","tag":tag,"operator":kind,"selection":"frozen EMA equal-session mean R2 over fixed 12 epochs; earliest tie","outer_query_opened":False,"n_scored_windows":len(dev_ds),"all_rows":rows,"selected":selected,"endpoint_epoch12":endpoint,"carrier_npz_sha256":loaded["receipt"]["digests"]["npz"]}
 receipt=Path(plan.RESULT_ROOT)/f"formal_{tag}_{kind}_postscore.json"
 if receipt.exists(): raise FileExistsError(receipt)
 receipt.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); return report
if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser();p.add_argument("--tag",required=True);p.add_argument("--kind",choices=("full_window","current_query"),required=True);a=p.parse_args();print(json.dumps(run(a.tag,a.kind),indent=2))
