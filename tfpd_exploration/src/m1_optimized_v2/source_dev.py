"""Frozen chronological known-source-development FULL-vs-current-query control.

This is not a clean source holdout: D0 is presealed from all source EMG. Per
source, decoder training begins after M10 and ends before an 80% trial cut;
development full W=100 windows start at/after that cut. Outer 20120924 unread.
"""
from __future__ import annotations
import copy, hashlib, json, sys, random
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from . import bank, plan
from .data import SourceCarrierDataset, build_source_only_datamodule, materialize_source_banks
from .model import assert_common_binding, build as build_query
from tfpd_exploration.src.m1_temporal_decoder_quick_product_v1.ema import DecoderEMA
from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_temporal import whole_unit_dropout
from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_temporal import M1TemporalFlatDecoder
_EXP=str(plan.REPO_ROOT / "streaming_calibration_exp")
if _EXP not in sys.path: sys.path.insert(0,_EXP)
from src.data.falcon_datamodule import SessionBatchSampler
EPOCHS=12; TAG="chron80_v2"; CALIB=10
def _name(x): return x.decode() if isinstance(x,bytes) else str(x)
def _sha(values):
 h=hashlib.sha256()
 for name,start in values: h.update(name.encode()); h.update(np.asarray([start],dtype=np.int64).tobytes())
 return h.hexdigest()
def _r2(p,y):
 sse=float(np.square(p-y).sum()); sst=float(np.square(y-y.mean(0,keepdims=True)).sum()); return 1-sse/sst
def _shape(x,y):
 x=x.float(); y=y.float()
 if x.dim()==4: x=x.squeeze(-1) if x.shape[-1]==1 else x.mean(-1)
 if x.shape[-1]!=64 and x.shape[1]==64: x=x.transpose(1,2)
 if y.dim()==3: y=y[:,-1,:]
 return x,y
def _banks(device):
 values=materialize_source_banks()
 return {n:type(v)(E0=v.E0.to(device),T=v.T.to(device),unit_mask=v.unit_mask.to(device)) for n,v in values.items()}
def _clone(dataset,items,loaded):
 inner=copy.copy(dataset.base); inner.window_indices=list(items)
 return SourceCarrierDataset(inner,loaded)
def _split(dm):
 base=dm.train_dataset.base; train=[]; dev=[]; rows={}
 for name in plan.SOURCE_SESSIONS:
  starts=np.asarray(base.trial_start_indices[name],dtype=np.int64); cut=int(np.floor(.8*len(starts)))
  if not CALIB<cut<len(starts): raise RuntimeError(f"invalid chronological split {name}")
  start_after=int(starts[CALIB]); cut_bin=int(starts[cut]); session=[(n,int(s)) for n,s in base.window_indices if n==name]
  a=[z for z in session if z[1]>=start_after and z[1]+plan.WINDOW-1<cut_bin]
  b=[z for z in session if z[1]>=cut_bin]
  if not a or not b: raise RuntimeError(f"empty chronological split {name}")
  train+=a; dev+=b; rows[name]={"n_trials":int(len(starts)),"calibration_trials":[0,CALIB],"cut_trial":cut,"cut_padded_bin":cut_bin,"train_windows":len(a),"dev_windows":len(b),"train_window_starts_sha256":_sha(a),"dev_window_starts_sha256":_sha(b),"purge_rule":"train end < cut; dev full W starts >= cut"}
 return train,dev,rows
def _model(kind):
 if kind=="current_query":
  m=build_query("flat"); assert_common_binding(m); return m
 if kind=="full_window": return M1TemporalFlatDecoder(seed=plan.SEED)
 raise ValueError(kind)
def _write_once(path,obj):
 encoded=json.dumps(obj,indent=2,sort_keys=True)+"\n"
 if path.exists() and path.read_text()!=encoded: raise FileExistsError(f"frozen artifact differs: {path}")
 if not path.exists(): path.write_text(encoded)
 return encoded
def _metrics(pred, target, train_target):
    mean=np.broadcast_to(train_target.mean(0,keepdims=True),target.shape)
    return {"r2":{"model":_r2(pred,target),"zero":_r2(np.zeros_like(target),target),"train_only_source_mean":_r2(mean,target)},"mse":{"model":float(np.square(pred-target).mean()),"zero":float(np.square(target).mean()),"train_only_source_mean":float(np.square(mean-target).mean())},"prediction_std":float(pred.std()),"target_std":float(target.std())}
def _keep(batch,epoch,batch_id,device):
    token=hashlib.sha256(f"m1_optimized_v2_keep|42|{epoch}|{batch_id}".encode()).digest()
    g=torch.Generator(device="cpu").manual_seed(int.from_bytes(token[:8],"little")%(2**63))
    # One independently sampled whole-unit mask per example, identically used
    # by FULL and current-query arms.  `whole_unit_dropout` itself normalizes
    # 1-D masks to [1,N], so supply [B,N] directly; never add a second axis.
    return whole_unit_dropout(torch.ones(batch,64,dtype=torch.bool),p=.10,generator=g).to(device)
def _all_dev_batches(dataset, batch_size=32):
    """Session-pure evaluation batches retaining every tail window."""
    groups={name:[] for name in plan.SOURCE_SESSIONS}
    for index,(name,_start) in enumerate(dataset.base.window_indices): groups[name].append(index)
    batches=[]
    for name in plan.SOURCE_SESSIONS:
        for left in range(0,len(groups[name]),batch_size): batches.append(groups[name][left:left+batch_size])
    if sum(map(len,batches)) != len(dataset): raise RuntimeError("dev sampler omitted a window")
    return batches
def run(kind,epochs=EPOCHS,device="cuda:0",tag=TAG,resume=None):
 loaded=bank.load(); dm=build_source_only_datamodule(loaded)
 if getattr(dm,"target_path",None) is not None or dm.val_heldin_dataset is not None: raise RuntimeError("outer/query materialized")
 tr,dv,rows=_split(dm); split={"schema":"m1_optimized_v2_known_source_chron80_split_v1","tag":tag,"source_sessions":list(plan.SOURCE_SESSIONS),"outer_session":plan.OUTER_SESSION,"outer_path_resolved":False,"outer_query_opened":False,"known_source_development_not_clean_holdout":True,"d0_source_full_emg_presealed":True,"rows":rows,"total_train_windows":len(tr),"total_dev_windows":len(dv),"train_window_ids_sha256":_sha(tr),"dev_window_ids_sha256":_sha(dv),"selection":"matched FULL/current-query source-dev only; never outer24"}
 split_path=plan.RESULT_ROOT/f"source_dev_{tag}_split.json"; split_encoded=_write_once(split_path,split)
 train_ds=_clone(dm.train_dataset,tr,loaded); dev_ds=_clone(dm.train_dataset,dv,loaded)
 train_sampler=SessionBatchSampler(train_ds,32,shuffle=True,seed=plan.SEED,balance_sessions=False,reshuffle_each_epoch=False); dev_sampler=_all_dev_batches(dev_ds)
 # Baseline is fit only on behavior targets available in decoder training.
 train_y=[]
 for item in DataLoader(train_ds,batch_sampler=train_sampler):
  _,y,*_=item
  if y.dim()==3: y=y[:,-1,:]
  train_y.append(y.numpy())
 train_y=np.concatenate(train_y)
 dev_y=[]
 for item in DataLoader(dev_ds,batch_sampler=dev_sampler):
  _,y,*_=item
  if y.dim()==3: y=y[:,-1,:]
  dev_y.append(y.numpy())
 y0=np.concatenate(dev_y); baseline=_metrics(np.zeros_like(y0),y0,train_y); baseline["n_windows"]=int(len(y0))
 _write_once(plan.RESULT_ROOT/f"source_dev_{tag}_baseline.json",{"schema":"m1_optimized_v2_source_dev_trainonly_baseline_v2","split_sha256":hashlib.sha256(split_encoded.encode()).hexdigest(),**baseline})
 dev=torch.device(device); banks=_banks(dev); model=_model(kind).to(dev); opt=torch.optim.AdamW([{"params":[p for n,p in model.named_parameters() if p.requires_grad and p.ndim>1],"weight_decay":1e-2},{"params":[p for n,p in model.named_parameters() if p.requires_grad and p.ndim<=1],"weight_decay":0.0}],lr=1e-4); ema=DecoderEMA(model,decay=.9995); sampler=train_sampler; history=[]; step=0; first=1
 if resume is not None:
  state=torch.load(resume,map_location="cpu",weights_only=False)
  if state["split_train_sha256"]!=split["train_window_ids_sha256"] or state["split_dev_sha256"]!=split["dev_window_ids_sha256"] or state.get("carrier_npz_sha256")!=loaded["receipt"]["digests"]["npz"] or state["operator"]!=kind or state["recipe"]!={"adamw_lr":1e-4,"weight_decay":.01,"bias_norm_weight_decay":0.,"warmup_epochs":1,"ema":.9995,"unit_dropout":.10,"clip":1.}: raise RuntimeError("resume split/carrier/operator/recipe drift")
  model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"]); ema.load_checkpoint_state(state["ema"]); step=int(state["global_step"]); first=int(state["epoch"])+1
  if "python" not in state["rng"] or "cuda" not in state["rng"]: raise RuntimeError("resume checkpoint has incomplete RNG snapshot; live prepatch checkpoints are not resumable")
  random.setstate(state["rng"]["python"]); np.random.set_state(state["rng"]["numpy"]); torch.set_rng_state(state["rng"]["torch"]); torch.cuda.set_rng_state_all(state["rng"]["cuda"])
 for epoch in range(first,epochs+1):
  model.train(); losses=[]
  for batch_id,item in enumerate(DataLoader(train_ds,batch_sampler=sampler)):
   x,y,_c,s,_t=item[:5]; x,y=_shape(x,y); x,y=x.to(dev),y.to(dev); step+=1
   for g in opt.param_groups: g["lr"]=1e-4*min(1.,step/max(1,len(sampler)))
   opt.zero_grad(set_to_none=True); loss=F.mse_loss(model.forward_last(x,banks[_name(s[0])],dropout_keep=_keep(len(x),epoch,batch_id,dev)),y)
   if not torch.isfinite(loss): raise RuntimeError("nonfinite source-dev loss")
   loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); ema.update_after_step(model); losses.append(float(loss.detach()))
  history.append({"epoch":epoch,"train_mse":sum(losses)/len(losses)})
  ckpt=plan.RESULT_ROOT/f"source_dev_{tag}_{kind}_epoch_{epoch:03d}.pt"
  if ckpt.exists(): raise FileExistsError(f"attempt checkpoint exists: {ckpt}")
  torch.save({"schema":"m1_optimized_v2_source_dev_checkpoint_v3","tag":tag,"operator":kind,"epoch":epoch,"global_step":step,"split_train_sha256":split["train_window_ids_sha256"],"split_dev_sha256":split["dev_window_ids_sha256"],"carrier_npz_sha256":loaded["receipt"]["digests"]["npz"],"model":model.state_dict(),"optimizer":opt.state_dict(),"ema":ema.checkpoint_state(),"rng":{"python":random.getstate(),"torch":torch.get_rng_state(),"numpy":np.random.get_state(),"cuda":torch.cuda.get_rng_state_all()},"recipe":{"adamw_lr":1e-4,"weight_decay":.01,"bias_norm_weight_decay":0.,"warmup_epochs":1,"ema":.9995,"unit_dropout":.10,"clip":1.},"outer_query_opened":False},ckpt)
 model.eval(); ps=[]; ys=[]; names=[]
 with torch.inference_mode():
  for item in DataLoader(dev_ds,batch_sampler=dev_sampler):
   x,y,_c,s,_t=item[:5]; name=_name(s[0])
   if any(_name(z)!=name for z in s): raise RuntimeError("development batch is not session-pure")
   x,y=_shape(x,y); ps.append(model.forward_last(x.to(dev),banks[name]).cpu().numpy()); ys.append(y.numpy()); names.extend([name]*len(y))
 p=np.concatenate(ps); y=np.concatenate(ys)
 if len(y)!=len(dev_ds) or len(y)!=split["total_dev_windows"]: raise RuntimeError(f"development scoring omitted windows: {len(y)} vs {len(dev_ds)}")
 per={}
 for name in plan.SOURCE_SESSIONS:
  take=np.asarray([x==name for x in names]); per[name]=_metrics(p[take],y[take],train_y)
 equal=float(np.mean([per[n]["r2"]["model"] for n in plan.SOURCE_SESSIONS])); metrics=_metrics(p,y,train_y)
 report={"schema":"m1_optimized_v2_source_dev_matched_v4","tag":tag,"operator":kind,"seed":plan.SEED,"epochs":epochs,"split_path":str(split_path),"baseline":baseline,"pooled":metrics,"per_session":per,"equal_session_mean_r2":equal,"n_scored_windows":int(len(y)),"history":history,"outer_query_opened":False,"updated":datetime.now(timezone.utc).isoformat()}
 out=plan.RESULT_ROOT/f"source_dev_{tag}_{kind}.json"
 if out.exists(): raise FileExistsError(f"attempt exists: {out}")
 out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); return report
if __name__=="__main__":
 import argparse
 parser=argparse.ArgumentParser(); parser.add_argument("--kind",choices=("full_window","current_query"),required=True); parser.add_argument("--epochs",type=int,default=EPOCHS); parser.add_argument("--tag",default=TAG); parser.add_argument("--resume",default=None); a=parser.parse_args(); print(json.dumps(run(a.kind,a.epochs,tag=a.tag,resume=a.resume),indent=2))
