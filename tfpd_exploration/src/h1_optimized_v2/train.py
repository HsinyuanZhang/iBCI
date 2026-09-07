"""Short-overfit and source-only training for corrected-unit H1 T-current-query."""
from __future__ import annotations
import argparse, json, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .data import build_window_manifest, collate_runtime_target, materialize_banks, shuffled_batches
from .model import H1CurrentQueryDecoder, make_matched_pair

ROOT = Path(__file__).resolve().parents[2] / "results/decoder_validation_v2/20260905_190000/h1"
SEED=42

def r2(pred, target):
    ss=((target-target.mean(0))**2).sum(); return float(1-((pred-target)**2).sum()/ss)

def main(epochs=12, tiny=0, max_batches=0, activity_scale=32.0, arm="t"):
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    device=torch.device("cuda:0")
    ROOT.mkdir(parents=True, exist_ok=True)
    pack=build_window_manifest(); sessions=pack["train_sessions"]; banks=materialize_banks(sessions)
    gb={k:type(v)(v.E0.to(device),v.T.to(device),v.unit_mask.to(device)) for k,v in banks.items()}
    full, query = make_matched_pair(activity_scale=activity_scale)
    model = (full if arm == "full" else query).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-2)
    fixed=None; log=[]
    for ep in range(1,epochs+1):
      batches=shuffled_batches(sessions,seed=SEED,epoch=ep,batch_size=8)
      if tiny:
        if fixed is None: fixed=batches[0][:tiny]
        batches=[fixed]
      if max_batches: batches=batches[:max_batches]
      losses=[]
      for items in batches:
        name,x,y,native,_=collate_runtime_target(sessions,items); x=x.to(device); y=y.to(device)
        pred=model.forward_last(x,gb[name]); loss=F.mse_loss(pred,y)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.detach()))
      with torch.no_grad():
        if tiny:
          name,x,y,native,_=collate_runtime_target(sessions,fixed); p=model.forward_last(x.to(device),gb[name]).cpu(); rr=r2(p/20,native)
        else: rr=None
      log.append({"epoch":ep,"loss":float(np.mean(losses)),"tiny_native_r2":rr,"batches":len(batches)})
      print(log[-1],flush=True)
    out={"schema":"h1_optimized_v2","arm":arm,"initialization":"full-window and T are constructed by make_matched_pair; frontend/final-norm/readout copied exactly and T temporal maps compatible full temporal weights","frontend_revision":"original_v1" if activity_scale == 1.0 else f"activity_balanced_scalar_v1_scale_{activity_scale:g}","target_contract":"train target=20*raw_nwb_velocity; score/runtime prediction=raw_output/20","epochs":epochs,"tiny":tiny,"log":log}
    revision = "original_v1" if activity_scale == 1.0 else f"activity_balanced_scalar_v1_scale_{activity_scale:g}"
    stem = ("tiny_overfit" if tiny else "train") + "__" + arm + "__" + revision
    (ROOT/(stem + ".json")).write_text(json.dumps(out,indent=2)+"\n")
    torch.save({"model":model.state_dict(),"contract":out["target_contract"],"frontend_revision":revision},ROOT/(stem + ".pt"))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=12);p.add_argument('--tiny',type=int,default=0);p.add_argument('--max-batches',type=int,default=0);p.add_argument('--activity-scale',type=float,default=32.0);p.add_argument('--arm',choices=('full','t'),default='t');a=p.parse_args();main(a.epochs,a.tiny,a.max_batches,a.activity_scale,a.arm)
