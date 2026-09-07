"""Source-only real-data learnability gate: zero/mean baselines and tiny-set overfit."""
from __future__ import annotations
import json
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn.functional as F
from . import bank, plan
from .data import build_source_only_datamodule, materialize_source_banks
from .model import build, assert_common_binding

def _r2(pred,y):
    sse=float(np.square(pred-y).sum()); sst=float(np.square(y-y.mean(0,keepdims=True)).sum())
    return float("nan") if sst==0 else 1-sse/sst
def run(device="cpu",steps=200,n_windows=16):
    loaded=bank.load(); dm=build_source_only_datamodule(loaded); banks=materialize_source_banks()
    if getattr(dm,"target_path",None) is not None or dm.val_heldin_dataset is not None: raise RuntimeError("learnability gate has target leakage")
    # Form a single-session tiny set: mixing banks is invalid and would conceal a carrier bug.
    loader=dm.train_dataloader(); picked=None
    for batch in loader:
        neural,target,_calib,session,_carrier=batch[:5]; name=str(session[0]);
        if name in plan.SOURCE_SESSIONS and neural.shape[0]>=n_windows: picked=(neural[:n_windows],target[:n_windows],name); break
    if picked is None: raise RuntimeError("could not form source tiny set")
    x,y,name=picked; x=x.float(); y=y.float()
    if x.dim()==4: x=x.squeeze(-1) if x.shape[-1]==1 else x.mean(-1)
    if x.shape[-1]!=plan.N_UNITS and x.shape[1]==plan.N_UNITS: x=x.transpose(1,2)
    if y.dim()==3: y=y[:,-1,:]
    dev=torch.device(device); x,y=x.to(dev),y.to(dev); b=banks[name]; b=type(b)(E0=b.E0.to(dev),T=b.T.to(dev),unit_mask=b.unit_mask.to(dev))
    model=build("flat").to(dev); assert_common_binding(model); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-2)
    hist=[]
    for step in range(1,steps+1):
        opt.zero_grad(set_to_none=True); pred=model.forward_last(x,b); loss=F.mse_loss(pred,y)
        if not torch.isfinite(loss): raise RuntimeError("nonfinite M1 loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        if step in {1,20,50,100,steps}: hist.append({"step":step,"mse":float(loss.detach())})
    with torch.inference_mode(): pred=model.forward_last(x,b).cpu().numpy(); pert=model.forward_last(x+1,b).cpu().numpy()
    actual=y.cpu().numpy(); zero=np.zeros_like(actual); mean=np.broadcast_to(actual.mean(0,keepdims=True),actual.shape)
    report={"schema":"m1_optimized_v2_realdata_learnability_v1","source_only":True,"outer_query_opened":False,"carrier_revision":plan.CARRIER_REVISION,"session":name,"n_windows":int(n_windows),"steps":steps,"history":hist,"r2":{"model":_r2(pred,actual),"zero":_r2(zero,actual),"source_mean":_r2(mean,actual)},"mse":{"model":float(np.square(pred-actual).mean()),"zero":float(np.square(zero-actual).mean()),"source_mean":float(np.square(mean-actual).mean())},"prediction_std":float(pred.std()),"target_std":float(actual.std()),"input_response_l2":float(np.square(pert-pred).sum()**.5),"updated":datetime.now(timezone.utc).isoformat()}
    report["status"]="LEARN_OK" if report["r2"]["model"]>0.20 and report["r2"]["model"]>report["r2"]["source_mean"] and report["prediction_std"]>0.2*report["target_std"] and report["input_response_l2"]>1e-3 else "LEARN_FAIL"
    out=plan.RESULT_ROOT/"learnability_source_only.json"; out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); return report

if __name__=="__main__": print(json.dumps(run("cuda:0" if torch.cuda.is_available() else "cpu"),indent=2))
