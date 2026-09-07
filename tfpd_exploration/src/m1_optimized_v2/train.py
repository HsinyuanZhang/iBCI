"""Fixed-budget full-window M1 training; no outer session is opened or scored."""
from __future__ import annotations
import json, time
from datetime import datetime, timezone
import torch
import torch.nn.functional as F
from . import bank, plan
from .data import build_source_only_datamodule, materialize_source_banks
from .model import build, assert_common_binding

EPOCHS=12; LR=1e-4; WEIGHT_DECAY=1e-2; MICROBATCH=8
def _name(x): return x.decode() if isinstance(x,bytes) else str(x)
def run(arm="flat",epochs=EPOCHS,device="cuda:0"):
    if arm not in {"flat","route"}: raise ValueError(arm)
    loaded=bank.load(); dm=build_source_only_datamodule(loaded); banks=materialize_source_banks()
    if getattr(dm,"target_path",None) is not None or dm.val_heldin_dataset is not None: raise RuntimeError("full train requires target-unread source loader")
    dev=torch.device(device); model=build(arm).to(dev); assert_common_binding(model)
    gb={n:type(x)(E0=x.E0.to(dev),T=x.T.to(dev),unit_mask=x.unit_mask.to(dev)) for n,x in banks.items()}
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    dest=plan.RESULT_ROOT/f"train_{arm}"; dest.mkdir(parents=True,exist_ok=True); history=[]; start=time.time()
    for epoch in range(1,epochs+1):
        losses=[]; model.train()
        for batch in dm.train_dataloader():
            x,y,_calib,session,_carrier=batch[:5]; n=_name(session[0])
            if n not in gb: raise RuntimeError("non-source batch")
            x=x.float().to(dev); y=y.float().to(dev)
            if x.dim()==4: x=x.squeeze(-1) if x.shape[-1]==1 else x.mean(-1)
            if x.shape[-1]!=plan.N_UNITS and x.shape[1]==plan.N_UNITS: x=x.transpose(1,2)
            if y.dim()==3: y=y[:,-1,:]
            opt.zero_grad(set_to_none=True); total=0.
            for offset in range(0,x.shape[0],MICROBATCH):
                s=slice(offset,min(offset+MICROBATCH,x.shape[0])); loss=F.mse_loss(model.forward_last(x[s],gb[n]),y[s])*((s.stop-s.start)/x.shape[0]); loss.backward(); total+=float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); losses.append(total)
        row={"epoch":epoch,"train_mse":sum(losses)/max(len(losses),1),"elapsed_s":time.time()-start}; history.append(row)
        torch.save({"schema":"m1_optimized_v2_fixed_source_only_checkpoint","arm":arm,"epoch":epoch,"carrier_revision":plan.CARRIER_REVISION,"outer_query_opened":False,"model":model.state_dict(),"optimizer":opt.state_dict()},dest/f"epoch_{epoch:03d}.pt")
    receipt={"schema":"m1_optimized_v2_fixed_training_v1","arm":arm,"epochs":epochs,"candidate_budget":{"architectures":["flat","route"],"fixed_epochs":EPOCHS,"selection":"source-development only; no target session or terminal evaluation"},"source_only_manifest":dm.get_split_manifest(),"history":history,"updated":datetime.now(timezone.utc).isoformat()}
    (dest/"receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n"); return receipt
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--arm",default="flat"); p.add_argument("--epochs",type=int,default=EPOCHS); a=p.parse_args(); print(json.dumps(run(a.arm,a.epochs),indent=2))
