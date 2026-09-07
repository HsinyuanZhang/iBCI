"""GPU0-only steady-state RIFT H1 microbatch benchmark; never writes checkpoints."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
import h1_train as h

def step(model,opt,bank,x,valid,y,keep,micro):
    opt.zero_grad(set_to_none=True); total=0.0
    for off in range(0,len(x),micro):
        xb=x[off:off+micro]; vm=valid[off:off+micro]; yb=y[off:off+micro]
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            loss=nn.functional.mse_loss(h._forward(model,xb,bank,keep,vm).float(),yb)
        if not bool(torch.isfinite(loss)): raise FloatingPointError("nonfinite CUDA benchmark loss")
        (loss*(len(xb)/len(x))).backward(); total+=float(loss.detach())*len(xb)/len(x)
    grad=float(nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)); opt.step()
    return total,grad

def fp32_local_dense_gate(bank,x,valid,y,keep):
    """Same initialization/data, one FP32 forward/backward; not an optimizer comparison."""
    torch.manual_seed(h.SEED); local=h._decoder("recency",x.device,200,"local")
    torch.manual_seed(h.SEED); dense=h._decoder("recency",x.device,200,"dense")
    checks=[]
    for model in (local,dense):
        model.zero_grad(set_to_none=True); out=h._forward(model,x[:2],bank,keep,valid[:2]); loss=nn.functional.mse_loss(out.float(),y[:2]); loss.backward(); checks.append((out.detach(),float(loss.detach()),{n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}))
    out_err=float((checks[0][0]-checks[1][0]).abs().max()); loss_err=abs(checks[0][1]-checks[1][1]); grad_err=max(float((checks[0][2][n]-checks[1][2][n]).abs().max()) for n in checks[0][2])
    if not all(np.isfinite([out_err,loss_err,grad_err])): raise FloatingPointError("nonfinite local/dense FP32 gate")
    return {"batch":2,"fp32_output_max_abs":out_err,"fp32_loss_abs":loss_err,"fp32_grad_max_abs":grad_err,"finite":True}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--dest",type=Path,required=True); p.add_argument("--microbatch",type=int,choices=(16,32),required=True); p.add_argument("--warmup",type=int,default=5); p.add_argument("--measure",type=int,default=15); p.add_argument("--context-bins",type=int,default=200,choices=(200,)); p.add_argument("--backend",choices=("local","dense"),default="local"); args=p.parse_args()
    if torch.cuda.current_device()!=0: raise RuntimeError("benchmark requires visible GPU0 as cuda:0")
    torch.set_num_threads(2); device=torch.device("cuda:0"); torch.manual_seed(h.SEED); np.random.seed(h.SEED)
    train=h.build_train(args.context_bins); cal=h.b2.build_cal1_banks(train["banks"]); s=train["sessions"][0]; bank=cal["banks"][(s,int(cal["starts"][s][0]),3)]
    idx=np.arange(h.BATCH); x=torch.from_numpy(train["X"][s][idx]).to(device); valid=torch.from_numpy(train["valid"][s][idx]).to(device); y=torch.from_numpy(train["y"][s][idx]*h.SCALE).to(device)
    keep=torch.ones(bank.unit_mask.shape[0],dtype=torch.bool); parity=fp32_local_dense_gate(bank,x,valid,y,keep)
    model=h._decoder("recency",device,args.context_bins,args.backend); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    for _ in range(args.warmup): step(model,opt,bank,x,valid,y,keep,args.microbatch)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); rows=[]
    for _ in range(args.measure):
        t=time.perf_counter(); loss,grad=step(model,opt,bank,x,valid,y,keep,args.microbatch); torch.cuda.synchronize(); rows.append((time.perf_counter()-t,loss,grad))
    report={"schema":"rift_h1_r200_cuda_microbatch_benchmark_v1","context_bins":200,"layer_bins":[50,50,50,49],"backend":args.backend,"microbatch":args.microbatch,"effective_batch":32,"warmup_updates":args.warmup,"measured_updates":args.measure,"fp32_local_dense_gate":parity,"steady_step_seconds_mean":float(np.mean([r[0] for r in rows])),"steady_step_seconds_median":float(np.median([r[0] for r in rows])),"loss_last":rows[-1][1],"grad_norm_last":rows[-1][2],"peak_alloc_mib":float(torch.cuda.max_memory_allocated()/2**20),"peak_reserved_mib":float(torch.cuda.max_memory_reserved()/2**20),"finite":True,"checkpoint_written":False}
    args.dest.mkdir(parents=True,exist_ok=False); (args.dest/"benchmark.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
