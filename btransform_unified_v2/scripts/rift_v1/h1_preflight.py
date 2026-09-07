"""CPU-only real-data contract probe for H1-RIFT-D4-R300."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
import h1_train as h

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--dest",type=Path,required=True); p.add_argument("--variant",choices=("recency","flat"),default="recency"); p.add_argument("--context-bins",type=int,choices=(200,300),default=300); p.add_argument("--attention-backend",choices=("local","dense"),default="local"); p.add_argument("--microbatches",type=int,default=2); args=p.parse_args()
    torch.set_num_threads(2); dest=args.dest.resolve(); dest.mkdir(parents=True,exist_ok=True)
    train=h.build_train(args.context_bins); cal=h.b2.build_cal1_banks(train["banks"])
    session=train["sessions"][0]; bank=cal["banks"][(session, int(cal["starts"][session][0]), 3)]
    # These are real endpoints, including an early/reset-derived case if present.
    n=min(max(1,args.microbatches),len(train["X"][session])); xb=torch.from_numpy(train["X"][session][:n]); vm=torch.from_numpy(train["valid"][session][:n]); keep=torch.ones(bank.unit_mask.shape[0],dtype=torch.bool)
    model=h._decoder(args.variant,torch.device("cpu"),args.context_bins,args.attention_backend); init_sha=h._sha_state(model)
    paired_variant="flat" if args.variant=="recency" else "recency"; paired_sha=h._sha_state(h._decoder(paired_variant,torch.device("cpu"),args.context_bins,args.attention_backend))
    model.train(); pred=h._forward(model,xb,bank,keep,vm)
    target=torch.from_numpy(train["y"][session][:n]*h.SCALE)
    loss=torch.nn.functional.mse_loss(pred.float(),target); loss.backward()
    grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")))
    before={name:p.detach().clone() for name,p in model.named_parameters()}; opt=torch.optim.AdamW(model.parameters(),lr=1e-4); ema=h.DecoderEMA(model,decay=.9995); opt.step(); ema.update_after_step(model)
    changed=any(not torch.equal(before[name],p) for name,p in model.named_parameters())
    # H1's declared bridge is raw/20 vs native: report its exact loss relation.
    native_loss=torch.nn.functional.mse_loss((pred.detach()/h.SCALE),target/h.SCALE)
    scale_ratio=float(loss.detach()/native_loss)
    ends=train["ids"][session]; lengths=vm.sum(dim=1).tolist()
    report={"schema":"rift_h1_context_cpu_preflight_v1","ok":bool(torch.isfinite(pred).all() and torch.isfinite(loss) and np.isfinite(grad_norm) and grad_norm>0 and changed and abs(scale_ratio-400.0)<1e-3 and init_sha==paired_sha),"device":"cpu","threads":2,"variant":args.variant,"context_bins":args.context_bins,"attention_backend":args.attention_backend,"layer_bins":list(h.layer_bins(args.context_bins)),"raw_contract":f"x[a-{args.context_bins-1}:a], right-aligned; input_valid_mask excludes pre-reset padding","train_sessions":train["sessions"],"updates_per_epoch":train["updates_per_epoch"],"sample_session":session,"sample_endpoints":[int(x) for x in ends[:n]],"sample_valid_bins":[int(x) for x in lengths],"sample_x_shape":list(xb.shape),"bank_budget":3,"bank_e0_sha256":bank.calibration_meta.get("array_sha256"),"initialization_sha256":init_sha,"paired_variant":paired_variant,"paired_initialization_sha256":paired_sha,"paired_initialization_equal":init_sha==paired_sha,"prediction_shape":list(pred.shape),"loss":float(loss.detach()),"gradient_norm":grad_norm,"optimizer_changed_parameter":changed,"ema_updates":ema.n_updates,"scale_loss_ratio_raw_to_native":scale_ratio,"c2_bank_source":"btransform_unified_v1/scripts/h1_c2_cal1_b2_l200_p16.py::build_cal1_banks","no_gpu_initialized":not torch.cuda.is_initialized(),"official_test_used":False}
    (dest/"preflight.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2)); return 0 if report["ok"] else 2
if __name__=="__main__": raise SystemExit(main())
