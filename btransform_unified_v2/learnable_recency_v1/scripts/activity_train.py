#!/usr/bin/env python3
"""Joint, source-only training for the pure-activity calibration decoder.

The identity encoder is trained only from source support activity and source
query labels.  Evaluation data is intentionally never opened on this path.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PKG, ROOT, WS = HERE.parent, HERE.parent.parent, HERE.parent.parent.parent
V1 = WS / "btransform_unified_v1"
for path in (HERE, PKG / "src", ROOT / "src", ROOT, V1 / "src", WS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import activity_data
from btransform_unified_v1 import plan as v1_plan
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.model import unit_dropout_seed, whole_unit_dropout
from btransform_unified_v1.r2 import variance_weighted_r2
from btransform_unified_v1.schedule import warmup_cosine_lr
from learnable_recency_v1.activity_model import ActivityLearnableRiftDecoder, DEFAULT_SUPPORT_BINS
from learnable_recency_v1.config import add_learnable_flags, config_from_args
from learnable_recency_v1.wrap import apply_group_lrs, split_optimizer_parameters
from tfpd_exploration.src.m2_dual_track_v1 import plan as old_plan

SEED, BATCH = 42, 32
TASK_EPOCHS = {"m1": 24, "m2": 24, "h1": 32}
M2_UPDATES = 3165
SCHEMA, CKPT_SCHEMA = "activity_joint_early_pool_train_v2", "activity_joint_early_pool_epoch_checkpoint_v2"

def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def _atom(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)

def _append(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf8") as f:
        f.write(json.dumps(value, sort_keys=True, default=str) + "\n")

def _with_ema(model: nn.Module, ema: DecoderEMA, fn):
    raw = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    try:
        ema.apply_to(model)
        return fn()
    finally:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in raw: p.copy_(raw[n])

def source_hashes() -> dict[str, str]:
    paths = (Path(__file__), HERE / "activity_data.py", PKG / "src/learnable_recency_v1/activity_model.py",
             PKG / "src/learnable_recency_v1/config.py", PKG / "src/learnable_recency_v1/wrap.py",
             PKG / "src/learnable_recency_v1/temporal.py", ROOT / "src/btransform_unified_v2/model.py",
             ROOT / "src/btransform_unified_v2/temporal.py", V1 / "src/btransform_unified_v1/identity_variant.py",
             V1 / "src/btransform_unified_v1/model.py", V1 / "src/btransform_unified_v1/ema.py",
             V1 / "src/btransform_unified_v1/schedule.py", V1 / "src/btransform_unified_v1/plan.py",
             WS / "tfpd_exploration/src/m2_dual_track_v1/plan.py", HERE / "activity_score.py",
             HERE / "m2_projadd_learnable_score.py", HERE / "m1_full_learnable_train.py", HERE / "h1_learnable_train.py")
    return {str(p): _sha(p) for p in paths if p.is_file()}

def _trial_mask(item: Mapping[str, Any], device: torch.device) -> torch.Tensor | None:
    mask = item.get("support_provenance", {}).get("valid_mask")
    if mask is None: return None
    # H1 is padded per bin; encoder's current interface accepts trial validity.
    a = np.asarray(mask)
    return torch.as_tensor(a.any(axis=1) if a.ndim == 2 else a, device=device, dtype=torch.bool)

def _support(item: Mapping[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    # Activity files are read-only mmaps.  Copy before crossing into torch so
    # no training kernel can ever write into the source calibration artifact.
    return torch.as_tensor(np.array(item["activity"], dtype=np.float32, copy=True), device=device), _trial_mask(item, device)

def score_surface(model: ActivityLearnableRiftDecoder, surface: Mapping[str, Any], *, context: int,
                  device: torch.device, behavior_scale: float, max_batches: int | None = None,
                  save_predictions: Path | None = None, identities: dict[str, torch.Tensor] | None = None,
                  capture_arrays: bool = False) -> dict[str, Any]:
    model.eval(); rows = {}; ps=[]; ts=[]
    for session, item in surface.items():
        if identities is not None and session in identities:
            identity = identities[session]
        else:
            activity, trial_mask = _support(item, device)
            # Keep target calibration visibly outside every query graph.  The
            # model also enforces this, but the runner owns the protocol boundary.
            with torch.no_grad():
                identity = model.calibrate(activity, trial_mask)
            if identities is not None: identities[session] = identity
        pred=[]; target=[]
        for start in range(0, len(item["starts"]), BATCH):
            if max_batches is not None and start // BATCH >= max_batches: break
            ids=np.arange(start, min(start+BATCH, len(item["starts"])), dtype=np.int64)
            x, _y, valid = activity_data.windows(item, ids, context, device, behavior_scale)
            with torch.no_grad(): p=model(x, identity=identity, input_valid_mask=valid).float().cpu().numpy()/behavior_scale
            pred.append(p); target.append(np.asarray(item["Y"], np.float32)[ids])
        if not pred: continue
        p=np.concatenate(pred); t=np.concatenate(target); r=float(variance_weighted_r2(t,p))
        rows[session]={"r2":r,"window_count":len(t),"identity_norm":float(identity.norm().cpu()),
                       "prediction_sha256":hashlib.sha256(p.tobytes()).hexdigest()}
        if capture_arrays: rows[session]["_prediction"],rows[session]["_target"]=p,t
        if save_predictions is not None:
            save_predictions.mkdir(parents=True, exist_ok=True); np.save(save_predictions / f"{session}.npy", p)
        ps.append(p); ts.append(t)
    if not rows: return {"per_session":{},"equal_session_mean":None,"pooled_r2":None,"n_windows":0,"partial":max_batches is not None}
    return {"per_session":rows,"equal_session_mean":float(np.mean([r["r2"] for r in rows.values()])),
            "pooled_r2":float(variance_weighted_r2(np.concatenate(ts),np.concatenate(ps))),
            "n_windows":int(sum(r["window_count"] for r in rows.values())),"partial":max_batches is not None}

def _m2_manifest_batches(train: Mapping[str, Any], epoch: int):
    from scripts.rift_v1 import m2_train as frozen
    manifest=frozen.old_sampler.load_manifest(frozen.MANIFEST)
    if manifest.get("digest") != frozen.MANIFEST_DIGEST or int(manifest.get("batch_size",0)) != BATCH or len(manifest["batches"][str(epoch)]) != M2_UPDATES:
        raise RuntimeError("M2 frozen 24x3165/B32 seed42 manifest drift")
    if set(train) != set(manifest["sessions"]): raise RuntimeError("M2 source session roster drift")
    lengths={s:len(v["starts"]) for s,v in train.items()}
    if lengths != manifest.get("lengths"): raise RuntimeError("M2 frozen manifest lengths drift")
    return [(r["session"],np.asarray(r["indices"],np.int64)) for r in manifest["batches"][str(epoch)]]

def _batches(task: str, train: Mapping[str, Any], epoch: int):
    if task == "m2": return _m2_manifest_batches(train, epoch)
    rng=np.random.default_rng(SEED + epoch); out=[]
    for session in sorted(train):
        ids=rng.permutation(len(train[session]["starts"]))
        out += [(session, ids[i:i+BATCH]) for i in range(0,len(ids),BATCH)]
    rng.shuffle(out)
    return out

def run(args: argparse.Namespace) -> dict[str, Any]:
    task=args.task.lower(); smoke=args.smoke_steps is not None
    if task not in TASK_EPOCHS: raise ValueError("--task must be m1, m2, or h1")
    if not smoke and args.epochs != TASK_EPOCHS[task]: raise ValueError(f"formal {task} needs {TASK_EPOCHS[task]} epochs")
    if args.seed != SEED or args.proj_dim != 16: raise ValueError("activity recipe is seed42/P16")
    device=torch.device(args.device); torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    # Crucial: this never requests evaluation, so M2 training cannot open EXT6.
    data=activity_data.load_task_data(task, include_eval=False); train=data["train"]; mini=data.get("validation",{})
    if not train: raise RuntimeError("source training surface is empty")
    context=int(data["metadata"]["context"]); scale=float(data["metadata"].get("behavior_scale",5.0))
    cfg=config_from_args(args, task); model=ActivityLearnableRiftDecoder(task,cfg,context_bins=context,seed=SEED,support_bins=args.support_bins,identity_hidden=args.identity_hidden).to(device)
    model.temporal.set_attention_backend("local")
    ema=DecoderEMA(model,decay=v1_plan.EMA_DECAY)
    groups=split_optimizer_parameters(model,peak_lr=v1_plan.LR_PEAK,weight_decay=v1_plan.WEIGHT_DECAY,lr_multiplier=cfg.lr_multiplier)
    encoder_names={n for n,_ in model.named_parameters() if n.startswith("identity_encoder.")}
    group_names={n for group in groups for n,p in model.named_parameters() if any(p is x for x in group["params"])}
    if not encoder_names <= group_names: raise RuntimeError("optimizer omits activity encoder parameters")
    opt=torch.optim.AdamW(groups,lr=v1_plan.LR_PEAK,weight_decay=v1_plan.WEIGHT_DECAY,betas=old_plan.ADAM_BETAS,eps=old_plan.ADAM_EPS)
    dest=args.dest.resolve()
    if dest.exists() and any(dest.iterdir()): raise FileExistsError("fresh destination must be empty")
    dest.mkdir(parents=True,exist_ok=True)
    meta={"schema":SCHEMA,"status":"SMOKE" if smoke else "FORMAL","task":task,"seed":SEED,"context_bins":context,"proj_dim":16,"epochs":TASK_EPOCHS[task],"batch":BATCH,"identity_interface":"activity_pre_pool_post_pool_joint","identity_architecture":"existing_early_pool_activity_trunk_v2","zero_carrier":True,"no_T":True,"no_frozen_E0":True,"no_TaskBank":True,"support_bins":model.support_bins,"identity_hidden":model.identity_hidden,"learnable_config":cfg.__dict__,"selection":{"primary":"FULL-compatible public calibration EMA scan; final claims EvalAI","source_minival":"audit only; never checkpoint selection"},"optimizer":{"name":"AdamW","lr":v1_plan.LR_PEAK,"weight_decay":v1_plan.WEIGHT_DECAY,"betas":list(old_plan.ADAM_BETAS),"eps":old_plan.ADAM_EPS,"grad_clip":v1_plan.GRAD_CLIP,"ema_decay":v1_plan.EMA_DECAY,"unit_dropout":v1_plan.UNIT_DROPOUT,"warmup_updates":M2_UPDATES if task=="m2" else None},"behavior_scale":scale,"sampler":"frozen M2 manifest 24x3165/B32 seed42" if task=="m2" else "per-epoch deterministic global session-batch shuffle","source_hashes":source_hashes(),"input_hashes":{k:{s:v.get("hashes",{}) for s,v in data.get(k,{}).items()} for k in ("train","validation")},"support_provenance":{k:{s:v.get("support_provenance",{}) for s,v in data.get(k,{}).items()} for k in ("train","validation")},"utc":datetime.now(timezone.utc).isoformat()}
    _atom(dest/"run_meta.json",meta); step=0; started=time.monotonic(); initial={n:p.detach().clone() for n,p in model.named_parameters() if "identity_encoder" in n}
    total=(M2_UPDATES if task=="m2" else max(1,len(_batches(task,train,1))))*TASK_EPOCHS[task]
    for epoch in range(1,TASK_EPOCHS[task]+1):
        model.train(); losses=[]; grad_norms=[]; encoder_grad={"pre_pool":[],"post_pool":[]}
        for bi,(session,ids) in enumerate(_batches(task,train,epoch)):
            item=train[session]; x,y,valid=activity_data.windows(item,ids,context,device,scale); a,tm=_support(item,device)
            step+=1; apply_group_lrs(opt,warmup_cosine_lr(step,total_steps=total,warmup_steps=(M2_UPDATES if task=="m2" else max(1,total//TASK_EPOCHS[task])),peak=v1_plan.LR_PEAK,min_factor=v1_plan.LR_MIN_FACTOR))
            g=torch.Generator(device="cpu"); g.manual_seed(unit_dropout_seed(SEED,epoch,bi)); keep=whole_unit_dropout(torch.ones(model.units,dtype=torch.bool),p=v1_plan.UNIT_DROPOUT,generator=g)
            opt.zero_grad(set_to_none=True)
            with (torch.autocast(device_type="cuda",dtype=torch.bfloat16) if device.type=="cuda" else contextlib.nullcontext()): loss=nn.functional.mse_loss(model(x,a,trial_mask=tm,dropout_keep=keep,input_valid_mask=valid).float(),y)
            if not torch.isfinite(loss): raise FloatingPointError("nonfinite loss")
            loss.backward()
            for branch in encoder_grad:
                grads=[p.grad.detach().norm() for n,p in model.named_parameters() if n.startswith(f"identity_encoder.{branch}.") and p.grad is not None]
                encoder_grad[branch].append(float(torch.stack(grads).norm().cpu()) if grads else 0.0)
            grad_norms.append(float(nn.utils.clip_grad_norm_(model.parameters(),v1_plan.GRAD_CLIP,error_if_nonfinite=True))); opt.step(); ema.update_after_step(model); losses.append(float(loss.detach().cpu()))
            if smoke and step >= args.smoke_steps: break
        raw=score_surface(model,mini,context=context,device=device,behavior_scale=scale,max_batches=1 if smoke else None) if mini else None
        es=_with_ema(model,ema,lambda:score_surface(model,mini,context=context,device=device,behavior_scale=scale,max_batches=1 if smoke else None)) if mini else None
        ckpt=dest/f"epoch_{epoch:03d}.pt"; torch.save({"schema":CKPT_SCHEMA,"task":task,"epoch":epoch,"global_step":step,"smoke":smoke,"raw_state_dict":model.state_dict(),"ema":ema.state_dict(),"optimizer":opt.state_dict(),"run_meta":meta},ckpt)
        changed={n:float((p.detach()-initial[n]).norm().cpu()) for n,p in model.named_parameters() if n in initial}
        audit={k:{"mean":float(np.mean(v)),"max":float(np.max(v)),"finite":bool(np.isfinite(v).all()),"nonzero":bool(np.max(v)>0)} for k,v in encoder_grad.items()}
        if not all(v["finite"] and v["nonzero"] for v in audit.values()): raise RuntimeError("encoder pre_pool/post_pool did not receive finite nonzero gradients")
        row={"epoch":epoch,"global_step":step,"train_mse":float(np.mean(losses)),"grad_norm":float(np.mean(grad_norms)),"encoder_grad_audit":audit,"encoder_parameter_delta_norms":changed,"source_minival_raw":raw,"source_minival_ema":es}; _append(dest/"metrics.jsonl",row); _atom(dest/"heartbeat.json",row)
        if not smoke and task=="m2" and (len(losses)!=M2_UPDATES or step!=epoch*M2_UPDATES or ema.n_updates!=step): raise RuntimeError("M2 exact manifest/EMA accounting drift")
        if smoke: break
    final_epoch=epoch
    if not smoke and task=="m2" and step!=24*M2_UPDATES: raise RuntimeError("M2 total update count drift")
    # A reload/prediction check makes the CPU smoke receipt reviewable.
    state=torch.load(dest/f"epoch_{final_epoch:03d}.pt",map_location=device,weights_only=False); restored=ActivityLearnableRiftDecoder(task,cfg,context_bins=context,seed=SEED,support_bins=model.support_bins,identity_hidden=args.identity_hidden).to(device); restored.temporal.set_attention_backend("local"); restored.load_state_dict(state["raw_state_dict"]); sample=next(iter(train.values())); sx,_,sv=activity_data.windows(sample,np.arange(1),context,device,scale); sa,stm=_support(sample,device); fixed_keep=torch.ones(model.units,dtype=torch.bool,device=device)
    with torch.no_grad():
        raw_parity=bool(torch.equal(model.eval()(sx,sa,trial_mask=stm,dropout_keep=fixed_keep,input_valid_mask=sv),restored.eval()(sx,sa,trial_mask=stm,dropout_keep=fixed_keep,input_valid_mask=sv)))
        shadow=state["ema"]["shadow"]
        for n,p in restored.named_parameters():
            if n in shadow: p.copy_(shadow[n].to(p.device,p.dtype))
        ema_parity=bool(torch.equal(_with_ema(model,ema,lambda:model.eval()(sx,sa,trial_mask=stm,dropout_keep=fixed_keep,input_valid_mask=sv)),restored.eval()(sx,sa,trial_mask=stm,dropout_keep=fixed_keep,input_valid_mask=sv)))
    parity=raw_parity and ema_parity
    receipt={"schema":"activity_joint_smoke_receipt_v1" if smoke else "activity_joint_train_receipt_v1","status":"COMPLETED","formal_claim":not smoke,"selection":{"checkpoint":f"epoch_{final_epoch:03d}.pt","view":"EMA","rule":"smoke only" if smoke else "last completed training checkpoint; local selection performed by FULL-compatible EMA scan"},"global_step":step,"checkpoint_parity":{"raw":raw_parity,"ema":ema_parity},"finite_loss":bool(math.isfinite(row["train_mse"])),"encoder_gradient_audit":row["encoder_grad_audit"],"encoder_updated":bool(any(v>0 for v in row["encoder_parameter_delta_norms"].values())),"runtime_seconds":time.monotonic()-started}
    if smoke and not (parity and receipt["finite_loss"] and receipt["encoder_updated"]): raise RuntimeError("activity smoke validation failed")
    _atom(dest/("smoke_receipt.json" if smoke else "train_receipt.json"),receipt); return receipt

def build_parser():
    p=argparse.ArgumentParser(description=__doc__); add_learnable_flags(p); p.set_defaults(ladder="default")
    p.add_argument("--task",required=True,choices=("m1","m2","h1")); p.add_argument("--dest",type=Path,required=True); p.add_argument("--device",default="cpu"); p.add_argument("--cpu-threads",type=int,default=2); p.add_argument("--epochs",type=int); p.add_argument("--seed",type=int,default=SEED); p.add_argument("--proj-dim",type=int,default=16); p.add_argument("--support-bins",type=int); p.add_argument("--identity-hidden",type=int,default=None); p.add_argument("--smoke-steps",type=int)
    return p
def main():
    a=build_parser().parse_args(); a.epochs=TASK_EPOCHS[a.task] if a.epochs is None else a.epochs
    if a.smoke_steps is not None and a.smoke_steps<1: raise ValueError("--smoke-steps must be positive")
    print(json.dumps(run(a),indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
