#!/usr/bin/env python3
"""Calibrate each target session once, then score frozen EMA predictions."""
from __future__ import annotations
import argparse, json, sys, time
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent; PKG,ROOT,WS=HERE.parent,HERE.parent.parent,HERE.parent.parent.parent; V1=WS/"btransform_unified_v1"
for p in (HERE,PKG/"src",ROOT/"src",ROOT,V1/"src",WS):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
import activity_data
import activity_train as train
from learnable_recency_v1.activity_model import ActivityLearnableRiftDecoder
from learnable_recency_v1.config import config_from_run_meta

SCHEMA="activity_joint_score_v1"
def _read(p): return json.loads(p.read_text())
def _ema_into(model,state):
    shadow=state.get("ema",{}).get("shadow",{}); named={n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(shadow)!=set(named): raise RuntimeError("checkpoint EMA/model parameter mismatch")
    with torch.no_grad():
        for n,p in named.items(): p.copy_(shadow[n].to(p.device,p.dtype))
def _atom(p,x):
    p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+".tmp"); q.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+"\n"); q.replace(p)
def run(args):
    started=time.monotonic(); run_dir=args.run_dir.resolve(); meta=_read(run_dir/"run_meta.json")
    if meta.get("schema")!=train.SCHEMA: raise RuntimeError("not an activity-joint training run")
    task=meta["task"]; smoke=meta.get("status")=="SMOKE"; receipt=_read(run_dir/("smoke_receipt.json" if smoke else "train_receipt.json"))
    if receipt.get("status")!="COMPLETED" or (smoke and not args.allow_smoke): raise RuntimeError("completed formal run required (or --allow-smoke)")
    fixed_epoch=train.TASK_EPOCHS[task]
    if not smoke and (meta.get("status")!="FORMAL" or int(meta.get("epochs",-1))!=fixed_epoch or receipt.get("selection",{}).get("checkpoint") != f"epoch_{fixed_epoch:03d}.pt"): raise RuntimeError("formal score requires a completed all-epoch training run")
    if smoke and (args.max_batches is None or args.max_batches<1): raise RuntimeError("smoke score requires --max-batches >= 1")
    device=torch.device(args.device); torch.set_num_threads(args.cpu_threads); cfg=config_from_run_meta(meta,task)
    model=ActivityLearnableRiftDecoder(task,cfg,context_bins=int(meta["context_bins"]),seed=int(meta["seed"]),support_bins=int(meta["support_bins"]),identity_hidden=int(meta["identity_hidden"])).to(device); model.temporal.set_attention_backend("local")
    data=activity_data.load_task_data(task,include_eval=True); surface=data.get("evaluation",{})
    if not surface: raise RuntimeError(f"{task} evaluation surface is unavailable")
    common=dict(context=int(data["metadata"]["context"]),device=device,behavior_scale=float(data["metadata"].get("behavior_scale",5.0)),max_batches=args.max_batches)
    epochs=[int(receipt["selection"]["checkpoint"].split("_")[1].split(".")[0])] if smoke else list(range(1,fixed_epoch+1))
    curve={}
    for epoch in epochs:
        state=torch.load(run_dir/f"epoch_{epoch:03d}.pt",map_location=device,weights_only=False)
        if state.get("schema")!=train.CKPT_SCHEMA or state.get("task")!=task or int(state.get("epoch", -1))!=epoch or bool(state.get("smoke"))!=smoke: raise RuntimeError(f"epoch {epoch}: checkpoint contract mismatch")
        if int(state.get("ema",{}).get("n_updates", -1)) != int(state.get("global_step", -2)): raise RuntimeError(f"epoch {epoch}: EMA accounting mismatch")
        if task=="m2" and not smoke and int(state["global_step"]) != epoch*train.M2_UPDATES: raise RuntimeError(f"epoch {epoch}: M2 step mismatch")
        model.load_state_dict(state["raw_state_dict"],strict=True); _ema_into(model,state); model.eval()
        # Identity cache is deliberately scoped to this checkpoint only.
        report=train.score_surface(model,surface,**common,identities={},capture_arrays=(task=="h1"))
        if task=="h1":
            from btransform_unified_v1.c2_protocol import HELDOUT_SESSION_TO_FALCON_KEY, grouped_session_metrics
            def _row(session): return report["per_session"].get(session) or report["per_session"].get(session.removeprefix("ses-"))
            preds={key:_row(session).pop("_prediction") for session,key in HELDOUT_SESSION_TO_FALCON_KEY}
            targets={key:_row(session).pop("_target") for session,key in HELDOUT_SESSION_TO_FALCON_KEY}
            masks={key:np.ones(len(preds[key]),dtype=bool) for _session,key in HELDOUT_SESSION_TO_FALCON_KEY}
            grouped=grouped_session_metrics(preds,targets,masks,HELDOUT_SESSION_TO_FALCON_KEY); metric=float(grouped["r2_mean"]); report["grouped_seven"]=grouped
        else: metric=float(report["equal_session_mean"])
        if not np.isfinite(metric): raise FloatingPointError(f"epoch {epoch}: nonfinite selection metric")
        curve[str(epoch)]={**report,"selection_metric":metric,"checkpoint_sha256":train._sha(run_dir/f"epoch_{epoch:03d}.pt")}
    best=max(epochs,key=lambda e:(float(curve[str(e)]["selection_metric"]),-e))
    if args.save_predictions:
        # Explicit second scoring pass; it does re-calibrate, and the receipt says so.
        state=torch.load(run_dir/f"epoch_{best:03d}.pt",map_location=device,weights_only=False); model.load_state_dict(state["raw_state_dict"],strict=True); _ema_into(model,state); model.eval(); train.score_surface(model,surface,**common,identities={},save_predictions=args.dest/"predictions")
    out={"schema":SCHEMA,"status":"SMOKE_COMPLETED" if smoke else "COMPLETED","formal_claim":False,"partial":args.max_batches is not None,"metric_scope":"public_calibration_local_development","final_metric_source":"EvalAI submission result","official_test_used":False,"task":task,"run_dir":str(run_dir),"view":"EMA","selection":{"epoch":best,"rule":"smoke epoch only" if smoke else "earliest maximum FULL-compatible public-calibration EMA scan","metric":"grouped-seven r2_mean" if task=="h1" else "equal_session_mean variance-weighted r2"},"ema_by_epoch":curve,"calibration":{"per_target_session_once_per_checkpoint":True,"mode":"eval_no_grad","support":"activity only","additional_prediction_export_pass":bool(args.save_predictions)},"data_hashes":{s:v.get("hashes",{}) for s,v in surface.items()},"support_provenance":{s:v.get("support_provenance",{}) for s,v in surface.items()},"runtime_seconds":time.monotonic()-started,"utc":datetime.now(timezone.utc).isoformat()}
    _atom(args.dest.resolve()/"score_receipt.json",out); return out
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--run-dir",type=Path,required=True); p.add_argument("--dest",type=Path,required=True); p.add_argument("--device",default="cpu"); p.add_argument("--cpu-threads",type=int,default=2); p.add_argument("--max-batches",type=int); p.add_argument("--allow-smoke",action="store_true"); p.add_argument("--save-predictions",action="store_true"); a=p.parse_args()
    if a.max_batches is not None and a.max_batches<1: p.error("--max-batches must be positive")
    print(json.dumps(run(a),indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
