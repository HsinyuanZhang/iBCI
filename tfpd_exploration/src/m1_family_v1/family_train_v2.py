"""Executable, source-only chron80 family trainers; formal launch is external.

Both trainers make no outer/session-query loader.  The chronological tail is a
frozen *source-minival* surface: RAW and EMA are scored after every epoch and
the earliest highest EMA equal-session R2 is selected.  It is not a held-out
or official score.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tfpd_exploration.src.m1_optimized_v2 import bank
from tfpd_exploration.src.m1_optimized_v2 import plan
from tfpd_exploration.src.m1_optimized_v2.data import build_source_only_datamodule, materialize_source_banks
from tfpd_exploration.src.m1_optimized_v2.model import build as build_query
from tfpd_exploration.src.m1_optimized_v2.source_dev import _clone, _keep, _metrics, _split

ROOT = plan.RESULT_ROOT / "family_v1"
EPOCHS = 24
BATCH = 32
EMA_DECAY = 0.9995


def _name(x): return x.decode() if isinstance(x, bytes) else str(x)
def _sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def _json_sha(value) -> str:
    """Digest a declared topology without serializing it into the receipt."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
def _now() -> dict[str, object]:
    return {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "unix": time.time()}


def _p1_provenance(loaded, split, train_batches, dev_batches) -> dict[str, object]:
    """Exact file/code and realized-sampler identities for the launch receipt."""
    code_paths = [
        Path(__file__),
        Path(__file__).parents[1] / "m1_optimized_v2" / "model.py",
        Path(__file__).parents[1] / "m1_optimized_v2" / "source_dev.py",
        Path(__file__).parents[1] / "two_mainlines_long_v1" / "decoder" / "h1_temporal.py",
        Path(__file__).parents[1] / "two_mainlines_long_v1" / "current_query_v2" / "core.py",
    ]
    receipt = loaded["receipt"]
    return {
        "source_files": receipt.get("provenance", {}).get("source_files", {}),
        "bank_receipt_sha256": _sha(plan.BANK_RECEIPT),
        "bank_npz_sha256": receipt["digests"]["npz"],
        "modelcode_sha256": {str(path.relative_to(plan.REPO_ROOT)): _sha(path) for path in code_paths},
        "frozen_split_digest": _json_sha(split),
        "train_sampler_digest": _json_sha(train_batches),
        "dev_batch_digest": _json_sha(dev_batches),
        "actual_train_rows": sum(map(len, train_batches)),
        "dropped_train_rows": sum(int(row["train_windows"]) for row in split.values()) - sum(map(len, train_batches)),
        "actual_dev_rows": sum(map(len, dev_batches)),
    }


class EMA:
    """State-dict EMA that covers parameters and buffers deterministically."""
    def __init__(self, model): self.state = None
    @torch.no_grad()
    def update(self, model):
        if self.state is None:
            self.state={k:v.detach().clone() for k,v in model.state_dict().items()}; return
        for k, v in model.state_dict().items():
            old = self.state[k]
            if v.dtype.is_floating_point: old.mul_(EMA_DECAY).add_(v.detach(), alpha=1 - EMA_DECAY)
            else: old.copy_(v)
    def load(self, model):
        if self.state is None: raise RuntimeError("EMA before first successful step")
        model.load_state_dict(self.state, strict=True)


def _batches(dataset):
    """Fixed, session-pure order shared by both P1 arms."""
    groups = {name: [] for name in plan.SOURCE_SESSIONS}
    for i, (name, _start) in enumerate(dataset.base.window_indices): groups[_name(name)].append(i)
    result = [row[i:i+BATCH] for name in plan.SOURCE_SESSIONS for i in range(0, len(groups[name]), BATCH) for row in (groups[name],)]
    if sum(map(len, result)) != len(dataset): raise RuntimeError("batch order dropped source rows")
    return result


def _trusted_train_batches(dataset):
    """Exact source-dev sampler contract, materialized once for both P1 arms."""
    exp=str(plan.REPO_ROOT/"streaming_calibration_exp")
    if exp not in sys.path: sys.path.insert(0,exp)
    from src.data.falcon_datamodule import SessionBatchSampler
    sampler=SessionBatchSampler(dataset,BATCH,shuffle=True,seed=42,balance_sessions=False,reshuffle_each_epoch=False)
    batches=[list(x) for x in sampler]
    # This is deliberately the legacy/source-dev sampler's own drop-last
    # behavior (122688 of 122718 chron80 rows), not a replacement ordering.
    if not batches or any(len(x)!=BATCH for x in batches): raise RuntimeError("trusted sampler topology")
    return batches


def _trusted_adamw(model):
    decay=[p for n,p in model.named_parameters() if p.requires_grad and p.ndim>1]
    no_decay=[p for n,p in model.named_parameters() if p.requires_grad and p.ndim<=1]
    return torch.optim.AdamW([{"params":decay,"weight_decay":.01},{"params":no_decay,"weight_decay":0.}],lr=1e-4)


def _source_sets():
    loaded = bank.load(); dm = build_source_only_datamodule(loaded)
    if getattr(dm, "target_path", None) is not None or dm.val_heldin_dataset is not None: raise RuntimeError("outer target materialized")
    train_rows, dev_rows, split_rows = _split(dm)
    if set(train_rows) & set(dev_rows): raise RuntimeError("chron80 overlap")
    return loaded, _clone(dm.train_dataset, train_rows, loaded), _clone(dm.train_dataset, dev_rows, loaded), split_rows


def _p1_models(device):
    """Pair construction: FLAT-equivalent g=0 but a live ROUTE tangent.

    The legacy constructor calls ``initialize_route_only`` after copying FLAT:
    q_cal and route projections are independently reinitialized while g stays
    zero.  P1 preserves that trusted initialization exactly.
    """
    torch.manual_seed(42); flat=build_query("flat").to(device); route=build_query("route").to(device)
    # Do not rely on independent shell construction consuming identical RNG;
    # explicitly map every shared tensor from the one FLAT initialization.
    mapped=route.state_dict()
    for key,value in flat.state_dict().items():
        if key in mapped: mapped[key]=value.detach().clone()
    route.load_state_dict(mapped,strict=True)
    if not torch.equal(route.frontend.attn.g,torch.zeros_like(route.frontend.attn.g)): raise RuntimeError("P1 g drift")
    return flat, route


def _score(model, dataset, batches, device, *, b3=False, original_spint=False):
    model.eval(); pred=[]; y=[]; names=[]
    banks = None if b3 else {n: type(v)(E0=v.E0.to(device), T=v.T.to(device), unit_mask=v.unit_mask.to(device)) for n,v in materialize_source_banks().items()}
    with torch.inference_mode():
        for ids in batches:
            x, target, calibration, sessions, carrier = next(iter(DataLoader(dataset, batch_sampler=[ids])))
            name = _name(sessions[0])
            if any(_name(s) != name for s in sessions): raise RuntimeError("non-pure batch")
            if b3: out, _ = model(x.to(device).float(), calib_trials=calibration.to(device).float(), side_features=None, carrier=carrier.to(device).float())
            elif original_spint: out = model(x.to(device).float(), calib_trialized_neural_features=calibration.to(device).float())
            else: out = model.forward_last(x.to(device).float(), banks[name])
            pred.append(out[:, -1, :].cpu().numpy() if (b3 or original_spint) else out.cpu().numpy()); y.append(target[:, -1, :].numpy()); names += [name] * len(ids)
    p=np.concatenate(pred); target=np.concatenate(y); train_mean=np.zeros_like(target)
    per={}
    for n in plan.SOURCE_SESSIONS:
        take=np.asarray([x==n for x in names])
        if take.any(): per[n]=_metrics(p[take],target[take],train_mean)
    pooled=_metrics(p,target,train_mean)
    # No train mean was supplied to this scorer.  Never mislabel all-zero as
    # a train-only baseline; retain model/zero only.
    for metric in [pooled,*per.values()]:
        for group in ("r2","mse"): metric[group].pop("train_only_source_mean",None)
    return {"equal_session_mean_r2":float(np.mean([v["r2"]["model"] for v in per.values()])),"pooled":pooled,"per_session":per,"n":len(target),"complete_all_sessions":len(per)==len(plan.SOURCE_SESSIONS)}


def _ema_score(model, ema, *args, **kw):
    raw=copy.deepcopy(model.state_dict()); ema.load(model)
    try: return _score(model,*args,**kw)
    finally: model.load_state_dict(raw,strict=True)


def _record_epoch(model, ema, dev, dev_batches, device, *, b3, epoch, root, opt, recipe):
    raw=_score(model,dev,dev_batches,device,b3=b3); ema_m=_ema_score(model,ema,dev,dev_batches,device,b3=b3)
    root.mkdir(parents=True, exist_ok=True)
    path=root/f"epoch_{epoch:03d}.pt"
    if path.exists(): raise FileExistsError(path)
    torch.save({"schema":"m1_family_v1_chron80_checkpoint_v2","epoch":epoch,"model":model.state_dict(),"ema":ema.state,"optimizer":opt.state_dict(),"recipe":recipe,"outer_query_opened":False},path)
    # Immediate strict reload smoke guards serialization before the next epoch.
    blob=torch.load(path,map_location="cpu",weights_only=False)
    if blob["epoch"] != epoch or set(blob["model"]) != set(model.state_dict()): raise RuntimeError("checkpoint reload drift")
    return {"epoch":epoch,"checkpoint":str(path),"checkpoint_sha256":_sha(path),"raw":raw,"ema":ema_m}


def _select(history):
    best=max(x["ema"]["equal_session_mean_r2"] for x in history)
    return next(x for x in history if x["ema"]["equal_session_mean_r2"] == best)


def train_p0(*, device="cuda:0", epochs=EPOCHS):
    """Fresh Original SpintModel baseline; never loads teacher/B3/S-Fix weights."""
    torch.manual_seed(42); exp=str(plan.REPO_ROOT/"streaming_calibration_exp"); sys.path.insert(0,exp) if exp not in sys.path else None
    from src.models.components.spint import SpintModel
    _loaded, train, dev, split = _source_sets(); root=ROOT/"p0_original_spint_fresh_chron80_v2"; root.mkdir(parents=True,exist_ok=False)
    # Read-only teacher config established these topology literals; fresh
    # constructor is intentionally never load_state_dict'ed from that file.
    model=SpintModel(model_dim=1024,num_covariates=16,window_size=100,num_heads=8,num_layers=1,num_id_layers=3,use_learnable_id=True,learnable_id_type="mlp",learnable_rep=True,dropout_rate=0.,dynamic_dropout=False,tf_drop_rate=.1,readin_layer_type="mlp").to(device)
    recipe={"arm":"P0_OriginalSPINT_fresh","seed":42,"epochs":epochs,"optimizer":"Adam","lr":1e-5,"weight_decay":0.,"selection":"EMA equal-session source-minival highest; earliest tie","native_divisor":1,"teacher_config_read_only":True,"teacher_weights_loaded":False,"b3_or_rsyn3_projection":False}
    opt=torch.optim.Adam(model.parameters(),lr=1e-5,weight_decay=0.); ema=EMA(model)
    history=[]; train_batches=_trusted_train_batches(train); dev_batches=_batches(dev)
    (root/"run_meta.json").write_text(json.dumps({"recipe":recipe,"split":split,"bank_sha256":_loaded["receipt"]["digests"]["npz"],"status":"RUNNING","outer_query_opened":False},indent=2,sort_keys=True)+"\n")
    for epoch in range(1,epochs+1):
        model.train(); losses=[]
        for ids in train_batches:
            x,y,c,_s,_carrier=next(iter(DataLoader(train,batch_sampler=[ids]))); out=model(x.to(device).float(),calib_trialized_neural_features=c.to(device).float()); loss=F.mse_loss(out[:,-1,:],y[:,-1,:].to(device).float())
            if out.shape[-1]!=16 or y.shape[-1]!=16 or not torch.isfinite(loss): raise RuntimeError("P0 native loss contract")
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); ema.update(model); losses.append(float(loss.detach()))
            if len(losses)%20==0: (root/"heartbeat.json").write_text(json.dumps({"epoch":epoch,"step":len(losses),"loss":losses[-1]},sort_keys=True)+"\n")
        raw=_score(model,dev,dev_batches,device,original_spint=True); em=_ema_score(model,ema,dev,dev_batches,device,original_spint=True)
        path=root/f"epoch_{epoch:03d}.pt"; torch.save({"epoch":epoch,"model":model.state_dict(),"ema":ema.state,"optimizer":opt.state_dict(),"recipe":recipe},path); blob=torch.load(path,map_location="cpu",weights_only=False)
        if set(blob["model"])!=set(model.state_dict()): raise RuntimeError("P0 checkpoint reload")
        history.append({"epoch":epoch,"checkpoint":str(path),"checkpoint_sha256":_sha(path),"raw":raw,"ema":em,"train_native_mse":float(np.mean(losses))})
        (root/"epoch_metrics.json").write_text(json.dumps(history,indent=2,sort_keys=True)+"\n")
    report={"schema":"m1_family_v1_p0_original_spint_fresh_chron80_v2","status":"SOURCE_MINIVAL_ONLY","split":split,"history":history,"selected":_select(history),"endpoint24":history[-1],"outer_query_opened":False,"recipe":recipe}
    (root/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); return report


def paired_gradient_smoke():
    """Real paired optimizer/checkpoint/score smoke without persisting a run."""
    _loaded, train, dev, _ = _source_sets(); device=torch.device("cpu"); flat,route=_p1_models(device); ids=_batches(train)[0]
    x,y,_c,s,_carrier=next(iter(DataLoader(train,batch_sampler=[ids]))); name=_name(s[0]); banks={n:type(v)(E0=v.E0,T=v.T,unit_mask=v.unit_mask) for n,v in materialize_source_banks().items()}; keep=_keep(len(x),1,0,device)
    optf=torch.optim.AdamW(flat.parameters(),lr=1e-4); optr=torch.optim.AdamW(route.parameters(),lr=1e-4)
    losses=[]
    for model,opt in ((flat,optf),(route,optr)):
        out=model.forward_last(x.float(),banks[name],dropout_keep=keep); loss=F.mse_loss(out,y[:,-1,:]); opt.zero_grad(); loss.backward(); losses.append(float(loss)); opt.step()
    grad=float(route.frontend.attn.g.grad.abs().max())
    if not grad>0: raise RuntimeError("ROUTE gate has zero gradient")
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/"paired.pt"; torch.save({"flat":flat.state_dict(),"route":route.state_dict(),"optf":optf.state_dict(),"optr":optr.state_dict()},p); blob=torch.load(p,map_location="cpu",weights_only=False)
        if set(blob["flat"]) != set(flat.state_dict()): raise RuntimeError("paired checkpoint reload")
    score=_score(flat,dev,_batches(dev),device,b3=False)
    return {"route_gate_gradient_absmax":grad,"losses":losses,"dev_raw_score_smoke":score,"outer_query_opened":False}


def p1_final_short_smoke(device="cuda:0"):
    """Bounded GPU readiness gate: two trusted steps, two dev batches only."""
    started=time.monotonic(); _loaded,train,dev,split=_source_sets(); dev_t=torch.device(device); flat,route=_p1_models(dev_t)
    tb=_trusted_train_batches(train); db=_batches(dev)[:2]; banks={n:type(v)(E0=v.E0.to(dev_t),T=v.T.to(dev_t),unit_mask=v.unit_mask.to(dev_t)) for n,v in materialize_source_banks().items()}
    opts={"flat":_trusted_adamw(flat),"route":_trusted_adamw(route)}; models={"flat":flat,"route":route}; emas={k:EMA(v) for k,v in models.items()}; lrs={k:[] for k in models}; losses={}; route_grad=0.
    for batch_id,ids in enumerate(tb[:2]):
        x,y,_c,s,_carrier=next(iter(DataLoader(train,batch_sampler=[ids]))); name=_name(s[0]); keep=_keep(len(x),1,batch_id,dev_t)
        for arm in ("flat","route"):
            m,o=models[arm],opts[arm]; lr=1e-4*((batch_id+1)/len(tb)); [g.update(lr=lr) for g in o.param_groups]; lrs[arm].append(lr)
            out=m.forward_last(x.to(dev_t).float(),banks[name],dropout_keep=keep); loss=F.mse_loss(out,y[:,-1,:].to(dev_t).float()); o.zero_grad(set_to_none=True); loss.backward()
            if arm=="route": route_grad=max(route_grad,float(m.frontend.attn.g.grad.abs().max()))
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.); o.step(); emas[arm].update(m); losses.setdefault(arm,[]).append(float(loss.detach()))
    if lrs["flat"]!=lrs["route"] or not route_grad>0: raise RuntimeError("P1 paired LR or route gradient gate")
    raw={a:_score(models[a],dev,db,dev_t,b3=False) for a in models}; ema={a:_ema_score(models[a],emas[a],dev,db,dev_t,b3=False) for a in models}
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/"p1_smoke.pt"; torch.save({"flat":flat.state_dict(),"route":route.state_dict(),"ema_flat":emas["flat"].state,"ema_route":emas["route"].state,"opt_flat":opts["flat"].state_dict(),"opt_route":opts["route"].state_dict()},p); z=torch.load(p,map_location="cpu",weights_only=False)
        if set(z["flat"])!=set(flat.state_dict()) or set(z["route"])!=set(route.state_dict()): raise RuntimeError("P1 serialization")
    out={"schema":"m1_family_v1_p1_final_short_smoke_v1","device":str(dev_t),"wall_seconds":time.monotonic()-started,"steps_per_arm":2,"lrs":lrs,"shared_dropout":"_keep(batch,epoch=1,batch_id) used identically","route_gate_gradient_absmax":route_grad,"losses":losses,"raw":raw,"ema":ema,"serialization_roundtrip":True,"split":split,"bank_sha256":_loaded["receipt"]["digests"]["npz"],"outer_query_opened":False,"formal_24ep_started":False}
    p=ROOT/"p1_final_short_smoke_v1.json"; p.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); return out


def train_p1(*, device="cuda:0", epochs=EPOCHS, deadline_seconds=6 * 60 * 60):
    """Synchronized new QueryAge16 FLAT/ROUTE pair; call only after launch GO."""
    _loaded, train, dev, split = _source_sets(); root=ROOT/"p1_queryage16_pair_chron80_v2"; root.mkdir(parents=True,exist_ok=False)
    flat,route=_p1_models(device)
    shared=[k for k in flat.state_dict() if k in route.state_dict()]
    if any(not torch.equal(flat.state_dict()[k],route.state_dict()[k]) for k in shared): raise RuntimeError("P1 initial shared state drift")
    of=_trusted_adamw(flat); oroute=_trusted_adamw(route); ef=EMA(flat); er=EMA(route)
    banks={n:type(v)(E0=v.E0.to(device),T=v.T.to(device),unit_mask=v.unit_mask.to(device)) for n,v in materialize_source_banks().items()}
    recipe={"pair":"P1 QueryAge16 FLAT/ROUTE","seed":42,"epochs":epochs,"optimizer":"AdamW","lr":1e-4,"weight_decay":.01,"bias_norm_weight_decay":0.,"warmup_epochs":1,"clip":1.,"sampler":"SessionBatchSampler(seed42, shuffle=True, balance=False, reshuffle_each_epoch=False)","dropout":"shared _keep(seed,epoch,batch_id) [B,64]","selection_primary":"EMA equal-session source-minival highest; earliest tie","selection_secondary":"raw scores reported every epoch; not used to select the primary checkpoint","native_divisor":1,"deadline_seconds":deadline_seconds}
    history=[]; tb=_trusted_train_batches(train); db=_batches(dev); steps={"flat":0,"route":0}
    provenance=_p1_provenance(_loaded, split, tb, db)
    if provenance["actual_train_rows"] != 122688 or provenance["dropped_train_rows"] != 30 or provenance["actual_dev_rows"] != 31252:
        raise RuntimeError(f"unexpected frozen sampler realization: {provenance}")
    started_mono=time.monotonic(); deadline_mono=started_mono + deadline_seconds
    meta={"schema":"m1_family_v1_p1_run_meta_v2","recipe":recipe,"split":split,"provenance":provenance,"status":"RUNNING","started":_now(),"outer_query_opened":False}
    (root/"run_meta.json").write_text(json.dumps(meta,indent=2,sort_keys=True)+"\n")
    def heartbeat(epoch, arm, loss):
        (root/"heartbeat.json").write_text(json.dumps({"epoch":epoch,"arm":arm,"step":steps[arm],"loss":loss,**_now()},sort_keys=True)+"\n")
    def timeout(epoch, arm, batch_id):
        meta.update({"status":"TIME_BUDGET_INCOMPLETE","deadline":_now(),"deadline_epoch":epoch,"deadline_arm":arm,"deadline_batch_id":batch_id,"completed_epochs":len(history)})
        (root/"run_meta.json").write_text(json.dumps(meta,indent=2,sort_keys=True)+"\n")
        (root/"TIME_BUDGET_INCOMPLETE.json").write_text(json.dumps({"reason":"hard six-hour training budget elapsed; completed checkpoints preserved","epoch":epoch,"arm":arm,"batch_id":batch_id,"completed_epochs":len(history),**_now()},indent=2,sort_keys=True)+"\n")
        raise TimeoutError("TIME_BUDGET_INCOMPLETE")
    for epoch in range(1,epochs+1):
        rows={}; route_grad=0.
        for arm,model,opt,ema in (("flat",flat,of,ef),("route",route,oroute,er)):
            model.train(); losses=[]
            for batch_id,ids in enumerate(tb):
                if time.monotonic() >= deadline_mono: timeout(epoch,arm,batch_id)
                x,y,_c,s,_carrier=next(iter(DataLoader(train,batch_sampler=[ids]))); name=_name(s[0]); keep=_keep(len(x),epoch,batch_id,torch.device(device)); out=model.forward_last(x.to(device).float(),banks[name],dropout_keep=keep); loss=F.mse_loss(out,y[:,-1,:].to(device).float())
                if out.shape[-1]!=16 or not torch.isfinite(loss): raise RuntimeError("P1 native loss contract")
                steps[arm]+=1
                for group in opt.param_groups: group["lr"]=1e-4*min(1.,steps[arm]/max(1,len(tb)))
                opt.zero_grad(set_to_none=True); loss.backward()
                if arm=="route": route_grad=max(route_grad,float(model.frontend.attn.g.grad.abs().max()))
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); ema.update(model); losses.append(float(loss.detach()))
                if steps[arm] % 20 == 0: heartbeat(epoch,arm,losses[-1])
            rows[arm]=_record_epoch(model,ema,dev,db,device,b3=False,epoch=epoch,root=root/f"{arm}",opt=opt,recipe=recipe); rows[arm]["train_native_mse"]=float(np.mean(losses))
        if not route_grad>0: raise RuntimeError("P1 routing gate had zero epoch gradient")
        history.append({"epoch":epoch,"route_gate_gradient_absmax":route_grad,**rows})
        (root/"epoch_metrics.json").write_text(json.dumps(history,indent=2,sort_keys=True)+"\n")
    def pick(arm):
        best=max(r[arm]["ema"]["equal_session_mean_r2"] for r in history); return next(r[arm] for r in history if r[arm]["ema"]["equal_session_mean_r2"]==best)
    meta.update({"status":"COMPLETE","completed":_now(),"completed_epochs":len(history)})
    (root/"run_meta.json").write_text(json.dumps(meta,indent=2,sort_keys=True)+"\n")
    report={"schema":"m1_family_v1_p1_queryage16_pair_chron80_v2","status":"SOURCE_MINIVAL_ONLY","split":split,"recipe":recipe,"provenance":provenance,"history":history,"selected_primary_ema":{"flat":pick("flat"),"route":pick("route")},"endpoint24":{"flat":history[-1]["flat"],"route":history[-1]["route"]},"outer_query_opened":False}
    (root/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--smoke",action="store_true"); parser.add_argument("--p1-final-smoke",action="store_true"); parser.add_argument("--train-p0",action="store_true"); parser.add_argument("--train-p1",action="store_true"); parser.add_argument("--device",default="cuda:0"); args=parser.parse_args()
    if args.smoke: print(json.dumps(paired_gradient_smoke(),indent=2,sort_keys=True))
    elif args.p1_final_smoke: print(json.dumps(p1_final_short_smoke(args.device),indent=2,sort_keys=True))
    elif args.train_p0: print(json.dumps(train_p0(device=args.device),indent=2,sort_keys=True))
    elif args.train_p1: print(json.dumps(train_p1(device=args.device),indent=2,sort_keys=True))
    else: parser.error("explicit --smoke or --train-p0 required")
