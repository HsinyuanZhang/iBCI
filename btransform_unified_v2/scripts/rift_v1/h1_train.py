"""H1-RIFT-D4-R{200,300} trainer (C2-CAL-1 B2 data protocol, CPU/GPU explicit).

This is deliberately a new result family.  It consumes raw end-anchored H1
contexts plus ``input_valid_mask``; it does not reuse V1's left-zero-padded
L200 tensors as temporal evidence.
"""
from __future__ import annotations

import argparse, contextlib, dataclasses, hashlib, json, math, os, random, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT.parent / "btransform_unified_v1"
for p in (ROOT / "src", V1 / "src", V1 / "scripts", ROOT.parent, V1):
    if str(p) not in sys.path: sys.path.insert(0, str(p))

from btransform_unified_v1 import adapters, cal1_b2, h1_config, plan
from btransform_unified_v1.bank import TaskBank, array_sha256
from btransform_unified_v1.c2_protocol import (C2_CYCLE, HELDOUT_SESSION_TO_FALCON_KEY,
    HO_SELECTION_METRIC, grouped_session_metrics, pick_m7_start, prefix_schedule, select_epoch)
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.model import unit_dropout_seed, whole_unit_dropout
from btransform_unified_v1.schedule import warmup_cosine_lr
import h1_c2_cal1_b2_l200_p16 as b2

CONTEXT = 300; EPOCHS = 32; SEED = 42; BATCH = 32; SCALE = h1_config.TARGET_MULTIPLIER
LAYERS = (75, 75, 75, 74); UPDATES_PER_EPOCH = 731

def layer_bins(context_bins: int) -> tuple[int, int, int, int]:
    if int(context_bins) == 200: return (50, 50, 50, 49)
    if int(context_bins) == 300: return LAYERS
    raise ValueError("H1 RIFT context_bins must be 200 or 300")

def _decoder(variant: str, device: torch.device, context_bins: int = CONTEXT, attention_backend: str = "local"):
    # Imported lazily: the frontend/temporal owners may be editing this module.
    from btransform_unified_v2.model import RiftDecoder
    model=RiftDecoder(task="h1", context_bins=context_bins, bias_mode=variant, seed=SEED, proj_dim=16).to(device)
    model.temporal.set_attention_backend(attention_backend)
    return model

def _sha_state(model: nn.Module) -> str:
    """Trainable initialization only: recency's fixed slope buffers are excluded."""
    h = hashlib.sha256()
    for n, v in sorted(model.named_parameters()): h.update(n.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def source_manifest() -> dict[str, str]:
    paths=[Path(__file__), ROOT/"src"/"btransform_unified_v2"/"model.py",ROOT/"src"/"btransform_unified_v2"/"temporal.py",ROOT/"src"/"btransform_unified_v2"/"config.py"]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

def _digest_train_contract(train: dict[str, Any], cal1: dict[str, Any]) -> dict[str, str]:
    endpoints=hashlib.sha256(); masks=hashlib.sha256(); banks=hashlib.sha256()
    for session in train["sessions"]:
        endpoints.update(session.encode()); endpoints.update(np.asarray(train["ids"][session],np.int64).tobytes())
        masks.update(session.encode()); masks.update(np.asarray(train["valid"][session],np.bool_).tobytes())
    for (session,start,budget), bank in sorted(cal1["banks"].items()):
        banks.update(f"{session}|{start}|{budget}|".encode()); banks.update(str(bank.calibration_meta.get("array_sha256")).encode()); banks.update(str(bank.calibration_meta.get("carrier_sha256")).encode())
    return {"endpoint_inventory_sha256":endpoints.hexdigest(),"valid_mask_inventory_sha256":masks.hexdigest(),"bank_roster_sha256":banks.hexdigest()}

def endpoint_context(neural: np.ndarray, ends: np.ndarray, context: int = CONTEXT) -> tuple[np.ndarray, np.ndarray]:
    """Return [N,R,U] and valid mask; zero cells are never valid KV evidence."""
    out = np.zeros((len(ends), context, neural.shape[1]), np.float32)
    valid = np.zeros((len(ends), context), np.bool_)
    for row, end0 in enumerate(np.asarray(ends, np.int64)):
        end = int(end0); lo = max(0, end - context + 1); n = end - lo + 1
        if n > 0:
            out[row, context-n:] = neural[lo:end+1]
            valid[row, context-n:] = True
    return out, valid

def _source_endpoints(surface: str, session: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row = adapters._h1_source_cache()[surface][session]
    neural = np.ascontiguousarray(row["neural"], dtype=np.float32)
    if surface == "train":
        ends = np.asarray(row["query_starts"], np.int64) + h1_config.FULL_WINDOW - 1
        mask = np.asarray(row["eval_mask"], bool); ends = ends[(ends >= 0) & (ends < len(mask)) & mask[ends]]
    else:
        ends = np.flatnonzero(np.asarray(row["eval_mask"], bool)).astype(np.int64)
    return neural, ends, np.ascontiguousarray(row["velocity"], dtype=np.float32)[ends]

def build_train(context_bins: int = CONTEXT) -> dict[str, Any]:
    """Reuse B2's C2-CAL-1 bank builder; only replace old L200 input tensors."""
    # Do not call B2 build_train_only: retaining its [N,200,176] tensors beside
    # R300 would add several GiB with no RIFT consumer.  _rebank is the exact
    # same frozen B2 source/bank route; its temporary L200 tensor is discarded
    # one session at a time and every retained bank aliases its R300 context.
    sessions=list(h1_config.H1_ALL_SESSIONS); banks={}; xs: dict[str,np.ndarray] = {}; valid: dict[str,np.ndarray] = {}; ys = {}; ids = {}
    b2.s2.PROJ_DIM=16; b2.s2.IDENTITY_MODE="proj_add"; b2.s2.WINDOW=200
    for s in sessions:
        base_bank, _ = b2.s2._rebank("train", s, 200)
        neural, ends, y = _source_endpoints("train", s)
        x, m = endpoint_context(neural, ends, context_bins)
        if not np.array_equal(ends, base_bank.window_ids): raise RuntimeError(f"B2 endpoint drift {s}")
        banks[s]=dataclasses.replace(base_bank, X_store=x, target_store=y, window_ids=ends)
        xs[s], valid[s], ys[s], ids[s] = x, m, y, ends
    updates = sum(math.ceil(len(xs[s])/BATCH) for s in sessions)
    if updates != UPDATES_PER_EPOCH: raise RuntimeError(f"R{context_bins} endpoint updates {updates} != B2 formal {UPDATES_PER_EPOCH}")
    return {"context_bins":context_bins,"sessions":sessions,"banks":banks,"X":xs,"valid":valid,"y":ys,"ids":ids,"updates_per_epoch":updates}

def build_ho(context_bins: int = CONTEXT) -> dict[str, Any]:
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb
    banks={}; xs={}; vm={}; ys={}; keys=[]
    for session, key in HELDOUT_SESSION_TO_FALCON_KEY:
        path=b2.HO_DIR / f"sub-HumanPitt-held-out-calib_{session}.nwb"
        neural, velocity, _change, eval_mask=load_nwb(path, FalconTask.h1)
        ends=np.flatnonzero(np.asarray(eval_mask, bool)).astype(np.int64); x,m=endpoint_context(np.asarray(neural,np.float32), ends, context_bins)
        activity, carrier=adapters._h1_payload_arrays(session); e0,hc=b2._materialize_e0(activity,carrier)
        banks[key]=TaskBank(session, e0,hc,np.ones(e0.shape[0],bool),x,np.asarray(velocity,np.float32)[ends],ends,{"shape":tuple(e0.shape),"trial_count":3,"budget":3,"estimator":"C2 HO-M3 payload + C2 materializer","array_sha256":array_sha256(e0)})
        xs[key],vm[key],ys[key]=x,m,banks[key].target_store; keys.append(key)
    return {"context_bins":context_bins,"banks":banks,"X":xs,"valid":vm,"y":ys,"keys":keys}

def _forward(model, xb, bank, keep, valid):
    return model(xb, bank, dropout_keep=keep, input_valid_mask=valid)

def score_ho_m3(model, ema, ho, device) -> dict[str,Any]:
    named=dict(model.named_parameters()); saved={n:p.detach().clone() for n,p in named.items()}; was=model.training
    try:
        ema.apply_to(model); model.eval(); preds={}; targets={}; masks={}
        for key in ho["keys"]:
            chunks=[]
            for off in range(0,len(ho["X"][key]),32):
                xb=torch.from_numpy(ho["X"][key][off:off+32]).to(device); valid=torch.from_numpy(ho["valid"][key][off:off+32]).to(device)
                with torch.inference_mode(): chunks.append(_forward(model,xb,ho["banks"][key],None,valid).cpu().numpy()/SCALE)
            preds[key]=np.concatenate(chunks); targets[key]=ho["y"][key]; masks[key]=np.ones(len(preds[key]),bool)
        return grouped_session_metrics(preds,targets,masks,HELDOUT_SESSION_TO_FALCON_KEY)
    finally:
        with torch.no_grad():
            for n,p in named.items(): p.copy_(saved[n])
        model.train(was)

def score_stage(args) -> dict[str, Any]:
    """Restartable post-training HO-M3 sweep; never retrains or opens official test."""
    dest=args.dest.resolve(); meta=json.loads((dest/"run_meta.json").read_text())
    if meta.get("status") != "FORMAL" or meta.get("variant") != args.variant or int(meta.get("context_bins",0)) != args.context_bins: raise RuntimeError("dest is not a matching formal RIFT run")
    # R300 runs produced before the local backend existed were dense; preserve
    # their exact execution choice when their sealed metadata has no field.
    backend=meta.get("attention_backend", "dense")
    device=torch.device(args.device); model=_decoder(args.variant,device,args.context_bins,backend); ema=DecoderEMA(model,decay=.9995); ho=build_ho(args.context_bins); curve=[]
    for ep in range(1,int(meta["epochs"])+1):
        path=dest/f"epoch_{ep:03d}.pt"
        if not path.is_file(): raise FileNotFoundError(f"missing checkpoint for score stage: {path}")
        state=torch.load(path,map_location=device,weights_only=False)
        if state.get("smoke"): raise RuntimeError(f"smoke checkpoint cannot be scored: {path}")
        model.load_state_dict(state["raw_state_dict"]); ema.load_state_dict(state["ema"]); report=score_ho_m3(model,ema,ho,device)
        curve.append({"epoch":ep,"epoch_zero_based":ep-1,HO_SELECTION_METRIC:report["r2_mean"],"worst_session_r2":report["worst_session_r2"],"session_std_population":report["r2_std_population"],"per_session_r2":report["per_session_r2"]})
    selected=select_epoch(curve); (dest/"ho_m3_selection.json").write_text(json.dumps({"status":"HO_M3_DEVELOPMENT_SELECTION","selected":selected,"curve":curve},indent=2,sort_keys=True)+"\n")
    (dest/"train_receipt.json").write_text(json.dumps({"status":"COMPLETED","updates":int(meta["epochs"])*UPDATES_PER_EPOCH,"epochs":int(meta["epochs"]),"selected_epoch":selected["epoch"],"official_test_used":False},indent=2)+"\n")
    return {"status":"SCORE_COMPLETED","selected_epoch":selected["epoch"]}

def _checkpoint(path, model, optimizer, ema, epoch, step, rng, *, variant, context_bins, attention_backend, microbatch, smoke):
    if path.exists(): raise FileExistsError(f"refusing to overwrite checkpoint {path}")
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    state={"schema":"rift_h1_context_train_v1","variant":variant,"context_bins":context_bins,"attention_backend":attention_backend,"microbatch":microbatch,"smoke":smoke,"epoch":epoch,"global_step":step,"raw_state_dict":model.state_dict(),"optimizer":optimizer.state_dict(),"ema":ema.state_dict(),"rng":rng.bit_generator.state,"torch_rng_cpu":torch.get_rng_state(),"torch_rng_cuda":cuda_rng,"numpy_rng":np.random.get_state(),"python_rng":random.getstate()}
    tmp=path.with_suffix(path.suffix+".tmp"); torch.save(state,tmp); tmp.replace(path)

def run(args) -> dict[str,Any]:
    device=torch.device(args.device); torch.set_num_threads(2); torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    dest=args.dest.resolve()
    if args.resume:
        if not dest.is_dir() or not (dest/"run_meta.json").is_file(): raise RuntimeError("--resume requires its existing formal run --dest")
        existing=json.loads((dest/"run_meta.json").read_text())
        if existing.get("status") != "FORMAL" or existing.get("variant") != args.variant or int(existing.get("context_bins",0)) != args.context_bins or existing.get("attention_backend", "dense") != args.attention_backend or int(existing.get("microbatch",0)) != args.microbatch: raise RuntimeError("--resume run metadata mismatch")
    elif dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"refusing nonempty non-resume result directory {dest}")
    train=build_train(args.context_bins); cal1=b2.build_cal1_banks(train["banks"]); contract_digests=_digest_train_contract(train,cal1)
    model=_decoder(args.variant,device,args.context_bins,args.attention_backend); init_sha=_sha_state(model); ema=DecoderEMA(model,decay=.9995)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8); rng=np.random.default_rng(SEED)
    dest.mkdir(parents=True,exist_ok=True)
    meta={"schema":"rift_h1_context_train_v1","status":"SMOKE" if args.max_updates_smoke else "FORMAL","variant":args.variant,"context_bins":args.context_bins,"attention_backend":args.attention_backend,"layer_bins":list(layer_bins(args.context_bins)),"raw_span":f"x[a-{args.context_bins-1}:a] with input_valid_mask","train_sessions":train["sessions"],"updates_per_epoch":731,"epochs":args.epochs,"batch":32,"microbatch":args.microbatch,"precision":"bf16 autocast on CUDA; fp32 CPU","peak_lr":1e-4,"floor_lr":1e-5,"warmup_epochs":1,"ema":.9995,"unit_dropout":.1,"seed":42,"initialization_sha256":init_sha,"cal1":"V1 build_cal1_banks C2-CAL-1 B2","official_test_used":False,"cold_start":"raw pre-session padding invalid via input_valid_mask","pairing_digests":contract_digests,"source_manifest_sha256":source_manifest(),"launch":{"argv":sys.argv,"pid":os.getpid(),"cuda_visible_devices":os.environ.get('CUDA_VISIBLE_DEVICES'),"python_no_user_site":os.environ.get('PYTHONNOUSERSITE')},"utc":datetime.now(timezone.utc).isoformat()}
    if not args.resume: (dest/"run_meta.json").write_text(json.dumps(meta,indent=2,sort_keys=True)+"\n")
    total=train["updates_per_epoch"]*args.epochs; warm=train["updates_per_epoch"]; step=0; curve=[]; first_epoch=1
    if args.resume:
        state=torch.load(args.resume, map_location=device, weights_only=False)
        if state.get("schema") not in ("rift_h1_r300_train_v1", "rift_h1_context_train_v1") or state.get("smoke"): raise RuntimeError("resume requires a formal H1-RIFT epoch checkpoint")
        if state.get("variant") != args.variant or state.get("context_bins") != args.context_bins or state.get("attention_backend", "dense") != args.attention_backend or state.get("microbatch") != args.microbatch: raise RuntimeError("resume variant/context/backend/microbatch mismatch")
        model.load_state_dict(state["raw_state_dict"]); opt.load_state_dict(state["optimizer"]); ema.load_state_dict(state["ema"])
        rng.bit_generator.state=state["rng"]; torch.set_rng_state(state["torch_rng_cpu"].cpu()); np.random.set_state(state["numpy_rng"]); random.setstate(state["python_rng"])
        if device.type == "cuda" and state.get("torch_rng_cuda") is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_rng_cuda"]])
        step=int(state["global_step"]); first_epoch=int(state["epoch"])+1
        if first_epoch > args.epochs: raise RuntimeError("resume checkpoint already reaches requested --epochs")
    started=time.monotonic()
    for ep in range(first_epoch,args.epochs+1):
        model.train(); losses=[]; schedule=prefix_schedule(ep-1,train["updates_per_epoch"]); order=list(train["sessions"]); rng.shuffle(order); si=0; pairing=hashlib.sha256()
        for s in order:
            idx=rng.permutation(len(train["X"][s]))
            for off in range(0,len(idx),BATCH):
                take=idx[off:off+BATCH]; budget=int(schedule[si]); starts=cal1_b2.legal_starts(cal1["starts"][s],cal1["n_trials"][s],budget); bank=cal1["banks"][(s,pick_m7_start(s,epoch0=ep-1,step=si,starts=starts),budget)]
                pairing.update(s.encode()); pairing.update(np.asarray(train["ids"][s][take],np.int64).tobytes()); pairing.update(str(bank.calibration_meta.get("array_sha256")).encode())
                step+=1; lr=warmup_cosine_lr(step,total,warm,peak=1e-4,min_factor=.1); opt.param_groups[0]["lr"]=lr
                keep=whole_unit_dropout(torch.from_numpy(bank.unit_mask.copy()),p=.1,generator=torch.Generator().manual_seed(unit_dropout_seed(SEED,ep,si)))
                pairing.update(keep.numpy().tobytes())
                opt.zero_grad(set_to_none=True); batch_loss=0.0
                for moff in range(0,len(take),args.microbatch):
                    mtake=take[moff:moff+args.microbatch]; xb=torch.from_numpy(train["X"][s][mtake]).to(device); vm=torch.from_numpy(train["valid"][s][mtake]).to(device); yb=torch.from_numpy(train["y"][s][mtake]*SCALE).to(device)
                    amp=torch.autocast(device_type="cuda",dtype=torch.bfloat16) if device.type=="cuda" else contextlib.nullcontext()
                    with amp: pred=_forward(model,xb,bank,keep,vm); raw_loss=nn.functional.mse_loss(pred.float(),yb)
                    if not bool(torch.isfinite(raw_loss)): raise FloatingPointError(f"nonfinite loss at epoch={ep} step={step}")
                    (raw_loss * (len(mtake)/len(take))).backward(); batch_loss += float(raw_loss.detach()) * len(mtake)/len(take)
                grad_norm=float(nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)); opt.step(); ema.update_after_step(model); losses.append(batch_loss); si+=1
                if step == 1 or step % 50 == 0:
                    progress={"event":"step","epoch":ep,"step":step,"loss":batch_loss,"grad_norm":grad_norm,"elapsed_seconds":time.monotonic()-started,"cuda_peak_alloc_mib":float(torch.cuda.max_memory_allocated(device)/(1024*1024)) if device.type=="cuda" else None}
                    (dest/"heartbeat.json").write_text(json.dumps(progress,indent=2)+"\n")
                    print(f"[rift {args.variant}] ep={ep}/{args.epochs} step={step}/{total} loss={batch_loss:.6f} grad={grad_norm:.3f} elapsed={progress['elapsed_seconds']:.0f}s",flush=True)
                if args.max_updates_smoke and step>=args.max_updates_smoke: break
            if args.max_updates_smoke and step>=args.max_updates_smoke: break
        _checkpoint(dest/f"epoch_{ep:03d}.pt",model,opt,ema,ep,step,rng,variant=args.variant,context_bins=args.context_bins,attention_backend=args.attention_backend,microbatch=args.microbatch,smoke=bool(args.max_updates_smoke))
        row={"event":"epoch","epoch":ep,"global_step":step,"train_mse":float(np.mean(losses)),"lr":lr,"smoke":bool(args.max_updates_smoke),"endpoint_bank_mask_sequence_sha256":pairing.hexdigest()}
        with (dest/"metrics.jsonl").open("a") as f: f.write(json.dumps(row)+"\n"); f.flush()
        (dest/"heartbeat.json").write_text(json.dumps(row,indent=2)+"\n")
        if args.max_updates_smoke: break
        if si != train["updates_per_epoch"]: raise RuntimeError(f"epoch {ep} updates {si} != {train['updates_per_epoch']}")
    if not args.max_updates_smoke:
        if step != args.epochs * UPDATES_PER_EPOCH: raise RuntimeError(f"formal update total {step} != {args.epochs * UPDATES_PER_EPOCH}")
        return score_stage(args)
    return {"status":"SMOKE_COMPLETED" if args.max_updates_smoke else "TRAIN_COMPLETED","steps":step,"initialization_sha256":init_sha}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--dest",type=Path,required=True); p.add_argument("--device",default="cuda:0"); p.add_argument("--variant",choices=("recency","flat"),required=True); p.add_argument("--context-bins",type=int,choices=(200,300),default=300); p.add_argument("--attention-backend",choices=("local","dense"),default="local"); p.add_argument("--stage",choices=("train","score"),default="train"); p.add_argument("--epochs",type=int,default=EPOCHS); p.add_argument("--microbatch",type=int,default=32); p.add_argument("--max-updates-smoke",type=int); p.add_argument("--resume",type=Path); args=p.parse_args()
    if args.epochs<1 or args.epochs>EPOCHS or args.microbatch<1 or args.microbatch>BATCH: p.error("epochs must be 1..32 and microbatch must be 1..32")
    print(json.dumps(score_stage(args) if args.stage=="score" else run(args),indent=2))
if __name__=="__main__": main()
