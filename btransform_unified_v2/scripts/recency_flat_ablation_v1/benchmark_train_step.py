#!/usr/bin/env python3
"""Paired, fresh-parameter CUDA train-step cost benchmark for RIFT bias modes.

This is an engineering compute comparison, not an accuracy experiment: it
uses the first real source training batch from each current recipe and fresh
seed-42 timed parameters.  Offline reference audits may inspect existing
receipts/checkpoints before timing; no trained epoch state is used as a timed
model initialization.
"""
from __future__ import annotations

import argparse, contextlib, hashlib, importlib.util, json, os, platform, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]; WS = ROOT.parent; HERE = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "src", WS / "btransform_unified_v1/src", WS / "btransform_unified_v1/scripts", WS, HERE):
    if str(p) not in sys.path: sys.path.insert(0, str(p))

from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1.model import unit_dropout_seed, whole_unit_dropout

SEED, ROUNDS, WARMUP, TIMED = 42, 5, 10, 30
OUT = ROOT / "results/recency_flat_ablation_v1"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


def sha_bytes(value: Any) -> str:
    if isinstance(value, torch.Tensor): value = value.detach().cpu().contiguous().numpy().tobytes()
    elif isinstance(value, np.ndarray): value = np.ascontiguousarray(value).tobytes()
    elif isinstance(value, str): value = value.encode()
    return hashlib.sha256(value).hexdigest()

def source_map(fixture: Mapping[str,Any]) -> dict[str,str]:
    """Seal this benchmark and every current decoder/runtime source in use."""
    paths={Path(__file__).resolve(), ROOT/"src/btransform_unified_v2/config.py", ROOT/"src/btransform_unified_v2/temporal.py", ROOT/"src/btransform_unified_v2/model.py", ROOT/"src/btransform_unified_v2/concat_model.py", ROOT/"src/btransform_unified_v2/streaming.py", ROOT/"src/btransform_unified_v2/cpu_runtime.py", HERE/"m1_flat_model.py", HERE/"m1_full_flat_train.py", HERE/"m2_flat_train.py", HERE/"h1_flat_train.py"}
    bound=fixture.get("source_bindings",{})
    if isinstance(bound,Mapping): paths.update(Path(name) for name in bound)
    absent=[str(path) for path in paths if not path.is_file()]
    if absent: raise RuntimeError(f"runtime source seal path missing: {absent}")
    return {str(path):sha_bytes(path.read_bytes()) for path in sorted(paths)}


def device_contract() -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1": raise RuntimeError("require CUDA_VISIBLE_DEVICES=1 (physical GPU 1)")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise RuntimeError("GPU 1 must be visible as exactly logical cuda:0")
    device = torch.device("cuda:0")
    if torch.cuda.current_device() != 0: torch.cuda.set_device(device)
    import subprocess
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader,nounits"], text=True).splitlines()
    uuid = next((row for row in rows if row.split(",", 1)[0].strip() == "1"), None)
    if uuid is None: raise RuntimeError("nvidia-smi cannot identify physical GPU 1")
    return {"device": device, "cuda_visible_devices": "1", "logical_device": "cuda:0", "physical_gpu_1": uuid}


def state_hash(model: nn.Module, *, params_only: bool = False) -> str:
    h = hashlib.sha256(); rows = model.named_parameters() if params_only else model.state_dict().items()
    for name, value in sorted(rows): h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def assert_pair(rec: nn.Module, flat: nn.Module) -> dict[str, Any]:
    a, b = dict(rec.named_parameters()), dict(flat.named_parameters())
    if a.keys() != b.keys() or any(not torch.equal(a[k], b[k]) for k in a): raise RuntimeError("fresh paired named parameters differ")
    sa, sb = rec.state_dict(), flat.state_dict(); delta = [k for k in sa if not torch.equal(sa[k], sb[k])]
    if delta != ["temporal.recency_slopes"]: raise RuntimeError(f"unexpected paired state delta {delta}")
    slopes = flat.temporal.recency_slopes
    if slopes.dtype != torch.float32 or tuple(slopes.shape) != (8,) or torch.count_nonzero(slopes).item() != 0: raise RuntimeError("flat slope buffer drift")
    return {"named_parameters_byte_equal": True, "sole_buffer_delta": delta[0], "recency_slope_sha256": sha_bytes(rec.temporal.recency_slopes), "flat_slope_sha256": sha_bytes(slopes)}


def stats(values: list[float]) -> dict[str, float | int]:
    a = np.asarray(values, np.float64); return {"n":len(values),"mean_ms":float(a.mean()),"median_ms":float(np.median(a)),"p95_ms":float(np.percentile(a,95))}


def tensor_meta(value: Any) -> Any:
    if isinstance(value, torch.Tensor): return {"shape":list(value.shape),"dtype":str(value.dtype),"sha256":sha_bytes(value)}
    if isinstance(value, np.ndarray): return {"shape":list(value.shape),"dtype":str(value.dtype),"sha256":sha_bytes(value)}
    return value

def jsonable(value: Any) -> Any:
    """Make provenance compact and JSON-safe without expanding array contents."""
    if isinstance(value, (torch.Tensor, np.ndarray)): return tensor_meta(value)
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    if isinstance(value, Mapping): return {str(key):jsonable(item) for key,item in value.items()}
    if isinstance(value, (tuple,list)): return [jsonable(item) for item in value]
    return value

def bank_meta(bank: Any) -> dict[str,Any]:
    out={"unit_mask_sha256":sha_bytes(bank.unit_mask)}
    for key in ("E0","carrier"):
        if hasattr(bank,key): out[key+"_sha256"]=sha_bytes(getattr(bank,key))
    out["calibration_meta"]=jsonable(dict(getattr(bank,"calibration_meta",{})))
    return out


def m1_fixture(device: torch.device):
    flat = load("_bench_m1_full_flat", HERE / "m1_full_flat_train.py"); base = flat.frozen
    pack=ROOT/"results/m1_muscle_r100_v1/carrier_official4/carrier_pack.npz"; reference_path=ROOT/"results/m1_muscle_r100_v1/formal_s42_gpu1"
    carriers, binding = flat._carrier_binding(pack); reference=flat._paired_reference(reference_path,arm="D_JOINT",binding=binding); dataset, sampler = base.legacy.build_fullsession_face()
    banks = base._replace_carriers(base.legacy.build_fullsession_banks(dataset)[0], carriers, base.SOURCE_SESSIONS); calib = base.m1_plan.calib_trials_from_dataset(dataset)
    loader = iter(torch.utils.data.DataLoader(dataset, batch_sampler=sampler, collate_fn=base.legacy._collate, num_workers=0)); x,y,sessions = next(loader)
    if any(s != sessions[0] for s in sessions): raise RuntimeError("M1 first source batch is mixed-session")
    if len(x)!=32 or tuple(x.shape[1:])!=(100,64) or x.dtype!=torch.float32: raise RuntimeError("M1 first batch geometry/dtype drift")
    valid = torch.ones((len(x), base.CONTEXT), dtype=torch.bool); keep = whole_unit_dropout(banks[sessions[0]].unit_mask, p=.1, generator=torch.Generator().manual_seed(unit_dropout_seed(SEED,1,0)))
    payload={"x":x.float().to(device),"y":y.to(device),"valid":valid.to(device),"bank":banks[sessions[0]],"keep":keep,"session":sessions[0]}
    def factory(mode: str):
        model = base._decoder(device, "D_JOINT", SEED) if mode=="recency" else flat._decoder(device,"D_JOINT",SEED)
        model.install_session_memory(banks,calib); return model
    def step(model,opt,ema):
        opt.zero_grad(set_to_none=True); amp=torch.autocast("cuda",torch.bfloat16)
        with amp: loss=nn.functional.mse_loss(model(payload["x"],payload["bank"],dropout_keep=payload["keep"],input_valid_mask=payload["valid"]).float(),payload["y"])
        if not bool(torch.isfinite(loss)): raise FloatingPointError("nonfinite M1 benchmark loss")
        loss.backward(); g=float(nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)); opt.step(); ema.update_after_step(model); return float(loss.detach()),g
    return {"task":"m1","context":100,"backend":"local","lr":1e-4,"factory":factory,"optimizer":lambda m: flat.optimizer_factory.build_optimizer(m.named_parameters(),lr=1e-4,weight_decay=flat.plan.WEIGHT_DECAY),"step":step,"payload":payload,"mask":lambda i: whole_unit_dropout(banks[sessions[0]].unit_mask,p=.1,generator=torch.Generator().manual_seed(unit_dropout_seed(SEED,1,i))),"fixture":{"raw":tensor_meta(x),"target":tensor_meta(y),"valid":tensor_meta(valid),"dropout_mask_sha256":sha_bytes(keep),"bank":"live B3S M10 / muscle_response16_svd4/global_rms official4 carrier", "bank_hashes":bank_meta(banks[sessions[0]]),"calib_sha256":sha_bytes(json.dumps({k:sha_bytes(v) for k,v in calib.items()},sort_keys=True)),"session":sessions[0],"carrier_binding_sha256":sha_bytes(json.dumps(binding,sort_keys=True)),"paired_reference":reference,"source_bindings":flat._source_hashes()}}


def m2_fixture(device: torch.device):
    flat=load("_bench_m2_flat",HERE/"m2_flat_train.py"); base=flat.frozen; reference=flat.reference_binding()
    dual,banks=base._load_surface("source_train",device); manifest=base.old_sampler.load_manifest(base.MANIFEST); batch=next(base.old_sampler.iter_manifest_batches(dual,manifest,1,device=device,target_space=base.old_plan.TRAINING_TARGET_SPACE))
    if batch.X.shape[0]!=32 or tuple(batch.X.shape[1:])!=(50,96) or batch.X.dtype!=torch.float32: raise RuntimeError("M2 first batch geometry/dtype drift")
    valid=base._batch_valid(batch,surface="source_train",padding=base._surface_padding("source_train",banks),device=device); keep=whole_unit_dropout(batch.unit_mask,p=.1,generator=torch.Generator().manual_seed(unit_dropout_seed(SEED,1,0)))
    payload={"x":batch.X,"y":batch.last_target,"valid":valid,"bank":banks[batch.session_id],"keep":keep,"session":batch.session_id}
    def factory(mode:str): return base._decoder(device) if mode=="recency" else flat.flat_decoder(device)
    def step(model,opt,ema):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",torch.bfloat16): loss=nn.functional.mse_loss(model(payload["x"],payload["bank"],dropout_keep=payload["keep"],input_valid_mask=payload["valid"]).float(),payload["y"])
        if not bool(torch.isfinite(loss)): raise FloatingPointError("nonfinite M2 benchmark loss")
        loss.backward();g=float(nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step();ema.update_after_step(model);return float(loss.detach()),g
    return {"task":"m2","context":50,"backend":"local","lr":3e-4,"factory":factory,"optimizer":lambda m: base.old_training.build_optimizer(m.named_parameters(),lr=3e-4,weight_decay=flat.v1_plan.WEIGHT_DECAY),"step":step,"payload":payload,"mask":lambda i: whole_unit_dropout(batch.unit_mask,p=.1,generator=torch.Generator().manual_seed(unit_dropout_seed(SEED,1,i))),"fixture":{"raw":tensor_meta(batch.X),"target":tensor_meta(batch.last_target),"valid":tensor_meta(valid),"dropout_mask_sha256":sha_bytes(keep),"bank":"frozen concat source manifest first batch","bank_hashes":bank_meta(banks[batch.session_id]),"sampler_manifest_sha256":sha_bytes(base.MANIFEST.read_bytes()),"paired_reference":reference,"source_bindings":flat.source_hashes()}}


def h1_fixture(device: torch.device):
    flat=load("_bench_h1_flat",HERE/"h1_flat_train.py"); s,ht=flat.signed,flat.ht
    reference=flat.validate_reference(flat.REF_BANKS); plan,_receipt,_carriers=s._load_banks(flat.REF_BANKS); train=ht.build_train(300); cal=s.build_cal1_signed(train["banks"],plan); rng=np.random.default_rng(SEED); order=list(train["sessions"]);rng.shuffle(order); session=order[0]; take=rng.permutation(len(train["X"][session]))[:32]; budget=int(s.prefix_schedule(0,train["updates_per_epoch"])[0]); starts=s.cal1_b2.legal_starts(cal["starts"][session],cal["n_trials"][session],budget); bank=cal["banks"][(session,s.pick_m7_start(session,epoch0=0,step=0,starts=starts),budget)]
    keep=s.whole_unit_dropout(torch.from_numpy(bank.unit_mask.copy()),p=.1,generator=torch.Generator().manual_seed(s.unit_dropout_seed(SEED,1,0))); payload={"x":torch.from_numpy(train["X"][session][take]).to(device),"y":torch.from_numpy(train["y"][session][take]*s.SCALE).to(device),"valid":torch.from_numpy(train["valid"][session][take]).to(device),"bank":bank,"keep":keep,"session":session,"budget":budget}
    if payload["x"].shape[0]!=32 or tuple(payload["x"].shape[1:])!=(300,s.h1_config.UNITS) or payload["x"].dtype!=torch.float32: raise RuntimeError("H1 first batch geometry/dtype drift")
    def factory(mode:str): return ht._decoder(mode,device,300,"dense")
    def step(model,opt,ema):
        opt.zero_grad(set_to_none=True); total=0.
        for off in range(0,len(payload["x"]),32):
            with torch.autocast("cuda",torch.bfloat16): loss=nn.functional.mse_loss(ht._forward(model,payload["x"][off:off+32],payload["bank"],payload["keep"],payload["valid"][off:off+32]).float(),payload["y"][off:off+32])
            if not bool(torch.isfinite(loss)): raise FloatingPointError("nonfinite H1 benchmark loss")
            loss.backward();total+=float(loss.detach())
        g=float(nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True));opt.step();ema.update_after_step(model);return total,g
    return {"task":"h1","context":300,"backend":"dense","lr":1e-4,"factory":factory,"optimizer":lambda m: torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8),"step":step,"payload":payload,"mask":lambda i: s.whole_unit_dropout(torch.from_numpy(bank.unit_mask.copy()),p=.1,generator=torch.Generator().manual_seed(s.unit_dropout_seed(SEED,1,i))),"fixture":{"raw":tensor_meta(payload["x"]),"target":tensor_meta(payload["y"]),"valid":tensor_meta(payload["valid"]),"dropout_mask_sha256":sha_bytes(keep),"bank":"current signed_state C2 M7 first epoch/session/permutation/budget","bank_hashes":bank_meta(bank),"session":session,"budget":budget,"banks_receipt_sha256":sha_bytes((flat.REF_BANKS/"receipt.json").read_bytes()),"plan_sha256":sha_bytes(json.dumps(plan,sort_keys=True,default=str)),"paired_reference":reference,"source_bindings":flat.source_manifest()}}


def model_info(model: nn.Module, context: int, backend: str, batch: int) -> dict[str,Any]:
    cfg=model.temporal_config; windows=list(cfg.windows); kv=sum(w-1 for w in windows)*2*cfg.heads*cfg.head_dim*4
    mac=2*batch*cfg.width*(len(windows)*context*context if backend=="dense" else context*sum(windows))
    return {"parameters":sum(p.numel() for p in model.parameters()),"trainable":sum(p.numel() for p in model.parameters() if p.requires_grad),"buffers":sum(b.numel() for b in model.buffers()),"windows":windows,"batch_size":batch,"output_geometry":str(getattr(model,"geometry",{})),"temporal_attention_macs_formula":"dense=2*B*width*layers*R^2; local=2*B*width*R*sum(windows): QK+AV only; excludes frontend, projections, norms, softmax, FFN, backward, optimizer and memory traffic","temporal_attention_macs":mac,"logical_kv_bytes_per_stream":kv}


def one_round(case:dict[str,Any],mode:str,device:torch.device)->dict[str,Any]:
    torch.manual_seed(SEED); model=case["factory"](mode).train(); opt=case["optimizer"](model);ema=DecoderEMA(model,decay=.9995)
    before=state_hash(model); parameter_before=state_hash(model,params_only=True)
    if parameter_before != case["initial_parameter_sha"][mode]: raise RuntimeError("round fresh parameter initialization differs from paired initial receipt")
    masks=[case["mask"](i) for i in range(WARMUP+TIMED)]; torch.cuda.reset_peak_memory_stats(device)
    for index in range(WARMUP): case["payload"]["keep"]=masks[index]; case["step"](model,opt,ema)
    torch.cuda.synchronize(device); baseline=torch.cuda.memory_allocated(device); baseline_reserved=torch.cuda.memory_reserved(device); torch.cuda.reset_peak_memory_stats(device); samples=[]; events=[]
    for index in range(TIMED):
        case["payload"]["keep"]=masks[WARMUP+index]
        torch.cuda.synchronize(device); a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();start=time.perf_counter_ns();loss,grad=case["step"](model,opt,ema);b.record();torch.cuda.synchronize(device);samples.append((time.perf_counter_ns()-start)/1e6);events.append(a.elapsed_time(b))
        if not np.isfinite(loss) or not np.isfinite(grad):raise RuntimeError("nonfinite timed recipe update")
    slopes=model.temporal.recency_slopes; zero=bool(torch.count_nonzero(slopes).item()==0)
    if zero != (mode=="flat"): raise RuntimeError("post-update temporal slope mode drift")
    if ema.n_updates != WARMUP + TIMED: raise RuntimeError("EMA update count differs from benchmark update count")
    return {"wall_ms":samples,"cuda_event_ms":events,"cuda_event_note":"GPU elapsed interval from event before transaction to event after transaction; not a sum of kernels","wall_summary":stats(samples),"cuda_event_summary":stats(events),"dropout_mask_sequence_sha256":sha_bytes(b"".join(mask.detach().cpu().contiguous().numpy().tobytes() for mask in masks)),"memory":{"warmup_baseline_allocated":baseline,"warmup_baseline_reserved":baseline_reserved,"peak_allocated":torch.cuda.max_memory_allocated(device),"peak_reserved":torch.cuda.max_memory_reserved(device),"incremental_peak_allocated":torch.cuda.max_memory_allocated(device)-baseline},"loss_last":loss,"grad_norm_last":grad,"ema_updates":ema.n_updates,"state_hash_before":before,"parameter_state_sha256_before":parameter_before,"state_hash_after":state_hash(model),"slopes_zero_after":zero,"model":model_info(model,case["context"],case["backend"],int(case["payload"]["x"].shape[0]))}


def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--task",choices=("m1","m2","h1"),required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--affinity",default="8,9,10,11");a=p.parse_args();out=a.output.resolve()
    if out.exists():raise FileExistsError("--output must be fresh")
    cpus=[int(x) for x in a.affinity.split(",")];os.sched_setaffinity(0,cpus); torch.set_num_threads(2 if a.task=="h1" else 4); torch.set_num_interop_threads(1)
    env=device_contract(); fixture_started=time.perf_counter();case={"m1":m1_fixture,"m2":m2_fixture,"h1":h1_fixture}[a.task](env["device"]); fixture_prepare_seconds=time.perf_counter()-fixture_started
    fixture_record=jsonable(case["fixture"])
    json.dumps(fixture_record, sort_keys=True)
    runtime_source_map_before=source_map(case["fixture"])
    # Explicit whole-model pairing proof is outside all timed rounds.
    torch.manual_seed(SEED); reference_recency=case["factory"]("recency"); torch.manual_seed(SEED); reference_flat=case["factory"]("flat")
    initial_pair={**assert_pair(reference_recency,reference_flat),"recency_parameter_sha256":state_hash(reference_recency,params_only=True),"flat_parameter_sha256":state_hash(reference_flat,params_only=True)}
    case["initial_parameter_sha"]={"recency":initial_pair["recency_parameter_sha256"],"flat":initial_pair["flat_parameter_sha256"]}
    del reference_recency, reference_flat; torch.cuda.empty_cache()
    rounds={"recency":[],"flat":[]}
    for r in range(ROUNDS):
        order=("recency","flat") if r%2==0 else ("flat","recency")
        for mode in order: rounds[mode].append({"round":r+1,"position":order.index(mode)+1,**one_round(case,mode,env["device"])})
    runtime_source_map_after=source_map(case["fixture"])
    if runtime_source_map_after != runtime_source_map_before: raise RuntimeError("runtime source map changed during benchmark; refusing receipt")
    report={"schema":"rift_recency_flat_real_train_step_gpu_v1","status":"COMPLETED_FRESH_PARAMETER_CONTROLLED_COMPUTE_ONLY","utc":datetime.now(timezone.utc).isoformat(),"scope":"Fresh seed-42 timed models use fresh parameters. Fixture preparation and all I/O occur before timing. Offline reference audits may inspect completed recency checkpoints and frozen initializers/source assets. This is not target-performance evidence.","task":a.task,"script_sha256":sha_bytes(Path(__file__).read_bytes()),"runtime_source_map_before":runtime_source_map_before,"runtime_source_map_after":runtime_source_map_after,"runtime_source_map_unchanged":True,"device":{**{k:(str(v) if isinstance(v,torch.device) else v) for k,v in env.items()},"torch":torch.__version__,"cuda":torch.version.cuda,"cpu":platform.processor(),"affinity":sorted(os.sched_getaffinity(0)),"threads":torch.get_num_threads()},"protocol":{"rounds":ROUNDS,"warmup_updates":WARMUP,"timed_updates":TIMED,"timing":"synchronize immediately before and after each strict update transaction; event interval is supplementary","included":"zero_grad, BF16 autocast forward, loss, backward, clip=1, optimizer.step, EMA.update","excluded":"fixture/data/bank/mask preparation and trained checkpoint I/O","rotation":"recency/flat first position alternates"},"fixture_prepare_seconds":fixture_prepare_seconds,"fixture":fixture_record,"initial_pairing":initial_pair,"rounds":rounds}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps({"output":str(out),"task":a.task},sort_keys=True))

if __name__=="__main__":main()
