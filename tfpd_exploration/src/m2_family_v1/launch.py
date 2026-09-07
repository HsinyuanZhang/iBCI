"""Explicitly gated source-only paired 24-epoch runner; never scores ext4/HO."""
from __future__ import annotations
import json, os, time, tempfile, hashlib
from pathlib import Path
from typing import Any
import numpy as np
import torch

from tfpd_exploration.src.m2_b_small_stability_v1 import training
from tfpd_exploration.src.m2_b_small_stability_v1.ema import DecoderEMA
from tfpd_exploration.src.m2_dual_track_v1 import data, plan, sampler, training as old_training
from . import config
from .decoder import make_paired_decoders, shared_parameter_max_abs_diff

def formal_train_cli() -> str:
    return (f"PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=1 {config.TRAIN_ENV_FLAG}=1 "
            "/home/xinyuan/miniconda3/envs/spint/bin/python -m tfpd_exploration.src.m2_family_v1.launch --train")

def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def _append(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(value,sort_keys=True)+"\n")

def _sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1<<20),b""): digest.update(block)
    return digest.hexdigest()

def _recipe_hashes(manifest_path: Path, banks: dict[str, Any]) -> dict[str, Any]:
    """Bind exact inputs/code without opening any non-source surface."""
    package=Path(__file__).resolve().parent
    code={name:_sha256_file(package/name) for name in ("config.py","decoder.py","routing.py","launch.py")}
    cache={}
    for session,bank in sorted(banks.items()):
        root=data._session_dir("source_train",session)
        cache[session]={name:_sha256_file(root/name) for name in ("X_store.npy","target_store.npy","eligible_starts.npy","T.npy","e0_u.pt","mapping.json","provenance.json")}
        cache[session]["provenance_e0_sha256"]=bank.provenance.get("e0_sha256")
        cache[session]["provenance_t4_sha256"]=bank.provenance.get("t4_sha256")
    return {"code_sha256":code,"sampler_sha256":_sha256_file(manifest_path),"source_cache_sha256":cache}

def _score(model: torch.nn.Module, banks: dict[str, Any], device: torch.device) -> float:
    from tfpd_exploration.src.m2_dual_track_v1.contracts import summarize_sessions, variance_weighted_r2
    values = {}
    model.eval()
    for name, bank in banks.items():
        ys=[]; ps=[]
        for batch in data.iter_session_batches(bank, batch_size=plan.EFFECTIVE_BATCH, device=device, target_space=plan.SCORING_TARGET_SPACE):
            with torch.inference_mode(): ps.append((model.forward_last(batch.X, batch.bank, batch.unit_mask)/plan.BEHAVIOR_SCALE).cpu().numpy())
            ys.append(batch.last_target.cpu().numpy())
        values[name] = variance_weighted_r2(np.concatenate(ys), np.concatenate(ps))
    return float(summarize_sessions(values)["equal_session_mean"])

def run_pair() -> dict[str, Any]:
    if os.environ.get(config.TRAIN_ENV_FLAG) != "1":
        raise RuntimeError(f"REFUSED: set {config.TRAIN_ENV_FLAG}=1 after review authorization")
    raw_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if raw_gpu.strip() not in {"0", "1"}: raise RuntimeError("exactly one leased GPU index is required")
    root=config.RESULT_ROOT
    started=time.monotonic()
    if root.exists(): raise RuntimeError(f"refuse overwrite {root}")
    device=torch.device("cuda:0")
    if config.TRAIN_DTYPE != "float32": raise RuntimeError("only frozen FP32 training is implemented")
    torch.manual_seed(config.SEED); np.random.seed(config.SEED)
    flat, route=make_paired_decoders(config.SEED)
    if shared_parameter_max_abs_diff(flat, route) != 0.0: raise RuntimeError("paired shared initialization drift")
    models={"FLAT":flat.to(device), "ROUTE":route.to(device)}
    opt={name:training.make_optimizer(model.trainable_parameters().items()) for name,model in models.items()}
    ema={name:DecoderEMA(model, decay=config.EMA_DECAY) for name,model in models.items()}
    manifest_path=Path("tfpd_exploration/results/m2_dual_track_v1/20260905_101500/sampler/shuffled_batch_manifest_24.json")
    manifest=sampler.load_manifest(manifest_path)
    banks={surface:{s:data.load_session_bank(surface,s,device=device) for s in plan.HELDIN_SESSIONS} for surface in ("source_train","source_minival")}
    actual_updates=old_training.count_updates(banks["source_train"])
    if actual_updates != config.UPDATES_PER_EPOCH: raise RuntimeError(f"updates/epoch drift {actual_updates} != {config.UPDATES_PER_EPOCH}")
    states={name:training.TrainState(cell=f"CRST_B4_{name}", seed=config.SEED, manifest_digest=manifest["digest"]) for name in models}
    scores={name:{"RAW":{},"EMA":{}} for name in models}
    _write(root/"pretrain_manifest.json", {"schema":config.SCHEMA,"status":"RUNNING","members":list(config.MEMBERS),"epochs":config.EPOCHS,"selection":{"primary":"EMA maximize source-minival equal-session R2 over epochs 1..24; earliest epoch tie","secondary":"RAW endpoint24 reported only; RAW/EMA never selected between"},"forbidden":["ext4","official","heldout"],"shared_max_abs":0.0,"manifest_digest":manifest["digest"],"updates_per_epoch":actual_updates,"train_dtype":config.TRAIN_DTYPE,"target_unit":"decoder_raw_equals_native_times_5","loss":"MSE(pred_decoder_raw,batch.last_target_decoder_raw)","gate_initialization":"ROUTE g exactly zero; g0 full-forward parity bound by current contract test and smoke_v2 receipt","shared_initialization":"same seed42 and byte-identical non-routing state","recipe_hashes":_recipe_hashes(manifest_path,banks["source_train"])})
    for epoch in range(1, config.EPOCHS+1):
        # Same epoch/batch order and the *same* whole-unit mask for both members.
        for name,model in models.items():
            model.train()
            for batch_id,batch in enumerate(old_training.epoch_batches_shuffled(banks["source_train"], manifest, epoch, device=device)):
                if time.monotonic()-started >= config.PAIR_WALLCLOCK_BUDGET_SECONDS: raise RuntimeError("hard 6h pair budget reached")
                # Preserve historical BxN whole-unit semantics: each sample gets
                # an independent deterministic mask, not one broadcast N mask.
                base_mask=batch.unit_mask if batch.unit_mask is not None else batch.bank.unit_mask
                if base_mask.ndim == 1: base_mask=base_mask.unsqueeze(0).expand(batch.X.size(0),-1)
                keep=training.unit_dropout_mask(base_mask,seed=config.SEED,epoch=epoch,batch_id=batch_id).to(device)
                opt[name].zero_grad(set_to_none=True)
                # Frozen S1 cosine schedule; both paired members receive its same step.
                training.apply_cell_lr(opt[name], "S1-SMALL-COS", (epoch-1)*3165 + batch_id + 1)
                # epoch_batches_shuffled's default target space is decoder_raw: both terms are already 5x native.
                loss=torch.nn.functional.mse_loss(model.forward_last(batch.X,batch.bank,dropout_keep=keep).float(), batch.last_target.float())
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP); opt[name].step(); ema[name].update_after_step(model)
                if batch_id % config.LOG_EVERY_STEPS == 0:
                    _append(root/name/"metrics.jsonl",{"epoch":epoch,"batch_id":batch_id,"global_step":(epoch-1)*actual_updates+batch_id+1,"loss":float(loss.detach().cpu()),"lr":float(opt[name].param_groups[0]["lr"])})
                    _write(root/name/"heartbeat.json",{"epoch":epoch,"batch_id":batch_id,"global_step":(epoch-1)*actual_updates+batch_id+1,"loss":float(loss.detach().cpu()),"unix":time.time()})
            scores[name]["RAW"][epoch]=_score(model,banks["source_minival"],device)
            scores[name]["EMA"][epoch]=ema[name].score_with_ema(model, lambda m:_score(m,banks["source_minival"],device))
            _write(root/name/"epoch_metrics.json",{"epoch":epoch,"raw_equal_session_r2":scores[name]["RAW"][epoch],"ema_equal_session_r2":scores[name]["EMA"][epoch],"elapsed_seconds":time.monotonic()-started})
            state=states[name]; state.epoch=epoch; state.global_step=epoch*old_training.count_updates(banks["source_train"])
            ckpt=training.save_checkpoint(model,opt[name],ema[name],state); (root/name).mkdir(parents=True,exist_ok=True); torch.save(ckpt,root/name/f"epoch_{epoch:03d}.pt")
    result={"schema":config.SCHEMA,"status":"SOURCE_ONLY_COMPLETE","scores":scores,"source_pick":{n:{v:min(d,key=lambda e:(-d[e],e)) for v,d in vals.items()} for n,vals in scores.items()},"comparison_scored":False}
    _write(root/"summary.json",result); return result

def run_smoke(max_steps: int = config.SMOKE_MAX_STEPS) -> dict[str, Any]:
    """Disposable real-data paired train path check.  Saves no weights/checkpoints."""
    if os.environ.get(config.TRAIN_ENV_FLAG) != "1": raise RuntimeError(f"REFUSED: set {config.TRAIN_ENV_FLAG}=1")
    if not (1 <= max_steps <= config.SMOKE_MAX_STEPS): raise RuntimeError("smoke step budget")
    root=config.SMOKE_V2_ROOT
    if root.exists(): raise RuntimeError(f"refuse overwrite smoke receipt {root}")
    if os.environ.get("CUDA_VISIBLE_DEVICES","") not in {"0","1"}: raise RuntimeError("one leased GPU required")
    device=torch.device("cuda:0"); torch.manual_seed(config.SEED); np.random.seed(config.SEED)
    flat,route=make_paired_decoders(config.SEED); models={"FLAT":flat.to(device),"ROUTE":route.to(device)}
    optimizers={n:training.make_optimizer(m.trainable_parameters().items()) for n,m in models.items()}
    emas={n:DecoderEMA(m,decay=config.EMA_DECAY) for n,m in models.items()}
    manifest=sampler.load_manifest(Path("tfpd_exploration/results/m2_dual_track_v1/20260905_101500/sampler/shuffled_batch_manifest_24.json"))
    banks={s:data.load_session_bank("source_train",s,device=device) for s in plan.HELDIN_SESSIONS}
    count=old_training.count_updates(banks)
    if count != config.UPDATES_PER_EPOCH: raise RuntimeError(f"updates/epoch drift {count}")
    _write(root/"receipt.json",{"schema":config.SCHEMA+"_smoke_v2","status":"RUNNING","max_steps":max_steps,"warmup_steps":20,"timed_steps":80,"weights_saved":False,"checkpoint_saved":False,"target_unit":"decoder_raw_equals_native_times_5","train_dtype":config.TRAIN_DTYPE,"updates_per_epoch":count})
    logs=[]
    warm_seconds=0.0; timed_seconds=0.0; mask_diverse=False; arms_same_mask=True
    for batch_id,batch in enumerate(old_training.epoch_batches_shuffled(banks,manifest,1,device=device)):
        if batch_id >= max_steps: break
        base_mask=batch.unit_mask if batch.unit_mask is not None else batch.bank.unit_mask
        if base_mask.ndim == 1: base_mask=base_mask.unsqueeze(0).expand(batch.X.size(0),-1)
        keep=training.unit_dropout_mask(base_mask,seed=config.SEED,epoch=1,batch_id=batch_id).to(device)
        mask_diverse |= bool(keep.size(0)>1 and torch.any(keep[1:] != keep[:1]))
        arm_keep={"FLAT":keep.clone(),"ROUTE":keep.clone()}; arms_same_mask &= bool(torch.equal(arm_keep["FLAT"],arm_keep["ROUTE"]))
        t0=time.perf_counter(); row={"batch_id":batch_id,"target_abs_max":float(batch.last_target.abs().max())}
        for name,model in models.items():
            model.train(); optimizers[name].zero_grad(set_to_none=True); training.apply_cell_lr(optimizers[name],"S1-SMALL-COS",batch_id+1)
            loss=torch.nn.functional.mse_loss(model.forward_last(batch.X,batch.bank,dropout_keep=arm_keep[name]).float(),batch.last_target.float()); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),config.GRAD_CLIP); optimizers[name].step(); emas[name].update_after_step(model); row[name+"_loss"]=float(loss.detach().cpu())
        elapsed=time.perf_counter()-t0
        if batch_id < 20: warm_seconds += elapsed
        else: timed_seconds += elapsed
        if batch_id % config.LOG_EVERY_STEPS == 0: _append(root/"steps.jsonl",row); _write(root/"heartbeat.json",row|{"unix":time.time()})
        logs.append(row)
    # One real minival batch checks RAW and EMA native-unit outputs without a full score sweep.
    mini={s:data.load_session_bank("source_minival",s,device=device) for s in plan.HELDIN_SESSIONS}
    min_batch=next(data.iter_session_batches(next(iter(mini.values())),batch_size=plan.EFFECTIVE_BATCH,device=device,target_space=plan.SCORING_TARGET_SPACE))
    score={}
    for name,model in models.items():
        model.eval()
        with torch.inference_mode(): raw=(model.forward_last(min_batch.X,min_batch.bank,min_batch.unit_mask)/plan.BEHAVIOR_SCALE).float()
        ema_native=emas[name].score_with_ema(model,lambda current:(current.forward_last(min_batch.X,min_batch.bank,min_batch.unit_mask)/plan.BEHAVIOR_SCALE).float())
        score[name]={"raw_finite":bool(torch.isfinite(raw).all()),"ema_finite":bool(torch.isfinite(ema_native).all()),"native_shape":list(raw.shape),"target_native_shape":list(min_batch.last_target.shape)}
    # Serialized temporary artifact must roundtrip raw model, optimizer and EMA exactly; it is deleted.
    roundtrip={}
    for name,model in models.items():
        state=training.TrainState(cell="SMOKE",global_step=max_steps,epoch=1,batch_id=max_steps,seed=config.SEED,manifest_digest=manifest["digest"])
        payload=training.save_checkpoint(model,optimizers[name],emas[name],state)
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            torch.save(payload,tmp.name); restored=make_paired_decoders(config.SEED)[1 if name=="ROUTE" else 0].to(device); ropt=training.make_optimizer(restored.trainable_parameters().items()); rema=DecoderEMA(restored,decay=config.EMA_DECAY); rstate=training.TrainState(cell="",seed=0); training.load_checkpoint(torch.load(tmp.name,map_location="cpu",weights_only=False),restored,ropt,rema,rstate)
            exact=all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
            roundtrip[name]={"raw_state_exact":exact,"optimizer_groups":len(ropt.param_groups),"ema_updates":rema.n_updates,"state_step":rstate.global_step}
    result={"schema":config.SCHEMA+"_smoke_v2","status":"COMPLETE_DISPOSABLE_NO_WEIGHTS","steps":len(logs),"first":logs[0],"last":logs[-1],"timing":{"warmup_total_s":warm_seconds,"timed_total_s":timed_seconds,"timed_pair_step_mean_s":timed_seconds/80.0,"estimated_24epoch_pair_hours":timed_seconds/80.0*config.UPDATES_PER_EPOCH*config.EPOCHS/3600.0},"mask":{"b_by_n":True,"row_diversity_observed":mask_diverse,"arms_identical":arms_same_mask},"one_batch_native_raw_ema":score,"checkpoint_roundtrip":roundtrip,"weights_saved":False,"checkpoint_saved":False}
    _write(root/"receipt.json",result); return result

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--train",action="store_true"); p.add_argument("--smoke",action="store_true"); a=p.parse_args()
    if not a.train and not a.smoke: print(formal_train_cli())
    else: print(json.dumps(run_smoke() if a.smoke else run_pair(),sort_keys=True))
