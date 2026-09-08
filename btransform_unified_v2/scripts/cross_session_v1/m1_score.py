#!/usr/bin/env python3
"""Audited score-only entrypoint for a completed cross-session M1 arm.

This wrapper deliberately does not call ``m1_data.materialize_target`` for D:
that legacy helper refits the source carrier.  D projects target M10 through
the exact dictionary/scale/normalizer arrays sealed during source prepare.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"src"),str(ROOT.parent/"btransform_unified_v1/src"),str(ROOT.parent)]
import m1_train as train
from m1_data import FOLDS, QUERY, _bank, _dataset, _records, _slice_trials, _assert_reset, _rectify, materialize_sources, nwb_path, source_for
from btransform_unified_v2.cross_session_m1_model import ARMS, CrossSessionM1Decoder, Z_NONE, B_ACTIVITY_ONLY, D_JOINT

def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def _cache_path(a: argparse.Namespace) -> Path:
    key=hashlib.sha256("|".join(source_for(a.target)).encode()).hexdigest()
    return a.dest.parent.parent/"source_prepare_cache"/f"rsyn3_{key}.npz"

def _require_receipts(a: argparse.Namespace) -> tuple[dict, dict, dict]:
    train_receipt=json.loads((a.dest/"train_receipt.json").read_text())
    preflight=json.loads((a.dest/"preflight.json").read_text())
    if train_receipt.get("status")!="COMPLETED" or preflight.get("status")!="PASSED": raise RuntimeError("completed train and passed preflight receipts are required")
    expected=train.metadata(a)
    for name, value in (("train",train_receipt),("preflight",preflight)):
        for key in ("arm","seed","fold","data_contract"):
            if value.get(key)!=expected[key]: raise RuntimeError(f"{name} receipt metadata mismatch: {key}")
    cp=a.dest/"resume_latest.pt"; selected=a.dest/"selected_ema.pt"
    recorded=train_receipt.get("checkpoint_sha256",{})
    actual={p.name:_sha(p) for p in (cp,selected)}
    if recorded != actual: raise RuntimeError("checkpoint checksum differs from train receipt")
    final=torch.load(cp,map_location=a.device,weights_only=False); picked=torch.load(selected,map_location=a.device,weights_only=False)
    for name, value in (("resume checkpoint",final),("selected EMA",picked)):
        meta=value.get("meta",{})
        for key in ("arm","seed","fold","data_contract"):
            if meta.get(key)!=expected[key]: raise RuntimeError(f"{name} metadata mismatch: {key}")
    if int(final.get("epoch",0)) != train.EPOCHS: raise RuntimeError("resume checkpoint is not fixed epoch 24")
    return train_receipt, final, picked

def _target_dataset(target: str):
    """The B/D FalconDataset route with explicit support/query physical slices."""
    record=_records((target,))[target]
    ds=_dataset({target:_slice_trials(record,*QUERY)},{target:_slice_trials(record,0,10)})
    _assert_reset(ds,QUERY)
    return ds

def _d_target_bank(target: str, src: dict):
    """Frozen source dictionary + normalizer, target M10 only; no source refit."""
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import data as raw_data, syn3
    frozen=src["rsyn3"]
    required=("basis_dictionary","basis_scale","normalizer_mean","normalizer_scale")
    if set(required)-set(frozen): raise RuntimeError("source cache lacks frozen rSyn3 arrays")
    target_m10=raw_data.load_support_bins(nwb_path(target),emg_trial_stop=10,neural_trial_stop=10)
    emg,_=_rectify(target_m10.emg)
    # ``project_basis`` only needs these NNMF fields.  Calling NNLS directly
    # makes the frozen provenance explicit and cannot trigger an NMF fit.
    scores=syn3.nnls_activations(syn3.apply_scale(emg,frozen["basis_scale"]),frozen["basis_dictionary"])
    w,b=syn3.fit_all_units(scores,target_m10.rates)
    raw=syn3.carrier_from_encoding(w,b)
    carrier=np.ascontiguousarray(syn3.normalize_carriers(raw,frozen["normalizer_mean"],frozen["normalizer_scale"]),dtype=np.float32)
    return _bank(target,carrier)

def _target_rows(model,ema,ds,bank,device: torch.device, target: str):
    """Evaluate EMA and retain exact FalconDataset-derived output coordinates."""
    raw={n:p.detach().clone() for n,p in model.named_parameters()}
    model.eval(); train._ema_apply(model,ema)
    targets=[]; predictions=[]; starts=[]; padded_outputs=[]; query_outputs=[]
    # The prefix is read from FalconDataset's actual padded trial start, never
    # inferred from the nominal 100-bin window or a literal 99.
    prefix=int(np.asarray(ds.trial_start_indices[target])[0])
    if prefix <= 0: raise RuntimeError("target reader lacks an actual Falcon padding prefix")
    with torch.inference_mode():
        for x,y,s,v,window_start in train.loader(ds,42,shuffle=False):
            if set(s)!={target}: raise RuntimeError("target score batch session drift")
            pred=model(x.to(device),bank,input_valid_mask=v.to(device)).cpu().numpy()
            st=window_start.numpy().astype(np.int64)
            po=st + ds.window_size - 1
            qo=po-prefix
            if (qo < 0).any(): raise RuntimeError("target output precedes physical query slice")
            targets.append(y[:,-1].numpy()); predictions.append(pred); starts.append(st); padded_outputs.append(po); query_outputs.append(qo)
    with torch.no_grad():
        for n,p in model.named_parameters(): p.copy_(raw[n])
    return {"target":np.concatenate(targets),"prediction":np.concatenate(predictions),"window_start_padded":np.concatenate(starts),"output_index_padded":np.concatenate(padded_outputs),"output_index_query_relative":np.concatenate(query_outputs),"prefix_bins":np.asarray([prefix],dtype=np.int64)}

def _metric(rows: dict, target: str) -> dict:
    from btransform_unified_v1.r2 import variance_weighted_r2
    value=float(variance_weighted_r2(rows["target"],rows["prediction"]))
    return {"per_session":{target:value},"equal_session_mean":value}

def _write_npz(path: Path, rows: dict) -> str:
    tmp=path.with_suffix(".tmp.npz");np.savez_compressed(tmp,**rows);tmp.replace(path);return _sha(path)

def score(a: argparse.Namespace) -> None:
    receipt, final, selected=_require_receipts(a)
    cache=_cache_path(a)
    if not cache.is_file(): raise FileNotFoundError(f"frozen source cache required: {cache}")
    src=materialize_sources(a.target,cache=cache.parent)
    if receipt.get("actual_source_arrays") != train.source_evidence(src): raise RuntimeError("rematerialized source arrays differ from train receipt")
    if {k:train.array_hash(v) for k,v in src["rsyn3"].items()} != receipt["actual_source_arrays"]["rsyn3_dictionary_and_normalizer"]: raise RuntimeError("frozen source rSyn3 hash mismatch")
    device=torch.device(a.device); model=CrossSessionM1Decoder(a.arm,seed=a.seed).to(device);train._configure(model)
    if a.arm==Z_NONE:
        # Reuse the query-only target path; its bank is irrelevant to Z and no
        # target calibration/carrier tensor is materialized.
        from m1_data import materialize_target
        target=materialize_target(a.target,src,Z_NONE)["target"]; bank=src["banks"][src["sources"][0]]; train._install(model,a.arm,src["banks"],src["calib"])
    else:
        target=_target_dataset(a.target); calib=np.ascontiguousarray(target.calib_trialized_neural_features[a.target][:10],dtype=np.float32)
        bank=_bank(a.target,np.zeros((64,4),dtype=np.float32)) if a.arm==B_ACTIVITY_ONLY else _d_target_bank(a.target,src)
        banks={**src["banks"],a.target:bank}; train._install(model,a.arm,banks,{**src["calib"],a.target:calib})
    from btransform_unified_v1.ema import DecoderEMA
    ema=DecoderEMA(model,train.EMA_DECAY);model.load_state_dict(final["model"],strict=True);ema.load_state_dict(final["ema"])
    e24_rows=_target_rows(model,ema,target,bank,device,a.target); e24=_metric(e24_rows,a.target);e24_hash=_write_npz(a.dest/"target_epoch24_predictions.npz",e24_rows)
    ema.load_state_dict(selected["ema"]);chosen_rows=_target_rows(model,ema,target,bank,device,a.target);chosen=_metric(chosen_rows,a.target);chosen_hash=_write_npz(a.dest/"target_selected_predictions.npz",chosen_rows)
    source_hashes={str(p):_sha(p) for p in (Path(__file__),ROOT/"scripts/cross_session_v1/m1_data.py",ROOT/"src/btransform_unified_v2/cross_session_m1_model.py")}
    train.atomic_json(a.dest/"score_receipt.json",{**train.metadata(a),"status":"COMPLETED","source_selected":receipt["selected"],"target_selected_ema":chosen,"target_epoch24_ema":e24,"target_labels_used_for_selection":False,"score_source_sha256":source_hashes,"frozen_rsyn3_sha256":receipt["actual_source_arrays"]["rsyn3_dictionary_and_normalizer"],"checkpoint_sha256":{p.name:_sha(p) for p in (a.dest/"resume_latest.pt",a.dest/"selected_ema.pt")},"target_audit_arrays":{"selected":{"file":"target_selected_predictions.npz","sha256":chosen_hash},"epoch24":{"file":"target_epoch24_predictions.npz","sha256":e24_hash}},"coordinate_schema":{"window_start_padded":"FalconDataset padded-window start","output_index_padded":"window_start_padded + actual window_size - 1","output_index_query_relative":"output_index_padded - actual trial_start_indices[target][0]","prefix_bins":"actual FalconDataset first trial padded index"}})

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--dest",type=Path,required=True);p.add_argument("--target",choices=FOLDS,required=True);p.add_argument("--arm",choices=ARMS,required=True);p.add_argument("--seed",type=int,default=42);p.add_argument("--device",default="cuda:0");a=p.parse_args()
    if a.seed!=42: raise ValueError("protocol primary seed is 42")
    score(a)
if __name__=="__main__": main()
