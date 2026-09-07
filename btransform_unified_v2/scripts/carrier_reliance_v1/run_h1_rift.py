"""CPU-only direct H-C carrier reliance diagnostic for frozen H1 RIFT EMA-22.

This is a development-only intervention: E0, raw neural stream, unit mask,
checkpoint and normalizer stay fixed.  It changes the direct four-coordinate
H-C carrier path only, so it cannot identify any H-C information already
embedded in E0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
V1=ROOT.parent / "btransform_unified_v1"
for path in (ROOT/"src", V1/"src", V1/"scripts", ROOT/"scripts"/"rift_v1", Path(__file__).parent):
    if str(path) not in sys.path: sys.path.insert(0,str(path))

import common
from btransform_unified_v1.bank import TaskBank, array_sha256
from btransform_unified_v1.c2_protocol import grouped_session_metrics
from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v1 import h1_config
from btransform_unified_v2 import RiftDecoder
from btransform_unified_v2.cpu_runtime import CpuRiftRuntime

CHECKPOINT=ROOT/"results"/"rift_v1"/"h1_r300_pair_20260907T075312Z"/"recency_20260907_155347"/"formal"/"epoch_022.pt"
ARMS=("REAL","C_ZERO","C_SHUF101","C_SHUF102","C_SHUF103")
ARM_COMMON={"REAL":("normal",None),"C_ZERO":("zero",None),"C_SHUF101":("shuffle",101),"C_SHUF102":("shuffle",102),"C_SHUF103":("shuffle",103)}
SCALE=float(h1_config.TARGET_MULTIPLIER)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(checkpoint: Path) -> RiftDecoder:
    state=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if state.get("smoke") or state.get("variant")!="recency" or state.get("context_bins")!=300:
        raise ValueError("requires formal H1 R300 recency checkpoint")
    model=RiftDecoder("h1",context_bins=300,bias_mode="recency",seed=42).eval()
    model.load_state_dict(state["raw_state_dict"])
    ema=DecoderEMA(model,decay=.9995); ema.load_state_dict(state["ema"]); ema.apply_to(model)
    if model.training: raise RuntimeError("EMA model must be eval")
    return model


def make_bank(record: dict[str, Any], arm: str) -> TaskBank:
    kind,seed=ARM_COMMON[arm]; arrays=common.clone_bank_for_arm(record,kind,seed)
    e0,carrier,mask=arrays["E0"],arrays["carrier"],arrays["unit_mask"]
    # The stores are contract-required but never consumed: score windows are
    # materialized lazily from raw neural only in bounded chunks.
    meta={"shape":list(e0.shape),"trial_count":3,"budget":3,
          "estimator":"C2 HO-M3 frozen bank; direct H-C intervention only",
          "array_sha256":array_sha256(e0),"carrier_arm":arm}
    return TaskBank(record["session"],e0,carrier,mask,np.zeros((1,300,176),np.float32),np.zeros((1,7),np.float32),np.zeros(1,np.int64),meta)


def windows(record: dict[str, Any], endpoints: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Bounded [B,300,176] construction, never an N×300×176 expansion."""
    end=np.asarray(endpoints,np.int64); neural=np.asarray(record["neural"],np.float32)
    x=np.zeros((len(end),300,176),np.float32); valid=np.zeros((len(end),300),bool)
    for row,value in enumerate(end):
        lo=max(0,int(value)-299); n=int(value)-lo+1; x[row,300-n:]=neural[lo:int(value)+1]; valid[row,300-n:]=True
    return x,valid


@torch.inference_mode()
def oracle(model: RiftDecoder, record: dict[str,Any], bank: TaskBank, endpoints: np.ndarray, chunk: int=16) -> np.ndarray:
    rows=[]
    for offset in range(0,len(endpoints),chunk):
        x,valid=windows(record,endpoints[offset:offset+chunk])
        rows.append((model(torch.from_numpy(x),bank,input_valid_mask=torch.from_numpy(valid)).numpy()/SCALE).astype(np.float32))
    return np.concatenate(rows,axis=0)


@torch.inference_mode()
def stream_group(model: RiftDecoder, records: list[dict[str,Any]], banks: list[TaskBank]) -> dict[str,np.ndarray]:
    """Fresh B=2 cached runtime; each row replays only its selected span."""
    ids=[record["key"] for record in records]; starts=[]; selected=[]; last=[]
    for record in records:
        take=np.asarray(record["selected_endpoints"],np.int64)
        if take.size==0: raise ValueError(f"empty fixed surface for {record['key']}")
        start=max(0,int(take.min())-299); starts.append(start); selected.append({int(v-start):i for i,v in enumerate(take)}); last.append(int(take.max())-start)
    out={record["key"]:np.empty((len(record["selected_endpoints"]),7),np.float32) for record in records}
    runtime=CpuRiftRuntime(model,banks,ids,temporal_backend="cached")
    for tick in range(max(last)+1):
        observed=np.zeros((len(records),176),np.float32); valid=np.zeros(len(records),bool)
        for row,record in enumerate(records):
            if tick<=last[row]: observed[row]=record["neural"][starts[row]+tick]; valid[row]=True
        score=runtime.advance(torch.from_numpy(observed),valid_mask=torch.from_numpy(valid)).numpy()/SCALE
        for row,record in enumerate(records):
            index=selected[row].get(tick)
            if index is not None: out[record["key"]][index]=score[row]
    return out


def runtime_oracle_preflight(model: RiftDecoder, surface: dict[str,Any]) -> dict[str,Any]:
    """Validate cached continuous streams against full-window oracle, including reset."""
    checked=[]
    for group in surface["groups"]:
        records=[row for row in surface["records"] if row["group"]==group]
        banks=[make_bank(row,"REAL") for row in records]
        # First two selected endpoints per recording: 28 real endpoints total.
        narrow=[]
        for row in records:
            copied=dict(row); copied["selected_endpoints"]=np.asarray(row["selected_endpoints"][:2],np.int64); narrow.append(copied)
        streamed=stream_group(model,narrow,banks)
        for row,bank in zip(narrow,banks):
            want=oracle(model,row,bank,row["selected_endpoints"]); got=streamed[row["key"]]
            torch.testing.assert_close(torch.from_numpy(got),torch.from_numpy(want),rtol=1e-5,atol=1e-5)
            checked.append({"key":row["key"],"count":len(want),"max_abs":float(np.max(np.abs(got-want)))})
    first=surface["records"][0]; endpoint=np.asarray(first["selected_endpoints"][:1],np.int64)
    arm_parity=[]; real_before=None
    for arm in ARMS:
        bank=make_bank(first,arm); want=oracle(model,first,bank,endpoint)
        single=dict(first); single["selected_endpoints"]=endpoint
        got=stream_group(model,[single],[bank])[first["key"]]
        torch.testing.assert_close(torch.from_numpy(got),torch.from_numpy(want),rtol=1e-5,atol=1e-5)
        if arm=="REAL":
            if real_before is None: real_before=got.copy()
            else: torch.testing.assert_close(torch.from_numpy(got),torch.from_numpy(real_before),rtol=0,atol=0)
        arm_parity.append({"arm":arm,"max_abs":float(np.max(np.abs(got-want)))})
    assert real_before is not None
    restored=stream_group(model,[dict(first, selected_endpoints=endpoint)],[make_bank(first,"REAL")])[first["key"]]
    torch.testing.assert_close(torch.from_numpy(restored),torch.from_numpy(real_before),rtol=0,atol=0)
    arm_parity.append({"arm":"REAL_RESTORED_AFTER_INTERVENTIONS","max_abs":float(np.max(np.abs(restored-real_before)))})
    bank=make_bank(first,"REAL"); ident=first["key"]
    one=np.asarray(first["neural"][:128],np.float32); runtime=CpuRiftRuntime(model,[bank],[ident],temporal_backend="cached")
    times=[]
    for value in one:
        start=time.perf_counter(); runtime.advance(torch.from_numpy(value[None])); times.append(time.perf_counter()-start)
    runtime.reset_rows([ident]); reset_score=runtime.advance(torch.from_numpy(one[0:1]))
    fresh=CpuRiftRuntime(model,[bank],[ident],temporal_backend="cached").advance(torch.from_numpy(one[0:1]))
    torch.testing.assert_close(reset_score,fresh,rtol=1e-5,atol=1e-5)
    return {"passed":True,"records":checked,"real_endpoints":sum(row["count"] for row in checked),
            "reset_semantics":"explicit reset_rows output equals a fresh runtime at the next real bin; each group/arm also creates fresh banks and runtime",
            "all_arm_complete_window_parity_and_real_restore":arm_parity,
            "cpu_no_score_timing":{"advances":len(times),"mean_ms":float(np.mean(times)*1e3),"median_ms":float(np.median(times)*1e3),"p95_ms":float(np.percentile(times,95)*1e3)}}


def summarize(surface: dict[str,Any], predictions: dict[str,np.ndarray]) -> dict[str,Any]:
    targets={}; masks={}; mapping=[]
    for record in surface["records"]:
        key=record["key"]; targets[key]=np.asarray(record["targets"])[record["selected_endpoints"]]; masks[key]=np.ones(len(targets[key]),bool); mapping.append((record["session"],key))
    return grouped_session_metrics(predictions,targets,masks,mapping)


def source_manifest(surface: dict[str,Any], checkpoint: Path) -> dict[str,Any]:
    return {"schema":"rift_h1_carrier_reliance_v1","development_only":True,"official_test_used":False,"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"checkpoint":str(checkpoint.resolve()),"checkpoint_sha256":digest(checkpoint),"weights":"EMA e22 applied; raw state is not scored","arms":list(ARMS),"intervention":"direct normalized H-C carrier only; E0/raw/unit_mask/checkpoint/normalizer fixed","e0_hc_caveat":"E0 retains existing H-C-derived information; this estimates direct-path reliance only","surface":{"query_inventory_sha256":surface["query_inventory_sha256"],"full_valid_inventory_sha256":surface["full_valid_inventory_sha256"],"groups":surface["groups"],"max_endpoints_per_group":surface["max_endpoints_per_group"],"permutation_seeds":surface["permutation_seeds"],"permutations":surface["permutations"],"carrier_zero_semantics":surface["carrier_zero_semantics"],"payload_derivation":surface["payload_derivation"]},"source_sha256":{str(Path(__file__).resolve()):digest(Path(__file__)),str(Path(common.__file__).resolve()):digest(Path(common.__file__))},"cpu":{"affinity":sorted(os.sched_getaffinity(0)),"torch_threads":torch.get_num_threads(),"host_logical_cpus":os.cpu_count(),"exclusive_machine_claim":False}}


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--dest",type=Path,required=True); p.add_argument("--checkpoint",type=Path,default=CHECKPOINT); p.add_argument("--preflight-only",action="store_true"); args=p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("",None): raise RuntimeError("CPU experiment requires CUDA_VISIBLE_DEVICES='' ")
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(42)
    if args.dest.exists(): raise FileExistsError("result destination already exists")
    surface=common.load_surface(max_endpoints_per_group=2048); model=load_model(args.checkpoint.resolve()); manifest=source_manifest(surface,args.checkpoint.resolve())
    args.dest.mkdir(parents=True); (args.dest/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    preflight=runtime_oracle_preflight(model,surface); (args.dest/"preflight.json").write_text(json.dumps(preflight,indent=2,sort_keys=True)+"\n")
    if args.preflight_only: return 0
    indices={record["key"]:np.asarray(record["selected_endpoints"],np.int64) for record in surface["records"]}
    np.savez_compressed(args.dest/"selected_endpoints.npz",**indices)
    all_predictions={}; reports={}
    for arm in ARMS:
        predictions={}
        for group in surface["groups"]:
            records=[row for row in surface["records"] if row["group"]==group]; banks=[make_bank(row,arm) for row in records]
            predictions.update(stream_group(model,records,banks))
            (args.dest/"heartbeat.json").write_text(json.dumps({"status":"RUNNING","arm":arm,"completed_group":group,"utc":datetime.now(timezone.utc).isoformat()},indent=2)+"\n")
        all_predictions[arm]=predictions; reports[arm]=summarize(surface,predictions)
        np.savez_compressed(args.dest/f"predictions_{arm}.npz",**predictions)
        (args.dest/"heartbeat.json").write_text(json.dumps({"status":"RUNNING","completed_arm":arm,"utc":datetime.now(timezone.utc).isoformat()},indent=2)+"\n")
    deltas={arm:{"definition":"REAL_minus_arm, paired within H1 group", "group_equal_mean_delta_vs_real":float(reports["REAL"]["r2_mean"]-reports[arm]["r2_mean"]),"per_group_delta":{g:float(reports["REAL"]["per_session_r2"][g]-reports[arm]["per_session_r2"][g]) for g in surface["groups"]}} for arm in ARMS if arm!="REAL"}
    for arm,row in deltas.items(): row["seven_group_bootstrap"]=common.grouped_bootstrap(row["per_group_delta"])
    shuffle_group_mean={g:float(np.mean([deltas[f"C_SHUF{seed}"]["per_group_delta"][g] for seed in common.PERMUTATION_SEEDS])) for g in surface["groups"]}
    result={"schema":"rift_h1_carrier_reliance_v1","status":"COMPLETED","manifest":manifest,"preflight":preflight,"reports":reports,"paired_deltas":deltas,"shuffle_seed_mean":{"definition":"mean(REAL minus C_SHUF101/102/103) within each group before seven-group bootstrap","per_group_delta":shuffle_group_mean,"seven_group_bootstrap":common.grouped_bootstrap(shuffle_group_mean)},"prediction_files":{arm:f"predictions_{arm}.npz" for arm in ARMS},"selected_endpoint_file":"selected_endpoints.npz","completed_utc":datetime.now(timezone.utc).isoformat()}
    (args.dest/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); (args.dest/"heartbeat.json").write_text(json.dumps({"status":"COMPLETED","utc":result["completed_utc"]},indent=2)+"\n"); print(json.dumps({"status":"COMPLETED","dest":str(args.dest)},indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
