"""Reproducible CPU H1 runtime comparison: trained RIFT versus sealed BT-EORT.

Both arms consume public H1 raw neural windows under two pinned physical CPU
cores.  RIFT is loaded from the named EMA checkpoint; BT-EORT uses its sealed
submission payload and shipped ONNX graphs.  This is an engineering latency
comparison, not a prediction-quality comparison between different models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT.parent / "btransform_unified_v1"
BT = ROOT.parent / "tfpd_exploration" / "submissions" / "evalai_h1_c2_cal1_b2_ort_v1"
for path in (ROOT / "src", V1 / "src", V1 / "scripts", ROOT.parent, BT):
    sys.path.insert(0, str(path))

from btransform_unified_v1.ema import DecoderEMA
from btransform_unified_v2 import RiftDecoder, RiftStreamDecoder
from btransform_unified_v2.cpu_runtime import CpuRiftRuntime
import h1_train


def summary(samples: list[float]) -> dict[str, float | int]:
    a = np.asarray(samples, dtype=np.float64) * 1e3
    return {"n": len(samples), "mean_ms": float(a.mean()), "median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95))}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_name() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def cpu_sharing() -> dict[str, object]:
    quota = None
    for path in (Path("/sys/fs/cgroup/cpu.max"), Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")):
        if path.is_file():
            quota = path.read_text().strip(); break
    return {"host_logical_cpus": os.cpu_count(), "process_affinity": sorted(os.sched_getaffinity(0)),
            "cgroup_cpu_quota": quota, "exclusive_machine_claim": False,
            "note": "benchmark pins this process to two physical CPUs; concurrent host workloads may remain"}


def load_rift(checkpoint: Path) -> RiftDecoder:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("smoke") or state.get("variant") != "recency" or state.get("context_bins") != 300:
        raise ValueError("checkpoint must be formal H1 R300 recency")
    model = RiftDecoder("h1", context_bins=300, bias_mode="recency", seed=42).eval()
    model.load_state_dict(state["raw_state_dict"])
    ema = DecoderEMA(model, decay=.9995); ema.load_state_dict(state["ema"]); ema.apply_to(model)
    return model.eval()


def flows(train: dict, batch: int, length: int) -> tuple[list, np.ndarray, list[str], dict]:
    sessions = train["sessions"][:batch]
    # Pull one shared continuous public raw-neural segment for each selected
    # session; no endpoint windows are concatenated, replayed, or synthesized.
    start = 0
    rows=[]
    for session in sessions:
        neural, _ends, _y = h1_train._source_endpoints("train", session)
        if neural.shape[0] < start + length: raise RuntimeError(f"H1 source too short: {session}")
        rows.append(np.ascontiguousarray(neural[start:start + length], dtype=np.float32))
    receipt=json.loads((BT / "artifacts" / "payload.receipt.json").read_text())
    by_session={row["session"]:tag for tag,row in receipt["banks"]["per_tag"].items()}
    tags=[by_session[session] for session in sessions]
    stacked=np.stack(rows, axis=1)
    manifest={"sessions":sessions,"dataset_tags":tags,"raw_start_inclusive":start,"raw_stop_exclusive":start+length,
              "raw_mean":float(stacked.mean()),"raw_std":float(stacked.std()),"raw_nonzero_fraction":float(np.count_nonzero(stacked)/stacked.size)}
    return [train["banks"][session] for session in sessions], stacked, tags, manifest


def rift_parity(model: RiftDecoder, banks: list, x: np.ndarray) -> dict[str, float | bool]:
    b = len(banks); ids = [f"s{n}" for n in range(b)]
    ref = RiftStreamDecoder(model)
    state = CpuRiftRuntime(model, banks, ids, temporal_backend="state")
    cached = CpuRiftRuntime(model, banks, ids, temporal_backend="cached")
    bank_by_id={key:bank for key,bank in zip(ids,banks)}; max_abs = 0.0; max_ratio = 0.0
    with torch.inference_mode():
        for t, item in enumerate(x):
            order = ids[::-1] if t == 11 else ids
            # The caller reorders physical rows along with stream_ids, then
            # keeps that order for all following observed bins.
            reordered = order != [f"s{n}" for n in range(b)]
            input_ = torch.from_numpy(item[::-1].copy() if reordered else item)
            a = ref.stream_step(input_, [bank_by_id[key] for key in order], order)
            for runtime in (state, cached):
                got = runtime.advance(input_, order if t == 11 else None)
                torch.testing.assert_close(got, a, rtol=1e-5, atol=1e-5)
                delta = (got - a).abs(); max_abs = max(max_abs, float(delta.max()))
                max_ratio = max(max_ratio, float((delta / (1e-5 + 1e-5 * a.abs())).max()))
            ids = order
        # Exercise an asynchronous stream finish/reset under the registered
        # row ordering and compare the first post-reset advance.
        state.reset_rows([ids[-1]]); cached.reset_rows([ids[-1]]); ref.reset(ids[-1])
        item = torch.from_numpy(x[-1][::-1].copy() if ids != [f"s{n}" for n in range(b)] else x[-1])
        a = ref.stream_step(item, [bank_by_id[key] for key in ids], ids)
        for runtime in (state, cached):
            got = runtime.advance(item); torch.testing.assert_close(got, a, rtol=1e-5, atol=1e-5); delta=(got-a).abs(); max_abs=max(max_abs,float(delta.max())); max_ratio=max(max_ratio,float((delta/(1e-5+1e-5*a.abs())).max()))
    return {"passed": True, "max_abs_error": max_abs, "max_tolerance_ratio": max_ratio, "checked_advances":len(x)+1, "reorder_and_async_reset_exercised": True}


def measure_rift(model: RiftDecoder, banks: list, x: np.ndarray, backend: str, warmup: int, steps: int) -> dict:
    """One round, from a NumPy public boundary to native NumPy predictions."""
    ids=[f"s{n}" for n in range(len(banks))]; t0=time.perf_counter()
    if backend == "reference":
        runner=RiftStreamDecoder(model)
        call=lambda item: runner.stream_step(torch.from_numpy(item),banks,ids).numpy()/20.0
    else:
        runner=CpuRiftRuntime(model,banks,ids,temporal_backend=backend)
        call=lambda item: runner.advance(torch.from_numpy(item)).numpy()/20.0
    construct=time.perf_counter()-t0; t0=time.perf_counter(); first=call(x[0]); startup=time.perf_counter()-t0
    if not np.isfinite(first).all(): raise RuntimeError("nonfinite RIFT native prediction")
    for item in x[1:warmup]: call(item)
    samples=[]
    for item in x[warmup:warmup+steps]:
        t0=time.perf_counter(); out=call(item); samples.append(time.perf_counter()-t0)
        if not np.isfinite(out).all(): raise RuntimeError("nonfinite RIFT native prediction")
    return {"construct_and_register":construct, "startup_first_predict":startup, "steady_samples_s":samples, "predict_calls":warmup+steps, "observed_bins":warmup+steps, "temporal_backend":backend}


def measure_bt(x: np.ndarray, batch: int, tags: list[str], warmup: int, steps: int) -> dict:
    os.environ["RT_PACKED_DECODER"] = str(BT / "h1_trf_falcon_decoder.py")
    from falcon_challenge.config import FalconConfig, FalconTask
    from h1_exacte_ort import OrtH1ProjAddFalconDecoder
    payload = BT / "artifacts" / "h1_c2_cal1_b2_s42_ema_e18_L200.pkl"; graphs=BT / "artifacts" / "ort_graphs"
    import h1_trf_falcon_decoder as packed
    missing=set(tags)-set(packed.load_payload(payload)["bank_by_dataset_tag"])
    if missing: raise ValueError(f"sealed BT-EORT payload lacks selected tags: {sorted(missing)}")
    # FalconConfig.hash_dataset strips the final underscore component, so a
    # harmless suffix preserves the sealed payload key after its normalization.
    decoder_tags=[f"{tag}_benchmark" for tag in tags]
    t0=time.perf_counter(); decoder=OrtH1ProjAddFalconDecoder(FalconConfig(task=FalconTask.h1),str(payload),batch_size=batch,graph_dir=graphs,intra_op=2,inter_op=1); decoder.reset(decoder_tags); construct=time.perf_counter()-t0
    t0=time.perf_counter(); first=decoder.predict(x[0]); startup=time.perf_counter()-t0
    if not np.isfinite(first).all(): raise RuntimeError("nonfinite BT-EORT native prediction")
    for item in x[1:warmup]: decoder.predict(item)
    samples=[]
    for item in x[warmup:warmup+steps]:
        t0=time.perf_counter(); out=decoder.predict(item); samples.append(time.perf_counter()-t0)
        if not np.isfinite(out).all(): raise RuntimeError("nonfinite BT-EORT native prediction")
    engine=decoder._engine
    if engine is None or not engine.sess_adv or not engine.sess_reb:
        raise RuntimeError("BT-EORT ONNX sessions were not exercised; refusing eager fallback benchmark")
    expected_calls=warmup+steps
    if decoder._n_predicts != expected_calls: raise RuntimeError("BT-EORT predict counter mismatch")
    return {"construct_and_reset_s":construct, "startup_first_predict_including_ort_session_s":startup, "steady_samples_s":samples, "predict_calls":expected_calls, "actual_predict_counter":decoder._n_predicts, "observed_bins":expected_calls, "ort_version":engine._ort.__version__, "ort_session_batches":{"advance":sorted(engine.sess_adv),"rebuild":sorted(engine.sess_reb)}, "ort_session_init_seconds":engine.session_init_s, "sealed_payload":str(payload), "sealed_payload_sha256":sha256(payload), "onnx_graph_sha256":{p.name:sha256(p) for p in sorted(graphs.glob(f"ort_*_b{batch}.onnx"))}}


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--dest",type=Path,required=True); p.add_argument("--warmup",type=int,default=300); p.add_argument("--steps",type=int,default=100); p.add_argument("--rounds",type=int,default=3); args=p.parse_args()
    if args.warmup < 300 or args.steps < 100 or args.rounds != 3: raise ValueError("require warmup >=300, steps >=100, rounds=3")
    torch.set_num_threads(2); torch.set_num_interop_threads(1); model=load_rift(args.checkpoint.resolve()); train=h1_train.build_train(300)
    cases={}
    for batch in (1,8):
        banks, raw, tags, flow_manifest=flows(train,batch,args.warmup+args.steps)
        parity=rift_parity(model,banks,raw)
        if not parity["passed"]: raise RuntimeError(f"strict RIFT parity failed: {parity}")
        arms={"rift_reference":[],"rift_state":[],"rift_cached":[],"bt_eort_ort":[]}
        order=list(arms)
        for round_index in range(args.rounds):
            rotation=order[round_index:]+order[:round_index]
            for arm in rotation:
                if arm == "bt_eort_ort": result=measure_bt(raw,batch,tags,args.warmup,args.steps)
                else: result=measure_rift(model,banks,raw,arm.removeprefix("rift_"),args.warmup,args.steps)
                arms[arm].append(result)
        def compact(rows):
            per_round=[]; samples=[]
            for row in rows:
                row=dict(row); one=row.pop("steady_samples_s"); samples.extend(one); row["steady_predict"]=summary(one); per_round.append(row)
            return {"per_round":per_round,"steady_predict":summary(samples)}
        cases[f"B{batch}"]={"public_h1_raw_flow":flow_manifest, "rift_parity":parity, **{key:compact(value) for key,value in arms.items()}}
    report={"schema":"h1_r300_trained_rift_vs_bt_eort_cpu_v1","utc":datetime.now(timezone.utc).isoformat(),"device":"cpu","cpu":cpu_name(),"affinity":sorted(os.sched_getaffinity(0)),"cpu_sharing":cpu_sharing(),"threads":{"torch_intra":2,"torch_interop":1,"ort_intra":2,"ort_inter":1},"torch":torch.__version__,"dtype":"float32","autograd":False,"checkpoint":str(args.checkpoint.resolve()),"checkpoint_sha256":sha256(args.checkpoint.resolve()),"rift_weights":"EMA applied from formal H1 R300 recency checkpoint", "protocol":{"rounds":args.rounds,"warmup":args.warmup,"measured_steps_each_round":args.steps,"cold_start":True,"window_wrap":True,"stream_reorder":True,"async_reset":True},"cases":cases}
    args.dest.mkdir(parents=True,exist_ok=False); (args.dest/"benchmark.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); print(json.dumps(report,indent=2)); return 0

if __name__ == "__main__": raise SystemExit(main())
