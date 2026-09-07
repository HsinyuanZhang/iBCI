"""Pinned-process M2 E-path timing receipt on real source continuous streams.

Invocation is intentionally externalized with ``taskset`` by the lease owner.
No score is inferred from this benchmark; scoring lives in validate.py.
"""
from __future__ import annotations

import argparse, json, os, platform, statistics, time, hashlib
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from tfpd_exploration.src.m2_dual_track_v1 import data, plan
from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime import constants as C
from tfpd_exploration.src.two_mainlines_long_v1.m2_runtime.load_weights import load_pick
from .decoder import M2ExactOptimizedDecoder

OUT = C.REPO_ROOT / "tfpd_exploration/results/decoder_validation_v2/20260905_190000/m2"

def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _quantiles(ms: list[float]) -> dict[str, float]:
    a = np.asarray(ms, dtype=np.float64)
    return {"p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)), "max_ms": float(a.max()), "mean_ms": float(a.mean())}

def _run(kind: str, batch: int, variant: str, calls: int) -> dict:
    model, meta = load_pick(kind, device="cpu")
    sessions = list(plan.HELDIN_SESSIONS[:batch])
    banks = [data.load_session_bank("source_minival", s, device="cpu") for s in sessions]
    # Real neural recordings, simultaneously one trace per legal evaluator slot.
    traces = [np.asarray(b.X_store, dtype=np.float32) for b in banks]
    # Minival source recordings are 165--243 bins, shorter than the required
    # 2,048-call sustained timing cell.  Cycle each *real recorded* trace with
    # no synthetic values; wrap points are timing-only discontinuities, never
    # used for an R2 claim (validate.py owns exact score streams).
    buf = np.zeros((C.WINDOW, batch, C.CHANNELS), dtype=np.float32)
    opt = M2ExactOptimizedDecoder(model, banks) if variant == "optimized" else None
    def step(t: int, initialized: bool) -> None:
        nonlocal buf
        observation = np.stack([trace[t % len(trace)] for trace in traces], axis=0)
        # Same online observation rolling/copy-to-torch boundary as the Falcon runtime;
        # M2 frozen picks declare smooth_observations=False.
        buf[:-1] = buf[1:]
        buf[-1] = observation
        x = torch.from_numpy(np.array(buf.transpose(1, 0, 2), copy=True))
        with torch.inference_mode():
            if variant == "reference":
                # Original official runtime semantics: per-active-session forward.
                prediction = torch.cat([model.forward_last(x[i:i+1], b, b.unit_mask) for i, b in enumerate(banks)], dim=0)
            elif initialized:
                prediction = opt.advance(x[:, -1:])
            else:
                prediction = opt.rebuild(x)
        # Match the official adapter's terminal output conversion and validation.
        native = prediction.cpu().numpy() / C.BEHAVIOR_SCALE
        if not np.isfinite(native).all():
            raise RuntimeError("non-finite benchmark prediction")
        return native.astype(np.float32, copy=False)
    # Startup: zero-padded window first, then true history fill.  Reset and cold
    # compilation are separately recorded, never hidden in steady-step percentiles.
    cold0 = time.perf_counter_ns(); opt2 = M2ExactOptimizedDecoder(model, banks) if variant == "optimized" else None
    cold_ms = (time.perf_counter_ns()-cold0)/1e6
    reset0 = time.perf_counter_ns(); buf.fill(0.0); (opt.invalidate() if opt else None)
    reset_ms = (time.perf_counter_ns()-reset0)/1e6
    for t in range(C.WINDOW): step(t, initialized=(variant == "optimized" and t > 0))
    # Ensure the optimized cache has a full true window (the first rebuild was all-zero/startup).
    if variant == "optimized": opt.rebuild(torch.from_numpy(np.array(buf.transpose(1,0,2), copy=True)))
    timings: list[float] = []
    for t in range(C.WINDOW, C.WINDOW + calls):
        t0 = time.perf_counter_ns(); step(t, initialized=True); timings.append((time.perf_counter_ns()-t0)/1e6)
    return {"kind": kind, "batch": batch, "variant": variant, "calls": calls, "cold_compile_ms": cold_ms,
            "reset_ms": reset_ms, "startup_warmup_calls": C.WINDOW, "step_wall": _quantiles(timings),
            "sessions": sessions, "weight_sha256": meta["weight_sha256"],
            "stream": "source_minival real neural observations, cyclic replay only because source traces < 2048 bins"}

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--calls", type=int, default=128); p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--tag", default="probe"); args=p.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    records=[]
    for repeat in range(args.repeats):
        for kind in ("small", "large"):
            for batch in (1, 7):
                for variant in ("reference", "optimized"):
                    row=_run(kind,batch,variant,args.calls); row["repeat"]=repeat; records.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)
    receipt={"schema":"m2_optimized_v2_cpu_benchmark_v1","lease":"CPU sequential taskset 8-11", "affinity":sorted(os.sched_getaffinity(0)),
             "torch":torch.__version__,"torch_threads":torch.get_num_threads(),"interop_threads":torch.get_num_interop_threads(),
             "python":platform.python_version(),"platform":platform.platform(),"cpu_model":_cpu_model(),"calls_per_repeat":args.calls,"repeats":args.repeats,
             "measurement":"end-to-end adapter-step wall: buffer update, torch conversion, model, CPU NumPy conversion, /5, finite check, return; batch wall is not divided by batch",
             "code_sha256":{"benchmark.py":_sha(Path(__file__)),"decoder.py":_sha(Path(__file__).with_name("decoder.py"))},"records":records}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/f"cpu_{args.tag}.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")

if __name__ == "__main__": main()
