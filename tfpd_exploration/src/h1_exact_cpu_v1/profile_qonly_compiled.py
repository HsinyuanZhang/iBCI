"""Isolated 128-call CPU receipt for the static-carrier, Q-only compiled E path.

This is a backbone-only prototype receipt.  It is deliberately not an API
claim for any subsequent fitted affine readout: its model/state and public
readout are exactly the sealed V4 FULL control named below.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import triton

from tfpd_exploration.src.h1_optimized_v2.cache import build_or_load
from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2.benchmark_stream import FullWindowAPI
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank

from .backend import CompiledStaticCarrierQOnlyExactFullWindowStream
from .profile_static import CALLS, OUT, STATE, WARMUP, model, q, r2, run, sha
from .static_frontend import StaticCarrierExactFullWindowStream
from .static_frontend_qonly import StaticCarrierQOnlyExactFullWindowStream


def _run_compiled(model_instance, bank, session: str, trace: np.ndarray) -> tuple[dict, np.ndarray]:
    started = time.perf_counter_ns()
    stream = CompiledStaticCarrierQOnlyExactFullWindowStream(
        model_instance, bank, task="h1", session_id=session, unit_ids=range(trace.shape[1]),
        compile_backend=True,
    )
    constructor_ms = (time.perf_counter_ns() - started) / 1e6
    started = time.perf_counter_ns(); stream.reset(); reset_ms = (time.perf_counter_ns() - started) / 1e6
    warmup_ms = []
    for index in range(WARMUP):
        started = time.perf_counter_ns(); stream.predict(trace[index][None])
        warmup_ms.append((time.perf_counter_ns() - started) / 1e6)
    timing_ms, predictions = [], []
    for index in range(CALLS):
        started = time.perf_counter_ns()
        predictions.append(stream.predict(trace[WARMUP + index][None])[0])
        timing_ms.append((time.perf_counter_ns() - started) / 1e6)
    return {
        "ctor_ms": constructor_ms,
        "reset_ms": reset_ms,
        "compile_wrapper_ms_after_reset": stream.compile_ms,
        "backend_kind": stream.backend_kind,
        "compile_error": stream.compile_error,
        "warmup_first_call_ms_includes_kernel_compile": warmup_ms[0],
        "warmup_p50_ms": float(np.percentile(warmup_ms, 50)),
        "timed": q(timing_ms),
    }, np.asarray(predictions, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cache = build_or_load()
    session = next(iter(cache["minival"]))
    row = cache["minival"][session]
    trace = np.asarray(row["neural"], dtype=np.float32)
    target = np.asarray(row["velocity"][WARMUP:WARMUP + CALLS], dtype=np.float32)
    bank = H1Bank(*[row["bank"][key] for key in ("E0", "T", "unit_mask")])
    full, pfull = run(FullWindowAPI, model(), bank, session, trace)
    static, pstatic = run(StaticCarrierExactFullWindowStream, model(), bank, session, trace)
    qonly, pqonly = run(StaticCarrierQOnlyExactFullWindowStream, model(), bank, session, trace)
    compiled, pcompiled = _run_compiled(model(), bank, session, trace)
    parity = {
        "full_vs_compiled_max_abs": float(np.max(np.abs(pfull - pcompiled))),
        "static_vs_compiled_max_abs": float(np.max(np.abs(pstatic - pcompiled))),
        "qonly_vs_compiled_max_abs": float(np.max(np.abs(pqonly - pcompiled))),
        "gate": "<=1e-5",
    }
    parity["passes"] = max(v for k, v in parity.items() if k.endswith("max_abs")) <= 1e-5
    r2_values = {"full": r2(pfull, target), "static": r2(pstatic, target),
                 "qonly": r2(pqonly, target), "compiled": r2(pcompiled, target)}
    r2_values["qonly_compiled_abs_delta"] = abs(r2_values["qonly"] - r2_values["compiled"])
    r2_values["gate"] = "<=1e-5"
    r2_values["passes"] = r2_values["qonly_compiled_abs_delta"] <= 1e-5
    here = Path(__file__).resolve().parent
    receipt = {
        "schema": "h1_exact_cpu_v1_static_carrier_qonly_compiled_profile_v1",
        "scope": "128 real-source calls; backbone-only V4 FULL prototype, not a later affine-readout API receipt",
        "affinity": sorted(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "platform": platform.platform(), "python_executable": sys.executable,
        "torch": {"version": torch.__version__, "origin": torch.__file__},
        "triton": {"version": triton.__version__, "origin": triton.__file__},
        "calls": CALLS, "warmup_calls": WARMUP, "session": session,
        "state_sha256": sha(STATE),
        "code_sha256": {name: sha(here / name) for name in (
            "backend.py", "static_frontend.py", "static_frontend_qonly.py", "profile_qonly_compiled.py"
        )},
        "measurement": "whole public native API wall; cold construction/reset/warmup separately recorded; B1 only",
        "original_full_recompute": full, "static_carrier_exact_e": static,
        "static_carrier_qonly_exact_e": qonly, "static_carrier_qonly_compiled_exact_e": compiled,
        "native_parity": parity, "r2": r2_values,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"profile128_threads{args.threads}_static_carrier_qonly_compiled_v1.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"path": str(path), "backend": compiled["backend_kind"], "parity": parity,
                      "r2": r2_values, "p95_ms": {"full": full["timed"]["p95_ms"],
                      "static": static["timed"]["p95_ms"], "qonly": qonly["timed"]["p95_ms"],
                      "compiled": compiled["timed"]["p95_ms"]}}, sort_keys=True))


if __name__ == "__main__":
    main()
