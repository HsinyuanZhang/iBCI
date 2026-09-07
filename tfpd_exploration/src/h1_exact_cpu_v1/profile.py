"""Bounded real-stream CPU profile for the isolated exact H1 backend.

This is intentionally a 128-call exploratory comparison, not a replacement
for any formal sustained timing receipt.  It uses the sealed V4 FULL epoch-12
EMA standalone state and one real chronological source-minival stream.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from tfpd_exploration.src.h1_optimized_v2.cache import ROOT, build_or_load
from tfpd_exploration.src.h1_optimized_v4.model import H1SignedFull
from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2 import ExactFullWindowStream
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank

from .backend import CompiledExactFullWindowStream


OUT = ROOT / "exact_cpu_v1"
STATE = ROOT / "paired_v4_signed_12ep_v1/independent_score_export/full_epoch12_plain_ema_model_state.pt"
CALLS, WARMUP = 128, 50


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {key: float(np.percentile(array, percentile)) for key, percentile in
            (("p50_ms", 50), ("p95_ms", 95), ("p99_ms", 99))} | {"mean_ms": float(array.mean())}


def _r2(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(1.0 - np.square(prediction - target).sum() /
                 np.square(target - target.mean(axis=0, keepdims=True)).sum())


def _model() -> H1SignedFull:
    model = H1SignedFull().eval()
    model.load_state_dict(torch.load(STATE, map_location="cpu", weights_only=True), strict=True)
    return model


def _run(stream_cls, *, model, bank, session: str, trace: np.ndarray, target: np.ndarray, compiled: bool) -> tuple[dict, np.ndarray]:
    created = time.perf_counter_ns()
    stream = (stream_cls(model, bank, task="h1", session_id=session, unit_ids=range(trace.shape[1]), compile_backend=True)
              if compiled else stream_cls(model, bank, task="h1", session_id=session, unit_ids=range(trace.shape[1])))
    ctor_ms = (time.perf_counter_ns() - created) / 1e6
    reset = time.perf_counter_ns(); stream.reset(); reset_ms = (time.perf_counter_ns() - reset) / 1e6
    warm = []
    for index in range(WARMUP):
        started = time.perf_counter_ns(); stream.predict(trace[index % len(trace)][None])
        warm.append((time.perf_counter_ns() - started) / 1e6)
    elapsed, prediction = [], []
    for index in range(CALLS):
        started = time.perf_counter_ns()
        prediction.append(stream.predict(trace[(WARMUP + index) % len(trace)][None])[0])
        elapsed.append((time.perf_counter_ns() - started) / 1e6)
    record = {"ctor_ms": ctor_ms, "reset_ms": reset_ms, "startup_first_call_ms": warm[0],
              "startup_p50_ms": float(np.percentile(warm, 50)), "timed": _quantiles(elapsed)}
    if compiled:
        record.update({"backend_kind": stream.backend_kind, "compile_ms": stream.compile_ms,
                       "compile_error": stream.compile_error})
    return record, np.asarray(prediction, dtype=np.float32)


def main() -> None:
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    cache = build_or_load()
    session = next(iter(cache["minival"]))
    row = cache["minival"][session]
    trace = np.asarray(row["neural"], dtype=np.float32)
    target = np.asarray(row["velocity"][WARMUP:WARMUP + CALLS], dtype=np.float32)
    bank = H1Bank(*[row["bank"][key] for key in ("E0", "T", "unit_mask")])
    baseline, reference = _run(ExactFullWindowStream, model=_model(), bank=bank, session=session,
                               trace=trace, target=target, compiled=False)
    candidate, actual = _run(CompiledExactFullWindowStream, model=_model(), bank=bank, session=session,
                             trace=trace, target=target, compiled=True)
    max_abs = float(np.max(np.abs(actual - reference)))
    baseline_r2, candidate_r2 = _r2(reference, target), _r2(actual, target)
    receipt = {"schema": "h1_exact_cpu_v1_exploratory_profile_v1", "scope": "128 real-source calls only; not a 2048 formal receipt",
               "affinity": sorted(os.sched_getaffinity(0)), "torch_threads": torch.get_num_threads(),
               "interop_threads": torch.get_num_interop_threads(), "platform": platform.platform(),
               "calls": CALLS, "startup_warmup_calls": WARMUP, "session": session,
               "stream": "real chronological source-minival observations; no synthetic values",
               "state_sha256": _sha(STATE), "code_sha256": {"backend.py": _sha(Path(__file__).with_name("backend.py")),
               "profile.py": _sha(Path(__file__))}, "baseline": baseline, "candidate": candidate,
               "native_parity": {"max_abs": max_abs, "gate": "<=1e-5", "passes": max_abs <= 1e-5},
               "r2": {"baseline": baseline_r2, "candidate": candidate_r2, "abs_delta": abs(candidate_r2 - baseline_r2),
                      "gate": "<=1e-5", "passes": abs(candidate_r2 - baseline_r2) <= 1e-5}}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "profile128_threads2_trace_candidate_v1.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"path": str(path), "parity": receipt["native_parity"], "r2": receipt["r2"],
                      "baseline_p95": baseline["timed"]["p95_ms"], "candidate_p95": candidate["timed"]["p95_ms"]}))


if __name__ == "__main__":
    main()
