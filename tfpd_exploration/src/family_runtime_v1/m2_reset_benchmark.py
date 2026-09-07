"""Read-only paired reset/first-call benchmark for exact M2 zero history.

This is a local runtime diagnostic, not a model-selection or remote request.
Repeated reset timings are warm-process timings; initial construction/load
times are separately recorded and should not be confused with cold OS cache.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .benchmark import M2_PAYLOAD, process_memory, sha, stats
from .m2_cold_start import ColdStartLinearConvM2Decoder
from .m2_linear_conv import LinearConvFiveTokenM2Decoder
from .m2_spint_comparison import ROOT

PAYLOAD_SHA = "4db109e75276d6e8f47df6540b0a4c8f6e955f81af1212127723976795f2eaa4"


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if torch.cuda.is_available():
        raise RuntimeError("requires CUDA_VISIBLE_DEVICES='' at process start")
    payload = ROOT / M2_PAYLOAD
    if sha(payload) != PAYLOAD_SHA:
        raise RuntimeError("frozen payload drift")
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    sessions = sorted((ROOT / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache/source_train").glob("ses-*"))
    if len(sessions) != 7:
        raise RuntimeError("seven source rows required")
    tags, values, sources = [], [], {}
    for session in sessions:
        parts = session.name.removeprefix("ses-").split("-")
        tags.append(f"{parts[-1]}_{''.join(parts[:3])}")
        path = session / "X_store.npy"
        raw = np.load(path, mmap_mode="r")
        if not np.all(raw[:49] == 0):
            raise RuntimeError("source W49 padding drift")
        values.append(np.array(raw[49:57], dtype=np.float32, copy=True))
        sources[session.name] = sha(path)
    data = np.stack(values, axis=1)
    engines, construct, first_reset, zero_error = {}, {}, {}, 0.0
    for name, cls in (("linear_conv", LinearConvFiveTokenM2Decoder), ("zero_history_reset", ColdStartLinearConvM2Decoder)):
        begun = time.perf_counter_ns(); engines[name] = cls(payload, batch_size=7)
        construct[name] = (time.perf_counter_ns() - begun) / 1e6
        begun = time.perf_counter_ns(); engines[name].reset(tags)
        first_reset[name] = (time.perf_counter_ns() - begun) / 1e6
    reset_times = {name: [] for name in engines}
    public_times = {name: [] for name in engines}
    error = 0.0
    with torch.no_grad():
        for repeat in range(args.repeats):
            names = list(engines)
            if repeat % 2:
                names.reverse()
            outputs = {}
            for name in names:
                engine = engines[name]
                begun = time.perf_counter_ns(); engine.reset(tags)
                reset_times[name].append((time.perf_counter_ns() - begun) / 1e6)
                # The first call is timed independently of the reset boundary.
                begun = time.perf_counter_ns(); outputs[name] = [engine.predict(data[0])]
                public_times[name].append((time.perf_counter_ns() - begun) / 1e6)
                for value in data[1:]:
                    outputs[name].append(engine.predict(value))
            for actual, expected in zip(outputs["zero_history_reset"], outputs["linear_conv"], strict=True):
                np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
                error = max(error, float(np.abs(actual - expected).max()))
            # Match full current frontend states, not just the readout, after
            # the same first eight real bins from each independent reset.
            a, b = engines["zero_history_reset"]._engine.frontend, engines["linear_conv"]._engine.frontend
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
            zero_error = max(zero_error, float((a - b).abs().max()))
    result = {
        "schema": "m2_exact_zero_history_reset_paired_v1", "batch": 7,
        "repeats": args.repeats, "torch": torch.__version__, "threads": args.threads,
        "affinity": sorted(os.sched_getaffinity(0)), "payload_sha256": PAYLOAD_SHA,
        "constructor_load_ms": construct, "initial_reset_ms": first_reset,
        "warm_process_reset_ms": {name: stats(samples) for name, samples in reset_times.items()},
        "post_reset_first_public_call_ms": {name: stats(samples) for name, samples in public_times.items()},
        "max_native_first8_abs_error": error, "max_frontend_after8_abs_error": zero_error,
        "runtime_state_bytes": {name: dataclasses.asdict(engine.state_bytes()) for name, engine in engines.items()},
        "joint_process_memory": process_memory(), "joint_memory_not_standalone_decoder_memory": True,
        "source_raw_sha256": sources,
        "code_sha256": {name: sha(Path(__file__).with_name(name)) for name in ("m2_reset_benchmark.py", "m2_cold_start.py", "m2_linear_conv.py", "linear_conv.py", "m2_grouped.py", "grouped_value.py", "m2_lifted.py", "m2.py")},
        "targets_or_nwb_opened": False, "parameter_updates": 0, "not_official_latency": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--repeats", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
