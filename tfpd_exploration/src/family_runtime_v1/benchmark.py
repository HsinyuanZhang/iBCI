"""Same-process paired public latency on continuous local source bins.

No targets, model selection, retraining, official requests or package writes.
This measures an exact implementation change against V3, not against SPINT.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .m1 import FiveTokenCurrentQueryStream
from .m2 import FiveTokenM2Decoder
from .m1_lifted import LiftedFiveTokenCurrentQueryStream
from tfpd_exploration.src.m1_runtime_v3 import HeterogeneousCurrentQueryStream
from tfpd_exploration.src.m2_runtime_v3.runtime import RuntimeV3Decoder


M2_PAYLOAD = Path("tfpd_exploration/submissions/evalai_m2_small_trf_ext6_epochpick_v1/artifacts/m2_small_trf_s42_ema_e08_ext6.pkl")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(values):
    return {"mean_ms": float(np.mean(values)), "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)), "p99_ms": float(np.percentile(values, 99)), "max_ms": float(max(values))}


def process_memory():
    fields = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, number, _ = line.split()
            fields[key.rstrip(":") + "_bytes"] = int(number) * 1024
    return fields


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    count = 1 + args.calls + args.warmup
    if args.task == "m1":
        from tfpd_exploration.src.m1_runtime_v3.public_benchmark import material
        model, bank, _, values, selected, receipt, names = material(4, count)
        types = (HeterogeneousCurrentQueryStream, LiftedFiveTokenCurrentQueryStream if args.variant == "lifted" else FiveTokenCurrentQueryStream)
        t0 = time.perf_counter_ns()
        baseline = types[0](model, bank)
        cold_base = (time.perf_counter_ns() - t0) / 1e6
        t0 = time.perf_counter_ns()
        candidate = types[1](model, bank)
        cold_candidate = (time.perf_counter_ns() - t0) / 1e6
        authority = {"state_sha256": selected["plain_ema_model_state_sha256"], "source_cache_sha256": receipt["npz_sha256"], "rows": names, "batch": 4, "fourth_lane": "repeat first source session, not fourth distinct bank"}
    else:
        if args.variant == "lifted":
            from .m2_lifted import LiftedFiveTokenM2Decoder
            candidate_type = LiftedFiveTokenM2Decoder
        else:
            candidate_type = FiveTokenM2Decoder
        source = Path("tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache/source_train")
        sessions = sorted(source.glob("ses-*"))
        if len(sessions) != 7:
            raise RuntimeError("seven M2 source sessions required")
        tags, rows, hashes = [], [], {}
        for session in sessions:
            bits = session.name.removeprefix("ses-").split("-")
            tags.append(f"{bits[-1]}_{''.join(bits[:3])}")
            path = session / "X_store.npy"
            neural = np.load(path, mmap_mode="r")
            # The cache contains W-1 raw startup padding, removed exactly once.
            if not np.all(neural[:49] == 0):
                raise RuntimeError("M2 source startup padding contract drift")
            row = np.array(neural[49:49 + count], dtype=np.float32, copy=True)
            if row.shape != (count, 96):
                raise RuntimeError("insufficient continuous source bins; no repetition allowed")
            rows.append(row)
            hashes[session.name] = sha(path)
        values = np.stack(rows, axis=1)
        t0 = time.perf_counter_ns()
        baseline = RuntimeV3Decoder(M2_PAYLOAD, batch_size=7)
        baseline.reset(tags)
        cold_base = (time.perf_counter_ns() - t0) / 1e6
        t0 = time.perf_counter_ns()
        candidate = candidate_type(M2_PAYLOAD, batch_size=7)
        candidate.reset(tags)
        cold_candidate = (time.perf_counter_ns() - t0) / 1e6
        authority = {"payload": str(M2_PAYLOAD), "payload_sha256": sha(M2_PAYLOAD), "neural_sha256": hashes, "rows": tags, "batch": 7}
    if len(values) != count:
        raise RuntimeError("insufficient continuous stream; repetition is not allowed")
    times = {"v3": [], "candidate": []}
    first_call_ms = {}
    memory_before = process_memory()
    max_error = 0.
    with torch.no_grad():
        for index, value in enumerate(values):
            predictions = {}
            # Alternate order to avoid consistently privileging one warmed path.
            order = (("v3", baseline), ("candidate", candidate))
            if index % 2:
                order = order[::-1]
            for name, engine in order:
                t0 = time.perf_counter_ns()
                predictions[name] = engine.predict(value)
                elapsed = (time.perf_counter_ns() - t0) / 1e6
                prediction = predictions[name]
                if prediction.dtype != np.float32 or not prediction.flags.c_contiguous or not prediction.flags.owndata or not np.isfinite(prediction).all():
                    raise RuntimeError("public native output ownership/dtype/finite contract drift")
                if index == 0:
                    first_call_ms[name] = elapsed
                if index >= 1 + args.warmup:
                    times[name].append(elapsed)
            difference = np.abs(predictions["candidate"] - predictions["v3"])
            max_error = max(max_error, float(difference.max()))
            if not np.all(difference <= 1e-5 + 1e-5 * np.abs(predictions["v3"])):
                raise RuntimeError(f"native parity failure at observation {index}")
    metrics = {name: stats(xs) for name, xs in times.items()}
    persistent = {}
    for name, engine in (("v3", baseline), ("candidate", candidate)):
        state = engine.state_bytes
        state = state() if callable(state) else state
        persistent[name] = dataclasses.asdict(state) if dataclasses.is_dataclass(state) else state
    result = {"schema": "family_exact_runtime_public_paired_v2", "task": args.task,
              "calls": args.calls, "warmup": args.warmup, "authority": authority, "candidate_variant": args.variant,
              "max_native_abs_error": max_error, "native_parity_pass": True,
              "public_whole_batch": metrics, "p95_speedup_vs_v3": metrics["v3"]["p95_ms"] / metrics["candidate"]["p95_ms"],
              "cold_v3_ms": cold_base, "cold_candidate_ms": cold_candidate,
              "first_public_call_ms": first_call_ms, "first_call_excluded_from_warmup_and_timing": True,
              "runtime_state_bytes": persistent, "paired_process_memory_before": memory_before,
              "paired_process_memory_after": process_memory(),
              "torch": torch.__version__, "threads": args.threads, "affinity": sorted(os.sched_getaffinity(0)),
              "python_executable": sys.executable, "python_user_site_disabled": bool(sys.flags.no_user_site),
              "torch_module_path": str(torch.__file__),
              "code_sha256": {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
              "code_hash_scope": "directory snapshot; unimported opposite-task modules may differ while their owner works; not all listed modules execute",
              "not_spint_comparison": True, "not_container_or_official_acceptance": True,
              "host_not_exclusive": True, "concurrent_work_scope": "other task training/validation pinned outside CPU0-3; benchmark pinned separately; shared-host effects remain possible",
              "continuous_stream": True, "labels_opened": False, "parameter_updates": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("m1", "m2"), required=True)
    parser.add_argument("--variant", choices=("five_token", "lifted"), default="five_token")
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    parser.add_argument("--calls", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
