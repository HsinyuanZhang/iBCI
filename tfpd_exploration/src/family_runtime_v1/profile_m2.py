"""Read-only cost profile for the current exact lifted M2 public stream."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .m2_lifted import LiftedFiveTokenM2Decoder


ROOT = Path(__file__).resolve().parents[3]
PAYLOAD = ROOT / (
    "tfpd_exploration/submissions/evalai_m2_small_trf_ext6_epochpick_v1/"
    "artifacts/m2_small_trf_s42_ema_e08_ext6.pkl"
)
SOURCE = ROOT / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache/source_train"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_values(count: int) -> tuple[list[str], np.ndarray, dict[str, str]]:
    sessions = sorted(SOURCE.glob("ses-*"))
    if len(sessions) != 7:
        raise RuntimeError("exactly seven source sessions required")
    tags, rows, hashes = [], [], {}
    for session in sessions:
        parts = session.name.removeprefix("ses-").split("-")
        tags.append(f"{parts[-1]}_{''.join(parts[:3])}")
        path = session / "X_store.npy"
        raw = np.load(path, mmap_mode="r")
        if raw.ndim != 2 or raw.shape[1] != 96 or not np.all(raw[:49] == 0):
            raise RuntimeError(f"{session.name}: source raw-stream contract drift")
        values = np.array(raw[49:49 + count], dtype=np.float32, copy=True)
        if values.shape != (count, 96):
            raise RuntimeError(f"{session.name}: insufficient continuous source bins")
        rows.append(values)
        hashes[session.name] = sha(path)
    return tags, np.stack(rows, axis=1), hashes


def summarize_ns(values: list[int]) -> dict[str, float]:
    milliseconds = np.asarray(values, dtype=np.float64) / 1.0e6
    return {
        "mean_ms": float(milliseconds.mean()),
        "p50_ms": float(np.percentile(milliseconds, 50)),
        "p95_ms": float(np.percentile(milliseconds, 95)),
        "min_ms": float(milliseconds.min()),
        "max_ms": float(milliseconds.max()),
    }


def final_q_readout(engine, hidden: torch.Tensor) -> torch.Tensor:
    """The exact final-Q/readout suffix, copied solely for component timing."""
    model = engine.model
    block = model.temporal.blocks[-1]
    normed = block.norm1(hidden)
    attention = block.attn
    batch, width, dim = normed.shape
    q_weight, kv_weight = attention.qkv.weight[:dim], attention.qkv.weight[dim:]
    q_bias, kv_bias = attention.qkv.bias[:dim], attention.qkv.bias[dim:]
    query = F.linear(normed[:, -1:], q_weight, q_bias).view(
        batch, 1, attention.n_heads, attention.head_dim
    ).transpose(1, 2)
    key_value = F.linear(normed, kv_weight, kv_bias).view(
        batch, width, 2, attention.n_heads, attention.head_dim
    )
    key, value = key_value.unbind(dim=2)
    key, value = key.transpose(1, 2), value.transpose(1, 2)
    attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    attended = attention.proj(attended.transpose(1, 2).contiguous().view(batch, 1, dim))
    last = hidden[:, -1:] + attended
    last = last + block.ffn(block.norm2(last))
    return model.readout(model.final_norm(last))[:, 0, :]


def temporal_prefix(engine, frontend: torch.Tensor) -> torch.Tensor:
    model = engine.model
    hidden = frontend + model.temporal.pe[: frontend.size(1)].unsqueeze(0).to(
        device=frontend.device, dtype=frontend.dtype
    )
    for block in model.temporal.blocks[:-1]:
        hidden = block(hidden)
    return hidden


def instrumented_calls(decoder, values: np.ndarray, warmup: int) -> dict[str, dict[str, float]]:
    engine = decoder._engine
    original_refresh, original_repair, original_last = (
        engine._refresh_if_mutated, engine._repair_frontend, engine._last
    )
    measured: dict[str, list[int]] = {key: [] for key in (
        "public_total", "mutation_audit", "frontend_repair", "temporal_last",
        "temporal_three_full_blocks", "temporal_final_q_readout",
    )}
    active: dict[str, int] = {}

    def timed(name, function):
        def wrapped(*args, **kwargs):
            start = time.perf_counter_ns()
            value = function(*args, **kwargs)
            elapsed = time.perf_counter_ns() - start
            if active["record"]:
                measured[name].append(elapsed)
            return value
        return wrapped

    engine._refresh_if_mutated = timed("mutation_audit", original_refresh)
    engine._repair_frontend = timed("frontend_repair", original_repair)
    engine._last = timed("temporal_last", original_last)
    try:
        with torch.no_grad():
            for index, value in enumerate(values):
                active["record"] = index >= warmup
                start = time.perf_counter_ns()
                output = decoder.predict(value)
                elapsed = time.perf_counter_ns() - start
                if active["record"]:
                    measured["public_total"].append(elapsed)
                    raw = engine.raw
                    frontend = engine.frontend
                    start = time.perf_counter_ns()
                    prefix = temporal_prefix(engine, frontend)
                    measured["temporal_three_full_blocks"].append(time.perf_counter_ns() - start)
                    # The independently timed suffix receives the same frontend;
                    # this is component profiling, not a changed public path.
                    start = time.perf_counter_ns()
                    direct = final_q_readout(engine, prefix)
                    measured["temporal_final_q_readout"].append(time.perf_counter_ns() - start)
                    if not np.isfinite(output).all() or not torch.isfinite(prefix).all() or not torch.isfinite(direct).all():
                        raise RuntimeError("non-finite profile output")
    finally:
        engine._refresh_if_mutated = original_refresh
        engine._repair_frontend = original_repair
        engine._last = original_last
    result = {name: summarize_ns(values) for name, values in measured.items()}
    result["public_unattributed_host_dispatch_advance_ms"] = {
        key: result["public_total"][key] - result["mutation_audit"][key]
        - result["frontend_repair"][key] - result["temporal_last"][key]
        for key in ("mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms")
    }
    return result


def aten_profile(tags: list[str], values: np.ndarray, warmup: int) -> list[dict[str, object]]:
    decoder = LiftedFiveTokenM2Decoder(PAYLOAD, batch_size=7)
    decoder.reset(tags)
    with torch.no_grad():
        for value in values[:warmup]:
            decoder.predict(value)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
            decoder.predict(values[warmup])
    rows = []
    for event in profiler.key_averages():
        rows.append({
            "aten": event.key,
            "self_cpu_time_us": float(event.self_cpu_time_total),
            "cpu_time_us": float(event.cpu_time_total),
            "calls": int(event.count),
        })
    return sorted(rows, key=lambda row: row["self_cpu_time_us"], reverse=True)[:25]


def run(args) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    tags, values, neural_hashes = source_values(args.warmup + args.calls)
    decoder = LiftedFiveTokenM2Decoder(PAYLOAD, batch_size=7)
    decoder.reset(tags)
    costs = instrumented_calls(decoder, values, args.warmup)
    operations = aten_profile(tags, values, args.warmup)
    result = {
        "schema": "family_runtime_v1_m2_lifted_component_profile_v1",
        "status": "PROFILE_ONLY_NO_IMPLEMENTATION_CHANGE",
        "payload_sha256": sha(PAYLOAD),
        "source_neural_sha256": neural_hashes,
        "tags": tags,
        "batch": 7,
        "warmup": args.warmup,
        "measured_public_calls": args.calls,
        "continuous_source_bins_only": True,
        "labels_or_targets_opened": False,
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "component_costs": costs,
        "one_public_call_aten_top25": operations,
        "code_sha256": {Path(__file__).name: sha(Path(__file__)), "m2_lifted.py": sha(Path(__file__).with_name("m2_lifted.py"))},
        "parameter_updates": 0,
        "no_quantization_amp_or_window_change": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--calls", type=int, default=128)
    run(parser.parse_args())
