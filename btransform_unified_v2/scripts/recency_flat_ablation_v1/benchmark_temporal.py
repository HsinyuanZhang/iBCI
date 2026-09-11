#!/usr/bin/env python3
"""CPU-only synthetic temporal-core benchmark for RIFT recency versus flat.

This intentionally measures only ``RiftTemporal``.  It does not load a
dataset, checkpoint, decoder, or trained weight, and it makes no network or
GPU call.  The JSON's ``scope`` field is the accompanying README: the output
is an engineering latency/allocation observation, never an end-to-end or
prediction-quality claim.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
V1 = WORKSPACE / "btransform_unified_v1"
for path in (ROOT / "src", V1 / "src", V1 / "scripts", WORKSPACE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from btransform_unified_v2.config import DEFAULT_HALF_LIVES, RiftTemporalConfig
from btransform_unified_v2.cpu_temporal import CpuRiftTemporalRuntime
from btransform_unified_v2.temporal import RiftTemporal, RiftTemporalState


SEED = 20260910
CONTEXTS = (50, 100, 300)
BATCHES = (1, 4, 8)
WIDTH, HEADS, FFN_WIDTH, LAYERS = 256, 8, 512, 4
ROUNDS, ONLINE_STEPS, FULL_FORWARD_STEPS = 5, 128, 32


def require(condition: bool, text: str) -> None:
    if not condition:
        raise RuntimeError(text)


def parse_affinity(text: str | None) -> list[int]:
    allowed = sorted(os.sched_getaffinity(0))
    if text is None:
        # A deterministic two-CPU subset of the process's current allowance.
        require(len(allowed) >= 2, "at least two CPUs must be available for default affinity")
        return allowed[:2]
    try:
        cpus = [int(item) for item in text.split(",") if item]
    except ValueError as exc:
        raise ValueError("--affinity must be comma-separated CPU integers") from exc
    require(len(cpus) == 2 and len(set(cpus)) == 2, "--affinity must name exactly two distinct CPUs")
    require(set(cpus).issubset(set(allowed)), f"requested affinity {cpus} is outside current allowed CPUs {allowed}")
    return cpus


def cpu_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def quantiles_ns(samples: list[int]) -> dict[str, float | int]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "n": len(samples),
        "mean_ms": float(values.mean() / 1e6),
        "median_ms": float(np.median(values) / 1e6),
        "p95_ms": float(np.percentile(values, 95) / 1e6),
    }


def storage_bytes(tensors: Iterable[torch.Tensor]) -> int:
    """Count each underlying storage once, including zero-length tensors."""
    seen: set[tuple[str, int]] = set()
    total = 0
    for tensor in tensors:
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total


def runtime_memory(runtime: CpuRiftTemporalRuntime) -> dict[str, int]:
    state = [*runtime.keys, *runtime.values, *runtime.lengths]
    scratch = [*runtime._scratch_k, *runtime._scratch_v, *runtime._scratch_valid, *runtime._ages, *runtime._positions]
    return {
        "cached_state_unique_storage_bytes": storage_bytes(state),
        "cached_scratch_unique_storage_bytes": storage_bytes(scratch),
        "cached_state_and_scratch_unique_storage_bytes": storage_bytes([*state, *scratch]),
    }


def generic_state_bytes(state: RiftTemporalState) -> int:
    return storage_bytes(entry for layers in (state.keys, state.values) for rows in layers for row in rows for entry in row)


def expected_slopes(config: RiftTemporalConfig) -> list[float]:
    return [0.0 if half_life is None else math.log(2.0) * config.bin_seconds / half_life for half_life in DEFAULT_HALF_LIVES]


def make_pair(context: int) -> tuple[RiftTemporalConfig, RiftTemporal, RiftTemporal]:
    common = dict(width=WIDTH, heads=HEADS, ffn_width=FFN_WIDTH, layers=LAYERS)
    recency_config = RiftTemporalConfig.for_context(context, bias_mode="recency", **common)
    flat_config = RiftTemporalConfig.for_context(context, bias_mode="flat", **common)
    torch.manual_seed(SEED + context)
    recency = RiftTemporal(recency_config).eval()
    torch.manual_seed(SEED + context)
    flat = RiftTemporal(flat_config).eval()
    recency_parameters, flat_parameters = dict(recency.named_parameters()), dict(flat.named_parameters())
    require(recency_parameters.keys() == flat_parameters.keys(), "recency/flat named parameter keys differ")
    require(all(torch.equal(recency_parameters[name], flat_parameters[name]) for name in recency_parameters), "same-seed parameters are not byte-identical")
    recency_slopes = recency.recency_slopes.detach().cpu().tolist()
    require(np.allclose(recency_slopes, expected_slopes(recency_config), rtol=0.0, atol=1e-8), "recency slopes differ from config half-lives")
    require(torch.count_nonzero(flat.recency_slopes).item() == 0, "flat slopes must be exactly zero")
    return recency_config, recency, flat


def parity(temporal: RiftTemporal, tokens: torch.Tensor) -> dict[str, Any]:
    """Compare generic cached state, preallocated cache, and local full forward.

    The flow begins with valid left padding and is longer than every local
    window twice over, so both padding and steady finite-cache replacement are
    exercised.  Dense is deliberately absent: it is a correctness oracle, not
    a latency comparator here.
    """
    batch, length, _ = tokens.shape
    valid = torch.ones(batch, length, dtype=torch.bool)
    valid[:, : min(3, length - 1)] = False
    generic = temporal.init_state(batch, "cpu", torch.float32)
    cached = CpuRiftTemporalRuntime(temporal, batch)
    max_cached_generic = 0.0
    with torch.inference_mode():
        local = temporal(tokens, valid, backend="local")
        generic_last = cached_last = None
        for index in range(length):
            generic_last, generic = temporal.step(tokens[:, index], generic, valid[:, index])
            cached_last = cached.step(tokens[:, index], valid[:, index])
            max_cached_generic = max(max_cached_generic, float((cached_last - generic_last).abs().max()))
    require(generic_last is not None and cached_last is not None, "empty parity flow")
    full_generic = float((local[:, -1] - generic_last).abs().max())
    full_cached = float((local[:, -1] - cached_last).abs().max())
    require(max(max_cached_generic, full_generic, full_cached) <= 2e-5, "cached/state/full local parity exceeds 2e-5")
    return {
        "passed": True, "atol": 2e-5, "valid_left_pad_bins": int((~valid[0]).sum()),
        "flow_bins": length, "flow_exceeds_each_window_twice": True,
        "cached_vs_generic_state_max_abs": max_cached_generic,
        "full_local_last_bin_vs_generic_state_max_abs": full_generic,
        "full_local_last_bin_vs_cached_max_abs": full_cached,
        "generic_state_unique_storage_bytes_at_steady_state": generic_state_bytes(generic),
        **runtime_memory(cached),
    }


def online_round(temporal: RiftTemporal, batch: int, stream: torch.Tensor, warmup: int) -> list[int]:
    runtime = CpuRiftTemporalRuntime(temporal, batch)
    with torch.inference_mode():
        for token in stream[:warmup]:
            runtime.step(token)
        samples: list[int] = []
        for token in stream[warmup:warmup + ONLINE_STEPS]:
            start = time.perf_counter_ns()
            output = runtime.step(token)
            samples.append(time.perf_counter_ns() - start)
            require(torch.isfinite(output).all().item(), "nonfinite cached temporal output")
    return samples


def full_local_latency(temporal: RiftTemporal, batch: int, sequence: torch.Tensor) -> list[int]:
    valid = torch.ones(batch, sequence.shape[1], dtype=torch.bool)
    valid[:, :3] = False
    with torch.inference_mode():
        # Warm only the local full-window path.  Its timings are kept separate
        # from online cached timings because their workloads are different.
        temporal(sequence, valid, backend="local")
        samples: list[int] = []
        for _ in range(FULL_FORWARD_STEPS):
            start = time.perf_counter_ns()
            output = temporal(sequence, valid, backend="local")
            samples.append(time.perf_counter_ns() - start)
            require(torch.isfinite(output).all().item(), "nonfinite local full-window output")
    return samples


def parameter_count(temporal: RiftTemporal) -> int:
    return sum(parameter.numel() for parameter in temporal.parameters())


def case(context: int, batch: int, generator: torch.Generator) -> dict[str, Any]:
    config, recency, flat = make_pair(context)
    longest_window = max(config.windows)
    parity_length = max(context, 2 * longest_window + 1)
    parity_tokens = torch.randn(batch, parity_length, WIDTH, generator=generator, dtype=torch.float32)
    flow = torch.randn(context + ONLINE_STEPS, batch, WIDTH, generator=generator, dtype=torch.float32)
    full_sequence = torch.randn(batch, context, WIDTH, generator=generator, dtype=torch.float32)
    logical_kv = batch * sum(window - 1 for window in config.windows) * 2 * HEADS * config.head_dim * 4
    variants = {"recency": recency, "flat": flat}
    online: dict[str, Any] = {name: {"rounds": []} for name in variants}
    # Rotate the first arm each round to avoid a fixed recency-first bias.
    for round_index in range(ROUNDS):
        order = ("recency", "flat") if round_index % 2 == 0 else ("flat", "recency")
        for name in order:
            samples = online_round(variants[name], batch, flow, context)
            online[name]["rounds"].append({"round": round_index + 1, "position": order.index(name) + 1, "latency_ns": samples, "summary": quantiles_ns(samples)})
    for value in online.values():
        all_samples = [sample for round_ in value["rounds"] for sample in round_["latency_ns"]]
        value["steady_cached_online"] = quantiles_ns(all_samples)
    full = {}
    for name, temporal in variants.items():
        samples = full_local_latency(temporal, batch, full_sequence)
        full[name] = {"local_full_window_latency": quantiles_ns(samples), "latency_ns": samples, "repetitions": FULL_FORWARD_STEPS}
    return {
        "context_bins": context, "batch_size": batch,
        "config": {"width": WIDTH, "heads": HEADS, "head_dim": config.head_dim, "ffn_width": FFN_WIDTH, "layers": LAYERS, "windows": list(config.windows), "token_receptive_field": config.token_receptive_field},
        "parameters": {"count_each_variant": parameter_count(recency), "same_seed_named_parameters_byte_identical": True},
        "recency_slopes": {"expected": expected_slopes(config), "observed": recency.recency_slopes.tolist(), "flat_observed": flat.recency_slopes.tolist(), "flat_all_exact_zero": True},
        "memory": {"logical_kv_bytes": logical_kv, "logical_kv_definition": "B * sum(window-1) * 2(K,V) * heads * head_dim * FP32_bytes", "recency": parity(recency, parity_tokens), "flat": parity(flat, parity_tokens)},
        "attention_macs_per_online_bin": {"formula": "B * sum_layers(2 * heads * window_l * head_dim); QK plus AV only", "value": batch * sum(2 * HEADS * window * config.head_dim for window in config.windows), "excludes": "linear projections, LayerNorm, softmax, FFN, residuals, cache copies, and memory traffic"},
        "online_cached": online, "full_window_local": full,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="fresh JSON output path")
    parser.add_argument("--affinity", help="exactly two comma-separated CPUs; default is first two currently allowed CPUs")
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), f"fresh output required; refusing to overwrite {output}")
    cpus = parse_affinity(args.affinity)
    os.sched_setaffinity(0, cpus)
    require(sorted(os.sched_getaffinity(0)) == sorted(cpus), "failed to apply requested CPU affinity")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    cases = {f"R{context}_B{batch}": case(context, batch, generator) for context in CONTEXTS for batch in BATCHES}
    report = {
        "schema": "rift_temporal_recency_flat_cpu_benchmark_v1", "status": "COMPLETED_SYNTHETIC_TEMPORAL_ONLY", "utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Synthetic CPU FP32 inference benchmark of untrained RiftTemporal only. It loads no dataset/checkpoint/model payload, makes no network/GPU call, and does not measure end-to-end decoder latency or prediction quality.",
        "device": "cpu", "dtype": "float32", "autograd": False, "seed": SEED,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "cpu": cpu_name(), "cpu_affinity": sorted(os.sched_getaffinity(0)), "torch_intraop_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()},
        "protocol": {"contexts": list(CONTEXTS), "batches": list(BATCHES), "online_backend": "CpuRiftTemporalRuntime preallocated cached KV", "online_warmup_bins": "context_bins (therefore >= R)", "online_rounds": ROUNDS, "online_bins_per_round": ONLINE_STEPS, "online_order": "recency/flat alternates first position every round", "full_window_backend": "RiftTemporal.forward backend=local", "full_window_repetitions": FULL_FORWARD_STEPS, "full_window_note": "reported separately; not an online-cache comparison", "parity": "generic step state, preallocated cached runtime, and local full forward final bin with valid left padding and flow longer than twice every local window", "parity_atol": 2e-5},
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output), "cases": len(cases)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
