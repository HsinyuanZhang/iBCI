"""Short CPU-only cost preflight for untrained H1 RIFT streaming.

This is intentionally a synthetic, inference-only timing probe.  It measures
real ``RiftStreamDecoder.stream_step`` advances (including stream-id lookup,
bank/mask hashing, state packing and readout), not repeated ``predict`` calls.
It is not an E-ORT benchmark: the frozen 582044 implementation requires its
own exported EMA ONNX payload/engine, which is outside this short preflight.
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
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "btransform_unified_v1" / "src")]

from btransform_unified_v1.bank import TaskBank, array_sha256
from btransform_unified_v2 import RiftDecoder, RiftStreamDecoder


def synthetic_h1_bank(seed: int = 17) -> TaskBank:
    rng = np.random.default_rng(seed)
    e0 = rng.standard_normal((176, 700), dtype=np.float32)
    carrier = rng.standard_normal((176, 4), dtype=np.float32)
    mask = np.ones(176, dtype=np.bool_)
    x = np.zeros((1, 1, 176), dtype=np.float32)
    y = np.zeros((1, 7), dtype=np.float32)
    meta = {"shape": [176, 700], "trial_count": 1, "estimator": "synthetic_cpu_cost",
            "array_sha256": array_sha256(e0), "budget": 1}
    return TaskBank("cpu-cost-h1", e0, carrier, mask, x, y, np.array([0], dtype=np.int64), meta)


def percentile(values: list[float], fraction: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), fraction))


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.uname().processor


def state_bytes(stream: RiftStreamDecoder) -> tuple[int, int, int]:
    """Return logical KV bytes, unique backing-storage bytes, and storage count."""
    logical = 0
    storage: dict[tuple[str, int], int] = {}
    for item in stream._streams.values():  # benchmark-only introspection
        assert item.state is not None
        for layers in (item.state.keys, item.state.values):
            for layer in layers:
                for row in layer:
                    for value in row:
                        logical += value.numel() * value.element_size()
                        key = (str(value.device), value.untyped_storage().data_ptr())
                        storage[key] = max(storage.get(key, 0), value.untyped_storage().nbytes())
    return logical, sum(storage.values()), len(storage)


def timing_summary(values: list[float]) -> dict[str, float]:
    return {"n": len(values), "median_ms": statistics.median(values) * 1e3,
            "mean_ms": statistics.mean(values) * 1e3, "p95_ms": percentile(values, 95) * 1e3}


@torch.inference_mode()
def measure(context: int, batch: int, steps: int, rounds: int, bank: TaskBank) -> dict[str, object]:
    torch.manual_seed(1000 + context * 10 + batch)
    # Parameters must be created outside inference_mode so the adapter's
    # explicit weight-version invalidation check remains available.
    with torch.inference_mode(False):
        model = RiftDecoder("h1", context_bins=context, seed=42).eval()
    stream = RiftStreamDecoder(model)
    ids = [f"b{row}" for row in range(batch)]
    # Warm precisely to the configured raw receptive field, then measure real
    # post-warm advances.  Every call consumes a fresh raw observation.
    with torch.inference_mode():
        for _ in range(context):
            stream.stream_step(torch.randn(batch, 176), bank, ids)
    before = state_bytes(stream)
    full: list[float] = []
    frontend: list[float] = []
    temporal: list[float] = []
    for _round in range(rounds):
        for _ in range(steps):
            next_x = torch.randn(batch, 176)
            start = time.perf_counter()
            stream.stream_step(next_x, bank, ids)
            full.append(time.perf_counter() - start)
            # Component timing uses a separate real temporal state advance;
            # it never replaces the full-adapter samples above.
            raw5 = torch.randn(batch, 5, 176)
            start = time.perf_counter(); z = model.frontend_last(raw5, bank); frontend.append(time.perf_counter() - start)
            packed = model.temporal.init_state(batch, "cpu", torch.float32)
            for _ in range(max(window - 1 for window in model.temporal_config.windows)):
                _, packed = model.temporal.step(torch.randn(batch, 256), packed)
            start = time.perf_counter(); _, packed = model.temporal.step(z, packed); temporal.append(time.perf_counter() - start)
    after = state_bytes(stream)
    return {"context_bins": context, "windows": list(model.temporal_config.windows), "batch": batch,
            "warmup_advances": context, "measured_advances": steps * rounds, "rounds": rounds,
            "full_adapter_advance": timing_summary(full), "frontend_last": timing_summary(frontend),
            "temporal_step": timing_summary(temporal),
            "persistent_kv_after_warm": {"logical_bytes": before[0], "unique_storage_bytes": before[1], "unique_storages": before[2]},
            "persistent_kv_after_measure": {"logical_bytes": after[0], "unique_storage_bytes": after[1], "unique_storages": after[2]},
            "kv_stable_after_measure": before == after}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.steps < 30 or args.steps > 50 or args.rounds != 3:
        raise ValueError("use 30..50 steps and exactly three interleaved rounds")
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    bank = synthetic_h1_bank()
    # Interleave configurations by round at coarse level through measure's
    # internal three rounds; each configuration has the same sample count.
    results = [measure(context, batch, args.steps, args.rounds, bank) for context in (200, 300) for batch in (1, 8)]
    destination = args.dest or ROOT / "results" / "rift_v1" / f"cpu_cost_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    destination.mkdir(parents=True, exist_ok=False)
    report = {"schema": "rift_cpu_cost_v1", "device": "cpu", "cpu": cpu_model(),
              "threads": 2, "torch": torch.__version__, "python": platform.python_version(), "dtype": "float32",
              "autograd": False, "weights": "untrained shared seed=42", "input": "synthetic iid N(0,1) H1 raw bins; synthetic all-true bank",
              "full_adapter_includes": ["stream_id lookup", "per-stream bank/mask hash", "raw history", "vectorized frontend_last", "state pack/unpack", "temporal.step", "readout"],
              "eort_582044": {"measured": False, "reason": "requires sealed EMA ONNX graph payload/engine; not reused in short RIFT-only CPU preflight",
                              "official_record": "submission 582044, E-ORT H1 L200"}, "cases": results}
    (destination / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
