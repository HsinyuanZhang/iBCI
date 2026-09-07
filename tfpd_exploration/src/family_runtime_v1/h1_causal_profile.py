"""Synthetic W700 H1 profile; initialized weights, no data or quality claim."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from tfpd_exploration.src.h1_family_v1.model import make_v2_unscaled_dot_localbalanced_pair
from .h1_causal import H1CausalRuntime, W
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "tfpd_exploration/results/decoder_validation_v2/20260905_190000/h1/family_v1/h1_causal_initialized_profile_v1.json"


def closure():
    from tfpd_exploration.src.h1_family_v1 import model
    from tfpd_exploration.src.h1_optimized_v2 import model as base_model
    from tfpd_exploration.src.two_mainlines_long_v1.decoder import h1_temporal, h1_config
    from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2 import core
    from . import h1_causal, h1_lifted_frontend, linear_conv
    paths = [Path(__file__)] + [Path(m.__file__) for m in
        (model, base_model, h1_temporal, h1_config, core, h1_causal, h1_lifted_frontend, linear_conv)]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


@torch.no_grad()
def native(model, bank, raw):
    return (model.forward_last(raw, bank) / 20).numpy().copy()


def check(got, want):
    np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)
    if got.shape != (1, 7) or got.dtype != np.float32 or not got.flags.owndata or not got.flags.c_contiguous:
        raise RuntimeError("whole-public output contract drift")
    return float(np.max(np.abs(got - want)))


def stats(values):
    return {"mean_ms": float(np.mean(values)), "p95_ms": float(np.percentile(values, 95))}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    if torch.cuda.is_available():
        raise RuntimeError("CUDA must be disabled")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    before = closure()
    flat, route = make_v2_unscaled_dot_localbalanced_pair(seed=42)
    del route
    flat.eval()
    torch.manual_seed(913)
    bank = H1Bank(torch.randn(176, 700) * .2, torch.randn(176, 4) * .1,
                  torch.ones(176, dtype=torch.bool))
    rng = np.random.default_rng(913)
    bins = rng.poisson(.3, (11, 1, 176)).astype(np.float32)
    runtime = H1CausalRuntime(flat, bank)
    start = time.perf_counter_ns()
    runtime.reset(bank, batch=1)
    reset_ms = (time.perf_counter_ns() - start) / 1e6
    first_raw = torch.zeros(1, W, 176)
    first_raw[:, -1] = torch.from_numpy(bins[0])
    start = time.perf_counter_ns()
    first = runtime.predict(bins[0])
    first_ms = (time.perf_counter_ns() - start) / 1e6
    max_error = check(first, native(flat, bank, first_raw))
    torch.testing.assert_close(runtime.raw, first_raw, atol=0, rtol=0)

    # Explicit synthetic setup, excluded from timings. The separate 703-bin
    # lifecycle test establishes that real streaming reaches the same state.
    raw = torch.from_numpy(rng.poisson(.3, (1, W, 176)).astype(np.float32))
    with torch.no_grad():
        runtime.raw = raw.clone()
        runtime.frontend = flat.encode_frontend(raw, bank).detach().clone()
    samples = {"full_model": [], "runtime_public": []}
    for index, value in enumerate(bins[1:]):  # Exactly 2 warmups + 8 timed pairs.
        raw = torch.cat((raw[:, 1:], torch.from_numpy(value)[:, None]), dim=1)
        calls = {"full_model": lambda: native(flat, bank, raw),
                 "runtime_public": lambda: runtime.predict(value)}
        outputs = {}
        order = ("full_model", "runtime_public") if index % 2 == 0 else ("runtime_public", "full_model")
        for key in order:
            start = time.perf_counter_ns()
            outputs[key] = calls[key]()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if index >= 2:
                samples[key].append(elapsed)
        max_error = max(max_error, check(outputs["runtime_public"], outputs["full_model"]))
        torch.testing.assert_close(runtime.raw, raw, atol=0, rtol=0)
    if any(len(v) != 8 for v in samples.values()):
        raise RuntimeError("exact timing count drift")
    after = closure()
    if before != after:
        raise RuntimeError("profile code changed during execution")
    result = {"schema": "h1_causal_initialized_profile_v1", "status": "SYNTHETIC_INITIALIZED_NOT_OFFICIAL",
        "geometry": {"W": W, "B": 1, "units": 176, "E0": 700, "T": 4, "set_dim": 256, "temporal_layers": 4},
        "model": "fresh seed42 common localbalanced unscaled FLAT", "warmups": 2, "timed": 8,
        "samples_ms": samples, "timing": {k: stats(v) for k, v in samples.items()},
        "reset_after_constructor_ms": reset_ms, "first_public_call_ms": first_ms,
        "max_native_abs_error": max_error, "code_sha256_pre": before, "code_sha256_post": after,
        "affinity": sorted(os.sched_getaffinity(0)), "torch": torch.__version__,
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "synthetic_cache_seeded_steady_not_lifecycle_proof": True, "no_trained_weight_or_data_loaded": True,
        "quality_or_official_latency_claim": False,
        "timing_boundary": "host bin through whole public predict to owning native NumPy; oracle receives preassembled W700 tensor",
        "decomposition": "not instrumented; whole path timing only"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "timing", "reset_after_constructor_ms", "first_public_call_ms", "max_native_abs_error")}))


if __name__ == "__main__":
    main()
