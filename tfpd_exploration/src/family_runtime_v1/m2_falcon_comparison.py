"""Final local Falcon adapter against the frozen SPINT image, source bins only."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .benchmark import M2_PAYLOAD, sha, stats
from .m2_falcon import FamilyM2FalconDecoder
from .m2_spint_comparison import ROOT, WRAPPER, AS_SHIPPED, DECLARED, EXPECTED
from tfpd_exploration.src.m2_runtime_v3.runtime import RuntimeV3Decoder, _reference_module


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if torch.cuda.is_available():
        raise RuntimeError("CUDA must be disabled before startup")
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    import src.models.components.spint as spint_module
    spint_path = Path(spint_module.__file__)
    if sha(spint_path) != "855b9d06e9c22ca1c1ded78df4ec15af875707bace78b5528499f05f7a212519":
        raise RuntimeError("requires the frozen image's actual Original-SPINT module")
    for path, digest in EXPECTED.items():
        if sha(path) != digest:
            raise RuntimeError(f"frozen payload/wrapper drift: {path}")
    spec = importlib.util.spec_from_file_location("_m2_final_spint_ref", WRAPPER)
    wrapper = importlib.util.module_from_spec(spec); spec.loader.exec_module(wrapper)
    ref = _reference_module()  # Explicit module-import boundary, before timers.
    from falcon_challenge.config import FalconConfig, FalconTask
    config = FalconConfig(task=FalconTask.m2)
    code = [Path(__file__), Path(ref.__file__), spint_path,
            Path(__file__).parents[1] / "m2_runtime_v3/runtime.py"]
    code.extend(Path(__file__).with_name(name + ".py") for name in (
        "m2_falcon", "m2_cold_start", "m2_linear_conv", "linear_conv", "m2_grouped",
        "grouped_value", "m2_lifted", "m2", "repair", "benchmark", "m2_spint_comparison"))
    code_before = {str(path): sha(path) for path in code}
    source = ROOT / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache/source_train"
    sessions = sorted(source.glob("ses-*"))
    if len(sessions) != 7:
        raise RuntimeError("exactly seven source streams required")
    count = 1 + args.warmup + args.calls
    tags, rows, source_hash = [], [], {}
    for folder in sessions:
        bits = folder.name.removeprefix("ses-").split("-")
        tags.append(f"{bits[-1]}_{''.join(bits[:3])}")
        path = folder / "X_store.npy"; raw = np.load(path, mmap_mode="r")
        if not np.all(raw[:49] == 0):
            raise RuntimeError("source's W49 padding drift")
        value = np.array(raw[49:49 + count], dtype=np.float32, copy=True)
        if value.shape != (count, 96):
            raise RuntimeError("insufficient real source bins; repetition forbidden")
        rows.append(value); source_hash[str(path)] = sha(path)
    values = np.stack(rows, axis=1)
    factories = {
        "spint_image_default_seed42": lambda: wrapper.T4CachedIdentityDecoder(config, str(AS_SHIPPED), batch_size=7),
        "spint_declared_seed44_e8": lambda: wrapper.T4CachedIdentityDecoder(config, str(DECLARED), batch_size=7),
        "transformer_v3_backend": lambda: RuntimeV3Decoder(ROOT / M2_PAYLOAD, batch_size=7),
        "transformer_final_falcon": lambda: FamilyM2FalconDecoder(config, ROOT / M2_PAYLOAD, batch_size=7),
    }
    engines, construct, reset = {}, {}, {}
    for name, factory in factories.items():
        start = time.perf_counter_ns(); engines[name] = factory()
        construct[name] = (time.perf_counter_ns() - start) / 1e6
        start = time.perf_counter_ns(); engines[name].reset(tags)
        reset[name] = (time.perf_counter_ns() - start) / 1e6
    samples = {name: [] for name in engines}; first = {}; error = 0.0
    names = list(engines)
    with torch.no_grad():
        for index, value in enumerate(values):
            output = {}; shift = index % len(names)
            for name in names[shift:] + names[:shift]:
                start = time.perf_counter_ns(); output[name] = engines[name].predict(value)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                out = output[name]
                if out.shape != (7, 2) or out.dtype != np.float32 or not np.isfinite(out).all():
                    raise RuntimeError("whole-batch public output contract drift")
                if index == 0: first[name] = elapsed
                elif index >= args.warmup + 1: samples[name].append(elapsed)
            got, want = output["transformer_final_falcon"], output["transformer_v3_backend"]
            if not got.flags.owndata or not got.flags.c_contiguous:
                raise RuntimeError("final adapter output must own contiguous native storage")
            np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)
            error = max(error, float(np.abs(got - want).max()))
    code_after = {str(path): sha(path) for path in code}
    if code_before != code_after or any(sha(Path(path)) != digest for path, digest in source_hash.items()):
        raise RuntimeError("code/source changed during benchmark")
    if any(sha(path) != digest for path, digest in EXPECTED.items()):
        raise RuntimeError("frozen input changed during benchmark")
    timing = {name: stats(value) for name, value in samples.items()}
    result = {"schema": "m2_final_falcon_same_spint_image_source_latency_v1", "status": "PASS",
        "calls": args.calls, "warmup": args.warmup, "batch": 7, "threads": args.threads,
        "torch": torch.__version__, "affinity": sorted(os.sched_getaffinity(0)),
        "timing": timing, "first_public_call_ms": first, "constructor_after_explicit_module_import_ms": construct,
        "reset_after_constructor_ms": reset, "constructor_order": names, "cold_import_time_excluded": True,
        "final_falcon_max_native_abs_error": error, "code_sha256_pre": code_before, "code_sha256_post": code_after,
        "source_raw_sha256": source_hash, "payload_wrapper_sha256": {str(path): digest for path, digest in EXPECTED.items()},
        "p95_final_over_spint_default": timing["transformer_final_falcon"]["p95_ms"] / timing["spint_image_default_seed42"]["p95_ms"],
        "p95_v3_over_final": timing["transformer_v3_backend"]["p95_ms"] / timing["transformer_final_falcon"]["p95_ms"],
        "historical_image": "sha256:8de56c58939ebd8306954ea7d180aceb7269fd3df28192f11df0dcaa7b60dc7f",
        "boundary": "whole-B7 contiguous float32 host input through public predict to owning native NumPy output",
        "targets_or_nwb_opened": False, "parameter_updates": 0, "not_official_latency": True,
        "host_not_exclusive": True, "quality_selection_or_generalization_claim": False,
        "existing_images_payloads_submissions_modified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "timing", "first_public_call_ms", "reset_after_constructor_ms", "final_falcon_max_native_abs_error")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, choices=(1, 2), required=True)
    parser.add_argument("--calls", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
