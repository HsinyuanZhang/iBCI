"""Actual historical M2 wrapper vs exact runtime on identical source bins.

No targets, calibration fit, source datamodule, remote API or container writes.
The two SPINT payloads are reported separately because the recorded 581919
image's default path contains a different payload from its declared selection.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .benchmark import M2_PAYLOAD, sha, stats
from .m2_lifted import LiftedFiveTokenM2Decoder
from tfpd_exploration.src.m2_runtime_v3.runtime import RuntimeV3Decoder


ROOT = Path(__file__).resolve().parents[3]
WRAPPER = ROOT / "tfpd_exploration/submissions/evalai_m2_movement_t4_empty_v1/t4_spint_decoder.py"
AS_SHIPPED = ROOT / "tfpd_exploration/submissions/evalai_m2_movement_t4_empty_v1/artifacts/t4_m2_seed42_movement_t4_empty_identity.pkl"
DECLARED = ROOT / "tfpd_exploration/submissions/evalai_m2_movement_t4_empty_epochpick_v1/artifacts/t4_m2_seed44_epoch08_movement_t4_empty_identity.pkl"
EXPECTED = {
    WRAPPER: "fd1d5b203d9c8daccdda2e212c7efd7720f43717dceef876bd136e99bf7224dc",
    AS_SHIPPED: "4e4dae8f7239582a26d44cdb449e674710f28223523dd691dd4f8758b05220e0",
    DECLARED: "f2f8cd4c046a5880e9716d61981cee4aa0652d33e411212fa7be217cef05b051",
    ROOT / M2_PAYLOAD: "4db109e75276d6e8f47df6540b0a4c8f6e955f81af1212127723976795f2eaa4",
}


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if torch.cuda.is_available():
        raise RuntimeError("CPU reference requires CUDA_VISIBLE_DEVICES='' before process start")
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise RuntimeError(f"frozen reference drift: {path}")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if not args.container_reference:
        sys.path.insert(0, str(ROOT / "streaming_calibration_exp"))
    import src.models.components.spint as spint_module
    spint_source_sha = sha(Path(spint_module.__file__))
    if args.container_reference and spint_source_sha != "855b9d06e9c22ca1c1ded78df4ec15af875707bace78b5528499f05f7a212519":
        raise RuntimeError("container's original SPINT module authority drift")
    spec = importlib.util.spec_from_file_location("_m2_packaged_spint_reference", WRAPPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from falcon_challenge.config import FalconConfig, FalconTask
    config = FalconConfig(task=FalconTask.m2)
    count = 1 + args.warmup + args.calls
    source = ROOT / "tfpd_exploration/results/m2_dual_track_v1/20260905_101500/cache/source_train"
    sessions = sorted(source.glob("ses-*"))
    if len(sessions) != 7:
        raise RuntimeError("exactly seven source rows required")
    tags, rows, source_hashes = [], [], {}
    for session in sessions:
        bits = session.name.removeprefix("ses-").split("-")
        tags.append(f"{bits[-1]}_{''.join(bits[:3])}")
        path = session / "X_store.npy"
        raw = np.load(path, mmap_mode="r")
        if not np.all(raw[:49] == 0):
            raise RuntimeError("source padding authority drift")
        row = np.array(raw[49:49 + count], dtype=np.float32, copy=True)
        if row.shape != (count, 96):
            raise RuntimeError("insufficient continuous raw bins; no repetition")
        rows.append(row)
        source_hashes[session.name] = sha(path)
    values = np.stack(rows, axis=1)
    engines, cold = {}, {}
    for name, payload in (("spint_image_default_seed42", AS_SHIPPED), ("spint_declared_seed44_e8", DECLARED)):
        begun = time.perf_counter_ns()
        engine = module.T4CachedIdentityDecoder(config, str(payload), batch_size=7)
        engine.reset([Path(tag) for tag in tags])
        if engine.device.type != "cpu" or engine.smooth_observations:
            raise RuntimeError("frozen CPU/unsmoothed wrapper contract drift")
        engines[name] = engine
        cold[name] = (time.perf_counter_ns() - begun) / 1e6
    transformer_types = [("transformer_v3", RuntimeV3Decoder), ("transformer_lifted", LiftedFiveTokenM2Decoder)]
    if args.include_grouped:
        from .m2_grouped import GroupedValueFiveTokenM2Decoder
        transformer_types.append(("transformer_grouped", GroupedValueFiveTokenM2Decoder))
    if args.include_linear_conv:
        from .m2_linear_conv import LinearConvFiveTokenM2Decoder
        transformer_types.append(("transformer_linear_conv", LinearConvFiveTokenM2Decoder))
    for name, cls in transformer_types:
        begun = time.perf_counter_ns()
        engine = cls(ROOT / M2_PAYLOAD, batch_size=7)
        engine.reset(tags)
        engines[name] = engine
        cold[name] = (time.perf_counter_ns() - begun) / 1e6
    names = list(engines)
    samples = {name: [] for name in names}
    first = {}
    transformer_errors = {name: 0. for name, _ in transformer_types}
    with torch.no_grad():
        for index, value in enumerate(values):
            outputs = {}
            shift = index % len(names)
            for name in names[shift:] + names[:shift]:
                begun = time.perf_counter_ns()
                outputs[name] = engines[name].predict(value)
                duration = (time.perf_counter_ns() - begun) / 1e6
                out = outputs[name]
                if out.shape != (7, 2) or out.dtype != np.float32 or not np.isfinite(out).all():
                    raise RuntimeError("whole-batch native output contract drift")
                if index == 0:
                    first[name] = duration
                elif index >= 1 + args.warmup:
                    samples[name].append(duration)
            expected = outputs["transformer_v3"]
            for name, _ in transformer_types:
                actual = outputs[name]
                np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
                transformer_errors[name] = max(transformer_errors[name], float(np.abs(actual - expected).max()))
    metrics = {name: stats(value) for name, value in samples.items()}
    result = {
        "schema": "m2_packaged_spint_same_host_source_b7_latency_v1", "batch": 7,
        "threads": torch.get_num_threads(), "calls": args.calls, "warmup": args.warmup,
        "torch": torch.__version__, "affinity": sorted(os.sched_getaffinity(0)),
        "spint_module_path": str(spint_module.__file__), "spint_module_sha256": spint_source_sha,
        "spint_model_geometry": {"dim": engines["spint_image_default_seed42"].decoder.model_dim,
                                 "layers": engines["spint_image_default_seed42"].decoder.num_layers,
                                 "heads": engines["spint_image_default_seed42"].decoder.num_heads},
        "public_predict_host_to_native_numpy_whole_batch": True, "continuous_source_only": True,
        "timing": metrics, "construct_load_reset_ms": cold, "first_public_call_ms": first,
        "transformer_max_native_abs_error": transformer_errors["transformer_lifted"],
        "all_transformer_max_native_abs_error": transformer_errors,
        "grouped_candidate_included": args.include_grouped,
        "linear_conv_candidate_included": args.include_linear_conv,
        "all_p95_ratios_to_spint_image_default": {name: item["p95_ms"] / metrics["spint_image_default_seed42"]["p95_ms"] for name, item in metrics.items()},
        "lifted_p95_ratio_to_spint_image_default": metrics["transformer_lifted"]["p95_ms"] / metrics["spint_image_default_seed42"]["p95_ms"],
        "lifted_p95_ratio_to_spint_declared": metrics["transformer_lifted"]["p95_ms"] / metrics["spint_declared_seed44_e8"]["p95_ms"],
        "source_raw_sha256": source_hashes, "session_tags": tags,
        "frozen_payload_and_wrapper_sha256": {str(path): expected for path, expected in EXPECTED.items()},
        "code_sha256": {str(path): sha(path) for path in (Path(__file__), Path(__file__).with_name("m2_lifted.py"), Path(__file__).with_name("m2.py"),
            *((Path(__file__).with_name("m2_grouped.py"), Path(__file__).with_name("grouped_value.py")) if args.include_grouped or args.include_linear_conv else ()),
            *((Path(__file__).with_name("m2_linear_conv.py"), Path(__file__).with_name("linear_conv.py")) if args.include_linear_conv else ()))},
        "historical_default_image_id": "sha256:8de56c58939ebd8306954ea7d180aceb7269fd3df28192f11df0dcaa7b60dc7f",
        "image_default_payload_mismatch_receipt": "m2_581919_default_payload_audit_v1.json",
        "not_official_latency": True, "same_frozen_spint_container_process": args.container_reference,
        "not_quality_or_generalization_test": True,
        "not_confirmed_user_reference_for_all_tasks": True, "host_not_exclusive": True,
        "parameter_updates": 0, "targets_or_nwb_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--calls", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--container-reference", action="store_true")
    parser.add_argument("--include-grouped", action="store_true")
    parser.add_argument("--include-linear-conv", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
