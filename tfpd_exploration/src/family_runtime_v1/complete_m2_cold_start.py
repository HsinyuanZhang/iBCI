"""Fail-closed ext4 completion proof for the reset-only M2 cold-start variant.

This is an independent runner.  It reuses only the already-audited pure
authority/session/direct-window helpers and never edits the linear-conv
candidate validator or runtime implementation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from tfpd_exploration.src.m2_dual_track_v1 import plan

from . import complete_m2_candidate as base
from . import complete_m2_lifted as frozen
from .m2_cold_start import ColdStartLinearConvM2Decoder

ROOT = frozen.ROOT
OUTPUT = ROOT / "tfpd_exploration/results/family_runtime_v1/m2_cold_start_complete_ext4_v1.json"
EXPECTED_ENDPOINTS = base.EXPECTED_ENDPOINTS
EXPECTED_PUBLIC_RAW_BINS = base.EXPECTED_PUBLIC_RAW_BINS
VARIANT = "zero_history_reset"


def dependencies() -> dict[str, str]:
    """The linear-conv base closure plus the cold implementation/validator."""
    values = dict(base.dependencies("linear_conv"))
    siblings = Path(__file__).parent
    values["m2_cold_start.py"] = frozen.file_sha256(siblings / "m2_cold_start.py")
    values["complete_m2_cold_start.py"] = frozen.file_sha256(Path(__file__))
    if len(values) != 16:
        raise RuntimeError(f"expected 14 base + cold + validator hashes, got {len(values)}")
    return values


def preflight() -> dict[str, object]:
    report = base.preflight("linear_conv")
    return {
        "schema": "family_runtime_v1_m2_cold_start_complete_prepare_v1",
        "status": "PASS_METADATA_ONLY_NO_DECODER",
        "variant": VARIANT,
        "implementation_scope": "reset-only exact zero-history frontend; inherited linear/grouped steady path",
        "endpoint_count": report["endpoint_count"],
        "public_raw_bin_count": report["public_raw_bin_count"],
        "payload_sha256": report["payload_sha256"],
        "addendum_sha256": report["addendum_sha256"],
        "sealed_e8_pooled_native_r2": report["sealed_e8_pooled_native_r2"],
        "base_dependency_sha256_14": base.dependencies("linear_conv"),
        "dependency_sha256_16": dependencies(),
        "per_session": report["per_session"],
        "parameter_updates": 0,
        "not_new_model_or_selection": True,
    }


def run(output: Path = OUTPUT) -> None:
    archive = output.with_suffix(".npz")
    if output.exists() or archive.exists():
        raise FileExistsError(f"output exists: {output}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rows, sealed = frozen.load_sealed_rows()
    deps_before = dependencies()
    base_before = base.dependencies("linear_conv")
    authority_before = {s: frozen.authority_hashes(s) for s in plan.EXT4_SESSIONS}
    reports, values, started = {}, [], time.perf_counter()
    for session in plan.EXT4_SESSIONS:
        report, arrays = base.validate_session(session, rows[session], authority_before[session], ColdStartLinearConvM2Decoder)
        reports[session] = report
        values.append(arrays)
    prediction, direct, v3, target, starts, sessions = (
        np.concatenate([value[i] for value in values]) if i < 5 else np.asarray(sum((value[i] for value in values), []))
        for i in range(6)
    )
    if prediction.shape != (EXPECTED_ENDPOINTS, plan.OUT_DIM):
        raise RuntimeError("global endpoint cardinality/output shape drift")
    if sum(row["public_raw_bins"] for row in reports.values()) != EXPECTED_PUBLIC_RAW_BINS:
        raise RuntimeError("global public raw-bin cardinality drift")
    direct_error = frozen.require_close(prediction, direct, "global cold/direct")
    v3_error = frozen.require_close(prediction, v3, "global cold/v3")
    score = frozen.native_r2(target, prediction)
    if abs(score - float(sealed["e8_pooled_r2"])) > frozen.ABS_TOL:
        raise RuntimeError("sealed pooled native R2 drift")
    deps_after = dependencies()
    base_after = base.dependencies("linear_conv")
    authority_after = {s: frozen.authority_hashes(s) for s in plan.EXT4_SESSIONS}
    payload_after, addendum_after = frozen.file_sha256(frozen.PAYLOAD), frozen.file_sha256(frozen.ADDENDUM)
    _, sealed_after = frozen.load_sealed_rows()
    if (deps_after != deps_before or base_after != base_before or authority_after != authority_before
            or payload_after != frozen.PAYLOAD_SHA256 or addendum_after != frozen.ADDENDUM_SHA256
            or sealed_after["e8_payload_sha256"] != frozen.PAYLOAD_SHA256):
        raise RuntimeError("cold-start dependency/payload/addendum/authority mutation")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive, prediction=prediction, direct_full=direct, v3_oracle=v3,
                        target=target, session=sessions, start=starts)
    body = {
        "schema": "family_runtime_v1_m2_cold_start_complete_ext4_v1", "status": "PASS",
        "variant": VARIANT, "implementation_scope": "reset-only exact zero-history frontend; inherited linear/grouped steady path",
        "endpoint_count": EXPECTED_ENDPOINTS, "public_raw_bin_count": EXPECTED_PUBLIC_RAW_BINS,
        "pooled_native_r2": score, "sealed_e8_pooled_native_r2": sealed["e8_pooled_r2"],
        "per_session": reports, "global_public_v3_candidate_max_abs_error": max(r["max_public_v3_candidate_abs_error"] for r in reports.values()),
        "global_endpoint_direct_max_abs_error": direct_error, "global_endpoint_v3_max_abs_error": v3_error,
        "base_dependency_sha256_14_pre": base_before, "base_dependency_sha256_14_post": base_after,
        "dependency_sha256_16_pre": deps_before, "dependency_sha256_16_post": deps_after,
        "cache_provenance_map_bank_sha256_pre": authority_before, "cache_provenance_map_bank_sha256_post": authority_after,
        "payload_sha256_pre": frozen.PAYLOAD_SHA256, "payload_sha256_post": payload_after,
        "addendum_sha256_pre": frozen.ADDENDUM_SHA256, "addendum_sha256_post": addendum_after,
        "stream_npz_sha256": frozen.file_sha256(archive), "elapsed_seconds": time.perf_counter()-started,
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "parameter_updates": 0, "not_new_model_or_selection": True,
    }
    output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.preflight:
        if args.output != OUTPUT: parser.error("--preflight does not accept a custom output")
        print(json.dumps(preflight(), indent=2, sort_keys=True))
    else:
        run(args.output)
