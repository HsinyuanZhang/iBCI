"""Unrun fail-closed ext4 complete validator for isolated exact candidates."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from tfpd_exploration.src.m2_dual_track_v1 import data, plan
from tfpd_exploration.src.m2_runtime_v3.runtime import RuntimeV3Decoder
from tfpd_exploration.src.m2_runtime_v3 import runtime as v3_runtime
from tfpd_exploration.src.m2_same_query_comparator_v1 import core

from . import complete_m2_lifted as frozen
from .m2_grouped import GroupedValueFiveTokenM2Decoder
from .m2_linear_conv import LinearConvFiveTokenM2Decoder


VARIANTS = {
    "grouped": GroupedValueFiveTokenM2Decoder,
    "linear_conv": LinearConvFiveTokenM2Decoder,
}
EXPECTED_ENDPOINTS = 2069
EXPECTED_PUBLIC_RAW_BINS = 10839


def dependencies(variant: str) -> dict[str, str]:
    siblings = Path(__file__).parent
    names = [
        "complete_m2_candidate.py", "complete_m2_lifted.py", "m2.py", "m2_lifted.py",
        "m2_grouped.py", "grouped_value.py", "m2_linear_conv.py", "linear_conv.py",
        "repair.py",
    ]
    paths = {name: siblings / name for name in names}
    paths["runtime_v3.py"] = frozen.ROOT / "tfpd_exploration/src/m2_runtime_v3/runtime.py"
    paths["runtime_v3_reference.py"] = Path(v3_runtime._REFERENCE)
    paths["dual_track_data.py"] = Path(data.__file__)
    paths["dual_track_plan.py"] = Path(plan.__file__)
    paths["historical_hash_core.py"] = Path(core.__file__)
    selected = {"grouped": ("m2_grouped.py", "grouped_value.py"),
                "linear_conv": ("m2_linear_conv.py", "linear_conv.py", "m2_grouped.py", "grouped_value.py")}[variant]
    fixed = ("complete_m2_candidate.py", "complete_m2_lifted.py", "m2.py", "m2_lifted.py",
             "repair.py", "runtime_v3.py", "runtime_v3_reference.py", "dual_track_data.py",
             "dual_track_plan.py", "historical_hash_core.py")
    return {name: frozen.file_sha256(paths[name]) for name in (*fixed, *selected)}


@torch.no_grad()
def direct_native(decoder, window: np.ndarray) -> np.ndarray:
    engine = decoder._engine
    return (
        decoder.model.forward_last(
            torch.from_numpy(np.ascontiguousarray(window[None, :, :])),
            engine.bank,
            engine.bank.unit_mask,
        ).numpy() / plan.BEHAVIOR_SCALE
    )


def validate_session(session: str, sealed: dict[str, object], before: dict[str, str], candidate_type):
    bank = data.load_session_bank("ext4", session)
    starts = np.ascontiguousarray(bank.eligible_starts, dtype=np.int64)
    target = np.ascontiguousarray(bank.target_store, dtype=np.float32)
    raw = np.ascontiguousarray(bank.X_store, dtype=np.float32)
    endpoints = starts + plan.WINDOW - 1
    if (plan.WINDOW != 50 or plan.CHANNELS != 96 or plan.OUT_DIM != 2 or plan.BEHAVIOR_SCALE != 5.0
            or raw.ndim != 2 or raw.shape[1] != 96 or raw.shape[0] < 50 or not np.all(raw[:49] == 0)
            or target.shape != (starts.size, 2) or starts.size != int(sealed["window_count"])
            or starts.size == 0 or int(starts[0]) < 0
            or not np.all(np.diff(starts) > 0) or int(endpoints[-1]) >= raw.shape[0]
            or core.array_sha256(starts) != sealed["ordered_window_starts_sha256"]
            or core.array_sha256(target) != sealed["target_sha256"]):
        raise RuntimeError(f"{session}: sealed ext4 geometry/start/target authority drift")
    tag = frozen.payload_tag(session)
    v3 = RuntimeV3Decoder(frozen.PAYLOAD, batch_size=1)
    candidate = candidate_type(frozen.PAYLOAD, batch_size=1)
    v3.reset([tag])
    candidate.reset([tag])
    prediction, full, v3_endpoint = (np.empty_like(target) for _ in range(3))
    index = {int(tick): i for i, tick in enumerate(endpoints)}
    public_error = 0.0
    visited = 0
    for tick, value in enumerate(raw[49:], start=49):
        obs = np.ascontiguousarray(value[None, :], dtype=np.float32)
        got, reference = candidate.predict(obs), v3.predict(obs)
        public_error = max(public_error, frozen.require_close(got, reference, f"{session}: public {tick}"))
        row = index.get(tick)
        if row is not None:
            window = np.ascontiguousarray(raw[int(starts[row]):int(starts[row]) + plan.WINDOW])
            if not np.array_equal(candidate._engine.raw[0].numpy(), window):
                raise RuntimeError(f"{session}: independent W50 stream window drift")
            native = direct_native(candidate, window)
            frozen.require_close(got[:1], native, f"{session}: direct {tick}")
            prediction[row], full[row], v3_endpoint[row] = got[0], native[0], reference[0]
            visited += 1
    if visited != starts.size:
        raise RuntimeError(f"{session}: visited endpoint cardinality drift")
    direct_error = frozen.require_close(prediction, full, f"{session}: endpoint/direct")
    v3_error = frozen.require_close(prediction, v3_endpoint, f"{session}: endpoint/v3")
    score = frozen.native_r2(target, prediction)
    if abs(score - float(sealed["e8_r2"])) > frozen.ABS_TOL:
        raise RuntimeError(f"{session}: sealed e8 native R2 drift")
    if frozen.authority_hashes(session) != before:
        raise RuntimeError(f"{session}: cache/provenance/map/bank mutation")
    return {
        "tag": tag, "public_raw_bins": int(raw.shape[0] - 49), "endpoint_count": int(starts.size),
        "visited_endpoint_count": visited, "starts_sha256": core.array_sha256(starts),
        "targets_sha256": core.array_sha256(target), "candidate_prediction_sha256": core.array_sha256(prediction),
        "native_r2": score, "max_public_v3_candidate_abs_error": public_error,
        "max_endpoint_direct_abs_error": direct_error, "max_endpoint_v3_abs_error": v3_error,
    }, (prediction, full, v3_endpoint, target, starts, [session] * starts.size)


def preflight(variant: str) -> dict[str, object]:
    rows, receipt = frozen.load_sealed_rows()
    total, raw_total, report = 0, 0, {}
    for session in plan.EXT4_SESSIONS:
        paths = frozen.authority_paths(session)
        starts = np.ascontiguousarray(np.load(paths["eligible_starts"]), dtype=np.int64)
        target = np.ascontiguousarray(np.load(paths["target_store"], mmap_mode="r"), dtype=np.float32)
        raw = np.load(paths["X_store"], mmap_mode="r")
        sealed = rows[session]
        if (plan.WINDOW != 50 or plan.CHANNELS != 96 or plan.OUT_DIM != 2 or plan.BEHAVIOR_SCALE != 5.0
                or starts.size == 0 or int(starts[0]) < 0 or starts.size != int(sealed["window_count"])
                or target.shape != (starts.size, 2) or raw.ndim != 2 or raw.shape[1] != 96
                or core.array_sha256(starts) != sealed["ordered_window_starts_sha256"]
                or core.array_sha256(target) != sealed["target_sha256"] or not np.all(raw[:49] == 0)):
            raise RuntimeError(f"{session}: candidate preflight authority drift")
        total += starts.size; raw_total += raw.shape[0] - 49
        report[session] = {"endpoints": int(starts.size), "public_raw_bins": int(raw.shape[0] - 49),
                           "cache_provenance_map_bank_sha256": frozen.authority_hashes(session)}
    if total != EXPECTED_ENDPOINTS or raw_total != EXPECTED_PUBLIC_RAW_BINS:
        raise RuntimeError(f"expected {EXPECTED_ENDPOINTS} endpoints/{EXPECTED_PUBLIC_RAW_BINS} raw bins, got {total}/{raw_total}")
    return {"schema": "family_runtime_v1_m2_candidate_preflight_v1", "status": "PASS_METADATA_ONLY_NO_DECODER",
            "variant": variant, "endpoint_count": total, "public_raw_bin_count": raw_total,
            "payload_sha256": frozen.PAYLOAD_SHA256, "addendum_sha256": frozen.ADDENDUM_SHA256,
            "sealed_e8_pooled_native_r2": receipt["e8_pooled_r2"], "dependency_sha256": dependencies(variant),
            "per_session": report}


def run(variant: str, output: Path) -> None:
    archive = output.with_suffix(".npz")
    if output.exists() or archive.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rows, sealed = frozen.load_sealed_rows()
    deps_before = dependencies(variant)
    cache_before = {session: frozen.authority_hashes(session) for session in plan.EXT4_SESSIONS}
    reports, values, started = {}, [], time.perf_counter()
    for session in plan.EXT4_SESSIONS:
        report, arrays = validate_session(session, rows[session], cache_before[session], VARIANTS[variant])
        reports[session] = report; values.append(arrays)
    prediction, full, v3, target, starts, sessions = (
        np.concatenate([item[i] for item in values]) if i < 5 else np.asarray(sum((item[i] for item in values), []))
        for i in range(6)
    )
    if prediction.shape[0] != EXPECTED_ENDPOINTS or sum(row["public_raw_bins"] for row in reports.values()) != EXPECTED_PUBLIC_RAW_BINS:
        raise RuntimeError("global endpoint/raw-bin cardinality drift")
    direct_error = frozen.require_close(prediction, full, "global direct")
    v3_error = frozen.require_close(prediction, v3, "global v3")
    score = frozen.native_r2(target, prediction)
    if abs(score - float(sealed["e8_pooled_r2"])) > frozen.ABS_TOL:
        raise RuntimeError("sealed pooled native R2 drift")
    deps_after = dependencies(variant)
    cache_after = {session: frozen.authority_hashes(session) for session in plan.EXT4_SESSIONS}
    _, sealed_after = frozen.load_sealed_rows()
    payload_after, addendum_after = frozen.file_sha256(frozen.PAYLOAD), frozen.file_sha256(frozen.ADDENDUM)
    if (deps_after != deps_before or cache_after != cache_before
            or payload_after != frozen.PAYLOAD_SHA256 or addendum_after != frozen.ADDENDUM_SHA256
            or sealed_after["e8_payload_sha256"] != frozen.PAYLOAD_SHA256):
        raise RuntimeError("candidate dependency or authority mutation")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive, prediction=prediction, direct_full=full, v3_oracle=v3, target=target, session=sessions, start=starts)
    body = {"schema": "family_runtime_v1_m2_candidate_complete_ext4_v1", "status": "PASS", "variant": variant,
            "endpoint_count": EXPECTED_ENDPOINTS, "public_raw_bin_count": EXPECTED_PUBLIC_RAW_BINS,
            "pooled_native_r2": score, "sealed_e8_pooled_native_r2": sealed["e8_pooled_r2"],
            "per_session": reports, "global_public_v3_candidate_max_abs_error": max(row["max_public_v3_candidate_abs_error"] for row in reports.values()),
            "global_endpoint_direct_max_abs_error": direct_error, "global_endpoint_v3_max_abs_error": v3_error,
            "dependency_sha256_pre": deps_before, "dependency_sha256_post": deps_after,
            "cache_provenance_map_bank_sha256_pre": cache_before, "cache_provenance_map_bank_sha256_post": cache_after,
            "payload_sha256_pre": frozen.PAYLOAD_SHA256, "payload_sha256_post": payload_after,
            "addendum_sha256_pre": frozen.ADDENDUM_SHA256, "addendum_sha256_post": addendum_after,
            "runtime_bank_authority": "hash-bound frozen e8 payload bank; compact-cache E0/T hashes are provenance-only",
            "stream_npz_sha256": frozen.file_sha256(archive), "elapsed_seconds": time.perf_counter()-started,
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "parameter_updates": 0, "not_selection_or_ranking": True, "not_hidden_or_official": True}
    output.write_text(json.dumps(body, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        if args.output is not None: parser.error("--preflight does not write --output")
        print(json.dumps(preflight(args.variant), indent=2, sort_keys=True))
    elif args.output is None:
        parser.error("--output is required unless --preflight")
    else:
        run(args.variant, args.output)
