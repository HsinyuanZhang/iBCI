#!/usr/bin/env python3
"""CPU-only RT sparse-endpoint Stage 0B constructibility audit.

The primary path reads only trial events, cursor-position endpoints, and
sorted-unit spikes.  It freezes endpoint and neural one-row-per-reach
eligibility before it opens cursor velocity.  Dense velocity is then used only
for the predeclared semantic audit and dense-row reporting; it cannot alter
primary eligibility, coverage, or a fit design.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pynwb
from pynwb import NWBHDF5IO


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "sua_exploration/data/dandi_000688/sub-C"
DEFAULT_OUTPUT = ROOT / "sua_exploration/results/rt_simple_label_v1/stage0b"
PROTOCOL = ROOT / "sua_exploration/docs/RT_SPARSE_ENDPOINT_LABEL_CPU_PROTOCOL_ADDENDUM_20260810.md"
PROTOCOL_SHA256 = "741e41249ab0d5fd771f5298f885afc1e468f09f51d29cb4beb78dd63da89581"
STAGE0_RECEIPT = ROOT / "sua_exploration/results/rt_simple_label_v1/stage0/RT_SIMPLE_LABEL_STAGE0_METADATA_RECEIPT_v1.json"
STAGE0_RECEIPT_SHA256 = "ad8b2c583fddb8eef852fa4c39408beda9927a2c0dfbc714e4725df4cec9d37b"
ROOT_REVIEW = ROOT / "sua_exploration/results/rt_simple_label_v1/RT_SPARSE_ENDPOINT_PROTOCOL_ROOT_REVIEW_v1.json"
ROOT_REVIEW_SHA256 = "f5d6368ce93e733d5c6de9e68b750e44cd90ec689070ceab6a33f3468a88407f"
FOCUSED_TEST = ROOT / "sua_exploration/tests/test_rt_sparse_endpoint_stage0b.py"

SCHEMA = "rt_sparse_endpoint_stage0b_v1"
EXPECTED_SESSIONS = 15
M24 = 24
BIN_SECONDS = 0.020
BLOCK_BINS = 5
LEAD_BINS = 2
MAX_BRACKET_SECONDS = 0.020
MIN_DISPLACEMENT_CM = 0.50
MIN_LABEL_REACHES = 24
MIN_ENDPOINT_COVERAGE = 0.80
MIN_DENSE_AUDIT_COVERAGE = 0.80
MIN_DENSE_COSINE = 0.70
ACTIVE_EPSILON = 1.0e-3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _ceil_bin(time_s: float) -> int:
    return int(np.ceil(float(time_s) / BIN_SECONDS - 1.0e-10))


def _floor_bin(time_s: float) -> int:
    return int(np.floor(float(time_s) / BIN_SECONDS + 1.0e-10))


def _finite_float(value: float | np.floating[Any]) -> float | None:
    result = float(value)
    return result if np.isfinite(result) else None


def _canonical_unit(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "")


def require_unit(value: Any, *, allowed: set[str], label: str) -> str:
    unit = _canonical_unit(value)
    _need(unit in allowed, f"{label} unit must be one of {sorted(allowed)}, got {value!r}")
    return unit


def apply_nwb_conversion(data: np.ndarray, conversion: Any, offset: Any, *, label: str) -> np.ndarray:
    """Apply the scalar NWB physical-value transform, or fail rather than guess."""
    scale_array, offset_array = np.asarray(conversion), np.asarray(offset)
    _need(scale_array.ndim == 0 and offset_array.ndim == 0, f"{label} conversion/offset must be scalar")
    try:
        scale, shift = float(scale_array), float(offset_array)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} conversion/offset must be numeric scalars") from error
    _need(np.isfinite(scale) and np.isfinite(shift), f"{label} conversion/offset must be finite")
    values = np.asarray(data, dtype=np.float64)
    converted = values * scale + shift
    _need(np.all(np.isfinite(converted) | np.isnan(converted)), f"{label} conversion produced infinity")
    return converted


def _summary(values: Iterable[float]) -> dict[str, Any]:
    vector = np.asarray(list(values), dtype=np.float64)
    vector = vector[np.isfinite(vector)]
    if vector.size == 0:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(vector.size), "mean": float(np.mean(vector)), "median": float(np.median(vector)),
        "min": float(np.min(vector)), "max": float(np.max(vector)),
    }


def as_go_cue_matrix(values: Any, *, n_trials: int) -> np.ndarray:
    """Normalise a fixed-width/ragged cue column without inventing a cue."""
    raw = np.asarray(values)
    if raw.ndim == 2:
        matrix = np.asarray(raw, dtype=np.float64)
    elif raw.ndim == 1 and raw.dtype == object:
        rows = [np.asarray(value, dtype=np.float64).reshape(-1) for value in raw]
        _need(len(rows) == n_trials, "go-cue row count does not equal trial count")
        matrix = np.full((n_trials, max((row.size for row in rows), default=0)), np.nan)
        for index, row in enumerate(rows):
            matrix[index, :row.size] = row
    else:
        raise ValueError("go_cue_time_array must be a numeric matrix or ragged object vector")
    _need(matrix.shape[0] == n_trials and matrix.shape[1] >= 1, "invalid go-cue matrix")
    return matrix


def parse_reaches(
    starts: np.ndarray, stops: np.ndarray, cues: np.ndarray, num_targets: np.ndarray
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fail-closed go-cue parser.  It intentionally does not inspect velocity."""
    starts = np.asarray(starts, dtype=np.float64).reshape(-1)
    stops = np.asarray(stops, dtype=np.float64).reshape(-1)
    targets = np.asarray(num_targets).reshape(-1)
    matrix = np.asarray(cues, dtype=np.float64)
    _need(starts.shape == stops.shape == targets.shape, "trial metadata lengths differ")
    _need(matrix.ndim == 2 and matrix.shape[0] == starts.size, "cue matrix has wrong shape")
    _need(np.isfinite(starts).all() and np.isfinite(stops).all() and np.all(stops > starts), "invalid trial boundaries")
    reaches: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for trial, (start, stop, target_count) in enumerate(zip(starts, stops, targets)):
        record: dict[str, Any] = {"trial_index": trial, "complete_cue": False, "declared_reaches": 0,
                                  "accepted_event_reaches": 0, "exclusion_reason": None,
                                  "segment_exclusion_reasons": {}}
        if not np.isfinite(target_count):
            record["exclusion_reason"] = "nonfinite_num_targets"; records.append(record); continue
        count = int(target_count)
        record["declared_reaches"] = count
        if count < 1 or count > matrix.shape[1]:
            record["exclusion_reason"] = "num_targets_out_of_go_cue_width"; records.append(record); continue
        trial_cues, unused = matrix[trial, :count], matrix[trial, count:]
        if not np.isfinite(trial_cues).all():
            record["exclusion_reason"] = "nonfinite_required_go_cue"; records.append(record); continue
        if np.isfinite(unused).any():
            record["exclusion_reason"] = "finite_undeclared_go_cue"; records.append(record); continue
        if not np.all(np.diff(trial_cues) > 0.0):
            record["exclusion_reason"] = "nonmonotonic_go_cues"; records.append(record); continue
        if trial_cues[0] < start or trial_cues[-1] >= stop:
            record["exclusion_reason"] = "go_cue_outside_trial"; records.append(record); continue
        record["complete_cue"] = True
        ends = np.concatenate([trial_cues[1:], np.asarray([stop])])
        excluded: Counter[str] = Counter()
        for reach_index, (reach_start, reach_end) in enumerate(zip(trial_cues, ends)):
            left, right = _ceil_bin(float(reach_start)), _floor_bin(float(reach_end))
            if right <= left:
                excluded["segment_empty_after_whole_bin_alignment"] += 1
                continue
            reaches.append({"trial_index": trial, "reach_index": reach_index,
                            "start_s": float(reach_start), "end_s": float(reach_end),
                            "left_bin": left, "right_bin": right})
            record["accepted_event_reaches"] += 1
        record["segment_exclusion_reasons"] = {key: int(excluded[key]) for key in sorted(excluded)}
        records.append(record)
    return reaches, records


def interpolate_endpoint(times: np.ndarray, positions: np.ndarray, time_s: float) -> tuple[np.ndarray | None, str | None]:
    """Return an exact/linear sparse endpoint; never choose a nearest substitute."""
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    p = np.asarray(positions, dtype=np.float64)
    if t.size < 1 or p.shape != (t.size, 2) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        return None, "invalid_cursor_position_series"
    value = float(time_s)
    index = int(np.searchsorted(t, value, side="left"))
    if index < t.size and t[index] == value:
        if not np.isfinite(p[index]).all():
            return None, "endpoint_exact_position_nonfinite"
        return p[index].copy(), None
    if index == 0 or index == t.size:
        return None, "endpoint_outside_cursor_position_range"
    left, right = index - 1, index
    width = float(t[right] - t[left])
    if width > MAX_BRACKET_SECONDS:
        return None, "endpoint_bracket_exceeds_20ms"
    if not np.isfinite(p[left]).all() or not np.isfinite(p[right]).all():
        return None, "endpoint_bracket_position_nonfinite"
    alpha = (value - t[left]) / width
    return p[left] + alpha * (p[right] - p[left]), None


def attach_endpoint_labels(reaches: list[dict[str, Any]], times: np.ndarray, positions: np.ndarray) -> None:
    """Freeze endpoint eligibility before neural/dense processing."""
    for reach in reaches:
        origin, origin_reason = interpolate_endpoint(times, positions, reach["start_s"])
        endpoint, endpoint_reason = interpolate_endpoint(times, positions, reach["end_s"])
        reach["endpoint_label"] = False
        reach["endpoints_readable"] = False
        reach["endpoint_reason"] = origin_reason or endpoint_reason
        if origin_reason or endpoint_reason:
            continue
        assert origin is not None and endpoint is not None
        reach["endpoints_readable"] = True
        displacement = endpoint - origin
        length = float(np.linalg.norm(displacement))
        if not np.isfinite(length):
            reach["endpoint_reason"] = "nonfinite_endpoint_displacement"; continue
        if length < MIN_DISPLACEMENT_CM:
            reach["endpoint_reason"] = "short_endpoint_displacement"; continue
        theta = float(math.atan2(float(displacement[1]), float(displacement[0])))
        if not np.isfinite(theta):
            reach["endpoint_reason"] = "nonfinite_endpoint_direction"; continue
        reach.update({"endpoint_label": True, "endpoint_reason": None, "origin_cm": origin.tolist(),
                      "endpoint_cm": endpoint.tolist(), "displacement_cm": displacement.tolist(),
                      "length_cm": length, "theta_rad": theta})


def bin_spikes(spike_times: list[Any], *, n_bins: int) -> np.ndarray:
    neural = np.zeros((n_bins, len(spike_times)), dtype=np.float64)
    for channel, raw in enumerate(spike_times):
        spikes = np.asarray(raw, dtype=np.float64).reshape(-1)
        _need(np.isfinite(spikes).all(), f"unit {channel} has nonfinite spike times")
        bins = np.floor(spikes / BIN_SECONDS + 1.0e-10).astype(np.int64)
        keep = (bins >= 0) & (bins < n_bins)
        np.add.at(neural[:, channel], bins[keep], 1.0)
    return neural


def attach_primary_neural_rows(reaches: list[dict[str, Any]], neural: np.ndarray) -> None:
    """Freeze one-row-per-reach primary eligibility without dense velocity."""
    values = np.asarray(neural, dtype=np.float64)
    _need(values.ndim == 2 and values.shape[0] > 0, "invalid neural bins")
    for reach in reaches:
        reach["candidate_blocks"] = 0
        reach["eligible_blocks"] = 0
        reach["primary_row"] = False
        reach["primary_reason"] = None
        if not reach.get("endpoint_label", False):
            reach["primary_reason"] = "no_endpoint_label"; continue
        rates: list[np.ndarray] = []
        for left in range(int(reach["left_bin"]), int(reach["right_bin"]) - BLOCK_BINS - LEAD_BINS + 1, BLOCK_BINS):
            reach["candidate_blocks"] += 1
            right = left + BLOCK_BINS
            # The parser-derived reach alone implements same-reach +40ms containment.
            if left < 0 or right + LEAD_BINS > int(reach["right_bin"]) or right > values.shape[0]:
                continue
            rates.append(values[left:right].sum(axis=0) / (BLOCK_BINS * BIN_SECONDS))
        reach["eligible_blocks"] = len(rates)
        if not rates:
            reach["primary_reason"] = "no_neural_block_within_same_reach_plus40ms"; continue
        reach["reach_mean_rate"] = np.mean(np.asarray(rates), axis=0)
        reach["primary_row"] = True


def design_audit(reaches: list[dict[str, Any]], *, primary_only: bool) -> dict[str, Any]:
    selected = [row for row in reaches if row.get("primary_row", False)] if primary_only else [
        row for row in reaches if row.get("endpoint_label", False) and int(row.get("eligible_blocks", 0)) > 0
    ]
    if not primary_only:
        selected = [row for row in selected for _ in range(int(row["eligible_blocks"]))]
    if not selected:
        return {"rows": 0, "rank": 0, "condition": None}
    theta = np.asarray([row["theta_rad"] for row in selected], dtype=np.float64)
    design = np.column_stack([np.ones(theta.size), np.cos(theta), np.sin(theta)])
    rank = int(np.linalg.matrix_rank(design))
    condition = float(np.linalg.cond(design)) if rank == 3 else None
    return {"rows": int(theta.size), "rank": rank, "condition": _finite_float(condition) if condition is not None else None}


def circular_coverage(reaches: list[dict[str, Any]]) -> dict[str, Any]:
    theta = np.asarray([row["theta_rad"] for row in reaches if row.get("endpoint_label", False)], dtype=np.float64)
    if theta.size == 0:
        return {"directions": 0, "resultant_length": None, "largest_gap_rad": None, "histogram_8": [0] * 8}
    ordered = np.sort((theta + 2 * np.pi) % (2 * np.pi))
    gaps = np.diff(np.r_[ordered, ordered[0] + 2 * np.pi])
    histogram, _ = np.histogram(theta, bins=np.linspace(-np.pi, np.pi, 9))
    return {"directions": int(theta.size), "resultant_length": float(np.linalg.norm(np.mean(np.column_stack([np.cos(theta), np.sin(theta)]), axis=0))),
            "largest_gap_rad": float(np.max(gaps)), "histogram_8": [int(value) for value in histogram]}


def endpoint_scalar_report(reaches: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [row for row in reaches if row.get("endpoint_label", False)]
    times = sorted({float(time) for row in labels for time in (row["start_s"], row["end_s"])})
    return {"unique_endpoint_timestamps_s": times, "unique_endpoint_count": len(times),
            "raw_scalar_coordinates": int(2 * len(times)), "derived_direction_count": len(labels)}


def _bin_velocity(data: np.ndarray, timestamps: np.ndarray, *, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(data, dtype=np.float64)
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    _need(values.ndim == 2 and values.shape[1] == 2 and times.shape == (values.shape[0],), "invalid cursor velocity")
    _need(np.isfinite(values).all() and np.isfinite(times).all() and np.all(np.diff(times) > 0), "invalid cursor velocity samples")
    ids = np.floor(times / BIN_SECONDS + 1.0e-10).astype(np.int64)
    keep = (ids >= 0) & (ids < n_bins)
    sums = np.zeros((n_bins, 2), dtype=np.float64); counts = np.zeros(n_bins, dtype=np.int64)
    np.add.at(sums, ids[keep], values[keep]); np.add.at(counts, ids[keep], 1)
    velocity = np.zeros((n_bins, 2), dtype=np.float64); valid = counts > 0
    velocity[valid] = sums[valid] / counts[valid, None]
    return velocity, valid


def dense_audit_after_freeze(reaches: list[dict[str, Any]], velocity: np.ndarray, valid: np.ndarray) -> None:
    """Populate audit-only fields.  This function must follow primary freezing."""
    for reach in reaches:
        reach["dense_audit_reason"] = None; reach["dense_direction_cosine"] = None
        reach["dense_retained_rows"] = 0
        # Dense retained 100-ms target rows are counted only for reporting.
        # ``span=[left,right+2)`` is exactly the union of the sealed K4 tests
        # ``active[left:right] && active[left+2:right+2]``.
        for left in range(int(reach["left_bin"]), int(reach["right_bin"]) - BLOCK_BINS - LEAD_BINS + 1, BLOCK_BINS):
            right = left + BLOCK_BINS
            span = slice(left, right + LEAD_BINS)
            if valid[span].shape[0] == BLOCK_BINS + LEAD_BINS and valid[span].all() and np.all(np.any(np.abs(velocity[span]) >= ACTIVE_EPSILON, axis=1)):
                reach["dense_retained_rows"] += 1
        if not reach.get("endpoint_label", False):
            reach["dense_audit_reason"] = "no_endpoint_label"; continue
        left, right = _ceil_bin(float(reach["start_s"])), _floor_bin(float(reach["end_s"]))
        if right <= left or not valid[left:right].all():
            reach["dense_audit_reason"] = "missing_or_empty_dense_velocity_interval"; continue
        dense = BIN_SECONDS * velocity[left:right].sum(axis=0)
        dense_length = float(np.linalg.norm(dense))
        if not np.isfinite(dense_length) or dense_length < MIN_DISPLACEMENT_CM:
            reach["dense_audit_reason"] = "short_or_nonfinite_dense_displacement"; continue
        endpoint = np.asarray(reach["displacement_cm"], dtype=np.float64)
        cosine = float(np.dot(endpoint, dense) / (np.linalg.norm(endpoint) * dense_length))
        if not np.isfinite(cosine):
            reach["dense_audit_reason"] = "nonfinite_dense_direction_cosine"; continue
        reach["dense_direction_cosine"] = cosine


def _scope_rows(reaches: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    if scope == "support_m24": return [row for row in reaches if int(row["trial_index"]) < M24]
    if scope == "later": return [row for row in reaches if int(row["trial_index"]) >= M24]
    if scope == "all": return list(reaches)
    raise ValueError(f"unknown scope {scope}")


def _scope_records(records: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    if scope == "support_m24": return records[:M24]
    if scope == "later": return records[M24:]
    if scope == "all": return list(records)
    raise ValueError(f"unknown scope {scope}")


def session_stage0b_summary(session: str, reaches: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build report and gates only after dense audit fields have been added."""
    scopes: dict[str, Any] = {}
    for scope in ("support_m24", "later", "all"):
        rows = _scope_rows(reaches, scope)
        accepted = len(rows); labelled = [row for row in rows if row.get("endpoint_label", False)]
        readable = [row for row in rows if row.get("endpoints_readable", False)]
        short = [row for row in rows if row.get("endpoint_reason") == "short_endpoint_displacement"]
        primary = [row for row in rows if row.get("primary_row", False)]
        dense_defined = [row for row in labelled if row.get("dense_direction_cosine") is not None]
        scalar = endpoint_scalar_report(rows)
        dense_rows = int(sum(int(row.get("dense_retained_rows", 0)) for row in rows))
        dense_scalars = 2 * dense_rows
        scalar.update({"dense_rt_retained_rows": dense_rows, "dense_rt_target_scalars": dense_scalars,
                       "raw_to_dense_scalar_ratio": (float(scalar["raw_scalar_coordinates"] / dense_scalars) if dense_scalars else None)})
        declared = sum(int(record["declared_reaches"]) for record in _scope_records(records, scope))
        reasons = Counter(str(row.get("endpoint_reason")) for row in rows if row.get("endpoint_reason"))
        primary_reasons = Counter(str(row.get("primary_reason")) for row in rows if row.get("primary_reason"))
        scopes[scope] = {"declared_reaches": int(declared), "accepted_event_reaches": accepted,
                         "endpoint_readable_reaches": len(readable), "short_endpoint_displacement_reaches": len(short),
                         "endpoint_labelled_reaches": len(labelled),
                         "endpoint_label_coverage": (float(len(labelled) / accepted) if accepted else None),
                         "endpoint_exclusion_reasons": dict(sorted(reasons.items())),
                         "primary_reach_rows": len(primary), "primary_exclusion_reasons": dict(sorted(primary_reasons.items())),
                         "eligible_100ms_blocks": int(sum(int(row.get("eligible_blocks", 0)) for row in primary)),
                         "candidate_100ms_blocks": int(sum(int(row.get("candidate_blocks", 0)) for row in rows)),
                         "block_weighted_design_audit": design_audit(rows, primary_only=False),
                         "primary_reach_design": design_audit(rows, primary_only=True),
                         "circular_coverage": circular_coverage(rows), "endpoint_scalar_accounting": scalar,
                         "dense_audit_defined_pairs": len(dense_defined),
                         "dense_audit_coverage": (float(len(dense_defined) / len(labelled)) if labelled else None),
                         "dense_direction_cosine": _summary(row["dense_direction_cosine"] for row in dense_defined)}
    support = scopes["support_m24"]
    design = support["primary_reach_design"]
    conditions = {
        "endpoint_labels": support["endpoint_labelled_reaches"] >= MIN_LABEL_REACHES and (support["endpoint_label_coverage"] or 0.0) >= MIN_ENDPOINT_COVERAGE,
        "direction_design": design["rank"] == 3 and design["condition"] is not None,
        "dense_audit_availability": (support["dense_audit_coverage"] or 0.0) >= MIN_DENSE_AUDIT_COVERAGE,
        "dense_audit_agreement": (support["dense_direction_cosine"]["median"] or -math.inf) >= MIN_DENSE_COSINE,
    }
    trial_scalars = {}
    for trial in range(len(records)):
        rows = [row for row in reaches if int(row["trial_index"]) == trial]
        accounting = endpoint_scalar_report(rows)
        dense_rows = int(sum(int(row.get("dense_retained_rows", 0)) for row in rows))
        dense_scalars = 2 * dense_rows
        accounting.update({"dense_rt_retained_rows": dense_rows, "dense_rt_target_scalars": dense_scalars,
                           "raw_to_dense_scalar_ratio": (float(accounting["raw_scalar_coordinates"] / dense_scalars) if dense_scalars else None)})
        trial_scalars[str(trial)] = accounting
    return {"session": session, "m24": support, "scopes": scopes, "trial_records": records,
            "per_trial_endpoint_scalar_accounting": trial_scalars,
            "gate_conditions": conditions, "gate_pass": bool(all(conditions.values()))}


def aggregate_all15(session_rows: dict[str, dict[str, Any]], *, allowed_sessions: set[str]) -> dict[str, Any]:
    _need(len(session_rows) == EXPECTED_SESSIONS, f"expected {EXPECTED_SESSIONS} sessions, got {len(session_rows)}")
    _need(set(session_rows) == allowed_sessions, "aggregate session names differ from frozen Stage-0 allowlist")
    failures = {name: [key for key, value in row["gate_conditions"].items() if not value]
                for name, row in sorted(session_rows.items()) if not row["gate_pass"]}
    return {"expected_sessions": EXPECTED_SESSIONS, "session_count": len(session_rows),
            "all_sessions_pass": not failures, "failing_sessions": failures,
            "status": "PASS_STAGE0B_ENDPOINT_CONSTRUCTIBLE_NO_GPU" if not failures else "STOP_STAGE0B_ENDPOINT_SEMANTIC_OR_COVERAGE_FAILURE_NO_GPU"}


def _bound_inputs() -> dict[str, dict[str, str]]:
    paths = {"protocol": (PROTOCOL, PROTOCOL_SHA256), "stage0_receipt": (STAGE0_RECEIPT, STAGE0_RECEIPT_SHA256),
             "root_review": (ROOT_REVIEW, ROOT_REVIEW_SHA256)}
    result: dict[str, dict[str, str]] = {}
    for name, (path, expected) in paths.items():
        actual = sha256_file(path)
        _need(actual == expected, f"{name} SHA-256 drift: {actual} != {expected}")
        result[name] = {"path": str(path), "sha256": actual}
    return result


def implementation_provenance() -> dict[str, Any]:
    """Bind this implementation/runtime in a receipt; it is not a scientific gate."""
    script = Path(__file__).resolve()
    test = FOCUSED_TEST.resolve()
    _need(test.is_file(), f"focused test is missing: {test}")
    return {
        "script": {"path": str(script), "sha256": sha256_file(script)},
        "focused_test": {"path": str(test), "sha256": sha256_file(test)},
        "runtime": {"python_version": sys.version, "numpy_version": np.__version__,
                    "pynwb_version": getattr(pynwb, "__version__", None)},
        "not_a_gate": True,
    }


def load_stage0_allowlist() -> dict[str, dict[str, Any]]:
    """Bind Stage 0's exact session scope; count alone is never sufficient."""
    payload = json.loads(STAGE0_RECEIPT.read_text(encoding="utf-8"))
    _need(payload.get("status") == "STAGE0_COMPLETE_STOP_BEFORE_STAGE1", "Stage-0 receipt status is not complete-stop")
    _need(payload.get("stage0_read_rule_branch") == "B_NO_PER_REACH_TARGET_FIELD_DIRECTION_REQUIRES_BEHAVIOR_TRACE_ANNOTATION_COST_CLAIM_VOID", "Stage-0 receipt branch is not the required B branch")
    _need(payload.get("session_count") == EXPECTED_SESSIONS, "Stage-0 receipt session_count drift")
    rows = payload.get("sessions")
    _need(isinstance(rows, list) and len(rows) == EXPECTED_SESSIONS, "Stage-0 receipt sessions malformed")
    allowlist: dict[str, dict[str, Any]] = {}
    for row in rows:
        _need(isinstance(row, dict), "Stage-0 session row malformed")
        session, path = row.get("session"), row.get("nwb_path")
        _need(isinstance(session, str) and isinstance(path, str) and Path(path).is_absolute(), "Stage-0 session/path malformed")
        _need(session not in allowlist, f"duplicate Stage-0 session {session}")
        _need(isinstance(row.get("nwb_size_bytes"), int), f"Stage-0 session {session} lacks byte count")
        allowlist[session] = {"nwb_path": path, "nwb_size_bytes": int(row["nwb_size_bytes"])}
    return allowlist


def validate_run_scope(data_root: Path, output_dir: Path) -> None:
    _need(data_root.resolve() == DATA_ROOT.resolve(), f"data-root must equal frozen RT root {DATA_ROOT}")
    _need(output_dir.resolve() == DEFAULT_OUTPUT.resolve(), f"output-dir must equal frozen isolated Stage0B path {DEFAULT_OUTPUT}")


def validate_discovered_scope(paths: list[Path], allowlist: dict[str, dict[str, Any]]) -> None:
    _need(len(paths) == EXPECTED_SESSIONS, f"expected {EXPECTED_SESSIONS} discovered RT NWBs, got {len(paths)}")
    actual: dict[str, Path] = {}
    for path in paths:
        resolved = path.resolve()
        name = resolved.name.removeprefix("sub-C_").removesuffix("_behavior+ecephys.nwb")
        _need(name not in actual, f"duplicate discovered session {name}")
        actual[name] = resolved
    _need(set(actual) == set(allowlist), "discovered session names differ from Stage-0 allowlist")
    for name, resolved in actual.items():
        expected = allowlist[name]
        _need(str(resolved) == expected["nwb_path"], f"{name}: discovered NWB path differs from Stage-0 receipt")
        _need(resolved.stat().st_size == expected["nwb_size_bytes"], f"{name}: NWB byte count differs from Stage-0 receipt")


def validate_cpu_environment() -> tuple[dict[str, str | None], int]:
    _need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA_VISIBLE_DEVICES must be empty")
    caps = {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    _need(all(value in {"1", "2"} for value in caps.values()), f"thread caps must be 1 or 2: {caps}")
    niceness = int(os.nice(0))
    _need(niceness >= 10, f"current niceness must be >=10, got {niceness}")
    return caps, niceness


def audit_session(path: Path) -> dict[str, Any]:
    """Open one permitted NWB; dense velocity is read only after primary freeze."""
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        _need(nwb.trials is not None and nwb.units is not None, f"{path.name}: missing trials or units")
        trials = nwb.trials
        required = {"start_time", "stop_time", "go_cue_time_array", "num_targets"}
        _need(required.issubset(trials.colnames), f"{path.name}: missing trial columns")
        starts = np.asarray(trials["start_time"][:], dtype=np.float64); stops = np.asarray(trials["stop_time"][:], dtype=np.float64)
        reaches, records = parse_reaches(starts, stops, as_go_cue_matrix(trials["go_cue_time_array"][:], n_trials=len(starts)), np.asarray(trials["num_targets"][:]))
        position = nwb.processing["behavior"]["Position"].spatial_series["cursor_pos"]
        _need(position.timestamps is not None, f"{path.name}: cursor_pos lacks timestamps")
        position_unit = require_unit(position.unit, allowed={"cm"}, label="cursor_pos")
        position_conversion, position_offset = position.conversion, position.offset
        pos_times = np.asarray(position.timestamps[:], dtype=np.float64)
        pos_data = apply_nwb_conversion(np.asarray(position.data[:]), position_conversion, position_offset, label="cursor_pos")
        attach_endpoint_labels(reaches, pos_times, pos_data)
        _need("spike_times" in nwb.units.colnames, f"{path.name}: missing spike_times")
        n_bins = _ceil_bin(float(np.max(stops))) + 1
        neural = bin_spikes(list(nwb.units["spike_times"][:]), n_bins=n_bins)
        attach_primary_neural_rows(reaches, neural)  # Primary eligibility is now frozen.

        velocity_series = nwb.processing["behavior"].data_interfaces["Velocity"].time_series["cursor_vel"]
        _need(velocity_series.timestamps is not None, f"{path.name}: cursor_vel lacks timestamps")
        velocity_unit = require_unit(velocity_series.unit, allowed={"cm/s", "cm/sec", "centimeter/s", "centimeters/s"}, label="cursor_vel")
        velocity_conversion, velocity_offset = velocity_series.conversion, velocity_series.offset
        velocity_data = apply_nwb_conversion(np.asarray(velocity_series.data[:]), velocity_conversion, velocity_offset, label="cursor_vel")
        velocity, valid = _bin_velocity(velocity_data, np.asarray(velocity_series.timestamps[:]), n_bins=n_bins)
        dense_audit_after_freeze(reaches, velocity, valid)
    name = path.name.removeprefix("sub-C_").removesuffix("_behavior+ecephys.nwb")
    row = session_stage0b_summary(name, reaches, records)
    row["nwb"] = {"path": str(path.resolve()), "bytes": int(path.stat().st_size), "sha256": sha256_file(path),
                  "cursor_position": {"unit": position_unit, "conversion": float(position_conversion), "offset": float(position_offset)},
                  "cursor_velocity": {"unit": velocity_unit, "conversion": float(velocity_conversion), "offset": float(velocity_offset)}}
    return row


def write_atomic_receipt(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        receipt = temporary / "RT_SPARSE_ENDPOINT_STAGE0B_RECEIPT_v1.json"
        receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(receipt, 0o444)
        os.replace(temporary, output_dir)
    except Exception:
        for child in temporary.glob("*"):
            child.unlink()
        temporary.rmdir()
        raise
    return output_dir / receipt.name


def run(data_root: Path, output_dir: Path) -> Path:
    validate_run_scope(data_root, output_dir)
    caps, niceness = validate_cpu_environment()
    bound = _bound_inputs()
    allowlist = load_stage0_allowlist()
    paths = sorted(data_root.resolve().glob("sub-C_ses-RT-*_behavior+ecephys.nwb"))
    validate_discovered_scope(paths, allowlist)
    sessions = {row["session"]: row for row in (audit_session(path) for path in paths)}
    aggregate = aggregate_all15(sessions, allowed_sessions=set(allowlist))
    payload = {"schema": SCHEMA, "status": aggregate["status"], "bound_inputs": bound, "compute": {"cpu_only": True,
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "thread_caps": caps, "niceness": niceness, "torch_imported": False,
               "gpu_context_created": False}, "non_interference": {"processes_signalled": False, "gpu_jobs_started": False,
               "watched_directories_written": False, "decoder_constructed": False, "optimizer_constructed": False,
               "datamodule_constructed": False, "stage1_started": False}, "protocol": {"m24": M24, "bin_ms": 20,
               "block_bins": BLOCK_BINS, "behavior_lead_bins": LEAD_BINS, "min_endpoint_displacement_cm": MIN_DISPLACEMENT_CM,
               "max_endpoint_bracket_ms": 20, "primary_fit": "one_reach_one_mean_rate_row; cursor_vel_forbidden",
               "dense_velocity": "read_only_after_primary_eligibility_freeze_audit_only"}, "aggregate": aggregate,
               "implementation": implementation_provenance(), "sessions": sessions}
    return write_atomic_receipt(output_dir, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(run(args.data_root, args.output_dir))


if __name__ == "__main__":
    main()
