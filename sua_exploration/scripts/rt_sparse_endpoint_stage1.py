#!/usr/bin/env python3
"""CPU-only RT sparse-endpoint Stage 1 AC4 constructibility screen.

This program reconstructs one reach-mean neural response row per direction.
It does not open the continuous behaviour stream used by the Stage-0B semantic
audit.  All Stage-0B identities are checked before any OLS fit is created.
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
DEFAULT_OUTPUT = ROOT / "sua_exploration/results/rt_simple_label_v1/stage1"
PROTOCOL = ROOT / "sua_exploration/docs/RT_SPARSE_ENDPOINT_LABEL_CPU_PROTOCOL_ADDENDUM_20260810.md"
PROTOCOL_SHA256 = "741e41249ab0d5fd771f5298f885afc1e468f09f51d29cb4beb78dd63da89581"
STAGE0B = ROOT / "sua_exploration/results/rt_simple_label_v1/stage0b/RT_SPARSE_ENDPOINT_STAGE0B_RECEIPT_v1.json"
STAGE0B_SHA256 = "b88c91ab5cfb30b4a9ef978622e00488193c4ad18b09498c84ba76e10b9943b1"
STAGE0B_REVIEW = ROOT / "sua_exploration/results/rt_simple_label_v1/RT_SPARSE_ENDPOINT_STAGE0B_ROOT_REVIEW_v1.json"
STAGE0B_REVIEW_SHA256 = "5becc24de9db8ad6bc114faa083f0bf59c83a0e6b207f834293081d9ad870656"
STAGE0B_IMPLEMENTATION_SHA256 = "12575b0677cb347fedff7efe84f865af86e7f01c32a06c905dfd148fc524e52d"
FOCUSED_TEST = ROOT / "sua_exploration/tests/test_rt_sparse_endpoint_stage1.py"

SCHEMA = "rt_sparse_endpoint_stage1_v1"
EXPECTED_SESSIONS = 15
M24 = 24
BIN_SECONDS = 0.020
BLOCK_BINS = 5
LEAD_BINS = 2
MAX_BRACKET_SECONDS = 0.020
MIN_DISPLACEMENT_CM = 0.50
SPLIT_MEDIAN_GATE = 0.50
SPLIT_FRACTION_GATE = 0.50
SPLIT_CHANNEL_COSINE = 0.40
SHUFFLE_TRANSFER_GATE = 0.01
NORM_EPSILON = 1.0e-12
SHUFFLE_SEED = 42
SHUFFLE_NAMESPACE = "rt-sparse-endpoint-v1"


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ceil_bin(time_s: float) -> int:
    return int(np.ceil(float(time_s) / BIN_SECONDS - 1.0e-10))


def _floor_bin(time_s: float) -> int:
    return int(np.floor(float(time_s) / BIN_SECONDS + 1.0e-10))


def _canonical_unit(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "")


def apply_position_conversion(data: np.ndarray, conversion: Any, offset: Any) -> np.ndarray:
    scale, shift = np.asarray(conversion), np.asarray(offset)
    _need(scale.ndim == 0 and shift.ndim == 0, "cursor position conversion/offset must be scalar")
    try:
        multiplier, addition = float(scale), float(shift)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("cursor position conversion/offset must be numeric") from error
    _need(np.isfinite(multiplier) and np.isfinite(addition), "cursor position conversion/offset must be finite")
    result = np.asarray(data, dtype=np.float64) * multiplier + addition
    _need(np.all(np.isfinite(result) | np.isnan(result)), "cursor position conversion produced infinity")
    return result


def as_go_cue_matrix(values: Any, *, n_trials: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim == 2:
        matrix = np.asarray(raw, dtype=np.float64)
    elif raw.ndim == 1 and raw.dtype == object:
        items = [np.asarray(value, dtype=np.float64).reshape(-1) for value in raw]
        _need(len(items) == n_trials, "go-cue row count mismatch")
        matrix = np.full((n_trials, max((item.size for item in items), default=0)), np.nan)
        for index, item in enumerate(items):
            matrix[index, :item.size] = item
    else:
        raise ValueError("go-cue array must be numeric or ragged")
    _need(matrix.shape[0] == n_trials and matrix.shape[1] >= 1, "invalid go-cue matrix")
    return matrix


def parse_reaches(starts: np.ndarray, stops: np.ndarray, cues: np.ndarray, targets: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    starts, stops = np.asarray(starts, dtype=np.float64).reshape(-1), np.asarray(stops, dtype=np.float64).reshape(-1)
    targets, matrix = np.asarray(targets).reshape(-1), np.asarray(cues, dtype=np.float64)
    _need(starts.shape == stops.shape == targets.shape and matrix.shape[0] == starts.size, "trial metadata shape mismatch")
    _need(np.isfinite(starts).all() and np.isfinite(stops).all() and np.all(stops > starts), "invalid trial bounds")
    reaches: list[dict[str, Any]] = []; records: list[dict[str, Any]] = []
    for trial, (start, stop, count_raw) in enumerate(zip(starts, stops, targets)):
        record: dict[str, Any] = {"trial_index": trial, "declared_reaches": 0, "complete_cue": False, "exclusion_reason": None}
        if not np.isfinite(count_raw): record["exclusion_reason"] = "nonfinite_num_targets"; records.append(record); continue
        count = int(count_raw); record["declared_reaches"] = count
        if count < 1 or count > matrix.shape[1]: record["exclusion_reason"] = "num_targets_out_of_go_cue_width"; records.append(record); continue
        declared, unused = matrix[trial, :count], matrix[trial, count:]
        if not np.isfinite(declared).all(): record["exclusion_reason"] = "nonfinite_required_go_cue"; records.append(record); continue
        if np.isfinite(unused).any(): record["exclusion_reason"] = "finite_undeclared_go_cue"; records.append(record); continue
        if not np.all(np.diff(declared) > 0): record["exclusion_reason"] = "nonmonotonic_go_cues"; records.append(record); continue
        if declared[0] < start or declared[-1] >= stop: record["exclusion_reason"] = "go_cue_outside_trial"; records.append(record); continue
        record["complete_cue"] = True
        for index, (left_time, right_time) in enumerate(zip(declared, np.r_[declared[1:], stop])):
            left, right = _ceil_bin(float(left_time)), _floor_bin(float(right_time))
            if right > left:
                reaches.append({"trial_index": trial, "reach_index": index, "start_s": float(left_time), "end_s": float(right_time), "left_bin": left, "right_bin": right})
        records.append(record)
    return reaches, records


def interpolate_endpoint(times: np.ndarray, positions: np.ndarray, time_s: float) -> tuple[np.ndarray | None, str | None]:
    times, positions = np.asarray(times, dtype=np.float64).reshape(-1), np.asarray(positions, dtype=np.float64)
    if times.size < 1 or positions.shape != (times.size, 2) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        return None, "invalid_cursor_position_series"
    index = int(np.searchsorted(times, float(time_s), side="left"))
    if index < times.size and times[index] == time_s:
        return (positions[index].copy(), None) if np.isfinite(positions[index]).all() else (None, "endpoint_exact_position_nonfinite")
    if index == 0 or index == times.size: return None, "endpoint_outside_cursor_position_range"
    left, right = index - 1, index
    if float(times[right] - times[left]) > MAX_BRACKET_SECONDS: return None, "endpoint_bracket_exceeds_20ms"
    if not np.isfinite(positions[left]).all() or not np.isfinite(positions[right]).all(): return None, "endpoint_bracket_position_nonfinite"
    return positions[left] + ((float(time_s) - times[left]) / (times[right] - times[left])) * (positions[right] - positions[left]), None


def attach_endpoint_labels(reaches: list[dict[str, Any]], times: np.ndarray, positions: np.ndarray) -> None:
    for row in reaches:
        start, start_reason = interpolate_endpoint(times, positions, row["start_s"])
        end, end_reason = interpolate_endpoint(times, positions, row["end_s"])
        row.update({"endpoint_label": False, "endpoints_readable": False, "endpoint_reason": start_reason or end_reason})
        if start_reason or end_reason: continue
        assert start is not None and end is not None
        row["endpoints_readable"] = True; difference = end - start; length = float(np.linalg.norm(difference))
        if not np.isfinite(length): row["endpoint_reason"] = "nonfinite_endpoint_displacement"; continue
        if length < MIN_DISPLACEMENT_CM: row["endpoint_reason"] = "short_endpoint_displacement"; continue
        theta = float(math.atan2(float(difference[1]), float(difference[0])))
        if not np.isfinite(theta): row["endpoint_reason"] = "nonfinite_endpoint_direction"; continue
        row.update({"endpoint_label": True, "endpoint_reason": None, "theta_rad": theta, "displacement_cm": difference.tolist()})


def bin_spikes(spike_times: list[Any], *, n_bins: int) -> np.ndarray:
    output = np.zeros((n_bins, len(spike_times)), dtype=np.float64)
    for channel, item in enumerate(spike_times):
        spikes = np.asarray(item, dtype=np.float64).reshape(-1); _need(np.isfinite(spikes).all(), "nonfinite spike time")
        bins = np.floor(spikes / BIN_SECONDS + 1e-10).astype(np.int64); keep = (bins >= 0) & (bins < n_bins)
        np.add.at(output[:, channel], bins[keep], 1.0)
    return output


def attach_primary_rows(reaches: list[dict[str, Any]], neural: np.ndarray) -> None:
    for row in reaches:
        row.update({"candidate_blocks": 0, "eligible_blocks": 0, "primary_row": False, "primary_reason": None})
        if not row.get("endpoint_label"): row["primary_reason"] = "no_endpoint_label"; continue
        rates: list[np.ndarray] = []
        for left in range(int(row["left_bin"]), int(row["right_bin"]) - BLOCK_BINS - LEAD_BINS + 1, BLOCK_BINS):
            row["candidate_blocks"] += 1; right = left + BLOCK_BINS
            if left < 0 or right + LEAD_BINS > row["right_bin"] or right > neural.shape[0]: continue
            rates.append(neural[left:right].sum(axis=0) / (BLOCK_BINS * BIN_SECONDS))
        row["eligible_blocks"] = len(rates)
        if rates: row["reach_mean_rate"] = np.mean(np.asarray(rates), axis=0); row["primary_row"] = True
        else: row["primary_reason"] = "no_neural_block_within_same_reach_plus40ms"


def primary_design(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row.get("primary_row")]
    if not selected: return {"rows": 0, "rank": 0, "condition": None}
    theta = np.asarray([row["theta_rad"] for row in selected]); matrix = np.column_stack([np.ones(theta.size), np.cos(theta), np.sin(theta)])
    rank = int(np.linalg.matrix_rank(matrix))
    return {"rows": int(theta.size), "rank": rank, "condition": float(np.linalg.cond(matrix)) if rank == 3 else None}


def endpoint_scalar_accounting(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [row for row in rows if row.get("endpoint_label")]
    times = sorted({float(time) for row in labels for time in (row["start_s"], row["end_s"])})
    return {"unique_endpoint_timestamps_s": times, "unique_endpoint_count": len(times), "raw_scalar_coordinates": 2 * len(times), "derived_direction_count": len(labels)}


def _rows_for(rows: list[dict[str, Any]], start: int, stop: int | None) -> list[dict[str, Any]]:
    return [row for row in rows if start <= int(row["trial_index"]) < (stop if stop is not None else math.inf) and row.get("primary_row")]


def fit_ac4(rows: list[dict[str, Any]]) -> tuple[np.ndarray | None, dict[str, Any]]:
    if len(rows) < 3: return None, {"status": "undefined", "reason": "fewer_than_three_reach_rows"}
    theta = np.asarray([row["theta_rad"] for row in rows]); response = np.asarray([row["reach_mean_rate"] for row in rows])
    design = np.column_stack([np.ones(theta.size), np.cos(theta), np.sin(theta)]); rank = int(np.linalg.matrix_rank(design))
    if rank != 3: return None, {"status": "undefined", "reason": "rank_deficient", "rank": rank}
    condition = float(np.linalg.cond(design))
    if not np.isfinite(condition): return None, {"status": "undefined", "reason": "nonfinite_condition", "rank": rank}
    coefficient, *_ = np.linalg.lstsq(design, response, rcond=None)
    if not np.isfinite(coefficient).all(): return None, {"status": "undefined", "reason": "nonfinite_ols", "rank": rank, "condition": condition}
    return coefficient, {"status": "defined", "rows": int(theta.size), "rank": rank, "condition": condition}


def split_cosines(first: np.ndarray | None, second: np.ndarray | None, channels: int) -> tuple[np.ndarray, dict[str, Any]]:
    result = np.full(channels, np.nan)
    if first is None or second is None: return result, {"defined_channels": 0, "median": None, "fraction_ge_040": None}
    vectors_one, vectors_two = first[1:3].T, second[1:3].T
    norm_one, norm_two = np.linalg.norm(vectors_one, axis=1), np.linalg.norm(vectors_two, axis=1)
    defined = (norm_one > NORM_EPSILON) & (norm_two > NORM_EPSILON)
    result[defined] = np.sum(vectors_one[defined] * vectors_two[defined], axis=1) / (norm_one[defined] * norm_two[defined])
    finite = result[np.isfinite(result)]
    return result, {"defined_channels": int(finite.size), "median": float(np.median(finite)) if finite.size else None,
                    "fraction_ge_040": float(np.mean(finite >= SPLIT_CHANNEL_COSINE)) if finite.size else None}


def deterministic_rotation(reach_count: int, *, session: str, seed: int = 42) -> tuple[np.ndarray, int]:
    _need(reach_count >= 2, "reach rotation needs at least two reaches")
    value = int.from_bytes(hashlib.sha256(f"{SHUFFLE_NAMESPACE}:{session}:{seed}".encode()).digest()[:8], "little")
    shift = 1 + value % (reach_count - 1); order = np.roll(np.arange(reach_count, dtype=np.int64), shift)
    _need(not np.any(order == np.arange(reach_count)), "rotation retained a reach identity")
    return order, int(shift)


def r2_by_channel(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    observed, predicted = np.asarray(observed, dtype=np.float64), np.asarray(predicted, dtype=np.float64)
    _need(observed.shape == predicted.shape and observed.ndim == 2, "R2 shape mismatch")
    total = np.sum((observed - observed.mean(axis=0)) ** 2, axis=0); residual = np.sum((observed - predicted) ** 2, axis=0)
    output = np.full(observed.shape[1], np.nan); defined = total > 0
    output[defined] = 1.0 - residual[defined] / total[defined]
    return output


def forward_transfer(session: str, support: list[dict[str, Any]], later: list[dict[str, Any]], channels: int) -> dict[str, Any]:
    correct, fit = fit_ac4(support)
    if correct is None or not later: return {"status": "undefined", "reason": "support_or_later_undefined", "session_median_shuffle": None, "session_median_intercept": None}
    order, shift = deterministic_rotation(len(support), session=session, seed=SHUFFLE_SEED)
    shuffled_rows = [{**row, "theta_rad": support[int(index)]["theta_rad"]} for row, index in zip(support, order)]
    shuffled, shuffled_fit = fit_ac4(shuffled_rows)
    if shuffled is None: return {"status": "undefined", "reason": "shuffle_fit_undefined", "session_median_shuffle": None, "session_median_intercept": None}
    theta = np.asarray([row["theta_rad"] for row in later]); design = np.column_stack([np.ones(theta.size), np.cos(theta), np.sin(theta)])
    observed = np.asarray([row["reach_mean_rate"] for row in later]); r_correct = r2_by_channel(observed, design @ correct)
    r_shuffle = r2_by_channel(observed, design @ shuffled); r_intercept = r2_by_channel(observed, np.broadcast_to(np.mean([row["reach_mean_rate"] for row in support], axis=0), observed.shape))
    delta_shuffle, delta_intercept = r_correct - r_shuffle, r_correct - r_intercept
    defined = np.isfinite(delta_shuffle) & np.isfinite(delta_intercept)
    return {"status": "defined", "correct_fit": fit, "shuffle_fit": shuffled_fit, "rotation_shift": shift,
            "rotation_order": order.tolist(), "defined_channels": int(defined.sum()), "undefined_channels": int(channels - defined.sum()),
            "session_median_shuffle": float(np.median(delta_shuffle[defined])) if defined.any() else None,
            "session_median_intercept": float(np.median(delta_intercept[defined])) if defined.any() else None,
            "per_channel_delta_shuffle": [float(value) if np.isfinite(value) else None for value in delta_shuffle],
            "per_channel_delta_intercept": [float(value) if np.isfinite(value) else None for value in delta_intercept]}


def _summary(values: list[float | None]) -> dict[str, Any]:
    defined = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if defined.size == 0: return {"defined_sessions": 0, "mean": None, "median": None, "positive": 0, "zero": 0, "negative": 0, "leave_largest_out_mean": None, "removed_session": None, "removed_session_index": None}
    indices = [index for index, value in enumerate(values) if value is not None and np.isfinite(value)]
    pairs = list(zip(indices, defined.tolist()))
    removed_index, _ = max(pairs, key=lambda pair: (abs(pair[1]), -pair[0]))
    keep = [value for index, value in pairs if index != removed_index]
    return {"defined_sessions": int(defined.size), "mean": float(np.mean(defined)), "median": float(np.median(defined)),
            "positive": int(np.sum(defined > 0)), "zero": int(np.sum(defined == 0)), "negative": int(np.sum(defined < 0)),
            "leave_largest_out_mean": float(np.mean(keep)) if keep else None, "removed_session_index": int(removed_index)}


def aggregate_stage1(rows: dict[str, dict[str, Any]], *, allowlist: set[str]) -> dict[str, Any]:
    _need(set(rows) == allowlist and len(rows) == EXPECTED_SESSIONS, "Stage1 aggregate allowlist mismatch")
    names = sorted(rows); split_values = [rows[name]["split"]["median"] for name in names]
    shuffle_values = [rows[name]["forward"]["session_median_shuffle"] for name in names]
    intercept_values = [rows[name]["forward"]["session_median_intercept"] for name in names]
    pooled = [value for name in names for value in rows[name]["split"]["per_channel"] if value is not None]
    split_median = _summary(split_values); shuffled = _summary(shuffle_values); intercept = _summary(intercept_values)
    for report in (shuffled, intercept):
        index = report.pop("removed_session_index")
        report["removed_session"] = names[index] if index is not None else None
    split_defined = split_median["defined_sessions"] == EXPECTED_SESSIONS
    pooled_fraction = float(np.mean(np.asarray(pooled) >= SPLIT_CHANNEL_COSINE)) if pooled else None
    conditions = {"split_direction": split_defined and split_median["median"] is not None and split_median["median"] >= SPLIT_MEDIAN_GATE,
                  "split_coverage": pooled_fraction is not None and pooled_fraction >= SPLIT_FRACTION_GATE,
                  "correct_pair_transfer": shuffled["defined_sessions"] == EXPECTED_SESSIONS and shuffled["median"] is not None and shuffled["median"] >= SHUFFLE_TRANSFER_GATE,
                  "directional_value": intercept["defined_sessions"] == EXPECTED_SESSIONS and intercept["median"] is not None and intercept["median"] > 0}
    return {"session_order": names, "split_session_median": {**split_median, "ordered_values": [{"session": name, "value": split_values[index]} for index, name in enumerate(names)]}, "pooled_split_fraction_ge_040": pooled_fraction,
            "forward_correct_minus_shuffle": {**shuffled, "ordered_values": [{"session": name, "value": shuffle_values[index]} for index, name in enumerate(names)]},
            "forward_correct_minus_intercept": {**intercept, "ordered_values": [{"session": name, "value": intercept_values[index]} for index, name in enumerate(names)]},
            "gate_conditions": conditions, "status": "PASS_STAGE1_SPARSE_ENDPOINT_AC4_CONSTRUCTIBLE_NO_GPU" if all(conditions.values()) else "STOP_STAGE1_SPARSE_ENDPOINT_AC4_CONSTRUCTIBILITY_FAILED_NO_GPU"}


def validate_stage0b_identity(session: str, rows: list[dict[str, Any]], receipt_row: dict[str, Any]) -> None:
    support = [row for row in rows if row["trial_index"] < M24]; primary = _rows_for(rows, 0, M24)
    expected = receipt_row["m24"]; design = primary_design(support); scalar = endpoint_scalar_accounting(support)
    _need(sum(bool(row.get("endpoint_label")) for row in support) == expected["endpoint_labelled_reaches"], f"{session}: Stage0B endpoint-label identity drift")
    _need(len(primary) == expected["primary_reach_rows"], f"{session}: Stage0B primary-row identity drift")
    _need(design["rank"] == expected["primary_reach_design"]["rank"] and design["rows"] == expected["primary_reach_design"]["rows"], f"{session}: Stage0B design identity drift")
    expected_condition = expected["primary_reach_design"]["condition"]
    _need((design["condition"] is None and expected_condition is None) or (design["condition"] is not None and expected_condition is not None and np.isclose(design["condition"], expected_condition, rtol=0.0, atol=1e-12)), f"{session}: Stage0B condition identity drift")
    for key in ("unique_endpoint_timestamps_s", "unique_endpoint_count", "raw_scalar_coordinates", "derived_direction_count"):
        _need(scalar[key] == expected["endpoint_scalar_accounting"][key], f"{session}: Stage0B scalar accounting identity drift for {key}")
    dense = expected["endpoint_scalar_accounting"]
    _need(dense["dense_rt_target_scalars"] == 2 * dense["dense_rt_retained_rows"], f"{session}: Stage0B dense scalar accounting internally inconsistent")


def _bound_inputs() -> tuple[dict[str, dict[str, str]], dict[str, Any], dict[str, Any]]:
    bindings = {"protocol": (PROTOCOL, PROTOCOL_SHA256), "stage0b": (STAGE0B, STAGE0B_SHA256), "stage0b_review": (STAGE0B_REVIEW, STAGE0B_REVIEW_SHA256)}
    output: dict[str, dict[str, str]] = {}
    for name, (path, expected) in bindings.items():
        actual = sha256_file(path); _need(actual == expected, f"{name} SHA drift")
        output[name] = {"path": str(path), "sha256": actual}
    stage0b, review = json.loads(STAGE0B.read_text()), json.loads(STAGE0B_REVIEW.read_text())
    _need(stage0b.get("schema") == "rt_sparse_endpoint_stage0b_v1" and stage0b.get("status") == "PASS_STAGE0B_ENDPOINT_CONSTRUCTIBLE_NO_GPU", "Stage0B receipt is not passing")
    _need(stage0b.get("bound_inputs", {}).get("protocol", {}).get("sha256") == PROTOCOL_SHA256, "Stage0B protocol identity drift")
    _need(review.get("schema") == "rt_sparse_endpoint_stage0b_root_review_v1" and review.get("status") == "PASS_REVIEW_AUTHORIZE_CPU_STAGE1_ONLY", "Stage0B review does not authorize Stage1")
    _need(review.get("stage0b_receipt", {}).get("sha256") == STAGE0B_SHA256 and review.get("stage0b_receipt", {}).get("terminal_status") == "PASS_STAGE0B_ENDPOINT_CONSTRUCTIBLE_NO_GPU", "Stage0B review receipt identity drift")
    _need(stage0b.get("implementation", {}).get("script", {}).get("sha256") == STAGE0B_IMPLEMENTATION_SHA256, "Stage0B implementation identity drift")
    return output, stage0b, review


def implementation_provenance() -> dict[str, Any]:
    script, test = Path(__file__).resolve(), FOCUSED_TEST.resolve(); _need(test.is_file(), "focused test missing")
    return {"script": {"path": str(script), "sha256": sha256_file(script)}, "focused_test": {"path": str(test), "sha256": sha256_file(test)},
            "runtime": {"python_version": sys.version, "numpy_version": np.__version__, "pynwb_version": getattr(pynwb, "__version__", None)}, "not_a_gate": True}


def validate_scope(data_root: Path, output_dir: Path, stage0b: dict[str, Any]) -> dict[str, dict[str, Any]]:
    _need(data_root.resolve() == DATA_ROOT.resolve(), "data-root differs from frozen RT root")
    _need(output_dir.resolve() == DEFAULT_OUTPUT.resolve(), "output-dir differs from frozen Stage1 directory")
    sessions = stage0b.get("sessions"); _need(isinstance(sessions, dict) and len(sessions) == EXPECTED_SESSIONS, "Stage0B allowlist malformed")
    allowlist: dict[str, dict[str, Any]] = {}
    for name, row in sessions.items():
        nwb = row.get("nwb", {}); _need(isinstance(nwb.get("path"), str) and Path(nwb["path"]).is_absolute() and isinstance(nwb.get("bytes"), int) and isinstance(nwb.get("sha256"), str), "Stage0B NWB provenance malformed")
        allowlist[name] = nwb
    return allowlist


def validate_cpu() -> tuple[dict[str, str | None], int]:
    _need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA_VISIBLE_DEVICES must be empty")
    caps = {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    _need(all(value in {"1", "2"} for value in caps.values()), "thread caps must be 1 or 2")
    niceness = int(os.nice(0)); _need(niceness >= 10, "current niceness must be >=10")
    return caps, niceness


def audit_session(path: Path, stage0b_row: dict[str, Any]) -> dict[str, Any]:
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read(); _need(nwb.trials is not None and nwb.units is not None, "missing trials or units")
        trials = nwb.trials; required = {"start_time", "stop_time", "go_cue_time_array", "num_targets"}; _need(required.issubset(trials.colnames), "RT trial fields missing")
        starts, stops = np.asarray(trials["start_time"][:], dtype=np.float64), np.asarray(trials["stop_time"][:], dtype=np.float64)
        reaches, _records = parse_reaches(starts, stops, as_go_cue_matrix(trials["go_cue_time_array"][:], n_trials=len(starts)), np.asarray(trials["num_targets"][:]))
        position = nwb.processing["behavior"]["Position"].spatial_series["cursor_pos"]
        _need(_canonical_unit(position.unit) == "cm" and position.timestamps is not None, "cursor position must be timestamped cm")
        attach_endpoint_labels(reaches, np.asarray(position.timestamps[:]), apply_position_conversion(np.asarray(position.data[:]), position.conversion, position.offset))
        _need("spike_times" in nwb.units.colnames, "sorted spike times missing")
        neural = bin_spikes(list(nwb.units["spike_times"][:]), n_bins=_ceil_bin(float(np.max(stops))) + 1)
        attach_primary_rows(reaches, neural)
    name = path.name.removeprefix("sub-C_").removesuffix("_behavior+ecephys.nwb")
    validate_stage0b_identity(name, reaches, stage0b_row)
    support, first, second, later = _rows_for(reaches, 0, M24), _rows_for(reaches, 0, 12), _rows_for(reaches, 12, 24), _rows_for(reaches, M24, None)
    full, full_fit = fit_ac4(support); left, left_fit = fit_ac4(first); right, right_fit = fit_ac4(second)
    channels = neural.shape[1]; cosine, split = split_cosines(left, right, channels); split["per_channel"] = [float(value) if np.isfinite(value) else None for value in cosine]
    forward = forward_transfer(name, support, later, channels)
    return {"session": name, "nwb": {"path": str(path.resolve()), "bytes": int(path.stat().st_size), "sha256": sha256_file(path)},
            "identity_verified_before_fit": True, "full_fit": full_fit, "full_coefficients_ac": (full[1:3].T.tolist() if full is not None else None),
            "split": split, "split_fits": {"first": left_fit, "second": right_fit}, "forward": forward,
            "reach_counts": {"support": len(support), "first_half": len(first), "second_half": len(second), "later": len(later)}}


def write_atomic(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_dir = output_dir.resolve(); _need(not output_dir.exists(), f"output exists: {output_dir}"); output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        receipt = temporary / "RT_SPARSE_ENDPOINT_STAGE1_RECEIPT_v1.json"; receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.chmod(receipt, 0o444); os.replace(temporary, output_dir)
    except Exception:
        for child in temporary.glob("*"): child.unlink()
        temporary.rmdir(); raise
    return output_dir / receipt.name


def run(data_root: Path, output_dir: Path) -> Path:
    bound, stage0b, _review = _bound_inputs(); allow = validate_scope(data_root, output_dir, stage0b); caps, niceness = validate_cpu()
    paths = sorted(data_root.resolve().glob("sub-C_ses-RT-*_behavior+ecephys.nwb")); _need(len(paths) == EXPECTED_SESSIONS, "discovered RT NWB count mismatch")
    actual = {path.name.removeprefix("sub-C_").removesuffix("_behavior+ecephys.nwb"): path.resolve() for path in paths}; _need(set(actual) == set(allow), "discovered Stage1 allowlist mismatch")
    for name, path in actual.items(): _need(str(path) == allow[name]["path"] and path.stat().st_size == allow[name]["bytes"] and sha256_file(path) == allow[name]["sha256"], f"{name}: NWB provenance drift")
    sessions = {name: audit_session(path, stage0b["sessions"][name]) for name, path in actual.items()}
    aggregate = aggregate_stage1(sessions, allowlist=set(allow))
    payload = {"schema": SCHEMA, "status": aggregate["status"], "bound_inputs": bound, "implementation": implementation_provenance(),
               "compute": {"cpu_only": True, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "thread_caps": caps, "niceness": niceness, "torch_imported": False, "gpu_context_created": False},
               "non_interference": {"processes_signalled": False, "gpu_jobs_started": False, "watched_directories_written": False, "decoder_constructed": False, "optimizer_constructed": False, "datamodule_constructed": False, "continuous_behavior_stream_read": False, "stage2_started": False, "decoder_eval": False, "paper_modified": False},
               "protocol": {"m24_trials": M24, "split_trial_ranges": [[0, 12], [12, 24]], "raw_bin_ms": 20, "block_bins": BLOCK_BINS, "behavior_lead_bins": LEAD_BINS, "min_endpoint_displacement_cm": MIN_DISPLACEMENT_CM, "split_norm_epsilon": NORM_EPSILON, "one_reach_one_response_row": True, "shuffle_seed": SHUFFLE_SEED, "shuffle_namespace": SHUFFLE_NAMESPACE},
               "gates": {"split_session_median_cosine_gte": SPLIT_MEDIAN_GATE, "split_pooled_fraction_cosine_gte": SPLIT_CHANNEL_COSINE, "split_pooled_fraction_minimum": SPLIT_FRACTION_GATE, "correct_minus_shuffle_session_median_gte": SHUFFLE_TRANSFER_GATE, "correct_minus_intercept_session_median_gt": 0.0},
               "aggregate": aggregate, "sessions": sessions}
    return write_atomic(output_dir, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--data-root", type=Path, default=DATA_ROOT); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); print(run(args.data_root, args.output_dir))


if __name__ == "__main__": main()
