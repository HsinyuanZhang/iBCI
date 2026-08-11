"""Production endpoint-only T4d loader for the RT nested-LOSO path.

The first NWB pass reads trial events, *timestamp metadata* for cursor
position, the deduplicated coordinate samples bracketing M24 endpoints, and
sorted spikes.  It never reads dense cursor-position coordinates or velocity.
Only after its T4d carrier is frozen does the ordinary decoder loader open
dense velocity for target/query scoring arrays.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from pynwb import NWBHDF5IO

from src.data.rt_k4_loader import BIN_SIZE_S, load_rt_session

M24 = 24
BLOCK_BINS = 5
LEAD_BINS = 2
MAX_BRACKET_SECONDS = 0.020
MIN_DISPLACEMENT_CM = 0.50


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _canonical_unit(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "")


def _go_cues(values: Any, n_trials: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim == 2:
        result = np.asarray(raw, dtype=np.float64)
    elif raw.ndim == 1 and raw.dtype == object:
        rows = [np.asarray(value, dtype=np.float64).reshape(-1) for value in raw]
        if len(rows) != n_trials: raise ValueError("go-cue row count mismatch")
        result = np.full((n_trials, max((row.size for row in rows), default=0)), np.nan)
        for index, row in enumerate(rows): result[index, :row.size] = row
    else:
        raise ValueError("go_cue_time_array must be numeric or ragged")
    if result.shape[0] != n_trials or result.shape[1] < 1: raise ValueError("invalid go-cue matrix")
    return result


def _complete_reaches(starts: np.ndarray, stops: np.ndarray, cues: np.ndarray, targets: np.ndarray) -> list[tuple[float, float]]:
    """Stage1-identical complete-cue parsing, restricted only after M24."""
    starts, stops = np.asarray(starts, dtype=np.float64).reshape(-1), np.asarray(stops, dtype=np.float64).reshape(-1)
    targets, cues = np.asarray(targets).reshape(-1), np.asarray(cues, dtype=np.float64)
    if starts.shape != stops.shape or starts.shape != targets.shape or cues.shape[0] != starts.size: raise ValueError("trial metadata shape mismatch")
    if not np.isfinite(starts).all() or not np.isfinite(stops).all() or not np.all(stops > starts): raise ValueError("invalid trial bounds")
    reaches: list[tuple[float, float]] = []
    for trial, (start, stop, count_raw) in enumerate(zip(starts, stops, targets)):
        if trial >= M24: break
        if not np.isfinite(count_raw): continue
        count = int(count_raw)
        if count < 1 or count > cues.shape[1]: continue
        declared, unused = cues[trial, :count], cues[trial, count:]
        if not np.isfinite(declared).all() or np.isfinite(unused).any() or not np.all(np.diff(declared) > 0): continue
        if declared[0] < start or declared[-1] >= stop: continue
        reaches.extend((float(left), float(right)) for left, right in zip(declared, np.r_[declared[1:], stop]))
    return reaches


def _endpoint_indices(times: np.ndarray, time_s: float) -> tuple[tuple[int, ...] | None, str | None]:
    index = int(np.searchsorted(times, time_s, side="left"))
    if index < times.size and times[index] == time_s: return (index,), None
    if index == 0 or index == times.size: return None, "endpoint_outside_cursor_position_range"
    if float(times[index] - times[index - 1]) > MAX_BRACKET_SECONDS: return None, "endpoint_bracket_exceeds_20ms"
    return (index - 1, index), None


def _endpoint_from_sparse(times: np.ndarray, coordinates: Mapping[int, np.ndarray], time_s: float) -> tuple[np.ndarray | None, str | None]:
    indices, reason = _endpoint_indices(times, time_s)
    if reason: return None, reason
    assert indices is not None
    values = [np.asarray(coordinates[index], dtype=np.float64) for index in indices]
    if any(value.shape != (2,) or not np.isfinite(value).all() for value in values): return None, "endpoint_position_nonfinite_or_non2d"
    if len(indices) == 1: return values[0].copy(), None
    left, right = indices
    return values[0] + ((time_s - times[left]) / (times[right] - times[left])) * (values[1] - values[0]), None


def _carrier_from_endpoint_payload(*, starts: np.ndarray, stops: np.ndarray, cues: np.ndarray,
                                   target_count: np.ndarray, position_times: np.ndarray,
                                   positions: np.ndarray | Mapping[int, np.ndarray], spike_times: list[Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit per-unit Stage1 AC4 on sparse endpoint labels, returning `[a,c,0,0]`."""
    times = np.asarray(position_times, dtype=np.float64).reshape(-1)
    if times.size < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0): raise ValueError("cursor_pos timestamps invalid")
    n_bins = int(np.ceil(float(np.max(np.asarray(stops, dtype=np.float64))) / BIN_SIZE_S - 1e-10)) + 1
    neural = np.zeros((n_bins, len(spike_times)), dtype=np.float64)
    for unit, source in enumerate(spike_times):
        spikes = np.asarray(source, dtype=np.float64).reshape(-1)
        if not np.isfinite(spikes).all(): raise ValueError("nonfinite spike time")
        bins = np.floor(spikes / BIN_SIZE_S + 1e-10).astype(np.int64)
        bins = bins[(bins >= 0) & (bins < n_bins)]
        np.add.at(neural[:, unit], bins, 1.0)
    if isinstance(positions, Mapping): sparse = positions
    else: sparse = {i: np.asarray(value, dtype=np.float64) for i, value in enumerate(np.asarray(positions, dtype=np.float64))}
    rows: list[tuple[float, np.ndarray]] = []
    for left_t, right_t in _complete_reaches(starts, stops, cues, target_count):
        left, right = int(np.ceil(left_t / BIN_SIZE_S - 1e-10)), int(np.floor(right_t / BIN_SIZE_S + 1e-10))
        first, first_reason = _endpoint_from_sparse(times, sparse, left_t)
        last, last_reason = _endpoint_from_sparse(times, sparse, right_t)
        if first_reason or last_reason or first is None or last is None or right <= left: continue
        displacement = last - first
        if not np.isfinite(displacement).all() or np.linalg.norm(displacement) < MIN_DISPLACEMENT_CM: continue
        blocks = [neural[index:index + BLOCK_BINS].sum(0) / (BLOCK_BINS * BIN_SIZE_S)
                  for index in range(left, right - BLOCK_BINS - LEAD_BINS + 1, BLOCK_BINS)
                  if index + BLOCK_BINS + LEAD_BINS <= right]
        if blocks: rows.append((float(math.atan2(float(displacement[1]), float(displacement[0]))), np.mean(blocks, axis=0)))
    if len(rows) < 3: raise ValueError("T4d requires at least three eligible endpoint reach rows in M24")
    theta = np.asarray([row[0] for row in rows], dtype=np.float64)
    design = np.column_stack([np.ones(theta.size), np.cos(theta), np.sin(theta)])
    rank = int(np.linalg.matrix_rank(design))
    if rank != 3: raise ValueError("T4d endpoint design is rank deficient")
    condition = float(np.linalg.cond(design))
    coefficients, *_ = np.linalg.lstsq(design, np.asarray([row[1] for row in rows]), rcond=None)
    if not np.isfinite(coefficients).all(): raise ValueError("T4d endpoint OLS is nonfinite")
    raw = np.column_stack([coefficients[1], coefficients[2], np.zeros(neural.shape[1]), np.zeros(neural.shape[1])]).astype(np.float32)
    return raw, {"carrier": "T4d=[a,c,0,0]", "m24_trials": M24, "eligible_reach_rows": len(rows), "design_rank": rank,
                 "design_condition": condition, "full_coefficients_ac": coefficients[1:3].T.tolist(), "dense_velocity_carrier": False,
                 "endpoint_direction_support": True, "dense_velocity_read": False, "query_velocity_used_in_carrier": False,
                 "access_log": ["trial_events", "cursor_position_timestamps", "cursor_position_endpoint_coordinate_samples", "spikes", "carrier_frozen"]}


def _sparse_coordinate_map(pos: Any, times: np.ndarray, endpoint_times: list[float]) -> tuple[dict[int, np.ndarray], dict[str, int]]:
    requested: set[int] = set()
    for endpoint in endpoint_times:
        indices, _reason = _endpoint_indices(times, endpoint)
        if indices is not None: requested.update(indices)
    ordered = np.asarray(sorted(requested), dtype=np.int64)
    # This is intentionally the only position-coordinate data read.  Timestamp
    # metadata is permitted in full solely to find the endpoint brackets.
    values = np.asarray(pos.data[ordered], dtype=np.float64) if ordered.size else np.empty((0, 2), dtype=np.float64)
    if values.shape != (ordered.size, 2): raise ValueError("cursor_pos selected endpoint coordinates must be [samples,2]")
    conversion, offset = np.asarray(pos.conversion), np.asarray(pos.offset)
    if conversion.ndim != 0 or offset.ndim != 0 or not np.isfinite(float(conversion)) or not np.isfinite(float(offset)): raise ValueError("cursor_pos conversion/offset must be finite scalars")
    values = values * float(conversion) + float(offset)
    return {int(index): values[row] for row, index in enumerate(ordered)}, {"unique_endpoint_coordinate_samples": int(ordered.size), "unique_endpoint_coordinate_scalars": int(2 * ordered.size)}


def load_rt_sparse_endpoint_t4d_session(nwb_path: str | Path) -> dict[str, Any]:
    """Freeze endpoint-only T4d, then attach it to the ordinary decoder raw dict."""
    path = Path(nwb_path)
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        if nwb.trials is None or nwb.units is None or "spike_times" not in nwb.units.colnames: raise ValueError("RT endpoint carrier requires trials and sorted spikes")
        trials = nwb.trials; required = {"start_time", "stop_time", "go_cue_time_array", "num_targets"}
        if not required.issubset(trials.colnames): raise ValueError("RT endpoint carrier trial schema missing")
        starts, stops = np.asarray(trials["start_time"][:], dtype=np.float64), np.asarray(trials["stop_time"][:], dtype=np.float64)
        cues = _go_cues(trials["go_cue_time_array"][:], starts.size); targets = np.asarray(trials["num_targets"][:])
        pos = nwb.processing["behavior"]["Position"].spatial_series["cursor_pos"]
        if _canonical_unit(pos.unit) != "cm" or pos.timestamps is None: raise ValueError("cursor_pos must be explicit-timestamped cm")
        times = np.asarray(pos.timestamps[:], dtype=np.float64)
        reaches = _complete_reaches(starts, stops, cues, targets)
        sparse_coordinates, coordinate_audit = _sparse_coordinate_map(pos, times, [time for reach in reaches for time in reach])
        raw_feature, audit = _carrier_from_endpoint_payload(starts=starts, stops=stops, cues=cues, target_count=targets, position_times=times, positions=sparse_coordinates, spike_times=list(nwb.units["spike_times"][:]))
    before = _hash(raw_feature.tolist())
    dense_raw = load_rt_session(path)
    after = _hash(raw_feature.tolist())
    if before != after or raw_feature.shape[0] != dense_raw["neural"].shape[1]: raise RuntimeError("frozen T4d carrier changed or has wrong unit count")
    audit.update(coordinate_audit)
    audit.update({"carrier_sha256_before_dense_target": before, "carrier_sha256_after_dense_target": after,
                  "carrier_unchanged_after_dense_target": True, "access_log": audit["access_log"] + ["decoder_dense_target_loader"],
                  "decoder_target_loaded_after_carrier_freeze": True,
                  "loaded_after_carrier_freeze_for_decoder_query_construction_and_scoring": True,
                  "dense_target_never_enters_carrier": True})
    dense_raw["t4d_raw_feature"] = raw_feature; dense_raw["t4d_audit"] = audit
    return dense_raw
