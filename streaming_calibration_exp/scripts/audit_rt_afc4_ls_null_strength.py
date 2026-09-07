#!/usr/bin/env python3
"""CPU-only support audit of RT AFC4's continuous-velocity label nulls.

The production ``afc4_ls`` control cyclically rotates 100-ms velocity labels
*within each event-qualified reach*.  It changes every block index, but a reach
is often directionally homogeneous; therefore index non-identity alone does
not establish a strong label-association null.  This audit quantifies that
fact on the exact M24 support calculation and constructs a stronger, fairer
comparison null before any decoder/GPU experiment is considered.

Safety contract
---------------
Only trials ``[0, 24)`` are opened.  The NWB reader takes the stop time of
trial 24 as a strict cutoff and reads only velocity samples and spike times
strictly before that cutoff.  It does not call ``load_rt_session``, construct
a DataModule/decoder/optimizer, instantiate CUDA tensors, or read a target
query velocity sample.  The output is a support-only diagnostic receipt, not
an RT decoding result and not an authorization for a GPU run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pynwb


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.data.falcon_k4_features import (  # noqa: E402
    K4_ACTIVE_EPSILON,
    K4_BEHAVIOR_LEAD_BINS,
    K4_BLOCK_WIDTH_BINS,
    K4_RAW_BIN_MS,
    deterministic_k4_block_label_permutation,
    k4_from_raw_calibration,
)
from src.data.rt_k4_loader import (  # noqa: E402
    BIN_SIZE_S,
    RT_EXPECTED_SESSION_COUNT,
    _as_go_cue_matrix,
    build_rt_reach_segments,
    find_rt_sessions,
)


CALIBRATION_TRIALS = 24
SEED = 42
EPS = 1.0e-12
# This is deliberately a mechanism gate, not an accuracy threshold.  It says
# the *reach-average* velocity assigned to a neural reach cannot still be
# mostly collinear with its original reach-average velocity.
MAX_STRONG_NULL_GROUP_DIRECTION_COSINE = 0.50
LOCAL_SWAP_PASSES = 12
LOCAL_SWAP_TRIALS_PER_BLOCK = 12
SCHEMA = "rt_afc4_ls_null_strength_support_audit_v1"
STATUS = "PASS_CPU_SUPPORT_ONLY_RT_AFC4_LS_NULL_STRENGTH_AUDIT"
DEFAULT_DATA_DIR = WORKSPACE / "sua_exploration/data/dandi_000688/sub-C"
DEFAULT_OUTPUT = (
    WORKSPACE
    / "sua_exploration/results/rt_afc4_ls_null_strength_audit_v1"
    / "RT_AFC4_LS_NULL_STRENGTH_SUPPORT_AUDIT_v1.json"
)


class StrongNullError(RuntimeError):
    """The predeclared cross-reach null conditions could not be met."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise StrongNullError(message)


def _sha256_bytes(*values: np.ndarray | bytes | str) -> str:
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, str):
            value = value.encode("utf-8")
        if isinstance(value, bytes):
            digest.update(value)
            continue
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def _strict_prefix_count(dataset: Any, *, left: int, right: int, cutoff_s: float) -> int:
    """Binary-search a sorted HDF5 vector without materialising post-cutoff data."""

    low, high = int(left), int(right)
    while low < high:
        middle = (low + high) // 2
        if float(dataset[middle]) < float(cutoff_s):
            low = middle + 1
        else:
            high = middle
    return low


def _support_velocity(
    velocity_data: Any, velocity_timestamps: Any, *, cutoff_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Bin only support velocity samples, retaining missing-bin semantics."""

    count = _strict_prefix_count(
        velocity_timestamps, left=0, right=len(velocity_timestamps), cutoff_s=cutoff_s
    )
    if count < 1:
        raise StrongNullError("M24 support contains no cursor-velocity sample")
    timestamps = np.asarray(velocity_timestamps[:count], dtype=np.float64)
    values = np.asarray(velocity_data[:count], dtype=np.float64)
    _need(values.ndim == 2 and values.shape == (count, 2), "support cursor velocity must be [time,2]")
    _need(np.isfinite(values).all() and np.isfinite(timestamps).all(), "support velocity contains non-finite values")
    _need(np.all(np.diff(timestamps) > 0.0), "support velocity timestamps must be strictly increasing")
    _need(np.all(timestamps < cutoff_s), "support reader crossed the trial-24 cutoff")
    n_bins = int(math.ceil(float(cutoff_s) / BIN_SIZE_S - 1.0e-10))
    bin_index = np.floor(timestamps / BIN_SIZE_S + 1.0e-10).astype(np.int64)
    _need(np.all((0 <= bin_index) & (bin_index < n_bins)), "support velocity bin crossed cutoff")
    sums = np.zeros((n_bins, 2), dtype=np.float64)
    np.add.at(sums, bin_index, values)
    counts = np.bincount(bin_index, minlength=n_bins).astype(np.int64)
    valid = counts > 0
    velocity = np.zeros((n_bins, 2), dtype=np.float64)
    velocity[valid] = sums[valid] / counts[valid, None]
    return velocity.astype(np.float32), valid


def load_rt_m24_support(nwb_path: str | Path) -> dict[str, Any]:
    """Load just the support prefix required by M24 AFC4, never query labels.

    Scalar/binary-search metadata accesses may inspect vector lengths and the
    cutoff boundary, but all velocity values, spike values, and trial cues
    materialised by this function are strictly before trial-24's stop time.
    """

    path = Path(nwb_path)
    with pynwb.NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        _need(nwb.trials is not None, f"RT NWB has no trials table: {path}")
        trials = nwb.trials
        required = {"start_time", "stop_time", "go_cue_time_array", "num_targets"}
        _need(required.issubset(trials.colnames), f"RT trial columns missing: {sorted(required.difference(trials.colnames))}")
        _need(len(trials) >= CALIBRATION_TRIALS, f"RT NWB has fewer than M24 trials: {path}")
        starts = np.asarray(trials["start_time"][:CALIBRATION_TRIALS], dtype=np.float64)
        stops = np.asarray(trials["stop_time"][:CALIBRATION_TRIALS], dtype=np.float64)
        num_targets = np.asarray(trials["num_targets"][:CALIBRATION_TRIALS])
        go_cues = _as_go_cue_matrix(
            trials["go_cue_time_array"][:CALIBRATION_TRIALS], n_trials=CALIBRATION_TRIALS
        )
        _need(np.isfinite(stops).all() and np.all(stops > starts), "invalid M24 support trial boundaries")
        cutoff_s = float(stops[-1])

        try:
            velocity_series = nwb.processing["behavior"].data_interfaces["Velocity"].time_series["cursor_vel"]
        except KeyError as error:
            raise StrongNullError(f"RT cursor velocity is missing: {path}") from error
        _need(velocity_series.timestamps is not None, "RT cursor velocity requires explicit timestamps")
        velocity, velocity_valid = _support_velocity(
            velocity_series.data, velocity_series.timestamps, cutoff_s=cutoff_s
        )
        n_bins = int(velocity.shape[0])

        _need(nwb.units is not None and "spike_times" in nwb.units.colnames, "RT NWB lacks sorted units")
        spike_column = nwb.units["spike_times"]
        spike_index = spike_column.data
        spike_data = spike_column.target.data
        n_units = int(len(spike_index))
        _need(n_units >= 2, "RT AFC4 requires at least two sorted units")
        neural = np.zeros((n_bins, n_units), dtype=np.float32)
        left = 0
        for channel in range(n_units):
            right = int(spike_index[channel])
            end = _strict_prefix_count(spike_data, left=left, right=right, cutoff_s=cutoff_s)
            spikes = np.asarray(spike_data[left:end], dtype=np.float64)
            _need(np.isfinite(spikes).all(), f"RT unit {channel} has non-finite support spikes")
            _need(np.all(spikes < cutoff_s), f"RT unit {channel} spike crossed trial-24 cutoff")
            spike_bins = np.floor(spikes / BIN_SIZE_S + 1.0e-10).astype(np.int64)
            valid_spikes = (spike_bins >= 0) & (spike_bins < n_bins)
            np.add.at(neural[:, channel], spike_bins[valid_spikes], 1.0)
            left = right

    trial_change, segment_ids, _eval_mask, segment_audit = build_rt_reach_segments(
        n_bins=n_bins,
        trial_start_times=starts,
        trial_stop_times=stops,
        go_cue_time_array=go_cues,
        num_targets=num_targets,
        velocity_bin_valid=velocity_valid,
    )
    stem = path.name.removesuffix("_behavior+ecephys.nwb")
    return {
        "session_name": stem.removeprefix("sub-C_"),
        "nwb_path": str(path),
        "support_trial_index_range": [0, CALIBRATION_TRIALS],
        "support_cutoff_s": cutoff_s,
        "neural": neural,
        "velocity": velocity,
        "trial_change": trial_change,
        "segment_ids": segment_ids,
        "segment_audit": segment_audit,
        "support_input_sha256": _sha256_bytes(neural, velocity, trial_change, segment_ids, starts, stops, go_cues),
    }


def collect_m24_blocks(raw: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce K4's same-reach active 100-ms block collection exactly."""

    neural = np.asarray(raw["neural"], dtype=np.float64)
    velocity = np.asarray(raw["velocity"], dtype=np.float64)
    trial_change = np.asarray(raw["trial_change"], dtype=bool)
    segment_ids = np.asarray(raw["segment_ids"], dtype=np.int64)
    starts = np.flatnonzero(trial_change)
    _need(starts.size == CALIBRATION_TRIALS, "support reader must expose exactly M24 chronological trial starts")
    ends = np.r_[starts[1:], neural.shape[0]]
    active = ~np.all(np.abs(velocity) < K4_ACTIVE_EPSILON, axis=1)
    rates: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    groups: list[int] = []
    for start, end in zip(starts, ends):
        for left in range(
            int(start), int(end) - K4_BLOCK_WIDTH_BINS - K4_BEHAVIOR_LEAD_BINS + 1, K4_BLOCK_WIDTH_BINS
        ):
            right = left + K4_BLOCK_WIDTH_BINS
            y_left = left + K4_BEHAVIOR_LEAD_BINS
            y_right = y_left + K4_BLOCK_WIDTH_BINS
            membership = segment_ids[left:y_right]
            if membership.shape[0] != K4_BLOCK_WIDTH_BINS + K4_BEHAVIOR_LEAD_BINS:
                raise StrongNullError("invalid M24 support block span")
            if membership[0] < 0 or not np.all(membership == membership[0]):
                continue
            if not (active[left:right].all() and active[y_left:y_right].all()):
                continue
            rates.append(neural[left:right].sum(axis=0) / (K4_BLOCK_WIDTH_BINS * K4_RAW_BIN_MS / 1000.0))
            labels.append(velocity[y_left:y_right].mean(axis=0))
            groups.append(int(membership[0]))
    rate = np.asarray(rates, dtype=np.float64)
    label = np.asarray(labels, dtype=np.float64)
    group = np.asarray(groups, dtype=np.int64)
    _need(rate.ndim == 2 and rate.shape[0] >= 3 and label.shape == (rate.shape[0], 2), "invalid active M24 block set")
    _need(np.all(group >= 0), "M24 blocks lack valid reach groups")
    return rate, label, group


def fit_descriptor(rates: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Fit the four AFC4 coordinates from already validated support blocks."""

    _need(rates.ndim == 2 and labels.shape == (rates.shape[0], 2), "descriptor input shape mismatch")
    design = np.column_stack([np.ones(labels.shape[0], dtype=np.float64), labels])
    rank = int(np.linalg.matrix_rank(design))
    _need(rank == 3, f"AFC4 design rank is {rank}, not 3")
    coefficients, _, rank_lstsq, _ = np.linalg.lstsq(design, rates, rcond=None)
    _need(int(rank_lstsq) == 3, "AFC4 least-squares fit became rank-deficient")
    weights = coefficients[1:].T
    return np.column_stack([weights[:, 0], weights[:, 1], np.linalg.norm(weights, axis=1), coefficients[0]]).astype(np.float32)


def _repair_group_collisions(permutation: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Turn an arbitrary block permutation into a no-same-reach permutation."""

    result = permutation.copy()
    count = result.size
    if np.max(np.bincount(np.unique(groups, return_inverse=True)[1])) * 2 > count:
        raise StrongNullError("cross-reach derangement is impossible: one reach owns more than half of active blocks")
    while True:
        bad = np.flatnonzero(groups[result] == groups)
        if bad.size == 0:
            return result
        changed = False
        for left_index in range(bad.size):
            left = int(bad[left_index])
            for right_index in range(left_index + 1, bad.size):
                right = int(bad[right_index])
                if groups[left] != groups[right]:
                    result[left], result[right] = result[right], result[left]
                    changed = True
                    break
            if changed:
                break
        if changed:
            continue
        # One residual collision (or collisions from a single group) can be
        # repaired by swapping with a currently valid destination.
        left = int(bad[0])
        for right in range(count):
            if right == left or groups[right] == groups[left]:
                continue
            if groups[result[right]] != groups[left]:
                result[left], result[right] = result[right], result[left]
                changed = True
                break
        if not changed:
            raise StrongNullError("cannot repair cross-reach derangement without a same-reach label")


def _unit_vectors(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    output = np.zeros_like(values, dtype=np.float64)
    valid = norms[:, 0] > EPS
    output[valid] = values[valid] / norms[valid]
    return output


def deterministic_cross_reach_derangement(
    groups: np.ndarray, velocity: np.ndarray, *, session_name: str, seed: int
) -> np.ndarray:
    """A stronger support-only label null: exact velocity multiset, no reach match.

    The result is an exact block-label permutation.  Every destination neural
    block receives a velocity from a different reach; a deterministic local
    2-opt phase minimises donor/recipient direction cosine while retaining the
    cross-reach constraint.  It is not a global fixed rotation or an IID bin
    shuffle: the unit of assignment is an already event-qualified 100-ms
    block, and all feasibility conditions are checked after optimisation.
    """

    group = np.asarray(groups, dtype=np.int64).reshape(-1)
    label = np.asarray(velocity, dtype=np.float64)
    _need(group.size >= 3 and label.shape == (group.size, 2), "cross-reach null needs >=3 [block,2] labels")
    _need(np.unique(group).size >= 3, "cross-reach null needs at least three accepted reaches")
    digest = hashlib.sha256(f"rt-afc4-ls-strong-v1:{seed}:{session_name}".encode()).digest()
    rng = np.random.RandomState(int.from_bytes(digest[:4], "little"))
    permutation = _repair_group_collisions(rng.permutation(group.size).astype(np.int64), group)
    unit = _unit_vectors(label)
    speed = np.linalg.norm(label, axis=1)
    # Bounded deterministic 2-opt: it cannot create a same-reach pair and
    # prefers anti-aligned direction with a deliberately small speed penalty.
    for _pass in range(LOCAL_SWAP_PASSES):
        improved = False
        for _ in range(LOCAL_SWAP_TRIALS_PER_BLOCK * group.size):
            left, right = rng.randint(0, group.size, size=2).tolist()
            if left == right:
                continue
            left_source, right_source = int(permutation[left]), int(permutation[right])
            if group[right_source] == group[left] or group[left_source] == group[right]:
                continue
            old = float(unit[left] @ unit[left_source] + unit[right] @ unit[right_source])
            new = float(unit[left] @ unit[right_source] + unit[right] @ unit[left_source])
            old += 0.02 * (abs(speed[left] - speed[left_source]) + abs(speed[right] - speed[right_source]))
            new += 0.02 * (abs(speed[left] - speed[right_source]) + abs(speed[right] - speed[left_source]))
            if new < old - 1.0e-12:
                permutation[left], permutation[right] = permutation[right], permutation[left]
                improved = True
        if not improved:
            break
    _need(np.array_equal(np.sort(permutation), np.arange(group.size)), "strong null is not an exact permutation")
    _need(np.all(permutation != np.arange(group.size)), "strong null left an active block unchanged")
    _need(np.all(group[permutation] != group), "strong null retained a same-reach label")
    diagnostics = permutation_diagnostics(group, label, permutation)
    direction = diagnostics["reach_direction_transfer"]
    _need(direction["defined_reaches"] >= 3, "strong null has too few defined reach directions")
    _need(direction["mean_cosine"] <= MAX_STRONG_NULL_GROUP_DIRECTION_COSINE and
          direction["median_cosine"] <= MAX_STRONG_NULL_GROUP_DIRECTION_COSINE,
          "strong null did not sufficiently break coarse reach-direction association")
    return permutation


def _safe_cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    norm = float(np.linalg.norm(x) * np.linalg.norm(y))
    if norm <= EPS:
        return None
    return float(np.clip(np.dot(x, y) / norm, -1.0, 1.0))


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    x, y = np.asarray(left, dtype=np.float64).reshape(-1), np.asarray(right, dtype=np.float64).reshape(-1)
    if x.size < 2 or x.shape != y.shape or np.std(x) <= EPS or np.std(y) <= EPS:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _numeric_summary(values: Iterable[float | None]) -> dict[str, Any]:
    usable = np.asarray([float(value) for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if usable.size == 0:
        return {"defined": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "defined": int(usable.size), "mean": float(np.mean(usable)), "median": float(np.median(usable)),
        "minimum": float(np.min(usable)), "maximum": float(np.max(usable)),
    }


def permutation_diagnostics(groups: np.ndarray, velocity: np.ndarray, permutation: np.ndarray) -> dict[str, Any]:
    """Quantify label-marginal preservation and coarse reach association."""

    group = np.asarray(groups, dtype=np.int64).reshape(-1)
    label = np.asarray(velocity, dtype=np.float64)
    order = np.asarray(permutation, dtype=np.int64).reshape(-1)
    _need(order.shape == group.shape and label.shape == (group.size, 2), "permutation diagnostic shape mismatch")
    _need(np.array_equal(np.sort(order), np.arange(order.size)), "permutation diagnostic requires a true permutation")
    assigned = label[order]
    reaches: list[dict[str, Any]] = []
    for reach in np.unique(group).tolist():
        indices = np.flatnonzero(group == reach)
        own = label[indices].mean(axis=0)
        donor = assigned[indices].mean(axis=0)
        cosine = _safe_cosine(own, donor)
        angle = None if cosine is None else float(math.atan2(own[0] * donor[1] - own[1] * donor[0], np.dot(own, donor)))
        reaches.append({
            "reach_id": int(reach), "active_blocks": int(indices.size), "same_reach_assigned_blocks": int(np.sum(group[order[indices]] == reach)),
            "correct_mean_velocity": own.tolist(), "assigned_mean_velocity": donor.tolist(),
            "direction_cosine": cosine, "signed_direction_angle_rad": angle,
        })
    reach_cosines = [row["direction_cosine"] for row in reaches]
    same = int(np.sum(group[order] == group))
    return {
        "active_blocks": int(group.size),
        "labels_changed_blocks": int(np.sum(order != np.arange(order.size))),
        "same_reach_assigned_blocks": same,
        "same_reach_assigned_fraction": float(same / group.size),
        "velocity_marginal": {
            "preserved_exactly_by_index_permutation": True,
            "permutation_is_bijective": True,
            "global_mean_delta": (assigned.mean(axis=0) - label.mean(axis=0)).tolist(),
            "global_speed_mean_delta": float(np.linalg.norm(assigned, axis=1).mean() - np.linalg.norm(label, axis=1).mean()),
            "tolerance": 0.0,
        },
        "block_length_statistics": {
            "unit": "event-qualified fixed 100-ms blocks",
            "total_blocks_unchanged": True,
            "per_reach_neural_block_counts_unchanged": True,
            "tolerance": 0.0,
        },
        "reach_direction_transfer": {
            "defined_reaches": int(sum(value is not None for value in reach_cosines)),
            "mean_cosine": _numeric_summary(reach_cosines)["mean"],
            "median_cosine": _numeric_summary(reach_cosines)["median"],
            "per_reach": reaches,
        },
    }


def descriptor_comparison(aligned: np.ndarray, null: np.ndarray) -> dict[str, Any]:
    """Per-channel signed W geometry and full four-coordinate comparison."""

    correct, shuffled = np.asarray(aligned, dtype=np.float64), np.asarray(null, dtype=np.float64)
    _need(correct.ndim == 2 and correct.shape == shuffled.shape and correct.shape[1] == 4, "descriptor comparison requires [N,4]")
    rows: list[dict[str, Any]] = []
    for channel in range(correct.shape[0]):
        w0, w1 = correct[channel, :2], shuffled[channel, :2]
        w_cosine = _safe_cosine(w0, w1)
        angle = None if w_cosine is None else float(math.atan2(w0[0] * w1[1] - w0[1] * w1[0], np.dot(w0, w1)))
        rows.append({
            "channel": int(channel), "w_cosine": w_cosine,
            "signed_w_angle_rad": angle, "signed_w_angle_deg": None if angle is None else float(np.degrees(angle)),
            "aligned_w_norm": float(correct[channel, 2]), "null_w_norm": float(shuffled[channel, 2]),
            "delta_w_norm": float(shuffled[channel, 2] - correct[channel, 2]),
            "aligned_b": float(correct[channel, 3]), "null_b": float(shuffled[channel, 3]),
            "delta_b": float(shuffled[channel, 3] - correct[channel, 3]),
            "descriptor_row_cosine": _safe_cosine(correct[channel], shuffled[channel]),
        })
    delta_norm = shuffled[:, 2] - correct[:, 2]
    delta_b = shuffled[:, 3] - correct[:, 3]
    return {
        "per_channel": rows,
        "w_geometry": {
            "flattened_w_cosine": _safe_cosine(correct[:, :2].reshape(-1), shuffled[:, :2].reshape(-1)),
            "flattened_w_pearson": _pearson(correct[:, :2], shuffled[:, :2]),
            "per_channel_w_cosine": _numeric_summary(row["w_cosine"] for row in rows),
            "per_channel_signed_w_angle_rad": _numeric_summary(row["signed_w_angle_rad"] for row in rows),
        },
        "w_norm_difference": {
            "signed_delta": _numeric_summary(delta_norm.tolist()),
            "mean_absolute_delta": float(np.mean(np.abs(delta_norm))),
            "rmse": float(np.sqrt(np.mean(delta_norm ** 2))),
            "pearson": _pearson(correct[:, 2], shuffled[:, 2]),
        },
        "baseline_b_difference": {
            "signed_delta": _numeric_summary(delta_b.tolist()),
            "mean_absolute_delta": float(np.mean(np.abs(delta_b))),
            "rmse": float(np.sqrt(np.mean(delta_b ** 2))),
            "pearson": _pearson(correct[:, 3], shuffled[:, 3]),
        },
        "descriptor_row_cosine": _numeric_summary(row["descriptor_row_cosine"] for row in rows),
    }


def audit_one_support(*, fold: int, path: Path) -> dict[str, Any]:
    raw = load_rt_m24_support(path)
    rates, velocity, groups = collect_m24_blocks(raw)
    session = str(raw["session_name"])
    aligned, aligned_audit = k4_from_raw_calibration(
        raw["neural"], raw["velocity"], raw["trial_change"], calibration_n_trials=CALIBRATION_TRIALS,
        segment_ids=raw["segment_ids"],
    )
    legacy, legacy_audit = k4_from_raw_calibration(
        raw["neural"], raw["velocity"], raw["trial_change"], calibration_n_trials=CALIBRATION_TRIALS,
        segment_ids=raw["segment_ids"], label_shuffle=True, label_shuffle_seed=SEED,
        label_shuffle_session_name=session,
    )
    legacy_permutation = deterministic_k4_block_label_permutation(groups, session_name=session, seed=SEED)
    _need(np.array_equal(fit_descriptor(rates, velocity), aligned), f"{session}: custom aligned support fit drift")
    _need(np.array_equal(fit_descriptor(rates, velocity[legacy_permutation]), legacy), f"{session}: custom LS support fit drift")
    _need(legacy_audit.extra is not None and legacy_audit.extra.get("label_changed_blocks") == rates.shape[0],
          f"{session}: current LS did not alter all active blocks")
    _need(legacy_audit.extra.get("label_permutation_sha256") == hashlib.sha256(legacy_permutation.tobytes()).hexdigest(),
          f"{session}: current LS permutation receipt drift")
    strong_status: dict[str, Any]
    try:
        strong_permutation = deterministic_cross_reach_derangement(groups, velocity, session_name=session, seed=SEED)
        strong = fit_descriptor(rates, velocity[strong_permutation])
        strong_status = {
            "status": "defined_passed_predeclared_strength_gate",
            "permutation_sha256": hashlib.sha256(strong_permutation.tobytes()).hexdigest(),
            "comparison_to_aligned": descriptor_comparison(aligned, strong),
            "permutation_diagnostics": permutation_diagnostics(groups, velocity, strong_permutation),
        }
    except StrongNullError as error:
        strong_status = {"status": "undefined_fail_closed", "reason": str(error)}
    return {
        "fold": int(fold), "session_name": session, "nwb_path": str(path),
        "support_trial_index_range": raw["support_trial_index_range"], "support_cutoff_s": raw["support_cutoff_s"],
        "support_input_sha256": raw["support_input_sha256"], "num_channels": int(aligned.shape[0]),
        "active_blocks": int(rates.shape[0]), "accepted_reaches": int(np.unique(groups).size),
        "aligned_k4_audit": aligned_audit.as_dict(), "legacy_ls_k4_audit": legacy_audit.as_dict(),
        "legacy_ls": {
            "policy": "within_segment_cyclic_block_labels__singleton_cross_segment_fallback",
            "permutation_sha256": hashlib.sha256(legacy_permutation.tobytes()).hexdigest(),
            "comparison_to_aligned": descriptor_comparison(aligned, legacy),
            "permutation_diagnostics": permutation_diagnostics(groups, velocity, legacy_permutation),
        },
        "strong_cross_reach_null": strong_status,
    }


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    legacy = [row["legacy_ls"] for row in rows]
    strong_rows = [row["strong_cross_reach_null"] for row in rows if row["strong_cross_reach_null"]["status"].startswith("defined")]
    def values(items: Sequence[dict[str, Any]], path: Sequence[str]) -> list[float | None]:
        result: list[float | None] = []
        for item in items:
            current: Any = item
            for key in path:
                if not isinstance(current, dict):
                    current = None
                    break
                current = current.get(key)
            result.append(current if isinstance(current, (float, int)) else None)
        return result
    summary_paths = {
        "flattened_w_cosine": ("comparison_to_aligned", "w_geometry", "flattened_w_cosine"),
        "flattened_w_pearson": ("comparison_to_aligned", "w_geometry", "flattened_w_pearson"),
        "median_per_channel_w_cosine": ("comparison_to_aligned", "w_geometry", "per_channel_w_cosine", "median"),
        "mean_abs_w_norm_delta": ("comparison_to_aligned", "w_norm_difference", "mean_absolute_delta"),
        "mean_abs_b_delta": ("comparison_to_aligned", "baseline_b_difference", "mean_absolute_delta"),
        "median_descriptor_row_cosine": ("comparison_to_aligned", "descriptor_row_cosine", "median"),
        "same_reach_label_fraction": ("permutation_diagnostics", "same_reach_assigned_fraction"),
        "mean_reach_direction_cosine": ("permutation_diagnostics", "reach_direction_transfer", "mean_cosine"),
        "median_reach_direction_cosine": ("permutation_diagnostics", "reach_direction_transfer", "median_cosine"),
    }
    return {
        "legacy_within_reach_ls": {key: _numeric_summary(values(legacy, path)) for key, path in summary_paths.items()},
        "strong_cross_reach_null": {
            "defined_sessions": int(len(strong_rows)), "undefined_sessions": int(len(rows) - len(strong_rows)),
            **{key: _numeric_summary(values(strong_rows, path)) for key, path in summary_paths.items()},
        },
    }


def _write_immutable(path: Path, body: dict[str, Any]) -> str:
    _need(not path.exists(), f"refusing to overwrite audit receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(body, indent=2, sort_keys=True, default=_json_default) + "\n")
    path.chmod(0o444)
    _need(stat.S_IMODE(path.stat().st_mode) == 0o444, "audit receipt chmod to 0444 failed")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-support-audit", action="store_true", help="explicitly open only M24 support prefixes")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _need(args.execute_support_audit, "refusing to open NWB without --execute-support-audit")
    _need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "audit is CPU-only; set CUDA_VISIBLE_DEVICES='' explicitly")
    paths = find_rt_sessions(args.data_dir)
    _need(len(paths) == RT_EXPECTED_SESSION_COUNT, f"expected {RT_EXPECTED_SESSION_COUNT} RT sessions, found {len(paths)}")
    rows = [audit_one_support(fold=fold, path=path) for fold, path in enumerate(paths)]
    _need([row["fold"] for row in rows] == list(range(RT_EXPECTED_SESSION_COUNT)), "RT support fold enumeration drift")
    strong_defined = [row for row in rows if row["strong_cross_reach_null"]["status"].startswith("defined")]
    receipt = {
        "schema": SCHEMA, "status": STATUS,
        "execution_contract": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "decoder_constructed": False,
            "optimizer_constructed": False, "datamodule_constructed": False, "formal_target_query_opened": False,
            "velocity_values_materialized": "strictly before trial-24 stop time only",
            "spike_times_materialized": "strictly before trial-24 stop time only",
            "trial_metadata_materialized": "trial indices [0,24) only",
        },
        "preexisting_artifact_assessment": {
            "artifact": str(PROJECT / "outputs/rt_k4_preflight/rt_k4_m24_fold0_seed42_cpu_receipt.json"),
            "sufficient_for_null_strength_question": False,
            "reason": "It records all blocks changed and max descriptor delta, but not W angle/cosine, flattened geometry, W-norm/b changes, or reach-direction preservation.",
        },
        "current_ls_definition": "within-reach deterministic nonzero cyclic block-label rotation; singleton reaches use cross-reach fallback",
        "strong_null_definition": {
            "name": "deterministic_cross_reach_block_derangement",
            "unit": "event-qualified 100-ms block", "not": ["global fixed rotation", "IID bin shuffle"],
            "preserves_exactly": ["global velocity-label multiset", "active block count", "neural block membership", "per-reach neural block counts"],
            "requires": ["all active block labels changed", "no label from its own reach", "bijection", "mean and median reach-direction cosine <= 0.50"],
            "does_not_preserve": "within-reach velocity temporal autocorrelation; this is deliberately sacrificed to test cross-reach label association.",
        },
        "interpretation": {
            "weak_null": "A legacy-LS descriptor change with high within-reach direction preservation only says the cyclic null is weak; it cannot support a claim that labels are useless.",
            "label_uninformative": "Only a predeclared, comparably trained decoder result in which correctly paired AFC4 fails to beat a verified strong association null could support that narrower conclusion. This receipt contains no decoder score.",
        },
        "future_gpu_routing_after_current_control_queues_terminal": {
            "priority_1_component_attribution": (
                "Run the existing matched afc4_mb4=[0,0,||W||,b] arm across the same 15-fold protocol. "
                "Full minus MB4 is the direct signed-direction contribution; it should precede any claim from a weak LS null."
            ),
            "priority_2_association_control": (
                "If every support session passes this receipt's strength gate, attach the isolated cross-reach "
                "derangement only after a new matched-arm receipt and rerun Full versus strong-LS."
            ),
            "current_implementation_inventory": {
                "mb4_config": "configs/experiment/rt_joint_afc4_mb4_m24_loso.yaml exists",
                "mb4_datamodule_and_evaluator": "afc4_mb4 is already accepted and masks normalized [wx,wy] only",
                "runner_gap": "scripts/run_rt_clean_nested_loso.py omits afc4_mb4 from both its arm validation and CLI choices",
                "remote_supervisor_gap": "scripts/run_rt_clean_nested_loso_remote_missing.py intentionally accepts only afc4_rs/afc4_ls",
            },
            "minimal_safe_integration_after_queues": [
                "Review the terminal imported R-RS/R-LS matrix; do not alter its active import graph.",
                "Add afc4_mb4 explicitly to the clean runner's accepted arms and parser choices; add a contract test for exact command binding.",
                "Create a separate MB4 supervisor/output root and receipt schema; never resume or reuse R-RS/R-LS cell directories.",
                "For strong-LS, move the audited standalone derangement into a new, versioned feature function only after its support receipt is reviewed; add parity and fail-closed tests before exposing a new arm.",
                "Predeclare Full−MB4 and Full−strong-LS paired statistics, folds, seed, source normalizer, checkpoint rule, and one-shot outer evaluation before GPU launch.",
            ],
        },
        "strong_null_gpu_readiness": {
            "all_sessions_passed_support_strength_gate": len(strong_defined) == len(rows),
            "defined_sessions": len(strong_defined), "total_sessions": len(rows),
            "recommendation": (
                "CONSIDER_STRONG_NULL_GPU_CONTROL_ONLY_AFTER_MATCHED_TRAINING_RECEIPT"
                if len(strong_defined) == len(rows) else "DO_NOT_OPEN_GPU_STRONG_NULL__SUPPORT_GATE_INCOMPLETE"
            ),
        },
        "fold_rows": rows,
        "aggregate": _aggregate(rows),
    }
    digest = _write_immutable(args.output, receipt)
    print(f"WROTE_RT_AFC4_LS_NULL_STRENGTH_RECEIPT={args.output}")
    print(f"SHA256={digest}")
    print(f"STRONG_NULL_DEFINED={len(strong_defined)}/{len(rows)}")


if __name__ == "__main__":
    main()
