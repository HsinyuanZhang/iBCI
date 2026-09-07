"""Deterministic direction-covered activity support selectors.

Only source calibration activity rates, trial durations, and the direction
labels already consumed by T4@50 enter this module.  All arithmetic is NumPy
float64 on CPU.  No model, query, target, prediction, loss, or gradient is an
accepted input.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from . import plan


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical_json(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class ActivityAuthority:
    session_id: str
    rates_hz: np.ndarray
    durations_s: np.ndarray
    directions: np.ndarray
    chronological_indices: np.ndarray

    def validated(self, *, expected_c: int = plan.CANDIDATE_POOL_N) -> "ActivityAuthority":
        rates = np.asarray(self.rates_hz)
        durations = np.asarray(self.durations_s)
        directions = np.asarray(self.directions)
        indices = np.asarray(self.chronological_indices)
        if rates.dtype != np.float64 or rates.ndim != 2:
            raise ValueError("rates_hz must be CPU float64 [C,N]")
        if rates.shape[0] != expected_c or rates.shape[1] < 1:
            raise ValueError(f"rates_hz must have shape [{expected_c},N>=1], got {rates.shape}")
        for name, array in (
            ("durations_s", durations),
            ("directions", directions),
            ("chronological_indices", indices),
        ):
            if array.shape != (expected_c,):
                raise ValueError(f"{name} must have shape [{expected_c}], got {array.shape}")
        if durations.dtype != np.float64:
            raise ValueError("durations_s must be CPU float64")
        if directions.dtype != np.int64 or indices.dtype != np.int64:
            raise ValueError("directions and chronological_indices must be int64")
        if not np.all(np.isfinite(rates)) or np.any(rates < 0.0):
            raise ValueError("rates_hz must be finite and nonnegative")
        if not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
            raise ValueError("durations_s must be finite and positive")
        if np.unique(indices).size != expected_c or not np.all(indices[:-1] < indices[1:]):
            raise ValueError("chronological_indices must be unique and strictly increasing")
        if np.any((directions < -1) | (directions >= plan.NUM_DIRECTIONS)):
            raise ValueError("directions must be -1 or a canonical direction in [0,7]")
        present = set(int(x) for x in directions if x >= 0)
        if present != set(range(plan.NUM_DIRECTIONS)):
            raise ValueError(f"all eight directions are required, got {sorted(present)}")
        return self

    def receipt(self) -> dict[str, object]:
        self.validated()
        return {
            "session_id": self.session_id,
            "candidate_pool_n": int(self.rates_hz.shape[0]),
            "unit_count": int(self.rates_hz.shape[1]),
            "rates_hz_sha256": _array_sha256(self.rates_hz),
            "durations_s_sha256": _array_sha256(self.durations_s),
            "directions_sha256": _array_sha256(self.directions),
            "chronological_indices_sha256": _array_sha256(self.chronological_indices),
            "direction_counts": [
                int(np.sum(self.directions == direction))
                for direction in range(plan.NUM_DIRECTIONS)
            ],
            "invalid_direction_count": int(np.sum(self.directions == -1)),
            "rates_finite": True,
            "rates_nonnegative": True,
        }


@dataclass(frozen=True)
class SelectionResult:
    law: str
    indices: tuple[int, ...]
    trace: tuple[dict[str, object], ...]

    def receipt(self) -> dict[str, object]:
        payload = {
            "law": self.law,
            "indices": list(self.indices),
            "trace": list(self.trace),
        }
        return {**payload, "payload_sha256": hashlib.sha256(_canonical_json(payload)).hexdigest()}


def robust_activity_z(authority: ActivityAuthority) -> tuple[np.ndarray, dict[str, object]]:
    authority.validated()
    features = np.log1p(authority.rates_hz, dtype=np.float64)
    log_duration = np.log(authority.durations_s, dtype=np.float64)
    duration_centered = log_duration - np.mean(log_duration, dtype=np.float64)
    duration_denominator = float(
        np.sum(duration_centered * duration_centered, dtype=np.float64)
    )
    if not math.isfinite(duration_denominator) or duration_denominator <= 0.0:
        raise ValueError("log-duration residualization denominator must be finite and positive")
    feature_centered = features - np.mean(features, axis=0, dtype=np.float64)[None, :]
    duration_slope = (
        duration_centered[:, None].T @ feature_centered / duration_denominator
    ).reshape(-1)
    residual = features - duration_centered[:, None] * duration_slope[None, :]
    center = np.median(residual, axis=0)
    mad = np.median(np.abs(residual - center[None, :]), axis=0)
    zero_mad = mad == 0.0
    scale = plan.ROBUST_MAD_FACTOR * mad + plan.ROBUST_SCALE_EPSILON
    z = (residual - center[None, :]) / scale[None, :]
    z[:, zero_mad] = 0.0
    if not np.all(np.isfinite(z)):
        raise ValueError("robust activity z contains nonfinite values")
    return z, {
        "feature": "duration_residualized_log1p_raw_spike_rate_hz",
        "normalizer_pool": "all_first_50_activity_trials_including_direction_minus_one",
        "duration_transform": "log_seconds_centered_by_arithmetic_mean",
        "duration_residualization": "per_unit_closed_form_ols_slope_with_intercept_removed_by_centering",
        "duration_denominator": duration_denominator,
        "duration_slope_sha256": _array_sha256(duration_slope),
        "residual_sha256": _array_sha256(residual),
        "mad_factor": plan.ROBUST_MAD_FACTOR,
        "epsilon": plan.ROBUST_SCALE_EPSILON,
        "zero_mad_coordinate_count": int(np.sum(zero_mad)),
        "center_sha256": _array_sha256(center),
        "mad_sha256": _array_sha256(mad),
        "z_sha256": _array_sha256(z),
    }


def pairwise_activity_distance(z: np.ndarray) -> np.ndarray:
    values = np.asarray(z)
    if values.dtype != np.float64 or values.ndim != 2:
        raise ValueError("z must be CPU float64 [C,N]")
    if not np.all(np.isfinite(values)):
        raise ValueError("z must be finite")
    distance = np.median(
        np.abs(values[:, None, :] - values[None, :, :]), axis=2
    ).astype(np.float64, copy=False)
    if not np.all(np.isfinite(distance)) or not np.array_equal(distance, distance.T):
        raise ValueError("pairwise activity distance must be finite and symmetric")
    np.fill_diagonal(distance, 0.0)
    return distance


def _extremum_index(
    candidates: Sequence[int], values: Sequence[float], *, maximize: bool
) -> int:
    if len(candidates) == 0 or len(candidates) != len(values):
        raise ValueError("extremum candidates and values must be nonempty and aligned")
    finite_values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(finite_values)):
        raise ValueError("extremum values must be finite")
    extremum = float(np.max(finite_values) if maximize else np.min(finite_values))
    tied = [
        int(index)
        for index, value in zip(candidates, finite_values, strict=True)
        if math.isclose(
            float(value), extremum,
            rel_tol=plan.TIE_REL_TOL,
            abs_tol=plan.TIE_ABS_TOL,
        )
    ]
    if not tied:
        raise AssertionError("global-extremum tie set unexpectedly empty")
    return min(tied)


def _valid_by_direction(authority: ActivityAuthority) -> dict[int, list[int]]:
    by_direction = {
        direction: [
            int(index)
            for index in np.flatnonzero(authority.directions == direction)
        ]
        for direction in range(plan.NUM_DIRECTIONS)
    }
    if any(not members for members in by_direction.values()):
        raise ValueError("direction-covered selection requires every direction")
    return by_direction


def _finish(result: SelectionResult, authority: ActivityAuthority) -> SelectionResult:
    indices = np.asarray(result.indices, dtype=np.int64)
    if indices.shape != (plan.ACTIVITY_SUPPORT_N,):
        raise AssertionError("support must contain exactly K indices")
    if np.unique(indices).size != plan.ACTIVITY_SUPPORT_N or not np.all(indices[:-1] < indices[1:]):
        raise AssertionError("support indices must be unique and chronologically sorted")
    if np.any(authority.directions[indices] < 0):
        raise AssertionError("direction-invalid trial entered a covered support")
    counts = np.bincount(authority.directions[indices], minlength=plan.NUM_DIRECTIONS)
    if np.any(counts < 1) or np.any(counts > plan.MAX_PER_DIRECTION):
        raise AssertionError(f"coverage/cap violation: {counts.tolist()}")
    return result


def select_cov_top10(authority: ActivityAuthority) -> SelectionResult:
    authority.validated()
    z, _ = robust_activity_z(authority)
    distance = pairwise_activity_distance(z)
    by_direction = _valid_by_direction(authority)
    selected: list[int] = []
    trace: list[dict[str, object]] = []
    for direction in range(plan.NUM_DIRECTIONS):
        candidates = by_direction[direction]
        sums = [float(np.sum(distance[index, candidates], dtype=np.float64)) for index in candidates]
        choice = _extremum_index(candidates, sums, maximize=False)
        selected.append(choice)
        trace.append({
            "stage": "direction_medoid",
            "direction": direction,
            "candidates": candidates,
            "distance_sums": sums,
            "choice": choice,
        })
    valid_candidates = [int(x) for x in np.flatnonzero(authority.directions >= 0)]
    while len(selected) < plan.ACTIVITY_SUPPORT_N:
        direction_counts = np.bincount(
            authority.directions[np.asarray(selected, dtype=np.int64)],
            minlength=plan.NUM_DIRECTIONS,
        )
        candidates = [
            index for index in valid_candidates
            if index not in selected
            and direction_counts[int(authority.directions[index])] < plan.MAX_PER_DIRECTION
        ]
        current_nearest = np.min(distance[:, selected], axis=1)
        gains = [
            float(np.sum(np.maximum(0.0, current_nearest - distance[:, index]), dtype=np.float64))
            for index in candidates
        ]
        choice = _extremum_index(candidates, gains, maximize=True)
        selected.append(choice)
        trace.append({
            "stage": "greedy_coverage_gain",
            "slot": len(selected) - 1,
            "candidates": candidates,
            "gains": gains,
            "choice": choice,
        })
    result = SelectionResult("COV-TOP10", tuple(sorted(selected)), tuple(trace))
    return _finish(result, authority)


def select_cov_anti10(authority: ActivityAuthority) -> SelectionResult:
    """Coverage-matched least-representative control.

    One anti-medoid (maximum within-direction distance sum) is selected per
    direction.  The final two slots maximize each candidate's total distance
    to all valid candidates, subject to the same two-per-direction cap.
    """
    authority.validated()
    z, _ = robust_activity_z(authority)
    distance = pairwise_activity_distance(z)
    by_direction = _valid_by_direction(authority)
    selected: list[int] = []
    trace: list[dict[str, object]] = []
    for direction in range(plan.NUM_DIRECTIONS):
        candidates = by_direction[direction]
        sums = [float(np.sum(distance[index, candidates], dtype=np.float64)) for index in candidates]
        choice = _extremum_index(candidates, sums, maximize=True)
        selected.append(choice)
        trace.append({
            "stage": "direction_antimedoid",
            "direction": direction,
            "candidates": candidates,
            "distance_sums": sums,
            "choice": choice,
        })
    valid_candidates = [int(x) for x in np.flatnonzero(authority.directions >= 0)]
    total_distance = np.sum(distance[:, valid_candidates], axis=1, dtype=np.float64)
    while len(selected) < plan.ACTIVITY_SUPPORT_N:
        counts = np.bincount(
            authority.directions[np.asarray(selected, dtype=np.int64)],
            minlength=plan.NUM_DIRECTIONS,
        )
        candidates = [
            index for index in valid_candidates
            if index not in selected and counts[int(authority.directions[index])] < plan.MAX_PER_DIRECTION
        ]
        values = [float(total_distance[index]) for index in candidates]
        choice = _extremum_index(candidates, values, maximize=True)
        selected.append(choice)
        trace.append({
            "stage": "global_outlier_fill",
            "slot": len(selected) - 1,
            "candidates": candidates,
            "total_distances": values,
            "choice": choice,
        })
    return _finish(
        SelectionResult("COV-ANTI10", tuple(sorted(selected)), tuple(trace)), authority
    )


def select_cov_fixed(authority: ActivityAuthority, *, late: bool) -> SelectionResult:
    authority.validated()
    by_direction = _valid_by_direction(authority)
    selected = [
        (members[-1] if late else members[0])
        for members in by_direction.values()
    ]
    valid = [int(x) for x in np.flatnonzero(authority.directions >= 0)]
    remaining = list(reversed(valid)) if late else valid
    trace: list[dict[str, object]] = [{
        "stage": "one_per_direction",
        "order": "latest" if late else "earliest",
        "choices": list(selected),
    }]
    for candidate in remaining:
        if len(selected) == plan.ACTIVITY_SUPPORT_N:
            break
        if candidate in selected:
            continue
        count = sum(
            int(authority.directions[index]) == int(authority.directions[candidate])
            for index in selected
        )
        if count < plan.MAX_PER_DIRECTION:
            selected.append(candidate)
            trace.append({"stage": "ordered_fill", "choice": candidate})
    law = "COV-FIX-L10" if late else "COV-FIX-E10"
    return _finish(SelectionResult(law, tuple(sorted(selected)), tuple(trace)), authority)


def stateless_seed(
    *, training_seed: int, epoch: int, session_id: str, sample_or_window_id: str
) -> tuple[int, str]:
    payload = {
        "route_domain": plan.RANDOM_DOMAIN,
        "training_seed": int(training_seed),
        "epoch": int(epoch),
        "session_id": str(session_id),
        "sample_or_window_id": str(sample_or_window_id),
    }
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return int(digest[:16], 16), digest


def select_cov_random10(
    authority: ActivityAuthority,
    *, training_seed: int,
    epoch: int,
    sample_or_window_id: str,
) -> SelectionResult:
    authority.validated()
    seed, domain_digest = stateless_seed(
        training_seed=training_seed,
        epoch=epoch,
        session_id=authority.session_id,
        sample_or_window_id=sample_or_window_id,
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    by_direction = _valid_by_direction(authority)
    selected = [
        int(rng.choice(np.asarray(by_direction[direction], dtype=np.int64)))
        for direction in range(plan.NUM_DIRECTIONS)
    ]
    valid = [
        int(x) for x in np.flatnonzero(authority.directions >= 0)
        if int(x) not in selected
    ]
    legal_pairs = [
        (left, right)
        for offset, left in enumerate(valid)
        for right in valid[offset + 1 :]
        if int(authority.directions[left]) != int(authority.directions[right])
    ]
    if not legal_pairs:
        raise ValueError("no legal two-direction fill pair exists")
    extra = list(legal_pairs[int(rng.integers(0, len(legal_pairs)))])
    selected.extend(extra)
    trace = ({
        "stage": "stateless_direction_covered_draw",
        "domain_digest": domain_digest,
        "one_per_direction": selected[:8],
        "extra": extra,
        "sorted_before_indexing": True,
    },)
    return _finish(
        SelectionResult("COV-RAND10", tuple(sorted(selected)), trace), authority
    )


def point_biserial_and_smd_full(
    durations_s: np.ndarray, selected_indices: Iterable[int]
) -> tuple[float, float]:
    durations = np.asarray(durations_s, dtype=np.float64)
    if durations.ndim != 1 or not np.all(np.isfinite(durations)):
        raise ValueError("durations must be a finite vector")
    membership = np.zeros(durations.shape[0], dtype=np.float64)
    membership[np.asarray(tuple(selected_indices), dtype=np.int64)] = 1.0
    if membership.min() == membership.max():
        raise ValueError("selected membership must contain both groups")
    corr = float(np.corrcoef(membership, durations)[0, 1])
    if not math.isfinite(corr):
        raise ValueError("point-biserial correlation is nonfinite")
    p = float(membership.mean())
    smd_full = corr / math.sqrt(p * (1.0 - p))
    return corr, smd_full


def jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def selector_stage0_diagnostics(authority: ActivityAuthority) -> dict[str, object]:
    authority.validated()
    original = select_cov_top10(authority)
    jackknife: list[dict[str, object]] = []
    for dropped_unit in range(authority.rates_hz.shape[1]):
        reduced = ActivityAuthority(
            session_id=authority.session_id,
            rates_hz=np.delete(authority.rates_hz, dropped_unit, axis=1),
            durations_s=authority.durations_s,
            directions=authority.directions,
            chronological_indices=authority.chronological_indices,
        )
        selected = select_cov_top10(reduced)
        jackknife.append({
            "dropped_unit": dropped_unit,
            "indices": list(selected.indices),
            "jaccard": jaccard(original.indices, selected.indices),
        })
    jaccards = np.asarray([row["jaccard"] for row in jackknife], dtype=np.float64)
    pb, smd_full = point_biserial_and_smd_full(
        authority.durations_s, original.indices
    )
    direction_counts = np.bincount(
        authority.directions[np.asarray(original.indices, dtype=np.int64)],
        minlength=plan.NUM_DIRECTIONS,
    )
    repeated = select_cov_top10(authority)
    deterministic = original.receipt()["payload_sha256"] == repeated.receipt()["payload_sha256"]
    median_jaccard = float(np.median(jaccards))
    passed = bool(
        median_jaccard >= plan.JACKKNIFE_MEDIAN_JACCARD_MIN
        and abs(pb) <= plan.POINT_BISERIAL_MAX_ABS
        and np.all(direction_counts >= 1)
        and deterministic
    )
    return {
        "session_id": authority.session_id,
        "top10": original.receipt(),
        "anti10": select_cov_anti10(authority).receipt(),
        "fixed_early10": select_cov_fixed(authority, late=False).receipt(),
        "fixed_late10": select_cov_fixed(authority, late=True).receipt(),
        "median_leave_one_unit_out_jaccard": median_jaccard,
        "jaccard_gate_min": plan.JACKKNIFE_MEDIAN_JACCARD_MIN,
        "leave_one_unit_out": jackknife,
        "point_biserial_duration": pb,
        "point_biserial_gate_max_abs": plan.POINT_BISERIAL_MAX_ABS,
        "smd_full_duration": smd_full,
        "direction_counts": direction_counts.astype(int).tolist(),
        "deterministic_trace": deterministic,
        "passed": passed,
    }
