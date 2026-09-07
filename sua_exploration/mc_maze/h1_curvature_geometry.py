"""CPU-only within-event position trajectory geometry diagnostic for H1.

Measures how much native 7-DoF position trajectories deviate from a
straight-line constant-speed reference between event endpoints.  This module
does not construct carriers, regressions, or decoders.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1


SCHEMA = "h1_curvature_geometry_diagnostic_v1"
PROTOCOL = "h1_curvature_geometry_diagnostic_20260812_v1"
SAMPLE_COUNT = 9
CHORD_FLOOR = 1.0e-9
DEGENERATE_DIM_FLOOR = 1.0e-9
PERCENTILE_METHOD = "linear"
EXCEEDANCE_THRESHOLDS: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50)
DOF_NAMES: tuple[str, ...] = ("tx", "ty", "tz", "rx", "g1", "g2", "g3")
SUPPORT_TRIAL_CUTOFF = 4


@dataclass(frozen=True)
class EventGeometry:
    session_name: str
    tag: str
    trial_index: int
    chord_norm: float
    max_deviation: float
    mean_deviation: float
    max_deviation_ratio: float | None
    mean_deviation_ratio: float | None
    arc_length: float
    arc_chord_ratio: float | None
    speed_cv: float
    degenerate_chord: bool
    dof_curvature_ratio: np.ndarray  # [7]
    dof_degenerate: np.ndarray  # [7] bool


def _percentile_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "median": None, "p90": None, "max": None}
    return {
        "count": int(array.size),
        "median": float(np.percentile(array, 50, method=PERCENTILE_METHOD)),
        "p90": float(np.percentile(array, 90, method=PERCENTILE_METHOD)),
        "max": float(np.max(array)),
    }


def _median_or_none(values: Sequence[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None
    return float(np.median(array))


def _exceedance_fractions(values: Sequence[float], thresholds: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {f"{threshold:g}": 0.0 for threshold in thresholds}
    return {f"{threshold:g}": float(np.mean(array > threshold)) for threshold in thresholds}


def sample_event_times(start_time: float, stop_time: float, sample_count: int = SAMPLE_COUNT) -> np.ndarray:
    v1._need(sample_count >= 2, "sample_count must be at least 2")
    v1._need(math.isfinite(start_time) and math.isfinite(stop_time) and stop_time > start_time,
             "invalid event interval for sampling")
    fractions = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
    return start_time + fractions * (stop_time - start_time)


def segment_lengths(positions: np.ndarray) -> np.ndarray:
    samples = np.asarray(positions, dtype=np.float64)
    v1._need(samples.ndim == 2 and samples.shape[1] == v1.POSITION_DIM,
             f"positions must be [{SAMPLE_COUNT},{v1.POSITION_DIM}]")
    v1._need(samples.shape[0] >= 2, "positions must contain at least two samples")
    deltas = np.diff(samples, axis=0)
    return np.linalg.norm(deltas, axis=1)


def speed_cv_from_positions(positions: np.ndarray) -> float:
    lengths = segment_lengths(positions)
    mean = float(np.mean(lengths))
    if mean <= 0.0:
        return 0.0
    return float(np.std(lengths, ddof=0) / mean)


def measure_trajectory(positions: np.ndarray) -> dict[str, Any]:
    """Compute geometry metrics from ``S`` evenly spaced position samples."""

    samples = np.asarray(positions, dtype=np.float64)
    v1._need(samples.shape == (SAMPLE_COUNT, v1.POSITION_DIM), f"expected shape ({SAMPLE_COUNT}, 7)")
    v1._need(np.isfinite(samples).all(), "position samples must be finite")

    p0 = samples[0]
    p8 = samples[-1]
    delta = p8 - p0
    chord_norm = float(np.linalg.norm(delta))
    fractions = np.linspace(0.0, 1.0, SAMPLE_COUNT, dtype=np.float64)
    reference = p0[None, :] + fractions[:, None] * delta[None, :]

    deviations = np.linalg.norm(samples[1:-1] - reference[1:-1], axis=1)
    max_deviation = float(np.max(deviations)) if deviations.size else 0.0
    mean_deviation = float(np.mean(deviations)) if deviations.size else 0.0

    lengths = segment_lengths(samples)
    arc_length = float(np.sum(lengths))

    degenerate_chord = chord_norm <= CHORD_FLOOR
    if degenerate_chord:
        max_deviation_ratio = None
        mean_deviation_ratio = None
        arc_chord_ratio = None
    else:
        max_deviation_ratio = max_deviation / chord_norm
        mean_deviation_ratio = mean_deviation / chord_norm
        arc_chord_ratio = arc_length / chord_norm

    dof_delta = np.abs(delta)
    dof_degenerate = dof_delta <= DEGENERATE_DIM_FLOOR
    dof_deviation = np.max(np.abs(samples[1:-1] - reference[1:-1]), axis=0)
    dof_denominator = np.maximum(dof_delta, DEGENERATE_DIM_FLOOR)
    dof_curvature_ratio = dof_deviation / dof_denominator

    return {
        "chord_norm": chord_norm,
        "max_deviation": max_deviation,
        "mean_deviation": mean_deviation,
        "max_deviation_ratio": max_deviation_ratio,
        "mean_deviation_ratio": mean_deviation_ratio,
        "arc_length": arc_length,
        "arc_chord_ratio": arc_chord_ratio,
        "speed_cv": speed_cv_from_positions(samples),
        "degenerate_chord": degenerate_chord,
        "dof_curvature_ratio": dof_curvature_ratio,
        "dof_degenerate": dof_degenerate,
    }


def sample_event_positions(
    times: np.ndarray,
    positions: np.ndarray,
    start_time: float,
    stop_time: float,
    *,
    sample_count: int = SAMPLE_COUNT,
) -> np.ndarray | None:
    sample_times = sample_event_times(start_time, stop_time, sample_count=sample_count)
    rows: list[np.ndarray] = []
    for time_s in sample_times.tolist():
        try:
            rows.append(v1.interpolate_position(times, positions, time_s))
        except v1.SparseEventEndpointError:
            return None
    return np.stack(rows, axis=0)


def measure_session(session: v1.EventSession) -> tuple[tuple[EventGeometry, ...], Counter[str]]:
    rejections: Counter[str] = Counter()
    measured: list[EventGeometry] = []

    with h5py.File(session.path, "r") as handle:
        group = handle["acquisition/OpenLoopKinematics"]
        times, position_values, _step, _conversion, _offset = v1._converted_series(group, expected_dim=v1.POSITION_DIM)

    for event in session.events:
        samples = sample_event_positions(times, position_values, event.start_time, event.stop_time)
        if samples is None:
            rejections["sample_read_failed"] += 1
            continue
        metrics = measure_trajectory(samples)
        measured.append(EventGeometry(
            session_name=session.session_name,
            tag=event.tag,
            trial_index=event.trial_index,
            chord_norm=float(metrics["chord_norm"]),
            max_deviation=float(metrics["max_deviation"]),
            mean_deviation=float(metrics["mean_deviation"]),
            max_deviation_ratio=metrics["max_deviation_ratio"],
            mean_deviation_ratio=metrics["mean_deviation_ratio"],
            arc_length=float(metrics["arc_length"]),
            arc_chord_ratio=metrics["arc_chord_ratio"],
            speed_cv=float(metrics["speed_cv"]),
            degenerate_chord=bool(metrics["degenerate_chord"]),
            dof_curvature_ratio=np.asarray(metrics["dof_curvature_ratio"], dtype=np.float64),
            dof_degenerate=np.asarray(metrics["dof_degenerate"], dtype=bool),
        ))

    return tuple(measured), rejections


def _ratio_values(events: Sequence[EventGeometry], field: str) -> list[float]:
    values: list[float] = []
    for event in events:
        if event.degenerate_chord:
            continue
        value = getattr(event, field)
        if value is not None:
            values.append(float(value))
    return values


def _aggregate_block(events: Sequence[EventGeometry]) -> dict[str, Any]:
    max_deviation_ratios = _ratio_values(events, "max_deviation_ratio")
    arc_chord_ratios = _ratio_values(events, "arc_chord_ratio")
    speed_cvs = [event.speed_cv for event in events]
    return {
        "event_count": len(events),
        "degenerate_chord_count": sum(1 for event in events if event.degenerate_chord),
        "max_deviation_ratio": _percentile_summary(max_deviation_ratios),
        "arc_chord_ratio": _percentile_summary(arc_chord_ratios),
        "speed_cv": _percentile_summary(speed_cvs),
        "max_deviation_ratio_exceedance": _exceedance_fractions(max_deviation_ratios, EXCEEDANCE_THRESHOLDS),
    }


def _per_tag_block(events: Sequence[EventGeometry]) -> dict[str, Any]:
    by_tag: dict[str, list[EventGeometry]] = {tag: [] for tag in v1.MOVEMENT_TAGS}
    for event in events:
        by_tag[event.tag].append(event)
    rows: dict[str, Any] = {}
    for tag in v1.MOVEMENT_TAGS:
        tagged = by_tag[tag]
        rows[tag] = {
            "event_count": len(tagged),
            "median_max_deviation_ratio": _median_or_none(_ratio_values(tagged, "max_deviation_ratio")),
            "median_arc_chord_ratio": _median_or_none(_ratio_values(tagged, "arc_chord_ratio")),
            "median_speed_cv": _median_or_none([item.speed_cv for item in tagged]),
        }
    return rows


def _per_dimension_block(events: Sequence[EventGeometry]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for index, name in enumerate(DOF_NAMES):
        ratios = [
            float(event.dof_curvature_ratio[index])
            for event in events
            if not event.degenerate_chord and not event.dof_degenerate[index]
        ]
        rows[name] = {
            "median_curvature_ratio": _median_or_none(ratios),
            "degenerate_delta_count": sum(1 for event in events if event.dof_degenerate[index]),
        }
    return rows


def _trial_split_block(events: Sequence[EventGeometry]) -> dict[str, Any]:
    support = tuple(event for event in events if event.trial_index < SUPPORT_TRIAL_CUTOFF)
    later = tuple(event for event in events if event.trial_index >= SUPPORT_TRIAL_CUTOFF)
    return {
        "support_events_trial_index_lt_4": _aggregate_block(support),
        "later_events_trial_index_ge_4": _aggregate_block(later),
    }


def _per_session_block(events: Sequence[EventGeometry]) -> dict[str, Any]:
    by_session: dict[str, list[EventGeometry]] = defaultdict(list)
    for event in events:
        by_session[event.session_name].append(event)
    return {
        session: _aggregate_block(session_events)
        for session in v1.H1_HELDIN_SESSIONS
        for session_events in [by_session.get(session, [])]
    }


def run_diagnostic(sessions: Mapping[str, v1.EventSession]) -> dict[str, Any]:
    v1._need(set(sessions) == set(v1.H1_HELDIN_SESSIONS) and len(sessions) == 13,
             "expected all 13 public H1 held-in calibration sessions")

    all_events: list[EventGeometry] = []
    rejection_counts: Counter[str] = Counter()
    for session_name in v1.H1_HELDIN_SESSIONS:
        measured, rejections = measure_session(sessions[session_name])
        all_events.extend(measured)
        rejection_counts.update(rejections)

    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "sample_count": SAMPLE_COUNT,
        "percentile_method": PERCENTILE_METHOD,
        "scope": {
            "cuda_used": False,
            "decoder_constructed": False,
            "carrier_fitted": False,
            "dense_velocity_series_opened": False,
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "overall": _aggregate_block(all_events),
        "by_tag": _per_tag_block(all_events),
        "by_dimension": _per_dimension_block(all_events),
        "by_trial_split": _trial_split_block(all_events),
        "by_session": _per_session_block(all_events),
    }
