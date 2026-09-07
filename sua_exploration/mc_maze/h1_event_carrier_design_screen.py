"""Source-only H1 event-carrier design screen.

This module does not modify the frozen H-SE5 implementation.  It compares
parameter-matched event-level label maps before any new GPU proposal.  Every
deployable candidate still fits only ``q + 1`` coefficients per channel in a
target session and uses no target-session neural-network optimization.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1


SCHEMA = "h1_event_carrier_cpu_design_screen_v1"
PROTOCOL = "h1_event_carrier_global_tag_context_source_screen_20260811_v1"
TARGET_RIDGE_LAMBDA = 3.0
SOURCE_RIDGE_LAMBDA = 1.0
FEATURE_SCALE_FLOOR = 1.0e-8
LATENT_SCALE_FLOOR = 1.0e-8
TAG_NAMES = v1.MOVEMENT_TAGS
TAG_INDEX = {tag: index for index, tag in enumerate(TAG_NAMES)}


@dataclass(frozen=True)
class ContextEvent:
    base: v1.MovementEvent
    start_state: np.ndarray
    midpoint_state: np.ndarray

    def __post_init__(self) -> None:
        v1._need(self.start_state.shape == (7,) and np.isfinite(self.start_state).all(), "invalid event start state")
        v1._need(self.midpoint_state.shape == (7,) and np.isfinite(self.midpoint_state).all(), "invalid event midpoint")


@dataclass(frozen=True)
class ContextSession:
    base: v1.EventSession
    events: tuple[ContextEvent, ...]


@dataclass(frozen=True)
class Candidate:
    name: str
    basis_mode: str
    feature_family: str
    rank: int
    duration_weighted_target: bool = False
    diagnostic_only: bool = False

    @property
    def tag_dependent(self) -> bool:
        return self.feature_family in {"context", "tag_delta", "tag_context"}

    @property
    def carrier_dim(self) -> int:
        return self.rank + 1


CANDIDATES: tuple[Candidate, ...] = (
    Candidate("pca_delta_q4", "pca", "delta", 4),
    Candidate("ser_delta_q4", "source_encoding", "delta", 4),
    Candidate("ser_context_q4", "source_encoding", "context", 4),
    Candidate("ser_tag_delta_q4", "source_encoding", "tag_delta", 4),
    Candidate("ser_tag_context_q4", "source_encoding", "tag_context", 4),
    Candidate("ser_tag_context_q4_wls", "source_encoding", "tag_context", 4, duration_weighted_target=True),
    Candidate("pca_delta_q7_diagnostic", "pca", "delta", 7, diagnostic_only=True),
    Candidate("ser_tag_context_q7_diagnostic", "source_encoding", "tag_context", 7, diagnostic_only=True),
)


@dataclass(frozen=True)
class LatentMap:
    candidate: Candidate
    outer_date: str
    source_sessions: tuple[str, ...]
    raw_dim: int
    active_mask: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    projection: np.ndarray
    latent_scale: np.ndarray
    energy_ratio: np.ndarray
    source_event_count: int
    map_sha256: str

    def transform(self, events: Sequence[ContextEvent], *, tag_overrides: Sequence[str] | None = None) -> np.ndarray:
        raw = raw_features(events, self.candidate.feature_family, tag_overrides=tag_overrides)
        v1._need(raw.shape[1] == self.raw_dim, f"{self.candidate.name}: raw feature dimension drift")
        standardized = (raw[:, self.active_mask] - self.feature_mean[None, :]) / self.feature_scale[None, :]
        latent = standardized @ self.projection
        latent = latent / self.latent_scale[None, :]
        v1._need(latent.shape == (len(events), self.candidate.rank) and np.isfinite(latent).all(),
                 f"{self.candidate.name}: invalid latent values")
        return latent

    def manifest(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.name,
            "basis_mode": self.candidate.basis_mode,
            "feature_family": self.candidate.feature_family,
            "rank": self.candidate.rank,
            "outer_date": self.outer_date,
            "source_sessions": list(self.source_sessions),
            "source_event_count": self.source_event_count,
            "raw_dim": self.raw_dim,
            "active_dim": int(self.active_mask.sum()),
            "cumulative_energy_at_rank": float(self.energy_ratio[: self.candidate.rank].sum()),
            "array_sha256": {
                "active_mask": v1.array_sha256(self.active_mask),
                "feature_mean": v1.array_sha256(self.feature_mean),
                "feature_scale": v1.array_sha256(self.feature_scale),
                "projection": v1.array_sha256(self.projection),
                "latent_scale": v1.array_sha256(self.latent_scale),
            },
            "map_sha256": self.map_sha256,
        }


def load_context_session(session: v1.EventSession) -> ContextSession:
    """Attach start/midpoint state using the same native position endpoints."""

    with h5py.File(session.path, "r") as handle:
        group = handle["acquisition/OpenLoopKinematics"]
        times, positions, _step, _conversion, _offset = v1._converted_series(group, expected_dim=7)
    rows: list[ContextEvent] = []
    for event in session.events:
        start = v1.interpolate_position(times, positions, event.start_time)
        stop = v1.interpolate_position(times, positions, event.stop_time)
        v1._need(np.allclose(stop - start, event.displacement, rtol=0.0, atol=1.0e-12),
                 f"{session.session_name}: endpoint reconstruction drift")
        rows.append(ContextEvent(
            base=event,
            start_state=np.asarray(start, np.float64),
            midpoint_state=np.asarray((start + stop) / 2.0, np.float64),
        ))
    return ContextSession(base=session, events=tuple(rows))


def load_context_sessions(data_root: str | Path) -> dict[str, ContextSession]:
    paths = v1.index_heldin_calib(data_root)
    return {
        name: load_context_session(v1.load_event_session(paths[name]))
        for name in v1.H1_HELDIN_SESSIONS
    }


def _onehot(tags: Sequence[str]) -> np.ndarray:
    output = np.zeros((len(tags), len(TAG_NAMES)), dtype=np.float64)
    for row, tag in enumerate(tags):
        v1._need(tag in TAG_INDEX, f"unknown event tag {tag}")
        output[row, TAG_INDEX[tag]] = 1.0
    return output


def raw_features(
    events: Sequence[ContextEvent],
    family: str,
    *,
    tag_overrides: Sequence[str] | None = None,
) -> np.ndarray:
    v1._need(bool(events), "carrier feature set is empty")
    tags = tuple(tag_overrides) if tag_overrides is not None else tuple(event.base.tag for event in events)
    v1._need(len(tags) == len(events), "tag override length mismatch")
    delta = np.stack([event.base.displacement for event in events]).astype(np.float64)
    midpoint = np.stack([event.midpoint_state for event in events]).astype(np.float64)
    onehot = _onehot(tags)
    if family == "delta":
        output = delta
    elif family == "context":
        output = np.concatenate((delta, midpoint, onehot), axis=1)
    elif family == "tag_delta":
        interaction = (onehot[:, :, None] * delta[:, None, :]).reshape(len(events), -1)
        output = np.concatenate((onehot, interaction), axis=1)
    elif family == "tag_context":
        delta_interaction = (onehot[:, :, None] * delta[:, None, :]).reshape(len(events), -1)
        midpoint_interaction = (onehot[:, :, None] * midpoint[:, None, :]).reshape(len(events), -1)
        output = np.concatenate((onehot, delta_interaction, midpoint_interaction), axis=1)
    else:
        raise v1.SparseEventEndpointError(f"unknown feature family {family}")
    v1._need(np.isfinite(output).all(), f"{family}: nonfinite raw feature")
    return output


def _canonical_projection(projection: np.ndarray) -> np.ndarray:
    value = np.asarray(projection, np.float64).copy()
    for column in range(value.shape[1]):
        pivot = int(np.argmax(np.abs(value[:, column])))
        if value[pivot, column] < 0:
            value[:, column] *= -1.0
    return value


def _source_rows(
    sessions: Mapping[str, ContextSession], outer_date: str,
) -> tuple[tuple[str, ...], list[ContextEvent]]:
    names = tuple(name for name in v1.H1_HELDIN_SESSIONS if v1.session_date(name) != outer_date)
    events = [event for name in names for event in sessions[name].events]
    v1._need(bool(events), f"{outer_date}: empty source event pool")
    return names, events


def fit_latent_map(
    sessions: Mapping[str, ContextSession], *, outer_date: str, candidate: Candidate,
) -> LatentMap:
    names, pooled_events = _source_rows(sessions, outer_date)
    raw = raw_features(pooled_events, candidate.feature_family)
    raw_mean = raw.mean(axis=0)
    raw_scale = raw.std(axis=0)
    active = raw_scale > FEATURE_SCALE_FLOOR
    v1._need(int(active.sum()) >= candidate.rank, f"{candidate.name}: too few active source features")
    mean = raw_mean[active]
    scale = raw_scale[active]
    x = (raw[:, active] - mean[None, :]) / scale[None, :]

    if candidate.basis_mode == "pca":
        _u, singular, right = np.linalg.svd(x, full_matrices=False)
        v1._need(right.shape[0] >= candidate.rank, f"{candidate.name}: PCA rank deficient")
        projection = right[: candidate.rank].T
        energy = np.square(singular) / np.square(singular).sum()
    elif candidate.basis_mode == "source_encoding":
        coefficients: list[np.ndarray] = []
        offset = 0
        for name in names:
            count = len(sessions[name].events)
            xs = x[offset : offset + count]
            response = np.stack([event.base.log_rates for event in sessions[name].events]).astype(np.float64)
            response_mean = response.mean(axis=0, keepdims=True)
            response_scale = np.maximum(response.std(axis=0, keepdims=True), 1.0e-6)
            ys = (response - response_mean) / response_scale
            penalty = np.eye(xs.shape[1], dtype=np.float64) * (count * SOURCE_RIDGE_LAMBDA)
            coefficient = np.linalg.solve(xs.T @ xs + penalty, xs.T @ ys)
            coefficient /= max(float(np.linalg.norm(coefficient, ord="fro")), 1.0e-12)
            coefficients.append(coefficient)
            offset += count
        v1._need(offset == len(pooled_events), f"{candidate.name}: source session slicing drift")
        stacked = np.concatenate(coefficients, axis=1)
        left, singular, _right = np.linalg.svd(stacked, full_matrices=False)
        v1._need(left.shape[1] >= candidate.rank, f"{candidate.name}: source encoding rank deficient")
        projection = left[:, : candidate.rank]
        energy = np.square(singular) / np.square(singular).sum()
    else:
        raise v1.SparseEventEndpointError(f"unknown basis mode {candidate.basis_mode}")

    projection = _canonical_projection(projection)
    latent = x @ projection
    latent_scale = np.maximum(latent.std(axis=0), LATENT_SCALE_FLOOR)
    body = {
        "protocol": PROTOCOL,
        "candidate": candidate.name,
        "outer_date": outer_date,
        "source_sessions": list(names),
        "source_event_count": len(pooled_events),
        "active_mask": v1.array_sha256(active),
        "feature_mean": v1.array_sha256(mean),
        "feature_scale": v1.array_sha256(scale),
        "projection": v1.array_sha256(projection),
        "latent_scale": v1.array_sha256(latent_scale),
    }
    return LatentMap(
        candidate=candidate,
        outer_date=outer_date,
        source_sessions=names,
        raw_dim=raw.shape[1],
        active_mask=np.asarray(active, bool),
        feature_mean=np.asarray(mean, np.float64),
        feature_scale=np.asarray(scale, np.float64),
        projection=np.asarray(projection, np.float64),
        latent_scale=np.asarray(latent_scale, np.float64),
        energy_ratio=np.asarray(energy, np.float64),
        source_event_count=len(pooled_events),
        map_sha256=v1.canonical_sha256(body),
    )


def select_range(session: ContextSession, *, start: int, budget: int) -> tuple[ContextEvent, ...]:
    v1._need(budget in (3, 4), f"unsupported budget M={budget}")
    return tuple(event for event in session.events if start <= event.base.trial_index < start + budget)


def fit_target_carrier(
    latent: np.ndarray,
    response: np.ndarray,
    *,
    durations: np.ndarray,
    weighted: bool,
) -> np.ndarray:
    z = np.asarray(latent, np.float64)
    y = np.asarray(response, np.float64)
    v1._need(z.ndim == 2 and y.shape == (z.shape[0], v1.EXPECTED_NEURONS), "target carrier shape mismatch")
    design = np.column_stack((np.ones(z.shape[0]), z))
    v1._need(np.linalg.matrix_rank(design) == design.shape[1], "target event design rank deficient")
    if weighted:
        weights = np.asarray(durations, np.float64)
        weights = weights / weights.mean()
    else:
        weights = np.ones(z.shape[0], dtype=np.float64)
    gram = design.T @ (weights[:, None] * design)
    cross = design.T @ (weights[:, None] * y)
    penalty = np.diag([0.0] + [1.0] * z.shape[1]) * (weights.sum() * TARGET_RIDGE_LAMBDA)
    coefficient = np.linalg.solve(gram + penalty, cross)
    carrier = np.column_stack((coefficient[1:].T, coefficient[0]))
    v1._need(np.isfinite(carrier).all(), "target carrier is nonfinite")
    return carrier


def within_trial_tag_shuffle(
    events: Sequence[ContextEvent], *, session: str, budget: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rotate tag identities within each trial with zero tag fixed points."""

    order = np.arange(len(events), dtype=np.int64)
    groups: dict[int, list[int]] = {}
    for index, event in enumerate(events):
        groups.setdefault(event.base.trial_index, []).append(index)
    shifts: dict[str, int] = {}
    for trial, indices in sorted(groups.items()):
        v1._need(len(indices) >= 2, f"trial {trial}: fewer than two events for tag shuffle")
        source_tags = [events[index].base.tag for index in indices]
        key = f"h1-event-carrier-tag-shuffle:{session}:M{budget}:trial{trial}"
        start = 1 + int.from_bytes(__import__("hashlib").sha256(key.encode("utf-8")).digest()[:8], "little") % (len(indices) - 1)
        chosen = None
        for offset in range(len(indices) - 1):
            shift = 1 + (start - 1 + offset) % (len(indices) - 1)
            rotated = np.roll(np.asarray(indices, np.int64), shift)
            if all(events[int(right)].base.tag != source_tags[left] for left, right in enumerate(rotated)):
                chosen = (shift, rotated)
                break
        v1._need(chosen is not None, f"trial {trial}: no fixed-point-free tag rotation")
        shift, rotated = chosen
        order[np.asarray(indices, np.int64)] = rotated
        shifts[str(trial)] = int(shift)
    fixed = sum(events[index].base.tag == events[int(order[index])].base.tag for index in range(len(events)))
    v1._need(fixed == 0 and np.array_equal(np.sort(order), np.arange(len(events))), "tag shuffle is invalid")
    return order, {
        "namespace": "h1-event-carrier-tag-shuffle-v1",
        "session": session,
        "budget": budget,
        "trial_shifts": shifts,
        "order_sha256": v1.array_sha256(order),
        "fixed_points": fixed,
    }


def predict(carrier: np.ndarray, latent: np.ndarray) -> np.ndarray:
    values = np.asarray(carrier, np.float64)
    z = np.asarray(latent, np.float64)
    return z @ values[:, :-1].T + values[:, -1][None, :]


def evaluate_session(
    session: ContextSession, latent_map: LatentMap, *, budget: int,
) -> dict[str, Any]:
    support = select_range(session, start=0, budget=budget)
    later = tuple(event for event in session.events if event.base.trial_index >= budget)
    v1._need(len(support) >= latent_map.candidate.carrier_dim and len(later) >= 4,
             f"{session.base.session_name}: insufficient support/query events")
    z_support = latent_map.transform(support)
    z_later = latent_map.transform(later)
    y_support = np.stack([event.base.log_rates for event in support]).astype(np.float64)
    y_later = np.stack([event.base.log_rates for event in later]).astype(np.float64)
    durations = np.asarray([event.base.duration_seconds for event in support], np.float64)
    carrier = fit_target_carrier(
        z_support, y_support, durations=durations,
        weighted=latent_map.candidate.duration_weighted_target,
    )
    r_correct = v1.r2_by_channel(y_later, predict(carrier, z_later))

    order, label_manifest = v1.within_trial_label_shuffle(
        tuple(event.base for event in support), session=session.base.session_name, budget=budget,
    )
    label_carrier = fit_target_carrier(
        z_support[order], y_support, durations=durations,
        weighted=latent_map.candidate.duration_weighted_target,
    )
    r_label = v1.r2_by_channel(y_later, predict(label_carrier, z_later))
    if latent_map.candidate.duration_weighted_target:
        support_mean = np.average(y_support, axis=0, weights=durations)
    else:
        support_mean = y_support.mean(axis=0)
    r_intercept = v1.r2_by_channel(y_later, np.broadcast_to(support_mean, y_later.shape))

    r_tag = None
    tag_manifest = None
    if latent_map.candidate.tag_dependent:
        tag_order, tag_manifest = within_trial_tag_shuffle(
            support, session=session.base.session_name, budget=budget,
        )
        wrong_tags = tuple(support[int(index)].base.tag for index in tag_order)
        z_wrong_tag = latent_map.transform(support, tag_overrides=wrong_tags)
        tag_carrier = fit_target_carrier(
            z_wrong_tag, y_support, durations=durations,
            weighted=latent_map.candidate.duration_weighted_target,
        )
        r_tag = v1.r2_by_channel(y_later, predict(tag_carrier, z_later))

    defined = np.isfinite(r_correct) & np.isfinite(r_label) & np.isfinite(r_intercept)
    if r_tag is not None:
        defined &= np.isfinite(r_tag)
    v1._need(np.any(defined), f"{session.base.session_name}: no defined forward channels")
    result = {
        "session": session.base.session_name,
        "budget": budget,
        "support_events": len(support),
        "later_events": len(later),
        "design_rank": int(np.linalg.matrix_rank(np.column_stack((np.ones(len(support)), z_support)))),
        "carrier_dim": latent_map.candidate.carrier_dim,
        "defined_channels": int(defined.sum()),
        "median_r2_correct": float(np.median(r_correct[defined])),
        "median_delta_label_shuffle": float(np.median((r_correct - r_label)[defined])),
        "median_delta_intercept": float(np.median((r_correct - r_intercept)[defined])),
        "correct_carrier_sha256": v1.array_sha256(carrier),
        "label_shuffle": label_manifest,
        "duration_weighted_target": latent_map.candidate.duration_weighted_target,
    }
    if r_tag is not None:
        result["median_delta_tag_shuffle"] = float(np.median((r_correct - r_tag)[defined]))
        result["tag_shuffle"] = tag_manifest
    return result


def _summary(rows: Mapping[str, Mapping[str, Any]], field: str) -> dict[str, Any]:
    return v1.paired_summary([(name, row.get(field)) for name, row in rows.items()])


def aggregate_candidate(rows: Mapping[str, Mapping[str, Any]], baseline: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    augmented = {
        name: {
            **row,
            "delta_vs_pca_delta_q4": float(row["median_r2_correct"] - baseline[name]["median_r2_correct"]),
        }
        for name, row in rows.items()
    }
    output = {
        "correct_r2": _summary(augmented, "median_r2_correct"),
        "correct_minus_pca_delta_q4": _summary(augmented, "delta_vs_pca_delta_q4"),
        "correct_minus_label_shuffle": _summary(augmented, "median_delta_label_shuffle"),
        "correct_minus_intercept": _summary(augmented, "median_delta_intercept"),
        "support_event_count": {
            "minimum": min(int(row["support_events"]) for row in rows.values()),
            "median": int(np.median([row["support_events"] for row in rows.values()])),
            "maximum": max(int(row["support_events"]) for row in rows.values()),
        },
    }
    if all("median_delta_tag_shuffle" in row for row in rows.values()):
        output["correct_minus_tag_shuffle"] = _summary(augmented, "median_delta_tag_shuffle")
    return output


def _positive_control(summary: Mapping[str, Any], *, minimum_positive: int = 10) -> bool:
    return bool(
        summary["defined_sessions"] == 13
        and summary["mean"] > 0
        and summary["median"] > 0
        and summary["positive"] >= minimum_positive
        and summary["leave_largest_absolute_out_mean"] > 0
    )


def budget_gate(candidate: Candidate, aggregate: Mapping[str, Any]) -> dict[str, Any]:
    vs = aggregate["correct_minus_pca_delta_q4"]
    material = bool(
        _positive_control(vs)
        and vs["mean"] >= 0.02
        and vs["median"] >= 0.01
    )
    label = _positive_control(aggregate["correct_minus_label_shuffle"])
    intercept = _positive_control(aggregate["correct_minus_intercept"])
    tag = True
    if candidate.tag_dependent:
        tag = _positive_control(aggregate["correct_minus_tag_shuffle"])
    passed = bool(not candidate.diagnostic_only and candidate.rank == 4 and material and label and intercept and tag)
    return {
        "passed": passed,
        "material_gain_vs_pca_delta_q4": material,
        "correct_minus_label_shuffle": label,
        "correct_minus_intercept": intercept,
        "correct_minus_tag_shuffle": tag if candidate.tag_dependent else "not_applicable",
        "thresholds": {
            "mean_delta_vs_baseline": 0.02,
            "median_delta_vs_baseline": 0.01,
            "minimum_positive_sessions": 10,
            "leave_largest_absolute_out_mean_positive": True,
        },
    }


def run_screen(sessions: Mapping[str, ContextSession]) -> dict[str, Any]:
    maps: dict[str, dict[str, LatentMap]] = {candidate.name: {} for candidate in CANDIDATES}
    for candidate in CANDIDATES:
        for outer_date in v1.H1_DATES:
            maps[candidate.name][outer_date] = fit_latent_map(sessions, outer_date=outer_date, candidate=candidate)

    budgets: dict[str, Any] = {}
    for budget in (3, 4):
        rows_by_candidate: dict[str, dict[str, Any]] = {candidate.name: {} for candidate in CANDIDATES}
        for candidate in CANDIDATES:
            for name in v1.H1_HELDIN_SESSIONS:
                mapping = maps[candidate.name][v1.session_date(name)]
                rows_by_candidate[candidate.name][name] = evaluate_session(sessions[name], mapping, budget=budget)
        baseline = rows_by_candidate["pca_delta_q4"]
        candidate_bodies: dict[str, Any] = {}
        for candidate in CANDIDATES:
            aggregate = aggregate_candidate(rows_by_candidate[candidate.name], baseline)
            candidate_bodies[candidate.name] = {
                "spec": {
                    "basis_mode": candidate.basis_mode,
                    "feature_family": candidate.feature_family,
                    "rank": candidate.rank,
                    "carrier_dim": candidate.carrier_dim,
                    "duration_weighted_target": candidate.duration_weighted_target,
                    "diagnostic_only": candidate.diagnostic_only,
                },
                "sessions": rows_by_candidate[candidate.name],
                "aggregate": aggregate,
                "gate": budget_gate(candidate, aggregate),
            }
        budgets[f"M{budget}"] = {"budget_trials": budget, "candidates": candidate_bodies}

    passing: list[str] = []
    for candidate in CANDIDATES:
        if candidate.name == "pca_delta_q4" or candidate.diagnostic_only:
            continue
        if budgets["M3"]["candidates"][candidate.name]["gate"]["passed"] and budgets["M4"]["candidates"][candidate.name]["gate"]["passed"]:
            passing.append(candidate.name)
    selected = None
    if passing:
        def rank_key(name: str) -> tuple[float, float, str]:
            m3 = budgets["M3"]["candidates"][name]["aggregate"]["correct_minus_pca_delta_q4"]
            m4 = budgets["M4"]["candidates"][name]["aggregate"]["correct_minus_pca_delta_q4"]
            return (min(float(m3["median"]), float(m4["median"])), min(float(m3["mean"]), float(m4["mean"])), name)
        selected = max(passing, key=rank_key)

    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "frozen_constants": {
            "target_ridge_lambda": TARGET_RIDGE_LAMBDA,
            "source_ridge_lambda": SOURCE_RIDGE_LAMBDA,
            "support_budgets": [3, 4],
            "tags": list(TAG_NAMES),
            "candidates_in_fixed_order": [candidate.name for candidate in CANDIDATES],
            "selection_rule": "pass both M3 and M4 gates; maximize the minimum budget median delta, then mean delta",
        },
        "basis_by_candidate_and_outer_date": {
            candidate.name: {date: maps[candidate.name][date].manifest() for date in v1.H1_DATES}
            for candidate in CANDIDATES
        },
        "budgets": budgets,
        "passing_candidates": passing,
        "selected_candidate": selected,
        "status": "PASS_CPU_CARRIER_CANDIDATE_SELECTED" if selected else "STOP_CPU_NO_MATERIAL_CARRIER_CANDIDATE",
        "gpu_authorized_by_this_screen": False,
        "interpretation": {
            "development_design_screen": True,
            "current_hse5_gpu_baseline_unchanged": True,
            "q7_arms_are_diagnostic_only": True,
            "source_encoding_basis_treats_each_source_session_channel_set_as_separate_columns": True,
            "no_cross_session_channel_correspondence_required_for_basis": True,
        },
    }
