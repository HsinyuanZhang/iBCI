"""Source-only H1 sparse-event label-semantics screen (V4).

The earlier event-carrier screens changed linear bases while keeping endpoint
displacement as the task covariate.  This module tests a distinct hypothesis:
event firing *rates* may align better with endpoint mean velocity than with raw
displacement.  Every candidate consumes only the already-required native event
start/stop positions and timestamps, emits a five-value carrier row, and uses
no target-session neural-network optimization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from sua_exploration.mc_maze import h1_event_carrier_design_screen as design
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1


SCHEMA = "h1_event_carrier_semantic_v4_source_screen_v1"
PROTOCOL = "h1_event_carrier_endpoint_velocity_semantics_20260812_v1"
RANK = 4
CARRIER_DIM = 5
TARGET_RIDGE_LAMBDA = 3.0
SOURCE_RIDGE_LAMBDA = 1.0
SCALE_FLOOR = 1.0e-8


@dataclass(frozen=True)
class SemanticCandidate:
    name: str
    basis_mode: str
    feature_family: str

    @property
    def carrier_dim(self) -> int:
        return CARRIER_DIM


CANDIDATES: tuple[SemanticCandidate, ...] = (
    SemanticCandidate("pca_delta_q4", "pca", "delta"),
    SemanticCandidate("pca_mean_velocity_q4", "pca", "mean_velocity"),
    SemanticCandidate("ser_mean_velocity_q4", "source_encoding", "mean_velocity"),
    SemanticCandidate("ser_direction_speed_q4", "source_encoding", "direction_speed"),
)


def _endpoint_rows(events: Sequence[design.ContextEvent]) -> tuple[np.ndarray, np.ndarray]:
    v1._need(bool(events), "semantic feature set is empty")
    delta = np.stack([event.base.displacement for event in events]).astype(np.float64)
    duration = np.asarray([event.base.duration_seconds for event in events], dtype=np.float64)
    v1._need(
        delta.shape == (len(events), v1.POSITION_DIM)
        and np.isfinite(delta).all()
        and np.isfinite(duration).all()
        and np.all(duration > 0),
        "invalid endpoint/duration rows",
    )
    return delta, duration


def _base_velocity(events: Sequence[design.ContextEvent]) -> np.ndarray:
    delta, duration = _endpoint_rows(events)
    return delta / duration[:, None]


def semantic_raw_features(
    events: Sequence[design.ContextEvent],
    family: str,
    *,
    velocity_mean: np.ndarray | None = None,
    velocity_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Build sparse endpoint features without opening within-event trajectories."""

    delta, duration = _endpoint_rows(events)
    if family == "delta":
        output = delta
    elif family == "mean_velocity":
        output = delta / duration[:, None]
    elif family == "direction_speed":
        v1._need(velocity_mean is not None and velocity_scale is not None,
                 "direction-speed requires a source-frozen velocity normalizer")
        mean = np.asarray(velocity_mean, dtype=np.float64)
        scale = np.asarray(velocity_scale, dtype=np.float64)
        v1._need(mean.shape == scale.shape == (v1.POSITION_DIM,) and np.all(scale > 0),
                 "invalid direction-speed normalizer")
        standardized = (delta / duration[:, None] - mean[None, :]) / scale[None, :]
        speed = np.linalg.norm(standardized, axis=1)
        safe = np.maximum(speed, SCALE_FLOOR)
        direction = standardized / safe[:, None]
        output = np.column_stack((direction, np.log1p(speed)))
    else:
        raise v1.SparseEventEndpointError(f"unknown semantic feature family {family}")
    v1._need(output.shape[0] == len(events) and np.isfinite(output).all(),
             f"{family}: invalid semantic feature matrix")
    return np.asarray(output, dtype=np.float64)


def _canonical_projection(projection: np.ndarray) -> np.ndarray:
    output = np.asarray(projection, dtype=np.float64).copy()
    for column in range(output.shape[1]):
        pivot = int(np.argmax(np.abs(output[:, column])))
        if output[pivot, column] < 0:
            output[:, column] *= -1.0
    return output


@dataclass(frozen=True)
class SemanticMap:
    candidate: SemanticCandidate
    outer_date: str
    source_sessions: tuple[str, ...]
    velocity_mean: np.ndarray
    velocity_scale: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    projection: np.ndarray
    latent_scale: np.ndarray
    energy_ratio: np.ndarray
    source_event_count: int
    map_sha256: str

    def transform(self, events: Sequence[design.ContextEvent]) -> np.ndarray:
        raw = semantic_raw_features(
            events,
            self.candidate.feature_family,
            velocity_mean=self.velocity_mean,
            velocity_scale=self.velocity_scale,
        )
        standardized = (raw - self.feature_mean[None, :]) / self.feature_scale[None, :]
        latent = standardized @ self.projection / self.latent_scale[None, :]
        v1._need(latent.shape == (len(events), RANK) and np.isfinite(latent).all(),
                 f"{self.candidate.name}: invalid semantic latent")
        return latent

    def manifest(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.name,
            "basis_mode": self.candidate.basis_mode,
            "feature_family": self.candidate.feature_family,
            "rank": RANK,
            "carrier_dim": CARRIER_DIM,
            "outer_date": self.outer_date,
            "source_sessions": list(self.source_sessions),
            "source_event_count": self.source_event_count,
            "cumulative_energy_at_rank": float(self.energy_ratio[:RANK].sum()),
            "array_sha256": {
                "velocity_mean": v1.array_sha256(self.velocity_mean),
                "velocity_scale": v1.array_sha256(self.velocity_scale),
                "feature_mean": v1.array_sha256(self.feature_mean),
                "feature_scale": v1.array_sha256(self.feature_scale),
                "projection": v1.array_sha256(self.projection),
                "latent_scale": v1.array_sha256(self.latent_scale),
            },
            "map_sha256": self.map_sha256,
        }


def _source_pool(
    sessions: Mapping[str, design.ContextSession], outer_date: str,
) -> tuple[tuple[str, ...], list[design.ContextEvent]]:
    names = tuple(name for name in v1.H1_HELDIN_SESSIONS if v1.session_date(name) != outer_date)
    v1._need(set(names).issubset(sessions), f"{outer_date}: missing semantic source sessions")
    events = [event for name in names for event in sessions[name].events]
    v1._need(bool(events), f"{outer_date}: empty semantic source pool")
    return names, events


def fit_semantic_map(
    sessions: Mapping[str, design.ContextSession],
    *,
    outer_date: str,
    candidate: SemanticCandidate,
) -> SemanticMap:
    names, pooled_events = _source_pool(sessions, outer_date)
    velocity = _base_velocity(pooled_events)
    velocity_mean = velocity.mean(axis=0)
    velocity_scale = np.maximum(velocity.std(axis=0), SCALE_FLOOR)
    raw = semantic_raw_features(
        pooled_events,
        candidate.feature_family,
        velocity_mean=velocity_mean,
        velocity_scale=velocity_scale,
    )
    feature_mean = raw.mean(axis=0)
    feature_scale = np.maximum(raw.std(axis=0), SCALE_FLOOR)
    x = (raw - feature_mean[None, :]) / feature_scale[None, :]

    if candidate.basis_mode == "pca":
        _left, singular, right = np.linalg.svd(x, full_matrices=False)
        v1._need(right.shape[0] >= RANK, f"{candidate.name}: PCA rank deficient")
        projection = right[:RANK].T
        energy = np.square(singular) / np.square(singular).sum()
    elif candidate.basis_mode == "source_encoding":
        coefficients: list[np.ndarray] = []
        offset = 0
        for name in names:
            count = len(sessions[name].events)
            xs = x[offset : offset + count]
            response = np.stack([event.base.log_rates for event in sessions[name].events]).astype(np.float64)
            response = (response - response.mean(axis=0, keepdims=True)) / np.maximum(
                response.std(axis=0, keepdims=True), 1.0e-6,
            )
            penalty = np.eye(xs.shape[1], dtype=np.float64) * (count * SOURCE_RIDGE_LAMBDA)
            coefficient = np.linalg.solve(xs.T @ xs + penalty, xs.T @ response)
            coefficient /= max(float(np.linalg.norm(coefficient, ord="fro")), 1.0e-12)
            coefficients.append(coefficient)
            offset += count
        v1._need(offset == len(pooled_events), f"{candidate.name}: source slicing drift")
        stacked = np.concatenate(coefficients, axis=1)
        projection, singular, _right = np.linalg.svd(stacked, full_matrices=False)
        v1._need(projection.shape[1] >= RANK, f"{candidate.name}: source-encoding rank deficient")
        projection = projection[:, :RANK]
        energy = np.square(singular) / np.square(singular).sum()
    else:
        raise v1.SparseEventEndpointError(f"unknown semantic basis mode {candidate.basis_mode}")

    projection = _canonical_projection(projection)
    latent_scale = np.maximum((x @ projection).std(axis=0), SCALE_FLOOR)
    body = {
        "protocol": PROTOCOL,
        "candidate": candidate.name,
        "outer_date": outer_date,
        "source_sessions": list(names),
        "source_event_count": len(pooled_events),
        "velocity_mean": v1.array_sha256(velocity_mean),
        "velocity_scale": v1.array_sha256(velocity_scale),
        "feature_mean": v1.array_sha256(feature_mean),
        "feature_scale": v1.array_sha256(feature_scale),
        "projection": v1.array_sha256(projection),
        "latent_scale": v1.array_sha256(latent_scale),
    }
    return SemanticMap(
        candidate=candidate,
        outer_date=outer_date,
        source_sessions=names,
        velocity_mean=np.asarray(velocity_mean, dtype=np.float64),
        velocity_scale=np.asarray(velocity_scale, dtype=np.float64),
        feature_mean=np.asarray(feature_mean, dtype=np.float64),
        feature_scale=np.asarray(feature_scale, dtype=np.float64),
        projection=np.asarray(projection, dtype=np.float64),
        latent_scale=np.asarray(latent_scale, dtype=np.float64),
        energy_ratio=np.asarray(energy, dtype=np.float64),
        source_event_count=len(pooled_events),
        map_sha256=v1.canonical_sha256(body),
    )


def evaluate_session(
    session: design.ContextSession,
    semantic_map: SemanticMap,
    *,
    budget: int,
) -> dict[str, Any]:
    support = design.select_range(session, start=0, budget=budget)
    later = tuple(event for event in session.events if event.base.trial_index >= budget)
    v1._need(len(support) >= CARRIER_DIM and len(later) >= 4,
             f"{session.base.session_name}: insufficient semantic support/query")
    z_support = semantic_map.transform(support)
    z_later = semantic_map.transform(later)
    y_support = np.stack([event.base.log_rates for event in support]).astype(np.float64)
    y_later = np.stack([event.base.log_rates for event in later]).astype(np.float64)
    durations = np.asarray([event.base.duration_seconds for event in support], dtype=np.float64)
    carrier = design.fit_target_carrier(
        z_support, y_support, durations=durations, weighted=False,
    )
    correct = v1.r2_by_channel(y_later, design.predict(carrier, z_later))

    order, shuffle = v1.within_trial_label_shuffle(
        tuple(event.base for event in support),
        session=session.base.session_name,
        budget=budget,
    )
    shuffled_carrier = design.fit_target_carrier(
        z_support[order], y_support, durations=durations, weighted=False,
    )
    shuffled = v1.r2_by_channel(y_later, design.predict(shuffled_carrier, z_later))
    intercept = v1.r2_by_channel(
        y_later,
        np.broadcast_to(y_support.mean(axis=0), y_later.shape),
    )
    defined = np.isfinite(correct) & np.isfinite(shuffled) & np.isfinite(intercept)
    v1._need(np.any(defined), f"{session.base.session_name}: no defined semantic forward channels")
    return {
        "session": session.base.session_name,
        "budget": budget,
        "support_events": len(support),
        "later_events": len(later),
        "design_rank": int(np.linalg.matrix_rank(np.column_stack((np.ones(len(support)), z_support)))),
        "carrier_dim": CARRIER_DIM,
        "defined_channels": int(defined.sum()),
        "median_r2_correct": float(np.median(correct[defined])),
        "median_delta_label_shuffle": float(np.median((correct - shuffled)[defined])),
        "median_delta_intercept": float(np.median((correct - intercept)[defined])),
        "correct_carrier_sha256": v1.array_sha256(carrier),
        "label_shuffle": shuffle,
    }


def _summary(rows: Mapping[str, Mapping[str, Any]], field: str) -> dict[str, Any]:
    return v1.paired_summary([(name, row.get(field)) for name, row in rows.items()])


def aggregate_candidate(
    rows: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    augmented = {
        name: {
            **row,
            "delta_vs_hse5": float(row["median_r2_correct"] - baseline[name]["median_r2_correct"]),
        }
        for name, row in rows.items()
    }
    return {
        "correct_r2": _summary(augmented, "median_r2_correct"),
        "correct_minus_hse5": _summary(augmented, "delta_vs_hse5"),
        "correct_minus_label_shuffle": _summary(augmented, "median_delta_label_shuffle"),
        "correct_minus_intercept": _summary(augmented, "median_delta_intercept"),
        "support_event_count": {
            "minimum": min(int(row["support_events"]) for row in rows.values()),
            "median": int(np.median([row["support_events"] for row in rows.values()])),
            "maximum": max(int(row["support_events"]) for row in rows.values()),
        },
    }


def _positive(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary["defined_sessions"] == 13
        and summary["mean"] > 0
        and summary["median"] > 0
        and summary["positive"] >= 10
        and summary["leave_largest_absolute_out_mean"] > 0
    )


def budget_gate(candidate: SemanticCandidate, aggregate: Mapping[str, Any]) -> dict[str, Any]:
    delta = aggregate["correct_minus_hse5"]
    material = bool(_positive(delta) and delta["mean"] >= 0.02 and delta["median"] >= 0.01)
    label = _positive(aggregate["correct_minus_label_shuffle"])
    intercept = _positive(aggregate["correct_minus_intercept"])
    passed = bool(candidate.name != "pca_delta_q4" and material and label and intercept)
    return {
        "passed": passed,
        "material_gain_vs_hse5": material,
        "correct_minus_label_shuffle": label,
        "correct_minus_intercept": intercept,
        "thresholds": {
            "mean_delta_vs_hse5": 0.02,
            "median_delta_vs_hse5": 0.01,
            "minimum_positive_sessions": 10,
            "leave_largest_absolute_out_mean_positive": True,
        },
    }


def run_screen(sessions: Mapping[str, design.ContextSession]) -> dict[str, Any]:
    maps = {
        candidate.name: {
            date: fit_semantic_map(sessions, outer_date=date, candidate=candidate)
            for date in v1.H1_DATES
        }
        for candidate in CANDIDATES
    }
    budgets: dict[str, Any] = {}
    for budget in (3, 4):
        rows_by_candidate = {
            candidate.name: {
                name: evaluate_session(
                    sessions[name], maps[candidate.name][v1.session_date(name)], budget=budget,
                )
                for name in v1.H1_HELDIN_SESSIONS
            }
            for candidate in CANDIDATES
        }
        baseline = rows_by_candidate["pca_delta_q4"]
        candidates: dict[str, Any] = {}
        for candidate in CANDIDATES:
            aggregate = aggregate_candidate(rows_by_candidate[candidate.name], baseline)
            candidates[candidate.name] = {
                "spec": {
                    "basis_mode": candidate.basis_mode,
                    "feature_family": candidate.feature_family,
                    "rank": RANK,
                    "carrier_dim": CARRIER_DIM,
                    "reads_only_endpoint_positions_and_event_timestamps": True,
                },
                "sessions": rows_by_candidate[candidate.name],
                "aggregate": aggregate,
                "gate": budget_gate(candidate, aggregate),
            }
        budgets[f"M{budget}"] = {"budget_trials": budget, "candidates": candidates}

    passing = [
        candidate.name
        for candidate in CANDIDATES
        if candidate.name != "pca_delta_q4"
        and budgets["M3"]["candidates"][candidate.name]["gate"]["passed"]
        and budgets["M4"]["candidates"][candidate.name]["gate"]["passed"]
    ]
    selected = None
    if passing:
        selected = max(
            passing,
            key=lambda name: (
                min(
                    float(budgets["M3"]["candidates"][name]["aggregate"]["correct_minus_hse5"]["median"]),
                    float(budgets["M4"]["candidates"][name]["aggregate"]["correct_minus_hse5"]["median"]),
                ),
                name,
            ),
        )
    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "frozen_constants": {
            "rank": RANK,
            "carrier_dim": CARRIER_DIM,
            "target_ridge_lambda": TARGET_RIDGE_LAMBDA,
            "source_ridge_lambda": SOURCE_RIDGE_LAMBDA,
            "support_budgets": [3, 4],
            "candidates_in_fixed_order": [candidate.name for candidate in CANDIDATES],
            "selection_rule": "pass frozen gate at both M3 and M4; maximize minimum-budget median delta",
        },
        "maps": {
            candidate.name: {date: maps[candidate.name][date].manifest() for date in v1.H1_DATES}
            for candidate in CANDIDATES
        },
        "budgets": budgets,
        "passing_candidates": passing,
        "selected_candidate": selected,
        "status": "PASS_CPU_SEMANTIC_CANDIDATE_SELECTED" if selected else "STOP_CPU_SEMANTIC_CANDIDATE_NOT_MATERIAL",
        "gpu_authorized_by_this_screen": False,
        "scope": {
            "native_position_endpoints_per_event": 2,
            "native_event_timestamps_per_event": 2,
            "dense_velocity_opened": False,
            "within_event_position_trajectory_opened": False,
            "target_session_optimizer_steps": 0,
            "target_session_backward_steps": 0,
        },
    }
