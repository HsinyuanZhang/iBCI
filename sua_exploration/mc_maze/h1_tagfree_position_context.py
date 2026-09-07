"""CPU-only H1 tag-free position-context carrier screen.

Reimplements feature construction and latent-map fitting with a pluggable
feature function because the sealed design screen hard-codes family dispatch.
Reference arms must reproduce the sealed design-screen receipt exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from sua_exploration.mc_maze import h1_event_carrier_design_screen as sealed
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1

SCHEMA = "h1_tagfree_position_context_cpu_screen_v1"
PROTOCOL = "h1_tagfree_position_context_carrier_20260812_v1"
FEATURE_SCALE_FLOOR = sealed.FEATURE_SCALE_FLOOR
LATENT_SCALE_FLOOR = sealed.LATENT_SCALE_FLOOR
SOURCE_RIDGE_LAMBDA = sealed.SOURCE_RIDGE_LAMBDA
TARGET_RIDGE_LAMBDA = sealed.TARGET_RIDGE_LAMBDA
TAG_NAMES = sealed.TAG_NAMES
TAG_INDEX = sealed.TAG_INDEX

PRIMARY_GATE_MEAN_M3 = 0.012312
PRIMARY_GATE_MEAN_M4 = 0.011550
PRIMARY_GATE_MIN_POSITIVE = 11
STRETCH_GATE_MEAN_M3 = 0.015390
STRETCH_GATE_MEAN_M4 = 0.014438
POSITIVE_CONTROL_MIN_POSITIVE = 11
INTEGRITY_TOLERANCE = 1.0e-10

ContextEvent = sealed.ContextEvent
ContextSession = sealed.ContextSession
load_context_sessions = sealed.load_context_sessions
fit_target_carrier = sealed.fit_target_carrier
predict = sealed.predict
select_range = sealed.select_range
_canonical_projection = sealed._canonical_projection

SEALED_REFERENCE_CANDIDATES = ("pca_delta_q4", "ser_context_q4")
TAG_FREE_CANDIDATES = (
    "ser_poscontext_q4",
    "pca_poscontext_q4",
    "ser_startstop_q4",
    "ser_deltastart_q4",
)


@dataclass(frozen=True)
class Candidate:
    name: str
    basis_mode: str
    feature_family: str
    rank: int

    @property
    def tag_dependent(self) -> bool:
        return self.feature_family == "context"

    @property
    def carrier_dim(self) -> int:
        return self.rank + 1


CANDIDATES: tuple[Candidate, ...] = (
    Candidate("pca_delta_q4", "pca", "delta", 4),
    Candidate("ser_context_q4", "source_encoding", "context", 4),
    Candidate("ser_poscontext_q4", "source_encoding", "poscontext", 4),
    Candidate("pca_poscontext_q4", "pca", "poscontext", 4),
    Candidate("ser_startstop_q4", "source_encoding", "startstop", 4),
    Candidate("ser_deltastart_q4", "source_encoding", "deltastart", 4),
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
        v1._need(
            latent.shape == (len(events), self.candidate.rank) and np.isfinite(latent).all(),
            f"{self.candidate.name}: invalid latent values",
        )
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
            "retained_energy_at_rank": float(self.energy_ratio[: self.candidate.rank].sum()),
            "array_sha256": {
                "active_mask": v1.array_sha256(self.active_mask),
                "feature_mean": v1.array_sha256(self.feature_mean),
                "feature_scale": v1.array_sha256(self.feature_scale),
                "projection": v1.array_sha256(self.projection),
                "latent_scale": v1.array_sha256(self.latent_scale),
            },
            "map_sha256": self.map_sha256,
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
    start = np.stack([event.start_state for event in events]).astype(np.float64)
    midpoint = np.stack([event.midpoint_state for event in events]).astype(np.float64)
    stop = start + delta
    onehot = _onehot(tags)
    if family == "delta":
        output = delta
    elif family == "context":
        output = np.concatenate((delta, midpoint, onehot), axis=1)
    elif family == "poscontext":
        output = np.concatenate((delta, midpoint), axis=1)
    elif family == "startstop":
        output = np.concatenate((start, stop), axis=1)
    elif family == "deltastart":
        output = np.concatenate((delta, start), axis=1)
    else:
        raise v1.SparseEventEndpointError(f"unknown feature family {family}")
    v1._need(np.isfinite(output).all(), f"{family}: nonfinite raw feature")
    return output


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


def evaluate_session(
    session: ContextSession, latent_map: LatentMap, *, budget: int,
) -> dict[str, Any]:
    support = select_range(session, start=0, budget=budget)
    later = tuple(event for event in session.events if event.base.trial_index >= budget)
    v1._need(
        len(support) >= latent_map.candidate.carrier_dim and len(later) >= 4,
        f"{session.base.session_name}: insufficient support/query events",
    )
    z_support = latent_map.transform(support)
    z_later = latent_map.transform(later)
    y_support = np.stack([event.base.log_rates for event in support]).astype(np.float64)
    y_later = np.stack([event.base.log_rates for event in later]).astype(np.float64)
    durations = np.asarray([event.base.duration_seconds for event in support], np.float64)
    carrier = fit_target_carrier(z_support, y_support, durations=durations, weighted=False)
    r_correct = v1.r2_by_channel(y_later, predict(carrier, z_later))

    order, label_manifest = v1.within_trial_label_shuffle(
        tuple(event.base for event in support), session=session.base.session_name, budget=budget,
    )
    label_carrier = fit_target_carrier(z_support[order], y_support, durations=durations, weighted=False)
    r_label = v1.r2_by_channel(y_later, predict(label_carrier, z_later))
    support_mean = y_support.mean(axis=0)
    r_intercept = v1.r2_by_channel(y_later, np.broadcast_to(support_mean, y_later.shape))

    defined = np.isfinite(r_correct) & np.isfinite(r_label) & np.isfinite(r_intercept)
    v1._need(np.any(defined), f"{session.base.session_name}: no defined forward channels")
    return {
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
    }


def _summary(rows: Mapping[str, Mapping[str, Any]], field: str) -> dict[str, Any]:
    return v1.paired_summary([(name, row.get(field)) for name, row in rows.items()])


def aggregate_candidate(
    rows: Mapping[str, Mapping[str, Any]], baseline: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    augmented = {
        name: {
            **row,
            "delta_vs_pca_delta_q4": float(row["median_r2_correct"] - baseline[name]["median_r2_correct"]),
        }
        for name, row in rows.items()
    }
    return {
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


def _positive_control(summary: Mapping[str, Any], *, minimum_positive: int = POSITIVE_CONTROL_MIN_POSITIVE) -> bool:
    return bool(
        summary["defined_sessions"] == 13
        and summary["mean"] > 0
        and summary["median"] > 0
        and summary["positive"] >= minimum_positive
        and summary["leave_largest_absolute_out_mean"] > 0
    )


def evaluate_budget_gate(
    aggregate: Mapping[str, Any], *, budget: int, candidate: Candidate,
) -> dict[str, Any]:
    vs = aggregate["correct_minus_pca_delta_q4"]
    mean_threshold = PRIMARY_GATE_MEAN_M3 if budget == 3 else PRIMARY_GATE_MEAN_M4
    stretch_threshold = STRETCH_GATE_MEAN_M3 if budget == 3 else STRETCH_GATE_MEAN_M4
    mean_pass = bool(vs["mean"] >= mean_threshold)
    positive_pass = bool(vs["positive"] >= PRIMARY_GATE_MIN_POSITIVE)
    loo_pass = bool(vs["leave_largest_absolute_out_mean"] > 0)
    label_pass = _positive_control(aggregate["correct_minus_label_shuffle"])
    intercept_pass = _positive_control(aggregate["correct_minus_intercept"])
    stretch_pass = bool(vs["mean"] >= stretch_threshold)
    passes_primary = bool(
        candidate.name in TAG_FREE_CANDIDATES
        and mean_pass
        and positive_pass
        and loo_pass
        and label_pass
        and intercept_pass
    )
    return {
        "budget": budget,
        "mean_vs_pca_delta_q4": vs["mean"],
        "median_vs_pca_delta_q4": vs["median"],
        "positive_vs_pca_delta_q4": vs["positive"],
        "leave_largest_absolute_out_mean_vs_pca_delta_q4": vs["leave_largest_absolute_out_mean"],
        "passes_mean_threshold": mean_pass,
        "passes_positive_sessions": positive_pass,
        "passes_leave_largest_out": loo_pass,
        "passes_correct_minus_label_shuffle": label_pass,
        "passes_correct_minus_intercept": intercept_pass,
        "passes_primary_gate": passes_primary,
        "passes_stretch_gate": stretch_pass,
        "thresholds": {
            "mean_vs_pca_delta_q4": mean_threshold,
            "stretch_mean_vs_pca_delta_q4": stretch_threshold,
            "minimum_positive_sessions": PRIMARY_GATE_MIN_POSITIVE,
            "positive_control_minimum_positive_sessions": POSITIVE_CONTROL_MIN_POSITIVE,
        },
    }


def evaluate_candidate_gate(candidate: Candidate, budgets: Mapping[str, Any]) -> dict[str, Any]:
    m3 = budgets["M3"]["candidates"][candidate.name]["gate"]
    m4 = budgets["M4"]["candidates"][candidate.name]["gate"]
    return {
        "passes_primary_gate": bool(m3["passes_primary_gate"] and m4["passes_primary_gate"]),
        "passes_stretch_gate": bool(m3["passes_stretch_gate"] and m4["passes_stretch_gate"]),
        "M3": m3,
        "M4": m4,
    }


def verify_sealed_integrity(
    body: Mapping[str, Any], sealed_receipt: Mapping[str, Any], *, tolerance: float = INTEGRITY_TOLERANCE,
) -> dict[str, Any]:
    fields = ("median_r2_correct", "median_delta_label_shuffle", "median_delta_intercept")
    differences: list[dict[str, Any]] = []
    for budget in (3, 4):
        for candidate_name in SEALED_REFERENCE_CANDIDATES:
            observed_rows = body["budgets"][f"M{budget}"]["candidates"][candidate_name]["sessions"]
            sealed_rows = sealed_receipt["budgets"][f"M{budget}"]["candidates"][candidate_name]["sessions"]
            for session in v1.H1_HELDIN_SESSIONS:
                for field in fields:
                    observed = float(observed_rows[session][field])
                    expected = float(sealed_rows[session][field])
                    differences.append({
                        "budget": budget,
                        "candidate": candidate_name,
                        "session": session,
                        "field": field,
                        "absolute_difference": abs(observed - expected),
                    })
    maximum = max(row["absolute_difference"] for row in differences)
    return {
        "passed": maximum <= tolerance,
        "absolute_tolerance": tolerance,
        "comparisons": len(differences),
        "maximum_absolute_difference": maximum,
        "sealed_reference_candidates": list(SEALED_REFERENCE_CANDIDATES),
    }


def _mean_retained_energy(body: Mapping[str, Any], candidate_name: str) -> float:
    values = [
        float(body["basis_by_candidate_and_outer_date"][candidate_name][date]["retained_energy_at_rank"])
        for date in v1.H1_DATES
    ]
    return float(np.mean(values))


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
                    "tag_dependent": candidate.tag_dependent,
                },
                "sessions": rows_by_candidate[candidate.name],
                "aggregate": aggregate,
                "gate": evaluate_budget_gate(aggregate, budget=budget, candidate=candidate),
                "mean_retained_energy_at_rank": _mean_retained_energy(
                    {
                        "basis_by_candidate_and_outer_date": {
                            candidate.name: {
                                date: maps[candidate.name][date].manifest() for date in v1.H1_DATES
                            }
                        }
                    },
                    candidate.name,
                ),
            }
        budgets[f"M{budget}"] = {"budget_trials": budget, "candidates": candidate_bodies}

    candidate_gates = {
        candidate.name: evaluate_candidate_gate(candidate, budgets)
        for candidate in CANDIDATES
        if candidate.name in TAG_FREE_CANDIDATES
    }
    passing = [
        name for name in TAG_FREE_CANDIDATES
        if candidate_gates[name]["passes_primary_gate"]
    ]
    selected = None
    if passing:
        def rank_key(name: str) -> tuple[float, str]:
            m3_mean = budgets["M3"]["candidates"][name]["aggregate"]["correct_minus_pca_delta_q4"]["mean"]
            m4_mean = budgets["M4"]["candidates"][name]["aggregate"]["correct_minus_pca_delta_q4"]["mean"]
            return (min(float(m3_mean), float(m4_mean)), name)
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
            "primary_gate_mean_m3": PRIMARY_GATE_MEAN_M3,
            "primary_gate_mean_m4": PRIMARY_GATE_MEAN_M4,
            "stretch_gate_mean_m3": STRETCH_GATE_MEAN_M3,
            "stretch_gate_mean_m4": STRETCH_GATE_MEAN_M4,
            "selection_rule": "pass primary gate at both M3 and M4; maximize minimum budget mean contrast vs pca_delta_q4; tie-break by name",
        },
        "basis_by_candidate_and_outer_date": {
            candidate.name: {date: maps[candidate.name][date].manifest() for date in v1.H1_DATES}
            for candidate in CANDIDATES
        },
        "budgets": budgets,
        "candidate_gates": candidate_gates,
        "passing_tag_free_candidates": passing,
        "selected_candidate": selected,
        "status": "PASS_CPU_PCTX_CANDIDATE_SELECTED" if selected else "STOP_CPU_PCTX_NOT_MATERIAL",
        "scope": {
            "cuda_used": False,
            "decoder_constructed": False,
            "dense_velocity_series_opened": False,
        },
    }
