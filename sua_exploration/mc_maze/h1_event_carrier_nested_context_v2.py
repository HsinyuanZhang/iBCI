"""Nested source-only refinement of the H1 event-context carrier.

V1 showed that source-supervised ``delta + midpoint + tag`` was the only arm
with 13/13 positive gains at both M3 and M4, but it missed the frozen material
effect threshold.  V2 does not lower that threshold.  It selects one of a
small fixed context/regularization matrix using only inner source dates for
each outer date, then evaluates that selected algorithm on the outer date.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from sua_exploration.mc_maze import h1_event_carrier_design_screen as v1screen
from sua_exploration.mc_maze import h1_sparse_event_endpoint as event_v1


SCHEMA = "h1_event_carrier_nested_context_cpu_v2"
PROTOCOL = "h1_event_carrier_nested_source_context_20260811_v2"
RANK = 4
FEATURE_SCALE_FLOOR = 1.0e-8
LATENT_SCALE_FLOOR = 1.0e-8


@dataclass(frozen=True)
class Configuration:
    name: str
    feature_family: str
    source_ridge: float
    target_ridge: float
    response_mode: str = "zscore"
    balance_session_coefficients: bool = True


CONFIGURATIONS: tuple[Configuration, ...] = (
    Configuration("context_mid_anchor", "context_mid", 1.0, 3.0),
    Configuration("context_start", "context_start", 1.0, 3.0),
    Configuration("context_stop", "context_stop", 1.0, 3.0),
    Configuration("tag_delta", "tag_delta", 1.0, 3.0),
    Configuration("context_mid_source_l0p1", "context_mid", 0.1, 3.0),
    Configuration("context_mid_source_l3", "context_mid", 3.0, 3.0),
    Configuration("context_mid_source_l10", "context_mid", 10.0, 3.0),
    Configuration("context_mid_target_l1", "context_mid", 1.0, 1.0),
    Configuration("context_mid_target_l10", "context_mid", 1.0, 10.0),
    Configuration("context_mid_center_response", "context_mid", 1.0, 3.0, response_mode="center"),
    Configuration("context_mid_unbalanced_sessions", "context_mid", 1.0, 3.0, balance_session_coefficients=False),
)


@dataclass(frozen=True)
class ContextMap:
    configuration: Configuration
    source_sessions: tuple[str, ...]
    raw_dim: int
    active_mask: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    projection: np.ndarray
    latent_scale: np.ndarray
    energy_ratio: np.ndarray
    source_event_count: int
    map_sha256: str

    def transform(
        self,
        events: Sequence[v1screen.ContextEvent],
        *,
        tag_overrides: Sequence[str] | None = None,
    ) -> np.ndarray:
        raw = raw_features(events, self.configuration.feature_family, tag_overrides=tag_overrides)
        event_v1._need(raw.shape[1] == self.raw_dim, "nested context raw feature drift")
        x = (raw[:, self.active_mask] - self.mean[None, :]) / self.scale[None, :]
        z = x @ self.projection / self.latent_scale[None, :]
        event_v1._need(z.shape == (len(events), RANK) and np.isfinite(z).all(), "invalid nested context latent")
        return z

    def manifest(self) -> dict[str, Any]:
        return {
            "configuration": self.configuration.name,
            "feature_family": self.configuration.feature_family,
            "source_ridge": self.configuration.source_ridge,
            "target_ridge": self.configuration.target_ridge,
            "response_mode": self.configuration.response_mode,
            "balance_session_coefficients": self.configuration.balance_session_coefficients,
            "source_sessions": list(self.source_sessions),
            "source_event_count": self.source_event_count,
            "raw_dim": self.raw_dim,
            "active_dim": int(self.active_mask.sum()),
            "energy_at_rank4": float(self.energy_ratio[:RANK].sum()),
            "array_sha256": {
                "active_mask": event_v1.array_sha256(self.active_mask),
                "mean": event_v1.array_sha256(self.mean),
                "scale": event_v1.array_sha256(self.scale),
                "projection": event_v1.array_sha256(self.projection),
                "latent_scale": event_v1.array_sha256(self.latent_scale),
            },
            "map_sha256": self.map_sha256,
        }


def raw_features(
    events: Sequence[v1screen.ContextEvent],
    family: str,
    *,
    tag_overrides: Sequence[str] | None = None,
) -> np.ndarray:
    event_v1._need(bool(events), "nested context event set empty")
    tags = tuple(tag_overrides) if tag_overrides is not None else tuple(event.base.tag for event in events)
    event_v1._need(len(tags) == len(events), "nested context tag override mismatch")
    onehot = np.zeros((len(events), len(event_v1.MOVEMENT_TAGS)), dtype=np.float64)
    for row, tag in enumerate(tags):
        onehot[row, event_v1.MOVEMENT_TAGS.index(tag)] = 1.0
    delta = np.stack([event.base.displacement for event in events]).astype(np.float64)
    start = np.stack([event.start_state for event in events]).astype(np.float64)
    midpoint = np.stack([event.midpoint_state for event in events]).astype(np.float64)
    stop = start + delta
    if family == "context_mid":
        output = np.concatenate((delta, midpoint, onehot), axis=1)
    elif family == "context_start":
        output = np.concatenate((delta, start, onehot), axis=1)
    elif family == "context_stop":
        output = np.concatenate((delta, stop, onehot), axis=1)
    elif family == "tag_delta":
        interaction = (onehot[:, :, None] * delta[:, None, :]).reshape(len(events), -1)
        output = np.concatenate((onehot, interaction), axis=1)
    elif family == "delta":
        output = delta
    else:
        raise event_v1.SparseEventEndpointError(f"unknown nested context feature family {family}")
    event_v1._need(np.isfinite(output).all(), "nested context feature nonfinite")
    return output


def _canonical_projection(value: np.ndarray) -> np.ndarray:
    output = np.asarray(value, np.float64).copy()
    for column in range(output.shape[1]):
        pivot = int(np.argmax(np.abs(output[:, column])))
        if output[pivot, column] < 0:
            output[:, column] *= -1.0
    return output


def fit_context_map(
    sessions: Mapping[str, v1screen.ContextSession],
    *,
    source_names: Sequence[str],
    configuration: Configuration,
) -> ContextMap:
    names = tuple(source_names)
    event_v1._need(bool(names) and len(set(names)) == len(names), "invalid nested source roster")
    pooled = [event for name in names for event in sessions[name].events]
    raw = raw_features(pooled, configuration.feature_family)
    mean_all, scale_all = raw.mean(axis=0), raw.std(axis=0)
    active = scale_all > FEATURE_SCALE_FLOOR
    event_v1._need(int(active.sum()) >= RANK, "nested context active feature rank too small")
    mean, scale = mean_all[active], scale_all[active]
    x = (raw[:, active] - mean[None, :]) / scale[None, :]
    coefficients: list[np.ndarray] = []
    offset = 0
    for name in names:
        count = len(sessions[name].events)
        xs = x[offset : offset + count]
        response = np.stack([event.base.log_rates for event in sessions[name].events]).astype(np.float64)
        response = response - response.mean(axis=0, keepdims=True)
        if configuration.response_mode == "zscore":
            response = response / np.maximum(response.std(axis=0, keepdims=True), 1.0e-6)
        elif configuration.response_mode != "center":
            raise event_v1.SparseEventEndpointError(f"unknown response mode {configuration.response_mode}")
        penalty = np.eye(xs.shape[1]) * (count * configuration.source_ridge)
        coefficient = np.linalg.solve(xs.T @ xs + penalty, xs.T @ response)
        if configuration.balance_session_coefficients:
            coefficient /= max(float(np.linalg.norm(coefficient, ord="fro")), 1.0e-12)
        coefficients.append(coefficient)
        offset += count
    event_v1._need(offset == len(pooled), "nested source slicing drift")
    stacked = np.concatenate(coefficients, axis=1)
    left, singular, _right = np.linalg.svd(stacked, full_matrices=False)
    projection = _canonical_projection(left[:, :RANK])
    latent = x @ projection
    latent_scale = np.maximum(latent.std(axis=0), LATENT_SCALE_FLOOR)
    energy = np.square(singular) / np.square(singular).sum()
    body = {
        "protocol": PROTOCOL,
        "configuration": configuration.name,
        "source_sessions": list(names),
        "active_mask": event_v1.array_sha256(active),
        "mean": event_v1.array_sha256(mean),
        "scale": event_v1.array_sha256(scale),
        "projection": event_v1.array_sha256(projection),
        "latent_scale": event_v1.array_sha256(latent_scale),
    }
    return ContextMap(
        configuration=configuration,
        source_sessions=names,
        raw_dim=raw.shape[1],
        active_mask=np.asarray(active, bool),
        mean=np.asarray(mean, np.float64),
        scale=np.asarray(scale, np.float64),
        projection=np.asarray(projection, np.float64),
        latent_scale=np.asarray(latent_scale, np.float64),
        energy_ratio=np.asarray(energy, np.float64),
        source_event_count=len(pooled),
        map_sha256=event_v1.canonical_sha256(body),
    )


def fit_pca_baseline(
    sessions: Mapping[str, v1screen.ContextSession], *, source_names: Sequence[str],
) -> ContextMap:
    configuration = Configuration("pca_delta_q4", "delta", 0.0, 3.0)
    names = tuple(source_names)
    pooled = [event for name in names for event in sessions[name].events]
    raw = raw_features(pooled, "delta")
    mean, scale = raw.mean(axis=0), raw.std(axis=0)
    active = scale > FEATURE_SCALE_FLOOR
    x = (raw[:, active] - mean[active][None, :]) / scale[active][None, :]
    _u, singular, right = np.linalg.svd(x, full_matrices=False)
    projection = _canonical_projection(right[:RANK].T)
    latent = x @ projection
    latent_scale = np.maximum(latent.std(axis=0), LATENT_SCALE_FLOOR)
    energy = np.square(singular) / np.square(singular).sum()
    body = {
        "protocol": PROTOCOL,
        "configuration": configuration.name,
        "source_sessions": list(names),
        "active_mask": event_v1.array_sha256(active),
        "mean": event_v1.array_sha256(mean[active]),
        "scale": event_v1.array_sha256(scale[active]),
        "projection": event_v1.array_sha256(projection),
        "latent_scale": event_v1.array_sha256(latent_scale),
    }
    return ContextMap(
        configuration=configuration,
        source_sessions=names,
        raw_dim=7,
        active_mask=np.asarray(active, bool),
        mean=np.asarray(mean[active], np.float64),
        scale=np.asarray(scale[active], np.float64),
        projection=np.asarray(projection, np.float64),
        latent_scale=np.asarray(latent_scale, np.float64),
        energy_ratio=np.asarray(energy, np.float64),
        source_event_count=len(pooled),
        map_sha256=event_v1.canonical_sha256(body),
    )


def fit_target(z: np.ndarray, response: np.ndarray, *, ridge: float) -> np.ndarray:
    z = np.asarray(z, np.float64)
    response = np.asarray(response, np.float64)
    design = np.column_stack((np.ones(len(z)), z))
    event_v1._need(np.linalg.matrix_rank(design) == RANK + 1, "nested target design rank deficient")
    penalty = np.diag([0.0] + [1.0] * RANK) * (len(z) * ridge)
    coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ response)
    return np.column_stack((coefficient[1:].T, coefficient[0]))


def evaluate(
    session: v1screen.ContextSession,
    mapping: ContextMap,
    *,
    budget: int,
) -> dict[str, Any]:
    support = tuple(event for event in session.events if event.base.trial_index < budget)
    later = tuple(event for event in session.events if event.base.trial_index >= budget)
    z_support, z_later = mapping.transform(support), mapping.transform(later)
    y_support = np.stack([event.base.log_rates for event in support]).astype(np.float64)
    y_later = np.stack([event.base.log_rates for event in later]).astype(np.float64)
    carrier = fit_target(z_support, y_support, ridge=mapping.configuration.target_ridge)
    r_correct = event_v1.r2_by_channel(y_later, v1screen.predict(carrier, z_later))
    label_order, label_manifest = event_v1.within_trial_label_shuffle(
        tuple(event.base for event in support), session=session.base.session_name, budget=budget,
    )
    label_carrier = fit_target(z_support[label_order], y_support, ridge=mapping.configuration.target_ridge)
    r_label = event_v1.r2_by_channel(y_later, v1screen.predict(label_carrier, z_later))
    tag_order, tag_manifest = v1screen.within_trial_tag_shuffle(
        support, session=session.base.session_name, budget=budget,
    )
    wrong_tags = tuple(support[int(index)].base.tag for index in tag_order)
    z_tag = mapping.transform(support, tag_overrides=wrong_tags)
    tag_carrier = fit_target(z_tag, y_support, ridge=mapping.configuration.target_ridge)
    r_tag = event_v1.r2_by_channel(y_later, v1screen.predict(tag_carrier, z_later))
    mean = y_support.mean(axis=0)
    r_intercept = event_v1.r2_by_channel(y_later, np.broadcast_to(mean, y_later.shape))
    defined = np.isfinite(r_correct) & np.isfinite(r_label) & np.isfinite(r_tag) & np.isfinite(r_intercept)
    event_v1._need(np.any(defined), "nested context forward score undefined")
    return {
        "session": session.base.session_name,
        "budget": budget,
        "configuration": mapping.configuration.name,
        "support_events": len(support),
        "later_events": len(later),
        "defined_channels": int(defined.sum()),
        "median_r2_correct": float(np.median(r_correct[defined])),
        "median_delta_label_shuffle": float(np.median((r_correct - r_label)[defined])),
        "median_delta_tag_shuffle": float(np.median((r_correct - r_tag)[defined])),
        "median_delta_intercept": float(np.median((r_correct - r_intercept)[defined])),
        "label_shuffle": label_manifest,
        "tag_shuffle": tag_manifest,
    }


def _paired(rows: Sequence[tuple[str, float]]) -> dict[str, Any]:
    return event_v1.paired_summary([(name, value) for name, value in rows])


def compare_rows(
    candidate: Mapping[str, Mapping[str, Any]], baseline: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "candidate_minus_baseline": _paired([
            (name, float(candidate[name]["median_r2_correct"] - baseline[name]["median_r2_correct"]))
            for name in candidate
        ]),
        "candidate_minus_intercept": _paired([
            (name, float(candidate[name]["median_delta_intercept"])) for name in candidate
        ]),
        "candidate_minus_label_shuffle": _paired([
            (name, float(candidate[name]["median_delta_label_shuffle"])) for name in candidate
        ]),
        "candidate_minus_tag_shuffle": _paired([
            (name, float(candidate[name]["median_delta_tag_shuffle"])) for name in candidate
        ]),
    }


def _inner_eligible(summary_by_budget: Mapping[str, Mapping[str, Any]]) -> bool:
    for budget in ("M3", "M4"):
        summaries = summary_by_budget[budget]
        total = int(summaries["candidate_minus_baseline"]["defined_sessions"])
        minimum_positive = math.ceil(0.6 * total)
        for key in (
            "candidate_minus_baseline", "candidate_minus_intercept",
            "candidate_minus_label_shuffle", "candidate_minus_tag_shuffle",
        ):
            row = summaries[key]
            if not (
                row["mean"] > 0 and row["median"] > 0
                and row["positive"] >= minimum_positive
                and row["leave_largest_absolute_out_mean"] > 0
            ):
                return False
    return True


def _selection_key(summary_by_budget: Mapping[str, Mapping[str, Any]], name: str) -> tuple[float, float, str]:
    medians = [summary_by_budget[budget]["candidate_minus_baseline"]["median"] for budget in ("M3", "M4")]
    means = [summary_by_budget[budget]["candidate_minus_baseline"]["mean"] for budget in ("M3", "M4")]
    return min(medians), min(means), name


def run_nested(sessions: Mapping[str, v1screen.ContextSession]) -> dict[str, Any]:
    outer_rows: dict[str, dict[str, Any]] = {"M3": {}, "M4": {}}
    outer_baseline: dict[str, dict[str, Any]] = {"M3": {}, "M4": {}}
    outer_receipts: dict[str, Any] = {}

    for outer_date in event_v1.H1_DATES:
        source_dates = tuple(date for date in event_v1.H1_DATES if date != outer_date)
        inner_summaries: dict[str, Any] = {}
        for configuration in CONFIGURATIONS:
            rows: dict[str, dict[str, Any]] = {"M3": {}, "M4": {}}
            baselines: dict[str, dict[str, Any]] = {"M3": {}, "M4": {}}
            map_hashes: dict[str, str] = {}
            for inner_date in source_dates:
                train_names = tuple(
                    name for name in event_v1.H1_HELDIN_SESSIONS
                    if event_v1.session_date(name) not in {outer_date, inner_date}
                )
                candidate_map = fit_context_map(sessions, source_names=train_names, configuration=configuration)
                baseline_map = fit_pca_baseline(sessions, source_names=train_names)
                map_hashes[inner_date] = candidate_map.map_sha256
                for name in event_v1.H1_HELDIN_SESSIONS:
                    if event_v1.session_date(name) != inner_date:
                        continue
                    for budget in (3, 4):
                        rows[f"M{budget}"][name] = evaluate(sessions[name], candidate_map, budget=budget)
                        baselines[f"M{budget}"][name] = evaluate(sessions[name], baseline_map, budget=budget)
            summary = {budget: compare_rows(rows[budget], baselines[budget]) for budget in ("M3", "M4")}
            inner_summaries[configuration.name] = {
                "configuration": configuration.__dict__,
                "summaries": summary,
                "eligible": _inner_eligible(summary),
                "inner_map_sha256_by_date": map_hashes,
            }

        eligible = [name for name, row in inner_summaries.items() if row["eligible"]]
        event_v1._need(bool(eligible), f"{outer_date}: no inner-eligible context configuration")
        selected_name = max(
            eligible,
            key=lambda name: _selection_key(inner_summaries[name]["summaries"], name),
        )
        selected = next(item for item in CONFIGURATIONS if item.name == selected_name)
        train_names = tuple(
            name for name in event_v1.H1_HELDIN_SESSIONS
            if event_v1.session_date(name) != outer_date
        )
        selected_map = fit_context_map(sessions, source_names=train_names, configuration=selected)
        baseline_map = fit_pca_baseline(sessions, source_names=train_names)
        target_names = tuple(
            name for name in event_v1.H1_HELDIN_SESSIONS
            if event_v1.session_date(name) == outer_date
        )
        for name in target_names:
            for budget in (3, 4):
                outer_rows[f"M{budget}"][name] = evaluate(sessions[name], selected_map, budget=budget)
                outer_baseline[f"M{budget}"][name] = evaluate(sessions[name], baseline_map, budget=budget)
        outer_receipts[outer_date] = {
            "selected_configuration": selected_name,
            "eligible_configurations": eligible,
            "inner_summaries": inner_summaries,
            "final_map": selected_map.manifest(),
            "baseline_map": baseline_map.manifest(),
            "target_sessions": list(target_names),
        }

    aggregate = {
        budget: compare_rows(outer_rows[budget], outer_baseline[budget])
        for budget in ("M3", "M4")
    }
    gates: dict[str, Any] = {}
    for budget in ("M3", "M4"):
        rows = aggregate[budget]
        material = rows["candidate_minus_baseline"]
        clauses = {
            "mean_delta_at_least_0p02": material["mean"] >= 0.02,
            "median_delta_at_least_0p01": material["median"] >= 0.01,
            "positive_sessions_at_least_10": material["positive"] >= 10,
            "leave_largest_delta_positive": material["leave_largest_absolute_out_mean"] > 0,
            "beats_intercept_robustly": v1screen._positive_control(rows["candidate_minus_intercept"]),
            "beats_label_shuffle_robustly": v1screen._positive_control(rows["candidate_minus_label_shuffle"]),
            "beats_tag_shuffle_robustly": v1screen._positive_control(rows["candidate_minus_tag_shuffle"]),
        }
        gates[budget] = {"clauses": clauses, "passed": all(clauses.values())}
    passed = gates["M3"]["passed"] and gates["M4"]["passed"]
    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "configuration_matrix": [configuration.__dict__ for configuration in CONFIGURATIONS],
        "selection_rule": {
            "per_outer_date": "inner-source-date eligible configs only; maximize minimum M3/M4 median delta, then mean delta",
            "outer_target_used_for_configuration_selection": False,
        },
        "outer_dates": outer_receipts,
        "outer_rows": outer_rows,
        "outer_baseline_rows": outer_baseline,
        "aggregate": aggregate,
        "gate": {"budgets": gates, "passed": passed},
        "status": "PASS_CPU_NESTED_CONTEXT_MATERIAL" if passed else "STOP_CPU_NESTED_CONTEXT_NOT_MATERIAL",
        "gpu_authorized_by_this_screen": False,
    }
