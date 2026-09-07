"""Source-only low-rank channel-tuning (LRT5) carrier screen for H1.

This module tests a deliberately constrained alternative to ordinary H-SE5.
The endpoint representation stays exactly q=4 PCA endpoint displacement, but
the ``[4,176]`` slope matrix is constrained to a source-frozen *channel* tuning
subspace.  A target support block estimates only a small ``[4,r]`` coefficient
matrix in closed form:

    Yc ~= Zc A U.T,    W = A U.T,    b = mean(Y - Z W).

``U`` comes from an SVD of non-outer-date, full-event H-SE5 slope maps; the
source slope mean is removed before that SVD and is not inserted into the
deployable correct carrier.  Thus the target's endpoint/neural pairing is the
only way to determine ``A``.  ``U`` is channel-indexed, so every evaluation
contains both a fixed-point-free ``U`` row-shuffle refit and a source-prior-only
control.  The programme opens only the public H1 held-in calibration files,
never dense velocity, and never creates a target-session optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import hashlib
import numpy as np

from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as hse5


# V2 changes only the static source-prior *control*: its intercept now aligns
# nonzero prior slopes at the target support mean, b = mean(Y) - mean(Z) W.
# The literal candidate/grid, all endpoint maps/subspaces, and the correct
# target-pairing LRT5 arm are unchanged from r1.
SCHEMA = "h1_event_carrier_lrt5_source_screen_v2"
PROTOCOL = "h1_event_carrier_low_rank_channel_tuning_source_screen_20260812_v2"
RANK = 4
CARRIER_DIM = 5
SOURCE_SLOPE_RIDGE_LAMBDA = 3.0
TARGET_RANK_GRID: tuple[int, ...] = (1, 2, 4, 8)
TARGET_RIDGE_GRID: tuple[float, ...] = (0.1, 1.0, 3.0, 10.0)
SUPPORT_BUDGETS: tuple[int, ...] = (3, 4)
MATERIAL_MEAN = 0.02
MATERIAL_MEDIAN = 0.01
MINIMUM_POSITIVE = 10
SCALE_FLOOR = 1.0e-8


def _need(condition: bool, message: str) -> None:
    v1._need(condition, message)


def _canonical_component_columns(values: np.ndarray) -> np.ndarray:
    """Fix SVD signs so both receipts and controls are deterministic."""

    output = np.asarray(values, dtype=np.float64).copy()
    for column in range(output.shape[1]):
        pivot = int(np.argmax(np.abs(output[:, column])))
        if output[pivot, column] < 0:
            output[:, column] *= -1.0
    return output


@dataclass(frozen=True)
class EndpointMap:
    """Source-frozen q=4 endpoint PCA map with an arbitrary source allowlist."""

    outer_date: str
    source_sessions: tuple[str, ...]
    mean: np.ndarray  # [7]
    scale: np.ndarray  # [7]
    components: np.ndarray  # [4,7]
    score_scale: np.ndarray  # [4]
    retained_variance: float
    source_event_count: int
    map_sha256: str

    def transform(self, events: Sequence[v1.MovementEvent]) -> np.ndarray:
        _need(bool(events), "LRT5 endpoint transform received no events")
        delta = np.stack([event.displacement for event in events]).astype(np.float64)
        z = ((delta - self.mean[None, :]) / self.scale[None, :]) @ self.components.T
        z /= self.score_scale[None, :]
        _need(z.shape == (len(events), RANK) and np.isfinite(z).all(), "LRT5 endpoint latent drift")
        return z

    def manifest(self) -> dict[str, Any]:
        return {
            "outer_date": self.outer_date,
            "source_sessions": list(self.source_sessions),
            "source_event_count": self.source_event_count,
            "retained_variance": self.retained_variance,
            "array_sha256": {
                "mean": v1.array_sha256(self.mean),
                "scale": v1.array_sha256(self.scale),
                "components": v1.array_sha256(self.components),
                "score_scale": v1.array_sha256(self.score_scale),
            },
            "map_sha256": self.map_sha256,
        }


def source_names_for_outer(outer_date: str) -> tuple[str, ...]:
    _need(outer_date in v1.H1_DATES, f"unknown outer date {outer_date}")
    return tuple(name for name in v1.H1_HELDIN_SESSIONS if v1.session_date(name) != outer_date)


def fit_endpoint_map_from_names(
    sessions: Mapping[str, v1.EventSession], *, outer_date: str, source_names: Sequence[str],
) -> EndpointMap:
    """Fit only endpoint normalization/PCA from the literal source list."""

    names = tuple(source_names)
    _need(bool(names) and len(names) == len(set(names)), "LRT5 source allowlist invalid")
    _need(set(names).issubset(sessions), "LRT5 endpoint map missing source session")
    _need(all(v1.session_date(name) != outer_date for name in names), "LRT5 outer date entered endpoint map")
    pooled = np.stack([event.displacement for name in names for event in sessions[name].events]).astype(np.float64)
    _need(pooled.shape[0] >= 2 * RANK and pooled.shape[1] == v1.POSITION_DIM,
          "LRT5 endpoint source pool underspecified")
    mean = pooled.mean(axis=0)
    scale = np.maximum(pooled.std(axis=0), SCALE_FLOOR)
    standardized = (pooled - mean[None, :]) / scale[None, :]
    _left, singular, right = np.linalg.svd(standardized, full_matrices=False)
    _need(right.shape[0] >= RANK and np.linalg.matrix_rank(standardized) >= RANK,
          "LRT5 endpoint PCA rank deficient")
    components = hse5.v1._canonicalize_component_signs(right[:RANK])
    score_scale = np.maximum((standardized @ components.T).std(axis=0), SCALE_FLOOR)
    ratio = np.square(singular) / np.square(singular).sum()
    body = {
        "protocol": PROTOCOL,
        "outer_date": outer_date,
        "source_sessions": list(names),
        "mean": v1.array_sha256(mean), "scale": v1.array_sha256(scale),
        "components": v1.array_sha256(components), "score_scale": v1.array_sha256(score_scale),
    }
    return EndpointMap(
        outer_date=outer_date, source_sessions=names, mean=np.asarray(mean, np.float64),
        scale=np.asarray(scale, np.float64), components=np.asarray(components, np.float64),
        score_scale=np.asarray(score_scale, np.float64), retained_variance=float(ratio[:RANK].sum()),
        source_event_count=int(pooled.shape[0]), map_sha256=v1.canonical_sha256(body),
    )


@dataclass(frozen=True)
class ChannelSlopeSubspace:
    """Source-frozen, zero-mean channel basis for q=4 H-SE5 slope maps."""

    outer_date: str
    source_sessions: tuple[str, ...]
    rank: int
    u: np.ndarray  # [176,r], columns orthonormal
    source_slope_mean: np.ndarray  # [4,176], receipt/strict-control only
    source_prior_slopes: np.ndarray  # [4,176], static prior control only
    source_slope_carrier_sha256: tuple[str, ...]
    source_slope_row_count: int
    subspace_sha256: str

    def manifest(self) -> dict[str, Any]:
        return {
            "outer_date": self.outer_date,
            "source_sessions": list(self.source_sessions),
            "rank": self.rank,
            "source_slope_row_count": self.source_slope_row_count,
            "source_slope_carrier_sha256": list(self.source_slope_carrier_sha256),
            "deployable_correct_carrier_uses_source_slope_mean": False,
            "array_sha256": {
                "u": v1.array_sha256(self.u),
                "source_slope_mean": v1.array_sha256(self.source_slope_mean),
                "source_prior_slopes": v1.array_sha256(self.source_prior_slopes),
            },
            "subspace_sha256": self.subspace_sha256,
        }


def _all_event_hse5_slopes(session: v1.EventSession, endpoint_map: EndpointMap) -> np.ndarray:
    z = endpoint_map.transform(session.events)
    y = np.stack([event.log_rates for event in session.events]).astype(np.float64)
    carrier = hse5.fit_carrier_arrays(z, y)
    return np.asarray(carrier[:, :RANK].T, dtype=np.float64)  # [4,176]


def fit_channel_slope_subspace(
    sessions: Mapping[str, v1.EventSession], endpoint_map: EndpointMap, *, rank: int,
) -> ChannelSlopeSubspace:
    """SVD the non-outer source H-SE5 slope rows; never deploy their mean."""

    _need(rank in TARGET_RANK_GRID, f"LRT5 rank outside literal grid: {rank}")
    names = endpoint_map.source_sessions
    _need(len(names) >= 2 and all(v1.session_date(name) != endpoint_map.outer_date for name in names),
          "LRT5 subspace source scope drift")
    maps = [_all_event_hse5_slopes(sessions[name], endpoint_map) for name in names]
    slope_rows = np.concatenate(maps, axis=0)
    _need(slope_rows.shape == (len(names) * RANK, v1.EXPECTED_NEURONS), "LRT5 source slope matrix drift")
    source_row_mean = slope_rows.mean(axis=0)
    centered = slope_rows - source_row_mean[None, :]
    _left, singular, right = np.linalg.svd(centered, full_matrices=False)
    _need(right.shape[0] >= rank and np.linalg.matrix_rank(centered) >= rank,
          "LRT5 source channel subspace rank deficient")
    u = _canonical_component_columns(right[:rank].T)
    _need(np.allclose(u.T @ u, np.eye(rank), rtol=0.0, atol=1.0e-10), "LRT5 U is not orthonormal")
    prior = np.mean(np.stack(maps, axis=0), axis=0)
    body = {
        "protocol": PROTOCOL, "outer_date": endpoint_map.outer_date,
        "source_sessions": list(names), "rank": rank,
        "u": v1.array_sha256(u), "source_slope_mean": v1.array_sha256(source_row_mean),
        "source_prior_slopes": v1.array_sha256(prior),
        "source_slope_carrier_sha256": [v1.array_sha256(np.column_stack((item.T, np.zeros(v1.EXPECTED_NEURONS)))) for item in maps],
    }
    return ChannelSlopeSubspace(
        outer_date=endpoint_map.outer_date, source_sessions=names, rank=rank, u=u,
        source_slope_mean=np.asarray(source_row_mean, np.float64), source_prior_slopes=np.asarray(prior, np.float64),
        source_slope_carrier_sha256=tuple(body["source_slope_carrier_sha256"]),
        source_slope_row_count=int(slope_rows.shape[0]), subspace_sha256=v1.canonical_sha256(body),
    )


def _centered_arrays(z: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z_value = np.asarray(z, dtype=np.float64)
    y_value = np.asarray(response, dtype=np.float64)
    _need(z_value.ndim == 2 and z_value.shape[1] == RANK and z_value.shape[0] >= CARRIER_DIM,
          "LRT5 target latent shape drift")
    _need(y_value.shape == (z_value.shape[0], v1.EXPECTED_NEURONS) and np.isfinite(y_value).all(),
          "LRT5 target response shape drift")
    z_mean = z_value.mean(axis=0)
    y_mean = y_value.mean(axis=0)
    zc, yc = z_value - z_mean[None, :], y_value - y_mean[None, :]
    _need(np.linalg.matrix_rank(zc) == RANK, "LRT5 centered target design is rank deficient")
    return zc, yc, z_mean, y_mean


def fit_lrt_carrier(
    z: np.ndarray, response: np.ndarray, subspace: ChannelSlopeSubspace, *, ridge_lambda: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Closed-form target pairing-dependent LRT5 fit, returning [176,5]."""

    _need(ridge_lambda in TARGET_RIDGE_GRID, f"LRT5 lambda outside literal grid: {ridge_lambda}")
    zc, yc, z_mean, y_mean = _centered_arrays(z, response)
    u = np.asarray(subspace.u, dtype=np.float64)
    _need(u.shape == (v1.EXPECTED_NEURONS, subspace.rank), "LRT5 U shape drift")
    penalty = np.eye(RANK, dtype=np.float64) * (zc.shape[0] * float(ridge_lambda))
    a = np.linalg.solve(zc.T @ zc + penalty, zc.T @ (yc @ u))
    slopes = a @ u.T  # [4,176]
    intercept = y_mean - z_mean @ slopes
    carrier = np.column_stack((slopes.T, intercept))
    _need(carrier.shape == (v1.EXPECTED_NEURONS, CARRIER_DIM) and np.isfinite(carrier).all(),
          "LRT5 carrier is nonfinite")
    return carrier, {
        "target_coefficient_shape": list(a.shape),
        "target_coefficient_sha256": v1.array_sha256(a),
        "target_pairing_required": True,
        "target_ridge_lambda": float(ridge_lambda),
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
    }


def carrier_from_source_prior(
    z: np.ndarray, response: np.ndarray, subspace: ChannelSlopeSubspace,
) -> np.ndarray:
    """Static-slope control with a target mean-aligned, unpaired intercept.

    The arm has no target endpoint/neural *pairing*: ``W`` is the literal
    source prior and ``b`` is its analytic target mean alignment.  Omitting
    ``mean(Z) @ W`` would make a nonzero static slope predict the wrong target
    mean and would artificially weaken this required control.
    """

    z_value = np.asarray(z, dtype=np.float64)
    y = np.asarray(response, dtype=np.float64)
    _need(z_value.ndim == 2 and z_value.shape[1] == RANK and z_value.shape[0] == y.shape[0],
          "LRT5 prior latent shape drift")
    _need(y.ndim == 2 and y.shape[1] == v1.EXPECTED_NEURONS, "LRT5 prior response shape drift")
    slopes = subspace.source_prior_slopes
    _need(slopes.shape == (RANK, v1.EXPECTED_NEURONS), "LRT5 source prior slope shape drift")
    intercept = y.mean(axis=0) - z_value.mean(axis=0) @ slopes
    carrier = np.column_stack((slopes.T, intercept))
    _need(np.isfinite(carrier).all(), "LRT5 source prior carrier nonfinite")
    return carrier


def carrier_from_intercept(response: np.ndarray) -> np.ndarray:
    y = np.asarray(response, dtype=np.float64)
    _need(y.ndim == 2 and y.shape[1] == v1.EXPECTED_NEURONS, "LRT5 intercept response shape drift")
    return np.column_stack((np.zeros((v1.EXPECTED_NEURONS, RANK), dtype=np.float64), y.mean(axis=0)))


def predict(carrier: np.ndarray, z: np.ndarray) -> np.ndarray:
    values = np.asarray(carrier, dtype=np.float64)
    latent = np.asarray(z, dtype=np.float64)
    _need(values.shape == (v1.EXPECTED_NEURONS, CARRIER_DIM) and latent.ndim == 2 and latent.shape[1] == RANK,
          "LRT5 prediction shape drift")
    return latent @ values[:, :RANK].T + values[:, RANK][None, :]


def row_shuffle_subspace(
    subspace: ChannelSlopeSubspace, *, session: str, budget: int,
) -> tuple[ChannelSlopeSubspace, dict[str, Any]]:
    """Fixed-point-free channel attachment control, followed by a target refit."""

    key = f"{v1.SHUFFLE_NAMESPACE}:lrt5-u-row:{session}:M{budget}:r{subspace.rank}"
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little"))
    identity = np.arange(v1.EXPECTED_NEURONS, dtype=np.int64)
    order = identity.copy()
    for _ in range(100):
        rng.shuffle(order)
        if not np.any(order == identity):
            break
    _need(not np.any(order == identity), "LRT5 U row shuffle retained a channel attachment")
    u = subspace.u[order]
    body = {
        "protocol": PROTOCOL, "parent_subspace_sha256": subspace.subspace_sha256,
        "session": session, "budget": budget, "order": v1.array_sha256(order), "u": v1.array_sha256(u),
    }
    shuffled = ChannelSlopeSubspace(
        outer_date=subspace.outer_date, source_sessions=subspace.source_sessions, rank=subspace.rank,
        u=u, source_slope_mean=subspace.source_slope_mean[order], source_prior_slopes=subspace.source_prior_slopes[:, order],
        source_slope_carrier_sha256=subspace.source_slope_carrier_sha256,
        source_slope_row_count=subspace.source_slope_row_count, subspace_sha256=v1.canonical_sha256(body),
    )
    return shuffled, {
        "namespace": "h1-event-carrier-lrt5-u-row-shuffle-v1", "session": session, "budget": budget,
        "order_sha256": v1.array_sha256(order), "fixed_points": 0,
        "parent_subspace_sha256": subspace.subspace_sha256,
    }


def _session_arrays(
    session: v1.EventSession, endpoint_map: EndpointMap, *, budget: int,
) -> tuple[tuple[v1.MovementEvent, ...], tuple[v1.MovementEvent, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    support = session.events_before(budget)
    later = session.events_after(budget)
    _need(len(support) >= CARRIER_DIM and len(later) >= 4, f"{session.session_name}: LRT5 support/query underspecified")
    z_support = endpoint_map.transform(support)
    y_support = np.stack([event.log_rates for event in support]).astype(np.float64)
    z_later = endpoint_map.transform(later)
    y_later = np.stack([event.log_rates for event in later]).astype(np.float64)
    return support, later, z_support, y_support, z_later, y_later


def evaluate_lrt_session(
    session: v1.EventSession, endpoint_map: EndpointMap, subspace: ChannelSlopeSubspace, *, budget: int, ridge_lambda: float,
) -> dict[str, Any]:
    """Evaluate the correct model and every pairing/static control on later events."""

    _need(endpoint_map.outer_date == session.date == subspace.outer_date,
          "LRT5 outer-date scope mismatch")
    support, later, z_support, y_support, z_later, y_later = _session_arrays(session, endpoint_map, budget=budget)
    correct, fit = fit_lrt_carrier(z_support, y_support, subspace, ridge_lambda=ridge_lambda)
    order, label_manifest = v1.within_trial_label_shuffle(support, session=session.session_name, budget=budget)
    label, label_fit = fit_lrt_carrier(z_support[order], y_support, subspace, ridge_lambda=ridge_lambda)
    shuffled_u, row_manifest = row_shuffle_subspace(subspace, session=session.session_name, budget=budget)
    row, row_fit = fit_lrt_carrier(z_support, y_support, shuffled_u, ridge_lambda=ridge_lambda)
    intercept = carrier_from_intercept(y_support)
    prior = carrier_from_source_prior(z_support, y_support, subspace)
    r_correct = v1.r2_by_channel(y_later, predict(correct, z_later))
    r_label = v1.r2_by_channel(y_later, predict(label, z_later))
    r_row = v1.r2_by_channel(y_later, predict(row, z_later))
    r_intercept = v1.r2_by_channel(y_later, predict(intercept, z_later))
    r_prior = v1.r2_by_channel(y_later, predict(prior, z_later))
    defined = np.isfinite(r_correct) & np.isfinite(r_label) & np.isfinite(r_row) & np.isfinite(r_intercept) & np.isfinite(r_prior)
    _need(np.any(defined), f"{session.session_name}: LRT5 lacks finite forward channels")
    return {
        "session": session.session_name, "date": session.date, "budget": budget,
        "support_events": len(support), "later_events": len(later),
        "centered_design_rank": int(np.linalg.matrix_rank(z_support - z_support.mean(axis=0, keepdims=True))),
        "carrier_dim": CARRIER_DIM, "subspace_rank": subspace.rank, "target_ridge_lambda": ridge_lambda,
        "defined_channels": int(defined.sum()),
        "median_r2_correct": float(np.median(r_correct[defined])),
        "median_r2_label_shuffle": float(np.median(r_label[defined])),
        "median_r2_u_row_shuffle": float(np.median(r_row[defined])),
        "median_r2_intercept": float(np.median(r_intercept[defined])),
        "median_r2_source_prior": float(np.median(r_prior[defined])),
        "median_delta_label_shuffle": float(np.median((r_correct - r_label)[defined])),
        "median_delta_u_row_shuffle": float(np.median((r_correct - r_row)[defined])),
        "median_delta_intercept": float(np.median((r_correct - r_intercept)[defined])),
        "median_delta_source_prior": float(np.median((r_correct - r_prior)[defined])),
        "correct_carrier_sha256": v1.array_sha256(correct),
        "label_carrier_sha256": v1.array_sha256(label),
        "u_row_carrier_sha256": v1.array_sha256(row),
        "intercept_carrier_sha256": v1.array_sha256(intercept),
        "source_prior_carrier_sha256": v1.array_sha256(prior),
        "correct_fit": fit, "label_fit": label_fit, "u_row_fit": row_fit,
        "label_shuffle": label_manifest, "u_row_shuffle": row_manifest,
        "outer_future_used_only_for_scoring": True,
    }


def hse5_baseline_session(
    session: v1.EventSession, endpoint_map: EndpointMap, *, budget: int,
) -> dict[str, Any]:
    """Use the bound V2 H-SE5 codepath for exact source-receipt reproduction."""

    support, later, z_support, y_support, z_later, y_later = _session_arrays(session, endpoint_map, budget=budget)
    carrier = hse5.fit_carrier_arrays(z_support, y_support)
    order, _manifest = v1.within_trial_label_shuffle(support, session=session.session_name, budget=budget)
    label = hse5.fit_carrier_arrays(z_support[order], y_support)
    r_correct = v1.r2_by_channel(y_later, hse5.predict(carrier, z_later))
    r_label = v1.r2_by_channel(y_later, hse5.predict(label, z_later))
    r_intercept = v1.r2_by_channel(y_later, np.broadcast_to(y_support.mean(axis=0), y_later.shape))
    defined = np.isfinite(r_correct) & np.isfinite(r_label) & np.isfinite(r_intercept)
    _need(np.any(defined), f"{session.session_name}: H-SE5 reference lacked finite channels")
    return {
        "median_r2_correct": float(np.median(r_correct[defined])),
        "median_delta_label_shuffle": float(np.median((r_correct - r_label)[defined])),
        "median_delta_intercept": float(np.median((r_correct - r_intercept)[defined])),
        "carrier_sha256": v1.array_sha256(carrier),
    }


@dataclass(frozen=True)
class HyperparameterSelection:
    outer_date: str
    budget: int
    selected_rank: int
    selected_ridge_lambda: float
    grid: Mapping[str, Mapping[str, Any]]
    selection_sha256: str

    def manifest(self) -> dict[str, Any]:
        return {
            "outer_date": self.outer_date, "budget": self.budget,
            "selected_rank": self.selected_rank, "selected_ridge_lambda": self.selected_ridge_lambda,
            "selection_scope": "inner source-date LODO only; outer date absent from every inner map/subspace/score",
            "grid": self.grid, "selection_sha256": self.selection_sha256,
        }


def select_hyperparameters_source_lodo(
    sessions: Mapping[str, v1.EventSession], *, outer_date: str, budget: int,
) -> HyperparameterSelection:
    """Select rank/lambda using only future-within-source dates, never outer data."""

    _need(budget in SUPPORT_BUDGETS, f"LRT5 unsupported budget {budget}")
    outer_source = source_names_for_outer(outer_date)
    inner_dates = tuple(date for date in v1.H1_DATES if date != outer_date)
    grid: dict[str, dict[str, Any]] = {}
    # Cache the fully source-only maps/subspaces once per inner-date/rank.
    cache: dict[tuple[str, int], tuple[EndpointMap, ChannelSlopeSubspace, tuple[str, ...]]] = {}
    for inner_date in inner_dates:
        train_names = tuple(name for name in outer_source if v1.session_date(name) != inner_date)
        validation_names = tuple(name for name in outer_source if v1.session_date(name) == inner_date)
        _need(bool(train_names) and bool(validation_names), "LRT5 inner date split invalid")
        endpoint_map = fit_endpoint_map_from_names(sessions, outer_date=inner_date, source_names=train_names)
        for rank in TARGET_RANK_GRID:
            cache[(inner_date, rank)] = (endpoint_map, fit_channel_slope_subspace(sessions, endpoint_map, rank=rank), validation_names)
    for rank in TARGET_RANK_GRID:
        for ridge_lambda in TARGET_RIDGE_GRID:
            rows: list[tuple[str, float]] = []
            inner_manifest: dict[str, Any] = {}
            for inner_date in inner_dates:
                endpoint_map, subspace, validation_names = cache[(inner_date, rank)]
                score_rows: dict[str, float] = {}
                for name in validation_names:
                    session = sessions[name]
                    _support, _later, z_support, y_support, z_later, y_later = _session_arrays(session, endpoint_map, budget=budget)
                    carrier, _fit = fit_lrt_carrier(z_support, y_support, subspace, ridge_lambda=ridge_lambda)
                    r2 = v1.r2_by_channel(y_later, predict(carrier, z_later))
                    value = float(np.median(r2[np.isfinite(r2)]))
                    rows.append((name, value)); score_rows[name] = value
                inner_manifest[inner_date] = {
                    "train_source_sessions": list(endpoint_map.source_sessions),
                    "validation_source_sessions": list(validation_names),
                    "endpoint_map_sha256": endpoint_map.map_sha256,
                    "subspace_sha256": subspace.subspace_sha256,
                    "session_median_r2": score_rows,
                }
            summary = v1.paired_summary(rows)
            key = f"r{rank}_lambda{ridge_lambda:g}"
            grid[key] = {
                "rank": rank, "ridge_lambda": ridge_lambda,
                "inner_source_date_lodo_median_r2": summary,
                "inner_folds": inner_manifest,
            }
    # Fixed descending performance, then simpler model/lower ridge tie break.
    candidates = list(grid.values())
    best = min(candidates, key=lambda row: (
        -float(row["inner_source_date_lodo_median_r2"]["mean"]),
        -float(row["inner_source_date_lodo_median_r2"]["median"]),
        int(row["rank"]), float(row["ridge_lambda"]),
    ))
    body = {
        "protocol": PROTOCOL, "outer_date": outer_date, "budget": budget,
        "grid": grid, "selected_rank": best["rank"], "selected_ridge_lambda": best["ridge_lambda"],
        "tie_break": "maximize inner mean, then median; then smaller rank, then smaller lambda",
    }
    return HyperparameterSelection(
        outer_date=outer_date, budget=budget, selected_rank=int(best["rank"]),
        selected_ridge_lambda=float(best["ridge_lambda"]), grid=grid,
        selection_sha256=v1.canonical_sha256(body),
    )


def _summary(rows: Mapping[str, Mapping[str, Any]], field: str) -> dict[str, Any]:
    return v1.paired_summary([(name, row.get(field)) for name, row in rows.items()])


def _positive(summary: Mapping[str, Any]) -> bool:
    return bool(summary["defined_sessions"] == 13 and summary["mean"] > 0 and summary["median"] > 0
                and summary["positive"] >= MINIMUM_POSITIVE and summary["leave_largest_absolute_out_mean"] > 0)


def aggregate(rows: Mapping[str, Mapping[str, Any]], baseline: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    augmented = {name: {**row, "delta_vs_hse5": float(row["median_r2_correct"] - baseline[name]["median_r2_correct"])}
                 for name, row in rows.items()}
    return {
        "correct_r2": _summary(augmented, "median_r2_correct"),
        "correct_minus_hse5": _summary(augmented, "delta_vs_hse5"),
        "correct_minus_label_shuffle": _summary(augmented, "median_delta_label_shuffle"),
        "correct_minus_u_row_shuffle": _summary(augmented, "median_delta_u_row_shuffle"),
        "correct_minus_intercept": _summary(augmented, "median_delta_intercept"),
        "correct_minus_source_prior": _summary(augmented, "median_delta_source_prior"),
        "support_event_count": {
            "minimum": min(int(row["support_events"]) for row in rows.values()),
            "median": float(np.median([row["support_events"] for row in rows.values()])),
            "maximum": max(int(row["support_events"]) for row in rows.values()),
        },
    }


def gate(aggregate_value: Mapping[str, Any]) -> dict[str, Any]:
    delta = aggregate_value["correct_minus_hse5"]
    material = bool(_positive(delta) and delta["mean"] >= MATERIAL_MEAN and delta["median"] >= MATERIAL_MEDIAN)
    controls = {key: _positive(aggregate_value[key]) for key in (
        "correct_minus_label_shuffle", "correct_minus_u_row_shuffle", "correct_minus_intercept", "correct_minus_source_prior",
    )}
    return {
        "passed": bool(material and all(controls.values())),
        "material_gain_vs_hse5": material,
        **controls,
        "thresholds": {
            "mean_delta_vs_hse5": MATERIAL_MEAN, "median_delta_vs_hse5": MATERIAL_MEDIAN,
            "minimum_positive_sessions": MINIMUM_POSITIVE, "leave_largest_absolute_out_mean_positive": True,
        },
    }


def run_screen(sessions: Mapping[str, v1.EventSession]) -> dict[str, Any]:
    _need(tuple(sessions) == v1.H1_HELDIN_SESSIONS, "LRT5 source session order/allowlist drift")
    budgets: dict[str, Any] = {}
    for budget in SUPPORT_BUDGETS:
        selections = {date: select_hyperparameters_source_lodo(sessions, outer_date=date, budget=budget)
                      for date in v1.H1_DATES}
        maps = {date: fit_endpoint_map_from_names(sessions, outer_date=date, source_names=source_names_for_outer(date))
                for date in v1.H1_DATES}
        subspaces = {date: fit_channel_slope_subspace(sessions, maps[date], rank=selections[date].selected_rank)
                     for date in v1.H1_DATES}
        rows: dict[str, Any] = {}
        baseline: dict[str, Any] = {}
        for name in v1.H1_HELDIN_SESSIONS:
            date = v1.session_date(name)
            rows[name] = evaluate_lrt_session(
                sessions[name], maps[date], subspaces[date], budget=budget,
                ridge_lambda=selections[date].selected_ridge_lambda,
            )
            baseline[name] = hse5_baseline_session(sessions[name], maps[date], budget=budget)
        combined = aggregate(rows, baseline)
        budgets[f"M{budget}"] = {
            "budget_trials": budget,
            "selection_by_outer_date": {date: selections[date].manifest() for date in v1.H1_DATES},
            "endpoint_map_by_outer_date": {date: maps[date].manifest() for date in v1.H1_DATES},
            "subspace_by_outer_date": {date: subspaces[date].manifest() for date in v1.H1_DATES},
            "hse5_baseline_sessions": baseline,
            "lrt5_sessions": rows,
            "aggregate": combined,
            "gate": gate(combined),
        }
    passed = all(budgets[f"M{budget}"]["gate"]["passed"] for budget in SUPPORT_BUDGETS)
    return {
        "schema": SCHEMA, "protocol": PROTOCOL,
        "candidate_matrix_predeclared_before_data_run": True,
        "frozen_constants": {
            "endpoint_representation": "H-SE5 q4 endpoint-displacement PCA",
            "carrier_dim": CARRIER_DIM, "rank_grid": list(TARGET_RANK_GRID),
            "target_ridge_grid": list(TARGET_RIDGE_GRID), "source_slope_ridge_lambda": SOURCE_SLOPE_RIDGE_LAMBDA,
            "support_budgets": list(SUPPORT_BUDGETS),
            "selection_rule": "inner source-date LODO maximize mean then median; lower rank/lambda ties",
            "correct_slope_form": "W=A@U.T; source slope mean removed for SVD and never added to correct carrier",
        },
        "budgets": budgets,
        "status": "PASS_CPU_LRT5_MATERIAL" if passed else "STOP_CPU_LRT5_NOT_MATERIAL",
        "gpu_authorized_by_this_screen": False,
        "scope": {
            "public_held_in_calibration_nwbs_opened": 13,
            "minival_nwbs_opened": 0, "held_out_nwbs_opened": 0, "formal_test_labels_opened": 0,
            "dense_velocity_opened": False, "target_session_optimizer_steps": 0,
            "target_session_backward_steps": 0, "decoder_constructed": False, "trainer_constructed": False,
            "cuda_used": False,
        },
        "interpretation": {
            "screen_is_source_only_development": True,
            "passing_is_necessary_not_sufficient_for_a_new_gpu_arm": True,
            "correct_lrt5_has_no_deployed_static_source_slope_mean": True,
            "source_channel_subspace_requires_pairing_controls_due_to_channel_index_structure": True,
        },
    }
