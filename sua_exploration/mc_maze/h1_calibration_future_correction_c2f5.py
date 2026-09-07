"""Source-only calibration-to-future correction (C2F5) for H1 event carriers.

This is deliberately a *CPU screen*, not a new decoder or GPU path.  The
deployment object remains the five-wide H-SE5 endpoint carrier
``[w1, w2, w3, w4, b]``.  C2F5 asks a tightly bounded question:

    Can source sessions teach one channel-shared, analytic operator to correct
    a target session's low-budget H-SE5 sufficient statistics toward the
    carrier that would be fitted from that session's later events?

For a target session the operator sees only calibration-support quantities:
the raw H-SE5 carrier and analytic ridge-quality statistics.  It never sees
later target labels, a channel index, dense velocity, a neural-network
optimizer, or a backward pass.  In the outer source-date LODO procedure,
later events are used only (i) as a *teacher* in source sessions and (ii) to
score the held-out source date.  They never enter its fitted correction map.

The primary operator is a residual affine ridge map,
``B_corrected = B_support + F([B_support, quality])``.  A matched
``quality_only`` control replaces ``[B_support, quality]`` by ``quality`` in
``F`` but retains the identical raw H-SE5 base.  Consequently it can expose a
spurious source prior or quality-statistic effect without granting the
operator access to the target carrier values themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as v2


SCHEMA = "h1_calibration_future_correction_c2f5_source_screen_v1"
PROTOCOL = "h1_c2f5_shared_residual_operator_source_lodo_20260812_v1"
RANK = 4
CARRIER_DIM = 5
QUALITY_NAMES: tuple[str, ...] = (
    "log_ridge_covariance_w1",
    "log_ridge_covariance_w2",
    "log_ridge_covariance_w3",
    "log_ridge_covariance_w4",
    "log_ridge_covariance_b",
    "log_residual_variance",
    "log_response_variance",
)
QUALITY_DIM = len(QUALITY_NAMES)
QUALITY_FLOOR = 1.0e-12
SCALE_FLOOR = 1.0e-8
RESIDUAL_SCALE_FLOOR = 1.0e-8

# This grid and its ordered small-lambda tie break are frozen before any data
# are loaded by the runner.  The grid controls only the source-fitted shared
# residual operator; it is not a target-session calibration hyperparameter.
C2F_RIDGE_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
SELECTION_TIE_TOLERANCE = 1.0e-12
SUPPORT_BUDGETS: tuple[int, ...] = (3, 4)
PRIMARY_INPUT_KIND = "carrier_plus_quality"
QUALITY_ONLY_INPUT_KIND = "quality_only_no_carrier"

MATERIAL_MEAN_DELTA = 0.02
MATERIAL_MEDIAN_DELTA = 0.01
MIN_POSITIVE_SESSIONS = 10


def _need(condition: bool, message: str) -> None:
    v1._need(condition, message)


def _design(z: np.ndarray) -> np.ndarray:
    values = np.asarray(z, dtype=np.float64)
    _need(values.ndim == 2 and values.shape[1] == RANK and values.shape[0] >= CARRIER_DIM,
          f"C2F5 latent design must be [events,{RANK}], got {values.shape}")
    design = np.column_stack((np.ones(values.shape[0], dtype=np.float64), values))
    _need(np.linalg.matrix_rank(design) == CARRIER_DIM, "C2F5 ridge design is rank deficient")
    return design


def _carrier_to_beta(carrier: np.ndarray) -> np.ndarray:
    values = np.asarray(carrier, dtype=np.float64)
    _need(values.shape == (v1.EXPECTED_NEURONS, CARRIER_DIM), "C2F5 carrier shape drift")
    return np.vstack((values[:, -1], values[:, :RANK].T))


def _beta_to_carrier(beta: np.ndarray) -> np.ndarray:
    values = np.asarray(beta, dtype=np.float64)
    _need(values.shape == (CARRIER_DIM, v1.EXPECTED_NEURONS), "C2F5 beta shape drift")
    return np.column_stack((values[1:].T, values[0]))


def _exact_hse5_carrier_and_quality(z: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return the exact V2 carrier plus target-legal analytic quality rows.

    Carrier fitting dispatches directly to V2, which is the byte-bound H-SE5
    reference implementation.  The extra variance computation only produces
    quality features; it cannot alter the raw baseline carrier.
    """

    x = _design(z)
    y = np.asarray(response, dtype=np.float64)
    _need(y.shape == (x.shape[0], v1.EXPECTED_NEURONS) and np.isfinite(y).all(),
          "C2F5 response shape/nonfinite drift")
    carrier = v2.fit_carrier_arrays(z, y)
    beta = _carrier_to_beta(carrier)
    penalty = np.diag([0.0] + [1.0] * RANK) * (x.shape[0] * v2.RIDGE_LAMBDA)
    system = x.T @ x + penalty
    inverse = np.linalg.inv(system)
    residual = y - x @ beta
    effective_df = float(np.trace((x.T @ x) @ inverse))
    residual_df = float(x.shape[0]) - effective_df
    _need(np.isfinite(residual_df) and residual_df > 1.0e-8,
          f"C2F5 residual degrees of freedom invalid: {residual_df}")
    residual_variance = np.maximum(np.sum(np.square(residual), axis=0) / residual_df, QUALITY_FLOOR)
    response_variance = np.maximum(np.var(y, axis=0), QUALITY_FLOOR)
    covariance_kernel = inverse @ (x.T @ x) @ inverse
    beta_covariance = np.maximum(np.diag(covariance_kernel)[:, None] * residual_variance[None, :], QUALITY_FLOOR)
    # Carrier order is slopes then intercept, whereas beta order is intercept
    # then slopes.  Quality rows mirror carrier order exactly.
    carrier_covariance = np.vstack((beta_covariance[1:], beta_covariance[:1])).T
    quality = np.column_stack((
        np.log(carrier_covariance),
        np.log(residual_variance),
        np.log(response_variance),
    ))
    _need(quality.shape == (v1.EXPECTED_NEURONS, QUALITY_DIM) and np.isfinite(quality).all(),
          "C2F5 quality shape/nonfinite drift")
    return carrier, quality, {
        "support_events": int(x.shape[0]),
        "design_rank": int(np.linalg.matrix_rank(x)),
        "effective_residual_degrees_of_freedom": residual_df,
        "ridge_lambda": v2.RIDGE_LAMBDA,
        "quality_names": list(QUALITY_NAMES),
        "carrier_sha256": v1.array_sha256(carrier),
        "quality_sha256": v1.array_sha256(quality),
    }


def _events_before(session: v1.EventSession, budget: int) -> tuple[v1.MovementEvent, ...]:
    return v2.select_trial_range(session, start_index=0, budget=budget)


def _events_after(session: v1.EventSession, budget: int) -> tuple[v1.MovementEvent, ...]:
    _need(budget in SUPPORT_BUDGETS, f"C2F5 unsupported budget M={budget}")
    return tuple(event for event in session.events if event.trial_index >= budget)


def fit_support_sufficient_statistics(
    session: v1.EventSession,
    basis: v2.EndpointBasisV2,
    *,
    budget: int,
    shuffled_labels: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit target-legal H-SE5 sufficient statistics from support events only."""

    support = _events_before(session, budget)
    z, response = v2.event_arrays(support, basis)
    shuffle: Mapping[str, Any] | None = None
    if shuffled_labels:
        order, shuffle = v1.within_trial_label_shuffle(support, session=session.session_name, budget=budget)
        z = z[order]
    carrier, quality, fit = _exact_hse5_carrier_and_quality(z, response)
    return carrier, quality, {
        **fit,
        "budget": budget,
        "support_events": len(support),
        "shuffle": shuffle,
    }


def fit_future_teacher(
    session: v1.EventSession, basis: v2.EndpointBasisV2, *, budget: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a source-only future-event *teacher*; never call on an outer target."""

    later = _events_after(session, budget)
    _need(len(later) >= CARRIER_DIM, f"{session.session_name}: C2F5 later teacher has too few events")
    z, response = v2.event_arrays(later, basis)
    carrier, _quality, fit = _exact_hse5_carrier_and_quality(z, response)
    return carrier, {
        **fit,
        "later_events": len(later),
        "teacher_role": "source_session_future_events_only",
    }


@dataclass(frozen=True)
class SourceExamples:
    """Source-session rows for fitting a channel-shared residual operator."""

    session_names: tuple[str, ...]
    outer_date: str
    budget: int
    support_carriers: np.ndarray  # [sessions, channels, 5]
    quality: np.ndarray  # [sessions, channels, 7]
    future_teachers: np.ndarray  # [sessions, channels, 5]
    manifests: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        count = len(self.session_names)
        _need(count >= 1 and self.support_carriers.shape == (count, v1.EXPECTED_NEURONS, CARRIER_DIM),
              "C2F5 source support carrier shape drift")
        _need(self.quality.shape == (count, v1.EXPECTED_NEURONS, QUALITY_DIM),
              "C2F5 source quality shape drift")
        _need(self.future_teachers.shape == self.support_carriers.shape,
              "C2F5 source future teacher shape drift")
        _need(len(self.manifests) == count and np.isfinite(self.support_carriers).all()
              and np.isfinite(self.quality).all() and np.isfinite(self.future_teachers).all(),
              "C2F5 source values are nonfinite")

    @property
    def residuals(self) -> np.ndarray:
        return self.future_teachers - self.support_carriers

    def rows(self, input_kind: str) -> tuple[np.ndarray, np.ndarray]:
        if input_kind == PRIMARY_INPUT_KIND:
            features = np.concatenate((self.support_carriers, self.quality), axis=-1)
        elif input_kind == QUALITY_ONLY_INPUT_KIND:
            features = self.quality
        else:
            raise v1.SparseEventEndpointError(f"unknown C2F5 input kind {input_kind}")
        return features.reshape(-1, features.shape[-1]), self.residuals.reshape(-1, CARRIER_DIM)


def build_source_examples(
    sessions: Mapping[str, v1.EventSession],
    basis: v2.EndpointBasisV2,
    *,
    source_names: Sequence[str],
    outer_date: str,
    budget: int,
) -> SourceExamples:
    """Build teacher examples exclusively from declared source sessions."""

    names = tuple(source_names)
    _need(bool(names) and len(set(names)) == len(names), "C2F5 source names are empty or duplicated")
    _need(all(v1.session_date(name) != outer_date for name in names),
          "C2F5 outer date leaked into source operator examples")
    support: list[np.ndarray] = []
    quality: list[np.ndarray] = []
    future: list[np.ndarray] = []
    manifests: list[Mapping[str, Any]] = []
    for name in names:
        _need(name in sessions, f"C2F5 source session missing: {name}")
        b_support, q, support_fit = fit_support_sufficient_statistics(sessions[name], basis, budget=budget)
        b_future, future_fit = fit_future_teacher(sessions[name], basis, budget=budget)
        support.append(b_support)
        quality.append(q)
        future.append(b_future)
        manifests.append({
            "session": name,
            "date": v1.session_date(name),
            "support_fit": support_fit,
            "future_teacher_fit": future_fit,
            "coefficient_shift_l2_mean": float(np.mean(np.linalg.norm(b_future - b_support, axis=1))),
            "coefficient_shift_l2_median": float(np.median(np.linalg.norm(b_future - b_support, axis=1))),
        })
    return SourceExamples(
        session_names=names,
        outer_date=outer_date,
        budget=budget,
        support_carriers=np.stack(support, axis=0),
        quality=np.stack(quality, axis=0),
        future_teachers=np.stack(future, axis=0),
        manifests=tuple(manifests),
    )


@dataclass(frozen=True)
class C2FOperator:
    """A small source-fitted affine ridge map shared by all 176 channels."""

    input_kind: str
    ridge_lambda: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    residual_mean: np.ndarray
    residual_scale: np.ndarray
    standardized_coefficients: np.ndarray  # [input_dim+1, 5], intercept first
    source_sessions: tuple[str, ...]
    outer_date: str
    budget: int
    operator_sha256: str

    @property
    def input_dim(self) -> int:
        return int(self.feature_mean.size)

    def _features(self, carrier: np.ndarray, quality: np.ndarray) -> np.ndarray:
        b = np.asarray(carrier, dtype=np.float64)
        q = np.asarray(quality, dtype=np.float64)
        _need(b.shape == (v1.EXPECTED_NEURONS, CARRIER_DIM), "C2F5 operator carrier input drift")
        _need(q.shape == (v1.EXPECTED_NEURONS, QUALITY_DIM), "C2F5 operator quality input drift")
        if self.input_kind == PRIMARY_INPUT_KIND:
            result = np.concatenate((b, q), axis=1)
        elif self.input_kind == QUALITY_ONLY_INPUT_KIND:
            result = q
        else:  # defensive: serialized operator should never be malformed.
            raise v1.SparseEventEndpointError(f"C2F5 unknown operator input kind {self.input_kind}")
        _need(result.shape == (v1.EXPECTED_NEURONS, self.input_dim), "C2F5 operator feature width drift")
        return result

    def predict_residual(self, carrier: np.ndarray, quality: np.ndarray) -> np.ndarray:
        features = self._features(carrier, quality)
        standardized = (features - self.feature_mean[None, :]) / self.feature_scale[None, :]
        design = np.column_stack((np.ones(len(standardized), dtype=np.float64), standardized))
        residual = (design @ self.standardized_coefficients) * self.residual_scale[None, :] + self.residual_mean[None, :]
        _need(residual.shape == (v1.EXPECTED_NEURONS, CARRIER_DIM) and np.isfinite(residual).all(),
              "C2F5 operator residual is invalid")
        return residual

    def apply(self, carrier: np.ndarray, quality: np.ndarray) -> np.ndarray:
        corrected = np.asarray(carrier, dtype=np.float64) + self.predict_residual(carrier, quality)
        _need(corrected.shape == (v1.EXPECTED_NEURONS, CARRIER_DIM) and np.isfinite(corrected).all(),
              "C2F5 corrected carrier is invalid")
        return corrected

    def manifest(self) -> dict[str, Any]:
        return {
            "input_kind": self.input_kind,
            "ridge_lambda": self.ridge_lambda,
            "input_dim": self.input_dim,
            "carrier_dim": CARRIER_DIM,
            "quality_dim": QUALITY_DIM,
            "source_sessions": list(self.source_sessions),
            "outer_date": self.outer_date,
            "budget": self.budget,
            "has_channel_index_feature": False,
            "target_session_later_labels_enter_operator_fit": False,
            "array_sha256": {
                "feature_mean": v1.array_sha256(self.feature_mean),
                "feature_scale": v1.array_sha256(self.feature_scale),
                "residual_mean": v1.array_sha256(self.residual_mean),
                "residual_scale": v1.array_sha256(self.residual_scale),
                "standardized_coefficients": v1.array_sha256(self.standardized_coefficients),
            },
            "operator_sha256": self.operator_sha256,
        }


def fit_operator(examples: SourceExamples, *, input_kind: str, ridge_lambda: float) -> C2FOperator:
    """Fit a source-only, channel-shared standardised residual ridge operator."""

    _need(ridge_lambda in C2F_RIDGE_GRID, "C2F5 operator lambda outside frozen grid")
    features, targets = examples.rows(input_kind)
    _need(features.ndim == 2 and targets.shape == (features.shape[0], CARRIER_DIM), "C2F5 operator row shape drift")
    feature_mean = features.mean(axis=0)
    feature_scale = np.maximum(features.std(axis=0), SCALE_FLOOR)
    target_mean = targets.mean(axis=0)
    target_scale = np.maximum(targets.std(axis=0), RESIDUAL_SCALE_FLOOR)
    x = (features - feature_mean[None, :]) / feature_scale[None, :]
    y = (targets - target_mean[None, :]) / target_scale[None, :]
    design = np.column_stack((np.ones(len(x), dtype=np.float64), x))
    penalty = np.diag([0.0] + [1.0] * x.shape[1]) * (len(x) * float(ridge_lambda))
    coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    _need(np.isfinite(coefficient).all(), "C2F5 operator coefficient is nonfinite")
    body = {
        "protocol": PROTOCOL,
        "input_kind": input_kind,
        "ridge_lambda": ridge_lambda,
        "outer_date": examples.outer_date,
        "budget": examples.budget,
        "source_sessions": list(examples.session_names),
        "feature_mean": v1.array_sha256(feature_mean),
        "feature_scale": v1.array_sha256(feature_scale),
        "residual_mean": v1.array_sha256(target_mean),
        "residual_scale": v1.array_sha256(target_scale),
        "standardized_coefficients": v1.array_sha256(coefficient),
    }
    return C2FOperator(
        input_kind=input_kind,
        ridge_lambda=float(ridge_lambda),
        feature_mean=np.asarray(feature_mean, np.float64),
        feature_scale=np.asarray(feature_scale, np.float64),
        residual_mean=np.asarray(target_mean, np.float64),
        residual_scale=np.asarray(target_scale, np.float64),
        standardized_coefficients=np.asarray(coefficient, np.float64),
        source_sessions=examples.session_names,
        outer_date=examples.outer_date,
        budget=examples.budget,
        operator_sha256=v1.canonical_sha256(body),
    )


def row_shuffle_carrier(carrier: np.ndarray, *, session: str, budget: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Deterministic fixed-point-free channel attachment control for C2F5."""

    values = np.asarray(carrier, dtype=np.float64)
    _need(values.shape == (v1.EXPECTED_NEURONS, CARRIER_DIM), "C2F5 row shuffle carrier shape drift")
    key = f"{v1.SHUFFLE_NAMESPACE}:c2f5-row:{session}:M{budget}"
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little"))
    identity = np.arange(v1.EXPECTED_NEURONS, dtype=np.int64)
    order = identity.copy()
    for _ in range(100):
        rng.shuffle(order)
        if not np.any(order == identity):
            break
    _need(not np.any(order == identity), "C2F5 row shuffle retained fixed points")
    return values[order], {
        "namespace": "h1-c2f5-row-shuffle-v1",
        "session": session,
        "budget": budget,
        "fixed_points": 0,
        "order_sha256": v1.array_sha256(order),
    }


def _source_names_for_outer(outer_date: str) -> tuple[str, ...]:
    _need(outer_date in v1.H1_DATES, "C2F5 invalid outer date")
    return tuple(name for name in v1.H1_HELDIN_SESSIONS if v1.session_date(name) != outer_date)


def _median_channel_r2(
    observed: np.ndarray, carrier: np.ndarray, z_later: np.ndarray,
) -> float:
    r2 = v1.r2_by_channel(observed, v2.predict(carrier, z_later))
    finite = r2[np.isfinite(r2)]
    _need(finite.size > 0, "C2F5 has no defined later-event channels")
    return float(np.median(finite))


def _source_inner_validation(
    sessions: Mapping[str, v1.EventSession],
    basis: v2.EndpointBasisV2,
    *,
    outer_date: str,
    budget: int,
    ridge_lambda: float,
) -> dict[str, Any]:
    """Choose map strength with source-date LODO, with outer date absent."""

    source_dates = tuple(date for date in v1.H1_DATES if date != outer_date)
    rows: list[dict[str, Any]] = []
    for inner_date in source_dates:
        train_names = tuple(name for name in v1.H1_HELDIN_SESSIONS
                            if v1.session_date(name) not in {outer_date, inner_date})
        validation_names = tuple(name for name in v1.H1_HELDIN_SESSIONS if v1.session_date(name) == inner_date)
        _need(train_names and validation_names, "C2F5 inner source-date LODO is underspecified")
        examples = build_source_examples(
            sessions, basis, source_names=train_names, outer_date=outer_date, budget=budget,
        )
        operator = fit_operator(examples, input_kind=PRIMARY_INPUT_KIND, ridge_lambda=ridge_lambda)
        for name in validation_names:
            support, quality, _fit = fit_support_sufficient_statistics(sessions[name], basis, budget=budget)
            later = _events_after(sessions[name], budget)
            z_later, observed = v2.event_arrays(later, basis)
            corrected = operator.apply(support, quality)
            rows.append({
                "session": name,
                "inner_validation_date": inner_date,
                "outer_date": outer_date,
                "operator_source_dates": sorted({v1.session_date(source) for source in train_names}),
                "median_later_event_raw_log_rate_r2": _median_channel_r2(observed, corrected, z_later),
                "operator_sha256": operator.operator_sha256,
            })
    values = np.asarray([row["median_later_event_raw_log_rate_r2"] for row in rows], dtype=np.float64)
    _need(values.size == len(_source_names_for_outer(outer_date)) and np.isfinite(values).all(),
          "C2F5 inner validation score coverage drift")
    return {
        "ridge_lambda": ridge_lambda,
        "outer_date": outer_date,
        "inner_source_dates": list(source_dates),
        "selection_metric": "mean_per_recording_median_channel_later_event_raw_log_rate_r2",
        "rows": rows,
        "mean_score": float(values.mean()),
        "median_score": float(np.median(values)),
        "defined_recordings": int(values.size),
        "outer_date_visible_to_operator_training": False,
    }


def select_operator_lambda(
    sessions: Mapping[str, v1.EventSession],
    basis: v2.EndpointBasisV2,
    *,
    outer_date: str,
    budget: int,
) -> tuple[float, list[dict[str, Any]]]:
    """Source-only inner-date LODO selection with frozen small-lambda tie break."""

    rows = [
        _source_inner_validation(sessions, basis, outer_date=outer_date, budget=budget, ridge_lambda=value)
        for value in C2F_RIDGE_GRID
    ]
    return _select_lambda_from_source_scores(rows), rows


def _select_lambda_from_source_scores(rows: Sequence[Mapping[str, Any]]) -> float:
    """Frozen score/tie rule factored for direct unit-test coverage."""

    _need(len(rows) == len(C2F_RIDGE_GRID), "C2F5 lambda selection grid coverage drift")
    _need(tuple(float(row["ridge_lambda"]) for row in rows) == C2F_RIDGE_GRID,
          "C2F5 lambda selection grid order drift")
    # Ordered grid is ascending.  The score tolerance makes the tie break
    # explicit and stable: choose the smallest lambda within tolerance of best.
    best_score = max(float(row["mean_score"]) for row in rows)
    return min(
        float(row["ridge_lambda"])
        for row in rows
        if float(row["mean_score"]) >= best_score - SELECTION_TIE_TOLERANCE
    )


def _positive(summary: Mapping[str, Any]) -> bool:
    return bool(
        summary["defined_sessions"] == 13
        and summary["mean"] > 0
        and summary["median"] > 0
        and summary["positive"] >= MIN_POSITIVE_SESSIONS
        and summary["leave_largest_absolute_out_mean"] > 0
    )


def _summary(rows: Mapping[str, Mapping[str, Any]], field: str) -> dict[str, Any]:
    return v1.paired_summary([(name, row.get(field)) for name, row in rows.items()])


def evaluate_outer_session(
    session: v1.EventSession,
    basis: v2.EndpointBasisV2,
    *,
    budget: int,
    primary: C2FOperator,
    quality_only: C2FOperator,
) -> dict[str, Any]:
    """Score all frozen C2F5 controls on one outer session's later events."""

    _need(v1.session_date(session.session_name) == basis.outer_date, "C2F5 target/basis outer-date mismatch")
    _need(not any(v1.session_date(name) == session.date for name in primary.source_sessions),
          "C2F5 target session date leaked into primary operator")
    _need(not any(v1.session_date(name) == session.date for name in quality_only.source_sessions),
          "C2F5 target session date leaked into quality-only operator")
    support, quality, support_fit = fit_support_sufficient_statistics(session, basis, budget=budget)
    label_support, label_quality, label_fit = fit_support_sufficient_statistics(
        session, basis, budget=budget, shuffled_labels=True,
    )
    corrected = primary.apply(support, quality)
    label_corrected = primary.apply(label_support, label_quality)
    quality_corrected = quality_only.apply(support, quality)
    row_corrected, row_manifest = row_shuffle_carrier(corrected, session=session.session_name, budget=budget)
    later = _events_after(session, budget)
    z_later, observed = v2.event_arrays(later, basis)
    support_mean = np.mean(np.stack([event.log_rates for event in _events_before(session, budget)]), axis=0)
    carriers = {
        "hse5": support,
        "c2f5": corrected,
        "label_shuffled_c2f5": label_corrected,
        "row_shuffled_c2f5": row_corrected,
        "quality_only_c2f5": quality_corrected,
    }
    scores = {name: v1.r2_by_channel(observed, v2.predict(value, z_later)) for name, value in carriers.items()}
    scores["intercept_only"] = v1.r2_by_channel(observed, np.broadcast_to(support_mean, observed.shape))
    defined = np.logical_and.reduce([np.isfinite(value) for value in scores.values()])
    _need(np.any(defined), f"{session.session_name}: C2F5 has no jointly defined channels")
    medians = {name: float(np.median(value[defined])) for name, value in scores.items()}
    return {
        "status": "defined",
        "session": session.session_name,
        "date": session.date,
        "budget": budget,
        "support_events": len(_events_before(session, budget)),
        "later_events": len(later),
        "defined_channels": int(defined.sum()),
        "basis_sha256": basis.basis_sha256,
        "raw_hse5_support_carrier_sha256": v1.array_sha256(support),
        "corrected_carrier_sha256": v1.array_sha256(corrected),
        "label_shuffled_corrected_carrier_sha256": v1.array_sha256(label_corrected),
        "quality_only_corrected_carrier_sha256": v1.array_sha256(quality_corrected),
        "row_shuffled_corrected_carrier_sha256": v1.array_sha256(row_corrected),
        "support_fit": support_fit,
        "label_shuffled_support_fit": label_fit,
        "primary_operator_sha256": primary.operator_sha256,
        "quality_only_operator_sha256": quality_only.operator_sha256,
        "row_shuffle": row_manifest,
        "median_r2_hse5": medians["hse5"],
        "median_r2_c2f5": medians["c2f5"],
        "median_r2_label_shuffled_c2f5": medians["label_shuffled_c2f5"],
        "median_r2_row_shuffled_c2f5": medians["row_shuffled_c2f5"],
        "median_r2_quality_only_c2f5": medians["quality_only_c2f5"],
        "median_r2_intercept_only": medians["intercept_only"],
        "delta_c2f5_minus_hse5": medians["c2f5"] - medians["hse5"],
        "delta_c2f5_minus_label_shuffled": medians["c2f5"] - medians["label_shuffled_c2f5"],
        "delta_c2f5_minus_row_shuffled": medians["c2f5"] - medians["row_shuffled_c2f5"],
        "delta_c2f5_minus_quality_only": medians["c2f5"] - medians["quality_only_c2f5"],
        "delta_c2f5_minus_intercept": medians["c2f5"] - medians["intercept_only"],
        "target_session_later_labels_enter_operator_fit": False,
    }


def _basis_for_outer(
    sessions: Mapping[str, v1.EventSession], *, outer_date: str,
) -> v2.EndpointBasisV2:
    # Use V2 itself rather than reimplementing final bases.  This binds raw
    # H-SE5 baseline reproduction to the deployed q4 endpoint carrier exactly.
    return v2.fit_source_all_event_basis(sessions, outer_date=outer_date)


def _teacher_diagnostic(examples: SourceExamples) -> dict[str, Any]:
    shift = np.linalg.norm(examples.residuals, axis=-1)
    support_norm = np.linalg.norm(examples.support_carriers, axis=-1)
    return {
        "role": "diagnostic_only_not_a_deployment_or_outer_score",
        "source_future_teacher_used_only_for_operator_training": True,
        "mean_channel_coefficient_shift_l2": float(np.mean(shift)),
        "median_channel_coefficient_shift_l2": float(np.median(shift)),
        "mean_relative_shift_l2": float(np.mean(shift / np.maximum(support_norm, 1.0e-12))),
        "per_source_session": list(examples.manifests),
    }


def budget_gate(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    delta = _summary(rows, "delta_c2f5_minus_hse5")
    material = bool(_positive(delta) and delta["mean"] >= MATERIAL_MEAN_DELTA and delta["median"] >= MATERIAL_MEDIAN_DELTA)
    controls = {
        "correct_minus_label_shuffled": _summary(rows, "delta_c2f5_minus_label_shuffled"),
        "correct_minus_row_shuffled": _summary(rows, "delta_c2f5_minus_row_shuffled"),
        "correct_minus_quality_only": _summary(rows, "delta_c2f5_minus_quality_only"),
        "correct_minus_intercept": _summary(rows, "delta_c2f5_minus_intercept"),
    }
    control_pass = {name: _positive(summary) for name, summary in controls.items()}
    passed = bool(material and all(control_pass.values()))
    return {
        "passed": passed,
        "material_gain_vs_hse5": material,
        "controls": control_pass,
        "thresholds": {
            "mean_delta_vs_hse5": MATERIAL_MEAN_DELTA,
            "median_delta_vs_hse5": MATERIAL_MEDIAN_DELTA,
            "minimum_positive_sessions": MIN_POSITIVE_SESSIONS,
            "leave_largest_absolute_out_mean_positive": True,
            "all_matched_controls_require_positive_mean_median_and_10_of_13": True,
        },
        "correct_minus_hse5": delta,
        **controls,
    }


def _baseline_reproduction(
    rows: Mapping[str, Mapping[str, Any]], reference: Mapping[str, Any], *, budget: int,
) -> dict[str, Any]:
    differences: list[dict[str, Any]] = []
    fields = (
        ("median_r2_hse5", "median_r2_correct"),
        # The raw (not C2F-transformed) H-SE5 controls must reproduce V2 too.
        ("raw_hse5_label_delta", "median_delta_shuffle"),
        ("raw_hse5_intercept_delta", "median_delta_intercept"),
    )
    for name in v1.H1_HELDIN_SESSIONS:
        observed = rows[name]
        ref = reference["budgets"][f"M{budget}"]["sessions"][name]["forward"]
        for own_key, ref_key in fields:
            differences.append({
                "session": name,
                "field": own_key,
                "absolute_difference": abs(float(observed[own_key]) - float(ref[ref_key])),
            })
    maximum = max(value["absolute_difference"] for value in differences)
    return {
        "comparisons": len(differences),
        "absolute_tolerance": 1.0e-10,
        "maximum_absolute_difference": maximum,
        "passed": bool(maximum <= 1.0e-10),
        "differences": differences,
    }


def run_screen(
    sessions: Mapping[str, v1.EventSession], *, hse5_reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the frozen date-LODO C2F5 source-only screen."""

    _need(hse5_reference.get("schema") == v2.SCHEMA, "C2F5 H-SE5 V2 reference schema mismatch")
    budgets: dict[str, Any] = {}
    for budget in SUPPORT_BUDGETS:
        rows: dict[str, dict[str, Any]] = {}
        outer_details: dict[str, Any] = {}
        for outer_date in v1.H1_DATES:
            basis = _basis_for_outer(sessions, outer_date=outer_date)
            selected_lambda, selection_rows = select_operator_lambda(
                sessions, basis, outer_date=outer_date, budget=budget,
            )
            source_names = _source_names_for_outer(outer_date)
            examples = build_source_examples(
                sessions, basis, source_names=source_names, outer_date=outer_date, budget=budget,
            )
            primary = fit_operator(examples, input_kind=PRIMARY_INPUT_KIND, ridge_lambda=selected_lambda)
            quality_only = fit_operator(examples, input_kind=QUALITY_ONLY_INPUT_KIND, ridge_lambda=selected_lambda)
            outer_details[outer_date] = {
                "basis": basis.manifest(),
                "operator_lambda_selection": {
                    "grid": list(C2F_RIDGE_GRID),
                    "tie_break": "smallest_lambda_within_1e-12_of_best_mean_source_date_LODO_score",
                    "selected_lambda": selected_lambda,
                    "source_date_lodo_rows": selection_rows,
                },
                "primary_operator": primary.manifest(),
                "quality_only_operator": quality_only.manifest(),
                "teacher_headroom_and_learnability_diagnostic": _teacher_diagnostic(examples),
            }
            for name in v1.H1_HELDIN_SESSIONS:
                if v1.session_date(name) == outer_date:
                    rows[name] = evaluate_outer_session(
                        sessions[name], basis, budget=budget, primary=primary, quality_only=quality_only,
                    )
        _need(tuple(rows) == v1.H1_HELDIN_SESSIONS, "C2F5 outer row order/coverage drift")
        # Add the exact raw H-SE5 deltas separately from the C2F control
        # deltas.  They make baseline reproduction an explicit invariant.
        for name, row in rows.items():
            basis = _basis_for_outer(sessions, outer_date=row["date"])
            raw, _quality, _fit = fit_support_sufficient_statistics(sessions[name], basis, budget=budget)
            raw_label, _label_quality, _label_fit = fit_support_sufficient_statistics(
                sessions[name], basis, budget=budget, shuffled_labels=True,
            )
            later = _events_after(sessions[name], budget)
            z_later, observed = v2.event_arrays(later, basis)
            raw_r2 = v1.r2_by_channel(observed, v2.predict(raw, z_later))
            label_r2 = v1.r2_by_channel(observed, v2.predict(raw_label, z_later))
            support_mean = np.mean(np.stack([event.log_rates for event in _events_before(sessions[name], budget)]), axis=0)
            intercept_r2 = v1.r2_by_channel(observed, np.broadcast_to(support_mean, observed.shape))
            raw_defined = np.isfinite(raw_r2) & np.isfinite(label_r2) & np.isfinite(intercept_r2)
            _need(np.any(raw_defined), f"{name}: C2F5 raw baseline no finite channels")
            row["raw_hse5_label_delta"] = float(np.median((raw_r2 - label_r2)[raw_defined]))
            row["raw_hse5_intercept_delta"] = float(np.median((raw_r2 - intercept_r2)[raw_defined]))
        reproduction = _baseline_reproduction(rows, hse5_reference, budget=budget)
        _need(reproduction["passed"], f"C2F5 M{budget} failed exact H-SE5 reproduction: {reproduction['maximum_absolute_difference']}")
        budgets[f"M{budget}"] = {
            "budget_trials": budget,
            "outer_date_details": outer_details,
            "sessions": rows,
            "raw_hse5_v2_reproduction": reproduction,
            "gate": budget_gate(rows),
        }
    passing = all(budgets[f"M{budget}"]["gate"]["passed"] for budget in SUPPORT_BUDGETS)
    return {
        "schema": SCHEMA,
        "protocol": PROTOCOL,
        "candidate_matrix_predeclared_before_data_run": True,
        "frozen_constants": {
            "rank": RANK,
            "carrier_dim": CARRIER_DIM,
            "quality_names": list(QUALITY_NAMES),
            "quality_dim": QUALITY_DIM,
            "support_budgets": list(SUPPORT_BUDGETS),
            "input_kinds": [PRIMARY_INPUT_KIND, QUALITY_ONLY_INPUT_KIND],
            "primary_operator": "B_corrected = B_support + F([B_support, quality])",
            "quality_only_control": "B_support + F(quality), with no B_support supplied to F",
            "operator_ridge_grid": list(C2F_RIDGE_GRID),
            "lambda_selection": "outer-date-excluded source-date LODO mean later-event raw-log-rate R2; smallest lambda tie break",
            "outer_future_teacher_role": "forbidden; source session only",
            "target_session_optimizer_steps": 0,
            "target_session_backward_steps": 0,
            "channel_index_feature": False,
            "gate": {
                "mean_delta_vs_hse5": MATERIAL_MEAN_DELTA,
                "median_delta_vs_hse5": MATERIAL_MEDIAN_DELTA,
                "minimum_positive_sessions": MIN_POSITIVE_SESSIONS,
                "leave_largest_absolute_out_mean_positive": True,
                "controls": [
                    "label_shuffled_support_refit_through_same_operator",
                    "row_shuffled_corrected_carrier",
                    "quality_only_no_B_support_operator",
                    "intercept_only",
                ],
            },
        },
        "budgets": budgets,
        "passing_candidate": passing,
        "status": "PASS_CPU_C2F5_MATERIAL" if passing else "STOP_CPU_C2F5_NOT_MATERIAL",
        "gpu_authorized_by_this_screen": False,
        "scope": {
            "native_position_endpoints_per_event": 2,
            "native_event_timestamps_per_event": 2,
            "dense_velocity_opened": False,
            "within_event_position_trajectory_opened": False,
            "target_session_later_labels_enter_operator_fit": False,
            "target_session_optimizer_steps": 0,
            "target_session_backward_steps": 0,
        },
    }
