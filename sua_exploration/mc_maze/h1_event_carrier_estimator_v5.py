"""Frozen source-only H1 sparse-event estimator screen (V5).

V1--V4 changed the *label representation* (endpoint basis, context, tag, or
mean-velocity semantics) while retaining the same ordinary ridge estimator.
This module holds the proven H-SE5 q=4 endpoint representation fixed and tests
three different estimator axes before any further GPU proposal:

``poisson_exposure_irls``
    A log-link Poisson count model with event duration as a fixed exposure
    offset.  Its five carrier values are four log-rate slopes and an intercept.

``within_trial_contrast_ridge``
    Ridge slopes are estimated after removing a separate intercept from each
    calibration trial, then restored to one deployable global intercept.  This
    asks whether trial-wide rate excursions are a nuisance rather than tuning.

``eb_channel_shrinkage``
    The ordinary target ridge estimate is analytically shrunk toward a
    source-date-LODO, channel-correspondent empirical-Bayes prior.  The prior,
    its variance and every hyperparameter are learned from source sessions
    alone; no target-session optimizer or backpropagation is used.

All candidates output exactly ``[w1,w2,w3,w4,b]`` per channel.  They consume
only native event endpoints/timestamps and event spike counts reconstructed
from the already parsed event rates; they never read dense H1 velocity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from sua_exploration.mc_maze import h1_event_carrier_design_screen as design
from sua_exploration.mc_maze import h1_event_carrier_semantic_v4 as semantic
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1


# V5r2 corrects an exposure-offset error in the V5r1 Poisson working response.
# V5r1's immutable receipt remains preserved as an invalidated historical
# artifact; it is not a scientific result and must never be compared further.
SCHEMA = "h1_event_carrier_estimator_v5_source_screen_v2"
PROTOCOL = "h1_event_carrier_likelihood_nuisance_prior_source_screen_20260812_v2"
RANK = 4
CARRIER_DIM = 5
TARGET_RIDGE_LAMBDA = 3.0
POISSON_RIDGE_LAMBDA = 3.0
CONTRAST_RIDGE_LAMBDA = 3.0
EB_TARGET_RIDGE_LAMBDA = 3.0
EB_VARIANCE_FLOOR = 1.0e-8
POISSON_MAX_ITERATIONS = 20
POISSON_TOLERANCE = 1.0e-8
LOG_RATE_CLIP = 12.0


@dataclass(frozen=True)
class EstimatorCandidate:
    name: str
    family: str
    requires_source_prior: bool = False

    @property
    def carrier_dim(self) -> int:
        return CARRIER_DIM


# The matrix is intentionally literal and ordered.  Do not append a candidate
# after looking at the receipt; a new hypothesis must receive a new screen.
CANDIDATES: tuple[EstimatorCandidate, ...] = (
    EstimatorCandidate("hse5_pca_delta_q4", "ordinary_ridge_baseline"),
    EstimatorCandidate("poisson_exposure_irls", "likelihood"),
    EstimatorCandidate("within_trial_contrast_ridge", "nuisance"),
    EstimatorCandidate("eb_channel_shrinkage", "source_prior", requires_source_prior=True),
)


def _need(condition: bool, message: str) -> None:
    v1._need(condition, message)


def _design(z: np.ndarray) -> np.ndarray:
    values = np.asarray(z, dtype=np.float64)
    _need(values.ndim == 2 and values.shape[1] == RANK and values.shape[0] >= CARRIER_DIM,
          f"V5 latent design must be [E,{RANK}], got {values.shape}")
    output = np.column_stack((np.ones(values.shape[0], dtype=np.float64), values))
    _need(np.linalg.matrix_rank(output) == CARRIER_DIM, "V5 support design is rank deficient")
    return output


def _carrier_from_beta(beta: np.ndarray) -> np.ndarray:
    value = np.asarray(beta, dtype=np.float64)
    _need(value.shape == (CARRIER_DIM, v1.EXPECTED_NEURONS) and np.isfinite(value).all(),
          "V5 beta shape/nonfinite drift")
    return np.column_stack((value[1:].T, value[0]))


def _beta_from_carrier(carrier: np.ndarray) -> np.ndarray:
    value = np.asarray(carrier, dtype=np.float64)
    _need(value.shape == (v1.EXPECTED_NEURONS, CARRIER_DIM), "V5 carrier shape drift")
    return np.vstack((value[:, -1], value[:, :RANK].T))


def _linear_beta(z: np.ndarray, response: np.ndarray, *, ridge_lambda: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the exact H-SE5 ridge form and return beta, residual variance, covariance diagonal."""

    x = _design(z)
    y = np.asarray(response, dtype=np.float64)
    _need(y.shape == (x.shape[0], v1.EXPECTED_NEURONS) and np.isfinite(y).all(), "V5 response shape drift")
    penalty = np.diag([0.0] + [1.0] * RANK) * (x.shape[0] * float(ridge_lambda))
    system = x.T @ x + penalty
    inverse = np.linalg.inv(system)
    beta = inverse @ (x.T @ y)
    residual = y - x @ beta
    # Ridge's sampling covariance is A^{-1} X'X A^{-1} sigma², not simply
    # diag(A^{-1}) sigma².  Its residual degrees of freedom use the smoother
    # trace tr[X A^{-1} X']; this is the frequentist variance contract used by
    # the H1 EB protocol.  Fail closed rather than invent a denominator.
    effective_df = float(np.trace((x.T @ x) @ inverse))
    residual_df = float(x.shape[0]) - effective_df
    _need(np.isfinite(effective_df) and residual_df > 1.0e-8,
          f"V5 ridge effective residual degrees of freedom invalid: {residual_df}")
    sigma2 = np.maximum(np.sum(np.square(residual), axis=0) / residual_df, EB_VARIANCE_FLOOR)
    covariance_kernel = inverse @ (x.T @ x) @ inverse
    covariance_diag = np.maximum(np.diag(covariance_kernel), 0.0)[:, None] * sigma2[None, :]
    _need(np.isfinite(beta).all() and np.isfinite(covariance_diag).all(), "V5 linear fit is nonfinite")
    return beta, sigma2, covariance_diag


def log_rates_to_counts(response: np.ndarray, durations: np.ndarray) -> np.ndarray:
    """Recover integer event counts from the bound ``log1p(count / duration)`` contract."""

    y = np.asarray(response, dtype=np.float64)
    exposure = np.asarray(durations, dtype=np.float64).reshape(-1)
    _need(y.ndim == 2 and y.shape[1] == v1.EXPECTED_NEURONS and y.shape[0] == exposure.size,
          "V5 count reconstruction shape drift")
    _need(np.all(exposure > 0) and np.isfinite(y).all(), "V5 count reconstruction input drift")
    raw = np.expm1(y) * exposure[:, None]
    rounded = np.rint(raw)
    _need(np.all(raw >= -1.0e-9) and np.max(np.abs(raw - rounded), initial=0.0) <= 1.0e-8,
          "event log-rate contract does not recover integer spike counts")
    return np.asarray(rounded, dtype=np.float64)


def fit_poisson_exposure_beta(z: np.ndarray, response: np.ndarray, durations: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Vectorized, fixed-iteration Poisson IRLS with a duration exposure offset.

    The output beta parameterizes log firing *rate*.  For forward prediction
    against the stored ``log1p(rate)`` response, callers use
    ``log1p(exp(X beta))``; no nonlinear extra carrier dimensions are added.
    """

    x = _design(z)
    y = np.asarray(response, dtype=np.float64)
    exposure = np.asarray(durations, dtype=np.float64).reshape(-1)
    counts = log_rates_to_counts(y, exposure)
    _need(exposure.shape == (x.shape[0],) and np.all(exposure > 0), "V5 exposure drift")
    penalty = np.diag([0.0] + [1.0] * RANK) * (x.shape[0] * POISSON_RIDGE_LAMBDA)
    rate = np.maximum(counts.sum(axis=0) / exposure.sum(), 1.0e-8)
    beta = np.zeros((CARRIER_DIM, v1.EXPECTED_NEURONS), dtype=np.float64)
    beta[0] = np.log(rate)
    iterations = 0
    final_update = float("inf")
    for iterations in range(1, POISSON_MAX_ITERATIONS + 1):
        eta = np.clip(x @ beta, -LOG_RATE_CLIP, LOG_RATE_CLIP)
        mean_count = np.maximum(exposure[:, None] * np.exp(eta), 1.0e-8)
        # ``eta`` is deliberately *X beta*, i.e. log rate rather than the
        # complete GLM predictor ``log(exposure) + X beta``.  Consequently the
        # offset has already cancelled in this working response.  Subtracting
        # ``log(exposure)`` here would apply the offset a second time and make
        # a constant rate appear duration dependent (V5r1's invalid bug).
        target = eta + (counts - mean_count) / mean_count
        updated = np.empty_like(beta)
        # The per-channel IRLS weights are distinct, so each small 5x5 normal
        # equation is intentionally explicit.  This is CPU-only and tiny.
        for channel in range(v1.EXPECTED_NEURONS):
            weight = mean_count[:, channel]
            system = x.T @ (weight[:, None] * x) + penalty
            cross = x.T @ (weight * target[:, channel])
            updated[:, channel] = np.linalg.solve(system, cross)
        final_update = float(np.max(np.abs(updated - beta)))
        beta = updated
        if final_update <= POISSON_TOLERANCE:
            break
    _need(np.isfinite(beta).all(), "V5 Poisson IRLS produced nonfinite beta")
    return beta, {
        "iterations": int(iterations),
        "maximum_final_beta_update": final_update,
        "converged": bool(final_update <= POISSON_TOLERANCE),
        "poisson_ridge_lambda": POISSON_RIDGE_LAMBDA,
        "exposure_offset": "log(duration_seconds)",
    }


def poisson_predict(beta: np.ndarray, z: np.ndarray) -> np.ndarray:
    x = _design(z)
    value = np.asarray(beta, dtype=np.float64)
    _need(value.shape == (CARRIER_DIM, v1.EXPECTED_NEURONS), "V5 Poisson beta shape drift")
    log_rate = np.clip(x @ value, -LOG_RATE_CLIP, LOG_RATE_CLIP)
    return np.log1p(np.exp(log_rate))


def fit_within_trial_contrast_beta(
    z: np.ndarray, response: np.ndarray, trial_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit slope only after projecting out calibration-trial intercepts."""

    latent = np.asarray(z, dtype=np.float64)
    y = np.asarray(response, dtype=np.float64)
    trials = np.asarray(trial_indices, dtype=np.int64).reshape(-1)
    _need(latent.shape[0] == y.shape[0] == trials.size and latent.shape[1] == RANK,
          "V5 contrast input shape drift")
    unique = np.unique(trials)
    _need(unique.size >= 2, "V5 within-trial contrast needs at least two trials")
    centered_z = latent.copy()
    centered_y = y.copy()
    counts_by_trial: dict[str, int] = {}
    for trial in unique.tolist():
        mask = trials == trial
        _need(int(mask.sum()) >= 2, f"V5 trial {trial} lacks within-trial contrast")
        centered_z[mask] -= centered_z[mask].mean(axis=0, keepdims=True)
        centered_y[mask] -= centered_y[mask].mean(axis=0, keepdims=True)
        counts_by_trial[str(int(trial))] = int(mask.sum())
    penalty = np.eye(RANK, dtype=np.float64) * (latent.shape[0] * CONTRAST_RIDGE_LAMBDA)
    weights = np.linalg.solve(centered_z.T @ centered_z + penalty, centered_z.T @ centered_y)
    intercept = (y - latent @ weights).mean(axis=0)
    beta = np.vstack((intercept, weights))
    _need(np.isfinite(beta).all(), "V5 contrast beta is nonfinite")
    return beta, {
        "trial_count": int(unique.size),
        "events_per_trial": counts_by_trial,
        "centered_design_rank": int(np.linalg.matrix_rank(centered_z)),
        "contrast_ridge_lambda": CONTRAST_RIDGE_LAMBDA,
        "intercept_restoration": "global mean(y - z @ within_trial_contrast_slopes)",
    }


@dataclass(frozen=True)
class EBPrior:
    outer_date: str
    budget: int
    source_sessions: tuple[str, ...]
    mean: np.ndarray  # [5,176]
    tau2: np.ndarray  # [5,176]
    source_carrier_sha256: tuple[str, ...]
    prior_sha256: str

    def manifest(self) -> dict[str, Any]:
        return {
            "outer_date": self.outer_date,
            "budget": self.budget,
            "source_sessions": list(self.source_sessions),
            "source_carrier_sha256": list(self.source_carrier_sha256),
            "array_sha256": {
                "mean": v1.array_sha256(self.mean),
                "tau2": v1.array_sha256(self.tau2),
            },
            "prior_sha256": self.prior_sha256,
            "variance_model": "max(source_sample_variance - mean_source_estimation_variance, floor)",
            "source_only": True,
        }


def fit_source_eb_prior(
    sessions: Mapping[str, design.ContextSession],
    endpoint_map: semantic.SemanticMap,
    *,
    outer_date: str,
    budget: int,
) -> EBPrior:
    """Fit a channel-correspondent prior without the outer date's sessions."""

    source_names = tuple(name for name in v1.H1_HELDIN_SESSIONS if v1.session_date(name) != outer_date)
    _need(len(source_names) >= 10, "V5 source EB pool is underspecified")
    betas: list[np.ndarray] = []
    covariance: list[np.ndarray] = []
    carrier_shas: list[str] = []
    for name in source_names:
        support = design.select_range(sessions[name], start=0, budget=budget)
        z = endpoint_map.transform(support)
        y = np.stack([event.base.log_rates for event in support]).astype(np.float64)
        beta, _sigma2, covariance_diag = _linear_beta(z, y, ridge_lambda=EB_TARGET_RIDGE_LAMBDA)
        betas.append(beta)
        covariance.append(covariance_diag)
        carrier_shas.append(v1.array_sha256(_carrier_from_beta(beta)))
    stacked = np.stack(betas, axis=0)
    cov = np.stack(covariance, axis=0)
    mean = stacked.mean(axis=0)
    observed_var = stacked.var(axis=0, ddof=1)
    tau2 = np.maximum(observed_var - cov.mean(axis=0), EB_VARIANCE_FLOOR)
    body = {
        "protocol": PROTOCOL,
        "outer_date": outer_date,
        "budget": budget,
        "source_sessions": list(source_names),
        "mean": v1.array_sha256(mean),
        "tau2": v1.array_sha256(tau2),
        "source_carrier_sha256": carrier_shas,
        "eb_target_ridge_lambda": EB_TARGET_RIDGE_LAMBDA,
        "eb_variance_floor": EB_VARIANCE_FLOOR,
    }
    return EBPrior(
        outer_date=outer_date,
        budget=budget,
        source_sessions=source_names,
        mean=np.asarray(mean, dtype=np.float64),
        tau2=np.asarray(tau2, dtype=np.float64),
        source_carrier_sha256=tuple(carrier_shas),
        prior_sha256=v1.canonical_sha256(body),
    )


def fit_eb_beta(z: np.ndarray, response: np.ndarray, prior: EBPrior) -> tuple[np.ndarray, dict[str, Any]]:
    beta, _sigma2, target_covariance = _linear_beta(z, response, ridge_lambda=EB_TARGET_RIDGE_LAMBDA)
    _need(prior.mean.shape == prior.tau2.shape == beta.shape, "V5 EB prior shape drift")
    alpha = prior.tau2 / (prior.tau2 + np.maximum(target_covariance, EB_VARIANCE_FLOOR))
    alpha = np.clip(alpha, 0.0, 1.0)
    posterior = alpha * beta + (1.0 - alpha) * prior.mean
    _need(np.isfinite(posterior).all(), "V5 EB posterior is nonfinite")
    return posterior, {
        "prior_sha256": prior.prior_sha256,
        "mean_target_weight": float(alpha.mean()),
        "median_target_weight": float(np.median(alpha)),
        "minimum_target_weight": float(alpha.min()),
        "maximum_target_weight": float(alpha.max()),
        "target_ridge_lambda": EB_TARGET_RIDGE_LAMBDA,
    }


def _arrays(
    events: Sequence[design.ContextEvent], endpoint_map: semantic.SemanticMap,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = endpoint_map.transform(events)
    y = np.stack([event.base.log_rates for event in events]).astype(np.float64)
    durations = np.asarray([event.base.duration_seconds for event in events], dtype=np.float64)
    trials = np.asarray([event.base.trial_index for event in events], dtype=np.int64)
    _need(y.shape == (z.shape[0], v1.EXPECTED_NEURONS) and np.all(durations > 0), "V5 event arrays drift")
    return z, y, durations, trials


def _fit_candidate(
    candidate: EstimatorCandidate,
    z: np.ndarray,
    y: np.ndarray,
    durations: np.ndarray,
    trials: np.ndarray,
    *,
    prior: EBPrior | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return carrier, forward prediction beta, and fit metadata."""

    if candidate.name == "hse5_pca_delta_q4":
        # This exact shared implementation is deliberately the H-SE5 reference.
        carrier = design.fit_target_carrier(z, y, durations=durations, weighted=False)
        beta = _beta_from_carrier(carrier)
        return carrier, beta, {"estimator": "design.fit_target_carrier exact H-SE5 baseline"}
    if candidate.name == "poisson_exposure_irls":
        beta, metadata = fit_poisson_exposure_beta(z, y, durations)
        return _carrier_from_beta(beta), beta, metadata
    if candidate.name == "within_trial_contrast_ridge":
        beta, metadata = fit_within_trial_contrast_beta(z, y, trials)
        return _carrier_from_beta(beta), beta, metadata
    if candidate.name == "eb_channel_shrinkage":
        _need(prior is not None, "V5 EB candidate requires source prior")
        beta, metadata = fit_eb_beta(z, y, prior)
        return _carrier_from_beta(beta), beta, metadata
    raise v1.SparseEventEndpointError(f"unknown V5 candidate {candidate.name}")


def _predict_candidate(candidate: EstimatorCandidate, beta: np.ndarray, z: np.ndarray) -> np.ndarray:
    if candidate.name == "poisson_exposure_irls":
        return poisson_predict(beta, z)
    return _design(z) @ beta


def _row_shuffle(carrier: np.ndarray, *, session: str, budget: int, candidate: str) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(carrier, dtype=np.float64)
    _need(values.shape == (v1.EXPECTED_NEURONS, CARRIER_DIM), "V5 row shuffle input drift")
    import hashlib

    key = f"{v1.SHUFFLE_NAMESPACE}:v5-row:{candidate}:{session}:M{budget}"
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little"))
    order = np.arange(v1.EXPECTED_NEURONS, dtype=np.int64)
    for _ in range(100):
        rng.shuffle(order)
        if not np.any(order == np.arange(v1.EXPECTED_NEURONS)):
            break
    _need(not np.any(order == np.arange(v1.EXPECTED_NEURONS)), "V5 row shuffle retained fixed points")
    return values[order], {
        "namespace": "h1-event-carrier-estimator-v5-row-shuffle-v1",
        "candidate": candidate,
        "session": session,
        "budget": budget,
        "order_sha256": v1.array_sha256(order),
        "fixed_points": 0,
    }


def evaluate_session(
    session: design.ContextSession,
    endpoint_map: semantic.SemanticMap,
    candidate: EstimatorCandidate,
    *,
    budget: int,
    prior: EBPrior | None,
) -> dict[str, Any]:
    support = design.select_range(session, start=0, budget=budget)
    later = tuple(event for event in session.events if event.base.trial_index >= budget)
    _need(len(support) >= CARRIER_DIM and len(later) >= 4, f"{session.base.session_name}: insufficient V5 events")
    z_support, y_support, durations, trials = _arrays(support, endpoint_map)
    z_later, y_later, _later_duration, _later_trials = _arrays(later, endpoint_map)
    carrier, beta, fit = _fit_candidate(candidate, z_support, y_support, durations, trials, prior=prior)

    order, label_manifest = v1.within_trial_label_shuffle(
        tuple(event.base for event in support), session=session.base.session_name, budget=budget,
    )
    label_carrier, label_beta, label_fit = _fit_candidate(
        candidate, z_support[order], y_support, durations, trials, prior=prior,
    )
    # A fixed-iteration IRLS candidate is scientifically undefined if either
    # required fit did not converge.  We retain convergence metadata in the
    # receipt but do not calculate or interpret a negative forward score.
    if candidate.name == "poisson_exposure_irls" and not (fit["converged"] and label_fit["converged"]):
        return {
            "status": "undefined_nonconverged_poisson_fit",
            "session": session.base.session_name,
            "budget": budget,
            "support_events": len(support),
            "later_events": len(later),
            "design_rank": int(np.linalg.matrix_rank(_design(z_support))),
            "carrier_dim": CARRIER_DIM,
            "correct_carrier_sha256": v1.array_sha256(carrier),
            "label_carrier_sha256": v1.array_sha256(label_carrier),
            "label_shuffle": label_manifest,
            "fit": fit,
            "label_fit": label_fit,
        }
    correct = v1.r2_by_channel(y_later, _predict_candidate(candidate, beta, z_later))
    shuffled = v1.r2_by_channel(y_later, _predict_candidate(candidate, label_beta, z_later))
    intercept = v1.r2_by_channel(y_later, np.broadcast_to(y_support.mean(axis=0), y_later.shape))

    shuffled_rows, row_manifest = _row_shuffle(
        carrier, session=session.base.session_name, budget=budget, candidate=candidate.name,
    )
    row_beta = _beta_from_carrier(shuffled_rows)
    row = v1.r2_by_channel(y_later, _predict_candidate(candidate, row_beta, z_later))
    defined = np.isfinite(correct) & np.isfinite(shuffled) & np.isfinite(intercept) & np.isfinite(row)
    _need(np.any(defined), f"{session.base.session_name}: no finite V5 forward channels")
    return {
        "status": "defined",
        "session": session.base.session_name,
        "budget": budget,
        "support_events": len(support),
        "later_events": len(later),
        "design_rank": int(np.linalg.matrix_rank(_design(z_support))),
        "carrier_dim": CARRIER_DIM,
        "defined_channels": int(defined.sum()),
        "median_r2_correct": float(np.median(correct[defined])),
        "median_delta_label_shuffle": float(np.median((correct - shuffled)[defined])),
        "median_delta_intercept": float(np.median((correct - intercept)[defined])),
        "median_delta_row_shuffle": float(np.median((correct - row)[defined])),
        "correct_carrier_sha256": v1.array_sha256(carrier),
        "label_carrier_sha256": v1.array_sha256(label_carrier),
        "label_shuffle": label_manifest,
        "row_shuffle": row_manifest,
        "fit": fit,
        "label_fit": label_fit,
    }


def _summary(rows: Mapping[str, Mapping[str, Any]], field: str) -> dict[str, Any]:
    return v1.paired_summary([(name, row.get(field)) for name, row in rows.items()])


def aggregate_candidate(
    rows: Mapping[str, Mapping[str, Any]], baseline: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    undefined = [name for name, row in rows.items() if row.get("status", "defined") != "defined"]
    if undefined:
        _need(all(row.get("status", "defined") in {"defined", "undefined_nonconverged_poisson_fit"} for row in rows.values()),
              "V5 undefined aggregate has an unrecognized row status")
        correct = [bool(row["fit"].get("converged")) for row in rows.values()]
        shuffled = [bool(row["label_fit"].get("converged")) for row in rows.values()]
        return {
            "status": "undefined_nonconverged_poisson_fits",
            "defined_session_count": len(rows) - len(undefined),
            "undefined_session_count": len(undefined),
            "undefined_sessions": undefined,
            "poisson_convergence": {
                "correct_fit": {
                    "evaluated_sessions": len(rows), "converged_sessions": int(sum(correct)),
                    "nonconverged_sessions": int(len(rows) - sum(correct)),
                    "maximum_final_beta_update": float(max(row["fit"]["maximum_final_beta_update"] for row in rows.values())),
                },
                "label_shuffle_fit": {
                    "evaluated_sessions": len(rows), "converged_sessions": int(sum(shuffled)),
                    "nonconverged_sessions": int(len(rows) - sum(shuffled)),
                    "maximum_final_beta_update": float(max(row["label_fit"]["maximum_final_beta_update"] for row in rows.values())),
                },
                "fail_closed": True,
                "interpretation": "no Poisson forward R2/delta is computed or interpreted when any required IRLS fit is nonconverged",
            },
        }
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
        "correct_minus_row_shuffle": _summary(augmented, "median_delta_row_shuffle"),
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


def budget_gate(candidate: EstimatorCandidate, aggregate: Mapping[str, Any]) -> dict[str, Any]:
    if aggregate.get("status", "defined") != "defined":
        return {
            "passed": False,
            "material_gain_vs_hse5": False,
            "correct_minus_label_shuffle": False,
            "correct_minus_intercept": False,
            "correct_minus_row_shuffle": False,
            "undefined_reason": aggregate["status"],
            "thresholds": {
                "mean_delta_vs_hse5": 0.02,
                "median_delta_vs_hse5": 0.01,
                "minimum_positive_sessions": 10,
                "leave_largest_absolute_out_mean_positive": True,
            },
        }
    delta = aggregate["correct_minus_hse5"]
    material = bool(_positive(delta) and delta["mean"] >= 0.02 and delta["median"] >= 0.01)
    label = _positive(aggregate["correct_minus_label_shuffle"])
    intercept = _positive(aggregate["correct_minus_intercept"])
    row = _positive(aggregate["correct_minus_row_shuffle"])
    passed = bool(candidate.name != "hse5_pca_delta_q4" and material and label and intercept and row)
    return {
        "passed": passed,
        "material_gain_vs_hse5": material,
        "correct_minus_label_shuffle": label,
        "correct_minus_intercept": intercept,
        "correct_minus_row_shuffle": row,
        "thresholds": {
            "mean_delta_vs_hse5": 0.02,
            "median_delta_vs_hse5": 0.01,
            "minimum_positive_sessions": 10,
            "leave_largest_absolute_out_mean_positive": True,
        },
    }


def run_screen(sessions: Mapping[str, design.ContextSession]) -> dict[str, Any]:
    endpoint_candidate = semantic.SemanticCandidate("pca_delta_q4", "pca", "delta")
    maps = {
        date: semantic.fit_semantic_map(sessions, outer_date=date, candidate=endpoint_candidate)
        for date in v1.H1_DATES
    }
    priors: dict[str, dict[str, EBPrior]] = {"M3": {}, "M4": {}}
    for budget in (3, 4):
        for date in v1.H1_DATES:
            priors[f"M{budget}"][date] = fit_source_eb_prior(
                sessions, maps[date], outer_date=date, budget=budget,
            )

    budgets: dict[str, Any] = {}
    for budget in (3, 4):
        rows_by_candidate: dict[str, dict[str, Any]] = {candidate.name: {} for candidate in CANDIDATES}
        for candidate in CANDIDATES:
            for name in v1.H1_HELDIN_SESSIONS:
                date = v1.session_date(name)
                rows_by_candidate[candidate.name][name] = evaluate_session(
                    sessions[name], maps[date], candidate, budget=budget,
                    prior=priors[f"M{budget}"][date] if candidate.requires_source_prior else None,
                )
        baseline = rows_by_candidate["hse5_pca_delta_q4"]
        candidate_bodies: dict[str, Any] = {}
        for candidate in CANDIDATES:
            aggregate = aggregate_candidate(rows_by_candidate[candidate.name], baseline)
            candidate_bodies[candidate.name] = {
                "spec": {
                    "family": candidate.family,
                    "carrier_dim": candidate.carrier_dim,
                    "requires_source_prior": candidate.requires_source_prior,
                    "target_session_optimizer_steps": 0,
                    "target_session_backward_steps": 0,
                    "reads_only_endpoint_positions_event_timestamps_and_spike_times": True,
                },
                "sessions": rows_by_candidate[candidate.name],
                "aggregate": aggregate,
                "gate": budget_gate(candidate, aggregate),
            }
        budgets[f"M{budget}"] = {
            "budget_trials": budget,
            "endpoint_maps": {date: maps[date].manifest() for date in v1.H1_DATES},
            "source_eb_priors": {date: priors[f"M{budget}"][date].manifest() for date in v1.H1_DATES},
            "candidates": candidate_bodies,
        }

    passing = [
        candidate.name for candidate in CANDIDATES[1:]
        if all(budgets[f"M{budget}"]["candidates"][candidate.name]["gate"]["passed"] for budget in (3, 4))
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
            "poisson_ridge_lambda": POISSON_RIDGE_LAMBDA,
            "contrast_ridge_lambda": CONTRAST_RIDGE_LAMBDA,
            "eb_target_ridge_lambda": EB_TARGET_RIDGE_LAMBDA,
            "eb_variance_floor": EB_VARIANCE_FLOOR,
            "poisson_max_iterations": POISSON_MAX_ITERATIONS,
            "support_budgets": [3, 4],
            "candidates_in_fixed_order": [candidate.name for candidate in CANDIDATES],
            "selection_rule": "pass unchanged material+all-controls gate at both M3/M4; maximize minimum-budget median delta",
        },
        "candidate_matrix_predeclared_before_data_run": True,
        "budgets": budgets,
        "passing_candidates": passing,
        "selected_candidate": selected,
        "status": "PASS_CPU_ESTIMATOR_CANDIDATE_SELECTED" if selected else "STOP_CPU_ESTIMATOR_CANDIDATES_NOT_MATERIAL",
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
