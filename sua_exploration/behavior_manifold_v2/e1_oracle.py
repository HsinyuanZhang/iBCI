"""E1: oracle-latent ceiling -- the decision gate.

For each session the frozen q8 encoder produces latent codes for the FULL
session behaviour (calibration plus query).  The deployment estimator (one
closed-form neural-to-latent ridge at the deployed lambda) is then fitted on
all of it and the strict post-M10 query is decoded and scored.  The target
query therefore enters the estimator's training rows: this cell is a
LEAKAGE DIAGNOSTIC in the sub-M C3 convention, never a deployable number.  The
matching oracle DirectRidge control (full-session 16-D ridge) gives the other
family's ceiling, so both rows of the 2x2 ceiling matrix have a deployable cell
and an oracle cell.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from sua_exploration.behavior_autoencoder_v1.core import fit_affine_ridge, need, predict_affine_ridge
from sua_exploration.h1_m1_priority_v1.core import (
    DirectRidgeSession,
    _fit_ridge,
    _predict_ridge,
    array_sha256,
    regression_metrics,
)

from .frozen import FrozenDeployment
from .protocol import aligned_budget, budget_normalizers, full_session_arrays

ORACLE_LAMBDA_SENSITIVITY: tuple[float, ...] = (3.0e-2, 1.0e-1, 3.0e-1, 1.0, 3.0)
PREREGISTERED_HIGH = 0.60
PREREGISTERED_LOW = 0.50


def oracle_q8_ensemble(
    session: DirectRidgeSession,
    frozen_target,
    *,
    ridge_lambda: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Oracle-support q8 route: fit the latent ridge on the full session."""

    full_x, full_y, full_indices = full_session_arrays(session)
    query_x, query_y, query_indices = aligned_budget(session, 10, split="query")
    seed_predictions = []
    for seed in frozen_target.seeds:
        manifold = seed.manifold
        latent_full = manifold.encode(full_y)
        weight, intercept = fit_affine_ridge(
            (full_x - frozen_target.x_mean) / frozen_target.x_scale,
            latent_full,
            ridge_lambda=ridge_lambda,
        )
        latent_query = predict_affine_ridge(
            (query_x - frozen_target.x_mean) / frozen_target.x_scale, weight, intercept
        )
        seed_predictions.append(manifold.decode(latent_query))
    prediction = np.mean(np.stack(seed_predictions, axis=0), axis=0, dtype=np.float64)
    metrics = regression_metrics(query_y, prediction)
    metrics.update(
        {
            "full_session_bins": int(full_x.shape[0]),
            "full_indices_sha256": array_sha256(full_indices),
            "query_indices_sha256": array_sha256(query_indices),
            "prediction_sha256": array_sha256(prediction),
        }
    )
    return metrics, query_y, prediction


def oracle_direct_ridge(
    session: DirectRidgeSession,
    source_sessions: Sequence[DirectRidgeSession],
    *,
    ridge_lambda: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Oracle-support DirectRidge control: full-session 16-D ridge."""

    normalizer = budget_normalizers(source_sessions, 10)
    full_x, full_y, full_indices = full_session_arrays(session)
    query_x, query_y, query_indices = aligned_budget(session, 10, split="query")
    weight, intercept = _fit_ridge(full_x, full_y, ridge_lambda=ridge_lambda, **normalizer)
    prediction = _predict_ridge(query_x, weight=weight, intercept=intercept, **normalizer)
    metrics = regression_metrics(query_y, prediction)
    metrics.update(
        {
            "full_session_bins": int(full_x.shape[0]),
            "full_indices_sha256": array_sha256(full_indices),
            "query_indices_sha256": array_sha256(query_indices),
            "weight_sha256": array_sha256(weight),
            "prediction_sha256": array_sha256(prediction),
        }
    )
    return metrics, query_y, prediction


def experiment_e1(frozen: FrozenDeployment, *, sessions: Mapping[str, DirectRidgeSession]) -> dict[str, Any]:
    names = frozen.names
    oracle_q8: dict[str, float] = {}
    oracle_q8_lambda_curves: dict[str, Any] = {}
    oracle_dr: dict[str, float] = {}
    oracle_dr_by_lambda: dict[str, Any] = {}
    for name in names:
        target = frozen.targets[name]
        sources = [sessions[source] for source in target.sources]
        deployed_lambdas = {seed.ridge_lambda for seed in target.seeds}
        need(
            len(deployed_lambdas) == 1,
            f"{name}: oracle cell requires a shared deployed lambda across seeds",
        )
        # q8 oracle at the shared deployed lambda (0.3 for every frozen seed).
        metrics, _, _ = oracle_q8_ensemble(
            sessions[name], target, ridge_lambda=target.seeds[0].ridge_lambda
        )
        oracle_q8[name] = float(metrics["pooled_variance_weighted_r2"])
        curve = {}
        for value in ORACLE_LAMBDA_SENSITIVITY:
            lambda_metrics, _, _ = oracle_q8_ensemble(sessions[name], target, ridge_lambda=value)
            curve[str(value)] = float(lambda_metrics["pooled_variance_weighted_r2"])
        oracle_q8_lambda_curves[name] = curve
        # Oracle DirectRidge at the frozen baseline's deployed lambda per target.
        deployed_lambda = float(frozen.direct_baseline["folds"][name]["selected"]["ridge_lambda_per_sample"])
        dr_metrics, _, _ = oracle_direct_ridge(
            sessions[name], sources, ridge_lambda=deployed_lambda
        )
        oracle_dr[name] = float(dr_metrics["pooled_variance_weighted_r2"])
        dr_curve = {}
        for value in (1.0e-1, 3.0e-1):
            lambda_metrics, _, _ = oracle_direct_ridge(sessions[name], sources, ridge_lambda=value)
            dr_curve[str(value)] = float(lambda_metrics["pooled_variance_weighted_r2"])
        oracle_dr_by_lambda[name] = dr_curve

    deployed_q8 = np.asarray(
        [frozen.targets[name].ensemble_metrics["pooled_variance_weighted_r2"] for name in names],
        dtype=np.float64,
    )
    deployed_dr = np.asarray(
        [frozen.targets[name].directridge_metrics["pooled_variance_weighted_r2"] for name in names],
        dtype=np.float64,
    )
    oracle_q8_values = np.asarray([oracle_q8[name] for name in names], dtype=np.float64)
    oracle_dr_values = np.asarray([oracle_dr[name] for name in names], dtype=np.float64)

    oracle_q8_mean = float(oracle_q8_values.mean())
    if oracle_q8_mean >= PREREGISTERED_HIGH:
        reading = "ESTIMATION_LIMITED: zero-training estimator upgrades are the route; Stage-2 aux-head cancelled"
    elif oracle_q8_mean <= PREREGISTERED_LOW:
        reading = "REPRESENTATIONAL_GAP: aux-head (manifold as source-training regularizer) justified"
    else:
        reading = "AMBIGUOUS_BAND: oracle q8 lies between the two pre-registered thresholds"

    matrix = {
        "DirectRidge": {
            "m10_deployable": {
                "equal_session_mean_r2": float(deployed_dr.mean()),
                "per_session": dict(zip(names, deployed_dr.tolist())),
            },
            "oracle_full_session": {
                "equal_session_mean_r2": float(oracle_dr_values.mean()),
                "per_session": dict(zip(names, oracle_dr_values.tolist())),
            },
        },
        "q8_route": {
            "m10_deployable": {
                "equal_session_mean_r2": float(deployed_q8.mean()),
                "per_session": dict(zip(names, deployed_q8.tolist())),
            },
            "oracle_full_session": {
                "equal_session_mean_r2": oracle_q8_mean,
                "per_session": dict(zip(names, oracle_q8_values.tolist())),
            },
        },
    }
    return {
        "schema": "m1_behavior_manifold_v2_e1_oracle_ceiling_v1",
        "status": "COMPLETE_E1_ORACLE_LATENT_CEILING",
        "leakage_disclosure": (
            "Oracle cells fit the estimator on the FULL session (calibration plus strict post-M10 query); "
            "they are leakage diagnostics in the sub-M C3 convention and are never deployable numbers. "
            "The M10-deployable cells are the frozen published results recomputed bit-exactly."
        ),
        "ceiling_matrix": matrix,
        "on_manifold_gap_deployed": float(deployed_q8.mean() - deployed_dr.mean()),
        "oracle_gap_within_q8_route": float(oracle_q8_mean - deployed_q8.mean()),
        "oracle_gain_directridge_to_q8": float(oracle_q8_mean - oracle_dr_values.mean()),
        "preregistered_reading": {
            "thresholds": {"high": PREREGISTERED_HIGH, "low": PREREGISTERED_LOW},
            "oracle_q8_equal_session_mean": oracle_q8_mean,
            "reading": reading,
        },
        "oracle_q8_lambda_sensitivity_per_session": oracle_q8_lambda_curves,
        "oracle_directridge_lambda_sensitivity_per_session": oracle_dr_by_lambda,
        "oracle_directridge_deployed_lambda_per_session": {
            name: float(frozen.direct_baseline["folds"][name]["selected"]["ridge_lambda_per_sample"])
            for name in names
        },
        "protocol": {
            "estimator": "the deployment rule's closed-form neural-to-latent ridge (slope-only lambda*N penalty, frozen source-support x normalizer)",
            "latent_support": "z = frozen q8 encoder(full session behaviour); three-seed mean of decoded query predictions",
            "lambda": "each seed's deployed lambda for the primary oracle cell; sensitivity grid recorded",
            "query": "strict post-M10 (frozen rule, unchanged)",
            "scoring": "variance-weighted 16-output R2 per session; equal session aggregation",
        },
        "disclosure": {
            "target_backward_steps": 0,
            "target_optimizer_steps": 0,
            "formal_heldout_opened": False,
            "minival_opened": False,
            "frozen_artifacts_changed": False,
            "oracle_cells_labelled_leakage_diagnostic": True,
        },
    }
