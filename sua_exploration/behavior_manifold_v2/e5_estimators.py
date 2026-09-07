"""E5: closed-form upgrades of the deployment latent ridge.

Four estimators of the SAME frozen route (frozen encoder, one affine ridge in
latent space, frozen decoder, three-seed mean), differing only in how the
ridge's parameters are chosen:

  (a) per-session GCV lambda -- selected on the target's own M10 support alone
      via the closed-form generalized-cross-validation curve (exact leave-one-out
      residuals also recorded); fully deployable;
  (b) pooled lambda -- one lambda selected across the three source sessions by
      the frozen nested rule (inner manifolds refit without the validation
      session, each source scored on its own strict post-M10 query);
  (c) empirical-Bayes shrinkage -- the target's M10 latent ridge convexly
      shrunk toward a source-population latent regression fitted on the other
      three sessions' FULL data (outer target excluded), with the shrinkage
      weight selected on the three sources under the same inner-refit rule;
  (d) the deployed fixed lambda (the frozen published estimator, recomputed).

Each variant is scored on the strict post-M10 query with the governing
convention, and the M10 -> oracle gap-closing fraction from E1 is reported.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from sua_exploration.behavior_autoencoder_v1.calibration_aware import (
    CalibrationAwareSpec,
    fit_calibration_aware_manifold,
)
from sua_exploration.behavior_autoencoder_v1.core import (
    fit_affine_ridge,
    need,
    predict_affine_ridge,
)
from sua_exploration.behavior_autoencoder_v1.m1_screen import (
    _stable_seed,
    _x_normalizer,
    score_one_session,
)
from sua_exploration.h1_m1_priority_v1.core import (
    DirectRidgeSession,
    LAMBDA_GRID,
    array_sha256,
    regression_metrics,
)

from .frozen import FrozenDeployment
from .protocol import aligned_budget, full_session_arrays, latent_route_prediction

GCV_LAMBDA_GRID: tuple[float, ...] = tuple(value for value in LAMBDA_GRID if value > 0.0)
GAMMA_GRID: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)


def gcv_lambda_curves(
    x_std: np.ndarray, z: np.ndarray, lambdas: tuple[float, ...] = GCV_LAMBDA_GRID
) -> dict[str, Any]:
    """Closed-form GCV and exact LOO curves for the deployment ridge.

    The deployment estimator is the affine ridge whose slope carries the
    lambda*N penalty and whose intercept is unpenalized (fit_affine_ridge).
    Its exact linear smoother is S = A (A'A + P)^{-1} A' with A = [X, 1] and
    P = lambda*N * diag(1, ..., 1, 0), so the leave-one-out residuals have the
    exact closed form e_i / (1 - S_ii) (verified against brute-force refits in
    the tests).  GCV replaces S_ii by the trace average 1 - tr(S)/N.
    """

    x_std = np.asarray(x_std, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    n, d = x_std.shape
    need(n > d + 1 and z.shape[0] == n and np.isfinite(x_std).all() and np.isfinite(z).all(), "invalid GCV input")
    augmented = np.concatenate((x_std, np.ones((n, 1), dtype=np.float64)), axis=1)
    gram = augmented.T @ augmented
    atz = augmented.T @ z
    penalty = np.eye(d + 1, dtype=np.float64)
    penalty[-1, -1] = 0.0  # the deployment ridge never penalizes the intercept
    curves: dict[str, dict[str, float]] = {}
    for value in lambdas:
        value = float(value)
        need(value > 0.0, "GCV is undefined at lambda = 0")
        matrix = gram + value * float(n) * penalty
        inverse = np.linalg.inv(matrix)
        beta = inverse @ atz
        fitted = augmented @ beta
        residual = z - fitted
        hat_diagonal = np.einsum("ij,jk,ik->i", augmented, inverse, augmented, optimize=True)
        need(np.all(hat_diagonal < 1.0 - 1.0e-9), "LOO denominator collapsed")
        dof = float(hat_diagonal.sum(dtype=np.float64))
        gcv_denominator = 1.0 - dof / float(n)
        need(gcv_denominator > 1.0e-12, "GCV denominator collapsed")
        gcv = float(np.square(residual / gcv_denominator).mean(dtype=np.float64))
        loo = float(np.square(residual / (1.0 - hat_diagonal)[:, None]).mean(dtype=np.float64))
        curves[str(value)] = {
            "gcv_mean_squared_error": gcv,
            "exact_loo_mean_squared_error": loo,
            "effective_degrees_of_freedom": dof,
        }
    best_gcv = min(curves, key=lambda key: (curves[key]["gcv_mean_squared_error"], float(key)))
    best_loo = min(curves, key=lambda key: (curves[key]["exact_loo_mean_squared_error"], float(key)))
    return {
        "curves": curves,
        "selected_lambda_gcv": float(best_gcv),
        "selected_lambda_exact_loo": float(best_loo),
        "rows": int(n),
        "features": int(d),
    }


def _latent_ridge_weights(
    x: np.ndarray,
    y: np.ndarray,
    manifold,
    x_mean: np.ndarray,
    x_scale: np.ndarray,
    *,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    latent = manifold.encode(y)
    return fit_affine_ridge((x - x_mean) / x_scale, latent, ridge_lambda=ridge_lambda)


def _shrunk_query_prediction(
    session: DirectRidgeSession,
    manifold,
    x_mean: np.ndarray,
    x_scale: np.ndarray,
    *,
    budget: int,
    target_weight: np.ndarray,
    target_intercept: np.ndarray,
    source_weight: np.ndarray,
    source_intercept: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    query_x, query_y, _ = aligned_budget(session, budget, split="query")
    weight = gamma * target_weight + (1.0 - gamma) * source_weight
    intercept = gamma * target_intercept + (1.0 - gamma) * source_intercept
    latent_query = predict_affine_ridge((query_x - x_mean) / x_scale, weight, intercept)
    return manifold.decode(latent_query), query_y


def _inner_manifolds_for_target(
    sessions: Mapping[str, DirectRidgeSession],
    frozen_target,
    seed_row,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Deterministic nested inner manifolds: validation session excluded from its own fit."""

    spec = CalibrationAwareSpec(**seed_row.spec)
    inner: dict[str, Any] = {}
    for validation_name in frozen_target.sources:
        train_names = tuple(name for name in frozen_target.sources if name != validation_name)
        train = {name: sessions[name] for name in train_names}
        if seed_row.seed_offset is None:
            seed = _stable_seed("ca-inner", frozen_target.target, validation_name, dict(seed_row.spec))
        else:
            seed = _stable_seed(
                "ca-inner", int(seed_row.seed_offset), frozen_target.target, validation_name, dict(seed_row.spec)
            )
        manifold = fit_calibration_aware_manifold(train, spec, device=device, seed=seed)
        x_mean, x_scale = _x_normalizer(train)
        inner[validation_name] = {
            "manifold": manifold,
            "train_sessions": list(train_names),
            "x_mean": x_mean,
            "x_scale": x_scale,
            "state_sha256": manifold.fit_evidence["state_sha256"],
        }
    return inner


def _pooled_lambda(
    sessions: Mapping[str, DirectRidgeSession],
    frozen_target,
    seed_row,
    inner: Mapping[str, Any],
    *,
    lambda_grid: tuple[float, ...],
) -> dict[str, Any]:
    rows = []
    for value in lambda_grid:
        scores = [
            float(
                score_one_session(
                    sessions[name],
                    inner[name]["manifold"],
                    x_mean=inner[name]["x_mean"],
                    x_scale=inner[name]["x_scale"],
                    ridge_lambda=float(value),
                )[0]["pooled_variance_weighted_r2"]
            )
            for name in frozen_target.sources
        ]
        rows.append(
            {
                "ridge_lambda_per_sample": float(value),
                "source_validation_r2": scores,
                "equal_source_session_mean_r2": float(np.mean(scores, dtype=np.float64)),
            }
        )
    best_index = max(
        range(len(rows)), key=lambda index: (rows[index]["equal_source_session_mean_r2"], -index)
    )
    return {"candidate_table": rows, "selected_lambda": float(rows[best_index]["ridge_lambda_per_sample"])}


def _select_gamma(
    sessions: Mapping[str, DirectRidgeSession],
    frozen_target,
    seed_row,
    inner: Mapping[str, Any],
    *,
    gamma_grid: tuple[float, ...],
) -> dict[str, Any]:
    contexts = []
    for name in frozen_target.sources:
        context = inner[name]
        manifold = context["manifold"]
        x_mean, x_scale = context["x_mean"], context["x_scale"]
        support_x, support_y, _ = aligned_budget(sessions[name], 10, split="support")
        target_weight, target_intercept = _latent_ridge_weights(
            support_x, support_y, manifold, x_mean, x_scale, ridge_lambda=seed_row.ridge_lambda
        )
        others = [other for other in frozen_target.sources if other != name]
        full_x = []
        full_y = []
        for other in others:
            ox, oy, _ = full_session_arrays(sessions[other])
            full_x.append(ox)
            full_y.append(oy)
        source_weight, source_intercept = _latent_ridge_weights(
            np.concatenate(full_x),
            np.concatenate(full_y),
            manifold,
            x_mean,
            x_scale,
            ridge_lambda=seed_row.ridge_lambda,
        )
        contexts.append((name, manifold, x_mean, x_scale, target_weight, target_intercept, source_weight, source_intercept))
    rows = []
    for gamma in gamma_grid:
        scores = []
        for name, manifold, x_mean, x_scale, target_weight, target_intercept, source_weight, source_intercept in contexts:
            prediction, truth = _shrunk_query_prediction(
                sessions[name],
                manifold,
                x_mean,
                x_scale,
                budget=10,
                target_weight=target_weight,
                target_intercept=target_intercept,
                source_weight=source_weight,
                source_intercept=source_intercept,
                gamma=float(gamma),
            )
            scores.append(float(regression_metrics(truth, prediction)["pooled_variance_weighted_r2"]))
        rows.append(
            {
                "gamma": float(gamma),
                "source_validation_r2": scores,
                "equal_source_session_mean_r2": float(np.mean(scores, dtype=np.float64)),
            }
        )
    # Grid is ordered descending in gamma so an exact tie keeps the value closest
    # to the deployed estimator (gamma = 1).
    best_index = max(
        range(len(rows)), key=lambda index: (rows[index]["equal_source_session_mean_r2"], -index)
    )
    return {"candidate_table": rows, "selected_gamma": float(rows[best_index]["gamma"])}


def experiment_e5(
    frozen: FrozenDeployment,
    *,
    sessions: Mapping[str, DirectRidgeSession],
    device: str,
    oracle_q8_equal_session_mean: float | None = None,
) -> dict[str, Any]:
    device_object = torch.device(device)
    names = frozen.names
    variants = ("gcv_lambda", "pooled_lambda", "eb_shrinkage", "deployed_fixed_lambda")
    scores: dict[str, dict[str, float]] = {variant: {} for variant in variants}
    per_seed_rows: dict[str, Any] = {}
    for name in names:
        target = frozen.targets[name]
        seed_predictions: dict[str, list[np.ndarray]] = {variant: [] for variant in variants}
        truth_reference = None
        rows: list[dict[str, Any]] = []
        for seed in target.seeds:
            support_x, support_y, _ = aligned_budget(sessions[name], 10, split="support")
            latent_support = seed.manifold.encode(support_y)
            # (a) GCV on the support alone.
            gcv = gcv_lambda_curves(
                (support_x - target.x_mean) / target.x_scale, latent_support
            )
            # (b)+(c) need the nested inner manifolds.
            inner = _inner_manifolds_for_target(sessions, target, seed, device=device_object)
            pooled = _pooled_lambda(
                sessions, target, seed, inner, lambda_grid=tuple(float(v) for v in LAMBDA_GRID)
            )
            gamma = _select_gamma(sessions, target, seed, inner, gamma_grid=GAMMA_GRID)
            # (c) deployment side: shrink the target ridge toward the other-3 full-session regression.
            target_weight, target_intercept = _latent_ridge_weights(
                support_x, support_y, seed.manifold, target.x_mean, target.x_scale,
                ridge_lambda=seed.ridge_lambda,
            )
            full_x = []
            full_y = []
            for other in target.sources:
                ox, oy, _ = full_session_arrays(sessions[other])
                full_x.append(ox)
                full_y.append(oy)
            source_weight, source_intercept = _latent_ridge_weights(
                np.concatenate(full_x), np.concatenate(full_y), seed.manifold,
                target.x_mean, target.x_scale, ridge_lambda=seed.ridge_lambda,
            )
            eb_prediction, eb_truth = _shrunk_query_prediction(
                sessions[name], seed.manifold, target.x_mean, target.x_scale,
                budget=10,
                target_weight=target_weight, target_intercept=target_intercept,
                source_weight=source_weight, source_intercept=source_intercept,
                gamma=gamma["selected_gamma"],
            )
            gcv_prediction, gcv_truth, _, _ = latent_route_prediction(
                sessions[name], seed.manifold, target.x_mean, target.x_scale,
                budget=10, ridge_lambda=gcv["selected_lambda_gcv"],
            )
            pooled_prediction, pooled_truth, _, _ = latent_route_prediction(
                sessions[name], seed.manifold, target.x_mean, target.x_scale,
                budget=10, ridge_lambda=pooled["selected_lambda"],
            )
            deployed_prediction = seed.deployment_prediction
            for value, prediction in (
                ("gcv_lambda", gcv_prediction),
                ("pooled_lambda", pooled_prediction),
                ("eb_shrinkage", eb_prediction),
                ("deployed_fixed_lambda", deployed_prediction),
            ):
                seed_predictions[value].append(prediction)
            for prediction_truth in (gcv_truth, pooled_truth, eb_truth, seed.deployment_truth):
                if truth_reference is None:
                    truth_reference = prediction_truth
                else:
                    assert np.array_equal(truth_reference, prediction_truth), "query truth drift"
            rows.append(
                {
                    "seed_index": seed.seed_index,
                    "gcv": {
                        "selected_lambda": gcv["selected_lambda_gcv"],
                        "selected_lambda_exact_loo": gcv["selected_lambda_exact_loo"],
                        "effective_dof_at_selected": gcv["curves"][str(gcv["selected_lambda_gcv"])]["effective_degrees_of_freedom"],
                        "deployed_lambda": seed.ridge_lambda,
                    },
                    "pooled_lambda": pooled["selected_lambda"],
                    "eb_shrinkage": {
                        "selected_gamma": gamma["selected_gamma"],
                        "source_population_rows": int(sum(len(row) for row in full_y)),
                    },
                    "inner_manifolds": {
                        key: {
                            "train_sessions": value["train_sessions"],
                            "state_sha256": value["state_sha256"],
                        }
                        for key, value in inner.items()
                    },
                }
            )
            if device_object.type == "cuda":
                torch.cuda.empty_cache()
        assert truth_reference is not None
        per_seed_rows[name] = rows
        for variant in variants:
            prediction = np.mean(np.stack(seed_predictions[variant], axis=0), axis=0, dtype=np.float64)
            metrics = regression_metrics(truth_reference, prediction)
            scores[variant][name] = float(metrics["pooled_variance_weighted_r2"])

    result_variants: dict[str, Any] = {}
    deployed_array = np.asarray(
        [scores["deployed_fixed_lambda"][name] for name in names], dtype=np.float64
    )
    for variant in variants:
        values = np.asarray([scores[variant][name] for name in names], dtype=np.float64)
        delta = values - deployed_array
        row = {
            "equal_session_mean_r2": float(values.mean()),
            "per_session_r2": dict(zip(names, values.tolist())),
            "paired_delta_vs_deployed": {
                "mean": float(delta.mean()),
                "per_session": dict(zip(names, delta.tolist())),
                "positive_sessions": int((delta > 0.0).sum()),
                "total_sessions": len(names),
            },
        }
        if oracle_q8_equal_session_mean is not None:
            gap = oracle_q8_equal_session_mean - float(deployed_array.mean())
            row["oracle_gap_of_deployed"] = float(gap)
            row["oracle_gap_closed_fraction"] = (
                float(delta.mean()) / gap if abs(gap) > 1.0e-12 else None
            )
        result_variants[variant] = row
    return {
        "schema": "m1_behavior_manifold_v2_e5_estimator_upgrades_v1",
        "status": "COMPLETE_E5_LATENT_RIDGE_ESTIMATOR_UPGRADES",
        "variants": result_variants,
        "per_seed_selections": per_seed_rows,
        "selection_rules": {
            "gcv_lambda": "closed-form GCV on the target's own M10 support only (grid = frozen lambda grid without 0); fully deployable",
            "pooled_lambda": "one lambda per (target, seed) maximizing equal-source-session mean query R2 under nested inner-refit manifolds",
            "eb_shrinkage": "convex shrinkage of the M10 latent ridge toward the other-three-session full-data latent regression; gamma selected on the three sources under the same inner-refit rule (ties prefer gamma closest to 1)",
            "deployed_fixed_lambda": "the frozen published estimator (lambda = 0.3 per seed)",
        },
        "ambiguity_resolutions": {
            "lambda_of_reference_regressions": "the EB source-population latent regression uses each seed's deployed lambda",
            "gcv_zero_lambda": "lambda = 0 excluded from the GCV grid because the GCV denominator degenerates; smallest grid value is 1e-4",
            "inner_refits": "pooled-lambda and gamma selection rebuild the nested inner calibration-aware manifolds (validation session excluded) with the frozen seed rule, so no source query is scored through a manifold trained on it",
        },
        "disclosure": {
            "target_backward_steps": 0,
            "target_optimizer_steps": 0,
            "formal_heldout_opened": False,
            "minival_opened": False,
            "frozen_artifacts_changed": False,
            "gcv_uses_target_query": False,
            "pooled_lambda_and_gamma_use_target_query": False,
        },
    }
