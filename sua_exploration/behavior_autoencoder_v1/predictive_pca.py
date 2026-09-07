"""Closed-form predictive output basis for the M1 M10 comparison."""
from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from sua_exploration.h1_m1_priority_v1.core import DirectRidgeSession, array_sha256, canonical_sha256, direct_ridge_loso, regression_metrics

from .core import AutoencoderSpec, fit_affine_ridge, fit_pca_manifold, need, predict_affine_ridge
from .m1_screen import FittedManifold, _moments, _session_behavior, _x_normalizer, score_one_session


def fit_predictive_pca(
    sessions: Mapping[str, DirectRidgeSession],
    latent_dim: int,
    *,
    meta_ridge_lambda: float = 0.3,
) -> FittedManifold:
    names = tuple(sorted(sessions)); need(len(names) >= 2, "predictive PCA needs at least two source sessions")
    raw = np.concatenate([_session_behavior(sessions[name]) for name in names])
    behavior_mean, behavior_scale = _moments(raw); x_mean, x_scale = _x_normalizer(sessions)
    predictions = []
    for name in names:
        support_x, support_y, _ = sessions[name].aligned(0, split="support")
        query_x, _, _ = sessions[name].aligned(0, split="query")
        weight, intercept = fit_affine_ridge(
            (support_x - x_mean) / x_scale,
            (support_y - behavior_mean) / behavior_scale,
            ridge_lambda=meta_ridge_lambda,
        )
        predictions.append(predict_affine_ridge((query_x - x_mean) / x_scale, weight, intercept))
    predicted = np.concatenate(predictions)
    pca = fit_pca_manifold(predicted, latent_dim)
    evidence = {
        "kind": "predictive_pca", "source_sessions": list(names), "latent_dim": int(latent_dim),
        "meta_ridge_lambda_per_sample": float(meta_ridge_lambda), "predicted_source_rows": int(predicted.shape[0]),
        "predicted_source_sha256": array_sha256(predicted), "components_sha256": array_sha256(pca.components),
        "behavior_mean": behavior_mean.tolist(), "behavior_scale": behavior_scale.tolist(),
        "behavior_mean_sha256": array_sha256(behavior_mean), "behavior_scale_sha256": array_sha256(behavior_scale),
        "source_query_labels_used_to_fit_basis": False,
    }
    return FittedManifold(AutoencoderSpec("pca", latent_dim), behavior_mean, behavior_scale, pca, None, torch.device("cpu"), evidence)


def nested_predictive_pca_loso(
    sessions: Mapping[str, DirectRidgeSession],
    *,
    latent_dims: Sequence[int] = (2, 4, 6, 8, 10, 12, 14),
    deployment_lambda_grid: Sequence[float] = (0.1, 0.3),
) -> dict[str, Any]:
    names = tuple(sorted(sessions)); need(len(names) == 4, "predictive PCA requires four sessions")
    baseline = direct_ridge_loso(sessions, lag_grid=(0,), lambda_grid=deployment_lambda_grid)
    folds = {}; truth_rows = []; prediction_rows = []; started = time.perf_counter()
    for target_name in names:
        source_names = tuple(name for name in names if name != target_name); candidates = []
        for latent_dim in latent_dims:
            scores_by_lambda = {float(value): [] for value in deployment_lambda_grid}; fit_evidence = []
            for validation_name in source_names:
                train_names = tuple(name for name in source_names if name != validation_name); train = {name: sessions[name] for name in train_names}
                manifold = fit_predictive_pca(train, int(latent_dim)); x_mean, x_scale = _x_normalizer(train); validation_scores = {}
                for ridge_lambda in deployment_lambda_grid:
                    metrics, _, _ = score_one_session(sessions[validation_name], manifold, x_mean=x_mean, x_scale=x_scale, ridge_lambda=float(ridge_lambda))
                    value = float(metrics["pooled_variance_weighted_r2"]); scores_by_lambda[float(ridge_lambda)].append(value); validation_scores[str(float(ridge_lambda))] = value
                fit_evidence.append({"validation_session": validation_name, "train_sessions": list(train_names), "fit": manifold.fit_evidence, "validation_r2": validation_scores})
            fit_sha = canonical_sha256(fit_evidence)
            for ridge_lambda in deployment_lambda_grid:
                values = scores_by_lambda[float(ridge_lambda)]
                candidates.append({"latent_dim": int(latent_dim), "spec_name": f"predictive_pca_q{latent_dim}", "ridge_lambda_per_sample": float(ridge_lambda), "source_validation_r2": values, "equal_source_session_mean_r2": float(np.mean(values)), "inner_fits_sha256": fit_sha})
        winner_index = max(range(len(candidates)), key=lambda index: (candidates[index]["equal_source_session_mean_r2"], -index)); winner = candidates[winner_index]
        sources = {name: sessions[name] for name in source_names}; manifold = fit_predictive_pca(sources, int(winner["latent_dim"])); x_mean, x_scale = _x_normalizer(sources)
        metrics, truth, prediction = score_one_session(sessions[target_name], manifold, x_mean=x_mean, x_scale=x_scale, ridge_lambda=float(winner["ridge_lambda_per_sample"]))
        baseline_metrics = baseline["folds"][target_name]["target_metrics"]
        folds[target_name] = {"source_sessions": list(source_names), "target_excluded": True, "selected": winner, "candidate_table": candidates, "candidate_table_sha256": canonical_sha256(candidates), "final_fit": manifold.fit_evidence, "target_metrics": metrics, "baseline_target_metrics": baseline_metrics, "delta_vs_directridge": float(metrics["pooled_variance_weighted_r2"] - baseline_metrics["pooled_variance_weighted_r2"]), "input_path": sessions[target_name].input_path, "input_sha256": sessions[target_name].input_sha256}
        truth_rows.append(truth); prediction_rows.append(prediction)
    scores = np.asarray([folds[name]["target_metrics"]["pooled_variance_weighted_r2"] for name in names]); baseline_scores = np.asarray([folds[name]["baseline_target_metrics"]["pooled_variance_weighted_r2"] for name in names]); delta = scores - baseline_scores
    return {
        "schema": "m1_m10_source_frozen_predictive_pca_nested_loso_v1", "status": "COMPLETE_PREDICTIVE_PCA_SCREEN",
        "protocol": {"basis_input": "source query predictions from session-local M10 ridge; no source query labels", "outer_target_excluded": True, "inner_validation_excluded": True, "target_backward_steps": 0, "target_optimizer_steps": 0},
        "latent_dims": [int(value) for value in latent_dims], "deployment_lambda_grid": [float(value) for value in deployment_lambda_grid], "sessions": list(names), "output_names": list(sessions[names[0]].output_names), "folds": folds,
        "equal_session": {"mean_r2": float(scores.mean()), "median_r2": float(np.median(scores)), "per_session_r2": dict(zip(names, scores.tolist()))}, "directridge_equal_session": baseline["equal_session"],
        "paired_delta_vs_directridge": {"mean": float(delta.mean()), "median": float(np.median(delta)), "positive_sessions": int((delta > 0).sum()), "total_sessions": len(names), "per_session": dict(zip(names, delta.tolist()))},
        "pooled": regression_metrics(np.concatenate(truth_rows), np.concatenate(prediction_rows)), "target_used_for_selection": False, "elapsed_seconds": float(time.perf_counter() - started),
    }

