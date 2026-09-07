"""H1 ridge-family arms for the fold-0 M4 fairness experiment.

All arms share the sealed v2r2 query contract, calibration containment, and dense
7-DoF velocity supervision.  Hyperparameter and PCA selection consume only the
four-trial calibration block; query windows are never used for fitting bases,
normalisation statistics, or model selection.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from sua_exploration.mc_maze.priority_a2_normalized_ridge_v2 import FEATURE_STD_EPS


WINDOW = 700
HISTORY_BINS_DEFAULT = 50
VELOCITY_DIM = 7
EXPECTED_NEURONS = 176
NORMALIZED_LAMBDA_SEALED = 1.0
SUPPORT_TRIALS = 4
EXPECTED_QUERY_SHA256 = "665fe535e90a221123b778171685577f67c2cc1902648cde2c8b2523e990e4da"
SEALED_POOLED_R2 = 0.25823473332303337
SEALED_PER_SESSION_R2: dict[str, float] = {
    "ses-19250101T111740": 0.23732910506183247,
    "ses-19250101T112404": 0.3175631653063281,
}
CARRIER_HSE5_POOLED_R2 = 0.500037

PCA_K_GRID: tuple[int, ...] = (8, 16, 32, 64)
W_GRID: tuple[int, ...] = (10, 25, 50)
LAMBDA_GRID: tuple[float, ...] = tuple(float(value) for value in np.logspace(-3.0, 3.0, 25))


class H1RidgeFamilyError(RuntimeError):
    """Raised when an H1 ridge-family contract is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise H1RidgeFamilyError(message)


def pooled_r2(truth: np.ndarray, estimate: np.ndarray) -> float:
    truth64 = np.asarray(truth, dtype=np.float64)
    estimate64 = np.asarray(estimate, dtype=np.float64)
    sse = float(np.square(truth64 - estimate64).sum())
    centered = truth64 - truth64.mean(axis=0, keepdims=True)
    tss = float(np.square(centered).sum())
    require(tss > 0.0, "R2 denominator is not positive")
    return 1.0 - sse / tss


def fitted_parameter_count(feature_dim: int, output_dim: int = VELOCITY_DIM) -> int:
    require(feature_dim > 0 and output_dim > 0, "invalid parameter count inputs")
    return int(feature_dim * output_dim)


def observations_per_parameter(calibration_rows: int, feature_dim: int) -> float:
    require(calibration_rows > 0 and feature_dim > 0, "invalid observations-per-parameter inputs")
    return float(calibration_rows) / float(feature_dim)


def supervision_coordinates(calibration_rows: int, output_dim: int = VELOCITY_DIM) -> int:
    return int(calibration_rows * output_dim)


def fit_normalized_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    normalized_lambda: float,
) -> dict[str, np.ndarray | float]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    weights = np.ones(x.shape[0], dtype=np.float64)
    total = float(weights.sum())
    x_mean = (weights[:, None] * x).sum(axis=0) / total
    y_mean = (weights[:, None] * y).sum(axis=0) / total
    centered = x - x_mean[None, :]
    variance = (weights[:, None] * centered * centered).sum(axis=0) / total
    scale = np.sqrt(variance)
    scale[scale < FEATURE_STD_EPS] = 1.0
    standardized = centered / scale[None, :]
    centered_y = y - y_mean[None, :]
    root_weight = np.sqrt(weights / total)
    design = standardized * root_weight[:, None]
    target = centered_y * root_weight[:, None]
    if design.shape[0] >= design.shape[1]:
        gram = design.T @ design
        gram.flat[:: gram.shape[0] + 1] += float(normalized_lambda)
        coefficient = np.linalg.solve(gram, design.T @ target)
        solver_form = "primal"
    else:
        gram = design @ design.T
        gram.flat[:: gram.shape[0] + 1] += float(normalized_lambda)
        coefficient = design.T @ np.linalg.solve(gram, target)
        solver_form = "dual"
    prediction = standardized @ coefficient + y_mean[None, :]
    intercept_foc = (weights[:, None] * (y - prediction)).sum(axis=0) / total
    foc_max = float(np.abs(intercept_foc).max())
    require(foc_max <= 1.0e-8, f"seven-output intercept FOC failed: {foc_max}")
    return {
        "x_mean": x_mean,
        "scale": scale,
        "coefficient": coefficient,
        "intercept": y_mean,
        "intercept_foc_max_abs_error": foc_max,
        "solver_form": solver_form,
        "feature_dim": int(x.shape[1]),
        "calibration_rows": int(x.shape[0]),
    }


def predict_normalized_ridge(features: np.ndarray, fit: Mapping[str, np.ndarray | float]) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    x_mean = np.asarray(fit["x_mean"], dtype=np.float64)
    scale = np.asarray(fit["scale"], dtype=np.float64)
    coefficient = np.asarray(fit["coefficient"], dtype=np.float64)
    intercept = np.asarray(fit["intercept"], dtype=np.float64)
    return ((x - x_mean[None, :]) / scale[None, :]) @ coefficient + intercept[None, :]


@dataclass(frozen=True)
class ChannelPCA:
    mean: np.ndarray
    components: np.ndarray

    @property
    def k(self) -> int:
        return int(self.components.shape[0])

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(self.mean).tobytes())
        digest.update(np.ascontiguousarray(self.components).tobytes())
        return digest.hexdigest()


def fit_channel_pca(neural_calibration_bins: np.ndarray, k: int) -> ChannelPCA:
    data = np.asarray(neural_calibration_bins, dtype=np.float64)
    require(data.ndim == 2 and data.shape[1] == EXPECTED_NEURONS, "PCA input must be [rows,176]")
    require(data.shape[0] >= k, "not enough calibration bins for requested PCA rank")
    mean = data.mean(axis=0)
    centered = data - mean[None, :]
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = np.ascontiguousarray(vt[: int(k)], dtype=np.float64)
    return ChannelPCA(mean=np.ascontiguousarray(mean, dtype=np.float64), components=components)


def project_neural(neural: np.ndarray, pca: ChannelPCA) -> np.ndarray:
    array = np.asarray(neural, dtype=np.float64)
    if array.ndim == 1:
        return np.ascontiguousarray((array - pca.mean) @ pca.components.T, dtype=np.float64)
    return np.ascontiguousarray((array - pca.mean[None, :]) @ pca.components.T, dtype=np.float64)


def history_feature(
    neural: np.ndarray,
    target_bin: int,
    *,
    history_bins: int,
    pca: ChannelPCA | None = None,
) -> np.ndarray:
    history = neural[target_bin - history_bins + 1 : target_bin + 1]
    require(history.shape == (history_bins, EXPECTED_NEURONS), "history shape mismatch")
    if pca is None:
        return np.ascontiguousarray(history.reshape(-1), dtype=np.float64)
    projected = project_neural(history, pca)
    return np.ascontiguousarray(projected.reshape(-1), dtype=np.float64)


def feature_dim(*, history_bins: int, pca_k: int | None = None) -> int:
    channels = int(pca_k) if pca_k is not None else EXPECTED_NEURONS
    return int(history_bins * channels)


@dataclass(frozen=True)
class SessionSplit:
    session_name: str
    neural: np.ndarray
    velocity: np.ndarray
    trial_num: np.ndarray
    eval_mask: np.ndarray
    support_values: tuple[float, ...]
    support_boundary: int
    calibration_target_bins: np.ndarray
    calibration_trial_ids: np.ndarray
    calibration_truth: np.ndarray
    query_output_bins: np.ndarray
    query_truth: np.ndarray
    pca_calibration_bins: np.ndarray


def _support_set(support_values: Sequence[float]) -> set[float]:
    return {float(value) for value in support_values}


def collect_calibration_targets(
    neural: np.ndarray,
    velocity: np.ndarray,
    trial_num: np.ndarray,
    eval_mask: np.ndarray,
    support_values: Sequence[float],
    *,
    history_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    support_set = _support_set(support_values)
    calibration_targets: list[int] = []
    calibration_trial_ids: list[float] = []
    for target_bin in range(history_bins - 1, neural.shape[0]):
        if not eval_mask[target_bin]:
            continue
        history_trials = trial_num[target_bin - history_bins + 1 : target_bin + 1]
        if not np.isfinite(history_trials).all():
            continue
        unique = np.unique(history_trials)
        if unique.size != 1 or float(unique[0]) not in support_set:
            continue
        calibration_targets.append(target_bin)
        calibration_trial_ids.append(float(unique[0]))
    targets = np.asarray(calibration_targets, dtype=np.int64)
    trial_ids = np.asarray(calibration_trial_ids, dtype=np.float64)
    truth = velocity[targets]
    require(targets.size > 0, "no legal calibration rows")
    return targets, trial_ids, truth


def collect_query_outputs(
    neural: np.ndarray,
    trial_num: np.ndarray,
    eval_mask: np.ndarray,
    *,
    support_boundary: int,
) -> np.ndarray:
    query_outputs: list[int] = []
    for start in range(support_boundary, neural.shape[0] - WINDOW + 1):
        output = start + WINDOW - 1
        if not eval_mask[output]:
            continue
        history_start = output - HISTORY_BINS_DEFAULT + 1
        require(history_start >= support_boundary, "query history overlaps support")
        query_outputs.append(output)
    outputs = np.asarray(query_outputs, dtype=np.int64)
    require(outputs.size > 0, "no legal query rows")
    return outputs


def collect_pca_calibration_bins(
    neural: np.ndarray,
    trial_num: np.ndarray,
    eval_mask: np.ndarray,
    support_values: Sequence[float],
) -> np.ndarray:
    support_set = _support_set(support_values)
    rows: list[np.ndarray] = []
    for bin_index in range(neural.shape[0]):
        if not eval_mask[bin_index]:
            continue
        trial_value = float(trial_num[bin_index])
        if not math.isfinite(trial_value) or trial_value not in support_set:
            continue
        rows.append(neural[bin_index])
    require(rows, "no calibration-block neural bins for PCA")
    return np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.float64)


def rebuild_split_for_history(base: SessionSplit, *, history_bins: int) -> SessionSplit:
    calibration_targets, calibration_trial_ids, calibration_truth = collect_calibration_targets(
        base.neural,
        base.velocity,
        base.trial_num,
        base.eval_mask,
        base.support_values,
        history_bins=history_bins,
    )
    return SessionSplit(
        session_name=base.session_name,
        neural=base.neural,
        velocity=base.velocity,
        trial_num=base.trial_num,
        eval_mask=base.eval_mask,
        support_values=base.support_values,
        support_boundary=base.support_boundary,
        calibration_target_bins=calibration_targets,
        calibration_trial_ids=calibration_trial_ids,
        calibration_truth=calibration_truth,
        query_output_bins=base.query_output_bins,
        query_truth=base.query_truth,
        pca_calibration_bins=base.pca_calibration_bins,
    )


def build_session_split(record: Any, *, history_bins: int = HISTORY_BINS_DEFAULT) -> SessionSplit:
    neural = np.asarray(record.neural, dtype=np.float64)
    velocity = np.asarray(record.velocity, dtype=np.float64)
    trial_num = np.asarray(record.trial_num)
    eval_mask = np.asarray(record.eval_mask, dtype=bool)
    support_values = tuple(float(value) for value in record.trial_values[:SUPPORT_TRIALS])
    calibration_targets, calibration_trial_ids, calibration_truth = collect_calibration_targets(
        neural,
        velocity,
        trial_num,
        eval_mask,
        support_values,
        history_bins=history_bins,
    )
    fifth_trial = float(record.trial_values[SUPPORT_TRIALS])
    fifth_bins = np.flatnonzero(eval_mask & np.isfinite(trial_num) & (trial_num == fifth_trial))
    require(fifth_bins.size > 0, "fifth trial has no eval-valid bin")
    support_boundary = int(fifth_bins[0])
    query_outputs = collect_query_outputs(
        neural,
        trial_num,
        eval_mask,
        support_boundary=support_boundary,
    )
    return SessionSplit(
        session_name=str(record.session_name),
        neural=neural,
        velocity=velocity,
        trial_num=trial_num,
        eval_mask=eval_mask,
        support_values=support_values,
        support_boundary=support_boundary,
        calibration_target_bins=calibration_targets,
        calibration_trial_ids=calibration_trial_ids,
        calibration_truth=calibration_truth,
        query_output_bins=query_outputs,
        query_truth=velocity[query_outputs],
        pca_calibration_bins=collect_pca_calibration_bins(neural, trial_num, eval_mask, support_values),
    )


def build_feature_matrix(
    split: SessionSplit,
    target_bins: np.ndarray,
    *,
    history_bins: int,
    pca: ChannelPCA | None = None,
) -> np.ndarray:
    rows = [
        history_feature(split.neural, int(target_bin), history_bins=history_bins, pca=pca)
        for target_bin in target_bins
    ]
    return np.stack(rows, axis=0)


def leave_one_trial_out_cv_score(
    features: np.ndarray,
    targets: np.ndarray,
    trial_ids: np.ndarray,
    *,
    normalized_lambda: float,
) -> float:
    unique_trials = np.unique(trial_ids)
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    for held_out in unique_trials:
        train_mask = trial_ids != held_out
        test_mask = ~train_mask
        require(train_mask.any() and test_mask.any(), "invalid LOO trial split")
        fit = fit_normalized_ridge(features[train_mask], targets[train_mask], normalized_lambda=normalized_lambda)
        predictions.append(predict_normalized_ridge(features[test_mask], fit))
        truths.append(targets[test_mask])
    return pooled_r2(np.concatenate(truths, axis=0), np.concatenate(predictions, axis=0))


def select_lambda(
    features: np.ndarray,
    targets: np.ndarray,
    trial_ids: np.ndarray,
    *,
    lambda_grid: Sequence[float] = LAMBDA_GRID,
) -> tuple[float, list[dict[str, float]]]:
    grid_rows: list[dict[str, float]] = []
    best_lambda = float(lambda_grid[0])
    best_score = -math.inf
    for candidate in lambda_grid:
        score = leave_one_trial_out_cv_score(features, targets, trial_ids, normalized_lambda=float(candidate))
        grid_rows.append({"lambda": float(candidate), "loo_calibration_r2": float(score)})
        if score > best_score + 1.0e-15 or (
            math.isclose(score, best_score, rel_tol=0.0, abs_tol=1.0e-15) and float(candidate) < best_lambda
        ):
            best_score = float(score)
            best_lambda = float(candidate)
    return best_lambda, grid_rows


def session_metrics_from_fit(
    *,
    split: SessionSplit,
    fit: Mapping[str, np.ndarray | float],
    query_features: np.ndarray,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    query_prediction = predict_normalized_ridge(query_features, fit)
    session_r2 = pooled_r2(split.query_truth, query_prediction)
    calibration_rows = int(split.calibration_target_bins.size)
    feature_dimension = int(fit["feature_dim"])
    return {
        "r2": float(session_r2),
        "calibration_rows": calibration_rows,
        "supervision_coordinates": supervision_coordinates(calibration_rows),
        "feature_dim": feature_dimension,
        "fitted_parameters": fitted_parameter_count(feature_dimension),
        "observations_per_parameter": observations_per_parameter(calibration_rows, feature_dimension),
        "query_windows": int(split.query_output_bins.size),
        "selection": dict(selection),
        "intercept_foc_max_abs_error": float(fit["intercept_foc_max_abs_error"]),
        "solver_form": str(fit["solver_form"]),
        "query_prediction": np.ascontiguousarray(query_prediction, dtype=np.float64),
    }


def run_ridge_v2r2_sealed(split: SessionSplit) -> dict[str, Any]:
    features = build_feature_matrix(
        split,
        split.calibration_target_bins,
        history_bins=HISTORY_BINS_DEFAULT,
        pca=None,
    )
    query_features = build_feature_matrix(
        split,
        split.query_output_bins,
        history_bins=HISTORY_BINS_DEFAULT,
        pca=None,
    )
    fit = fit_normalized_ridge(features, split.calibration_truth, normalized_lambda=NORMALIZED_LAMBDA_SEALED)
    selection = {
        "consumed": "fixed sealed comparator; no calibration-block selection",
        "history_bins": HISTORY_BINS_DEFAULT,
        "pca_k": None,
        "normalized_lambda": NORMALIZED_LAMBDA_SEALED,
    }
    return session_metrics_from_fit(split=split, fit=fit, query_features=query_features, selection=selection)


def run_ridge_pca(split: SessionSplit, *, pca_k: int) -> dict[str, Any]:
    pca = fit_channel_pca(split.pca_calibration_bins, pca_k)
    features = build_feature_matrix(
        split,
        split.calibration_target_bins,
        history_bins=HISTORY_BINS_DEFAULT,
        pca=pca,
    )
    query_features = build_feature_matrix(
        split,
        split.query_output_bins,
        history_bins=HISTORY_BINS_DEFAULT,
        pca=pca,
    )
    fit = fit_normalized_ridge(features, split.calibration_truth, normalized_lambda=NORMALIZED_LAMBDA_SEALED)
    selection = {
        "consumed": "PCA basis fit on eval-valid neural bins from the four support trials only",
        "pca_calibration_bins": int(split.pca_calibration_bins.shape[0]),
        "pca_basis_sha256": pca.digest(),
        "history_bins": HISTORY_BINS_DEFAULT,
        "pca_k": int(pca_k),
        "normalized_lambda": NORMALIZED_LAMBDA_SEALED,
    }
    return session_metrics_from_fit(split=split, fit=fit, query_features=query_features, selection=selection)


def run_ridge_lambda_cv(split: SessionSplit) -> dict[str, Any]:
    features = build_feature_matrix(
        split,
        split.calibration_target_bins,
        history_bins=HISTORY_BINS_DEFAULT,
        pca=None,
    )
    query_features = build_feature_matrix(
        split,
        split.query_output_bins,
        history_bins=HISTORY_BINS_DEFAULT,
        pca=None,
    )
    selected_lambda, grid_rows = select_lambda(features, split.calibration_truth, split.calibration_trial_ids)
    fit = fit_normalized_ridge(features, split.calibration_truth, normalized_lambda=selected_lambda)
    selection = {
        "consumed": "leave-one-support-trial-out CV on calibration rows only",
        "history_bins": HISTORY_BINS_DEFAULT,
        "pca_k": None,
        "lambda_grid": [float(value) for value in LAMBDA_GRID],
        "lambda_cv_rows": grid_rows,
        "selected_lambda": float(selected_lambda),
    }
    return session_metrics_from_fit(split=split, fit=fit, query_features=query_features, selection=selection)


def run_ridge_w_sweep(split: SessionSplit) -> dict[str, Any]:
    best_score = -math.inf
    best_history = int(W_GRID[0])
    best_lambda = float(LAMBDA_GRID[0])
    best_fit: dict[str, np.ndarray | float] | None = None
    best_query_features: np.ndarray | None = None
    active_split = split
    cv_rows: list[dict[str, Any]] = []
    for history_bins in W_GRID:
        session = rebuild_split_for_history(split, history_bins=history_bins)
        features = build_feature_matrix(session, session.calibration_target_bins, history_bins=history_bins, pca=None)
        selected_lambda, lambda_rows = select_lambda(features, session.calibration_truth, session.calibration_trial_ids)
        score = next(row["loo_calibration_r2"] for row in lambda_rows if row["lambda"] == selected_lambda)
        cv_rows.append(
            {
                "history_bins": int(history_bins),
                "selected_lambda": float(selected_lambda),
                "loo_calibration_r2": float(score),
                "calibration_rows": int(session.calibration_target_bins.size),
                "feature_dim": int(feature_dim(history_bins=history_bins)),
            }
        )
        if score > best_score + 1.0e-15 or (
            math.isclose(score, best_score, rel_tol=0.0, abs_tol=1.0e-15)
            and (history_bins < best_history or (history_bins == best_history and selected_lambda < best_lambda))
        ):
            best_score = float(score)
            best_history = int(history_bins)
            best_lambda = float(selected_lambda)
            active_split = session
            best_fit = fit_normalized_ridge(features, session.calibration_truth, normalized_lambda=selected_lambda)
            best_query_features = build_feature_matrix(
                session,
                session.query_output_bins,
                history_bins=history_bins,
                pca=None,
            )
    require(best_fit is not None and best_query_features is not None, "W sweep found no candidate")
    selection = {
        "consumed": "leave-one-support-trial-out CV on calibration rows only across W grid",
        "history_bins_grid": list(W_GRID),
        "lambda_grid": [float(value) for value in LAMBDA_GRID],
        "cv_rows": cv_rows,
        "selected_history_bins": int(best_history),
        "selected_lambda": float(best_lambda),
        "pca_k": None,
        "query_windows_fixed_to_sealed_contract": True,
    }
    return session_metrics_from_fit(
        split=active_split,
        fit=best_fit,
        query_features=best_query_features,
        selection=selection,
    )


def run_ridge_pca_lambda_cv(split: SessionSplit) -> dict[str, Any]:
    best_score = -math.inf
    best_k = int(PCA_K_GRID[0])
    best_lambda = float(LAMBDA_GRID[0])
    best_fit: dict[str, np.ndarray | float] | None = None
    best_query_features: np.ndarray | None = None
    best_pca: ChannelPCA | None = None
    cv_rows: list[dict[str, Any]] = []
    for pca_k in PCA_K_GRID:
        pca = fit_channel_pca(split.pca_calibration_bins, pca_k)
        features = build_feature_matrix(
            split,
            split.calibration_target_bins,
            history_bins=HISTORY_BINS_DEFAULT,
            pca=pca,
        )
        selected_lambda, lambda_rows = select_lambda(features, split.calibration_truth, split.calibration_trial_ids)
        score = next(row["loo_calibration_r2"] for row in lambda_rows if row["lambda"] == selected_lambda)
        cv_rows.append(
            {
                "pca_k": int(pca_k),
                "selected_lambda": float(selected_lambda),
                "loo_calibration_r2": float(score),
                "feature_dim": int(feature_dim(history_bins=HISTORY_BINS_DEFAULT, pca_k=pca_k)),
                "pca_basis_sha256": pca.digest(),
            }
        )
        if score > best_score + 1.0e-15 or (
            math.isclose(score, best_score, rel_tol=0.0, abs_tol=1.0e-15)
            and (pca_k < best_k or (pca_k == best_k and selected_lambda < best_lambda))
        ):
            best_score = float(score)
            best_k = int(pca_k)
            best_lambda = float(selected_lambda)
            best_pca = pca
            best_fit = fit_normalized_ridge(features, split.calibration_truth, normalized_lambda=selected_lambda)
            best_query_features = build_feature_matrix(
                split,
                split.query_output_bins,
                history_bins=HISTORY_BINS_DEFAULT,
                pca=pca,
            )
    require(best_fit is not None and best_query_features is not None and best_pca is not None, "PCA+lambda CV found no candidate")
    selection = {
        "consumed": (
            "PCA basis fit on eval-valid neural bins from the four support trials only; "
            "leave-one-support-trial-out CV on calibration rows for k and lambda"
        ),
        "pca_calibration_bins": int(split.pca_calibration_bins.shape[0]),
        "pca_k_grid": list(PCA_K_GRID),
        "lambda_grid": [float(value) for value in LAMBDA_GRID],
        "cv_rows": cv_rows,
        "selected_pca_k": int(best_k),
        "selected_lambda": float(best_lambda),
        "selected_pca_basis_sha256": best_pca.digest(),
        "history_bins": HISTORY_BINS_DEFAULT,
    }
    return session_metrics_from_fit(
        split=split,
        fit=best_fit,
        query_features=best_query_features,
        selection=selection,
    )


def window_manifest_sha256(rows: Sequence[tuple[str, int]]) -> str:
    digest = hashlib.sha256()
    for session_name, start in rows:
        digest.update(session_name.encode("ascii"))
        digest.update(np.int64(start).tobytes())
    return digest.hexdigest()


def aggregate_arm(per_session: Mapping[str, Mapping[str, Any]], *, query_truth: Mapping[str, np.ndarray], query_predictions: Mapping[str, np.ndarray]) -> dict[str, Any]:
    pooled_truth = np.concatenate([query_truth[name] for name in query_truth], axis=0)
    pooled_prediction = np.concatenate([query_predictions[name] for name in query_predictions], axis=0)
    calibration_rows = sum(int(row["calibration_rows"]) for row in per_session.values())
    feature_dim_total = sum(int(row["feature_dim"]) * int(row["calibration_rows"]) for row in per_session.values())
    weighted_feature_dim = feature_dim_total / calibration_rows if calibration_rows else 0.0
    fitted_parameters = sum(int(row["fitted_parameters"]) for row in per_session.values())
    return {
        "r2": float(pooled_r2(pooled_truth, pooled_prediction)),
        "calibration_rows": int(calibration_rows),
        "supervision_coordinates": supervision_coordinates(calibration_rows),
        "mean_feature_dim": float(weighted_feature_dim),
        "fitted_parameters": int(fitted_parameters),
        "observations_per_parameter": float(calibration_rows) / float(weighted_feature_dim) if weighted_feature_dim else 0.0,
        "query_windows": int(sum(int(row["query_windows"]) for row in per_session.values())),
    }


def run_arm(arm_name: str, splits: Mapping[str, SessionSplit]) -> dict[str, Any]:
    runners = {
        "ridge_v2r2_sealed": lambda split: run_ridge_v2r2_sealed(split),
        "ridge_pca_k8": lambda split: run_ridge_pca(split, pca_k=8),
        "ridge_pca_k16": lambda split: run_ridge_pca(split, pca_k=16),
        "ridge_pca_k32": lambda split: run_ridge_pca(split, pca_k=32),
        "ridge_pca_k64": lambda split: run_ridge_pca(split, pca_k=64),
        "ridge_lambda_cv": run_ridge_lambda_cv,
        "ridge_W_sweep": run_ridge_w_sweep,
        "ridge_pca_lambda_cv": run_ridge_pca_lambda_cv,
    }
    require(arm_name in runners, f"unknown arm {arm_name}")
    per_session: dict[str, Any] = {}
    query_truth: dict[str, np.ndarray] = {}
    query_predictions: dict[str, np.ndarray] = {}
    for name, split in splits.items():
        result = runners[arm_name](split)
        query_predictions[name] = np.asarray(result["query_prediction"], dtype=np.float64)
        query_truth[name] = split.query_truth
        per_session[name] = {key: value for key, value in result.items() if key != "query_prediction"}
    pooled = aggregate_arm(per_session, query_truth=query_truth, query_predictions=query_predictions)
    return {"arm": arm_name, "per_session": per_session, "pooled": pooled}


def all_arm_names() -> tuple[str, ...]:
    return (
        "ridge_v2r2_sealed",
        "ridge_pca_k8",
        "ridge_pca_k16",
        "ridge_pca_k32",
        "ridge_pca_k64",
        "ridge_lambda_cv",
        "ridge_W_sweep",
        "ridge_pca_lambda_cv",
    )


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
