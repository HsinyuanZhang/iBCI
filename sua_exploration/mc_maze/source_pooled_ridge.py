"""Source-pooled ridge (SPR) for subject-M fairness experiment — CPU only.

Extends the frozen A2a-v2 normalized weighted ridge with a source-pooled prior.
Part A audits channel correspondence; Parts B/C run scratch and SPR arms.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from sua_exploration.mc_maze import priority_a2_normalized_ridge_v2 as ridge
from sua_exploration.mc_maze.priority_a2_normalized_ridge_v2 import (
    FEATURE_STD_EPS,
    OUTPUT_DIM,
    NormalizedWeightedRidge,
    fit_normalized_weighted_ridge,
    predict_normalized_weighted_ridge,
)

HISTORY_BINS = 50
BUDGETS = (15, 30, 50)
VIEWS = ("sua", "pseudo_mua")
INTEGRITY_ATOL = 5.0e-5
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 20260812

# Frozen hyperparameter grids (predeclared; gamma=0 / lambda=1 recover scratch).
LAMBDA_GRID = (0.1, 1.0, 10.0)
GAMMA_GRID = (0.0, 0.01, 0.1, 1.0)

SEALED_SCRATCH_REFERENCES: dict[str, dict[int, float]] = {
    "ridge_dense_scratch": {
        "sua": {15: 0.0726, 30: 0.3222, 50: 0.4179},
        "pseudo_mua": {15: 0.1291, 30: 0.3414, 50: 0.4102},
    },
    "ridge_direction_scratch": {
        "sua": {15: -0.4599, 30: -0.2086, 50: -0.1220},
        "pseudo_mua": {15: -0.4244, 30: -0.1635, 50: -0.0879},
    },
}

SEALED_T4_REFERENCES: dict[str, dict[int, float]] = {
    "sua": {15: 0.3381, 30: 0.3582, 50: 0.3568},
    "pseudo_mua": {15: 0.2882, 30: 0.3053, 50: 0.3061},
}

A2A_V2_RECEIPT_SHA256 = "b6a080c48d36adc74050f0a6672623585320e32bada45001a962622379bcba58"


class SourcePooledRidgeError(RuntimeError):
    """Raised when SPR protocol invariants are violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SourcePooledRidgeError(message)


@dataclass(frozen=True)
class SessionChannelInfo:
    asset_id: str
    session_id: str
    view: str
    n_channels: int
    source_unit_count: int
    channel_ids: tuple[int, ...]
    electrode_ids_per_unit: tuple[int, ...] | None
    channel_ids_sha256: str


@dataclass(frozen=True)
class CorrespondenceAudit:
    view: str
    spr_definable: bool
    status: str
    n_sessions: int
    per_session: tuple[SessionChannelInfo, ...]
    channel_id_sets_intersection_size: int
    channel_id_sets_union_size: int
    stable_index_mapping_exists: bool
    correspondent_channel_count: int
    pairwise_channel_id_set_intersection_min: int
    pairwise_channel_id_set_intersection_max: int
    all_sessions_identical_channel_ids: bool
    feature_dimension_matches_across_sessions: bool


@dataclass(frozen=True)
class SessionCalibration:
    asset_id: str
    session_id: str
    support_starts: np.ndarray
    support_features: np.ndarray
    dense_targets: np.ndarray
    direction_targets: np.ndarray
    uniform_weights: np.ndarray
    query_starts: np.ndarray
    query_features_fn: Callable[[np.ndarray], np.ndarray]
    query_target: np.ndarray
    channel_ids: np.ndarray


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _channel_ids_tuple(channel_ids: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in np.asarray(channel_ids, dtype=np.int64).tolist())


def detect_correspondence(
    channel_id_rows: Sequence[np.ndarray],
    *,
    identifiers_are_positional_only: bool,
) -> tuple[bool, bool, int]:
    """Return (stable_mapping, all_identical, correspondent_count) on identifier rows."""
    require(len(channel_id_rows) > 0, "correspondence requires at least one session")
    arrays = [np.ascontiguousarray(row, dtype=np.int64) for row in channel_id_rows]
    dims_match = len({array.shape[0] for array in arrays}) == 1
    all_identical = all(np.array_equal(arrays[0], array) for array in arrays)
    intersection = set(arrays[0].tolist())
    union = set(arrays[0].tolist())
    for array in arrays[1:]:
        as_set = set(array.tolist())
        intersection &= as_set
        union |= as_set
    correspondent_count = len(intersection) if all_identical and dims_match else 0
    stable_mapping = bool(all_identical and dims_match and not identifiers_are_positional_only)
    if identifiers_are_positional_only:
        stable_mapping = False
        correspondent_count = 0
    return stable_mapping, all_identical, correspondent_count


def audit_view_correspondence(
    per_session: Sequence[SessionChannelInfo],
    *,
    view: str,
) -> CorrespondenceAudit:
    require(per_session, "audit requires sessions")
    require(all(row.view == view for row in per_session), "mixed views in audit")
    channel_rows = [np.asarray(row.channel_ids, dtype=np.int64) for row in per_session]
    positional_only = view == "sua"
    stable_mapping, all_identical, correspondent_count = detect_correspondence(
        channel_rows, identifiers_are_positional_only=positional_only
    )
    sets = [set(row.channel_ids) for row in per_session]
    intersection = set.intersection(*sets)
    union = set.union(*sets)
    pairwise_mins: list[int] = []
    pairwise_maxs: list[int] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = len(sets[i] & sets[j])
            pairwise_mins.append(overlap)
            pairwise_maxs.append(overlap)
    dims_match = len({row.n_channels for row in per_session}) == 1
    spr_definable = stable_mapping and correspondent_count > 0
    status = "SPR_DEFINABLE" if spr_definable else "SPR_UNDEFINED_NO_CHANNEL_CORRESPONDENCE"
    return CorrespondenceAudit(
        view=view,
        spr_definable=spr_definable,
        status=status,
        n_sessions=len(per_session),
        per_session=tuple(per_session),
        channel_id_sets_intersection_size=len(intersection),
        channel_id_sets_union_size=len(union),
        stable_index_mapping_exists=stable_mapping,
        correspondent_channel_count=correspondent_count,
        pairwise_channel_id_set_intersection_min=min(pairwise_mins) if pairwise_mins else len(intersection),
        pairwise_channel_id_set_intersection_max=max(pairwise_maxs) if pairwise_maxs else len(intersection),
        all_sessions_identical_channel_ids=all_identical,
        feature_dimension_matches_across_sessions=dims_match,
    )


def correspondence_audit_to_dict(audit: CorrespondenceAudit) -> dict[str, Any]:
    return {
        "view": audit.view,
        "spr_definable": audit.spr_definable,
        "status": audit.status,
        "n_sessions": audit.n_sessions,
        "channel_id_sets_intersection_size": audit.channel_id_sets_intersection_size,
        "channel_id_sets_union_size": audit.channel_id_sets_union_size,
        "stable_index_mapping_exists": audit.stable_index_mapping_exists,
        "correspondent_channel_count": audit.correspondent_channel_count,
        "pairwise_channel_id_set_intersection_min": audit.pairwise_channel_id_set_intersection_min,
        "pairwise_channel_id_set_intersection_max": audit.pairwise_channel_id_set_intersection_max,
        "all_sessions_identical_channel_ids": audit.all_sessions_identical_channel_ids,
        "feature_dimension_matches_across_sessions": audit.feature_dimension_matches_across_sessions,
        "per_session": [
            {
                "asset_id": row.asset_id,
                "session_id": row.session_id,
                "n_channels": row.n_channels,
                "source_unit_count": row.source_unit_count,
                "channel_ids_sha256": row.channel_ids_sha256,
                "channel_ids_head": list(row.channel_ids[:8]),
                "channel_ids_tail": list(row.channel_ids[-8:]) if row.n_channels > 8 else list(row.channel_ids),
                "electrode_ids_unique_count": (
                    len(set(row.electrode_ids_per_unit)) if row.electrode_ids_per_unit is not None else None
                ),
            }
            for row in audit.per_session
        ],
    }


def fit_prior_anchored_normalized_weighted_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    normalized_lambda: float = 1.0,
    gamma: float = 0.0,
    beta_prior: np.ndarray | None = None,
) -> NormalizedWeightedRidge:
    """Prior-anchored ridge in the A2a-v2 standardized coordinates."""
    if float(gamma) == 0.0:
        return fit_normalized_weighted_ridge(
            features, targets, weights, normalized_lambda=normalized_lambda
        )
    x, y, w, total = ridge._validated_fit_inputs(features, targets, weights, normalized_lambda)
    require(math.isfinite(float(gamma)) and float(gamma) >= 0.0, "gamma must be finite and nonnegative")
    xbar = (w[:, None] * x).sum(axis=0, dtype=np.float64) / total
    ybar = (w[:, None] * y).sum(axis=0, dtype=np.float64) / total
    centered = x - xbar
    variance = (w[:, None] * centered * centered).sum(axis=0, dtype=np.float64) / total
    require(np.isfinite(variance).all() and np.all(variance >= 0.0), "weighted feature variance is invalid")
    scale = np.sqrt(variance)
    scale[scale < FEATURE_STD_EPS] = 1.0
    z = centered / scale
    yc = y - ybar
    prior = np.zeros((x.shape[1], OUTPUT_DIM), dtype=np.float64) if beta_prior is None else np.asarray(
        beta_prior, dtype=np.float64
    )
    require(prior.shape == (x.shape[1], OUTPUT_DIM), "beta_prior shape mismatch")
    require(np.isfinite(prior).all(), "beta_prior must be finite")
    sqrt_weight = np.sqrt(w / total)
    a = z * sqrt_weight[:, None]
    weighted_target = yc * sqrt_weight[:, None]
    effective_lambda = float(normalized_lambda) + float(gamma)
    rhs = a.T @ weighted_target + float(gamma) * prior
    try:
        primal_gram = a.T @ a
        primal_gram.flat[:: primal_gram.shape[0] + 1] += effective_lambda
        coefficients = np.linalg.solve(primal_gram, rhs)
        solver_form = "primal_prior"
    except np.linalg.LinAlgError as exc:
        raise SourcePooledRidgeError("prior-anchored ridge solve failed") from exc
    require(np.isfinite(coefficients).all(), "prior-anchored coefficients are nonfinite")
    intercept = ybar
    result = NormalizedWeightedRidge(
        feature_mean=np.ascontiguousarray(xbar, dtype=np.float64),
        feature_scale=np.ascontiguousarray(scale, dtype=np.float64),
        coefficients=np.ascontiguousarray(coefficients, dtype=np.float64),
        intercept=np.ascontiguousarray(intercept, dtype=np.float64),
        normalized_lambda=float(normalized_lambda),
        total_weight=total,
        solver_form=solver_form,
    )
    residual_foc = (w[:, None] * (y - predict_normalized_weighted_ridge(x, result))).sum(axis=0) / total
    require(np.allclose(residual_foc, 0.0, rtol=0.0, atol=2.0e-12), "intercept FOC failed for prior-anchored ridge")
    return result


def fit_scratch_coefficients(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    normalized_lambda: float = 1.0,
) -> np.ndarray:
    readout = fit_normalized_weighted_ridge(features, targets, weights, normalized_lambda=normalized_lambda)
    return np.ascontiguousarray(readout.coefficients, dtype=np.float64)


def pool_beta_prior(source_coefficients: Sequence[np.ndarray]) -> np.ndarray:
    require(source_coefficients, "cannot pool an empty coefficient list")
    stacked = np.stack([np.asarray(value, dtype=np.float64) for value in source_coefficients], axis=0)
    require(stacked.ndim == 3, "source coefficients must be [n_source, n_features, 2]")
    shapes = {tuple(row.shape) for row in source_coefficients}
    require(len(shapes) == 1, "source coefficient shapes must match for pooling")
    return np.ascontiguousarray(stacked.mean(axis=0), dtype=np.float64)


def _arm_targets_and_weights(
    arm: str,
    dense_targets: np.ndarray,
    direction_targets: np.ndarray,
    uniform_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if arm in {"ridge_dense_scratch", "ridge_dense_spr"}:
        return dense_targets, uniform_weights
    if arm in {"ridge_direction_scratch", "ridge_direction_spr"}:
        return direction_targets, uniform_weights
    raise SourcePooledRidgeError(f"unknown arm: {arm}")


def score_readout_on_query(
    readout: NormalizedWeightedRidge,
    query_features_fn: Callable[[np.ndarray], np.ndarray],
    query_starts: np.ndarray,
    query_target: np.ndarray,
    *,
    recompute_r2: Callable[[np.ndarray, np.ndarray], float],
    batch_size: int = 2048,
) -> float:
    pieces: list[np.ndarray] = []
    for start in range(0, int(query_starts.size), batch_size):
        batch = query_starts[start : start + batch_size]
        pieces.append(predict_normalized_weighted_ridge(query_features_fn(batch), readout))
    prediction = np.ascontiguousarray(np.concatenate(pieces), dtype=np.float64)
    return float(recompute_r2(prediction.astype(np.float32), query_target))


def select_lambda_gamma_source_loso(
    source_sessions: Sequence[SessionCalibration],
    *,
    arm: str,
    budget: int,
    recompute_r2: Callable[[np.ndarray, np.ndarray], float],
) -> tuple[float, float, dict[str, Any]]:
    """Nested LOSO among source sessions only; never reads the target query."""
    require(source_sessions, "source LOSO requires source sessions")
    best_score = -math.inf
    best_lambda = LAMBDA_GRID[0]
    best_gamma = GAMMA_GRID[0]
    fold_records: list[dict[str, Any]] = []
    grid_records: list[dict[str, Any]] = []
    for normalized_lambda in LAMBDA_GRID:
        for gamma in GAMMA_GRID:
            fold_scores: list[float] = []
            for held_out_index, held_out in enumerate(source_sessions):
                train_sessions = [
                    session for index, session in enumerate(source_sessions) if index != held_out_index
                ]
                train_coefficients = []
                for session in train_sessions:
                    targets, weights = _arm_targets_and_weights(
                        arm, session.dense_targets, session.direction_targets, session.uniform_weights
                    )
                    train_coefficients.append(
                        fit_scratch_coefficients(
                            session.support_features, targets, weights, normalized_lambda=1.0
                        )
                    )
                beta_prior = pool_beta_prior(train_coefficients)
                targets, weights = _arm_targets_and_weights(
                    arm, held_out.dense_targets, held_out.direction_targets, held_out.uniform_weights
                )
                readout = fit_prior_anchored_normalized_weighted_ridge(
                    held_out.support_features,
                    targets,
                    weights,
                    normalized_lambda=normalized_lambda,
                    gamma=gamma,
                    beta_prior=beta_prior,
                )
                fold_score = score_readout_on_query(
                    readout,
                    held_out.query_features_fn,
                    held_out.query_starts,
                    held_out.query_target,
                    recompute_r2=recompute_r2,
                )
                fold_scores.append(fold_score)
                fold_records.append(
                    {
                        "lambda": normalized_lambda,
                        "gamma": gamma,
                        "held_out_asset_id": held_out.asset_id,
                        "validation_r2": fold_score,
                        "target_query_consumed": False,
                    }
                )
            mean_score = float(np.mean(fold_scores))
            grid_records.append(
                {
                    "lambda": normalized_lambda,
                    "gamma": gamma,
                    "mean_source_loso_r2": mean_score,
                    "per_fold_r2": fold_scores,
                }
            )
            if mean_score > best_score:
                best_score = mean_score
                best_lambda = normalized_lambda
                best_gamma = gamma
    provenance = {
        "selection_policy": "nested_leave_one_source_session_out",
        "target_query_consumed": False,
        "target_calibration_beyond_budget_consumed": False,
        "lambda_grid": list(LAMBDA_GRID),
        "gamma_grid": list(GAMMA_GRID),
        "selected_lambda": best_lambda,
        "selected_gamma": best_gamma,
        "selected_mean_source_loso_r2": best_score,
        "grid_evaluations": grid_records,
        "fold_records": fold_records,
        "budget": budget,
        "arm": arm,
        "n_source_sessions": len(source_sessions),
    }
    return best_lambda, best_gamma, provenance


def run_scratch_arm(
    session: SessionCalibration,
    *,
    arm: str,
    query_starts: np.ndarray,
    recompute_r2: Callable[[np.ndarray, np.ndarray], float],
) -> dict[str, Any]:
    targets, weights = _arm_targets_and_weights(
        arm, session.dense_targets, session.direction_targets, session.uniform_weights
    )
    readout = fit_normalized_weighted_ridge(session.support_features, targets, weights, normalized_lambda=1.0)
    score = score_readout_on_query(
        readout, session.query_features_fn, query_starts, session.query_target, recompute_r2=recompute_r2
    )
    return {
        "r2": score,
        "lambda": 1.0,
        "gamma": 0.0,
        "coefficients_sha256": sha256_array(readout.coefficients),
        "intercept": readout.intercept.tolist(),
        "solver_form": readout.solver_form,
    }


def run_spr_arm(
    target: SessionCalibration,
    source_sessions: Sequence[SessionCalibration],
    *,
    arm: str,
    budget: int,
    query_starts: np.ndarray,
    recompute_r2: Callable[[np.ndarray, np.ndarray], float],
    selection_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if selection_provenance is None:
        selected_lambda, selected_gamma, selection_provenance = select_lambda_gamma_source_loso(
            source_sessions,
            arm=arm,
            budget=budget,
            recompute_r2=recompute_r2,
        )
    else:
        selected_lambda = float(selection_provenance["selected_lambda"])
        selected_gamma = float(selection_provenance["selected_gamma"])
    source_coefficients = []
    for session in source_sessions:
        targets, weights = _arm_targets_and_weights(
            arm, session.dense_targets, session.direction_targets, session.uniform_weights
        )
        source_coefficients.append(
            fit_scratch_coefficients(session.support_features, targets, weights, normalized_lambda=1.0)
        )
    beta_prior = pool_beta_prior(source_coefficients)
    targets, weights = _arm_targets_and_weights(
        arm, target.dense_targets, target.direction_targets, target.uniform_weights
    )
    readout = fit_prior_anchored_normalized_weighted_ridge(
        target.support_features,
        targets,
        weights,
        normalized_lambda=selected_lambda,
        gamma=selected_gamma,
        beta_prior=beta_prior,
    )
    score = score_readout_on_query(
        readout, target.query_features_fn, query_starts, target.query_target, recompute_r2=recompute_r2
    )
    return {
        "r2": score,
        "lambda": selected_lambda,
        "gamma": selected_gamma,
        "beta_prior_sha256": sha256_array(beta_prior),
        "coefficients_sha256": sha256_array(readout.coefficients),
        "intercept": readout.intercept.tolist(),
        "solver_form": readout.solver_form,
        "selection_provenance": selection_provenance,
        "n_source_sessions": len(source_sessions),
    }


def paired_bootstrap_interval(
    deltas: Sequence[float],
    *,
    n_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    values = np.asarray(deltas, dtype=np.float64)
    require(values.ndim == 1 and values.size > 0, "bootstrap requires nonempty deltas")
    rng = np.random.default_rng(seed)
    means = np.empty(n_samples, dtype=np.float64)
    n = values.size
    for index in range(n_samples):
        sample = values[rng.integers(0, n, size=n)]
        means[index] = sample.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_contrast(
    *,
    view: str,
    budget: int,
    per_session_r2: Mapping[str, float],
    scratch_direction_r2: Mapping[str, float] | None,
    spr_direction_r2: Mapping[str, float] | None,
) -> dict[str, Any]:
    asset_ids = sorted(per_session_r2.keys())
    t4_values = np.asarray([SEALED_T4_REFERENCES[view][budget]] * len(asset_ids), dtype=np.float64)
    if spr_direction_r2 is not None:
        spr_values = np.asarray([spr_direction_r2[asset_id] for asset_id in asset_ids], dtype=np.float64)
        deltas_t4_minus_spr = t4_values - spr_values
        signs_t4 = int((deltas_t4_minus_spr > 0).sum())
        ci_t4 = paired_bootstrap_interval(deltas_t4_minus_spr.tolist())
    else:
        deltas_t4_minus_spr = None
        signs_t4 = None
        ci_t4 = None
    if scratch_direction_r2 is not None and spr_direction_r2 is not None:
        scratch_values = np.asarray([scratch_direction_r2[asset_id] for asset_id in asset_ids], dtype=np.float64)
        deltas_spr_minus_scratch = spr_values - scratch_values
        ci_prior = paired_bootstrap_interval(deltas_spr_minus_scratch.tolist())
    else:
        deltas_spr_minus_scratch = None
        ci_prior = None
    return {
        "view": view,
        "budget": budget,
        "t4_minus_ridge_direction_spr": (
            None
            if deltas_t4_minus_spr is None
            else {
                "mean": float(deltas_t4_minus_spr.mean()),
                "median": float(np.median(deltas_t4_minus_spr)),
                "positive_session_signs": signs_t4,
                "n_sessions": len(asset_ids),
                "bootstrap_95_ci": list(ci_t4),
            }
        ),
        "ridge_direction_spr_minus_ridge_direction_scratch": (
            None
            if deltas_spr_minus_scratch is None
            else {
                "mean": float(deltas_spr_minus_scratch.mean()),
                "median": float(np.median(deltas_spr_minus_scratch)),
                "positive_session_signs": int((deltas_spr_minus_scratch > 0).sum()),
                "n_sessions": len(asset_ids),
                "bootstrap_95_ci": list(ci_prior),
            }
        ),
    }


def verify_integrity_gate(
    arm: str,
    view: str,
    budget: int,
    observed_mean: float,
) -> dict[str, Any]:
    expected = SEALED_SCRATCH_REFERENCES[arm][view][budget]
    deviation = abs(float(observed_mean) - expected)
    passed = deviation <= INTEGRITY_ATOL
    return {
        "arm": arm,
        "view": view,
        "budget": budget,
        "expected": expected,
        "observed": float(observed_mean),
        "absolute_deviation": deviation,
        "atol": INTEGRITY_ATOL,
        "passed": passed,
    }


def build_session_channel_info(
    *,
    asset_id: str,
    session_id: str,
    view: str,
    record: Any,
    electrode_ids_per_unit: np.ndarray | None,
) -> SessionChannelInfo:
    channel_ids = np.asarray(record.channel_ids, dtype=np.int64)
    return SessionChannelInfo(
        asset_id=asset_id,
        session_id=session_id,
        view=view,
        n_channels=int(channel_ids.size),
        source_unit_count=int(record.source_unit_count or channel_ids.size),
        channel_ids=_channel_ids_tuple(channel_ids),
        electrode_ids_per_unit=(
            _channel_ids_tuple(electrode_ids_per_unit) if electrode_ids_per_unit is not None else None
        ),
        channel_ids_sha256=sha256_array(channel_ids),
    )
