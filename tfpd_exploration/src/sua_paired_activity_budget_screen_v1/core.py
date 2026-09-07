"""Pure numerical and protocol helpers for the paired activity-budget screen."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence

import numpy as np

from . import plan


class ScreenError(RuntimeError):
    """Raised when an input, chronology, or numerical contract drifts."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScreenError(message)


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def raw_array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def select_m4_support(theta_first30: Sequence[float]) -> np.ndarray:
    """Select four labelled directions from the fixed first-30 cue pool."""
    from sua_exploration.mc_maze.d_optimal_calibration_design import (
        greedy_forward_d_optimal_indices,
    )

    theta = np.asarray(theta_first30, dtype=np.float64).reshape(-1)
    require(theta.size == plan.ACTIVITY_HORIZON, "M4 cue pool must contain exactly 30 rows")
    candidates = np.flatnonzero(np.isfinite(theta)).astype(np.int64)
    require(candidates.size >= plan.M4, "M4 cue pool has fewer than four finite directions")
    local = greedy_forward_d_optimal_indices(theta[candidates], plan.M4)
    selected = np.sort(candidates[local]).astype(np.int64, copy=False)
    require(selected.size == plan.M4 and np.unique(selected).size == plan.M4,
            "M4 D-opt support cardinality drift")
    require(int(selected.min()) >= 0 and int(selected.max()) < plan.ACTIVITY_HORIZON,
            "M4 D-opt support left the first-30 cue pool")
    design = np.column_stack(
        (np.cos(theta[selected]), np.sin(theta[selected]), np.ones(plan.M4, dtype=np.float64))
    )
    require(int(np.linalg.matrix_rank(design)) == 3, "M4 D-opt design is rank deficient")
    return np.ascontiguousarray(selected)


def select_activity(
    calibration_first30: np.ndarray,
    *,
    selected_support: Sequence[int],
    activity_budget: int,
) -> np.ndarray:
    values = np.asarray(calibration_first30, dtype=np.float32)
    require(values.ndim == 3 and values.shape[0] == plan.ACTIVITY_HORIZON,
            "activity calibration must be [30,time,channels]")
    require(values.shape[1] == plan.TRIAL_LENGTH, "activity trial length drift")
    selected = np.asarray(selected_support, dtype=np.int64).reshape(-1)
    require(selected.size in (plan.M4, plan.M10), "unsupported support cardinality")
    require(np.unique(selected).size == selected.size, "support contains duplicate rows")
    require(int(selected.min()) >= 0 and int(selected.max()) < plan.ACTIVITY_HORIZON,
            "support left first-30 activity pool")
    if activity_budget == selected.size:
        return np.ascontiguousarray(values[selected])
    require(activity_budget == plan.ACTIVITY_HORIZON,
            "activity budget must equal selected M or the fixed first-30 horizon")
    return np.ascontiguousarray(values[: plan.ACTIVITY_HORIZON])


def variance_weighted_r2(target: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    require(y.ndim == 2 and y.shape[1] == 2 and p.shape == y.shape, "R2 shape drift")
    require(y.shape[0] >= 2 and np.isfinite(y).all() and np.isfinite(p).all(),
            "R2 requires finite two-coordinate rows")
    residual = np.square(y - p).sum(dtype=np.float64)
    centered = y - y.mean(axis=0, keepdims=True)
    total = np.square(centered).sum(dtype=np.float64)
    require(float(total) > 0.0, "target variance is zero")
    return float(1.0 - residual / total)


def summarize_rows(values: Mapping[str, Mapping[int, float]]) -> dict[str, object]:
    require(len(values) == plan.EXPECTED_SESSIONS, "summary session count drift")
    require(all(set(row) == set(plan.SEEDS) for row in values.values()),
            "summary seed topology drift")
    ordered_sessions = sorted(values)
    matrix = np.asarray(
        [[float(values[session][seed]) for seed in plan.SEEDS] for session in ordered_sessions],
        dtype=np.float64,
    )
    require(np.isfinite(matrix).all(), "summary contains nonfinite R2")
    return {
        "session_count": int(matrix.shape[0]),
        "seed_count": int(matrix.shape[1]),
        "equal_session_seed_mean": float(matrix.mean()),
        "equal_session_mean_after_seed_average": float(matrix.mean(axis=1).mean()),
        "equal_session_median_after_seed_average": float(np.median(matrix.mean(axis=1))),
        "seed_means": {str(seed): float(matrix[:, index].mean()) for index, seed in enumerate(plan.SEEDS)},
        "per_session_seed_r2": {
            session: {str(seed): float(values[session][seed]) for seed in plan.SEEDS}
            for session in ordered_sessions
        },
    }


def hierarchical_bootstrap(delta: np.ndarray) -> dict[str, object]:
    values = np.asarray(delta, dtype=np.float64)
    require(values.shape == (plan.EXPECTED_SESSIONS, len(plan.SEEDS)),
            "bootstrap expects [15,3]")
    require(np.isfinite(values).all(), "bootstrap contains nonfinite values")
    rng = np.random.Generator(np.random.PCG64(plan.BOOTSTRAP_SEED))
    draws = np.empty(plan.BOOTSTRAP_DRAWS, dtype=np.float64)
    for index in range(plan.BOOTSTRAP_DRAWS):
        sessions = rng.integers(0, values.shape[0], size=values.shape[0])
        seeds = rng.integers(0, values.shape[1], size=(values.shape[0], values.shape[1]))
        selected = values[sessions]
        selected = np.take_along_axis(selected, seeds, axis=1)
        draws[index] = selected.mean(dtype=np.float64)
    lower, upper = np.quantile(draws, (0.025, 0.975), method="linear")
    return {
        "seed": plan.BOOTSTRAP_SEED,
        "draws": plan.BOOTSTRAP_DRAWS,
        "lower_95": float(lower),
        "upper_95": float(upper),
    }


def paired_contrast(
    candidate: Mapping[str, Mapping[int, float]],
    reference: Mapping[str, Mapping[int, float]],
) -> dict[str, object]:
    require(set(candidate) == set(reference) and len(candidate) == plan.EXPECTED_SESSIONS,
            "paired session topology drift")
    ordered = sorted(candidate)
    matrix = np.asarray(
        [
            [float(candidate[session][seed] - reference[session][seed]) for seed in plan.SEEDS]
            for session in ordered
        ],
        dtype=np.float64,
    )
    session_means = matrix.mean(axis=1)
    seed_means = matrix.mean(axis=0)
    return {
        "equal_session_seed_mean": float(matrix.mean()),
        "equal_session_median_after_seed_average": float(np.median(session_means)),
        "positive_session_seed_cells": int(np.count_nonzero(matrix > 0.0)),
        "session_seed_cell_count": int(matrix.size),
        "positive_sessions_after_seed_average": int(np.count_nonzero(session_means > 0.0)),
        "session_count": int(session_means.size),
        "positive_seed_means": int(np.count_nonzero(seed_means > 0.0)),
        "seed_count": int(seed_means.size),
        "seed_means": {str(seed): float(seed_means[index]) for index, seed in enumerate(plan.SEEDS)},
        "per_session_seed_delta": {
            session: {str(seed): float(matrix[row, column]) for column, seed in enumerate(plan.SEEDS)}
            for row, session in enumerate(ordered)
        },
        "hierarchical_bootstrap_95": hierarchical_bootstrap(matrix),
    }

