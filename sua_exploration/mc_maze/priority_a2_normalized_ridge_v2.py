"""Fail-closed normalized weighted ridge for Priority A2 version 2.

The fit is deliberately independent of the invalidated A2 scripts.  It uses a
weighted standardization and an explicit, unpenalized intercept.  In the
standardized coordinates ``Z`` it solves

    (Z.T @ diag(w) @ Z / sum(w) + lambda I) beta
        = Z.T @ diag(w) @ (Y - ybar_w) / sum(w).

The intercept is then the weighted first-order-condition solution.  Keeping
this module free of NWB/metric I/O makes the numerical contract testable before
any large dataset is opened.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np


FEATURE_STD_EPS = 1.0e-8
OUTPUT_DIM = 2


class PriorityA2NumericalError(RuntimeError):
    """Raised when an A2 numerical/provenance input violates the frozen contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PriorityA2NumericalError(message)


@dataclass(frozen=True)
class NormalizedWeightedRidge:
    """A CPU normalized weighted ridge readout with an unpenalized intercept."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    normalized_lambda: float
    total_weight: float
    solver_form: str


def _validated_fit_inputs(
    features: np.ndarray, targets: np.ndarray, weights: np.ndarray, normalized_lambda: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    require(x.ndim == 2 and x.shape[0] >= 1 and x.shape[1] >= 1, "features must be nonempty [rows, features]")
    require(y.shape == (x.shape[0], OUTPUT_DIM), "targets must have shape [rows,2]")
    require(w.shape == (x.shape[0],), "weights must have shape [rows]")
    require(np.isfinite(x).all(), "features must be finite")
    require(np.isfinite(y).all(), "targets must be finite")
    require(np.isfinite(w).all() and np.all(w > 0.0), "weights must be finite and strictly positive")
    require(math.isfinite(float(normalized_lambda)) and float(normalized_lambda) > 0.0, "lambda must be finite and positive")
    total = float(w.sum(dtype=np.float64))
    require(math.isfinite(total) and total > 0.0, "total weight must be finite and positive")
    return np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(w), total


def fit_normalized_weighted_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    normalized_lambda: float = 1.0,
) -> NormalizedWeightedRidge:
    """Fit normalized weighted ridge, with all inputs validated fail-closed."""
    x, y, w, total = _validated_fit_inputs(features, targets, weights, normalized_lambda)
    xbar = (w[:, None] * x).sum(axis=0, dtype=np.float64) / total
    ybar = (w[:, None] * y).sum(axis=0, dtype=np.float64) / total
    centered = x - xbar
    variance = (w[:, None] * centered * centered).sum(axis=0, dtype=np.float64) / total
    require(np.isfinite(variance).all() and np.all(variance >= 0.0), "weighted feature variance is invalid")
    scale = np.sqrt(variance)
    scale[scale < FEATURE_STD_EPS] = 1.0
    z = centered / scale
    yc = y - ybar
    # A makes the normalized weighted least-squares problem explicit:
    # min ||A beta - sqrt(w/W)Yc||² + lambda ||beta||².
    sqrt_weight = np.sqrt(w / total)
    a = z * sqrt_weight[:, None]
    weighted_target = yc * sqrt_weight[:, None]
    try:
        if a.shape[0] < a.shape[1]:
            # Dual form avoids a p×p solve for the deliberately high-dimensional
            # K<=16 A2b arms.  It is algebraically identical to the primal form.
            dual_gram = a @ a.T
            dual_gram.flat[:: dual_gram.shape[0] + 1] += float(normalized_lambda)
            coefficients = a.T @ np.linalg.solve(dual_gram, weighted_target)
            solver_form = "dual"
        else:
            primal_gram = a.T @ a
            primal_gram.flat[:: primal_gram.shape[0] + 1] += float(normalized_lambda)
            coefficients = np.linalg.solve(primal_gram, a.T @ weighted_target)
            solver_form = "primal"
    except np.linalg.LinAlgError as exc:  # lambda should normally make this impossible, but fail closed.
        raise PriorityA2NumericalError("normalized weighted ridge solve failed") from exc
    require(np.isfinite(coefficients).all(), "normalized weighted ridge coefficients are nonfinite")
    # Prediction is parameterized as ``((X - xbar) / scale) beta + intercept``.
    # In those centered coordinates the explicit unpenalized intercept is ybar.
    # (The algebraically equivalent raw-X intercept is ybar - (xbar/scale) @ beta.)
    intercept = ybar
    require(np.isfinite(intercept).all(), "normalized weighted ridge intercept is nonfinite")
    result = NormalizedWeightedRidge(
        feature_mean=np.ascontiguousarray(xbar, dtype=np.float64),
        feature_scale=np.ascontiguousarray(scale, dtype=np.float64),
        coefficients=np.ascontiguousarray(coefficients, dtype=np.float64),
        intercept=np.ascontiguousarray(intercept, dtype=np.float64),
        normalized_lambda=float(normalized_lambda),
        total_weight=total,
        solver_form=solver_form,
    )
    # Test the FOC on its normalized scale so multiplying every row weight does
    # not make an otherwise identical solve fail an absolute-tolerance check.
    residual_foc = (w[:, None] * (y - predict_normalized_weighted_ridge(x, result))).sum(axis=0) / total
    require(np.allclose(residual_foc, 0.0, rtol=0.0, atol=2.0e-12), "unpenalized intercept first-order condition failed")
    return result


def predict_normalized_weighted_ridge(features: np.ndarray, readout: NormalizedWeightedRidge) -> np.ndarray:
    """Predict with the explicit unpenalized intercept."""
    x = np.asarray(features, dtype=np.float64)
    require(x.ndim == 2 and x.shape[1] == readout.feature_mean.size, "prediction feature shape mismatch")
    require(np.isfinite(x).all(), "prediction features must be finite")
    value = ((x - readout.feature_mean) / readout.feature_scale) @ readout.coefficients + readout.intercept
    require(value.shape == (x.shape[0], OUTPUT_DIM) and np.isfinite(value).all(), "prediction is invalid")
    return np.ascontiguousarray(value, dtype=np.float64)


def fit_normalized_unweighted_reference(
    features: np.ndarray, targets: np.ndarray, *, normalized_lambda: float = 1.0
) -> NormalizedWeightedRidge:
    """Sealed-form normalized unweighted reference used only for reproduction gates."""
    x = np.asarray(features, dtype=np.float64)
    return fit_normalized_weighted_ridge(x, targets, np.ones(x.shape[0], dtype=np.float64), normalized_lambda=normalized_lambda)


def require_target_directions(trials: Sequence[Mapping[str, object]]) -> np.ndarray:
    """Return finite target directions; absent/non-numeric/nonfinite values are fatal."""
    require(len(trials) > 0, "support trials are empty")
    values: list[float] = []
    for index, trial in enumerate(trials):
        raw = trial.get("target_dir")
        try:
            value = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise PriorityA2NumericalError(f"support trial {index} has missing/non-numeric target_dir") from exc
        require(math.isfinite(value), f"support trial {index} has nonfinite target_dir")
        values.append(value)
    return np.ascontiguousarray(values, dtype=np.float64)


def assign_windows_to_trials(starts: np.ndarray, trials: Sequence[Mapping[str, object]], *, window_size: int) -> np.ndarray:
    """Map every causal window to exactly one trial and prove it stays inside it."""
    starts = np.asarray(starts)
    require(starts.ndim == 1 and starts.size > 0 and np.issubdtype(starts.dtype, np.integer), "starts must be nonempty integral rank-1")
    require(window_size > 0, "window size must be positive")
    bounds: list[tuple[int, int]] = []
    for index, trial in enumerate(trials):
        try:
            low, high = int(trial["start"]), int(trial["stop"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PriorityA2NumericalError(f"support trial {index} has invalid bounds") from exc
        require(low < high, f"support trial {index} has nonpositive duration")
        bounds.append((low, high))
    require(bounds, "support trials are empty")
    assigned = np.empty(starts.size, dtype=np.int64)
    for row, start_raw in enumerate(starts):
        start = int(start_raw)
        owners = [i for i, (low, high) in enumerate(bounds) if low <= start and start + window_size <= high]
        require(len(owners) == 1, f"calibration window {start} does not belong to exactly one support trial")
        assigned[row] = owners[0]
    return np.ascontiguousarray(assigned)


def direction_targets_for_windows(trials: Sequence[Mapping[str, object]], trial_of_window: np.ndarray) -> np.ndarray:
    directions = require_target_directions(trials)
    owner = np.asarray(trial_of_window, dtype=np.int64)
    require(owner.ndim == 1 and owner.size > 0 and int(owner.min()) >= 0 and int(owner.max()) < directions.size, "trial ownership is invalid")
    target = np.column_stack((np.cos(directions[owner]), np.sin(directions[owner])))
    require(np.isfinite(target).all(), "direction target construction is nonfinite")
    return np.ascontiguousarray(target, dtype=np.float64)


def equal_trial_weights(trial_of_window: np.ndarray, n_trials: int) -> np.ndarray:
    """Give every represented support trial the same *total* positive weight."""
    owner = np.asarray(trial_of_window, dtype=np.int64)
    require(owner.ndim == 1 and owner.size > 0 and n_trials > 0, "invalid equal-trial inputs")
    require(int(owner.min()) >= 0 and int(owner.max()) < n_trials, "trial owner is outside support")
    counts = np.bincount(owner, minlength=n_trials)
    require(np.all(counts > 0), "each support trial must retain at least one selected row")
    weights = 1.0 / counts[owner].astype(np.float64)
    require(np.isfinite(weights).all() and np.all(weights > 0.0), "equal-trial weights are invalid")
    totals = np.bincount(owner, weights=weights, minlength=n_trials)
    require(np.allclose(totals, totals[0], rtol=0.0, atol=1.0e-12), "equal-trial totals drift")
    return np.ascontiguousarray(weights)


def stable_trial_permutation(session_or_asset: str, trial_index: int, mask_seed: int, n_rows: int) -> np.ndarray:
    """Target-blind stable permutation keyed by session/asset, trial index, and seed."""
    require(isinstance(session_or_asset, str) and session_or_asset, "session/asset identifier is required")
    require(trial_index >= 0 and n_rows > 0, "invalid trial permutation inputs")
    digest = hashlib.sha256(f"priority-a2b-v2|{session_or_asset}|{trial_index}|{mask_seed}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:16], byteorder="little", signed=False)
    return np.ascontiguousarray(np.random.Generator(np.random.PCG64(seed)).permutation(n_rows), dtype=np.int64)


def nested_density_masks(trial_of_window: np.ndarray, n_trials: int, *, session_or_asset: str, mask_seed: int, ks: Sequence[int] = (1, 2, 4, 8, 16)) -> dict[int, np.ndarray]:
    """Return target-blind nested finite-K masks; ``all`` is represented by caller."""
    owner = np.asarray(trial_of_window, dtype=np.int64)
    require(owner.ndim == 1 and owner.size > 0, "invalid window ownership")
    require(tuple(ks) == tuple(sorted(set(ks))) and all(k > 0 for k in ks), "K values must be increasing distinct positives")
    require(int(owner.min()) >= 0 and int(owner.max()) < n_trials, "trial owner is outside support")
    masks = {int(k): np.zeros(owner.size, dtype=bool) for k in ks}
    for trial_index in range(n_trials):
        rows = np.flatnonzero(owner == trial_index)
        require(rows.size > 0, f"support trial {trial_index} has no calibration rows")
        require(rows.size >= max(ks), f"support trial {trial_index} has fewer than the required max K rows")
        ordered = rows[stable_trial_permutation(session_or_asset, trial_index, mask_seed, rows.size)]
        for k in ks:
            masks[int(k)][ordered[:int(k)]] = True
    previous: np.ndarray | None = None
    for k in ks:
        current = masks[int(k)]
        # Every trial supplies exactly K rows because the per-trial capacity was
        # checked above; spell this out instead of silently accepting a prefix.
        per_trial_count = np.bincount(owner[current], minlength=n_trials)
        require(np.all(per_trial_count == int(k)), "not every trial supplied exactly K rows")
        if previous is not None:
            require(np.all(~previous | current), "density masks are not nested")
        previous = current
    return {k: np.ascontiguousarray(v) for k, v in masks.items()}


def numerical_contract_self_test() -> dict[str, float]:
    """Small independent pre-data test; callers must execute it before loaders."""
    rng = np.random.default_rng(20260811)
    x = rng.normal(size=(31, 5))
    y = rng.normal(size=(31, 2)) + np.array([1.25, -0.75])
    w = rng.uniform(0.2, 3.0, size=31)
    uniform = np.ones(31)
    probe = rng.normal(size=(9, 5))
    weighted_uniform = fit_normalized_weighted_ridge(x, y, uniform)
    # Deliberately independent, local implementation of the sealed unweighted
    # equation.  Do not call another weighted helper here: this pre-data gate is
    # intended to catch a shared normalization bug before a large loader opens.
    reference_mean = x.mean(axis=0)
    reference_scale = x.std(axis=0)
    reference_scale[reference_scale < FEATURE_STD_EPS] = 1.0
    reference_z = (x - reference_mean) / reference_scale
    reference_ymean = y.mean(axis=0)
    reference_beta = np.linalg.solve(
        (reference_z.T @ reference_z) / x.shape[0] + np.eye(x.shape[1]),
        (reference_z.T @ (y - reference_ymean)) / x.shape[0],
    )
    reference_prediction = ((probe - reference_mean) / reference_scale) @ reference_beta + reference_ymean
    max_uniform_error = float(np.max(np.abs(predict_normalized_weighted_ridge(probe, weighted_uniform) - reference_prediction)))
    require(max_uniform_error <= 1.0e-10, "uniform-weight sealed normalized reproduction self-test failed")
    first = fit_normalized_weighted_ridge(x, y, w)
    scaled = fit_normalized_weighted_ridge(x, y, w * 17.0)
    max_scale_error = float(np.max(np.abs(predict_normalized_weighted_ridge(probe, first) - predict_normalized_weighted_ridge(probe, scaled))))
    require(max_scale_error <= 2.0e-12, "global weight-scale invariance self-test failed")
    foc = (w[:, None] * (y - predict_normalized_weighted_ridge(x, first))).sum(axis=0) / float(w.sum())
    max_intercept_foc = float(np.max(np.abs(foc)))
    require(max_intercept_foc <= 2.0e-9, "intercept FOC self-test failed")
    return {"uniform_reference_max_abs_error": max_uniform_error, "weight_scaling_max_abs_error": max_scale_error, "intercept_foc_max_abs_error": max_intercept_foc}
