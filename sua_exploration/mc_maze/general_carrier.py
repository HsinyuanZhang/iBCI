"""Leakage-safe primitives for the development-only general-carrier Gate A.

This module deliberately does *not* implement a network side feature.  It is a
CPU screening layer for the falsifiable claim that a unit's instantaneous rate
can be described by a movement-aligned, low-dimensional behavioural carrier::

    r_i(t) = b_i + W_i y(t - lag) + epsilon_i(t).

The functions operate on already-selected time bins.  Callers must provide
chronologically disjoint fit/selection/evaluation masks; this module validates
that no response/label row crosses the supplied split boundary.  In particular,
it contains no trial-average helper -- using one for the RT task would destroy
the four within-trial reaches that motivated this audit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


GATE_A_PROTOCOL_VERSION = 1
BIN_SIZE_MS = 20
LAG_GRID_BINS: tuple[int, ...] = (-10, -5, 0, 5, 10)
RIDGE_ALPHA_GRID: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
PRACTICAL_R2_DELTA = 0.03
PRACTICAL_MSE_RATIO = 0.95


@dataclass(frozen=True)
class CarrierFit:
    """Per-unit encoding fit, where rows correspond to neural units."""

    weights: np.ndarray  # [units, behaviour_dimensions]
    intercept: np.ndarray  # [units]
    alpha: float
    lag_bins: int

    def features_2d(self) -> np.ndarray:
        """Return the planned [Wx, Wy, ||W||, b] 2-D carrier feature."""
        if self.weights.ndim != 2 or self.weights.shape[1] != 2:
            raise ValueError("features_2d requires exactly two behavioural dimensions")
        return np.column_stack(
            [
                self.weights[:, 0],
                self.weights[:, 1],
                np.linalg.norm(self.weights, axis=1),
                self.intercept,
            ]
        ).astype(np.float32)


def _as_2d(name: str, value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def ridge_with_intercept(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``y = x @ weights + intercept`` with an unpenalized intercept.

    ``alpha`` must be chosen without touching a later evaluation split.  The
    primal/dual switch is mathematically identical and keeps the direct-Wiener
    control inexpensive when there are many units.
    """
    x = _as_2d("x", x)
    y = _as_2d("y", y)
    if x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("ridge requires equal row counts and at least two rows")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be a finite non-negative value")
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    xc = x - x_mean
    yc = y - y_mean
    n_rows, n_features = xc.shape
    if alpha == 0:
        # The minimum-norm OLS solution is well-defined even for the direct
        # Wiener control's p>n calibration matrix; normal equations are not.
        weights, *_ = np.linalg.lstsq(xc, yc, rcond=None)
    elif n_features <= n_rows:
        gram = xc.T @ xc
        gram[np.diag_indices_from(gram)] += alpha
        weights = np.linalg.solve(gram, xc.T @ yc)
    else:
        gram = xc @ xc.T
        gram[np.diag_indices_from(gram)] += alpha
        weights = xc.T @ np.linalg.solve(gram, yc)
    intercept = y_mean - x_mean @ weights
    return weights, intercept


def predict_linear(x: np.ndarray, weights: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    x = _as_2d("x", x)
    weights = _as_2d("weights", weights)
    intercept = np.asarray(intercept, dtype=np.float64)
    if x.shape[1] != weights.shape[0] or intercept.shape != (weights.shape[1],):
        raise ValueError("linear prediction dimensions disagree")
    return x @ weights + intercept


def pair_indices(
    mask: np.ndarray, lag_bins: int, segment_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return response/label indices strictly contained in the same split mask.

    Positive ``lag_bins`` realizes ``r(t) ~ y(t-lag)`` exactly.  Requiring both
    indices to be in ``mask`` prevents a fit row from borrowing a behaviour
    label across the A/B chronological boundary.
    """
    mask = np.asarray(mask, dtype=bool)
    segment_ids = np.asarray(segment_ids)
    if mask.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    if segment_ids.ndim != 1 or len(segment_ids) != len(mask):
        raise ValueError("segment_ids must be one-dimensional and match mask length")
    response = np.flatnonzero(mask)
    labels = response - int(lag_bins)
    keep = (labels >= 0) & (labels < len(mask))
    response, labels = response[keep], labels[keep]
    keep = mask[labels] & (segment_ids[response] == segment_ids[labels])
    response, labels = response[keep], labels[keep]
    if len(response) < 10:
        raise ValueError("fewer than ten within-split lagged movement bins")
    return response, labels


def fit_encoding(
    rate: np.ndarray,
    behavior: np.ndarray,
    mask: np.ndarray,
    segment_ids: np.ndarray,
    *,
    lag_bins: int,
    alpha: float,
) -> CarrierFit:
    """Fit the movement-aligned per-unit encoding signature on one split only."""
    rate = _as_2d("rate", rate)
    behavior = _as_2d("behavior", behavior)
    if rate.shape[0] != behavior.shape[0] or len(mask) != rate.shape[0]:
        raise ValueError("rate, behavior, and mask must share a time axis")
    response, labels = pair_indices(mask, lag_bins, segment_ids)
    # y -> r gives weights [behaviour, units], stored as [units, behaviour].
    weights, intercept = ridge_with_intercept(behavior[labels], rate[response], alpha)
    return CarrierFit(weights=weights.T, intercept=intercept, alpha=float(alpha), lag_bins=int(lag_bins))


def predict_encoding(fit: CarrierFit, behavior: np.ndarray, response_indices: np.ndarray, label_indices: np.ndarray) -> np.ndarray:
    behavior = _as_2d("behavior", behavior)
    response_indices = np.asarray(response_indices, dtype=np.int64)
    label_indices = np.asarray(label_indices, dtype=np.int64)
    if response_indices.shape != label_indices.shape:
        raise ValueError("response and label index arrays must agree")
    if np.any(label_indices < 0) or np.any(label_indices >= len(behavior)):
        raise ValueError("label index out of bounds")
    return behavior[label_indices] @ fit.weights.T + fit.intercept


def mean_squared_error(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = _as_2d("observed", observed)
    predicted = _as_2d("predicted", predicted)
    if observed.shape != predicted.shape:
        raise ValueError("MSE inputs have incompatible shapes")
    return float(np.mean((observed - predicted) ** 2))


def select_encoding_hyperparameters(
    rate: np.ndarray,
    behavior: np.ndarray,
    fit_mask: np.ndarray,
    select_mask: np.ndarray,
    segment_ids: np.ndarray,
    *,
    lags: Iterable[int] = LAG_GRID_BINS,
    alphas: Iterable[float] = RIDGE_ALPHA_GRID,
) -> tuple[int, float, dict[str, float]]:
    """Choose lag/regularization on A only, using an earlier/later A split.

    The returned curve is diagnostic provenance.  No B/evaluation row is read
    here, which is intentionally testable through ``pair_indices``.
    """
    rate = _as_2d("rate", rate)
    behavior = _as_2d("behavior", behavior)
    if rate.shape[0] != behavior.shape[0]:
        raise ValueError("rate and behavior must share a time axis")
    if np.any(np.asarray(fit_mask, bool) & np.asarray(select_mask, bool)):
        raise ValueError("fit and selection masks must be disjoint")
    curve: dict[str, float] = {}
    best: tuple[float, int, float] | None = None
    for lag in lags:
        try:
            select_response, select_labels = pair_indices(select_mask, int(lag), segment_ids)
        except ValueError:
            continue
        for alpha in alphas:
            try:
                fit = fit_encoding(
                    rate, behavior, fit_mask, segment_ids, lag_bins=int(lag), alpha=float(alpha)
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            mse = mean_squared_error(rate[select_response], predict_encoding(fit, behavior, select_response, select_labels))
            key = f"lag={int(lag)},alpha={float(alpha):g}"
            curve[key] = mse
            candidate = (mse, int(lag), float(alpha))
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError("no valid within-A lag/alpha candidate")
    return best[1], best[2], curve


def carrier_inverse(fit: CarrierFit, rate: np.ndarray) -> np.ndarray:
    """Decode the carrier without fitting a neural-to-behaviour Wiener matrix.

    This is the least-squares inverse of the *encoding* signature.  It uses
    only the already-calibrated per-unit ``W,b`` values.  A later affine output
    calibration is allowed only on A and is kept separate in the audit.
    """
    rate = _as_2d("rate", rate)
    if rate.shape[1] != fit.weights.shape[0]:
        raise ValueError("rate unit dimension disagrees with carrier fit")
    residual = rate - fit.intercept[None, :]
    gram = fit.weights.T @ fit.weights
    # Tiny fixed numerical stabilizer; not selected on B.
    gram[np.diag_indices_from(gram)] += 1e-8
    return residual @ fit.weights @ np.linalg.inv(gram)


def fit_affine_output(raw_carrier: np.ndarray, behavior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the small carrier-output scale/intercept on A only."""
    return ridge_with_intercept(raw_carrier, behavior, alpha=0.0)


def variance_weighted_r2(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = _as_2d("prediction", prediction)
    target = _as_2d("target", target)
    if prediction.shape != target.shape:
        raise ValueError("R2 inputs have incompatible shapes")
    ss_res = np.sum((target - prediction) ** 2, axis=0)
    centered = target - target.mean(axis=0)
    ss_tot = np.sum(centered ** 2, axis=0)
    valid = ss_tot > 1e-12
    if not np.any(valid):
        raise ValueError("R2 target has no non-zero variance")
    return float(1.0 - np.sum(ss_res[valid]) / np.sum(ss_tot[valid]))


def deterministic_row_shuffle(fit: CarrierFit, seed: int) -> CarrierFit:
    """Move complete descriptor rows across units; raw rate rows are untouched."""
    rng = np.random.RandomState(int(seed))
    order = rng.permutation(fit.weights.shape[0])
    return CarrierFit(
        weights=fit.weights[order].copy(),
        intercept=fit.intercept[order].copy(),
        alpha=fit.alpha,
        lag_bins=fit.lag_bins,
    )


def deterministic_weight_shuffle(fit: CarrierFit, seed: int) -> CarrierFit:
    """Shuffle only behavioural weights, retaining every unit's firing baseline.

    This is the required mechanism null for the rate-prediction gate.  A full
    descriptor-row shuffle remains useful later as a network-side counterpart
    to TS4, but it confounds functional tuning with baseline-rate destruction.
    """
    rng = np.random.RandomState(int(seed))
    order = rng.permutation(fit.weights.shape[0])
    return CarrierFit(
        weights=fit.weights[order].copy(),
        intercept=fit.intercept.copy(),
        alpha=fit.alpha,
        lag_bins=fit.lag_bins,
    )
