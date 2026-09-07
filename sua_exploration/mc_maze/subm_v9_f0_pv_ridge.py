"""Numerical primitives for the external sub-M F0/PV/ridge controls.

This module intentionally contains no NWB, checkpoint, result-root, or metric
I/O.  The companion runner owns those operations.  Keeping the numerical core
small makes the three control definitions inspectable and unit-testable:

* ``PV50`` is a standard baseline-subtracted, unit-preferred-direction
  population vector followed by an affine 2-D gain calibration;
* ``Ridge50`` is a causal, full 50-bin Wiener/ridge readout with a fixed
  normalized penalty; and
* both methods operate only on rows passed explicitly by the runner.

The controls use no optimization loop or gradient at calibration time.  Ridge
is solved once by its normal equations; CUDA is optional and only accelerates
that closed-form linear algebra.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


HISTORY_BINS = 50
BIN_SIZE_S = 0.020
RIDGE_NORMALIZED_LAMBDA = 1.0
FEATURE_STD_EPS = 1.0e-8
MODULATION_EPS = 1.0e-6


class ClassicalControlError(RuntimeError):
    """Raised when a frozen numerical/control invariant is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ClassicalControlError(message)


def validate_starts(
    neural: np.ndarray,
    starts: np.ndarray,
    *,
    window_size: int = HISTORY_BINS,
) -> np.ndarray:
    """Validate and return contiguous ``int64`` causal window starts."""
    array = np.asarray(starts)
    require(array.ndim == 1 and array.size > 0, "window starts must be a nonempty rank-1 array")
    require(np.issubdtype(array.dtype, np.integer), "window starts must be integral")
    array = np.ascontiguousarray(array, dtype=np.int64)
    require(neural.ndim == 2 and neural.shape[0] >= window_size, "neural must be [time,channels]")
    require(int(array.min()) >= 0, "negative window start")
    require(int(array.max()) + window_size <= neural.shape[0], "window reaches outside neural array")
    return array


def targets_at_window_end(behavior: np.ndarray, starts: np.ndarray, *, window_size: int = HISTORY_BINS) -> np.ndarray:
    """Return the decoder's last-timestep target for each causal window."""
    require(behavior.ndim == 2 and behavior.shape[1] == 2, "behavior must have shape [time,2]")
    starts = np.asarray(starts, dtype=np.int64)
    require(starts.size > 0 and int(starts.min()) >= 0, "invalid starts for behavior")
    end = starts + window_size - 1
    require(int(end.max()) < behavior.shape[0], "window target reaches outside behavior array")
    target = np.ascontiguousarray(behavior[end], dtype=np.float32)
    require(target.shape == (starts.size, 2) and np.isfinite(target).all(), "invalid window targets")
    return target


def raw_window_features(
    neural: np.ndarray,
    starts: np.ndarray,
    *,
    window_size: int = HISTORY_BINS,
) -> np.ndarray:
    """Materialize causal flattened ``[window_size * N]`` spike windows."""
    starts = validate_starts(neural, starts, window_size=window_size)
    offsets = np.arange(window_size, dtype=np.int64)
    values = neural[starts[:, None] + offsets[None, :]]
    features = np.ascontiguousarray(values.reshape(starts.size, -1), dtype=np.float32)
    require(np.isfinite(features).all(), "nonfinite raw window feature")
    return features


def prefix_sums(neural: np.ndarray) -> np.ndarray:
    """Return an exact float64 prefix sum suitable for arbitrary causal windows."""
    require(neural.ndim == 2 and np.isfinite(neural).all(), "neural must be finite [time,channels]")
    prefix = np.empty((neural.shape[0] + 1, neural.shape[1]), dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(neural, axis=0, dtype=np.float64, out=prefix[1:])
    return prefix


def window_rates_from_prefix(
    prefix: np.ndarray,
    starts: np.ndarray,
    *,
    window_size: int = HISTORY_BINS,
    bin_size_s: float = BIN_SIZE_S,
) -> np.ndarray:
    """Return window-mean rates [rows,N] without materializing 3-D windows."""
    starts = np.ascontiguousarray(np.asarray(starts), dtype=np.int64)
    require(prefix.ndim == 2 and starts.ndim == 1 and starts.size > 0, "invalid prefix/rate starts")
    require(int(starts.min()) >= 0 and int(starts.max()) + window_size < prefix.shape[0], "rate window out of bounds")
    require(bin_size_s > 0.0, "bin size must be positive")
    rates = (prefix[starts + window_size] - prefix[starts]) / (window_size * bin_size_s)
    rates = np.ascontiguousarray(rates, dtype=np.float64)
    require(np.isfinite(rates).all(), "nonfinite window rates")
    return rates


def batched(array: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    """Yield contiguous views of a rank-1 index array."""
    require(batch_size > 0, "batch size must be positive")
    for start in range(0, int(array.size), batch_size):
        yield array[start : start + batch_size]


@dataclass(frozen=True)
class PopulationVectorReadout:
    """Population directions/baselines and its calibration affine map."""

    preferred_direction: np.ndarray  # [N,2], unit normalized where modulation is nonzero
    baseline_rate: np.ndarray  # [N]
    gain: np.ndarray  # [2,2], p @ gain + intercept
    intercept: np.ndarray  # [2]
    calibration_rank: int
    zero_modulation_channels: int


def preferred_directions_from_cosine(
    a: np.ndarray,
    c: np.ndarray,
    m: np.ndarray,
    *,
    modulation_eps: float = MODULATION_EPS,
) -> tuple[np.ndarray, int]:
    """Construct unit preferred directions from T4's ``[a,c,m,b]`` terms."""
    a, c, m = (np.asarray(value, dtype=np.float64) for value in (a, c, m))
    require(a.ndim == c.ndim == m.ndim == 1 and a.shape == c.shape == m.shape, "cosine coefficient shape drift")
    require(np.isfinite(a).all() and np.isfinite(c).all() and np.isfinite(m).all(), "nonfinite cosine coefficient")
    active = m > modulation_eps
    preferred = np.zeros((m.size, 2), dtype=np.float64)
    if np.any(active):
        preferred[active, 0] = a[active] / m[active]
        preferred[active, 1] = c[active] / m[active]
    norms = np.linalg.norm(preferred[active], axis=1) if np.any(active) else np.empty(0)
    require(np.allclose(norms, 1.0, atol=2.0e-6, rtol=2.0e-6), "preferred-direction normalization drift")
    return preferred, int((~active).sum())


def population_vectors(rates: np.ndarray, preferred_direction: np.ndarray, baseline_rate: np.ndarray) -> np.ndarray:
    """Classical baseline-subtracted population vector for each window."""
    rates = np.asarray(rates, dtype=np.float64)
    preferred_direction = np.asarray(preferred_direction, dtype=np.float64)
    baseline_rate = np.asarray(baseline_rate, dtype=np.float64)
    require(rates.ndim == 2 and preferred_direction.shape == (rates.shape[1], 2), "PV channel shape drift")
    require(baseline_rate.shape == (rates.shape[1],), "PV baseline shape drift")
    vector = (rates - baseline_rate[None, :]) @ preferred_direction
    require(np.isfinite(vector).all(), "nonfinite population vector")
    return np.ascontiguousarray(vector, dtype=np.float64)


def fit_population_vector_gain(
    calibration_vectors: np.ndarray,
    calibration_targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit the classical 2-D affine gain/intercept using calibration rows only."""
    vector = np.asarray(calibration_vectors, dtype=np.float64)
    target = np.asarray(calibration_targets, dtype=np.float64)
    require(vector.ndim == target.ndim == 2 and vector.shape[1] == target.shape[1] == 2, "PV gain expects [rows,2]")
    require(vector.shape[0] == target.shape[0] and vector.shape[0] >= 3, "PV calibration row mismatch")
    design = np.column_stack((vector, np.ones(vector.shape[0], dtype=np.float64)))
    coefficients, _residual, rank, _singular = np.linalg.lstsq(design, target, rcond=None)
    require(int(rank) == 3, "PV affine calibration is rank deficient")
    gain = np.ascontiguousarray(coefficients[:2], dtype=np.float64)
    intercept = np.ascontiguousarray(coefficients[2], dtype=np.float64)
    require(np.isfinite(gain).all() and np.isfinite(intercept).all(), "nonfinite PV affine fit")
    return gain, intercept, int(rank)


def predict_population_vector(vectors: np.ndarray, gain: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    gain = np.asarray(gain, dtype=np.float64)
    intercept = np.asarray(intercept, dtype=np.float64)
    require(vectors.ndim == 2 and vectors.shape[1] == 2 and gain.shape == (2, 2) and intercept.shape == (2,), "PV prediction shape drift")
    prediction = vectors @ gain + intercept
    require(np.isfinite(prediction).all(), "nonfinite PV prediction")
    return np.ascontiguousarray(prediction, dtype=np.float32)


@dataclass(frozen=True)
class RidgeReadout:
    """Fixed-normalized-penalty linear Wiener/ridge readout."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: np.ndarray
    weights: np.ndarray
    normalized_lambda: float
    solver_device: str


def _ridge_cpu(features: np.ndarray, targets: np.ndarray, normalized_lambda: float) -> RidgeReadout:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < FEATURE_STD_EPS] = 1.0
    standardized = (x - mean) / scale
    target_mean = y.mean(axis=0)
    n_rows = standardized.shape[0]
    gram = (standardized.T @ standardized) / n_rows
    gram.flat[:: gram.shape[0] + 1] += normalized_lambda
    rhs = (standardized.T @ (y - target_mean)) / n_rows
    weights = np.linalg.solve(gram, rhs)
    require(np.isfinite(weights).all(), "nonfinite CPU ridge weights")
    return RidgeReadout(
        feature_mean=np.ascontiguousarray(mean, dtype=np.float32),
        feature_scale=np.ascontiguousarray(scale, dtype=np.float32),
        target_mean=np.ascontiguousarray(target_mean, dtype=np.float32),
        weights=np.ascontiguousarray(weights, dtype=np.float32),
        normalized_lambda=float(normalized_lambda),
        solver_device="cpu",
    )


def _ridge_cuda(features: np.ndarray, targets: np.ndarray, normalized_lambda: float, device: str) -> RidgeReadout:
    import torch

    require(torch.cuda.is_available(), "CUDA ridge requested but no CUDA device is available")
    runtime_device = torch.device(device)
    require(runtime_device.type == "cuda", "CUDA ridge device must be cuda:*")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)).to(runtime_device)
        y = torch.from_numpy(np.ascontiguousarray(targets, dtype=np.float32)).to(runtime_device)
        mean = x.mean(dim=0)
        scale = x.std(dim=0, correction=0)
        scale = torch.where(scale < FEATURE_STD_EPS, torch.ones_like(scale), scale)
        standardized = (x - mean) / scale
        target_mean = y.mean(dim=0)
        n_rows = int(standardized.shape[0])
        gram = (standardized.transpose(0, 1) @ standardized) / float(n_rows)
        gram.diagonal().add_(float(normalized_lambda))
        rhs = (standardized.transpose(0, 1) @ (y - target_mean)) / float(n_rows)
        factor, info = torch.linalg.cholesky_ex(gram, check_errors=False)
        require(int(info.max().item()) == 0, "CUDA ridge Gram is not positive definite")
        weights = torch.cholesky_solve(rhs, factor)
        require(bool(torch.isfinite(weights).all().item()), "nonfinite CUDA ridge weights")
        result = RidgeReadout(
            feature_mean=mean.detach().cpu().contiguous().numpy().astype(np.float32, copy=False),
            feature_scale=scale.detach().cpu().contiguous().numpy().astype(np.float32, copy=False),
            target_mean=target_mean.detach().cpu().contiguous().numpy().astype(np.float32, copy=False),
            weights=weights.detach().cpu().contiguous().numpy().astype(np.float32, copy=False),
            normalized_lambda=float(normalized_lambda),
            solver_device=str(runtime_device),
        )
    torch.cuda.synchronize(runtime_device)
    return result


def fit_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    normalized_lambda: float = RIDGE_NORMALIZED_LAMBDA,
    device: str = "cpu",
) -> RidgeReadout:
    """Fit one deterministic ridge readout with an unpenalized intercept.

    The penalty is applied to *standardized* features in the normalized objective

    ``mean(||Y - XW - b||^2) + lambda ||W||^2``.

    Thus ``lambda=1`` has the same meaning across a session's window count.
    """
    x = np.asarray(features)
    y = np.asarray(targets)
    require(x.ndim == 2 and y.ndim == 2 and x.shape[0] == y.shape[0] and y.shape[1] == 2, "ridge feature/target shape drift")
    require(x.shape[0] >= 3 and x.shape[1] > 0, "ridge requires nonempty calibration design")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "ridge input is nonfinite")
    require(np.isfinite(normalized_lambda) and normalized_lambda > 0.0, "ridge lambda must be finite and positive")
    if device.startswith("cuda"):
        return _ridge_cuda(x, y, float(normalized_lambda), device)
    require(device == "cpu", "ridge device must be cpu or cuda:*" )
    return _ridge_cpu(x, y, float(normalized_lambda))


def predict_ridge(features: np.ndarray, readout: RidgeReadout, *, device: str = "cpu") -> np.ndarray:
    """Apply a frozen ridge readout to rows that were never used in its fit."""
    x = np.asarray(features, dtype=np.float32)
    require(x.ndim == 2 and x.shape[1] == readout.feature_mean.size and np.isfinite(x).all(), "ridge prediction feature shape drift")
    if device.startswith("cuda"):
        import torch

        require(torch.cuda.is_available(), "CUDA ridge prediction requested but no CUDA device")
        runtime_device = torch.device(device)
        with torch.no_grad():
            tx = torch.from_numpy(np.ascontiguousarray(x)).to(runtime_device)
            mean = torch.from_numpy(readout.feature_mean).to(runtime_device)
            scale = torch.from_numpy(readout.feature_scale).to(runtime_device)
            weights = torch.from_numpy(readout.weights).to(runtime_device)
            target_mean = torch.from_numpy(readout.target_mean).to(runtime_device)
            value = ((tx - mean) / scale) @ weights + target_mean
            result = value.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        return np.ascontiguousarray(result)
    require(device == "cpu", "ridge device must be cpu or cuda:*")
    result = ((x - readout.feature_mean) / readout.feature_scale) @ readout.weights + readout.target_mean
    require(np.isfinite(result).all(), "nonfinite ridge prediction")
    return np.ascontiguousarray(result, dtype=np.float32)
