"""Calib-only labeled hold-vs-reach contrasts. No fake angles. No absolute rate."""

from __future__ import annotations

import numpy as np

from . import plan


class ContrastError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContrastError(message)


def robust_z(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    _require(vector.size == plan.CHANNELS, "contrast vector must be one value per channel")
    median = float(np.median(vector))
    mad = float(np.median(np.abs(vector - median)))
    scale = max(mad * 1.4826, 1.0e-6)
    return np.ascontiguousarray((vector - median) / scale, dtype=np.float32)


def hold_reach_masks(
    angles: np.ndarray, *, horizon: int = plan.ACTIVITY_HORIZON
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(angles, dtype=np.float64).reshape(-1)
    _require(int(horizon) >= 2, "contrast horizon must be >= 2")
    _require(theta.size >= horizon, f"need first-{horizon} target angles")
    prefix = theta[:horizon]
    reach = np.isfinite(prefix)
    hold = ~reach
    _require(int(reach.sum()) >= 1, f"first-{horizon} has no directional trials")
    _require(int(hold.sum()) >= 1, f"first-{horizon} has no center/hold trials")
    return hold, reach


def contrast_from_trial_rates(
    spike_sums: np.ndarray,
    lengths: np.ndarray,
    angles: np.ndarray,
    *,
    horizon: int = plan.ACTIVITY_HORIZON,
) -> np.ndarray:
    """Return session-normalized [N, 4] = delta, log-ratio, hold-std, reach-std."""
    sums = np.asarray(spike_sums, dtype=np.float64)
    lens = np.asarray(lengths, dtype=np.float64).reshape(-1)
    _require(sums.ndim == 2 and sums.shape[1] == plan.CHANNELS, "spike sums must be [trials, 96]")
    _require(sums.shape[0] >= horizon, f"need first-{horizon} spike sums")
    _require(lens.size >= horizon, f"need first-{horizon} lengths")
    _require(np.all(lens[:horizon] > 0) and np.isfinite(sums[:horizon]).all(),
             "invalid calib spike sums")
    hold, reach = hold_reach_masks(angles, horizon=horizon)
    rates = sums[:horizon] / lens[:horizon, None]
    hold_rates = rates[hold]
    reach_rates = rates[reach]
    hold_mean = hold_rates.mean(axis=0)
    reach_mean = reach_rates.mean(axis=0)
    hold_std = hold_rates.std(axis=0, ddof=0) if hold_rates.shape[0] > 1 else np.zeros(plan.CHANNELS)
    reach_std = reach_rates.std(axis=0, ddof=0) if reach_rates.shape[0] > 1 else np.zeros(plan.CHANNELS)
    delta = reach_mean - hold_mean
    log_ratio = np.log1p(np.maximum(reach_mean, 0.0)) - np.log1p(np.maximum(hold_mean, 0.0))
    columns = np.stack(
        [robust_z(delta), robust_z(log_ratio), robust_z(hold_std), robust_z(reach_std)],
        axis=1,
    )
    _require(columns.shape == (plan.CHANNELS, plan.CONTRAST_DIM), "contrast shape drift")
    _require(np.isfinite(columns).all(), "non-finite hold-vs-reach contrast")
    return np.ascontiguousarray(columns, dtype=np.float32)


def shuffle_contrast(contrast: np.ndarray, *, seed: int = plan.SHUFFLE_SEED) -> np.ndarray:
    values = np.asarray(contrast, dtype=np.float32)
    _require(values.shape == (plan.CHANNELS, plan.CONTRAST_DIM), "shuffle contrast shape drift")
    permutation = np.random.RandomState(int(seed)).permutation(plan.CHANNELS)
    return np.ascontiguousarray(values[permutation], dtype=np.float32)
