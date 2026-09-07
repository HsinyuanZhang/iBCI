"""Trial-local smoothing kernels for the P2' decomposition matrix.

Two consumers share this module:

* the fixed OUTPUT filter (row A1 and every other non-raw row): a causal EMA
  over one completed query trial's governing-bin prediction sequence, reset at
  every trial boundary;
* the PSEUDO-direction constructions (sub-study of section 10.4): the same
  kernels applied to a completed trial's complementary-group velocity
  trajectory before the frozen displacement-integration pipeline.

Both consumers must never mix information across trials.  The API therefore
takes exactly ONE trial (one ``[P, C]`` block) per call; there is deliberately
no function that smooths a concatenated window stream.  The reset is a
structural property of the call boundary, not a flag.
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np

from . import plan


class P2PrimeFilterError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise P2PrimeFilterError(message)


Any2D = Union[np.ndarray, Sequence[Sequence[float]]]

ALPHA = float(plan.OUTPUT_FILTER["alpha"])


def ema_causal_one_trial(x: Any2D, *, alpha: float = ALPHA) -> np.ndarray:
    """Causal EMA over the rows of one trial's block.

    out[k] = alpha * x[k] + (1 - alpha) * out[k-1], out[0] = x[0].
    Row k reads rows <= k only; nothing outside the supplied block is read,
    so the trial boundary is a hard reset by construction.
    """
    values = np.asarray(x, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[0] >= 1, "one-trial EMA needs a nonempty [P, C] block")
    _require(0.0 < alpha <= 1.0, "EMA alpha must be in (0, 1]")
    out = np.empty_like(values)
    out[0] = values[0]
    retention = 1.0 - alpha
    for index in range(1, values.shape[0]):
        out[index] = alpha * values[index] + retention * out[index - 1]
    _require(bool(np.isfinite(out).all()), "causal EMA produced nonfinite output")
    return out


def ema_zero_phase_one_trial(x: Any2D, *, alpha: float = ALPHA) -> np.ndarray:
    """Trial-complete zero-phase two-pass EMA with no padding.

    out = EMA_forward then EMA_backward over the same block.  The whole trial
    is past at carrier-update time (section 10.4), so reading the trial's own
    future rows is legal for the pseudo-direction construction.  Both passes
    are confined to the supplied block, so the trial boundary is again a hard
    reset.
    """
    forward = ema_causal_one_trial(x, alpha=alpha)
    backward = ema_causal_one_trial(forward[::-1], alpha=alpha)[::-1]
    return np.ascontiguousarray(backward)


def apply_output_filter_one_trial(
    prediction: Any2D, *, alpha: float = ALPHA, enabled: bool = True,
) -> np.ndarray:
    """The one fixed output filter of the matrix, applied to one trial.

    ``enabled=False`` is the identity path used only by the raw anchor A0; it
    still normalizes to float64 so every row's metric runs on one dtype path.
    """
    values = np.asarray(prediction, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[0] >= 1, "output filter needs a nonempty [W, C] block")
    if not enabled:
        return values
    return ema_causal_one_trial(values, alpha=alpha)


def smooth_velocity_trajectory_one_trial(
    velocity: Any2D, *, construction: str, alpha: float = ALPHA,
) -> np.ndarray:
    """Apply one pseudo-direction construction to one group's trajectory.

    ``raw`` returns the trajectory unchanged (float64).  ``smoothed_causal``
    and ``smoothed_zero_phase`` apply the two kernels above.  ``true`` is NOT
    handled here -- it is a different input object, not a kernel.
    """
    values = np.asarray(velocity, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[1] == 2, "velocity trajectory must be [P, 2]")
    _require(construction in ("raw", "smoothed_causal", "smoothed_zero_phase"),
             f"unknown pseudo construction kernel: {construction}")
    if construction == "raw":
        return values
    if construction == "smoothed_causal":
        return ema_causal_one_trial(values, alpha=alpha)
    return ema_zero_phase_one_trial(values, alpha=alpha)


def true_physical_velocity_one_trial(
    behavior_rows: Any2D, *, behavior_mean: Sequence[float], behavior_std: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Restore the sealed z-scored behavior rows to physical velocity units.

    Returns ``(restored_readonly_array, padded_row_mask)``.  Mirrors
    ``restore_physical_velocity`` of the frozen CDM physical seam so the TRUE
    completed-trial direction is built from exactly the estimator's own units.
    Rows whose z-scored value is the parser's -1 padding become zero velocity
    (pre-registered defensive rule; the receipt records how often it triggers).
    """
    rows = np.asarray(behavior_rows, dtype=np.float64)
    mean = np.asarray(behavior_mean, dtype=np.float64)
    std = np.asarray(behavior_std, dtype=np.float64)
    _require(rows.ndim == 2 and rows.shape[1] == 2, "behavior rows must be [P, 2]")
    _require(mean.shape == std.shape == (2,) and bool(np.all(std > 0.0)),
             "sealed behavior normalizer must be [2] with positive std")
    padded = np.all(rows == -1.0, axis=1)
    restored = rows * std[None, :] + mean[None, :]
    restored[padded] = 0.0
    _require(bool(np.isfinite(restored).all()), "true velocity restoration became nonfinite")
    result = np.ascontiguousarray(restored)
    result.setflags(write=False)
    return result, padded
