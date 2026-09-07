"""Independent B1 official spectrogram metric and raw/log transforms."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error

from .constants import N_FREQ, N_SPEC_FRAMES, N_VALID, VALID_END, VALID_START


class MetricError(ValueError):
    pass


def valid_frame_mask_2d() -> np.ndarray:
    """Time-major [880, 158] boolean mask with frames [90, 790) True for all frequencies."""
    mask = np.zeros((N_SPEC_FRAMES, N_FREQ), dtype=bool)
    mask[VALID_START:VALID_END, :] = True
    return mask


def valid_frame_mask_freq_major() -> np.ndarray:
    return valid_frame_mask_2d().T.copy()


def normalize_signal(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def _fail_closed_trials(pred_trials: np.ndarray, tgt_trials: np.ndarray) -> None:
    if not np.isfinite(pred_trials).all() or not np.isfinite(tgt_trials).all():
        raise MetricError("non-finite prediction or target; fail-closed")
    for trial in range(pred_trials.shape[0]):
        pr = pred_trials[trial]
        tg = tgt_trials[trial]
        if np.max(pr) == np.min(pr):
            raise MetricError("constant prediction range is zero; fail-closed")
        if np.max(tg) == np.min(tg):
            raise MetricError("constant target range is zero; fail-closed")


def b1_official_metric(
    pred: np.ndarray,
    tgt: np.ndarray,
    mask: np.ndarray,
    *,
    fail_closed: bool = True,
) -> dict:
    """Independent reimplementation of FalconEvaluator.compute_metrics_spectrogram_distance.

    pred, tgt, mask: time-major [T, 158] with T = n_trials * 880.
    """
    pred = np.asarray(pred)
    tgt = np.asarray(tgt)
    mask = np.asarray(mask).astype(bool)
    if pred.ndim != 2 or tgt.ndim != 2 or mask.ndim != 2:
        raise MetricError(f"expected 2-D [T,158] arrays, got {pred.shape} {tgt.shape} {mask.shape}")
    if pred.shape != tgt.shape or pred.shape != mask.shape:
        raise MetricError(f"shape mismatch pred={pred.shape} tgt={tgt.shape} mask={mask.shape}")
    if pred.shape[1] != N_FREQ:
        raise MetricError(f"expected {N_FREQ} frequencies, got {pred.shape[1]}")
    if pred.shape[0] % N_SPEC_FRAMES != 0:
        raise MetricError(f"T={pred.shape[0]} is not a multiple of {N_SPEC_FRAMES}")

    if pred.shape != mask.shape:
        raise MetricError("predictions and masks have different lengths")
    masked_tgt = tgt[mask]
    masked_prd = pred[mask]
    n_trials = pred.shape[0] // N_SPEC_FRAMES
    expected = n_trials * N_VALID * N_FREQ
    if masked_tgt.size != expected or masked_prd.size != expected:
        raise MetricError(
            f"mask true-count {masked_prd.size} != n_trials*{N_VALID}*{N_FREQ}={expected}"
        )
    tgt_trials = masked_tgt.reshape(n_trials, N_VALID, N_FREQ)
    prd_trials = masked_prd.reshape(n_trials, N_VALID, N_FREQ)
    if fail_closed:
        _fail_closed_trials(prd_trials, tgt_trials)
    errors = [
        mean_squared_error(normalize_signal(tgt_trials[t]), normalize_signal(prd_trials[t]))
        for t in range(n_trials)
    ]
    errors = np.asarray(errors, dtype=np.float64)
    return {
        "MSE Mean": float(np.mean(errors)),
        "MSE Std.": float(np.std(errors)),
        "n_trials": int(n_trials),
        "per_trial_mse": errors,
    }


def official_metric_from_trials(pred_trials, tgt_trials, *, fail_closed: bool = True) -> dict:
    """pred_trials/tgt_trials: list or array of [158, 880] frequency-major trials."""
    pred_stream, tgt_stream, mask_stream = trials_to_time_major(pred_trials, tgt_trials)
    return b1_official_metric(pred_stream, tgt_stream, mask_stream, fail_closed=fail_closed)


def trials_to_time_major(pred_trials, tgt_trials=None):
    preds = [np.asarray(p) for p in pred_trials]
    for item in preds:
        if item.shape != (N_FREQ, N_SPEC_FRAMES):
            raise MetricError(f"trial spectrogram must be [158,880], got {item.shape}")
    pred_stream = np.concatenate([p.T for p in preds], axis=0)
    mask = np.concatenate([valid_frame_mask_2d() for _ in preds], axis=0)
    if tgt_trials is None:
        return pred_stream, mask
    tgts = [np.asarray(t) for t in tgt_trials]
    tgt_stream = np.concatenate([t.T for t in tgts], axis=0)
    return pred_stream, tgt_stream, mask


def log_from_raw(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(raw).all() or np.any(raw <= 0):
        raise MetricError("raw spectrogram must be finite and strictly positive")
    return np.log(raw)


def raw_from_log(log_spec: np.ndarray) -> np.ndarray:
    return np.exp(np.asarray(log_spec, dtype=np.float64))


def standardize(log_spec: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(log_spec, dtype=np.float64) - mean) / std


def inverse_standardize(z: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.asarray(z, dtype=np.float64) * std + mean


def standardized_log_to_raw(z: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return raw_from_log(inverse_standardize(z, mean, std))


def compare_official_bitwise(pred: np.ndarray, tgt: np.ndarray, mask: np.ndarray) -> dict:
    """Require bitwise identity with FalconEvaluator.compute_metrics_spectrogram_distance."""
    from falcon_challenge.evaluator import FalconEvaluator

    ours = b1_official_metric(pred, tgt, mask, fail_closed=True)
    packed_pred = pred[None, :, None, :]
    packed_tgt = tgt[None, :, None, :]
    packed_mask = mask[None, :, :]
    theirs = FalconEvaluator.compute_metrics_spectrogram_distance(packed_pred, packed_tgt, packed_mask)
    mean_equal = np.array(ours["MSE Mean"]).tobytes() == np.array(theirs["MSE Mean"], dtype=np.float64).tobytes()
    # sklearn / numpy may return python float vs np.float64; compare via float64 bit pattern of the value
    ours_mean = np.float64(ours["MSE Mean"])
    theirs_mean = np.float64(theirs["MSE Mean"])
    ours_std = np.float64(ours["MSE Std."])
    theirs_std = np.float64(theirs["MSE Std."])
    bitwise = bool(
        ours_mean.tobytes() == theirs_mean.tobytes() and ours_std.tobytes() == theirs_std.tobytes()
    )
    return {
        "bitwise_equal": bitwise,
        "ours": {"MSE Mean": float(ours_mean), "MSE Std.": float(ours_std)},
        "theirs": {"MSE Mean": float(theirs_mean), "MSE Std.": float(theirs_std)},
    }
