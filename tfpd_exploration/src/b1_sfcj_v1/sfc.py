"""SFC4/SFC9/Direct159 closed-form fits, normalizer, Zero9/pad, lag selection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import r2_score

from .acoustic_basis import AcousticBasis
from .constants import (
    CARRIER_DIM,
    LAG_GRID_MS,
    N_CHANNELS,
    N_FREQ,
    SFC_RIDGE_LAMBDA,
    VALID_END,
    VALID_START,
)
from .data import rate_view, spec_frame_to_bin
from .metric import log_from_raw
from .util import sha256_array


def ridge_with_intercept(X: np.ndarray, Y: np.ndarray, lam: float):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    squeeze = False
    if Y.ndim == 1:
        Y = Y[:, None]
        squeeze = True
    n, p = X.shape
    Xd = np.concatenate([np.ones((n, 1)), X], axis=1)
    xtx = Xd.T @ Xd
    pen = np.zeros((p + 1, p + 1), dtype=np.float64)
    np.fill_diagonal(pen, lam)
    pen[0, 0] = 0.0
    coef = np.linalg.solve(xtx + pen, Xd.T @ Y)
    intercept = coef[0]
    weights = coef[1:]
    if squeeze:
        return intercept[0], weights[:, 0]
    return intercept, weights


def design_condition_number(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    Xd = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    return float(np.linalg.cond(Xd))


def aligned_rate_and_z(trials, basis: AcousticBasis, q: int, lag_ms: int):
    """Return r [T, 85] and z [T, q] on official valid frames, lagged on rate."""
    rs = []
    zs = []
    for trial in trials:
        rate = rate_view(trial.tx)  # [900, 85]
        log_spec = log_from_raw(trial.spectrogram)  # [158, 880]
        z_all = basis.project(log_spec.T, q)  # [880, q]
        for k in range(VALID_START, VALID_END):
            b = spec_frame_to_bin(k) - int(lag_ms)
            if b < 0 or b >= rate.shape[0]:
                raise ValueError(f"lag {lag_ms} maps frame {k} to invalid bin {b}")
            rs.append(rate[b])
            zs.append(z_all[k])
    return np.asarray(rs, dtype=np.float64), np.asarray(zs, dtype=np.float64)


@dataclass
class SFCFit:
    q: int
    lag_ms: int
    intercept: np.ndarray  # [85]
    weights: np.ndarray  # [85, q]
    padded: np.ndarray  # [85, 9] after optional normalize
    raw_vector: np.ndarray  # [85, q+1] intercept then weights
    condition_number: float
    design_rank: int
    coefficient_norm: float
    encoding_r2_mean: float
    finite: bool


def fit_sfc_on_trials(trials, basis: AcousticBasis, q: int, lag_ms: int, lam: float = SFC_RIDGE_LAMBDA) -> SFCFit:
    r, z = aligned_rate_and_z(trials, basis, q, lag_ms)
    intercepts = np.zeros(N_CHANNELS, dtype=np.float64)
    weights = np.zeros((N_CHANNELS, q), dtype=np.float64)
    preds = np.zeros_like(r)
    for i in range(N_CHANNELS):
        b, w = ridge_with_intercept(z, r[:, i], lam)
        intercepts[i] = b
        weights[i] = w
        preds[:, i] = b + z @ w
    r2s = [float(r2_score(r[:, i], preds[:, i])) for i in range(N_CHANNELS)]
    raw = np.concatenate([intercepts[:, None], weights], axis=1)
    cond = design_condition_number(z)
    rank = int(np.linalg.matrix_rank(np.concatenate([np.ones((z.shape[0], 1)), z], axis=1)))
    padded = pad_carrier(raw, q)
    return SFCFit(
        q=q,
        lag_ms=int(lag_ms),
        intercept=intercepts,
        weights=weights,
        padded=padded,
        raw_vector=raw,
        condition_number=cond,
        design_rank=rank,
        coefficient_norm=float(np.linalg.norm(weights)),
        encoding_r2_mean=float(np.mean(r2s)),
        finite=bool(np.isfinite(raw).all()),
    )


def pad_carrier(raw: np.ndarray, q: int) -> np.ndarray:
    """raw [N, q+1] -> [N, 9] with trailing zeros for SFC4. Zero9 is literal zeros."""
    out = np.zeros((raw.shape[0], CARRIER_DIM), dtype=np.float64)
    out[:, : q + 1] = raw
    return out


def zero9(n_units: int = N_CHANNELS) -> np.ndarray:
    return np.zeros((n_units, CARRIER_DIM), dtype=np.float64)


def standardize_coefficients(raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.where(std == 0.0, 1.0, std)
    return (raw - mean) / std


def fit_coefficient_normalizer(fits: list[SFCFit]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate([f.raw_vector for f in fits], axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0, ddof=0)
    std = np.where(std == 0.0, 1.0, std)
    return mean, std


def encoding_r2(trials, fit: SFCFit, basis: AcousticBasis) -> float:
    r, z = aligned_rate_and_z(trials, basis, fit.q, fit.lag_ms)
    preds = fit.intercept[None, :] + z @ fit.weights.T
    scores = [float(r2_score(r[:, i], preds[:, i])) for i in range(N_CHANNELS)]
    return float(np.mean(scores))


def split_half_report(first: SFCFit, second: SFCFit) -> dict:
    w1 = first.weights.reshape(-1)
    w2 = second.weights.reshape(-1)
    if w1.std() == 0 or w2.std() == 0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(w1, w2)[0, 1])
    cosines = []
    for i in range(first.weights.shape[0]):
        a, b = first.weights[i], second.weights[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            cosines.append(float("nan"))
        else:
            cosines.append(float(np.dot(a, b) / (na * nb)))
    cosines = np.asarray(cosines, dtype=np.float64)
    return {
        "flattened_W_correlation": corr,
        "median_per_channel_cosine": float(np.nanmedian(cosines)),
        "per_channel_cosine_sha256": sha256_array(np.nan_to_num(cosines, nan=0.0)),
    }


def derange_trials(trials):
    """Cyclic derangement of spectrograms vs neural (3-cycle when M=3)."""
    n = len(trials)
    if n < 2:
        raise ValueError("need at least 2 trials to derange")
    perm = [(i + 1) % n for i in range(n)]
    out = []
    for i, trial in enumerate(trials):
        src = trials[perm[i]]
        out.append(
            type(trial)(
                date=trial.date,
                split=trial.split,
                trial_index=trial.trial_index,
                path=trial.path,
                tx=trial.tx,
                timestamps=trial.timestamps,
                spectrogram=src.spectrogram,
                spectrogram_times=src.spectrogram_times,
                spectrogram_frequencies=src.spectrogram_frequencies,
                eval_mask=src.eval_mask,
                start_time=trial.start_time,
                stop_time=trial.stop_time,
            )
        )
    return out


def select_fold_lag(train_dates, load_m3, load_held, basis: AcousticBasis, q: int = 8) -> dict:
    """Per-date first-M3 fit, score on same-date calib 4..N; equal-weight dates; val date excluded."""
    per_lag = []
    for lag in LAG_GRID_MS:
        date_scores = []
        for date in train_dates:
            fit = fit_sfc_on_trials(load_m3(date), basis, q, lag)
            held = load_held(date)
            score = encoding_r2(held, fit, basis)
            date_scores.append({"date": date, "encoding_r2": score, "finite": fit.finite})
        mean_score = float(np.mean([d["encoding_r2"] for d in date_scores]))
        per_lag.append({"lag_ms": int(lag), "mean_encoding_r2": mean_score, "per_date": date_scores})
    # higher encoding R² is better; tie -> smaller lag
    best = sorted(per_lag, key=lambda r: (-r["mean_encoding_r2"], r["lag_ms"]))[0]
    return {"selected_lag_ms": best["lag_ms"], "grid": per_lag}


def fit_direct159(trials, basis: AcousticBasis, lag_ms: int, lam: float) -> dict:
    """Stage 3 diagnostic only: intercept + 158 standardized-log weights per unit."""
    r_list = []
    y_list = []
    for trial in trials:
        rate = rate_view(trial.tx)
        z_log = basis.transform_log(log_from_raw(trial.spectrogram).T)
        for k in range(VALID_START, VALID_END):
            b = spec_frame_to_bin(k) - int(lag_ms)
            r_list.append(rate[b])
            y_list.append(z_log[k])
    r = np.asarray(r_list)
    y = np.asarray(y_list)
    intercepts = np.zeros((N_CHANNELS, 1))
    weights = np.zeros((N_CHANNELS, N_FREQ))
    for i in range(N_CHANNELS):
        b, w = ridge_with_intercept(y, r[:, i], lam)
        intercepts[i] = b
        weights[i] = w
    return {
        "lag_ms": int(lag_ms),
        "lambda": float(lam),
        "intercept_shape": list(intercepts.shape),
        "weights_shape": list(weights.shape),
        "finite": bool(np.isfinite(weights).all()),
        "condition_number": design_condition_number(y),
    }
