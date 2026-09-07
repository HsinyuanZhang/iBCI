"""Source-prior SFC9 profiles for stable-index B1 channels."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import r2_score

from tfpd_exploration.src.b1_sfcj_v1 import sfc
from tfpd_exploration.src.b1_sfcj_v1.acoustic_basis import AcousticBasis

from .plan import PROFILE_GAMMA_GRID, PROFILE_Q


@dataclass(frozen=True)
class ProfileAuthority:
    lag_ms: int
    gamma: float
    coefficient_mean: np.ndarray
    coefficient_std: np.ndarray
    source_prior: np.ndarray
    raw_by_date: dict[str, np.ndarray]
    standardized_by_date: dict[str, np.ndarray]
    gamma_grid: tuple[dict, ...]


def _encoding_r2_for_raw(trials, basis: AcousticBasis, raw: np.ndarray, lag_ms: int) -> float:
    rates, z = sfc.aligned_rate_and_z(trials, basis, PROFILE_Q, lag_ms)
    pred = raw[:, 0][None, :] + z @ raw[:, 1:].T
    values = [float(r2_score(rates[:, i], pred[:, i])) for i in range(rates.shape[1])]
    return float(np.mean(values))


def fit_source_prior_profiles(
    *,
    train_dates: tuple[str, ...],
    all_dates: tuple[str, ...],
    basis: AcousticBasis,
    first_m3,
    held_calib,
) -> ProfileAuthority:
    """Fit M3 SFC9 then shrink toward stable-index source channel priors.

    Gamma is selected using only the outer-training dates.  For each training
    date, its prior is the mean profile of the other training date(s), and the
    score is measured on that same date's released calibration trials 4..N.
    The validation date never participates in gamma, lag, or normalization.
    """
    lag_report = sfc.select_fold_lag(train_dates, first_m3, held_calib, basis, q=PROFILE_Q)
    lag_ms = int(lag_report["selected_lag_ms"])
    base = {
        date: sfc.fit_sfc_on_trials(first_m3(date), basis, PROFILE_Q, lag_ms).raw_vector
        for date in all_dates
    }
    train_prior = np.mean(np.stack([base[d] for d in train_dates], axis=0), axis=0)

    grid = []
    for gamma in PROFILE_GAMMA_GRID:
        date_scores = []
        for date in train_dates:
            others = [base[d] for d in train_dates if d != date]
            prior = np.mean(np.stack(others, axis=0), axis=0) if others else train_prior
            mixed = (1.0 - gamma) * base[date] + gamma * prior
            date_scores.append(_encoding_r2_for_raw(held_calib(date), basis, mixed, lag_ms))
        grid.append(
            {
                "gamma": float(gamma),
                "mean_encoding_r2": float(np.mean(date_scores)),
                "per_date_encoding_r2": [float(v) for v in date_scores],
            }
        )
    # Higher is better; smaller gamma wins exact ties.
    winner = sorted(grid, key=lambda row: (-row["mean_encoding_r2"], row["gamma"]))[0]
    gamma = float(winner["gamma"])

    raw_by_date = {}
    for date in all_dates:
        raw_by_date[date] = (1.0 - gamma) * base[date] + gamma * train_prior

    train_rows = np.concatenate([raw_by_date[d] for d in train_dates], axis=0)
    mean = train_rows.mean(axis=0)
    std = train_rows.std(axis=0, ddof=0)
    std = np.where(std == 0.0, 1.0, std)
    standardized = {date: (raw - mean) / std for date, raw in raw_by_date.items()}
    for date, value in standardized.items():
        if value.shape != (85, 9) or not np.isfinite(value).all():
            raise RuntimeError(f"invalid SP-SFC9 profile for {date}: {value.shape}")

    return ProfileAuthority(
        lag_ms=lag_ms,
        gamma=gamma,
        coefficient_mean=mean,
        coefficient_std=std,
        source_prior=train_prior,
        raw_by_date=raw_by_date,
        standardized_by_date=standardized,
        gamma_grid=tuple(grid),
    )
