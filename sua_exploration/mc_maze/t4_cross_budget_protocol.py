"""Pure, causal T4@M fitting contracts for the Step-2A audit.

This module deliberately has no datamodule or model dependency.  It consumes already
selected chronological rewarded trials in their given order and reproduces the existing
equal-per-direction-mean cosine T4 semantics.  A rank-deficient prefix is represented as
``undefined`` (NaN descriptor), never as a usable zero-filled descriptor.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from mc_maze.unit_side_features import (
    CANONICAL_DIRECTIONS_RAD,
    MODULATION_EPS,
    TUNING_NUM_DIRECTIONS,
    _unit_tuning_features,
)


T4_CROSS_BUDGET_FIT_SEMANTICS_VERSION = "equal_direction_mean_cosine_rank3_v2_labelled_reliability"
DEFAULT_T4_BUDGETS: tuple[int, ...] = (10, 15, 20, 50)
_EPS = 1.0e-8
RELIABILITY_FEATURE_NAMES: tuple[str, ...] = (
    "log1p_labelled_fit_residual_variance",
    "log1p_labelled_spike_count",
    "log1p_labelled_time_exposure_s",
    "log1p_total_prefix_spike_count",
    "log1p_total_prefix_time_exposure_s",
    "log1p_modulation_to_residual",
    "rank_valid",
)


@dataclass(frozen=True)
class T4PrefixFit:
    """One actual chronological T4 prefix fit and its causal reliability fields."""

    budget: int
    t4: np.ndarray  # [units, 4], NaN throughout when the shared design is undefined.
    reliability: np.ndarray  # [units, 5], named by RELIABILITY_FEATURE_NAMES.
    fit_defined: bool
    design_rank: int
    design_condition: float
    direction_counts: np.ndarray  # [8], counts among actual first-M labelled trials.
    direction_balance: float
    zero_spike_unit_count: int
    zero_modulation_unit_count: int


def validate_budgets(budgets: Sequence[int]) -> tuple[int, ...]:
    """Validate strictly increasing positive chronological-prefix budgets."""
    normalized = tuple(int(value) for value in budgets)
    if not normalized:
        raise ValueError("at least one T4 budget is required")
    if any(value <= 0 for value in normalized):
        raise ValueError(f"T4 budgets must be positive, got {normalized}")
    if tuple(sorted(set(normalized))) != normalized:
        raise ValueError(f"T4 budgets must be strictly increasing and unique, got {normalized}")
    return normalized


def canonical_direction_indices(target_dirs_rad: Sequence[float | None]) -> np.ndarray:
    """Map already-selected target directions to the existing eight canonical bins.

    The NWB-facing feature module imports the existing private direction snapper directly so
    that the data path shares its exact semantics.  This public array helper is intentionally
    only for synthetic/pure tests where direction indices are provided directly to the fitter.
    """
    from mc_maze.unit_side_features import _nearest_canonical_direction_index

    return np.asarray(
        [
            _nearest_canonical_direction_index(float(theta)) if theta is not None and np.isfinite(theta) else -1
            for theta in target_dirs_rad
        ],
        dtype=np.int64,
    )


def direction_design_metadata(direction_indices: np.ndarray) -> tuple[np.ndarray, int, float, float]:
    """Return actual coverage, rank, condition and min/max coverage balance.

    Rank/condition are computed from the same unique canonical directions used by the existing
    T4 equal-direction-mean fit.  A rank below three is not deployment-eligible in Step 2A.
    """
    directions = np.asarray(direction_indices, dtype=np.int64)
    if directions.ndim != 1:
        raise ValueError(f"direction_indices must be rank 1, got {directions.shape}")
    if np.any((directions < -1) | (directions >= TUNING_NUM_DIRECTIONS)):
        raise ValueError("direction indices must be -1 or canonical indices 0..7")
    valid = directions[directions >= 0]
    counts = np.bincount(valid, minlength=TUNING_NUM_DIRECTIONS).astype(np.int64)
    present = np.flatnonzero(counts > 0)
    if present.size:
        theta = np.asarray([CANONICAL_DIRECTIONS_RAD[int(index)] for index in present], dtype=np.float64)
        design = np.stack([np.ones_like(theta), np.cos(theta), np.sin(theta)], axis=1)
        rank = int(np.linalg.matrix_rank(design))
        condition = float(np.linalg.cond(design)) if rank == 3 else math.inf
        balance = float(counts[present].min() / counts[present].max())
    else:
        rank = 0
        condition = math.inf
        balance = 0.0
    return counts, rank, condition, balance


def fit_t4_prefix(
    trial_rates: np.ndarray,
    trial_durations_s: np.ndarray,
    direction_indices: np.ndarray,
    *,
    budget: int,
) -> T4PrefixFit:
    """Fit T4 from an actual first-M prefix of trial rates.

    ``trial_rates`` is `[units, trials]` in Hz and ``trial_durations_s`` is the parallel trial
    duration vector.  The function never reorders or resamples trials.  It calls the existing
    per-unit equal-direction-mean T4 helper, then separately calculates trial-level residual
    variance/exposure for audit-only reliability fields.  The residual fit is evaluated *only*
    on rows with a finite/snap-valid target direction.  An unlabeled rewarded trial is part of
    the chronological prefix but must never acquire a spurious canonical direction via Python's
    negative indexing.
    """
    rates = np.asarray(trial_rates, dtype=np.float64)
    durations = np.asarray(trial_durations_s, dtype=np.float64)
    directions = np.asarray(direction_indices, dtype=np.int64)
    if rates.ndim != 2:
        raise ValueError(f"trial_rates must be [units,trials], got {rates.shape}")
    if durations.ndim != 1 or directions.ndim != 1:
        raise ValueError("trial_durations_s and direction_indices must be rank 1")
    if rates.shape[1] != durations.size or rates.shape[1] != directions.size:
        raise ValueError("trial rate, duration and direction lengths must agree")
    if not np.isfinite(rates).all() or not np.isfinite(durations).all() or np.any(durations <= 0):
        raise ValueError("rates must be finite and durations strictly positive")
    if budget <= 0 or budget > rates.shape[1]:
        raise ValueError(f"budget={budget} requires 1..{rates.shape[1]} chronological trials")

    prefix_rates = rates[:, :budget]
    prefix_durations = durations[:budget]
    prefix_directions = directions[:budget]
    counts, rank, condition, balance = direction_design_metadata(prefix_directions)
    num_units = prefix_rates.shape[0]
    if rank != 3 or not math.isfinite(condition):
        return T4PrefixFit(
            budget=budget,
            t4=np.full((num_units, 4), np.nan, dtype=np.float32),
            reliability=np.full((num_units, len(RELIABILITY_FEATURE_NAMES)), np.nan, dtype=np.float32),
            fit_defined=False,
            design_rank=rank,
            design_condition=condition,
            direction_counts=counts,
            direction_balance=balance,
            zero_spike_unit_count=0,
            zero_modulation_unit_count=0,
        )

    present = np.flatnonzero(counts > 0).astype(np.int64).tolist()
    t4 = np.empty((num_units, 4), dtype=np.float32)
    reliability = np.empty((num_units, len(RELIABILITY_FEATURE_NAMES)), dtype=np.float32)
    valid_labelled = prefix_directions >= 0
    labelled_directions = prefix_directions[valid_labelled]
    theta = np.asarray(
        [CANONICAL_DIRECTIONS_RAD[int(index)] for index in labelled_directions], dtype=np.float64
    )
    design = np.stack([np.ones_like(theta), np.cos(theta), np.sin(theta)], axis=1)
    zero_spike = 0
    zero_modulation = 0
    for unit_index in range(num_units):
        unit_t4, _t8, is_zero_spike, is_zero_modulation = _unit_tuning_features(
            prefix_rates[unit_index], prefix_directions, present
        )
        t4[unit_index] = unit_t4
        b, a, c = (float(unit_t4[3]), float(unit_t4[0]), float(unit_t4[1]))
        labelled_rates = prefix_rates[unit_index, valid_labelled]
        residual = labelled_rates - design @ np.asarray([b, a, c])
        residual_variance = float(
            np.dot(residual, residual) / max(1, labelled_rates.size - 3)
        )
        labelled_spike_count = float(
            np.dot(labelled_rates, prefix_durations[valid_labelled])
        )
        labelled_time_exposure_s = float(prefix_durations[valid_labelled].sum())
        total_prefix_spike_count = float(np.dot(prefix_rates[unit_index], prefix_durations))
        total_prefix_time_exposure_s = float(prefix_durations.sum())
        modulation_to_residual = float(unit_t4[2]) / math.sqrt(residual_variance + _EPS)
        reliability[unit_index] = np.asarray(
            [
                math.log1p(max(residual_variance, 0.0)),
                math.log1p(max(labelled_spike_count, 0.0)),
                math.log1p(max(labelled_time_exposure_s, 0.0)),
                math.log1p(max(total_prefix_spike_count, 0.0)),
                math.log1p(max(total_prefix_time_exposure_s, 0.0)),
                math.log1p(max(modulation_to_residual, 0.0)),
                1.0,
            ],
            dtype=np.float32,
        )
        zero_spike += int(is_zero_spike)
        zero_modulation += int(is_zero_modulation or float(unit_t4[2]) <= MODULATION_EPS)
    if reliability.shape != (num_units, len(RELIABILITY_FEATURE_NAMES)):
        raise AssertionError("T4 reliability shape/name contract violated")
    return T4PrefixFit(
        budget=budget,
        t4=t4,
        reliability=reliability,
        fit_defined=True,
        design_rank=rank,
        design_condition=condition,
        direction_counts=counts,
        direction_balance=balance,
        zero_spike_unit_count=zero_spike,
        zero_modulation_unit_count=zero_modulation,
    )


def fit_t4_prefixes(
    trial_rates: np.ndarray,
    trial_durations_s: np.ndarray,
    direction_indices: np.ndarray,
    *,
    budgets: Sequence[int] = DEFAULT_T4_BUDGETS,
) -> dict[int, T4PrefixFit]:
    """Fit every requested nested chronological prefix without coefficient-noise synthesis."""
    checked = validate_budgets(budgets)
    if checked[-1] > np.asarray(trial_rates).shape[1]:
        raise ValueError(f"largest budget={checked[-1]} exceeds supplied chronological trial count")
    return {
        budget: fit_t4_prefix(trial_rates, trial_durations_s, direction_indices, budget=budget)
        for budget in checked
    }


def thin_trial_counts_within_trial(
    trial_counts: np.ndarray,
    *,
    keep_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Independent spike thinning that preserves each trial's label/order/exposure boundary."""
    counts = np.asarray(trial_counts)
    if counts.ndim != 2 or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
        raise ValueError("trial_counts must be a nonnegative integer [units,trials] array")
    if not 0.0 < keep_probability <= 1.0:
        raise ValueError("keep_probability must be in (0,1]")
    return rng.binomial(counts, keep_probability).astype(np.int64, copy=False)


def within_trial_thinned_prefix_fit(
    trial_counts: np.ndarray,
    trial_durations_s: np.ndarray,
    direction_indices: np.ndarray,
    *,
    budget: int,
    keep_probability: float,
    rng: np.random.Generator,
) -> T4PrefixFit:
    """Refit the same prefix after within-trial spike thinning; labels are unchanged."""
    counts = thin_trial_counts_within_trial(
        trial_counts, keep_probability=keep_probability, rng=rng
    )
    durations = np.asarray(trial_durations_s, dtype=np.float64)
    if counts.shape[1] != durations.size:
        raise ValueError("trial count/duration lengths must agree")
    return fit_t4_prefix(counts / durations[None, :], durations, direction_indices, budget=budget)
