"""Neural-only carrier (N4) for M2 — behavior-free channel descriptors.

N4 reuses K4's raw block construction (20-ms bins, 5-bin blocks, no cross-trial
boundaries) but removes the velocity-activation filter so that **no behavioral
variable is ever read**.  Each channel is described by four statistics computed
on its block rate/count series:

  1. mean_rate        — mean block rate (Hz), behavior-free analog of baseline_rate
  2. fano             — var(count)/mean(count) on raw counts (not rates)
  3. lag1_autocorr    — Pearson(r[k], r[k+1]) within-trial pairs only
  4. population_coupling — Pearson(r_i[k], mean_{j!=i} r_j[k])

Degenerate channels (zero spikes, Fano denominator underflow, insufficient
autocorr pairs) raise by default — fail-closed, no silent zero-fill.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

N4_DIM = 4
N4_FEATURE_NAMES = ("mean_rate", "fano", "lag1_autocorr", "population_coupling")
N4_CALIBRATION_TRIALS = 24
N4_ALLOWED_CALIBRATION_TRIALS = (N4_CALIBRATION_TRIALS, 33, 12)
N4_RAW_BIN_MS = 20
N4_BLOCK_WIDTH_BINS = 5
N4_ESTIMATOR_VERSION = 1

DegeneracyPolicy = Literal["raise", "fill_zero", "fill_median"]


@dataclass(frozen=True)
class N4Audit:
    """Serializable provenance for one N4 raw calibration support."""

    calibration_trials: int
    n_blocks: int
    raw_bin_ms: int
    block_width_bins: int
    n_channels: int
    n_degenerate_channels: int
    degeneracy_policy: str
    active_rule: str

    def as_dict(self) -> dict[str, int | bool | str]:
        return {
            "calibration_trials": self.calibration_trials,
            "n_blocks": self.n_blocks,
            "raw_bin_ms": self.raw_bin_ms,
            "block_width_bins": self.block_width_bins,
            "n_channels": self.n_channels,
            "n_degenerate_channels": self.n_degenerate_channels,
            "degeneracy_policy": self.degeneracy_policy,
            "active_rule": self.active_rule,
        }


def n4_from_raw_calibration(
    neural: np.ndarray,
    trial_change: np.ndarray,
    *,
    calibration_n_trials: int = N4_CALIBRATION_TRIALS,
    degeneracy_policy: DegeneracyPolicy = "raise",
) -> tuple[np.ndarray, N4Audit]:
    """Compute behavior-free N4 features from raw neural calibration.

    Unlike K4, this function does NOT read covariates at all. It uses every
    block within each trial (no velocity-activation filter).
    """
    neural = np.asarray(neural, dtype=np.float64)
    trial_change = np.asarray(trial_change, dtype=bool)
    if neural.ndim != 2:
        raise ValueError(f"N4 neural must be [time, channels], got {neural.shape}")
    if neural.shape[0] != trial_change.shape[0]:
        raise ValueError("N4 neural/trial_change must share time length")
    if not np.isfinite(neural).all():
        raise ValueError("N4 raw calibration neural contains non-finite values")
    if calibration_n_trials not in N4_ALLOWED_CALIBRATION_TRIALS:
        raise ValueError(
            f"N4 supports calibration_n_trials in {N4_ALLOWED_CALIBRATION_TRIALS}, "
            f"got {calibration_n_trials}"
        )

    n_channels = neural.shape[1]
    starts = np.flatnonzero(trial_change)
    if len(starts) < calibration_n_trials:
        raise ValueError(
            f"N4 requires {calibration_n_trials} raw calibration trials, got {len(starts)}"
        )
    ends = np.r_[starts[1:], len(trial_change)]

    # Collect block counts and rates per channel — ALL blocks, no velocity filter.
    block_counts: list[np.ndarray] = []
    block_rates: list[np.ndarray] = []
    # Track which blocks belong to which trial (for within-trial autocorr).
    block_trial_id: list[int] = []
    for trial_idx, (start, end) in enumerate(
        zip(starts[:calibration_n_trials], ends[:calibration_n_trials])
    ):
        for left in range(
            int(start),
            int(end) - N4_BLOCK_WIDTH_BINS + 1,
            N4_BLOCK_WIDTH_BINS,
        ):
            right = left + N4_BLOCK_WIDTH_BINS
            counts = neural[left:right].sum(axis=0)  # [n_channels]
            rate = counts / (N4_BLOCK_WIDTH_BINS * N4_RAW_BIN_MS / 1000.0)
            block_counts.append(counts)
            block_rates.append(rate)
            block_trial_id.append(trial_idx)

    if len(block_counts) < 3:
        raise ValueError("N4 support has fewer than three valid blocks")

    counts_arr = np.asarray(block_counts, dtype=np.float64)  # [n_blocks, n_channels]
    rates_arr = np.asarray(block_rates, dtype=np.float64)    # [n_blocks, n_channels]
    trial_ids = np.asarray(block_trial_id, dtype=np.int64)
    n_blocks = rates_arr.shape[0]

    features = np.zeros((n_channels, N4_DIM), dtype=np.float32)
    n_degenerate = 0

    for ch in range(n_channels):
        ch_counts = counts_arr[:, ch]
        ch_rates = rates_arr[:, ch]
        degenerate_reasons: list[str] = []

        # Dim 1: mean_rate
        mean_rate = float(ch_rates.mean())

        # Dim 2: fano (on raw counts)
        mean_count = float(ch_counts.mean())
        if mean_count <= 1e-6:
            degenerate_reasons.append("fano_zero_mean_count")
            fano_val = np.nan
        else:
            var_count = float(ch_counts.var())
            fano_val = var_count / mean_count

        # Dim 3: lag1_autocorr (within-trial pairs only)
        autocorr_pairs_r = []
        autocorr_pairs_k = []
        for t in range(n_blocks - 1):
            if trial_ids[t] != trial_ids[t + 1]:
                continue
            autocorr_pairs_r.append(ch_rates[t])
            autocorr_pairs_k.append(ch_rates[t + 1])
        if len(autocorr_pairs_r) < 3:
            degenerate_reasons.append("autocorr_insufficient_pairs")
            autocorr_val = np.nan
        else:
            r_a = np.asarray(autocorr_pairs_r, dtype=np.float64)
            r_b = np.asarray(autocorr_pairs_k, dtype=np.float64)
            if np.std(r_a) <= 1e-12 or np.std(r_b) <= 1e-12:
                degenerate_reasons.append("autocorr_zero_variance")
                autocorr_val = np.nan
            else:
                autocorr_val = float(np.corrcoef(r_a, r_b)[0, 1])

        # Dim 4: population_coupling
        others = np.delete(rates_arr, ch, axis=1)
        pop_mean = others.mean(axis=1)
        if np.std(ch_rates) <= 1e-12 or np.std(pop_mean) <= 1e-12:
            degenerate_reasons.append("popcoupling_zero_variance")
            popcoup_val = np.nan
        else:
            popcoup_val = float(np.corrcoef(ch_rates, pop_mean)[0, 1])

        if degenerate_reasons:
            n_degenerate += 1
            if degeneracy_policy == "raise":
                raise ValueError(
                    f"N4 channel {ch} degenerate: {'; '.join(degenerate_reasons)}"
                )
            elif degeneracy_policy == "fill_zero":
                vals = [mean_rate, fano_val, autocorr_val, popcoup_val]
                features[ch] = [0.0 if (v is None or (isinstance(v, float) and not np.isfinite(v))) else v for v in vals]
                continue
            elif degeneracy_policy == "fill_median":
                # Will be filled after loop
                features[ch] = np.nan
                continue

        features[ch] = [mean_rate, fano_val, autocorr_val, popcoup_val]

    # Fill median for degenerate channels if requested
    if degeneracy_policy == "fill_median" and n_degenerate > 0:
        for d in range(N4_DIM):
            col = features[:, d]
            valid = col[np.isfinite(col)]
            if len(valid) > 0:
                med = float(np.median(valid))
            else:
                med = 0.0
            col[~np.isfinite(col)] = med
            features[:, d] = col

    audit = N4Audit(
        calibration_trials=calibration_n_trials,
        n_blocks=n_blocks,
        raw_bin_ms=N4_RAW_BIN_MS,
        block_width_bins=N4_BLOCK_WIDTH_BINS,
        n_channels=n_channels,
        n_degenerate_channels=n_degenerate,
        degeneracy_policy=degeneracy_policy,
        active_rule="all_blocks_no_behavior_filter__behavior_free",
    )
    return features.astype(np.float32), audit


def fit_train_n4_stats(
    session_n4_features: dict[str, np.ndarray], session_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Fit N4 z-score statistics from current LOSO train sessions only."""
    chunks: list[np.ndarray] = []
    for name in session_names:
        if name not in session_n4_features:
            raise ValueError(f"Missing train-session N4 features for {name}")
        values = np.asarray(session_n4_features[name], dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != N4_DIM or not np.isfinite(values).all():
            raise ValueError(f"Invalid N4 feature matrix for train session {name}: {values.shape}")
        chunks.append(values)
    if not chunks:
        raise ValueError("No train N4 feature matrices supplied")
    stacked = np.concatenate(chunks, axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std[std <= 1e-6] = 1.0
    return mean, std


def deterministic_n4_row_permutation(
    num_channels: int, *, session_name: str, seed: int
) -> np.ndarray:
    """Full-row deterministic nonidentity NS4 permutation, namespaced from KS4."""
    if num_channels < 2:
        raise ValueError("NS4 requires at least two channels")
    import hashlib

    digest = hashlib.sha256(f"n4-ns4-v1:{seed}:{session_name}".encode()).digest()
    rng = np.random.RandomState(int.from_bytes(digest[:4], "little"))
    order = rng.permutation(num_channels)
    if np.array_equal(order, np.arange(num_channels)):
        order = np.roll(order, 1)
    return order.astype(np.int64, copy=False)
