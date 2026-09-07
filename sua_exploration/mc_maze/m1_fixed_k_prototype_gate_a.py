"""Pure contracts for the source-only M1 fixed-K temporal-prototype Gate A.

The runner is intentionally separate.  This module accepts already-extracted
raw, unpadded trial bins and has no NWB, decoder, CUDA, or behaviour dependency.
It freezes the primary later-neural oracle and the inference-unit statistics
used by the future CPU-only source audit.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t


SUPPORT_TRIALS = 10
TARGET_START_TRIAL = 210
BIN_SECONDS = 0.020
OBJ_LEVELS = (1, 2, 3, 4)
RAW_PAD_VALUE = -1.0
RIDGE = 1.0
REPEATABILITY_RESAMPLES = 256
GATE_A_SEMANTICS_VERSION = "m1_fixed_k_gate_a_raw_trial_prefix_log1p_future_obj_rate_v3"


@dataclass(frozen=True)
class M1RawSession:
    """Only valid, raw, non-interpolated per-trial binned counts are retained."""

    session: str
    support_trials: tuple[np.ndarray, ...]  # exact first 10: [bins, channels]
    future_trial_counts: np.ndarray  # [trials >= 210, channels] total valid raw counts
    future_trial_bins: np.ndarray  # [trials >= 210] valid bin exposure
    support_obj_ids: np.ndarray  # [10]
    future_obj_ids: np.ndarray  # [trials >= 210], scorer-only
    source_path: str
    source_sha256: str


def _as_integer_counts(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 1 or not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError(f"{name} must be finite nonnegative raw counts")
    if not np.array_equal(array, np.rint(array)):
        raise ValueError(f"{name} contains non-integer values; interpolation/smoothing is forbidden")
    return array.astype(np.int64, copy=False)


def validate_obj_coverage(labels: np.ndarray, *, name: str, expected_counts: Sequence[int] | None = None) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    if values.size == 0 or set(values.tolist()) != set(OBJ_LEVELS):
        raise ValueError(f"{name} must contain exactly obj_id levels {OBJ_LEVELS}")
    counts = np.asarray([(values == level).sum() for level in OBJ_LEVELS], dtype=np.int64)
    if expected_counts is not None and not np.array_equal(counts, np.asarray(expected_counts, dtype=np.int64)):
        raise ValueError(f"{name} obj_id counts {counts.tolist()} differ from frozen expected {list(expected_counts)}")
    return counts


def extract_valid_raw_trials(
    padded_trials: np.ndarray,
    valid_lengths: np.ndarray,
    trial_spike_sums: np.ndarray,
    *,
    expected_channels: int = 64,
) -> tuple[np.ndarray, ...]:
    """Strip only verified -1 padding from production raw FalconDataset trials.

    The production dataset materializes a [trials,1024,64] convenience array.
    This function rejects any cubic/interpolated/noninteger valid prefix, any
    non--1 tail, or a mismatch against the independently retained raw sums.
    Returned variable-length trial arrays contain no padding.
    """
    values = np.asarray(padded_trials, dtype=np.float64)
    lengths = np.asarray(valid_lengths, dtype=np.int64).reshape(-1)
    sums = np.asarray(trial_spike_sums, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != lengths.size or values.shape[2] != expected_channels:
        raise ValueError(f"padded trials must be [trials,T,{expected_channels}] aligned to lengths")
    if sums.shape != (lengths.size, expected_channels) or np.any(lengths <= 0) or np.any(lengths > values.shape[1]):
        raise ValueError("raw trial sums/lengths are inconsistent")
    result: list[np.ndarray] = []
    for index, length in enumerate(lengths.tolist()):
        prefix = values[index, :length]
        tail = values[index, length:]
        counts = _as_integer_counts(prefix, name=f"trial {index} valid prefix")
        if tail.size and not np.array_equal(tail, np.full_like(tail, RAW_PAD_VALUE)):
            raise ValueError(f"trial {index} tail is not exactly raw padding {RAW_PAD_VALUE}")
        if not np.array_equal(counts.sum(axis=0, dtype=np.int64), np.rint(sums[index]).astype(np.int64)):
            raise ValueError(f"trial {index} valid-prefix count sum disagrees with raw trial_spike_sums")
        result.append(counts)
    return tuple(result)


def per_trial_hz(trials: Sequence[np.ndarray]) -> np.ndarray:
    """Convert variable-length raw-count trials to unweighted per-trial Hz."""
    if not trials:
        raise ValueError("at least one trial is required")
    output = []
    channels: int | None = None
    for index, trial in enumerate(trials):
        counts = _as_integer_counts(trial, name=f"trial {index}")
        if counts.ndim != 2 or counts.shape[0] <= 0:
            raise ValueError("each raw trial must be nonempty [bins,channels]")
        if channels is None:
            channels = counts.shape[1]
        if counts.shape[1] != channels:
            raise ValueError("raw trial channel widths differ")
        output.append(counts.sum(axis=0, dtype=np.float64) / (counts.shape[0] * BIN_SECONDS))
    return np.stack(output, axis=0)


def category_mean_hz(trial_hz: np.ndarray, labels: np.ndarray, *, name: str) -> np.ndarray:
    """Per-channel, per-obj_id unweighted mean of raw per-trial Hz."""
    rates = np.asarray(trial_hz, dtype=np.float64)
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    if rates.ndim != 2 or rates.shape[0] != values.size or not np.isfinite(rates).all() or np.any(rates < 0.0):
        raise ValueError(f"{name} rates/labels are invalid")
    validate_obj_coverage(values, name=name)
    return np.stack([rates[values == level].mean(axis=0) for level in OBJ_LEVELS], axis=1)


def category_mean_log1p_hz(trial_hz: np.ndarray, labels: np.ndarray, *, name: str) -> np.ndarray:
    """Frozen later-neural oracle: log1p of the category mean per-trial Hz."""
    return np.log1p(category_mean_hz(trial_hz, labels, name=name))


def d4_support_carrier(session: M1RawSession) -> np.ndarray:
    """The untransformed label-informed D4 reference from first ten trials only."""
    if len(session.support_trials) != SUPPORT_TRIALS or session.support_obj_ids.shape != (SUPPORT_TRIALS,):
        raise ValueError("D4 support requires exactly first ten aligned trials/obj_id labels")
    return category_mean_hz(per_trial_hz(session.support_trials), session.support_obj_ids, name="support obj_id")


def future_neural_oracle_target(session: M1RawSession) -> np.ndarray:
    """Scorer-only [channel,4] target from chronological trials [210,end)."""
    counts = _as_integer_counts(session.future_trial_counts, name="future trial counts")
    bins = np.asarray(session.future_trial_bins, dtype=np.int64).reshape(-1)
    if counts.ndim != 2 or counts.shape[0] != bins.size or bins.size != session.future_obj_ids.size or np.any(bins <= 0):
        raise ValueError("future count/bin/label arrays are inconsistent")
    rates = counts / (bins[:, None] * BIN_SECONDS)
    return category_mean_log1p_hz(rates, session.future_obj_ids, name="future scorer-only obj_id")


def disjoint_binomial_partition(trials: Sequence[np.ndarray], *, rng: np.random.Generator) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Partition every support-bin count into two complementary spike views."""
    first: list[np.ndarray] = []
    second: list[np.ndarray] = []
    for index, trial in enumerate(trials):
        counts = _as_integer_counts(trial, name=f"partition trial {index}")
        left = rng.binomial(counts, 0.5).astype(np.int64, copy=False)
        first.append(left)
        second.append(counts - left)
    return tuple(first), tuple(second)


def paired_session_summary(deltas: Sequence[float]) -> dict[str, float | int | list[float]]:
    """Session-level paired CI/MDE. Resample seeds are not independent sessions."""
    values = np.asarray(deltas, dtype=np.float64).reshape(-1)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Gate A paired inference is locked to four finite outer-session deltas")
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    se = std / math.sqrt(values.size)
    crit95 = float(student_t.ppf(0.975, df=values.size - 1))
    crit80 = float(student_t.ppf(0.80, df=values.size - 1))
    return {
        "n_sessions": int(values.size),
        "deltas": values.tolist(),
        "mean_delta": mean,
        "session_sd": std,
        "ci95": [mean - crit95 * se, mean + crit95 * se],
        "mde_80pct_two_sided_alpha05": (crit95 + crit80) * se,
        "positive_sessions": int((values > 0.0).sum()),
    }


def strict_gate(prototype_minus_rate: Sequence[float], prototype_minus_shuffle: Sequence[float], *, repeatability_contract_pass: bool, oracle_contract_pass: bool, state_contract_pass: bool) -> dict[str, object]:
    """Frozen Gate-A decision: both controls, measured MDE, and all contracts."""
    contrasts = {
        "prototype_minus_rate_only": paired_session_summary(prototype_minus_rate),
        "prototype_minus_slot_shuffle": paired_session_summary(prototype_minus_shuffle),
    }
    distinguishable = {}
    for name, entry in contrasts.items():
        mean = float(entry["mean_delta"])
        lower = float(entry["ci95"][0])
        mde = float(entry["mde_80pct_two_sided_alpha05"])
        distinguishable[name] = bool(lower > 0.0 and mean >= mde)
    passed = bool(
        all(distinguishable.values())
        and repeatability_contract_pass
        and oracle_contract_pass
        and state_contract_pass
    )
    return {
        "decision": "pass_for_separate_decoder_review" if passed else "stop_cpu_gate_not_met",
        "prototype_distinguishably_beats_both_controls": bool(all(distinguishable.values())),
        "contrast_distinguishability": distinguishable,
        "repeatability_contract_pass": bool(repeatability_contract_pass),
        "oracle_contract_pass": bool(oracle_contract_pass),
        "state_contract_pass": bool(state_contract_pass),
        "contrasts": contrasts,
        "gpu_authorized": False,
    }
