"""Pure contracts for the M2 R10@M24 source-only CPU gate.

This module deliberately has no NWB, FalconDataset, decoder, or CUDA import.
All functions operate on already-trialized non-negative integer spike counts.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import beta as beta_distribution
from scipy.stats import t as student_t

M = 24
WIDTH = 10
TARGET_WIDTH = 8
CHANNELS = 96
RIDGE = 1.0
BIN_SECONDS = 0.020
LAGS = (1, 2, 4, 8, 16, 32, 64, 128)
RANDOM_SCHEDULES = 4095
TOTAL_SCHEDULES = 4096
THINNING_REPEATS = 256
SEED = "m2-r10-m24-global-full-permutation-v2"
THIN_SEED = "m2-r10-m24-thinning-v1"
EPS = 1e-12
UINT32_MAX = np.iinfo(np.uint32).max


def _seed(*values: object) -> int:
    payload = ":".join((SEED, *map(str, values))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _validated_trials(
    value: Sequence[np.ndarray], *, name: str, exact_count: int | None
) -> tuple[np.ndarray, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, np.ndarray))
        or (exact_count is not None and len(value) != exact_count)
        or (exact_count is None and len(value) < 1)
    ):
        requirement = f"exactly {exact_count}" if exact_count is not None else "one or more"
        raise ValueError(f"{name} requires {requirement} trials")

    output: list[np.ndarray] = []
    channels: int | None = None
    for index, item in enumerate(value):
        array = np.asarray(item)
        if (
            array.ndim != 2
            or array.shape[0] < 1
            or array.shape[1] < 1
            or not np.isfinite(array).all()
            or np.any(array < 0)
            or not np.equal(array, np.floor(array)).all()
            or np.any(array > UINT32_MAX)
        ):
            raise ValueError(f"{name}[{index}] is not a valid integer-count prefix")
        if channels is None:
            channels = int(array.shape[1])
        if array.shape[1] != channels:
            raise ValueError(f"{name} channel mismatch")
        integer = array.astype(np.int64)
        totals = integer.sum(axis=0, dtype=np.uint64)
        if np.any(totals > UINT32_MAX):
            raise OverflowError(f"{name}[{index}] uint32 trial-total overflow")
        output.append(integer)
    return tuple(output)


def _trials(value: Sequence[np.ndarray], name: str = "trials") -> tuple[np.ndarray, ...]:
    return _validated_trials(value, name=name, exact_count=M)


def _future_trials(value: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    return _validated_trials(value, name="future", exact_count=None)


def _trial_log_rates(trials: Sequence[np.ndarray]) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    checked = _trials(trials)
    rates = np.stack(
        [
            np.log((trial.sum(axis=0, dtype=np.uint64).astype(np.float64) + 0.5)
                   / (BIN_SECONDS * trial.shape[0]))
            for trial in checked
        ]
    )
    if not np.isfinite(rates).all():
        raise ValueError("nonfinite trial log rate")
    return checked, rates


def r10(trials: Sequence[np.ndarray]) -> np.ndarray:
    """Ten fixed linear quantiles of the 24 exposure-normalized trial log-rates."""
    _, rates = _trial_log_rates(trials)
    probabilities = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
    result = np.quantile(rates, probabilities, axis=0, method="linear").T
    if result.shape[1] != WIDTH or not np.isfinite(result).all():
        raise ValueError("invalid R10 output")
    return result


def l10(trials: Sequence[np.ndarray]) -> np.ndarray:
    """Frozen width-10 low-order moment/polynomial rate control."""
    checked, rates = _trial_log_rates(trials)
    mu = rates.mean(axis=0)
    sigma = np.sqrt(np.square(rates - mu).mean(axis=0))
    total_bins = sum(trial.shape[0] for trial in checked)
    total_counts = sum(
        (trial.sum(axis=0, dtype=np.uint64) for trial in checked),
        start=np.zeros(checked[0].shape[1], dtype=np.uint64),
    )
    pooled_rate = np.log((total_counts.astype(np.float64) + 0.5) / (BIN_SECONDS * total_bins))
    result = np.stack(
        (
            mu,
            sigma,
            mu**2,
            mu * sigma,
            sigma**2,
            mu**3,
            mu**2 * sigma,
            mu * sigma**2,
            sigma**3,
            pooled_rate,
        ),
        axis=1,
    )
    if result.shape[1] != WIDTH or not np.isfinite(result).all():
        raise ValueError("invalid L10 output")
    return result


def future_autocorr(future: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """Trial-demeaned, exposure-weighted C/v target with whole-row masking."""
    trials = _future_trials(future)
    channels = trials[0].shape[1]
    deviations: list[np.ndarray] = []
    variance_numerator = np.zeros(channels, dtype=np.float64)
    bins = 0
    for trial in trials:
        deviation = trial.astype(np.float64) - trial.mean(axis=0, keepdims=True)
        deviations.append(deviation)
        variance_numerator += np.square(deviation).sum(axis=0)
        bins += trial.shape[0]
    variance = variance_numerator / bins

    target = np.full((channels, TARGET_WIDTH), np.nan, dtype=np.float64)
    exposure = np.zeros((channels, TARGET_WIDTH), dtype=np.int64)
    for lag_index, lag in enumerate(LAGS):
        numerator = np.zeros(channels, dtype=np.float64)
        pairs = 0
        for deviation in deviations:
            if deviation.shape[0] > lag:
                numerator += (deviation[:-lag] * deviation[lag:]).sum(axis=0)
                pairs += deviation.shape[0] - lag
        exposure[:, lag_index] = pairs
        if pairs > 0:
            finite_variance = variance > EPS
            target[finite_variance, lag_index] = (
                numerator[finite_variance] / pairs / variance[finite_variance]
            )

    row_defined = (variance > EPS) & np.all(exposure > 0, axis=1)
    target[~row_defined, :] = np.nan
    mask = np.isfinite(target)
    if not np.array_equal(mask.all(axis=1), row_defined):
        raise RuntimeError("whole-row target mask construction failure")
    return {
        "target": target,
        "mask": mask,
        "pair_exposure": exposure,
        "variance": variance,
        "defined_row_fraction_by_lag": mask.mean(axis=0),
    }


def _finite_matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite matrix")
    return array


@dataclass(frozen=True)
class Ridge:
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    target_mean: np.ndarray


def _fit(x: np.ndarray, y: np.ndarray) -> Ridge:
    features = _finite_matrix(x, "ridge features")
    targets = _finite_matrix(y, "ridge targets")
    if (
        features.shape[0] < 1
        or features.shape[0] != targets.shape[0]
        or features.shape[1] != WIDTH
        or targets.shape[1] != TARGET_WIDTH
    ):
        raise ValueError("ridge shape/row contract")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale <= EPS] = 1.0
    standardized = (features - mean) / scale
    target_mean = targets.mean(axis=0)
    weight = np.linalg.solve(
        standardized.T @ standardized + RIDGE * np.eye(WIDTH),
        standardized.T @ (targets - target_mean),
    )
    if not np.isfinite(weight).all():
        raise ValueError("nonfinite ridge weight")
    return Ridge(mean, scale, weight, target_mean)


def _predict(model: Ridge, x: np.ndarray) -> np.ndarray:
    features = _finite_matrix(x, "prediction features")
    if features.shape[1] != WIDTH:
        raise ValueError("prediction feature width")
    prediction = (features - model.mean) / model.scale @ model.weight + model.target_mean
    if not np.isfinite(prediction).all():
        raise ValueError("nonfinite prediction")
    return prediction


def _validated_session_inputs(
    features: Mapping[str, np.ndarray], targets: Mapping[str, Mapping[str, np.ndarray]]
) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    names = tuple(sorted(features))
    if len(names) != 7 or set(targets) != set(names):
        raise ValueError("requires exactly seven matching sessions")
    complete: dict[str, np.ndarray] = {}
    for session in names:
        feature = np.asarray(features[session])
        target = np.asarray(targets[session].get("target"))
        raw_mask = np.asarray(targets[session].get("mask"))
        if raw_mask.dtype != np.bool_:
            raise ValueError("target mask must be boolean")
        if (
            feature.shape != (CHANNELS, WIDTH)
            or not np.isfinite(feature).all()
            or target.shape != (CHANNELS, TARGET_WIDTH)
            or raw_mask.shape != (CHANNELS, TARGET_WIDTH)
            or not np.array_equal(raw_mask, np.isfinite(target))
        ):
            raise ValueError("feature/target/mask shape-finiteness contract")
        rows = raw_mask.all(axis=1)
        if not np.array_equal(raw_mask, np.repeat(rows[:, None], TARGET_WIDTH, axis=1)):
            raise ValueError("target mask is not whole-row")
        if raw_mask.mean(axis=0).min() < 0.90 or rows.sum() < 1:
            raise ValueError("target defined-row fraction below 0.90")
        complete[session] = rows
    return names, complete


def score_loso(
    features: Mapping[str, np.ndarray], targets: Mapping[str, Mapping[str, np.ndarray]]
) -> list[dict[str, float | int | str]]:
    names, complete = _validated_session_inputs(features, targets)
    rows: list[dict[str, float | int | str]] = []
    for left_out in names:
        train_sessions = [name for name in names if name != left_out]
        train_x = np.concatenate([np.asarray(features[name])[complete[name]] for name in train_sessions])
        train_y = np.concatenate(
            [np.asarray(targets[name]["target"])[complete[name]] for name in train_sessions]
        )
        test_x = np.asarray(features[left_out])[complete[left_out]]
        test_y = np.asarray(targets[left_out]["target"])[complete[left_out]]
        model = _fit(train_x, train_y)
        prediction = _predict(model, test_x)
        baseline = train_y.mean(axis=0)
        tss = float(np.square(test_y - baseline).sum())
        rss = float(np.square(test_y - prediction).sum())
        if not math.isfinite(tss) or not math.isfinite(rss) or tss <= 0 or rss < 0:
            raise ValueError("invalid TSS/RSS")
        rows.append(
            {
                "left_out_session": left_out,
                "r2": 1.0 - rss / tss,
                "bounded_u": tss / (tss + rss),
                "tss": tss,
                "rss": rss,
                "defined_rows": int(complete[left_out].sum()),
            }
        )
    return rows


def paired(r10_scores: Sequence[float], l10_scores: Sequence[float]) -> dict[str, object]:
    r10_array = np.asarray(r10_scores, dtype=np.float64)
    l10_array = np.asarray(l10_scores, dtype=np.float64)
    if (
        r10_array.shape != (7,)
        or l10_array.shape != (7,)
        or not np.isfinite(r10_array).all()
        or not np.isfinite(l10_array).all()
    ):
        raise ValueError("paired requires two vectors of seven finite scores")
    delta = r10_array - l10_array
    sd = float(delta.std(ddof=1))
    sem = sd / math.sqrt(7)
    ci_multiplier = float(student_t.ppf(0.975, 6))
    power_multiplier = float(student_t.ppf(0.80, 6))
    mean = float(delta.mean())
    return {
        "delta_by_session": delta.tolist(),
        "mean": mean,
        "median": float(np.median(delta)),
        "sd": sd,
        "operational_interval_95": [
            mean - ci_multiplier * sem,
            mean + ci_multiplier * sem,
        ],
        "operational_mde80": (ci_multiplier + power_multiplier) * sem,
        "interpretation": "seven-fold operational stability/precision only; folds are not independent",
    }


def formal_permutation(session: str, replicate: int, n: int) -> np.ndarray:
    """Uniform full-S_n draw; fixed points, identity, and duplicates are legal."""
    if replicate < 1 or n < 1:
        raise ValueError("formal schedule requires positive replicate and n")
    return np.random.default_rng(_seed("full_S", session, replicate, n)).permutation(n)


def schedule_receipt(sessions: Sequence[str], n: int) -> dict[str, object]:
    names = tuple(sorted(sessions))
    if len(names) != 7 or len(set(names)) != 7 or n < 1:
        raise ValueError("schedule requires seven unique sessions and positive n")
    lines: list[str] = []
    schedule_hashes: list[str] = []
    vectors: list[dict[str, list[int]]] = []
    fixed_points: dict[str, dict[str, int]] = {}
    per_session_identity: dict[str, list[int]] = {session: [] for session in names}
    global_identity: list[int] = []
    identity = np.arange(n)

    for replicate in range(1, RANDOM_SCHEDULES + 1):
        record: dict[str, list[int]] = {}
        record_lines: list[str] = []
        fixed_points[str(replicate)] = {}
        all_identity = True
        for session in names:
            permutation = formal_permutation(session, replicate, n)
            if permutation.shape != (n,) or not np.array_equal(np.sort(permutation), identity):
                raise RuntimeError("non-bijective formal permutation")
            values = permutation.tolist()
            record[session] = values
            count = int(np.sum(permutation == identity))
            fixed_points[str(replicate)][session] = count
            is_identity = count == n
            if is_identity:
                per_session_identity[session].append(replicate)
            all_identity = all_identity and is_identity
            record_lines.append(f"{replicate}:{session}:{','.join(map(str, values))}")
        if all_identity:
            global_identity.append(replicate)
        vectors.append(record)
        lines.extend(record_lines)
        schedule_hashes.append(hashlib.sha256("\n".join(record_lines).encode()).hexdigest())

    multiplicities = Counter(schedule_hashes)
    return {
        "seed_namespace": SEED,
        "formal_distribution": "independent_uniform_full_S_n_per_session",
        "random_schedules": RANDOM_SCHEDULES,
        "global_identity_observation_separate": True,
        "complete_permutation_vectors_by_schedule": vectors,
        "schedule_sha256": schedule_hashes,
        "all_vectors_complete_permutations": True,
        "fixed_point_count_by_schedule_session": fixed_points,
        "per_session_identity_draw_indices": per_session_identity,
        "per_session_identity_draw_counts": {
            session: len(indices) for session, indices in per_session_identity.items()
        },
        "global_identity_schedule_indices": global_identity,
        "duplicate_schedule_hash_multiplicities": {
            digest: count for digest, count in multiplicities.items() if count > 1
        },
        "full_plan_sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
    }


def null_features(features: Mapping[str, np.ndarray], replicate: int) -> dict[str, np.ndarray]:
    return {
        session: np.asarray(feature)[formal_permutation(session, replicate, np.asarray(feature).shape[0])]
        for session, feature in features.items()
    }


def mc(observed: float, null_values: Sequence[float]) -> dict[str, float | int | str]:
    values = np.asarray(null_values, dtype=np.float64)
    if (
        values.shape != (RANDOM_SCHEDULES,)
        or not np.isfinite(values).all()
        or np.any((values <= 0) | (values > 1))
        or not math.isfinite(float(observed))
        or not 0 < float(observed) <= 1
    ):
        raise ValueError("MC requires one observed and exactly 4095 null H values in (0,1]")
    exceedances = int((values >= observed).sum())
    p_attach = (1 + exceedances) / TOTAL_SCHEDULES
    upper = (
        float(beta_distribution.ppf(0.975, exceedances + 1, RANDOM_SCHEDULES - exceedances))
        if exceedances < RANDOM_SCHEDULES
        else 1.0
    )
    return {
        "p_attach": p_attach,
        "exceedances": exceedances,
        "mc_upper_97p5": upper,
        "sorted_H_sha256": hashlib.sha256(
            np.sort(values).astype("<f8").tobytes()
        ).hexdigest(),
    }


def split_reliability(
    trials: Sequence[np.ndarray], session: str, repeats: int = THINNING_REPEATS
) -> dict[str, object]:
    checked = _trials(trials)
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    medians: list[float] = []
    fractions: list[float] = []
    pearsons: list[list[float]] = []
    for repeat in range(repeats):
        first: list[np.ndarray] = []
        second: list[np.ndarray] = []
        for trial_index, trial in enumerate(checked):
            seed_payload = f"{THIN_SEED}:{session}:{repeat}:{trial_index}".encode()
            seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "little")
            half = np.random.default_rng(seed).binomial(trial, 0.5)
            first.append(2 * half)
            second.append(2 * (trial - half))
        first_r10 = r10(tuple(first))
        second_r10 = r10(tuple(second))
        pooled = np.concatenate((first_r10, second_r10), axis=0)
        mean = pooled.mean(axis=0)
        scale = pooled.std(axis=0)
        if np.any(scale <= EPS) or not np.isfinite(scale).all():
            raise ValueError("undefined R10 thinning coordinate")
        first_z = (first_r10 - mean) / scale
        second_z = (second_r10 - mean) / scale
        denominator = np.linalg.norm(first_z, axis=1) * np.linalg.norm(second_z, axis=1)
        defined = denominator > EPS
        fraction = float(defined.mean())
        if not defined.any():
            raise ValueError("no defined thinning rows")
        median = float(
            np.median((first_z[defined] * second_z[defined]).sum(axis=1) / denominator[defined])
        )
        coordinate_pearsons: list[float] = []
        for coordinate in range(WIDTH):
            if first_z[:, coordinate].std() <= EPS or second_z[:, coordinate].std() <= EPS:
                raise ValueError("undefined thinning Pearson coordinate")
            correlation = float(np.corrcoef(first_z[:, coordinate], second_z[:, coordinate])[0, 1])
            if not math.isfinite(correlation):
                raise ValueError("nonfinite thinning Pearson coordinate")
            coordinate_pearsons.append(correlation)
        if not math.isfinite(median) or not math.isfinite(fraction):
            raise ValueError("nonfinite thinning summary")
        medians.append(median)
        fractions.append(fraction)
        pearsons.append(coordinate_pearsons)

    quantiles = [float(np.quantile(medians, probability)) for probability in (0.025, 0.5, 0.975)]
    if not np.isfinite(quantiles).all():
        raise ValueError("nonfinite thinning quantiles")
    return {
        "repeats": repeats,
        "median_standardized_row_cosine": medians,
        "quantiles_2p5_50_97p5": quantiles,
        "minimum_defined_row_fraction": float(min(fractions)),
        "per_coordinate_pearson": pearsons,
    }


def streaming_receipt(n: int = CHANNELS) -> dict[str, int | bool | str]:
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    accumulator = n * 104 + 56
    final_fp32 = n * WIDTH * 4
    return {
        "per_channel_bytes": 104,
        "channel_state_bytes": n * 104,
        "shared_rounded_bytes": 56,
        "accumulator_bytes": accumulator,
        "final_fp32_bytes": final_fp32,
        "peak_fp32_bytes": accumulator + final_fp32,
        "bitonic_padded_inputs": 32,
        "bitonic_compare_exchanges": 240,
        "trial_rate_log_evaluations_per_channel": M,
        "raw_bin_retained": False,
        "overflow": "uint32 hard failure",
    }
