#!/usr/bin/env python3
"""CPU-only, source-disjoint Gate-A audit for an M1 support-set encoder.

The deployed input boundary is the first ten calibration trials.  Later trial
labels are used only to construct an offline neural-rate scoring target.  This
script does not train or evaluate a behavioral decoder and does not access the
official held-out query set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[2]
SCE = ROOT / "streaming_calibration_exp"
DATA = ROOT / "SPINT-main" / "data" / "000941"
PROTOCOL = ROOT / "sua_exploration" / "docs" / "M1_SUPPORT_ENCODER_GATE_A_PROTOCOL.md"
CORRECTION = ROOT / "sua_exploration" / "docs" / "M1_SUPPORT_ENCODER_GATE_A_NUMERICAL_CORRECTION.md"
DEFAULT_OUT = ROOT / "sua_exploration" / "results" / "m1_support_encoder_gate_a_v2"
LEVELS = (1, 2, 3, 4)
SUPPORT = 10
TARGET_START = 210
LAMBDAS = (0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0)
CANDIDATES = (
    ("means_direct", "means", "direct"),
    ("stats_direct", "stats", "direct"),
    ("stats_residual", "stats", "residual"),
)
N_RESAMPLES = 256
N_BOOTSTRAP = 20_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def strict_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, np.ndarray):
        return strict_json(value.tolist())
    if isinstance(value, dict):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    raise TypeError(f"not JSON serializable: {type(value)!r}")


@dataclass(frozen=True)
class SessionData:
    name: str
    sums: np.ndarray
    lengths: np.ndarray
    labels: np.ndarray
    nwb_path: Path


@dataclass(frozen=True)
class RidgeMap:
    feature_kind: str
    target_mode: str
    ridge_lambda: float
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    weights: np.ndarray


def session_name(path: Path) -> str:
    return "ses-" + path.name.split("_ses-")[1].split("_behavior")[0]


def _datamodule() -> Any:
    sys.path.insert(0, str(SCE))
    from src.data.falcon_datamodule import FalconDataModule

    dm = FalconDataModule(
        task="m1",
        data_dir=str(DATA) + "/",
        heldin_session_names=[""],
        batch_size=32,
        window_size=100,
        calibration_n_trials=SUPPORT,
        random_calibration=False,
        smooth_calibration=False,
        max_trial_length=1024,
        standardize_covariates=False,
        use_intertrials=True,
        use_calib_intertrials=False,
        trial_feature_type="raw",
        interpolate_trials=True,
        interpolate_trials_kind="cubic",
        pad_value=-1.0,
        validation_protocol="loso",
        loso_fold=1,
        rotation_id=0,
        include_heldout_in_fit=False,
        include_heldout_in_test=False,
        query_start_trial=0,
        heldin_query_start_trial=SUPPORT,
        heldin_query_end_trial=TARGET_START,
        allow_empty_heldout_query=False,
        num_workers=0,
        pin_memory=False,
        sampler_seed=42,
        balance_session_batches=False,
        reshuffle_train_sampler_each_epoch=False,
        side_feature_group="d4",
        side_feature_shuffle_seed=42,
    )
    # Lightning's Trainer is deliberately not constructed.  The data module
    # only needs world_size to derive its CPU batch size.
    dm.trainer = SimpleNamespace(world_size=1)
    dm.setup("fit")
    return dm


def load_sessions() -> dict[str, SessionData]:
    paths = {session_name(path): path for path in sorted(DATA.glob("sub-MonkeyL-held-in-calib/*.nwb"))}
    if len(paths) != 4:
        raise RuntimeError(f"expected four source held-in-calib NWBs, found {len(paths)}")
    dm = _datamodule()
    datasets = (dm.train_dataset, dm.val_heldin_dataset)
    result: dict[str, SessionData] = {}
    for dataset in datasets:
        for name, sums in dataset.calib_trial_spike_sums.items():
            if name in result:
                raise RuntimeError(f"duplicate source session: {name}")
            if name not in paths:
                raise RuntimeError(f"data module returned unrecognized source session: {name}")
            lengths = np.asarray(dataset.calib_trial_lengths[name], dtype=np.float64)
            labels = np.asarray(dataset.calib_trial_obj_ids[name], dtype=np.int64)
            values = np.asarray(sums, dtype=np.float64)
            if values.ndim != 2 or values.shape[1] != 64:
                raise RuntimeError(f"{name}: expected [trials,64] spike sums, got {values.shape}")
            if values.shape[0] != lengths.size or values.shape[0] != labels.size:
                raise RuntimeError(f"{name}: trial arrays do not align")
            if values.shape[0] <= TARGET_START or np.any(lengths <= 0) or not np.all(np.isfinite(values)):
                raise RuntimeError(f"{name}: invalid source trial data")
            result[name] = SessionData(name, values, lengths, labels, paths[name])
    if set(result) != set(paths):
        raise RuntimeError(f"source-session mismatch: loaded={sorted(result)}, files={sorted(paths)}")
    return dict(sorted(result.items()))


def trial_rates(data: SessionData, stop: int | None = None) -> np.ndarray:
    sums = data.sums if stop is None else data.sums[:stop]
    lengths = data.lengths if stop is None else data.lengths[:stop]
    return sums / lengths[:, None]


def category_means(rates: np.ndarray, labels: np.ndarray) -> np.ndarray:
    present = tuple(sorted(set(np.asarray(labels, dtype=np.int64).tolist())))
    if present != LEVELS:
        raise ValueError(f"all four levels are required; found {present}")
    return np.stack([rates[labels == level].mean(axis=0) for level in LEVELS], axis=1)


def future_target(data: SessionData) -> np.ndarray:
    rates = trial_rates(data)[TARGET_START:]
    labels = data.labels[TARGET_START:]
    return category_means(rates, labels)


def support_features(
    sums: np.ndarray,
    lengths: np.ndarray,
    labels: np.ndarray,
    feature_kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.asarray(sums[:SUPPORT], dtype=np.float64)
    lengths = np.asarray(lengths[:SUPPORT], dtype=np.float64)
    labels = np.asarray(labels[:SUPPORT], dtype=np.int64)
    if sums.shape[0] != SUPPORT or lengths.shape != (SUPPORT,) or labels.shape != (SUPPORT,):
        raise ValueError("support arrays must contain exactly ten aligned trials")
    rates = sums / lengths[:, None]
    d4 = category_means(rates, labels)
    log_d4 = np.log1p(np.maximum(d4, 0.0))
    if feature_kind == "means":
        return log_d4, d4

    global_mean = rates.mean(axis=0)
    global_std = rates.std(axis=0, ddof=1)
    half = SUPPORT // 2
    trend = rates[half:].mean(axis=0) - rates[:half].mean(axis=0)
    relative_trend = np.arcsinh(trend / np.maximum(global_mean, 1.0e-6))
    counts = np.asarray([(labels == level).sum() for level in LEVELS], dtype=np.float64)
    proportions = counts / SUPPORT
    exposure = np.asarray([lengths[labels == level].mean() for level in LEVELS], dtype=np.float64)
    exposure_log_ratio = np.log(exposure / lengths.mean())
    repeated_design = np.tile(np.concatenate([proportions, exposure_log_ratio]), (rates.shape[1], 1))

    if feature_kind == "rate_only":
        features = np.column_stack(
            [
                np.log1p(np.maximum(global_mean, 0.0)),
                np.log1p(np.maximum(global_std, 0.0)),
                relative_trend,
                repeated_design,
            ]
        )
        return features, d4

    if feature_kind != "stats":
        raise ValueError(f"unknown feature kind: {feature_kind}")
    standard_errors = []
    for level in LEVELS:
        rows = rates[labels == level]
        if len(rows) < 2:
            raise ValueError(f"support level {level} has fewer than two trials")
        standard_errors.append(rows.std(axis=0, ddof=1) / math.sqrt(len(rows)))
    se4 = np.stack(standard_errors, axis=1)
    features = np.column_stack(
        [
            log_d4,
            np.log1p(np.maximum(se4, 0.0)),
            np.log1p(np.maximum(global_mean, 0.0)),
            np.log1p(np.maximum(global_std, 0.0)),
            relative_trend,
            repeated_design,
        ]
    )
    return features, d4


def feature_matrix(data: SessionData, kind: str, labels: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    used_labels = data.labels if labels is None else np.asarray(labels)
    return support_features(data.sums, data.lengths, used_labels, kind)


def fit_ridge(
    sessions: Iterable[SessionData],
    feature_kind: str,
    target_mode: str,
    ridge_lambda: float,
) -> RidgeMap:
    session_list = list(sessions)
    if not session_list:
        raise ValueError("at least one training session is required")
    xs, ys = [], []
    for data in session_list:
        x, d4 = feature_matrix(data, feature_kind)
        target_log = np.log1p(np.maximum(future_target(data), 0.0))
        if target_mode == "direct":
            y = target_log
        elif target_mode == "residual":
            y = target_log - np.log1p(np.maximum(d4, 0.0))
        else:
            raise ValueError(f"unknown target mode: {target_mode}")
        xs.append(x)
        ys.append(y)
    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    x_mean = x_all.mean(axis=0)
    x_std = x_all.std(axis=0)
    x_std[x_std <= 1.0e-10] = 1.0
    xz = (x_all - x_mean) / x_std
    y_mean = y_all.mean(axis=0)
    lhs = xz.T @ xz + float(ridge_lambda) * np.eye(xz.shape[1])
    rhs = xz.T @ (y_all - y_mean)
    try:
        weights = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(lhs) @ rhs
    return RidgeMap(feature_kind, target_mode, float(ridge_lambda), x_mean, x_std, y_mean, weights)


def predict_ridge(
    model: RidgeMap,
    data: SessionData,
    *,
    labels: np.ndarray | None = None,
    support_sums: np.ndarray | None = None,
    support_lengths: np.ndarray | None = None,
) -> np.ndarray:
    if support_sums is None:
        x, d4 = feature_matrix(data, model.feature_kind, labels)
    else:
        used_labels = data.labels if labels is None else labels
        used_lengths = data.lengths if support_lengths is None else support_lengths
        x, d4 = support_features(support_sums, used_lengths, used_labels, model.feature_kind)
    predicted_log = model.y_mean + ((x - model.x_mean) / model.x_std) @ model.weights
    if model.target_mode == "residual":
        predicted_log = predicted_log + np.log1p(np.maximum(d4, 0.0))
    # Do not hide unstable extrapolation with an empirical upper cap. Candidate
    # selection detects and explicitly rejects every non-finite result.
    with np.errstate(over="ignore", invalid="ignore"):
        prediction = np.expm1(predicted_log)
    return np.maximum(prediction, 0.0)


def mse(predicted: np.ndarray, target: np.ndarray) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.mean(np.square(np.asarray(predicted) - np.asarray(target))))


def log_mse(predicted: np.ndarray, target: np.ndarray) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.mean(np.square(np.log1p(np.maximum(predicted, 0.0)) - np.log1p(np.maximum(target, 0.0)))))


def select_finite_record(
    records: list[dict[str, Any]], score_key: str, tie_keys: tuple[str, ...], context: str
) -> dict[str, Any]:
    valid = [
        row
        for row in records
        if row.get("status") == "ok"
        and row.get(score_key) is not None
        and math.isfinite(float(row[score_key]))
    ]
    if not valid:
        raise RuntimeError(f"no finite candidate for {context}")
    return min(valid, key=lambda row: (float(row[score_key]), *(row[key] for key in tie_keys)))


def select_model(train_names: list[str], sessions: dict[str, SessionData]) -> tuple[RidgeMap, dict[str, Any]]:
    records = []
    for candidate_name, feature_kind, target_mode in CANDIDATES:
        for ridge_lambda in LAMBDAS:
            fold_ratios = []
            fold_mse = []
            for val_name in train_names:
                inner_train = [sessions[name] for name in train_names if name != val_name]
                model = fit_ridge(inner_train, feature_kind, target_mode, ridge_lambda)
                val = sessions[val_name]
                prediction = predict_ridge(model, val)
                target = future_target(val)
                _, d4 = feature_matrix(val, "means")
                current = mse(prediction, target)
                baseline = mse(d4, target)
                fold_mse.append(current)
                fold_ratios.append(current / max(baseline, 1.0e-15))
            finite = bool(np.all(np.isfinite(fold_mse)) and np.all(np.isfinite(fold_ratios)))
            score = float(np.mean(np.log(np.maximum(fold_ratios, 1.0e-15)))) if finite else None
            records.append(
                {
                    "status": "ok" if finite and math.isfinite(score) else "invalid_nonfinite",
                    "reason": None if finite and math.isfinite(score) else "nonfinite_inner_fold_prediction_or_error",
                    "candidate": candidate_name,
                    "feature_kind": feature_kind,
                    "target_mode": target_mode,
                    "ridge_lambda": ridge_lambda,
                    "mean_log_mse_ratio_to_d4": score,
                    "geometric_mse_ratio_to_d4": float(math.exp(score)) if score is not None and math.isfinite(score) else None,
                    "inner_fold_mse": fold_mse,
                    "inner_fold_mse_ratio_to_d4": fold_ratios,
                }
            )
    selected = select_finite_record(
        records,
        "mean_log_mse_ratio_to_d4",
        ("candidate", "ridge_lambda"),
        f"E4 outer train sessions {train_names}",
    )
    fitted = fit_ridge(
        [sessions[name] for name in train_names],
        selected["feature_kind"],
        selected["target_mode"],
        selected["ridge_lambda"],
    )
    return fitted, {"selected": selected, "candidates": records}


def select_rate_only(train_names: list[str], sessions: dict[str, SessionData]) -> tuple[RidgeMap, dict[str, Any]]:
    records = []
    for ridge_lambda in LAMBDAS:
        errors = []
        for val_name in train_names:
            model = fit_ridge(
                [sessions[name] for name in train_names if name != val_name],
                "rate_only",
                "direct",
                ridge_lambda,
            )
            errors.append(mse(predict_ridge(model, sessions[val_name]), future_target(sessions[val_name])))
        finite = bool(np.all(np.isfinite(errors)))
        score = float(np.mean(np.log(np.maximum(errors, 1.0e-15)))) if finite else None
        records.append(
            {
                "status": "ok" if finite and math.isfinite(score) else "invalid_nonfinite",
                "reason": None if finite and math.isfinite(score) else "nonfinite_inner_fold_prediction_or_error",
                "ridge_lambda": ridge_lambda,
                "inner_fold_mse": errors,
                "mean_log_mse": score,
            }
        )
    selected = select_finite_record(
        records,
        "mean_log_mse",
        ("ridge_lambda",),
        f"rate-only outer train sessions {train_names}",
    )
    model = fit_ridge([sessions[name] for name in train_names], "rate_only", "direct", selected["ridge_lambda"])
    return model, {"selected": selected, "candidates": records}


def nonidentity_permutation(size: int, *seed_parts: object) -> np.ndarray:
    rng = np.random.RandomState(stable_seed(*seed_parts))
    permutation = rng.permutation(size)
    if np.array_equal(permutation, np.arange(size)):
        permutation = np.roll(permutation, 1)
    return permutation.astype(np.int64)


def shuffled_labels(data: SessionData) -> tuple[np.ndarray, np.ndarray]:
    original = np.asarray(data.labels[:SUPPORT], dtype=np.int64)
    permutation = nonidentity_permutation(SUPPORT, "m1-e4-label-shuffle-v1", data.name, 42)
    shuffled = original[permutation]
    if np.array_equal(shuffled, original):
        # A non-identity trial permutation can preserve a repeated label vector.
        # Deterministically search cyclic shifts until the labels differ.
        for shift in range(1, SUPPORT):
            candidate = np.roll(original, shift)
            if not np.array_equal(candidate, original):
                shuffled = candidate
                permutation = np.roll(np.arange(SUPPORT), shift)
                break
    if np.array_equal(shuffled, original) or sorted(shuffled.tolist()) != sorted(original.tolist()):
        raise RuntimeError(f"failed to construct count-preserving label control for {data.name}")
    return shuffled, permutation


def correlation(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    xc, yc = x - x.mean(), y - y.mean()
    pearson_den = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    cosine_den = float(np.linalg.norm(x) * np.linalg.norm(y))
    pearson = float(xc @ yc / pearson_den) if pearson_den > 0 else 0.0
    cosine = float(x @ y / cosine_den) if cosine_den > 0 else 0.0
    return pearson, cosine


def summarize_samples(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q025": float(np.quantile(array, 0.025)),
        "q975": float(np.quantile(array, 0.975)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def split_half_repeatability(data: SessionData, model: RidgeMap) -> dict[str, Any]:
    counts = np.asarray(np.rint(data.sums[:SUPPORT]), dtype=np.int64)
    if not np.allclose(counts, data.sums[:SUPPORT], atol=1.0e-5) or np.any(counts < 0):
        raise RuntimeError(f"{data.name}: production trial sums are not nonnegative integer spike counts")
    d4_pearson, d4_cosine, e4_pearson, e4_cosine = [], [], [], []
    for repetition in range(N_RESAMPLES):
        rng = np.random.RandomState(stable_seed("m1-e4-thinning-v1", data.name, repetition))
        half_a = rng.binomial(counts, 0.5).astype(np.float64)
        half_b = counts.astype(np.float64) - half_a
        half_lengths = data.lengths[:SUPPORT] * 0.5
        _, d4_a = support_features(half_a, half_lengths, data.labels, "means")
        _, d4_b = support_features(half_b, half_lengths, data.labels, "means")
        e4_a = predict_ridge(model, data, support_sums=half_a, support_lengths=half_lengths)
        e4_b = predict_ridge(model, data, support_sums=half_b, support_lengths=half_lengths)
        if not np.all(np.isfinite(e4_a)) or not np.all(np.isfinite(e4_b)):
            raise RuntimeError(f"{data.name}: selected E4 is non-finite under thinning repetition {repetition}")
        dp, dc = correlation(d4_a, d4_b)
        ep, ec = correlation(e4_a, e4_b)
        d4_pearson.append(dp)
        d4_cosine.append(dc)
        e4_pearson.append(ep)
        e4_cosine.append(ec)
    return {
        "n_deterministic_thinnings": N_RESAMPLES,
        "definition": "binomial spike thinning into complementary half-exposure estimates; all ten trials and labels retained",
        "d4": {"pearson": summarize_samples(d4_pearson), "cosine": summarize_samples(d4_cosine)},
        "e4": {"pearson": summarize_samples(e4_pearson), "cosine": summarize_samples(e4_cosine)},
    }


def paired_ratio(numerator: list[float], denominator: list[float], name: str) -> dict[str, Any]:
    a = np.asarray(numerator, dtype=np.float64)
    b = np.asarray(denominator, dtype=np.float64)
    if a.shape != (4,) or b.shape != (4,) or np.any(a <= 0) or np.any(b <= 0):
        raise ValueError(f"{name}: four finite positive paired errors are required")
    ratios = a / b
    logs = np.log(ratios)
    mean_log = float(logs.mean())
    sd_log = float(logs.std(ddof=1))
    critical = float(student_t.ppf(0.975, df=3))
    half_width = critical * sd_log / math.sqrt(4)
    rng = np.random.RandomState(stable_seed("m1-e4-bootstrap-v1", name))
    indices = rng.randint(0, 4, size=(N_BOOTSTRAP, 4))
    boot = np.exp(logs[indices].mean(axis=1))
    return {
        "name": name,
        "per_session_ratio": ratios,
        "geometric_mean_ratio": float(math.exp(mean_log)),
        "n_numerator_better": int(np.sum(ratios < 1.0)),
        "paired_t_95_ratio_interval": [float(math.exp(mean_log - half_width)), float(math.exp(mean_log + half_width))],
        "bootstrap_95_ratio_interval": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "multiplicative_95_precision_factor": float(math.exp(half_width)),
        "minimum_centered_ratio_with_upper_t_bound_at_one": float(math.exp(-half_width)),
        "n_pairs": 4,
    }


def evaluate_fold(test_name: str, sessions: dict[str, SessionData]) -> tuple[dict[str, Any], RidgeMap]:
    train_names = [name for name in sessions if name != test_name]
    e4_model, selection = select_model(train_names, sessions)
    rate_model, rate_selection = select_rate_only(train_names, sessions)
    test = sessions[test_name]
    target = future_target(test)
    _, d4 = feature_matrix(test, "means")
    e4 = predict_ridge(e4_model, test)
    rate_only = predict_ridge(rate_model, test)
    es4_permutation = nonidentity_permutation(64, "m1-es4-v1", test_name, 42)
    es4 = e4[es4_permutation]
    label_control, label_permutation = shuffled_labels(test)
    label_shuffle = predict_ridge(e4_model, test, labels=label_control)

    permutation_mse = []
    for repetition in range(N_RESAMPLES):
        permutation = nonidentity_permutation(64, "m1-es4-distribution-v1", test_name, repetition)
        permutation_mse.append(mse(e4[permutation], target))

    predictions = {
        "target_future_rate": target,
        "d4": d4,
        "e4": e4,
        "es4": es4,
        "label_shuffle_e4": label_shuffle,
        "rate_only": rate_only,
    }
    nonfinite_arms = [arm for arm, values in predictions.items() if not np.all(np.isfinite(values))]
    if nonfinite_arms:
        raise RuntimeError(f"{test_name}: selected outer predictions are non-finite for {nonfinite_arms}")
    errors = {
        arm: {"raw_rate_mse": mse(prediction, target), "log1p_rate_mse": log_mse(prediction, target)}
        for arm, prediction in predictions.items()
        if arm != "target_future_rate"
    }
    record = {
        "test_session": test_name,
        "train_sessions": train_names,
        "outer_train_rows": len(train_names) * 64,
        "n_trials": int(test.sums.shape[0]),
        "support": {
            "trial_interval": [0, SUPPORT],
            "label_counts": {str(level): int(np.sum(test.labels[:SUPPORT] == level)) for level in LEVELS},
            "valid_lengths": test.lengths[:SUPPORT],
        },
        "target": {
            "trial_interval": [TARGET_START, int(test.sums.shape[0])],
            "label_counts": {str(level): int(np.sum(test.labels[TARGET_START:] == level)) for level in LEVELS},
        },
        "e4_selection": selection,
        "rate_only_selection": rate_selection,
        "es4_primary_permutation": es4_permutation,
        "label_shuffle_trial_permutation": label_permutation,
        "label_shuffle_labels": label_control,
        "es4_256_permutation_mse": summarize_samples(permutation_mse),
        "errors": errors,
        "predictions": predictions,
        "repeatability": split_half_repeatability(test, e4_model),
    }
    return record, e4_model


def build_audit() -> dict[str, Any]:
    sessions = load_sessions()
    folds = []
    for test_name in sessions:
        fold, _ = evaluate_fold(test_name, sessions)
        folds.append(fold)

    arm_errors: dict[str, list[float]] = {arm: [] for arm in ("d4", "e4", "es4", "label_shuffle_e4", "rate_only")}
    for fold in folds:
        for arm in arm_errors:
            arm_errors[arm].append(float(fold["errors"][arm]["raw_rate_mse"]))
    contrasts = {
        "e4_over_d4": paired_ratio(arm_errors["e4"], arm_errors["d4"], "e4_over_d4"),
        "e4_over_es4": paired_ratio(arm_errors["e4"], arm_errors["es4"], "e4_over_es4"),
        "e4_over_label_shuffle": paired_ratio(arm_errors["e4"], arm_errors["label_shuffle_e4"], "e4_over_label_shuffle"),
        "e4_over_rate_only": paired_ratio(arm_errors["e4"], arm_errors["rate_only"], "e4_over_rate_only"),
    }
    thresholds = {
        "e4_over_d4": 0.95,
        "e4_over_es4": 0.90,
        "e4_over_label_shuffle": 0.90,
        "e4_over_rate_only": 0.90,
    }
    point_checks = {
        name: bool(
            contrast["geometric_mean_ratio"] <= thresholds[name]
            and contrast["n_numerator_better"] >= 3
        )
        for name, contrast in contrasts.items()
    }
    uncertainty_checks = {
        name: bool(contrast["paired_t_95_ratio_interval"][1] < 1.0)
        for name, contrast in contrasts.items()
    }
    coverage_ok = bool(all(
        set(fold["support"]["label_counts"].values()) and
        all(count >= 2 for count in fold["support"]["label_counts"].values()) and
        all(count > 0 for count in fold["target"]["label_counts"].values()) and
        len(fold["train_sessions"]) == 3 and fold["test_session"] not in fold["train_sessions"]
        for fold in folds
    ))
    pass_gate = bool(coverage_ok and all(point_checks.values()) and all(uncertainty_checks.values()))
    if pass_gate:
        disposition = "pass_gpu_protocol_review_allowed"
    elif not all(point_checks.values()):
        disposition = "ineffective_stop_m1_representation_branch"
    else:
        disposition = "indeterminate_unresolvable_stop_m1_representation_branch"

    input_files = {
        name: {"path": str(data.nwb_path.relative_to(ROOT)), "sha256": sha256(data.nwb_path)}
        for name, data in sessions.items()
    }
    return {
        "schema_version": "m1_support_encoder_gate_a_v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "compute": "CPU only; FalconDataModule constructed without a Lightning Trainer and CUDA hidden",
            "dataset": "FALCON M1 source held-in-calib sessions only",
            "support_trial_interval": [0, SUPPORT],
            "future_proxy_trial_interval": [TARGET_START, None],
            "outer_split": "leave one complete source session out",
            "inner_selection": "leave one complete outer-training source session out",
            "future_label_oracle": "Future obj_id labels construct and score the offline neural-rate target only; they never enter E4 or a deployed decoder.",
            "not_a_decoder_metric": True,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "generation_command": "/home/xinyuan/miniconda3/envs/spint/bin/python sua_exploration/scripts/audit_m1_support_encoder_gate_a.py",
        },
        "inputs": {
            "protocol": {"path": str(PROTOCOL.relative_to(ROOT)), "sha256": sha256(PROTOCOL)},
            "numerical_correction": {"path": str(CORRECTION.relative_to(ROOT)), "sha256": sha256(CORRECTION)},
            "script": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": sha256(Path(__file__).resolve())},
            "nwb": input_files,
        },
        "frozen_model_space": {
            "candidates": [row[0] for row in CANDIDATES],
            "ridge_lambdas": LAMBDAS,
            "output_dimension": 4,
            "resamples": N_RESAMPLES,
        },
        "folds": folds,
        "summary": {
            "raw_rate_mse_by_arm_and_session_order": arm_errors,
            "session_order": [fold["test_session"] for fold in folds],
            "paired_contrasts": contrasts,
        },
        "decision": {
            "supersedes": "m1_support_encoder_gate_a_v1 in full because non-finite rate-only candidate selection was invalid",
            "practical_ratio_thresholds": thresholds,
            "minimum_positive_sessions": 3,
            "point_checks": point_checks,
            "uncertainty_checks_upper_paired_t_ratio_below_one": uncertainty_checks,
            "coverage_and_source_disjointness_ok": coverage_ok,
            "gpu_gate_pass": pass_gate,
            "disposition": disposition,
            "prohibited_followups_on_failure": [
                "GPU decoder fusion",
                "encoder-width or latent-dimension sweep",
                "autoencoder or contrastive rescue on the same four sessions",
                "support-budget sweep",
                "adaptive reuse of official held-out scores",
            ],
        },
    }


def report_markdown(audit: dict[str, Any]) -> str:
    decision = audit["decision"]
    lines = [
        "# M1 support-set encoder Gate-A result",
        "",
        f"**Decision:** `{decision['disposition']}`  ",
        f"**GPU gate:** `{str(decision['gpu_gate_pass']).lower()}`",
        "",
        "This is a source-only future-neural-rate proxy, not decoder R² and not an official held-out result.",
        "Future category labels are scorer-only oracle information.",
        "",
        "| contrast | geometric MSE ratio | better sessions | paired-t 95% ratio CI | threshold | point | uncertainty |",
        "|---|---:|---:|---:|---:|:---:|:---:|",
    ]
    contrasts = audit["summary"]["paired_contrasts"]
    for name in ("e4_over_d4", "e4_over_es4", "e4_over_label_shuffle", "e4_over_rate_only"):
        row = contrasts[name]
        interval = row["paired_t_95_ratio_interval"]
        lines.append(
            f"| {name} | {row['geometric_mean_ratio']:.4f} | {row['n_numerator_better']}/4 | "
            f"[{interval[0]:.4f}, {interval[1]:.4f}] | {decision['practical_ratio_thresholds'][name]:.2f} | "
            f"{decision['point_checks'][name]} | {decision['uncertainty_checks_upper_paired_t_ratio_below_one'][name]} |"
        )
    lines.extend(["", "## Outer folds", "", "| test session | selected E4 | lambda | D4 MSE | E4 MSE | ES4 MSE | label-shuffle MSE | rate-only MSE |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for fold in audit["folds"]:
        selected = fold["e4_selection"]["selected"]
        errors = fold["errors"]
        lines.append(
            f"| {fold['test_session']} | {selected['candidate']} | {selected['ridge_lambda']:.4g} | "
            f"{errors['d4']['raw_rate_mse']:.6g} | {errors['e4']['raw_rate_mse']:.6g} | "
            f"{errors['es4']['raw_rate_mse']:.6g} | {errors['label_shuffle_e4']['raw_rate_mse']:.6g} | "
            f"{errors['rate_only']['raw_rate_mse']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A pass would authorize only a separately frozen one-cell GPU protocol. A failure stops this M1",
            "representation branch; it does not establish that all learned functional carriers are universally useless.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.out.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing result directory: {output}")
    audit = strict_json(build_audit())
    output.mkdir(parents=True, exist_ok=False)
    audit_path = output / "audit.json"
    report_path = output / "report.md"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n")
    report_path.write_text(report_markdown(audit))
    manifest = {
        "schema_version": "m1_support_encoder_gate_a_manifest_v2",
        "audit.json": sha256(audit_path),
        "report.md": sha256(report_path),
        str(PROTOCOL.relative_to(ROOT)): sha256(PROTOCOL),
        str(CORRECTION.relative_to(ROOT)): sha256(CORRECTION),
        str(Path(__file__).resolve().relative_to(ROOT)): sha256(Path(__file__).resolve()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "decision": audit["decision"], "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
