#!/usr/bin/env python3
"""Train-only audit: does calibration confidence predict future T4 fit error?

For every strict-manifest *training* session, fit T4 and its confidence from
the first M rewarded labelled trials, then score the frozen cosine prediction
on rewarded trials ``[M:50]``.  No validation or formal-test NWB is opened.

This is a descriptor diagnostic, not a decoding result.  It is intended to
decide whether a failed confidence-FiLM arm should be optimized with better
confidence inputs or killed as an unsupported mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import spearmanr

from mc_maze.unit_side_features import (
    CANONICAL_DIRECTIONS_RAD,
    _nearest_canonical_direction_index,
    _pool_trial_rate_matrix,
    _unit_tuning_features,
    list_datamodule_rewarded_trials,
    tuning_fit_confidence_descriptor,
)


EPS = 1.0e-8
CURRENT_FEATURES = (
    "t4_a",
    "t4_c",
    "log1p_t4_m",
    "t4_b",
    "log_residual_variance",
    "log_ac_covariance_shape",
)
EXPANDED_FEATURES = (
    *CURRENT_FEATURES,
    "log1p_spike_exposure",
    "direction_entropy",
    "log_ac_standard_error_trace",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_budgets(value: str) -> tuple[int, ...]:
    budgets = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not budgets or len(set(budgets)) != len(budgets):
        raise argparse.ArgumentTypeError("budgets must be nonempty and unique")
    if any(value < 3 for value in budgets):
        raise argparse.ArgumentTypeError("each budget must be at least 3")
    return budgets


def _fit_t4_matrix(
    rates: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    present = sorted({int(value) for value in directions if value >= 0})
    if len(present) < 3:
        raise ValueError("T4 predictive audit requires at least three directions")
    theta = np.asarray(
        [CANONICAL_DIRECTIONS_RAD[index] for index in present],
        dtype=np.float64,
    )
    design = np.stack(
        [np.ones_like(theta), np.cos(theta), np.sin(theta)], axis=1
    )
    if int(np.linalg.matrix_rank(design)) != 3:
        raise ValueError("T4 predictive audit requires a rank-3 direction design")
    t4 = np.empty((rates.shape[0], 4), dtype=np.float64)
    confidence = np.empty((rates.shape[0], 2), dtype=np.float64)
    for unit_idx in range(rates.shape[0]):
        unit_t4, _t8, _zero_spike, _zero_modulation = _unit_tuning_features(
            rates[unit_idx], directions, present
        )
        t4[unit_idx] = unit_t4
        confidence[unit_idx] = tuning_fit_confidence_descriptor(
            rates[unit_idx],
            directions,
            selected_t4=unit_t4,
        )
    return t4, confidence


def _direction_design_descriptors(
    directions: np.ndarray,
    residual_variance: np.ndarray,
) -> tuple[float, np.ndarray]:
    valid = directions >= 0
    theta = np.asarray(
        [CANONICAL_DIRECTIONS_RAD[int(index)] for index in directions[valid]],
        dtype=np.float64,
    )
    design = np.stack(
        [np.ones_like(theta), np.cos(theta), np.sin(theta)], axis=1
    )
    if int(np.linalg.matrix_rank(design)) != 3:
        raise ValueError("trial-level design is not rank 3")
    covariance_shape = np.linalg.inv(design.T @ design)[1:3, 1:3]
    log_se_trace = 0.5 * np.log(
        residual_variance * float(np.trace(covariance_shape)) + EPS
    )
    counts = np.bincount(directions[valid], minlength=8).astype(np.float64)
    probabilities = counts[counts > 0] / counts.sum()
    entropy = float(
        -np.sum(probabilities * np.log(probabilities)) / math.log(8.0)
    )
    return entropy, log_se_trace


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 3 or float(np.std(x)) <= 1.0e-12:
        return None
    result = spearmanr(x, y)
    value = float(result.statistic)
    return value if math.isfinite(value) else None


def _standardized_ridge_loso(
    rows: list[dict],
    feature_names: Iterable[str],
    *,
    ridge: float = 1.0,
) -> dict:
    feature_names = tuple(feature_names)
    sessions = sorted({str(row["session"]) for row in rows})
    observed: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    per_session: dict[str, dict[str, float | int]] = {}
    for session in sessions:
        train = [row for row in rows if row["session"] != session]
        test = [row for row in rows if row["session"] == session]
        x_train = np.asarray(
            [[row[name] for name in feature_names] for row in train],
            dtype=np.float64,
        )
        x_test = np.asarray(
            [[row[name] for name in feature_names] for row in test],
            dtype=np.float64,
        )
        y_train = np.asarray(
            [row["log_future_prediction_mse"] for row in train],
            dtype=np.float64,
        )
        y_test = np.asarray(
            [row["log_future_prediction_mse"] for row in test],
            dtype=np.float64,
        )
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        std[std < 1.0e-8] = 1.0
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
        x_train = np.concatenate(
            [np.ones((x_train.shape[0], 1)), x_train], axis=1
        )
        x_test = np.concatenate(
            [np.ones((x_test.shape[0], 1)), x_test], axis=1
        )
        penalty = np.eye(x_train.shape[1], dtype=np.float64) * ridge
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            x_train.T @ x_train + penalty,
            x_train.T @ y_train,
        )
        y_pred = x_test @ coefficients
        denominator = float(np.sum((y_test - y_test.mean()) ** 2))
        r2 = (
            1.0 - float(np.sum((y_test - y_pred) ** 2)) / denominator
            if denominator > 0.0
            else float("nan")
        )
        per_session[session] = {
            "unit_count": len(test),
            "r2": r2,
            "mae": float(np.mean(np.abs(y_test - y_pred))),
        }
        observed.append(y_test)
        predicted.append(y_pred)
    y = np.concatenate(observed)
    y_hat = np.concatenate(predicted)
    global_denominator = float(np.sum((y - y.mean()) ** 2))
    return {
        "features": list(feature_names),
        "ridge": ridge,
        "global_r2": (
            1.0 - float(np.sum((y - y_hat) ** 2)) / global_denominator
        ),
        "global_mae": float(np.mean(np.abs(y - y_hat))),
        "per_session_median_r2": float(
            np.nanmedian([value["r2"] for value in per_session.values()])
        ),
        "per_session_median_mae": float(
            np.median([value["mae"] for value in per_session.values()])
        ),
        "per_session": per_session,
    }


def _session_rows(
    path: Path,
    *,
    session: str,
    budgets: tuple[int, ...],
    reference_pool: int,
) -> dict[int, list[dict]]:
    trials = list_datamodule_rewarded_trials(
        path,
        bin_size_ms=20,
        window_size=50,
        trial_result_filter="R",
    )
    if len(trials) < reference_pool:
        raise ValueError(
            f"{session}: requires {reference_pool} rewarded trials, found {len(trials)}"
        )
    trials = trials[:reference_pool]
    directions = np.asarray(
        [
            _nearest_canonical_direction_index(trial["target_dir"])
            if trial.get("target_dir") is not None
            else -1
            for trial in trials
        ],
        dtype=np.int64,
    )
    durations = np.asarray(
        [trial["stop_time"] - trial["start_time"] for trial in trials],
        dtype=np.float64,
    )
    rates, _ = _pool_trial_rate_matrix(path, trials)
    t4_reference, _ = _fit_t4_matrix(rates, directions)
    output: dict[int, list[dict]] = {}
    for budget in budgets:
        fit_rates = rates[:, :budget]
        fit_directions = directions[:budget]
        t4, confidence = _fit_t4_matrix(fit_rates, fit_directions)
        valid_future = directions[budget:reference_pool] >= 0
        future_directions = directions[budget:reference_pool][valid_future]
        future_rates = rates[:, budget:reference_pool][:, valid_future]
        theta = np.asarray(
            [CANONICAL_DIRECTIONS_RAD[int(index)] for index in future_directions],
            dtype=np.float64,
        )
        predicted = (
            t4[:, 3:4]
            + t4[:, 0:1] * np.cos(theta)[None, :]
            + t4[:, 1:2] * np.sin(theta)[None, :]
        )
        future_mse = np.mean((future_rates - predicted) ** 2, axis=1)
        drift = np.mean(
            (t4[:, (0, 1, 3)] - t4_reference[:, (0, 1, 3)]) ** 2,
            axis=1,
        )
        residual_variance = np.exp(confidence[:, 0])
        entropy, log_se_trace = _direction_design_descriptors(
            fit_directions, residual_variance
        )
        exposure = np.sum(
            fit_rates * durations[:budget][None, :], axis=1
        )
        output[budget] = [
            {
                "session": session,
                "unit_index": unit_idx,
                "t4_a": float(t4[unit_idx, 0]),
                "t4_c": float(t4[unit_idx, 1]),
                "log1p_t4_m": float(np.log1p(max(t4[unit_idx, 2], 0.0))),
                "t4_b": float(t4[unit_idx, 3]),
                "log_residual_variance": float(confidence[unit_idx, 0]),
                "log_ac_covariance_shape": float(confidence[unit_idx, 1]),
                "log1p_spike_exposure": float(np.log1p(exposure[unit_idx])),
                "direction_entropy": entropy,
                "log_ac_standard_error_trace": float(log_se_trace[unit_idx]),
                "log_future_prediction_mse": float(
                    np.log(future_mse[unit_idx] + EPS)
                ),
                "log_t4_reference_drift": float(np.log(drift[unit_idx] + EPS)),
            }
            for unit_idx in range(rates.shape[0])
        ]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--budgets", type=_parse_budgets, default=(10, 15, 20, 30))
    parser.add_argument("--reference-pool", type=int, default=50)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if any(budget >= args.reference_pool for budget in args.budgets):
        raise ValueError("every budget must be smaller than reference_pool")

    manifest_path = args.manifest.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sessions = list(manifest["session_splits"]["train"])
    if len(sessions) != 27 or len(set(sessions)) != 27:
        raise ValueError("predictive audit requires the strict 27-session train split")
    by_budget: dict[int, list[dict]] = {budget: [] for budget in args.budgets}
    receipts: dict[str, str] = {}
    for index, session in enumerate(sessions, start=1):
        path = (data_dir / f"{session}_behavior+ecephys.nwb").resolve()
        if path.parent != data_dir or not path.is_file():
            raise FileNotFoundError(path)
        session_result = _session_rows(
            path,
            session=session,
            budgets=args.budgets,
            reference_pool=args.reference_pool,
        )
        for budget, rows in session_result.items():
            by_budget[budget].extend(rows)
        receipts[session] = str(path)
        print(f"[{index:02d}/{len(sessions)}] {session}", flush=True)

    budget_results: dict[str, dict] = {}
    for budget, rows in by_budget.items():
        target = np.asarray(
            [row["log_future_prediction_mse"] for row in rows],
            dtype=np.float64,
        )
        drift = np.asarray(
            [row["log_t4_reference_drift"] for row in rows],
            dtype=np.float64,
        )
        descriptor_correlations = {}
        for name in EXPANDED_FEATURES[4:]:
            values = np.asarray([row[name] for row in rows], dtype=np.float64)
            descriptor_correlations[name] = {
                "vs_log_future_prediction_mse": _spearman(values, target),
                "vs_log_t4_reference_drift": _spearman(values, drift),
            }
        feature_sets = {
            "t4_only": CURRENT_FEATURES[:4],
            "t4_plus_residual_only": (
                *CURRENT_FEATURES[:4],
                "log_residual_variance",
            ),
            "t4_plus_geometry_only": (
                *CURRENT_FEATURES[:4],
                "log_ac_covariance_shape",
            ),
            "t4_plus_current_confidence": CURRENT_FEATURES,
            "t4_plus_exposure": (
                *CURRENT_FEATURES[:4],
                "log1p_spike_exposure",
            ),
            "t4_plus_analytic_se": (
                *CURRENT_FEATURES[:4],
                "log_ac_standard_error_trace",
            ),
            "t4_plus_expanded_confidence": EXPANDED_FEATURES,
        }
        models = {
            name: _standardized_ridge_loso(rows, features)
            for name, features in feature_sets.items()
        }
        baseline = models["t4_only"]
        current = models["t4_plus_current_confidence"]
        expanded = models["t4_plus_expanded_confidence"]
        budget_results[str(budget)] = {
            "unit_rows": len(rows),
            "future_trial_count": args.reference_pool - budget,
            "descriptor_spearman": descriptor_correlations,
            "loso_ridge": {
                **models,
                "current_minus_t4_global_r2": (
                    current["global_r2"] - baseline["global_r2"]
                ),
                "expanded_minus_current_global_r2": (
                    expanded["global_r2"] - current["global_r2"]
                ),
            },
        }

    result = {
        "schema_version": 1,
        "purpose": "train_only_t4_confidence_predictive_validity",
        "interpretation_boundary": (
            "Descriptor diagnostic only; does not establish decoding improvement."
        ),
        "protocol": {
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "data_dir": str(data_dir),
            "split": "train",
            "session_count": 27,
            "budgets": list(args.budgets),
            "reference_pool": args.reference_pool,
            "fit_trials": "chronological_rewarded_trials[0:M]",
            "future_score_trials": "chronological_rewarded_trials[M:50]",
            "validation_session_nwb_opened": False,
            "formal_test_session_nwb_opened": False,
        },
        "budgets": budget_results,
        "train_nwb_receipts": receipts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        budget: {
            "current_minus_t4_global_r2": values["loso_ridge"][
                "current_minus_t4_global_r2"
            ],
            "expanded_minus_current_global_r2": values["loso_ridge"][
                "expanded_minus_current_global_r2"
            ],
            "current_global_r2": values["loso_ridge"][
                "t4_plus_current_confidence"
            ]["global_r2"],
            "expanded_global_r2": values["loso_ridge"][
                "t4_plus_expanded_confidence"
            ]["global_r2"],
        }
        for budget, values in budget_results.items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
