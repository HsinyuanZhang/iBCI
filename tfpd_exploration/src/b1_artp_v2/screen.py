"""Source screen for fixed-profile ARTP-P and causal-growing control."""
from __future__ import annotations

import time

import numpy as np

from tfpd_exploration.src.b1_artp_v1.screen import _score
from tfpd_exploration.src.b1_sfcj_v1 import data

from . import plan
from .core import (
    ARTPProfileConfig,
    build_profiled_payload,
    predict_from_profiled_payload,
    predict_from_profiled_rate,
)


def _date(date: str, config: ARTPProfileConfig) -> dict:
    calibration = data.calib_trials(date)
    references = calibration[:3]
    query = calibration[3:] + data.minival_trials(date)
    payload = build_profiled_payload(references)
    fixed, permuted, neural_free, growing = [], [], [], []
    rates = [data.rate_view(trial.tx).astype(np.float64) for trial in query]
    reference_rates = np.asarray(payload["reference_rates"])
    count = int(reference_rates.shape[0] * reference_rates.shape[1])
    running_sum = reference_rates.sum(axis=(0, 1))
    running_square_sum = (reference_rates * reference_rates).sum(axis=(0, 1))
    for index, trial in enumerate(query):
        fixed.append(predict_from_profiled_payload(trial.tx, payload, config=config)[0])
        permuted.append(
            predict_from_profiled_payload(
                query[(index + 1) % len(query)].tx, payload, config=config
            )[0]
        )
        neural_free.append(
            predict_from_profiled_payload(
                trial.tx, payload, config=config, use_query_neural=False
            )[0]
        )
        mean = running_sum / count
        variance = np.maximum(running_square_sum / count - mean * mean, 0.0)
        std = np.sqrt(variance)
        std = np.where(std < 1e-6, 1.0, std)
        growing.append(
            predict_from_profiled_rate(
                rates[index], payload, config=config, channel_mean=mean, channel_std=std
            )[0]
        )
        # Decode-before-commit: current query activity affects only future profile states.
        running_sum += rates[index].sum(axis=0)
        running_square_sum += (rates[index] * rates[index]).sum(axis=0)
        count += rates[index].shape[0]
    targets = [trial.spectrogram for trial in query]
    template = np.asarray(payload["median_template"])
    systems = {
        "TPL-M3-median": _score([template] * len(query), targets),
        "ARTP-P": _score(fixed, targets),
        "ARTP-P-cyclic-query-signature": _score(permuted, targets),
        "ARTP-P-neural-free-reliability": _score(neural_free, targets),
        "ARTP-P-growing-profile": _score(growing, targets),
    }
    primary = systems["ARTP-P"]["mse_mean"]
    return {
        "date": date,
        "n_query": len(query),
        "systems": systems,
        "gain_vs_tpl": systems["TPL-M3-median"]["mse_mean"] - primary,
        "relative_gain_vs_tpl": (systems["TPL-M3-median"]["mse_mean"] - primary)
        / systems["TPL-M3-median"]["mse_mean"],
        "correct_vs_cyclic": systems["ARTP-P-cyclic-query-signature"]["mse_mean"] - primary,
        "correct_vs_neural_free": systems["ARTP-P-neural-free-reliability"]["mse_mean"] - primary,
        "fixed_vs_growing": systems["ARTP-P-growing-profile"]["mse_mean"] - primary,
        "profile_authority": {
            "mean_sha256": payload["array_sha256"]["channel_profile_mean"],
            "std_sha256": payload["array_sha256"]["channel_profile_std"],
            "frozen_after_m3": True,
            "decode_before_commit": True,
        },
    }


def _bootstrap_profile(rows: list[dict]) -> dict:
    rng = np.random.default_rng(plan.BOOTSTRAP_SEED)
    names = {
        "gain_vs_tpl": ("TPL-M3-median", "ARTP-P"),
        "correct_vs_cyclic": ("ARTP-P-cyclic-query-signature", "ARTP-P"),
        "correct_vs_neural_free": ("ARTP-P-neural-free-reliability", "ARTP-P"),
    }
    by_date = []
    for row in rows:
        by_date.append(
            {
                key: np.asarray(row["systems"][left]["per_trial_mse"])
                - np.asarray(row["systems"][right]["per_trial_mse"])
                for key, (left, right) in names.items()
            }
        )
    samples = {key: np.empty(plan.BOOTSTRAP_REPLICATES) for key in names}
    for repeat in range(plan.BOOTSTRAP_REPLICATES):
        selected_dates = rng.integers(0, len(by_date), size=len(by_date))
        for key in names:
            date_means = []
            for date_index in selected_dates:
                values = by_date[int(date_index)][key]
                selected_trials = rng.integers(0, len(values), size=len(values))
                date_means.append(float(np.mean(values[selected_trials])))
            samples[key][repeat] = float(np.mean(date_means))
    return {
        key: {
            "mean": float(np.mean(values)),
            "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
            "probability_strictly_positive": float(np.mean(values > 0.0)),
        }
        for key, values in samples.items()
    }
def run_source_screen(config: ARTPProfileConfig = ARTPProfileConfig()) -> dict:
    config.validate()
    started = time.time()
    rows = [_date(date, config) for date in plan.HELD_IN_DATES]
    aggregate = {}
    for key in ("gain_vs_tpl", "correct_vs_cyclic", "correct_vs_neural_free", "fixed_vs_growing"):
        values = [float(row[key]) for row in rows]
        aggregate[key] = {
            "equal_date_mean": float(np.mean(values)),
            "positive_dates": int(sum(value > 0.0 for value in values)),
            "per_date": values,
        }
    gates = {key: aggregate[key]["positive_dates"] >= count for key, count in plan.PRIMARY_GATES.items()}
    return {
        "schema": plan.SCHEMA,
        "status": "SOURCE_SCREEN_COMPLETE",
        "method": "ARTP-P",
        "config": {
            "m": plan.M,
            "tau": config.tau,
            "reliability_strength": config.reliability_strength,
            "raw_profile_mix": config.raw_profile_mix,
        },
        "selection_disclosure": "Profile mix and ARTP hyperparameters are exploratory source-selected; held-out is the confirmation.",
        "rows": rows,
        "aggregate": aggregate,
        "hierarchical_bootstrap": _bootstrap_profile(rows),
        "gates": gates,
        "all_primary_gates_pass": bool(all(gates.values())),
        "continual_profile_promoted": bool(aggregate["fixed_vs_growing"]["equal_date_mean"] < 0.0),
        "continual_profile_decision": "STOP_GROWING_PROFILE" if aggregate["fixed_vs_growing"]["equal_date_mean"] >= 0.0 else "PROMOTE_GROWING_PROFILE",
        "wall_seconds": float(time.time() - started),
    }
