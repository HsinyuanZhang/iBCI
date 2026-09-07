"""Held-in source screen and controls for ARTP."""
from __future__ import annotations

import time

import numpy as np

from tfpd_exploration.src.b1_sfcj_v1 import data
from tfpd_exploration.src.b1_sfcj_v1.metric import official_metric_from_trials
from tfpd_exploration.src.b1_sfcj_v1.util import sha256_array

from . import plan
from .core import ARTPConfig, build_payload, predict_from_signature


def _score(predictions, targets) -> dict:
    value = official_metric_from_trials(predictions, targets)
    return {
        "mse_mean": float(value["MSE Mean"]),
        "mse_std": float(value["MSE Std."]),
        "per_trial_mse": [float(x) for x in value["per_trial_mse"]],
        "n_trials": int(value["n_trials"]),
        "prediction_sha256": sha256_array(np.stack(predictions)),
        "target_sha256": sha256_array(np.stack(targets)),
    }


def _date_screen(date: str, config: ARTPConfig) -> dict:
    calibration = data.calib_trials(date)
    references = calibration[: plan.M]
    query = calibration[plan.M :] + data.minival_trials(date)
    payload = build_payload(references)
    # Compute the public signature once per query; the cyclic control changes
    # only query-to-target pairing and never touches targets during inference.
    from .core import activity_signature

    signatures = [activity_signature(trial.tx) for trial in query]
    artp, permuted, neural_free, evidence = [], [], [], []
    for index, signature in enumerate(signatures):
        prediction, row = predict_from_signature(signature, payload, config=config)
        artp.append(prediction)
        evidence.append(row)
        permuted.append(
            predict_from_signature(signatures[(index + 1) % len(signatures)], payload, config=config)[0]
        )
        neural_free.append(
            predict_from_signature(signature, payload, config=config, use_query_neural=False)[0]
        )
    template = np.asarray(payload["median_template"], dtype=np.float64)
    targets = [np.asarray(trial.spectrogram, dtype=np.float64) for trial in query]
    systems = {
        "TPL-M3-median": _score([template] * len(query), targets),
        "ARTP": _score(artp, targets),
        "ARTP-cyclic-query-signature": _score(permuted, targets),
        "ARTP-neural-free-reliability": _score(neural_free, targets),
    }
    primary = systems["ARTP"]["mse_mean"]
    return {
        "date": date,
        "m": plan.M,
        "n_query": len(query),
        "query_roster": [f"{trial.split}:{trial.trial_index}" for trial in query],
        "systems": systems,
        "gain_vs_tpl": systems["TPL-M3-median"]["mse_mean"] - primary,
        "relative_gain_vs_tpl": (
            systems["TPL-M3-median"]["mse_mean"] - primary
        )
        / systems["TPL-M3-median"]["mse_mean"],
        "correct_vs_cyclic": systems["ARTP-cyclic-query-signature"]["mse_mean"] - primary,
        "correct_vs_neural_free": systems["ARTP-neural-free-reliability"]["mse_mean"] - primary,
        "payload_authority": {
            key: payload[key]
            for key in (
                "schema",
                "m",
                "signature_law",
                "reference_signature_sha256",
                "reference_spectrogram_sha256",
                "median_template_sha256",
                "query_labels_used",
                "model_updates",
            )
        },
        "weight_rows": [
            {
                "query": f"{query[i].split}:{query[i].trial_index}",
                "similarities": row["similarities"],
                "weights": row["weights"],
                "prediction_sha256": row["prediction_sha256"],
            }
            for i, row in enumerate(evidence)
        ],
    }


def _bootstrap(rows: list[dict]) -> dict:
    rng = np.random.default_rng(plan.BOOTSTRAP_SEED)
    by_date = []
    for row in rows:
        systems = row["systems"]
        by_date.append(
            {
                "gain_vs_tpl": np.asarray(systems["TPL-M3-median"]["per_trial_mse"])
                - np.asarray(systems["ARTP"]["per_trial_mse"]),
                "correct_vs_cyclic": np.asarray(
                    systems["ARTP-cyclic-query-signature"]["per_trial_mse"]
                )
                - np.asarray(systems["ARTP"]["per_trial_mse"]),
                "correct_vs_neural_free": np.asarray(
                    systems["ARTP-neural-free-reliability"]["per_trial_mse"]
                )
                - np.asarray(systems["ARTP"]["per_trial_mse"]),
            }
        )
    samples = {key: np.empty(plan.BOOTSTRAP_REPLICATES) for key in by_date[0]}
    for repeat in range(plan.BOOTSTRAP_REPLICATES):
        picked_dates = rng.integers(0, len(by_date), size=len(by_date))
        for key in samples:
            date_means = []
            for date_index in picked_dates:
                values = by_date[int(date_index)][key]
                picked_trials = rng.integers(0, len(values), size=len(values))
                date_means.append(float(np.mean(values[picked_trials])))
            samples[key][repeat] = float(np.mean(date_means))
    return {
        key: {
            "mean": float(np.mean(values)),
            "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
            "probability_strictly_positive": float(np.mean(values > 0.0)),
        }
        for key, values in samples.items()
    }


def run_source_screen(config: ARTPConfig = ARTPConfig()) -> dict:
    config.validate()
    started = time.time()
    rows = [_date_screen(date, config) for date in plan.HELD_IN_DATES]
    aggregate = {}
    for key in ("gain_vs_tpl", "correct_vs_cyclic", "correct_vs_neural_free"):
        values = [float(row[key]) for row in rows]
        aggregate[key] = {
            "equal_date_mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "positive_dates": int(sum(value > 0.0 for value in values)),
            "per_date": values,
        }
    gates = {
        "gain_vs_tpl": aggregate["gain_vs_tpl"]["positive_dates"]
        >= plan.PRIMARY_GATES["gain_vs_tpl_m3_median_strictly_positive_dates"],
        "correct_vs_cyclic": aggregate["correct_vs_cyclic"]["positive_dates"]
        >= plan.PRIMARY_GATES[
            "correct_vs_cyclic_query_signature_strictly_positive_dates"
        ],
        "correct_vs_neural_free": aggregate["correct_vs_neural_free"]["positive_dates"]
        >= plan.PRIMARY_GATES[
            "correct_vs_neural_free_reliability_strictly_positive_dates"
        ],
    }
    return {
        "schema": plan.SCHEMA,
        "status": "SOURCE_SCREEN_COMPLETE",
        "method": "ARTP",
        "method_expansion": "Activity-Reliability Template Profiling",
        "config": {
            "m": plan.M,
            "signature": plan.SIGNATURE,
            "tau": config.tau,
            "reliability_strength": config.reliability_strength,
            "mixing": config.mixing,
        },
        "selection_disclosure": (
            "The primary hyperparameters were frozen after exploratory held-in source analysis. "
            "This screen is constructibility/mechanism evidence, not an unbiased estimate. "
            "Any held-out EvalAI submission must be treated as the confirmation."
        ),
        "target_contract": {
            "calibration_labels": "first three released calibration spectrograms only",
            "calibration_neural": "first three released calibration neural trials only",
            "query_neural": True,
            "query_labels": False,
            "model_updates": 0,
        },
        "rows": rows,
        "aggregate": aggregate,
        "hierarchical_bootstrap": _bootstrap(rows),
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "wall_seconds": float(time.time() - started),
    }
