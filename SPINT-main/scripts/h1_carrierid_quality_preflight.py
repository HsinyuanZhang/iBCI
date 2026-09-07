#!/usr/bin/env python3
"""Source-only CPU feasibility receipt for H1 CarrierID quality features.

This program is deliberately upstream of any new model or GPU run.  It opens
only the 11 fold-0 source held-in-calibration recordings, reconstructs the
already sealed source transform, and tests three candidate *source-legal*
signals:

``C1`` analytic EB confidence
    One extra per-channel dimension, the M=4 posterior shrinkage weight.

``Q1`` support-split forward-transfer quality
    Fit the fixed ridge mapping on support trials 1--2 and compare its
    channel-wise carrier vector to an independently fitted trials-3--4 vector.
    The eventual normal M=4 carrier still uses trials 1--4; Q1 is a quality
    feature only and does not touch query trials.

``P1`` per-channel source-date prior
    The mean Q1 vector over source dates, assessed by leave-one-source-date-out
    correlation before it is ever proposed for a new date.

There is no target, minival, formal, EvalAI, Trainer, CUDA, checkpoint, or
model-selection path.  A PASS is feasibility evidence only and never
authorizes a GPU run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.h1_carrierid_quality import (
    QUALITY_FEATURE_DIM,
    QUALITY_SCHEMA,
    confidence_from_m4_fit,
    leave_one_date_out_channel_prior,
    normalize_scalar_feature,
    pearson_correlation,
    source_scalar_feature_normalizer,
    split_forward_transfer_quality,
)
from src.data.h1_m4_eb_pilot import (
    FOLD0_DATE,
    H1_M4_FOLD0_SOURCE,
    fit_frozen_carrier,
    load_source_records,
    reconstruct_frozen_plan,
    session_date,
)
from src.h1_m4_eb_normalized_v2_contract import (
    array_sha256,
    sha256_file,
    write_immutable_json,
)


PREFLIGHT_SCHEMA = "h1_carrierid_quality_source_only_cpu_preflight_v1"
PREFLIGHT_STATUS = "PASS_H1_CARRIERID_QUALITY_SOURCE_ONLY_CPU_FEASIBILITY_NOT_GPU_AUTHORIZED"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _summary(values: np.ndarray) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("feature summary requires finite nonempty values")
    return {
        "shape": list(vector.shape),
        "sha256": array_sha256(vector),
        "count": int(vector.size),
        "mean": float(vector.mean()),
        "std": float(vector.std(ddof=0)),
        "minimum": float(vector.min()),
        "maximum": float(vector.max()),
        "quantiles": {
            "q01": float(np.quantile(vector, 0.01)),
            "q25": float(np.quantile(vector, 0.25)),
            "q50": float(np.quantile(vector, 0.50)),
            "q75": float(np.quantile(vector, 0.75)),
            "q99": float(np.quantile(vector, 0.99)),
        },
    }


def _json_number(value: float) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _serializable_lodo(result: dict[str, Any]) -> dict[str, Any]:
    per_date: dict[str, Any] = {}
    for date, value in result["per_date"].items():
        prior = np.asarray(value["prior"], dtype=np.float64)
        per_date[date] = {
            "held_sessions": list(value["held_sessions"]),
            "training_session_count": len(value["training_sessions"]),
            "prior_summary": _summary(prior),
            "prior_vs_session_pearson": {
                name: _json_number(score) for name, score in value["prior_vs_session_pearson"].items()
            },
            "mean_session_pearson": _json_number(value["mean_session_pearson"]),
        }
    return {
        "source_date_count": int(result["source_date_count"]),
        "source_session_count": int(result["source_session_count"]),
        "channels": int(result["channels"]),
        "pooled_lodo_pearson": _json_number(result["pooled_lodo_pearson"]),
        "median_session_lodo_pearson": _json_number(result["median_session_lodo_pearson"]),
        "mean_session_lodo_pearson": _json_number(result["mean_session_lodo_pearson"]),
        "per_date": per_date,
    }


def _write_prior_once(path: Path, prior: np.ndarray) -> dict[str, Any]:
    output = path.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite source-date prior: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, np.asarray(prior, dtype=np.float64), allow_pickle=False)
    output.chmod(0o444)
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "mode": "0444",
        "array": _summary(prior),
    }


def run(
    *,
    data_dir: Path,
    raw_receipt: Path,
    eb_receipt: Path,
    output: Path,
    prior_output: Path,
) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise ValueError("quality preflight requires CUDA_VISIBLE_DEVICES to be unset")
    if output.exists() or prior_output.exists():
        raise FileExistsError("quality preflight output and prior paths must both be fresh")

    # `load_source_records` resolves and opens the explicit 11-source list.
    # It neither calls nor imports the target-only terminal loader.
    records = load_source_records(data_dir)
    if tuple(records) != H1_M4_FOLD0_SOURCE:
        raise ValueError("quality preflight source session order/scope drift")
    if any(session_date(name) == FOLD0_DATE for name in records):
        raise ValueError("target fold date leaked into quality source preflight")
    plan = reconstruct_frozen_plan(records, raw_receipt, eb_receipt)

    confidence_by_session: dict[str, np.ndarray] = {}
    quality_by_session: dict[str, np.ndarray] = {}
    date_by_session: dict[str, str] = {}
    per_session: dict[str, Any] = {}
    confidence_quality_by_session: dict[str, float] = {}
    for name in H1_M4_FOLD0_SOURCE:
        record = records[name]
        support_values = tuple(record.trial_values[:4])
        full_fit = fit_frozen_carrier(record, plan, support_values)
        confidence = confidence_from_m4_fit(full_fit)
        split = split_forward_transfer_quality(record, plan, support_values)
        quality = np.asarray(split["quality"], dtype=np.float64)
        valid = np.asarray(split["valid"], dtype=bool)
        _require(confidence.shape == quality.shape == (record.num_neurons,), f"{name}: feature shape drift")
        confidence_by_session[name] = confidence
        quality_by_session[name] = quality
        date_by_session[name] = record.date
        confidence_quality_by_session[name] = pearson_correlation(confidence, quality)
        per_session[name] = {
            "date": record.date,
            "support_trial_values": list(support_values),
            "forward_fit_trials": [float(value) for value in split["fit_trial_values"]],
            "forward_score_trials": [float(value) for value in split["score_trial_values"]],
            "fit_block_count": int(split["fit_block_count"][0]),
            "score_block_count": int(split["score_block_count"][0]),
            "confidence": _summary(confidence),
            "forward_transfer_quality": _summary(quality),
            "valid_channels": int(valid.sum()),
            "valid_channel_fraction": float(valid.mean()),
            "confidence_vs_quality_pearson": _json_number(confidence_quality_by_session[name]),
        }

    confidence_matrix = np.stack([confidence_by_session[name] for name in H1_M4_FOLD0_SOURCE], axis=0)
    quality_matrix = np.stack([quality_by_session[name] for name in H1_M4_FOLD0_SOURCE], axis=0)
    confidence_normalizer = source_scalar_feature_normalizer(confidence_matrix)
    quality_normalizer = source_scalar_feature_normalizer(quality_matrix)
    normalized_confidence = normalize_scalar_feature(confidence_matrix, confidence_normalizer)
    normalized_quality = normalize_scalar_feature(quality_matrix, quality_normalizer)
    lodo = leave_one_date_out_channel_prior(quality_by_session, date_by_session)
    source_prior = np.asarray(lodo["per_channel_all_source_prior"], dtype=np.float64)
    prior_artifact = _write_prior_once(prior_output, source_prior)
    per_session_correlation = [
        confidence_quality_by_session[name]
        for name in H1_M4_FOLD0_SOURCE
    ]
    pooled_confidence_quality = pearson_correlation(confidence_matrix, quality_matrix)
    lodo_json = _serializable_lodo(lodo)
    date_means = [float(value["mean_session_pearson"]) for value in lodo["per_date"].values()]

    # These are frozen decision rules for whether a future *source-trained*
    # candidate deserves implementation.  They cannot be overridden by an H1
    # target score, and even all PASS clauses below do not authorize a GPU.
    confidence_has_directional_source_evidence = (
        np.isfinite(pooled_confidence_quality)
        and pooled_confidence_quality > 0.0
        and int(sum(score > 0.0 for score in per_session_correlation if np.isfinite(score))) >= 6
    )
    dynamic_quality_is_constructible = bool(
        np.isfinite(quality_matrix).all() and float(np.mean(quality_matrix >= 0.0)) == 1.0
        and min(float(per_session[name]["valid_channel_fraction"]) for name in per_session) >= 0.95
    )
    static_prior_has_cross_date_evidence = (
        np.isfinite(lodo["pooled_lodo_pearson"])
        and lodo["pooled_lodo_pearson"] > 0.0
        and np.isfinite(lodo["median_session_lodo_pearson"])
        and lodo["median_session_lodo_pearson"] > 0.0
        and int(sum(score > 0.0 for score in date_means if np.isfinite(score))) >= 3
    )
    receipt = {
        "schema": PREFLIGHT_SCHEMA,
        "status": PREFLIGHT_STATUS,
        "scope": {
            "opened": "exactly 11 public fold-0 source held-in-calibration NWBs",
            "source_sessions": list(H1_M4_FOLD0_SOURCE),
            "target_recordings_opened": 0,
            "target_recordings_enumerated": 0,
            "minival_opened_or_enumerated": False,
            "formal_heldout_opened_or_enumerated": False,
            "evalai_opened_or_enumerated": False,
            "cuda_constructed_or_launched": False,
            "trainer_launched": False,
            "checkpoint_created_or_selected": False,
        },
        "source_binding": {
            "fold_date": FOLD0_DATE,
            "raw_receipt_path": str(raw_receipt.resolve()),
            "raw_receipt_sha256": sha256_file(raw_receipt),
            "eb_receipt_path": str(eb_receipt.resolve()),
            "eb_receipt_sha256": sha256_file(eb_receipt),
            "frozen_transform_sha256": plan.transform_sha256,
            "source_files": {
                name: records[name].input_sha256 for name in H1_M4_FOLD0_SOURCE
            },
        },
        "feature_contract": {
            "schema": QUALITY_SCHEMA,
            "ordinary_carrier": "unchanged frozen M=4 [N,4] EB carrier fitted from support trials 1--4",
            "candidate_c1": {
                "extra_dimensions": QUALITY_FEATURE_DIM,
                "raw_feature": "analytic EB posterior shrinkage weight in [0,1] from ordinary M=4 fit",
                "source_only_normalizer": confidence_normalizer,
                "model_change_if_later_implemented": "CarrierID carrier_dim 4 -> 5; only first carrier-post linear input adds 32 weights",
            },
            "candidate_q1": {
                "extra_dimensions": QUALITY_FEATURE_DIM,
                "raw_feature": "per-channel (cosine(raw_carrier_1_2-mu, raw_carrier_3_4-mu)+1)/2",
                "source_only_normalizer": quality_normalizer,
                "fit_scope": "support trials 1--2 only",
                "score_scope": "support trials 3--4 only",
                "ordinary_m4_carrier_still_uses": "support trials 1--4",
                "uses_query_trials": False,
            },
            "candidate_p1": {
                "extra_dimensions": QUALITY_FEATURE_DIM,
                "raw_feature": "per-channel all-source mean Q1 forward-transfer agreement",
                "artifact": prior_artifact,
                "runtime_scope": "all source dates only; no target-date quality fit",
            },
        },
        "per_source_recording": per_session,
        "pooled_source_statistics": {
            "records": len(H1_M4_FOLD0_SOURCE),
            "channels_per_record": int(confidence_matrix.shape[1]),
            "source_channel_samples": int(confidence_matrix.size),
            "confidence": _summary(confidence_matrix),
            "forward_transfer_quality": _summary(quality_matrix),
            "normalized_confidence": _summary(normalized_confidence),
            "normalized_forward_transfer_quality": _summary(normalized_quality),
            "confidence_vs_quality_pooled_pearson": _json_number(pooled_confidence_quality),
            "confidence_vs_quality_per_recording_pearson": {
                name: _json_number(confidence_quality_by_session[name])
                for name in H1_M4_FOLD0_SOURCE
            },
        },
        "per_channel_source_date_prior": {
            "artifact": prior_artifact,
            "leave_one_source_date_out": lodo_json,
        },
        "frozen_kill_rules": {
            "C1_confidence": {
                "stop_before_gpu_if": "source confidence normalizer is degenerate, or pooled confidence-vs-Q1 Pearson <= 0 and fewer than 6/11 recordings have positive Pearson",
                "observed_directional_source_evidence": confidence_has_directional_source_evidence,
                "interpretation": "A positive association is only a mechanism screen; it is not a decoding gain claim.",
            },
            "Q1_split_forward_transfer": {
                "stop_before_gpu_if": "any source feature is nonfinite, normalizer is degenerate, or fewer than 95% of channels have a defined split cosine in any source recording",
                "observed_constructible": dynamic_quality_is_constructible,
                "interpretation": "Q1 must remain a support-only quality feature. Any query-derived version is leakage diagnostic only.",
            },
            "P1_static_source_date_prior": {
                "stop_before_gpu_if": "pooled LODO Pearson <= 0, median held-session LODO Pearson <= 0, or fewer than 3/5 held dates have positive mean held-session Pearson",
                "observed_cross_date_evidence": static_prior_has_cross_date_evidence,
                "interpretation": "If this fails, do not add a static electrode/channel lookup as a fallback.",
            },
            "common": {
                "no_gpu_authorized_by_this_receipt": True,
                "no_target_score_may_override_cpu_kill_rule": True,
                "no_target_or_formal_data_used_for_model_selection": True,
            },
        },
    }
    receipt_path, digest = write_immutable_json(output, receipt)
    return {"receipt_path": str(receipt_path), "receipt_sha256": digest, "status": PREFLIGHT_STATUS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "000954")
    parser.add_argument("--raw-receipt", type=Path, required=True)
    parser.add_argument("--eb-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
