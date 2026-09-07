#!/usr/bin/env python3
"""CPU-only H1 M=2 date-LODO gate for a rotationally invariant carrier.

This is a distinct, fail-closed preflight from the signed-direction AFC4
audit.  It imports that audit's held-in-only loader, M=2 block construction,
date-LODO partitioning, source PCA, and source-only lag/ridge selection;
therefore it deliberately cannot silently reintroduce the stopped M=1/raw-bin
protocol.

The fixed candidate is exactly four per-channel values:

    [log1p(||W_combined||),
     log1p(max(b_combined, 0)),
     symmetric_per_channel_cross_trial_gain,
     log1p(combined_residual_variance)]

``W_combined``/``b_combined`` are fit from both deployment-permitted target
calibration trials after all source-only choices are frozen.  The crucial
third value is never an in-sample fit statistic: fit trial 1 with correctly
paired velocity blocks and predict trial 2 with correctly paired velocity
blocks, repeat in the reverse direction, then average the two per-channel
held-trial gains.  In each direction the baseline is the *training trial's*
per-channel mean firing rate.

For every target recording, the same evaluation is repeated for 31 declared,
nonzero deterministic circular rotations of velocity blocks *inside each fit
trial*.  Evaluation labels remain correctly paired.  Thus the null preserves
the rate/velocity marginal distributions but destroys only fit-side label
association and can never move a block across a TrialNum boundary.

I0 is a conjunction, not a score-shopping opportunity.  All six dates must
be defined and at least four dates must independently satisfy each of:

* trial-1 vs trial-2 ``||W||`` Spearman >= 0.5;
* symmetric correct cross-trial gain > 0;
* date correct gain > that date's 31-null q95.

The source-only normalizer is recorded for a later descriptor experiment but
is not applied to, or allowed to influence, I0.  A receipt from this program
never authorizes GPU use by itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


# Reuse the signed audit as the sole authority for H1 file scope, M=2 trial
# extraction, 100-ms blocks, source-only date-LODO planning, PCA, and ridge
# fitting.  Do not copy its data protocol into this independent gate.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import audit_h1_afc4_m2_date_lodo as signed_audit  # noqa: E402


PROTOCOL_VERSION = "h1_invariant_carrier_m2_date_lodo_cpu_audit_v1"
NULL_REPLICATES = 31
NULL_SEED = 20260807
I0_NORM_SPEARMAN_THRESHOLD = 0.50
I0_MIN_PASSING_DATES = 4
NORMALIZER_SCALE_FLOOR = 1.0e-6
EPS = signed_audit.EPS
CARRIER_FIELD_NAMES: tuple[str, ...] = (
    "log1p_w_combined_norm",
    "log1p_positive_b_combined",
    "symmetric_per_channel_cross_trial_gain",
    "log1p_combined_residual_variance",
)


class InvariantAuditError(ValueError):
    """A numerical or protocol violation which must fail this route closed."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON encode {type(value)!r}")


def _finite_summary(values: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _correct_aligned_trials(
    record: signed_audit.H1M2Record,
    plan: signed_audit.SourcePlan,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return rate/score pairs with the true within-trial association intact."""

    rate_one, score_one = signed_audit._align_trial(
        record.trials[0], plan.basis, lag_blocks=plan.selected_lag_blocks
    )
    rate_two, score_two = signed_audit._align_trial(
        record.trials[1], plan.basis, lag_blocks=plan.selected_lag_blocks
    )
    return rate_one, score_one, rate_two, score_two


def _per_channel_transfer_gain(
    train_fit: signed_audit.AffineFit,
    train_rates: np.ndarray,
    evaluation_rates: np.ndarray,
    evaluation_scores_correct: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Held-trial gain using exactly the train-trial per-channel mean baseline."""

    train = np.asarray(train_rates, dtype=np.float64)
    target = np.asarray(evaluation_rates, dtype=np.float64)
    if train.ndim != 2 or target.ndim != 2 or train.shape[1] != target.shape[1]:
        raise InvariantAuditError("cross-trial train/evaluation rate arrays are incompatible")
    prediction = signed_audit._predict(train_fit, evaluation_scores_correct)
    if prediction.shape != target.shape:
        raise InvariantAuditError("cross-trial prediction/evaluation shapes disagree")
    # This is intentionally *not* the ridge-fit intercept.  The requested
    # baseline is the raw per-channel firing-rate mean of the fit trial.
    baseline_rate = train.mean(axis=0)
    baseline = np.broadcast_to(baseline_rate[None, :], target.shape)
    full_sse = np.square(prediction - target).sum(axis=0)
    baseline_sse = np.square(baseline - target).sum(axis=0)
    defined = np.isfinite(full_sse) & np.isfinite(baseline_sse) & (baseline_sse > EPS)
    gain = np.full(target.shape[1], np.nan, dtype=np.float64)
    gain[defined] = 1.0 - full_sse[defined] / baseline_sse[defined]
    return gain, {
        "baseline": "fit-trial raw per-channel rate mean",
        "evaluation_label_association": "correct within-trial velocity pairing",
        "defined_channels": int(defined.sum()),
        "total_channels": int(target.shape[1]),
        "gain_summary": _finite_summary(gain),
    }


def _symmetric_cross_trial_gain(
    *,
    rate_one: np.ndarray,
    fit_score_one: np.ndarray,
    eval_score_one_correct: np.ndarray,
    rate_two: np.ndarray,
    fit_score_two: np.ndarray,
    eval_score_two_correct: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit both directions and return a per-channel symmetric held-trial gain."""

    fit_one = signed_audit._fit_affine(rate_one, fit_score_one, ridge=ridge)
    fit_two = signed_audit._fit_affine(rate_two, fit_score_two, ridge=ridge)
    forward, forward_audit = _per_channel_transfer_gain(
        fit_one, rate_one, rate_two, eval_score_two_correct
    )
    reverse, reverse_audit = _per_channel_transfer_gain(
        fit_two, rate_two, rate_one, eval_score_one_correct
    )
    defined = np.isfinite(forward) & np.isfinite(reverse)
    symmetric = np.full(forward.shape, np.nan, dtype=np.float64)
    symmetric[defined] = 0.5 * (forward[defined] + reverse[defined])
    return symmetric, {
        "status": "defined" if defined.any() else "undefined",
        "symmetric_rule": "0.5 * (trial1->trial2 gain + trial2->trial1 gain), both directions defined",
        "defined_channels": int(defined.sum()),
        "total_channels": int(symmetric.size),
        "coverage_fraction": float(defined.mean()),
        "symmetric_gain_summary": _finite_summary(symmetric),
        "trial1_to_trial2": forward_audit,
        "trial2_to_trial1": reverse_audit,
    }


def _combined_fit_and_descriptor(
    *,
    record: signed_audit.H1M2Record,
    plan: signed_audit.SourcePlan,
    rate_one: np.ndarray,
    score_one: np.ndarray,
    rate_two: np.ndarray,
    score_two: np.ndarray,
    symmetric_gain: np.ndarray,
) -> tuple[np.ndarray, signed_audit.AffineFit, dict[str, Any]]:
    """Construct the immutable four-column raw carrier from correct M=2 data."""

    combined_rate = np.concatenate((rate_one, rate_two), axis=0)
    combined_score = np.concatenate((score_one, score_two), axis=0)
    combined_fit = signed_audit._fit_affine(combined_rate, combined_score, ridge=plan.selected_ridge)
    residual = combined_rate - signed_audit._predict(combined_fit, combined_score)
    residual_variance = np.mean(np.square(residual), axis=0)
    w_norm = np.linalg.norm(combined_fit.weights, axis=1)
    b_positive = np.maximum(combined_fit.intercept, 0.0)
    # A channel with a zero-variance evaluation target has no valid held-trial
    # R²-like gain.  It receives a neutral descriptor value while the coverage
    # remains explicit and the date scalar is computed only over valid channels.
    gain_finite = np.where(np.isfinite(symmetric_gain), symmetric_gain, 0.0)
    descriptor = np.column_stack(
        (
            np.log1p(w_norm),
            np.log1p(b_positive),
            gain_finite,
            np.log1p(residual_variance),
        )
    ).astype(np.float64)
    if not np.isfinite(descriptor).all():
        raise InvariantAuditError("invariant carrier contains non-finite values")
    descriptor_hash = _sha256_bytes(descriptor.tobytes(order="C"))
    return descriptor, combined_fit, {
        "field_names": list(CARRIER_FIELD_NAMES),
        "raw_shape": [int(descriptor.shape[0]), int(descriptor.shape[1])],
        "raw_descriptor_sha256": descriptor_hash,
        "field_summaries": {
            name: _finite_summary(descriptor[:, column])
            for column, name in enumerate(CARRIER_FIELD_NAMES)
        },
        "combined_fit": {
            "label_association": "correct within-trial velocity pairing",
            "blocks": int(combined_rate.shape[0]),
            "design_rank": int(combined_fit.rank),
            "design_condition": float(combined_fit.condition),
            "residual_variance_summary": _finite_summary(residual_variance),
        },
        "undefined_cross_trial_gain_replaced_with_neutral_zero": int((~np.isfinite(symmetric_gain)).sum()),
    }


def deterministic_rotation_offset(
    *,
    session_name: str,
    trial_number: float,
    aligned_block_count: int,
    replicate: int,
) -> int:
    """A declared nonidentity rotation local to one already-aligned trial."""

    if not 0 <= int(replicate) < NULL_REPLICATES:
        raise InvariantAuditError(f"null replicate must be in [0,{NULL_REPLICATES}), got {replicate}")
    if int(aligned_block_count) < 2:
        raise InvariantAuditError("fit-side circular null needs at least two blocks in each trial")
    payload = (
        f"{PROTOCOL_VERSION}:fit-side-ls:{NULL_SEED}:{session_name}:{trial_number}:"
        f"{aligned_block_count}:{replicate}"
    ).encode("utf-8")
    return 1 + (int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (aligned_block_count - 1))


def fit_side_rotated_scores(
    scores_correct: np.ndarray,
    *,
    session_name: str,
    trial_number: float,
    replicate: int,
) -> tuple[np.ndarray, int]:
    """Rotate only one trial's score rows; no cross-trial exchange is possible."""

    scores = np.asarray(scores_correct, dtype=np.float64)
    if scores.ndim != 2:
        raise InvariantAuditError("fit-side score rows must be a matrix")
    offset = deterministic_rotation_offset(
        session_name=session_name,
        trial_number=trial_number,
        aligned_block_count=int(scores.shape[0]),
        replicate=replicate,
    )
    return np.roll(scores, shift=offset, axis=0), offset


def _correct_record_quantities(
    record: signed_audit.H1M2Record,
    plan: signed_audit.SourcePlan,
) -> dict[str, Any]:
    """Fit correct pairs once, then expose only derived carrier/audit quantities."""

    rate_one, score_one, rate_two, score_two = _correct_aligned_trials(record, plan)
    fit_one = signed_audit._fit_affine(rate_one, score_one, ridge=plan.selected_ridge)
    fit_two = signed_audit._fit_affine(rate_two, score_two, ridge=plan.selected_ridge)
    norm_reliability = signed_audit.signed_w_reliability(fit_one, fit_two)
    symmetric_gain, transfer = _symmetric_cross_trial_gain(
        rate_one=rate_one,
        fit_score_one=score_one,
        eval_score_one_correct=score_one,
        rate_two=rate_two,
        fit_score_two=score_two,
        eval_score_two_correct=score_two,
        ridge=plan.selected_ridge,
    )
    descriptor, combined_fit, descriptor_audit = _combined_fit_and_descriptor(
        record=record,
        plan=plan,
        rate_one=rate_one,
        score_one=score_one,
        rate_two=rate_two,
        score_two=score_two,
        symmetric_gain=symmetric_gain,
    )
    return {
        "rate_one": rate_one,
        "score_one": score_one,
        "rate_two": rate_two,
        "score_two": score_two,
        "fit_one": fit_one,
        "fit_two": fit_two,
        "norm_reliability": norm_reliability,
        "symmetric_gain": symmetric_gain,
        "transfer": transfer,
        "descriptor": descriptor,
        "combined_fit": combined_fit,
        "descriptor_audit": descriptor_audit,
    }


def _null_replicate_gain(
    *,
    record: signed_audit.H1M2Record,
    plan: signed_audit.SourcePlan,
    correct: Mapping[str, Any],
    replicate: int,
) -> dict[str, Any]:
    """Fit rotated labels only; target-trial evaluation labels stay correct."""

    score_one_rotated, offset_one = fit_side_rotated_scores(
        correct["score_one"],
        session_name=record.session_name,
        trial_number=record.trials[0].trial_number,
        replicate=replicate,
    )
    score_two_rotated, offset_two = fit_side_rotated_scores(
        correct["score_two"],
        session_name=record.session_name,
        trial_number=record.trials[1].trial_number,
        replicate=replicate,
    )
    gain, transfer = _symmetric_cross_trial_gain(
        rate_one=correct["rate_one"],
        fit_score_one=score_one_rotated,
        eval_score_one_correct=correct["score_one"],
        rate_two=correct["rate_two"],
        fit_score_two=score_two_rotated,
        eval_score_two_correct=correct["score_two"],
        ridge=plan.selected_ridge,
    )
    summary = _finite_summary(gain)
    return {
        "replicate": int(replicate),
        "fit_side_only": True,
        "evaluation_labels": "correct within-trial velocity pairing",
        "trial1_rotation_offset_blocks": int(offset_one),
        "trial2_rotation_offset_blocks": int(offset_two),
        "trial1_rotation_nonzero": bool(offset_one != 0),
        "trial2_rotation_nonzero": bool(offset_two != 0),
        "symmetric_per_channel_gain_median": summary["median"],
        "symmetric_per_channel_gain_mean": summary["mean"],
        "defined_channels": int(transfer["defined_channels"]),
        "total_channels": int(transfer["total_channels"]),
        "coverage_fraction": float(transfer["coverage_fraction"]),
    }


def _source_only_normalizer(
    source_records: Iterable[signed_audit.H1M2Record],
    plan: signed_audit.SourcePlan,
) -> dict[str, Any]:
    """Record a target-excluded normalizer; I0 deliberately does not consume it."""

    records = tuple(source_records)
    raw: list[np.ndarray] = []
    hashes: dict[str, str] = {}
    for record in records:
        values = _correct_record_quantities(record, plan)["descriptor"]
        raw.append(values)
        hashes[record.session_name] = _sha256_bytes(values.tobytes(order="C"))
    if not raw:
        raise InvariantAuditError("source-only normalizer received no source recordings")
    values = np.concatenate(raw, axis=0)
    if values.ndim != 2 or values.shape[1] != len(CARRIER_FIELD_NAMES) or not np.isfinite(values).all():
        raise InvariantAuditError("source-only normalizer received invalid descriptor values")
    mean = values.mean(axis=0)
    raw_std = values.std(axis=0)
    scale = np.maximum(raw_std, NORMALIZER_SCALE_FLOOR)
    body = {
        "field_names": list(CARRIER_FIELD_NAMES),
        "source_sessions": [record.session_name for record in records],
        "source_descriptor_hashes": hashes,
        "mean": mean.tolist(),
        "raw_std": raw_std.tolist(),
        "scale_after_floor": scale.tolist(),
        "scale_floor": NORMALIZER_SCALE_FLOOR,
        "fit_rows": int(values.shape[0]),
        "fit_scope": "outer-source-dates-only",
        "used_by_i0_gate": False,
        "used_to_transform_i0_metrics": False,
    }
    return {**body, "normalizer_sha256": _sha256_bytes(_canonical_json(body).encode("utf-8"))}


def evaluate_target_record(
    record: signed_audit.H1M2Record,
    plan: signed_audit.SourcePlan,
) -> dict[str, Any]:
    """Evaluate one target recording with correct transfer and 31 fit-side nulls."""

    if record.date != plan.outer_date:
        raise InvariantAuditError(
            f"record {record.session_name} belongs to {record.date}, not outer date {plan.outer_date}"
        )
    correct = _correct_record_quantities(record, plan)
    transfer = correct["transfer"]
    correct_summary = _finite_summary(correct["symmetric_gain"])
    null_rows = tuple(
        _null_replicate_gain(record=record, plan=plan, correct=correct, replicate=replicate)
        for replicate in range(NULL_REPLICATES)
    )
    null_values = [row["symmetric_per_channel_gain_median"] for row in null_rows]
    if any(value is None or not np.isfinite(float(value)) for value in null_values):
        null_status = "undefined"
        null_median = None
        null_q95 = None
    else:
        null_array = np.asarray(null_values, dtype=np.float64)
        null_status = "defined"
        null_median = float(np.quantile(null_array, 0.50))
        null_q95 = float(np.quantile(null_array, 0.95))
    norm_spearman = correct["norm_reliability"]["w_norm_spearman_diagnostic"]
    correct_gain = correct_summary["median"]
    record_defined = (
        transfer["status"] == "defined"
        and isinstance(norm_spearman, (float, int))
        and np.isfinite(float(norm_spearman))
        and isinstance(correct_gain, (float, int))
        and np.isfinite(float(correct_gain))
        and null_status == "defined"
    )
    return {
        "status": "defined" if record_defined else "undefined",
        "session_name": record.session_name,
        "date": record.date,
        "input_nwb_sha256": record.input_sha256,
        "source_plan_sha256": plan.plan_sha256,
        "support": {
            "trial_count": 2,
            "trial_numbers": [record.trials[0].trial_number, record.trials[1].trial_number],
            "block_bins": signed_audit.BLOCK_BINS,
            "block_seconds": signed_audit.BLOCK_SECONDS,
            "all_finite_blocks_only": True,
            "activity_threshold_used": None,
            "query_labels_read": False,
            "trial_1": dict(record.trials[0].audit),
            "trial_2": dict(record.trials[1].audit),
        },
        "invariant_reliability": {
            "metric": "trial1-vs-trial2 per-channel ||W|| Spearman",
            "spearman": norm_spearman,
            "pearson": correct["norm_reliability"]["w_norm_pearson_diagnostic"],
            "defined_channels": correct["norm_reliability"]["primary_defined_channels"],
            "total_channels": correct["norm_reliability"]["primary_total_channels"],
        },
        "correct_cross_trial_transfer": {
            **transfer,
            "date_aggregate_input": "record-level median symmetric per-channel held-trial gain",
            "symmetric_per_channel_gain_median": correct_gain,
            "symmetric_per_channel_gain_mean": correct_summary["mean"],
            "in_sample_fit_gain_used": False,
            "fit_label_association": "correct within-trial velocity pairing",
            "evaluation_label_association": "correct within-trial velocity pairing",
        },
        "fit_side_label_rotation_null": {
            "status": null_status,
            "replicates": NULL_REPLICATES,
            "null_seed": NULL_SEED,
            "rotation_scope": "within each aligned fit trial only; no TrialNum crossing",
            "evaluation_label_association": "correct within-trial velocity pairing",
            "record_null_median": null_median,
            "record_null_q95": null_q95,
            "replicate_rows": [dict(row) for row in null_rows],
        },
        "carrier": correct["descriptor_audit"],
    }


def _finite_rows(rows: Iterable[Mapping[str, Any]], path: Sequence[str]) -> list[float]:
    values: list[float] = []
    for row in rows:
        current: Any = row
        for key in path:
            if not isinstance(current, Mapping):
                current = None
                break
            current = current.get(key)
        if isinstance(current, (int, float)) and np.isfinite(float(current)):
            values.append(float(current))
    return values


def _equal_recording_mean(values: Sequence[float], *, expected: int) -> float | None:
    if len(values) != expected:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def aggregate_date(
    record_rows: Sequence[Mapping[str, Any]],
    *,
    date: str,
    plan: signed_audit.SourcePlan,
    normalizer: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate same-day recordings equally before testing the three I0 clauses."""

    total = len(record_rows)
    defined = [row for row in record_rows if row.get("status") == "defined"]
    norm_values = _finite_rows(defined, ("invariant_reliability", "spearman"))
    correct_values = _finite_rows(
        defined, ("correct_cross_trial_transfer", "symmetric_per_channel_gain_median")
    )
    null_by_replicate: list[float] = []
    null_complete = len(defined) == total and total > 0
    if null_complete:
        for replicate in range(NULL_REPLICATES):
            values: list[float] = []
            for row in defined:
                null_rows = row["fit_side_label_rotation_null"]["replicate_rows"]
                if len(null_rows) != NULL_REPLICATES:
                    null_complete = False
                    break
                value = null_rows[replicate].get("symmetric_per_channel_gain_median")
                if not isinstance(value, (float, int)) or not np.isfinite(float(value)):
                    null_complete = False
                    break
                values.append(float(value))
            if not null_complete:
                break
            null_by_replicate.append(float(np.mean(np.asarray(values, dtype=np.float64))))
    status = (
        "defined"
        if len(defined) == total
        and total > 0
        and len(norm_values) == total
        and len(correct_values) == total
        and null_complete
        and len(null_by_replicate) == NULL_REPLICATES
        else "undefined"
    )
    correct_gain = _equal_recording_mean(correct_values, expected=total)
    norm_spearman = _equal_recording_mean(norm_values, expected=total)
    null_median = None if not null_by_replicate else float(np.quantile(null_by_replicate, 0.50))
    null_q95 = None if not null_by_replicate else float(np.quantile(null_by_replicate, 0.95))
    difference_q95 = (
        None
        if correct_gain is None or null_q95 is None
        else float(correct_gain - null_q95)
    )
    return {
        "status": status,
        "date": date,
        "recordings_total": int(total),
        "recordings_defined": int(len(defined)),
        "recording_sessions": [str(row.get("session_name")) for row in record_rows],
        "source_plan": plan.as_dict(),
        "source_only_normalizer": dict(normalizer),
        "date_invariant_w_norm_spearman": norm_spearman,
        "date_invariant_w_norm_spearman_rule": "equal-recording mean of record-level trial1-vs-trial2 Spearman",
        "date_correct_symmetric_cross_trial_gain": correct_gain,
        "date_correct_gain_rule": "equal-recording mean of record-level median symmetric held-trial gain",
        "date_null_gain_replicates": null_by_replicate,
        "date_null_gain_median": null_median,
        "date_null_gain_q95": null_q95,
        "date_correct_minus_null_q95": difference_q95,
        "date_null_rule": "for each replicate, equal-recording mean of record-level median fit-side-rotated/eval-correct gain",
        "in_sample_fit_gain_used": False,
        "recording_results": [dict(row) for row in record_rows],
    }


def i0_gate(date_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The three predeclared I0 clauses, reported separately and conjoined."""

    if len(date_rows) != 6:
        raise InvariantAuditError(f"I0 requires exactly six date rows, got {len(date_rows)}")
    all_dates_defined = all(row.get("status") == "defined" for row in date_rows)

    def finite_metric(row: Mapping[str, Any], name: str) -> float | None:
        value = row.get(name)
        return float(value) if isinstance(value, (int, float)) and np.isfinite(float(value)) else None

    norm_values = [finite_metric(row, "date_invariant_w_norm_spearman") for row in date_rows]
    correct_values = [finite_metric(row, "date_correct_symmetric_cross_trial_gain") for row in date_rows]
    margin_values = [finite_metric(row, "date_correct_minus_null_q95") for row in date_rows]
    metrics_defined = all(
        value is not None for values in (norm_values, correct_values, margin_values) for value in values
    )
    norm_count = sum(value >= I0_NORM_SPEARMAN_THRESHOLD for value in norm_values if value is not None)
    gain_count = sum(value > 0.0 for value in correct_values if value is not None)
    null_count = sum(value > 0.0 for value in margin_values if value is not None)
    norm_pass = bool(all_dates_defined and metrics_defined and norm_count >= I0_MIN_PASSING_DATES)
    gain_pass = bool(all_dates_defined and metrics_defined and gain_count >= I0_MIN_PASSING_DATES)
    null_pass = bool(all_dates_defined and metrics_defined and null_count >= I0_MIN_PASSING_DATES)
    passed = bool(norm_pass and gain_pass and null_pass)
    return {
        "status": "PASS_I0_INVARIANT_CARRIER_FEASIBILITY" if passed else "STOP_I0_INVARIANT_CARRIER_FEASIBILITY_FAILED",
        "all_six_dates_defined": bool(all_dates_defined and metrics_defined),
        "dates_total": 6,
        "required_passing_dates_per_clause": I0_MIN_PASSING_DATES,
        "subgates": {
            "invariant_w_norm_spearman": {
                "metric": "date invariant ||W|| Spearman",
                "threshold": I0_NORM_SPEARMAN_THRESHOLD,
                "comparison": ">=",
                "dates_passing": int(norm_count),
                "pass": norm_pass,
            },
            "correct_symmetric_cross_trial_gain": {
                "metric": "date correct symmetric held-trial transfer gain",
                "threshold": 0.0,
                "comparison": ">",
                "dates_passing": int(gain_count),
                "pass": gain_pass,
                "in_sample_fit_gain_substitute_allowed": False,
            },
            "correct_gain_above_fit_side_rotation_null_q95": {
                "metric": "date correct gain minus date 31-null q95",
                "threshold": 0.0,
                "comparison": ">",
                "dates_passing": int(null_count),
                "pass": null_pass,
                "evaluation_labels_remain_correct": True,
            },
        },
        "conjunction": "all_six_dates_defined AND each of the three subgates passes",
        "pass": passed,
        "gpu_authorized_by_this_gate": False,
    }


def run_audit(records: Mapping[str, signed_audit.H1M2Record]) -> dict[str, Any]:
    """Run only the CPU held-in M=2 six-date invariant-carrier gate."""

    groups = signed_audit.group_records_by_date(records)
    if set(records) == set(signed_audit.H1_HELDIN_SESSIONS) and set(groups) != set(signed_audit.H1_DATES):
        raise InvariantAuditError(f"unexpected official H1 held-in date groups: {groups}")
    partitions = signed_audit.date_lodo_partitions(records)
    date_rows: list[dict[str, Any]] = []
    for partition in partitions:
        outer_date = str(partition["outer_date"])
        target_sessions = tuple(partition["target_sessions"])
        try:
            plan = signed_audit.fit_source_plan(records, outer_date)
            source_records = tuple(records[name] for name in partition["source_sessions"])
            normalizer = _source_only_normalizer(source_records, plan)
            target_rows: list[dict[str, Any]] = []
            for session_name in target_sessions:
                try:
                    target_rows.append(evaluate_target_record(records[session_name], plan))
                except (signed_audit.AuditError, InvariantAuditError, np.linalg.LinAlgError) as exc:
                    target_rows.append(
                        {
                            "status": "undefined",
                            "session_name": session_name,
                            "date": outer_date,
                            "reason": str(exc),
                        }
                    )
            date_rows.append(
                aggregate_date(
                    target_rows,
                    date=outer_date,
                    plan=plan,
                    normalizer=normalizer,
                )
            )
        except (signed_audit.AuditError, InvariantAuditError, np.linalg.LinAlgError) as exc:
            date_rows.append(
                {
                    "status": "undefined",
                    "date": outer_date,
                    "recordings_total": int(len(target_sessions)),
                    "recordings_defined": 0,
                    "recording_sessions": list(target_sessions),
                    "reason": str(exc),
                }
            )
    gate = i0_gate(date_rows)
    input_hashes = {name: records[name].input_sha256 for name in sorted(records)}
    constants = {
        "carrier_field_names": list(CARRIER_FIELD_NAMES),
        "null_replicates": NULL_REPLICATES,
        "null_seed": NULL_SEED,
        "norm_spearman_threshold": I0_NORM_SPEARMAN_THRESHOLD,
        "required_passing_dates": I0_MIN_PASSING_DATES,
        "normalizer_scale_floor": NORMALIZER_SCALE_FLOOR,
    }
    return {
        "schema": "h1_invariant_carrier_m2_date_lodo_cpu_feasibility_v1",
        "status": (
            "PASS_CPU_I0_FEASIBILITY__NO_GPU_AUTHORIZATION"
            if gate["pass"]
            else "STOP_CPU_I0_FEASIBILITY_FAILED__NO_GPU_AUTHORIZATION"
        ),
        "protocol_version": PROTOCOL_VERSION,
        "task": "h1",
        "scope": {
            "opened_data": "exactly 13 public held-in-calib NWBs",
            "formal_heldout_opened": False,
            "minival_or_query_opened": False,
            "evalai_opened": False,
            "decoder_constructed": False,
            "optimizer_constructed": False,
            "gpu_constructed": False,
            "target_backpropagation": False,
        },
        "reuse_contract": {
            "signed_audit_data_protocol": str(Path(signed_audit.__file__).resolve()),
            "signed_audit_protocol_version": signed_audit.PROTOCOL_VERSION,
            "data_loading_and_m2_blocks_reimplemented_here": False,
            "source_pca_lag_ridge_reimplemented_here": False,
        },
        "carrier_contract": {
            "field_names": list(CARRIER_FIELD_NAMES),
            "width": 4,
            "combined_fit_labels": "correct within-trial velocity pairing",
            "cross_trial_gain": "fit trial1 correct -> evaluate trial2 correct and reverse; train-trial rate mean baseline",
            "in_sample_fit_gain_allowed_for_i0": False,
            "fit_side_null": f"{NULL_REPLICATES} deterministic nonzero within-trial circular velocity-block rotations",
            "null_evaluation_labels": "correct within-trial velocity pairing",
        },
        "source_only_selection_contract": {
            "outer_unit": "calendar date",
            "outer_dates": list(sorted(groups)),
            "source_dates_per_fold": 5,
            "normalizer_scope": "outer-source-dates-only",
            "normalizer_used_by_i0_gate": False,
            "target_date_used_for_source_choices": False,
        },
        "input_nwb_sha256": input_hashes,
        "source_hashes": {
            "invariant_audit_script_sha256": signed_audit.sha256_file(Path(__file__).resolve()),
            "signed_data_protocol_script_sha256": signed_audit.sha256_file(Path(signed_audit.__file__).resolve()),
            "protocol_constants_sha256": _sha256_bytes(_canonical_json(constants).encode("utf-8")),
        },
        "date_lodo": date_rows,
        "i0_gate": gate,
        "decision": {
            "i0_pass": bool(gate["pass"]),
            "gpu_authorized_by_this_receipt": False,
            "if_i0_fails": (
                "stop this invariant-carrier route at CPU gate; do not tune null count, threshold, "
                "aggregation, PCA, lag, ridge, descriptor width, or fusion after seeing this receipt"
            ),
            "if_i0_passes": (
                "this receipt alone still does not authorize GPU; a separately reviewed joint exact-SPINT "
                "residual protocol is required"
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "SPINT-main" / "data" / "000954",
        help="Exact SPINT-main/data/000954 root; held-out/formal roots are refused by the imported audit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repo_root
            / "sua_exploration"
            / "results"
            / "h1_invariant_carrier_m2_date_lodo_v1"
            / "cpu_feasibility_receipt.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, ""):
        raise RuntimeError(
            "This is a CPU-only audit. Invoke with CUDA_VISIBLE_DEVICES='' (or unset it); "
            f"observed {visible!r}."
        )
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable H1 invariant audit receipt: {output}")
    paths = signed_audit.index_h1_heldin_calib(args.data_dir)
    records = {name: signed_audit.load_h1_record(path) for name, path in paths.items()}
    receipt = run_audit(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    output.chmod(0o444)
    print(json.dumps({"status": receipt["status"], "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
