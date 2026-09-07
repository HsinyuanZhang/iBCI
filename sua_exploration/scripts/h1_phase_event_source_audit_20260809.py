#!/usr/bin/env python3
"""CPU-only H1 trial-phase carrier audit on public held-in calibration data.

The public H1 calibration NWBs do not contain named reach, go-cue, or phase
events.  They contain ``TrialNum`` boundaries and dense seven-dimensional
velocity.  This audit therefore tests one deliberately narrow, reproducible
proxy: split every TrialNum's legal 100-ms blocks into chronological early and
late halves, fit a continuous-velocity coefficient row in each half, concatenate
the two seven-dimensional rows, and learn a source-date-only 14 -> 4 projection.

For each of six leave-one-date-out folds, the projection and its centering are
fit on the other five dates.  A held public calibration recording is evaluated
only by fitting trials 1--2 and comparing the resulting per-channel carrier with
an independent trials-3--4 fit.  No minival, held-out, formal, or EvalAI path is
opened or enumerated.  No decoder, Trainer, checkpoint, CUDA context, or GPU is
used.

The ordinary unsplit M4 q16/ridge100 carrier is the matched reference.  The
primary statistic is the median per-channel signed cosine after centering each
split fit by its source-only carrier mean.  Deterministic row-shuffle and
within-trial/within-phase label-rotation controls are reported; the zero carrier
is the signed-cosine origin (0, or 0.5 on the legacy Q1 agreement scale).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
SPINT_ROOT = WORKSPACE / "SPINT-main"
if str(SPINT_ROOT) not in sys.path:
    sys.path.insert(0, str(SPINT_ROOT))

from src.data.h1_m4_cce_date_lodo import (  # noqa: E402
    reconstruct_plan_for_date,
    source_sessions_for_date,
    target_sessions_for_date,
)
from src.data.h1_m4_eb_pilot import (  # noqa: E402
    BLOCK_BINS,
    BLOCK_SECONDS,
    EPS,
    FOLD0_DATE,
    H1_HELDIN_SESSIONS,
    H1_M4_FOLD0_SOURCE,
    H1PilotRecord,
    PilotDataError,
    index_heldin_calib,
    load_record,
    reconstruct_frozen_plan,
    session_date,
)
from src.h1_m4_cce_contract import (  # noqa: E402
    CONFIRMATORY_DATES,
    array_sha256,
    sha256_file,
    write_immutable_json,
)


SCHEMA = "h1_phase_event_source_audit_v3"
STATUS_PASS = "PASS_CPU_TRIAL_PHASE_PROXY_WORTH_GPU_DESIGN_REVIEW__GPU_NOT_AUTHORIZED"
STATUS_STOP = "STOP_CPU_TRIAL_PHASE_PROXY_NOT_MORE_STABLE__NO_GPU_AUTHORIZATION"
ALL_DATES = (FOLD0_DATE, *CONFIRMATORY_DATES)
SUPPORT_TRIALS = 4
SPLIT_TRIALS = 2
PHASES = 2
LABEL_DIM = 7
CARRIER_DIM = 4
NULL_REPLICATES = 31
NULL_SEED = 20260809

# Frozen before executing the real-data audit.  These are routing rules, not
# claims of statistical significance.
MIN_PRIMARY_DELTA = 0.03
MIN_POSITIVE_DELTA_DATES = 4
MIN_CONTROL_PASS_DATES = 5
MIN_VALID_CHANNEL_FRACTION = 0.95


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _summary(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    _require(vector.size > 0 and np.isfinite(vector).all(), "summary requires finite nonempty values")
    return {
        "count": int(vector.size),
        "mean": float(vector.mean()),
        "std_population": float(vector.std(ddof=0)),
        "minimum": float(vector.min()),
        "q25": float(np.quantile(vector, 0.25)),
        "median": float(np.quantile(vector, 0.50)),
        "q75": float(np.quantile(vector, 0.75)),
        "maximum": float(vector.max()),
    }


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    _require(x.shape == y.shape and x.size >= 2, "Pearson inputs must have matching nontrivial shape")
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > EPS else 0.0


def _signed_cosines(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    _require(left.shape == right.shape and left.ndim == 2, "carrier cosine inputs must match [N,C]")
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    valid = (left_norm > EPS) & (right_norm > EPS)
    cosine = np.zeros(left.shape[0], dtype=np.float64)
    cosine[valid] = np.einsum("ij,ij->i", left[valid], right[valid]) / (left_norm[valid] * right_norm[valid])
    return np.clip(cosine, -1.0, 1.0), valid


def _forward_gain_vs_zero(first: np.ndarray, second: np.ndarray) -> float:
    """Return 1-SSE(first predicts second)/SSE(zero predicts second)."""

    prediction = np.asarray(first, dtype=np.float64)
    target = np.asarray(second, dtype=np.float64)
    _require(prediction.shape == target.shape and prediction.ndim == 2, "forward-gain arrays must match [N,C]")
    zero_sse = float(np.square(target).sum())
    _require(np.isfinite(zero_sse) and zero_sse > EPS, "forward zero-control energy is undefined")
    return float(1.0 - np.square(target - prediction).sum() / zero_sse)


def _phase_indices(block_count: int, phase_count: int) -> np.ndarray:
    _require(block_count >= phase_count and phase_count >= 1, "phase split requires at least one block per phase")
    # Equal-count chronological bins are invariant to trial duration and need
    # no target-date threshold or event detector.
    phase = (np.arange(block_count, dtype=np.int64) * int(phase_count)) // int(block_count)
    return np.minimum(phase, int(phase_count) - 1)


def _rotation_shift(length: int, *tokens: object) -> int:
    _require(length >= 2, "label rotation requires at least two rows")
    digest = hashlib.sha256("|".join(map(str, tokens)).encode("utf-8")).digest()
    return 1 + int.from_bytes(digest[:8], "big") % (int(length) - 1)


def _project(rates: np.ndarray, plan: Any) -> np.ndarray:
    return ((np.asarray(rates, np.float64) - plan.mean[None, :]) / plan.scale[None, :]) @ plan.pcs[: plan.q].T


def _ridge_raw_rows(rates: np.ndarray, labels: np.ndarray, plan: Any) -> tuple[np.ndarray, float, int]:
    x = np.asarray(rates, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    _require(x.ndim == 2 and y.ndim == 2 and x.shape[0] == y.shape[0], "malformed phase fit arrays")
    _require(y.shape[1] == LABEL_DIM and x.shape[0] > int(plan.q) + 1, "phase fit is underdetermined")
    z = _project(x, plan)
    design = np.column_stack((np.ones(z.shape[0], dtype=np.float64), z))
    regularizer = np.eye(design.shape[1], dtype=np.float64) * float(plan.ridge_lambda)
    regularizer[0, 0] = 0.0
    system = design.T @ design + regularizer
    condition = float(np.linalg.cond(system))
    beta = np.linalg.solve(system, design.T @ y)
    raw_rows = (np.asarray(plan.pcs[: plan.q], np.float64).T @ beta[1:]) / np.asarray(plan.scale, np.float64)[:, None]
    _require(raw_rows.shape == (x.shape[1], LABEL_DIM) and np.isfinite(raw_rows).all(), "raw coefficient rows invalid")
    return raw_rows, condition, int(x.shape[0])


def _fit_content_rows(
    record: H1PilotRecord,
    plan: Any,
    trial_values: Sequence[float],
    *,
    phase_count: int,
    label_rotation_replicate: int | None = None,
) -> dict[str, Any]:
    values = tuple(float(value) for value in trial_values)
    _require(len(values) in (SPLIT_TRIALS, SUPPORT_TRIALS), "fit requires exactly two or four trials")
    rate_parts: list[list[np.ndarray]] = [[] for _ in range(phase_count)]
    label_parts: list[list[np.ndarray]] = [[] for _ in range(phase_count)]
    per_trial_phase_blocks: dict[str, list[int]] = {}
    for value in values:
        trial = record.blocks_for(value)
        rates = np.asarray(trial.rates, dtype=np.float64)
        labels = np.asarray(trial.velocity, dtype=np.float64)
        _require(rates.shape[0] == labels.shape[0] and rates.shape[0] >= 2 * phase_count, "trial lacks phase blocks")
        phases = _phase_indices(rates.shape[0], phase_count)
        counts: list[int] = []
        for phase in range(phase_count):
            selected = np.flatnonzero(phases == phase)
            _require(selected.size >= 2, "trial phase has fewer than two label rows")
            phase_labels = labels[selected]
            if label_rotation_replicate is not None:
                shift = _rotation_shift(
                    len(selected), NULL_SEED, "label", label_rotation_replicate,
                    record.session_name, value, phase_count, phase,
                )
                phase_labels = np.roll(phase_labels, shift, axis=0)
            rate_parts[phase].append(rates[selected])
            label_parts[phase].append(phase_labels)
            counts.append(int(selected.size))
        per_trial_phase_blocks[str(value)] = counts
    rows: list[np.ndarray] = []
    conditions: list[float] = []
    block_counts: list[int] = []
    for phase in range(phase_count):
        raw_rows, condition, count = _ridge_raw_rows(
            np.concatenate(rate_parts[phase], axis=0),
            np.concatenate(label_parts[phase], axis=0),
            plan,
        )
        rows.append(raw_rows)
        conditions.append(condition)
        block_counts.append(count)
    content = np.concatenate(rows, axis=1)
    _require(content.shape == (record.num_neurons, LABEL_DIM * phase_count), "content width drift")
    return {
        "content_rows": content,
        "condition_by_phase": conditions,
        "block_count_by_phase": block_counts,
        "per_trial_phase_blocks": per_trial_phase_blocks,
    }


def _canonicalize_columns(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float64).copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


def _fit_source_projection(
    records: Mapping[str, H1PilotRecord], plan: Any, *, phase_count: int
) -> dict[str, Any]:
    pooled: list[np.ndarray] = []
    source_fit: dict[str, Any] = {}
    for name in plan.source_sessions:
        record = records[name]
        fit = _fit_content_rows(record, plan, record.trial_values[:SUPPORT_TRIALS], phase_count=phase_count)
        rows = np.asarray(fit["content_rows"], dtype=np.float64)
        pooled.append(rows)
        source_fit[name] = {
            "support_trial_values": [float(v) for v in record.trial_values[:SUPPORT_TRIALS]],
            "block_count_by_phase": fit["block_count_by_phase"],
            "condition_by_phase": fit["condition_by_phase"],
        }
    pooled_rows = np.concatenate(pooled, axis=0)
    _, singular_values, right = np.linalg.svd(pooled_rows, full_matrices=False)
    projection = _canonicalize_columns(right[:CARRIER_DIM].T)
    carriers = pooled_rows @ projection
    center = carriers.mean(axis=0)
    _require(projection.shape == (LABEL_DIM * phase_count, CARRIER_DIM), "source projection shape drift")
    return {
        "projection": projection,
        "center": center,
        "source_rows": pooled_rows,
        "source_carriers": carriers,
        "singular_values": singular_values,
        "source_fit": source_fit,
    }


def _row_permutation(length: int, session: str, family: str, replicate: int) -> np.ndarray:
    digest = hashlib.sha256(f"{NULL_SEED}|row|{family}|{session}|{replicate}".encode("utf-8")).digest()
    permutation = np.random.default_rng(int.from_bytes(digest[:8], "big")).permutation(length)
    if np.array_equal(permutation, np.arange(length)):
        permutation = np.roll(permutation, 1)
    return permutation


def _evaluate_family(
    record: H1PilotRecord,
    plan: Any,
    *,
    family: str,
    phase_count: int,
    projection: np.ndarray,
    center: np.ndarray,
) -> dict[str, Any]:
    values = tuple(float(v) for v in record.trial_values[:SUPPORT_TRIALS])
    fit = _fit_content_rows(record, plan, values[:2], phase_count=phase_count)
    score = _fit_content_rows(record, plan, values[2:], phase_count=phase_count)
    first = np.asarray(fit["content_rows"], np.float64) @ projection - center[None, :]
    second = np.asarray(score["content_rows"], np.float64) @ projection - center[None, :]
    cosine, valid = _signed_cosines(first, second)
    _require(float(valid.mean()) >= MIN_VALID_CHANNEL_FRACTION, f"{record.session_name}/{family}: valid coverage below gate")
    nominal_median = float(np.median(cosine[valid]))
    nominal_mean = float(np.mean(cosine[valid]))
    nominal_forward_gain = _forward_gain_vs_zero(first, second)

    row_medians: list[float] = []
    row_means: list[float] = []
    label_medians: list[float] = []
    label_means: list[float] = []
    row_forward_gains: list[float] = []
    label_forward_gains: list[float] = []
    row_hashes: list[str] = []
    for replicate in range(NULL_REPLICATES):
        permutation = _row_permutation(record.num_neurons, record.session_name, family, replicate)
        row_hashes.append(array_sha256(permutation.astype(np.int64)))
        row_cosine, row_valid = _signed_cosines(first, second[permutation])
        row_medians.append(float(np.median(row_cosine[row_valid])))
        row_means.append(float(np.mean(row_cosine[row_valid])))
        row_forward_gains.append(_forward_gain_vs_zero(first, second[permutation]))

        rotated = _fit_content_rows(
            record,
            plan,
            values[2:],
            phase_count=phase_count,
            label_rotation_replicate=replicate,
        )
        rotated_carrier = np.asarray(rotated["content_rows"], np.float64) @ projection - center[None, :]
        label_cosine, label_valid = _signed_cosines(first, rotated_carrier)
        label_medians.append(float(np.median(label_cosine[label_valid])))
        label_means.append(float(np.mean(label_cosine[label_valid])))
        label_forward_gains.append(_forward_gain_vs_zero(first, rotated_carrier))

    row_q95 = float(np.quantile(row_medians, 0.95))
    label_q95 = float(np.quantile(label_medians, 0.95))
    return {
        "family": family,
        "phase_count": phase_count,
        "fit_trial_values": list(values[:2]),
        "score_trial_values": list(values[2:]),
        "fit_block_count_by_phase": fit["block_count_by_phase"],
        "score_block_count_by_phase": score["block_count_by_phase"],
        "fit_condition_by_phase": fit["condition_by_phase"],
        "score_condition_by_phase": score["condition_by_phase"],
        "per_trial_phase_blocks": {
            "fit": fit["per_trial_phase_blocks"],
            "score": score["per_trial_phase_blocks"],
        },
        "valid_channels": int(valid.sum()),
        "valid_channel_fraction": float(valid.mean()),
        "nominal": {
            "median_per_channel_signed_cosine": nominal_median,
            "mean_per_channel_signed_cosine": nominal_mean,
            "legacy_q1_agreement_from_median": float((nominal_median + 1.0) / 2.0),
            "flattened_pearson": _pearson(first, second),
            "forward_transfer_gain_vs_zero": nominal_forward_gain,
            "first_centered_carrier_sha256": array_sha256(first),
            "second_centered_carrier_sha256": array_sha256(second),
        },
        "zero_control": {
            "forward_transfer_gain_vs_zero": 0.0,
            "definition": "actual zero prediction of the source-centered score-half carrier; nominal gain is 1-SSE(first_half,score_half)/SSE(zero,score_half)",
            "cosine_not_reported": "mathematical cosine to a zero vector is undefined",
        },
        "row_shuffle_control": {
            "replicates": NULL_REPLICATES,
            "median_per_channel_signed_cosine": _summary(row_medians),
            "mean_per_channel_signed_cosine": _summary(row_means),
            "forward_transfer_gain_vs_zero": _summary(row_forward_gains),
            "q95_of_replicate_medians": row_q95,
            "nominal_minus_q95": float(nominal_median - row_q95),
            "permutation_hashes_sha256": _canonical_sha256(row_hashes),
        },
        "label_rotation_control": {
            "replicates": NULL_REPLICATES,
            "operation": "rotate score-half velocity rows by a deterministic nonzero amount within each TrialNum and within each phase; neural rows and label multiset remain fixed",
            "median_per_channel_signed_cosine": _summary(label_medians),
            "mean_per_channel_signed_cosine": _summary(label_means),
            "forward_transfer_gain_vs_zero": _summary(label_forward_gains),
            "q95_of_replicate_medians": label_q95,
            "nominal_minus_q95": float(nominal_median - label_q95),
        },
    }


def _metadata_scope(path: Path) -> dict[str, Any]:
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        acquisition = sorted(str(key) for key in nwb.acquisition.keys())
        processing = sorted(str(key) for key in nwb.processing.keys())
        intervals = sorted(str(key) for key in nwb.intervals.keys()) if nwb.intervals is not None else []
        trials_columns = [] if nwb.trials is None else [str(name) for name in nwb.trials.colnames]
        epochs_columns = [] if nwb.epochs is None else [str(name) for name in nwb.epochs.colnames]
    return {
        "acquisition_keys": acquisition,
        "processing_keys": processing,
        "interval_keys": intervals,
        "trials_columns": trials_columns,
        "epochs_columns": epochs_columns,
        "named_reach_go_cue_or_phase_field_found": any(
            token in key.lower()
            for key in (*acquisition, *processing, *intervals, *trials_columns, *epochs_columns)
            for token in ("reach", "go_cue", "gocue", "phase", "onset")
        ),
    }


def _plan_for_date(
    records: Mapping[str, H1PilotRecord], outer_date: str, raw_receipt: Path, eb_receipt: Path
) -> Any:
    source_names = tuple(name for name in H1_HELDIN_SESSIONS if session_date(name) != outer_date)
    source_records = {name: records[name] for name in source_names}
    if outer_date == FOLD0_DATE:
        _require(source_names == H1_M4_FOLD0_SOURCE, "fold0 source partition drift")
        return reconstruct_frozen_plan(source_records, raw_receipt, eb_receipt)
    _require(source_names == source_sessions_for_date(outer_date), "confirmatory source partition drift")
    return reconstruct_plan_for_date(source_records, outer_date, raw_receipt, eb_receipt)


def run(*, data_dir: Path, raw_receipt: Path, eb_receipt: Path, output: Path) -> dict[str, Any]:
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") in (None, ""), "CPU audit requires CUDA_VISIBLE_DEVICES unset")
    _require(not output.exists(), f"refusing to overwrite {output}")
    _require(stat.S_IMODE(raw_receipt.stat().st_mode) == 0o444, "raw receipt must be immutable 0444")
    _require(stat.S_IMODE(eb_receipt.stat().st_mode) == 0o444, "EB receipt must be immutable 0444")

    paths = index_heldin_calib(data_dir)
    _require(tuple(paths) == H1_HELDIN_SESSIONS, "public held-in-calib scope drift")
    records = {name: load_record(paths[name]) for name in H1_HELDIN_SESSIONS}
    metadata = {name: _metadata_scope(paths[name]) for name in H1_HELDIN_SESSIONS}
    _require(not any(row["named_reach_go_cue_or_phase_field_found"] for row in metadata.values()), "native event field unexpectedly appeared")

    per_date: dict[str, Any] = {}
    all_record_deltas: list[float] = []
    for outer_date in ALL_DATES:
        plan = _plan_for_date(records, outer_date, raw_receipt, eb_receipt)
        _require(tuple(plan.source_sessions) == tuple(name for name in H1_HELDIN_SESSIONS if session_date(name) != outer_date), "plan source leakage")
        whole = _fit_source_projection(records, plan, phase_count=1)
        phase = _fit_source_projection(records, plan, phase_count=PHASES)

        # The independently reconstructed whole projection must span the same
        # four-dimensional source subspace as the immutable current plan.
        overlap = np.linalg.svd(np.asarray(plan.U).T @ np.asarray(whole["projection"]), compute_uv=False)
        _require(float(overlap.min()) > 1.0 - 1.0e-8, f"{outer_date}: whole projection parity failed")

        targets = tuple(name for name in H1_HELDIN_SESSIONS if session_date(name) == outer_date)
        if outer_date != FOLD0_DATE:
            _require(targets == target_sessions_for_date(outer_date), "held public calibration date partition drift")
        per_record: dict[str, Any] = {}
        date_delta: list[float] = []
        date_phase_nominal: list[float] = []
        date_ordinary_forward_gain: list[float] = []
        date_phase_forward_gain: list[float] = []
        date_phase_row_gap: list[float] = []
        date_phase_label_gap: list[float] = []
        for name in targets:
            ordinary_result = _evaluate_family(
                records[name], plan, family="whole_m4_q16", phase_count=1,
                projection=np.asarray(whole["projection"]), center=np.asarray(whole["center"]),
            )
            phase_result = _evaluate_family(
                records[name], plan, family="trial_phase2_continuous_velocity", phase_count=PHASES,
                projection=np.asarray(phase["projection"]), center=np.asarray(phase["center"]),
            )
            ordinary_value = float(ordinary_result["nominal"]["median_per_channel_signed_cosine"])
            phase_value = float(phase_result["nominal"]["median_per_channel_signed_cosine"])
            delta = phase_value - ordinary_value
            date_delta.append(delta)
            all_record_deltas.append(delta)
            date_phase_nominal.append(phase_value)
            date_ordinary_forward_gain.append(float(ordinary_result["nominal"]["forward_transfer_gain_vs_zero"]))
            date_phase_forward_gain.append(float(phase_result["nominal"]["forward_transfer_gain_vs_zero"]))
            date_phase_row_gap.append(float(phase_result["row_shuffle_control"]["nominal_minus_q95"]))
            date_phase_label_gap.append(float(phase_result["label_rotation_control"]["nominal_minus_q95"]))
            per_record[name] = {
                "input_sha256": records[name].input_sha256,
                "trial_values_used": [float(v) for v in records[name].trial_values[:SUPPORT_TRIALS]],
                "ordinary": ordinary_result,
                "trial_phase2": phase_result,
                "phase_minus_ordinary_median_signed_cosine": float(delta),
            }
        date_row = {
            "outer_date": outer_date,
            "source_sessions": list(plan.source_sessions),
            "held_public_calibration_sessions": list(targets),
            "source_transform_sha256": str(plan.transform_sha256),
            "whole_projection_sha256": array_sha256(np.asarray(whole["projection"])),
            "phase_projection_sha256": array_sha256(np.asarray(phase["projection"])),
            "phase_center_sha256": array_sha256(np.asarray(phase["center"])),
            "whole_projection_subspace_overlap_singular_values": overlap.tolist(),
            "phase_source_singular_values_first8": np.asarray(phase["singular_values"][:8], np.float64).tolist(),
            "per_recording": per_record,
            "equal_recording": {
                "ordinary_median_signed_cosine_mean": float(np.mean([
                    row["ordinary"]["nominal"]["median_per_channel_signed_cosine"] for row in per_record.values()
                ])),
                "phase_median_signed_cosine_mean": float(np.mean(date_phase_nominal)),
                "phase_minus_ordinary": float(np.mean(date_delta)),
                "ordinary_forward_transfer_gain_vs_zero": float(np.mean(date_ordinary_forward_gain)),
                "phase_forward_transfer_gain_vs_zero": float(np.mean(date_phase_forward_gain)),
                "phase_minus_ordinary_forward_transfer_gain": float(np.mean(date_phase_forward_gain) - np.mean(date_ordinary_forward_gain)),
                "phase_nominal_minus_row_q95": float(np.mean(date_phase_row_gap)),
                "phase_nominal_minus_label_q95": float(np.mean(date_phase_label_gap)),
            },
        }
        per_date[outer_date] = date_row

    date_deltas = np.asarray([row["equal_recording"]["phase_minus_ordinary"] for row in per_date.values()], np.float64)
    date_phase = np.asarray([row["equal_recording"]["phase_median_signed_cosine_mean"] for row in per_date.values()], np.float64)
    date_ordinary_forward_gain = np.asarray([row["equal_recording"]["ordinary_forward_transfer_gain_vs_zero"] for row in per_date.values()], np.float64)
    date_phase_forward_gain = np.asarray([row["equal_recording"]["phase_forward_transfer_gain_vs_zero"] for row in per_date.values()], np.float64)
    date_row_gap = np.asarray([row["equal_recording"]["phase_nominal_minus_row_q95"] for row in per_date.values()], np.float64)
    date_label_gap = np.asarray([row["equal_recording"]["phase_nominal_minus_label_q95"] for row in per_date.values()], np.float64)
    gate = {
        "minimum_primary_equal_date_delta": MIN_PRIMARY_DELTA,
        "minimum_positive_delta_dates": MIN_POSITIVE_DELTA_DATES,
        "minimum_control_pass_dates": MIN_CONTROL_PASS_DATES,
        "minimum_valid_channel_fraction_per_recording_family": MIN_VALID_CHANNEL_FRACTION,
        "observed_equal_date_mean_phase_minus_whole": float(date_deltas.mean()),
        "observed_positive_delta_dates": int(np.sum(date_deltas > 0.0)),
        "observed_phase_forward_gain_above_zero_dates": int(np.sum(date_phase_forward_gain > 0.0)),
        "observed_phase_above_row_q95_dates": int(np.sum(date_row_gap > 0.0)),
        "observed_phase_above_label_q95_dates": int(np.sum(date_label_gap > 0.0)),
    }
    gate["passes"] = bool(
        gate["observed_equal_date_mean_phase_minus_whole"] >= MIN_PRIMARY_DELTA
        and gate["observed_positive_delta_dates"] >= MIN_POSITIVE_DELTA_DATES
        and gate["observed_phase_forward_gain_above_zero_dates"] == len(ALL_DATES)
        and gate["observed_phase_above_row_q95_dates"] >= MIN_CONTROL_PASS_DATES
        and gate["observed_phase_above_label_q95_dates"] >= MIN_CONTROL_PASS_DATES
    )

    receipt = {
        "schema": SCHEMA,
        "status": STATUS_PASS if gate["passes"] else STATUS_STOP,
        "scope": {
            "opened": "exactly 13 public sub-HumanPitt-held-in-calib NWBs",
            "opened_session_count": len(records),
            "opened_sessions": list(records),
            "minival_opened_or_enumerated": False,
            "held_out_calib_or_query_opened_or_enumerated": False,
            "formal_or_evalai_opened_or_enumerated": False,
            "decoder_or_checkpoint_opened": False,
            "trainer_constructed": False,
            "cuda_constructed_or_gpu_launched": False,
            "target_backpropagation": False,
        },
        "native_event_metadata_audit": {
            "per_recording": metadata,
            "conclusion": "No named reach/go-cue/onset/phase field exists. TrialNum is the only semantic event boundary used; the candidate is a trial-relative temporal phase proxy, not a native reach-phase annotation.",
        },
        "statistical_definition": {
            "unit_of_outer_inference": "date; six public held-in-calibration dates",
            "source_transform": "leave one date out; q16/ridge100 PCA and ordinary 7->4 carrier projection reconstructed from immutable receipts on the other five dates",
            "ordinary_reference": "one ridge fit over all legal 100-ms blocks; 7D continuous velocity coefficient rows; source-only 7->4 projection",
            "phase_candidate": "within each TrialNum, chronological legal 100-ms blocks are split into equal-count early/late halves; independent ridge fit per half; concatenate 7+7 coefficient rows; source-only 14->4 SVD projection",
            "held_date_data": "only chronological calibration trials 1-4; trials 1-2 fit and trials 3-4 forward score",
            "primary": "per recording median over valid channels of signed cosine between source-centered trials1-2 and trials3-4 carrier rows; equal-recording mean within date, then equal-date mean",
            "attachment_control": f"{NULL_REPLICATES} deterministic row permutations of the score-half carrier",
            "label_control": f"{NULL_REPLICATES} deterministic nonzero within-TrialNum/within-phase rotations of score-half continuous velocity labels",
            "zero_control": "actual source-centered zero predictor; report gain 1-SSE(first_half,score_half)/SSE(zero,score_half); zero itself is exactly 0 gain; no zero-vector cosine is reported",
            "block_contract": {"bins_per_block": BLOCK_BINS, "seconds_per_block": BLOCK_SECONDS},
        },
        "frozen_inputs": {
            "raw_receipt": {"path": str(raw_receipt.resolve()), "sha256": sha256_file(raw_receipt), "mode": "0444"},
            "eb_receipt": {"path": str(eb_receipt.resolve()), "sha256": sha256_file(eb_receipt), "mode": "0444"},
            "source_code": {
                "audit_script_sha256_before_receipt_write": sha256_file(Path(__file__).resolve()),
                "h1_m4_eb_pilot_sha256": sha256_file(SPINT_ROOT / "src/data/h1_m4_eb_pilot.py"),
                "h1_m4_cce_date_lodo_sha256": sha256_file(SPINT_ROOT / "src/data/h1_m4_cce_date_lodo.py"),
            },
            "input_nwb_sha256": {name: records[name].input_sha256 for name in records},
        },
        "per_date": per_date,
        "aggregate": {
            "equal_date_phase_minus_whole": _summary(date_deltas),
            "equal_recording_phase_minus_whole": _summary(all_record_deltas),
            "equal_date_phase_nominal_median_signed_cosine": _summary(date_phase),
            "equal_date_ordinary_forward_transfer_gain_vs_zero": _summary(date_ordinary_forward_gain),
            "equal_date_phase_forward_transfer_gain_vs_zero": _summary(date_phase_forward_gain),
            "equal_date_phase_minus_ordinary_forward_transfer_gain": _summary(date_phase_forward_gain - date_ordinary_forward_gain),
            "equal_date_phase_nominal_minus_row_q95": _summary(date_row_gap),
            "equal_date_phase_nominal_minus_label_q95": _summary(date_label_gap),
        },
        "gpu_routing_gate": gate,
        "interpretation_contract": {
            "if_pass": "CPU evidence supports designing a matched GPU trial-phase carrier arm; this receipt itself does not authorize or report a GPU result.",
            "if_stop": "Do not start a GPU trial-phase carrier arm. The proxy is not more stable than the current whole-M4 carrier under source-only forward transfer.",
            "forbidden_claims": [
                "native reach phase was observed",
                "decoder R2 improved",
                "formal or hidden H1 generalized",
                "the trial-phase proxy is a biological event representation",
            ],
        },
    }
    path, digest = write_immutable_json(output, receipt)
    return {"receipt_path": str(path), "receipt_sha256": digest, "status": receipt["status"], "gpu_gate": gate}


V4_SCHEMA = "h1_native_event_source_audit_v4r2"
V4_STATUS_STOP_MISSING = "STOP_CPU_NATIVE_EVENT_CARRIER_UNDEFINED_MISSING_COMPLETE_PHASE_BLOCKS__NO_GPU_AUTHORIZATION"
V4_STATUS_PASS = "PASS_CPU_NATIVE_EVENT_CARRIER_WORTH_GPU_DESIGN_REVIEW__GPU_NOT_AUTHORIZED"
BIN_SECONDS = BLOCK_SECONDS / BLOCK_BINS
EVENT_CATEGORIES = (
    "Presentation",
    "Reach",
    "Orient",
    "SnapTo",
    "Shape",
    "Grasp",
    "Carry",
    "Orient2",
    "Release",
)
MOVEMENT_EVENT_CATEGORIES = EVENT_CATEGORIES[1:]
EXCLUDED_EVENT_TAGS = ("Intertrial",)
EVENT_FEATURE_DIM = len(EVENT_CATEGORIES) * LABEL_DIM
EVENT_SVD_DIM = 3
EVENT_RIDGE_LAMBDA = 100.0
TIME_TOLERANCE_SECONDS = 1.0e-9


def _dereference_single_tag(value: Any) -> str:
    tags = np.asarray(value, dtype=object).reshape(-1)
    _require(tags.size == 1, f"each H1 epoch must have one tag, observed {tags.tolist()}")
    tag = str(tags[0])
    _require(tag != "" and not tag.startswith("["), "epoch tag was not correctly dereferenced")
    return tag


def _event_category(tag: str) -> str | None:
    if tag.startswith("Presentation"):
        return "Presentation"
    if tag in MOVEMENT_EVENT_CATEGORIES:
        return tag
    if tag in EXCLUDED_EVENT_TAGS:
        return None
    raise ValueError(f"unexpected native H1 epoch tag {tag!r}")


def _load_event_epochs(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        _require(nwb.epochs is not None, f"{path}: epochs table missing")
        table = nwb.epochs.to_dataframe()
        trial_series = nwb.acquisition["TrialNum"]
        interval = float(trial_series.rate)
        starting_time = float(trial_series.starting_time)
    _require(abs(interval - BIN_SECONDS) <= 1.0e-12 and abs(starting_time) <= 1.0e-12, "20-ms timebase drift")
    epochs: list[dict[str, Any]] = []
    for row_id, row in table.iterrows():
        start, stop = float(row["start_time"]), float(row["stop_time"])
        tag = _dereference_single_tag(row["tags"])
        _require(np.isfinite(start) and np.isfinite(stop) and stop > start, "invalid epoch interval")
        epochs.append({
            "row_id": int(row_id),
            "start_time": start,
            "stop_time": stop,
            "duration_seconds": stop - start,
            "tag": tag,
            "category": _event_category(tag),
        })
    _require(all(epochs[i]["start_time"] >= epochs[i - 1]["stop_time"] - TIME_TOLERANCE_SECONDS for i in range(1, len(epochs))), "epoch intervals overlap")
    return epochs, {
        "trialnum_timebase_field_named_rate": interval,
        "trialnum_starting_time": starting_time,
        "interpretation": "the files store 0.02 in the TimeSeries rate field; this audit binds it only as the observed per-bin time increment already used by the 20-ms arrays",
    }


def _containing_epoch(epochs: Sequence[Mapping[str, Any]], start: float, stop: float) -> Mapping[str, Any] | None:
    starts = np.asarray([float(epoch["start_time"]) for epoch in epochs], dtype=np.float64)
    index = int(np.searchsorted(starts, start + TIME_TOLERANCE_SECONDS, side="right") - 1)
    if index < 0:
        return None
    epoch = epochs[index]
    if start >= float(epoch["start_time"]) - TIME_TOLERANCE_SECONDS and stop <= float(epoch["stop_time"]) + TIME_TOLERANCE_SECONDS:
        return epoch
    return None


def _design_diagnostic(rows: list[tuple[np.ndarray, np.ndarray, str]]) -> dict[str, Any]:
    if not rows:
        return {"status": "undefined_no_complete_event_blocks"}
    velocity = np.stack([row[1] for row in rows]).astype(np.float64)
    categories = [row[2] for row in rows]
    raw = np.zeros((len(rows), EVENT_FEATURE_DIM), dtype=np.float64)
    for index, category in enumerate(categories):
        event_index = EVENT_CATEGORIES.index(category)
        raw[index, event_index * LABEL_DIM : (event_index + 1) * LABEL_DIM] = velocity[index]
    design = np.column_stack((np.ones(len(raw), dtype=np.float64), raw))
    regularizer = np.eye(design.shape[1], dtype=np.float64) * EVENT_RIDGE_LAMBDA
    regularizer[0, 0] = 0.0
    system = design.T @ design + regularizer
    active = np.any(np.abs(raw) > EPS, axis=0)
    return {
        "status": "defined_diagnostic_only_before_source_rms_normalization",
        "rows": int(len(raw)),
        "columns_including_intercept": int(design.shape[1]),
        "raw_design_rank": int(np.linalg.matrix_rank(design)),
        "active_event_velocity_columns": int(active.sum()),
        "ridge100_system_condition_unscaled": float(np.linalg.cond(system)),
        "raw_design_sha256": array_sha256(raw),
    }


def _audit_native_events(record: H1PilotRecord, epochs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tag_counts: dict[str, int] = {}
    tag_durations: dict[str, float] = {}
    category_counts: dict[str, int] = {category: 0 for category in EVENT_CATEGORIES}
    category_durations: dict[str, float] = {category: 0.0 for category in EVENT_CATEGORIES}
    excluded_counts: dict[str, int] = {tag: 0 for tag in EXCLUDED_EVENT_TAGS}
    epoch_trial_alignment: list[dict[str, Any]] = []
    for epoch in epochs:
        tag = str(epoch["tag"])
        duration = float(epoch["duration_seconds"])
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        tag_durations[tag] = tag_durations.get(tag, 0.0) + duration
        category = epoch["category"]
        if category is None:
            excluded_counts[tag] += 1
        else:
            category_counts[str(category)] += 1
            category_durations[str(category)] += duration
        first = max(0, int(np.ceil((float(epoch["start_time"]) - TIME_TOLERANCE_SECONDS) / BIN_SECONDS)))
        last = min(len(record.trial_num), int(np.floor((float(epoch["stop_time"]) + TIME_TOLERANCE_SECONDS) / BIN_SECONDS)))
        complete = [
            index for index in range(first, last)
            if index * BIN_SECONDS >= float(epoch["start_time"]) - TIME_TOLERANCE_SECONDS
            and (index + 1) * BIN_SECONDS <= float(epoch["stop_time"]) + TIME_TOLERANCE_SECONDS
        ]
        trial_values = sorted({float(record.trial_num[index]) for index in complete if np.isfinite(record.trial_num[index])})
        epoch_trial_alignment.append({
            "row_id": int(epoch["row_id"]),
            "tag": tag,
            "category": category,
            "complete_20ms_bins": len(complete),
            "finite_trial_values": trial_values,
            "crosses_trialnum_values": len(trial_values) > 1,
        })

    bin_counts = {
        "total": int(len(record.trial_num)),
        "eval_valid": int(np.asarray(record.eval_mask, bool).sum()),
        "complete_in_one_epoch": 0,
        "eval_valid_and_complete_in_one_epoch": 0,
        "eval_valid_complete_candidate_category": 0,
        "eval_valid_complete_excluded_intertrial": 0,
    }
    bin_category_counts = {category: 0 for category in EVENT_CATEGORIES}
    for index in range(len(record.trial_num)):
        epoch = _containing_epoch(epochs, index * BIN_SECONDS, (index + 1) * BIN_SECONDS)
        if epoch is None:
            continue
        bin_counts["complete_in_one_epoch"] += 1
        if not bool(record.eval_mask[index]):
            continue
        bin_counts["eval_valid_and_complete_in_one_epoch"] += 1
        category = epoch["category"]
        if category is None:
            bin_counts["eval_valid_complete_excluded_intertrial"] += 1
        else:
            bin_counts["eval_valid_complete_candidate_category"] += 1
            bin_category_counts[str(category)] += 1

    rows_by_trial: dict[float, list[tuple[np.ndarray, np.ndarray, str]]] = {float(value): [] for value in record.trial_values}
    block_counts_by_trial: dict[str, dict[str, int]] = {}
    total_legal_blocks = 0
    complete_event_blocks = 0
    excluded_or_boundary_blocks = 0
    for trial in record.trials:
        counts = {category: 0 for category in EVENT_CATEGORIES}
        for row_index, indices in enumerate(np.asarray(trial.block_indices, dtype=np.int64)):
            total_legal_blocks += 1
            start = float(indices[0]) * BIN_SECONDS
            stop = float(indices[-1] + 1) * BIN_SECONDS
            epoch = _containing_epoch(epochs, start, stop)
            if epoch is None or epoch["category"] is None:
                excluded_or_boundary_blocks += 1
                continue
            category = str(epoch["category"])
            complete_event_blocks += 1
            counts[category] += 1
            rows_by_trial[float(trial.trial_number)].append((
                np.asarray(trial.rates[row_index], dtype=np.float64),
                np.asarray(trial.velocity[row_index], dtype=np.float64),
                category,
            ))
        block_counts_by_trial[str(float(trial.trial_number))] = counts

    support = tuple(float(value) for value in record.trial_values[:SUPPORT_TRIALS])
    fit_rows = [row for value in support[:2] for row in rows_by_trial[value]]
    score_rows = [row for value in support[2:] for row in rows_by_trial[value]]
    full_rows = fit_rows + score_rows
    def counts_for(rows: Sequence[tuple[np.ndarray, np.ndarray, str]]) -> dict[str, int]:
        return {category: int(sum(row[2] == category for row in rows)) for category in EVENT_CATEGORIES}
    fit_counts, score_counts, full_counts = counts_for(fit_rows), counts_for(score_rows), counts_for(full_rows)
    missing_fit = [category for category, count in fit_counts.items() if count == 0]
    missing_score = [category for category, count in score_counts.items() if count == 0]
    missing_full = [category for category, count in full_counts.items() if count == 0]
    return {
        "epoch_table": {
            "rows": len(epochs),
            "raw_tag_counts": tag_counts,
            "raw_tag_duration_seconds": tag_durations,
            "collapsed_category_counts": category_counts,
            "collapsed_category_duration_seconds": category_durations,
            "excluded_tag_counts": excluded_counts,
            "epoch_rows_crossing_multiple_finite_trialnum_values": int(sum(row["crosses_trialnum_values"] for row in epoch_trial_alignment)),
            "epoch_rows_without_complete_20ms_bin": int(sum(row["complete_20ms_bins"] == 0 for row in epoch_trial_alignment)),
            "alignment_rows": epoch_trial_alignment,
        },
        "twenty_ms_alignment": {
            **bin_counts,
            "candidate_category_counts": bin_category_counts,
            "candidate_coverage_of_eval_valid": float(bin_counts["eval_valid_complete_candidate_category"] / max(1, bin_counts["eval_valid"])),
        },
        "hundred_ms_alignment": {
            "total_legal_trial_bounded_blocks": total_legal_blocks,
            "complete_single_candidate_epoch_blocks": complete_event_blocks,
            "excluded_intertrial_or_cross_boundary_blocks": excluded_or_boundary_blocks,
            "coverage": float(complete_event_blocks / max(1, total_legal_blocks)),
            "counts_by_trial": block_counts_by_trial,
        },
        "support_contract": {
            "support_trial_values": list(support),
            "fit_trial_values": list(support[:2]),
            "score_trial_values": list(support[2:]),
            "fit_category_block_counts": fit_counts,
            "score_category_block_counts": score_counts,
            "full_m4_category_block_counts": full_counts,
            "missing_fit_categories": missing_fit,
            "missing_score_categories": missing_score,
            "missing_full_m4_categories": missing_full,
            "all_categories_present_fit_and_score": not missing_fit and not missing_score,
            "fit_design_diagnostic": _design_diagnostic(fit_rows),
            "score_design_diagnostic": _design_diagnostic(score_rows),
        },
    }


def run_native_event_v4(*, data_dir: Path, raw_receipt: Path, eb_receipt: Path, output: Path) -> dict[str, Any]:
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") in (None, ""), "native-event audit requires CUDA_VISIBLE_DEVICES unset")
    _require(not output.exists(), f"refusing to overwrite {output}")
    _require(stat.S_IMODE(raw_receipt.stat().st_mode) == 0o444, "raw receipt must be immutable 0444")
    _require(stat.S_IMODE(eb_receipt.stat().st_mode) == 0o444, "EB receipt must be immutable 0444")
    paths = index_heldin_calib(data_dir)
    _require(tuple(paths) == H1_HELDIN_SESSIONS, "public held-in-calib scope drift")
    records = {name: load_record(paths[name]) for name in H1_HELDIN_SESSIONS}

    per_record: dict[str, Any] = {}
    global_tag_counts: dict[str, int] = {}
    global_tag_durations: dict[str, float] = {}
    missing_cells: list[dict[str, Any]] = []
    for name in H1_HELDIN_SESSIONS:
        epochs, timebase = _load_event_epochs(paths[name])
        audit = _audit_native_events(records[name], epochs)
        audit["input_sha256"] = records[name].input_sha256
        audit["timebase"] = timebase
        per_record[name] = audit
        for tag, count in audit["epoch_table"]["raw_tag_counts"].items():
            global_tag_counts[tag] = global_tag_counts.get(tag, 0) + int(count)
        for tag, duration in audit["epoch_table"]["raw_tag_duration_seconds"].items():
            global_tag_durations[tag] = global_tag_durations.get(tag, 0.0) + float(duration)
        for half in ("fit", "score"):
            missing = audit["support_contract"][f"missing_{half}_categories"]
            if missing:
                missing_cells.append({"session": name, "date": session_date(name), "half": half, "missing_categories": list(missing)})

    constructible = len(missing_cells) == 0
    # The candidate and controls are intentionally not evaluated after this
    # gate fails: removing a missing event or changing grouping after observing
    # coverage would be a post-hoc redesign.
    status = V4_STATUS_PASS if constructible else V4_STATUS_STOP_MISSING
    date_partitions = {
        date: {
            "held_public_calibration_sessions": [name for name in H1_HELDIN_SESSIONS if session_date(name) == date],
            "source_sessions": [name for name in H1_HELDIN_SESSIONS if session_date(name) != date],
        }
        for date in ALL_DATES
    }
    coverage_20 = np.asarray([row["twenty_ms_alignment"]["candidate_coverage_of_eval_valid"] for row in per_record.values()], np.float64)
    coverage_100 = np.asarray([row["hundred_ms_alignment"]["coverage"] for row in per_record.values()], np.float64)
    receipt = {
        "schema": V4_SCHEMA,
        "status": status,
        "correction_of_v3": {
            "v3_receipt": str((output.parent / "H1_PHASE_EVENT_SOURCE_AUDIT_v3.json").resolve()),
            "v3_metadata_premise_valid": False,
            "error": "v3 inspected epochs column names and raw indexed storage but did not dereference epochs.tags values",
            "v3_numerical_scope_still_valid": "TrialNum-relative chronological early/late proxy only",
            "v3_native_event_conclusion_superseded": True,
        },
        "supersedes_v4_receipt": {
            "path": str((output.parent / "H1_NATIVE_EVENT_SOURCE_AUDIT_v4.json").resolve()),
            "reason": "v4 used a misleading field name saying sessions while counting fit/score half-cells; v4r2 changes only that field name and regenerates hashes",
        },
        "scope": {
            "opened": "exactly 13 public sub-HumanPitt-held-in-calib NWBs",
            "opened_sessions": list(H1_HELDIN_SESSIONS),
            "minival_opened_or_enumerated": False,
            "held_out_calib_or_query_opened_or_enumerated": False,
            "formal_or_evalai_opened_or_enumerated": False,
            "decoder_or_checkpoint_opened": False,
            "trainer_constructed": False,
            "cuda_constructed_or_gpu_launched": False,
            "target_backpropagation": False,
        },
        "frozen_candidate_before_score": {
            "event_categories": list(EVENT_CATEGORIES),
            "presentation_rule": "collapse Presentation and Presentation2..Presentation7 into one Presentation category",
            "intertrial_rule": "exclude Intertrial as a non-movement period",
            "block_rule": "retain a legal TrialNum/eval-valid 100-ms block only if all five 20-ms bin intervals lie completely inside one named epoch row",
            "feature_design": "one_hot(named_event_category) tensor_product continuous_7d_velocity",
            "feature_width": EVENT_FEATURE_DIM,
            "encoding_fit_if_constructible": "source-RMS-normalized event-velocity design with unpenalized intercept and fixed ridge lambda 100, predicting per-channel firing rate",
            "carrier_if_constructible": "source-only SVD compresses high-dimensional coefficient rows to 3; per-channel fitted intercept b is the fourth value",
            "outer_protocol": "six-date LODO; held public calibration trials1-2 fit and trials3-4 score",
            "controls_if_constructible": f"actual zero predictor, {NULL_REPLICATES} row shuffles, and {NULL_REPLICATES} nonzero velocity rotations within the same TrialNum and same named event",
            "no_sweep": {"grouping": True, "rank": EVENT_SVD_DIM, "lag_bins": 0, "block_seconds": BLOCK_SECONDS},
            "fail_closed_rule": "if any frozen event category has zero complete blocks in either fit half or score half of any held recording, stop before estimator/projection/control scoring and do not drop or merge that category",
        },
        "corrected_native_event_metadata": {
            "global_raw_tag_vocabulary": sorted(global_tag_counts),
            "global_raw_tag_counts": global_tag_counts,
            "global_raw_tag_duration_seconds": global_tag_durations,
            "per_recording": per_record,
            "coverage_summary": {
                "candidate_20ms_coverage_of_eval_valid": _summary(coverage_20),
                "complete_single_event_100ms_coverage_of_legal_blocks": _summary(coverage_100),
            },
        },
        "six_date_source_only_partitions": date_partitions,
        "constructibility_gate": {
            "all_13_recordings_dereferenced_epoch_tags": True,
            "all_frozen_categories_present_in_fit_and_score_for_every_recording": constructible,
            "missing_phase_cells": missing_cells,
            "missing_phase_cell_count": len(missing_cells),
            "fit_score_half_cells_missing_snapto": int(sum("SnapTo" in cell["missing_categories"] for cell in missing_cells)),
            "passes": constructible,
        },
        "forward_transfer_and_controls": {
            "status": "NOT_RUN_FAIL_CLOSED_BEFORE_ESTIMATOR" if not constructible else "CONSTRUCTIBLE_REQUIRES_SEPARATE_NUMERICAL_STAGE",
            "whole_m4_comparison": "undefined because the frozen native-event candidate lacks required phase rows" if not constructible else "pending",
            "zero_control": "undefined_not_run" if not constructible else "pending",
            "row_shuffle_control": "undefined_not_run" if not constructible else "pending",
            "within_same_event_label_rotation_control": "undefined_not_run" if not constructible else "pending",
        },
        "gpu_routing_gate": {
            "passes": False,
            "gpu_authorized": False,
            "reason": "native-event design is not constructible under the frozen complete-block/missing-phase rule" if not constructible else "constructibility alone never authorizes GPU; forward-transfer stage required",
        },
        "frozen_inputs": {
            "raw_receipt": {"path": str(raw_receipt.resolve()), "sha256": sha256_file(raw_receipt), "mode": "0444"},
            "eb_receipt": {"path": str(eb_receipt.resolve()), "sha256": sha256_file(eb_receipt), "mode": "0444"},
            "source_code": {
                "audit_script_sha256_before_receipt_write": sha256_file(Path(__file__).resolve()),
                "h1_m4_eb_pilot_sha256": sha256_file(SPINT_ROOT / "src/data/h1_m4_eb_pilot.py"),
                "h1_m4_cce_date_lodo_sha256": sha256_file(SPINT_ROOT / "src/data/h1_m4_cce_date_lodo.py"),
            },
            "input_nwb_sha256": {name: records[name].input_sha256 for name in records},
        },
        "interpretation": {
            "supported": "H1 does contain native task-stage tags; the frozen all-stage event-conditioned 100-ms candidate cannot be formed from the legal first-four-trial blocks because required stages are absent after complete-block/eval masking.",
            "not_supported": [
                "H1 lacks native events",
                "native-event carrier decoding is negative",
                "dropping SnapTo or regrouping events would preserve this preregistered candidate",
                "formal or hidden H1 performance was evaluated",
            ],
        },
    }
    path, digest = write_immutable_json(output, receipt)
    return {"receipt_path": str(path), "receipt_sha256": digest, "status": status, "constructibility_gate": receipt["constructibility_gate"], "gpu_gate": receipt["gpu_routing_gate"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=SPINT_ROOT / "data/000954")
    parser.add_argument(
        "--raw-receipt", type=Path,
        default=WORKSPACE / "sua_exploration/results/h1_m4_population_decoder_carrier_date_lodo_v1/H1_M4_POPULATION_DECODER_CARRIER_CPU_RECEIPT.json",
    )
    parser.add_argument(
        "--eb-receipt", type=Path,
        default=WORKSPACE / "sua_exploration/results/h1_m4_empirical_bayes_confidence_carrier_date_lodo_v1/H1_M4_EMPIRICAL_BAYES_CONFIDENCE_CARRIER_CPU_RECEIPT.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=WORKSPACE / "sua_exploration/results/h1_phase_event_source_audit_v1/H1_NATIVE_EVENT_SOURCE_AUDIT_v4r2.json",
    )
    args = parser.parse_args()
    print(json.dumps(run_native_event_v4(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
