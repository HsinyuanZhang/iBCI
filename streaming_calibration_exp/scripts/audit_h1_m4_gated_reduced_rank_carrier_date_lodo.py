#!/usr/bin/env python3
"""Frozen source-only H1 M=4 gate x reduced-rank EB carrier audit.

This module intentionally has no SPINT/GPU path.  It evaluates the actual
carrier-implied reconstruction E @ U.T on strict chronological query trials,
rather than merely changing a split-half attachment statistic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import audit_h1_afc4_m2_date_lodo as h1


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "h1_m4_gated_reduced_rank_carrier_date_lodo_cpu_v1"
PROTOCOL_PATH = ROOT / "sua_exploration/docs/H1_M4_GATED_REDUCED_RANK_CARRIER_FROZEN_PROTOCOL.md"
TEST_PATH = ROOT / "streaming_calibration_exp/tests/test_h1_m4_gated_reduced_rank_carrier_date_lodo.py"
Q_NEURAL = 16
RIDGE_LAMBDA = 100.0
KIN_RANKS = (1, 2, 3)
GATES = ("G0_all_finite", "G1_rt_active_5bin")
PRIMARY_GATE = "G1_rt_active_5bin"
PRIMARY_KIN_RANK = 3
NULL_REPLICATES = 31
NULL_SEED = 20260807
MIN_DATES = 4
ATTACHMENT_THRESHOLD = 0.5
ACTIVE_EPSILON = 1e-3
EPS = 1e-12


class AuditError(ValueError):
    """A frozen-contract violation that makes a cell undefined."""


@dataclass(frozen=True)
class TrialBlocks:
    """Legal 100-ms blocks, retaining G1 eligibility made from raw 20-ms bins."""

    trial_number: float
    rates: np.ndarray  # [blocks, neurons]
    velocity: np.ndarray  # [blocks, 7], the correctly paired block means
    active_5bin: np.ndarray  # [blocks], fixed before every null rotation
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class FullRecord:
    session_name: str
    date: str
    path: Path | None
    input_sha256: str | None
    trials: tuple[TrialBlocks, ...]


@dataclass(frozen=True)
class SourcePlan:
    outer_date: str
    gate: str
    source_sessions: tuple[str, ...]
    source_input_sha256: tuple[str | None, ...]
    mean: np.ndarray
    scale: np.ndarray
    pcs: np.ndarray
    source_rows: np.ndarray
    right_vectors: np.ndarray
    sha256: str


@dataclass(frozen=True)
class CellPlan:
    source: SourcePlan
    q_kin: int
    U: np.ndarray
    prior_mu: np.ndarray
    prior_tau2: float
    source_energy_fraction: float
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _require_gate(gate: str) -> None:
    if gate not in GATES:
        raise AuditError(f"unknown frozen gate {gate}")


def _require_rank(q_kin: int) -> None:
    if q_kin not in KIN_RANKS:
        raise AuditError(f"q_kin={q_kin} is not in frozen ranks {KIN_RANKS}")


def _support_query(record: FullRecord) -> tuple[tuple[TrialBlocks, ...], tuple[TrialBlocks, ...]]:
    if len(record.trials) < 7:
        raise AuditError(f"{record.session_name}: M=4 requires at least three chronological query trials")
    support, query = tuple(record.trials[:4]), tuple(record.trials[4:])
    if any(trial.rates.shape[0] == 0 for trial in support) or any(trial.rates.shape[0] == 0 for trial in query):
        raise AuditError(f"{record.session_name}: support/query has an empty legal trial")
    return support, query


def _selection(trial: TrialBlocks, gate: str) -> np.ndarray:
    _require_gate(gate)
    if trial.rates.ndim != 2 or trial.velocity.ndim != 2 or trial.rates.shape[0] != trial.velocity.shape[0]:
        raise AuditError(f"trial {trial.trial_number}: malformed block arrays")
    if trial.velocity.shape[1] != h1.VELOCITY_DIM:
        raise AuditError(f"trial {trial.trial_number}: expected {h1.VELOCITY_DIM}-D velocity")
    eligible = np.ones(trial.rates.shape[0], dtype=bool) if gate == "G0_all_finite" else np.asarray(trial.active_5bin, dtype=bool)
    if eligible.shape != (trial.rates.shape[0],):
        raise AuditError(f"trial {trial.trial_number}: activity eligibility shape mismatch")
    # Every retained support trial needs a nonidentity in-trial label rotation.
    if int(eligible.sum()) < 2:
        raise AuditError(f"trial {trial.trial_number}: fewer than two retained support blocks")
    return eligible


def _selected_support(
    record: FullRecord, gate: str, trial_indices: Sequence[int] = (0, 1, 2, 3)
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    support, _ = _support_query(record)
    trials = [support[index] for index in trial_indices]
    indices = tuple(_selection(trial, gate) for trial in trials)
    rates = np.concatenate([trial.rates[index] for trial, index in zip(trials, indices)], axis=0)
    labels = np.concatenate([trial.velocity[index] for trial, index in zip(trials, indices)], axis=0)
    if rates.shape[0] < 2 or not np.isfinite(rates).all() or not np.isfinite(labels).all():
        raise AuditError(f"{record.session_name}: selected support is nonfinite or too small")
    return rates, labels, tuple(trials), indices


def _query_arrays(record: FullRecord) -> tuple[np.ndarray, np.ndarray, tuple[TrialBlocks, ...]]:
    _, query = _support_query(record)
    rates = np.concatenate([trial.rates for trial in query], axis=0)
    labels = np.concatenate([trial.velocity for trial in query], axis=0)
    if rates.shape[0] == 0 or not np.isfinite(rates).all() or not np.isfinite(labels).all():
        raise AuditError(f"{record.session_name}: query is nonfinite or empty")
    return rates, labels, query


def _project(rates: np.ndarray, plan: SourcePlan) -> np.ndarray:
    return ((rates - plan.mean[None, :]) / plan.scale[None, :]) @ plan.pcs.T


def _fit_ridge(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = np.column_stack((np.ones(features.shape[0]), features))
    regularizer = np.eye(design.shape[1], dtype=np.float64) * RIDGE_LAMBDA
    regularizer[0, 0] = 0.0
    matrix = design.T @ design + regularizer
    beta = np.linalg.solve(matrix, design.T @ labels)
    residual = labels - design @ beta
    rss = np.square(residual).sum(axis=0)
    inverse = np.linalg.inv(matrix)
    degrees = float(design.shape[0] - np.trace(design @ inverse @ design.T))
    if not np.isfinite(degrees) or degrees <= EPS:
        raise AuditError("ridge residual covariance degrees of freedom undefined")
    sigma2 = rss / degrees
    covariance = inverse @ (design.T @ design) @ inverse
    covariance = (covariance + covariance.T) / 2.0
    if not np.isfinite(beta).all() or not np.isfinite(sigma2).all() or np.any(sigma2 < 0.0):
        raise AuditError("ridge fit is nonfinite")
    return beta, covariance, sigma2


def _raw_rows(beta: np.ndarray, plan: SourcePlan) -> np.ndarray:
    return (plan.pcs.T @ beta[1:]) / plan.scale[:, None]


def _fixed_sign_columns(U: np.ndarray) -> np.ndarray:
    fixed = np.array(U, dtype=np.float64, copy=True)
    for column in range(fixed.shape[1]):
        pivot = int(np.argmax(np.abs(fixed[:, column])))
        if fixed[pivot, column] < 0.0:
            fixed[:, column] *= -1.0
    return fixed


def _build_source_plan(records: Mapping[str, FullRecord], outer_date: str, gate: str) -> SourcePlan:
    _require_gate(gate)
    source = tuple(sorted(name for name, record in records.items() if record.date != outer_date))
    if not source or any(records[name].date == outer_date for name in source):
        raise AuditError("outer date leaked into source plan")
    support_rates = np.concatenate([_selected_support(records[name], gate)[0] for name in source], axis=0)
    mean = support_rates.mean(axis=0)
    scale = np.maximum(support_rates.std(axis=0), 1e-6)
    _, _, pcs = np.linalg.svd((support_rates - mean[None, :]) / scale[None, :], full_matrices=False)
    if pcs.shape[0] < Q_NEURAL:
        raise AuditError(f"source PCA has fewer than q_neural={Q_NEURAL} components")
    pcs = pcs[:Q_NEURAL]
    provisional = SourcePlan(outer_date, gate, source, (), mean, scale, pcs, np.empty((0, h1.VELOCITY_DIM)), np.empty((0, h1.VELOCITY_DIM)), "")
    rows: list[np.ndarray] = []
    for name in source:
        rates, labels, _trials, _indices = _selected_support(records[name], gate)
        beta, _covariance, _sigma2 = _fit_ridge(_project(rates, provisional), labels)
        rows.append(_raw_rows(beta, provisional))
    source_rows = np.concatenate(rows, axis=0)
    _left, singular_values, right_vectors = np.linalg.svd(source_rows, full_matrices=False)
    if right_vectors.shape[0] < max(KIN_RANKS) or not np.isfinite(singular_values).all():
        raise AuditError("source coefficient rows cannot form all frozen reduced ranks")
    inputs = tuple(records[name].input_sha256 for name in source)
    body = {
        "protocol": PROTOCOL,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "outer_date": outer_date,
        "gate": gate,
        "source_sessions": source,
        "source_input_sha256": inputs,
        "normalizer_scope": "outer-source dates, M=4 correct-label retained support blocks only",
        "pca_scope": "outer-source dates, M=4 correct-label retained support blocks only",
        "q_neural": Q_NEURAL,
        "ridge_lambda": RIDGE_LAMBDA,
        "carrier_basis_scope": "outer-source pooled decoder rows only",
        "source_singular_values": singular_values.tolist(),
    }
    return SourcePlan(outer_date, gate, source, inputs, mean, scale, pcs, source_rows, right_vectors, _json_sha(body))


def _cell_plan(source: SourcePlan, q_kin: int) -> CellPlan:
    _require_rank(q_kin)
    U = _fixed_sign_columns(source.right_vectors[:q_kin].T)
    source_carrier = source.source_rows @ U
    mu = source_carrier.mean(axis=0)
    tau2 = float(np.square(source_carrier - mu[None, :]).sum() / (source_carrier.shape[0] * q_kin))
    total_energy = float(np.square(np.linalg.svd(source.source_rows, compute_uv=False)).sum())
    kept_energy = float(np.square(np.linalg.svd(source.source_rows, compute_uv=False)[:q_kin]).sum())
    if not np.isfinite(tau2) or tau2 <= EPS or total_energy <= EPS:
        raise AuditError("source EB prior is undefined")
    body = {"source_plan_sha256": source.sha256, "q_kin": q_kin, "U": U.tolist(), "prior_mu": mu.tolist(), "prior_tau2": tau2}
    return CellPlan(source, q_kin, U, mu, tau2, kept_energy / total_energy, _json_sha(body))


def _shrink(beta: np.ndarray, covariance: np.ndarray, sigma2: np.ndarray, plan: CellPlan) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = _raw_rows(beta, plan.source)
    carrier = raw @ plan.U
    P = plan.source.pcs.T
    feature_covariance = covariance[1:, 1:]
    raw_feature_variance = ((P @ feature_covariance) * P).sum(axis=1) / np.square(plan.source.scale)
    projected_output_variance = plan.U.T @ np.diag(sigma2) @ plan.U
    variance = raw_feature_variance * (float(np.trace(projected_output_variance)) / plan.q_kin)
    if not np.isfinite(variance).all() or np.any(variance < 0.0):
        raise AuditError("projected EB variance is undefined")
    weight = plan.prior_tau2 / (plan.prior_tau2 + variance)
    result = plan.prior_mu[None, :] + weight[:, None] * (carrier - plan.prior_mu[None, :])
    if not np.isfinite(result).all() or np.any(weight <= 0.0) or np.any(weight > 1.0):
        raise AuditError("EB carrier is undefined")
    return result, weight, variance


def _fit_target(
    record: FullRecord,
    plan: CellPlan,
    trial_indices: Sequence[int] = (0, 1, 2, 3),
    rotated_labels: Mapping[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    rates, correct_labels, trials, indices = _selected_support(record, plan.source.gate, trial_indices)
    labels = correct_labels if rotated_labels is None else np.concatenate(
        [rotated_labels[trial_indices[position]] for position in range(len(trials))], axis=0
    )
    if labels.shape != correct_labels.shape:
        raise AuditError("null labels changed retained support rows")
    beta, covariance, sigma2 = _fit_ridge(_project(rates, plan.source), labels)
    return beta, covariance, sigma2, indices


def _support_mean(record: FullRecord, gate: str) -> np.ndarray:
    _rates, labels, _trials, _indices = _selected_support(record, gate)
    return labels.mean(axis=0)


def _metric(record: FullRecord, carrier: np.ndarray, intercept: np.ndarray, plan: CellPlan) -> dict[str, float] | None:
    query_rates, query_labels, _query = _query_arrays(record)
    reconstruction = carrier @ plan.U.T
    prediction = (query_rates - plan.source.mean[None, :]) @ reconstruction + intercept[None, :]
    support_mean = _support_mean(record, plan.source.gate)
    sse = float(np.square(query_labels - prediction).sum())
    tss = float(np.square(query_labels - support_mean[None, :]).sum())
    if not np.isfinite(sse) or not np.isfinite(tss) or tss <= EPS:
        return None
    return {"r2": float(1.0 - sse / tss), "sse": sse, "tss": tss}


def _row_cosine(first: np.ndarray, second: np.ndarray) -> float | None:
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    valid = denominator > EPS
    return None if not valid.any() else float(np.median((first[valid] * second[valid]).sum(axis=1) / denominator[valid]))


def _rotation_offset(session: str, trial_number: float, length: int, replicate: int) -> int:
    if length < 2:
        raise AuditError("label rotation needs at least two frozen retained blocks")
    token = f"{NULL_SEED}|{session}|{trial_number}|{replicate}".encode()
    return 1 + int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % (length - 1)


def _rotated_retained_labels(record: FullRecord, gate: str, replicate: int) -> tuple[dict[int, np.ndarray], tuple[int, ...]]:
    support, _query = _support_query(record)
    result: dict[int, np.ndarray] = {}
    counts: list[int] = []
    for index, trial in enumerate(support):
        selected = _selection(trial, gate)  # always correct-label selection; never recompute after rotation
        labels = trial.velocity[selected]
        rotated = np.roll(labels, _rotation_offset(record.session_name, trial.trial_number, labels.shape[0], replicate), axis=0)
        if np.array_equal(rotated, labels):
            raise AuditError("label rotation offset was nonzero but did not change retained labels")
        result[index] = rotated
        counts.append(int(labels.shape[0]))
    return result, tuple(counts)


def _row_shuffle(carrier: np.ndarray, record: FullRecord, plan: CellPlan) -> tuple[np.ndarray, str]:
    seed = int.from_bytes(hashlib.sha256(f"{NULL_SEED}|row|{record.session_name}|{plan.sha256}".encode()).digest()[:8], "big")
    permutation = np.random.default_rng(seed).permutation(carrier.shape[0])
    if np.array_equal(permutation, np.arange(carrier.shape[0])):
        permutation = np.roll(permutation, 1)
    return carrier[permutation], hashlib.sha256(np.ascontiguousarray(permutation).tobytes()).hexdigest()


def _record_result(record: FullRecord, plan: CellPlan) -> dict[str, Any]:
    support, query = _support_query(record)
    beta, covariance, sigma2, pooled_indices = _fit_target(record, plan)
    carrier, weight, variance = _shrink(beta, covariance, sigma2, plan)
    correct = _metric(record, carrier, beta[0], plan)
    beta_a, cov_a, sig_a, _indices_a = _fit_target(record, plan, (0, 1))
    beta_b, cov_b, sig_b, _indices_b = _fit_target(record, plan, (2, 3))
    carrier_a, _wa, _va = _shrink(beta_a, cov_a, sig_a, plan)
    carrier_b, _wb, _vb = _shrink(beta_b, cov_b, sig_b, plan)
    attachment = _row_cosine(carrier_a, carrier_b)
    shuffled, permutation_sha = _row_shuffle(carrier, record, plan)
    row_shuffle = _metric(record, shuffled, beta[0], plan)
    retained_counts = tuple(int(index.sum()) for index in pooled_indices)
    nulls: list[dict[str, float] | None] = []
    null_counts: list[list[int]] = []
    for replicate in range(NULL_REPLICATES):
        rotated, counts = _rotated_retained_labels(record, plan.source.gate, replicate)
        null_beta, null_covariance, null_sigma2, null_indices = _fit_target(record, plan, rotated_labels=rotated)
        if tuple(int(index.sum()) for index in null_indices) != retained_counts or counts != retained_counts:
            raise AuditError("label null changed frozen retained support rows or block counts")
        null_carrier, _null_weight, _null_variance = _shrink(null_beta, null_covariance, null_sigma2, plan)
        nulls.append(_metric(record, null_carrier, null_beta[0], plan))
        null_counts.append(list(counts))
    defined = correct is not None and row_shuffle is not None and attachment is not None and all(item is not None for item in nulls)
    null_values = [item["r2"] for item in nulls if item is not None]
    return {
        "status": "defined" if defined else "undefined",
        "session_name": record.session_name,
        "support_trial_numbers": [trial.trial_number for trial in support],
        "query_trial_numbers": [trial.trial_number for trial in query],
        "support_candidate_100ms_blocks": int(sum(trial.rates.shape[0] for trial in support)),
        "support_candidate_by_trial": [int(trial.rates.shape[0]) for trial in support],
        "support_retained_100ms_blocks": int(sum(retained_counts)),
        "support_retained_by_trial": list(retained_counts),
        "support_discarded_by_gate": int(sum(trial.rates.shape[0] for trial in support) - sum(retained_counts)),
        "split_a_retained_100ms_blocks": int(sum(retained_counts[:2])),
        "split_b_retained_100ms_blocks": int(sum(retained_counts[2:])),
        "support_retained_exposure_seconds": float(sum(retained_counts) * 0.1),
        "query_100ms_blocks_ungated": int(sum(trial.rates.shape[0] for trial in query)),
        "query_exposure_seconds_ungated": float(sum(trial.rates.shape[0] for trial in query) * 0.1),
        "correct_reconstructed_query_r2": None if correct is None else correct["r2"],
        "correct_sse": None if correct is None else correct["sse"],
        "correct_tss_fit_side_mean": None if correct is None else correct["tss"],
        "split_pair_eb_carrier_cosine_median": attachment,
        "row_shuffle_reconstructed_query_r2": None if row_shuffle is None else row_shuffle["r2"],
        "row_shuffle_sse": None if row_shuffle is None else row_shuffle["sse"],
        "row_shuffle_tss_fit_side_mean": None if row_shuffle is None else row_shuffle["tss"],
        "carrier_shape": [int(carrier.shape[0]), int(carrier.shape[1])],
        "carrier_sha256": hashlib.sha256(np.ascontiguousarray(carrier).tobytes()).hexdigest(),
        "carrier_reconstruction": "E@U.T",
        "mean_eb_weight": float(weight.mean()),
        "mean_projected_variance": float(variance.mean()),
        "row_shuffle_permutation_sha256": permutation_sha,
        "rotation_null_reconstructed_query_r2": null_values,
        "rotation_null_sse": [None if item is None else item["sse"] for item in nulls],
        "rotation_null_tss_fit_side_mean": [None if item is None else item["tss"] for item in nulls],
        "rotation_null_retained_by_trial": null_counts,
        "rotation_null_reuses_correct_frozen_retained_blocks": True,
    }


def _weighted(rows: Sequence[Mapping[str, float]]) -> dict[str, float] | None:
    sse = float(sum(row["sse"] for row in rows))
    tss = float(sum(row["tss"] for row in rows))
    return None if not np.isfinite(sse) or not np.isfinite(tss) or tss <= EPS else {"r2": float(1.0 - sse / tss), "sse": sse, "tss": tss}


def _date_result(rows: list[dict[str, Any]], plan: CellPlan) -> dict[str, Any]:
    defined = bool(rows) and all(row["status"] == "defined" for row in rows)
    correct = _weighted([{"sse": row["correct_sse"], "tss": row["correct_tss_fit_side_mean"]} for row in rows]) if defined else None
    shuffled = _weighted([{"sse": row["row_shuffle_sse"], "tss": row["row_shuffle_tss_fit_side_mean"]} for row in rows]) if defined else None
    null_values: list[float] | None = [] if defined else None
    if null_values is not None:
        for replicate in range(NULL_REPLICATES):
            metric = _weighted([{"sse": row["rotation_null_sse"][replicate], "tss": row["rotation_null_tss_fit_side_mean"][replicate]} for row in rows])
            if metric is None:
                null_values = None
                break
            null_values.append(metric["r2"])
    attachment = None if not defined else float(np.mean([row["split_pair_eb_carrier_cosine_median"] for row in rows]))
    q95 = None if null_values is None else float(np.quantile(null_values, 0.95))
    return {
        "date": plan.source.outer_date,
        "gate": plan.source.gate,
        "q_kin": plan.q_kin,
        "status": "defined" if correct is not None and shuffled is not None and null_values is not None and attachment is not None else "undefined",
        "plan": {
            "source_sessions": list(plan.source.source_sessions),
            "source_input_sha256": list(plan.source.source_input_sha256),
            "source_plan_sha256": plan.source.sha256,
            "cell_plan_sha256": plan.sha256,
            "q_neural": Q_NEURAL,
            "ridge_lambda": RIDGE_LAMBDA,
            "U_shape": list(plan.U.shape),
            "U": plan.U.tolist(),
            "U_sha256": hashlib.sha256(np.ascontiguousarray(plan.U).tobytes()).hexdigest(),
            "source_singular_values": np.linalg.svd(plan.source.source_rows, compute_uv=False).tolist(),
            "prior_mu": plan.prior_mu.tolist(),
            "prior_tau2": plan.prior_tau2,
            "source_rank_energy_fraction": plan.source_energy_fraction,
        },
        "correct_reconstructed_query_r2": None if correct is None else correct["r2"],
        "rotation_null_q95": q95,
        "correct_minus_rotation_null_q95": None if correct is None or q95 is None else correct["r2"] - q95,
        "split_pair_eb_carrier_cosine_mean_median": attachment,
        "row_shuffle_reconstructed_query_r2": None if shuffled is None else shuffled["r2"],
        "correct_minus_row_shuffle": None if correct is None or shuffled is None else correct["r2"] - shuffled["r2"],
        "rotation_null_reconstructed_query_r2": null_values,
        "records": rows,
    }


def _cell_result(records: Mapping[str, FullRecord], gate: str, q_kin: int) -> dict[str, Any]:
    dates_out: list[dict[str, Any]] = []
    for outer_date in h1.H1_DATES:
        try:
            source = _build_source_plan(records, outer_date, gate)
            plan = _cell_plan(source, q_kin)
            target = [_record_result(record, plan) for record in records.values() if record.date == outer_date]
            dates_out.append(_date_result(target, plan))
        except Exception as exc:
            dates_out.append({"date": outer_date, "gate": gate, "q_kin": q_kin, "status": "undefined", "reason": str(exc)})
    count_ge = lambda key, threshold: sum(row.get(key, -np.inf) >= threshold for row in dates_out)
    count_gt_zero = lambda key: sum(row.get(key, -np.inf) > 0.0 for row in dates_out)
    positive = sum(row.get("correct_reconstructed_query_r2", -np.inf) > 0.0 for row in dates_out)
    gate_counts = {
        "all_six_dates_defined": all(row["status"] == "defined" for row in dates_out),
        "correct_reconstructed_r2_positive_dates": positive,
        "correct_above_rotation_q95_dates": count_gt_zero("correct_minus_rotation_null_q95"),
        "split_pair_attachment_at_least_050_dates": count_ge("split_pair_eb_carrier_cosine_mean_median", ATTACHMENT_THRESHOLD),
        "correct_above_row_shuffle_dates": count_gt_zero("correct_minus_row_shuffle"),
        "minimum_required_dates": MIN_DATES,
        "attachment_threshold": ATTACHMENT_THRESHOLD,
    }
    gate_counts["pass"] = bool(gate_counts["all_six_dates_defined"] and all(gate_counts[key] >= MIN_DATES for key in (
        "correct_reconstructed_r2_positive_dates", "correct_above_rotation_q95_dates", "split_pair_attachment_at_least_050_dates", "correct_above_row_shuffle_dates")))
    return {"gate": gate, "q_kin": q_kin, "is_primary": gate == PRIMARY_GATE and q_kin == PRIMARY_KIN_RANK, "date_lodo": dates_out, "cpu_gate": gate_counts}


def run_audit(records: Mapping[str, FullRecord], *, require_authority_records: bool = True) -> dict[str, Any]:
    if set(records) != {record.session_name for record in records.values()}:
        raise AuditError("record mapping key/session name mismatch")
    if require_authority_records and set(records) != set(h1.H1_HELDIN_SESSIONS):
        raise AuditError("real audit requires exactly the 13 H1 held-in-calib authority sessions")
    if tuple(sorted({record.date for record in records.values()})) != tuple(h1.H1_DATES):
        raise AuditError("requires exactly the six H1 authority dates")
    cells = [_cell_result(records, gate, q_kin) for gate in GATES for q_kin in KIN_RANKS]
    primary = next(cell for cell in cells if cell["is_primary"])
    primary_pass = bool(primary["cpu_gate"]["pass"])
    return {
        "schema": "h1_m4_gated_reduced_rank_carrier_date_lodo_cpu_v1",
        "protocol": PROTOCOL,
        "status": "PASS_CPU_H1_M4_GATED_REDUCED_RANK__ELIGIBLE_FOR_SEPARATE_GPU_PROPOSAL" if primary_pass else "STOP_CPU_H1_M4_GATED_REDUCED_RANK_FAILED__NO_GPU_AUTHORIZATION",
        "scope": {"opened_data": "exactly 13 public held-in-calib NWBs", "heldin_trials5plus_scored": True, "external_query_recordings_opened": False, "formal_heldout_opened": False, "minival_opened": False, "evalai_opened": False, "gpu_constructed": False, "target_backpropagation": False},
        "source_hashes": {"audit_script_sha256": sha256_file(Path(__file__)), "protocol_sha256": sha256_file(PROTOCOL_PATH), "test_sha256": sha256_file(TEST_PATH), "h1_block_authority_sha256": sha256_file(Path(h1.__file__))},
        "input_nwb_sha256": {name: record.input_sha256 for name, record in sorted(records.items())},
        "design": {
            "matrix": {"gates": list(GATES), "q_kin": list(KIN_RANKS), "primary": {"gate": PRIMARY_GATE, "q_kin": PRIMARY_KIN_RANK}},
            "G1": "each raw 20-ms bin in a retained 5-bin support block is ~all(abs(v)<1e-3); correct-label eligibility frozen before nulls",
            "query": "trials5+ all legal blocks, never activity-gated",
            "source_plan": "outer-date source-only normalizer/PCA/U/EB prior; q_neural=16 and lambda=100 fixed",
            "score": "actual EB carrier reconstruction R_carrier=E@U.T",
            "rotation_null": "31 within-trial retained-block label rotations; retained neural rows and block counts frozen",
            "row_shuffle": "complete nonidentity E row permutation scored through shuffled_E@U.T",
        },
        "cells": cells,
        "primary_cpu_gate": primary["cpu_gate"],
        "decision": {"gpu_authorized_by_this_receipt": False, "eligible_for_separate_gpu_proposal": primary_pass, "if_fail": "STOP; do not select another matrix cell or tune gate/rank/source/EB/fusion/GPU.", "sealed_prior_routes_unchanged": True},
    }


def _all_trial_values(trial_num: np.ndarray, eval_mask: np.ndarray) -> tuple[float, ...]:
    labels = np.asarray(trial_num, dtype=np.float64).reshape(-1)
    valid = np.flatnonzero(np.asarray(eval_mask, dtype=bool).reshape(-1) & np.isfinite(labels))
    ordered = labels[valid]
    if valid.size == 0 or np.any(np.diff(ordered) < 0.0):
        raise AuditError("TrialNum has no chronological eval-valid values")
    values: list[float] = []
    for value in ordered.tolist():
        if not values or value != values[-1]:
            values.append(float(value))
    return tuple(values)


def _trial_to_blocks(*, trial_number: float, neural: np.ndarray, velocity: np.ndarray, eval_mask: np.ndarray, trial_num: np.ndarray) -> TrialBlocks:
    belongs = np.asarray(trial_num == float(trial_number), dtype=bool)
    legal = belongs & np.asarray(eval_mask, dtype=bool) & np.isfinite(neural).all(axis=1) & np.isfinite(velocity).all(axis=1)
    indices = np.flatnonzero(legal)
    chunks: list[np.ndarray] = []
    if indices.size:
        breaks = np.flatnonzero(np.diff(indices) != 1) + 1
        chunks = [np.asarray(part, dtype=np.int64) for part in np.split(indices, breaks)]
    rates: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    active: list[bool] = []
    for chunk in chunks:
        for offset in range(0, (chunk.size // h1.BLOCK_BINS) * h1.BLOCK_BINS, h1.BLOCK_BINS):
            block = chunk[offset : offset + h1.BLOCK_BINS]
            raw_velocity = velocity[block]
            rates.append(neural[block].sum(axis=0) / h1.BLOCK_SECONDS)
            labels.append(raw_velocity.mean(axis=0))
            raw_active = ~np.all(np.abs(raw_velocity) < ACTIVE_EPSILON, axis=1)
            active.append(bool(raw_active.all()))
    neurons = neural.shape[1]
    return TrialBlocks(float(trial_number), np.asarray(rates, dtype=np.float64).reshape((-1, neurons)), np.asarray(labels, dtype=np.float64).reshape((-1, h1.VELOCITY_DIM)), np.asarray(active, dtype=bool), {"raw_trial_bins": int(belongs.sum()), "complete_100ms_blocks": len(rates), "G1_active_5bin_blocks": int(sum(active)), "activity_rule": "each_raw_bin_not_all_abs_velocity_lt_1e-3"})


def load_full_h1_record(path: str | Path) -> FullRecord:
    resolved = Path(path).resolve()
    h1._require_heldin_calib_path(resolved)
    try:
        from falcon_challenge.config import FalconTask
        from falcon_challenge.dataloaders import load_nwb
        from pynwb import NWBHDF5IO
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("H1 audit requires falcon_challenge and pynwb") from exc
    neural, velocity, _trial_change, eval_mask = load_nwb(resolved, FalconTask.h1)
    with NWBHDF5IO(str(resolved), "r", load_namespaces=True) as io:
        nwb = io.read()
        if "TrialNum" not in nwb.acquisition:
            raise AuditError(f"{resolved}: TrialNum missing")
        trial_num = np.asarray(nwb.acquisition["TrialNum"].data[:], dtype=np.float64)
    neural = h1._finite_matrix(neural, name="neural")
    velocity = h1._finite_matrix(velocity, name="velocity")
    if neural.shape[0] != velocity.shape[0] or velocity.shape[1] != h1.VELOCITY_DIM:
        raise AuditError("H1 neural/velocity shape mismatch")
    session = h1._session_name_from_path(resolved)
    record = FullRecord(session, h1.date_from_session(session), resolved, h1.sha256_file(resolved), tuple(_trial_to_blocks(trial_number=value, neural=neural, velocity=velocity, eval_mask=np.asarray(eval_mask, dtype=bool), trial_num=trial_num) for value in _all_trial_values(trial_num, eval_mask)))
    if record.trials[0].rates.shape[1] != h1.EXPECTED_NEURONS:
        raise AuditError(f"{session}: unexpected H1 channel count")
    _support_query(record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "SPINT-main/data/000954")
    parser.add_argument("--output", type=Path, default=ROOT / "sua_exploration/results/h1_m4_gated_reduced_rank_carrier_date_lodo_v1/H1_M4_GATED_REDUCED_RANK_CARRIER_CPU_RECEIPT.json")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise RuntimeError("CPU-only audit requires CUDA_VISIBLE_DEVICES unset")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {args.output}")
    paths = h1.index_h1_heldin_calib(args.data_dir)
    output = run_audit({name: load_full_h1_record(path) for name, path in paths.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.chmod(0o444)
    print(json.dumps({"status": output["status"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
