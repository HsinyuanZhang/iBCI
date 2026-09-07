#!/usr/bin/env python3
"""Frozen CPU-only H1 M=4 population decoder-direction carrier audit.

This is a separate M=4 budget program.  It uses target trials 1--4 only for
closed-form fitting and trials 5+ only for scoring; it neither reopens M=2 nor
constructs a GPU/SPINT model.
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
import audit_h1_population_decoder_carrier_m2_date_lodo as m2


PROTOCOL = "h1_m4_population_decoder_carrier_date_lodo_cpu_v1"
Q_GRID = (2, 4, 8, 16)
LAMBDA_GRID = (0.1, 1.0, 10.0, 100.0)
NULL_REPLICATES = 31
NULL_SEED = 20260807
MIN_DATES = 4
ATTACHMENT_THRESHOLD = 0.5
EPS = 1e-12
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "sua_exploration/docs/H1_M4_POPULATION_DECODER_CARRIER_FROZEN_PROTOCOL.md"


class AuditError(ValueError):
    pass


@dataclass(frozen=True)
class Plan:
    outer_date: str
    source_sessions: tuple[str, ...]
    source_input_sha256: tuple[str | None, ...]
    mean: np.ndarray
    scale: np.ndarray
    pcs: np.ndarray
    q: int
    ridge_lambda: float
    carrier_basis: np.ndarray
    source_grid_best_r2: float
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _support_query(record: m2.FullRecord) -> tuple[tuple[m2.h1.TrialBlocks, ...], tuple[m2.h1.TrialBlocks, ...]]:
    if len(record.trials) < 7:
        raise AuditError(f"{record.session_name}: M=4 requires four support and at least three later legal trials")
    support, query = tuple(record.trials[:4]), tuple(record.trials[4:])
    if any(trial.rates.shape[0] < 2 for trial in support) or len(query) < 3 or any(trial.rates.shape[0] == 0 for trial in query):
        raise AuditError(f"{record.session_name}: M=4 support/query has insufficient legal 100-ms blocks")
    return support, query


def _support_rates(record: m2.FullRecord) -> np.ndarray:
    support, _ = _support_query(record)
    return np.concatenate([trial.rates for trial in support], axis=0)


def _project(rates: np.ndarray, plan: Plan) -> np.ndarray:
    return ((rates - plan.mean[None, :]) / plan.scale[None, :]) @ plan.pcs[: plan.q].T


def _fit_ridge(features: np.ndarray, labels: np.ndarray, ridge_lambda: float) -> np.ndarray:
    design = np.column_stack((np.ones(features.shape[0]), features))
    regularizer = np.eye(design.shape[1]) * ridge_lambda
    regularizer[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + regularizer, design.T @ labels)


def _raw_rows(beta: np.ndarray, plan: Plan) -> np.ndarray:
    return (plan.pcs[: plan.q].T @ beta[1:]) / plan.scale[:, None]


def _fit_support(
    record: m2.FullRecord,
    plan: Plan,
    trial_indices: Sequence[int] = (0, 1, 2, 3),
    labels_override: Mapping[int, np.ndarray] | None = None,
) -> np.ndarray:
    support, _ = _support_query(record)
    chosen = [support[index] for index in trial_indices]
    rates = np.concatenate([trial.rates for trial in chosen], axis=0)
    labels = np.concatenate(
        [trial.velocity if labels_override is None else labels_override[index] for index, trial in zip(trial_indices, chosen)], axis=0
    )
    return _fit_ridge(_project(rates, plan), labels, plan.ridge_lambda)


def _score_query(record: m2.FullRecord, beta: np.ndarray, plan: Plan) -> dict[str, float] | None:
    support, query = _support_query(record)
    fit_mean = np.concatenate([trial.velocity for trial in support], axis=0).mean(axis=0)
    sse = 0.0
    tss = 0.0
    for trial in query:
        prediction = np.column_stack((np.ones(trial.rates.shape[0]), _project(trial.rates, plan))) @ beta
        sse += float(np.square(trial.velocity - prediction).sum())
        tss += float(np.square(trial.velocity - fit_mean[None, :]).sum())
    if not np.isfinite(sse) or not np.isfinite(tss) or tss <= EPS:
        return None
    return {"r2": float(1.0 - sse / tss), "sse": sse, "tss": tss}


def _row_cosine(first: np.ndarray, second: np.ndarray) -> float | None:
    numerator = (first * second).sum(axis=1)
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    values = numerator[denominator > EPS] / denominator[denominator > EPS]
    return None if values.size == 0 else float(np.median(values))


def _rotation(session: str, trial: float, length: int, replicate: int) -> int:
    if length < 2:
        raise AuditError("rotation null needs at least two support blocks")
    value = hashlib.sha256(f"{NULL_SEED}|{session}|{trial}|{replicate}".encode()).digest()
    return 1 + int.from_bytes(value[:8], "big") % (length - 1)


def _source_score(record: m2.FullRecord, plan: Plan) -> float:
    metric = _score_query(record, _fit_support(record, plan), plan)
    if metric is None:
        raise AuditError(f"{record.session_name}: source grid score undefined")
    return metric["r2"]


def _build_plan(records: Mapping[str, m2.FullRecord], outer_date: str) -> Plan:
    source = tuple(sorted(name for name, record in records.items() if record.date != outer_date))
    if not source or any(records[name].date == outer_date for name in source):
        raise AuditError("outer date leaked into source plan")
    source_rates = np.concatenate([_support_rates(records[name]) for name in source], axis=0)
    mean = source_rates.mean(axis=0)
    scale = np.maximum(source_rates.std(axis=0), 1e-6)
    _, _, pcs = np.linalg.svd((source_rates - mean[None, :]) / scale[None, :], full_matrices=False)
    if pcs.shape[0] < max(Q_GRID):
        raise AuditError("source PCA has fewer than 16 components")
    candidates: list[tuple[float, int, float]] = []
    for q in Q_GRID:
        for ridge_lambda in LAMBDA_GRID:
            provisional = Plan(outer_date, source, (), mean, scale, pcs[: max(Q_GRID)], q, ridge_lambda, np.empty((7, 4)), 0.0, "")
            candidates.append((float(np.mean([_source_score(records[name], provisional) for name in source])), q, ridge_lambda))
    best_score, q, ridge_lambda = sorted(candidates, key=lambda value: (-value[0], value[1], value[2]))[0]
    provisional = Plan(outer_date, source, (), mean, scale, pcs[: max(Q_GRID)], q, ridge_lambda, np.empty((7, 4)), best_score, "")
    source_rows = np.concatenate([_raw_rows(_fit_support(records[name], provisional), provisional) for name in source], axis=0)
    _, _, right_vectors = np.linalg.svd(source_rows, full_matrices=False)
    if right_vectors.shape[0] < 4:
        raise AuditError("source coefficient rows cannot form a 4-D basis")
    input_hashes = tuple(records[name].input_sha256 for name in source)
    body = {
        "protocol": PROTOCOL,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "outer_date": outer_date,
        "source_sessions": source,
        "source_input_sha256": input_hashes,
        "normalizer_scope": "outer-source dates, trials 1-4 only",
        "pca_scope": "outer-source dates, trials 1-4 only",
        "grid_scope": "outer-source M=4 support to trials5+ query only",
        "carrier_basis_scope": "outer-source M=4 pooled decoder rows only",
        "q_grid": Q_GRID,
        "lambda_grid": LAMBDA_GRID,
        "selected_q": q,
        "selected_lambda": ridge_lambda,
        "source_grid_best_equal_recording_r2": best_score,
    }
    return Plan(outer_date, source, input_hashes, mean, scale, pcs[: max(Q_GRID)], q, ridge_lambda, right_vectors[:4].T, best_score, _json_sha(body))


def _record_result(record: m2.FullRecord, plan: Plan) -> dict[str, Any]:
    support, query = _support_query(record)
    combined = _fit_support(record, plan)
    split_a = _fit_support(record, plan, (0, 1))
    split_b = _fit_support(record, plan, (2, 3))
    correct = _score_query(record, combined, plan)
    attachment = _row_cosine(_raw_rows(split_a, plan), _raw_rows(split_b, plan))
    carrier = _raw_rows(combined, plan) @ plan.carrier_basis
    nulls: list[dict[str, float] | None] = []
    for replicate in range(NULL_REPLICATES):
        rotated = {index: np.roll(trial.velocity, _rotation(record.session_name, trial.trial_number, trial.velocity.shape[0], replicate), axis=0) for index, trial in enumerate(support)}
        nulls.append(_score_query(record, _fit_support(record, plan, labels_override=rotated), plan))
    defined = correct is not None and attachment is not None and all(null is not None for null in nulls)
    null_r2 = [null["r2"] for null in nulls if null is not None]
    return {
        "status": "defined" if defined else "undefined",
        "session_name": record.session_name,
        "support_trial_numbers": [trial.trial_number for trial in support],
        "query_trial_numbers": [trial.trial_number for trial in query],
        "support_100ms_blocks": int(sum(trial.rates.shape[0] for trial in support)),
        "query_100ms_blocks": int(sum(trial.rates.shape[0] for trial in query)),
        "support_exposure_seconds": float(sum(trial.rates.shape[0] for trial in support) * 0.1),
        "query_exposure_seconds": float(sum(trial.rates.shape[0] for trial in query) * 0.1),
        "correct_later_query_variance_weighted_r2_relative_fit_side_mean": None if correct is None else correct["r2"],
        "correct_sse": None if correct is None else correct["sse"],
        "correct_tss_fit_side_mean": None if correct is None else correct["tss"],
        "split_pair_coefficient_row_cosine_median": attachment,
        "carrier_shape": [int(carrier.shape[0]), int(carrier.shape[1])],
        "carrier_sha256": hashlib.sha256(np.ascontiguousarray(carrier).tobytes()).hexdigest(),
        "rotation_null_r2_q95": None if not defined else float(np.quantile(null_r2, 0.95)),
        "rotation_null_r2": null_r2,
        "rotation_null_sse": [None if null is None else null["sse"] for null in nulls],
        "rotation_null_tss_fit_side_mean": [None if null is None else null["tss"] for null in nulls],
    }


def _weighted(rows: Sequence[Mapping[str, float]]) -> dict[str, float] | None:
    sse = float(sum(row["sse"] for row in rows))
    tss = float(sum(row["tss"] for row in rows))
    return None if not np.isfinite(sse) or not np.isfinite(tss) or tss <= EPS else {"r2": float(1.0 - sse / tss), "sse": sse, "tss": tss}


def _date_result(rows: list[dict[str, Any]], plan: Plan) -> dict[str, Any]:
    defined = bool(rows) and all(row["status"] == "defined" for row in rows)
    correct = _weighted([{"sse": row["correct_sse"], "tss": row["correct_tss_fit_side_mean"]} for row in rows]) if defined else None
    null_r2: list[float] | None = [] if defined else None
    if null_r2 is not None:
        for replicate in range(NULL_REPLICATES):
            metric = _weighted([{"sse": row["rotation_null_sse"][replicate], "tss": row["rotation_null_tss_fit_side_mean"][replicate]} for row in rows])
            if metric is None:
                null_r2 = None
                break
            null_r2.append(metric["r2"])
    q95 = None if null_r2 is None else float(np.quantile(null_r2, 0.95))
    attachment = None if not defined else float(np.mean([row["split_pair_coefficient_row_cosine_median"] for row in rows]))
    return {
        "date": plan.outer_date,
        "status": "defined" if correct is not None and null_r2 is not None and attachment is not None else "undefined",
        "plan": {"source_sessions": list(plan.source_sessions), "source_input_sha256": list(plan.source_input_sha256), "q": plan.q, "lambda": plan.ridge_lambda, "source_grid_best_equal_recording_r2": plan.source_grid_best_r2, "plan_sha256": plan.sha256},
        "correct_later_query_variance_weighted_r2_relative_fit_side_mean": None if correct is None else correct["r2"],
        "rotation_null_q95": q95,
        "correct_minus_rotation_null_q95": None if correct is None or q95 is None else correct["r2"] - q95,
        "split_pair_coefficient_row_attachment_mean_median_cosine": attachment,
        "records": rows,
    }


def run_audit(records: Mapping[str, m2.FullRecord]) -> dict[str, Any]:
    dates = tuple(sorted({record.date for record in records.values()}))
    if dates != tuple(m2.h1.H1_DATES) or set(records) != {record.session_name for record in records.values()}:
        raise AuditError("requires exactly the six authority dates with matching record keys")
    dates_out: list[dict[str, Any]] = []
    for outer in m2.h1.H1_DATES:
        try:
            plan = _build_plan(records, outer)
            dates_out.append(_date_result([_record_result(record, plan) for record in records.values() if record.date == outer], plan))
        except Exception as exc:
            dates_out.append({"date": outer, "status": "undefined", "reason": str(exc)})
    positive = sum(row.get("correct_later_query_variance_weighted_r2_relative_fit_side_mean", -np.inf) > 0 for row in dates_out)
    above_null = sum(row.get("correct_minus_rotation_null_q95", -np.inf) > 0 for row in dates_out)
    stable = sum(row.get("split_pair_coefficient_row_attachment_mean_median_cosine", -np.inf) >= ATTACHMENT_THRESHOLD for row in dates_out)
    gate = {"all_six_dates_defined": all(row["status"] == "defined" for row in dates_out), "correct_gain_positive_dates": positive, "correct_above_rotation_q95_dates": above_null, "split_pair_attachment_stable_dates": stable, "minimum_required_dates": MIN_DATES, "attachment_threshold": ATTACHMENT_THRESHOLD}
    gate["pass"] = bool(gate["all_six_dates_defined"] and positive >= MIN_DATES and above_null >= MIN_DATES and stable >= MIN_DATES)
    return {
        "schema": "h1_m4_population_decoder_carrier_date_lodo_cpu_feasibility_v1",
        "protocol": PROTOCOL,
        "status": "PASS_CPU_H1_M4_POPULATION_DECODER_CARRIER_FEASIBILITY__NO_GPU_AUTHORIZATION" if gate["pass"] else "STOP_CPU_H1_M4_POPULATION_DECODER_CARRIER_FEASIBILITY_FAILED__NO_GPU_AUTHORIZATION",
        "scope": {"opened_data": "exactly 13 public held-in-calib NWBs", "formal_heldout_opened": False, "minival_or_query_opened": False, "gpu_constructed": False, "target_backpropagation": False},
        "source_hashes": {"audit_script_sha256": sha256_file(Path(__file__).resolve()), "protocol_sha256": sha256_file(PROTOCOL_PATH), "m2_population_audit_sha256": sha256_file(Path(m2.__file__).resolve())},
        "input_nwb_sha256": {name: record.input_sha256 for name, record in sorted(records.items())},
        "design": {"carrier": "M=4 population decoder-direction coefficient-row carrier; not an M=2 result or T4", "timebase": "trial-bounded 100-ms blocks", "target_support": "trials 1-4 only", "target_query": "trials 5+ only; at least three later trials required", "attachment": "independent fits trials1-2 vs trials3-4, raw coefficient-row median cosine", "rotation_null": "31 deterministic nonzero within-support-trial label rotations; query labels/rates unchanged", "source_selection": "outer-date-excluded source-only normalizer/PCA/q/ridge/4-D row basis", "future_comparator": "an equally M=4 neural-only SPINT baseline only"},
        "date_lodo": dates_out,
        "cpu_gate": gate,
        "decision": {"gpu_authorized_by_this_receipt": False, "if_fail": "STOP; do not tune this M=4 program or attach it to SPINT.", "m2_sealed_routes_unchanged": True},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "SPINT-main/data/000954")
    parser.add_argument("--output", type=Path, default=ROOT / "sua_exploration/results/h1_m4_population_decoder_carrier_date_lodo_v1/H1_M4_POPULATION_DECODER_CARRIER_CPU_RECEIPT.json")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise RuntimeError("CPU-only audit requires CUDA_VISIBLE_DEVICES unset")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {args.output}")
    paths = m2.h1.index_h1_heldin_calib(args.data_dir)
    output = run_audit({name: m2.load_full_h1_record(path) for name, path in paths.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output.chmod(0o444)
    print(json.dumps({"status": output["status"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
