#!/usr/bin/env python3
"""CPU-only H1 population-decoder-direction carrier feasibility audit.

This is a new, deliberately narrow H1 hypothesis.  It does *not* revive the
sealed AFC4/LFMC4 per-channel encoding routes.  For every outer date, source
recordings alone choose a neural normalizer, low-rank population PCA dimension
and ridge penalty.  A target decoder is then fit only on chronological trials
one and two and scored only on chronological trials three onward.  The fitted
decoder's per-channel 7-vector is represented in a source-only 4-D row basis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import audit_h1_afc4_m2_date_lodo as h1


PROTOCOL = "h1_population_decoder_carrier_m2_date_lodo_cpu_v1"
Q_GRID = (2, 4, 8, 16)
LAMBDA_GRID = (0.1, 1.0, 10.0, 100.0)
NULL_REPLICATES = 31
NULL_SEED = 20260807
MIN_DATES = 4
ATTACHMENT_STABILITY_THRESHOLD = 0.5
EPS = 1e-12


class GateError(ValueError):
    """Raised for a predeclared protocol condition that makes a score invalid."""


@dataclass(frozen=True)
class FullRecord:
    """All chronological H1 trials, retaining the authority's legal blocks."""

    session_name: str
    date: str
    path: Path | None
    input_sha256: str | None
    trials: tuple[h1.TrialBlocks, ...]

    @property
    def num_neurons(self) -> int:
        return int(self.trials[0].rates.shape[1])


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
    carrier_basis: np.ndarray  # (7, 4), source-only right singular directions
    source_grid_best_r2: float
    sha256: str


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _median_finite(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if finite.size == 0 else float(np.median(finite))


def _metric(y: np.ndarray, prediction: np.ndarray, support_mean: np.ndarray) -> dict[str, float] | None:
    """Return R2 against the fit-side mean and the components for weighting."""

    sse = float(np.square(y - prediction).sum())
    tss = float(np.square(y - support_mean[None, :]).sum())
    if not np.isfinite(sse) or not np.isfinite(tss) or tss <= EPS:
        return None
    return {"r2": float(1.0 - sse / tss), "sse": sse, "tss": tss}


def _variance_weighted(metrics: list[dict[str, float]]) -> dict[str, float] | None:
    if not metrics:
        return None
    sse = float(sum(item["sse"] for item in metrics))
    tss = float(sum(item["tss"] for item in metrics))
    if not np.isfinite(sse) or not np.isfinite(tss) or tss <= EPS:
        return None
    return {"r2": float(1.0 - sse / tss), "sse": sse, "tss": tss}


def _fit_ridge(z: np.ndarray, y: np.ndarray, ridge_lambda: float) -> np.ndarray:
    design = np.column_stack((np.ones(z.shape[0]), z))
    regularizer = np.eye(design.shape[1], dtype=np.float64) * ridge_lambda
    regularizer[0, 0] = 0.0  # never shrink the intercept
    return np.linalg.solve(design.T @ design + regularizer, design.T @ y)


def _predict(z: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(z.shape[0]), z)) @ beta


def _row_cosines(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    result = np.full(first.shape[0], np.nan, dtype=np.float64)
    valid = denominator > EPS
    result[valid] = (first[valid] * second[valid]).sum(axis=1) / denominator[valid]
    return result


def _support_and_query(record: FullRecord) -> tuple[h1.TrialBlocks, h1.TrialBlocks, tuple[h1.TrialBlocks, ...]]:
    if len(record.trials) < 3:
        raise GateError(f"{record.session_name}: need at least three chronological trials for M=2 support and later query")
    first, second = record.trials[:2]
    later = tuple(trial for trial in record.trials[2:] if trial.rates.shape[0] > 0)
    if first.rates.shape[0] < 2 or second.rates.shape[0] < 2 or not later:
        raise GateError(f"{record.session_name}: support/query has too few legal 100-ms blocks")
    return first, second, later


def _source_support_rates(record: FullRecord) -> np.ndarray:
    first, second, _ = _support_and_query(record)
    return np.concatenate((first.rates, second.rates), axis=0)


def _pca_rows(values: np.ndarray, maximum_q: int) -> np.ndarray:
    _, _, right_vectors = np.linalg.svd(values, full_matrices=False)
    if right_vectors.shape[0] < maximum_q:
        raise GateError(f"source PCA rank has only {right_vectors.shape[0]} components, need {maximum_q}")
    return right_vectors[:maximum_q]


def _zscore_and_project(values: np.ndarray, plan: Plan) -> np.ndarray:
    return ((values - plan.mean[None, :]) / plan.scale[None, :]) @ plan.pcs[: plan.q].T


def _raw_channel_coefficients(beta: np.ndarray, plan: Plan) -> np.ndarray:
    """Convert ridge feature weights from source PCA space to raw-channel rows."""

    return (plan.pcs[: plan.q].T @ beta[1:, :]) / plan.scale[:, None]


def _fit_target_decoder(
    record: FullRecord,
    plan: Plan,
    first_labels: np.ndarray | None = None,
    second_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first, second, _ = _support_and_query(record)
    y_first = first.velocity if first_labels is None else first_labels
    y_second = second.velocity if second_labels is None else second_labels
    support_rates = np.concatenate((first.rates, second.rates), axis=0)
    support_velocity = np.concatenate((y_first, y_second), axis=0)
    combined = _fit_ridge(_zscore_and_project(support_rates, plan), support_velocity, plan.ridge_lambda)
    first_only = _fit_ridge(_zscore_and_project(first.rates, plan), y_first, plan.ridge_lambda)
    second_only = _fit_ridge(_zscore_and_project(second.rates, plan), y_second, plan.ridge_lambda)
    return combined, first_only, second_only


def _score_later_trials(record: FullRecord, beta: np.ndarray, plan: Plan) -> dict[str, float] | None:
    first, second, later = _support_and_query(record)
    fit_side_mean = np.concatenate((first.velocity, second.velocity), axis=0).mean(axis=0)
    metrics: list[dict[str, float]] = []
    for trial in later:
        metrics.append(_metric(trial.velocity, _predict(_zscore_and_project(trial.rates, plan), beta), fit_side_mean))
    if any(metric is None for metric in metrics):
        return None
    return _variance_weighted([metric for metric in metrics if metric is not None])


def _rotation_offset(session_name: str, trial_number: float, length: int, replicate: int) -> int:
    if length < 2:
        raise GateError("label-rotation null needs at least two support blocks")
    token = f"{NULL_SEED}|{session_name}|{trial_number}|{replicate}".encode("utf-8")
    return 1 + int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % (length - 1)


def _source_record_r2(record: FullRecord, plan: Plan) -> float:
    combined, _, _ = _fit_target_decoder(record, plan)
    metric = _score_later_trials(record, combined, plan)
    if metric is None:
        raise GateError(f"{record.session_name}: source selection score is undefined")
    return metric["r2"]


def _build_plan(records: Mapping[str, FullRecord], outer_date: str) -> Plan:
    source_names = tuple(sorted(name for name, record in records.items() if record.date != outer_date))
    if not source_names or any(records[name].date == outer_date for name in source_names):
        raise GateError("outer date leaked into its source plan")
    source_rates = np.concatenate([_source_support_rates(records[name]) for name in source_names], axis=0)
    mean = source_rates.mean(axis=0)
    scale = np.maximum(source_rates.std(axis=0), 1e-6)
    pcs = _pca_rows((source_rates - mean[None, :]) / scale[None, :], max(Q_GRID))

    candidates: list[tuple[float, int, float]] = []
    for q in Q_GRID:
        for ridge_lambda in LAMBDA_GRID:
            temporary = Plan(outer_date, source_names, (), mean, scale, pcs, q, ridge_lambda, np.empty((7, 4)), 0.0, "")
            source_scores = [_source_record_r2(records[name], temporary) for name in source_names]
            candidates.append((float(np.mean(source_scores)), q, ridge_lambda))
    source_grid_best_r2, q, ridge_lambda = sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))[0]

    provisional = Plan(outer_date, source_names, (), mean, scale, pcs, q, ridge_lambda, np.empty((7, 4)), source_grid_best_r2, "")
    source_coefficient_rows: list[np.ndarray] = []
    for name in source_names:
        combined, _, _ = _fit_target_decoder(records[name], provisional)
        source_coefficient_rows.append(_raw_channel_coefficients(combined, provisional))
    _, _, right_vectors = np.linalg.svd(np.concatenate(source_coefficient_rows, axis=0), full_matrices=False)
    if right_vectors.shape[0] < 4:
        raise GateError("source decoder coefficient matrix cannot form a 4-D carrier basis")
    carrier_basis = right_vectors[:4].T
    source_hashes = tuple(records[name].input_sha256 for name in source_names)
    plan_body = {
        "protocol": PROTOCOL,
        "outer_date": outer_date,
        "source_sessions": source_names,
        "source_input_sha256": source_hashes,
        "normalizer_scope": "source M=2 support trials only; outer date excluded",
        "population_pca_scope": "source M=2 support trials only; outer date excluded",
        "ridge_grid_scope": "source M=2 support-to-later-trial scores only; outer date excluded",
        "carrier_basis_scope": "source fitted population decoder rows only; outer date excluded",
        "q_grid": Q_GRID,
        "lambda_grid": LAMBDA_GRID,
        "selected_q": q,
        "selected_lambda": ridge_lambda,
        "source_grid_best_equal_recording_r2": source_grid_best_r2,
    }
    return Plan(outer_date, source_names, source_hashes, mean, scale, pcs, q, ridge_lambda, carrier_basis, source_grid_best_r2, _json_sha256(plan_body))


def _record_result(record: FullRecord, plan: Plan) -> dict[str, Any]:
    first, second, later = _support_and_query(record)
    combined, first_only, second_only = _fit_target_decoder(record, plan)
    correct = _score_later_trials(record, combined, plan)
    stability = _median_finite(_row_cosines(_raw_channel_coefficients(first_only, plan), _raw_channel_coefficients(second_only, plan)))
    carrier = _raw_channel_coefficients(combined, plan) @ plan.carrier_basis
    null_metrics: list[dict[str, float] | None] = []
    for replicate in range(NULL_REPLICATES):
        rotated_first = np.roll(first.velocity, _rotation_offset(record.session_name, first.trial_number, first.velocity.shape[0], replicate), axis=0)
        rotated_second = np.roll(second.velocity, _rotation_offset(record.session_name, second.trial_number, second.velocity.shape[0], replicate), axis=0)
        null_combined, _, _ = _fit_target_decoder(record, plan, rotated_first, rotated_second)
        # Queries and their scoring baseline remain correctly paired and unrotated.
        null_metrics.append(_score_later_trials(record, null_combined, plan))
    defined = correct is not None and stability is not None and all(metric is not None for metric in null_metrics)
    null_r2 = [metric["r2"] for metric in null_metrics if metric is not None]
    return {
        "status": "defined" if defined else "undefined",
        "session_name": record.session_name,
        "support_trial_numbers": [first.trial_number, second.trial_number],
        "query_trial_numbers": [trial.trial_number for trial in later],
        "query_100ms_blocks": int(sum(trial.rates.shape[0] for trial in later)),
        "correct_held_trial_variance_weighted_r2_relative_fit_side_mean": None if correct is None else correct["r2"],
        "correct_sse": None if correct is None else correct["sse"],
        "correct_tss_fit_side_mean": None if correct is None else correct["tss"],
        "trial1_2_coefficient_row_cosine_median": stability,
        "carrier_shape": [int(carrier.shape[0]), int(carrier.shape[1])],
        "carrier_sha256": hashlib.sha256(np.ascontiguousarray(carrier).tobytes()).hexdigest(),
        "rotation_null_r2_q95": None if not defined else float(np.quantile(null_r2, 0.95)),
        "rotation_null_replicate_r2": null_r2,
        "rotation_null_sse": [None if metric is None else metric["sse"] for metric in null_metrics],
        "rotation_null_tss_fit_side_mean": [None if metric is None else metric["tss"] for metric in null_metrics],
    }


def _date_result(records: list[dict[str, Any]], plan: Plan) -> dict[str, Any]:
    defined = bool(records) and all(row["status"] == "defined" for row in records)
    correct = None
    null_r2: list[float] | None = None
    attachment = None
    if defined:
        correct = _variance_weighted([{"sse": row["correct_sse"], "tss": row["correct_tss_fit_side_mean"]} for row in records])
        null_r2 = []
        for replicate in range(NULL_REPLICATES):
            value = _variance_weighted([
                {"sse": row["rotation_null_sse"][replicate], "tss": row["rotation_null_tss_fit_side_mean"][replicate]}
                for row in records
            ])
            if value is None:
                defined = False
                null_r2 = None
                break
            null_r2.append(value["r2"])
        attachment = float(np.mean([row["trial1_2_coefficient_row_cosine_median"] for row in records]))
    q95 = None if null_r2 is None else float(np.quantile(null_r2, 0.95))
    return {
        "date": plan.outer_date,
        "status": "defined" if defined and correct is not None else "undefined",
        "plan": {
            "source_sessions": list(plan.source_sessions),
            "source_input_sha256": list(plan.source_input_sha256),
            "q": plan.q,
            "lambda": plan.ridge_lambda,
            "source_grid_best_equal_recording_r2": plan.source_grid_best_r2,
            "plan_sha256": plan.sha256,
        },
        "correct_held_trial_variance_weighted_r2_relative_fit_side_mean": None if correct is None else correct["r2"],
        "rotation_null_q95": q95,
        "correct_minus_rotation_null_q95": None if correct is None or q95 is None else correct["r2"] - q95,
        "coefficient_attachment_stability_mean_median_cosine": attachment,
        "records": records,
    }


def run_audit(records: Mapping[str, FullRecord]) -> dict[str, Any]:
    if set(records) != {record.session_name for record in records.values()}:
        raise GateError("record mapping key/session name mismatch")
    dates = tuple(sorted({record.date for record in records.values()}))
    if dates != tuple(h1.H1_DATES):
        raise GateError(f"H1 population audit requires the six authority dates {h1.H1_DATES}, got {dates}")
    rows: list[dict[str, Any]] = []
    for outer_date in h1.H1_DATES:
        try:
            plan = _build_plan(records, outer_date)
            target_rows = [_record_result(record, plan) for record in records.values() if record.date == outer_date]
            rows.append(_date_result(target_rows, plan))
        except Exception as exc:  # receipt must retain a structured STOP reason per date
            rows.append({"date": outer_date, "status": "undefined", "reason": str(exc)})
    positive = sum(row.get("correct_held_trial_variance_weighted_r2_relative_fit_side_mean", -np.inf) > 0 for row in rows)
    above_null = sum(row.get("correct_minus_rotation_null_q95", -np.inf) > 0 for row in rows)
    stable = sum(row.get("coefficient_attachment_stability_mean_median_cosine", -np.inf) >= ATTACHMENT_STABILITY_THRESHOLD for row in rows)
    gate = {
        "all_six_dates_defined": all(row["status"] == "defined" for row in rows),
        "correct_gain_positive_dates": positive,
        "correct_above_rotation_q95_dates": above_null,
        "attachment_stable_dates": stable,
        "minimum_required_dates": MIN_DATES,
        "attachment_stability_threshold": ATTACHMENT_STABILITY_THRESHOLD,
    }
    gate["pass"] = bool(gate["all_six_dates_defined"] and positive >= MIN_DATES and above_null >= MIN_DATES and stable >= MIN_DATES)
    return {
        "schema": "h1_population_decoder_carrier_m2_date_lodo_cpu_feasibility_v1",
        "protocol": PROTOCOL,
        "status": "PASS_CPU_H1_POPULATION_DECODER_CARRIER_FEASIBILITY__NO_GPU_AUTHORIZATION" if gate["pass"] else "STOP_CPU_H1_POPULATION_DECODER_CARRIER_FEASIBILITY_FAILED__NO_GPU_AUTHORIZATION",
        "scope": {
            "opened_data": "exactly 13 public held-in-calib NWBs",
            "formal_heldout_opened": False,
            "minival_or_query_opened": False,
            "gpu_constructed": False,
            "target_backpropagation": False,
        },
        "design": {
            "carrier": "population decoder-direction coefficient-row carrier; not T4, AFC4, or LFMC4",
            "timebase": "20-ms bins aggregated into trial-bounded 100-ms blocks by the signed H1 authority",
            "target_support": "only first two chronological trials; pooled ridge y<-X fit",
            "target_query": "only all chronological trials after the first two; never used for target fitting or source selection",
            "score": "held-trial variance-weighted R2 relative to the pooled fit-side target mean",
            "rotation_null": "31 deterministic within-support-trial label rotations; later query labels remain correct",
            "source_selection": "outer-date-excluded source-only normalizer, population PCA, ridge grid, and 4-D coefficient-row basis",
            "gate": "all six dates defined; >=4/6 correct R2>0 and correct>null q95; additionally >=4/6 attachment median cosine>=0.5",
        },
        "date_lodo": rows,
        "cpu_gate": gate,
        "decision": {
            "gpu_authorized_by_this_receipt": False,
            "if_fail": "STOP: do not tune this program, do not attach it to SPINT, and do not use a GPU.",
            "sealed_routes_unchanged": True,
        },
    }


def _all_trial_values(trial_num: np.ndarray, eval_mask: np.ndarray) -> tuple[float, ...]:
    labels = np.asarray(trial_num, dtype=np.float64).reshape(-1)
    valid = np.flatnonzero(np.asarray(eval_mask, dtype=bool).reshape(-1) & np.isfinite(labels))
    if valid.size == 0:
        raise GateError("TrialNum has no finite eval-valid bins")
    ordered = labels[valid]
    if np.any(np.diff(ordered) < 0.0):
        raise GateError("TrialNum is not chronological/nondecreasing on eval-valid bins")
    values: list[float] = []
    for value in ordered.tolist():
        if not values or float(value) != values[-1]:
            values.append(float(value))
    return tuple(values)


def load_full_h1_record(path: str | Path) -> FullRecord:
    """Load every chronological held-in H1 trial using the signed block constructor."""

    resolved = Path(path).resolve()
    h1._require_heldin_calib_path(resolved)
    try:
        from falcon_challenge.config import FalconTask
        from falcon_challenge.dataloaders import load_nwb
        from pynwb import NWBHDF5IO
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("H1 population audit needs falcon_challenge and pynwb installed") from exc
    neural, velocity, _trial_change, eval_mask = load_nwb(resolved, FalconTask.h1)
    with NWBHDF5IO(str(resolved), "r", load_namespaces=True) as io:
        nwb = io.read()
        if "TrialNum" not in nwb.acquisition:
            raise GateError(f"{resolved}: TrialNum acquisition is missing")
        trial_num = np.asarray(nwb.acquisition["TrialNum"].data[:], dtype=np.float64)
    spikes = h1._finite_matrix(neural, name="neural")
    kinematics = h1._finite_matrix(velocity, name="velocity")
    if spikes.shape[0] != kinematics.shape[0] or kinematics.shape[1] != h1.VELOCITY_DIM:
        raise GateError(f"neural/velocity shape mismatch: {spikes.shape}/{kinematics.shape}")
    values = _all_trial_values(trial_num, eval_mask)
    trials = tuple(h1._trial_to_blocks(trial_number=value, neural=spikes, velocity=kinematics, eval_mask=np.asarray(eval_mask, dtype=bool), trial_num=trial_num) for value in values)
    record = FullRecord(h1._session_name_from_path(resolved), h1.date_from_session(h1._session_name_from_path(resolved)), resolved, h1.sha256_file(resolved), trials)
    if record.num_neurons != h1.EXPECTED_NEURONS:
        raise GateError(f"{record.session_name}: expected {h1.EXPECTED_NEURONS} H1 channels, got {record.num_neurons}")
    _support_and_query(record)  # fail closed before any outer-date fitting
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--data-dir", type=Path, default=root / "SPINT-main/data/000954")
    parser.add_argument("--output", type=Path, default=root / "sua_exploration/results/h1_population_decoder_carrier_m2_date_lodo_v1/H1_POPULATION_DECODER_CARRIER_CPU_RECEIPT.json")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise RuntimeError("CPU-only H1 population audit requires CUDA_VISIBLE_DEVICES to be unset")
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
