#!/usr/bin/env python3
"""Protocol-corrected H1 Ridge50 v2r2 replay and verifier.

This is an immutable replay of the v2 numerical result with three evidence
corrections:

* the query SHA is computed from the windows actually scored;
* fold-0 M4 H-SE5 label accounting is read from and bound to its immutable
  source receipt, rather than copied from the all-recording M3 summary; and
* the scored arrays are saved in an immutable bundle and checked by a second
  from-source replay.

The fixed normalized lambda remains 1.0.  This is a descriptive deployment-
boundary comparator, not an optimized, information-matched, or architecture-
matched H1 baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "sua_exploration", REPO_ROOT / "SPINT-main"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from sua_exploration.mc_maze import priority_a2_normalized_ridge_v2 as ridge
from sua_exploration.mc_maze.priority_a2_normalized_ridge_v2 import FEATURE_STD_EPS


WINDOW = 700
HISTORY_BINS = 50
VELOCITY_DIM = 7
EXPECTED_NEURONS = 176
NORMALIZED_LAMBDA = 1.0
SUPPORT_TRIALS = 4
EXPECTED_QUERY_SHA256 = "665fe535e90a221123b778171685577f67c2cc1902648cde2c8b2523e990e4da"
EXPECTED_HSE5_ACCOUNTING_SHA256 = "de4c23ac3fd21f96c68e54b5190538189543be6f7b19e21cb5533665872a28a4"

DATA_DIR = REPO_ROOT / "SPINT-main/data/000954"
OUTPUT_DIR = REPO_ROOT / "sua_exploration/results/h1_ridge_baseline_v2r2"
RECEIPT_PATH = OUTPUT_DIR / "h1_ridge_baseline_v2r2_receipt.json"
ARRAY_PATH = OUTPUT_DIR / "h1_ridge_baseline_v2r2_arrays.npz"
VERIFICATION_PATH = OUTPUT_DIR / "h1_ridge_baseline_v2r2_verification.json"

V1_RECEIPT = REPO_ROOT / "sua_exploration/results/h1_ridge_baseline_v1/h1_ridge_baseline_receipt.json"
V2_RECEIPT = REPO_ROOT / "sua_exploration/results/h1_ridge_baseline_v2/h1_ridge_baseline_v2_receipt.json"
HSE5_ACCOUNTING_RECEIPT = REPO_ROOT / "sua_exploration/results/h1_sparse_event_endpoint_v2/source_audit_v2r2.json"
HSE5_TERMINAL_RECEIPT = REPO_ROOT / "SPINT-main/pilot_artifacts/h1_sparse_event_endpoint/H1_SE5_M4_FOLD0_TERMINAL_v1.json"
HSE5_EVALUATOR = REPO_ROOT / "SPINT-main/scripts/h1_sparse_event_endpoint_evaluate.py"
HSE5_PROGRAM_COMPLETION = REPO_ROOT / "sua_exploration/results/h1_sparse_event_endpoint_program/H1_SPARSE_EVENT_ENDPOINT_PROGRAM_COMPLETION_v1.json"
LOADER = REPO_ROOT / "SPINT-main/src/data/h1_m4_eb_pilot.py"
RIDGE_CORE = REPO_ROOT / "sua_exploration/mc_maze/priority_a2_normalized_ridge_v2.py"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def window_manifest_sha256(rows: list[tuple[str, int]]) -> str:
    digest = hashlib.sha256()
    for session_name, start in rows:
        digest.update(session_name.encode("ascii"))
        digest.update(np.int64(start).tobytes())
    return digest.hexdigest()


def write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    os.chmod(path, 0o444)


def pooled_r2(truth: np.ndarray, estimate: np.ndarray) -> float:
    truth64 = np.asarray(truth, dtype=np.float64)
    estimate64 = np.asarray(estimate, dtype=np.float64)
    sse = float(np.square(truth64 - estimate64).sum())
    centered = truth64 - truth64.mean(axis=0, keepdims=True)
    tss = float(np.square(centered).sum())
    require(tss > 0.0, "R2 denominator is not positive")
    return 1.0 - sse / tss


def fit_normalized_ridge(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray | float]:
    weights = np.ones(x.shape[0], dtype=np.float64)
    total = float(weights.sum())
    x_mean = (weights[:, None] * x).sum(axis=0) / total
    y_mean = (weights[:, None] * y).sum(axis=0) / total
    centered = x - x_mean[None, :]
    variance = (weights[:, None] * centered * centered).sum(axis=0) / total
    scale = np.sqrt(variance)
    scale[scale < FEATURE_STD_EPS] = 1.0
    standardized = centered / scale[None, :]
    centered_y = y - y_mean[None, :]
    root_weight = np.sqrt(weights / total)
    design = standardized * root_weight[:, None]
    target = centered_y * root_weight[:, None]
    require(design.shape[0] < design.shape[1], "v2r2 expects the fixed H1 dual ridge branch")
    gram = design @ design.T
    gram.flat[:: gram.shape[0] + 1] += NORMALIZED_LAMBDA
    coefficient = design.T @ np.linalg.solve(gram, target)
    prediction = standardized @ coefficient + y_mean[None, :]
    intercept_foc = (weights[:, None] * (y - prediction)).sum(axis=0) / total
    foc_max = float(np.abs(intercept_foc).max())
    require(foc_max <= 1.0e-8, f"seven-output intercept FOC failed: {foc_max}")
    return {
        "x_mean": x_mean,
        "scale": scale,
        "coefficient": coefficient,
        "intercept": y_mean,
        "intercept_foc_max_abs_error": foc_max,
    }


def load_hse5_accounting(target_names: tuple[str, ...], records: dict[str, Any]) -> dict[str, Any]:
    require(HSE5_ACCOUNTING_RECEIPT.exists(), "H-SE5 accounting receipt is missing")
    require((HSE5_ACCOUNTING_RECEIPT.stat().st_mode & 0o777) == 0o444, "H-SE5 accounting receipt is not immutable")
    actual_sha = sha256_file(HSE5_ACCOUNTING_RECEIPT)
    require(actual_sha == EXPECTED_HSE5_ACCOUNTING_SHA256, "H-SE5 accounting receipt SHA drift")
    source = json.loads(HSE5_ACCOUNTING_RECEIPT.read_text(encoding="utf-8"))
    source_sessions = source["budgets"]["M4"]["sessions"]
    per_session: dict[str, Any] = {}
    aggregate = {
        "support_events": 0,
        "acquisition_endpoint_position_coordinates": 0,
        "derived_displacement_coordinates": 0,
        "projected_q4_model_input_coordinates": 0,
    }
    for name in target_names:
        row = source_sessions[name]
        require(row["session"] == name, f"{name}: H-SE5 accounting session drift")
        require(row["budget"] == SUPPORT_TRIALS, f"{name}: H-SE5 accounting is not M4")
        require(row["input_sha256"] == records[name].input_sha256, f"{name}: H-SE5/NWB binding drift")
        labels = row["label_accounting"]
        item = {
            "support_events": int(row["support_events"]),
            "acquisition_endpoint_position_coordinates": int(labels["acquisition_endpoint_position_scalars"]),
            "derived_displacement_coordinates": int(labels["derived_displacement_scalars"]),
            "projected_q4_model_input_coordinates": int(labels["projected_model_input_scalars"]),
            "dense_velocity_not_read_by_hse5_carrier": bool(row["position_source"]["dense_velocity_series_opened"] is False),
            "input_sha256": row["input_sha256"],
        }
        require(item["acquisition_endpoint_position_coordinates"] == item["support_events"] * 2 * VELOCITY_DIM, f"{name}: acquisition accounting mismatch")
        require(item["derived_displacement_coordinates"] == item["support_events"] * VELOCITY_DIM, f"{name}: displacement accounting mismatch")
        require(item["projected_q4_model_input_coordinates"] == item["support_events"] * 4, f"{name}: q4 accounting mismatch")
        per_session[name] = item
        for key in aggregate:
            aggregate[key] += int(item[key])
    require(aggregate == {
        "support_events": 41,
        "acquisition_endpoint_position_coordinates": 574,
        "derived_displacement_coordinates": 287,
        "projected_q4_model_input_coordinates": 164,
    }, "fold-0 M4 H-SE5 accounting drift")
    return {
        "scope": "fold-0 public held-in development; two target recordings; chronological first four trials",
        "per_session": per_session,
        "aggregate": aggregate,
        "source_receipt": str(HSE5_ACCOUNTING_RECEIPT),
        "source_receipt_sha256": actual_sha,
        "source_receipt_mode": "0444",
        "counting_boundary": (
            "Algorithmic coordinate accounting, not independent-sample count, effective sample size, "
            "human-annotation cost, or an information-matched comparison."
        ),
    }


def compute_replay() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from src.data.h1_m4_eb_pilot import H1_M4_FOLD0_TARGET, WINDOW as PILOT_WINDOW, load_target_records

    require(PILOT_WINDOW == WINDOW, "H1 query-window length drift")
    records = load_target_records(DATA_DIR)
    target_names = tuple(H1_M4_FOLD0_TARGET)
    require(set(records) == set(target_names), "fold-0 target session set drift")
    hse5_accounting = load_hse5_accounting(target_names, records)

    per_session: dict[str, Any] = {}
    saved_arrays: dict[str, np.ndarray] = {}
    all_truth: list[np.ndarray] = []
    all_prediction: list[np.ndarray] = []
    actual_scored_windows: list[tuple[str, int]] = []
    ridge_aggregate_rows = 0

    for session_index, name in enumerate(target_names):
        record = records[name]
        neural = np.asarray(record.neural, dtype=np.float64)
        velocity = np.asarray(record.velocity, dtype=np.float64)
        trial_num = np.asarray(record.trial_num)
        eval_mask = np.asarray(record.eval_mask, dtype=bool)
        support_values = tuple(float(value) for value in record.trial_values[:SUPPORT_TRIALS])
        support_set = set(support_values)

        calibration_targets: list[int] = []
        calibration_trial_ids: list[float] = []
        for target_bin in range(HISTORY_BINS - 1, neural.shape[0]):
            if not eval_mask[target_bin]:
                continue
            start = target_bin - HISTORY_BINS + 1
            history_trials = trial_num[start : target_bin + 1]
            if not np.isfinite(history_trials).all():
                continue
            unique = np.unique(history_trials)
            if unique.size != 1 or float(unique[0]) not in support_set:
                continue
            calibration_targets.append(target_bin)
            calibration_trial_ids.append(float(unique[0]))
        calibration_targets_array = np.asarray(calibration_targets, dtype=np.int64)
        calibration_starts = calibration_targets_array - HISTORY_BINS + 1
        require(calibration_targets_array.size > 0, f"{name}: no legal calibration rows")
        feature = np.stack(
            [neural[target - HISTORY_BINS + 1 : target + 1].reshape(-1) for target in calibration_targets_array]
        )
        target = velocity[calibration_targets_array]
        fit = fit_normalized_ridge(feature, target)

        fifth_trial = float(record.trial_values[SUPPORT_TRIALS])
        fifth_bins = np.flatnonzero(eval_mask & np.isfinite(trial_num) & (trial_num == fifth_trial))
        require(fifth_bins.size > 0, f"{name}: fifth trial has no eval-valid bin")
        support_boundary = int(fifth_bins[0])
        query_starts: list[int] = []
        query_outputs: list[int] = []
        for start in range(support_boundary, neural.shape[0] - WINDOW + 1):
            output = start + WINDOW - 1
            if not eval_mask[output]:
                continue
            history_start = output - HISTORY_BINS + 1
            require(history_start >= support_boundary, f"{name}: query Ridge50 history overlaps support")
            query_starts.append(start)
            query_outputs.append(output)
            actual_scored_windows.append((name, start))
        query_starts_array = np.asarray(query_starts, dtype=np.int64)
        query_outputs_array = np.asarray(query_outputs, dtype=np.int64)
        require(query_starts_array.size > 0, f"{name}: no legal query rows")
        query_feature = np.stack(
            [neural[output - HISTORY_BINS + 1 : output + 1].reshape(-1) for output in query_outputs_array]
        )
        query_truth = velocity[query_outputs_array]
        query_prediction = (
            (query_feature - np.asarray(fit["x_mean"])[None, :]) / np.asarray(fit["scale"])[None, :]
        ) @ np.asarray(fit["coefficient"]) + np.asarray(fit["intercept"])[None, :]

        session_r2 = pooled_r2(query_truth, query_prediction)
        unique_trials, trial_counts = np.unique(np.asarray(calibration_trial_ids), return_counts=True)
        calibration_by_trial = {str(float(value)): int(count) for value, count in zip(unique_trials, trial_counts)}
        prefix = f"session_{session_index}"
        saved_arrays.update({
            f"{prefix}_calibration_starts": calibration_starts,
            f"{prefix}_calibration_target_bins": calibration_targets_array,
            f"{prefix}_calibration_truth": target,
            f"{prefix}_query_starts": query_starts_array,
            f"{prefix}_query_output_bins": query_outputs_array,
            f"{prefix}_query_truth": query_truth,
            f"{prefix}_query_prediction": query_prediction,
            f"{prefix}_coefficient": np.asarray(fit["coefficient"]),
            f"{prefix}_feature_mean": np.asarray(fit["x_mean"]),
            f"{prefix}_feature_scale": np.asarray(fit["scale"]),
            f"{prefix}_intercept": np.asarray(fit["intercept"]),
        })
        per_session[name] = {
            "r2": session_r2,
            "calibration_rows": int(calibration_targets_array.size),
            "calibration_velocity_coordinates": int(calibration_targets_array.size * VELOCITY_DIM),
            "calibration_by_trial": calibration_by_trial,
            "query_windows": int(query_starts_array.size),
            "support_trials": list(support_values),
            "fifth_trial": fifth_trial,
            "support_boundary_bin": support_boundary,
            "calibration_starts_sha256": sha256_array(calibration_starts),
            "calibration_target_bins_sha256": sha256_array(calibration_targets_array),
            "calibration_truth_sha256": sha256_array(target),
            "query_starts_sha256": sha256_array(query_starts_array),
            "query_output_bins_sha256": sha256_array(query_outputs_array),
            "query_truth_sha256": sha256_array(query_truth),
            "query_prediction_sha256": sha256_array(query_prediction),
            "coefficient_sha256": sha256_array(np.asarray(fit["coefficient"])),
            "feature_mean_sha256": sha256_array(np.asarray(fit["x_mean"])),
            "feature_scale_sha256": sha256_array(np.asarray(fit["scale"])),
            "intercept_sha256": sha256_array(np.asarray(fit["intercept"])),
            "seven_output_intercept_foc_max_abs_error": float(fit["intercept_foc_max_abs_error"]),
        }
        ridge_aggregate_rows += int(calibration_targets_array.size)
        all_truth.append(query_truth)
        all_prediction.append(query_prediction)

    actual_query_sha = window_manifest_sha256(actual_scored_windows)
    require(actual_query_sha == EXPECTED_QUERY_SHA256, "actual scored-query manifest SHA drift")
    pooled_truth = np.concatenate(all_truth, axis=0)
    pooled_prediction = np.concatenate(all_prediction, axis=0)
    saved_arrays["pooled_query_truth"] = pooled_truth
    saved_arrays["pooled_query_prediction"] = pooled_prediction
    require(ridge_aggregate_rows == 6088, "Ridge v2r2 calibration-row count drift")

    required_bindings = {
        "runner_sha256": sha256_file(Path(__file__)),
        "ridge_core_sha256": sha256_file(RIDGE_CORE),
        "loader_sha256": sha256_file(LOADER),
        "hse5_accounting_receipt_sha256": sha256_file(HSE5_ACCOUNTING_RECEIPT),
        "hse5_terminal_receipt_sha256": sha256_file(HSE5_TERMINAL_RECEIPT),
        "hse5_evaluator_sha256": sha256_file(HSE5_EVALUATOR),
        "hse5_program_completion_sha256": sha256_file(HSE5_PROGRAM_COMPLETION),
        "v1_receipt_sha256": sha256_file(V1_RECEIPT),
        "v2_receipt_sha256": sha256_file(V2_RECEIPT),
        "nwb_sha256": {name: records[name].input_sha256 for name in target_names},
    }
    evidence = {
        "schema": "h1_ridge_baseline_v2r2",
        "status": "COMPLETED_PROTOCOL_CORRECTED_IMMUTABLE_REPLAY",
        "date": "2026-08-12",
        "definition": (
            "Fold-0 two-recording H1 development comparator. A separate per-session linear ridge "
            "readout maps a same-trial causal 50-bin x 176-channel spike history to dense 7-DoF "
            "velocity, using the chronological first four trials, calibration-only standardization, "
            "an unpenalized intercept, and fixed normalized lambda=1. Query histories and scored "
            "windows are strictly post-support."
        ),
        "role": "descriptive deployment-boundary reference",
        "claim_boundary": [
            "not information-matched",
            "not architecture-matched",
            "not an optimized H1 ridge baseline",
            "not a causal explanation of the score gap",
            "not a carrier-superiority claim",
            "not H1 sparse-label evidence",
            "not a ridge-family upper bound",
        ],
        "supersedes": {
            "v1": "calibration/query feature-window mismatch",
            "v2": "scientific value retained; v2r2 corrects actual-query and fold0-M4 accounting provenance",
        },
        "pooled_r2": pooled_r2(pooled_truth, pooled_prediction),
        "per_session": per_session,
        "query_contract": {
            "actual_scored_window_count": len(actual_scored_windows),
            "actual_scored_window_manifest_sha256": actual_query_sha,
            "expected_hse5_query_manifest_sha256": EXPECTED_QUERY_SHA256,
            "exact_match": True,
            "scoring": "float64 pooled multi-output SSE/TSS at the 700-bin decoder-window last bin",
        },
        "ridge_config": {
            "history_bins": HISTORY_BINS,
            "feature_window": "[t-49:t+1] including the target-time neural bin",
            "feature_dim": HISTORY_BINS * EXPECTED_NEURONS,
            "velocity_dim": VELOCITY_DIM,
            "normalized_lambda": NORMALIZED_LAMBDA,
            "lambda_selection": "fixed canonical comparator; no H1 target or source selection",
            "support_trials": SUPPORT_TRIALS,
            "calibration_history": "entirely within one support trial",
            "query_history": "entirely post-support",
            "target_session_optimizer_steps": 0,
            "target_session_backward_steps": 0,
        },
        "target_supervision_accounting": {
            "ridge_fold0_m4": {
                "calibration_rows": ridge_aggregate_rows,
                "dense_7d_velocity_coordinates": ridge_aggregate_rows * VELOCITY_DIM,
                "per_session": {
                    name: {
                        "calibration_rows": per_session[name]["calibration_rows"],
                        "dense_7d_velocity_coordinates": per_session[name]["calibration_velocity_coordinates"],
                    }
                    for name in target_names
                },
            },
            "hse5_fold0_m4": hse5_accounting,
            "interpretation": (
                "Counts describe different algorithmic target representations. They are not independent "
                "sample counts, effective sample sizes, human-annotation costs, or equal-information arms."
            ),
        },
        "numerical_contract": {
            "canonical_two_output_core_self_test": ridge.numerical_contract_self_test(),
            "seven_output_intercept_foc_max_abs_error": max(
                row["seven_output_intercept_foc_max_abs_error"] for row in per_session.values()
            ),
        },
        "input_bindings": required_bindings,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "byteorder": sys.byteorder,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "thread_environment": {
                key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            },
            "nice": os.getpriority(os.PRIO_PROCESS, 0),
        },
    }
    return evidence, saved_arrays


def create_array_bundle(saved_arrays: dict[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(stream, **saved_arrays)
    return stream.getvalue()


def run_mode() -> None:
    require(not RECEIPT_PATH.exists(), f"refusing to overwrite {RECEIPT_PATH}")
    require(not ARRAY_PATH.exists(), f"refusing to overwrite {ARRAY_PATH}")
    require(not VERIFICATION_PATH.exists(), f"verification already exists at {VERIFICATION_PATH}")
    started = time.monotonic()
    evidence, saved_arrays = compute_replay()
    array_bytes = create_array_bundle(saved_arrays)
    write_immutable(ARRAY_PATH, array_bytes)
    evidence["array_bundle"] = {
        "path": str(ARRAY_PATH),
        "sha256": sha256_bytes(array_bytes),
        "mode": "0444",
        "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": sha256_array(value)} for name, value in saved_arrays.items()},
    }
    evidence["elapsed_seconds"] = time.monotonic() - started
    receipt_bytes = canonical_json_bytes(evidence)
    write_immutable(RECEIPT_PATH, receipt_bytes)
    print(json.dumps({
        "status": evidence["status"],
        "pooled_r2": evidence["pooled_r2"],
        "per_session": {name: row["r2"] for name, row in evidence["per_session"].items()},
        "receipt": str(RECEIPT_PATH),
        "receipt_sha256": sha256_bytes(receipt_bytes),
        "array_bundle_sha256": sha256_bytes(array_bytes),
    }, indent=2, sort_keys=True))


def verify_mode() -> None:
    require(RECEIPT_PATH.exists() and ARRAY_PATH.exists(), "v2r2 receipt/array bundle is missing")
    require(not VERIFICATION_PATH.exists(), f"refusing to overwrite {VERIFICATION_PATH}")
    require((RECEIPT_PATH.stat().st_mode & 0o777) == 0o444, "v2r2 receipt is not immutable")
    require((ARRAY_PATH.stat().st_mode & 0o777) == 0o444, "v2r2 array bundle is not immutable")
    stored = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    recomputed, arrays = compute_replay()
    compared_fields = (
        "schema", "status", "definition", "role", "claim_boundary", "supersedes",
        "pooled_r2", "per_session", "query_contract", "ridge_config",
        "target_supervision_accounting", "numerical_contract", "input_bindings",
    )
    for field in compared_fields:
        require(stored[field] == recomputed[field], f"v2r2 replay mismatch in {field}")
    require(stored["array_bundle"]["sha256"] == sha256_file(ARRAY_PATH), "array-bundle SHA drift")
    with np.load(ARRAY_PATH, allow_pickle=False) as saved:
        require(set(saved.files) == set(arrays), "array-bundle key drift")
        for name, expected in arrays.items():
            actual = saved[name]
            require(actual.dtype == expected.dtype and actual.shape == expected.shape, f"{name}: dtype/shape drift")
            require(np.array_equal(actual, expected), f"{name}: replay is not byte-identical")
            require(stored["array_bundle"]["arrays"][name]["sha256"] == sha256_array(actual), f"{name}: stored SHA drift")
    verification = {
        "schema": "h1_ridge_baseline_v2r2_verification_v1",
        "status": "PASS_FROM_SOURCE_BYTE_IDENTICAL_REPLAY",
        "receipt": str(RECEIPT_PATH),
        "receipt_sha256": sha256_file(RECEIPT_PATH),
        "array_bundle": str(ARRAY_PATH),
        "array_bundle_sha256": sha256_file(ARRAY_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "compared_fields": list(compared_fields),
        "array_count": len(arrays),
        "pooled_r2": recomputed["pooled_r2"],
        "per_session_r2": {name: row["r2"] for name, row in recomputed["per_session"].items()},
        "actual_scored_query_manifest_sha256": recomputed["query_contract"]["actual_scored_window_manifest_sha256"],
        "no_gpu": True,
    }
    verification_bytes = canonical_json_bytes(verification)
    write_immutable(VERIFICATION_PATH, verification_bytes)
    print(json.dumps({
        "status": verification["status"],
        "verification": str(VERIFICATION_PATH),
        "verification_sha256": sha256_bytes(verification_bytes),
        "receipt_sha256": verification["receipt_sha256"],
    }, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "verify"))
    args = parser.parse_args()
    if args.mode == "run":
        run_mode()
    else:
        verify_mode()


if __name__ == "__main__":
    main()
