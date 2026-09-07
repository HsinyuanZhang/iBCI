#!/usr/bin/env python3
"""H1 per-session ridge baseline comparator (v2, protocol-corrected).

Fixes from v1:
  - Calibration feature is [t-49:t+1] (includes current bin), matching query.
  - Calibration history must be entirely within the same support trial.
  - Query history must be entirely within the post-support scope.
  - Query SHA mismatch fails closed (hard assert).
  - Dead code removed (build_calib_features with undefined neural_vel).
  - Records calibration/query starts, truth, prediction, coefficient/state hashes.
  - Binds loader, ridge core, and H-SE5 terminal/evaluator receipts.
  - Correctly records H-SE5 three-layer label accounting.
  - Lambda=1.0 is explicitly a fixed canonical comparator.
  - New v2 directory and receipt; v1 marked superseded.

CPU-only, no GPU.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "sua_exploration", REPO_ROOT / "SPINT-main"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sua_exploration.mc_maze import priority_a2_normalized_ridge_v2 as ridge
from sua_exploration.mc_maze.priority_a2_normalized_ridge_v2 import FEATURE_STD_EPS

WINDOW = 700
HISTORY_BINS = 50
VELOCITY_DIM = 7
EXPECTED_NEURONS = 176
LAMBDA = 1.0
SUPPORT_TRIALS = 4
EXPECTED_QUERY_SHA = "665fe535e90a221123b778171685577f67c2cc1902648cde2c8b2523e990e4da"
DATA_DIR = REPO_ROOT / "SPINT-main/data/000954"
OUTPUT_DIR = REPO_ROOT / "sua_exploration/results/h1_ridge_baseline_v2"

# H-SE5 terminal receipt and evaluator SHA bindings
HSE5_TERMINAL_RECEIPT = REPO_ROOT / "SPINT-main/pilot_artifacts/h1_sparse_event_endpoint/H1_SE5_M4_FOLD0_TERMINAL_v1.json"
HSE5_EVALUATOR = REPO_ROOT / "SPINT-main/scripts/h1_sparse_event_endpoint_evaluate.py"
HSE5_PROGRAM_COMPLETION = REPO_ROOT / "sua_exploration/results/h1_sparse_event_endpoint_program/H1_SPARSE_EVENT_ENDPOINT_PROGRAM_COMPLETION_v1.json"

# H-SE5 three-layer label accounting (from v2r2 receipt)
# Layer 1: estimator consumes ~796 projected V2 model-input scalars (events × 4)
# Layer 2: semantic payload is ~1393 derived displacement scalars (events × 7)
# Layer 3: acquisition-level is ~2786 endpoint-position scalars (2 endpoints × 7 × events)
# Dense velocity reference: ~216153 raw per-bin velocity scalars
HSE5_LABEL_ACCOUNTING = {
    "estimator_input_scalars": "events × 4 projected displacement components",
    "semantic_payload_scalars": "events × 7 derived endpoint displacement coordinates",
    "acquisition_position_scalars": "2 × 7 × events raw endpoint positions read from NWB",
    "dense_velocity_reference_scalars": "~216153 per-bin 7-DoF velocity coordinates",
    "dense_velocity_not_read_by_carrier": True,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes(order="C")).hexdigest()


def _window_manifest_hash(window_indices: list[tuple[str, int]]) -> str:
    digest = hashlib.sha256()
    for name, start in window_indices:
        digest.update(name.encode("ascii"))
        digest.update(np.int64(start).tobytes())
    return digest.hexdigest()


def _r2(truth: np.ndarray, estimate: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    sse = float(np.sum((truth - estimate) ** 2))
    tss = float(np.sum((truth - truth.mean(axis=0, keepdims=True)) ** 2))
    assert tss > 0, "zero total variance"
    return 1.0 - sse / tss


def _trial_of_bin(trial_num: np.ndarray, eval_mask: np.ndarray, t: int,
                  support_values: set, history_bins: int) -> int | None:
    """Return the trial index if bins [t-history_bins+1 : t+1] are all in the same support trial, else None."""
    start = t - history_bins + 1
    if start < 0:
        return None
    tvs = trial_num[start:t + 1]
    if not np.isfinite(tvs).all():
        return None
    unique = set(np.unique(tvs).tolist())
    if len(unique) != 1:
        return None
    tv = unique.pop()
    if tv not in support_values:
        return None
    return int(tv)


def _query_trial_safe(trial_num: np.ndarray, t: int, history_bins: int,
                      support_values: set) -> bool:
    """Check that bins [t-history_bins+1 : t+1] do not overlap any support trial."""
    start = t - history_bins + 1
    if start < 0:
        return False
    tvs = trial_num[start:t + 1]
    if not np.isfinite(tvs).all():
        return True  # NaN trial_num means post-recording padding; allow
    for tv in np.unique(tvs):
        if tv in support_values:
            return False
    return True


def _fit_normalized_ridge(x, y, w, normalized_lambda):
    """Normalized weighted ridge with unpenalized intercept. Supports any output dim."""
    total = w.sum()
    xbar = (w[:, None] * x).sum(axis=0) / total
    ybar = (w[:, None] * y).sum(axis=0) / total
    centered = x - xbar[None, :]
    variance = (w[:, None] * centered * centered).sum(axis=0) / total
    scale = np.sqrt(variance)
    scale[scale < FEATURE_STD_EPS] = 1.0
    z = centered / scale[None, :]
    yc = y - ybar[None, :]
    sqrt_w = np.sqrt(w / total)
    a = z * sqrt_w[:, None]
    t_weighted = yc * sqrt_w[:, None]
    if a.shape[0] < a.shape[1]:
        dual_gram = a @ a.T
        dual_gram.flat[::dual_gram.shape[0] + 1] += normalized_lambda
        coefficients = a.T @ np.linalg.solve(dual_gram, t_weighted)
    else:
        primal_gram = a.T @ a
        primal_gram.flat[::primal_gram.shape[0] + 1] += normalized_lambda
        coefficients = np.linalg.solve(primal_gram, a.T @ t_weighted)
    intercept = ybar
    # Verify FOC
    pred_calib = ((x - xbar[None, :]) / scale[None, :]) @ coefficients + intercept
    resid_foc = (w[:, None] * (y - pred_calib)).sum(axis=0) / total
    assert np.allclose(resid_foc, 0.0, atol=1e-8), f"intercept FOC failed: {resid_foc}"
    return xbar, scale, coefficients, intercept


def main():
    t0 = time.monotonic()
    numerical_contract = ridge.numerical_contract_self_test()
    print(f"[{time.strftime('%H:%M:%S')}] Numerical contract: {numerical_contract}", file=sys.stderr)

    from src.data.h1_m4_eb_pilot import (
        load_target_records, load_source_records, H1_M4_FOLD0_TARGET,
        WINDOW as PILOT_WINDOW, index_heldin_calib,
    )
    assert PILOT_WINDOW == WINDOW, f"window mismatch: {PILOT_WINDOW} vs {WINDOW}"

    records = load_target_records(DATA_DIR)
    assert set(records) == set(H1_M4_FOLD0_TARGET), "target session set mismatch"

    # Bind H-SE5 receipts
    hse5_terminal_sha = sha256_file(HSE5_TERMINAL_RECEIPT) if HSE5_TERMINAL_RECEIPT.exists() else "MISSING"
    hse5_evaluator_sha = sha256_file(HSE5_EVALUATOR) if HSE5_EVALUATOR.exists() else "MISSING"
    hse5_program_sha = sha256_file(HSE5_PROGRAM_COMPLETION) if HSE5_PROGRAM_COMPLETION.exists() else "MISSING"
    loader_sha = sha256_file(REPO_ROOT / "SPINT-main/src/data/h1_m4_eb_pilot.py")
    ridge_core_sha = sha256_file(REPO_ROOT / "sua_exploration/mc_maze/priority_a2_normalized_ridge_v2.py")

    per_session_results = {}
    all_truth = []
    all_estimate = []

    for name in H1_M4_FOLD0_TARGET:
        record = records[name]
        print(f"[{time.strftime('%H:%M:%S')}] Processing {name}", file=sys.stderr)

        trial_values = record.trial_values
        support_values = set(float(v) for v in trial_values[:SUPPORT_TRIALS])
        n_total = record.neural.shape[0]
        trial_num = record.trial_num
        eval_mask = record.eval_mask
        neural64 = record.neural.astype(np.float64)
        velocity64 = record.velocity.astype(np.float64)

        # --- Calibration: [t-49:t+1] → velocity[t], history within same support trial ---
        calib_indices = []
        calib_trial_ids = []
        for t in range(HISTORY_BINS - 1, n_total):
            if not eval_mask[t]:
                continue
            tv = _trial_of_bin(trial_num, eval_mask, t, support_values, HISTORY_BINS)
            if tv is None:
                continue
            calib_indices.append(t)
            calib_trial_ids.append(tv)

        ridge.require(len(calib_indices) > 0, f"{name}: no calibration bins")
        calib_indices = np.array(calib_indices, dtype=np.int64)
        calib_trial_ids = np.array(calib_trial_ids, dtype=np.int64)
        calib_starts = calib_indices - HISTORY_BINS + 1  # first bin of each history window

        n_calib = len(calib_indices)
        calib_features = np.empty((n_calib, HISTORY_BINS * EXPECTED_NEURONS), dtype=np.float64)
        calib_targets = np.empty((n_calib, VELOCITY_DIM), dtype=np.float64)
        for i, t in enumerate(calib_indices):
            calib_features[i] = neural64[t - HISTORY_BINS + 1:t + 1].flatten()
            calib_targets[i] = velocity64[t]

        print(f"  Calibration: {n_calib} bins, {calib_features.shape[1]} features", file=sys.stderr)

        # --- Fit normalized ridge (λ=1.0, fixed canonical comparator) ---
        weights = np.ones(n_calib, dtype=np.float64)
        xbar, scale, coefficients, intercept = _fit_normalized_ridge(
            calib_features, calib_targets, weights, LAMBDA,
        )

        # --- Query: strict post-4-trial boundary, history entirely post-support ---
        fifth = float(trial_values[SUPPORT_TRIALS])
        fifth_bins = np.flatnonzero(eval_mask & np.isfinite(trial_num) & (trial_num == fifth))
        ridge.require(fifth_bins.size > 0, f"{name}: fifth trial has no eval-valid bins")
        boundary = int(fifth_bins[0])

        query_windows = []
        query_indices_safe = []
        for start in range(boundary, n_total - WINDOW + 1):
            output = start + WINDOW - 1
            if not eval_mask[output]:
                continue
            # last bin of the 700-bin decoder window is `output`
            # ridge history for this bin is [output-49:output+1]
            if not _query_trial_safe(trial_num, output, HISTORY_BINS, support_values):
                continue
            query_windows.append((name, start))
            query_indices_safe.append(output)

        ridge.require(len(query_windows) > 0, f"{name}: no safe query windows")
        query_windows_arr = np.array([s for _, s in query_windows], dtype=np.int64)
        query_output_arr = np.array(query_indices_safe, dtype=np.int64)

        n_query = len(query_windows)
        query_features = np.empty((n_query, HISTORY_BINS * EXPECTED_NEURONS), dtype=np.float64)
        query_truth = np.empty((n_query, VELOCITY_DIM), dtype=np.float64)
        for i in range(n_query):
            last_bin = query_output_arr[i]
            query_features[i] = neural64[last_bin - HISTORY_BINS + 1:last_bin + 1].flatten()
            query_truth[i] = velocity64[last_bin]

        query_pred = ((query_features - xbar[None, :]) / scale[None, :]) @ coefficients + intercept
        session_r2 = _r2(query_truth, query_pred)

        # Per-trial calibration bin counts
        calib_by_trial = {}
        for tv, cnt in zip(*np.unique(calib_trial_ids, return_counts=True)):
            calib_by_trial[str(float(tv))] = int(cnt)

        per_session_results[name] = {
            "session_name": record.session_name,
            "r2": float(session_r2),
            "calibration_bins": int(n_calib),
            "calibration_by_trial": calib_by_trial,
            "query_windows": int(n_query),
            "support_trials": [float(v) for v in trial_values[:SUPPORT_TRIALS]],
            "fifth_trial": fifth,
            "query_first_bin": boundary,
            "calib_starts_sha256": sha256_array(calib_starts),
            "query_output_bins_sha256": sha256_array(query_output_arr),
            "calib_truth_sha256": sha256_array(calib_targets),
            "query_truth_sha256": sha256_array(query_truth),
            "query_pred_sha256": sha256_array(query_pred),
            "coefficients_sha256": sha256_array(coefficients),
            "intercept": intercept.tolist(),
            "feature_scale_sha256": sha256_array(scale),
            "feature_mean_sha256": sha256_array(xbar),
        }
        all_truth.append(query_truth)
        all_estimate.append(query_pred)
        print(f"  R² = {session_r2:.6f} ({n_query} query windows)", file=sys.stderr)

    # --- Verify query SHA (fail closed) ---
    all_query_windows = []
    for name in H1_M4_FOLD0_TARGET:
        record = records[name]
        tv = record.trial_values
        fifth = float(tv[SUPPORT_TRIALS])
        fifth_bins = np.flatnonzero(
            record.eval_mask & np.isfinite(record.trial_num) & (record.trial_num == fifth)
        )
        boundary = int(fifth_bins[0])
        for start in range(boundary, record.neural.shape[0] - WINDOW + 1):
            output = start + WINDOW - 1
            if record.eval_mask[output]:
                all_query_windows.append((name, start))

    query_sha = _window_manifest_hash(all_query_windows)
    print(f"\nQuery SHA:  {query_sha}", file=sys.stderr)
    print(f"Expected:   {EXPECTED_QUERY_SHA}", file=sys.stderr)
    assert query_sha == EXPECTED_QUERY_SHA, \
        f"Query SHA mismatch: {query_sha} != {EXPECTED_QUERY_SHA} — FAIL CLOSED"

    # --- Pooled R² ---
    pooled_truth = np.concatenate(all_truth, axis=0)
    pooled_estimate = np.concatenate(all_estimate, axis=0)
    pooled_r2 = _r2(pooled_truth, pooled_estimate)

    elapsed = time.monotonic() - t0

    # --- Write receipt ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "h1_ridge_baseline_v2",
        "status": "COMPLETED_CPU_ONLY",
        "date": "2026-08-12",
        "definition": (
            "Per-session ridge decoder on H1 fold-0. Dense per-bin 7-DoF velocity target, "
            "50-bin flattened spike history features [t-49:t+1], normalized lambda=1.0 "
            "(fixed canonical comparator, no source-only lambda selection). "
            "Calibration history entirely within the same support trial. "
            "Query history entirely within post-support scope. "
            "Same query boundary and R² computation as H-SE5 evaluation. "
            "Deployment boundary reference: ridge consumes dense velocity where "
            "carrier consumes one scalar per event. NOT equal-information."
        ),
        "v1_status": "SUPERSEDED by protocol-corrected v2 (v1 had calibration/query feature window mismatch)",
        "v1_receipt_sha256": "f12cfe39ada08fae38035e79ef33b0ab6ae3f9b29c219da4f81d69e2e109c00d",
        "v1_superseded_reason": (
            "v1 calibration feature used [t-50:t] (excludes current bin) while "
            "query used [last_bin-49:last_bin+1] (includes current bin). "
            "v2 uses [t-49:t+1] for both, with same-trial and post-support guards."
        ),
        "pooled_r2": float(pooled_r2),
        "per_session": per_session_results,
        "query_window_sha256": query_sha,
        "query_window_sha_expected": EXPECTED_QUERY_SHA,
        "query_window_sha_match": True,
        "ridge_config": {
            "history_bins": HISTORY_BINS,
            "feature_window": "[t-49:t+1] (50 bins including current)",
            "feature_dim": HISTORY_BINS * EXPECTED_NEURONS,
            "velocity_dim": VELOCITY_DIM,
            "lambda": LAMBDA,
            "lambda_selection": "fixed canonical comparator (no source-only selection)",
            "n_neurons": EXPECTED_NEURONS,
            "support_trials": SUPPORT_TRIALS,
            "calibration_history_constraint": "entirely within same support trial",
            "query_history_constraint": "entirely within post-support scope",
        },
        "hse5_label_accounting": HSE5_LABEL_ACCOUNTING,
        "numerical_contract": numerical_contract,
        "input_bindings": {
            "runner_sha256": sha256_file(Path(__file__)),
            "ridge_core_sha256": ridge_core_sha,
            "loader_sha256": loader_sha,
            "hse5_terminal_receipt_sha256": hse5_terminal_sha,
            "hse5_evaluator_sha256": hse5_evaluator_sha,
            "hse5_program_completion_sha256": hse5_program_sha,
            "nwb_sha256": {name: records[name].input_sha256 for name in H1_M4_FOLD0_TARGET},
        },
        "non_interference": {
            "gpu_used": False,
            "thread_caps": "OMP/MKL/OPENBLAS=4, nice -n 10",
            "output_directory": str(OUTPUT_DIR),
        },
        "elapsed_seconds": elapsed,
    }

    receipt_path = OUTPUT_DIR / "h1_ridge_baseline_v2_receipt.json"
    raw = (json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    receipt_path.write_bytes(raw)
    os.chmod(receipt_path, 0o444)

    print(f"\n{'='*80}", file=sys.stderr)
    print("H1 RIDGE BASELINE v2 (fold-0, date 19250101)", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    print(f"Pooled R²: {pooled_r2:.6f}", file=sys.stderr)
    for name in H1_M4_FOLD0_TARGET:
        r = per_session_results[name]
        print(f"  {name}: R²={r['r2']:.6f} ({r['query_windows']} windows, {r['calibration_bins']} calib bins)", file=sys.stderr)
    print(f"\nReceipt: {receipt_path}", file=sys.stderr)
    print(f"SHA-256: {hashlib.sha256(raw).hexdigest()}", file=sys.stderr)


if __name__ == "__main__":
    main()
