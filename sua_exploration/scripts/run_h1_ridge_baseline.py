#!/usr/bin/env python3
"""H1 per-session ridge baseline comparator.

Dense per-bin velocity ridge on H1 fold-0 (date 19250101). Same calibration
prefix (first 4 trials), same query windows (strict post-four-trial boundary,
SHA-matched to H-SE5), same R² computation (float64, multi-output variance-
weighted, last-bin scoring) as h1_sparse_event_endpoint_evaluate.py.

This is a deployment boundary reference: ridge consumes dense velocity trace
where the carrier consumes one scalar per event. NOT an equal-information
comparison. Per-session: tied to one unit set, does not transfer.

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

WINDOW = 700
HISTORY_BINS = 50
VELOCITY_DIM = 7
EXPECTED_NEURONS = 176
LAMBDA = 1.0
SUPPORT_TRIALS = 4
EXPECTED_QUERY_SHA = "665fe535e90a221123b778171685577f67c2cc1902648cde2c8b2523e990e4da"
DATA_DIR = REPO_ROOT / "SPINT-main/data/000954"
OUTPUT_DIR = REPO_ROOT / "sua_exploration/results/h1_ridge_baseline_v1"


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


def build_calib_features(neural: np.ndarray, eval_mask: np.ndarray,
                         trial_num: np.ndarray, trial_values: tuple,
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Build calibration features and targets from first SUPPORT_TRIALS trials.

    Features: HISTORY_BINS x N flattened spike history per calibration bin.
    Targets: VELOCITY_DIM velocity at that bin.

    Only eval-valid bins within the support trials are used. A bin is a valid
    calibration row if its preceding HISTORY_BINS bins are also available
    (not necessarily eval-valid, just present in the recording).
    """
    support_values = trial_values[:SUPPORT_TRIALS]
    n_total = neural.shape[0]
    calib_indices = []
    for t in range(HISTORY_BINS, n_total):
        if not eval_mask[t]:
            continue
        trial_val = trial_num[t]
        if not np.isfinite(trial_val):
            continue
        if trial_val not in support_values:
            continue
        calib_indices.append(t)

    ridge.require(bool(calib_indices), "no valid calibration bins in support trials")
    calib_indices = np.array(calib_indices, dtype=np.int64)

    features = np.empty((len(calib_indices), HISTORY_BINS * EXPECTED_NEURONS), dtype=np.float64)
    targets = np.empty((len(calib_indices), VELOCITY_DIM), dtype=np.float64)
    for i, t in enumerate(calib_indices):
        window = neural[t - HISTORY_BINS:t]  # [50, 176]
        features[i] = window.flatten()
        targets[i] = neural_vel[t]  # will be set by caller

    return calib_indices, features


def main():
    t0 = time.monotonic()
    numerical_contract = ridge.numerical_contract_self_test()
    print(f"[{time.strftime('%H:%M:%S')}] Numerical contract: {numerical_contract}", file=sys.stderr)

    from src.data.h1_m4_eb_pilot import load_target_records, H1_M4_FOLD0_TARGET, WINDOW as PILOT_WINDOW

    assert PILOT_WINDOW == WINDOW, f"window mismatch: {PILOT_WINDOW} vs {WINDOW}"

    records = load_target_records(DATA_DIR)
    assert set(records) == set(H1_M4_FOLD0_TARGET), "target session set mismatch"

    per_session_results = {}
    all_truth = []
    all_estimate = []

    for name in H1_M4_FOLD0_TARGET:
        record = records[name]
        print(f"[{time.strftime('%H:%M:%S')}] Processing {name} ({record.session_name})", file=sys.stderr)

        # Build calibration data
        trial_values = record.trial_values
        support_values = trial_values[:SUPPORT_TRIALS]
        n_total = record.neural.shape[0]

        calib_indices = []
        for t in range(HISTORY_BINS, n_total):
            if not record.eval_mask[t]:
                continue
            tv = record.trial_num[t]
            if not np.isfinite(tv) or tv not in support_values:
                continue
            calib_indices.append(t)
        calib_indices = np.array(calib_indices, dtype=np.int64)
        ridge.require(len(calib_indices) > 0, f"{name}: no calibration bins")

        calib_features = np.empty((len(calib_indices), HISTORY_BINS * EXPECTED_NEURONS), dtype=np.float64)
        calib_targets = np.empty((len(calib_indices), VELOCITY_DIM), dtype=np.float64)
        neural64 = record.neural.astype(np.float64)
        velocity64 = record.velocity.astype(np.float64)
        for i, t in enumerate(calib_indices):
            calib_features[i] = neural64[t - HISTORY_BINS:t].flatten()
            calib_targets[i] = velocity64[t]

        print(f"  Calibration: {len(calib_indices)} bins, {calib_features.shape[1]} features", file=sys.stderr)

        # Fit normalized ridge (direct numpy, OUTPUT_DIM=7 not supported by core module)
        from sua_exploration.mc_maze.priority_a2_normalized_ridge_v2 import FEATURE_STD_EPS
        weights = np.ones(len(calib_indices), dtype=np.float64)
        x = calib_features
        y = calib_targets
        w = weights
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
        t = yc * sqrt_w[:, None]
        if a.shape[0] < a.shape[1]:
            dual_gram = a @ a.T
            dual_gram.flat[::dual_gram.shape[0] + 1] += LAMBDA
            coefficients = a.T @ np.linalg.solve(dual_gram, t)
        else:
            primal_gram = a.T @ a
            primal_gram.flat[::primal_gram.shape[0] + 1] += LAMBDA
            coefficients = np.linalg.solve(primal_gram, a.T @ t)
        intercept = ybar

        def predict(feat):
            return ((feat - xbar[None, :]) / scale[None, :]) @ coefficients + intercept

        # Build query windows (strict post-four-trial boundary)
        fifth = float(trial_values[SUPPORT_TRIALS])
        fifth_bins = np.flatnonzero(
            record.eval_mask & np.isfinite(record.trial_num) & (record.trial_num == fifth)
        )
        ridge.require(fifth_bins.size > 0, f"{name}: fifth trial has no eval-valid bins")
        boundary = int(fifth_bins[0])

        query_indices = []
        for start in range(boundary, n_total - WINDOW + 1):
            output = start + WINDOW - 1
            if record.eval_mask[output]:
                query_indices.append((name, start))

        ridge.require(bool(query_indices), f"{name}: no query windows")
        query_indices_arr = np.array([s for _, s in query_indices], dtype=np.int64)

        # Predict at each query window's last bin
        query_features = np.empty((len(query_indices), HISTORY_BINS * EXPECTED_NEURONS), dtype=np.float64)
        query_truth = np.empty((len(query_indices), VELOCITY_DIM), dtype=np.float64)
        for i, start in enumerate(query_indices_arr):
            last_bin = start + WINDOW - 1
            feat_start = last_bin - HISTORY_BINS + 1
            query_features[i] = neural64[feat_start:last_bin + 1].flatten()
            query_truth[i] = velocity64[last_bin]

        query_pred = predict(query_features)
        session_r2 = _r2(query_truth, query_pred)

        per_session_results[name] = {
            "session_name": record.session_name,
            "calibration_bins": int(len(calib_indices)),
            "query_windows": int(len(query_indices)),
            "r2": float(session_r2),
            "support_trials": [float(v) for v in support_values],
            "fifth_trial": fifth,
            "query_first_bin": boundary,
        }
        all_truth.append(query_truth)
        all_estimate.append(query_pred)

        print(f"  R² = {session_r2:.6f} ({len(query_indices)} query windows)", file=sys.stderr)

    # Verify query window SHA
    all_query_windows = []
    for name in H1_M4_FOLD0_TARGET:
        record = records[name]
        trial_values = record.trial_values
        fifth = float(trial_values[SUPPORT_TRIALS])
        fifth_bins = np.flatnonzero(
            record.eval_mask & np.isfinite(record.trial_num) & (record.trial_num == fifth)
        )
        boundary = int(fifth_bins[0])
        for start in range(boundary, record.neural.shape[0] - WINDOW + 1):
            output = start + WINDOW - 1
            if record.eval_mask[output]:
                all_query_windows.append((name, start))

    query_sha = _window_manifest_hash(all_query_windows)
    print(f"\nQuery SHA: {query_sha}", file=sys.stderr)
    print(f"Expected:  {EXPECTED_QUERY_SHA}", file=sys.stderr)
    if query_sha != EXPECTED_QUERY_SHA:
        print("WARNING: query SHA mismatch — window definitions may differ", file=sys.stderr)

    # Pooled R²
    pooled_truth = np.concatenate(all_truth, axis=0)
    pooled_estimate = np.concatenate(all_estimate, axis=0)
    pooled_r2 = _r2(pooled_truth, pooled_estimate)

    elapsed = time.monotonic() - t0

    # Write receipt
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "h1_ridge_baseline_v1",
        "status": "COMPLETED_CPU_ONLY",
        "date": "2026-08-12",
        "definition": (
            "Per-session ridge decoder on H1 fold-0. Dense per-bin velocity target, "
            "50-bin flattened spike history features, normalized lambda=1.0. "
            "Same calibration prefix (4 trials), same query boundary (post-4-trial), "
            "same R² computation as H-SE5 evaluation. "
            "Deployment boundary reference: ridge consumes dense velocity where "
            "carrier consumes one scalar per event. NOT equal-information."
        ),
        "pooled_r2": float(pooled_r2),
        "per_session": per_session_results,
        "query_window_sha256": query_sha,
        "query_window_sha_expected": EXPECTED_QUERY_SHA,
        "query_window_sha_match": query_sha == EXPECTED_QUERY_SHA,
        "n_target_sessions": len(H1_M4_FOLD0_TARGET),
        "ridge_config": {
            "history_bins": HISTORY_BINS,
            "feature_dim": HISTORY_BINS * EXPECTED_NEURONS,
            "velocity_dim": VELOCITY_DIM,
            "lambda": LAMBDA,
            "n_neurons": EXPECTED_NEURONS,
            "support_trials": SUPPORT_TRIALS,
        },
        "numerical_contract": numerical_contract,
        "input_bindings": {
            "runner_sha256": sha256_file(Path(__file__)),
            "ridge_core_sha256": sha256_file(REPO_ROOT / "sua_exploration/mc_maze/priority_a2_normalized_ridge_v2.py"),
            "nwb_sha256": {name: records[name].input_sha256 for name in H1_M4_FOLD0_TARGET},
        },
        "non_interference": {
            "gpu_used": False,
            "thread_caps": "OMP/MKL/OPENBLAS=4, nice -n 10",
            "output_directory": str(OUTPUT_DIR),
        },
        "elapsed_seconds": elapsed,
    }

    receipt_path = OUTPUT_DIR / "h1_ridge_baseline_receipt.json"
    raw = (json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    receipt_path.write_bytes(raw)
    os.chmod(receipt_path, 0o444)

    print(f"\n{'='*80}", file=sys.stderr)
    print(f"H1 RIDGE BASELINE (fold-0, date 19250101)", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    print(f"Pooled R²: {pooled_r2:.6f}", file=sys.stderr)
    for name in H1_M4_FOLD0_TARGET:
        r = per_session_results[name]
        print(f"  {name}: R²={r['r2']:.6f} ({r['query_windows']} windows)", file=sys.stderr)
    print(f"\nReceipt: {receipt_path}", file=sys.stderr)
    print(f"SHA-256: {hashlib.sha256(raw).hexdigest()}", file=sys.stderr)


if __name__ == "__main__":
    main()
