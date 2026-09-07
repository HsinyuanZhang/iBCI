#!/usr/bin/env python3
"""Priority A: label-density ablation on SUA/pseudo-MUA.

Same 50*N flattened features as Ridge50. Same calibration windows. Same query.
Same support-only standardization. Same fixed lambda=1.

Only the TARGET changes:
  dense-Y: per-bin endpoint velocity (the existing Ridge50 target)
  sparse-Y: [cos(theta), sin(theta)] derived from the trial-table direction scalar,
            repeated for every window in that trial. Weighted so each trial is equal.

This isolates the effect of label density. If sparse-Y drops far below dense-Y,
the label density is the cause of the Ridge50 advantage.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import sys
import json
import hashlib
import time
import math
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "sua_exploration"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sua_exploration.mc_maze import subm_v9_f0_pv_ridge as numerical
from sua_exploration.mc_maze import trial_level_ridge_core as core
from sua_exploration.mc_maze.subm_co_score_only_v2 import recompute_torchmetrics_r2_cpu

def fit_ridge_weighted(features, targets, sample_weights, normalized_lambda=1.0):
    """Weighted ridge: mean(sum_i w_i * ||y_i - x_i W - b||^2) + lambda ||W||^2.
    Same standardization as numerical._ridge_cpu, but with per-sample weights."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    w = np.asarray(sample_weights, dtype=np.float64)
    w = w / w.sum() * len(w)  # normalize so weights sum to N

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < numerical.FEATURE_STD_EPS] = 1.0
    Z = (x - mean) / scale

    target_mean = y.mean(axis=0)
    n = Z.shape[0]

    # Weighted Gram: Z^T diag(w) Z / n + lambda * I
    Zw = Z * w[:, None]
    gram = (Zw.T @ Z) / n
    gram.flat[::gram.shape[0] + 1] += normalized_lambda
    rhs = (Zw.T @ (y - target_mean)) / n
    weights = np.linalg.solve(gram, rhs)

    return numerical.RidgeReadout(
        feature_mean=np.ascontiguousarray(mean, dtype=np.float32),
        feature_scale=np.ascontiguousarray(scale, dtype=np.float32),
        target_mean=np.ascontiguousarray(target_mean, dtype=np.float32),
        weights=np.ascontiguousarray(weights, dtype=np.float32),
        normalized_lambda=float(normalized_lambda),
        solver_device="cpu",
    )
from sua_exploration.scripts.run_subm_v9_f0_pv_ridge_controls import (
    load_v9_inputs,
    load_mean_std,
    load_v9_target,
    _build_view_base,
    _nwb_path_and_pin,
    _runtime_owners,
)
from sua_exploration.mc_maze.multisession_datamodule import _compute_valid_starts
from sua_exploration.mc_maze.unit_side_features import _nearest_canonical_direction_index

V9_RUN_ROOT = REPO_ROOT / "sua_exploration/results/dandi_000688_subm_co_three_arm_v9_formal_20260805"
NWB_ROOT = REPO_ROOT / "sua_exploration/data/dandi_000688"
OUTPUT_DIR = REPO_ROOT / "sua_exploration/results/trial_level_ridge_v1"
VIEWS = ("sua", "pseudo_mua")
BUDGETS = (15, 30, 50)
HISTORY_BINS = numerical.HISTORY_BINS
EXPECTED_SESSIONS = 15


def sha256_file(path):
    d = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            d.update(block)
    return d.hexdigest()


def run_priority_a():
    t0 = time.monotonic()
    print(f"[{time.strftime('%H:%M:%S')}] Loading V9 inputs...", file=sys.stderr)
    v9 = load_v9_inputs(V9_RUN_ROOT, repo_root=REPO_ROOT)
    owners = _runtime_owners(REPO_ROOT)
    behavior_stats = {v: load_mean_std(v9.behavior_normalizers[v][0], label=f"{v} behavior") for v in VIEWS}
    cohort = v9.cohort
    assert len(cohort) == EXPECTED_SESSIONS

    # Load sealed T4 for contrast
    budget_agg = json.loads((REPO_ROOT / "sua_exploration/results/dandi_000688_subm_v9_t4_label_budget_v1_full/aggregate/endpoint_aggregate_torchmetrics151.json").read_text())
    v9_agg = json.loads((V9_RUN_ROOT / "aggregate/endpoint_aggregate_torchmetrics151.json").read_text())
    t4_per_session = {}
    for view in VIEWS:
        t4_per_session[view] = {}
        for budget in (15, 30):
            vals = [np.mean([c["r2"] for c in budget_agg["cells"] if c["asset_id"]==row.asset_id and c["view"]==view and c["budget"]==budget]) for row in cohort]
            t4_per_session[view][budget] = np.array(vals)
        vals = [np.mean([c["r2"] for c in v9_agg["cells"] if c["asset_id"]==row.asset_id and c["view"]==view and c["arm"]=="shared_t4"]) for row in cohort]
        t4_per_session[view][50] = np.array(vals)

    results = {b: {v: {"dense": [], "sparse_dir": [], "cond": []} for v in VIEWS} for b in BUDGETS}
    cohort_assets = [row.asset_id for row in cohort]

    for si, row in enumerate(cohort):
        nwb_path = _nwb_path_and_pin(NWB_ROOT, row)
        for view in VIEWS:
            mean, std = behavior_stats[view]
            record, _rebuilt, builder_trials, _bridge = _build_view_base(
                repo_root=REPO_ROOT, nwb_path=nwb_path, view=view, mean=mean, std=std, owners=owners,
            )
            v9_target, v9_target_sha, _ = load_v9_target(v9, row, view)

            # Query: same as Ridge50 — byte-identical V9 target
            query_pred_fn = lambda readout: predict_ridge_query_batched(record, readout)

            for budget in BUDGETS:
                support = list(builder_trials[:budget])
                cal_starts = np.ascontiguousarray(
                    _compute_valid_starts(support, HISTORY_BINS), dtype=np.int64,
                )
                # Features: EXACT same as Ridge50 — 50*N flattened spike history
                cal_features = numerical.raw_window_features(record.neural, cal_starts)

                # --- dense-Y target: per-bin endpoint velocity (same as Ridge50) ---
                dense_Y = numerical.targets_at_window_end(record.behavior, cal_starts)

                # --- sparse-direction-Y: [cos(theta), sin(theta)] from trial direction scalar ---
                # Each window gets its trial's direction unit vector
                sparse_Y = np.empty_like(dense_Y)
                trial_bounds = [(int(t["start"]), int(t["stop"])) for t in support]
                bounds_arr = np.array(trial_bounds, dtype=np.int64)
                for wi, ws in enumerate(cal_starts):
                    ti = np.searchsorted(bounds_arr[:, 1], ws, side="right")
                    ti = min(ti, len(support) - 1)
                    theta = float(support[ti].get("target_dir", 0.0))
                    sparse_Y[wi, 0] = math.cos(theta)
                    sparse_Y[wi, 1] = math.sin(theta)

                # --- Fit dense-Y ridge ---
                readout_dense = numerical.fit_ridge(
                    cal_features, dense_Y,
                    normalized_lambda=numerical.RIDGE_NORMALIZED_LAMBDA, device="cpu",
                )
                pred_dense = query_pred_fn(readout_dense)
                r2_dense = float(recompute_torchmetrics_r2_cpu(pred_dense, v9_target))

                # --- Fit sparse-direction-Y ridge ---
                # For sparse-Y, weight each trial equally (inverse of per-trial window count)
                # so trial duration does not change the objective
                trial_of_window = np.empty(len(cal_starts), dtype=np.int64)
                for wi, ws in enumerate(cal_starts):
                    ti = np.searchsorted(bounds_arr[:, 1], ws, side="right")
                    trial_of_window[wi] = min(ti, len(support) - 1)
                # Equal per-trial weights
                trial_counts = np.bincount(trial_of_window, minlength=len(support)).astype(np.float64)
                sample_weights = np.ones(len(cal_starts), dtype=np.float64)
                for wi in range(len(cal_starts)):
                    sample_weights[wi] = 1.0 / trial_counts[trial_of_window[wi]]
                # Normalize so sum of weights = N
                sample_weights *= len(cal_starts) / sample_weights.sum()

                readout_sparse = fit_ridge_weighted(
                    cal_features, sparse_Y,
                    sample_weights=sample_weights,
                    normalized_lambda=numerical.RIDGE_NORMALIZED_LAMBDA,
                )
                pred_sparse = query_pred_fn(readout_sparse)
                r2_sparse = float(recompute_torchmetrics_r2_cpu(pred_sparse, v9_target))

                results[budget][view]["dense"].append(r2_dense)
                results[budget][view]["sparse_dir"].append(r2_sparse)

                # Conditioning (same for both — same features)
                cond = core.conditioning_report(cal_features, dense_Y)
                results[budget][view]["cond"].append(cond)

                print(f"[{time.strftime('%H:%M:%S')}] M={budget} {view} "
                      f"session {si+1}/15 ({row.session_id}): "
                      f"dense={r2_dense:.4f} sparse_dir={r2_sparse:.4f} "
                      f"feat={cal_features.shape[1]} rows={cal_features.shape[0]}",
                      file=sys.stderr)

            del record, _rebuilt

    elapsed = time.monotonic() - t0
    print(f"\n[{time.strftime('%H:%M:%S')}] Done in {elapsed:.0f}s", file=sys.stderr)

    # Build paired contrasts
    all_results = {}
    for view in VIEWS:
        for budget in BUDGETS:
            dense = np.array(results[budget][view]["dense"])
            sparse = np.array(results[budget][view]["sparse_dir"])
            t4 = t4_per_session[view][budget]

            delta_dense_sparse = dense - sparse  # how much does dense-Y help over sparse-Y
            delta_t4_sparse = t4 - sparse
            delta_t4_dense = t4 - dense

            all_results[f"{view}_M{budget}"] = {
                "dense_Y_mean": float(dense.mean()),
                "dense_Y_median": float(np.median(dense)),
                "sparse_dir_Y_mean": float(sparse.mean()),
                "sparse_dir_Y_median": float(np.median(sparse)),
                "t4_mean": float(t4.mean()),
                "delta_dense_minus_sparse_mean": float(delta_dense_sparse.mean()),
                "delta_dense_minus_sparse_median": float(np.median(delta_dense_sparse)),
                "delta_dense_minus_sparse_sign_dense_higher": int(np.sum(delta_dense_sparse > 0)),
                "delta_dense_minus_sparse_bootstrap": core.bootstrap_sessions(delta_dense_sparse),
                "delta_t4_minus_sparse_mean": float(delta_t4_sparse.mean()),
                "delta_t4_minus_sparse_sign_t4_higher": int(np.sum(delta_t4_sparse > 0)),
                "delta_t4_minus_dense_mean": float(delta_t4_dense.mean()),
                "delta_t4_minus_dense_sign_t4_higher": int(np.sum(delta_t4_dense > 0)),
                "per_session_dense_Y": {cohort_assets[i]: float(dense[i]) for i in range(EXPECTED_SESSIONS)},
                "per_session_sparse_dir_Y": {cohort_assets[i]: float(sparse[i]) for i in range(EXPECTED_SESSIONS)},
                "per_session_t4": {cohort_assets[i]: float(t4[i]) for i in range(EXPECTED_SESSIONS)},
                "conditioning_median": results[budget][view]["cond"][0],  # representative
            }

    # Print summary
    print("\n" + "=" * 100, file=sys.stderr)
    print("PRIORITY A: LABEL-DENSITY ABLATION (same 50*N features, different targets)", file=sys.stderr)
    print("=" * 100, file=sys.stderr)
    print(f"\n{'Budget':<8} {'View':<12} {'Dense-Y':<12} {'Sparse-dir-Y':<14} {'Δ(dense-sparse)':<16} {'T4':<10}", file=sys.stderr)
    for view in VIEWS:
        for budget in BUDGETS:
            r = all_results[f"{view}_M{budget}"]
            print(f"M={budget:<6} {view:<12} {r['dense_Y_mean']:<12.4f} {r['sparse_dir_Y_mean']:<14.4f} "
                  f"{r['delta_dense_minus_sparse_mean']:<+16.4f} {r['t4_mean']:<10.4f}",
                  file=sys.stderr)

    # Write receipt
    receipt = {
        "schema": "trial_level_ridge_priority_a_v1",
        "status": "COMPLETED_CPU_ONLY",
        "date": "2026-08-11",
        "definition": (
            "Same 50*N flattened spike-history features as Ridge50. Same calibration windows "
            "(first M trials). Same post-trial-50 query. Same support-only standardization. "
            "Same fixed lambda=1. Only the TARGET changes: "
            "dense-Y = per-bin endpoint velocity (Ridge50 target); "
            "sparse-dir-Y = [cos(theta), sin(theta)] from trial-table direction scalar, "
            "repeated per trial, equal-trial-weighted. "
            "This isolates label density."
        ),
        "cohort": [{"asset_id": r.asset_id, "session_id": r.session_id} for r in cohort],
        "results": all_results,
        "runner_sha256": sha256_file(Path(__file__)),
        "core_sha256": sha256_file(REPO_ROOT / "sua_exploration/mc_maze/trial_level_ridge_core.py"),
        "v9_manifest_sha256": v9.manifest_sha256,
        "non_interference": {
            "gpu_used": False,
            "thread_caps": "OMP/MKL/OPENBLAS=4, nice -n 10",
            "output_directory": str(OUTPUT_DIR),
        },
    }

    receipt_bytes = core.canonical_bytes(receipt)
    receipt_path = OUTPUT_DIR / "priority_a_label_density_receipt.json"
    receipt_path.write_bytes(receipt_bytes)
    os.chmod(receipt_path, 0o444)
    print(f"\nReceipt: {receipt_path}", file=sys.stderr)
    print(f"SHA-256: {core.sha256_bytes(receipt_bytes)}", file=sys.stderr)


def predict_ridge_query_batched(record, readout, batch_size=2048):
    parts = []
    for starts in numerical.batched(record.valid_starts, batch_size):
        feats = numerical.raw_window_features(record.neural, starts)
        parts.append(numerical.predict_ridge(feats, readout, device="cpu"))
    return np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)


if __name__ == "__main__":
    run_priority_a()
