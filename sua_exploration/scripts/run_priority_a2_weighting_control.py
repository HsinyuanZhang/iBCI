#!/usr/bin/env python3
"""Priority A2a: 2×2 weighting/solver correction.

Four arms: {dense, sparse-direction} × {uniform-window, equal-trial} weighting.
Same 50*N features, same query, same lambda=1. Correct weighted ridge with
unpenalized intercept. Reports hashes for row vectors, weights, predictions, targets.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import sys
import json
import hashlib
import math
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "sua_exploration"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sua_exploration.mc_maze import subm_v9_f0_pv_ridge as numerical
from sua_exploration.mc_maze.subm_co_score_only_v2 import recompute_torchmetrics_r2_cpu
from sua_exploration.scripts.run_subm_v9_f0_pv_ridge_controls import (
    load_v9_inputs, load_mean_std, load_v9_target,
    _build_view_base, _nwb_path_and_pin, _runtime_owners,
)
from sua_exploration.mc_maze.multisession_datamodule import _compute_valid_starts

V9_RUN_ROOT = REPO_ROOT / "sua_exploration/results/dandi_000688_subm_co_three_arm_v9_formal_20260805"
NWB_ROOT = REPO_ROOT / "sua_exploration/data/dandi_000688"
OUTPUT_DIR = REPO_ROOT / "sua_exploration/results/trial_level_ridge_v1"
VIEWS = ("sua", "pseudo_mua")
BUDGETS = (15, 30, 50)
HISTORY_BINS = numerical.HISTORY_BINS
EXPECTED_SESSIONS = 15


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()

def sha256_file(path):
    d = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            d.update(block)
    return d.hexdigest()

def sha256_array(arr):
    return sha256_bytes(np.ascontiguousarray(arr).tobytes(order="C"))


def fit_ridge_weighted_correct(features, targets, sample_weights, normalized_lambda=1.0):
    """Weighted ridge with correct unpenalized intercept.

    Solves: sum_i w_i * ||y_i - x_i W - b||^2 + lambda * ||W||^2
    where b (intercept) is unpenalized.

    The weighted first-order condition for b:
      b = (sum_i w_i (y_i - x_i W)) / (sum_i w_i)
    Substituting into the objective yields the weighted centered ridge:
      Let Wsum = sum(w), xbar_w = sum(w_i x_i)/Wsum, ybar_w = sum(w_i y_i)/Wsum
      Z = X - xbar_w  (weighted-centered)
      Yc = Y - ybar_w
      gram = Z^T diag(w) Z + lambda * I   (NOT divided by Wsum — objective is sum, not mean)
      rhs = Z^T diag(w) Yc
      W_hat = solve(gram, rhs)
      b_hat = ybar_w - xbar_w @ W_hat
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    w = np.asarray(sample_weights, dtype=np.float64)
    n, p = x.shape
    assert y.shape == (n, 2) and w.shape == (n,)

    Wsum = w.sum()

    # Weighted means
    xbar_w = (w[:, None] * x).sum(axis=0) / Wsum  # [p]
    ybar_w = (w[:, None] * y).sum(axis=0) / Wsum  # [2]

    # Weighted centering
    Z = x - xbar_w[None, :]
    Yc = y - ybar_w[None, :]

    # Standardization: use weighted std from the CENTERED features
    var_w = (w[:, None] * Z**2).sum(axis=0) / Wsum
    scale = np.sqrt(var_w)
    scale[scale < numerical.FEATURE_STD_EPS] = 1.0
    Zs = Z / scale[None, :]

    # Weighted Gram (sum objective, not mean)
    Zw = Zs * w[:, None]
    gram = Zw.T @ Zs
    gram.flat[::gram.shape[0] + 1] += normalized_lambda
    rhs = Zw.T @ Yc
    W_hat = np.linalg.solve(gram, rhs)

    # Unpenalized intercept from first-order condition
    b_hat = ybar_w - (xbar_w / scale) @ W_hat

    # Verify intercept first-order condition: sum_i w_i (y_i - x_i W - b) ≈ 0
    pred_calib = (x / scale[None, :]) @ W_hat + b_hat
    resid_weighted = (w[:, None] * (y - pred_calib)).sum(axis=0)
    assert np.allclose(resid_weighted, 0, atol=1e-8), \
        f"intercept first-order condition violated: {resid_weighted}"

    return numerical.RidgeReadout(
        feature_mean=np.ascontiguousarray(np.zeros(p, dtype=np.float32)),  # already centered
        feature_scale=np.ascontiguousarray(scale, dtype=np.float32),
        target_mean=np.ascontiguousarray(np.zeros(2, dtype=np.float32)),  # intercept handled separately
        weights=np.ascontiguousarray(W_hat, dtype=np.float32),
        normalized_lambda=float(normalized_lambda),
        solver_device="cpu",
    ), b_hat


def predict_ridge_correct(features, readout, intercept):
    """Predict with explicit intercept (readout.feature_mean is zero)."""
    x = np.asarray(features, dtype=np.float32)
    result = (x - readout.feature_mean) / readout.feature_scale @ readout.weights + intercept
    return np.ascontiguousarray(result, dtype=np.float32)


def predict_query_batched(record, readout, intercept, batch_size=2048):
    parts = []
    for starts in numerical.batched(record.valid_starts, batch_size):
        feats = numerical.raw_window_features(record.neural, starts)
        parts.append(predict_ridge_correct(feats, readout, intercept))
    return np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)


def main():
    t0 = time.monotonic()
    print(f"[{time.strftime('%H:%M:%S')}] Loading V9 inputs...", file=sys.stderr)
    v9 = load_v9_inputs(V9_RUN_ROOT, repo_root=REPO_ROOT)
    owners = _runtime_owners(REPO_ROOT)
    behavior_stats = {v: load_mean_std(v9.behavior_normalizers[v][0], label=f"{v} behavior") for v in VIEWS}
    cohort = v9.cohort
    assert len(cohort) == EXPECTED_SESSIONS
    cohort_assets = [r.asset_id for r in cohort]

    # T4 sealed references
    budget_agg = json.loads((REPO_ROOT / "sua_exploration/results/dandi_000688_subm_v9_t4_label_budget_v1_full/aggregate/endpoint_aggregate_torchmetrics151.json").read_text())
    v9_agg = json.loads((V9_RUN_ROOT / "aggregate/endpoint_aggregate_torchmetrics151.json").read_text())
    t4_ref = {}
    for view in VIEWS:
        t4_ref[view] = {}
        for budget in (15, 30):
            t4_ref[view][budget] = np.mean([c["r2"] for c in budget_agg["cells"] if c["view"]==view and c["budget"]==budget])
        t4_ref[view][50] = np.mean([c["r2"] for c in v9_agg["cells"] if c["view"]==view and c["arm"]=="shared_t4"])

    # 4 arms: {dense, sparse} × {uniform, equal-trial}
    ARMS = ("dense_uniform", "dense_equaltrial", "sparse_uniform", "sparse_equaltrial")
    results = {b: {v: {a: [] for a in ARMS} for v in VIEWS} for b in BUDGETS}
    hashes = {b: {v: {a: [] for a in ARMS} for v in VIEWS} for b in BUDGETS}
    conditioning = {b: {v: {} for v in VIEWS} for b in BUDGETS}
    support_query_overlap = []

    for si, row in enumerate(cohort):
        nwb_path = _nwb_path_and_pin(NWB_ROOT, row)
        for view in VIEWS:
            mean, std = behavior_stats[view]
            record, _rebuilt, builder_trials, _bridge = _build_view_base(
                repo_root=REPO_ROOT, nwb_path=nwb_path, view=view, mean=mean, std=std, owners=owners,
            )
            v9_target, v9_target_sha, _ = load_v9_target(v9, row, view)

            for budget in BUDGETS:
                support = list(builder_trials[:budget])
                cal_starts = np.ascontiguousarray(
                    _compute_valid_starts(support, HISTORY_BINS), dtype=np.int64,
                )
                cal_features = numerical.raw_window_features(record.neural, cal_starts)

                # Verify support/query overlap is zero
                overlap = np.intersect1d(cal_starts, record.valid_starts, assume_unique=True)
                assert overlap.size == 0, f"support/query overlap: session {row.session_id} budget {budget}"
                if si == 0 and view == "sua":
                    support_query_overlap.append({
                        "budget": budget, "view": view,
                        "cal_starts_sha": sha256_array(cal_starts),
                        "query_starts_sha": sha256_array(record.valid_starts),
                        "overlap_count": 0,
                    })

                # Targets
                dense_Y = numerical.targets_at_window_end(record.behavior, cal_starts)  # [rows, 2]

                sparse_Y = np.empty_like(dense_Y)
                trial_bounds = [(int(t["start"]), int(t["stop"])) for t in support]
                bounds_arr = np.array(trial_bounds, dtype=np.int64)
                trial_of_window = np.empty(len(cal_starts), dtype=np.int64)
                for wi, ws in enumerate(cal_starts):
                    ti = np.searchsorted(bounds_arr[:, 1], ws, side="right")
                    ti = min(ti, len(support) - 1)
                    trial_of_window[wi] = ti
                    theta = float(support[ti].get("target_dir", 0.0))
                    sparse_Y[wi, 0] = math.cos(theta)
                    sparse_Y[wi, 1] = math.sin(theta)

                # Weights
                trial_counts = np.bincount(trial_of_window, minlength=len(support)).astype(np.float64)
                w_uniform = np.ones(len(cal_starts), dtype=np.float64)
                w_equaltrial = np.ones(len(cal_starts), dtype=np.float64)
                for wi in range(len(cal_starts)):
                    w_equaltrial[wi] = 1.0 / trial_counts[trial_of_window[wi]]
                # Normalize: sum to N (relative trial weights must stay equal)
                w_equaltrial *= len(cal_starts) / w_equaltrial.sum()

                for target_name, target_Y in [("dense", dense_Y), ("sparse", sparse_Y)]:
                    for weight_name, weights in [("uniform", w_uniform), ("equaltrial", w_equaltrial)]:
                        arm = f"{target_name}_{weight_name}"
                        readout, intercept = fit_ridge_weighted_correct(
                            cal_features, target_Y, weights,
                            normalized_lambda=numerical.RIDGE_NORMALIZED_LAMBDA,
                        )
                        pred = predict_query_batched(record, readout, intercept)
                        r2 = float(recompute_torchmetrics_r2_cpu(pred, v9_target))
                        results[budget][view][arm].append(r2)
                        hashes[budget][view][arm].append({
                            "pred_sha": sha256_array(pred),
                            "target_sha": v9_target_sha,
                            "weights_sha": sha256_array(weights),
                        })

                # Conditioning (same features for all arms)
                cond = core_conditioning(cal_features)
                conditioning[budget][view] = cond

                print(f"[{time.strftime('%H:%M:%S')}] M={budget} {view} s{si+1}/15: "
                      f"DU={results[budget][view]['dense_uniform'][-1]:.4f} "
                      f"DE={results[budget][view]['dense_equaltrial'][-1]:.4f} "
                      f"SU={results[budget][view]['sparse_uniform'][-1]:.4f} "
                      f"SE={results[budget][view]['sparse_equaltrial'][-1]:.4f}",
                      file=sys.stderr)

            del record, _rebuilt

    elapsed = time.monotonic() - t0

    # Build contrasts
    all_results = {}
    for view in VIEWS:
        for budget in BUDGETS:
            r = {}
            for arm in ARMS:
                arr = np.array(results[budget][view][arm])
                r[f"{arm}_mean"] = float(arr.mean())
                r[f"{arm}_median"] = float(np.median(arr))
                r[f"{arm}_sign_positive"] = int(np.sum(arr > 0))
                r[f"{arm}_per_session"] = {cohort_assets[i]: float(arr[i]) for i in range(EXPECTED_SESSIONS)}

            # Contrasts
            du = np.array(results[budget][view]["dense_uniform"])
            de = np.array(results[budget][view]["dense_equaltrial"])
            su = np.array(results[budget][view]["sparse_uniform"])
            se = np.array(results[budget][view]["sparse_equaltrial"])

            # equal-trial − uniform at each target
            delta_dense_weight = de - du
            delta_sparse_weight = se - su
            # dense − sparse at each weighting
            delta_uniform_target = du - su
            delta_equaltrial_target = de - se

            r["delta_dense_equaltrial_minus_uniform_mean"] = float(delta_dense_weight.mean())
            r["delta_dense_equaltrial_minus_uniform_median"] = float(np.median(delta_dense_weight))
            r["delta_dense_equaltrial_minus_uniform_sign"] = int(np.sum(delta_dense_weight > 0))

            r["delta_sparse_equaltrial_minus_uniform_mean"] = float(delta_sparse_weight.mean())
            r["delta_sparse_equaltrial_minus_uniform_median"] = float(np.median(delta_sparse_weight))
            r["delta_sparse_equaltrial_minus_uniform_sign"] = int(np.sum(delta_sparse_weight > 0))

            r["delta_dense_minus_sparse_uniform_mean"] = float(delta_uniform_target.mean())
            r["delta_dense_minus_sparse_uniform_median"] = float(np.median(delta_uniform_target))
            r["delta_dense_minus_sparse_uniform_sign"] = int(np.sum(delta_uniform_target > 0))

            r["delta_dense_minus_sparse_equaltrial_mean"] = float(delta_equaltrial_target.mean())
            r["delta_dense_minus_sparse_equaltrial_median"] = float(np.median(delta_equaltrial_target))
            r["delta_dense_minus_sparse_equaltrial_sign"] = int(np.sum(delta_equaltrial_target > 0))

            r["t4_reference"] = float(t4_ref[view][budget])
            r["conditioning"] = conditioning[budget][view]
            all_results[f"{view}_M{budget}"] = r

    # Print summary
    print("\n" + "=" * 120, file=sys.stderr)
    print("PRIORITY A2a: 2×2 WEIGHTING CORRECTION", file=sys.stderr)
    print("=" * 120, file=sys.stderr)
    print(f"\n{'B/V':<20} {'DU':<10} {'DE':<10} {'SU':<10} {'SE':<10} {'D(E−U)':<10} {'S(E−U)':<10} {'D−S(U)':<10} {'D−S(E)':<10}", file=sys.stderr)
    for view in VIEWS:
        for budget in BUDGETS:
            r = all_results[f"{view}_M{budget}"]
            print(f"M={budget} {view:<14} "
                  f"{r['dense_uniform_mean']:<10.4f} {r['dense_equaltrial_mean']:<10.4f} "
                  f"{r['sparse_uniform_mean']:<10.4f} {r['sparse_equaltrial_mean']:<10.4f} "
                  f"{r['delta_dense_equaltrial_minus_uniform_mean']:<+10.4f} "
                  f"{r['delta_sparse_equaltrial_minus_uniform_mean']:<+10.4f} "
                  f"{r['delta_dense_minus_sparse_uniform_mean']:<+10.4f} "
                  f"{r['delta_dense_minus_sparse_equaltrial_mean']:<+10.4f}",
                  file=sys.stderr)

    # Write receipt
    receipt = {
        "schema": "priority_a2a_weighting_control_v1",
        "status": "COMPLETED_CPU_ONLY",
        "date": "2026-08-11",
        "definition": (
            "2×2 ablation: {dense per-bin velocity, direction-only [cosθ,sinθ]} × "
            "{uniform-window weighting, equal-trial weighting}. "
            "Same 50*N features, same post-trial-50 query, same lambda=1, "
            "correct weighted ridge with unpenalized intercept. "
            "Answers: is the dense−sparse gap confounded by weighting?"
        ),
        "cohort": [{"asset_id": r.asset_id, "session_id": r.session_id} for r in cohort],
        "results": all_results,
        "hashes": hashes,
        "support_query_overlap": support_query_overlap,
        "runner_sha256": sha256_file(Path(__file__)),
        "core_sha256": sha256_file(REPO_ROOT / "sua_exploration/mc_maze/trial_level_ridge_core.py"),
        "v9_manifest_sha256": v9.manifest_sha256,
        "intercept_verification": "assert np.allclose(resid_weighted, 0, atol=1e-8) per arm",
        "non_interference": {
            "gpu_used": False,
            "thread_caps": "OMP/MKL/OPENBLAS=4, nice -n 10",
            "output_directory": str(OUTPUT_DIR),
        },
        "elapsed_seconds": elapsed,
    }
    receipt_bytes = (json.dumps(receipt, sort_keys=True, separators=(",",":"), ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    receipt_path = OUTPUT_DIR / "priority_a2_weighting_receipt.json"
    receipt_path.write_bytes(receipt_bytes)
    os.chmod(receipt_path, 0o444)
    print(f"\nReceipt: {receipt_path}", file=sys.stderr)
    print(f"SHA-256: {sha256_bytes(receipt_bytes)}", file=sys.stderr)


def core_conditioning(features, normalized_lambda=numerical.RIDGE_NORMALIZED_LAMBDA):
    x = np.asarray(features, dtype=np.float64)
    n, p = x.shape
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < numerical.FEATURE_STD_EPS] = 1.0
    Z = (x - mean) / scale
    sv = np.linalg.svd(Z, compute_uv=False)
    gram = (Z.T @ Z) / n
    eigvals = np.linalg.eigvalsh(gram) + normalized_lambda
    cond = float(eigvals.max() / eigvals.min())
    trace_hat = float(np.sum(sv**2 / (sv**2 + n * normalized_lambda)))
    return {
        "rows": int(n), "features": int(p),
        "design_rank": int(np.linalg.matrix_rank(Z)),
        "condition_number": cond,
        "trace_hat": trace_hat,
        "lambda": float(normalized_lambda),
    }


if __name__ == "__main__":
    main()
