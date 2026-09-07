#!/usr/bin/env python3
"""Trial-level rate ridge comparator on SUA, RT, and M2.

CPU-only. Zero GPU. Thread-limited. All output to one isolated directory.

This comparator uses ONLY trial-level mean firing rates [M, N] and trial-level
mean velocity [M, 2] as calibration — the same M trials and same per-trial
information the T4/carrier consumes. The ridge solver is reused unchanged
(numerical.fit_ridge, mean-normalized lambda=1).
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
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "sua_exploration"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sua_exploration.mc_maze import subm_v9_f0_pv_ridge as numerical
from sua_exploration.mc_maze import trial_level_ridge_core as core
from sua_exploration.mc_maze.subm_co_score_only_v2 import recompute_torchmetrics_r2_cpu
from sua_exploration.scripts.run_subm_v9_f0_pv_ridge_controls import (
    load_v9_inputs,
    load_mean_std,
    load_v9_target,
    _build_view_base,
    _nwb_path_and_pin,
    _runtime_owners,
)

V9_RUN_ROOT = REPO_ROOT / "sua_exploration/results/dandi_000688_subm_co_three_arm_v9_formal_20260805"
NWB_ROOT = REPO_ROOT / "sua_exploration/data/dandi_000688"
OUTPUT_ROOT = REPO_ROOT / "sua_exploration/results/trial_level_ridge_v1"
RT_DATA_DIR = NWB_ROOT / "sub-C"
M2_DATA_DIR = REPO_ROOT / "SPINT-main/data/000953/sub-MonkeyN-held-out-calib"
VIEWS = ("sua", "pseudo_mua")
SUA_BUDGETS = (15, 30, 50)
RT_TRIALS = 24
M2_TRIALS = 24
HISTORY_BINS = numerical.HISTORY_BINS


def sha256_file(path: Path) -> str:
    return core.sha256_file(path)


def fit_predict_score(calib_rates, calib_velocity, query_rates, query_targets):
    """Fit ridge on trial-level rates, predict on query windows, score R²."""
    readout = numerical.fit_ridge(
        np.ascontiguousarray(calib_rates, dtype=np.float32),
        np.ascontiguousarray(calib_velocity, dtype=np.float32),
        normalized_lambda=numerical.RIDGE_NORMALIZED_LAMBDA,
        device="cpu",
    )
    pred = numerical.predict_ridge(
        np.ascontiguousarray(query_rates, dtype=np.float32),
        readout,
        device="cpu",
    )
    r2 = float(recompute_torchmetrics_r2_cpu(
        np.ascontiguousarray(pred, dtype=np.float32),
        np.ascontiguousarray(query_targets, dtype=np.float32),
    ))
    cond = core.conditioning_report(calib_rates, calib_velocity)
    return r2, cond, pred, readout


# ===== SUA =====

def run_sua():
    t0 = time.monotonic()
    print(f"[{time.strftime('%H:%M:%S')}] Loading V9 inputs for SUA...", file=sys.stderr)
    v9 = load_v9_inputs(V9_RUN_ROOT, repo_root=REPO_ROOT)
    owners = _runtime_owners(REPO_ROOT)
    behavior_stats = {v: load_mean_std(v9.behavior_normalizers[v][0], label=f"{v} behavior") for v in VIEWS}
    cohort = v9.cohort
    assert len(cohort) == 15

    # Load sealed T4 per-session for contrast
    budget_agg = json.loads((REPO_ROOT / "sua_exploration/results/dandi_000688_subm_v9_t4_label_budget_v1_full/aggregate/endpoint_aggregate_torchmetrics151.json").read_text())
    v9_agg = json.loads((V9_RUN_ROOT / "aggregate/endpoint_aggregate_torchmetrics151.json").read_text())
    ridge50_agg = json.loads((REPO_ROOT / "sua_exploration/results/dandi_000688_subm_f0_pv_ridge_v1_full_20260805/aggregate/endpoint_aggregate_torchmetrics151.json").read_text())

    t4_per_session = {}
    for view in VIEWS:
        t4_per_session[view] = {}
        for budget in (15, 30):
            vals = [np.mean([c["r2"] for c in budget_agg["cells"] if c["asset_id"] == row.asset_id and c["view"] == view and c["budget"] == budget]) for row in cohort]
            t4_per_session[view][budget] = np.array(vals)
        vals = [np.mean([c["r2"] for c in v9_agg["cells"] if c["asset_id"] == row.asset_id and c["view"] == view and c["arm"] == "shared_t4"]) for row in cohort]
        t4_per_session[view][50] = np.array(vals)

    results = {b: {v: [] for v in VIEWS} for b in SUA_BUDGETS}
    conditioning = {b: {v: [] for v in VIEWS} for b in SUA_BUDGETS}
    session_info = []

    for si, row in enumerate(cohort):
        nwb_path = _nwb_path_and_pin(NWB_ROOT, row)
        for view in VIEWS:
            mean, std = behavior_stats[view]
            record, _rebuilt, builder_trials, _bridge = _build_view_base(
                repo_root=REPO_ROOT, nwb_path=nwb_path, view=view, mean=mean, std=std, owners=owners,
            )
            v9_target, v9_target_sha, _ = load_v9_target(v9, row, view)

            prefix = numerical.prefix_sums(record.neural)
            query_rates_all = numerical.window_rates_from_prefix(prefix, record.valid_starts)

            for budget in SUA_BUDGETS:
                support = list(builder_trials[:budget])
                # Window-level 50-bin rates from M calibration trials, with trial-level velocity labels.
                # Each window from trial j gets trial j's mean velocity as its label.
                # This gives the same feature distribution at train and test time.
                from sua_exploration.mc_maze.multisession_datamodule import _compute_valid_starts
                cal_starts = np.ascontiguousarray(
                    _compute_valid_starts(list(support), HISTORY_BINS), dtype=np.int64,
                )
                cal_rates = numerical.window_rates_from_prefix(prefix, cal_starts)
                # Assign trial-mean velocity to each window based on which trial it falls in
                cal_velocity = np.empty((cal_rates.shape[0], 2), dtype=np.float64)
                trial_bounds = [(int(t["start"]), int(t["stop"])) for t in support]
                trial_velocities = []
                for s, e in trial_bounds:
                    trial_velocities.append(record.behavior[s:e].mean(axis=0))
                # Map each calibration window start to its trial
                bounds_arr = np.array(trial_bounds, dtype=np.int64)
                for wi, ws in enumerate(cal_starts):
                    ti = np.searchsorted(bounds_arr[:, 1], ws, side="right")
                    ti = min(ti, len(trial_velocities) - 1)
                    cal_velocity[wi] = trial_velocities[ti]

                r2, cond, pred, readout = fit_predict_score(
                    cal_rates, cal_velocity, query_rates_all, v9_target,
                )
                results[budget][view].append(r2)
                conditioning[budget][view].append(cond)

                print(f"[{time.strftime('%H:%M:%S')}] SUA M={budget} {view} "
                      f"session {si+1}/15 ({row.session_id}): R2={r2:.4f} "
                      f"rows={cond['rows']} feat={cond['features']} "
                      f"rank={cond['design_rank']} traceH={cond['trace_hat']:.1f}",
                      file=sys.stderr)

            if view == "sua":
                session_info.append({"asset_id": row.asset_id, "session_id": row.session_id})

            del record, _rebuilt

    elapsed = time.monotonic() - t0
    print(f"\n[{time.strftime('%H:%M:%S')}] SUA done in {elapsed:.0f}s", file=sys.stderr)

    # Build contrasts
    all_results = {}
    for view in VIEWS:
        for budget in SUA_BUDGETS:
            ridge_scores = np.array(results[budget][view])
            t4_scores = t4_per_session[view][budget]
            delta = t4_scores - ridge_scores
            ds = {
                "mean_r2": float(ridge_scores.mean()),
                "median_r2": float(np.median(ridge_scores)),
                "min_r2": float(ridge_scores.min()),
                "neg_count": int(np.sum(ridge_scores < 0)),
                "delta_t4_minus_ridge_mean": float(delta.mean()),
                "delta_t4_minus_ridge_median": float(np.median(delta)),
                "delta_sign_t4_higher": int(np.sum(delta > 0)),
                "bootstrap": core.bootstrap_sessions(delta),
                "per_session_r2": {cohort[i].asset_id: float(ridge_scores[i]) for i in range(15)},
                "per_session_delta": {cohort[i].asset_id: float(delta[i]) for i in range(15)},
                "conditioning": conditioning[budget][view],
            }
            all_results[f"sua_{view}_M{budget}"] = ds

    return all_results, session_info


# ===== RT =====

def run_rt():
    from streaming_calibration_exp.src.data.rt_k4_loader import load_rt_session, find_rt_sessions

    t0 = time.monotonic()
    print(f"\n[{time.strftime('%H:%M:%S')}] Loading RT sessions...", file=sys.stderr)
    rt_files = sorted(find_rt_sessions(RT_DATA_DIR))
    print(f"  Found {len(rt_files)} RT sessions", file=sys.stderr)

    results = []
    conditioning_list = []
    session_names = []

    for si, nwb_path in enumerate(rt_files):
        session_name = nwb_path.stem.replace("_behavior+ecephys", "").replace("sub-C_", "")
        data = load_rt_session(nwb_path)
        neural = data["neural"]
        covariates = data["covariates"]
        trial_change = data["trial_change"]
        eval_mask = data["eval_mask"]
        segment_id = data["k4_segment_id"]

        # First 24 trial-rows boundary
        trial_starts = np.flatnonzero(trial_change)
        assert len(trial_starts) > RT_TRIALS, f"RT session has only {len(trial_starts)} trials"
        support_stop_bin = int(trial_starts[RT_TRIALS])

        # Support segments: within first 24 trial-rows
        support_mask = np.zeros(len(segment_id), dtype=bool)
        support_mask[:support_stop_bin] = True
        support_seg_id = np.where(support_mask, segment_id, -1)

        # Use 5-bin window rates for calibration (same as query), with segment-level velocity labels
        # Each 5-bin window in segment j gets segment j's mean velocity
        support_eval = np.zeros(len(eval_mask), dtype=bool)
        support_eval[:support_stop_bin] = eval_mask[:support_stop_bin]
        cal_rates, cal_target_bins = core.block_query_rates_rt(
            neural, support_eval, support_seg_id, block_bins=5,
        )
        # Assign segment-mean velocity to each calibration window
        cal_velocity = np.empty((cal_rates.shape[0], 2), dtype=np.float64)
        _, seg_mean_vel, _ = core.segment_rates_and_velocity(neural, covariates, support_seg_id)
        unique_segs = sorted(int(s) for s in np.unique(support_seg_id) if int(s) >= 0)
        seg_to_idx = {s: i for i, s in enumerate(unique_segs)}
        for wi, tb in enumerate(cal_target_bins):
            seg = int(segment_id[tb])
            cal_velocity[wi] = seg_mean_vel[seg_to_idx[seg]]

        # Query: 5-bin causal blocks at all eval bins
        query_rates, target_bins = core.block_query_rates_rt(
            neural, eval_mask, segment_id, block_bins=5,
        )
        query_targets = covariates[target_bins]

        r2, cond, pred, readout = fit_predict_score(
            cal_rates, cal_velocity, query_rates, query_targets,
        )
        results.append(r2)
        conditioning_list.append(cond)
        session_names.append(session_name)

        print(f"[{time.strftime('%H:%M:%S')}] RT {session_name}: "
              f"R2={r2:.4f} calib_windows={cal_rates.shape[0]} N={cond['features']} "
              f"rank={cond['design_rank']} traceH={cond['trace_hat']:.1f} "
              f"query={query_rates.shape[0]}",
              file=sys.stderr)

    elapsed = time.monotonic() - t0
    results = np.array(results)
    print(f"\n[{time.strftime('%H:%M:%S')}] RT done in {elapsed:.0f}s, mean R2={results.mean():.4f}", file=sys.stderr)

    # RT sealed reference: Full carrier 0.4419 (single seed, 15-fold LOSO)
    # Note: the RT result is cross-session LOSO, our ridge is within-session.
    # The comparison is descriptive only.
    delta_vs_full = results - 0.4419  # ridge - carrier

    return {
        "rt_M24": {
            "mean_r2": float(results.mean()),
            "median_r2": float(np.median(results)),
            "min_r2": float(results.min()),
            "neg_count": int(np.sum(results < 0)),
            "delta_ridge_minus_carrier_full_mean": float(delta_vs_full.mean()),
            "delta_sign_ridge_higher": int(np.sum(delta_vs_full > 0)),
            "per_session_r2": {session_names[i]: float(results[i]) for i in range(len(results))},
            "per_session_segments": {session_names[i]: conditioning_list[i]["rows"] for i in range(len(results))},
            "conditioning": conditioning_list,
            "note": "RT sealed carrier (0.4419) is a cross-session nested-LOSO single-seed result. "
                    "This ridge is a within-session per-session refit. The comparison is descriptive.",
        }
    }


# ===== M2 =====

def run_m2():
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb

    t0 = time.monotonic()
    print(f"\n[{time.strftime('%H:%M:%S')}] Loading M2 sessions...", file=sys.stderr)

    m2_files = sorted(M2_DATA_DIR.glob("*.nwb"))
    assert len(m2_files) == 6, f"Expected 6 M2 held-out sessions, found {len(m2_files)}"
    print(f"  Found {len(m2_files)} M2 sessions", file=sys.stderr)

    results = []
    conditioning_list = []
    session_names = []

    for nwb_path in m2_files:
        session_name = nwb_path.stem.replace("sub-MonkeyN-held-out-calib_", "")
        neural, covariates, trial_change, eval_mask = load_nwb(str(nwb_path), FalconTask.m2)
        neural = np.asarray(neural, dtype=np.float32)
        covariates = np.asarray(covariates, dtype=np.float32)
        trial_change = np.asarray(trial_change, dtype=bool)
        eval_mask = np.asarray(eval_mask, dtype=bool)

        # Trial boundaries from trial_change
        trial_starts = np.flatnonzero(trial_change)
        trial_bounds = []
        for i in range(min(M2_TRIALS, len(trial_starts))):
            start = int(trial_starts[i])
            stop = int(trial_starts[i + 1]) if i + 1 < len(trial_starts) else len(trial_change)
            trial_bounds.append((start, stop))

        # Trial-mean velocities for labeling
        trial_velocities = np.array([covariates[s:e].mean(axis=0) for s, e in trial_bounds])

        # Calibration: 50-bin window rates from first 24 trials, with trial-level velocity labels
        bounds_arr = np.array(trial_bounds, dtype=np.int64)
        # Get all valid 50-bin causal windows within the first 24 trials
        cal_target_bins = []
        for ti, (ts, te) in enumerate(trial_bounds):
            for b in range(ts + 49, te):
                cal_target_bins.append((b, ti))
        cal_target_bins = np.array([(b, ti) for b, ti in cal_target_bins], dtype=np.int64)

        if cal_target_bins.shape[0] > 0:
            cal_rates = core.window_query_rates(neural, cal_target_bins[:, 0], window_size=50)
            cal_velocity = trial_velocities[cal_target_bins[:, 1]]
        else:
            raise RuntimeError(f"M2 session {session_name}: no calibration windows")

        # Query: 50-bin causal window rate at each eval bin
        eval_bins = np.flatnonzero(eval_mask)
        query_bins = np.array([b for b in eval_bins if b >= 49], dtype=np.int64)
        query_rates = core.window_query_rates(neural, query_bins, window_size=50)
        query_targets = covariates[query_bins]

        r2, cond, pred, readout = fit_predict_score(
            cal_rates, cal_velocity, query_rates, query_targets,
        )
        results.append(r2)
        conditioning_list.append(cond)
        session_names.append(session_name)

        print(f"[{time.strftime('%H:%M:%S')}] M2 {session_name}: "
              f"R2={r2:.4f} trials={M2_TRIALS} N={cond['features']} "
              f"rank={cond['design_rank']} traceH={cond['trace_hat']:.1f} "
              f"query={query_rates.shape[0]}",
              file=sys.stderr)

    elapsed = time.monotonic() - t0
    results = np.array(results)
    print(f"\n[{time.strftime('%H:%M:%S')}] M2 done in {elapsed:.0f}s, mean R2={results.mean():.4f}", file=sys.stderr)

    return {
        "m2_M24": {
            "mean_r2": float(results.mean()),
            "median_r2": float(np.median(results)),
            "min_r2": float(results.min()),
            "neg_count": int(np.sum(results < 0)),
            "per_session_r2": {session_names[i]: float(results[i]) for i in range(len(results))},
            "conditioning": conditioning_list,
            "sealed_references": {
                "t4": 0.2268, "ridge_w50_dense": 0.1139, "k4": 0.2458,
            },
            "note": "M2 ridge-W50 (0.1139) uses dense 50-bin windows. This trial-level ridge uses "
                    "24 trial-mean rates. T4 (0.2268) uses 24 trial-level velocity labels.",
        }
    }


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # Run all three cohorts
    sua_results, session_info = run_sua()
    all_results.update(sua_results)

    rt_results = run_rt()
    all_results.update(rt_results)

    m2_results = run_m2()
    all_results.update(m2_results)

    # Print summary
    print("\n" + "=" * 90, file=sys.stderr)
    print("TRIAL-LEVEL RATE RIDGE SUMMARY", file=sys.stderr)
    print("=" * 90, file=sys.stderr)

    print("\nSUA (trial-level ridge vs T4)", file=sys.stderr)
    print(f"{'Budget':<8} {'View':<12} {'Ridge mean':<12} {'T4 mean':<12} {'T4-Ridge Δ':<12} {'T4 higher':<10}", file=sys.stderr)
    for view in VIEWS:
        for budget in SUA_BUDGETS:
            key = f"sua_{view}_M{budget}"
            r = all_results[key]
            t4_mean = {
                ("sua", 15): 0.3381, ("sua", 30): 0.3582, ("sua", 50): 0.3568,
                ("pseudo_mua", 15): 0.2882, ("pseudo_mua", 30): 0.3053, ("pseudo_mua", 50): 0.3061,
            }[(view, budget)]
            print(f"M={budget:<6} {view:<12} {r['mean_r2']:<12.4f} {t4_mean:<12.4f} "
                  f"{t4_mean - r['mean_r2']:<+12.4f} {r['delta_sign_t4_higher']}/15",
                  file=sys.stderr)

    print(f"\nRT M24: ridge mean={all_results['rt_M24']['mean_r2']:.4f}, "
          f"carrier full=0.4419", file=sys.stderr)
    print(f"\nM2 M24: ridge mean={all_results['m2_M24']['mean_r2']:.4f}, "
          f"T4=0.2268, ridge-W50=0.1139", file=sys.stderr)

    # Write receipt
    receipt = {
        "schema": "trial_level_ridge_v1",
        "status": "COMPLETED_CPU_ONLY",
        "date": "2026-08-11",
        "equal_information_definition": (
            "Calibration uses ONLY trial-level mean firing rates [M,N] and trial-level mean "
            "velocity [M,2]. Same M trials and same per-trial information that T4/carrier consumes. "
            "Ridge solver reused unchanged (mean-normalized lambda=1)."
        ),
        "results": all_results,
        "non_interference_statement": {
            "gpu_used": False,
            "cuda_visible_devices": "",
            "thread_caps": "OMP/MKL/OPENBLAS=4, nice -n 10",
            "output_directory": str(OUTPUT_ROOT),
            "processes_signalled": [],
            "watched_directories_written": [],
        },
    }

    receipt_bytes = core.canonical_bytes(receipt)
    receipt_path = OUTPUT_ROOT / "trial_level_ridge_receipt.json"
    receipt_path.write_bytes(receipt_bytes)
    os.chmod(receipt_path, 0o444)
    print(f"\nReceipt: {receipt_path}", file=sys.stderr)
    print(f"SHA-256: {core.sha256_bytes(receipt_bytes)}", file=sys.stderr)


if __name__ == "__main__":
    main()
