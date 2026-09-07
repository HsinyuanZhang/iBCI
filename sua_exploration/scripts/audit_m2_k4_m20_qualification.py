#!/usr/bin/env python3
"""CPU-only M2 M20/M24 qualification audit for the disjoint held-out protocol.

This is deliberately a held-in-only feasibility check.  It uses the first 20
chronological calibration trials of each of the seven M2 held-in sessions:
The frozen M24 mode uses trials 0:12 to fit an OLS movement carrier and trials
12:24 untouched for prediction.  The retained M20 mode is an earlier
qualification receipt only, not a GPU-launch selector.
rate prediction.  It neither constructs a neural-network feature nor opens a
held-out-calibration or EvalAI file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sua_exploration"), str(ROOT / "streaming_calibration_exp")]

from falcon_challenge.config import FalconTask  # noqa: E402
from falcon_challenge.dataloaders import load_nwb  # noqa: E402
from mc_maze.general_carrier import (  # noqa: E402
    CarrierFit,
    deterministic_weight_shuffle,
    fit_encoding,
    mean_squared_error,
    predict_encoding,
)


TASK = "m2"
FIT_TRIALS = 10
PREDICT_TRIALS = 10
TOTAL_TRIALS = FIT_TRIALS + PREDICT_TRIALS
RAW_BIN_MS = 20
BLOCK_WIDTH_BINS = 5
BEHAVIOR_LEAD_BINS = 2
ACTIVE_EPSILON = 1e-3
N_NULLS = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def active_trial_blocks(neural: np.ndarray, velocity: np.ndarray, trial_change: np.ndarray) -> dict[str, np.ndarray]:
    neural = np.asarray(neural, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    trial_change = np.asarray(trial_change, dtype=bool)
    if neural.ndim != 2 or velocity.shape != (neural.shape[0], 2) or trial_change.shape != (neural.shape[0],):
        raise ValueError("M2 raw neural, 2-D velocity, and trial boundary arrays disagree")
    starts = np.flatnonzero(trial_change)
    if len(starts) < TOTAL_TRIALS:
        raise ValueError(f"requires at least {TOTAL_TRIALS} chronological trials, got {len(starts)}")
    ends = np.r_[starts[1:], len(trial_change)]
    active = ~np.all(np.abs(velocity) < ACTIVE_EPSILON, axis=1)
    rate, behavior, trials = [], [], []
    for trial, (start, end) in enumerate(zip(starts[:TOTAL_TRIALS], ends[:TOTAL_TRIALS])):
        for left in range(int(start), int(end) - BLOCK_WIDTH_BINS - BEHAVIOR_LEAD_BINS + 1, BLOCK_WIDTH_BINS):
            right = left + BLOCK_WIDTH_BINS
            y_left, y_right = left + BEHAVIOR_LEAD_BINS, left + BEHAVIOR_LEAD_BINS + BLOCK_WIDTH_BINS
            if active[left:right].all() and active[y_left:y_right].all():
                rate.append(neural[left:right].sum(axis=0) / (BLOCK_WIDTH_BINS * RAW_BIN_MS / 1000.0))
                behavior.append(velocity[y_left:y_right].mean(axis=0))
                trials.append(trial)
    if not rate:
        raise ValueError("no active movement-aligned blocks in first 20 trials")
    return {
        "rate": np.asarray(rate, dtype=np.float64),
        "behavior": np.asarray(behavior, dtype=np.float64),
        "trial": np.asarray(trials, dtype=np.int64),
    }


def process(path: Path) -> dict:
    if "held-out" in str(path).lower():
        raise ValueError(f"qualification audit rejects held-out input: {path}")
    neural, velocity, trial_change, _ = load_nwb(path, FalconTask.m2)
    blocks = active_trial_blocks(neural, velocity, trial_change)
    a, b = blocks["trial"] < FIT_TRIALS, blocks["trial"] >= FIT_TRIALS
    if not a.any() or not b.any():
        raise ValueError(f"{path.name}: missing fit or untouched prediction blocks")
    rate_a, y_a, seg_a = blocks["rate"][a], blocks["behavior"][a], blocks["trial"][a]
    rate_b, y_b, seg_b = blocks["rate"][b], blocks["behavior"][b], blocks["trial"][b]
    fit = fit_encoding(rate_a, y_a, np.ones(len(rate_a), dtype=bool), seg_a, lag_bins=0, alpha=0.0)
    indices = np.arange(len(rate_b), dtype=np.int64)
    k_mse = mean_squared_error(rate_b, predict_encoding(fit, y_b, indices, indices))
    base_mse = mean_squared_error(rate_b, np.broadcast_to(rate_a.mean(axis=0), rate_b.shape))
    null_mse = np.asarray([
        mean_squared_error(rate_b, predict_encoding(deterministic_weight_shuffle(fit, seed), y_b, indices, indices))
        for seed in range(N_NULLS)
    ])
    fit_b = fit_encoding(rate_b, y_b, np.ones(len(rate_b), dtype=bool), seg_b, lag_bins=0, alpha=0.0)
    corr = float(np.corrcoef(fit.weights.ravel(), fit_b.weights.ravel())[0, 1])
    design20 = np.column_stack([np.ones(len(blocks["behavior"])), blocks["behavior"]])
    rank = int(np.linalg.matrix_rank(design20))
    condition = float(np.linalg.cond(design20)) if rank == 3 else float("inf")
    if not np.isfinite(corr) or rank != 3 or not np.isfinite(condition):
        raise ValueError(f"{path.name}: invalid M20 stability/design audit")
    return {
        "session": path.name,
        "source_path": str(path.resolve()), "source_sha256": sha256(path),
        "fit_trials": [0, FIT_TRIALS], "prediction_trials": [FIT_TRIALS, TOTAL_TRIALS],
        "n_fit_blocks": int(len(rate_a)), "n_prediction_blocks": int(len(rate_b)),
        "all_support_active_blocks": int(len(blocks["rate"])), "all_support_design_rank": rank,
        "all_support_design_condition": condition,
        "kreg_mse": k_mse, "rate_only_fit_baseline_mse": base_mse,
        "kreg_over_baseline_ratio": k_mse / base_mse,
        "w_only_null_median_mse": float(np.median(null_mse)),
        "kreg_over_w_only_null_median_ratio": k_mse / float(np.median(null_mse)),
        "beats_all_100_w_only_nulls": bool(np.all(k_mse < null_mse)),
        "w_only_null_count": N_NULLS, "flattened_W_A_B_correlation": corr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "SPINT-main/data/000953")
    parser.add_argument("--protocol", choices=("m20", "m24"), default="m20")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    global FIT_TRIALS, PREDICT_TRIALS, TOTAL_TRIALS
    if args.protocol == "m24":
        FIT_TRIALS, PREDICT_TRIALS, TOTAL_TRIALS = 12, 12, 24
    else:
        FIT_TRIALS, PREDICT_TRIALS, TOTAL_TRIALS = 10, 10, 20
    if args.out is None:
        args.out = ROOT / "sua_exploration/results/general_carrier_proxy_v1" / f"audit_m2_{args.protocol}_heldin_v2.json"
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite qualification artifact: {args.out}")
    paths = sorted(args.data_dir.glob("**/*held-in-calib*.nwb"))
    if len(paths) != 7:
        raise ValueError(f"expected exactly seven M2 held-in sessions, found {len(paths)}")
    rows = [process(path) for path in paths]
    from scipy.stats import wilcoxon

    k = np.asarray([row["kreg_mse"] for row in rows])
    base = np.asarray([row["rate_only_fit_baseline_mse"] for row in rows])
    null = np.asarray([row["w_only_null_median_mse"] for row in rows])
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": f"held-in-only M2 M{TOTAL_TRIALS} qualification; no held-out or EvalAI file opened",
        "formal_heldout_evaluated": False, "uses_gpu": False, "uses_backward_gradients": False,
        "protocol": {
            "name": args.protocol,
            "fit_trials": f"0:{FIT_TRIALS}", "prediction_trials": f"{FIT_TRIALS}:{TOTAL_TRIALS}",
            "support_trials": TOTAL_TRIALS, "total_trials": TOTAL_TRIALS,
            "raw_bin_ms": RAW_BIN_MS, "block_width_bins": BLOCK_WIDTH_BINS,
            "behavior_lead_bins": BEHAVIOR_LEAD_BINS, "null": "100 deterministic W-only row shuffles; b unchanged",
        },
        "sessions": rows,
        "summary": {
            "n_sessions": len(rows),
            "mean_kreg_over_baseline_ratio": float(np.mean([row["kreg_over_baseline_ratio"] for row in rows])),
            "mean_kreg_over_w_only_null_median_ratio": float(np.mean([row["kreg_over_w_only_null_median_ratio"] for row in rows])),
            "kreg_beats_baseline_sessions": int(np.sum(k < base)),
            "kreg_beats_w_only_median_sessions": int(np.sum(k < null)),
            "all_7_sessions_beat_all_100_w_only_nulls": bool(all(row["beats_all_100_w_only_nulls"] for row in rows)),
            "wilcoxon_two_sided_kreg_vs_baseline_mse": float(wilcoxon(k, base, alternative="two-sided", method="exact").pvalue),
            "wilcoxon_two_sided_kreg_vs_w_only_median_mse": float(wilcoxon(k, null, alternative="two-sided", method="exact").pvalue),
            "W_A_B_correlations": [row["flattened_W_A_B_correlation"] for row in rows],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(args.out)


if __name__ == "__main__":
    main()
