#!/usr/bin/env python3
"""CPU-only reviewer diagnostic: invert RT's M24 per-channel velocity fit.

This is deliberately *not* a carrier-alone decoder: it uses both the live
neural activity and the support-fitted AFC4 forward carrier.  It is a
reviewer-facing diagnostic only.  Its result, favorable or unfavorable, is
not an RT L-D gate and never authorizes or blocks an L-D run.

The only target data read are the already-established RT development M24
support and the same chronological post-M24 report/query span used by the
matched Full/MB4 endpoint.  No GPU, optimizer, checkpoint selection, formal
heldout endpoint, full covariance, or target-session hyperparameter tuning is
used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STREAMING = ROOT / "streaming_calibration_exp"
if str(STREAMING) not in sys.path:
    sys.path.insert(0, str(STREAMING))

from src.data.falcon_k4_features import (  # noqa: E402
    K4_ACTIVE_EPSILON,
    K4_BEHAVIOR_LEAD_BINS,
    K4_BLOCK_WIDTH_BINS,
    K4_RAW_BIN_MS,
    collect_k4_support_blocks,
)
from src.data.rt_k4_loader import find_rt_sessions, load_rt_session  # noqa: E402


DATA_DIR = ROOT / "sua_exploration/data/dandi_000688/sub-C"
REFERENCE_AGGREGATE = ROOT / "sua_exploration/results/rt_mb4_matched_clean_nested_v1/RT_MB4_MATCHED_FULL15_AGGREGATE_v1.json"
MB4_ROOT = ROOT / "sua_exploration/results/rt_mb4_matched_clean_nested_v1/imported_cells/afc4_mb4"
RESULT_DIR = ROOT / "sua_exploration/results/rt_closed_form_inverse_diagnostic_v1"
PREFLIGHT = RESULT_DIR / "RT_CLOSED_FORM_INVERSE_CPU_PREFLIGHT_v1.json"
FINAL = RESULT_DIR / "RT_CLOSED_FORM_INVERSE_DIAGNOSTIC_v1.json"
LAMBDA = 0.0  # fixed analytic inverse; not selected from a target session.
VARIANCE_FLOOR = 1.0e-9  # numeric floor, fixed ex ante; no shrink parameter.
SHUFFLE_SEED = 20260810


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"append-only diagnostic refuses overwrite: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o444)


def _r2(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[0] < 3:
        raise ValueError("R2 needs matching [samples,2] arrays with at least three rows")
    denom = float(np.square(target - target.mean(axis=0, keepdims=True)).sum())
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError("query target has no finite variance")
    return float(1.0 - np.square(target - prediction).sum() / denom)


def fit_forward(rates: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel forward OLS: r_i = b_i + W_i v, with diagonal Sigma."""
    r, v = np.asarray(rates, dtype=np.float64), np.asarray(velocity, dtype=np.float64)
    if r.ndim != 2 or v.shape != (r.shape[0], 2) or r.shape[0] < 3:
        raise ValueError("forward fit needs rates=[blocks,channels], velocity=[blocks,2]")
    design = np.column_stack((np.ones(r.shape[0]), v))
    if np.linalg.matrix_rank(design) != 3:
        raise ValueError("M24 support forward design is rank deficient")
    coefficients, *_ = np.linalg.lstsq(design, r, rcond=None)
    b, w = coefficients[0], coefficients[1:].T
    residual = r - (b[None, :] + v @ w.T)
    # Diagonal only: one residual variance per channel.  This is not a
    # full-covariance estimate and no shrinkage coefficient is introduced.
    sigma_diag = np.maximum(np.mean(np.square(residual), axis=0), VARIANCE_FLOOR)
    return b, w, sigma_diag


def inverse_velocity(rates: np.ndarray, b: np.ndarray, w: np.ndarray, sigma_diag: np.ndarray, *, ridge_lambda: float = LAMBDA) -> np.ndarray:
    """Apply (WᵀΣ⁻¹W+λI)⁻¹WᵀΣ⁻¹(r-b), with *diagonal* Σ only."""
    r, b, w, diagonal = (np.asarray(rates, dtype=np.float64), np.asarray(b, dtype=np.float64), np.asarray(w, dtype=np.float64), np.asarray(sigma_diag, dtype=np.float64))
    if r.ndim != 2 or b.shape != (r.shape[1],) or w.shape != (r.shape[1], 2):
        raise ValueError("inverse input shapes do not describe channels x 2 forward weights")
    if diagonal.ndim != 1 or diagonal.shape != (r.shape[1],) or not np.isfinite(diagonal).all() or np.any(diagonal <= 0.0):
        raise ValueError("Sigma must be a positive diagonal vector; full covariance is forbidden")
    if not np.isfinite(ridge_lambda) or ridge_lambda < 0.0:
        raise ValueError("ridge lambda must be finite and non-negative")
    precision = 1.0 / diagonal
    system = w.T @ (precision[:, None] * w) + float(ridge_lambda) * np.eye(2)
    rhs = (r - b[None, :]) @ (precision[:, None] * w)
    if np.linalg.matrix_rank(system) != 2:
        raise ValueError("inverse system is rank deficient")
    return np.linalg.solve(system, rhs.T).T


def _query_blocks(raw: Mapping[str, Any], *, support_trials: int = 24) -> tuple[np.ndarray, np.ndarray, int]:
    """Event-qualified query blocks entirely in the existing post-M24 span."""
    neural = np.asarray(raw["neural"], dtype=np.float64)
    velocity = np.asarray(raw["covariates"], dtype=np.float64)
    changes = np.asarray(raw["trial_change"], dtype=bool)
    segment_ids = np.asarray(raw["k4_segment_id"], dtype=np.int64)
    starts = np.flatnonzero(changes)
    if starts.size <= support_trials:
        raise ValueError("no post-M24 query trial boundary")
    query_start = int(starts[support_trials])
    ends = np.r_[starts[1:], len(changes)]
    active = ~np.all(np.abs(velocity) < K4_ACTIVE_EPSILON, axis=1)
    rates: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for start, end in zip(starts[support_trials:], ends[support_trials:]):
        for left in range(int(start), int(end) - K4_BLOCK_WIDTH_BINS - K4_BEHAVIOR_LEAD_BINS + 1, K4_BLOCK_WIDTH_BINS):
            right, y_left, y_right = left + K4_BLOCK_WIDTH_BINS, left + K4_BEHAVIOR_LEAD_BINS, left + K4_BEHAVIOR_LEAD_BINS + K4_BLOCK_WIDTH_BINS
            groups = segment_ids[left:y_right]
            if groups.size != K4_BLOCK_WIDTH_BINS + K4_BEHAVIOR_LEAD_BINS or groups[0] < 0 or not np.all(groups == groups[0]):
                continue
            if not active[left:right].all() or not active[y_left:y_right].all():
                continue
            rates.append(neural[left:right].sum(axis=0) / (K4_BLOCK_WIDTH_BINS * K4_RAW_BIN_MS / 1000.0))
            labels.append(velocity[y_left:y_right].mean(axis=0))
    if len(rates) < 3:
        raise ValueError("matched post-M24 report span contains fewer than three event-qualified blocks")
    return np.asarray(rates), np.asarray(labels), query_start


def _fold_references() -> dict[int, dict[str, Any]]:
    aggregate = _json(REFERENCE_AGGREGATE)
    if aggregate.get("arm") != "afc4_mb4" or aggregate.get("full_comparator_arm") != "afc4_vel":
        raise ValueError("existing endpoint aggregate is not the matched RT Full/MB4 reference")
    result: dict[int, dict[str, Any]] = {}
    for row in aggregate.get("rows", []):
        fold = int(row["fold"])
        outer = MB4_ROOT / f"fold_{fold:02d}/seed_42/outer_target_eval.json"
        split = MB4_ROOT / f"fold_{fold:02d}/seed_42/fit/split_manifest.json"
        outer_value = _json(outer)
        if outer_value.get("outer_loso_fold") != fold or outer_value.get("outer_target_session") != row.get("target_session"):
            raise ValueError(f"matched MB4 reference split drift at fold {fold}")
        if outer_value.get("query_start_trial") != 24 or outer_value.get("window_size") != 50 or outer_value.get("target_backpropagation") is not False:
            raise ValueError(f"matched MB4 reference query contract drift at fold {fold}")
        result[fold] = {"row": row, "outer": outer, "split": split, "outer_value": outer_value}
    if sorted(result) != list(range(15)):
        raise ValueError("matched Full/MB4 reference must provide all 15 RT folds")
    return result


def prepare() -> None:
    if RESULT_DIR.exists():
        raise FileExistsError("diagnostic receipt directory is append-only")
    refs = _fold_references()
    sessions = {Path(path).name: Path(path) for path in find_rt_sessions(DATA_DIR)}
    if len(sessions) != 15:
        raise ValueError("RT development data must contain exactly 15 sessions")
    rows = []
    for fold, ref in refs.items():
        outer = ref["outer_value"]
        data = Path(str(outer["outer_target_path"])).resolve()
        if data.name not in sessions or sessions[data.name].resolve() != data:
            raise ValueError(f"fold {fold}: endpoint target is outside the indexed RT development data")
        rows.append({"fold": fold, "session": outer["outer_target_session"], "data_path": str(data), "data_sha256": _sha(data), "split_manifest_path": str(ref["split"].resolve()), "split_manifest_sha256": _sha(ref["split"]), "matched_mb4_outer_eval_path": str(ref["outer"].resolve()), "matched_mb4_outer_eval_sha256": _sha(ref["outer"]), "full_r2_50_window": ref["row"]["full_r2"], "mb4_r2_50_window": ref["row"]["mb4_r2"], "matched_full_mb4_query_windows": outer["query_windows_evaluated"]})
    payload = {
        "schema": "rt_closed_form_inverse_cpu_preflight_v1",
        "status": "CPU_PREFLIGHT_READY_NOT_A_GATE",
        "purpose": "reviewer-facing closed-form inverse diagnostic; not carrier-alone and not an L-D hard-kill gate",
        "runner_path": str(Path(__file__).resolve()), "runner_sha256": _sha(Path(__file__).resolve()),
        "test_path": str((ROOT / "streaming_calibration_exp/tests/test_rt_closed_form_inverse_diagnostic.py").resolve()), "test_sha256": _sha(ROOT / "streaming_calibration_exp/tests/test_rt_closed_form_inverse_diagnostic.py"),
        "data_scope": "RT development sub-C only; no new formal scope",
        "formal_heldout_opened": False, "gpu_launched": False, "optimizer_or_backpropagation": False,
        "support_contract": {"trials": [0, 24], "forward": "per-channel r_i=b_i+W_i v on event-qualified raw 5-bin blocks", "carrier": "AFC4/Full [W_x,W_y,||W||,b] uses live activity plus carrier"},
        "query_contract": {"trials": [24, "end"], "same_matched_full_mb4_report_span": True, "window_size_bins_reference": 50, "inverse_query_blocks": "event-qualified raw 5-bin blocks entirely after the same M24 boundary"},
        "inverse_contract": {"formula": "(W^T Sigma^-1 W+lambda I)^-1 W^T Sigma^-1(r-b)", "sigma": "support-residual diagonal only; full covariance forbidden", "lambda": LAMBDA, "lambda_selection": "fixed ex ante; no target-session selection", "shrinkage": "none", "variance_floor": VARIANCE_FLOOR},
        "sanity_controls": ["deterministic channel-row shuffle of W", "zero-W inverse"],
        "comparison_policy": "PV50/Ridge50 are unavailable for this RT matched report span; Full/MB4 50-window R2 are listed only as separately-windowed context and never pooled or differenced with this 5-bin diagnostic R2.",
        "reference_aggregate_path": str(REFERENCE_AGGREGATE.resolve()), "reference_aggregate_sha256": _sha(REFERENCE_AGGREGATE),
        "sessions": rows,
    }
    _exclusive_json(PREFLIGHT, payload)


def _validate_preflight() -> dict[str, Any]:
    preflight = _json(PREFLIGHT)
    if preflight.get("runner_sha256") != _sha(Path(__file__).resolve()) or preflight.get("test_sha256") != _sha(ROOT / "streaming_calibration_exp/tests/test_rt_closed_form_inverse_diagnostic.py"):
        raise ValueError("diagnostic runner/test hash drift")
    if preflight.get("status") != "CPU_PREFLIGHT_READY_NOT_A_GATE" or preflight.get("formal_heldout_opened") is not False:
        raise ValueError("preflight endpoint scope drift")
    return preflight


def execute() -> None:
    if FINAL.exists():
        raise FileExistsError("final diagnostic receipt is append-only")
    preflight = _validate_preflight()
    results = []
    for row in preflight["sessions"]:
        raw = load_rt_session(Path(row["data_path"]))
        if str(raw.get("session_name")) != row["session"]:
            raise ValueError("loaded session identity drift")
        support = collect_k4_support_blocks(raw["neural"], raw["covariates"], raw["trial_change"], calibration_n_trials=24, segment_ids=raw["k4_segment_id"])
        b, w, sigma = fit_forward(support.rates, support.velocity)
        query_rates, query_velocity, query_start = _query_blocks(raw)
        prediction = inverse_velocity(query_rates, b, w, sigma)
        rng = np.random.default_rng(SHUFFLE_SEED + int(row["fold"]))
        permutation = rng.permutation(w.shape[0])
        if np.array_equal(permutation, np.arange(w.shape[0])):
            permutation = np.roll(permutation, 1)
        shuffled = inverse_velocity(query_rates, b, w[permutation], sigma)
        # With W=0 the analytic observation model contains no velocity
        # information.  The closed-form limiting control is identically zero;
        # do not pretend its singular 2x2 normal equation can be inverted.
        zero = np.zeros_like(query_velocity)
        results.append({**row, "support_blocks": int(support.rates.shape[0]), "support_candidate_blocks": support.candidate_blocks, "support_segment_qualified_blocks": support.segment_qualified_blocks, "support_design_rank": int(np.linalg.matrix_rank(np.column_stack((np.ones(support.velocity.shape[0]), support.velocity)))), "query_start_raw_bin": query_start, "query_blocks": int(query_rates.shape[0]), "r2_inverse_5bin": _r2(prediction, query_velocity), "r2_shuffle_w_5bin": _r2(shuffled, query_velocity), "r2_zero_w_5bin": _r2(zero, query_velocity), "sigma_diagonal_min": float(sigma.min()), "sigma_diagonal_max": float(sigma.max()), "full_mb4_r2_window_policy": "different 50-bin model window; contextual only, never mixed with inverse 5-bin R2"})
    inverse_values = np.asarray([row["r2_inverse_5bin"] for row in results])
    shuffle_values = np.asarray([row["r2_shuffle_w_5bin"] for row in results])
    zero_values = np.asarray([row["r2_zero_w_5bin"] for row in results])
    payload = {"schema": "rt_closed_form_inverse_diagnostic_v1", "status": "PASS_CPU_REVIEWER_DIAGNOSTIC_NOT_A_GATE", "preflight_path": str(PREFLIGHT.resolve()), "preflight_sha256": _sha(PREFLIGHT), "mode": "closed_form_inverse_uses_live_activity_plus_support_fitted_carrier_not_carrier_alone", "formal_heldout_opened": False, "gpu_launched": False, "target_backpropagation": False, "target_session_hyperparameter_selection": False, "ld_hard_kill_gate": False, "interpretation": "Reviewer-facing diagnostic only. It cannot automatically block or authorize RT L-D regardless of result.", "comparison_policy": preflight["comparison_policy"], "per_session": results, "aggregate_5bin_only": {"mean_inverse_r2": float(inverse_values.mean()), "median_inverse_r2": float(np.median(inverse_values)), "mean_shuffle_w_r2": float(shuffle_values.mean()), "mean_zero_w_r2": float(zero_values.mean()), "mean_inverse_minus_shuffle": float((inverse_values - shuffle_values).mean()), "mean_inverse_minus_zero": float((inverse_values - zero_values).mean()), "session_count": len(results)}}
    _exclusive_json(FINAL, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="write append-only CPU preflight receipt")
    parser.add_argument("--execute", action="store_true", help="run CPU diagnostic only after preflight")
    args = parser.parse_args()
    if args.prepare == args.execute:
        parser.error("choose exactly one of --prepare or --execute")
    if args.prepare:
        prepare()
    else:
        execute()


if __name__ == "__main__":
    main()
