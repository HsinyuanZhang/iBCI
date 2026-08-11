#!/usr/bin/env python3
"""Priority A2a v2: normalized 2×2 weighting control (explicit ``--run`` only).

This version intentionally never imports either invalidated A2 runner.  With no
arguments it performs the independent numerical contract and exits before any
dataset/receipt is opened.  A full CPU run is opt-in so tests are the first gate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sua_exploration.mc_maze import priority_a2_normalized_ridge_v2 as ridge

V9_RUN_ROOT = REPO_ROOT / "sua_exploration/results/dandi_000688_subm_co_three_arm_v9_formal_20260805"
NWB_ROOT = REPO_ROOT / "sua_exploration/data/dandi_000688"
OUTPUT_DIR = REPO_ROOT / "sua_exploration/results/trial_level_ridge_v1"
RECEIPT_PATH = OUTPUT_DIR / "priority_a2_weighting_control_v2_receipt.json"
REFERENCE_AGGREGATE = REPO_ROOT / "sua_exploration/results/dandi_000688_subm_f0_pv_ridge_v1_full_20260805/aggregate/endpoint_aggregate_torchmetrics151.json"
REFERENCE_BUDGET_RECEIPT = REPO_ROOT / "sua_exploration/results/ridge_budget_curve_v1/ridge_budget_curve_receipt.json"
VIEWS, BUDGETS, HISTORY_BINS = ("sua", "pseudo_mua"), (15, 30, 50), 50
SEALED_DENSE_M50_R2_ATOL = 5.0e-5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: object) -> str:
    return hashlib.sha256(__import__("numpy").ascontiguousarray(value).tobytes(order="C")).hexdigest()


def package_versions() -> dict[str, str]:
    names = ("numpy", "torch", "torchmetrics", "pynwb")
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "NOT_INSTALLED"
    return result


def _predict_query(record: object, numerical: object, readout: ridge.NormalizedWeightedRidge):
    import numpy as np
    pieces = []
    for starts in numerical.batched(record.valid_starts, 2048):
        pieces.append(ridge.predict_normalized_weighted_ridge(numerical.raw_window_features(record.neural, starts), readout))
    result = np.ascontiguousarray(np.concatenate(pieces), dtype=np.float64)
    ridge.require(result.shape == (record.valid_starts.size, 2), "query prediction shape drift")
    return result


def _sealed_dense_scores() -> tuple[dict[tuple[int, str, str], float], dict[str, str]]:
    """Load all six authoritative uniform-dense reference cells by budget/view."""
    payload = json.loads(REFERENCE_AGGREGATE.read_text(encoding="utf-8"))
    values = {(50, str(cell["asset_id"]), str(cell["view"])): float(cell["r2"])
              for cell in payload["cells"] if cell["arm"] == "ridge50"}
    ridge.require(len(values) == 30, "sealed Ridge50 reference must contain 30 dense M50 cells")
    budget_payload = json.loads(REFERENCE_BUDGET_RECEIPT.read_text(encoding="utf-8"))
    for budget in (15, 30):
        for view in VIEWS:
            per_session = budget_payload["per_session_r2"][str(budget)][view]
            ridge.require(len(per_session) == 15, f"authoritative dense M{budget}/{view} reference is incomplete")
            for asset, score in per_session.items():
                values[(budget, str(asset), view)] = float(score)
    ridge.require(len(values) == 90, "all six dense uniform reference cells must be available")
    return values, {"ridge50_m50_aggregate_sha256": sha256_file(REFERENCE_AGGREGATE), "ridge_budget_m15_m30_receipt_sha256": sha256_file(REFERENCE_BUDGET_RECEIPT)}


def _write_new_receipt(payload: dict[str, object]) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if RECEIPT_PATH.exists():
        raise FileExistsError(f"refusing to overwrite existing v2 receipt: {RECEIPT_PATH}")
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    # Exclusive creation is the final TOCTOU-safe no-overwrite check.
    with RECEIPT_PATH.open("xb") as handle:
        handle.write(raw)
    os.chmod(RECEIPT_PATH, 0o444)
    return hashlib.sha256(raw).hexdigest()


def run() -> str:
    """Execute A2a only after the pre-data numerical contract has passed."""
    numerical_contract = ridge.numerical_contract_self_test()
    if RECEIPT_PATH.exists():
        raise FileExistsError(f"refusing to start because v2 receipt already exists: {RECEIPT_PATH}")
    import numpy as np
    from sua_exploration.mc_maze import subm_v9_f0_pv_ridge as numerical
    from sua_exploration.mc_maze.multisession_datamodule import _compute_valid_starts
    from sua_exploration.mc_maze.subm_co_score_only_v2 import recompute_torchmetrics_r2_cpu
    from sua_exploration.scripts.run_subm_v9_f0_pv_ridge_controls import (
        _build_view_base, _nwb_path_and_pin, _runtime_owners, load_mean_std, load_v9_inputs, load_v9_target,
    )

    start_time = time.monotonic()
    sealed_scores, sealed_reference_hashes = _sealed_dense_scores()
    v9 = load_v9_inputs(V9_RUN_ROOT, repo_root=REPO_ROOT)
    ridge.require(len(v9.cohort) == 15, "A2a v2 requires the fixed 15-session cohort")
    owners = _runtime_owners(REPO_ROOT)
    behavior_stats = {view: load_mean_std(v9.behavior_normalizers[view][0], label=f"{view} behavior") for view in VIEWS}
    normalizer_hashes = {view: sha256_file(Path(v9.behavior_normalizers[view][0])) for view in VIEWS}
    cells: list[dict[str, object]] = []
    for cohort_row in v9.cohort:
        nwb_path = _nwb_path_and_pin(NWB_ROOT, cohort_row)
        nwb_sha = sha256_file(nwb_path)
        for view in VIEWS:
            mean, std = behavior_stats[view]
            record, _rebuilt, builder_trials, _bridge = _build_view_base(
                repo_root=REPO_ROOT, nwb_path=nwb_path, view=view, mean=mean, std=std, owners=owners,
            )
            query_target, query_target_sha, _ = load_v9_target(v9, cohort_row, view)
            ridge.require(sha256_array(query_target) == query_target_sha, "loader query target SHA does not bind returned query targets")
            for budget in BUDGETS:
                support = list(builder_trials[:budget])
                # Required before constructing sparse targets: no zero substitution is possible.
                ridge.require_target_directions(support)
                support_starts = np.ascontiguousarray(_compute_valid_starts(support, HISTORY_BINS), dtype=np.int64)
                overlap = np.intersect1d(support_starts, record.valid_starts, assume_unique=True)
                ridge.require(overlap.size == 0, "support/query row overlap is nonzero")
                trial_of_window = ridge.assign_windows_to_trials(support_starts, support, window_size=HISTORY_BINS)
                x = numerical.raw_window_features(record.neural, support_starts)
                dense_y = numerical.targets_at_window_end(record.behavior, support_starts)
                sparse_y = ridge.direction_targets_for_windows(support, trial_of_window)
                arm_inputs = {
                    "dense_uniform": (dense_y, np.ones(len(support_starts), dtype=np.float64)),
                    "dense_equal_trial": (dense_y, ridge.equal_trial_weights(trial_of_window, budget)),
                    "direction_uniform": (sparse_y, np.ones(len(support_starts), dtype=np.float64)),
                    "direction_equal_trial": (sparse_y, ridge.equal_trial_weights(trial_of_window, budget)),
                }
                cell: dict[str, object] = {
                    "asset_id": cohort_row.asset_id, "session_id": cohort_row.session_id, "view": view, "budget": budget,
                    "nwb_sha256": nwb_sha, "support_starts_sha256": sha256_array(support_starts),
                    "query_starts_sha256": sha256_array(record.valid_starts), "support_feature_sha256": sha256_array(x),
                    "query_feature_source_neural_sha256": sha256_array(record.neural),
                    "query_target_sha256": query_target_sha, "support_query_overlap_count": int(overlap.size),
                    "support_rows": int(support_starts.size), "query_rows": int(record.valid_starts.size),
                    "distinct_dense_targets": int(np.unique(dense_y, axis=0).shape[0]),
                    "distinct_direction_targets": int(np.unique(sparse_y, axis=0).shape[0]), "arms": {},
                }
                for arm, (target, weights) in arm_inputs.items():
                    readout = ridge.fit_normalized_weighted_ridge(x, target, weights, normalized_lambda=1.0)
                    prediction = _predict_query(record, numerical, readout)
                    score = float(recompute_torchmetrics_r2_cpu(prediction.astype(np.float32), query_target))
                    totals = np.bincount(trial_of_window, weights=weights, minlength=budget)
                    cell["arms"][arm] = {
                        "r2": score, "prediction_sha256": sha256_array(prediction), "support_target_sha256": sha256_array(target),
                        "weights_sha256": sha256_array(weights), "weight_sum": float(weights.sum()),
                        "per_trial_total_weight": [float(value) for value in totals],
                        "intercept": readout.intercept.tolist(), "coefficients_sha256": sha256_array(readout.coefficients), "solver_form": readout.solver_form,
                    }
                # Gate every view×budget dense-uniform cell against its recorded
                # authoritative normalized-ridge reference, not merely plausibility.
                observed = float(cell["arms"]["dense_uniform"]["r2"])
                expected = sealed_scores[(budget, str(cohort_row.asset_id), view)]
                error = abs(observed - expected)
                ridge.require(error <= SEALED_DENSE_M50_R2_ATOL, f"sealed uniform-dense reproduction gate failed: {cohort_row.session_id}/{view}/M{budget}: {error}")
                cell["sealed_uniform_dense_reproduction"] = {"reference_r2": expected, "absolute_error": error, "atol": SEALED_DENSE_M50_R2_ATOL}
                cells.append(cell)
            del record, _rebuilt
    aggregates: dict[str, object] = {}
    for view in VIEWS:
        for budget in BUDGETS:
            matching = [cell for cell in cells if cell["view"] == view and cell["budget"] == budget]
            by_arm = {arm: np.asarray([float(cell["arms"][arm]["r2"]) for cell in matching], dtype=np.float64)
                      for arm in ("dense_uniform", "dense_equal_trial", "direction_uniform", "direction_equal_trial")}
            aggregates[f"{view}_M{budget}"] = {
                "arms": {arm: {"mean": float(values.mean()), "median": float(np.median(values)), "per_session": {str(cell["asset_id"]): float(value) for cell, value in zip(matching, values)}} for arm, values in by_arm.items()},
                "paired_deltas": {name: {"mean": float(values.mean()), "median": float(np.median(values)), "positive": int((values > 0).sum()), "zero": int((values == 0).sum()), "negative": int((values < 0).sum())} for name, values in {
                    "dense_equal_trial_minus_uniform": by_arm["dense_equal_trial"] - by_arm["dense_uniform"],
                    "direction_equal_trial_minus_uniform": by_arm["direction_equal_trial"] - by_arm["direction_uniform"],
                    "dense_minus_direction_uniform": by_arm["dense_uniform"] - by_arm["direction_uniform"],
                    "dense_minus_direction_equal_trial": by_arm["dense_equal_trial"] - by_arm["direction_equal_trial"],
                }.items()},
            }
    result = {"schema": "priority_a2a_weighting_control_v2", "status": "COMPLETED_CPU_ONLY", "definition": "Normalized weighted 2x2; lambda=1 in mean-weight objective; explicit unpenalized intercept.", "numerical_contract": numerical_contract, "cells": cells, "aggregates": aggregates, "input_bindings": {"runner_sha256": sha256_file(Path(__file__)), "ridge_core_sha256": sha256_file(REPO_ROOT / "sua_exploration/mc_maze/priority_a2_normalized_ridge_v2.py"), "v9_manifest_sha256": v9.manifest_sha256, "behavior_normalizer_sha256": normalizer_hashes, "dense_uniform_reference_sha256": sealed_reference_hashes, "package_versions": package_versions()}, "elapsed_seconds": time.monotonic() - start_time, "receipt_policy": "exclusive new v2 filename; never overwrite"}
    return _write_new_receipt(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="run the full CPU experiment after the pre-data self-test")
    args = parser.parse_args()
    contract = ridge.numerical_contract_self_test()
    print(json.dumps({"numerical_contract": contract, "full_run_started": bool(args.run)}, sort_keys=True))
    if not args.run:
        return
    print(json.dumps({"receipt_sha256": run(), "receipt": str(RECEIPT_PATH)}, sort_keys=True))


if __name__ == "__main__":
    main()
