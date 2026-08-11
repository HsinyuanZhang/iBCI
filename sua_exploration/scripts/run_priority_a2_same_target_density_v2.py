#!/usr/bin/env python3
"""Priority A2b v2: same-target density dose response (explicit ``--run`` only)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
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
RECEIPT_PATH = OUTPUT_DIR / "priority_a2_same_target_density_v2_receipt.json"
A2A_RECEIPT_PATH = OUTPUT_DIR / "priority_a2_weighting_control_v2_receipt.json"
VIEWS, BUDGETS, HISTORY_BINS = ("sua", "pseudo_mua"), (15, 30, 50), 50
FINITE_KS, MASK_SEEDS = (1, 2, 4, 8, 16), (42, 43, 44)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 100_000, 20260811
FROZEN_A2A_CONTRACT_BOUNDS = {
    "uniform_reference_max_abs_error": 1.0e-10,
    "weight_scaling_max_abs_error": 2.0e-12,
    "intercept_foc_max_abs_error": 2.0e-9,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: object) -> str:
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("numpy", "torch", "torchmetrics", "pynwb"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "NOT_INSTALLED"
    return result


def _predict_query(record: object, numerical: object, readout: ridge.NormalizedWeightedRidge):
    import numpy as np
    parts = [ridge.predict_normalized_weighted_ridge(numerical.raw_window_features(record.neural, starts), readout)
             for starts in numerical.batched(record.valid_starts, 2048)]
    result = np.ascontiguousarray(np.concatenate(parts), dtype=np.float64)
    ridge.require(result.shape == (record.valid_starts.size, 2), "query prediction shape drift")
    return result


def _bootstrap_mean_ci(values, *, seed: int = BOOTSTRAP_SEED, draws: int = BOOTSTRAP_DRAWS) -> list[float]:
    import numpy as np
    array = np.asarray(values, dtype=np.float64)
    ridge.require(array.ndim == 1 and array.size > 0 and np.isfinite(array).all(), "invalid paired-bootstrap values")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, array.size, size=(draws, array.size), endpoint=False)
    bootstrap_means = array[indices].mean(axis=1)
    return [float(np.quantile(bootstrap_means, 0.025)), float(np.quantile(bootstrap_means, 0.975))]


def _exact_two_sided_sign_test(values) -> dict[str, int | float]:
    import numpy as np
    array = np.asarray(values, dtype=np.float64)
    positive, negative, zero = int((array > 0).sum()), int((array < 0).sum()), int((array == 0).sum())
    nonzero, tail = positive + negative, min(positive, negative)
    probability = 1.0 if nonzero == 0 else min(1.0, 2.0 * sum(math.comb(nonzero, k) for k in range(tail + 1)) / (2 ** nonzero))
    return {"positive": positive, "zero": zero, "negative": negative, "nonzero": nonzero, "two_sided_p": probability}


def _slope_against_log2_finite_k(values_by_k: dict[int, float]) -> float:
    """OLS slope over 1,2,4,8,16 only; ``all`` is intentionally excluded."""
    import numpy as np
    x = np.log2(np.asarray(FINITE_KS, dtype=np.float64))
    y = np.asarray([values_by_k[k] for k in FINITE_KS], dtype=np.float64)
    return float(np.linalg.lstsq(np.column_stack((np.ones(x.size), x)), y, rcond=None)[0][1])


def _write_new_receipt(payload: dict[str, object]) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if RECEIPT_PATH.exists():
        raise FileExistsError(f"refusing to overwrite existing v2 receipt: {RECEIPT_PATH}")
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    with RECEIPT_PATH.open("xb") as handle:
        handle.write(raw)
    os.chmod(RECEIPT_PATH, 0o444)
    return hashlib.sha256(raw).hexdigest()


def verify_a2a_v2_receipt(path: Path = A2A_RECEIPT_PATH) -> dict[str, object]:
    """Fail closed before data I/O unless corrected A2a has passed every gate."""
    if not path.is_file():
        raise FileNotFoundError(f"A2b requires an existing verified A2a v2 receipt: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ridge.PriorityA2NumericalError("A2a v2 receipt is unreadable") from exc
    ridge.require(payload.get("schema") == "priority_a2a_weighting_control_v2", "A2a receipt schema is not corrected v2")
    ridge.require(payload.get("status") == "COMPLETED_CPU_ONLY", "A2a receipt status is not completed CPU-only")
    bindings = payload.get("input_bindings")
    ridge.require(isinstance(bindings, dict) and isinstance(bindings.get("runner_sha256"), str) and isinstance(bindings.get("ridge_core_sha256"), str), "A2a receipt lacks runner/core bindings")
    expected_runner_sha = sha256_file(SCRIPT_DIR / "run_priority_a2_weighting_control_v2.py")
    expected_core_sha = sha256_file(REPO_ROOT / "sua_exploration/mc_maze/priority_a2_normalized_ridge_v2.py")
    ridge.require(bindings["runner_sha256"] == expected_runner_sha, "A2a receipt runner SHA is not the current v2 runner")
    ridge.require(bindings["ridge_core_sha256"] == expected_core_sha, "A2a receipt core SHA is not the current normalized-ridge core")
    contract = payload.get("numerical_contract")
    ridge.require(isinstance(contract, dict), "A2a receipt lacks numerical contract")
    for key, maximum in FROZEN_A2A_CONTRACT_BOUNDS.items():
        value = contract.get(key)
        ridge.require(isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) <= maximum, f"A2a numerical contract failed: {key}")
    cells = payload.get("cells")
    ridge.require(isinstance(cells, list) and len(cells) == 90, "A2a receipt must contain 90 sealed reproduction cells")
    seen: set[tuple[object, object, object]] = set()
    assets: set[str] = set()
    views: set[str] = set()
    budgets: set[int] = set()
    expected_arms = {"dense_uniform", "dense_equal_trial", "direction_uniform", "direction_equal_trial"}
    for cell in cells:
        ridge.require(isinstance(cell, dict), "A2a receipt cell is malformed")
        key = (cell.get("asset_id"), cell.get("view"), cell.get("budget"))
        ridge.require(key not in seen, "A2a receipt has duplicate sealed reproduction cell")
        seen.add(key)
        ridge.require(isinstance(cell.get("asset_id"), str) and cell["asset_id"], "A2a receipt cell has invalid asset")
        ridge.require(cell.get("view") in {"sua", "pseudo_mua"}, "A2a receipt cell has invalid view")
        ridge.require(isinstance(cell.get("budget"), int) and cell["budget"] in {15, 30, 50}, "A2a receipt cell has invalid budget")
        assets.add(cell["asset_id"])
        views.add(cell["view"])
        budgets.add(cell["budget"])
        ridge.require(cell.get("support_query_overlap_count") == 0, "A2a receipt cell has nonzero support/query overlap")
        arms = cell.get("arms")
        ridge.require(
            isinstance(arms, dict) and set(arms) == expected_arms,
            "A2a receipt cell does not contain exactly the frozen 2x2 arms",
        )
        reproduction = cell.get("sealed_uniform_dense_reproduction")
        ridge.require(isinstance(reproduction, dict), "A2a receipt cell lacks sealed uniform-dense reproduction")
        error, atol = reproduction.get("absolute_error"), reproduction.get("atol")
        ridge.require(isinstance(error, (int, float)) and isinstance(atol, (int, float)) and math.isfinite(float(error)) and math.isfinite(float(atol)) and 0.0 < float(atol) <= 5.0e-5 and 0.0 <= float(error) <= float(atol), "A2a sealed uniform-dense reproduction gate failed")
    ridge.require(len(assets) == 15 and views == {"sua", "pseudo_mua"} and budgets == {15, 30, 50}, "A2a receipt does not have the required 15-assets × 2-views × 3-budgets grid")
    expected_grid = {(asset, view, budget) for asset in assets for view in views for budget in budgets}
    ridge.require(seen == expected_grid, "A2a receipt sealed reproduction grid has extras or missing cells")
    return {"path": str(path), "sha256": sha256_file(path), "asset_ids": sorted(assets)}


def run() -> str:
    """Run same-target A2b after its independent pre-data self-test succeeds."""
    numerical_contract = ridge.numerical_contract_self_test()
    if RECEIPT_PATH.exists():
        raise FileExistsError(f"refusing to start because v2 receipt already exists: {RECEIPT_PATH}")
    a2a_receipt_binding = verify_a2a_v2_receipt()
    import numpy as np
    from sua_exploration.mc_maze import subm_v9_f0_pv_ridge as numerical
    from sua_exploration.mc_maze.multisession_datamodule import _compute_valid_starts
    from sua_exploration.mc_maze.subm_co_score_only_v2 import recompute_torchmetrics_r2_cpu
    from sua_exploration.scripts.run_subm_v9_f0_pv_ridge_controls import (
        _build_view_base, _nwb_path_and_pin, _runtime_owners, load_mean_std, load_v9_inputs, load_v9_target,
    )

    started = time.monotonic()
    v9 = load_v9_inputs(V9_RUN_ROOT, repo_root=REPO_ROOT)
    ridge.require(len(v9.cohort) == 15, "A2b v2 requires the fixed 15-session cohort")
    ridge.require({str(row.asset_id) for row in v9.cohort} == set(a2a_receipt_binding["asset_ids"]), "A2b V9 cohort assets do not match the verified A2a receipt")
    owners = _runtime_owners(REPO_ROOT)
    behavior_stats = {view: load_mean_std(v9.behavior_normalizers[view][0], label=f"{view} behavior") for view in VIEWS}
    normalizer_hashes = {view: sha256_file(Path(v9.behavior_normalizers[view][0])) for view in VIEWS}
    cells: list[dict[str, object]] = []
    # indexed scores remain in memory only long enough to build audit summaries.
    scores: dict[tuple[str, int, int, str, int], float] = {}
    for cohort_row in v9.cohort:
        nwb_path = _nwb_path_and_pin(NWB_ROOT, cohort_row)
        nwb_sha = sha256_file(nwb_path)
        session_key = f"{cohort_row.asset_id}|{cohort_row.session_id}"
        for view in VIEWS:
            mean, std = behavior_stats[view]
            record, _rebuilt, builder_trials, _bridge = _build_view_base(
                repo_root=REPO_ROOT, nwb_path=nwb_path, view=view, mean=mean, std=std, owners=owners,
            )
            query_target, query_target_sha, _ = load_v9_target(v9, cohort_row, view)
            ridge.require(sha256_array(query_target) == query_target_sha, "loader query target SHA does not bind returned query targets")
            for budget in BUDGETS:
                support = list(builder_trials[:budget])
                support_starts = np.ascontiguousarray(_compute_valid_starts(support, HISTORY_BINS), dtype=np.int64)
                overlap = np.intersect1d(support_starts, record.valid_starts, assume_unique=True)
                ridge.require(overlap.size == 0, "support/query row overlap is nonzero")
                owner = ridge.assign_windows_to_trials(support_starts, support, window_size=HISTORY_BINS)
                x = numerical.raw_window_features(record.neural, support_starts)
                # Dense velocity is read only after masks/ownership are completely determined.
                masks_by_seed = {seed: ridge.nested_density_masks(owner, budget, session_or_asset=session_key, mask_seed=seed, ks=FINITE_KS) for seed in MASK_SEEDS}
                dense_y = numerical.targets_at_window_end(record.behavior, support_starts)
                base = {"asset_id": cohort_row.asset_id, "session_id": cohort_row.session_id, "view": view, "budget": budget,
                        "nwb_sha256": nwb_sha, "support_starts_sha256": sha256_array(support_starts), "query_starts_sha256": sha256_array(record.valid_starts),
                        "support_feature_sha256": sha256_array(x), "query_feature_source_neural_sha256": sha256_array(record.neural),
                        "support_target_sha256": sha256_array(dense_y), "query_target_sha256": query_target_sha,
                        "support_query_overlap_count": 0, "support_rows": int(support_starts.size), "query_rows": int(record.valid_starts.size),
                        "target_definition": "normalized dense 2-D velocity", "mask_selection_target_blind": True}
                def fit_one(mask, label: int | str) -> dict[str, object]:
                    """Fit one target-visible-row arm and return its hash-complete record."""
                    selected_owner = owner[mask]
                    weights = ridge.equal_trial_weights(selected_owner, budget)
                    readout = ridge.fit_normalized_weighted_ridge(x[mask], dense_y[mask], weights, normalized_lambda=1.0)
                    prediction = _predict_query(record, numerical, readout)
                    score = float(recompute_torchmetrics_r2_cpu(prediction.astype(np.float32), query_target))
                    effective_counts = np.bincount(selected_owner, minlength=budget)
                    totals = np.bincount(selected_owner, weights=weights, minlength=budget)
                    ridge.require(np.allclose(totals, totals[0], rtol=0.0, atol=1e-12), "per-trial total weights drift")
                    return {"K": label, "mask_sha256": sha256_array(mask.astype(np.uint8)), "selected_row_starts_sha256": sha256_array(support_starts[mask]),
                            "selected_target_sha256": sha256_array(dense_y[mask]), "weights_sha256": sha256_array(weights),
                            "prediction_sha256": sha256_array(prediction), "r2": score, "effective_rows_per_trial": [int(value) for value in effective_counts],
                            "per_trial_total_weight": [float(value) for value in totals], "coefficients_sha256": sha256_array(readout.coefficients),
                            "intercept": readout.intercept.tolist(), "solver_form": readout.solver_form}

                for seed in MASK_SEEDS:
                    for label, mask in masks_by_seed[seed].items():
                        result = fit_one(mask, label)
                        scores[(view, budget, seed, str(cohort_row.asset_id), int(label))] = float(result["r2"])
                        cells.append({**base, **result, "mask_seed": seed, "mask_seed_independent": False})
                # all has no seed-dependent mask.  Fit/predict once and reuse the
                # identical artifacts and score in all three seed-level reports.
                all_result = fit_one(np.ones(owner.size, dtype=bool), "all")
                for seed in MASK_SEEDS:
                    scores[(view, budget, seed, str(cohort_row.asset_id), -1)] = float(all_result["r2"])
                    cells.append({**base, **all_result, "mask_seed": seed, "mask_seed_independent": True,
                                  "reused_from_mask_seed": MASK_SEEDS[0], "fit_prediction_reused_across_seeds": True})
            del record, _rebuilt

    summaries: dict[str, object] = {}
    assets = [str(row.asset_id) for row in v9.cohort]
    for view in VIEWS:
        for budget in BUDGETS:
            per_session_dose: dict[str, object] = {}
            per_seed_means: dict[str, object] = {}
            for seed in MASK_SEEDS:
                per_seed_means[str(seed)] = {str(k): float(np.mean([scores[(view, budget, seed, asset, k)] for asset in assets])) for k in FINITE_KS}
                per_seed_means[str(seed)]["all"] = float(np.mean([scores[(view, budget, seed, asset, -1)] for asset in assets]))
            averaged_delta: list[float] = []
            averaged_slopes: list[float] = []
            for asset in assets:
                dose = {str(k): float(np.mean([scores[(view, budget, seed, asset, k)] for seed in MASK_SEEDS])) for k in FINITE_KS}
                dose["all"] = float(np.mean([scores[(view, budget, seed, asset, -1)] for seed in MASK_SEEDS]))
                per_session_dose[asset] = dose
                averaged_delta.append(dose["all"] - dose["1"])
                averaged_slopes.append(_slope_against_log2_finite_k({k: dose[str(k)] for k in FINITE_KS}))
            seed_mean_vector = {str(k): [float(per_seed_means[str(seed)][str(k)]) for seed in MASK_SEEDS] for k in FINITE_KS}
            seed_mean_vector["all"] = [float(per_seed_means[str(seed)]["all"]) for seed in MASK_SEEDS]
            summaries[f"{view}_M{budget}"] = {
                "per_session_dose_response_seed_mean": per_session_dose,
                "per_seed_mean_r2": per_seed_means,
                "mask_seed_variance_of_aggregate_mean": {label: float(np.var(values, ddof=0)) for label, values in seed_mean_vector.items()},
                "primary_all_minus_K1": {"aggregation": "per-session mean across mask seeds; all minus K1", "mean": float(np.mean(averaged_delta)), "median": float(np.median(averaged_delta)), "sign_counts": _exact_two_sided_sign_test(averaged_delta), "paired_bootstrap_95_ci": _bootstrap_mean_ci(averaged_delta), "per_session": dict(zip(assets, averaged_delta))},
                "paired_slope_r2_per_log2K": {"finite_K_only": list(FINITE_KS), "all_excluded_reason": "all has no invented numeric K; reported separately through all-minus-K1", "mean": float(np.mean(averaged_slopes)), "median": float(np.median(averaged_slopes)), "sign_counts": _exact_two_sided_sign_test(averaged_slopes), "paired_bootstrap_95_ci": _bootstrap_mean_ci(averaged_slopes), "per_session": dict(zip(assets, averaged_slopes))},
            }
    receipt = {"schema": "priority_a2b_same_target_density_v2", "status": "COMPLETED_CPU_ONLY", "definition": "Same normalized dense velocity target; only target-visible rows K vary, nested and target-blind; equal total trial weight.", "numerical_contract": numerical_contract, "mask_protocol": {"seeds": list(MASK_SEEDS), "finite_K": list(FINITE_KS), "all": "all calibration rows; no invented numeric K", "permutation_key": "sha256(priority-a2b-v2|asset|session|trial-index|mask-seed)", "nested_prefixes": True}, "fit_accounting": {"fits_per_session_view_budget_before_all_deduplication": 18, "fits_per_session_view_budget_after_all_deduplication": 16, "avoided_all_fits_per_session_view_budget": 2, "avoided_all_fits_total": 180}, "cells": cells, "summaries": summaries, "input_bindings": {"runner_sha256": sha256_file(Path(__file__)), "ridge_core_sha256": sha256_file(REPO_ROOT / "sua_exploration/mc_maze/priority_a2_normalized_ridge_v2.py"), "a2a_v2_receipt": a2a_receipt_binding, "v9_manifest_sha256": v9.manifest_sha256, "behavior_normalizer_sha256": normalizer_hashes, "package_versions": package_versions()}, "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED}, "elapsed_seconds": time.monotonic() - started, "receipt_policy": "exclusive new v2 filename; never overwrite"}
    return _write_new_receipt(receipt)


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
