#!/usr/bin/env python3
"""CPU-only runner for the frozen source-pooled ridge fairness experiment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
for _import_root in (REPO_ROOT, REPO_ROOT / "sua_exploration", REPO_ROOT / "sua_exploration/scripts"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

import numpy as np

from sua_exploration.mc_maze import priority_a2_normalized_ridge_v2 as ridge
from sua_exploration.mc_maze import source_pooled_ridge as spr
from sua_exploration.mc_maze.multisession_datamodule import _compute_valid_starts
from sua_exploration.mc_maze.unit_side_features import load_session_electrode_ids


V9_RUN_ROOT = REPO_ROOT / "sua_exploration/results/dandi_000688_subm_co_three_arm_v9_formal_20260805"
NWB_ROOT = REPO_ROOT / "sua_exploration/data/dandi_000688"
OUTPUT_DIR = REPO_ROOT / "sua_exploration/results/source_pooled_ridge"
RECEIPT_PATH = OUTPUT_DIR / "source_pooled_ridge_receipt.json"
A2A_V2_RECEIPT = REPO_ROOT / "sua_exploration/results/trial_level_ridge_v1/priority_a2_weighting_control_v2_receipt.json"
HISTORY_BINS = spr.HISTORY_BINS
SCRATCH_ARMS = ("ridge_dense_scratch", "ridge_direction_scratch")
SPR_ARMS = ("ridge_dense_spr", "ridge_direction_spr")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    names = ("numpy", "torch", "torchmetrics", "pynwb")
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "NOT_INSTALLED"
    return result


def _write_new_receipt(payload: dict[str, Any]) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if RECEIPT_PATH.exists():
        raise FileExistsError(f"refusing to overwrite existing receipt: {RECEIPT_PATH}")
    raw = (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    with RECEIPT_PATH.open("xb") as handle:
        handle.write(raw)
    os.chmod(RECEIPT_PATH, 0o444)
    return hashlib.sha256(raw).hexdigest()


def _build_calibration(
    *,
    record: Any,
    builder_trials: list[dict[str, Any]],
    budget: int,
    query_target: np.ndarray,
) -> spr.SessionCalibration:
    from sua_exploration.mc_maze import subm_v9_f0_pv_ridge as numerical

    support = list(builder_trials[:budget])
    ridge.require_target_directions(support)
    support_starts = np.ascontiguousarray(_compute_valid_starts(support, HISTORY_BINS), dtype=np.int64)
    overlap = np.intersect1d(support_starts, record.valid_starts, assume_unique=True)
    ridge.require(overlap.size == 0, "support/query overlap is nonzero")
    trial_of_window = ridge.assign_windows_to_trials(support_starts, support, window_size=HISTORY_BINS)
    x = numerical.raw_window_features(record.neural, support_starts)
    dense_y = numerical.targets_at_window_end(record.behavior, support_starts)
    sparse_y = ridge.direction_targets_for_windows(support, trial_of_window)
    uniform = np.ones(len(support_starts), dtype=np.float64)
    neural = record.neural

    def query_features_fn(starts: np.ndarray) -> np.ndarray:
        return numerical.raw_window_features(neural, starts)

    return spr.SessionCalibration(
        asset_id="",
        session_id="",
        support_starts=support_starts,
        support_features=x,
        dense_targets=dense_y,
        direction_targets=sparse_y,
        uniform_weights=uniform,
        query_starts=np.ascontiguousarray(record.valid_starts, dtype=np.int64),
        query_features_fn=query_features_fn,
        query_target=query_target,
        channel_ids=np.asarray(record.channel_ids, dtype=np.int64),
    )


def load_audit(repo_root: Path) -> dict[str, spr.CorrespondenceAudit]:
    from sua_exploration.scripts.run_subm_v9_f0_pv_ridge_controls import (
        _build_view_base,
        _nwb_path_and_pin,
        _runtime_owners,
        load_mean_std,
        load_v9_inputs,
    )

    v9 = load_v9_inputs(V9_RUN_ROOT, repo_root=repo_root)
    ridge.require(len(v9.cohort) == 15, "SPR requires the fixed 15-session cohort")
    owners = _runtime_owners(repo_root)
    behavior_stats = {
        view: load_mean_std(v9.behavior_normalizers[view][0], label=f"{view} behavior") for view in spr.VIEWS
    }
    channel_rows: dict[str, list[spr.SessionChannelInfo]] = {view: [] for view in spr.VIEWS}
    for cohort_row in v9.cohort:
        nwb_path = _nwb_path_and_pin(NWB_ROOT, cohort_row)
        electrode_ids = load_session_electrode_ids(nwb_path)
        for view in spr.VIEWS:
            mean, std = behavior_stats[view]
            record, _rebuilt, _builder_trials, _bridge = _build_view_base(
                repo_root=repo_root,
                nwb_path=nwb_path,
                view=view,
                mean=mean,
                std=std,
                owners=owners,
            )
            channel_rows[view].append(
                spr.build_session_channel_info(
                    asset_id=str(cohort_row.asset_id),
                    session_id=str(cohort_row.session_id),
                    view=view,
                    record=record,
                    electrode_ids_per_unit=electrode_ids,
                )
            )
            del record, _rebuilt
    return {view: spr.audit_view_correspondence(channel_rows[view], view=view) for view in spr.VIEWS}


def load_experiment_data(
    repo_root: Path,
) -> tuple[Any, dict[str, dict[int, dict[str, spr.SessionCalibration]]], dict[str, spr.CorrespondenceAudit], dict[str, np.ndarray], Callable[[np.ndarray, np.ndarray], float]]:
    from sua_exploration.mc_maze.subm_co_score_only_v2 import recompute_torchmetrics_r2_cpu
    from sua_exploration.scripts.run_subm_v9_f0_pv_ridge_controls import (
        _build_view_base,
        _nwb_path_and_pin,
        _runtime_owners,
        load_mean_std,
        load_v9_inputs,
        load_v9_target,
    )

    v9 = load_v9_inputs(V9_RUN_ROOT, repo_root=repo_root)
    ridge.require(len(v9.cohort) == 15, "SPR requires the fixed 15-session cohort")
    owners = _runtime_owners(repo_root)
    behavior_stats = {
        view: load_mean_std(v9.behavior_normalizers[view][0], label=f"{view} behavior") for view in spr.VIEWS
    }
    audits = load_audit(repo_root)
    by_view_budget: dict[str, dict[int, dict[str, spr.SessionCalibration]]] = {
        view: {budget: {} for budget in spr.BUDGETS} for view in spr.VIEWS
    }
    query_starts_by_asset: dict[str, np.ndarray] = {}
    for cohort_row in v9.cohort:
        asset_id = str(cohort_row.asset_id)
        nwb_path = _nwb_path_and_pin(NWB_ROOT, cohort_row)
        for view in spr.VIEWS:
            mean, std = behavior_stats[view]
            record, _rebuilt, builder_trials, _bridge = _build_view_base(
                repo_root=repo_root,
                nwb_path=nwb_path,
                view=view,
                mean=mean,
                std=std,
                owners=owners,
            )
            query_target, query_target_sha, _ = load_v9_target(v9, cohort_row, view)
            ridge.require(spr.sha256_array(query_target) == query_target_sha, "query target SHA drift")
            query_starts_by_asset[asset_id] = np.ascontiguousarray(record.valid_starts, dtype=np.int64)
            for budget in spr.BUDGETS:
                calibration = _build_calibration(
                    record=record,
                    builder_trials=builder_trials,
                    budget=budget,
                    query_target=query_target,
                )
                by_view_budget[view][budget][asset_id] = spr.SessionCalibration(
                    asset_id=asset_id,
                    session_id=str(cohort_row.session_id),
                    support_starts=calibration.support_starts,
                    support_features=calibration.support_features,
                    dense_targets=calibration.dense_targets,
                    direction_targets=calibration.direction_targets,
                    uniform_weights=calibration.uniform_weights,
                    query_starts=calibration.query_starts,
                    query_features_fn=calibration.query_features_fn,
                    query_target=calibration.query_target,
                    channel_ids=calibration.channel_ids,
                )
            del record, _rebuilt
    return v9, by_view_budget, audits, query_starts_by_asset, recompute_torchmetrics_r2_cpu


def run() -> str:
    numerical_contract = ridge.numerical_contract_self_test()
    if RECEIPT_PATH.exists():
        raise FileExistsError(f"refusing to start because receipt already exists: {RECEIPT_PATH}")
    start_time = time.monotonic()
    v9, by_view_budget, audits, query_starts_by_asset, recompute_r2 = load_experiment_data(REPO_ROOT)
    audit_payload = {view: spr.correspondence_audit_to_dict(audits[view]) for view in spr.VIEWS}
    cells: list[dict[str, Any]] = []
    aggregates: dict[str, Any] = {}
    integrity_gates: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    selection_records: list[dict[str, Any]] = []

    for view in spr.VIEWS:
        audit = audits[view]
        for budget in spr.BUDGETS:
            session_map = by_view_budget[view][budget]
            per_arm_scores: dict[str, dict[str, float]] = {arm: {} for arm in SCRATCH_ARMS + SPR_ARMS}
            for target_asset, target in sorted(session_map.items()):
                query_starts = query_starts_by_asset[target_asset]
                source_sessions = [
                    session_map[asset_id] for asset_id in sorted(session_map) if asset_id != target_asset
                ]
                cell: dict[str, Any] = {
                    "asset_id": target_asset,
                    "session_id": target.session_id,
                    "view": view,
                    "budget": budget,
                    "support_rows": int(target.support_starts.size),
                    "query_rows": int(query_starts.size),
                    "arms": {},
                }
                for arm in SCRATCH_ARMS:
                    result = spr.run_scratch_arm(
                        target,
                        arm=arm,
                        query_starts=query_starts,
                        recompute_r2=recompute_r2,
                    )
                    cell["arms"][arm] = result
                    per_arm_scores[arm][target_asset] = float(result["r2"])
                if audit.spr_definable:
                    for arm in SPR_ARMS:
                        result = spr.run_spr_arm(
                            target,
                            source_sessions,
                            arm=arm,
                            budget=budget,
                            query_starts=query_starts,
                            recompute_r2=recompute_r2,
                        )
                        cell["arms"][arm] = result
                        per_arm_scores[arm][target_asset] = float(result["r2"])
                        selection_records.append(
                            {
                                "asset_id": target_asset,
                                "view": view,
                                "budget": budget,
                                "arm": arm,
                                "selection_provenance": result["selection_provenance"],
                            }
                        )
                else:
                    for arm in SPR_ARMS:
                        cell["arms"][arm] = {"status": audit.status}
                cells.append(cell)

            for arm in SCRATCH_ARMS:
                values = np.asarray(list(per_arm_scores[arm].values()), dtype=np.float64)
                gate = spr.verify_integrity_gate(arm, view, budget, float(values.mean()))
                integrity_gates.append(gate)
                ridge.require(gate["passed"], f"integrity gate failed: {arm}/{view}/M{budget}")
            key = f"{view}_M{budget}"
            aggregates[key] = {
                "arms": {
                    arm: {
                        "mean": float(np.mean(list(per_arm_scores[arm].values()))),
                        "median": float(np.median(list(per_arm_scores[arm].values()))),
                        "per_session": per_arm_scores[arm],
                    }
                    for arm in SCRATCH_ARMS + (SPR_ARMS if audit.spr_definable else ())
                },
                "spr_status": audit.status,
            }
            contrasts.append(
                spr.summarize_contrast(
                    view=view,
                    budget=budget,
                    per_session_r2=per_arm_scores.get("ridge_direction_spr", per_arm_scores["ridge_direction_scratch"]),
                    scratch_direction_r2=per_arm_scores["ridge_direction_scratch"],
                    spr_direction_r2=per_arm_scores.get("ridge_direction_spr"),
                )
            )

    max_integrity_deviation = max(gate["absolute_deviation"] for gate in integrity_gates)
    payload = {
        "schema": "source_pooled_ridge_v1",
        "status": "COMPLETED_CPU_ONLY",
        "definition": "Source-pooled prior ridge fairness experiment for subject-M.",
        "numerical_contract": numerical_contract,
        "scope": {
            "cuda_used": False,
            "cohort_sessions": 15,
            "budgets": list(spr.BUDGETS),
            "views": list(spr.VIEWS),
        },
        "correspondence_audit": audit_payload,
        "sealed_reference_table": {
            "scratch": spr.SEALED_SCRATCH_REFERENCES,
            "t4": spr.SEALED_T4_REFERENCES,
            "a2a_v2_receipt_sha256": spr.A2A_V2_RECEIPT_SHA256,
        },
        "integrity_gates": integrity_gates,
        "integrity_gate_max_abs_deviation": max_integrity_deviation,
        "cells": cells,
        "aggregates": aggregates,
        "primary_contrasts": contrasts,
        "selection_provenance": selection_records,
        "input_bindings": {
            "runner_sha256": sha256_file(Path(__file__)),
            "implementation_sha256": sha256_file(REPO_ROOT / "sua_exploration/mc_maze/source_pooled_ridge.py"),
            "ridge_core_sha256": sha256_file(REPO_ROOT / "sua_exploration/mc_maze/priority_a2_normalized_ridge_v2.py"),
            "a2a_v2_receipt_sha256": sha256_file(A2A_V2_RECEIPT),
            "v9_manifest_sha256": v9.manifest_sha256,
            "package_versions": package_versions(),
        },
        "elapsed_seconds": time.monotonic() - start_time,
        "receipt_policy": "exclusive new filename; never overwrite; chmod 0444",
    }
    return _write_new_receipt(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Source-pooled ridge fairness experiment (CPU only)")
    parser.add_argument("--audit-only", action="store_true", help="run Part A correspondence audit only")
    parser.add_argument("--run", action="store_true", help="run the full experiment after tests")
    args = parser.parse_args()
    contract = ridge.numerical_contract_self_test()
    print(json.dumps({"numerical_contract": contract, "run_started": bool(args.run), "audit_only": bool(args.audit_only)}, sort_keys=True))
    if args.audit_only:
        audits = load_audit(REPO_ROOT)
        print(json.dumps({view: spr.correspondence_audit_to_dict(audits[view]) for view in spr.VIEWS}, indent=2, sort_keys=True))
        return
    if not args.run:
        return
    print(json.dumps({"receipt_sha256": run(), "receipt": str(RECEIPT_PATH)}, sort_keys=True))


if __name__ == "__main__":
    main()
