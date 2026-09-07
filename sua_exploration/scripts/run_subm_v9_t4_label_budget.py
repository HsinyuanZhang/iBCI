#!/usr/bin/env python3
"""Frozen-model external sub-M T4 label-budget robustness runner.

Only the chronological T4 fitting prefix changes (M=10/15/20/30/40).
Activity identity remains first-30 and every score target remains strictly
after trial 50.  The V9 M=50 artifacts are immutable references and are not
recomputed here.  This command never computes R2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PACKAGE_ROOT = ROOT / "sua_exploration"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from sua_exploration.mc_maze import subm_co_scorer_adapter_parity_v3 as parity_v3
from sua_exploration.mc_maze import subm_co_three_arm_v9_runtime as runtime
from sua_exploration.mc_maze import subm_v9_t4_label_budget as budget_core
from sua_exploration.mc_maze import unit_side_features


SCHEMA = "dandi_000688_subm_v9_t4_label_budget_v1"
EXPECTED_FULL_CELLS = 15 * len(runtime.VIEWS) * len(budget_core.BUDGETS) * len(runtime.SEEDS)


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def readonly(path: Path) -> bool:
    return path.is_file() and not path.is_symlink() and (path.stat().st_mode & 0o777) == 0o444


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(canonical_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    need(readonly(path), f"failed to seal {path}")


def reference_path(v9_root: Path, asset_id: str, view: str, seed: int) -> Path:
    return v9_root / "artifacts" / asset_id / view / "shared_t4" / f"seed_{seed}" / "predictions_targets.npz"


def cell_paths(output_root: Path, asset_id: str, view: str, budget: int, seed: int) -> tuple[Path, Path]:
    artifact = output_root / "artifacts" / asset_id / view / f"m_{budget}" / f"seed_{seed}" / "predictions_targets.npz"
    commit = output_root / "commits" / asset_id / view / f"m_{budget}" / f"seed_{seed}.json"
    return artifact, commit


def load_reference(v9_root: Path, asset_id: str, view: str, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    path = reference_path(v9_root, asset_id, view, seed)
    need(readonly(path), f"unsafe/missing V9 reference {path}")
    with np.load(path, allow_pickle=False) as archive:
        prediction = np.ascontiguousarray(archive["predictions"], dtype=np.float32)
        target = np.ascontiguousarray(archive["targets"], dtype=np.float32)
    need(prediction.shape == target.shape and prediction.ndim == 2 and prediction.shape[1] == 2, f"bad V9 artifact {path}")
    need(np.isfinite(prediction).all() and np.isfinite(target).all(), f"nonfinite V9 artifact {path}")
    return prediction, target, runtime.sha256_file(path)


def build_dataset(
    *, nwb_path: Path, view: str, budget: int, record: Any, rebuilt: Any,
    side_mean: np.ndarray, side_std: np.ndarray, owners: dict[str, Any], trials: list[dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    design = budget_core.audit_t4_design(
        trials, budget,
        nearest_direction=unit_side_features._nearest_canonical_direction_index,
        canonical_directions=unit_side_features.CANONICAL_DIRECTIONS_RAD,
    )
    features, metadata = owners["load_unit_side_features"](
        nwb_path,
        feature_group="t4",
        pool_size=budget,
        mean=side_mean,
        std=side_std,
        cache_dir=None,
        permutation_seed=None,
        bin_size_ms=parity_v3.BIN_SIZE_MS,
        window_size=parity_v3.HISTORY_BINS,
        trial_result_filter="R",
        signal_view=view,
    )
    features = np.ascontiguousarray(features, dtype=np.float32)
    need(features.shape == (record.neural.shape[1], 4), f"descriptor shape drift {record.name}/{view}/M{budget}")
    need(np.isfinite(features).all(), f"nonfinite descriptor {record.name}/{view}/M{budget}")
    dataset = owners["MCMazeSessionDataset"](
        neural_data=record.neural,
        behavior_data=record.behavior,
        valid_starts=record.valid_starts,
        calib_trials=rebuilt,
        window_size=parity_v3.HISTORY_BINS,
        session_name=record.name,
        side_features=features,
        electrode_ids=None,
    )
    need(len(dataset) == int(record.valid_starts.size), f"query count drift {record.name}/{view}/M{budget}")
    return dataset, {
        "design": design.as_dict(),
        "feature_shape": list(features.shape),
        "feature_sha256": sha_array(features),
        "loader_pool_size": int(budget),
        "loader_metadata_pool_size": metadata.get("pool_size") if isinstance(metadata, dict) else None,
    }


def initialize_output(
    *, output_root: Path, input_manifest: Path, input_sha: str, contract: runtime.RuntimeContract,
    v9_root: Path, nwb_root: Path, device: str,
) -> None:
    manifest = output_root / "run_manifest.json"
    payload = {
        "schema": SCHEMA,
        "status": "FORWARD_ONLY_NO_METRIC",
        "post_hoc_scope": "exploratory frozen-M50-trained-system label-budget robustness",
        "budgets_new": list(budget_core.BUDGETS),
        "reference_budget": budget_core.REFERENCE_BUDGET,
        "activity_trials": runtime.ACTIVITY_IDENTITY_TRIALS,
        "query_rule": "strictly_after_rewarded_trial_50",
        "views": list(runtime.VIEWS),
        "seeds": list(runtime.SEEDS),
        "expected_full_cells": EXPECTED_FULL_CELLS,
        "contract_sha256": contract.sha256,
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": input_sha,
        "reference_v9_root": str(v9_root),
        "nwb_root": str(nwb_root),
        "device": device,
        "metric_computed": False,
        "target_session_backward": False,
    }
    if manifest.exists():
        need(readonly(manifest), f"unsafe existing manifest {manifest}")
        need(json.loads(manifest.read_text()) == payload, "existing output manifest drift")
    else:
        write_json_exclusive(manifest, payload)


def preflight_or_forward(args: argparse.Namespace, *, do_forward: bool) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    nwb_root = args.nwb_root.resolve()
    input_manifest = args.input_manifest.resolve()
    v9_root = args.reference_v9_root.resolve()
    contract, input_sha = runtime.load_input_manifest(input_manifest)
    runtime.verify_runtime_inputs(contract=contract, nwb_root=nwb_root)
    owners = runtime._runtime_owners(repo_root)
    output_root = args.output_root.resolve() if args.output_root else None
    if do_forward:
        need(output_root is not None, "--output-root required for forward")
        initialize_output(
            output_root=output_root, input_manifest=input_manifest, input_sha=input_sha,
            contract=contract, v9_root=v9_root, nwb_root=nwb_root, device=args.device,
        )
    behavior_stats = {
        view: runtime._load_mean_std(contract.behavior_normalizer_paths[view], label=f"{view} behavior")
        for view in runtime.VIEWS
    }
    side_stats = {
        view: runtime._load_mean_std(contract.side_normalizer_paths[view], label=f"{view} T4")
        for view in runtime.VIEWS
    }
    models: dict[int, Any] = {}
    completed = 0
    skipped = 0
    design_rows: list[dict[str, Any]] = []
    cohort_stop = args.session_offset + args.max_sessions if args.max_sessions else len(contract.cohort)
    cohort = contract.cohort[args.session_offset:cohort_stop]
    need(bool(cohort), "session slice selected zero cohort rows")
    for row in cohort:
        nwb_path = nwb_root / row.frozen_path
        trials = owners["list_datamodule_rewarded_trials"](
            nwb_path,
            bin_size_ms=parity_v3.BIN_SIZE_MS,
            window_size=parity_v3.HISTORY_BINS,
            trial_result_filter="R",
        )
        for view in runtime.VIEWS:
            behavior_mean, behavior_std = behavior_stats[view]
            record, rebuilt, base_evidence = runtime._build_base(
                repo_root=repo_root, nwb_path=nwb_path, view=view,
                behavior_mean=behavior_mean, behavior_std=behavior_std, owners=owners,
            )
            need(record.name == row.session_id, f"session identity drift {record.name}")
            need(int(record.valid_starts.size) == row.query_window_count, f"query count drift {row.session_id}/{view}")
            reference_targets: dict[int, np.ndarray] = {}
            reference_artifact_sha: dict[int, str] = {}
            for seed in runtime.SEEDS:
                _prediction, target, artifact_sha = load_reference(v9_root, row.asset_id, view, seed)
                need(target.shape[0] == row.query_window_count, f"reference query count drift {row.session_id}/{view}/{seed}")
                reference_targets[seed] = target
                reference_artifact_sha[seed] = artifact_sha
            side_mean, side_std = side_stats[view]
            for budget in budget_core.BUDGETS:
                dataset, descriptor_evidence = build_dataset(
                    nwb_path=nwb_path, view=view, budget=budget, record=record, rebuilt=rebuilt,
                    side_mean=side_mean, side_std=side_std, owners=owners, trials=trials,
                )
                design_rows.append({
                    "asset_id": row.asset_id, "session_id": row.session_id, "view": view,
                    "budget": budget, **descriptor_evidence["design"],
                })
                if not do_forward:
                    continue
                for seed in runtime.SEEDS:
                    artifact, commit = cell_paths(output_root, row.asset_id, view, budget, seed)
                    if artifact.exists() or commit.exists():
                        need(readonly(artifact) and readonly(commit), f"orphan/unsafe existing cell {artifact}")
                        saved = json.loads(commit.read_text())
                        need(saved.get("artifact_sha256") == runtime.sha256_file(artifact), f"existing artifact SHA drift {artifact}")
                        with np.load(artifact, allow_pickle=False) as archive:
                            target = np.ascontiguousarray(archive["targets"], dtype=np.float32)
                        need(np.array_equal(target, reference_targets[seed]), f"existing target differs from V9 {artifact}")
                        skipped += 1
                        continue
                    if seed not in models:
                        models[seed] = runtime._load_model(
                            contract.checkpoint("shared_t4", seed), contract, args.device, owners
                        )
                    prediction, target, forward_evidence = runtime.collect_forward_predictions(
                        model=models[seed], dataset=dataset, device=args.device, owners=owners,
                    )
                    need(np.array_equal(target, reference_targets[seed]), f"target differs from V9 {row.session_id}/{view}/M{budget}/s{seed}")
                    artifact_sha = runtime._write_npz_exclusive(artifact, prediction, target)
                    payload = {
                        "schema": SCHEMA,
                        "asset_id": row.asset_id,
                        "session_id": row.session_id,
                        "view": view,
                        "budget": budget,
                        "seed": seed,
                        "query_window_count": int(target.shape[0]),
                        "artifact": str(artifact.relative_to(output_root)),
                        "artifact_sha256": artifact_sha,
                        "artifact_bytes": artifact.stat().st_size,
                        "prediction_sha256": sha_array(prediction),
                        "target_sha256": sha_array(target),
                        "reference_v9_artifact_sha256": reference_artifact_sha[seed],
                        "descriptor": descriptor_evidence,
                        "base_protocol": base_evidence,
                        "forward": forward_evidence,
                        "checkpoint_sha256": contract.checkpoint("shared_t4", seed).sha256,
                        "metric_computed": False,
                        "backward_called": False,
                    }
                    write_json_exclusive(commit, payload)
                    completed += 1
    result = {
        "status": "PREFLIGHT_PASS" if not do_forward else "FORWARD_ARTIFACTS_COMMITTED_NO_METRIC",
        "sessions": len(cohort),
        "views": len(runtime.VIEWS),
        "budgets": list(budget_core.BUDGETS),
        "design_rows": len(design_rows),
        "completed_cells": completed,
        "skipped_cells": skipped,
        "expected_selected_cells": len(cohort) * len(runtime.VIEWS) * len(budget_core.BUDGETS) * len(runtime.SEEDS),
        "max_design_condition": max(float(row["design_condition"]) for row in design_rows),
        "min_distinct_directions": min(len(row["present_direction_indices"]) for row in design_rows),
    }
    if args.preflight_receipt:
        receipt = args.preflight_receipt.resolve()
        if not receipt.exists():
            write_json_exclusive(receipt, {**result, "design": design_rows})
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "forward"), required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--nwb-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--reference-v9-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--session-offset", type=int, default=0)
    parser.add_argument("--max-sessions", type=int)
    args = parser.parse_args()
    if args.mode == "forward" and args.output_root is None:
        parser.error("--output-root is required for forward")
    if args.max_sessions is not None and not 1 <= args.max_sessions <= 15:
        parser.error("--max-sessions must be in [1,15]")
    if not 0 <= args.session_offset < 15:
        parser.error("--session-offset must be in [0,14]")
    if args.max_sessions is not None and args.session_offset + args.max_sessions > 15:
        parser.error("session slice exceeds frozen 15-session cohort")
    return args


def main() -> None:
    args = parse_args()
    result = preflight_or_forward(args, do_forward=args.mode == "forward")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
