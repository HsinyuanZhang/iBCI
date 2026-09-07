#!/usr/bin/env python3
"""Score-blind, BP-free external sub-M M=30 true-early-start forward runner.

This is deliberately distinct from the fixed-query T4 budget curve.  It
reuses frozen V9 source checkpoints/normalizers but reconstructs the target
session with the first 30 rewarded trials reserved for both B3 activity and
T4/TS4.  All decoded windows come from rewarded trials 31 onward and every
50-bin neural history is audited to be fully post-support.

Remote execution writes immutable prediction/target artifacts only; it never
opens a metric.  Local TorchMetrics-1.5.1 finalization is a separate command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "sua_exploration"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from sua_exploration.mc_maze import subm_co_scorer_adapter_parity_v3 as parity_v3
from sua_exploration.mc_maze import subm_co_three_arm_v9_runtime as runtime
from sua_exploration.mc_maze import subm_v9_m30_true_early_start as early
from sua_exploration.mc_maze import subm_v9_t4_label_budget as budget_core
from sua_exploration.mc_maze import unit_side_features


SOURCE_RELATIVE_PATHS = (
    "sua_exploration/mc_maze/subm_v9_m30_true_early_start.py",
    "sua_exploration/mc_maze/subm_v9_t4_label_budget.py",
    "sua_exploration/scripts/run_subm_v9_m30_true_early_start.py",
    "sua_exploration/mc_maze/subm_co_three_arm_v9_runtime.py",
    "sua_exploration/mc_maze/subm_co_score_only_v3r2.py",
    "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_v3.py",
    "sua_exploration/mc_maze/subm_co_scorer_adapter_parity_v5.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "sua_exploration/mc_maze/datamodule.py",
    "sua_exploration/mc_maze/unit_side_features.py",
    "sua_exploration/scripts/eval_adaptation_dandi688.py",
    "sua_exploration/scripts/dandi688_gradient_free_protocol.py",
    "sua_exploration/scripts/select_gradient_free_protocol_dandi688.py",
)


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_array(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def readonly(path: Path) -> bool:
    try:
        return path.is_file() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o444
    except OSError:
        return False


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> str:
    raw = canonical_bytes(dict(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    need(readonly(path), f"failed to seal immutable JSON {path}")
    return hashlib.sha256(raw).hexdigest()


def write_npz_exclusive(path: Path, prediction: np.ndarray, target: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        np.savez_compressed(handle, predictions=prediction, targets=target)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    need(readonly(path), f"failed to seal immutable NPZ {path}")
    return sha256_file(path)


def load_canonical_json(path: Path) -> dict[str, Any]:
    need(readonly(path), f"missing/unsafe immutable JSON {path}")
    raw = path.read_bytes()
    value = json.loads(raw)
    need(isinstance(value, dict) and canonical_bytes(value) == raw, f"noncanonical JSON {path}")
    return value


def canonical_json_equivalent(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Compare receipt evidence after the only permitted JSON normalization.

    ``json`` represents tuples as arrays and integer object keys as strings.
    The full canonical byte comparison preserves every JSON-visible value while
    avoiding a false runtime/preflight drift from those representation details.
    """
    return canonical_bytes(dict(left)) == canonical_bytes(dict(right))


def wait_for_exact_sealed_json(path: Path, payload: Mapping[str, Any], *, attempts: int = 80) -> None:
    """Wait briefly for an O_EXCL peer to finish fsync+chmod, then compare exactly."""
    need(attempts > 0, "manifest wait attempts must be positive")
    for attempt in range(attempts):
        if readonly(path):
            need(load_canonical_json(path) == dict(payload), "existing early-start output manifest drift")
            return
        # A peer may have created the inode but not yet completed fsync/chmod.
        # Do not accept its bytes before the immutable 0444 seal is visible.
        if attempt + 1 < attempts:
            time.sleep(0.05)
    raise RuntimeError(f"existing output manifest did not become immutable within {attempts * 0.05:.2f}s: {path}")


def source_pins(repo_root: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for relative in SOURCE_RELATIVE_PATHS:
        path = (repo_root / relative).resolve()
        need(path.is_file() and not path.is_symlink(), f"missing/unsafe source {relative}")
        pins[relative] = sha256_file(path)
    return pins


def reference_artifact_path(reference_root: Path, asset_id: str, view: str, arm: str, seed: int) -> Path:
    return reference_root / "artifacts" / asset_id / view / arm / f"seed_{seed}" / "predictions_targets.npz"


def load_reference_targets(
    reference_root: Path, *, asset_id: str, view: str, expected_rows: int,
) -> dict[str, str]:
    """Prove that all original V9 arms/seeds share one post-50 target trace."""
    targets: dict[tuple[str, int], np.ndarray] = {}
    artifact_hashes: dict[str, str] = {}
    for arm in early.ARMS:
        for seed in early.SEEDS:
            path = reference_artifact_path(reference_root, asset_id, view, arm, seed)
            need(readonly(path), f"missing/unsafe V9 reference artifact {path}")
            with np.load(path, allow_pickle=False) as archive:
                target = np.ascontiguousarray(archive["targets"], dtype=np.float32)
            need(target.shape == (expected_rows, 2) and np.isfinite(target).all(), f"bad V9 reference target {path}")
            targets[(arm, seed)] = target
            artifact_hashes[f"{arm}/seed_{seed}"] = sha256_file(path)
    canonical_target = targets[("shared_t4", 42)]
    for key, target in targets.items():
        need(np.array_equal(target, canonical_target), f"V9 target mismatch across arms/seeds at {asset_id}/{view}: {key}")
    return {
        "target_sha256": sha_array(canonical_target),
        "artifact_sha256_by_arm_seed": artifact_hashes,
    }


def _array_evidence(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": sha_array(array),
        "finite": bool(np.isfinite(array).all()),
    }


def _compact_bridge(bridge: Mapping[str, Any], *, selection_indices: list[int]) -> dict[str, Any]:
    need(selection_indices == list(range(early.ACTIVITY_TRIALS)), "activity selection is not chronological first 30")
    need(bridge.get("raw_owner_chronology_sha256_before") == bridge.get("raw_owner_chronology_sha256_after"), "bridge mutated chronology")
    need(bridge.get("only_start_stop_cast") is True and bridge.get("bin_recomputation") is False, "bridge rule drift")
    return {
        "schema_bridge_rule": bridge.get("schema_bridge_rule"),
        "raw_owner_chronology_sha256": bridge.get("raw_owner_chronology_sha256_before"),
        "builder_chronology_sha256": bridge.get("builder_chronology_sha256"),
        "only_start_stop_cast": True,
        "bin_recomputation": False,
        "activity_selection_indices": selection_indices,
    }


def prepare_true_early_view(
    *, repo_root: Path, nwb_path: Path, view: str, contract_row: Any,
    behavior_mean: np.ndarray, behavior_std: np.ndarray, owners: Mapping[str, Any],
    reference_root: Path,
) -> tuple[Any, Any, dict[str, Any]]:
    """Build a score-blind, target-bound M30 early-start view once per session."""
    from sua_exploration.mc_maze.subm_co_scorer_adapter_parity_v5 import bridge_owner_chronology_for_c1_builder

    need(view in early.VIEWS, f"unknown view {view}")
    raw_trials = owners["list_datamodule_rewarded_trials"](
        nwb_path,
        bin_size_ms=parity_v3.BIN_SIZE_MS,
        window_size=early.HISTORY_BINS,
        trial_result_filter="R",
    )
    need(len(raw_trials) >= 50, "external V9 cohort must retain 50 rewarded trials")
    _raw_evidence, builder_trials, bridge = bridge_owner_chronology_for_c1_builder(raw_trials)
    selection = owners["selection"].select_calibration_trial_indices(
        builder_trials, early.ACTIVITY_TRIALS, early.ACTIVITY_TRIALS, "first"
    )
    selection = [int(value) for value in selection]
    compact_bridge = _compact_bridge(bridge, selection_indices=selection)

    # The early record is the actual deployment data path: first 30 rewarded
    # trials are excluded wholesale from query windows by the datamodule.
    record = owners["load_dandi688_session"](
        nwb_path,
        bin_size_ms=parity_v3.BIN_SIZE_MS,
        window_size=early.HISTORY_BINS,
        calibration_n_trials=early.ACTIVITY_TRIALS,
        max_trial_length=parity_v3.TRIAL_LENGTH_BINS,
        pad_value=parity_v3.PAD_VALUE,
        interpolate_trials=True,
        behavior_mean=behavior_mean,
        behavior_std=behavior_std,
        trial_result_filter="R",
        exclude_calibration_trials_from_windows=True,
        cache_dir=None,
        signal_view=view,
    )
    need(record.name == contract_row.session_id, f"session identity drift {record.name}")
    query_audit = early.audit_true_early_start_query(
        raw_trials, record.valid_starts,
        support_trials=early.ACTIVITY_TRIALS, history_bins=early.HISTORY_BINS,
    )
    need(query_audit.query_history_fully_after_support, "early query history crosses support boundary")
    rebuilt = owners["c1"].build_calib_trials_for_indices(
        {"neural": record.neural, "trials": builder_trials, "n_units": int(record.neural.shape[1])},
        selection,
        early.ACTIVITY_TRIALS,
    )
    rebuilt = np.ascontiguousarray(rebuilt, dtype=np.float32)
    need(rebuilt.shape[0] == early.ACTIVITY_TRIALS and rebuilt.shape[2] == record.neural.shape[1], "activity calibration shape drift")

    # A separate post-50 record verifies that the original frozen V9 query
    # set is an exact suffix of the new one.  Thus the late part can be
    # compared to V9 without target-window ambiguity after local scoring.
    post50_record = owners["load_dandi688_session"](
        nwb_path,
        bin_size_ms=parity_v3.BIN_SIZE_MS,
        window_size=early.HISTORY_BINS,
        calibration_n_trials=50,
        max_trial_length=parity_v3.TRIAL_LENGTH_BINS,
        pad_value=parity_v3.PAD_VALUE,
        interpolate_trials=True,
        behavior_mean=behavior_mean,
        behavior_std=behavior_std,
        trial_result_filter="R",
        exclude_calibration_trials_from_windows=True,
        cache_dir=None,
        signal_view=view,
    )
    need(post50_record.name == record.name, "M30/M50 session identity drift")
    need(int(post50_record.valid_starts.size) == int(contract_row.query_window_count), "original V9 post50 query count drift")
    need(
        np.array_equal(np.asarray(record.neural), np.asarray(post50_record.neural))
        and np.array_equal(np.asarray(record.behavior), np.asarray(post50_record.behavior)),
        "M30/M50 loader changed neural or behavior arrays",
    )
    post50_mask = early.post50_suffix_mask(record.valid_starts, post50_record.valid_starts)
    early_target = early.query_targets_from_record(record)
    suffix_target = np.ascontiguousarray(early_target[post50_mask], dtype=np.float32)
    post50_target = early.query_targets_from_record(post50_record)
    need(np.array_equal(suffix_target, post50_target), "M30 suffix target differs from freshly reconstructed post50 target")
    reference = load_reference_targets(
        reference_root, asset_id=contract_row.asset_id, view=view,
        expected_rows=int(post50_record.valid_starts.size),
    )
    need(sha_array(suffix_target) == reference["target_sha256"], "M30 post50 suffix target differs from frozen V9 target")
    design = budget_core.audit_t4_design(
        raw_trials, early.T4_FIT_POOL_TRIALS,
        nearest_direction=unit_side_features._nearest_canonical_direction_index,
        canonical_directions=unit_side_features.CANONICAL_DIRECTIONS_RAD,
    )
    evidence = {
        "session_id": record.name,
        "view": view,
        "activity_identity": {
            "trials": early.ACTIVITY_TRIALS,
            "chronological_first": True,
            "calibration": _array_evidence(rebuilt),
            "bridge": compact_bridge,
        },
        "t4_ts4_fit": {
            "trials": early.T4_FIT_POOL_TRIALS,
            "chronological_first": True,
            "design": design.as_dict(),
        },
        "query": {
            **query_audit.as_dict(),
            "valid_starts_sha256": sha_array(np.asarray(record.valid_starts, dtype=np.int64)),
            "target_sha256": sha_array(early_target),
            "target_shape": list(early_target.shape),
            "strictly_after_rewarded_trial": early.ACTIVITY_TRIALS,
            "post50_suffix_window_count": int(post50_mask.sum()),
            "additional_true_early_windows": int((~post50_mask).sum()),
            "post50_suffix_valid_starts_sha256": sha_array(np.asarray(post50_record.valid_starts, dtype=np.int64)),
            "post50_suffix_target_sha256": sha_array(suffix_target),
            "frozen_v9_reference_artifact_sha256_by_arm_seed": reference["artifact_sha256_by_arm_seed"],
        },
        "base_arrays": {
            "neural": _array_evidence(record.neural),
            "behavior": _array_evidence(record.behavior),
        },
    }
    return record, rebuilt, evidence


def dataset_for_arm(
    *, nwb_path: Path, view: str, arm: str, seed: int, record: Any, rebuilt: np.ndarray,
    side_mean: np.ndarray | None, side_std: np.ndarray | None, owners: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    need(arm in early.ARMS and seed in early.SEEDS, "invalid arm/seed")
    if arm == "shared_zero4":
        features = np.zeros((record.neural.shape[1], 4), dtype=np.float32)
        need(np.all(features.view(np.uint32) == np.uint32(0)), "zero4 lost exact positive zeros")
        metadata: Mapping[str, Any] = {}
        descriptor = {
            "kind": "direct_standardized_zero4",
            "label_or_rate_fit_trials": 0,
            "permutation_seed": None,
            "source_side_normalizer_used": False,
        }
    else:
        need(side_mean is not None and side_std is not None, f"{arm} needs frozen source side normalizer")
        feature_group = "t4" if arm == "shared_t4" else "ts4"
        permutation_seed = None if arm == "shared_t4" else seed
        features, metadata = owners["load_unit_side_features"](
            nwb_path,
            feature_group=feature_group,
            pool_size=early.T4_FIT_POOL_TRIALS,
            mean=side_mean,
            std=side_std,
            cache_dir=None,
            permutation_seed=permutation_seed,
            bin_size_ms=parity_v3.BIN_SIZE_MS,
            window_size=early.HISTORY_BINS,
            trial_result_filter="R",
            signal_view=view,
        )
        descriptor = {
            "kind": feature_group,
            "label_or_rate_fit_trials": early.T4_FIT_POOL_TRIALS,
            "permutation_seed": permutation_seed,
            "source_side_normalizer_used": True,
        }
    features = np.ascontiguousarray(features, dtype=np.float32)
    need(features.shape == (record.neural.shape[1], 4) and np.isfinite(features).all(), "side feature shape/value drift")
    dataset = owners["MCMazeSessionDataset"](
        neural_data=record.neural,
        behavior_data=record.behavior,
        valid_starts=record.valid_starts,
        calib_trials=rebuilt,
        window_size=early.HISTORY_BINS,
        session_name=record.name,
        side_features=features,
        electrode_ids=None,
    )
    need(len(dataset) == int(record.valid_starts.size), "dataset query count drift")
    return dataset, {
        **descriptor,
        "feature_shape": list(features.shape),
        "feature_sha256": sha_array(features),
        "loader_metadata_pool_size": metadata.get("pool_size") if isinstance(metadata, Mapping) else None,
    }


def preflight_payload(
    *, repo_root: Path, nwb_root: Path, input_manifest: Path, reference_v9_root: Path,
) -> dict[str, Any]:
    contract, input_sha = runtime.load_input_manifest(input_manifest)
    runtime.verify_runtime_inputs(contract=contract, nwb_root=nwb_root)
    need(len(contract.cohort) == early.EXPECTED_SESSIONS, "V9 cohort count drift")
    reference_manifest = reference_v9_root / "run_manifest.json"
    need(readonly(reference_manifest), f"missing/unsafe V9 reference manifest {reference_manifest}")
    owners = runtime._runtime_owners(repo_root)
    behavior_stats = {
        view: runtime._load_mean_std(contract.behavior_normalizer_paths[view], label=f"{view} behavior")
        for view in early.VIEWS
    }
    rows: list[dict[str, Any]] = []
    for cohort_row in contract.cohort:
        session = {
            "asset_id": cohort_row.asset_id,
            "session_id": cohort_row.session_id,
            "frozen_path": cohort_row.frozen_path,
            "nwb_sha256": cohort_row.nwb_sha256,
            "nwb_bytes": cohort_row.nwb_bytes,
            "views": {},
        }
        nwb_path = nwb_root / cohort_row.frozen_path
        for view in early.VIEWS:
            mean, std = behavior_stats[view]
            _record, _rebuilt, evidence = prepare_true_early_view(
                repo_root=repo_root, nwb_path=nwb_path, view=view, contract_row=cohort_row,
                behavior_mean=mean, behavior_std=std, owners=owners, reference_root=reference_v9_root,
            )
            session["views"][view] = evidence
        rows.append(session)
    expected_full_cells = early.EXPECTED_CELLS
    expected_query_rows_per_view = {
        view: int(sum(row["views"][view]["query"]["observed_query_window_count"] for row in rows))
        for view in early.VIEWS
    }
    payload = {
        "schema": early.SCHEMA,
        "status": "FULL_15_SESSION_PREFLIGHT_PASS_NO_FORWARD_NO_METRIC",
        "claim_boundary": {
            "scope": "post-hoc same-external-cohort frozen-V9-system M30 true-early-start latency characterization",
            "not_new_external_confirmation": True,
            "not_low_budget_retraining": True,
            "target_session_backpropagation": False,
            "remote_metric_computation": False,
        },
        "protocol": {
            "activity_trials": early.ACTIVITY_TRIALS,
            "t4_ts4_fit_trials": early.T4_FIT_POOL_TRIALS,
            "query_rule": "rewarded_trials_strictly_after_30; every_50_bin_history_wholly_post_trial30",
            "views": list(early.VIEWS),
            "arms": list(early.ARMS),
            "seeds": list(early.SEEDS),
            "expected_full_cells": expected_full_cells,
        },
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": input_sha,
        "runtime_contract_sha256": contract.sha256,
        "reference_v9_root": str(reference_v9_root),
        "reference_v9_manifest_sha256": sha256_file(reference_manifest),
        "nwb_root": str(nwb_root),
        "source_sha256_by_relative_path": source_pins(repo_root),
        "expected_query_rows_per_view": expected_query_rows_per_view,
        "cohort": rows,
        "metric_computed": False,
        "backward_called": False,
    }
    return payload


def write_or_verify_receipt(path: Path, payload: dict[str, Any]) -> str:
    if path.exists():
        existing = load_canonical_json(path)
        need(existing == payload, "existing early-start preflight receipt drift")
        return sha256_file(path)
    try:
        return write_json_exclusive(path, payload)
    except FileExistsError:
        existing = load_canonical_json(path)
        need(existing == payload, "raced early-start preflight receipt drift")
        return sha256_file(path)


def load_receipt(path: Path, *, repo_root: Path, input_manifest: Path, reference_v9_root: Path) -> tuple[dict[str, Any], str]:
    receipt = load_canonical_json(path)
    need(receipt.get("schema") == early.SCHEMA, "early-start receipt schema drift")
    need(receipt.get("status") == "FULL_15_SESSION_PREFLIGHT_PASS_NO_FORWARD_NO_METRIC", "receipt is not full preflight")
    need(receipt.get("input_manifest") == str(input_manifest), "receipt input manifest path drift")
    need(receipt.get("reference_v9_root") == str(reference_v9_root), "receipt V9 root drift")
    need(receipt.get("source_sha256_by_relative_path") == source_pins(repo_root), "source hash differs from preflight receipt")
    need(receipt.get("protocol", {}).get("expected_full_cells") == early.EXPECTED_CELLS, "receipt cell topology drift")
    cohort = receipt.get("cohort")
    need(isinstance(cohort, list) and len(cohort) == early.EXPECTED_SESSIONS, "receipt cohort drift")
    return receipt, sha256_file(path)


def cell_paths(output_root: Path, asset_id: str, view: str, arm: str, seed: int) -> tuple[Path, Path]:
    artifact = output_root / "artifacts" / asset_id / view / arm / f"seed_{seed}" / "predictions_targets.npz"
    commit = output_root / "commits" / asset_id / view / arm / f"seed_{seed}.json"
    return artifact, commit


def receipt_session_view(receipt: Mapping[str, Any], *, asset_id: str, view: str) -> dict[str, Any]:
    matches = [row for row in receipt["cohort"] if row["asset_id"] == asset_id]
    need(len(matches) == 1, f"receipt lacks cohort asset {asset_id}")
    value = matches[0].get("views", {}).get(view)
    need(isinstance(value, dict), f"receipt lacks view {asset_id}/{view}")
    return value


def validate_existing_cell(
    *, artifact: Path, commit_path: Path, expected_target: np.ndarray, expected: Mapping[str, Any],
) -> None:
    need(readonly(artifact) and readonly(commit_path), f"orphan/unsafe existing cell {artifact}")
    commit = load_canonical_json(commit_path)
    need(commit.get("metric_computed") is False and commit.get("backward_called") is False, "existing cell policy drift")
    need(commit.get("artifact_sha256") == sha256_file(artifact), "existing cell artifact hash drift")
    need(commit.get("query_target_sha256") == expected["query"]["target_sha256"], "existing cell target receipt drift")
    with np.load(artifact, allow_pickle=False) as archive:
        target = np.ascontiguousarray(archive["targets"], dtype=np.float32)
    need(np.array_equal(target, expected_target), f"existing cell target differs from bound early target {artifact}")


def initialize_output(
    *, output_root: Path, receipt_path: Path, receipt_sha: str, receipt: Mapping[str, Any],
    input_manifest: Path, device: str,
) -> None:
    payload = {
        "schema": early.SCHEMA,
        "status": "FORWARD_ONLY_NO_METRIC",
        "post_hoc_scope": receipt["claim_boundary"]["scope"],
        "preflight_receipt": str(receipt_path),
        "preflight_receipt_sha256": receipt_sha,
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": receipt["input_manifest_sha256"],
        "runtime_contract_sha256": receipt["runtime_contract_sha256"],
        "reference_v9_root": receipt["reference_v9_root"],
        "reference_v9_manifest_sha256": receipt["reference_v9_manifest_sha256"],
        "source_sha256_by_relative_path": receipt["source_sha256_by_relative_path"],
        "protocol": receipt["protocol"],
        "execution_device": runtime.execution_device_binding(device),
        "metric_computed": False,
        "target_session_backpropagation": False,
        "overwrite_policy": "immutable_missing_only",
    }
    path = output_root / "run_manifest.json"
    if path.exists():
        wait_for_exact_sealed_json(path, payload)
        return
    try:
        write_json_exclusive(path, payload)
    except FileExistsError:
        wait_for_exact_sealed_json(path, payload)


def run_forward(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    nwb_root = args.nwb_root.resolve()
    input_manifest = args.input_manifest.resolve()
    reference_root = args.reference_v9_root.resolve()
    output_root = args.output_root.resolve()
    receipt_path = args.preflight_receipt.resolve()
    receipt, receipt_sha = load_receipt(
        receipt_path, repo_root=repo_root, input_manifest=input_manifest, reference_v9_root=reference_root
    )
    contract, input_sha = runtime.load_input_manifest(input_manifest)
    need(input_sha == receipt["input_manifest_sha256"] and contract.sha256 == receipt["runtime_contract_sha256"], "input contract drift after receipt")
    runtime.verify_runtime_inputs(contract=contract, nwb_root=nwb_root)
    need(sha256_file(reference_root / "run_manifest.json") == receipt["reference_v9_manifest_sha256"], "V9 reference manifest drift")
    initialize_output(
        output_root=output_root, receipt_path=receipt_path, receipt_sha=receipt_sha, receipt=receipt,
        input_manifest=input_manifest, device=args.device,
    )
    owners = runtime._runtime_owners(repo_root)
    behavior_stats = {
        view: runtime._load_mean_std(contract.behavior_normalizer_paths[view], label=f"{view} behavior")
        for view in early.VIEWS
    }
    side_stats = {
        view: runtime._load_mean_std(contract.side_normalizer_paths[view], label=f"{view} side")
        for view in early.VIEWS
    }
    stop = args.session_offset + args.max_sessions if args.max_sessions is not None else len(contract.cohort)
    cohort = contract.cohort[args.session_offset:stop]
    need(cohort, "selected zero sessions")
    models: dict[tuple[str, int], Any] = {}
    completed = 0
    skipped = 0
    for row in cohort:
        nwb_path = nwb_root / row.frozen_path
        for view in early.VIEWS:
            mean, std = behavior_stats[view]
            record, rebuilt, prepared = prepare_true_early_view(
                repo_root=repo_root, nwb_path=nwb_path, view=view, contract_row=row,
                behavior_mean=mean, behavior_std=std, owners=owners, reference_root=reference_root,
            )
            expected = receipt_session_view(receipt, asset_id=row.asset_id, view=view)
            need(
                canonical_json_equivalent(prepared, expected),
                f"runtime M30 source/query evidence drift after preflight: {row.asset_id}/{view}",
            )
            expected_target = early.query_targets_from_record(record)
            datasets: dict[tuple[str, int], tuple[Any, dict[str, Any]]] = {}
            for arm in early.ARMS:
                descriptor_seeds = (42,) if arm == "shared_t4" else ((0,) if arm == "shared_zero4" else early.SEEDS)
                for descriptor_seed in descriptor_seeds:
                    dataset, descriptor = dataset_for_arm(
                        nwb_path=nwb_path, view=view, arm=arm,
                        seed=42 if descriptor_seed == 0 else descriptor_seed,
                        record=record, rebuilt=rebuilt,
                        side_mean=None if arm == "shared_zero4" else side_stats[view][0],
                        side_std=None if arm == "shared_zero4" else side_stats[view][1], owners=owners,
                    )
                    datasets[(arm, descriptor_seed)] = (dataset, descriptor)
            for arm in early.ARMS:
                for seed in early.SEEDS:
                    artifact, commit_path = cell_paths(output_root, row.asset_id, view, arm, seed)
                    if artifact.exists() or commit_path.exists():
                        validate_existing_cell(
                            artifact=artifact, commit_path=commit_path, expected_target=expected_target, expected=expected,
                        )
                        skipped += 1
                        continue
                    model_key = (arm, seed)
                    if model_key not in models:
                        models[model_key] = runtime._load_model(contract.checkpoint(arm, seed), contract, args.device, owners)
                    descriptor_seed = seed if arm == "shared_ts4" else (42 if arm == "shared_t4" else 0)
                    dataset, descriptor = datasets[(arm, descriptor_seed)]
                    prediction, target, forward = runtime.collect_forward_predictions(
                        model=models[model_key], dataset=dataset, device=args.device, owners=owners,
                    )
                    need(np.array_equal(target, expected_target), f"target mismatch across arm/seed {row.asset_id}/{view}/{arm}/{seed}")
                    artifact_sha = write_npz_exclusive(artifact, prediction, target)
                    payload = {
                        "schema": early.SCHEMA,
                        "asset_id": row.asset_id,
                        "session_id": row.session_id,
                        "view": view,
                        "arm": arm,
                        "seed": seed,
                        "artifact": str(artifact.relative_to(output_root)),
                        "artifact_sha256": artifact_sha,
                        "artifact_bytes": artifact.stat().st_size,
                        "prediction_sha256": sha_array(prediction),
                        "query_target_sha256": sha_array(target),
                        "query_window_count": int(target.shape[0]),
                        "preflight_query_target_sha256": expected["query"]["target_sha256"],
                        "preflight_post50_suffix_target_sha256": expected["query"]["post50_suffix_target_sha256"],
                        "checkpoint_sha256": contract.checkpoint(arm, seed).sha256,
                        "checkpoint_path": contract.checkpoint(arm, seed).path,
                        "runtime_contract_sha256": contract.sha256,
                        "descriptor": descriptor,
                        "preflight_view_evidence": expected,
                        "forward": forward,
                        "metric_computed": False,
                        "backward_called": False,
                    }
                    write_json_exclusive(commit_path, payload)
                    completed += 1
    return {
        "status": "FORWARD_ARTIFACTS_COMMITTED_NO_METRIC",
        "selected_sessions": len(cohort),
        "expected_selected_cells": early.expected_cells_for_session_count(len(cohort)),
        "completed_cells": completed,
        "skipped_cells": skipped,
        "output_root": str(output_root),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("preflight", "forward"))
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--nwb-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--reference-v9-root", type=Path, required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--session-offset", type=int, default=0)
    parser.add_argument("--max-sessions", type=int)
    args = parser.parse_args()
    if args.mode == "forward" and args.output_root is None:
        parser.error("--output-root is required for forward")
    if args.max_sessions is not None and not 1 <= args.max_sessions <= early.EXPECTED_SESSIONS:
        parser.error("--max-sessions must be in [1,15]")
    if not 0 <= args.session_offset < early.EXPECTED_SESSIONS:
        parser.error("--session-offset must be in [0,14]")
    if args.max_sessions is not None and args.session_offset + args.max_sessions > early.EXPECTED_SESSIONS:
        parser.error("session slice exceeds frozen 15-session cohort")
    if args.mode == "preflight" and (args.session_offset != 0 or args.max_sessions is not None):
        parser.error("preflight is intentionally full-cohort; do not slice it")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "preflight":
        payload = preflight_payload(
            repo_root=args.repo_root.resolve(), nwb_root=args.nwb_root.resolve(),
            input_manifest=args.input_manifest.resolve(), reference_v9_root=args.reference_v9_root.resolve(),
        )
        digest = write_or_verify_receipt(args.preflight_receipt.resolve(), payload)
        print(json.dumps({
            "status": payload["status"],
            "receipt": str(args.preflight_receipt.resolve()),
            "sha256": digest,
            "expected_full_cells": payload["protocol"]["expected_full_cells"],
            "expected_query_rows_per_view": payload["expected_query_rows_per_view"],
        }, indent=2, sort_keys=True), flush=True)
        return
    result = run_forward(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
