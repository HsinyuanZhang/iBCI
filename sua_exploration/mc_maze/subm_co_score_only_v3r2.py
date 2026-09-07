"""Authorization-first V3R2 executor for the frozen external sub-M matrix.

This module imports only the standard library and the authorization-only V3
core at import time. NumPy, Torch, NWB owners, checkpoints, normalizers, and
the scorer become reachable only after a valid detached authorization has
been verified and its fresh nonce has been atomically claimed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from sua_exploration.mc_maze import subm_co_score_only_v3 as auth_v3


REPO_ROOT = Path(__file__).resolve().parents[2]


class ScoreV3R2ExecutionError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ScoreV3R2ExecutionError(message)


def _compact_bridge_trace(
    trace: Mapping[str, Any], *, indices: list[int], support_trials: int,
    identity_trials: int, query_count: int, view: str,
) -> dict[str, Any]:
    conversions = trace.get("start_stop_conversions")
    chronology = trace.get("raw_owner_chronology")
    _need(isinstance(conversions, list) and isinstance(chronology, list), "V5 bridge trace is incomplete")
    _need(indices == list(range(identity_trials)), "activity selection is not chronological first_n30")
    return {
        "schema": "dandi_000688_subm_co_v3r2_adapter_trace_v1",
        "view": view,
        "schema_bridge_rule": trace.get("schema_bridge_rule"),
        "raw_owner_trial_count": len(chronology),
        "start_stop_conversion_count": len(conversions),
        "raw_owner_chronology_sha256_before": trace.get("raw_owner_chronology_sha256_before"),
        "raw_owner_chronology_sha256_after": trace.get("raw_owner_chronology_sha256_after"),
        "builder_chronology_sha256": trace.get("builder_chronology_sha256"),
        "only_start_stop_cast": trace.get("only_start_stop_cast"),
        "bin_recomputation": trace.get("bin_recomputation"),
        "trial_selection_changed_by_schema_bridge": trace.get("trial_selection_changed_by_schema_bridge"),
        "t4_changed_by_schema_bridge": trace.get("t4_changed_by_schema_bridge"),
        "query_valid_starts_changed_by_schema_bridge": trace.get("query_valid_starts_changed_by_schema_bridge"),
        "activity_selection_indices": indices,
        "activity_identity_trials": identity_trials,
        "t4_fit_pool_trials": support_trials,
        "query_rule": "owner-loader valid_starts strictly after first 50 rewarded trials",
        "query_window_count": query_count,
    }


def _build_view_base(
    *, nwb_path: Path, view: str, normalizer: Mapping[str, Any], owners: Mapping[str, Any],
) -> tuple[Any, Any, list[dict[str, Any]], dict[str, Any]]:
    from sua_exploration.mc_maze import subm_co_scorer_adapter_parity_v3 as parity_v3
    from sua_exploration.mc_maze.subm_co_scorer_adapter_parity_v5 import (
        bridge_owner_chronology_for_c1_builder,
    )

    _need(view in {"sua", "pseudo_mua"}, "V3R2 view drift")
    record = owners["load_dandi688_session"](
        nwb_path,
        bin_size_ms=parity_v3.BIN_SIZE_MS,
        window_size=parity_v3.HISTORY_BINS,
        calibration_n_trials=parity_v3.SUPPORT_TRIALS,
        max_trial_length=parity_v3.TRIAL_LENGTH_BINS,
        pad_value=parity_v3.PAD_VALUE,
        interpolate_trials=True,
        behavior_mean=normalizer["behavior_mean"],
        behavior_std=normalizer["behavior_std"],
        trial_result_filter="R",
        exclude_calibration_trials_from_windows=True,
        cache_dir=None,
        signal_view=view,
    )
    raw_trials = owners["list_datamodule_rewarded_trials"](
        nwb_path,
        bin_size_ms=parity_v3.BIN_SIZE_MS,
        window_size=parity_v3.HISTORY_BINS,
        trial_result_filter="R",
    )
    _raw_evidence, builder_trials, bridge_trace = bridge_owner_chronology_for_c1_builder(raw_trials)
    indices = owners["selection"].select_calibration_trial_indices(
        builder_trials, parity_v3.IDENTITY_TRIALS, parity_v3.SUPPORT_TRIALS, "first"
    )
    indices = [int(value) for value in indices]
    rebuilt = owners["c1"].build_calib_trials_for_indices(
        {"neural": record.neural, "trials": builder_trials, "n_units": int(record.neural.shape[1])},
        indices,
        parity_v3.IDENTITY_TRIALS,
    )
    _need(int(rebuilt.shape[0]) == 30, "V3R2 activity calibration tensor is not 30 trials")
    _need(int(record.valid_starts.size) > 0, "V3R2 post-50 query is empty")
    compact = _compact_bridge_trace(
        bridge_trace,
        indices=indices,
        support_trials=parity_v3.SUPPORT_TRIALS,
        identity_trials=parity_v3.IDENTITY_TRIALS,
        query_count=int(record.valid_starts.size),
        view=view,
    )
    _need(compact["raw_owner_chronology_sha256_before"] == compact["raw_owner_chronology_sha256_after"], "V5 bridge mutated chronology")
    _need(compact["only_start_stop_cast"] is True and compact["bin_recomputation"] is False, "V5 bridge rule drift")
    _need(
        compact["trial_selection_changed_by_schema_bridge"] is False
        and compact["t4_changed_by_schema_bridge"] is False
        and compact["query_valid_starts_changed_by_schema_bridge"] is False,
        "V5 bridge changed deployment semantics",
    )
    return record, rebuilt, builder_trials, compact


def _dataset_for_cell(
    *, nwb_path: Path, view: str, arm: str, seed: int, record: Any,
    rebuilt_calibration: Any, normalizer: Mapping[str, Any], owners: Mapping[str, Any],
) -> tuple[Any, int]:
    from sua_exploration.mc_maze import subm_co_scorer_adapter_parity_v3 as parity_v3

    _need(arm in {"shared_t4", "shared_ts4"} and seed in {42, 43, 44}, "V3R2 score-cell identity drift")
    feature_group = "t4" if arm == "shared_t4" else "ts4"
    permutation_seed = None if arm == "shared_t4" else seed
    features, _metadata = owners["load_unit_side_features"](
        nwb_path,
        feature_group=feature_group,
        pool_size=parity_v3.SUPPORT_TRIALS,
        mean=normalizer["side_mean"],
        std=normalizer["side_std"],
        cache_dir=None,
        permutation_seed=permutation_seed,
        bin_size_ms=parity_v3.BIN_SIZE_MS,
        window_size=parity_v3.HISTORY_BINS,
        trial_result_filter="R",
        signal_view=view,
    )
    _need(features.shape == (record.neural.shape[1], 4), "V3R2 T4/TS4 descriptor shape drift")
    dataset = owners["MCMazeSessionDataset"](
        neural_data=record.neural,
        behavior_data=record.behavior,
        valid_starts=record.valid_starts,
        calib_trials=rebuilt_calibration,
        window_size=parity_v3.HISTORY_BINS,
        session_name=record.name,
        side_features=features,
        electrode_ids=None,
    )
    _need(len(dataset) == int(record.valid_starts.size), "V3R2 dataset/query count drift")
    return dataset, int(features.shape[0])


def score_frozen_subm_matrix_via_v5_bridge_v3r2(
    *, root: Path, external_nwb_root: Path, output_root: Path,
    authorization_path: Path, grant: auth_v3.AuthorizationGrant,
) -> dict[str, Any]:
    """Run and seal the fixed 180-cell matrix after V3 authorization only."""
    import numpy as np
    from sua_exploration.mc_maze import subm_co_score_only as v1
    from sua_exploration.mc_maze import subm_co_score_only_v2 as v2
    from sua_exploration.mc_maze import subm_co_scorer_adapter_parity_v3 as parity_v3

    root = root.resolve()
    output_root = output_root.resolve()
    _need(root == REPO_ROOT.resolve(), "V3R2 repository root substitution")
    counters = v1.RuntimeAuditCounters()
    contract = v1.validate_authority_chain(root, counters=counters)
    v1._verify_runtime_artifacts(root, contract, counters)
    nwb_paths = v1._frozen_nwb_paths(external_nwb_root, contract.frozen_sessions, counters)
    expected_counts = v2.expected_query_window_counts_from_preflight(contract, root)
    allowed = [root / item.checkpoint.relative_path for item in contract.terminal_checkpoints]
    allowed.append(root / contract.teacher_checkpoint.relative_path)
    cpu_binding = v2.execution_device_binding_from_mapping(
        {
            "torch_device": "cpu",
            "kind": "cpu",
            "matrix_torch_device": "cpu",
            "host": None,
            "physical_gpu_uuid": None,
            "physical_gpu_pci_bus_id": None,
            "cuda_visible_devices": None,
            "all_180_cells_same_device": True,
        }
    )
    writer = v2.SealedOutputWriterV2(
        output_root,
        contract,
        authorization_sha256=grant.authorization_sha256,
        execution_device=cpu_binding,
        authority_root=root,
    )
    trace_records: list[dict[str, Any]] = []
    with v1.RuntimeSafetyFence(allowed_checkpoint_paths=allowed, counters=counters) as fence:
        dependencies = v1._load_scoring_dependencies(root)
        owners = parity_v3._runtime_owners()
        fence.install_target_data_bans()
        _need(
            dependencies["normalizer_hashes_are_distinct"](
                contract.normalizers["sua"].side_feature_semantic_sha256,
                contract.normalizers["pseudo_mua"].side_feature_semantic_sha256,
            ),
            "V3R2 view-local normalizer isolation failed",
        )
        normalizers = v1._load_precomputed_normalizers(root, contract, dependencies)
        device = dependencies["torch"].device("cpu")
        models: dict[tuple[str, int], Any] = {}
        for terminal in contract.terminal_checkpoints:
            model = dependencies["load_frozen_model"](
                root / terminal.checkpoint.relative_path,
                root / contract.teacher_checkpoint.relative_path,
                "B3S",
                device,
            )
            fence.monitor_model_forward(model)
            models[(terminal.arm, terminal.seed)] = model

        for session in contract.frozen_sessions:
            nwb_path = nwb_paths[session.asset_id]
            bases: dict[str, tuple[Any, Any]] = {}
            for view in v1.VIEWS:
                record, rebuilt, _builder_trials, trace = _build_view_base(
                    nwb_path=nwb_path,
                    view=view,
                    normalizer=normalizers[view],
                    owners=owners,
                )
                _need(int(record.valid_starts.size) == expected_counts[session.asset_id], "V3R2 pinned query count drift")
                bases[view] = (record, rebuilt)
                trace_relative = f"sealed/adapter_traces/{session.asset_id}/{view}.json"
                trace_path = output_root / trace_relative
                trace_payload = {
                    **trace,
                    "asset_id": session.asset_id,
                    "session_id": session.session_id,
                    "frozen_path": session.frozen_path,
                }
                trace_sha = v2._write_immutable_json_exclusive(trace_path, trace_payload)
                trace_records.append({"relative_path": trace_relative, "sha256": trace_sha})

            datasets: dict[tuple[str, str, int], tuple[Any, int]] = {}
            for view in v1.VIEWS:
                record, rebuilt = bases[view]
                t4_dataset = _dataset_for_cell(
                    nwb_path=nwb_path, view=view, arm="shared_t4", seed=42,
                    record=record, rebuilt_calibration=rebuilt,
                    normalizer=normalizers[view], owners=owners,
                )
                for seed in v1.SEEDS:
                    datasets[(view, "shared_t4", seed)] = t4_dataset
                    datasets[(view, "shared_ts4", seed)] = _dataset_for_cell(
                        nwb_path=nwb_path, view=view, arm="shared_ts4", seed=seed,
                        record=record, rebuilt_calibration=rebuilt,
                        normalizer=normalizers[view], owners=owners,
                    )
            v1._verify_paired_view_alignment(
                datasets[("sua", "shared_t4", 42)][0],
                datasets[("pseudo_mua", "shared_t4", 42)][0],
                dependencies,
            )
            for seed in v1.SEEDS:
                for arm in v1.ARMS:
                    terminal = contract.checkpoint_by_key()[(arm, seed)]
                    for view in v1.VIEWS:
                        dataset, channel_count = datasets[(view, arm, seed)]
                        _score, predictions, targets = v1._score_one_session_dataset(
                            model=models[(arm, seed)],
                            dataset=dataset,
                            dependencies=dependencies,
                            counters=counters,
                        )
                        _need(np.isfinite(predictions).all() and np.isfinite(targets).all(), "V3R2 nonfinite prediction/target")
                        permutation = None if arm == "shared_t4" else v2.realized_ts4_permutation(channel_count, seed)
                        writer.write_session_result(
                            session=session,
                            view=view,
                            arm=arm,
                            seed=seed,
                            checkpoint_sha256=terminal.checkpoint.sha256,
                            normalizer=contract.normalizers[view],
                            predictions=predictions,
                            targets=targets,
                            query_window_count=len(dataset),
                            ts4_permutation=permutation,
                            ts4_feature_channel_count=None if arm == "shared_t4" else channel_count,
                        )
    seal = writer.finalize()
    aggregate = v2.aggregate_sealed_v2(output_root, contract)
    receipt_path = output_root / "v3r2_execution_receipt.json"
    receipt = {
        "schema": "dandi_000688_subm_co_score_only_execution_receipt_v3r2",
        "status": "COMPLETE_FROZEN_180_CELL_MATRIX_AND_AGGREGATE",
        "authorization": {
            "path": str(authorization_path.resolve()),
            "sha256": grant.authorization_sha256,
            "signature_sha256": grant.signature_sha256,
            "nonce_claim": {
                "path": str(grant.claim_path),
                "sha256": auth_v3.sha256_file(grant.claim_path),
            },
        },
        "deployment_budget": {
            "activity_identity_trials": 30,
            "t4_fit_pool_trials": 50,
            "query_start": "strictly_after_rewarded_trial_50",
        },
        "cell_count": 180,
        "adapter_traces": sorted(trace_records, key=lambda row: row["relative_path"]),
        "matrix_seal": seal,
        "aggregate": aggregate,
        "claim_boundary": "shared_t4_vs_shared_ts4_only",
        "external_subm_scoring_performed": True,
        "runtime_audit_counters": counters.snapshot(),
    }
    receipt_sha = v2._write_immutable_json_exclusive(receipt_path, receipt)
    return {
        "status": receipt["status"],
        "output_root": str(output_root),
        "execution_receipt": {"path": str(receipt_path), "sha256": receipt_sha},
        "aggregate_status": aggregate["status"],
        "overall_mechanism_pass": aggregate["overall_mechanism_pass"],
    }


def execute_from_fixed_prelaunch_v3r2(
    *, authorization_path: Path, signature_path: Path, output_root: Path,
    external_nwb_root: Path, prelaunch_dir: Path, root: Path = REPO_ROOT,
) -> dict[str, Any]:
    if root.resolve() != REPO_ROOT.resolve():
        raise auth_v3.ScoreV3AuthorizationError("V3R2 production runner requires fixed repository root")
    from sua_exploration.scripts.write_dandi688_subm_co_score_only_prelaunch_v3r2 import (
        canonical_bytes,
        load_stored_prelaunch,
    )

    stored = load_stored_prelaunch(prelaunch_dir, root)
    policy = stored["execution_policy"]
    without_sha = {key: value for key, value in policy.items() if key != "execution_policy_sha256"}
    actual_sha = hashlib.sha256(canonical_bytes(without_sha)).hexdigest()
    if (
        policy.get("execution_policy_sha256") != stored.get("execution_policy_sha256")
        or actual_sha != stored.get("execution_policy_sha256")
    ):
        raise auth_v3.ScoreV3AuthorizationError("stored V3R2 policy hash mismatch")
    grant = auth_v3._preauthorize_and_claim_with_policy(
        authorization_path=authorization_path,
        signature_path=signature_path,
        output_root=output_root,
        external_nwb_root=external_nwb_root,
        policy=policy,
    )
    auth_v3._set_cpu_environment()
    return score_frozen_subm_matrix_via_v5_bridge_v3r2(
        root=root,
        external_nwb_root=external_nwb_root,
        output_root=output_root,
        authorization_path=authorization_path,
        grant=grant,
    )
