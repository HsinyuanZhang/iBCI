"""Additive no-target executor/scorer contract for Track-B v2 RT development.

This module consumes the canonical development materializer and the sealed RT
15-session local-asset authority, then inspects the sole fixed d8/it250 GPU
engineering-cost receipt.  It has no target loader, CEBRA import, model,
readout, scorer, writer, or execution mode.  A missing/invalid cost receipt is
an explicit NO-GO before a held path is resolved or opened.

The future scientific geometry is fixed independently of scores at d8/it10000
with normalized ridge lambda 0.01 and cosine kNN k=3.  One legal joint
source-plus-held-M24-support encoder must serve all three mandatory readout
routes.  Held query rows never enter encoder or readout fit.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import track_b_v2_contract as base
import track_b_v2_development_target_materializer as materializer
import track_b_v2_fixed_gpu_engineering as fixed_gpu
import track_b_v2_source_adapter as source
import track_b_v2_subject_m_development_executor as subject_executor


SCHEMA = "track_b_v2_rt_development_executor_scorer_dry_plan_v1"
STATUS_COST_BLOCKED = "NO_GO__FIXED_D8IT250_GPU_COST_RECEIPT_REQUIRED_BEFORE_RT_TARGET_DESCRIPTOR"
STATUS_IMPLEMENTATION_BLOCKED = "GPU_COST_GATE_VALID__FUTURE_RT_EXECUTOR_AND_SCORER_STILL_NOT_IMPLEMENTED"
LOCAL_ASSET_AUTHORITY_SHA256 = "771ab920531a322fbccc9a745d80cc1cb6b3325bcc469283161e2561f7da9352"
SEEDS = (42, 43, 44)
DECODERS = ("linear_ridge", "knn_cosine_k3")
READOUT_ROUTES = (
    "source_only_consumer_mechanism_alignment",
    "target_support_only_standard_cebra_accuracy",
    "source_plus_target_support_hybrid_sensitivity",
)
REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / "cebra_exploration/scripts/run_track_b_v2_rt_development_executor.py"


class TrackBV2RTDevelopmentExecutorError(materializer.TrackBV2DevelopmentTargetMaterializerError):
    """Fail-closed error raised before any RT held descriptor may be resolved."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TrackBV2RTDevelopmentExecutorError(message)


def _sha_json(value: Any) -> str:
    return hashlib.sha256(base.canonical_json_bytes(value)).hexdigest()


def _implementation_closure() -> dict[str, dict[str, Any]]:
    paths = {
        "rt_executor_core": Path(__file__).absolute(),
        "rt_executor_cli": CLI,
        "canonical_development_materializer": Path(materializer.__file__).absolute(),
        "rt_local_asset_authority_core": Path(materializer.rt_asset_authority.__file__).absolute(),
        "subject_m_cost_and_private_snapshot_contract": Path(subject_executor.__file__).absolute(),
        "fixed_gpu_engineering_core": Path(fixed_gpu.__file__).absolute(),
    }
    closure: dict[str, dict[str, Any]] = {}
    for label, path in paths.items():
        raw = source._read_regular_file(path, label=f"RT executor live implementation {label}")
        closure[label] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return closure


def _validate_materializer_plan(plan: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    require(plan.get("schema") == materializer.DEVELOPMENT_TARGET_MATERIALIZER_DRY_PLAN_SCHEMA and
            plan.get("status") ==
            "CANONICAL_DEVELOPMENT_AUTHORITY_AND_RT_ASSET_LEDGER_VERIFIED__NO_TARGET_ARRAYS",
            "RT executor requires the canonical RT development materializer")
    require((plan.get("dataset"), plan.get("view")) == ("rt", None),
            "RT materializer scope drift")
    canonical = plan.get("canonical_development_authority")
    ledger = plan.get("target_asset_ledger_gate")
    require(isinstance(canonical, Mapping) and isinstance(ledger, Mapping),
            "RT materializer canonical authority/asset ledger missing")
    require(ledger.get("schema") == materializer.RT_TARGET_ASSET_LEDGER_SCHEMA and
            ledger.get("status") ==
            "ROOT_PUBLISHED_RT_LOCAL_ASSET_AUTHORITY_VALIDATED__NO_TARGET_OPEN",
            "RT executor requires the root-published local asset authority binding")
    require(ledger.get("caller_supplied_asset_ledger_mapping_permitted") is False and
            ledger.get("required_live_parser_entrypoint") ==
            "track_b_v2_rt_local_asset_authority.open_canonical_verified_asset_after_authority",
            "RT materializer same-FD live-entry contract drift")
    authority = ledger.get("authority")
    require(isinstance(authority, Mapping) and
            authority.get("body_path") == str(materializer.rt_asset_authority.CANONICAL_OUTPUT) and
            authority.get("body_sha256") == LOCAL_ASSET_AUTHORITY_SHA256 and
            authority.get("asset_count") == 15,
            "RT local asset authority path/SHA/count drift")
    asset = ledger.get("target_asset")
    require(isinstance(asset, Mapping) and
            asset.get("session_id") == plan.get("target_session_id") and
            asset.get("held_target_outer_fold_id") == plan.get("outer_fold_id"),
            "RT materializer selected target/fold binding drift")
    for key in (
        "target_data_discovery_permitted", "target_data_opened", "target_query_opened",
        "formal_data_opened", "cebra_imported", "cebra_trained", "gpu_used",
        "score_emitted", "official_execution_receipt_minted",
    ):
        require(plan.get(key) is False, f"RT materializer prohibited flag drift: {key}")
    require(ledger.get("target_data_discovered") is False and ledger.get("target_data_opened") is False,
            "RT materializer local-asset ledger claims target access")
    return canonical, ledger


def _cost_gate_valid(gate: Mapping[str, Any]) -> bool:
    return gate.get("status") == (
        "FIXED_D8IT250_GPU_COST_RECEIPT_VALID__REQUIRED_BUT_NOT_SUFFICIENT_FOR_TARGET_EXECUTION"
    )


def _fixed_scientific_contract() -> dict[str, Any]:
    return {
        "embedding_geometry": {
            "output_dimension": 8,
            "iterations": 10_000,
            "encoder_geometry_key": "d8-it10000",
            "source_or_target_score_selection_performed": False,
        },
        "decoders": {
            "linear_ridge": {
                "normalization": "train_fit_embedding_component_standardization_only",
                "normalized_ridge_lambda": 0.01,
            },
            "knn_cosine_k3": {
                "metric": "cosine",
                "k": 3,
                "normalization": "same_train_fit_embedding_component_standardization_policy",
            },
        },
        "seeds": list(SEEDS),
        "seed_interpretation": "stochastic_sensitivity_not_bitwise_reproducible_stream",
        "geometry_seed_route_or_decoder_winner_selection_permitted": False,
        "fixed_gpu_cost_receipt_is_engineering_cost_only_not_scientific_selection": True,
    }


def _m24_query_endpoint_contract() -> dict[str, Any]:
    return {
        "support": {
            "budget_trials": 24,
            "rewarded_trial_ordinals": list(range(24)),
            "sequence": "one_continuous_chronological_prefix_from_record_start_through_trial_23_stop",
            "all_intervening_raw_rows_included": True,
            "rewarded_segments_concatenated": False,
            "dense_continuous_velocity_float32_required": True,
            "neural_and_dense_velocity_enter_joint_encoder_fit": True,
        },
        "query": {
            "strict_post_M24_only": True,
            "rewarded_trial_ordinals": "ordered_trials_with_ordinal_greater_than_or_equal_to_24",
            "raw_query_start": "support_stop_raw_bin_plus_1",
            "valid_window_start_indices_must_be_strict_query_rows": True,
            "ordered_prediction_target_raw_bin_indices": "valid_window_start_plus_49",
            "prediction_targets_must_be_strict_query_rows": True,
            "query_neural_dense_velocity_or_behavior_enters_encoder_fit": False,
            "query_neural_or_labels_enters_any_readout_fit": False,
            "query_used_only_for_frozen_transform_and_score": True,
        },
        "exact_target_byte_authority": {
            "t4_and_cebra_ordered_float32_q_by_2_target_bytes_must_be_identical": True,
            "t4_reference_query_identity_sha256": "REQUIRED_FROM_SEALED_RT_T4_RUNTIME_LINEAGE",
            "t4_runtime_receipt_sha256": "REQUIRED_FROM_SEALED_RT_T4_RUNTIME_LINEAGE",
            "metric_implementation_authority_sha256": "REQUIRED_FROM_SEALED_RT_T4_RUNTIME_LINEAGE",
            "t4_reference_query_row_count_must_equal_endpoint_count": True,
            "unpaired_language_required_if_byte_or_endpoint_authority_differs": True,
        },
        "offset10_receptive_field": {
            "model_architecture": "offset10-model",
            "half_open_offsets_relative_to_prediction_target": [-5, 5],
            "exact_raw_indices_per_endpoint": "range(endpoint-5, endpoint+5)",
            "width_raw_bins": 10,
            "previous_raw_bins": 5,
            "prediction_target_bin_included": True,
            "strictly_future_raw_bins_after_prediction_target": 4,
            "every_receptive_field_wholly_inside_strict_query": True,
            "support_query_boundary_crossing_permitted": False,
            "causal_temporal_exposure_matched": False,
            "bias_direction": "favors_CEBRA_accuracy",
            "online_or_latency_equivalent_language_permitted": False,
        },
        "required_array_level_validator": {
            "module": str(Path(materializer.__file__).absolute()),
            "symbol": "materialize_synthetic_target_query_proof__future_live_successor_must_preserve_exact_checks",
            "current_symbol_accepts_real_target_arrays": False,
            "future_non_synthetic_bridge_requires_separate_root_review": True,
        },
    }


def _blocked_descriptor_contract(*, ledger: Mapping[str, Any]) -> dict[str, Any]:
    authority = ledger["authority"]
    asset = ledger["target_asset"]
    return {
        "status": "BLOCKED_BEFORE_HELD_PATH_RESOLUTION__FIXED_GPU_COST_RECEIPT_REQUIRED",
        "rt_local_asset_authority_body_sha256": authority["body_sha256"],
        "target_session_id": asset["session_id"],
        "held_target_outer_fold_id": asset["held_target_outer_fold_id"],
        "held_target_outer_fold_index": asset["held_target_outer_fold_index"],
        "canonical_target_path_copied_into_executor_plan": False,
        "target_path_resolved": False,
        "target_path_statted": False,
        "target_fd_opened": False,
        "private_snapshot_created": False,
        "parser_called": False,
    }


def _future_descriptor_contract(*, ledger: Mapping[str, Any]) -> dict[str, Any]:
    authority = ledger["authority"]
    asset = ledger["target_asset"]
    return {
        "status": "COST_GATE_VALID__DESCRIPTOR_CONTRACT_ONLY__LIVE_OPENER_NOT_IMPLEMENTED",
        "authority_must_be_loaded_before_target_path_resolution": True,
        "rt_local_asset_authority_body_path": authority["body_path"],
        "rt_local_asset_authority_body_sha256": authority["body_sha256"],
        "target_session_id": asset["session_id"],
        "held_target_outer_fold_id": asset["held_target_outer_fold_id"],
        "held_target_outer_fold_index": asset["held_target_outer_fold_index"],
        "future_canonical_target_path": asset["canonical_local_nwb_path"],
        "future_expected_byte_count": asset["expected_bytes"],
        "future_expected_sha256": asset["expected_sha256"],
        "held_source_descriptor": {
            "canonical_loader_and_opener": (
                "track_b_v2_rt_local_asset_authority.open_canonical_verified_asset_after_authority"
            ),
            "open_flags": ["O_RDONLY", "O_NOFOLLOW", "O_CLOEXEC"],
            "one_open_only": True,
            "fstat_size_then_stream_sha256_on_same_fd": True,
            "pre_and_post_hash_fd_identity_exact": True,
            "post_hash_path_inode_identity_exact": True,
            "ordinary_pathname_reopen_for_parser_permitted": False,
        },
        "private_snapshot": {
            "created_from_continuously_held_verified_source_fd": True,
            "create_flags": ["O_WRONLY", "O_CREAT", "O_EXCL", "O_NOFOLLOW", "O_CLOEXEC"],
            "initial_mode": "0600",
            "fsync_file_and_directory_then_mode": "0444",
            "snapshot_bytes_and_sha_must_equal_verified_source_fd": True,
            "parser_fd_held_open_and_rewound_to_zero": True,
            "parser_input": "/proc/self/fd/<held_snapshot_fd>_or_equivalent_descriptor_API",
            "parser_pathname_reopen_permitted": False,
            "post_parser_fd_and_path_identity_revalidation_required": True,
        },
        "canonical_rt_same_fd_contract_binding": {
            "module": str(Path(materializer.rt_asset_authority.__file__).absolute()),
            "symbol": "open_canonical_verified_asset_after_authority",
            "authority_body_sha256": LOCAL_ASSET_AUTHORITY_SHA256,
            "caller_supplied_asset_ledger_mapping_permitted": False,
            "direct_call_authorized_by_this_plan": False,
        },
        "target_path_resolved_by_this_plan": False,
        "target_fd_opened_by_this_plan": False,
        "private_snapshot_created_by_this_plan": False,
        "parser_called_by_this_plan": False,
    }


def _readout_and_receipt_topology() -> dict[str, Any]:
    routes = {
        "source_only_consumer_mechanism_alignment": {
            "encoder_fit_scope": "same_joint_14_source_plus_held_M24_support_encoder",
            "readout_fit_scope": "source_fit_only",
            "target_support_dense_labels_in_readout_fit": False,
            "unfitted_unseen_target_transform_permitted": False,
            "scientific_role": "mechanism_alignment_not_accuracy_headline",
        },
        "target_support_only_standard_cebra_accuracy": {
            "encoder_fit_scope": "same_joint_14_source_plus_held_M24_support_encoder",
            "readout_fit_scope": "held_target_M24_support_only",
            "target_support_dense_labels_in_readout_fit": True,
            "scientific_role": "accuracy_table_headline_with_extra_target_supervision_disclosed",
        },
        "source_plus_target_support_hybrid_sensitivity": {
            "encoder_fit_scope": "same_joint_14_source_plus_held_M24_support_encoder",
            "readout_fit_scope": "source_fit_plus_held_target_M24_support",
            "target_support_dense_labels_in_readout_fit": True,
            "scientific_role": "mandatory_sensitivity_not_headline_not_rescue",
        },
    }
    return {
        "outer_fold_count": 15,
        "cebra_seed_count": 3,
        "encoder_receipt_count": 45,
        "score_receipt_count": 15 * 3 * 3 * 2,
        "aggregate_receipt_count": 1,
        "per_outer_fold_and_seed": {
            "encoder_bundle": {
                "model_arm": "cebra_joint_behavior",
                "one_fresh_encoder_fit": True,
                "source_streams": "exact_ordered_14_outer_source_sessions",
                "held_stream_fit_rows": "continuous_chronological_M24_support_only",
                "held_support_neural_enters_encoder_fit": True,
                "held_support_dense_velocity_enters_encoder_fit": True,
                "held_query_neural_or_behavior_enters_fit": False,
                "same_encoder_state_sha256_required_for_all_routes_and_decoders": True,
            },
            "readout_routes": routes,
            "decoders": list(DECODERS),
            "one_score_receipt_per_route_decoder": True,
            "same_ordered_query_endpoint_and_target_byte_authority_for_all_six_scores": True,
            "posthoc_best_route_decoder_or_seed_selection_permitted": False,
        },
        "per_score_receipt_must_bind": [
            "rt_local_asset_authority_body_sha256",
            "canonical_development_materializer_plan_sha256",
            "fixed_d8it250_gpu_cost_receipt_body_sha256",
            "outer_fold_id_and_held_target_session_id",
            "joint_encoder_state_sha256",
            "source_and_held_M24_support_fit_index_authorities",
            "strict_post_M24_query_index_authority",
            "ordered_prediction_endpoint_authority",
            "full_offset10_receptive_field_authority",
            "ordered_T4_equal_CEBRA_float32_target_bytes_sha256",
            "sealed_RT_T4_query_and_metric_lineage",
            "route_decoder_seed_and_fixed_geometry",
            "zero_query_fit_or_backprop_proof",
        ],
        "aggregate": {
            "session_then_seed_aggregation_required": True,
            "all_15_folds_all_3_seeds_all_3_routes_both_decoders_required": True,
            "no_best_seed_route_or_decoder_substitution": True,
            "development_only_not_formal": True,
        },
    }


def build_rt_development_executor_dry_plan(
    *, dataset: str, view: str | None, outer_fold_id: str, target_session_id: object,
    proposed_target_path: object | None = None, proposed_target_discovery: object | None = None,
    execution_requested: bool = False, device: str = "cpu",
) -> dict[str, Any]:
    """Render the future RT execution/scoring boundary without touching target data."""
    dataset, view = base.validate_scope(dataset, view)
    require(dataset == "rt" and view is None, "RT executor accepts only dataset=rt and view=None")
    require(proposed_target_path is None, "RT executor dry plan accepts no target path")
    require(proposed_target_discovery is None, "RT executor dry plan accepts no target discovery callable")
    require(execution_requested is False, "RT executor scaffold has no execution mode")
    require(device == "cpu", "RT executor scaffold dry path is CPU/no-data only")
    try:
        plan = materializer.build_development_target_materializer_dry_plan(
            dataset=dataset, view=view, outer_fold_id=outer_fold_id,
            target_session_id=target_session_id,
        )
    except materializer.TrackBV2DevelopmentTargetMaterializerError as exc:
        raise TrackBV2RTDevelopmentExecutorError(str(exc)) from exc
    canonical, ledger = _validate_materializer_plan(plan)
    cost_gate = subject_executor.inspect_fixed_d8it250_gpu_cost_receipt()
    cost_valid = _cost_gate_valid(cost_gate)
    descriptor = (
        _future_descriptor_contract(ledger=ledger)
        if cost_valid else _blocked_descriptor_contract(ledger=ledger)
    )
    materializer_plan_sha256 = (
        _sha_json(plan) if cost_valid else
        "NOT_RENDERED_BEFORE_FIXED_GPU_COST_GATE__PARENT_AUTHORITIES_BOUND_SEPARATELY"
    )
    closure = _implementation_closure()
    payload = {
        "schema": SCHEMA,
        "status": STATUS_IMPLEMENTATION_BLOCKED if cost_valid else STATUS_COST_BLOCKED,
        "dataset": "rt",
        "view": None,
        "outer_fold_id": plan["outer_fold_id"],
        "target_session_id": plan["target_session_id"],
        "canonical_materializer_binding": {
            "schema": plan["schema"],
            "status": plan["status"],
            "plan_sha256": materializer_plan_sha256,
            "canonical_development_target_authority_sha256": canonical[
                "development_target_query_authority_sha256"],
            "canonical_metric_pointer_body_sha256": canonical["metric_pointer_body_sha256"],
            "fixed_geometry_contract_sha256": canonical["fixed_canonical_geometry_contract_sha256"],
            "rt_local_asset_authority_body_path": ledger["authority"]["body_path"],
            "rt_local_asset_authority_body_sha256": ledger["authority"]["body_sha256"],
            "rt_local_asset_authority_asset_count": ledger["authority"]["asset_count"],
            "caller_supplied_materializer_asset_path_or_sha_permitted": False,
        },
        "fixed_gpu_cost_gate": cost_gate,
        "fixed_gpu_cost_gate_required_before": [
            "held_target_path_resolution", "held_target_stat", "held_target_fd_open",
            "private_snapshot_creation", "NWB_parser_call", "CEBRA_import_or_fit",
            "readout_fit", "score",
        ],
        "scientific_contract": _fixed_scientific_contract(),
        "joint_encoder_contract": {
            "model_arm": "cebra_joint_behavior",
            "source_session_count_per_fold": 14,
            "held_target_support_budget_trials": 24,
            "encoder_fit_scope": "14_full_source_streams_plus_held_continuous_M24_support_stream",
            "held_target_is_not_an_unfitted_unseen_transform": True,
            "target_query_neural_behavior_or_labels_enter_fit": False,
            "all_three_readout_routes_share_exact_encoder_state": True,
        },
        "mandatory_readout_routes": list(READOUT_ROUTES),
        "rt_M24_post_M_query_endpoint_authority": _m24_query_endpoint_contract(),
        "future_held_descriptor_private_snapshot_parser_contract": descriptor,
        "immutable_future_receipt_topology": _readout_and_receipt_topology(),
        "implementation_closure": closure,
        "implementation_closure_sha256": _sha_json(closure),
        "target_execution_permitted": False,
        "target_data_discovery_permitted": False,
        "target_data_opened": False,
        "target_query_opened": False,
        "formal_data_opened": False,
        "cebra_imported": False,
        "cebra_fit_called": False,
        "gpu_used": False,
        "readout_fit_called": False,
        "score_emitted": False,
        "receipt_minted": False,
        "official_execution_receipt_minted": False,
    }
    return payload | {"dry_plan_sha256": _sha_json(payload)}


def refuse_rt_target_execution(**kwargs: Any) -> None:
    """Tripwire: this additive scaffold never authorizes a target operation."""
    build_rt_development_executor_dry_plan(**kwargs)
    raise TrackBV2RTDevelopmentExecutorError(
        "RT target executor/scorer is a no-target scaffold; separate root-reviewed live code is required"
    )
