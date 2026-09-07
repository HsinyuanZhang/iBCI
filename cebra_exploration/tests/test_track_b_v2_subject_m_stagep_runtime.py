"""No-target synthetic/adversarial tests for the Subject-M Stage-P runtime."""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "cebra_exploration" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_contract as base  # noqa: E402
import track_b_v2_subject_m_development_executor as development_executor  # noqa: E402
import track_b_v2_subject_m_stagep_runtime as runtime  # noqa: E402
import track_b_v2_post_synthetic_runtime_control_authority as control_authority  # noqa: E402


CLI = REPO_ROOT / "cebra_exploration/scripts/run_track_b_v2_subject_m_stagep_runtime.py"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sha_json(value: object) -> str:
    return hashlib.sha256(base.canonical_json_bytes(value)).hexdigest()


def _signed(payload: dict[str, object], key: str) -> dict[str, object]:
    payload[key] = hashlib.sha256(base.canonical_json_bytes(payload)).hexdigest()
    return payload


def _valid_cost_gate() -> dict[str, object]:
    evidence = {
        "schema": development_executor.FIXED_GPU_COST_GATE_SCHEMA,
        "status": "FIXED_D8IT250_GPU_COST_RECEIPT_VALID__REQUIRED_BUT_NOT_SUFFICIENT_FOR_TARGET_EXECUTION",
        "canonical_body_path": "/synthetic/cost/receipt.json",
        "canonical_body_sha256": _sha("cost-body"),
        "canonical_sidecar_path": "/synthetic/cost/receipt.json.sha256",
        "canonical_sidecar_sha256": _sha("cost-sidecar"),
        "fixed_final_geometry": {"output_dimension": 8, "source_iterations": 10_000,
                                 "encoder_geometry_key": "d8-it10000"},
        "cost_smoke_geometry": {"output_dimension": 8, "source_iterations": 250,
                                "encoder_geometry_key": "d8-it250"},
        "cost_smoke_seed": 42,
        "live_implementation_closure_sha256": _sha("closure"),
        "source_authority_binding_sha256": _sha("source-bindings"),
        "device_uuid": "GPU-synthetic",
        "measured_total_wall_clock_s": 1.0,
        "measured_gpu_fit_plus_transform_wall_clock_s": 0.5,
        "fresh_body_and_sidecar": True,
        "target_execution_permitted": False,
        "target_data_opened": False,
        "gpu_used_by_this_gate": False,
    }
    return evidence


def _control_evidence() -> dict[str, object]:
    return control_authority.load_raw_bound_evidence()[0]


def _synthetic_live_preflight(plan: dict[str, object]) -> dict[str, object]:
    copied = dict(plan)
    copied["official_stagep_preflight_body_sha256"] = _sha("synthetic immutable official preflight")
    return copied


def _target(preflight: dict[str, object], *, view: str = "sua") -> dict[str, object]:
    query_sha = _sha("ordered target float32")
    endpoint_sha = _sha("ordered endpoints")
    query: dict[str, object] = {
        "strictly_after_rewarded_trial": 50,
        "prediction_endpoint": "valid_window_start_plus_49",
        "offset10_receptive_field": "range(endpoint-5, endpoint+5)",
        "offset10_width_raw_bins": 10,
        "future_raw_bins_after_endpoint": 4,
        "receptive_field_violations": 0,
        "query_row_count": 10,
        "ordered_prediction_endpoint_int64_sha256": endpoint_sha,
        "ordered_offset10_RF_int64_sha256": _sha("ordered query RF"),
        "ordered_query_neural_float32_bytes_sha256": _sha("ordered query neural"),
        "ordered_target_behavior_float32_bytes_sha256": query_sha,
        "ordered_t4_target_float32_bytes_sha256": query_sha,
        "ordered_cebra_target_float32_bytes_sha256": query_sha,
        "sealed_A2_T4_target_query_receipt": {
            "receipt_body_path": "/synthetic/sealed_A2_T4_target_query.json",
            "receipt_body_sha256": _sha("sealed A2 receipt"),
            "ordered_target_float32_bytes_sha256": query_sha,
            "ordered_prediction_endpoint_int64_sha256": endpoint_sha,
            "query_window_count": 10,
        },
    }
    query["query_authority_sha256"] = _sha_json(query)
    payload: dict[str, object] = {
        "schema": runtime.STAGEP_TARGET_RECEIPT_SCHEMA,
        "cell": preflight["cell"], "primary_model_arm": runtime.PRIMARY_ARM,
        "official_stagep_preflight_body_sha256": preflight["official_stagep_preflight_body_sha256"],
        "canonical_development_target_authority_sha256": preflight["canonical_materializer_binding"]["canonical_development_target_authority_sha256"],  # type: ignore[index]
        "canonical_metric_pointer_body_sha256": preflight["canonical_materializer_binding"]["canonical_metric_pointer_body_sha256"],  # type: ignore[index]
        "fixed_geometry_contract_sha256": preflight["canonical_materializer_binding"]["fixed_geometry_contract_sha256"],  # type: ignore[index]
        "verified_target_asset": {
            "canonical_ledger_asset_path": "/synthetic/sub-M_ses-CO-20140307_behavior+ecephys.nwb",
            "expected_nwb_sha256": _sha("target nwb"), "expected_nwb_byte_count": 12345,
            "source_inode_identity_sha256": _sha("source inode"),
            "private_snapshot_sha256": _sha("target nwb"), "private_snapshot_byte_count": 12345,
            "private_snapshot_inode_identity_sha256": _sha("snapshot inode"),
        },
        "support": {"continuous_prefix_through_rewarded_trial": 50,
                    "rewarded_trial_segments_concatenated": False,
                    "all_intervening_chronological_rows_retained": True},
        "query": query,
        "private_snapshot_parser": {
            "source_sha_size_verified_same_fd": True,
            "snapshot_O_EXCL_fsync_0444": True,
            "parser_consumed_held_fd_not_path_reopen": True,
        },
    }
    if view == "pseudo_mua":
        payload["pmua_replay"] = {
            "same_target_record_as_sua": True,
            "behavior_endpoint_target_bytes_equal_to_sua": True,
            "replay_exact_equal": True,
            "input_sua_feature_sha256": _sha("sua feature"),
            "ordered_unit_ids_raw_bytes_sha256": _sha("unit ids"),
            "ordered_unit_electrode_ids_raw_bytes_sha256": _sha("electrodes"),
            "output_pmua_feature_sha256": _sha("pmua feature"),
            "input_sua_feature_shape": [100, 12], "input_sua_feature_shape_expected": [100, 12],
            "output_pmua_feature_shape": [100, 4], "output_pmua_feature_shape_expected": [100, 4],
            "sorted_unit_count": 12, "unique_electrode_channel_count": 4,
        }
    return _signed(payload, "target_receipt_sha256")


def _target_provenance_fields(target: dict[str, object]) -> dict[str, object]:
    """The downstream receipt surface is deliberately direct, not SHA-only."""
    query = target["query"]  # type: ignore[index]
    payload: dict[str, object] = {
        "target_query_authority_sha256": query["query_authority_sha256"],
        "query_row_count": query["query_row_count"],
        "ordered_prediction_endpoint_int64_sha256": query["ordered_prediction_endpoint_int64_sha256"],
        "ordered_offset10_RF_int64_sha256": query["ordered_offset10_RF_int64_sha256"],
        "ordered_query_neural_float32_bytes_sha256": query["ordered_query_neural_float32_bytes_sha256"],
        "ordered_target_behavior_float32_bytes_sha256": query["ordered_target_behavior_float32_bytes_sha256"],
        "ordered_t4_target_float32_bytes_sha256": query["ordered_t4_target_float32_bytes_sha256"],
        "ordered_cebra_target_float32_bytes_sha256": query["ordered_cebra_target_float32_bytes_sha256"],
        "sealed_A2_T4_target_query_receipt": copy.deepcopy(query["sealed_A2_T4_target_query_receipt"]),
        "verified_target_asset": copy.deepcopy(target["verified_target_asset"]),
        "verified_target_asset_sha256": target["verified_target_asset"]["expected_nwb_sha256"],  # type: ignore[index]
    }
    if target["cell"]["view"] == "pseudo_mua":  # type: ignore[index]
        payload["pmua_replay"] = copy.deepcopy(target["pmua_replay"])
    return payload


def _encoder_artifact_persistence(view: str) -> dict[str, object]:
    topology = runtime.stagep_output_topology(runtime.StagePCell.from_view(view))
    return {
        "encoder_checkpoint_persistence": {
            "canonical_artifact_path": topology["joint_encoder_checkpoint"],
            "artifact_sha256": _sha("encoder checkpoint artifact"),
            "sidecar_sha256": _sha("encoder checkpoint sidecar"),
            "format": "torch_state_dict_sorted_name_dtype_shape_bytes",
            "publication": "O_EXCL_0444_body_and_sha256_sidecar",
            "reader": "same_fd_ONOFOLLOW_sidecar_verified_before_torch_deserialize",
            "same_checkpoint_services_all_six_scores": True,
        },
        "embedding_persistence": {
            "canonical_artifact_path": topology["joint_embedding_bundle"],
            "artifact_sha256": _sha("embedding bundle artifact"),
            "sidecar_sha256": _sha("embedding bundle sidecar"),
            "format": "per_session_float32_contiguous_embeddings_role_dtype_shape_bytes",
            "publication": "O_EXCL_0444_body_and_sha256_sidecar",
            "reader": "same_fd_ONOFOLLOW_sidecar_verified_before_embedding_load",
            "same_bundle_services_all_six_scores": True,
        },
    }


def _encoder(preflight: dict[str, object], target: dict[str, object]) -> dict[str, object]:
    roster = runtime._strict27_source_roster_authority()
    return _signed({
        "schema": runtime.STAGEP_ENCODER_RECEIPT_SCHEMA,
        "cell": preflight["cell"], "primary_model_arm": runtime.PRIMARY_ARM,
        "official_stagep_preflight_body_sha256": preflight["official_stagep_preflight_body_sha256"],
        "canonical_development_target_authority_sha256": preflight["canonical_materializer_binding"]["canonical_development_target_authority_sha256"],  # type: ignore[index]
        "canonical_metric_pointer_body_sha256": preflight["canonical_materializer_binding"]["canonical_metric_pointer_body_sha256"],  # type: ignore[index]
        "fixed_geometry_contract_sha256": preflight["canonical_materializer_binding"]["fixed_geometry_contract_sha256"],  # type: ignore[index]
        "target_receipt_sha256": target["target_receipt_sha256"],
        **_target_provenance_fields(target),
        "geometry": {"output_dimension": 8, "iterations": 10_000},
        "source_session_count": 27, "held_support_trial_count": 50,
        "fit_stream_count": 28, "fit_count": 1,
        "target_support_dense_auxiliary_enters_encoder_fit": True,
        "target_query_enters_encoder_fit": False,
        "ordered_source_session_ids": roster["ordered_source_session_ids"],
        "source_roster_authority_sha256": roster["source_roster_authority_sha256"],
        "encoder_state_digest_algorithm": "sorted_session_index_parameter_name_dtype_shape_then_raw_bytes_sha256",
        **_encoder_artifact_persistence(preflight["cell"]["view"]),  # type: ignore[index]
        "encoder_state_sha256": _sha("one joint encoder state"),
        "embedding_bundle_sha256": _sha("one joint embedding bundle"),
    }, "encoder_receipt_sha256")


def _block(role: str, index: int) -> dict[str, object]:
    return {
        "block_role": role,
        "session_id": (runtime._canonical_strict27_source_ids()[index] if role == "source_session"
                       else "sub-M_ses-CO-20140307"),
        "fitted_offset": [5, 5],
        "trim_policy": "interior_rows_5_to_minus5_before_cross_session_combination",
        "padded_embedding_row_count": 20,
        "valid_embedding_row_count": 10,
        "valid_auxiliary_label_row_count": 10,
        "raw_index_authority_row_count": 20,
        "raw_index_authority_first": 0,
        "raw_index_authority_last": 19,
        "ordered_raw_index_authority_int64_sha256": _sha(f"raw-index-{role}-{index}"),
        "ordered_fit_endpoint_int64_sha256": _sha(f"endpoint-{role}-{index}"),
        "ordered_fit_RF_int64_sha256": _sha(f"rf-{role}-{index}"),
        "ordered_fit_label_float32_bytes_sha256": _sha(f"label-{role}-{index}"),
        "label_rows_equal_embedding_rows": True,
        "every_RF_wholly_inside_its_fit_block": True,
        "enters_any_fit": True,
    }


def _score(preflight: dict[str, object], target: dict[str, object], encoder: dict[str, object],
           route: str, decoder: str) -> dict[str, object]:
    query = target["query"]  # type: ignore[index]
    target_sha = query["ordered_cebra_target_float32_bytes_sha256"]
    blocks = ([_block("source_session", i) for i in range(27)]
              if route == runtime.ROUTES[0] else
              [_block("held_target_support", 0)] if route == runtime.ROUTES[1] else
              [_block("source_session", i) for i in range(27)] + [_block("held_target_support", 0)])
    prediction_sha = _sha(f"prediction-{route}-{decoder}")
    return _signed({
        "schema": runtime.STAGEP_SCORE_RECEIPT_SCHEMA,
        "cell": preflight["cell"], "primary_model_arm": runtime.PRIMARY_ARM,
        "official_stagep_preflight_body_sha256": preflight["official_stagep_preflight_body_sha256"],
        "canonical_development_target_authority_sha256": preflight["canonical_materializer_binding"]["canonical_development_target_authority_sha256"],  # type: ignore[index]
        "canonical_metric_pointer_body_sha256": preflight["canonical_materializer_binding"]["canonical_metric_pointer_body_sha256"],  # type: ignore[index]
        "fixed_geometry_contract_sha256": preflight["canonical_materializer_binding"]["fixed_geometry_contract_sha256"],  # type: ignore[index]
        "readout_route": route, "decoder": decoder,
        "target_receipt_sha256": target["target_receipt_sha256"],
        "encoder_receipt_sha256": encoder["encoder_receipt_sha256"],
        "encoder_state_sha256": encoder["encoder_state_sha256"],
        "embedding_bundle_sha256": encoder["embedding_bundle_sha256"],
        "readout_fit_scope": runtime._route_scope(route),
        "query_enters_readout_fit": False,
        "target_query_enters_encoder_fit": False,
        "target_backprop_or_update_after_encoder_fit": False,
        **_target_provenance_fields(target),
        "query_row_count": query["query_row_count"],
        "ordered_prediction_endpoint_int64_sha256": query["ordered_prediction_endpoint_int64_sha256"],
        "ordered_offset10_RF_int64_sha256": query["ordered_offset10_RF_int64_sha256"],
        "ordered_query_neural_float32_bytes_sha256": query["ordered_query_neural_float32_bytes_sha256"],
        "ordered_t4_target_float32_bytes_sha256": target_sha,
        "ordered_cebra_target_float32_bytes_sha256": target_sha,
        "ordered_cebra_prediction_float32_bytes_sha256": prediction_sha,
        "metric": {
            "implementation": "torchmetrics.regression.R2Score", "torchmetrics_version": "1.5.1",
            "multioutput": "variance_weighted", "device": "cpu", "dtype": "float32",
            "update_scope": "one_complete_ordered_external_target_session_query_then_compute_once",
            "update_call_count": 1, "compute_call_count": 1,
            "pooled_query_rows_across_sessions": False,
            "prediction_shape": [query["query_row_count"], 2], "target_shape": [query["query_row_count"], 2],
            "prediction_float32_bytes_sha256": prediction_sha,
            "target_float32_bytes_sha256": target_sha,
            "custom_numpy_float64_pooled_r2_used": False,
        },
        "r2_variance_weighted": 0.125,
        "readout_fit_valid_blocks": blocks,
        "readout_state": {
            "readout_state_sha256": _sha(f"readout-state-{route}-{decoder}"),
            "training_row_count": sum(int(block["valid_embedding_row_count"]) for block in blocks),
            "training_embedding_float32_sha256": _sha(f"training-x-{route}-{decoder}"),
            "training_label_float32_sha256": _sha(f"training-y-{route}-{decoder}"),
            "training_block_receipt_sha256": _sha(f"training-blocks-{route}-{decoder}"),
            "query_block_receipt_sha256": _sha("query block"),
            "query_valid_row_count": query["query_row_count"],
            "query_enters_fit": False,
            "readout_state_digest_algorithm": "canonical_route_decoder_training_arrays_and_fitted_readout_state_sha256",
            "readout_hyperparameters": (
                {"normalized_lambda": 0.01} if decoder == "linear_ridge" else {
                    "k": runtime._COSINE_KNN_K, "metric": "cosine", "algorithm": "exact_chunked_cosine_topk_v1",
                    "query_chunk_rows": runtime._COSINE_KNN_QUERY_CHUNK_ROWS,
                    "training_chunk_rows": runtime._COSINE_KNN_TRAIN_CHUNK_ROWS,
                    "tie_break": "cosine_similarity_descending_then_global_training_index_ascending",
                }
            ),
            **({
                "knn_execution": {
                    "algorithm": "exact_chunked_cosine_topk_v1", "k": runtime._COSINE_KNN_K,
                    "tie_break": "cosine_similarity_descending_then_global_training_index_ascending",
                    "query_chunk_rows": runtime._COSINE_KNN_QUERY_CHUNK_ROWS,
                    "training_chunk_rows": runtime._COSINE_KNN_TRAIN_CHUNK_ROWS,
                    "query_row_count": query["query_row_count"],
                    "training_row_count": sum(int(block["valid_embedding_row_count"]) for block in blocks),
                    "training_normalization_pass_count": 1,
                    "normalized_training_buffer_byte_count": (
                        sum(int(block["valid_embedding_row_count"]) for block in blocks) * 8 * 4
                    ),
                    "total_exhaustive_similarity_element_count": (
                        query["query_row_count"] * sum(int(block["valid_embedding_row_count"]) for block in blocks)
                    ),
                    "maximum_similarity_tile_query_rows": 10,
                    "maximum_similarity_tile_training_rows": 10,
                    "maximum_similarity_tile_elements": 100,
                    "full_query_by_training_similarity_matrix_materialized": False,
                    "neighbor_global_index_int64_sha256": _sha(f"knn-index-{route}"),
                    "neighbor_cosine_similarity_float32_sha256": _sha(f"knn-similarity-{route}"),
                }
            } if decoder == "knn_cosine_k3" else {}),
        },
    }, "score_receipt_sha256")


def _terminal(preflight: dict[str, object], target: dict[str, object], encoder: dict[str, object],
              score_receipt_sha256_by_role: dict[str, object]) -> dict[str, object]:
    return _signed({
        "schema": runtime.STAGEP_TERMINAL_RECEIPT_SCHEMA,
        "status": "ONE_STAGEP_PRIMARY_CELL_TERMINAL__SIX_MANDATORY_SCORES_COMPLETE",
        "cell": preflight["cell"], "primary_model_arm": runtime.PRIMARY_ARM,
        "official_stagep_preflight_body_sha256": preflight["official_stagep_preflight_body_sha256"],
        "canonical_development_target_authority_sha256": preflight["canonical_materializer_binding"]["canonical_development_target_authority_sha256"],  # type: ignore[index]
        "canonical_metric_pointer_body_sha256": preflight["canonical_materializer_binding"]["canonical_metric_pointer_body_sha256"],  # type: ignore[index]
        "fixed_geometry_contract_sha256": preflight["canonical_materializer_binding"]["fixed_geometry_contract_sha256"],  # type: ignore[index]
        "target_receipt_sha256": target["target_receipt_sha256"],
        **_target_provenance_fields(target),
        "encoder_receipt_sha256": encoder["encoder_receipt_sha256"],
        "encoder_state_sha256": encoder["encoder_state_sha256"],
        "score_receipt_sha256_by_role": score_receipt_sha256_by_role,
        "metric_aggregation_scope": "one_R2_per_external_target_session_seed_route_decoder__aggregate_session_then_seed",
        "pooled_query_rows_across_sessions": False,
        "development_pilot_only_not_population_inference": True,
    }, "terminal_receipt_sha256")


def test_real_preflight_is_cost_no_go_and_public_stagep_roster_has_no_date_or_seed_choice() -> None:
    plan = _synthetic_live_preflight(runtime.build_stagep_primary_preflight(view="sua"))
    assert plan["status"] in {
        "NO_GO__CANONICAL_D8IT250_COST_PAIR_REQUIRED__BEFORE_STAGEP_TARGET_RESOLUTION",
        "NO_GO__POST_COST_FIXED_RUNTIME_CONTROL_HARD_NULL_PAIR_REQUIRED__NO_TARGET",
    }
    assert plan["cell"] == {
        "dataset": "subject_m", "view": "sua", "outer_fold_id": "subject_m_sua_external_target_20140307",
        "target_session_id": "sub-M_ses-CO-20140307", "cebra_seed": 42,
        "canonical_stagep_cell_id": "stagep__sua__20140307__seed42",
    }
    assert plan["stagep_roster"]["selection_from_target_score_permitted"] is False
    assert plan["stagep_roster"]["session_or_seed_population_inference_permitted"] is False
    assert plan["required_runtime_control_body_path"] == str(runtime.RUNTIME_CONTROL_PATH)
    for flag in ("target_path_resolution_permitted", "target_data_opened", "formal_data_opened",
                 "cebra_imported", "cebra_fit_called", "readout_fit_called", "score_emitted", "gpu_used"):
        assert plan[flag] is False
    params = set(inspect.signature(runtime.build_stagep_primary_preflight).parameters)
    assert params == {"view"}


def test_arm_audit_proves_three_distinct_lifecycles_and_unseen_serviceability_boundary() -> None:
    audit = runtime.audit_vendored_cebra061_arm_serviceability()
    assert audit["vendored_cebra"]["version"] == "0.6.1"
    assert audit["arms"][runtime.PRIMARY_ARM]["target_serviceable"] is True
    assert audit["arms"]["cebra_frozen_source_adapt"]["may_reuse_primary_joint_encoder"] is False
    assert audit["arms"]["cebra_adapt_unaligned"]["target_serviceable_as_joint_multisession_encoder"] is False
    frozen = runtime.build_control_arm_deferred_contract(arm="cebra_frozen_source_adapt")
    assert frozen["status"].startswith("NO_GO")
    assert frozen["may_fill_primary_six_score_slots"] is False


def test_valid_cost_reaches_runtime_control_then_paired_official_root_authority_before_no_target_tripwire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for valid cost status not using the old wrong prefix check."""
    events: list[str] = []
    monkeypatch.setattr(development_executor, "inspect_fixed_d8it250_gpu_cost_receipt", _valid_cost_gate)

    def controls(*, preflight: dict[str, object]) -> dict[str, str]:
        events.append("controls")
        assert preflight["fixed_d8it250_gpu_cost_gate"]["canonical_body_sha256"] == _sha("cost-body")
        return {"body_sha256": _sha("control"), "sidecar_sha256": _sha("control-sidecar"),
                "body_path": "/synthetic/control", "sidecar_path": "/synthetic/control.sha256"}

    def official(*, view: str) -> dict[str, object]:
        events.append(f"official:{view}")
        return {"body_sha256": _sha(f"official:{view}"), "sidecar_sha256": _sha(f"sidecar:{view}"), "payload": {}}

    def root(*, preflight: dict[str, object], official_preflights: dict[str, object]) -> dict[str, str]:
        events.append("root")
        assert set(official_preflights) == {"sua", "pseudo_mua"}
        raise runtime.TrackBV2SubjectMStagePRuntimeError("reached deliberate no-target launch tripwire")

    monkeypatch.setattr(runtime, "_validate_runtime_control_pair", controls)
    monkeypatch.setattr(runtime, "_load_stagep_official_preflight", official)
    monkeypatch.setattr(runtime, "_validate_root_authorization", root)
    with pytest.raises(runtime.TrackBV2SubjectMStagePRuntimeError, match="reached deliberate"):
        runtime.refuse_stagep_primary_execution(view="sua")
    assert events == ["controls", "official:sua", "official:pseudo_mua", "root"]


def test_synthetic_valid_cost_controls_paired_official_preflights_and_one_root_pair_admit_then_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise full paired authority ordering with temp files only, never target data."""
    evidence = _control_evidence()
    monkeypatch.setattr(development_executor, "inspect_fixed_d8it250_gpu_cost_receipt", _valid_cost_gate)
    monkeypatch.setattr(runtime, "RESULT_ROOT", tmp_path / "stagep-results")
    monkeypatch.setattr(runtime, "RUNTIME_CONTROL_PATH", tmp_path / "controls" / "runtime_control.json")
    monkeypatch.setattr(runtime, "ROOT_AUTHORIZATION_PATH", tmp_path / "root" / "root_authorization.json")
    runtime.RUNTIME_CONTROL_PATH.parent.mkdir(parents=True)
    runtime.ROOT_AUTHORIZATION_PATH.parent.mkdir(parents=True)

    plan = runtime.build_stagep_primary_preflight(view="sua")
    contract = dict(plan["required_runtime_control_contract"])
    contract["control_execution_scale"] = {
        "frozen_by_root_after_cost_review": True, "root_pending": False, "actual_runtime_cell_count": 32,
    }
    hard_null = dict(contract["deranged_support_hard_null"])
    hard_null["permutation_authority_sha256"] = runtime._SYNTHETIC_V2_PERMUTATION_SHA256
    hard_null["threshold"] = 0.60
    contract["deranged_support_hard_null"] = hard_null
    contract["raw_bound_control_evidence"] = evidence
    closure = control_authority.implementation_closure()
    contract["publisher_implementation_closure_at_launch"] = closure
    contract["publisher_implementation_closure_at_final"] = closure
    contract["publisher_launch_final_live_closure_equal"] = True
    runtime._write_immutable_pair_once(runtime.RUNTIME_CONTROL_PATH, contract)

    for view in ("sua", "pseudo_mua"):
        path = Path(runtime.stagep_output_topology(runtime.StagePCell.from_view(view))["official_preflight"])
        path.parent.mkdir(parents=True, exist_ok=False)
        runtime.publish_stagep_official_preflight_pair(view=view)
    official = {view: runtime._load_stagep_official_preflight(view=view) for view in ("sua", "pseudo_mua")}
    descriptor = plan["required_root_authorization_descriptor"]
    root_payload = {
        "schema": descriptor["schema"], "status": descriptor["status"],
        "stagep_roster_sha256": descriptor["stagep_roster_sha256"],
        "authorized_cells": descriptor["authorized_cells_must_equal"],
        "official_preflight_body_sha256_by_view": {view: official[view]["body_sha256"] for view in ("sua", "pseudo_mua")},
        "fixed_d8it250_cost_body_sha256": _sha("cost-body"),
        "fixed_runtime_control_body_sha256": _sha_bytes(runtime.RUNTIME_CONTROL_PATH.read_bytes()),
        "implementation_closure_sha256": plan["implementation_closure_sha256"],
        "authorized_model_arm": descriptor["authorizes_only_model_arm"],
        "authorized_readout_routes": descriptor["authorizes_exact_readout_routes"],
        "authorized_decoders": descriptor["authorizes_exact_decoders"],
        "target_path_or_seed_or_output_override_permitted": False,
        "development_pilot_only_not_population_inference": True,
    }
    runtime._write_immutable_pair_once(runtime.ROOT_AUTHORIZATION_PATH, root_payload)
    admitted = runtime.build_stagep_live_admission(view="sua")
    assert admitted["status"].startswith("STAGEP_LIVE_ADMISSION_VALID")
    assert admitted["official_stagep_preflight_body_sha256"] == official["sua"]["body_sha256"]
    with pytest.raises(runtime.TrackBV2SubjectMStagePRuntimeError, match="remains disabled"):
        runtime.refuse_stagep_primary_execution(view="sua")


def test_successor_chain_binds_protocol_m50_endpoint_offset_metric_and_per_block_crop() -> None:
    plan = runtime.build_stagep_primary_preflight(view="pseudo_mua")
    chain = plan["primary_execution_chain"]
    assert plan["root_frozen_runtime_protocol"]["sha256"] == _sha(
        Path(plan["root_frozen_runtime_protocol"]["path"]).read_text())
    assert chain["target_support_query"]["ordered_prediction_target_raw_bin_indices"] == "valid_window_starts_plus_49"
    assert chain["target_support_query"]["offset10_receptive_field_per_endpoint"] == "range(endpoint-5, endpoint+5)"
    crop = chain["encoder_and_score"]["decoder_fit_valid_row_policy"]
    assert crop["per_sequence_trim"] == "interior_rows_5_to_minus5_before_any_cross_session_combination"
    assert crop["padded_edge_embeddings_permitted_in_any_readout_fit"] is False
    metric = chain["encoder_and_score"]["scoring_metric"]
    assert metric["implementation"] == "torchmetrics.regression.R2Score"
    assert metric["torchmetrics_version"] == "1.5.1" and metric["dtype"] == "float32"
    assert metric["custom_numpy_float64_pooled_r2_is_exact_parity"] is False
    closure = plan["implementation_closure"]
    assert {"development_target_materializer", "development_target_authority", "source_adapter",
            "canonical_subject_m_evaluator", "canonical_multisession_datamodule"} <= set(closure)
    recursive = closure["vendored_cebra_recursive_python_runtime"]["files"]
    assert {"models/model.py", "data/multi_session.py", "solver/multi_session.py",
            "integrations/sklearn/cebra.py"} <= set(recursive)


def test_synthetic_target_encoder_six_scores_require_one_joint_encoder_torchmetrics_and_per_block_crop() -> None:
    plan = _synthetic_live_preflight(runtime.build_stagep_primary_preflight(view="sua"))
    target = _target(plan)
    encoder = _encoder(plan, target)
    assert runtime.validate_primary_target_receipt(payload=target, preflight=plan, synthetic=True)["status"].endswith("VALID")
    assert runtime.validate_primary_encoder_receipt(
        payload=encoder, preflight=plan, target_receipt=target, synthetic=True,
    )["status"].endswith("VALID")
    rows = [_score(plan, target, encoder, route, decoder)
            for route in runtime.ROUTES for decoder in runtime.DECODERS]
    checked = runtime.validate_primary_score_receipts(
        payloads=rows, preflight=plan, target_receipt=target, encoder_receipt=encoder, synthetic=True,
    )
    assert len(checked["score_receipt_sha256_by_role"]) == 6
    terminal = _terminal(plan, target, encoder, checked["score_receipt_sha256_by_role"])
    assert runtime.validate_primary_terminal_receipt(
        payload=terminal, preflight=plan, target_receipt=target, encoder_receipt=encoder,
        score_receipts=rows, synthetic=True,
    )["status"].endswith("VALID")
    broken = copy.deepcopy(rows)
    broken[0]["metric"]["torchmetrics_version"] = "1.4.0"  # type: ignore[index]
    with pytest.raises(runtime.TrackBV2SubjectMStagePRuntimeError, match="torchmetrics"):
        runtime.validate_primary_score_receipts(
            payloads=broken, preflight=plan, target_receipt=target, encoder_receipt=encoder, synthetic=True,
        )
    broken = copy.deepcopy(rows)
    broken[0]["official_stagep_preflight_body_sha256"] = _sha("wrong official preflight")
    with pytest.raises(runtime.TrackBV2SubjectMStagePRuntimeError, match="official Stage-P preflight"):
        runtime.validate_primary_score_receipts(
            payloads=broken, preflight=plan, target_receipt=target, encoder_receipt=encoder, synthetic=True,
        )
    broken = copy.deepcopy(rows)
    broken[0]["readout_fit_valid_blocks"][0]["trim_policy"] = "crop_after_concatenation"  # type: ignore[index]
    with pytest.raises(runtime.TrackBV2SubjectMStagePRuntimeError, match="padded-edge"):
        runtime.validate_primary_score_receipts(
            payloads=broken, preflight=plan, target_receipt=target, encoder_receipt=encoder, synthetic=True,
        )
    broken = copy.deepcopy(rows)
    broken[-1]["encoder_state_sha256"] = _sha("illegal route refit")
    with pytest.raises(runtime.TrackBV2SubjectMStagePRuntimeError, match="one-encoder"):
        runtime.validate_primary_score_receipts(
            payloads=broken, preflight=plan, target_receipt=target, encoder_receipt=encoder, synthetic=True,
        )
    broken = copy.deepcopy(rows)
    broken[0]["query_row_count"] = 9
    with pytest.raises(runtime.TrackBV2SubjectMStagePRuntimeError, match="target asset/snapshot/A2/query-byte"):
        runtime.validate_primary_score_receipts(
            payloads=broken, preflight=plan, target_receipt=target, encoder_receipt=encoder, synthetic=True,
        )


def test_target_receipt_requires_exact_materializer_asset_snapshot_and_query_authorities() -> None:
    plan = _synthetic_live_preflight(runtime.build_stagep_primary_preflight(view="sua"))
    target = _target(plan)
    runtime.validate_primary_target_receipt(payload=target, preflight=plan, synthetic=True)
    broken = copy.deepcopy(target)
    broken["verified_target_asset"]["private_snapshot_sha256"] = _sha("replacement asset")  # type: ignore[index]
    broken = _signed({key: value for key, value in broken.items() if key != "target_receipt_sha256"},
                     "target_receipt_sha256")
    with pytest.raises(runtime.TrackBV2SubjectMStagePRuntimeError, match="verified target asset"):
        runtime.validate_primary_target_receipt(payload=broken, preflight=plan, synthetic=True)
    broken = copy.deepcopy(target)
    broken["query"]["query_row_count"] = 9  # type: ignore[index]
    query = broken["query"]  # type: ignore[index]
    query["query_authority_sha256"] = _sha_json({key: value for key, value in query.items()
                                                   if key != "query_authority_sha256"})
    broken = _signed({key: value for key, value in broken.items() if key != "target_receipt_sha256"},
                     "target_receipt_sha256")
    with pytest.raises(runtime.TrackBV2SubjectMStagePRuntimeError, match="query count/endpoint/RF"):
        runtime.validate_primary_target_receipt(payload=broken, preflight=plan, synthetic=True)


def test_pmua_target_receipt_requires_same_record_replay_provenance() -> None:
    plan = _synthetic_live_preflight(runtime.build_stagep_primary_preflight(view="pseudo_mua"))
    target = _target(plan, view="pseudo_mua")
    runtime.validate_primary_target_receipt(payload=target, preflight=plan, synthetic=True)
    broken = copy.deepcopy(target)
    broken["pmua_replay"]["replay_exact_equal"] = False  # type: ignore[index]
    broken = _signed({key: value for key, value in broken.items() if key != "target_receipt_sha256"},
                     "target_receipt_sha256")
    with pytest.raises(runtime.TrackBV2SubjectMStagePRuntimeError, match="pMUA"):
        runtime.validate_primary_target_receipt(payload=broken, preflight=plan, synthetic=True)


def test_downstream_receipts_copy_exact_target_asset_a2_query_and_pmua_authorities() -> None:
    plan = _synthetic_live_preflight(runtime.build_stagep_primary_preflight(view="pseudo_mua"))
    target = _target(plan, view="pseudo_mua")
    encoder = _encoder(plan, target)
    rows = [_score(plan, target, encoder, route, decoder)
            for route in runtime.ROUTES for decoder in runtime.DECODERS]
    checked = runtime.validate_primary_score_receipts(
        payloads=rows, preflight=plan, target_receipt=target, encoder_receipt=encoder, synthetic=True,
    )
    terminal = _terminal(plan, target, encoder, checked["score_receipt_sha256_by_role"])
    runtime.validate_primary_terminal_receipt(
        payload=terminal, preflight=plan, target_receipt=target, encoder_receipt=encoder,
        score_receipts=rows, synthetic=True,
    )

    broken_rows = copy.deepcopy(rows)
    del broken_rows[0]["pmua_replay"]
    with pytest.raises(runtime.TrackBV2SubjectMStagePRuntimeError, match="pMUA receipt lost"):
        runtime.validate_primary_score_receipts(
            payloads=broken_rows, preflight=plan, target_receipt=target, encoder_receipt=encoder, synthetic=True,
        )
    broken_terminal = copy.deepcopy(terminal)
    broken_terminal["sealed_A2_T4_target_query_receipt"]["receipt_body_sha256"] = _sha("wrong sealed body")  # type: ignore[index]
    broken_terminal = _signed({key: value for key, value in broken_terminal.items()
                               if key != "terminal_receipt_sha256"}, "terminal_receipt_sha256")
    with pytest.raises(runtime.TrackBV2SubjectMStagePRuntimeError, match="target asset/snapshot/A2/query-byte"):
        runtime.validate_primary_terminal_receipt(
            payload=broken_terminal, preflight=plan, target_receipt=target, encoder_receipt=encoder,
            score_receipts=rows, synthetic=True,
        )


def test_encoder_persistence_contract_rejects_alias_and_raw_pair_reader_requires_exact_bytes(tmp_path: Path) -> None:
    plan = _synthetic_live_preflight(runtime.build_stagep_primary_preflight(view="sua"))
    target = _target(plan)
    encoder = _encoder(plan, target)
    runtime.validate_primary_encoder_receipt(
        payload=encoder, preflight=plan, target_receipt=target, synthetic=True,
    )
    broken = copy.deepcopy(encoder)
    broken["embedding_persistence"]["canonical_artifact_path"] = "/tmp/alias_embeddings.npz"  # type: ignore[index]
    broken = _signed({key: value for key, value in broken.items() if key != "encoder_receipt_sha256"},
                     "encoder_receipt_sha256")
    with pytest.raises(runtime.TrackBV2SubjectMStagePRuntimeError, match="artifact persistence"):
        runtime.validate_primary_encoder_receipt(
            payload=broken, preflight=plan, target_receipt=target, synthetic=True,
        )

    artifact = tmp_path / "synthetic_state.pt"
    raw = b"synthetic checkpoint bytes"
    artifact.write_bytes(raw)
    sidecar = Path(f"{artifact}.sha256")
    sidecar_raw = f"{_sha_bytes(raw)}  {artifact.name}\n".encode("ascii")
    sidecar.write_bytes(sidecar_raw)
    os.chmod(artifact, 0o444)
    os.chmod(sidecar, 0o444)
    body, checked_sidecar = runtime._verify_immutable_raw_artifact_pair(
        artifact, label="synthetic checkpoint", expected_body_sha256=_sha_bytes(raw),
        expected_sidecar_sha256=_sha_bytes(sidecar_raw),
    )
    assert body.raw == raw and checked_sidecar.raw == sidecar_raw
    with pytest.raises(runtime.TrackBV2SubjectMStagePRuntimeError, match="artifact body/sidecar drift"):
        runtime._verify_immutable_raw_artifact_pair(
            artifact, label="synthetic checkpoint", expected_body_sha256=_sha("wrong"),
            expected_sidecar_sha256=_sha_bytes(sidecar_raw),
        )


def test_private_snapshot_parser_uses_held_fd_and_rejects_path_reopen_claim(tmp_path: Path) -> None:
    source = tmp_path / "synthetic_source.bin"
    source.write_bytes(b"synthetic verified target bytes")
    snapshot = tmp_path / "private_snapshot.bin"
    expected = source.read_bytes()

    def parser(fd_path: str) -> dict[str, object]:
        assert fd_path.startswith("/proc/self/fd/")
        with open(fd_path, "rb") as handle:
            assert handle.read() == expected
        return {"parser_consumed_fd_path": fd_path}

    with runtime.future_parse_canonical_asset_from_private_snapshot(
        expected_source_path=source, expected_sha256=_sha_bytes(expected), expected_bytes=len(expected),
        private_snapshot_path=snapshot, parser=parser,
    ) as (record, snapshot_contract):
        assert record["parser_consumed_fd_path"] == snapshot_contract["parser_fd_path"]
    assert snapshot.exists() and stat_mode(snapshot) == 0o444

    with pytest.raises(runtime.TrackBV2SubjectMStagePRuntimeError, match="parser did not prove"):
        with runtime.future_parse_canonical_asset_from_private_snapshot(
            expected_source_path=source, expected_sha256=_sha_bytes(expected), expected_bytes=len(expected),
            private_snapshot_path=tmp_path / "different_snapshot.bin",
            parser=lambda _fd: {"parser_consumed_fd_path": "/ordinary/path"},
        ):
            pass


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_same_fd_reader_rejects_open_then_rename_path_poison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    victim = tmp_path / "authority.json"
    replacement = tmp_path / "replacement.json"
    victim.write_bytes(b"original")
    replacement.write_bytes(b"replacement")
    real_read = os.read
    poisoned = {"done": False}

    def read_and_replace(fd: int, size: int) -> bytes:
        out = real_read(fd, size)
        if out and not poisoned["done"]:
            poisoned["done"] = True
            os.replace(replacement, victim)
        return out

    monkeypatch.setattr(runtime.os, "read", read_and_replace)
    with pytest.raises(runtime.TrackBV2SubjectMStagePRuntimeError, match="mutated|pathname identity changed"):
        runtime._read_regular_same_fd(victim, label="synthetic poison", required_mode=None)


def test_synthetic_oexcl_pair_rolls_back_body_when_sidecar_collides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = tmp_path / "receipt.json"
    sidecar = Path(f"{body}.sha256")
    real_open = os.open

    def collide(path: os.PathLike[str] | str, flags: int, *args: object) -> int:
        if Path(path) == sidecar and flags & os.O_CREAT:
            sidecar.write_text("competitor\n")
            raise FileExistsError("synthetic sidecar collision")
        return real_open(path, flags, *args)

    monkeypatch.setattr(runtime.os, "open", collide)
    with pytest.raises(FileExistsError):
        runtime._write_immutable_pair_once(body, {"synthetic": True})
    assert not body.exists()
    assert sidecar.read_text() == "competitor\n"


def test_synthetic_oexcl_pair_rolls_back_owned_partial_sidecar_and_body(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = tmp_path / "receipt.json"
    sidecar = Path(f"{body}.sha256")
    real_fsync = os.fsync
    calls = {"count": 0}

    def fail_sidecar_fsync(fd: int) -> None:
        calls["count"] += 1
        if calls["count"] == 3:
            raise OSError("synthetic sidecar fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(runtime.os, "fsync", fail_sidecar_fsync)
    with pytest.raises(OSError, match="sidecar"):
        runtime._write_immutable_pair_once(body, {"synthetic": True})
    assert not body.exists() and not sidecar.exists()


def test_execute_cli_has_no_target_date_seed_path_or_gpu_options_and_refuses_before_target() -> None:
    text = CLI.read_text()
    for forbidden in ("--target", "--date", "--seed", "--output", "--gpu", "--checkpoint", "--normalizer"):
        assert forbidden not in text
    command = [sys.executable, str(CLI), "--view", "sua", "--execute"]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert ("canonical d8/it250 cost pair" in result.stderr or
            "Stage-P runtime control" in result.stderr)


def test_runtime_module_imports_no_cebra_torch_numpy_or_target_loader() -> None:
    tree = ast.parse(Path(runtime.__file__).read_text())
    imported = {
        alias.name.split(".")[0]
        for item in tree.body if isinstance(item, (ast.Import, ast.ImportFrom))
        for alias in item.names
    }
    assert not {"cebra", "torch", "numpy", "sklearn", "pynwb", "h5py"} & imported


def test_execution_only_metric_helper_uses_torchmetrics151_cpu_float32_when_runtime_is_available() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    torchmetrics = pytest.importorskip("torchmetrics")
    if torchmetrics.__version__ != "1.5.1":
        pytest.skip("this environment does not carry the sealed TorchMetrics 1.5.1 runtime")
    prediction = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
    target = np.asarray([[0.0, 0.0], [1.0, 1.1], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)
    score, meta = runtime.future_score_torchmetrics151_cpu_float32(prediction=prediction, target=target)
    assert isinstance(score, float)
    assert meta["implementation"] == "torchmetrics.regression.R2Score"
    assert meta["dtype"] == "float32" and meta["device"] == "cpu"
    assert meta["update_call_count"] == meta["compute_call_count"] == 1
    assert meta["prediction_shape"] == meta["target_shape"] == [4, 2]

    endpoints = np.arange(20, dtype=np.int64)
    fields = endpoints[:, None] + np.arange(-5, 5, dtype=np.int64)[None, :]
    block = runtime._future_valid_offset10_block(
        embedding=np.zeros((20, 8), dtype=np.float32),
        auxiliary=np.zeros((20, 2), dtype=np.float32), endpoints=endpoints,
        receptive_fields=fields, raw_bin_indices=endpoints,
        block_role="source_session", session_id="synthetic-source",
    )
    assert block["embedding"].shape == (10, 8)
    assert block["receipt"]["padded_embedding_row_count"] == 20
    assert block["receipt"]["valid_embedding_row_count"] == block["receipt"]["valid_auxiliary_label_row_count"] == 10
    assert block["receipt"]["every_RF_wholly_inside_its_fit_block"] is True

    unordered = endpoints.copy()
    unordered[9], unordered[10] = unordered[10], unordered[9]
    with pytest.raises(runtime.TrackBV2SubjectMStagePRuntimeError, match="ordered unique consecutive"):
        runtime._future_valid_offset10_block(
            embedding=np.zeros((20, 8), dtype=np.float32), auxiliary=np.zeros((20, 2), dtype=np.float32),
            endpoints=unordered, receptive_fields=unordered[:, None] + np.arange(-5, 5, dtype=np.int64)[None, :],
            raw_bin_indices=unordered, block_role="source_session", session_id="synthetic-source",
        )
    gapped = endpoints.copy()
    gapped[10:] += 1
    with pytest.raises(runtime.TrackBV2SubjectMStagePRuntimeError, match="ordered unique consecutive"):
        runtime._future_valid_offset10_block(
            embedding=np.zeros((20, 8), dtype=np.float32), auxiliary=np.zeros((20, 2), dtype=np.float32),
            endpoints=gapped, receptive_fields=gapped[:, None] + np.arange(-5, 5, dtype=np.int64)[None, :],
            raw_bin_indices=gapped, block_role="source_session", session_id="synthetic-source",
        )

    def make_block(role: str, session: str, value: float) -> dict[str, object]:
        raw = np.arange(20, dtype=np.int64)
        return runtime._future_valid_offset10_block(
            embedding=np.full((20, 8), value, dtype=np.float32),
            auxiliary=np.column_stack((np.arange(20), np.arange(20))).astype(np.float32),
            endpoints=raw, receptive_fields=raw[:, None] + np.arange(-5, 5, dtype=np.int64)[None, :],
            raw_bin_indices=raw, block_role=role, session_id=session,
        )

    source_blocks = [make_block("source_session", session, float(index))
                     for index, session in enumerate(runtime._canonical_strict27_source_ids())]
    support_block = make_block("held_target_support", "sub-M_ses-CO-20140307", 1.0)
    query_block = make_block("strict_post_M50_query", "sub-M_ses-CO-20140307", 2.0)
    prediction, receipts, readout = runtime._future_readout_prediction(
        route="source_only_consumer_mechanism_alignment", decoder="linear_ridge",
        source_blocks=source_blocks, held_support_block=support_block, held_query_block=query_block,
    )
    assert prediction.shape == (10, 2) and len(receipts) == 27
    assert readout["query_valid_row_count"] == 10 and readout["query_enters_fit"] is False
    knn_prediction, knn_receipts, knn_readout = runtime._future_readout_prediction(
        route="source_only_consumer_mechanism_alignment", decoder="knn_cosine_k3",
        source_blocks=source_blocks, held_support_block=support_block, held_query_block=query_block,
    )
    assert knn_prediction.shape == (10, 2) and len(knn_receipts) == 27
    assert knn_readout["knn_execution"]["algorithm"] == "exact_chunked_cosine_topk_v1"
    assert knn_readout["knn_execution"]["full_query_by_training_similarity_matrix_materialized"] is False
    raw_query = dict(query_block)
    raw_query["embedding"] = np.zeros((20, 8), dtype=np.float32)
    raw_query["auxiliary"] = np.zeros((20, 2), dtype=np.float32)
    raw_query["endpoints"] = np.arange(20, dtype=np.int64)
    raw_query["receptive_fields"] = np.arange(20, dtype=np.int64)[:, None] + np.arange(-5, 5, dtype=np.int64)[None, :]
    with pytest.raises(runtime.TrackBV2SubjectMStagePRuntimeError, match="raw/padded"):
        runtime._future_readout_prediction(
            route="source_only_consumer_mechanism_alignment", decoder="linear_ridge",
            source_blocks=source_blocks, held_support_block=support_block, held_query_block=raw_query,
        )


def test_exact_chunked_cosine_knn_matches_dense_reference_and_never_materializes_full_matrix() -> None:
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(20260815)
    query = rng.normal(size=(17, 8)).astype(np.float32)
    train = rng.normal(size=(29, 8)).astype(np.float32)
    labels = rng.normal(size=(29, 2)).astype(np.float32)
    observed: list[dict[str, int]] = []
    normalizations: list[dict[str, int]] = []
    prediction, execution = runtime._future_exact_chunked_cosine_knn(
        query=query, training_embedding=train, training_labels=labels,
        query_chunk_rows=4, training_chunk_rows=7, chunk_observer=lambda record: observed.append(dict(record)),
        normalization_observer=lambda record: normalizations.append(dict(record)),
    )
    qnorm = np.linalg.norm(query, axis=1, keepdims=True)
    xnorm = np.linalg.norm(train, axis=1, keepdims=True)
    qnorm[qnorm == 0] = 1.0
    xnorm[xnorm == 0] = 1.0
    dense_similarity = (query / qnorm) @ (train / xnorm).T
    global_indices = np.broadcast_to(np.arange(train.shape[0], dtype=np.int64), dense_similarity.shape)
    dense_order = np.lexsort((global_indices, -dense_similarity), axis=1)[:, :3]
    dense_prediction = labels[dense_order].mean(axis=1).astype(np.float32)
    assert np.array_equal(prediction, dense_prediction)
    assert execution["tie_break"] == "cosine_similarity_descending_then_global_training_index_ascending"
    assert execution["neighbor_global_index_int64_sha256"] == _sha_bytes(
        memoryview(np.ascontiguousarray(dense_order, dtype=np.int64)).cast("B").tobytes())
    # BLAS may change a cosine's last bit when the output is tiled, but the
    # deterministic global neighbor set and the resulting float32 prediction
    # above are exactly equal.  The runtime therefore seals its actual tiled
    # similarities, rather than pretending a separately materialized dense
    # matrix supplied their bytes.
    assert len(execution["neighbor_cosine_similarity_float32_sha256"]) == 64
    assert observed and max(record["similarity_elements"] for record in observed) <= 4 * 7
    assert all(record["similarity_elements"] < query.shape[0] * train.shape[0] for record in observed)
    assert execution["full_query_by_training_similarity_matrix_materialized"] is False
    assert execution["training_normalization_pass_count"] == 1
    assert execution["normalized_training_buffer_byte_count"] == train.shape[0] * train.shape[1] * 4
    assert execution["total_exhaustive_similarity_element_count"] == query.shape[0] * train.shape[0]
    assert normalizations == [{
        "training_normalization_pass_count": 1, "training_row_count": train.shape[0],
        "normalized_training_buffer_byte_count": train.shape[0] * train.shape[1] * 4,
    }]

    # Equal cosine scores use the smallest *global* training index across
    # tiles, rather than whichever train chunk happens to be visited first.
    tied_prediction, tied_execution = runtime._future_exact_chunked_cosine_knn(
        query=np.asarray([[1.0, 0.0]], dtype=np.float32),
        training_embedding=np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        training_labels=np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]], dtype=np.float32),
        query_chunk_rows=1, training_chunk_rows=2,
    )
    assert np.array_equal(tied_prediction, np.asarray([[2.0, 0.0]], dtype=np.float32))
    expected_tied_indices = np.asarray([[0, 2, 4]], dtype=np.int64)
    assert tied_execution["neighbor_global_index_int64_sha256"] == _sha_bytes(
        memoryview(expected_tied_indices).cast("B").tobytes())

    # A larger synthetic shape would be >2 million entries if the original
    # dense Q×T matrix were allocated.  The observer proves every actual tile
    # stays below 17×113 entries, independent of total Q and T.
    large_observed: list[dict[str, int]] = []
    large_normalizations: list[dict[str, int]] = []
    large_prediction, large_execution = runtime._future_exact_chunked_cosine_knn(
        query=np.zeros((513, 8), dtype=np.float32),
        training_embedding=np.zeros((4097, 8), dtype=np.float32),
        training_labels=np.zeros((4097, 2), dtype=np.float32),
        query_chunk_rows=17, training_chunk_rows=113,
        chunk_observer=lambda record: large_observed.append(dict(record)),
        normalization_observer=lambda record: large_normalizations.append(dict(record)),
    )
    assert large_prediction.shape == (513, 2)
    assert large_observed and max(record["similarity_elements"] for record in large_observed) <= 17 * 113
    assert max(record["similarity_elements"] for record in large_observed) < 513 * 4097
    assert large_execution["maximum_similarity_tile_elements"] <= 17 * 113
    assert large_execution["full_query_by_training_similarity_matrix_materialized"] is False
    assert large_execution["training_normalization_pass_count"] == 1
    assert large_execution["normalized_training_buffer_byte_count"] == 4097 * 8 * 4
    assert large_execution["total_exhaustive_similarity_element_count"] == 513 * 4097
    assert large_normalizations == [{
        "training_normalization_pass_count": 1, "training_row_count": 4097,
        "normalized_training_buffer_byte_count": 4097 * 8 * 4,
    }]
