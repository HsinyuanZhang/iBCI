"""No-target synthetic tests for the Subject-M Track-B v2 one-cell successor."""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "cebra_exploration" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_subject_m_development_executor as development_executor  # noqa: E402
import track_b_v2_subject_m_one_cell_successor as successor  # noqa: E402


CLI = REPO_ROOT / "cebra_exploration" / "scripts" / "run_track_b_v2_subject_m_one_cell_successor.py"
SUA = {
    "view": "sua", "outer_fold_id": "subject_m_sua_external_target_20140307",
    "target_session_id": "sub-M_ses-CO-20140307", "cebra_seed": 42,
}
PMUA = {
    "view": "pseudo_mua", "outer_fold_id": "subject_m_pseudo_mua_external_target_20140307",
    "target_session_id": "sub-M_ses-CO-20140307", "cebra_seed": 43,
}


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _valid_cost_gate() -> dict[str, object]:
    return {
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


def _synthetic_rows() -> list[dict[str, object]]:
    return [
        {
            "readout_route": route,
            "decoder": decoder,
            "encoder_bundle_identity_sha256": _sha("one shared encoder"),
            "ordered_target_float32_raw_bytes_sha256": _sha("one ordered target bytes"),
            "synthetic_only": True,
        }
        for route in (
            "source_only_consumer_mechanism_alignment",
            "target_support_only_standard_cebra_accuracy",
            "source_plus_target_support_hybrid_sensitivity",
        )
        for decoder in ("linear_ridge", "knn_cosine_k3")
    ]


def test_real_sua_preflight_accepts_terminal_cost_but_stays_no_target_contract_only() -> None:
    plan = successor.build_subject_m_one_cell_preflight(**SUA)
    assert plan["status"] == "COST_GATE_VALID__ONE_CELL_EXECUTOR_CONTRACT_READY_FOR_ROOT_REVIEW_ONLY"
    cost = plan["fixed_d8it250_gpu_cost_gate"]
    assert cost["status"] == (
        "FIXED_D8IT250_GPU_COST_RECEIPT_VALID__REQUIRED_BUT_NOT_SUFFICIENT_FOR_TARGET_EXECUTION"
    )
    assert cost["canonical_body_sha256"] == (
        "ec7096a5e54e444fd6cdafa241aaa88a0720143e3c4e42e3565f022ee662c8e2"
    )
    assert cost["target_execution_permitted"] is False
    assert cost["target_data_opened"] is False
    assert plan["cell"] == {
        "dataset": "subject_m", "view": "sua", "outer_fold_id": SUA["outer_fold_id"],
        "target_session_id": SUA["target_session_id"], "cebra_seed": 42,
        "canonical_cell_id": "subject_m__sua__subject_m_sua_external_target_20140307__seed42",
    }
    assert plan["canonical_materializer_binding"]["caller_supplied_authority_or_asset_sha_permitted"] is False
    assert plan["fixed_final_geometry"] == {
        "output_dimension": 8, "iterations": 10_000,
        "linear_ridge_normalized_lambda": 0.01, "cosine_knn_k": 3,
        "source_or_target_geometry_selection_performed": False,
    }
    for flag in (
        "root_authorized_live_execution_permitted", "target_data_discovery_permitted", "target_data_opened",
        "target_query_opened", "formal_data_opened", "cebra_imported", "cebra_trained",
        "readout_fit_called", "score_emitted", "gpu_used", "official_execution_receipt_minted",
    ):
        assert plan[flag] is False


def test_pmua_chain_requires_same_record_pooling_replay_cross_view_and_exact_offset10() -> None:
    plan = successor.build_subject_m_one_cell_preflight(**PMUA)
    chain = plan["future_execution_chain"]
    lineage = chain["reference_lineage"]["lineage_contract"]
    parity = lineage["sua_pmua_cross_view_parity"]
    pooling = parity["pseudo_mua_target_pooling_replay"]
    assert parity["same_target_record_required"] is True
    assert parity["same_ordered_target_behavior_bytes_required"] is True
    assert parity["same_ordered_prediction_endpoint_authority_required"] is True
    assert pooling["input_sua_feature_sha256"] == "REQUIRED__EXACT_TARGET_RECORD_INPUT"
    assert pooling["ordered_unit_ids_raw_bytes_sha256"] == "REQUIRED"
    assert pooling["ordered_unit_electrode_ids_raw_bytes_sha256"] == "REQUIRED"
    assert pooling["replay_exact_equal_to_target_pmua_feature"] is True
    support_query = chain["target_support_and_query"]
    assert support_query["support"] == "one_continuous_chronological_prefix_through_stop_of_rewarded_trial_50"
    assert support_query["query"] == "rewarded_trials_strictly_after_50_only"
    assert support_query["prediction_target_timestamp"] == "valid_window_start_plus_49"
    assert support_query["receptive_field"] == "range(endpoint-5, endpoint+5)"
    assert support_query["strictly_future_raw_bins"] == 4
    assert support_query["causal_temporal_exposure_matched"] is False
    assert support_query["bias_direction"] == "favors_CEBRA_accuracy"


def test_internal_valid_cost_gate_changes_only_review_readiness_not_execution_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(development_executor, "inspect_fixed_d8it250_gpu_cost_receipt", _valid_cost_gate)
    plan = successor.build_subject_m_one_cell_preflight(**SUA)
    assert plan["status"] == "COST_GATE_VALID__ONE_CELL_EXECUTOR_CONTRACT_READY_FOR_ROOT_REVIEW_ONLY"
    assert plan["fixed_d8it250_gpu_cost_gate"]["canonical_body_sha256"] == _sha("cost-body")
    assert plan["root_authorized_live_execution_permitted"] is False
    assert plan["target_data_opened"] is plan["cebra_imported"] is plan["score_emitted"] is False


def test_identity_rejects_wrong_view_date_fold_seed_and_target_before_materializer_use() -> None:
    with pytest.raises(successor.TrackBV2SubjectMOneCellSuccessorError, match="unique canonical"):
        successor.build_subject_m_one_cell_preflight(**(SUA | {"outer_fold_id": "subject_m_sua_external_target_20140308"}))
    with pytest.raises(successor.TrackBV2SubjectMOneCellSuccessorError, match="seed"):
        successor.build_subject_m_one_cell_preflight(**(SUA | {"cebra_seed": 99}))
    with pytest.raises(successor.TrackBV2SubjectMOneCellSuccessorError, match="grammar"):
        successor.build_subject_m_one_cell_preflight(**(SUA | {"target_session_id": "ses-RT-20131009"}))
    with pytest.raises(successor.TrackBV2SubjectMOneCellSuccessorError, match="no execute mode"):
        successor.build_subject_m_one_cell_preflight(**SUA, execution_requested=True)


def test_public_preflight_accepts_no_target_path_sha_checkpoint_output_or_cebra_parameters() -> None:
    params = set(inspect.signature(successor.build_subject_m_one_cell_preflight).parameters)
    for forbidden in (
        "target_path", "target_sha256", "asset_sha256", "checkpoint", "normalizer",
        "output", "output_root", "cebra", "score", "authority",
    ):
        assert forbidden not in params
    source = Path(successor.__file__).read_text()
    parsed = ast.parse(source)
    imported = {
        alias.name.split(".")[0]
        for item in ast.walk(parsed) if isinstance(item, (ast.Import, ast.ImportFrom))
        for alias in item.names
    }
    assert not {"cebra", "torch", "sklearn", "numpy", "pynwb", "h5py"} & imported


def test_future_topology_requires_one_encoder_per_model_arm_and_all_18_score_pairs() -> None:
    plan = successor.build_subject_m_one_cell_preflight(**SUA)
    topology = plan["future_receipt_topology"]
    assert topology["writer_invariants"]["body_creation"] == "O_EXCL_ONLY"
    assert topology["writer_invariants"]["body_and_sidecar_mode"] == "0444"
    assert topology["writer_invariants"]["body_and_sidecar_fresh_before_target_resolution"] is True
    assert topology["future_terminal_preflight_receipt_body_and_sidecar_sha256"].startswith("REQUIRED")
    per_arm = topology["per_model_arm"]
    assert set(per_arm) == {"cebra_joint_behavior", "cebra_frozen_source_adapt", "cebra_adapt_unaligned"}
    assert sum(len(value["score_receipts"]) for value in per_arm.values()) == 18
    for value in per_arm.values():
        assert value["one_encoder_bundle_receipt"] is True
        assert value["encoder_may_not_be_refit_for_route_or_decoder"] is True
        assert {(x["readout_route"], x["decoder"]) for x in value["score_receipts"]} == {
            (route, decoder)
            for route in (
                "source_only_consumer_mechanism_alignment",
                "target_support_only_standard_cebra_accuracy",
                "source_plus_target_support_hybrid_sensitivity",
            )
            for decoder in ("linear_ridge", "knn_cosine_k3")
        }


def test_synthetic_scorer_layout_requires_six_pairs_one_encoder_and_one_target_bytes_authority() -> None:
    preflight = successor.build_subject_m_one_cell_preflight(**SUA)
    valid = successor.validate_synthetic_one_cell_score_layout(
        preflight=preflight, model_arm="cebra_joint_behavior", score_rows=_synthetic_rows(), synthetic_fixture=True,
    )
    assert valid["route_decoder_count"] == 6
    assert valid["score_emitted"] is False
    altered_encoder = _synthetic_rows()
    altered_encoder[-1]["encoder_bundle_identity_sha256"] = _sha("illegal refit")
    with pytest.raises(successor.TrackBV2SubjectMOneCellSuccessorError, match="illegally refit"):
        successor.validate_synthetic_one_cell_score_layout(
            preflight=preflight, model_arm="cebra_joint_behavior", score_rows=altered_encoder,
            synthetic_fixture=True,
        )
    duplicate = _synthetic_rows()
    duplicate[-1] = copy.deepcopy(duplicate[0])
    with pytest.raises(successor.TrackBV2SubjectMOneCellSuccessorError, match="missing or duplicate"):
        successor.validate_synthetic_one_cell_score_layout(
            preflight=preflight, model_arm="cebra_joint_behavior", score_rows=duplicate,
            synthetic_fixture=True,
        )
    with pytest.raises(successor.TrackBV2SubjectMOneCellSuccessorError, match="synthetic-fixture"):
        successor.validate_synthetic_one_cell_score_layout(
            preflight=preflight, model_arm="cebra_joint_behavior", score_rows=_synthetic_rows(),
            synthetic_fixture=False,
        )


def test_future_aggregate_interface_freezes_15_by_three_session_then_seed_admission() -> None:
    aggregate = successor.build_subject_m_view_future_aggregate_interface(view="sua")
    assert aggregate["expected_external_target_cell_count"] == 15
    assert aggregate["cebra_seed_set"] == [42, 43, 44]
    assert aggregate["required_one_cell_receipt_count_per_model_arm_route_decoder"] == 45
    assert aggregate["admission_requirements"]["aggregate_session_then_seed_not_bins"] is True
    assert aggregate["metrics_or_receipts_consumed"] is False
    assert aggregate["target_data_opened"] is aggregate["score_emitted"] is False


def test_cli_is_dry_by_default_and_execute_is_an_explicit_tripwire() -> None:
    help_result = subprocess.run([sys.executable, str(CLI), "--help"], check=True, capture_output=True, text=True)
    text = help_result.stdout.lower()
    for forbidden in ("--target-path", "--output", "--score", "--gpu", "--cebra", "--checkpoint"):
        assert forbidden not in text
    assert "--execute" in text
    dry = subprocess.run(
        [sys.executable, str(CLI), "--view", "sua", "--outer-fold-id", SUA["outer_fold_id"],
         "--target-session-id", SUA["target_session_id"], "--seed", "42"],
        check=True, capture_output=True, text=True,
    )
    assert '"status":"COST_GATE_VALID__ONE_CELL_EXECUTOR_CONTRACT_READY_FOR_ROOT_REVIEW_ONLY"' in dry.stdout
    execute = subprocess.run(
        [sys.executable, str(CLI), "--view", "sua", "--outer-fold-id", SUA["outer_fold_id"],
         "--target-session-id", SUA["target_session_id"], "--seed", "42", "--execute"],
        capture_output=True, text=True,
    )
    assert execute.returncode != 0
    assert "not implemented" in execute.stderr
