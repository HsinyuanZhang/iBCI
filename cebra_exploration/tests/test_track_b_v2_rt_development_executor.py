"""Focused no-target tests for the additive Track-B RT executor/scorer plan."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "cebra_exploration/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_contract as base  # noqa: E402
import track_b_v2_rt_development_executor as executor  # noqa: E402


CLI = REPO_ROOT / "cebra_exploration/scripts/run_track_b_v2_rt_development_executor.py"
FOLD = "rt_outer_fold_00"
TARGET = "ses-RT-20131009"


def _real_plan() -> dict:
    return executor.build_rt_development_executor_dry_plan(
        dataset="rt", view=None, outer_fold_id=FOLD, target_session_id=TARGET,
    )


def _valid_cost_gate() -> dict:
    return {
        "schema": "track_b_v2_subject_m_fixed_gpu_cost_gate_v1",
        "status": "FIXED_D8IT250_GPU_COST_RECEIPT_VALID__REQUIRED_BUT_NOT_SUFFICIENT_FOR_TARGET_EXECUTION",
        "canonical_body_path": str(executor.fixed_gpu.CANONICAL_COST_OUTPUT),
        "canonical_body_sha256": "a" * 64,
        "canonical_sidecar_path": str(executor.fixed_gpu.CANONICAL_COST_OUTPUT) + ".sha256",
        "canonical_sidecar_sha256": "b" * 64,
        "target_execution_permitted": False,
        "target_data_opened": False,
        "gpu_used_by_this_gate": False,
    }


def test_real_plan_consumes_materializer_and_rt_asset_but_cost_blocks_before_descriptor() -> None:
    plan = _real_plan()
    assert plan["status"] == executor.STATUS_COST_BLOCKED
    binding = plan["canonical_materializer_binding"]
    assert binding["rt_local_asset_authority_body_sha256"] == executor.LOCAL_ASSET_AUTHORITY_SHA256
    assert binding["rt_local_asset_authority_asset_count"] == 15
    assert binding["canonical_metric_pointer_body_sha256"] == (
        "d2e53165cba76aea5f28b893592b7bcf2951b8a4f13bafe8d5a14061b30072b6"
    )
    descriptor = plan["future_held_descriptor_private_snapshot_parser_contract"]
    assert descriptor["status"].startswith("BLOCKED_BEFORE_HELD_PATH_RESOLUTION")
    assert descriptor["canonical_target_path_copied_into_executor_plan"] is False
    assert descriptor["target_path_resolved"] is descriptor["target_fd_opened"] is False
    assert descriptor["private_snapshot_created"] is descriptor["parser_called"] is False
    for field in (
        "target_execution_permitted", "target_data_discovery_permitted", "target_data_opened",
        "target_query_opened", "formal_data_opened", "cebra_imported", "cebra_fit_called",
        "gpu_used", "readout_fit_called", "score_emitted", "receipt_minted",
        "official_execution_receipt_minted",
    ):
        assert plan[field] is False


def test_missing_cost_gate_does_not_coerce_or_copy_held_path(
        monkeypatch: pytest.MonkeyPatch) -> None:
    real_materializer = executor.materializer.build_development_target_materializer_dry_plan(
        dataset="rt", view=None, outer_fold_id=FOLD, target_session_id=TARGET,
    )

    class PoisonPath:
        def __str__(self) -> str:
            raise AssertionError("cost-blocked plan must not coerce held target path")

    real_materializer["target_asset_ledger_gate"]["target_asset"]["canonical_local_nwb_path"] = PoisonPath()
    monkeypatch.setattr(
        executor.materializer, "build_development_target_materializer_dry_plan",
        lambda **_kwargs: real_materializer,
    )
    monkeypatch.setattr(
        executor.subject_executor, "inspect_fixed_d8it250_gpu_cost_receipt",
        lambda: {"status": "NO_GO__FIXED_D8IT250_GPU_COST_RECEIPT_NOT_FRESH_AND_VALID"},
    )
    plan = _real_plan()
    descriptor = plan["future_held_descriptor_private_snapshot_parser_contract"]
    assert "future_canonical_target_path" not in descriptor
    assert descriptor["target_path_resolved"] is False


def test_valid_cost_still_only_renders_future_same_fd_private_snapshot_contract(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        executor.subject_executor, "inspect_fixed_d8it250_gpu_cost_receipt", _valid_cost_gate,
    )
    plan = _real_plan()
    assert plan["status"] == executor.STATUS_IMPLEMENTATION_BLOCKED
    descriptor = plan["future_held_descriptor_private_snapshot_parser_contract"]
    assert descriptor["future_expected_byte_count"] == 73479612
    assert descriptor["future_expected_sha256"] == (
        "5bd05b43ac590c70a3edb4e6fc4d0614dfe3e14d97ab5db54137dce5d6e0b11c"
    )
    assert descriptor["held_source_descriptor"]["one_open_only"] is True
    assert descriptor["held_source_descriptor"]["ordinary_pathname_reopen_for_parser_permitted"] is False
    assert descriptor["held_source_descriptor"]["canonical_loader_and_opener"] == (
        "track_b_v2_rt_local_asset_authority.open_canonical_verified_asset_after_authority"
    )
    snapshot = descriptor["private_snapshot"]
    assert snapshot["created_from_continuously_held_verified_source_fd"] is True
    assert snapshot["create_flags"] == ["O_WRONLY", "O_CREAT", "O_EXCL", "O_NOFOLLOW", "O_CLOEXEC"]
    assert snapshot["parser_fd_held_open_and_rewound_to_zero"] is True
    assert snapshot["parser_pathname_reopen_permitted"] is False
    binding = descriptor["canonical_rt_same_fd_contract_binding"]
    assert binding["symbol"] == "open_canonical_verified_asset_after_authority"
    assert binding["authority_body_sha256"] == executor.LOCAL_ASSET_AUTHORITY_SHA256
    assert binding["caller_supplied_asset_ledger_mapping_permitted"] is False
    assert binding["direct_call_authorized_by_this_plan"] is False
    assert plan["target_execution_permitted"] is plan["target_data_opened"] is False


def test_fixed_geometry_joint_encoder_and_all_three_routes_are_mandatory() -> None:
    plan = _real_plan()
    fixed = plan["scientific_contract"]
    assert fixed["embedding_geometry"] == {
        "output_dimension": 8,
        "iterations": 10_000,
        "encoder_geometry_key": "d8-it10000",
        "source_or_target_score_selection_performed": False,
    }
    assert fixed["decoders"]["linear_ridge"]["normalized_ridge_lambda"] == 0.01
    assert fixed["decoders"]["knn_cosine_k3"]["metric"] == "cosine"
    assert fixed["decoders"]["knn_cosine_k3"]["k"] == 3
    assert fixed["seeds"] == [42, 43, 44]
    joint = plan["joint_encoder_contract"]
    assert joint["model_arm"] == "cebra_joint_behavior"
    assert joint["source_session_count_per_fold"] == 14
    assert joint["held_target_support_budget_trials"] == 24
    assert joint["held_target_is_not_an_unfitted_unseen_transform"] is True
    assert joint["target_query_neural_behavior_or_labels_enter_fit"] is False
    assert plan["mandatory_readout_routes"] == list(executor.READOUT_ROUTES)


def test_rt_m24_post_m_endpoint_and_offset10_noncausal_authority_is_exact() -> None:
    contract = _real_plan()["rt_M24_post_M_query_endpoint_authority"]
    assert contract["support"]["rewarded_trial_ordinals"] == list(range(24))
    assert contract["support"]["all_intervening_raw_rows_included"] is True
    assert contract["support"]["rewarded_segments_concatenated"] is False
    assert contract["query"]["strict_post_M24_only"] is True
    assert contract["query"]["ordered_prediction_target_raw_bin_indices"] == "valid_window_start_plus_49"
    assert contract["query"]["query_neural_dense_velocity_or_behavior_enters_encoder_fit"] is False
    offset = contract["offset10_receptive_field"]
    assert offset["half_open_offsets_relative_to_prediction_target"] == [-5, 5]
    assert offset["exact_raw_indices_per_endpoint"] == "range(endpoint-5, endpoint+5)"
    assert offset["width_raw_bins"] == 10
    assert offset["strictly_future_raw_bins_after_prediction_target"] == 4
    assert offset["support_query_boundary_crossing_permitted"] is False
    assert offset["causal_temporal_exposure_matched"] is False
    assert offset["bias_direction"] == "favors_CEBRA_accuracy"


def test_score_topology_uses_one_joint_bundle_for_six_mandatory_scores_per_seed_fold() -> None:
    topology = _real_plan()["immutable_future_receipt_topology"]
    assert topology["outer_fold_count"] == 15
    assert topology["cebra_seed_count"] == 3
    assert topology["encoder_receipt_count"] == 45
    assert topology["score_receipt_count"] == 270
    per_seed = topology["per_outer_fold_and_seed"]
    assert per_seed["encoder_bundle"]["model_arm"] == "cebra_joint_behavior"
    assert per_seed["encoder_bundle"]["held_query_neural_or_behavior_enters_fit"] is False
    assert set(per_seed["readout_routes"]) == set(executor.READOUT_ROUTES)
    assert per_seed["decoders"] == list(executor.DECODERS)
    assert per_seed["same_ordered_query_endpoint_and_target_byte_authority_for_all_six_scores"] is True
    for route in per_seed["readout_routes"].values():
        assert route["encoder_fit_scope"] == "same_joint_14_source_plus_held_M24_support_encoder"


def test_rt_local_asset_authority_drift_fails_before_descriptor(
        monkeypatch: pytest.MonkeyPatch) -> None:
    plan = executor.materializer.build_development_target_materializer_dry_plan(
        dataset="rt", view=None, outer_fold_id=FOLD, target_session_id=TARGET,
    )
    plan["target_asset_ledger_gate"]["authority"]["body_sha256"] = "0" * 64
    monkeypatch.setattr(
        executor.materializer, "build_development_target_materializer_dry_plan",
        lambda **_kwargs: plan,
    )
    with pytest.raises(executor.TrackBV2RTDevelopmentExecutorError, match="authority path/SHA/count drift"):
        _real_plan()


def test_scope_and_all_execution_surfaces_fail_closed_before_target_like_input() -> None:
    class Poison:
        def __str__(self) -> str:
            raise AssertionError("wrong scope must fail before target coercion")

    with pytest.raises(executor.TrackBV2RTDevelopmentExecutorError, match="only dataset=rt"):
        executor.build_rt_development_executor_dry_plan(
            dataset="subject_m", view="sua", outer_fold_id="bad", target_session_id=Poison(),
        )
    with pytest.raises(base.TrackBV2ContractError, match="H1-excluded"):
        executor.build_rt_development_executor_dry_plan(
            dataset="falcon_h1", view=None, outer_fold_id="bad", target_session_id=Poison(),
        )
    with pytest.raises(executor.TrackBV2RTDevelopmentExecutorError, match="no execution mode"):
        executor.build_rt_development_executor_dry_plan(
            dataset="rt", view=None, outer_fold_id=FOLD, target_session_id=TARGET,
            execution_requested=True,
        )
    with pytest.raises(executor.TrackBV2RTDevelopmentExecutorError, match="no-target scaffold"):
        executor.refuse_rt_target_execution(
            dataset="rt", view=None, outer_fold_id=FOLD, target_session_id=TARGET,
        )


def test_cli_has_no_target_path_execution_score_gpu_cebra_or_output_surface() -> None:
    completed = subprocess.run(
        [sys.executable, str(CLI), "--help"], capture_output=True, text=True, check=True,
    )
    text = completed.stdout.lower()
    for forbidden in ("--target-path", "--execute", "--score", "--gpu", "--cebra", "--output"):
        assert forbidden not in text
    assert "--outer-fold-id" in text and "--target-session-id" in text


def test_real_cli_prints_no_go_without_target_cebra_gpu_score_or_write() -> None:
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(CLI), "--outer-fold-id", FOLD, "--target-session-id", TARGET],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == executor.STATUS_COST_BLOCKED
    assert payload["target_data_opened"] is payload["cebra_imported"] is False
    assert payload["gpu_used"] is payload["score_emitted"] is payload["receipt_minted"] is False


def test_new_rt_executor_source_has_no_loader_fit_score_or_writer_import() -> None:
    text = Path(executor.__file__).read_text()
    for forbidden in (
        "import cebra", "import torch", "import sklearn", "pynwb", "h5py",
        "write_immutable_receipt(", "os.listdir(", "os.scandir(", ".glob(", ".rglob(",
    ):
        assert forbidden not in text
    assert "inspect_fixed_d8it250_gpu_cost_receipt" in text
    assert executor.LOCAL_ASSET_AUTHORITY_SHA256 in text
