"""No-target tests for the canonical Track-B v2 development target authority.

These tests read only sealed Track-B receipt pairs.  They never open an NWB or
NPZ, import CEBRA, create a model, fit a readout, score a target, use a GPU, or
write an official receipt.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "cebra_exploration" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_development_target_authority as authority  # noqa: E402
import track_b_v2_metric_pointer_authority as pointer  # noqa: E402


def _assert_no_execution(payload: dict[str, object]) -> None:
    for key in (
        "target_data_discovery_permitted", "target_data_opened", "target_query_opened",
        "formal_data_opened", "cebra_imported", "cebra_trained", "gpu_used",
        "score_emitted", "official_execution_receipt_minted",
    ):
        assert payload[key] is False


def test_subject_m_sua_binds_real_v2_pointer_strict27_source_and_v9_target_membership() -> None:
    plan = authority.build_development_target_query_authority(
        dataset="subject_m", view="sua", outer_fold_id="subject_m_sua_external_target_20140307",
        target_session_id="sub-M_ses-CO-20140307",
    )
    assert plan["metric_pointer_validation"]["pointer_body_sha256"] == (
        "bc965600d6288576715a68e51343b1bd1a5b66ad5456fcb5ab63f5af4acdf45c"
    )
    source = plan["source_authority_binding"]
    assert source["source_session_count"] == 27
    assert source["canonical_source_authority_root"].endswith("strict27_sua_continuous_v2_dev")
    assert set(source["source_authority_receipts"]) == set(authority._MEMBER_FILENAMES)
    membership = plan["target_reference_lineage_membership"]
    assert membership["target_session_seed_set"] == [42, 43, 44]
    assert membership["target_asset_id"]
    assert membership["target_query_window_count"] > 0
    query = plan["target_query_authority_requirements"]["target_query_must_bind"]
    assert query["ordered_prediction_target_raw_bin_indices"] == "EXACTLY_VALID_WINDOW_START_PLUS_49"
    assert query["default_offset10_receptive_field_width_raw_bins"] == 10
    assert query["default_offset10_previous_raw_bins"] == 5
    assert query["default_offset10_right_extent_including_target_raw_bins"] == 5
    assert query["default_offset10_strictly_future_raw_bins"] == 4
    temporal = plan["readout_authority_plan"]["temporal_exposure_policy"]["default_offset10_model_offset"]
    assert temporal == {
        "left_previous_raw_bins": 5,
        "right_extent_including_target_raw_bins": 5,
        "strictly_future_raw_bins_after_target": 4,
        "receptive_field_width_raw_bins": 10,
        "half_open_offsets_relative_to_target": [-5, 5],
    }
    assert query["causal_temporal_exposure_matched"] is False
    assert plan["accuracy_table_headline_readout"] == "target_support_only_standard_cebra_accuracy"
    assert plan["mandatory_readout_routes"] == [
        "source_only_consumer_mechanism_alignment",
        "target_support_only_standard_cebra_accuracy",
        "source_plus_target_support_hybrid_sensitivity",
    ]
    fixed = plan["fixed_canonical_geometry_contract"]
    assert fixed["embedding_geometry"]["encoder_geometry_key"] == "d8-it10000"
    assert fixed["linear_ridge_normalized_lambda"] == 0.01
    assert fixed["cosine_knn_k"] == 3
    assert fixed["source_geometry_selection_performed"] is False
    assert fixed["target_geometry_selection_performed"] is False
    assert fixed["source_selector_fit_count"] == 0
    assert set(fixed["implementation_closure"]) == {
        "fixed_canonical_successor", "fixed_canonical_engineering", "actual_cpu_route_geometry",
    }
    historical = source["historical_selector_plan"]
    assert historical["historical_selector_plan_executed"] is False
    assert historical["historical_selector_plan_selected_geometry"] is False
    assert historical["historical_selector_plan_authorizes_fixed_execution"] is False
    assert source["source_authority_roles"]["source_only_dual_geometry_selection_plan"].startswith(
        "historical_unexecuted_lineage"
    )
    _assert_no_execution(plan)


def test_subject_m_pmua_binds_the_distinct_actual_pooling_source_authority() -> None:
    plan = authority.build_development_target_query_authority(
        dataset="subject_m", view="pseudo_mua", outer_fold_id="subject_m_pseudo_mua_external_target_20140307",
        target_session_id="sub-M_ses-CO-20140307",
    )
    assert plan["metric_pointer_validation"]["pointer_body_sha256"] == (
        "62f74eb29cd165a6f888234a693580ae3f90675fc92240dfc3a880694f943f67"
    )
    assert plan["source_authority_binding"]["canonical_source_authority_root"].endswith(
        "strict27_pmua_continuous_v2_dev"
    )
    neural = plan["source_authority_binding"]["source_authority_receipts"]["source_neural_input_authority"]
    assert neural["body_sha256"] == "47d7609e0a0dc6393d7d2c2d80b419088cff602813d88a8fd9200613522967ac"
    _assert_no_execution(plan)


def test_rt_binds_completed_15fold_manifest_exact_fold_and_14_source_pairs() -> None:
    plan = authority.build_development_target_query_authority(
        dataset="rt", view=None, outer_fold_id="rt_outer_fold_00",
        target_session_id="ses-RT-20131009",
    )
    assert plan["metric_pointer_validation"]["pointer_body_sha256"] == (
        "d2e53165cba76aea5f28b893592b7bcf2951b8a4f13bafe8d5a14061b30072b6"
    )
    source = plan["source_authority_binding"]
    assert source["rt_full_source_manifest"]["body_sha256"] == authority._RT_MANIFEST_SHA256
    assert source["rt_full_source_cost_aggregate"]["body_sha256"] == authority._RT_AGGREGATE_SHA256
    assert source["selected_outer_fold_id"] == "rt_outer_fold_00"
    assert source["opaque_held_out_target_session_id"] == "ses-RT-20131009"
    assert source["source_session_count"] == 14
    assert "ses-RT-20131009" not in source["source_session_ids"]
    assert plan["target_reference_lineage_membership"]["membership_semantics"].startswith("sealed_RT_15fold")
    _assert_no_execution(plan)


def test_missing_or_invalid_authority_fails_before_target_identifier_is_coerced(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_pointer(dataset: str, view: str | None):
        raise authority.TrackBV2DevelopmentTargetAuthorityError("pointer deliberately unavailable")

    class Poison:
        def __str__(self) -> str:
            raise AssertionError("target ID coercion must follow authority checks")

    monkeypatch.setattr(authority, "_canonical_pointer_validation", reject_pointer)
    with pytest.raises(authority.TrackBV2DevelopmentTargetAuthorityError, match="deliberately unavailable"):
        authority.build_development_target_query_authority(
            dataset="subject_m", view="sua", outer_fold_id="subject_m_sua_external_target_20140307", target_session_id=Poison(),
        )


def test_subject_m_rejects_a_grammar_valid_target_missing_from_pointer_bound_v9_lineage() -> None:
    with pytest.raises(authority.TrackBV2DevelopmentTargetAuthorityError, match="absent from the pointer-bound"):
        authority.build_development_target_query_authority(
            dataset="subject_m", view="sua", outer_fold_id="subject_m_sua_external_target_20990101",
            target_session_id="sub-M_ses-CO-20990101",
        )


def test_rt_rejects_target_id_that_does_not_equal_selected_sealed_held_out_fold() -> None:
    with pytest.raises(authority.TrackBV2DevelopmentTargetAuthorityError, match="does not equal the sealed selected fold"):
        authority.build_development_target_query_authority(
            dataset="rt", view=None, outer_fold_id="rt_outer_fold_00", target_session_id="ses-RT-20131010",
        )


def test_source_authority_alias_root_is_rejected_before_any_pair_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("platform cannot create symlink adversarial fixture")
    monkeypatch.setattr(authority, "_SUBJECT_M_SOURCE_ROOTS", {"sua": alias, "pseudo_mua": alias})
    with pytest.raises(authority.TrackBV2DevelopmentTargetAuthorityError, match="alias or symlink"):
        authority.build_development_target_query_authority(
            dataset="subject_m", view="sua", outer_fold_id="subject_m_sua_external_target_20140307",
            target_session_id="sub-M_ses-CO-20140307",
        )


def test_h1_is_rejected_before_target_like_object_is_touched() -> None:
    class Poison:
        def __str__(self) -> str:
            raise AssertionError("excluded H1 must fail before inspecting target-like input")

    with pytest.raises(authority.base.TrackBV2ContractError, match="H1-excluded"):
        authority.build_development_target_query_authority(
            dataset="falcon_h1", view=None, outer_fold_id="bad", target_session_id=Poison(),
        )


@pytest.mark.parametrize(
    ("view", "outer_fold_id", "message"),
    (
        ("sua", "subject_m_pseudo_mua_external_target_20140307", "bind its view"),
        ("sua", "subject_m_sua_external_target_20140308", "bind its view"),
        ("sua", "subm-20140307", "bind its view"),
    ),
)
def test_subject_m_outer_fold_must_canonically_bind_view_and_target_date(
    view: str, outer_fold_id: str, message: str,
) -> None:
    with pytest.raises(authority.TrackBV2DevelopmentTargetAuthorityError, match=message):
        authority.build_development_target_query_authority(
            dataset="subject_m", view=view, outer_fold_id=outer_fold_id,
            target_session_id="sub-M_ses-CO-20140307",
        )


def test_cli_has_no_target_path_execution_score_or_gpu_option() -> None:
    script = REPO_ROOT / "cebra_exploration" / "scripts" / "run_track_b_v2_development_target_authority.py"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=REPO_ROOT, env=env,
        text=True, capture_output=True, check=True,
    )
    for forbidden in ("--target-data", "--target-path", "--execute", "--score", "--gpu"):
        assert forbidden not in result.stdout
    assert "--target-session-id" in result.stdout
    assert pointer.canonical_metric_pointer_body_path("rt", None).exists()
