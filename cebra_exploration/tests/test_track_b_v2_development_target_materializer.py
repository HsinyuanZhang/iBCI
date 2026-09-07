"""Focused no-target and synthetic tests for the Track-B v2 materializer boundary."""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "cebra_exploration" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_development_target_authority as authority  # noqa: E402
import track_b_v2_development_target_materializer as materializer  # noqa: E402


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _subject_lineage(*, rows: int = 2) -> dict[str, object]:
    return {
        "t4_reference_query_identity_sha256": _sha("query-identity"),
        "metric_implementation_authority_sha256": _sha("torchmetrics-151"),
        "t4_runtime_receipt_sha256": _sha("runtime"),
        "v9_commit_receipt_sha256": _sha("commit"),
        "v9_runtime_base_input_trace_sha256": _sha("base-input"),
        "v9_runtime_query_behavior_trace_sha256": _sha("query-behavior"),
        "t4_predictions_targets_npz_sha256": _sha("npz"),
        "t4_reference_query_row_count": rows,
    }


def _subject_materialize_kwargs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "dataset": "subject_m",
        "view": "sua",
        "outer_fold_id": "subject_m_sua_external_target_20140307",
        "target_session_id": "sub-M_ses-CO-20140307",
        # One raw bin for every chronological trial is enough for the pure
        # boundary test: 0..49 support, then strict post-M query rows.
        "raw_trial_ordinal_by_bin": tuple(range(120)),
        "valid_window_start_indices": (50, 60),
        # Both RFs are inside the post-M query.  Offset(5,5) expands with the
        # half-open offsets [-5,+5), so there are ten bins total and four bins
        # strictly after each prediction-target endpoint.
        "full_receptive_field_raw_indices": (tuple(range(94, 104)), tuple(range(104, 114))),
        "t4_target_float32_bytes": b"\x00" * 16,
        "cebra_target_float32_bytes": b"\x00" * 16,
        "dense_support_velocity_float32_bytes": b"\x00" * (50 * 2 * 4),
        "dense_support_unique_row_count": 1,
        "target_feature_sha256": _sha("target-sua-feature"),
        "target_reference_lineage": _subject_lineage(),
        "vendored_model_alignment_authority_sha256": _sha("offset10"),
        "synthetic_fixture": True,
    }
    values.update(overrides)
    return values


def test_subject_m_dry_plan_rebuilds_canonical_authority_then_exact_a2_ledger_row() -> None:
    plan = materializer.build_development_target_materializer_dry_plan(
        dataset="subject_m", view="sua",
        outer_fold_id="subject_m_sua_external_target_20140307",
        target_session_id="sub-M_ses-CO-20140307",
    )
    canonical = authority.build_development_target_query_authority(
        dataset="subject_m", view="sua",
        outer_fold_id="subject_m_sua_external_target_20140307",
        target_session_id="sub-M_ses-CO-20140307",
    )
    assert plan["canonical_development_authority"]["development_target_query_authority_sha256"] == (
        canonical["development_target_query_authority_sha256"]
    )
    asset_gate = plan["target_asset_ledger_gate"]
    assert asset_gate["schema"] == materializer.SUBJECT_M_TARGET_ASSET_LEDGER_SCHEMA
    assert asset_gate["authorities"]["a2_official_preflight"]["body_sha256"] == materializer._A2_PREFLIGHT_SHA256
    assert asset_gate["authorities"]["subm_schema_ledger"]["body_sha256"] == materializer._SUBM_SCHEMA_LEDGER_SHA256
    assert asset_gate["authorities"]["subm_scope_manifest"]["body_sha256"] == materializer._SUBM_SCOPE_MANIFEST_SHA256
    assert asset_gate["target_asset"] == {
        "session_id": "sub-M_ses-CO-20140307",
        "asset_id": "a72cae17-6e18-4c36-bdaa-f0f31d557888",
        "frozen_relative_path": "sub-M/sub-M_ses-CO-20140307_behavior+ecephys.nwb",
        "a2_official_local_nwb_path": str(
            REPO_ROOT / "sua_exploration" / "data" / "dandi_000688" / "sub-M" /
            "sub-M_ses-CO-20140307_behavior+ecephys.nwb"
        ),
        "expected_sha256": "2f109d6daed0ad2c3dba12742d62b7be1f385117c1b0b018205093a873c29927",
        "expected_bytes": 74037256,
    }
    assert plan["target_data_opened"] is False
    assert plan["cebra_imported"] is False
    assert plan["score_emitted"] is False
    assert plan["readout_roles"]["fixed_geometry"]["selection_performed"] is False
    assert asset_gate["pathname_pre_and_post_hash_alone_is_sufficient"] is False
    assert "continuously_held_verified_inode" in asset_gate["verification_required_during_nwb_parser"]


def test_rt_dry_plan_binds_exact_root_published_local_byte_ledger_row() -> None:
    plan = materializer.build_development_target_materializer_dry_plan(
        dataset="rt", view=None, outer_fold_id="rt_outer_fold_00",
        target_session_id="ses-RT-20131009",
    )
    gate = plan["target_asset_ledger_gate"]
    assert plan["status"] == "CANONICAL_DEVELOPMENT_AUTHORITY_AND_RT_ASSET_LEDGER_VERIFIED__NO_TARGET_ARRAYS"
    assert gate["schema"] == materializer.RT_TARGET_ASSET_LEDGER_SCHEMA
    assert gate["status"] == "ROOT_PUBLISHED_RT_LOCAL_ASSET_AUTHORITY_VALIDATED__NO_TARGET_OPEN"
    assert gate["authority"]["body_sha256"] == materializer._RT_LOCAL_ASSET_AUTHORITY_SHA256
    assert gate["target_asset"] == {
        "session_id": "ses-RT-20131009",
        "held_target_outer_fold_id": "rt_outer_fold_00",
        "held_target_outer_fold_index": 0,
        "canonical_local_nwb_path": str(
            REPO_ROOT / "sua_exploration" / "data" / "dandi_000688" / "sub-C" /
            "sub-C_ses-RT-20131009_behavior+ecephys.nwb"
        ),
        "expected_sha256": "5bd05b43ac590c70a3edb4e6fc4d0614dfe3e14d97ab5db54137dce5d6e0b11c",
        "expected_bytes": 73479612,
    }
    assert gate["directory_discovery_or_glob_permitted"] is False
    assert gate["synthetic_or_derived_directory_roster_permitted"] is False
    assert gate["pathname_pre_and_post_hash_alone_is_sufficient"] is False
    assert gate["caller_supplied_asset_ledger_mapping_permitted"] is False
    assert gate["required_live_parser_entrypoint"] == (
        "track_b_v2_rt_local_asset_authority.open_canonical_verified_asset_after_authority"
    )
    assert plan["target_data_opened"] is False


def test_rt_authority_tamper_fails_before_target_path_or_array_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = materializer.rt_asset_authority.load_published_authority_before_target_access
    payload, body_sha = original()
    altered = copy.deepcopy(payload)
    altered["asset_rows"][0]["held_target_outer_fold_id"] = "rt_outer_fold_14"
    monkeypatch.setattr(
        materializer.rt_asset_authority,
        "load_published_authority_before_target_access",
        lambda: (altered, body_sha),
    )
    with pytest.raises(materializer.TrackBV2DevelopmentTargetMaterializerError, match="held-target fold"):
        materializer.build_development_target_materializer_dry_plan(
            dataset="rt", view=None, outer_fold_id="rt_outer_fold_00",
            target_session_id="ses-RT-20131009",
        )


def test_rt_materializer_exports_no_caller_mapping_or_pathname_reopen_helper() -> None:
    assert not hasattr(materializer, "verify_rt_target_asset_before_loader")
    assert not hasattr(materializer, "verify_rt_target_asset_after_loader")
    assert hasattr(materializer.rt_asset_authority, "open_canonical_verified_asset_after_authority")


def test_scope_rejects_h1_before_coercing_target_like_object() -> None:
    class Poison:
        def __str__(self) -> str:
            raise AssertionError("H1 must fail before target coercion")

    with pytest.raises(materializer.base.TrackBV2ContractError, match="H1-excluded"):
        materializer.build_development_target_materializer_dry_plan(
            dataset="falcon_h1", view=None, outer_fold_id="bad", target_session_id=Poison(),
        )


def test_subject_m_ledger_tamper_fails_before_any_target_path_or_array_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = materializer._read_legacy_sidecarless

    def tampered(path: Path, **kwargs: object):  # type: ignore[no-untyped-def]
        payload, verified = original(path, **kwargs)  # type: ignore[arg-type]
        if Path(path) != materializer._SUBM_SCHEMA_LEDGER:
            return payload, verified
        altered = copy.deepcopy(payload)
        row = next(item for item in altered["verified_downloads"] if item["asset_id"] == "a72cae17-6e18-4c36-bdaa-f0f31d557888")
        row["sha256"] = "0" * 64
        return altered, verified

    monkeypatch.setattr(materializer, "_read_legacy_sidecarless", tampered)
    with pytest.raises(materializer.TrackBV2DevelopmentTargetMaterializerError, match="verified-download path/size/SHA"):
        materializer.build_development_target_materializer_dry_plan(
            dataset="subject_m", view="sua",
            outer_fold_id="subject_m_sua_external_target_20140307",
            target_session_id="sub-M_ses-CO-20140307",
        )


def test_canonical_pointer_source_authority_rebuild_precedes_target_asset_id_coercion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise materializer.development_authority.TrackBV2DevelopmentTargetAuthorityError("pointer/source deliberately unavailable")

    class Poison:
        def __str__(self) -> str:
            raise AssertionError("target ID coercion must follow canonical authority rebuild")

    monkeypatch.setattr(materializer.development_authority, "build_development_target_query_authority", reject)
    with pytest.raises(materializer.TrackBV2DevelopmentTargetMaterializerError, match="pointer/source deliberately unavailable"):
        materializer.build_development_target_materializer_dry_plan(
            dataset="subject_m", view="sua", outer_fold_id="subject_m_sua_external_target_20140307",
            target_session_id=Poison(),
        )


def test_synthetic_materialization_proves_continuous_m50_post_m_endpoints_bytes_and_offset10_rf() -> None:
    proof = materializer.materialize_synthetic_target_query_proof(**_subject_materialize_kwargs())
    assert proof["status"].startswith("SYNTHETIC_IN_MEMORY")
    support = proof["target_support"]
    assert support["support_trial_budget"] == 50
    assert support["support_raw_coverage"]["first_index"] == 0
    assert support["support_raw_coverage"]["last_index"] == 49
    assert support["dense_label_row_count"] == 50
    assert support["dense_label_scalar_count"] == 100
    assert support["dense_label_unique_row_count"] == 1
    query = proof["target_query"]
    rf = query["receptive_field_and_target_byte_proof"]["vendored_model_receptive_field_alignment"]
    assert query["ordered_prediction_target_semantics"] == "valid_window_start_plus_49"
    assert rf["checked_prediction_endpoint_count"] == 2
    assert rf["receptive_field_containment_violation_count"] == 0
    assert rf["future_raw_bins_relative_to_prediction_target"]["maximum_future_raw_bins"] == 4
    assert rf["receptive_field_width_min"] == rf["receptive_field_width_max"] == 10
    assert query["t4_and_cebra_target_bytes_exactly_equal"] is True
    assert query["subject_m_sua_and_pseudo_mua_must_match_this_authority"] is True
    assert proof["readout_roles"]["routes"]["target_support_only_standard_cebra_accuracy"]["readout_fit_scope"] == (
        "target_support_only"
    )
    assert proof["readout_roles"]["routes"]["source_plus_target_support_hybrid_sensitivity"]["scientific_role"].startswith(
        "mandatory_sensitivity"
    )
    assert proof["target_asset_opened"] is False
    assert proof["cebra_imported"] is False
    assert proof["score_emitted"] is False


def test_synthetic_materialization_rejects_target_execution_start_endpoint_and_rf_leakage() -> None:
    with pytest.raises(materializer.TrackBV2DevelopmentTargetMaterializerError, match="not authorized outside synthetic"):
        materializer.materialize_synthetic_target_query_proof(**_subject_materialize_kwargs(synthetic_fixture=False))
    # Deliberately pass a non-vendored receptive field.  The successor rejects
    # it before the generic historical scaffold can treat an arbitrary
    # query-contained field as an offset10 field.
    with pytest.raises(materializer.TrackBV2DevelopmentTargetMaterializerError, match="exact half-open"):
        materializer.materialize_synthetic_target_query_proof(**_subject_materialize_kwargs(
            full_receptive_field_raw_indices=(tuple(range(45, 100)), tuple(range(104, 115))),
        ))
    with pytest.raises(materializer.TrackBV2DevelopmentTargetMaterializerError, match="bytes differ"):
        materializer.materialize_synthetic_target_query_proof(**_subject_materialize_kwargs(
            cebra_target_float32_bytes=b"\x01" * 16,
        ))
    with pytest.raises(materializer.TrackBV2DevelopmentTargetMaterializerError, match="unique-row count"):
        materializer.materialize_synthetic_target_query_proof(**_subject_materialize_kwargs(
            dense_support_unique_row_count=50,
        ))
    with pytest.raises(materializer.TrackBV2DevelopmentTargetMaterializerError, match="may not accept a real target asset"):
        materializer.materialize_synthetic_target_query_proof(**_subject_materialize_kwargs(
            target_asset_preopen_proof=object(),
        ))
    with pytest.raises(materializer.TrackBV2DevelopmentTargetMaterializerError, match="exact half-open"):
        materializer.materialize_synthetic_target_query_proof(**_subject_materialize_kwargs(
            full_receptive_field_raw_indices=(tuple(range(94, 105)), tuple(range(104, 115))),
        ))


def test_pmua_requires_actual_target_pooling_provenance_and_preserves_shared_target_behavior_order() -> None:
    values = _subject_materialize_kwargs(
        view="pseudo_mua", outer_fold_id="subject_m_pseudo_mua_external_target_20140307",
        target_feature_sha256=_sha("target-pmua-feature"),
    )
    with pytest.raises(materializer.TrackBV2DevelopmentTargetMaterializerError, match="requires actual pooling"):
        materializer.materialize_synthetic_target_query_proof(**values)
    values["pooling_provenance"] = {
        "method": "electrode_ids_from_units_then_pool_spikes_by_electrode",
        "output_pseudo_mua_feature_sha256": _sha("target-pmua-feature"),
        "channel_order": "np_unique_ascending_electrode_id",
        "output_dtype": "float32",
    }
    proof = materializer.materialize_synthetic_target_query_proof(**values)
    assert proof["target_query"]["ordered_target_float32_raw_bytes_sha256"] == hashlib.sha256(b"\x00" * 16).hexdigest()
    sua_proof = materializer.materialize_synthetic_target_query_proof(**_subject_materialize_kwargs())
    assert proof["target_query"]["subject_m_cross_view_behavior_order"]["cross_view_behavior_order_authority_sha256"] == (
        sua_proof["target_query"]["subject_m_cross_view_behavior_order"]["cross_view_behavior_order_authority_sha256"]
    )
    assert proof["target_neural_feature"]["pseudo_mua_pooling_provenance"]["output_pseudo_mua_feature_sha256"] == _sha(
        "target-pmua-feature"
    )


def test_same_fd_target_byte_reader_rejects_symlink_and_after_loader_inode_swap(tmp_path: Path) -> None:
    target = tmp_path / "target.nwb"
    target.write_bytes(b"immutable synthetic bytes")
    expected_sha = hashlib.sha256(target.read_bytes()).hexdigest()
    verified = materializer._stream_hash_same_fd(
        target, label="synthetic target", expected_sha256=expected_sha, expected_bytes=target.stat().st_size,
    )
    before = materializer.VerifiedTargetAssetBeforeOpen(
        session_id="sub-M_ses-CO-20990101", path=verified.path, sha256=verified.sha256,
        byte_count=verified.byte_count, identity=verified.identity,
    )
    replacement = tmp_path / "replacement.nwb"
    replacement.write_bytes(target.read_bytes())
    os.replace(replacement, target)
    with pytest.raises(materializer.TrackBV2DevelopmentTargetMaterializerError, match="inode/metadata changed"):
        materializer.verify_subject_m_target_asset_after_loader(before=before)
    alias = tmp_path / "alias.nwb"
    try:
        alias.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("platform cannot create symlink adversarial fixture")
    with pytest.raises(materializer.TrackBV2DevelopmentTargetMaterializerError, match="without following symlinks"):
        materializer._stream_hash_same_fd(
            alias, label="symlink synthetic target", expected_sha256=expected_sha, expected_bytes=target.stat().st_size,
        )


def test_post_loader_reopen_is_explicitly_not_a_parser_consumption_proof(tmp_path: Path) -> None:
    target = tmp_path / "target.nwb"
    target.write_bytes(b"stable synthetic bytes")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    verified = materializer._stream_hash_same_fd(
        target, label="stable synthetic target", expected_sha256=digest,
        expected_bytes=target.stat().st_size,
    )
    before = materializer.VerifiedTargetAssetBeforeOpen(
        session_id="sub-M_ses-CO-20990101", path=verified.path,
        sha256=verified.sha256, byte_count=verified.byte_count, identity=verified.identity,
    )
    after = materializer.verify_subject_m_target_asset_after_loader(before=before)
    assert after["same_inode_as_pre_loader"] is True
    assert after["pathname_pre_and_post_hash_alone_is_sufficient"] is False
    assert after["same_held_fd_or_private_snapshot_consumption_proof_still_required"] is True


def test_cli_is_dry_only_and_has_no_target_path_execute_score_or_gpu_option() -> None:
    script = REPO_ROOT / "cebra_exploration" / "scripts" / "run_track_b_v2_development_target_materializer.py"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=REPO_ROOT, env=env,
        text=True, capture_output=True, check=True,
    )
    for forbidden in ("--target-path", "--target-data", "--execute", "--score", "--gpu"):
        assert forbidden not in result.stdout
    assert "--target-session-id" in result.stdout
