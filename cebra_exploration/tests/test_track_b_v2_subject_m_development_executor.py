"""Focused no-target tests for the subject-M future executor/scorer boundary."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "cebra_exploration" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_contract as base  # noqa: E402
import track_b_v2_fixed_gpu_engineering as fixed_gpu  # noqa: E402
import track_b_v2_subject_m_development_executor as executor  # noqa: E402


CLI = REPO_ROOT / "cebra_exploration" / "scripts" / "run_track_b_v2_subject_m_development_executor.py"
_SUA_FOLD = "subject_m_sua_external_target_20140307"
_PMUA_FOLD = "subject_m_pseudo_mua_external_target_20140307"
_TARGET = "sub-M_ses-CO-20140307"


def _cost_payload() -> dict[str, object]:
    closure = fixed_gpu.route.snapshot_file_closure(
        fixed_gpu.implementation_paths(executor._FIXED_GPU_COST_RUNNER)
    )
    source_bindings = {
        label: {
            "path": str((fixed_gpu.AUTHORITY_ROOT / name).absolute()),
            "sha256": sha,
            "mode": "0444",
            "read_once_from_verified_fd": True,
        }
        for label, (name, sha, _schema) in fixed_gpu.AUTHORITY_FILES.items()
    }
    return {
        "schema": fixed_gpu.SCHEMA_COST,
        "status": fixed_gpu.STATUS_COST,
        "official": False,
        "fixed_final_geometry": fixed_gpu.FIXED_FINAL_GEOMETRY.as_dict(),
        "cost_smoke_geometry": fixed_gpu.COST_SMOKE_GEOMETRY.as_dict(),
        "fixed_normalized_ridge_lambda": fixed_gpu.FIXED_NORMALIZED_RIDGE_LAMBDA,
        "fixed_cosine_knn_k": fixed_gpu.FIXED_KNN_K,
        "seed": fixed_gpu.SEED,
        "scientific_metric_emitted": False,
        "winner_emitted": False,
        "selector_executed": False,
        "fixed_geometry_was_selected_from_source_data": False,
        "fixed_geometry_was_selected_from_target_data": False,
        "historical_selector_plan_executed": False,
        "historical_selector_plan_selected_geometry": False,
        "historical_selector_plan_authorizes_this_execution": False,
        "outer_target_discovered": False,
        "outer_target_path_resolved": False,
        "outer_target_opened": False,
        "formal_data_opened": False,
        "gpu_fit_call_count": 1,
        "preflight": {
            "schema": fixed_gpu.SCHEMA_PREFLIGHT,
            "status": fixed_gpu.STATUS_PREFLIGHT,
            "official": False,
            "implementation_closure_at_preflight": closure,
            "vendored_cuda_static_audit": fixed_gpu.vendored_cuda_static_audit(closure),
            "physical_cuda_visible_devices": fixed_gpu.CANONICAL_PHYSICAL_GPU_INDEX,
            "logical_sklearn_device": "cuda:0",
            "source_data_opened": False,
            "outer_target_discovered": False,
            "outer_target_opened": False,
            "formal_data_opened": False,
            "model_fit_called": False,
        },
        "source_authority_bindings": source_bindings,
        "source_authority_roles": {
            "selector_plan": (
                "historical_source_bundle_lineage_only__not_executed__not_selected__"
                "not_authorizing_fixed_canonical_gpu_cost"
            ),
        },
        "source_fold_boundary": {
            "held_source_session_id": fixed_gpu.HELD_SOURCE_ID,
            "peer_session_count": 26,
            "query_rows": 100,
            "query_neural_in_fit": False,
            "query_auxiliary_in_fit": False,
        },
        "gpu_fit_validation": {
            "fit_calls": [{
                "label": "gpu_joint_multisession_fit",
                "iterations": 250,
                "resolved_estimator_device": "cuda:0",
                "session_model_parameter_devices": ["cuda:0"] * 27,
            }],
            "query_neural_in_fit": False,
            "query_auxiliary_in_fit": False,
        },
        "cuda_identity": {
            "cuda_visible_devices": fixed_gpu.CANONICAL_PHYSICAL_GPU_INDEX,
            "logical_device": "cuda:0",
            "visible_device_count": 1,
            "torch_cuda_available": True,
            "device_uuid": "GPU-synthetic",
        },
        "cebra_backend_identity": {
            "actual_cebra": True,
            "cebra_version": fixed_gpu.route.VENDORED_CEBRA_VERSION,
            "cebra_commit": fixed_gpu.route.VENDORED_CEBRA_COMMIT,
            "requested_device": "cuda:0",
            "cpu_fallback_permitted": False,
        },
        "runtime": {
            "strict27_materialization_wall_clock_s": 1.0,
            "gpu_fit_plus_transform_wall_clock_s": 2.0,
            "total_wall_clock_s": 3.0,
            "peak_rss_kib": 1024,
            "python_executable": sys.executable,
        },
        "implementation_closure_at_launch": closure,
        "launch_closure_exact_equal_to_final_live": True,
    }


def _write_immutable_pair(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True)
    raw = base.canonical_json_bytes(payload)
    path.write_bytes(raw)
    os.chmod(path, 0o444)
    sidecar = path.with_name(f"{path.name}.sha256")
    sidecar.write_text(f"{hashlib.sha256(raw).hexdigest()}  {path.name}\n", encoding="ascii")
    os.chmod(sidecar, 0o444)


def test_real_subject_m_dry_plan_accepts_terminal_cost_but_remains_no_target() -> None:
    plan = executor.build_subject_m_development_executor_dry_plan(
        dataset="subject_m", view="sua", outer_fold_id=_SUA_FOLD, target_session_id=_TARGET,
    )
    assert plan["status"] == "GPU_COST_GATE_VALID__FUTURE_SUBJECT_M_EXECUTOR_AND_SCORER_STILL_NOT_IMPLEMENTED"
    assert plan["canonical_materializer_binding"]["target_asset_ledger_is_a2_v2_verified"] is True
    assert plan["fixed_gpu_cost_gate"]["canonical_body_sha256"] == (
        "ec7096a5e54e444fd6cdafa241aaa88a0720143e3c4e42e3565f022ee662c8e2"
    )
    assert plan["fixed_gpu_cost_gate"]["target_execution_permitted"] is False
    assert plan["fixed_gpu_cost_gate"]["target_data_opened"] is False
    for field in (
        "target_execution_permitted", "target_data_discovery_permitted", "target_data_opened",
        "target_query_opened", "formal_data_opened", "cebra_imported", "cebra_trained",
        "readout_fit_called", "score_emitted", "gpu_used", "official_execution_receipt_minted",
    ):
        assert plan[field] is False


def test_pmua_dry_plan_requires_same_record_exact_replay_and_cross_view_behavior_order() -> None:
    plan = executor.build_subject_m_development_executor_dry_plan(
        dataset="subject_m", view="pseudo_mua", outer_fold_id=_PMUA_FOLD, target_session_id=_TARGET,
    )
    cross = plan["exact_t4_target_byte_lineage"]["sua_pmua_cross_view_parity"]
    replay = cross["pseudo_mua_target_pooling_replay"]
    assert cross["same_target_record_required"] is True
    assert cross["same_ordered_target_behavior_bytes_required"] is True
    assert cross["same_ordered_prediction_endpoint_authority_required"] is True
    assert replay["input_sua_feature_sha256"] == "REQUIRED__EXACT_TARGET_RECORD_INPUT"
    assert replay["ordered_unit_ids_raw_bytes_sha256"] == "REQUIRED"
    assert replay["ordered_unit_electrode_ids_raw_bytes_sha256"] == "REQUIRED"
    assert replay["unit_count"] == "REQUIRED"
    assert replay["unique_electrode_channel_count"] == "REQUIRED"
    assert replay["replay_exact_equal_to_target_pmua_feature"] is True
    assert replay["caller_supplied_pooling_dict_without_same_record_replay_permitted"] is False
    assert plan["target_data_opened"] is False


def test_exact_offset10_contract_is_half_open_ten_bins_with_four_future_bins() -> None:
    contract = executor._exact_offset10_contract()
    assert contract["half_open_offsets_relative_to_prediction_endpoint"] == [-5, 5]
    assert contract["exact_per_endpoint_raw_receptive_field"] == "range(endpoint-5, endpoint+5)"
    assert contract["receptive_field_width_raw_bins"] == 10
    assert contract["previous_raw_bins"] == 5
    assert contract["strictly_future_raw_bins_after_endpoint"] == 4
    assert "five" not in json.dumps(contract).lower()


def test_topology_shares_one_legal_joint_encoder_bundle_and_splits_only_readouts() -> None:
    plan = executor.build_subject_m_development_executor_dry_plan(
        dataset="subject_m", view="sua", outer_fold_id=_SUA_FOLD, target_session_id=_TARGET,
    )
    per_seed = plan["immutable_future_receipt_topology"]["per_cebra_seed"]
    encoder = per_seed["encoder_bundle"]
    routes = per_seed["readout_routes"]
    assert per_seed["same_legal_target_serviceable_model_bundle_scores_all_readout_routes_and_decoders"] is True
    assert encoder["legal_multisession_target_serviceability_required"] is True
    assert encoder["never_transform_unfitted_unseen_target_session"] is True
    assert encoder["target_support_neural_enters_encoder_fit"] is True
    assert encoder["target_support_dense_velocity_enters_encoder_fit"] is True
    assert encoder["target_query_neural_or_labels_enter_fit"] is False
    source = routes["source_only_consumer_mechanism_alignment"]
    assert source["readout_fit_scope"] == "source_fit_only"
    assert source["encoder_fit_scope"] == "legal_joint_multisession_source_plus_target_support"
    assert source["target_support_dense_labels_in_readout_fit"] is False
    assert source["unfitted_target_transform_or_source_only_encoder_permitted"] is False
    for name in (
        "target_support_only_standard_cebra_accuracy",
        "source_plus_target_support_hybrid_sensitivity",
    ):
        assert routes[name]["encoder_fit_scope"] == "legal_joint_multisession_source_plus_target_support"
        assert routes[name]["independent_readout_fit_required"] is True


def test_scope_rejects_h1_and_rt_before_target_id_coercion() -> None:
    class Poison:
        def __str__(self) -> str:
            raise AssertionError("scope must reject before target coercion")

    with pytest.raises(base.TrackBV2ContractError, match="H1-excluded"):
        executor.build_subject_m_development_executor_dry_plan(
            dataset="falcon_h1", view=None, outer_fold_id="bad", target_session_id=Poison(),
        )
    with pytest.raises(executor.TrackBV2SubjectMDevelopmentExecutorError, match="rejects RT"):
        executor.build_subject_m_development_executor_dry_plan(
            dataset="rt", view=None, outer_fold_id="rt_outer_fold_00", target_session_id=Poison(),
        )


def test_cost_gate_accepts_only_exact_immutable_pair_but_never_authorizes_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = tmp_path / "canonical" / "receipt.json"
    _write_immutable_pair(body, _cost_payload())
    monkeypatch.setattr(fixed_gpu, "CANONICAL_COST_OUTPUT", body)
    gate = executor.inspect_fixed_d8it250_gpu_cost_receipt()
    assert gate["status"] == "FIXED_D8IT250_GPU_COST_RECEIPT_VALID__REQUIRED_BUT_NOT_SUFFICIENT_FOR_TARGET_EXECUTION"
    assert gate["fresh_body_and_sidecar"] is True
    assert len(gate["live_implementation_closure_sha256"]) == 64
    assert gate["device_uuid"] == "GPU-synthetic"
    assert gate["target_execution_permitted"] is False
    plan = executor.build_subject_m_development_executor_dry_plan(
        dataset="subject_m", view="sua", outer_fold_id=_SUA_FOLD, target_session_id=_TARGET,
    )
    assert plan["status"] == "GPU_COST_GATE_VALID__FUTURE_SUBJECT_M_EXECUTOR_AND_SCORER_STILL_NOT_IMPLEMENTED"
    assert plan["target_execution_permitted"] is False
    # A sidecar mismatch must fail closed, even if the immutable body remains valid.
    sidecar = body.with_name(f"{body.name}.sha256")
    os.chmod(sidecar, 0o644)
    sidecar.write_text("0" * 64 + "  receipt.json\n", encoding="ascii")
    os.chmod(sidecar, 0o444)
    assert executor.inspect_fixed_d8it250_gpu_cost_receipt()["status"].startswith("NO_GO__")


@pytest.mark.parametrize("poison", ("closure", "source", "backend", "runtime", "canonical_json"))
def test_cost_gate_recomputes_live_closure_authorities_backend_and_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, poison: str,
) -> None:
    body = tmp_path / poison / "receipt.json"
    payload = _cost_payload()
    if poison == "closure":
        payload["implementation_closure_at_launch"]["gpu_engineering_core"]["sha256"] = "0" * 64
    elif poison == "source":
        payload["source_authority_bindings"]["coverage"]["sha256"] = "0" * 64
    elif poison == "backend":
        payload["cebra_backend_identity"]["cebra_commit"] = "bad"
    elif poison == "runtime":
        payload["runtime"]["gpu_fit_plus_transform_wall_clock_s"] = 0.0
    _write_immutable_pair(body, payload)
    if poison == "canonical_json":
        os.chmod(body, 0o644)
        parsed = json.loads(body.read_text())
        raw = (json.dumps(parsed, sort_keys=False) + "\n").encode()
        body.write_bytes(raw)
        os.chmod(body, 0o444)
        sidecar = body.with_name(f"{body.name}.sha256")
        os.chmod(sidecar, 0o644)
        sidecar.write_text(f"{hashlib.sha256(raw).hexdigest()}  {body.name}\n", encoding="ascii")
        os.chmod(sidecar, 0o444)
    monkeypatch.setattr(fixed_gpu, "CANONICAL_COST_OUTPUT", body)
    gate = executor.inspect_fixed_d8it250_gpu_cost_receipt()
    assert gate["status"].startswith("NO_GO__")


def test_cost_gate_rejects_sidecar_symlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = tmp_path / "canonical" / "receipt.json"
    _write_immutable_pair(body, _cost_payload())
    sidecar = body.with_name(f"{body.name}.sha256")
    sidecar.unlink()
    replacement = tmp_path / "replacement.sha256"
    replacement.write_text(f"{hashlib.sha256(body.read_bytes()).hexdigest()}  receipt.json\n", encoding="ascii")
    sidecar.symlink_to(replacement)
    monkeypatch.setattr(fixed_gpu, "CANONICAL_COST_OUTPUT", body)
    gate = executor.inspect_fixed_d8it250_gpu_cost_receipt()
    assert gate["status"].startswith("NO_GO__")
    assert "symlinks" in gate["reason"]


def test_private_snapshot_parser_fd_survives_source_path_swap_and_rejects_pathname_proof(
    tmp_path: Path,
) -> None:
    source = tmp_path / "target.nwb"
    original = b"synthetic original target bytes"
    poison = b"synthetic replacement target bytes"
    source.write_bytes(original)
    snapshot = tmp_path / "private" / "target.snapshot"
    snapshot.parent.mkdir()
    expected = hashlib.sha256(original).hexdigest()
    with executor.open_verified_target_asset_private_snapshot(
        source_path=source, expected_sha256=expected, expected_bytes=len(original), snapshot_path=snapshot,
    ) as proof:
        contract = proof.as_contract_dict()
        assert contract["pathname_pre_and_post_hash_alone_is_sufficient"] is False
        assert contract["ordinary_source_or_snapshot_pathname_reopen_permitted"] is False
        replacement = tmp_path / "replacement.nwb"
        replacement.write_bytes(poison)
        os.replace(replacement, source)
        # Restore byte-identical contents at a different inode before a
        # hypothetical pathname-only post-check: this is the ABA case that
        # pre/post pathname SHA comparisons cannot exclude during parsing.
        restore = tmp_path / "restore-original.nwb"
        restore.write_bytes(original)
        os.replace(restore, source)
        # The parser consumes the held snapshot descriptor, not the swapped
        # target pathname.  /proc/self/fd/<n> resolves to this same inode.
        assert Path(proof.parser_fd_path).read_bytes() == original
        assert source.read_bytes() == original
        assert source.stat().st_ino != proof.source_identity[1]
        assert snapshot.stat().st_mode & 0o777 == 0o444
    assert snapshot.read_bytes() == original


def test_private_snapshot_rejects_source_symlink_and_snapshot_collision(tmp_path: Path) -> None:
    raw = b"synthetic bytes"
    source = tmp_path / "source.nwb"
    source.write_bytes(raw)
    expected = hashlib.sha256(raw).hexdigest()
    snapshot = tmp_path / "snapshot.nwb"
    snapshot.write_bytes(b"collision")
    with pytest.raises(executor.TrackBV2SubjectMDevelopmentExecutorError, match="fresh"):
        with executor.open_verified_target_asset_private_snapshot(
            source_path=source, expected_sha256=expected, expected_bytes=len(raw), snapshot_path=snapshot,
        ):
            raise AssertionError("must not reach parser")
    snapshot.unlink()
    link = tmp_path / "source-link.nwb"
    link.symlink_to(source)
    with pytest.raises(executor.TrackBV2SubjectMDevelopmentExecutorError, match="without following symlinks"):
        with executor.open_verified_target_asset_private_snapshot(
            source_path=link, expected_sha256=expected, expected_bytes=len(raw), snapshot_path=snapshot,
        ):
            raise AssertionError("must not reach parser")


def test_cli_exposes_no_target_path_execution_score_gpu_or_cebra_flags() -> None:
    completed = subprocess.run(
        [sys.executable, str(CLI), "--help"], check=True, capture_output=True, text=True,
    )
    text = completed.stdout.lower()
    for forbidden in ("--target-path", "--execute", "--score", "--gpu", "--cebra"):
        assert forbidden not in text
    assert "--target-session-id" in text
