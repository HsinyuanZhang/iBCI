"""Synthetic/fail-closed tests for the H1-excluded Track-B v2 contract.

No test opens an NWB, imports CEBRA, loads a checkpoint, or uses CUDA.
"""
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

import track_b_v2_contract as core  # noqa: E402


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _candidate() -> core.CandidateGeometry:
    return core.CandidateGeometry(output_dimension=8, source_iterations=1000, normalized_lambda=1.0e-2)


def _control_measurements(spec: core.ExactGeometryControlSpec) -> list[core.ControlMeasurement]:
    output: list[core.ControlMeasurement] = []
    for seed in spec.seeds:
        for arm in core.POSITIVE_ARMS:
            output.append(
                core.ControlMeasurement(
                    arm=arm,
                    seed=seed,
                    geometry=spec.scoring_geometry,
                    index_manifest_sha256=_sha("control-index"),
                    source_query_r2_by_decoder={"linear_ridge": 0.90, "knn_cosine_k3": 0.90},
                    target_query_r2_by_decoder={"linear_ridge": 0.80, "knn_cosine_k3": 0.80},
                    target_query_neural_in_fit=False,
                    target_query_labels_in_fit=False,
                    target_support_dense_labels_used_for_cebra_encoder_fit=True,
                    target_support_label_scalar_count=100,
                    target_support_label_unique_rows=50,
                    t4_sparse_reference_label_event_count=50,
                    t4_sparse_reference_label_row_count=50,
                    t4_sparse_reference_label_scalar_count=50,
                    t4_sparse_reference_label_semantics="one_trial_direction_angle_annotation__cos_sin_is_derived_not_two_annotations",
                    knn_geometry=spec.knn_scoring_geometry,
                )
            )
        output.append(
            core.ControlMeasurement(
                arm=core.NEGATIVE_ARM,
                seed=seed,
                geometry=spec.scoring_geometry,
                index_manifest_sha256=_sha("control-index"),
                source_query_r2_by_decoder={"linear_ridge": 0.10, "knn_cosine_k3": 0.10},
                target_query_r2_by_decoder={"linear_ridge": -0.10, "knn_cosine_k3": -0.10},
                target_query_neural_in_fit=False,
                target_query_labels_in_fit=False,
                target_support_dense_labels_used_for_cebra_encoder_fit=True,
                target_support_label_scalar_count=100,
                target_support_label_unique_rows=50,
                t4_sparse_reference_label_event_count=50,
                t4_sparse_reference_label_row_count=50,
                t4_sparse_reference_label_scalar_count=50,
                t4_sparse_reference_label_semantics="one_trial_direction_angle_annotation__cos_sin_is_derived_not_two_annotations",
                knn_geometry=spec.knn_scoring_geometry,
            )
        )
    return output


def test_scope_is_strictly_h1_and_m2_excluded() -> None:
    assert core.validate_scope("subject_m", "sua") == ("subject_m", "sua")
    assert core.validate_scope("subject_m", "pseudo_mua") == ("subject_m", "pseudo_mua")
    assert core.validate_scope("rt") == ("rt", None)
    for forbidden in ("falcon_h1", "falcon_m2", "h1", "rt "):
        with pytest.raises(core.TrackBV2ContractError, match="permits only"):
            core.validate_scope(forbidden)
    with pytest.raises(core.TrackBV2ContractError, match="requires view"):
        core.validate_scope("subject_m")
    with pytest.raises(core.TrackBV2ContractError, match="do not provide"):
        core.validate_scope("rt", "sua")


def test_support_query_contract_is_exact_budget_source_normalized_and_query_only() -> None:
    contract = core.build_support_query_contract(
        dataset="subject_m",
        view="sua",
        outer_fold_id="subject_m_sua_loso_target_20140307",
        target_session_id="sub-M_ses-CO-20140307",
        source_session_ids=("sub-C_ses-CO-20131003", "sub-C_ses-CO-20131022"),
    )
    payload = contract.as_dict()
    assert payload["target_support_budget_trials"] == 50
    assert payload["target_support"]["dense_labels_in_cebra_encoder_fit"] is True
    assert payload["target_support"]["readout_fit_policy"] == "frozen_by_track_b_v2_multisession_provenance_plan"
    assert payload["target_query"] == {
        "neural_in_fit": False,
        "labels_in_fit": False,
        "scored": True,
        "required_disjoint_from_support": True,
    }
    assert payload["neural_input_preprocessor"] == {
        "fit_scope": "outer_source_sessions_only",
        "target_support_neural_in_fit": False,
        "target_query_neural_in_fit": False,
        "behavior_auxiliary_scaler": "SEPARATE_LIVE_RECEIPT_REQUIRED",
        "readout_embedding_standardizer": "SEPARATE_LIVE_RECEIPT_REQUIRED",
    }
    assert payload["source_selector"]["may_read_target"] is False
    with pytest.raises(core.TrackBV2ContractError, match="target session cannot"):
        core.build_support_query_contract(
            dataset="rt",
            view=None,
            outer_fold_id="rt_loso_x",
            target_session_id="target",
            source_session_ids=("source_a", "target"),
        )
    with pytest.raises(core.TrackBV2ContractError, match="target query neural"):
        core.SupportQueryAdapterContract(
            dataset="rt",
            view=None,
            outer_fold_id="rt_loso_x",
            target_session_id="target",
            source_session_ids=("source_a", "source_b"),
            target_support_budget_trials=24,
            target_query_neural_in_fit=True,
        )


def test_source_only_selector_requires_complete_grid_and_has_deterministic_tie_break() -> None:
    selector = core.SourceOnlySelectorSpec(
        d_grid=(3, 8),
        iteration_grid=(250,),
        lambda_grid=(1.0e-2, 1.0e-1),
    )
    candidates = selector.candidates()
    results = [
        core.SourceOnlySelectionResult(
            candidate=candidate,
            linear_ridge_source_inner_query_r2=0.8 if candidate.output_dimension == 8 else 0.7,
            inner_source_fold_ids=("source_inner_0", "source_inner_1"),
        )
        for candidate in candidates
    ]
    selected = core.select_source_only_candidate(selector=selector, results=results)
    assert selected["target_data_used"] is False
    assert selected["report_decoders"] == ["linear_ridge", "knn_cosine_k3"]
    assert selected["selected"]["candidate"]["output_dimension"] == 8
    assert "target_query_r2" not in json.dumps(selected)
    knn_results = [
        core.KnnSourceOnlySelectionResult(
            candidate=candidate,
            knn_source_inner_query_r2=0.85 if candidate.output_dimension == 3 else 0.75,
            inner_source_fold_ids=("source_inner_0", "source_inner_1"),
        )
        for candidate in selector.knn_candidates()
    ]
    dual = core.select_source_only_geometries(
        selector=selector, linear_results=results, knn_results=knn_results,
    )
    assert dual["linear_ridge"]["selected"]["candidate"]["normalized_lambda"] == 1.0e-2
    assert dual["knn_cosine_k3"]["selected"]["candidate"]["normalized_lambda"] == "NOT_APPLICABLE"
    assert dual["knn_cosine_k3"]["selected"]["candidate"]["output_dimension"] == 3
    with pytest.raises(core.TrackBV2ContractError, match="one result for every"):
        core.select_source_only_candidate(selector=selector, results=results[:-1])


def test_exact_geometry_control_requires_all_arms_seeds_query_exclusion_and_selected_geometry() -> None:
    spec = core.ExactGeometryControlSpec(
        scoring_geometry=_candidate(),
        knn_scoring_geometry=core.KnnCandidateGeometry(output_dimension=3, source_iterations=250),
        seeds=(7, 11),
    )
    measurements = _control_measurements(spec)
    gate = core.assert_exact_geometry_controls(spec=spec, measurements=measurements)
    assert gate["status"] == "PASSED"
    assert gate["measurement_count"] == 6
    mismatch = list(measurements)
    mismatch[0] = core.ControlMeasurement(
        arm=mismatch[0].arm,
        seed=mismatch[0].seed,
        geometry=core.CandidateGeometry(output_dimension=3, source_iterations=1000, normalized_lambda=1.0e-2),
        index_manifest_sha256=mismatch[0].index_manifest_sha256,
        source_query_r2_by_decoder=mismatch[0].source_query_r2_by_decoder,
        target_query_r2_by_decoder=mismatch[0].target_query_r2_by_decoder,
        target_query_neural_in_fit=False,
        target_query_labels_in_fit=False,
        target_support_dense_labels_used_for_cebra_encoder_fit=True,
        target_support_label_scalar_count=100,
        target_support_label_unique_rows=50,
        t4_sparse_reference_label_event_count=50,
        t4_sparse_reference_label_row_count=50,
        t4_sparse_reference_label_scalar_count=50,
        t4_sparse_reference_label_semantics="one_trial_direction_angle_annotation__cos_sin_is_derived_not_two_annotations",
        knn_geometry=spec.knn_scoring_geometry,
    )
    with pytest.raises(core.TrackBV2ContractError, match="geometry differs"):
        core.assert_exact_geometry_controls(spec=spec, measurements=mismatch)
    with pytest.raises(core.TrackBV2ContractError, match="target query labels"):
        core.ControlMeasurement(
            arm=core.POSITIVE_ARMS[0],
            seed=7,
            geometry=_candidate(),
            index_manifest_sha256=_sha("control-index"),
            source_query_r2_by_decoder={"linear_ridge": 0.8, "knn_cosine_k3": 0.8},
            target_query_r2_by_decoder={"linear_ridge": 0.8, "knn_cosine_k3": 0.8},
            target_query_neural_in_fit=False,
            target_query_labels_in_fit=True,
            target_support_dense_labels_used_for_cebra_encoder_fit=True,
            target_support_label_scalar_count=100,
            target_support_label_unique_rows=50,
            t4_sparse_reference_label_event_count=50,
            t4_sparse_reference_label_row_count=50,
            t4_sparse_reference_label_scalar_count=50,
            t4_sparse_reference_label_semantics="one_trial_direction_angle_annotation__cos_sin_is_derived_not_two_annotations",
            knn_geometry=core.KnnCandidateGeometry(output_dimension=3, source_iterations=250),
        )


def test_unified_unseen_serviceability_receipt_has_no_accuracy_and_no_solver_call() -> None:
    receipt = core.build_unified_unseen_serviceability_receipt(
        dataset="rt",
        view=None,
        source_session_ids=("source_b", "source_a"),
        source_unit_counts=(31, 24),
        target_session_id="held_target",
        target_unit_count=37,
    )
    assert receipt["status"] == "UNIFIED_UNSEEN_SESSION_UNSERVABLE"
    assert receipt["score_status"] == "NO_ACCURACY_EMITTED"
    assert receipt["cebra_solver_called"] is False
    assert receipt["real_data_opened"] is False
    assert receipt["gpu_used"] is False
    assert receipt["unified_input_width_from_source_sessions"] == 55
    assert receipt["source_session_ids"] == ["source_a", "source_b"]
    assert receipt["source_unit_counts"] == [24, 31]
    assert "r2" not in json.dumps(receipt).lower()
    with pytest.raises(core.TrackBV2ContractError, match="unseen"):
        core.build_unified_unseen_serviceability_receipt(
            dataset="rt",
            view=None,
            source_session_ids=("source_a", "held_target"),
            source_unit_counts=(24, 31),
            target_session_id="held_target",
            target_unit_count=37,
        )


def test_immutable_authority_loader_requires_complete_sha_bound_readonly_receipt(tmp_path: Path) -> None:
    authority = {
        "schema": core.REFERENCE_AUTHORITY_SCHEMA,
        "dataset": "subject_m",
        "view": "sua",
        "canonical_reference_receipt_sha256": _sha("reference"),
        "source_session_roster_sha256": _sha("source-roster"),
        "target_support_index_sha256": _sha("support-index"),
        "target_query_index_sha256": _sha("query-index"),
        "source_fitted_neural_input_preprocessor_sha256": _sha("neural-preprocessor"),
        "reference_metrics": {"carrier": 0.356812345678, "ridge": 0.417912345678},
    }
    body = tmp_path / "authority.json"
    body_sha = core.write_immutable_json(body, authority)
    sidecar = tmp_path / "authority.json.sha256"
    sidecar.write_text(f"{body_sha}  {body.name}\n", encoding="ascii")
    sidecar.chmod(0o444)
    loaded = core.load_immutable_reference_authority(
        body_path=body,
        sidecar_path=sidecar,
        dataset="subject_m",
        view="sua",
    )
    assert loaded["body_sha256"] == body_sha
    assert loaded["authority"]["reference_metrics"] == authority["reference_metrics"]
    assert stat.S_IMODE(body.stat().st_mode) == 0o444
    with pytest.raises(core.TrackBV2ContractError, match="overwrite"):
        core.write_immutable_json(body, authority)


def test_receipt_writer_creates_standard_sha_sidecar_and_rejects_reuse(tmp_path: Path) -> None:
    body = tmp_path / "preflight.json"
    receipt = core.write_immutable_receipt(body, core.build_no_data_preflight(dataset="rt", view=None))
    sidecar = Path(receipt["sidecar_path"])
    assert sidecar.read_text(encoding="ascii") == f"{receipt['body_sha256']}  {body.name}\n"
    assert stat.S_IMODE(body.stat().st_mode) == stat.S_IMODE(sidecar.stat().st_mode) == 0o444
    with pytest.raises(core.TrackBV2ContractError, match="overwrite"):
        core.write_immutable_receipt(body, core.build_no_data_preflight(dataset="rt", view=None))


def test_receipt_pair_rejects_sidecar_collision_without_orphaning_body(tmp_path: Path, monkeypatch) -> None:
    body = tmp_path / "pair.json"
    sidecar = body.with_name("pair.json.sha256")
    sidecar.write_text("occupied\n", encoding="ascii")
    with pytest.raises(core.TrackBV2ContractError, match="receipt pair"):
        core.write_immutable_receipt(body, core.build_no_data_preflight(dataset="rt", view=None))
    assert not body.exists()
    assert sidecar.read_text(encoding="ascii") == "occupied\n"

    sidecar.unlink()
    original = core._write_immutable_bytes

    def collide_after_body(path: Path, payload: bytes) -> str:
        digest = original(path, payload)
        if path == body:
            sidecar.write_text("racing-sidecar\n", encoding="ascii")
            sidecar.chmod(0o444)
        return digest

    monkeypatch.setattr(core, "_write_immutable_bytes", collide_after_body)
    with pytest.raises(core.TrackBV2ContractError, match="overwrite"):
        core.write_immutable_receipt(body, core.build_no_data_preflight(dataset="rt", view=None))
    assert not body.exists(), "sidecar collision must not leave a receipt body orphan"
    assert sidecar.read_text(encoding="ascii") == "racing-sidecar\n"


def test_no_data_preflight_and_score_refusal_are_explicit() -> None:
    preflight = core.build_no_data_preflight(dataset="rt", view=None)
    assert preflight["status"] == "NO_DATA_NO_GPU_PREFLIGHT"
    assert preflight["real_data_opened"] is False
    assert preflight["gpu_used"] is False
    assert preflight["cebra_imported"] is False
    assert preflight["checkpoint_loaded"] is False
    assert preflight["scoring_executed"] is False
    assert preflight["allowed_datasets"] == ["subject_m", "rt"]
    with pytest.raises(core.TrackBV2ContractError, match="has no live scorer"):
        core.refuse_score_execution()


def test_cli_is_h1_excluded_no_data_and_uses_immutable_output(tmp_path: Path) -> None:
    script = REPO_ROOT / "cebra_exploration/scripts/run_track_b_v2_preflight.py"
    receipt = tmp_path / "preflight.json"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    completed = subprocess.run(
        [sys.executable, str(script), "--dataset", "subject_m", "--view", "sua", "--output", str(receipt)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["status"] == "NO_DATA_NO_GPU_PREFLIGHT"
    assert summary["real_data_opened"] is False
    assert summary["gpu_used"] is False
    assert receipt.exists()
    assert Path(summary["receipt_sidecar_path"]).exists()
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o444
    assert json.loads(receipt.read_text(encoding="utf-8"))["cuda_visible_devices"] == ""
    h1 = subprocess.run(
        [sys.executable, str(script), "--dataset", "falcon_h1"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert h1.returncode != 0
    assert "invalid choice" in h1.stderr
    score = subprocess.run(
        [sys.executable, str(script), "--dataset", "rt", "--score"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert score.returncode != 0
    assert "has no live scorer" in score.stderr
