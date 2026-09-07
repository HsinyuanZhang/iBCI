"""Synthetic/fail-closed tests for Track-B v2's no-data live-integration contracts.

These tests never import CEBRA or an NWB loader, never open a real dataset or
checkpoint, and never use a GPU.  Temporary files model immutable receipt
topology only.
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

import track_b_v2_contract as base  # noqa: E402
import track_b_v2_live_contract as live  # noqa: E402


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _adapter() -> dict:
    return live.canonical_adapter_spec("subject_m", "sua")


def _manifest() -> live.SourceSupportQueryIndexManifest:
    return live.SourceSupportQueryIndexManifest(
        dataset="subject_m",
        view="sua",
        outer_fold_id="subject_m_sua_loso_target",
        target_session_id="target",
        source_session_ids=("source_b", "source_a"),
        source_fit_blocks=(
            live.IndexBlock("source_a", "source_fit", "raw_bin_index", (0, 1, 2)),
            live.IndexBlock("source_b", "source_fit", "raw_bin_index", (4, 5, 6)),
        ),
        target_support_block=live.IndexBlock("target", "target_support", "raw_bin_index", tuple(range(100))),
        target_query_block=live.IndexBlock("target", "target_query", "raw_bin_index", tuple(range(100, 140))),
        target_support_trial_indices=tuple(range(50)),
        target_query_trial_indices=(50, 51),
        target_support_label_scalar_count=200,
        target_support_label_unique_rows=100,
        t4_sparse_reference_label_event_count=50,
        t4_sparse_reference_label_row_count=50,
        t4_sparse_reference_label_scalar_count=50,
        t4_sparse_reference_label_semantics=live.SUBJECT_M_T4_LABEL_SEMANTICS,
        t4_neural_support_trial_count=30,
        cebra_neural_support_trial_count=50,
        cebra_dense_label_support_trial_count=50,
        neural_exposure_matched=False,
        neural_exposure_bias_direction="favors_CEBRA_accuracy",
    )


def _pseudo_mua_manifest() -> live.SourceSupportQueryIndexManifest:
    return live.SourceSupportQueryIndexManifest(
        dataset="subject_m",
        view="pseudo_mua",
        outer_fold_id="subject_m_pmua_loso_target",
        target_session_id="target",
        source_session_ids=("source_a", "source_b"),
        source_fit_blocks=(
            live.IndexBlock("source_a", "source_fit", "raw_bin_index", (0, 1, 2)),
            live.IndexBlock("source_b", "source_fit", "raw_bin_index", (4, 5, 6)),
        ),
        target_support_block=live.IndexBlock("target", "target_support", "raw_bin_index", tuple(range(100))),
        target_query_block=live.IndexBlock("target", "target_query", "raw_bin_index", tuple(range(100, 140))),
        target_support_trial_indices=tuple(range(50)),
        target_query_trial_indices=(50, 51),
        target_support_label_scalar_count=200,
        target_support_label_unique_rows=100,
        t4_sparse_reference_label_event_count=50,
        t4_sparse_reference_label_row_count=50,
        t4_sparse_reference_label_scalar_count=50,
        t4_sparse_reference_label_semantics=live.SUBJECT_M_T4_LABEL_SEMANTICS,
        t4_neural_support_trial_count=30,
        cebra_neural_support_trial_count=50,
        cebra_dense_label_support_trial_count=50,
        neural_exposure_matched=False,
        neural_exposure_bias_direction="favors_CEBRA_accuracy",
        view_feature_sha256_by_session={
            "source_a": _sha("pmua-source-a"),
            "source_b": _sha("pmua-source-b"),
            "target": _sha("pmua-target"),
        },
    )


def _selector_results(selector: base.SourceOnlySelectorSpec) -> list[base.SourceOnlySelectionResult]:
    return [
        base.SourceOnlySelectionResult(
            candidate=candidate,
            linear_ridge_source_inner_query_r2=0.8,
            inner_source_fold_ids=("source_inner_a", "source_inner_b"),
        )
        for candidate in selector.candidates()
    ]


def _knn_selector_results(selector: base.SourceOnlySelectorSpec) -> list[base.KnnSourceOnlySelectionResult]:
    return [
        base.KnnSourceOnlySelectionResult(
            candidate=candidate,
            knn_source_inner_query_r2=0.7,
            inner_source_fold_ids=("source_inner_a", "source_inner_b"),
        )
        for candidate in selector.knn_candidates()
    ]


def _write_pair(path: Path, payload: dict) -> base.ExplicitSealedReceiptPair | live.ExplicitSealedReceiptPair:
    base.write_immutable_receipt(path, payload)
    return live.ExplicitSealedReceiptPair(
        role="unused",
        body_path=path,
        sidecar_path=path.with_name(f"{path.name}.sha256"),
    )


def test_canonical_adapters_bind_only_subject_m_views_and_rt_semantics() -> None:
    sua = live.canonical_adapter_spec("subject_m", "sua")
    pmua = live.canonical_adapter_spec("subject_m", "pseudo_mua")
    rt = live.canonical_adapter_spec("rt")
    assert sua["signal_view"] == "sua"
    assert pmua["signal_view"] == "pseudo_mua"
    assert pmua["feature_construction"] == "pool_spikes_by_electrode__canonical_subject_m_pseudo_mua"
    assert pmua["loader_bindings_required"]["pseudo_mua_constructor"]["symbol"] == "pool_spikes_by_electrode"
    assert rt["support_semantics"] == "chronological_first_24_trials__M24"
    assert rt["loader_bindings_required"]["query_layout"]["symbol"] == "rt_outer_window_layout"
    for adapter in (sua, pmua, rt):
        assert adapter["auxiliary"] == "continuous_velocity_dense_bin_level"
        assert adapter["information_matched"] is False
        assert adapter["bias_direction"] == "favors_CEBRA_accuracy"


def test_scope_rejection_precedes_any_arbitrary_path_or_discovery_coercion() -> None:
    class PoisonPath:
        def __fspath__(self) -> str:
            raise AssertionError("illegal scope must reject before a path is coerced")

        def __str__(self) -> str:
            raise AssertionError("illegal scope must reject before a path is stringified")

    with pytest.raises(base.TrackBV2ContractError, match="permits only"):
        live.canonical_adapter_spec("falcon_h1", proposed_data_path=PoisonPath())
    with pytest.raises(base.TrackBV2ContractError, match="permits only"):
        live.canonical_adapter_spec("falcon_m2", proposed_discovery=PoisonPath())
    with pytest.raises(live.TrackBV2LiveContractError, match="no arbitrary data path"):
        live.canonical_adapter_spec("rt", proposed_data_path=PoisonPath())


def test_sealed_loader_semantics_are_current_sha_bound_and_no_data() -> None:
    adapter = _adapter()
    receipt = live.build_sealed_loader_semantics_contract("subject_m", "sua")
    assert live.validate_sealed_loader_semantics_contract(adapter, receipt) == receipt
    assert receipt["nwb_opened"] is False
    poisoned = json.loads(json.dumps(receipt))
    poisoned["loader_bindings"]["record_loader"]["sha256"] = "0" * 64
    with pytest.raises(live.TrackBV2LiveContractError, match="SHA/symbol"):
        live.validate_sealed_loader_semantics_contract(adapter, poisoned)
    pmua_receipt = live.build_sealed_loader_semantics_contract("subject_m", "pseudo_mua")
    assert "pseudo_mua_constructor" in pmua_receipt["loader_bindings"]


def test_index_manifest_proves_raw_coverage_disjointness_and_declares_density_bias() -> None:
    manifest = _manifest()
    payload = manifest.as_dict()
    assert live.index_manifest_from_dict(payload) == manifest
    proof = live.verify_support_query_disjointness(manifest)
    assert proof["raw_coverage_overlap_count"] == 0
    assert proof["target_query_neural_in_fit"] is False
    assert proof["auxiliary"] == "continuous_velocity_dense_bin_level"
    assert proof["information_matched"] is False
    assert proof["bias_direction"] == "favors_CEBRA_accuracy"
    assert proof["target_support_dense_labels_used_for_cebra_encoder_fit"] is True
    with pytest.raises(live.TrackBV2LiveContractError, match="raw observation coverage overlaps"):
        live.SourceSupportQueryIndexManifest(
            dataset="subject_m", view="sua", outer_fold_id="x", target_session_id="target",
            source_session_ids=("source_a", "source_b"),
            source_fit_blocks=(
                live.IndexBlock("source_a", "source_fit", "raw_bin_index", (0,)),
                live.IndexBlock("source_b", "source_fit", "raw_bin_index", (1,)),
            ),
            target_support_block=live.IndexBlock("target", "target_support", "raw_bin_index", (0, 1, 2)),
            target_query_block=live.IndexBlock("target", "target_query", "raw_bin_index", (2, 3)),
            target_support_trial_indices=tuple(range(50)), target_query_trial_indices=(50,),
            target_support_label_scalar_count=100, target_support_label_unique_rows=50,
            t4_sparse_reference_label_event_count=50,
            t4_sparse_reference_label_row_count=50,
            t4_sparse_reference_label_scalar_count=50,
            t4_sparse_reference_label_semantics=live.SUBJECT_M_T4_LABEL_SEMANTICS,
            t4_neural_support_trial_count=30,
            cebra_neural_support_trial_count=50,
            cebra_dense_label_support_trial_count=50,
            neural_exposure_matched=False,
            neural_exposure_bias_direction="favors_CEBRA_accuracy",
        )

    pseudo_mua = _pseudo_mua_manifest()
    assert live.index_manifest_from_dict(pseudo_mua.as_dict()) == pseudo_mua
    incomplete_pmua = pseudo_mua.as_dict()
    incomplete_pmua["view_feature_sha256_by_session"].pop("target")
    incomplete_pmua.pop("index_manifest_sha256")
    with pytest.raises(live.TrackBV2LiveContractError, match="pseudo_mua feature SHA map"):
        live.index_manifest_from_dict(incomplete_pmua)
    with pytest.raises(live.TrackBV2LiveContractError, match="match label information"):
        live.SourceSupportQueryIndexManifest(
            **({name: getattr(manifest, name) for name in (
                "dataset", "view", "outer_fold_id", "target_session_id", "source_session_ids", "source_fit_blocks",
                "target_support_block", "target_query_block", "target_support_trial_indices", "target_query_trial_indices",
                "target_support_label_scalar_count", "target_support_label_unique_rows",
                "t4_sparse_reference_label_event_count", "t4_sparse_reference_label_row_count",
                "t4_sparse_reference_label_scalar_count", "t4_sparse_reference_label_semantics",
                "t4_neural_support_trial_count", "cebra_neural_support_trial_count",
                "cebra_dense_label_support_trial_count", "neural_exposure_matched",
                "neural_exposure_bias_direction",
            )}),
            information_matched=True,
        )


def test_source_neural_preprocessing_selector_and_multisession_provenance_are_target_closed() -> None:
    adapter = _adapter()
    manifest = _manifest()
    semantics = live.build_sealed_loader_semantics_contract("subject_m", "sua")
    neural_preprocessor = live.build_source_only_neural_input_preprocessor_receipt(
        adapter=adapter, loader_semantics=semantics, manifest=manifest,
        source_fitted_neural_input_preprocessor_sha256=_sha("source-neural-preprocessor"),
    )
    assert neural_preprocessor["target_data_used"] is False
    assert neural_preprocessor["target_support_label_scalar_count"] == 200
    assert neural_preprocessor["behavior_auxiliary_scaler_receipt"].startswith("LIVE_BLOCKER")
    selector = base.SourceOnlySelectorSpec(d_grid=(8,), iteration_grid=(1000,), lambda_grid=(1.0e-2,))
    chosen = live.build_inner_fold_selector_receipt(
        manifest=manifest, neural_input_preprocessor_receipt_sha256=_sha("preprocessor-receipt"),
        selector=selector, linear_results=_selector_results(selector), knn_results=_knn_selector_results(selector),
    )
    assert chosen["target_support_read"] is False
    assert chosen["selection"]["target_data_used"] is False
    plan = live.build_multisession_provenance_plan(
        manifest=manifest, source_neural_input_preprocessor_receipt_sha256=_sha("preprocessor-receipt"),
        selector_receipt_sha256=_sha("selector-receipt"), arm="cebra_joint_behavior",
    )
    assert plan["solver"] == live.MULTISESSION_SOLVER
    assert plan["cebra_solver_called"] is False
    assert plan["target_query_in_readout_fit"] is False
    primary = plan["readout_receipts_required"]["source_only_consumer_mechanism_alignment"]
    standard = plan["readout_receipts_required"]["target_support_only_standard_cebra_accuracy"]
    hybrid = plan["readout_receipts_required"]["source_plus_target_support_hybrid_sensitivity"]
    assert primary["fit_on"] == ["source_fit"]
    assert primary["target_support_dense_labels_used_for_cebra_encoder_fit"] is True
    assert primary["target_support_dense_labels_used_for_readout_fit"] is False
    assert primary["embedding_standardizer"]["fit_scope"] == "outer_source_embeddings_only"
    assert standard["fit_on"] == ["target_support"]
    assert standard["target_support_dense_labels_used_for_readout_fit"] is True
    assert standard["embedding_standardizer"]["fit_scope"] == "target_support_embeddings_only"
    assert hybrid["fit_on"] == ["source_fit", "target_support"]
    assert hybrid["not_an_accuracy_upper_bound"] is True
    assert hybrid["embedding_standardizer"]["fit_scope"] == "source_plus_target_support_embeddings_only"
    assert standard["accuracy_table_headline"] is True
    assert primary["accuracy_table_headline"] is False
    assert plan["accuracy_table_headline_estimand"] == "target_support_only_standard_cebra_accuracy"
    assert plan["headline_estimand_selection_permitted_at_runtime"] is False
    assert plan["accuracy_table_headline"]["model_arm"] == "cebra_joint_behavior"
    assert plan["model_arm_role"]["role"].startswith("standard_supported_cebra")
    frozen = live.build_multisession_provenance_plan(
        manifest=manifest, source_neural_input_preprocessor_receipt_sha256=_sha("preprocessor-receipt"),
        selector_receipt_sha256=_sha("selector-receipt"), arm="cebra_frozen_source_adapt",
    )
    assert frozen["model_arm_role"]["role"].startswith("vendored_deployment_oriented_sensitivity")
    assert frozen["accuracy_table_headline"]["model_arm"] == "cebra_joint_behavior"
    assert plan["model_receipt_required"]["behavior_auxiliary_scaler"]["fit_scope"] == "outer_source_behavior_auxiliary_only"
    assert plan["model_receipt_required"]["behavior_auxiliary_scaler"]["required_method"] == "canonical_fit_behavior_stats_componentwise_zscore"
    assert plan["model_receipt_required"]["behavior_auxiliary_scaler"]["second_refit_or_zscore_permitted"] is False
    assert plan["source_authority_binding_required"]["source_domain"] == "strict27_subc_co_train_only"
    assert plan["source_authority_binding_required"]["subm_lodo_source_permitted"] is False
    assert plan["cebra_score_seed_policy_required"]["subject_m_reference"]["reference"] == "sealed_T4_three_seed_aggregate"
    assert plan["cebra_score_seed_policy_required"]["terminal_development"]["cebra_seeds"] == [42, 43, 44]
    assert "may_fit_on" not in json.dumps(plan)
    with pytest.raises(live.TrackBV2LiveContractError, match="unauthorised"):
        live.build_multisession_provenance_plan(
            manifest=manifest, source_neural_input_preprocessor_receipt_sha256=_sha("preprocessor-receipt"),
            selector_receipt_sha256=_sha("selector-receipt"), arm="cebra_joint_behavior", execution_requested=True,
        )


def test_actual_cebra_cpu_control_contract_requires_exact_eight_seed_coverage_without_execution() -> None:
    geometry = base.CandidateGeometry(output_dimension=8, source_iterations=10000, normalized_lambda=1.0e-2)
    contract = live.build_actual_cebra_cpu_control_runner_contract(
        dataset="rt", view=None, linear_ridge_scoring_geometry=geometry,
        knn_scoring_geometry=base.KnnCandidateGeometry(3, 250),
    )
    assert contract["cebra_imported"] is False
    assert contract["expected_measurement_count"] == 24
    assert contract["expected_decoder_score_count"] == 48
    assert contract["required_decoders"] == ["linear_ridge", "knn_cosine_k3"]
    assert contract["knn_cosine_k3_scoring_geometry"]["normalized_lambda"] == "NOT_APPLICABLE"
    assert "target_support_label_scalar_count" in contract["required_live_label_density_bindings"]
    assert contract["unaligned_control"]["role"].startswith("diagnostic_distribution_only")
    assert contract["unaligned_control"]["hard_threshold"] is None
    assert contract["synthetic_deranged_support_auxiliary_hard_null"].startswith("SEPARATE_PENDING")
    measurements = []
    for arm in base.POSITIVE_ARMS + (base.NEGATIVE_ARM,):
        for seed in base.DEFAULT_CONTROL_SEEDS:
            measurements.append(base.ControlMeasurement(
                arm=arm, seed=seed, geometry=geometry,
                index_manifest_sha256=_sha("rt-control-index"),
                source_query_r2_by_decoder={"linear_ridge": 0.8, "knn_cosine_k3": 0.8},
                target_query_r2_by_decoder={
                    "linear_ridge": 0.8 if arm in base.POSITIVE_ARMS else 0.1,
                    "knn_cosine_k3": 0.8 if arm in base.POSITIVE_ARMS else 0.1,
                },
                target_query_neural_in_fit=False, target_query_labels_in_fit=False,
                target_support_dense_labels_used_for_cebra_encoder_fit=True,
                target_support_label_scalar_count=96,
                target_support_label_unique_rows=48,
                t4_sparse_reference_label_event_count=24,
                t4_sparse_reference_label_row_count=24,
                t4_sparse_reference_label_scalar_count=24,
                t4_sparse_reference_label_semantics=live.RT_T4D_LABEL_SEMANTICS,
                knn_geometry=base.KnnCandidateGeometry(3, 250),
            ))
    gate = live.assert_eight_seed_actual_cebra_cpu_controls(
        linear_ridge_scoring_geometry=geometry,
        knn_scoring_geometry=base.KnnCandidateGeometry(3, 250),
        measurements=measurements,
    )
    assert gate["measurement_count"] == 24
    assert gate["decoder_score_count"] == 48
    borrowed_knn_geometry = list(measurements)
    borrowed_knn_geometry[-1] = base.ControlMeasurement(
        arm=base.NEGATIVE_ARM, seed=base.DEFAULT_CONTROL_SEEDS[-1], geometry=geometry,
        index_manifest_sha256=_sha("rt-control-index"),
        source_query_r2_by_decoder={"linear_ridge": 0.8, "knn_cosine_k3": 0.8},
        target_query_r2_by_decoder={"linear_ridge": 0.1, "knn_cosine_k3": 0.1},
        target_query_neural_in_fit=False, target_query_labels_in_fit=False,
        target_support_dense_labels_used_for_cebra_encoder_fit=True,
        target_support_label_scalar_count=96,
        target_support_label_unique_rows=48,
        t4_sparse_reference_label_event_count=24,
        t4_sparse_reference_label_row_count=24,
        t4_sparse_reference_label_scalar_count=24,
        t4_sparse_reference_label_semantics=live.RT_T4D_LABEL_SEMANTICS,
        knn_geometry=base.KnnCandidateGeometry(8, 10000),
    )
    with pytest.raises(base.TrackBV2ContractError, match="kNN control geometry"):
        live.assert_eight_seed_actual_cebra_cpu_controls(
            linear_ridge_scoring_geometry=geometry,
            knn_scoring_geometry=base.KnnCandidateGeometry(3, 250),
            measurements=borrowed_knn_geometry,
        )
    broken_knn = list(measurements)
    broken_knn[-1] = base.ControlMeasurement(
        arm=base.NEGATIVE_ARM, seed=base.DEFAULT_CONTROL_SEEDS[-1], geometry=geometry,
        index_manifest_sha256=_sha("rt-control-index"),
        source_query_r2_by_decoder={"linear_ridge": 0.8, "knn_cosine_k3": 0.8},
        target_query_r2_by_decoder={"linear_ridge": 0.1, "knn_cosine_k3": 0.25},
        target_query_neural_in_fit=False, target_query_labels_in_fit=False,
        target_support_dense_labels_used_for_cebra_encoder_fit=True,
        target_support_label_scalar_count=96,
        target_support_label_unique_rows=48,
        t4_sparse_reference_label_event_count=24,
        t4_sparse_reference_label_row_count=24,
        t4_sparse_reference_label_scalar_count=24,
        t4_sparse_reference_label_semantics=live.RT_T4D_LABEL_SEMANTICS,
        knn_geometry=base.KnnCandidateGeometry(3, 250),
    )
    assert live.assert_eight_seed_actual_cebra_cpu_controls(
        linear_ridge_scoring_geometry=geometry,
        knn_scoring_geometry=base.KnnCandidateGeometry(3, 250),
        measurements=broken_knn,
    )["negative_arm_role"].startswith("diagnostic_distribution")
    with pytest.raises(base.TrackBV2ContractError, match="coverage"):
        live.assert_eight_seed_actual_cebra_cpu_controls(
            linear_ridge_scoring_geometry=geometry,
            knn_scoring_geometry=base.KnnCandidateGeometry(3, 250),
            measurements=measurements[:-1],
        )


def test_reference_authority_builder_accepts_only_pointer_bound_sidecarless_body_and_extracts_metrics(tmp_path: Path) -> None:
    adapter = _adapter()
    manifest = _manifest()
    legacy_body = tmp_path / "legacy_terminal.json"
    legacy_payload = {"receipt_kind": "legacy_terminal", "exact": {"carrier": 0.3568282358, "ridge": 0.417912345678}}
    legacy_body.write_bytes(json.dumps(legacy_payload, sort_keys=True).encode("utf-8"))
    legacy_body.chmod(0o444)
    pointer_payload = {
        "schema": live.SEALED_BODY_POINTER_SCHEMA,
        "status": "ROOT_AUDITED_POINTER__NOT_A_RESULT",
        "dataset": "subject_m", "view": "sua", "pointer_role": "canonical_reference_terminal",
        "sealed_body_path": str(legacy_body),
        "sealed_body_sha256": hashlib.sha256(legacy_body.read_bytes()).hexdigest(),
        "sealed_body_mode": "0444", "sealed_body_has_adjacent_sidecar": False,
        "sealed_body_schema": {"field": "receipt_kind", "value": "legacy_terminal"},
        "reference_metric_json_pointers": {"carrier": "/exact/carrier", "ridge": "/exact/ridge"},
    }
    pointer_path = tmp_path / "terminal_pointer.json"
    base.write_immutable_receipt(pointer_path, pointer_payload)
    roster_path = tmp_path / "source_roster.json"
    base.write_immutable_receipt(roster_path, {
        "schema": "track_b_v2_source_roster_receipt_v1", "dataset": "subject_m", "view": "sua",
        "outer_fold_id": manifest.outer_fold_id, "target_session_id": manifest.target_session_id,
        "source_session_ids": ["source_a", "source_b"],
    })
    neural_preprocessor_path = tmp_path / "neural_preprocessor.json"
    base.write_immutable_receipt(neural_preprocessor_path, {
        "schema": live.SOURCE_NEURAL_INPUT_PREPROCESSOR_SCHEMA, "dataset": "subject_m", "view": "sua",
        "outer_fold_id": manifest.outer_fold_id, "index_manifest_sha256": manifest.sha256,
        "source_session_roster_sha256": manifest.source_session_roster_sha256,
        "source_fitted_neural_input_preprocessor_sha256": _sha("preprocessor-values"),
        "target_data_used": False, "target_support_neural_in_neural_input_preprocessor_fit": False,
        })
    index_path = tmp_path / "index.json"
    base.write_immutable_receipt(index_path, manifest.as_dict())
    pointer_pair = live.ExplicitSealedReceiptPair(
        "canonical_reference_body_pointer", pointer_path, pointer_path.with_name("terminal_pointer.json.sha256"),
    )
    authority = live.build_canonical_metric_reference_authority_from_sealed_pointer(
        adapter=adapter,
        canonical_reference_body_pointer=pointer_pair,
    )
    assert authority["reference_metrics"] == legacy_payload["exact"]
    assert authority["canonical_reference_receipt_sha256"] == pointer_payload["sealed_body_sha256"]
    assert authority["rounded_literal_reference_metrics_accepted"] is False
    assert authority["status"].endswith("NOT_MINTED")
    assert authority["authority_scope"] == "aggregate_metric_reference_only__not_live_fold_lineage"
    assert "target_support_index_sha256" not in authority
    fold_binding = live.build_live_fold_reference_binding(
        adapter=adapter,
        canonical_reference_body_pointer=pointer_pair,
        source_roster=live.ExplicitSealedReceiptPair(
            "source_roster", roster_path, roster_path.with_name("source_roster.json.sha256"),
        ),
        source_neural_input_preprocessor=live.ExplicitSealedReceiptPair(
            "source_neural_input_preprocessor",
            neural_preprocessor_path,
            neural_preprocessor_path.with_name("neural_preprocessor.json.sha256"),
        ),
        source_support_query_index=live.ExplicitSealedReceiptPair(
            "source_support_query_index", index_path, index_path.with_name("index.json.sha256"),
        ),
    )
    assert fold_binding["authority_scope"] == "one_live_fold_lineage__not_aggregate_metric_authority"
    assert fold_binding["canonical_reference_receipt_sha256"] == pointer_payload["sealed_body_sha256"]
    assert fold_binding["target_support_index_sha256"] == manifest.target_support_block.sha256
    assert fold_binding["reference_metrics_present"] is False
    legacy_sidecar = legacy_body.with_name("legacy_terminal.json.sha256")
    legacy_sidecar.write_text("unexpected\n", encoding="ascii")
    with pytest.raises(live.TrackBV2LiveContractError, match="unexpected adjacent sidecar"):
        live.build_canonical_metric_reference_authority_from_sealed_pointer(
            adapter=adapter,
            canonical_reference_body_pointer=pointer_pair,
        )


def test_cpu_control_cli_is_no_data_cpu_only_and_execute_is_fail_closed() -> None:
    script = REPO_ROOT / "cebra_exploration" / "scripts" / "run_track_b_v2_cpu_controls.py"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(script), "--dataset", "rt"], cwd=REPO_ROOT, env=env,
        text=True, capture_output=True, check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "CONTRACT_ONLY__CEBRA_NOT_IMPORTED_OR_EXECUTED"
    assert payload["cuda_visible_devices"] == ""
    assert payload["cebra_solver_called"] is False
    refused = subprocess.run(
        [sys.executable, str(script), "--dataset", "rt", "--execute"], cwd=REPO_ROOT, env=env,
        text=True, capture_output=True,
    )
    assert refused.returncode != 0
    assert "not authorised" in refused.stderr


def test_single_fd_immutable_reader_rejects_open_then_rename_path_poison(tmp_path: Path, monkeypatch) -> None:
    body = tmp_path / "sealed.json"
    replacement = tmp_path / "replacement.json"
    body.write_bytes(b'{"receipt":"original"}')
    replacement.write_bytes(b'{"receipt":"replacement"}')
    body.chmod(0o444)
    replacement.chmod(0o444)
    real_fstat = live.os.fstat
    calls = 0

    def replace_after_second_fstat(descriptor: int):
        nonlocal calls
        info = real_fstat(descriptor)
        calls += 1
        if calls == 2:
            os.replace(replacement, body)
        return info

    monkeypatch.setattr(live.os, "fstat", replace_after_second_fstat)
    with pytest.raises(live.TrackBV2LiveContractError, match="pathname inode changed after read"):
        live._read_immutable_regular_same_fd(body, label="path-poison-test")
