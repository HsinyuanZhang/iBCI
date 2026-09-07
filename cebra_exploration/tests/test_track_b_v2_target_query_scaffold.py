"""Synthetic, no-data tests for Track-B v2 target/query lineage scaffolding."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "cebra_exploration" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_target_query_scaffold as target  # noqa: E402
import track_b_v2_metric_pointer_authority as pointer  # noqa: E402
import track_b_v2_source_adapter as source  # noqa: E402


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _authority_map() -> dict[str, str]:
    return {key: _sha(key) for key in target._SOURCE_AUTHORITY_KEYS}


def _redirect_canonical_pairs_to_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "canonical_root"
    monkeypatch.setattr(pointer, "_CANONICAL_POINTER_BODY_PATHS", {
        ("subject_m", "sua"): root / "subject_m_sua_metric_pointer.json",
        ("subject_m", "pseudo_mua"): root / "subject_m_pseudo_mua_metric_pointer.json",
        ("rt", None): root / "rt_metric_pointer.json",
    })
    monkeypatch.setattr(pointer, "_CANONICAL_AUDIT_ATTESTATION_BODY_PATHS", {
        ("subject_m", "sua"): root / "subject_m_sua_root_audit_attestation.json",
        ("subject_m", "pseudo_mua"): root / "subject_m_pseudo_mua_root_audit_attestation.json",
        ("rt", None): root / "rt_root_audit_attestation.json",
    })


def _mint_test_pointer(*, dataset: str, view: str | None) -> pointer.live.ExplicitSealedReceiptPair:
    dry = pointer.build_metric_pointer_dry_plan(dataset, view)
    attestation = pointer.publish_root_metric_pointer_audit_attestation_pair(
        dry_plan=dry, root_authorized_publish=True,
    )
    return pointer.publish_root_audited_metric_pointer_pair(
        dry_plan=dry, root_audit_attestation_pair=attestation, root_authorized_publish=True,
    )


def _proposal(dataset: str, view: str | None) -> dict[str, object]:
    if dataset == "subject_m":
        bodies = {
            "summary_cross_check_body": {
                "path": "/sealed/subm-summary.json", "sha256": _sha("summary"),
                "metric_json_pointer": "/summary/sua/50/mean_r2",
                "metric_pointer_verified_against_same_fd_bytes": True,
            },
            "per_session_seed_body": {
                "path": "/sealed/subm-cells.json", "sha256": _sha("cells"),
            },
        }
    else:
        bodies = {
            "aggregate_metric_query_identity_body": {
                "path": "/sealed/rt-aggregate.json", "sha256": _sha("rt-aggregate"),
                "metric_json_pointer": "/results/arms/t4d_reference/mean",
                "metric_pointer_verified_against_same_fd_bytes": True,
            },
            "per_fold_t4d_body": {
                "path": "/sealed/rt-fold.json", "sha256": _sha("rt-fold"),
                "required_json_pointer": "/rows/*/t4d_r2",
                "metric_pointer_verified_against_same_fd_bytes": True,
            },
            "stage2_delta_companion": {"path": "/sealed/rt-delta.json", "sha256": _sha("rt-delta")},
        }
    return {
        "schema": "track_b_v2_metric_pointer_mint_proposal_v1",
        "status": "ROOT_AUDITED_IMMUTABLE_POINTER_REQUIRED__NOT_MINTED__NOT_AUTHORITY",
        "dataset": dataset,
        "view": view,
        "bodies": bodies,
        "rounded_literals_accepted": False,
        "legacy_bodies_modified": False,
    }


def test_subject_m_no_data_plan_binds_not_minted_proposal_and_all_readout_roles() -> None:
    proposal = _proposal("subject_m", "sua")
    plan = target.build_no_data_target_query_scaffold(
        dataset="subject_m", view="sua", outer_fold_id="subm-20140307",
        target_session_id="sub-M_ses-CO-20140307", source_authority_sha256s=_authority_map(),
        pointer_proposal_payload=proposal, root_audit_attestation_sha256=_sha("root-review"),
    )
    assert plan["status"].startswith("NO_DATA_CPU_ONLY")
    binding = plan["pointer_proposal_binding"]
    assert binding["immutable_pointer_minted"] is False
    assert binding["metric_authority_available"] is False
    assert binding["proposal_payload_sha256"] == hashlib.sha256(
        __import__("track_b_v2_contract").canonical_json_bytes(proposal)
    ).hexdigest()
    support = plan["target_support_query_trial_window_contract"]["support_query_contract"]
    assert support["required_support_trial_ordinals"] == list(range(50))
    assert support["t4_neural_support_trial_count"] == 30
    assert support["cebra_neural_support_trial_count"] == 50
    assert support["bias_direction"] == "favors_CEBRA_accuracy"
    prefix = support["standard_cebra_support_sequence"]
    assert prefix["sequence_semantics"] == "one_continuous_chronological_prefix"
    assert prefix["stop"] == "STOP_OF_REWARDED_TRIAL_50"
    assert prefix["rewarded_segments_concatenated"] is False
    lineage = plan["target_support_query_trial_window_contract"]["future_live_lineage_must_bind"]
    assert lineage["standard_cebra_support_prefix_stop_raw_bin"].endswith("M50_OR_M24_TRIAL")
    assert lineage["v9_runtime_base_input_trace_sha256"].endswith("TRIALS_30")
    assert lineage["v9_runtime_query_behavior_trace_sha256"].endswith("TRIAL_50")
    routes = plan["readout_authority_plan"]["readout_routes"]
    assert routes["target_support_only_standard_cebra_accuracy"]["readout_fit_scope"] == "target_support_only"
    assert routes["source_plus_target_support_hybrid_sensitivity"]["scientific_role"].startswith("mandatory_sensitivity")
    temporal = plan["readout_authority_plan"]["temporal_exposure_policy"]
    assert temporal["default_offset10_model_offset"]["future_raw_bins"] == 5
    assert temporal["causal_alignment_sensitivity"]["posthoc_embedding_shift_permitted"] is False
    assert plan["target_data_opened"] is False
    assert plan["formal_data_opened"] is False
    assert plan["cebra_trained"] is False
    assert plan["score_emitted"] is False


def test_rt_no_data_plan_freezes_m24_and_disallows_target_paths_before_execution() -> None:
    plan = target.build_no_data_target_query_scaffold(
        dataset="rt", view=None, outer_fold_id="rt-20131009",
        target_session_id="ses-RT-20131009", source_authority_sha256s=_authority_map(),
        pointer_proposal_payload=_proposal("rt", None), root_audit_attestation_sha256=_sha("root-review"),
    )
    support = plan["target_support_query_trial_window_contract"]["support_query_contract"]
    assert support["required_support_trial_ordinals"] == list(range(24))
    assert support["query_window_semantics"].startswith("sealed_RT_outer_q24")
    assert support["standard_cebra_support_sequence"]["stop"] == "STOP_OF_CHRONOLOGICAL_TRIAL_24"
    with pytest.raises(target.TrackBV2TargetQueryScaffoldError, match="no target data path"):
        target.build_no_data_target_query_scaffold(
            dataset="rt", view=None, outer_fold_id="rt-20131009", target_session_id="ses-RT-20131009",
            source_authority_sha256s=_authority_map(), pointer_proposal_payload=_proposal("rt", None),
            root_audit_attestation_sha256=_sha("root-review"), proposed_target_path="/a/target.nwb",
        )


def test_scope_rejects_h1_before_touching_target_like_object() -> None:
    class Poison:
        def __str__(self) -> str:
            raise AssertionError("H1 rejection must precede target coercion")

    with pytest.raises(target.base.TrackBV2ContractError, match="H1-excluded"):
        target.build_no_data_target_query_scaffold(
            dataset="falcon_h1", view=None, outer_fold_id="bad", target_session_id=Poison(),
            source_authority_sha256s={}, pointer_proposal_payload={}, root_audit_attestation_sha256="bad",
        )


def test_receptive_field_proof_requires_target_byte_parity_order_and_full_query_containment() -> None:
    proof = target.QueryReceptiveFieldAndTargetByteProof(
        dataset="subject_m", view="pseudo_mua", target_session_id="sub-M_ses-CO-20140307",
        support_raw_indices=(0, 1, 2), query_raw_indices=tuple(range(3, 60)),
        valid_window_start_indices=(3, 5),
        t4_reference_ordered_prediction_target_raw_bin_indices=(52, 54),
        cebra_scored_ordered_prediction_target_raw_bin_indices=(52, 54),
        full_receptive_field_raw_indices=(tuple(range(3, 53)), tuple(range(5, 55))),
        vendored_model_alignment_authority_sha256=_sha("actual-vendored-cebra-alignment"),
        t4_target_float32_raw_bytes_sha256=_sha("targets"),
        cebra_target_float32_raw_bytes_sha256=_sha("targets"),
        t4_reference_query_identity_sha256=_sha("query-identity"),
        metric_implementation_authority_sha256=_sha("torchmetrics-151"),
    ).as_dict()
    assert proof["paired_accuracy_claim_permitted_by_this_boundary_proof"] is True
    assert proof["ordered_prediction_target_raw_bin_indices"]["index_count"] == 2
    assert "indices" not in proof["ordered_prediction_target_raw_bin_indices"]
    alignment = proof["vendored_model_receptive_field_alignment"]
    assert alignment["prediction_endpoint_semantics"].endswith("valid_start_plus_49")
    assert alignment["receptive_field_containment_violation_count"] == 0
    with pytest.raises(target.TrackBV2TargetQueryScaffoldError, match="differ from ordered T4 target bytes"):
        target.QueryReceptiveFieldAndTargetByteProof(
            dataset="rt", view=None, target_session_id="ses-RT-20131009",
            support_raw_indices=(0, 1), query_raw_indices=tuple(range(2, 50)),
            valid_window_start_indices=(0,),
            t4_reference_ordered_prediction_target_raw_bin_indices=(49,),
            cebra_scored_ordered_prediction_target_raw_bin_indices=(49,),
            full_receptive_field_raw_indices=((2, 3),),
            vendored_model_alignment_authority_sha256=_sha("alignment"),
            t4_target_float32_raw_bytes_sha256=_sha("a"), cebra_target_float32_raw_bytes_sha256=_sha("b"),
            t4_reference_query_identity_sha256=_sha("q"), metric_implementation_authority_sha256=_sha("metric"),
        )
    with pytest.raises(target.TrackBV2TargetQueryScaffoldError, match="escapes target query"):
        target.QueryReceptiveFieldAndTargetByteProof(
            dataset="rt", view=None, target_session_id="ses-RT-20131009",
            support_raw_indices=(0, 1), query_raw_indices=tuple(range(2, 51)),
            valid_window_start_indices=(0,),
            t4_reference_ordered_prediction_target_raw_bin_indices=(49,),
            cebra_scored_ordered_prediction_target_raw_bin_indices=(49,),
            full_receptive_field_raw_indices=((1, 49),),
            vendored_model_alignment_authority_sha256=_sha("alignment"),
            t4_target_float32_raw_bytes_sha256=_sha("same"), cebra_target_float32_raw_bytes_sha256=_sha("same"),
            t4_reference_query_identity_sha256=_sha("q"), metric_implementation_authority_sha256=_sha("metric"),
        )
    with pytest.raises(target.TrackBV2TargetQueryScaffoldError, match="exactly equal T4 endpoint ordering"):
        target.QueryReceptiveFieldAndTargetByteProof(
            dataset="rt", view=None, target_session_id="ses-RT-20131009",
            support_raw_indices=(0, 1), query_raw_indices=tuple(range(2, 52)),
            valid_window_start_indices=(2,),
            t4_reference_ordered_prediction_target_raw_bin_indices=(51,),
            cebra_scored_ordered_prediction_target_raw_bin_indices=(50,),
            full_receptive_field_raw_indices=(tuple(range(2, 52)),),
            vendored_model_alignment_authority_sha256=_sha("alignment"),
            t4_target_float32_raw_bytes_sha256=_sha("same"), cebra_target_float32_raw_bytes_sha256=_sha("same"),
            t4_reference_query_identity_sha256=_sha("q"), metric_implementation_authority_sha256=_sha("metric"),
        )


def test_receptive_field_proof_rejects_valid_start_as_prediction_target() -> None:
    with pytest.raises(target.TrackBV2TargetQueryScaffoldError, match=r"valid-window endpoints: valid_start \+ 49"):
        target.QueryReceptiveFieldAndTargetByteProof(
            dataset="subject_m", view="sua", target_session_id="sub-M_ses-CO-20140307",
            support_raw_indices=(0, 1), query_raw_indices=tuple(range(2, 60)),
            valid_window_start_indices=(2,),
            # Deliberate start-vs-end confusion: 2 is a valid start, not its endpoint 51.
            t4_reference_ordered_prediction_target_raw_bin_indices=(2,),
            cebra_scored_ordered_prediction_target_raw_bin_indices=(2,),
            full_receptive_field_raw_indices=(tuple(range(2, 52)),),
            vendored_model_alignment_authority_sha256=_sha("alignment"),
            t4_target_float32_raw_bytes_sha256=_sha("targets"),
            cebra_target_float32_raw_bytes_sha256=_sha("targets"),
            t4_reference_query_identity_sha256=_sha("query"),
            metric_implementation_authority_sha256=_sha("metric"),
        )


def test_receptive_field_proof_detects_future_bin_temporal_leak_without_relabeling_standard_arm_causal() -> None:
    proof = target.QueryReceptiveFieldAndTargetByteProof(
        dataset="rt", view=None, target_session_id="ses-RT-20131009",
        support_raw_indices=(0, 1), query_raw_indices=tuple(range(2, 60)),
        valid_window_start_indices=(2,),
        t4_reference_ordered_prediction_target_raw_bin_indices=(51,),
        cebra_scored_ordered_prediction_target_raw_bin_indices=(51,),
        # Synthetic default offset10-like future leakage: prediction endpoint
        # 51 reads through raw bin 56.  It is still query-disjoint, but not
        # causally exposure-matched to the endpoint reference.
        full_receptive_field_raw_indices=(tuple(range(46, 57)),),
        vendored_model_alignment_authority_sha256=_sha("offset10-alignment"),
        t4_target_float32_raw_bytes_sha256=_sha("same"),
        cebra_target_float32_raw_bytes_sha256=_sha("same"),
        t4_reference_query_identity_sha256=_sha("query"),
        metric_implementation_authority_sha256=_sha("metric"),
    ).as_dict()
    temporal = proof["vendored_model_receptive_field_alignment"]["future_raw_bins_relative_to_prediction_target"]
    assert temporal["prediction_endpoint_count_with_future_raw_bins"] == 1
    assert temporal["maximum_future_raw_bins"] == 5
    assert temporal["causal_temporal_exposure_matched"] is False
    assert temporal["online_or_latency_equivalent_language_permitted"] is False


def test_compact_receptive_field_receipt_does_not_scale_with_rows_times_rf_width() -> None:
    row_count, receptive_width = 3_000, 50
    starts = tuple(range(10, 10 + row_count))
    endpoints = tuple(start + 49 for start in starts)
    proof = target.QueryReceptiveFieldAndTargetByteProof(
        dataset="subject_m", view="sua", target_session_id="sub-M_ses-CO-20140307",
        support_raw_indices=tuple(range(10)), query_raw_indices=tuple(range(10, endpoints[-1] + 1)),
        valid_window_start_indices=starts,
        t4_reference_ordered_prediction_target_raw_bin_indices=endpoints,
        cebra_scored_ordered_prediction_target_raw_bin_indices=endpoints,
        full_receptive_field_raw_indices=tuple(tuple(range(start, start + receptive_width)) for start in starts),
        vendored_model_alignment_authority_sha256=_sha("actual-alignment"),
        t4_target_float32_raw_bytes_sha256=_sha("target-bytes"),
        cebra_target_float32_raw_bytes_sha256=_sha("target-bytes"),
        t4_reference_query_identity_sha256=_sha("query-identity"),
        metric_implementation_authority_sha256=_sha("metric"),
    ).as_dict()
    encoded = __import__("track_b_v2_contract").canonical_json_bytes(proof)
    assert proof["vendored_model_receptive_field_alignment"]["checked_prediction_endpoint_count"] == row_count
    assert proof["vendored_model_receptive_field_alignment"]["receptive_field_width_max"] == receptive_width
    # The receipt exposes only constant-size interval/index/RF authorities, not
    # 150k receptive-field integers or 3k endpoint integers.
    assert len(encoded) < 8_000


def test_target_query_cli_has_no_execute_no_data_path_or_gpu_switches() -> None:
    script = REPO_ROOT / "cebra_exploration" / "scripts" / "run_track_b_v2_target_query_scaffold.py"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=REPO_ROOT, env=env,
        text=True, capture_output=True, check=True,
    )
    assert "--execute" not in result.stdout
    assert "--target-data" not in result.stdout
    assert "--gpu" not in result.stdout
    assert "--target-session-id" in result.stdout
    rejected = subprocess.run(
        [sys.executable, str(script), "--dataset", "falcon_h1"], cwd=REPO_ROOT, env=env,
        text=True, capture_output=True,
    )
    assert rejected.returncode != 0
    assert "invalid choice" in rejected.stderr


def test_source_only_target_adapter_preflight_fails_before_target_coercion_without_official_pointer() -> None:
    class Poison:
        def __str__(self) -> str:
            raise AssertionError("missing pointer must fail before target ID coercion")

    with pytest.raises(target.TrackBV2TargetQueryScaffoldError, match="official root-audited immutable metric pointer pair"):
        target.build_source_only_target_adapter_preflight(
            dataset="subject_m", view="sua", outer_fold_id="subm-20140307", target_session_id=Poison(),
            source_authority_sha256s=_authority_map(), official_metric_pointer_pair=None,
        )


def test_source_only_target_adapter_preflight_consumes_valid_temp_pointer_without_target_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_canonical_pairs_to_tmp(monkeypatch, tmp_path)
    pair = _mint_test_pointer(dataset="subject_m", view="sua")
    preflight = target.build_source_only_target_adapter_preflight(
        dataset="subject_m", view="sua", outer_fold_id="subm-20140307",
        target_session_id="sub-M_ses-CO-20140307", source_authority_sha256s=_authority_map(),
        official_metric_pointer_pair=pair,
    )
    assert preflight["status"].startswith("OFFICIAL_POINTER_VALIDATED")
    assert preflight["official_metric_pointer_validation"]["reference_metrics"] == {
        "t4_m50_sua_mean_r2": 0.35682823575205275,
    }
    assert preflight["target_data_discovery_permitted"] is False
    assert preflight["target_data_opened"] is False
    assert preflight["cebra_trained"] is False
    assert preflight["continuous_prefix_policy_preserved"] is True
    assert preflight["causal_endpoint_policy"] == "valid_window_start_plus_49"
    assert preflight["standard_offset10_temporal_exposure"]["causal_temporal_exposure_matched"] is False
    assert preflight["accuracy_headline_readout"] == "target_support_only_standard_cebra_accuracy"
    assert preflight["posthoc_geometry_seed_or_readout_selection_permitted"] is False


def test_rt_source_only_preflight_requires_and_binds_full_15fold_source_plan_without_target_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_canonical_pairs_to_tmp(monkeypatch, tmp_path)
    pair = _mint_test_pointer(dataset="rt", view=None)
    plan = source.build_rt_15fold_source_authority_plan()
    fold = plan["outer_folds"][0]
    with pytest.raises(target.TrackBV2TargetQueryScaffoldError, match="requires the full 15-fold source-authority plan"):
        target.build_source_only_target_adapter_preflight(
            dataset="rt", view=None, outer_fold_id=fold["outer_fold_id"],
            target_session_id=fold["opaque_held_out_target_session_id"], source_authority_sha256s=_authority_map(),
            official_metric_pointer_pair=pair,
        )
    preflight = target.build_source_only_target_adapter_preflight(
        dataset="rt", view=None, outer_fold_id=fold["outer_fold_id"],
        target_session_id=fold["opaque_held_out_target_session_id"], source_authority_sha256s=_authority_map(),
        official_metric_pointer_pair=pair, rt_15fold_source_authority_plan=plan,
    )
    binding = preflight["rt_15fold_source_authority_plan_binding"]
    assert binding["outer_fold_source_session_count"] == 14
    assert binding["outer_fold_held_out_target_data_opened"] is False
    assert binding["per_fold_development_source_authority_bundles_required"] is True
    assert preflight["target_data_opened"] is False
    assert preflight["cebra_imported"] is False


def test_target_preflight_rejects_noncanonical_pointer_pair_before_target_id_coercion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _redirect_canonical_pairs_to_tmp(monkeypatch, tmp_path)
    pair = _mint_test_pointer(dataset="subject_m", view="sua")
    alias = pointer.live.ExplicitSealedReceiptPair(
        role="canonical_reference_body_pointer",
        body_path=tmp_path / "copy_of_official_pointer.json",
        sidecar_path=tmp_path / "copy_of_official_pointer.json.sha256",
    )

    class Poison:
        def __str__(self) -> str:
            raise AssertionError("noncanonical pointer must reject before target coercion")

    with pytest.raises(pointer.TrackBV2MetricPointerAuthorityError, match="one canonical output path"):
        target.build_source_only_target_adapter_preflight(
            dataset="subject_m", view="sua", outer_fold_id="subm-20140307", target_session_id=Poison(),
            source_authority_sha256s=_authority_map(), official_metric_pointer_pair=alias,
        )
    assert pair.body_path.exists()


def test_target_adapter_preflight_cli_has_no_data_execution_or_gpu_switches() -> None:
    script = REPO_ROOT / "cebra_exploration" / "scripts" / "run_track_b_v2_target_adapter_preflight.py"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=REPO_ROOT, env=env,
        text=True, capture_output=True, check=True,
    )
    assert "--execute" not in result.stdout
    assert "--target-data" not in result.stdout
    assert "--gpu" not in result.stdout
    assert "--official-pointer-body" in result.stdout
