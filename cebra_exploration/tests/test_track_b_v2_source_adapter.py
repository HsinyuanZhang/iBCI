"""Source-only tests for the additive Track-B v2 live adapter.

These tests exercise grammar, immutable-authority construction, legacy-pointer
inspection, and O_EXCL output topology.  They never invoke the real NWB
materializer, import CEBRA, load a checkpoint, score a target, or use CUDA.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "cebra_exploration" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import track_b_v2_source_adapter as source  # noqa: E402


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _rt_rows() -> tuple[source.SourceSessionMaterialization, ...]:
    rows = []
    for index, session_id in enumerate(("ses-RT-20131009", "ses-RT-20131010")):
        neural = np.full((30 + index, 4 + index), index + 1.0, dtype=np.float32)
        behavior = np.stack((np.arange(30 + index), -np.arange(30 + index)), axis=1).astype(np.float32)
        rows.append(source.SourceSessionMaterialization(
            dataset="rt", view=None, session_id=session_id,
            source_path=f"/canonical/source/{session_id}.nwb", source_nwb_sha256=_sha(session_id),
            neural=neural, dense_behavior=behavior, source_trial_count=30,
            rt_t4d_label_provenance={
                "m24_trial_event_count": 24,
                "eligible_endpoint_reach_row_count": 24,
                "unique_endpoint_coordinate_scalar_count": 48,
            },
        ))
    return tuple(rows)


def _rt_rows_for_ids(ids: tuple[str, ...]) -> tuple[source.SourceSessionMaterialization, ...]:
    """Small synthetic source arrays for one 14-session RT outer-fold bundle."""
    return tuple(
        source.SourceSessionMaterialization(
            dataset="rt", view=None, session_id=session_id,
            source_path=f"/canonical/source/{session_id}.nwb", source_nwb_sha256=_sha(session_id),
            neural=np.full((25 + index % 3, 2 + index % 4), index + 1.0, dtype=np.float32),
            dense_behavior=np.stack((np.arange(25 + index % 3), -np.arange(25 + index % 3)), axis=1).astype(np.float32),
            source_trial_count=24,
            rt_t4d_label_provenance={
                "m24_trial_event_count": 24,
                "eligible_endpoint_reach_row_count": 24,
                "unique_endpoint_coordinate_scalar_count": 48,
            },
        )
        for index, session_id in enumerate(ids)
    )


def _strict27_synthetic_subject_rows(ids: tuple[str, ...]) -> tuple[source.SourceSessionMaterialization, ...]:
    return tuple(
        source.SourceSessionMaterialization(
            dataset="subject_m", view="sua", session_id=session_id,
            source_path=f"/canonical/source/{session_id}.nwb", source_nwb_sha256=_sha(session_id),
            neural=np.full((3, 2 + index % 3), index + 1.0, dtype=np.float32),
            dense_behavior=np.asarray([[0.0, 1.0], [1.0, 0.0], [0.5, -0.5]], dtype=np.float32),
            source_trial_count=50,
        )
        for index, session_id in enumerate(ids)
    )


def test_subject_m_source_is_exact_strict27_subc_not_subm_lodo_and_scope_rejects_before_poison() -> None:
    strict27 = source._strict27_train_ids()
    assert len(strict27) == 27
    assert all(item.startswith("sub-C_ses-CO-") for item in strict27)
    full = source.canonical_source_only_adapter_spec(
        "subject_m", "sua", source_session_ids=strict27,
    )
    assert full["strict27_source_roster_exact"] is True
    assert full["canonical_source_root"].endswith("dandi_000688/sub-C")
    assert full["v9_source_manifest"]["sha256"] == source._V9_SOURCE_MANIFEST_SHA256
    smoke = source.canonical_source_only_adapter_spec(
        "subject_m", "pseudo_mua", source_session_ids=strict27[:2], source_only_smoke=True,
    )
    assert smoke["strict27_source_roster_exact"] is False
    with pytest.raises(source.TrackBV2SourceAdapterError, match="canonical dataset grammar"):
        source.canonical_source_only_adapter_spec(
            "subject_m", "sua", source_session_ids=("sub-M_ses-CO-20140307", "sub-M_ses-CO-20140308"),
            source_only_smoke=True,
        )

    class Poison:
        def __str__(self) -> str:
            raise AssertionError("target-like object must not be coerced")

    with pytest.raises(source.TrackBV2SourceAdapterError, match="no target session"):
        source.canonical_source_only_adapter_spec(
            "rt", None, source_session_ids=("ses-RT-20131009", "ses-RT-20131010"),
            target_session_id=Poison(),
        )
    with pytest.raises(source.base.TrackBV2ContractError, match="permits only"):
        source.canonical_source_only_adapter_spec(
            "falcon_h1", None, source_session_ids=("ses-RT-20131009", "ses-RT-20131010"),
        )


def test_source_bundle_has_three_distinct_authorities_dual_geometry_and_immutable_development_output(tmp_path: Path) -> None:
    rows = _rt_rows()
    request = source.canonical_source_only_adapter_spec(
        "rt", None, source_session_ids=tuple(row.session_id for row in rows),
    )
    bundle = source.build_source_only_authority_bundle(request=request, sessions=rows)
    assert set(bundle) == {
        "source_roster", "source_coverage", "source_neural_input_authority",
        "source_behavior_auxiliary_scaler_authority", "source_readout_embedding_identity_authority",
        "source_only_dual_geometry_selection_plan",
    }
    neural = bundle["source_neural_input_authority"]
    behavior = bundle["source_behavior_auxiliary_scaler_authority"]
    embedding = bundle["source_readout_embedding_identity_authority"]
    selector = bundle["source_only_dual_geometry_selection_plan"]
    assert neural["method"] == "dimension_agnostic_identity_float32_binned_counts"
    assert neural["parameter_count"] == 0
    assert neural["source_channel_vector_or_prefix_mapping_permitted"] is False
    assert behavior["behavior_auxiliary_scaler"]["fit_scope"] == "declared_source_sessions_only"
    assert embedding["readout_route"] == "source_only_consumer_mechanism_alignment"
    assert selector["separate_geometry_selection_required"] is True
    assert selector["knn_cosine_k3_selection"]["normalized_lambda"].startswith("NOT_APPLICABLE")
    pseudo_fold = selector["inner_source_folds"][0]
    held = pseudo_fold["pseudo_target_source_session_id"]
    assert held in pseudo_fold["joint_fit_session_ids"]
    assert pseudo_fold["joint_fit_session_input_blocks"]["held_source_pseudo_target_session"].endswith("M_SUPPORT_PREFIX_ONLY")
    assert pseudo_fold["pseudo_target_support"]["used_in_joint_fit"] is True
    assert pseudo_fold["pseudo_target_support"]["support_budget_trials"] == 24
    assert pseudo_fold["pseudo_target_support"]["cebra_input_sequence"]["rewarded_segments_concatenated"] is False
    assert pseudo_fold["pseudo_target_support"]["cebra_input_sequence"]["stop"] == "STOP_OF_CHRONOLOGICAL_TRIAL_24"
    assert pseudo_fold["pseudo_target_query"]["neural_in_joint_fit"] is False
    assert pseudo_fold["never_transform_unfitted_held_session"] is True
    assert pseudo_fold["adapt_true_permitted"] is False
    assert all(payload.get("target_data_opened") is False and payload.get("target_query_opened") is False
               for payload in bundle.values())
    source_row = bundle["source_coverage"]["source_sessions"][0]
    assert "label_accounting" not in source_row
    assert source_row["source_training_exposure"]["source_model_uses_full_source_session_raw_rows"] is True
    assert source_row["source_training_exposure"]["source_model_trial_count"] == 30
    assert source_row["source_training_exposure"]["target_support_budget_trials_present_in_this_source_row"] is False
    exposure = source_row["source_training_exposure"]["source_observation_exposure_vs_a2_t4"]
    assert exposure["source_observation_exposure_matched"] is False
    assert exposure["bias_direction"] == "favors_CEBRA_accuracy"
    assert exposure["rewarded_trial_only_cebra_source_sensitivity"]["status"].startswith("PREDECLARED_NOT_IMPLEMENTED")
    assert source_row["future_target_support_comparison"]["status"].startswith("NOT_MATERIALIZED")

    output_dir = tmp_path / "development_source_authority"
    receipts = source.write_development_source_only_authority_bundle(output_dir=output_dir, bundle=bundle)
    assert set(receipts) == set(bundle)
    for receipt in receipts.values():
        body = Path(receipt["body_path"])
        sidecar = Path(receipt["sidecar_path"])
        assert stat.S_IMODE(body.stat().st_mode) == 0o444
        assert stat.S_IMODE(sidecar.stat().st_mode) == 0o444
        assert sidecar.read_bytes() == f"{receipt['body_sha256']}  {body.name}\n".encode("ascii")
    with pytest.raises(source.TrackBV2SourceAdapterError, match="output directory"):
        source.write_development_source_only_authority_bundle(output_dir=output_dir, bundle=bundle)


def test_subject_m_behavior_authority_is_exact_canonical_loader_stage_not_second_zscore() -> None:
    ids = source._strict27_train_ids()
    request = source.canonical_source_only_adapter_spec("subject_m", "sua", source_session_ids=ids)
    mean = np.asarray([1.25, -2.5], dtype=np.float32)
    std = np.asarray([3.0, 4.0], dtype=np.float32)
    state = {
        "method": "canonical_fit_behavior_stats_componentwise_zscore",
        "input": "raw_binned_cursor_velocity_float32",
        "output": "load_session_with_trials_record_behavior__final_cebra_auxiliary",
        "fit_scope": "strict27_subc_co_train_only",
        "fit_session_ids": list(ids),
        "feature_dimension": 2,
        "mean_float32": mean.tolist(),
        "std_float32": std.tolist(),
        "mean_array_sha256": source._array_sha256(mean),
        "std_array_sha256": source._array_sha256(std),
        "parameter_count": 4,
    }
    request["canonical_loader_behavior_scaler"] = state | {"state_sha256": source._sha_json(state)}
    bundle = source.build_source_only_authority_bundle(
        request=request, sessions=_strict27_synthetic_subject_rows(ids),
    )
    behavior = bundle["source_behavior_auxiliary_scaler_authority"]
    assert behavior["behavior_auxiliary_method"] == "canonical_loader_final_auxiliary__no_second_stage_refit"
    assert behavior["second_behavior_refit_on_canonical_final_auxiliary_permitted"] is False
    assert behavior["behavior_auxiliary_scaler"]["mean_float32"] == mean.tolist()
    assert behavior["behavior_auxiliary_scaler"]["std_float32"] == std.tolist()


def test_metric_pointer_proposal_reads_exact_sidecarless_legacy_bodies_without_minting() -> None:
    subm = source.canonical_metric_pointer_mint_proposal("subject_m", "pseudo_mua")
    rt = source.canonical_metric_pointer_mint_proposal("rt")
    assert subm["status"].startswith("ROOT_AUDITED_IMMUTABLE_POINTER_REQUIRED")
    assert subm["rounded_literals_accepted"] is False
    assert subm["legacy_bodies_modified"] is False
    summary = subm["bodies"]["summary_cross_check_body"]
    assert summary["metric_json_pointer"] == "/summary/pseudo_mua/50/mean_r2"
    assert summary["metric_pointer_verified_against_same_fd_bytes"] is True
    assert summary["resolved_metric_values_from_verified_bytes"] == [0.3060729397667779]
    assert subm["bodies"]["per_session_seed_body"]["authority_scope"].startswith("M50_15_sessions_x_3_seeds")
    aggregate = rt["bodies"]["aggregate_metric_query_identity_body"]
    assert aggregate["metric_json_pointer"] == "/results/arms/t4d_reference/mean"
    assert aggregate["metric_pointer_verified_against_same_fd_bytes"] is True
    assert aggregate["resolved_metric_values_from_verified_bytes"] == [0.44817638439717034]
    assert aggregate["exact_per_fold_t4d_mean_from_verified_bytes"] == 0.44817638439717034
    assert aggregate["equals_per_fold_t4d_mean"] is True
    per_fold = rt["bodies"]["per_fold_t4d_body"]
    assert per_fold["metric_pointer_verified_against_same_fd_bytes"] is True
    assert len(per_fold["resolved_metric_values_from_verified_bytes"]) == 15
    assert rt["bodies"]["per_fold_t4d_body"]["path"].endswith("RT_T4D_VS_B2_D1024_FORWARD_ONLY_15FOLD_FINAL_v1.json")
    assert "not_absolute_T4d_metric" in rt["bodies"]["stage2_delta_companion"]["authority_scope"]


def test_rt_full_15fold_source_authority_plan_has_exactly_14_sources_per_opaque_held_target() -> None:
    plan = source.build_rt_15fold_source_authority_plan()
    assert plan["status"].startswith("RT_FULL_15FOLD_SOURCE_AUTHORITY_PLAN_ONLY")
    assert plan["outer_fold_count"] == 15
    all_sessions = set()
    for fold in plan["outer_folds"]:
        held = fold["opaque_held_out_target_session_id"]
        sources = tuple(fold["source_session_ids"])
        all_sessions.add(held)
        assert len(sources) == 14
        assert held not in sources
        assert fold["held_out_target_data_opened"] is False
        assert fold["held_out_target_passed_to_source_loader"] is False
        assert fold["target_support_budget_trials"] == 24
        assert fold["target_query_semantics"] == "sealed_RT_outer_q24_eligible_full_windows_only"
        prefix = fold["standard_cebra_support_sequence"]
        assert prefix["sequence_semantics"] == "one_continuous_chronological_prefix"
        assert prefix["stop"] == "STOP_OF_CHRONOLOGICAL_TRIAL_24"
        assert prefix["rewarded_segments_concatenated"] is False
    assert len(all_sessions) == 15
    assert plan["target_data_opened"] is False
    assert plan["cebra_imported"] is False
    assert plan["score_emitted"] is False

    fold = plan["outer_folds"][0]
    request = source.canonical_source_only_adapter_spec("rt", None, source_session_ids=fold["source_session_ids"])
    bundle = source.build_rt_outer_fold_source_authority_bundle(
        rt_15fold_plan=plan,
        outer_fold_id=fold["outer_fold_id"],
        request=request,
        sessions=_rt_rows_for_ids(tuple(fold["source_session_ids"])),
    )
    lineage = bundle["source_roster"]["outer_fold_lineage"]
    assert lineage["opaque_held_out_target_session_id"] == fold["opaque_held_out_target_session_id"]
    assert lineage["held_out_target_data_discovered"] is False
    assert lineage["held_out_target_data_opened"] is False
    assert lineage["held_out_target_passed_to_source_loader"] is False
    assert bundle["source_roster"]["target_session_id"] is None
    assert bundle["source_coverage"]["target_data_opened"] is False

    tampered = dict(plan)
    tampered["outer_folds"] = list(plan["outer_folds"])
    tampered["outer_folds"][0] = dict(tampered["outer_folds"][0])
    tampered["outer_folds"][0]["source_session_ids"] = tampered["outer_folds"][0]["source_session_ids"][:-1]
    with pytest.raises(source.TrackBV2SourceAdapterError, match="differs from sealed-lineage plan"):
        source.validate_rt_15fold_source_authority_plan(tampered)


def test_mutable_source_manifest_reader_rejects_open_then_rename_path_poison(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "strict_train_manifest.json"
    replacement = tmp_path / "replacement_manifest.json"
    manifest.write_bytes(b'{"source":"original"}')
    replacement.write_bytes(b'{"source":"replacement"}')
    manifest.chmod(0o664)
    replacement.chmod(0o600)
    real_fstat = source.os.fstat
    calls = 0

    def replace_after_second_fstat(descriptor: int):
        nonlocal calls
        info = real_fstat(descriptor)
        calls += 1
        if calls == 2:
            os.replace(replacement, manifest)
        return info

    monkeypatch.setattr(source.os, "fstat", replace_after_second_fstat)
    with pytest.raises(source.TrackBV2SourceAdapterError, match="pathname inode changed after read"):
        source._read_regular_file(manifest, label="mutable strict27 manifest")


def test_source_only_cli_has_no_target_or_gpu_execution_switches() -> None:
    script = REPO_ROOT / "cebra_exploration" / "scripts" / "run_track_b_v2_source_only_smoke.py"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=REPO_ROOT, env=env,
        text=True, capture_output=True, check=True,
    )
    assert "--target-session" not in result.stdout
    assert "--execute" not in result.stdout
    assert "--source-session" in result.stdout
    rejected = subprocess.run(
        [sys.executable, str(script), "--dataset", "falcon_h1", "--source-session", "ses-RT-20131009"],
        cwd=REPO_ROOT, env=env, text=True, capture_output=True,
    )
    assert rejected.returncode != 0
    assert "invalid choice" in rejected.stderr


def test_rt_15fold_plan_cli_has_no_source_data_target_or_execution_options() -> None:
    script = REPO_ROOT / "cebra_exploration" / "scripts" / "run_track_b_v2_rt_15fold_source_authority_plan.py"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=REPO_ROOT, env=env,
        text=True, capture_output=True, check=True,
    )
    assert "--target" not in result.stdout
    assert "--source" not in result.stdout
    assert "--execute" not in result.stdout
    assert "gpu" not in result.stdout.lower()
    rendered = subprocess.run(
        [sys.executable, str(script)], cwd=REPO_ROOT, env=env,
        text=True, capture_output=True, check=True,
    )
    assert '"target_data_opened":false' in rendered.stdout
    assert '"score_emitted":false' in rendered.stdout
