"""Synthetic/no-target adversarial tests for the RT one-cell successor."""
from __future__ import annotations

import copy
import hashlib
import inspect
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
import track_b_v2_rt_one_cell_live_executor as live  # noqa: E402
import track_b_v2_post_synthetic_runtime_control_authority as control_authority  # noqa: E402


CLI = REPO_ROOT / "cebra_exploration/scripts/run_track_b_v2_rt_one_cell_live_executor.py"


def _sign(payload: dict, key: str) -> dict:
    payload[key] = hashlib.sha256(base.canonical_json_bytes(payload)).hexdigest()
    return payload


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _control_evidence() -> dict:
    return control_authority.load_raw_bound_evidence()[0]


def _minimal_config(*, fold: int = 0, seed: int = 42) -> dict:
    payload = {
        "schema": live.CONFIG_SCHEMA,
        "cell_key": {
            "outer_fold_index": fold,
            "outer_fold_id": f"rt_outer_fold_{fold:02d}",
            "target_session_id": f"ses-RT-synthetic-{fold:02d}",
            "seed": seed,
        },
        "fixed_scientific_contract": {"source_stream_count": 14},
        "source_authority": {
            "source_session_ids": [f"source-{index:02d}" for index in range(14)],
        },
        "sealed_rt_t4_target_query_lineage": {
            "query_window_count": 9,
            "sealed_query_identity": {
                "ordered_window_start_sha256": _sha("starts"),
                "ordered_target_covariate_evalmask_sha256": _sha("target-mask"),
                "ordered_query_identity_sha256": _sha("query"),
            },
        },
    }
    return _sign(payload, "canonical_config_sha256")


def _target(config: dict) -> dict:
    query_lineage = config["sealed_rt_t4_target_query_lineage"]
    payload = {
        "schema": live.TARGET_RECEIPT_SCHEMA,
        "status": "RT_M24_SUPPORT_AND_STRICT_POST_M_QUERY_MATERIALIZED__NO_FIT",
        "cell_key": config["cell_key"],
        "canonical_config_sha256": config["canonical_config_sha256"],
        "formal": False,
        "rt_local_asset_authority_body_sha256": live.LOCAL_ASSET_AUTHORITY_SHA256,
        "canonical_internal_opener": (
            "track_b_v2_rt_local_asset_authority.open_canonical_verified_asset_after_authority"
        ),
        "caller_target_path_ledger_sha_or_bytes_used": False,
        "held_fd_private_snapshot": {
            "source_opened_once_O_NOFOLLOW": True,
            "source_size_sha_verified_on_same_fd": True,
            "snapshot_created_O_EXCL_0600_then_fsync_0444": True,
            "parser_consumed_held_snapshot_fd_not_path_reopen": True,
            "source_and_snapshot_inode_revalidated_after_parse": True,
        },
        "support": {
            "rewarded_trial_ordinals": list(range(24)),
            "continuous_prefix": True,
            "all_intervening_rows": True,
            "rewarded_segments_concatenated": False,
        },
        "query": {
            "query_window_count": query_lineage["query_window_count"],
            "sealed_query_identity": query_lineage["sealed_query_identity"],
            "strict_post_M24": True,
            "query_enters_any_fit": False,
            "prediction_endpoints_equal_valid_starts_plus_49": True,
            "every_offset10_RF_wholly_in_query": True,
            "support_query_boundary_crossings": 0,
            "offset10_strictly_future_bins": 4,
            "ordered_t4_target_float32_bytes_sha256": _sha("target-bytes"),
            "ordered_cebra_target_float32_bytes_sha256": _sha("target-bytes"),
        },
        "support_neural_float32_sha256": _sha("support-neural"),
        "support_dense_velocity_float32_sha256": _sha("support-velocity"),
        "query_neural_float32_sha256": _sha("query-neural"),
        "ordered_endpoint_int64_sha256": _sha("endpoints"),
        "full_offset10_RF_int64_sha256": _sha("rf"),
        "target_backprop_or_update": False,
        "score_emitted": False,
    }
    return _sign(payload, "target_materialization_receipt_sha256")


def _alignment(config: dict) -> dict:
    def block(role: str, rows: int, *, session_id: str | None = None) -> dict:
        payload = {
            "block_role": role,
            "padded_transform_rows": rows + 10,
            "valid_rows_after_separate_5_to_minus5_crop": rows,
            "embedding_row_count": rows,
            "auxiliary_label_row_count": rows,
            "ordered_valid_endpoint_int64_sha256": _sha(f"{role}-endpoint"),
            "full_valid_offset10_RF_int64_sha256": _sha(f"{role}-rf"),
            "valid_embedding_float32_sha256": _sha(f"{role}-embedding"),
            "aligned_auxiliary_float32_sha256": _sha(f"{role}-aux"),
        }
        if session_id is not None:
            payload["session_id"] = session_id
        return payload

    payload = {
        "schema": "track_b_v2_rt_offset10_valid_transform_alignment_v1",
        "offset_crop": {"left": 5, "right": 5},
        "each_block_cropped_separately": True,
        "concatenate_then_crop_used": False,
        "padded_edge_rows_used": 0,
        "source_blocks": [block("source_full", 20 + index, session_id=session)
                          for index, session in enumerate(config["source_authority"]["source_session_ids"])],
        "held_M24_support_block": block("held_M24_support", 13) | {"support_trial_count": 24},
        "strict_post_M24_query_block": block(
            "strict_post_M24_query",
            config["sealed_rt_t4_target_query_lineage"]["query_window_count"],
        ) | {"enters_any_fit": False, "RF_wholly_inside_query": True},
    }
    return _sign(payload, "alignment_authority_sha256")


def _encoder(config: dict, target_sha: str) -> dict:
    alignment = _alignment(config)
    payload = {
        "schema": live.ENCODER_RECEIPT_SCHEMA,
        "status": "ONE_JOINT_RT_ENCODER_TERMINAL__READY_FOR_SIX_READOUTS",
        "cell_key": config["cell_key"],
        "canonical_config_sha256": config["canonical_config_sha256"],
        "target_materialization_receipt_sha256": target_sha,
        "geometry": {"output_dimension": 8, "iterations": 10_000},
        "seed": config["cell_key"]["seed"],
        "source_session_ids": config["source_authority"]["source_session_ids"],
        "source_stream_count": 14,
        "held_support_trials": 24,
        "fit_stream_count": 15,
        "fit_count": 1,
        "held_query_enters_fit": False,
        "encoder_state_sha256": _sha("encoder"),
        "fit_index_authority_sha256": _sha("fit-indices"),
        "valid_transform_alignment": alignment,
        "valid_transform_alignment_sha256": alignment["alignment_authority_sha256"],
        "target_backprop_or_update_after_fit": False,
        "score_emitted": False,
    }
    return _sign(payload, "encoder_receipt_sha256")


def _score(config: dict, target_sha: str, encoder: dict, route: str, decoder: str) -> dict:
    scopes = {
        "source_only_consumer_mechanism_alignment": "source_fit_only",
        "target_support_only_standard_cebra_accuracy": "held_target_M24_support_only",
        "source_plus_target_support_hybrid_sensitivity": "source_fit_plus_held_target_M24_support",
    }
    payload = {
        "schema": live.SCORE_RECEIPT_SCHEMA,
        "status": "ONE_RT_ROUTE_DECODER_SCORE_TERMINAL",
        "cell_key": config["cell_key"],
        "canonical_config_sha256": config["canonical_config_sha256"],
        "route": route,
        "decoder": decoder,
        "target_materialization_receipt_sha256": target_sha,
        "encoder_receipt_sha256": encoder["encoder_receipt_sha256"],
        "encoder_state_sha256": encoder["encoder_state_sha256"],
        "valid_transform_alignment_sha256": encoder["valid_transform_alignment_sha256"],
        "sealed_query_identity": config["sealed_rt_t4_target_query_lineage"]["sealed_query_identity"],
        "ordered_t4_target_float32_bytes_sha256": _sha("target-bytes"),
        "ordered_cebra_target_float32_bytes_sha256": _sha("target-bytes"),
        "ordered_prediction_float32_bytes_sha256": _sha(f"prediction-{route}-{decoder}"),
        "r2_variance_weighted": 0.125,
        "metric_runtime": {
            "implementation": "torchmetrics.regression.R2Score",
            "torchmetrics_version": "1.5.1",
            "multioutput": "variance_weighted",
            "dtype": "float32",
            "device": "cpu",
            "update_scope": "one_complete_ordered_fold_session_query_then_compute_once",
            "update_call_count": 1,
            "compute_call_count": 1,
            "pooled_query_rows_across_folds": False,
            "custom_numpy_float64_clone_used": False,
            "prediction_shape": [config["sealed_rt_t4_target_query_lineage"]["query_window_count"], 2],
            "target_shape": [config["sealed_rt_t4_target_query_lineage"]["query_window_count"], 2],
            "prediction_float32_bytes_sha256": _sha(f"prediction-{route}-{decoder}"),
            "target_float32_bytes_sha256": _sha("target-bytes"),
        },
        "query_enters_fit": False,
        "target_backprop_or_encoder_update": False,
        "formal": False,
        "readout_fit_scope": scopes[route],
    }
    return _sign(payload, "score_receipt_sha256")


def _terminal_bundle(config: dict) -> tuple[dict, dict, list[dict], dict]:
    target = _target(config)
    encoder = _encoder(config, target["target_materialization_receipt_sha256"])
    scores = [
        _score(config, target["target_materialization_receipt_sha256"], encoder, route, decoder)
        for route in live.ROUTES for decoder in live.DECODERS
    ]
    payload = {
        "schema": live.TERMINAL_RECEIPT_SCHEMA,
        "status": "ONE_RT_CELL_TERMINAL__SIX_MANDATORY_SCORES_COMPLETE",
        "cell_key": config["cell_key"],
        "canonical_config_sha256": config["canonical_config_sha256"],
        "target_materialization_receipt_sha256": target["target_materialization_receipt_sha256"],
        "encoder_receipt_sha256": encoder["encoder_receipt_sha256"],
        "encoder_state_sha256": encoder["encoder_state_sha256"],
        "valid_transform_alignment_sha256": encoder["valid_transform_alignment_sha256"],
        "score_receipt_sha256_by_role": {
            f"{row['route']}__{row['decoder']}": row["score_receipt_sha256"] for row in scores
        },
        "scores_r2_by_role": {
            f"{row['route']}__{row['decoder']}": row["r2_variance_weighted"] for row in scores
        },
        "target_backprop_or_update": False,
        "formal": False,
        "metric_aggregation_scope": "one_R2_per_fold_session_seed_route_decoder__aggregate_session_then_seed",
        "pooled_query_rows_across_folds": False,
    }
    return target, encoder, scores, _sign(payload, "terminal_receipt_sha256")


def test_real_no_target_preflight_exact_fold_source_asset_t4_and_cost_no_go(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def poison(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("RT target opener must not run during preflight")

    monkeypatch.setattr(live.rt_asset_authority, "open_canonical_verified_asset_after_authority", poison)
    plan = live.build_no_target_preflight(outer_fold_index=0, seed=42)
    assert plan["status"] == live.STATUS_CONTROL_NO_GO
    config = plan["canonical_config"]
    assert config["cell_key"] == {
        "outer_fold_index": 0, "outer_fold_id": "rt_outer_fold_00",
        "target_session_id": "ses-RT-20131009", "seed": 42,
    }
    assert config["source_authority"]["source_session_count"] == 14
    assert len(config["source_authority"]["source_session_ids"]) == 14
    assert "ses-RT-20131009" not in config["source_authority"]["source_session_ids"]
    lineage = config["sealed_rt_t4_target_query_lineage"]
    assert lineage["target_nwb_sha256"] == "5bd05b43ac590c70a3edb4e6fc4d0614dfe3e14d97ab5db54137dce5d6e0b11c"
    assert lineage["query_window_count"] == 24632
    assert lineage["sealed_query_identity"]["ordered_query_identity_sha256"] == (
        "2d99f773b83fbd91dbe2fe56e61d0080d197457ccc3af5428a16a43dfafa3806"
    )
    assert plan["fixed_gpu_cost_gate"]["canonical_body_sha256"] == (
        "ec7096a5e54e444fd6cdafa241aaa88a0720143e3c4e42e3565f022ee662c8e2"
    )
    assert plan["root_frozen_runtime_control_gate"]["status"] == live.STATUS_CONTROL_NO_GO
    assert plan["root_frozen_runtime_control_gate"]["control_scale_and_threshold"].startswith("ROOT_PENDING")
    for field in ("target_data_opened", "target_query_opened", "formal_data_opened",
                  "cebra_imported", "cebra_fit_called", "gpu_used", "score_emitted", "receipt_minted"):
        assert plan[field] is False


def test_all_45_internal_cell_keys_are_canonical_and_unique() -> None:
    keys = [live.canonical_cell_key(outer_fold_index=fold, seed=seed)
            for fold in range(15) for seed in live.SEEDS]
    assert len(keys) == len(set((key.outer_fold_id, key.target_session_id, key.seed) for key in keys)) == 45
    assert {key.seed for key in keys} == {42, 43, 44}
    assert len({key.target_session_id for key in keys}) == 15
    with pytest.raises(live.TrackBV2RTOneCellError, match="0..14"):
        live.canonical_cell_key(outer_fold_index=15, seed=42)
    with pytest.raises(live.TrackBV2RTOneCellError, match="42,43,44"):
        live.canonical_cell_key(outer_fold_index=0, seed=41)


def test_config_fixes_geometry_joint_encoder_routes_offset_and_canonical_topology() -> None:
    config = live.build_canonical_config(outer_fold_index=0, seed=42)
    fixed = config["fixed_scientific_contract"]
    assert (fixed["output_dimension"], fixed["iterations"]) == (8, 10_000)
    assert fixed["encoder_fit_streams"].startswith("exact_14_full_source_streams")
    assert fixed["linear_ridge_normalized_lambda"] == 0.01
    assert fixed["cosine_knn_k"] == 3
    assert fixed["readout_routes"] == list(live.ROUTES)
    assert fixed["decoders"] == list(live.DECODERS)
    assert fixed["same_encoder_state_for_all_six_scores"] is True
    assert fixed["query_enters_encoder_or_readout_fit"] is False
    assert fixed["offset10"]["half_open_offsets_relative_to_prediction_target"] == [-5, 5]
    assert fixed["offset10"]["strictly_future_raw_bins"] == 4
    assert fixed["offset10"]["causal"] is False
    topology = config["output_topology"]
    assert topology["cell_root"].endswith("/cells/rt_outer_fold_00/seed_42")
    assert len(topology["score_receipts"]) == 6
    assert topology["caller_output_override_permitted"] is False
    assert config["development_pilot"] is True
    assert config["execution_scope"] == "SOLE_STAGE_P_DEVELOPMENT_PILOT__RT_OUTER_FOLD_00_SEED42"
    assert config["fold_or_seed_substitution_permitted_for_actual_pilot"] is False
    assert config["root_frozen_protocol"] == {
        "path": str(live.ROOT_FROZEN_PROTOCOL),
        "sha256": live.ROOT_FROZEN_PROTOCOL_SHA256,
        "section": "Root-frozen first live cells and score semantics (2026-08-15)",
    }


def test_sealed_t4_per_fold_row_binding_tamper_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    key = live.canonical_cell_key(outer_fold_index=0, seed=42)
    original = live._read_named_json

    def tampered(path: Path, expected_sha: str, *, label: str) -> dict:
        payload = original(path, expected_sha, label=label)
        if Path(path) == live.RT_T4_QUERY_BODY:
            payload = copy.deepcopy(payload)
            payload["per_fold"][0]["sealed_query_identity"]["ordered_query_identity_sha256"] = _sha("poison")
        return payload

    monkeypatch.setattr(live, "_read_named_json", tampered)
    with pytest.raises(live.TrackBV2RTOneCellError, match="row/query-identity binding mismatch"):
        live._sealed_t4_lineage(
            key=key,
            expected_asset_sha256="5bd05b43ac590c70a3edb4e6fc4d0614dfe3e14d97ab5db54137dce5d6e0b11c",
        )


def test_no_public_caller_target_session_path_ledger_sha_or_output_override_surface() -> None:
    assert set(inspect.signature(live.build_no_target_preflight).parameters) == {"outer_fold_index", "seed"}
    assert set(inspect.signature(live.build_canonical_config).parameters) == {"outer_fold_index", "seed"}
    completed = subprocess.run([sys.executable, str(CLI), "--help"], capture_output=True, text=True, check=True)
    text = completed.stdout.lower()
    for forbidden in ("--target", "--session", "--path", "--ledger", "--sha", "--bytes", "--output"):
        assert forbidden not in text


def test_actual_preflight_and_execute_reject_fold_or_seed_substitution_before_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        live, "build_canonical_config",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("authority/config reached")),
    )
    for fold, seed in ((1, 42), (0, 43), (14, 44)):
        with pytest.raises(live.TrackBV2RTOneCellError, match="frozen to rt_outer_fold_00 seed42"):
            live.build_no_target_preflight(outer_fold_index=fold, seed=seed)
        with pytest.raises(live.TrackBV2RTOneCellError, match="frozen to rt_outer_fold_00 seed42"):
            live.execution_is_not_authorized_this_successor(outer_fold_index=fold, seed=seed)
    cli = subprocess.run(
        [sys.executable, str(CLI), "--outer-fold-index", "1", "--seed", "42"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert cli.returncode != 0 and "invalid choice" in cli.stderr


@pytest.mark.parametrize("flag", ("--execute", "--i-have-root-review"))
def test_runner_single_execution_flag_rejected_before_preflight(flag: str) -> None:
    result = subprocess.run(
        [sys.executable, str(CLI), "--outer-fold-index", "0", "--seed", "42", flag],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "requires both" in result.stderr


def test_real_cli_default_is_no_target_no_write_and_dual_flags_stop_on_cost() -> None:
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    output_root_existed = os.path.lexists(live.RESULT_ROOT)
    dry = subprocess.run(
        [sys.executable, str(CLI), "--outer-fold-index", "0", "--seed", "42"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=True,
    )
    payload = json.loads(dry.stdout)
    assert payload["status"] == live.STATUS_CONTROL_NO_GO
    assert os.path.lexists(live.RESULT_ROOT) is output_root_existed
    execute = subprocess.run(
        [sys.executable, str(CLI), "--outer-fold-index", "0", "--seed", "42",
         "--execute", "--i-have-root-review"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True,
    )
    assert execute.returncode != 0
    assert "not jointly valid" in execute.stderr
    assert os.path.lexists(live.RESULT_ROOT) is output_root_existed


def test_valid_cost_but_missing_root_frozen_runtime_control_still_no_go_before_opener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = {
        "schema": "track_b_v2_subject_m_fixed_gpu_cost_gate_v1",
        "status": "FIXED_D8IT250_GPU_COST_RECEIPT_VALID__REQUIRED_BUT_NOT_SUFFICIENT_FOR_TARGET_EXECUTION",
        "canonical_body_sha256": _sha("cost"), "canonical_sidecar_sha256": _sha("cost-side"),
    }
    monkeypatch.setattr(live.subject_executor, "inspect_fixed_d8it250_gpu_cost_receipt", lambda: valid)
    monkeypatch.setattr(
        live.rt_asset_authority, "open_canonical_verified_asset_after_authority",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("target opened")),
    )
    plan = live.build_no_target_preflight(outer_fold_index=0, seed=42)
    assert plan["status"] == live.STATUS_CONTROL_NO_GO
    assert plan["root_frozen_runtime_control_gate"]["body_lexists"] is False
    with pytest.raises(live.TrackBV2RTOneCellError, match="not jointly valid"):
        live.execution_is_not_authorized_this_successor(outer_fold_index=0, seed=42)


def test_valid_cost_and_control_still_require_official_pairs_before_opener(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid_cost = {
        "schema": "track_b_v2_subject_m_fixed_gpu_cost_gate_v1",
        "status": "FIXED_D8IT250_GPU_COST_RECEIPT_VALID__REQUIRED_BUT_NOT_SUFFICIENT_FOR_TARGET_EXECUTION",
        "canonical_body_sha256": _sha("cost"), "canonical_sidecar_sha256": _sha("cost-side"),
    }
    valid_control = {
        "schema": live.RUNTIME_CONTROL_SCHEMA, "status": live.RUNTIME_CONTROL_STATUS,
        "canonical_body_sha256": _sha("control"), "target_access_permitted": False,
    }
    monkeypatch.setattr(live.subject_executor, "inspect_fixed_d8it250_gpu_cost_receipt", lambda: valid_cost)
    monkeypatch.setattr(live, "inspect_root_frozen_runtime_control_pair", lambda **_kwargs: valid_control)
    monkeypatch.setattr(
        live.rt_asset_authority, "open_canonical_verified_asset_after_authority",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("target opened")),
    )
    assert live.build_no_target_preflight(outer_fold_index=0, seed=42)["status"] == live.STATUS_REVIEW_ONLY
    with pytest.raises(live.TrackBV2RTOneCellError):
        live.execution_is_not_authorized_this_successor(outer_fold_index=0, seed=42)


def test_runtime_control_pair_requires_exact_post_cost_scale_threshold_and_hard_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cost = {"canonical_body_sha256": _sha("cost")}
    output = tmp_path / "control.json"
    evidence = _control_evidence()
    groups = evidence["target_support_only_groups"]
    positive_min = min(groups[f"{arm}__{decoder}"]["min"]
                       for arm in ("cebra_joint_behavior", "cebra_frozen_source_adapt")
                       for decoder in ("linear_ridge", "knn_cosine_k3"))
    ordinary_max = max(groups[f"cebra_adapt_unaligned__{decoder}"]["max"]
                       for decoder in ("linear_ridge", "knn_cosine_k3"))
    deranged_max = max(groups[
        f"cebra_joint_behavior__target_support_auxiliary_rows_deranged__{decoder}"]["max"]
                       for decoder in ("linear_ridge", "knn_cosine_k3"))
    payload = {
        "schema": live.RUNTIME_CONTROL_SCHEMA,
        "status": live.RUNTIME_CONTROL_STATUS,
        "fixed_geometry": {"output_dimension": 8, "iterations": 10_000},
        "vendored_cebra_version": "0.6.1",
        "minted_after_fixed_gpu_cost_review": True,
        "fixed_gpu_cost_receipt_body_sha256": cost["canonical_body_sha256"],
        "decision_threshold_r2": 0.70,
        "positive_threshold_origin": live.POSITIVE_THRESHOLD_ORIGIN,
        "deranged_hard_null_threshold_r2": 0.60,
        "hard_null_threshold_origin": live.HARD_NULL_THRESHOLD_ORIGIN,
        "controls": {
            "positive_control_pass": True,
            "ordinary_unaligned_distribution_reported": True,
            "ordinary_unaligned_role": "diagnostic_distribution_only__not_a_hard_null_gate",
            "deranged_support_hard_null_pass": True,
            "deranged_permutation_fixed_seed_independent_nonidentity": True,
            "thresholds_frozen_before_target": True,
        },
        "control_scale_and_threshold": {
            "frozen_after_cost_review_before_target": True,
            "synthetic_signal_scale": 1.0,
            "positive_control_threshold_r2": 0.70,
            "deranged_hard_null_threshold_r2": 0.60,
            "positive_threshold_origin": live.POSITIVE_THRESHOLD_ORIGIN,
            "hard_null_threshold_origin": live.HARD_NULL_THRESHOLD_ORIGIN,
            "positive_min": float(positive_min),
            "ordinary_negative_max": float(ordinary_max),
            "deranged_hard_null_max": float(deranged_max),
        },
        "raw_bound_control_evidence": evidence,
        "ordinary_unaligned_is_threshold_or_pass_gate": False,
        "target_data_opened": False,
        "formal_data_opened": False,
        "score_emitted": False,
    }
    closure = control_authority.implementation_closure()
    payload["publisher_implementation_closure_at_launch"] = closure
    payload["publisher_implementation_closure_at_final"] = closure
    payload["publisher_launch_final_live_closure_equal"] = True
    monkeypatch.setattr(live, "RUNTIME_CONTROL_BODY", output)
    live._write_immutable_pair_for_root_review(output, payload, root_reviewed=True)
    checked = live.inspect_root_frozen_runtime_control_pair(cost_gate=cost)
    assert checked["status"] == live.RUNTIME_CONTROL_STATUS
    assert checked["control_scale_and_threshold"] == payload["control_scale_and_threshold"]
    bad = copy.deepcopy(payload)
    bad["control_scale_and_threshold"]["frozen_after_cost_review_before_target"] = False
    output.chmod(0o644)
    output.unlink()
    output.with_name(f"{output.name}.sha256").chmod(0o644)
    output.with_name(f"{output.name}.sha256").unlink()
    live._write_immutable_pair_for_root_review(output, bad, root_reviewed=True)
    with pytest.raises(live.TrackBV2RTOneCellError, match="scale/threshold"):
        live.inspect_root_frozen_runtime_control_pair(cost_gate=cost)


def test_canonical_publication_missing_cost_fails_before_directory_or_file_creation() -> None:
    key = live.canonical_cell_key(outer_fold_index=0, seed=42)
    cell = live.canonical_cell_dir(key)
    existed = os.path.lexists(cell)
    with pytest.raises(live.TrackBV2RTOneCellError, match="cost gate is valid"):
        live.publish_canonical_config_and_preflight_for_root_review(
            outer_fold_index=0, seed=42, root_reviewed=True,
        )
    assert os.path.lexists(cell) is existed


def test_target_encoder_six_scores_and_terminal_validators_accept_exact_synthetic_null() -> None:
    config = _minimal_config()
    target, encoder, scores, terminal = _terminal_bundle(config)
    terminal_sha = live.validate_terminal_receipt(
        terminal, config, target_payload=target, encoder_payload=encoder, score_payloads=scores,
    )
    assert terminal_sha == terminal["terminal_receipt_sha256"]


@pytest.mark.parametrize(
    "mutator, message",
    (
        (lambda target: target["query"].__setitem__("query_window_count", 10), "sealed T4 identity"),
        (lambda target: target["query"].__setitem__("support_query_boundary_crossings", 1), "endpoint/offset10"),
        (lambda target: target["query"].__setitem__("ordered_cebra_target_float32_bytes_sha256", _sha("poison")),
         "not exact-equal"),
        (lambda target: target["held_fd_private_snapshot"].__setitem__(
            "parser_consumed_held_snapshot_fd_not_path_reopen", False), "private-snapshot"),
    ),
)
def test_target_materialization_adversarial_drift_fails_closed(mutator, message: str) -> None:
    config = _minimal_config()
    target = _target(config)
    mutator(target)
    with pytest.raises(live.TrackBV2RTOneCellError, match=message):
        live.validate_target_materialization_receipt(target, config)


def test_encoder_rejects_source_roster_query_fit_and_second_fit() -> None:
    config = _minimal_config()
    target = _target(config)
    encoder = _encoder(config, target["target_materialization_receipt_sha256"])
    for field, poison, message in (
        ("source_session_ids", encoder["source_session_ids"][:-1], "exact14"),
        ("held_query_enters_fit", True, "scope/count"),
        ("fit_count", 2, "scope/count"),
    ):
        bad = copy.deepcopy(encoder)
        bad[field] = poison
        with pytest.raises(live.TrackBV2RTOneCellError, match=message):
            live.validate_encoder_receipt(
                bad, config, target_receipt_sha256=target["target_materialization_receipt_sha256"],
            )

    bad_alignment = copy.deepcopy(encoder)
    bad_alignment["valid_transform_alignment"]["concatenate_then_crop_used"] = True
    with pytest.raises(live.TrackBV2RTOneCellError, match="crop policy"):
        live.validate_encoder_receipt(
            bad_alignment, config,
            target_receipt_sha256=target["target_materialization_receipt_sha256"],
        )
    bad_parity = copy.deepcopy(encoder)
    bad_parity["valid_transform_alignment"]["source_blocks"][3]["auxiliary_label_row_count"] += 1
    with pytest.raises(live.TrackBV2RTOneCellError, match="row parity"):
        live.validate_encoder_receipt(
            bad_parity, config,
            target_receipt_sha256=target["target_materialization_receipt_sha256"],
        )


def test_score_rejects_encoder_state_query_target_bytes_route_scope_and_updates() -> None:
    config = _minimal_config()
    target = _target(config)
    encoder = _encoder(config, target["target_materialization_receipt_sha256"])
    score = _score(config, target["target_materialization_receipt_sha256"], encoder,
                   live.ROUTES[0], live.DECODERS[0])
    mutations = (
        ("encoder_state_sha256", _sha("other"), "lineage"),
        ("sealed_query_identity", {}, "query identity"),
        ("ordered_cebra_target_float32_bytes_sha256", _sha("other-target"), "not exact"),
        ("readout_fit_scope", "target_support_only", "scope"),
        ("target_backprop_or_encoder_update", True, "update"),
    )
    for field, poison, message in mutations:
        bad = copy.deepcopy(score)
        bad[field] = poison
        with pytest.raises(live.TrackBV2RTOneCellError, match=message):
            live.validate_score_receipt(
                bad, config,
                target_receipt_sha256=target["target_materialization_receipt_sha256"],
                encoder_receipt_sha256=encoder["encoder_receipt_sha256"],
                encoder_state_sha256=encoder["encoder_state_sha256"],
                valid_transform_alignment_sha256=encoder["valid_transform_alignment_sha256"],
            )


@pytest.mark.parametrize(
    "field,value",
    (
        ("torchmetrics_version", "1.6.0"),
        ("dtype", "float64"),
        ("device", "cuda:0"),
        ("pooled_query_rows_across_folds", True),
        ("custom_numpy_float64_clone_used", True),
        ("update_call_count", 2),
    ),
)
def test_score_rejects_metric_version_dtype_device_pooling_clone_or_update_drift(field: str, value: object) -> None:
    config = _minimal_config()
    target = _target(config)
    encoder = _encoder(config, target["target_materialization_receipt_sha256"])
    score = _score(config, target["target_materialization_receipt_sha256"], encoder,
                   live.ROUTES[0], live.DECODERS[0])
    score["metric_runtime"][field] = value
    with pytest.raises(live.TrackBV2RTOneCellError, match="TorchMetrics 1.5.1"):
        live.validate_score_receipt(
            score, config,
            target_receipt_sha256=target["target_materialization_receipt_sha256"],
            encoder_receipt_sha256=encoder["encoder_receipt_sha256"],
            encoder_state_sha256=encoder["encoder_state_sha256"],
            valid_transform_alignment_sha256=encoder["valid_transform_alignment_sha256"],
        )


def test_terminal_rejects_missing_or_duplicate_route_decoder() -> None:
    config = _minimal_config()
    target, encoder, scores, terminal = _terminal_bundle(config)
    with pytest.raises(live.TrackBV2RTOneCellError, match="exact six"):
        live.validate_terminal_receipt(
            terminal, config, target_payload=target, encoder_payload=encoder, score_payloads=scores[:-1],
        )
    duplicated = scores[:-1] + [scores[0]]
    with pytest.raises(live.TrackBV2RTOneCellError, match="duplicate"):
        live.validate_terminal_receipt(
            terminal, config, target_payload=target, encoder_payload=encoder, score_payloads=duplicated,
        )


def test_aggregate_exact_45_grid_and_tamper_fail_closed() -> None:
    configs: dict[str, dict] = {}
    terminals: list[dict] = []
    for fold in range(15):
        for seed in live.SEEDS:
            config = _minimal_config(fold=fold, seed=seed)
            _, _, _, terminal = _terminal_bundle(config)
            cell_id = f"rt_outer_fold_{fold:02d}__seed_{seed}"
            configs[cell_id] = config
            terminals.append(terminal)
    aggregate = live.build_terminal_aggregate(terminal_records=terminals, config_by_cell=configs)
    assert aggregate["cell_count"] == 45
    assert aggregate["formal"] is False
    assert aggregate["aggregation_policy"].startswith("session_then_seed")
    assert aggregate["interface_only_does_not_authorize_45cell_queue_or_execution"] is True
    assert aggregate["one_R2_computed_per_fold_session_seed_route_decoder"] is True
    assert aggregate["pooled_query_rows_across_folds"] is False
    with pytest.raises(live.TrackBV2RTOneCellError, match="exact 15"):
        live.build_terminal_aggregate(terminal_records=terminals[:-1], config_by_cell=configs)
    poisoned = copy.deepcopy(terminals)
    poisoned[0]["scores_r2_by_role"][next(iter(poisoned[0]["scores_r2_by_role"]))] = 999.0
    with pytest.raises(live.TrackBV2RTOneCellError, match="payload"):
        live.build_terminal_aggregate(terminal_records=poisoned, config_by_cell=configs)


def test_transactional_immutable_writer_0444_sidecar_conflict_and_no_partial(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    result = live._write_immutable_pair_for_root_review(output, {"schema": "synthetic"}, root_reviewed=True)
    sidecar = Path(result["sidecar_path"])
    assert stat_mode(output) == stat_mode(sidecar) == 0o444
    assert sidecar.read_text() == f"{result['body_sha256']}  receipt.json\n"
    with pytest.raises(live.TrackBV2RTOneCellError, match="fresh"):
        live._write_immutable_pair_for_root_review(output, {"schema": "other"}, root_reviewed=True)
    assert json.loads(output.read_bytes()) == {"schema": "synthetic"}

    orphan_sidecar = tmp_path / "orphan.json.sha256"
    orphan_sidecar.write_text("poison")
    with pytest.raises(live.TrackBV2RTOneCellError, match="fresh"):
        live._write_immutable_pair_for_root_review(
            tmp_path / "orphan.json", {"schema": "x"}, root_reviewed=True,
        )
    assert not (tmp_path / "orphan.json").exists()

    body_only = tmp_path / "body-only.json"
    body_only.write_text("poison")
    with pytest.raises(live.TrackBV2RTOneCellError, match="fresh"):
        live._write_immutable_pair_for_root_review(
            body_only, {"schema": "x"}, root_reviewed=True,
        )
    assert not (tmp_path / "body-only.json.sha256").exists()


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_output_freshness_rejects_body_or_sidecar_before_future_target_access(tmp_path: Path) -> None:
    config = _minimal_config()
    score_paths = {
        f"{route}__{decoder}": str(tmp_path / "scores" / f"{route}__{decoder}.json")
        for route in live.ROUTES for decoder in live.DECODERS
    }
    (tmp_path / "scores").mkdir()
    config["output_topology"] = {
        "target_materialization_receipt": str(tmp_path / "target.json"),
        "joint_encoder_receipt": str(tmp_path / "encoder.json"),
        "terminal_receipt": str(tmp_path / "terminal.json"),
        "score_receipts": score_paths,
    }
    live.assert_future_result_outputs_fresh(config)
    (tmp_path / "encoder.json.sha256").write_text("poison")
    with pytest.raises(live.TrackBV2RTOneCellError, match="both be fresh"):
        live.assert_future_result_outputs_fresh(config)


def test_source_static_boundary_has_no_target_parser_cebra_torch_fit_gpu_score_or_glob_import() -> None:
    text = Path(live.__file__).read_text()
    for forbidden in (
        "import cebra", "import torch", "import sklearn", "import pynwb", "import h5py",
        "NWBHDF5IO", "np.load", "os.listdir", "os.scandir", ".glob(", ".rglob(",
    ):
        assert forbidden not in text
    assert text.count("open_canonical_verified_asset_after_authority(") == 1
    assert "_canonical_held_asset_context_for_separately_reviewed_runtime" in text
