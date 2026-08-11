"""Synthetic/no-NWB tests for strict RT Stage-R R-C versus R-S aggregation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import aggregate_rt_stage_r_b2_d1024 as module
from scripts import rt_stage_r_b2_d1024_fold10_race_recovery_checker as recovery_checker


def _write(path: Path, value: object, *, immutable: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    if immutable:
        path.chmod(0o444)
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rc_fixture(tmp_path: Path) -> tuple[Path, Path, dict[int, dict[str, object]]]:
    sessions = {str(fold): f"session-{fold}" for fold in module.FOLDS}
    windows = {str(fold): 1000 + fold for fold in module.FOLDS}
    rows: dict[int, dict[str, object]] = {}
    for fold in module.FOLDS:
        rows[fold] = {
            "arm": "afc4_vel", "fold": fold, "target_session": sessions[str(fold)],
            "inner_validation_session": f"inner-{fold}", "query_start_trial": 24, "window_size": 50,
            "query_windows_evaluated": windows[str(fold)], "r2_variance_weighted": 0.5 + fold / 100.0,
            "model_state_unchanged": True, "optimizer_present": False, "model_training_mode": False,
            "target_backpropagation": False, "target_query_labels_used_for_calibration": False,
            "target_query_labels_used_for_normalization": False, "target_query_labels_used_for_checkpoint_selection": False,
            "target_query_labels_used_for_scoring_only": True, "target_support_calibration_labels_used": True,
            "target_support_calibration_velocity_used": True,
            "provenance": {"outer_target_eval": {"status": "PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP", "sha256": "a" * 64}},
        }
    aggregate = _write(tmp_path / "rc.json", {
        "task": "rt", "seed": 42, "cells": list(rows.values()),
        "audits": {
            "exact_main_grid": True, "all_model_state_unchanged": True, "all_optimizer_absent": True,
            "all_target_backpropagation_false": True, "all_target_loaded_during_fit_false": True,
            "all_target_query_labels_read_during_fit_false": True,
            "all_target_query_labels_used_for_calibration_false": True,
            "all_target_query_labels_used_for_normalization_false": True,
            "all_target_query_labels_used_for_checkpoint_selection_false": True,
            "all_target_query_labels_used_for_scoring_only_true": True,
            "expected_target_sessions_by_fold": sessions,
            "query_windows_evaluated_by_arm_fold": {"afc4_vel": windows},
        },
    }, immutable=True)
    seal = _write(tmp_path / "seal.json", {
        "schema": "rt_seed42_clean_nested_loso_seal_marker_v1", "status": "PASS_RT_SEALED",
        "aggregate_sha256": _sha(aggregate), "folds": list(module.FOLDS), "seed": 42,
    }, immutable=True)
    return aggregate, seal, rows


def _rs_paths(tmp_path: Path, fold: int, rc: dict[str, object], *, terminal: bool = False) -> module.RsPaths:
    root = tmp_path / f"rs-{fold}"
    config = _write(root / "config.yaml", "\n".join((
        "run_id: rt_clean_nested_loso_m24_b2_d1024_zero4", "calibration_n_trials: 24",
        "query_start_trial: 24", "side_feature_group: zero4", "freeze_decoder: false",
        "loss_mode: task_only", "id_hidden_dim: 1024", "variant: B2",
    )))
    split = _write(root / "split.json", {
        "task": "rt", "development_only": True, "validation_protocol": "nested_loso",
        "outer_loso_fold": fold, "loso_fold": fold, "target_session": rc["target_session"],
        "inner_validation_session": rc["inner_validation_session"], "requested_side_feature_group": "zero4",
        "arm": {"canonical_arm": "zero4"}, "protocol": {"decode_target": "2D cursor velocity"},
        "calibration": {"budget_trials": 24, "trial_index_range": [0, 24], "target_calibration_optimizer_steps": 0},
        "query": {"query_start_trial": 24, "window_size_bins": 50},
        "nested_selection": {"clean": True, "outer_target_loaded_during_fit": False,
                             "outer_target_query_labels_read_during_fit": False,
                             "inner_validation_only_for_checkpoint_selection": True,
                             "checkpoint_metric": "val_heldin/r2_mean",
                             "checkpoint_metric_scope": "inner_validation_session_only"},
    })
    selection = _write(root / "selection.json", {
        "schema": "rt_clean_nested_loso_selection_receipt_v1", "status": "PASS_FIT_INNER_SELECTION_ONLY",
        "arm": "zero4", "seed": 42, "outer_loso_fold": fold, "selected_by_metric": "val_heldin/r2_mean",
        "selected_metric_scope": "inner_validation_session_only", "formal_heldout_opened": False,
        "outer_target_loaded_during_fit": False, "outer_target_query_labels_read_during_fit": False,
        "best_model_sha256": "b" * 64, "config_sha256": _sha(config), "split_manifest_sha256": _sha(split),
        "selected_epoch": 3, "selected_global_step": 4,
    })
    outer = _write(root / "outer.json", {
        "schema": "rt_clean_nested_loso_outer_eval_v1", "status": "PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP",
        "arm": "zero4", "outer_loso_fold": fold, "seed": 42, "query_start_trial": 24, "window_size": 50,
        "target_backpropagation": False, "optimizer_present": False, "model_training_mode": False,
        "model_state_unchanged": True, "target_query_labels_used_for_calibration": False,
        "target_query_labels_used_for_normalization": False, "target_query_labels_used_for_checkpoint_selection": False,
        "target_query_labels_used_for_scoring_only": True, "target_support_calibration_labels_used": True,
        "target_support_calibration_velocity_used": True, "outer_target_session": rc["target_session"],
        "query_windows_evaluated": rc["query_windows_evaluated"], "model_state_sha256_before": "c" * 64,
        "model_state_sha256_after": "c" * 64, "checkpoint_sha256": "b" * 64,
        "r2_variance_weighted": float(rc["r2_variance_weighted"]) - 0.1,
    })
    terminal_path = None
    if terminal:
        terminal_path = _write(root / "terminal.json", {
            "schema": "rt_stage_r_b2_d1024_fold_terminal_v1", "status": "PASS_CPU_ONE_SHOT_OUTER_EVAL",
            "fold": fold, "seed": 42, "arm": "zero4", "formal_heldout_opened": False,
        })
    return module.RsPaths(outer=outer, selection=selection, split=split, config=config, terminal=terminal_path, source="fixture")


def test_preview_is_read_only_partial_and_explicitly_not_a_claim(tmp_path):
    aggregate, seal, rc = _rc_fixture(tmp_path)
    paths = {fold: _rs_paths(tmp_path, fold, rc[fold], terminal=fold >= 3) for fold in (0, 3)}
    paths.update({fold: module.RsPaths(tmp_path / f"missing-{fold}.json", tmp_path / "a", tmp_path / "b", tmp_path / "c", None, "missing") for fold in set(module.FOLDS) - set(paths)})
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    preview = module.preview(rc_aggregate=aggregate, rc_seal=seal, rs_paths=paths)
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert preview["status"] == module.PREVIEW_STATUS and preview["paper_claim"] is False
    assert preview["validated_pair_count"] == 2 and preview["missing_folds"] == [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    assert before == after


def test_formal_aggregate_fail_closed_until_all_fifteen_pairs_exist(tmp_path):
    aggregate, seal, rc = _rc_fixture(tmp_path)
    paths = {fold: _rs_paths(tmp_path, fold, rc[fold], terminal=fold >= 3) for fold in (0,)}
    paths.update({fold: module.RsPaths(tmp_path / f"missing-{fold}.json", tmp_path / "a", tmp_path / "b", tmp_path / "c", None, "missing") for fold in set(module.FOLDS) - set(paths)})
    with pytest.raises(module.AggregateError, match="exactly 15/15"):
        module._paired_rows(rc, paths, require_all=True)
    assert not (tmp_path / "formal.json").exists()


def test_rs_validator_rejects_mismatched_windows_and_target_backprop(tmp_path):
    _, _, rc = _rc_fixture(tmp_path)
    paths = _rs_paths(tmp_path, 3, rc[3], terminal=True)
    outer = json.loads(paths.outer.read_text(encoding="utf-8"))
    outer["query_windows_evaluated"] += 1
    _write(paths.outer, outer)
    with pytest.raises(module.AggregateError, match="query windows"):
        module._validate_rs_fold(paths, rc=rc[3], fold=3)
    outer["query_windows_evaluated"] = rc[3]["query_windows_evaluated"]
    outer["target_backpropagation"] = True
    _write(paths.outer, outer)
    with pytest.raises(module.AggregateError, match="target_backpropagation"):
        module._validate_rs_fold(paths, rc=rc[3], fold=3)


def test_rs_validator_accepts_pipeline_v2_terminal_receipts(tmp_path):
    _, _, rc = _rc_fixture(tmp_path)
    paths = _rs_paths(tmp_path, 3, rc[3], terminal=True)
    terminal = json.loads(paths.terminal.read_text(encoding="utf-8"))
    terminal["schema"] = "rt_stage_r_b2_d1024_fold_terminal_v2"
    _write(paths.terminal, terminal)
    validated = module._validate_rs_fold(paths, rc=rc[3], fold=3)
    assert validated["r2"] == pytest.approx(float(rc[3]["r2_variance_weighted"]) - 0.1)


def _recovery_rs_paths(tmp_path: Path, rc: dict[str, object]) -> tuple[module.RsPaths, Path, dict[str, Path]]:
    """Build an entirely synthetic but hash-bound isolated fold-10 retry."""

    recovery_root = tmp_path / "fold10_race_recovery_v1"
    paths = recovery_checker.recovery_paths(recovery_root)
    config = _write(paths["config"], "\n".join((
        "run_id: rt_clean_nested_loso_m24_b2_d1024_zero4", "calibration_n_trials: 24",
        "query_start_trial: 24", "side_feature_group: zero4", "freeze_decoder: false",
        "loss_mode: task_only", "id_hidden_dim: 1024", "variant: B2",
    )))
    split = _write(paths["split"], {
        "task": "rt", "development_only": True, "validation_protocol": "nested_loso",
        "outer_loso_fold": 10, "loso_fold": 10, "target_session": rc["target_session"],
        "inner_validation_session": rc["inner_validation_session"], "requested_side_feature_group": "zero4",
        "arm": {"canonical_arm": "zero4"}, "protocol": {"decode_target": "2D cursor velocity"},
        "calibration": {"budget_trials": 24, "trial_index_range": [0, 24], "target_calibration_optimizer_steps": 0},
        "query": {"query_start_trial": 24, "window_size_bins": 50},
        "nested_selection": {"clean": True, "outer_target_loaded_during_fit": False,
                             "outer_target_query_labels_read_during_fit": False,
                             "inner_validation_only_for_checkpoint_selection": True,
                             "checkpoint_metric": "val_heldin/r2_mean",
                             "checkpoint_metric_scope": "inner_validation_session_only"},
    })
    checkpoint = paths["fit"] / "checkpoints/best_ckpt/epoch_003.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"fresh recovery checkpoint")
    selection = _write(paths["selection"], {
        "schema": "rt_clean_nested_loso_selection_receipt_v1", "status": "PASS_FIT_INNER_SELECTION_ONLY",
        "arm": "zero4", "seed": 42, "outer_loso_fold": 10, "selected_by_metric": "val_heldin/r2_mean",
        "selected_metric_scope": "inner_validation_session_only", "formal_heldout_opened": False,
        "outer_target_loaded_during_fit": False, "outer_target_query_labels_read_during_fit": False,
        "best_model_path": str(checkpoint.resolve()), "best_model_sha256": _sha(checkpoint),
        "config_sha256": _sha(config), "split_manifest_sha256": _sha(split),
        "selected_epoch": 3, "selected_global_step": 4,
    })
    outer = _write(paths["outer"], {
        "schema": "rt_clean_nested_loso_outer_eval_v1", "status": "PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP",
        "arm": "zero4", "outer_loso_fold": 10, "seed": 42, "query_start_trial": 24, "window_size": 50,
        "target_backpropagation": False, "optimizer_present": False, "model_training_mode": False,
        "model_state_unchanged": True, "target_query_labels_used_for_calibration": False,
        "target_query_labels_used_for_normalization": False, "target_query_labels_used_for_checkpoint_selection": False,
        "target_query_labels_used_for_scoring_only": True, "target_support_calibration_labels_used": True,
        "target_support_calibration_velocity_used": True, "outer_target_session": rc["target_session"],
        "query_windows_evaluated": rc["query_windows_evaluated"], "model_state_sha256_before": "c" * 64,
        "model_state_sha256_after": "c" * 64, "checkpoint_sha256": _sha(checkpoint),
        "r2_variance_weighted": float(rc["r2_variance_weighted"]) - 0.1,
    })
    preflight = _write(paths["preflight"], {"status": "READY_NOT_LAUNCHED"})
    old = _write(tmp_path / "old_raced_cell/cell_terminal.json", {
        "schema": recovery_checker.OLD_TERMINAL_SCHEMA,
        "status": recovery_checker.OLD_TERMINAL_STATUS,
        "fold": 10, "seed": 42, "arm": "zero4", "formal_heldout_opened": False,
    }, immutable=True)
    old_binding = recovery_checker.validate_old_failure(terminal_path=old)
    files = {
        name: {"path": str(paths[name].resolve()), "sha256": _sha(paths[name])}
        for name in ("preflight", "selection", "config", "split", "outer")
    }
    files["selected_checkpoint"] = {"path": str(checkpoint.resolve()), "sha256": _sha(checkpoint)}
    _write(paths["terminal"], {
        "schema": recovery_checker.RECOVERY_TERMINAL_SCHEMA,
        "status": recovery_checker.RECOVERY_TERMINAL_STATUS,
        "fold": 10, "seed": 42, "arm": "zero4", "formal_heldout_opened": False,
        "preserved_old_failure": old_binding,
        "freshness": {
            "recovery_root": str(recovery_root.resolve()), "cell": str(paths["cell"]),
            "old_cell_reused": False, "warmstart_forbidden": True,
            "configured_ckpt_path": None, "recovery_fit_attempts": 1,
        },
        "files": files,
    }, immutable=True)
    return module.RsPaths(
        outer=outer, selection=selection, split=split, config=config, terminal=paths["terminal"],
        source="local_3090_fold10_race_recovery_v1", recovery_root=recovery_root,
        old_failure_terminal=old,
    ), old, paths


def test_fold10_old_failed_receipt_cannot_substitute_for_missing_recovery(tmp_path):
    _, _, rc = _rc_fixture(tmp_path)
    old = _write(tmp_path / "old/cell_terminal.json", {
        "schema": recovery_checker.OLD_TERMINAL_SCHEMA,
        "status": recovery_checker.OLD_TERMINAL_STATUS,
        "fold": 10, "seed": 42, "arm": "zero4", "formal_heldout_opened": False,
    }, immutable=True)
    recovery_root = tmp_path / "missing_recovery"
    expected = recovery_checker.recovery_paths(recovery_root)
    fold10 = module.RsPaths(
        outer=expected["outer"], selection=expected["selection"], split=expected["split"],
        config=expected["config"], terminal=expected["terminal"],
        source="local_3090_fold10_race_recovery_v1", recovery_root=recovery_root,
        old_failure_terminal=old,
    )
    paths = {
        fold: module.RsPaths(tmp_path / f"missing-{fold}", tmp_path / "a", tmp_path / "b", tmp_path / "c", None, "missing")
        for fold in module.FOLDS
    }
    paths[10] = fold10
    rows, missing = module._paired_rows(rc, paths, require_all=False)
    assert rows == [] and 10 in missing
    assert old.is_file() and old.stat().st_mode & 0o777 == 0o444


def test_fold10_recovery_is_counted_only_after_terminal_source_and_config_hashes_bind(tmp_path):
    _, _, rc = _rc_fixture(tmp_path)
    fold10, old, paths = _recovery_rs_paths(tmp_path, rc[10])
    validated = module._validate_rs_fold(fold10, rc=rc[10], fold=10)
    assert validated["source"] == "local_3090_fold10_race_recovery_v1"
    assert validated["files"]["recovery_terminal"]["sha256"] == _sha(paths["terminal"])
    assert validated["files"]["recovery_source_config"]["sha256"] == _sha(paths["config"])
    assert validated["files"]["recovery_old_failure"]["sha256"] == _sha(old)

    # The generic RT semantic receipt still looks valid after this mutation,
    # but the recovery lineage checker blocks it because the config hash no
    # longer agrees with the immutable recovery terminal.
    paths["config"].write_text(paths["config"].read_text(encoding="utf-8") + "extra: drift\n", encoding="utf-8")
    with pytest.raises(module.AggregateError, match="config SHA mismatch"):
        module._validate_rs_fold(fold10, rc=rc[10], fold=10)


def test_default_paths_bind_only_fold10_to_the_new_recovery_root(tmp_path):
    paths = module.default_rs_paths(
        fold0_root=tmp_path / "fold0", local_root=tmp_path / "local",
        supervisor_root=tmp_path / "old_supervisor", fold10_recovery_root=tmp_path / "fresh_recovery",
    )
    assert paths[10].source == "local_3090_fold10_race_recovery_v1"
    assert paths[10].recovery_root == (tmp_path / "fresh_recovery").resolve()
    assert paths[10].old_failure_terminal == recovery_checker.old_failure_terminal()
    assert paths[9].source == "local_3090_supervisor_fold3_14"
    assert paths[11].source == "local_3090_supervisor_fold3_14"


def test_statistics_are_fixed_seed_and_sign_test_is_exact():
    first = module.bootstrap_mean_ci([0.1, 0.2, -0.1], draws=1000, seed=9)
    second = module.bootstrap_mean_ci([0.1, 0.2, -0.1], draws=1000, seed=9)
    assert first == second
    signs = module.exact_two_sided_sign_test([1.0, 1.0, -1.0, 0.0])
    assert signs == {"test": "exact_two_sided_binomial_sign_test", "positive": 2, "negative": 1,
                     "zero": 1, "nonzero_pairs": 3, "p_value": 1.0, "ties": "excluded_from_sign_test"}


def test_cost_ratio_is_the_precommitted_b2_to_rc_ratio(tmp_path):
    comparison = _write(tmp_path / "comparison.json", {
        "schema": "rt_stage_r_d1024_fold0_paired_comparison_v1",
        "arms": {"r_c_b3s_continuous_velocity_carrier": {"identity_encoder_parameters": 18290, "r2_variance_weighted": 0.4},
                 "r_s_b2_d1024_spint_scale_identity": {"identity_encoder_parameters": 4353074, "r2_variance_weighted": 0.1}},
        # Mirrors the already-immutable v1 receipt's rounded derived field;
        # the exact integer counts above are the authoritative binding.
        "paired_deltas": {"r_s_d1024_to_r_c_identity_parameter_ratio": 238.0029524322034},
    }, immutable=True)
    result = module._validate_cost(comparison, rc0=0.4, rs0=0.1)
    assert result["r_s_to_r_c_ratio"] == 4353074 / 18290

    drifted = _write(tmp_path / "comparison-drifted.json", {
        "schema": "rt_stage_r_d1024_fold0_paired_comparison_v1",
        "arms": {"r_c_b3s_continuous_velocity_carrier": {"identity_encoder_parameters": 18290, "r2_variance_weighted": 0.4},
                 "r_s_b2_d1024_spint_scale_identity": {"identity_encoder_parameters": 4353074, "r2_variance_weighted": 0.1}},
        "paired_deltas": {"r_s_d1024_to_r_c_identity_parameter_ratio": 238.01},
    }, immutable=True)
    with pytest.raises(module.AggregateError, match="parameter ratio drift"):
        module._validate_cost(drifted, rc0=0.4, rs0=0.1)


def test_aggregate_tool_is_receipt_only_and_has_no_data_or_gpu_execution_imports():
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "src.data" not in source and "import torch" not in source
    assert "import subprocess" not in source and "Trainer(" not in source
    assert '"nwb_opened": False' in source and '"cuda_constructed": False' in source
