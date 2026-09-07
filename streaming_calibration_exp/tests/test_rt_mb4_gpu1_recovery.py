"""Focused no-NWB/no-GPU contracts for the isolated RT MB4 GPU1 recovery."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]
BASE_PATH = PROJECT / "scripts/rt_mb4_matched_continuation.py"
RECOVERY_PATH = PROJECT / "scripts/rt_mb4_gpu1_recovery.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load(BASE_PATH, "rt_mb4_base_for_gpu1_recovery_test")
RECOVERY = _load(RECOVERY_PATH, "rt_mb4_gpu1_recovery_test")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, body: object, *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body if isinstance(body, str) else json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
    if immutable:
        path.chmod(0o444)


def _reference(tmp_path: Path) -> tuple[Path, Path]:
    aggregate = tmp_path / "full.json"
    cells = [
        {"arm": BASE.FULL_ARM, "fold": fold, "target_session": f"target-{fold}",
         "inner_validation_session": f"inner-{fold}", "query_start_trial": 24, "window_size": 50,
         "query_windows_evaluated": 1_000 + fold, "r2_variance_weighted": .5 + fold / 100.0}
        for fold in BASE.FOLDS
    ]
    _write(aggregate, {
        "schema": BASE.R_C_SCHEMA, "status": BASE.R_C_STATUS, "task": "rt", "seed": 42,
        "audits": {
            "exact_main_grid": True, "all_model_state_unchanged": True, "all_optimizer_absent": True,
            "all_target_backpropagation_false": True, "all_target_loaded_during_fit_false": True,
            "all_target_query_labels_read_during_fit_false": True,
            "all_target_query_labels_used_for_calibration_false": True,
            "all_target_query_labels_used_for_normalization_false": True,
            "all_target_query_labels_used_for_checkpoint_selection_false": True,
            "all_target_query_labels_used_for_scoring_only_true": True,
        },
        "cells": cells,
    }, immutable=True)
    seal = tmp_path / "full.seal.json"
    _write(seal, {"schema": BASE.R_C_SEAL_SCHEMA, "status": BASE.R_C_STATUS, "aggregate_sha256": _sha(aggregate)}, immutable=True)
    return aggregate, seal


def _complete_cell(root: Path, *, fold: int, r2: float = .3) -> None:
    paths = BASE.cell_paths(root, fold)
    _write(paths.config, "data:\n  side_feature_group: afc4_mb4\n")
    split = {
        "task": "rt", "development_only": True, "validation_protocol": "nested_loso",
        "outer_loso_fold": fold, "loso_fold": fold, "requested_side_feature_group": BASE.ARM,
        "arm": {"canonical_arm": BASE.ARM}, "target_session": f"target-{fold}",
        "inner_validation_session": f"inner-{fold}",
        "protocol": {"signal_type": "sorted_SUA", "decode_target": "2D cursor velocity"},
        "calibration": {"budget_trials": 24, "trial_index_range": [0, 24], "target_calibration_optimizer_steps": 0},
        "query": {"query_start_trial": 24, "window_size_bins": 50, "full_window_after_support_required": True},
        "nested_selection": {
            "clean": True, "outer_target_loaded_during_fit": False, "outer_target_query_labels_read_during_fit": False,
            "inner_validation_only_for_checkpoint_selection": True, "checkpoint_metric": "val_heldin/r2_mean",
            "checkpoint_metric_scope": "inner_validation_session_only",
        },
        "source_only_normalizer": {"fit_scope": "inner_train_sessions_only", "feature_group": BASE.ARM,
                                   "excluded_outer_target_session": f"target-{fold}"},
    }
    _write(paths.split, split)
    _write(paths.selection, {
        "schema": "rt_clean_nested_loso_selection_receipt_v1", "status": "PASS_FIT_INNER_SELECTION_ONLY",
        "arm": BASE.ARM, "outer_loso_fold": fold, "seed": 42, "selected_by_metric": "val_heldin/r2_mean",
        "selected_metric_scope": "inner_validation_session_only", "formal_heldout_opened": False,
        "outer_target_loaded_during_fit": False, "outer_target_query_labels_read_during_fit": False,
        "best_model_sha256": "a" * 64, "config_sha256": _sha(paths.config), "split_manifest_sha256": _sha(paths.split),
    })
    _write(paths.outer, {
        "schema": "rt_clean_nested_loso_outer_eval_v1", "status": "PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP",
        "arm": BASE.ARM, "outer_loso_fold": fold, "seed": 42, "query_start_trial": 24, "window_size": 50,
        "outer_target_session": f"target-{fold}", "query_windows_evaluated": 1_000 + fold,
        "normalizer_fit_scope": "inner_train_sessions_only", "target_backpropagation": False,
        "optimizer_present": False, "model_training_mode": False, "model_state_unchanged": True,
        "target_query_labels_used_for_calibration": False, "target_query_labels_used_for_normalization": False,
        "target_query_labels_used_for_checkpoint_selection": False, "target_query_labels_used_for_scoring_only": True,
        "target_support_calibration_labels_used": True, "target_support_calibration_velocity_used": True,
        "model_state_sha256_before": "b" * 64, "model_state_sha256_after": "b" * 64,
        "checkpoint_sha256": "a" * 64, "r2_variance_weighted": r2,
    })
    _write(paths.terminal, {"status": "PASS", "arm": BASE.ARM, "fold": fold, "seed": 42})


def _complete_control_cell(root: Path, *, arm: str, fold: int) -> None:
    controls = BASE._controls_module()
    paths = controls.cell_paths(root, arm, fold)
    _write(paths.config, {"run_id": f"rt_clean_nested_loso_m24_{arm}", "data": {"side_feature_group": arm}})
    split = {
        "task": "rt", "development_only": True, "validation_protocol": "nested_loso",
        "outer_loso_fold": fold, "loso_fold": fold, "requested_side_feature_group": arm,
        "arm": {"canonical_arm": arm}, "target_session": f"target-{fold}",
        "inner_validation_session": f"inner-{fold}",
        "protocol": {"signal_type": "sorted_SUA", "decode_target": "2D cursor velocity"},
        "calibration": {"budget_trials": 24, "trial_index_range": [0, 24], "target_calibration_optimizer_steps": 0},
        "query": {"query_start_trial": 24, "window_size_bins": 50, "full_window_after_support_required": True},
        "nested_selection": {
            "clean": True, "outer_target_loaded_during_fit": False, "outer_target_query_labels_read_during_fit": False,
            "inner_validation_only_for_checkpoint_selection": True, "checkpoint_metric": "val_heldin/r2_mean",
            "checkpoint_metric_scope": "inner_validation_session_only",
        },
        "source_only_normalizer": {"fit_scope": "inner_train_sessions_only", "feature_group": arm,
                                   "excluded_outer_target_session": f"target-{fold}"},
    }
    _write(paths.split, split)
    checkpoint = "a" * 64
    _write(paths.selection, {
        "schema": "rt_clean_nested_loso_selection_receipt_v1", "status": "PASS_FIT_INNER_SELECTION_ONLY",
        "arm": arm, "outer_loso_fold": fold, "seed": 42, "selected_by_metric": "val_heldin/r2_mean",
        "selected_metric_scope": "inner_validation_session_only", "formal_heldout_opened": False,
        "outer_target_loaded_during_fit": False, "outer_target_query_labels_read_during_fit": False,
        "best_model_sha256": checkpoint, "config_sha256": _sha(paths.config), "split_manifest_sha256": _sha(paths.split),
    })
    _write(paths.outer, {
        "schema": "rt_clean_nested_loso_outer_eval_v1", "status": "PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP",
        "arm": arm, "outer_loso_fold": fold, "seed": 42, "query_start_trial": 24, "window_size": 50,
        "outer_target_session": f"target-{fold}", "query_windows_evaluated": 1_000 + fold,
        "normalizer_fit_scope": "inner_train_sessions_only", "target_backpropagation": False,
        "optimizer_present": False, "model_training_mode": False, "model_state_unchanged": True,
        "target_query_labels_used_for_calibration": False, "target_query_labels_used_for_normalization": False,
        "target_query_labels_used_for_checkpoint_selection": False, "target_query_labels_used_for_scoring_only": True,
        "target_support_calibration_labels_used": True, "target_support_calibration_velocity_used": True,
        "model_state_sha256_before": "b" * 64, "model_state_sha256_after": "b" * 64,
        "checkpoint_sha256": checkpoint, "r2_variance_weighted": .3,
    })
    _write(paths.terminal, {"status": "PASS", "arm": arm, "fold": fold, "seed": 42})


def _setup(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    aggregate, seal = _reference(tmp_path)
    controls_root, predecessor = tmp_path / "controls", tmp_path / "asc-terminal.json"
    for item in BASE.expected_predecessor_cells("asc"):
        _complete_control_cell(controls_root, arm=item["arm"], fold=item["fold"])
    BASE.seal_predecessor(partition="asc", control_root=controls_root, aggregate_path=aggregate,
                          seal_path=seal, output=predecessor)
    work = tmp_path / "mb4"
    continuation = tmp_path / "base-v4.json"
    body = BASE.build_plan(work_root=work, aggregate_path=aggregate, seal_path=seal,
                           gpu1_predecessor_terminal=predecessor, gpu0_predecessor_terminal=tmp_path / "desc.json")
    _write(continuation, body, immutable=True)
    BASE._prepare_partition_root(work)
    _complete_cell(work, fold=0)
    recovery_plan = tmp_path / "recovery.json"
    recovery_body = RECOVERY.build_recovery_plan(work_root=work, continuation_plan=continuation,
                                                 predecessor_terminal=predecessor, aggregate_path=aggregate, seal_path=seal)
    _write(recovery_plan, recovery_body, immutable=True)
    return work, continuation, predecessor, aggregate, seal, recovery_plan


def test_fold0_pass_is_only_existing_cell_skipped_and_exact_tmux_worker_is_retained(tmp_path: Path) -> None:
    work, continuation, predecessor, aggregate, seal, recovery_plan = _setup(tmp_path)
    body = json.loads(recovery_plan.read_text())
    assert body["verified_fold0"]["fold"] == 0
    assert body["preflight_inventory"]["fresh_missing_folds"] == list(range(1, 8))
    assert body["recovery"]["fresh_execution_order"] == list(range(1, 8))
    assert body["recovery"]["wait_for_tmux_session_to_disappear"] == "rt_controls_split_gpu0_after_desc"
    assert recovery_plan.stat().st_mode & 0o777 == 0o444
    command = RECOVERY.tmux_send_command(recovery_plan=recovery_plan, work_root=work, continuation_plan=continuation,
                                         predecessor_terminal=predecessor, aggregate_path=aggregate, seal_path=seal)
    assert "tmux has-session -t rt_mb4_gpu1_after_controls" in command
    assert "tmux send-keys -t rt_mb4_gpu1_after_controls" in command
    assert "exec env CUDA_VISIBLE_DEVICES=1" in command
    assert "--recover-gpu1" in command


def test_partial_and_failed_existing_cells_are_fatal_not_skippable(tmp_path: Path) -> None:
    work, continuation, predecessor, aggregate, seal, _recovery_plan = _setup(tmp_path)
    partial = BASE.cell_paths(work, 1).cell
    partial.mkdir(parents=True)
    with pytest.raises(RECOVERY.Mb4Gpu1RecoveryError, match="partial"):
        RECOVERY.build_recovery_plan(work_root=work, continuation_plan=continuation,
                                     predecessor_terminal=predecessor, aggregate_path=aggregate, seal_path=seal)

    work2, continuation2, predecessor2, aggregate2, seal2, _recovery_plan2 = _setup(tmp_path / "failed")
    _complete_cell(work2, fold=1)
    paths = BASE.cell_paths(work2, 1)
    _write(paths.terminal, {"status": "FAILED", "arm": BASE.ARM, "fold": 1, "seed": 42})
    with pytest.raises(RECOVERY.Mb4Gpu1RecoveryError, match="failed/incomparable"):
        RECOVERY.build_recovery_plan(work_root=work2, continuation_plan=continuation2,
                                     predecessor_terminal=predecessor2, aggregate_path=aggregate2, seal_path=seal2)


def test_fold1_active_writer_gate_fails_before_directory_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work, continuation, predecessor, aggregate, seal, recovery_plan = _setup(tmp_path)
    # Keep the one loaded base module injectable so the recovery's active-writer
    # test can be tested without a GPU, Trainer, NWB, or subprocess execution.
    monkeypatch.setattr(RECOVERY, "_base", lambda: BASE)
    monkeypatch.setattr(RECOVERY, "_controls_session_exists", lambda _name: False)
    writer = "python train.py experiment=rt_clean_nested_loso_m24 data.loso_fold=1 data.outer_loso_fold=1"
    monkeypatch.setattr(BASE, "active_compute_commands_on_gpu", lambda _gpu: ([], [writer]))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(BASE.Mb4ContinuationError, match="exact-same-fold"):
        RECOVERY.recover_gpu1(recovery_plan=recovery_plan, work_root=work, continuation_plan=continuation,
                              predecessor_terminal=predecessor, aggregate_path=aggregate, seal_path=seal, poll_seconds=.001)
    assert not BASE.cell_paths(work, 1).cell.exists()
    assert not (work / "MB4_GPU1_RECOVERY_ACTIVE_v1.lock").exists()


def test_duplicate_recovery_lock_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    work, continuation, predecessor, aggregate, seal, recovery_plan = _setup(tmp_path)
    monkeypatch.setattr(RECOVERY, "_base", lambda: BASE)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    lock = RECOVERY._acquire_lock(work, recovery_plan)
    assert lock.is_file()
    with pytest.raises(RECOVERY.Mb4Gpu1RecoveryError, match="duplicate launch"):
        RECOVERY.recover_gpu1(recovery_plan=recovery_plan, work_root=work, continuation_plan=continuation,
                              predecessor_terminal=predecessor, aggregate_path=aggregate, seal_path=seal, poll_seconds=.001)
    lock.unlink()

