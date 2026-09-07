"""No-NWB contracts for the RT R-RS/R-LS continuation and receipt path."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import rt_controls_continuation as module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, body: dict, *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
    if immutable:
        path.chmod(0o444)


def _reference(tmp_path: Path) -> tuple[Path, Path]:
    aggregate = tmp_path / "rc.json"
    cells = []
    for fold in module.FOLDS:
        cells.append({
            "arm": "afc4_vel", "fold": fold, "target_session": f"target-{fold}",
            "inner_validation_session": f"inner-{fold}", "query_start_trial": 24, "window_size": 50,
            "query_windows_evaluated": 1000 + fold, "r2_variance_weighted": 0.5 + fold / 100.0,
        })
    _write_json(aggregate, {
        "schema": module.R_C_SCHEMA, "status": module.R_C_STATUS, "task": "rt", "seed": 42,
        "audits": {"exact_main_grid": True,
                   "all_model_state_unchanged": True, "all_optimizer_absent": True,
                   "all_target_backpropagation_false": True, "all_target_loaded_during_fit_false": True,
                   "all_target_query_labels_read_during_fit_false": True,
                   "all_target_query_labels_used_for_calibration_false": True,
                   "all_target_query_labels_used_for_normalization_false": True,
                   "all_target_query_labels_used_for_checkpoint_selection_false": True,
                   "all_target_query_labels_used_for_scoring_only_true": True},
        "cells": cells,
    }, immutable=True)
    seal = tmp_path / "rc.seal.json"
    _write_json(seal, {"schema": module.R_C_SEAL_SCHEMA, "status": module.R_C_STATUS,
                       "aggregate_sha256": _sha(aggregate)}, immutable=True)
    return aggregate, seal


def _complete_cell(root: Path, *, arm: str, fold: int) -> None:
    paths = module.cell_paths(root, arm, fold)
    config = {"run_id": f"rt_clean_nested_loso_m24_{arm}", "data": {"side_feature_group": arm}}
    _write_json(paths.config, config)
    split = {
        "task": "rt", "development_only": True, "validation_protocol": "nested_loso",
        "outer_loso_fold": fold, "loso_fold": fold, "requested_side_feature_group": arm,
        "arm": {"canonical_arm": arm}, "target_session": f"target-{fold}",
        "inner_validation_session": f"inner-{fold}",
        "protocol": {"signal_type": "sorted_SUA", "decode_target": "2D cursor velocity"},
        "calibration": {"budget_trials": 24, "trial_index_range": [0, 24], "target_calibration_optimizer_steps": 0},
        "query": {"query_start_trial": 24, "window_size_bins": 50, "full_window_after_support_required": True},
        "nested_selection": {"clean": True, "outer_target_loaded_during_fit": False,
                              "outer_target_query_labels_read_during_fit": False,
                              "inner_validation_only_for_checkpoint_selection": True,
                              "checkpoint_metric": "val_heldin/r2_mean",
                              "checkpoint_metric_scope": "inner_validation_session_only"},
        "source_only_normalizer": {"fit_scope": "inner_train_sessions_only", "feature_group": arm,
                                   "excluded_outer_target_session": f"target-{fold}"},
    }
    _write_json(paths.split, split)
    checkpoint_sha = "a" * 64
    selection = {
        "schema": "rt_clean_nested_loso_selection_receipt_v1", "status": "PASS_FIT_INNER_SELECTION_ONLY",
        "arm": arm, "outer_loso_fold": fold, "seed": 42, "selected_by_metric": "val_heldin/r2_mean",
        "selected_metric_scope": "inner_validation_session_only", "formal_heldout_opened": False,
        "outer_target_loaded_during_fit": False, "outer_target_query_labels_read_during_fit": False,
        "best_model_sha256": checkpoint_sha, "config_sha256": _sha(paths.config),
        "split_manifest_sha256": _sha(paths.split),
    }
    _write_json(paths.selection, selection)
    outer = {
        "schema": "rt_clean_nested_loso_outer_eval_v1", "status": "PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP",
        "arm": arm, "outer_loso_fold": fold, "seed": 42, "query_start_trial": 24, "window_size": 50,
        "outer_target_session": f"target-{fold}", "query_windows_evaluated": 1000 + fold,
        "normalizer_fit_scope": "inner_train_sessions_only", "target_backpropagation": False,
        "optimizer_present": False, "model_training_mode": False, "model_state_unchanged": True,
        "target_query_labels_used_for_calibration": False, "target_query_labels_used_for_normalization": False,
        "target_query_labels_used_for_checkpoint_selection": False, "target_query_labels_used_for_scoring_only": True,
        "target_support_calibration_labels_used": True, "target_support_calibration_velocity_used": True,
        "model_state_sha256_before": "b" * 64, "model_state_sha256_after": "b" * 64,
        "checkpoint_sha256": checkpoint_sha, "r2_variance_weighted": 0.3,
    }
    _write_json(paths.outer, outer)
    _write_json(paths.terminal, {"status": "PASS", "arm": arm, "fold": fold, "seed": 42})


def test_inventory_marks_complete_missing_and_never_treats_partial_as_complete(tmp_path: Path) -> None:
    aggregate, seal = _reference(tmp_path)
    root = tmp_path / "controls"
    _complete_cell(root, arm="afc4_rs", fold=11)
    report = module.inventory(control_root=root, aggregate_path=aggregate, seal_path=seal, require_import=False)
    rs = report["arms"]["afc4_rs"]
    ls = report["arms"]["afc4_ls"]
    assert rs["complete_folds"] == [11]
    assert rs["missing_folds"] == [fold for fold in module.FOLDS if fold != 11]
    assert rs["positive_rc_minus_control_folds"] == 1
    assert ls["completed_count"] == 0 and ls["missing_folds"] == list(module.FOLDS)
    assert report["final_aggregate_eligible"] is False


def test_copy_import_freezes_a_copy_without_modifying_raw_source(tmp_path: Path) -> None:
    aggregate, seal = _reference(tmp_path)
    raw, imported = tmp_path / "raw", tmp_path / "imported"
    _complete_cell(raw, arm="afc4_ls", fold=3)
    raw_outer = module.cell_paths(raw, "afc4_ls", 3).outer
    assert raw_outer.stat().st_mode & 0o777 != 0o444
    result = module.copy_import(source_root=raw, import_root=imported, arm="afc4_ls", fold=3,
                                aggregate_path=aggregate, seal_path=seal)
    paths = module.cell_paths(imported, "afc4_ls", 3)
    assert result["status"] == module.IMPORT_STATUS
    assert paths.outer.stat().st_mode & 0o777 == 0o444
    assert paths.imported.stat().st_mode & 0o777 == 0o444
    assert raw_outer.stat().st_mode & 0o777 != 0o444
    checked = module.validate_control_cell(control_root=imported, arm="afc4_ls", fold=3,
                                           reference=module._load_rc_reference(aggregate_path=aggregate, seal_path=seal),
                                           require_import=True)
    assert checked["state"] == "complete" and checked["copy_import_sha256"]


def test_execution_plan_is_target_free_and_refuses_active_rs_output_roots(tmp_path: Path) -> None:
    aggregate, seal = _reference(tmp_path)
    plan = module.build_execution_plan(work_root=tmp_path / "independent-controls", arm="afc4_rs", fold=0, gpu=0,
                                       aggregate_path=aggregate, seal_path=seal)
    assert plan["status"] == "PLAN_ONLY_NO_GPU_NO_NWB_NO_TRAINER"
    assert "experiment=rt_clean_nested_loso_m24" in plan["train_command"]
    assert "data.side_feature_group=afc4_rs" in plan["train_command"]
    with pytest.raises(module.ControlContinuationError, match="active R-S root"):
        module._ensure_independent_output_root(module.PROJECT / "outputs/rt_stage_r_b2_local3090/new")


def test_requested_gpu_any_compute_pid_is_fail_closed_and_same_fold_is_global() -> None:
    with pytest.raises(module.ControlContinuationError, match="occupied by compute PID"):
        module._ensure_launch_safe(gpu_commands=["<unreadable-gpu-pid:123>"], all_rt_commands=[], fold=5, gpu=0)
    same_fold = "python train.py experiment=rt_clean_nested_loso_b2_stage_r_zero4 data.loso_fold=5"
    with pytest.raises(module.ControlContinuationError, match="same-fold RT writer"):
        module._ensure_launch_safe(gpu_commands=[], all_rt_commands=[same_fold], fold=5, gpu=0)


def test_gpu0_control_is_not_blocked_by_rs_on_gpu1_unless_it_is_the_same_fold() -> None:
    # In real execution the first list comes only from selected-GPU nvidia-smi
    # PIDs.  A Stage-R writer on GPU1 must not idle GPU0; the separate global
    # scan only prohibits an exact same-fold collision.
    gpu0_commands: list[str] = []
    gpu1_rs = "python train.py experiment=rt_clean_nested_loso_b2_stage_r_zero4 data.loso_fold=6"
    module._ensure_launch_safe(gpu_commands=gpu0_commands, all_rt_commands=[gpu1_rs], fold=5, gpu=0)
    assert module._same_fold_rt_writer_commands([gpu1_rs], fold=5) == []
    assert module._same_fold_rt_writer_commands([gpu1_rs], fold=6) == [gpu1_rs]


def test_copy_import_requires_and_freezes_valid_cell_terminal(tmp_path: Path) -> None:
    aggregate, seal = _reference(tmp_path)
    raw, imported = tmp_path / "raw", tmp_path / "imported"
    _complete_cell(raw, arm="afc4_rs", fold=2)
    paths = module.cell_paths(raw, "afc4_rs", 2)
    paths.terminal.unlink()
    with pytest.raises(module.ControlContinuationError, match="without a valid cell terminal"):
        module.copy_import(source_root=raw, import_root=imported, arm="afc4_rs", fold=2,
                           aggregate_path=aggregate, seal_path=seal)


def test_cell_validation_rejects_query_and_session_mismatches(tmp_path: Path) -> None:
    aggregate, seal = _reference(tmp_path)
    reference = module._load_rc_reference(aggregate_path=aggregate, seal_path=seal)
    root = tmp_path / "controls"
    _complete_cell(root, arm="afc4_rs", fold=4)
    paths = module.cell_paths(root, "afc4_rs", 4)
    outer = json.loads(paths.outer.read_text(encoding="utf-8"))
    outer["query_windows_evaluated"] += 1
    _write_json(paths.outer, outer)
    with pytest.raises(module.ControlContinuationError, match="outer contract drift: query_windows_evaluated"):
        module.validate_control_cell(control_root=root, arm="afc4_rs", fold=4,
                                     reference=reference, require_import=False)

    _complete_cell(root, arm="afc4_rs", fold=5)
    paths = module.cell_paths(root, "afc4_rs", 5)
    split = json.loads(paths.split.read_text(encoding="utf-8"))
    split["inner_validation_session"] = "wrong-inner-session"
    _write_json(paths.split, split)
    selection = json.loads(paths.selection.read_text(encoding="utf-8"))
    selection["split_manifest_sha256"] = _sha(paths.split)
    _write_json(paths.selection, selection)
    with pytest.raises(module.ControlContinuationError, match="does not match frozen R-C target/inner-validation pair"):
        module.validate_control_cell(control_root=root, arm="afc4_rs", fold=5,
                                     reference=reference, require_import=False)


def test_partial_cells_and_duplicate_import_destinations_are_rejected(tmp_path: Path) -> None:
    aggregate, seal = _reference(tmp_path)
    partial = tmp_path / "partial"
    _complete_cell(partial, arm="afc4_ls", fold=1)
    module.cell_paths(partial, "afc4_ls", 1).split.unlink()
    report = module.inventory(control_root=partial, aggregate_path=aggregate, seal_path=seal, require_import=False)
    assert report["arms"]["afc4_ls"]["incomparable_folds"] == [1]
    assert report["final_aggregate_eligible"] is False

    raw, imported = tmp_path / "raw", tmp_path / "imported"
    _complete_cell(raw, arm="afc4_ls", fold=1)
    module.copy_import(source_root=raw, import_root=imported, arm="afc4_ls", fold=1,
                       aggregate_path=aggregate, seal_path=seal)
    with pytest.raises(module.ControlContinuationError, match="refusing to overwrite import destination"):
        module.copy_import(source_root=raw, import_root=imported, arm="afc4_ls", fold=1,
                           aggregate_path=aggregate, seal_path=seal)


def test_full_immutable_aggregate_reports_sign_test_and_deterministic_bootstrap(tmp_path: Path) -> None:
    aggregate, seal = _reference(tmp_path)
    raw, imported = tmp_path / "raw", tmp_path / "imported"
    for arm in module.CONTROL_ARMS:
        for fold in module.FOLDS:
            _complete_cell(raw, arm=arm, fold=fold)
            module.copy_import(source_root=raw, import_root=imported, arm=arm, fold=fold,
                               aggregate_path=aggregate, seal_path=seal)
    output = tmp_path / "aggregate.json"
    result = module.aggregate(import_root=imported, aggregate_path=aggregate, seal_path=seal, output=output)
    assert result["status"] == module.AGGREGATE_STATUS
    body = json.loads(output.read_text(encoding="utf-8"))
    for arm in module.CONTROL_ARMS:
        sign = body["arms"][arm]["exact_two_sided_sign_test"]
        bootstrap = body["arms"][arm]["paired_fold_bootstrap_95"]
        assert sign["positive"] == 15 and sign["p_value"] == pytest.approx(2.0 / (2 ** 15))
        assert bootstrap["rng_seed"] == module.BOOTSTRAP_SEED_BY_ARM[arm]
        assert bootstrap["draws"] == module.BOOTSTRAP_DRAWS
        assert bootstrap["lower_95"] <= bootstrap["upper_95"]
        assert body["arms"][arm]["median_rc_minus_control"] == pytest.approx(0.27)
    assert output.stat().st_mode & 0o777 == 0o444


def test_full_aggregate_retains_a_negative_ls_fold_in_paired_statistics(tmp_path: Path) -> None:
    aggregate, seal = _reference(tmp_path)
    raw, imported = tmp_path / "raw", tmp_path / "imported"
    for arm in module.CONTROL_ARMS:
        for fold in module.FOLDS:
            _complete_cell(raw, arm=arm, fold=fold)
    negative_fold = 4
    negative = module.cell_paths(raw, "afc4_ls", negative_fold).outer
    negative_outer = json.loads(negative.read_text(encoding="utf-8"))
    negative_outer["r2_variance_weighted"] = 0.9
    _write_json(negative, negative_outer)
    for arm in module.CONTROL_ARMS:
        for fold in module.FOLDS:
            module.copy_import(source_root=raw, import_root=imported, arm=arm, fold=fold,
                               aggregate_path=aggregate, seal_path=seal)
    output = tmp_path / "aggregate-negative-ls.json"
    module.aggregate(import_root=imported, aggregate_path=aggregate, seal_path=seal, output=output)
    body = json.loads(output.read_text(encoding="utf-8"))
    ls = body["arms"]["afc4_ls"]
    fold_row = next(row for row in ls["rows"] if row["fold"] == negative_fold)
    assert fold_row["rc_minus_control"] == pytest.approx(-0.36)
    assert ls["exact_two_sided_sign_test"] == {
        "test": "exact_two_sided_binomial_sign_test", "positive": 14, "negative": 1,
        "zero": 0, "nonzero_pairs": 15, "p_value": pytest.approx(2.0 * 16.0 / (2 ** 15)),
        "ties": "excluded_from_sign_test",
    }
    expected = sum(0.5 + fold / 100.0 - 0.3 for fold in module.FOLDS) - (0.5 + negative_fold / 100.0 - 0.3) - 0.36
    assert ls["mean_rc_minus_control"] == pytest.approx(expected / len(module.FOLDS))


def test_run_queue_only_runs_cli_order_and_stops_by_propagating_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    aggregate, seal = _reference(tmp_path)
    calls: list[int] = []

    def fake_execute_one(**kwargs):
        calls.append(kwargs["fold"])
        if kwargs["fold"] == 2:
            raise module.ControlContinuationError("synthetic fail-stop")
        return {"fold": kwargs["fold"]}

    monkeypatch.setattr(module, "execute_one", fake_execute_one)
    with pytest.raises(module.ControlContinuationError, match="synthetic fail-stop"):
        module.run_queue(work_root=tmp_path / "independent", arm="afc4_ls", folds=[0, 2, 3], gpu=0,
                         aggregate_path=aggregate, seal_path=seal)
    assert calls == [0, 2]


def test_source_has_no_target_loader_or_automatic_multi_cell_execution() -> None:
    source = (module.PROJECT / "scripts/rt_controls_continuation.py").read_text(encoding="utf-8")
    assert "load_rt_session" not in source
    assert "build_outer_target_dataset" not in source
    assert "for fold in FOLDS" in source  # receipt inventory only
    assert "execute_one" in source and "--execute requires --arm --fold" in source
