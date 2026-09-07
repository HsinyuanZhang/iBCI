"""No-NWB contracts for the isolated RT Full-vs-MB4 continuation/finalizer."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "rt_mb4_matched_continuation.py"
SPEC = importlib.util.spec_from_file_location("rt_mb4_matched_continuation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict | str, *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    if immutable:
        path.chmod(0o444)


def _reference(tmp_path: Path) -> tuple[Path, Path]:
    aggregate = tmp_path / "full.json"
    cells = [
        {"arm": MODULE.FULL_ARM, "fold": fold, "target_session": f"target-{fold}",
         "inner_validation_session": f"inner-{fold}", "query_start_trial": 24, "window_size": 50,
         "query_windows_evaluated": 1000 + fold, "r2_variance_weighted": .5 + fold / 100.0}
        for fold in MODULE.FOLDS
    ]
    _write(aggregate, {
        "schema": MODULE.R_C_SCHEMA, "status": MODULE.R_C_STATUS, "task": "rt", "seed": 42,
        "audits": {"exact_main_grid": True, "all_model_state_unchanged": True, "all_optimizer_absent": True,
                   "all_target_backpropagation_false": True, "all_target_loaded_during_fit_false": True,
                   "all_target_query_labels_read_during_fit_false": True,
                   "all_target_query_labels_used_for_calibration_false": True,
                   "all_target_query_labels_used_for_normalization_false": True,
                   "all_target_query_labels_used_for_checkpoint_selection_false": True,
                   "all_target_query_labels_used_for_scoring_only_true": True},
        "cells": cells,
    }, immutable=True)
    seal = tmp_path / "full.seal.json"
    _write(seal, {"schema": MODULE.R_C_SEAL_SCHEMA, "status": MODULE.R_C_STATUS, "aggregate_sha256": _sha(aggregate)}, immutable=True)
    return aggregate, seal


def _complete_cell(root: Path, *, fold: int, r2: float = .3) -> None:
    paths = MODULE.cell_paths(root, fold)
    _write(paths.config, "data:\n  side_feature_group: afc4_mb4\n")
    split = {
        "task": "rt", "development_only": True, "validation_protocol": "nested_loso",
        "outer_loso_fold": fold, "loso_fold": fold, "requested_side_feature_group": MODULE.ARM,
        "arm": {"canonical_arm": MODULE.ARM}, "target_session": f"target-{fold}",
        "inner_validation_session": f"inner-{fold}",
        "protocol": {"signal_type": "sorted_SUA", "decode_target": "2D cursor velocity"},
        "calibration": {"budget_trials": 24, "trial_index_range": [0, 24], "target_calibration_optimizer_steps": 0},
        "query": {"query_start_trial": 24, "window_size_bins": 50, "full_window_after_support_required": True},
        "nested_selection": {"clean": True, "outer_target_loaded_during_fit": False,
                             "outer_target_query_labels_read_during_fit": False,
                             "inner_validation_only_for_checkpoint_selection": True,
                             "checkpoint_metric": "val_heldin/r2_mean",
                             "checkpoint_metric_scope": "inner_validation_session_only"},
        "source_only_normalizer": {"fit_scope": "inner_train_sessions_only", "feature_group": MODULE.ARM,
                                    "excluded_outer_target_session": f"target-{fold}"},
    }
    _write(paths.split, split)
    selection = {
        "schema": "rt_clean_nested_loso_selection_receipt_v1", "status": "PASS_FIT_INNER_SELECTION_ONLY",
        "arm": MODULE.ARM, "outer_loso_fold": fold, "seed": 42, "selected_by_metric": "val_heldin/r2_mean",
        "selected_metric_scope": "inner_validation_session_only", "formal_heldout_opened": False,
        "outer_target_loaded_during_fit": False, "outer_target_query_labels_read_during_fit": False,
        "best_model_sha256": "a" * 64, "config_sha256": _sha(paths.config), "split_manifest_sha256": _sha(paths.split),
    }
    _write(paths.selection, selection)
    outer = {
        "schema": "rt_clean_nested_loso_outer_eval_v1", "status": "PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP",
        "arm": MODULE.ARM, "outer_loso_fold": fold, "seed": 42, "query_start_trial": 24, "window_size": 50,
        "outer_target_session": f"target-{fold}", "query_windows_evaluated": 1000 + fold,
        "normalizer_fit_scope": "inner_train_sessions_only", "target_backpropagation": False,
        "optimizer_present": False, "model_training_mode": False, "model_state_unchanged": True,
        "target_query_labels_used_for_calibration": False, "target_query_labels_used_for_normalization": False,
        "target_query_labels_used_for_checkpoint_selection": False, "target_query_labels_used_for_scoring_only": True,
        "target_support_calibration_labels_used": True, "target_support_calibration_velocity_used": True,
        "model_state_sha256_before": "b" * 64, "model_state_sha256_after": "b" * 64,
        "checkpoint_sha256": "a" * 64, "r2_variance_weighted": r2,
    }
    _write(paths.outer, outer)
    _write(paths.terminal, {"status": "PASS", "arm": MODULE.ARM, "fold": fold, "seed": 42})


def _complete_control_cell(root: Path, *, arm: str, fold: int) -> None:
    """Minimal valid raw R-RS/R-LS receipt bundle for predecessor sealing."""

    controls = MODULE._controls_module()
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
        "nested_selection": {"clean": True, "outer_target_loaded_during_fit": False,
                             "outer_target_query_labels_read_during_fit": False,
                             "inner_validation_only_for_checkpoint_selection": True,
                             "checkpoint_metric": "val_heldin/r2_mean",
                             "checkpoint_metric_scope": "inner_validation_session_only"},
        "source_only_normalizer": {"fit_scope": "inner_train_sessions_only", "feature_group": arm,
                                   "excluded_outer_target_session": f"target-{fold}"},
    }
    _write(paths.split, split)
    checkpoint_sha = "a" * 64
    _write(paths.selection, {
        "schema": "rt_clean_nested_loso_selection_receipt_v1", "status": "PASS_FIT_INNER_SELECTION_ONLY",
        "arm": arm, "outer_loso_fold": fold, "seed": 42, "selected_by_metric": "val_heldin/r2_mean",
        "selected_metric_scope": "inner_validation_session_only", "formal_heldout_opened": False,
        "outer_target_loaded_during_fit": False, "outer_target_query_labels_read_during_fit": False,
        "best_model_sha256": checkpoint_sha, "config_sha256": _sha(paths.config),
        "split_manifest_sha256": _sha(paths.split),
    })
    _write(paths.outer, {
        "schema": "rt_clean_nested_loso_outer_eval_v1", "status": "PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP",
        "arm": arm, "outer_loso_fold": fold, "seed": 42, "query_start_trial": 24, "window_size": 50,
        "outer_target_session": f"target-{fold}", "query_windows_evaluated": 1000 + fold,
        "normalizer_fit_scope": "inner_train_sessions_only", "target_backpropagation": False,
        "optimizer_present": False, "model_training_mode": False, "model_state_unchanged": True,
        "target_query_labels_used_for_calibration": False, "target_query_labels_used_for_normalization": False,
        "target_query_labels_used_for_checkpoint_selection": False, "target_query_labels_used_for_scoring_only": True,
        "target_support_calibration_labels_used": True, "target_support_calibration_velocity_used": True,
        "model_state_sha256_before": "b" * 64, "model_state_sha256_after": "b" * 64,
        "checkpoint_sha256": checkpoint_sha, "r2_variance_weighted": .3,
    })
    _write(paths.terminal, {"status": "PASS", "arm": arm, "fold": fold, "seed": 42})


def test_plan_exactly_binds_mb4_static_disjoint_partitions_and_hydra_contract(tmp_path: Path) -> None:
    aggregate, seal = _reference(tmp_path)
    work = tmp_path / "fresh-mb4-root"
    plan = MODULE.build_plan(work_root=work, aggregate_path=aggregate, seal_path=seal,
                             gpu1_predecessor_terminal=tmp_path / "gpu1-terminal.json",
                             gpu0_predecessor_terminal=tmp_path / "gpu0-terminal.json")
    assert plan["status"] == "PLAN_ONLY_NO_GPU_NO_NWB_NO_TRAINER"
    assert not work.exists()
    gpu1, gpu0 = plan["partitions"]["gpu1_after_asc"], plan["partitions"]["gpu0_after_desc"]
    assert gpu1["gpu"] == 1 and gpu0["gpu"] == 0
    assert set(gpu1["folds"]).isdisjoint(gpu0["folds"])
    assert set(gpu1["folds"]) | set(gpu0["folds"]) == set(MODULE.FOLDS)
    for partition in (gpu1, gpu0):
        for cell in partition["cells"]:
            train = cell["train_command"]
            assert "experiment=rt_clean_nested_loso_m24" in train
            assert "data.side_feature_group=afc4_mb4" in train
            assert f"data.loso_fold={cell['fold']}" in train
            assert f"data.outer_loso_fold={cell['fold']}" in train
            assert "seed=42" in train and "test=false" in train
            assert cell["eval_command_template"][-2:] == ["--device", "cuda"]
    assert "--seal-predecessor asc" in plan["watcher_templates"]["gpu1_after_asc"]
    assert "rt_controls_split_gpu1_after_asc" in plan["watcher_templates"]["gpu1_after_asc"]
    assert "--run-partition gpu1_after_asc" in plan["watcher_templates"]["gpu1_after_asc"]
    assert "--seal-predecessor desc" in plan["watcher_templates"]["gpu0_after_desc"]
    finalizer = plan["finalizer"]
    assert finalizer["requires_worker_sessions_observed"] == ["rt_mb4_gpu1_after_controls", "rt_mb4_gpu0_after_controls"]
    assert finalizer["waits_for_worker_sessions_to_disappear"] is True
    assert "--copy-import-all-and-finalize" in finalizer["tmux_template"]
    assert "rt_mb4_import_finalize_after_workers" in finalizer["tmux_template"]


def test_validation_rejects_q_session_and_target_bp_drift(tmp_path: Path) -> None:
    aggregate, seal = _reference(tmp_path)
    reference = MODULE.load_rc_reference(aggregate_path=aggregate, seal_path=seal)
    root = tmp_path / "raw"
    _complete_cell(root, fold=1)
    paths = MODULE.cell_paths(root, 1)
    outer = json.loads(paths.outer.read_text())
    outer["query_windows_evaluated"] += 1
    _write(paths.outer, outer)
    with pytest.raises(MODULE.Mb4ContinuationError, match="query_windows_evaluated"):
        MODULE.validate_cell(root=root, fold=1, reference=reference, require_import=False)

    _complete_cell(root, fold=2)
    paths = MODULE.cell_paths(root, 2)
    split = json.loads(paths.split.read_text())
    split["inner_validation_session"] = "wrong-session"
    _write(paths.split, split)
    selection = json.loads(paths.selection.read_text())
    selection["split_manifest_sha256"] = _sha(paths.split)
    _write(paths.selection, selection)
    with pytest.raises(MODULE.Mb4ContinuationError, match="target/inner session"):
        MODULE.validate_cell(root=root, fold=2, reference=reference, require_import=False)

    _complete_cell(root, fold=3)
    paths = MODULE.cell_paths(root, 3)
    outer = json.loads(paths.outer.read_text())
    outer["target_backpropagation"] = True
    _write(paths.outer, outer)
    with pytest.raises(MODULE.Mb4ContinuationError, match="target_backpropagation"):
        MODULE.validate_cell(root=root, fold=3, reference=reference, require_import=False)


def test_partial_and_duplicate_imports_fail_closed(tmp_path: Path) -> None:
    aggregate, seal = _reference(tmp_path)
    raw, imported = tmp_path / "raw", tmp_path / "imported"
    _complete_cell(raw, fold=4)
    MODULE.cell_paths(raw, 4).split.unlink()
    preview = MODULE.inventory(root=raw, aggregate_path=aggregate, seal_path=seal, require_import=False)
    assert preview["incomparable_folds"] == [4] and preview["finalize_eligible"] is False
    _complete_cell(raw, fold=4)
    MODULE.copy_import(source_root=raw, import_root=imported, fold=4, aggregate_path=aggregate, seal_path=seal)
    with pytest.raises(MODULE.Mb4ContinuationError, match="duplicate/overwrite"):
        MODULE.copy_import(source_root=raw, import_root=imported, fold=4, aggregate_path=aggregate, seal_path=seal)


def test_finalizer_requires_all_15_and_retains_negative_mb4_fold(tmp_path: Path) -> None:
    aggregate, seal = _reference(tmp_path)
    raw, imported = tmp_path / "raw", tmp_path / "imported"
    for fold in MODULE.FOLDS:
        _complete_cell(raw, fold=fold, r2=.9 if fold == 4 else .3)
        MODULE.copy_import(source_root=raw, import_root=imported, fold=fold, aggregate_path=aggregate, seal_path=seal)
    output = tmp_path / "mb4-aggregate.json"
    result = MODULE.finalize(import_root=imported, aggregate_path=aggregate, seal_path=seal, output=output)
    assert result["status"] == MODULE.AGGREGATE_STATUS
    body = json.loads(output.read_text())
    row = next(item for item in body["rows"] if item["fold"] == 4)
    assert row["full_minus_mb4"] == pytest.approx(-.36)
    test = body["full_minus_mb4"]["exact_two_sided_sign_test"]
    assert test["positive"] == 14 and test["negative"] == 1 and test["zero"] == 0
    assert body["full_minus_mb4"]["paired_fold_bootstrap_95"]["rng_seed"] == MODULE.BOOTSTRAP_SEED
    assert output.stat().st_mode & 0o777 == 0o444
    assert body["accounting"]["mb4"][0]["status"].startswith("not_present")


def test_predecessor_failure_and_fold1_vs_fold10_token_collision_contract(tmp_path: Path) -> None:
    predecessor = tmp_path / "failed-predecessor.json"
    _write(predecessor, {"schema": MODULE.PREDECESSOR_SCHEMA, "status": MODULE.PREDECESSOR_STATUS,
                         "partition": "asc", "all_cells_passed": False, "failed_cells": [1]})
    predecessor.chmod(0o444)
    with pytest.raises(MODULE.Mb4ContinuationError, match="not clean"):
        MODULE.validate_predecessor_terminal(predecessor, expected_partition="asc")

    fold10 = "python train.py experiment=rt_clean_nested_loso_m24 data.loso_fold=10 data.outer_loso_fold=10"
    fold1 = "python train.py experiment=rt_clean_nested_loso_m24 data.loso_fold=1 data.outer_loso_fold=1"
    assert MODULE.same_fold_rt_writer_commands([fold10], fold=1) == []
    assert MODULE.same_fold_rt_writer_commands([fold1], fold=1) == [fold1]
    MODULE.ensure_launch_safe(gpu_commands=[], all_rt_commands=[fold10], fold=1, gpu=0)
    with pytest.raises(MODULE.Mb4ContinuationError, match="exact-same-fold"):
        MODULE.ensure_launch_safe(gpu_commands=[], all_rt_commands=[fold1], fold=1, gpu=0)


def test_predecessor_seal_reuses_control_validator_records_sha_and_rejects_missing(tmp_path: Path) -> None:
    aggregate, seal = _reference(tmp_path)
    raw, output = tmp_path / "raw-controls", tmp_path / "asc-terminal.json"
    for item in MODULE.expected_predecessor_cells("asc"):
        _complete_control_cell(raw, arm=item["arm"], fold=item["fold"])
    result = MODULE.seal_predecessor(partition="asc", control_root=raw, aggregate_path=aggregate,
                                     seal_path=seal, output=output)
    assert result["status"] == MODULE.PREDECESSOR_STATUS and output.stat().st_mode & 0o777 == 0o444
    terminal = MODULE.validate_predecessor_terminal(output, expected_partition="asc")
    assert terminal["expected_cells"] == MODULE.expected_predecessor_cells("asc")
    assert len(terminal["validated_cells"]) == 12
    assert all(row["files"]["cell_terminal"]["sha256"] for row in terminal["validated_cells"])

    with pytest.raises(MODULE.Mb4ContinuationError, match="is not complete"):
        MODULE.seal_predecessor(partition="desc", control_root=raw, aggregate_path=aggregate,
                                seal_path=seal, output=tmp_path / "desc-terminal.json")


def test_launch_safety_failure_precedes_mb4_cell_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    aggregate, seal = _reference(tmp_path)
    raw_controls, predecessor = tmp_path / "raw-controls", tmp_path / "asc-terminal.json"
    for item in MODULE.expected_predecessor_cells("asc"):
        _complete_control_cell(raw_controls, arm=item["arm"], fold=item["fold"])
    MODULE.seal_predecessor(partition="asc", control_root=raw_controls, aggregate_path=aggregate,
                            seal_path=seal, output=predecessor)
    work = tmp_path / "mb4-work"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(MODULE, "active_compute_commands_on_gpu", lambda gpu: (["busy"], []))
    with pytest.raises(MODULE.Mb4ContinuationError, match="active compute owner"):
        MODULE.execute_partition(partition="gpu1_after_asc", work_root=work, predecessor_terminal=predecessor,
                                 aggregate_path=aggregate, seal_path=seal)
    assert not MODULE.cell_paths(work, 0).cell.exists()


def test_predecessor_rc_binding_must_match_mb4_comparator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    aggregate, seal = _reference(tmp_path)
    raw_controls, predecessor = tmp_path / "raw-controls", tmp_path / "asc-terminal.json"
    for item in MODULE.expected_predecessor_cells("asc"):
        _complete_control_cell(raw_controls, arm=item["arm"], fold=item["fold"])
    MODULE.seal_predecessor(partition="asc", control_root=raw_controls, aggregate_path=aggregate,
                            seal_path=seal, output=predecessor)
    predecessor.chmod(0o644)
    body = json.loads(predecessor.read_text())
    body["r_c_reference"]["aggregate_sha256"] = "f" * 64
    _write(predecessor, body)
    predecessor.chmod(0o444)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(MODULE.Mb4ContinuationError, match="comparator SHA"):
        MODULE.execute_partition(partition="gpu1_after_asc", work_root=tmp_path / "mb4-work",
                                 predecessor_terminal=predecessor, aggregate_path=aggregate, seal_path=seal)


def test_all_fold_copy_import_finalizer_requires_full_matrix_and_keeps_negative(tmp_path: Path) -> None:
    aggregate, seal = _reference(tmp_path)
    partial_root, partial_import, partial_output = tmp_path / "partial", tmp_path / "partial-import", tmp_path / "partial.json"
    _complete_cell(partial_root, fold=0)
    with pytest.raises(MODULE.Mb4ContinuationError, match="exactly 15 complete"):
        MODULE.copy_import_all_and_finalize(source_root=partial_root, import_root=partial_import,
                                            aggregate_path=aggregate, seal_path=seal, output=partial_output)
    assert not partial_import.exists() and not partial_output.exists()

    raw, imported, output = tmp_path / "raw", tmp_path / "imported", tmp_path / "final.json"
    for fold in MODULE.FOLDS:
        _complete_cell(raw, fold=fold, r2=.9 if fold == 4 else .3)
    result = MODULE.copy_import_all_and_finalize(source_root=raw, import_root=imported,
                                                  aggregate_path=aggregate, seal_path=seal, output=output)
    assert result["status"] == "PASS_RT_MB4_COPY_IMPORT_ALL_AND_FINALIZE"
    body = json.loads(output.read_text())
    assert len(result["imports"]) == 15 and len(body["rows"]) == 15
    assert next(row for row in body["rows"] if row["fold"] == 4)["full_minus_mb4"] == pytest.approx(-.36)
    assert output.stat().st_mode & 0o777 == 0o444
