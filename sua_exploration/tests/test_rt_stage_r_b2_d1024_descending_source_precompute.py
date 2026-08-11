"""Synthetic/no-data contracts for the RT descending source-precompute harness."""
from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest

from scripts import run_rt_stage_r_b2_d1024_descending_source_precompute as precompute
from scripts import run_rt_stage_r_b2_d1024_folds03_14_supervisor as stage_r


def _write_valid_fit(paths: dict[str, Path], *, fold: int) -> Path:
    """Materialize only source-fit artifacts with a real Hydra local reference."""

    paths["config"].parent.mkdir(parents=True)
    paths["config"].write_text(
        """data:
  calibration_n_trials: 24
  query_start_trial: ${.calibration_n_trials}
  side_feature_group: zero4
model:
  freeze_decoder: false
  loss_mode: task_only
  id_hidden_dim: 1024
  variant: B2
callbacks:
  unrelated_runtime_path: ${hydra:runtime.output_dir}
""",
        encoding="utf-8",
    )
    split = {
        "protocol": {"support_budget_trials": 24},
        "task": "rt",
        "development_only": True,
        "formal_heldout_opened": False,
        "validation_protocol": "nested_loso",
        "outer_loso_fold": fold,
        "loso_fold": fold,
        "requested_side_feature_group": "zero4",
        "nested_selection": {
            "clean": True,
            "outer_target_loaded_during_fit": False,
            "outer_target_query_labels_read_during_fit": False,
            "inner_validation_only_for_checkpoint_selection": True,
            "checkpoint_metric": "val_heldin/r2_mean",
            "checkpoint_metric_scope": "inner_validation_session_only",
        },
        "calibration": {
            "budget_trials": 24,
            "trial_index_range": [0, 24],
            "target_calibration_optimizer_steps": 0,
        },
        "query": {"query_start_trial": 24},
        "target_session": "ses-RT-target",
        "target_session_loaded_during_fit": False,
        "target_query_window_audit": None,
        "loaded_fit_sessions": ["ses-RT-source-a", "ses-RT-source-b"],
        "outer_source_sessions": ["ses-RT-source-a", "ses-RT-source-b"],
        "inner_train_sessions": ["ses-RT-source-a"],
        "inner_validation_session": "ses-RT-source-b",
    }
    paths["split"].write_text(json.dumps(split), encoding="utf-8")
    checkpoint = paths["fit"] / "checkpoints" / "best_ckpt" / "epoch_001.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"synthetic source-only checkpoint")
    receipt = {
        "schema": "rt_clean_nested_loso_selection_receipt_v1",
        "status": "PASS_FIT_INNER_SELECTION_ONLY",
        "arm": "zero4",
        "seed": 42,
        "outer_loso_fold": fold,
        "selected_by_metric": "val_heldin/r2_mean",
        "selected_metric_scope": "inner_validation_session_only",
        "formal_heldout_opened": False,
        "outer_target_loaded_during_fit": False,
        "outer_target_query_labels_read_during_fit": False,
        "selected_epoch": 1,
        "selected_global_step": 32,
        "best_model_path": str(checkpoint),
        "best_model_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "config_path": str(paths["config"]),
        "config_sha256": hashlib.sha256(paths["config"].read_bytes()).hexdigest(),
        "split_manifest_path": str(paths["split"]),
        "split_manifest_sha256": hashlib.sha256(paths["split"].read_bytes()).hexdigest(),
    }
    paths["selection"].write_text(json.dumps(receipt), encoding="utf-8")
    return checkpoint


def _write_valid_preflight(path: Path, *, fold: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "rt_stage_r_b2_zero4_constructibility_preflight_v2",
                "status": "READY_NOT_LAUNCHED",
                "fold": fold,
                "nwb_opened": False,
                "cuda_touched": False,
                "arms": {
                    "R-S": {
                        "id_hidden_dim": 1024,
                        "cost": {"variant": "B2"},
                        "object_construction": {
                            "datamodule_setup_called": False,
                            "outer_target_loaded": False,
                            "outer_target_query_labels_read": False,
                            "model_teacher_initialized": False,
                            "model_student_initialized": False,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_default_scope_starts_only_at_fold14_and_requires_contiguous_descending_order():
    assert precompute.DEFAULT_FOLDS == (14,)
    assert precompute.DEFAULT_GPU == 0
    assert precompute._validate_descending_folds([14, 13, 12]) == (14, 13, 12)
    assert precompute._validate_descending_folds([3]) == (3,)
    with pytest.raises(ValueError, match="contiguous descending"):
        precompute._validate_descending_folds([14, 12])
    with pytest.raises(ValueError, match="contiguous descending"):
        precompute._validate_descending_folds([13, 14])
    with pytest.raises(ValueError, match="3..14"):
        precompute._validate_descending_folds([15])


def test_plan_reuses_existing_preflight_and_train_commands_but_has_no_target_phase(monkeypatch, tmp_path):
    monkeypatch.setattr(precompute, "_require_source_program", lambda python: None)
    plan = precompute.plan(
        run_root=tmp_path / "standard_run_root", folds=[14, 13], python=Path("/usr/bin/python3"), gpu=0
    )

    assert plan["mode"] == "plan_only_no_execution"
    assert plan["execution_order"] == "strict_descending_no_resume_v1"
    assert plan["gpu_for_source_fit_only"] == 0
    assert list(plan["cells"]) == ["fold_14", "fold_13"]
    command_text = "\n".join(
        value for cell in plan["cells"].values() for value in cell.values() if isinstance(value, str)
    )
    assert str(stage_r.PREFLIGHT) in command_text
    assert str(stage_r.TRAIN) in command_text
    assert "rt_clean_nested_loso_eval.py" not in command_text
    assert not (tmp_path / "standard_run_root").exists()


def test_successful_fit_emits_only_immutable_source_ready_receipt(monkeypatch, tmp_path):
    run_root = tmp_path / "standard_run_root"
    fold = 14
    paths = precompute._paths(run_root, fold)
    calls: list[list[str]] = []

    monkeypatch.setattr(precompute, "_require_source_program", lambda python: None)

    def fake_run(command, *, cwd, env, log_path):
        calls.append(command)
        if str(stage_r.PREFLIGHT) in command:
            _write_valid_preflight(paths["preflight"], fold=fold)
            return 0
        if str(stage_r.TRAIN) in command:
            _write_valid_fit(paths, fold=fold)
            return 0
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(stage_r, "_run", fake_run)
    receipt_path = precompute.prepare_fold(
        run_root=run_root, fold=fold, python=Path("/usr/bin/python3"), gpu=0
    )

    receipt = precompute._validate_source_ready(receipt_path, fold=fold)
    assert receipt["status"] == "PASS_SOURCE_ONLY_FIT_READY"
    assert receipt["source_fit"]["checkpoint_epoch"] == 1
    assert receipt["source_fit"]["checkpoint_global_step"] == 32
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o444
    assert receipt["artifacts"]["selected_checkpoint"]["sha256"] == hashlib.sha256(
        (paths["fit"] / "checkpoints" / "best_ckpt" / "epoch_001.ckpt").read_bytes()
    ).hexdigest()
    assert len(calls) == 2
    assert all("rt_clean_nested_loso_eval.py" not in " ".join(command) for command in calls)
    assert not paths["outer"].exists()
    assert not paths["terminal"].exists()
    assert not (run_root / "supervisor_summary.json").exists()


@pytest.mark.parametrize("artifact", ("terminal", "source_ready", "outer"))
def test_any_existing_terminal_ready_or_target_artifact_is_fail_closed(monkeypatch, tmp_path, artifact):
    run_root = tmp_path / "standard_run_root"
    paths = precompute._paths(run_root, 14)
    paths[artifact].parent.mkdir(parents=True, exist_ok=True)
    paths[artifact].write_text("existing", encoding="utf-8")
    invoked = False

    monkeypatch.setattr(precompute, "_require_source_program", lambda python: None)

    def forbidden_run(*args, **kwargs):
        nonlocal invoked
        invoked = True
        raise AssertionError("collision must be rejected before preflight or fit")

    monkeypatch.setattr(stage_r, "_run", forbidden_run)
    with pytest.raises(FileExistsError, match="collision"):
        precompute.prepare_fold(
            run_root=run_root, fold=14, python=Path("/usr/bin/python3"), gpu=0
        )
    assert invoked is False


def test_existing_partial_cell_is_fail_closed_without_terminalizing(monkeypatch, tmp_path):
    run_root = tmp_path / "standard_run_root"
    paths = precompute._paths(run_root, 14)
    paths["preflight"].parent.mkdir(parents=True, exist_ok=True)
    paths["preflight"].write_text("partial", encoding="utf-8")
    monkeypatch.setattr(precompute, "_require_source_program", lambda python: None)
    monkeypatch.setattr(stage_r, "_run", lambda *args, **kwargs: pytest.fail("must not run a partial cell"))

    with pytest.raises(FileExistsError, match="partial"):
        precompute.prepare_fold(
            run_root=run_root, fold=14, python=Path("/usr/bin/python3"), gpu=0
        )
    assert not paths["source_ready"].exists()
    assert not paths["terminal"].exists()


def test_preflight_failure_leaves_partial_evidence_without_terminal_or_summary(monkeypatch, tmp_path):
    run_root = tmp_path / "standard_run_root"
    paths = precompute._paths(run_root, 14)
    monkeypatch.setattr(precompute, "_require_source_program", lambda python: None)
    monkeypatch.setattr(stage_r, "_run", lambda *args, **kwargs: 17)

    with pytest.raises(RuntimeError, match="preflight failed"):
        precompute.prepare_fold(
            run_root=run_root, fold=14, python=Path("/usr/bin/python3"), gpu=0
        )
    assert not paths["source_ready"].exists()
    assert not paths["terminal"].exists()
    assert not (run_root / "supervisor_summary.json").exists()


def test_harness_source_never_calls_stage_r_target_or_terminal_helpers():
    source = Path(precompute.__file__).read_text(encoding="utf-8")
    prohibited = (
        "stage_r.eval_command(",
        "stage_r._validate_outer(",
        "stage_r._terminal_payload(",
        "stage_r._write_summary(",
        "stage_r.OUTER_EVALUATOR",
    )
    assert all(token not in source for token in prohibited)
