"""Synthetic/no-data contracts for the RT Stage-R B2/D1024 supervisor."""
from __future__ import annotations

from concurrent.futures import Future
import hashlib
import json
from pathlib import Path
import sys
import threading

import pytest

from scripts import run_rt_stage_r_b2_d1024_folds03_14_supervisor as supervisor


def _write_b2_d1024_target_free_fit(tmp_path: Path, *, fold: int = 3):
    """Write the relevant shape of a real saved Hydra + RT split receipt.

    In particular, Hydra persists the local interpolation literally rather
    than materializing ``data.query_start_trial`` as a YAML integer.  The
    unrelated Hydra runtime interpolation makes this a regression guard
    against resolving the whole saved config.
    """

    paths = supervisor._paths(tmp_path, fold)
    paths["config"].parent.mkdir(parents=True)
    paths["config"].write_text(
        """task_name: train
data:
  calibration_n_trials: 24
  query_start_trial: ${.calibration_n_trials}
  side_feature_group: zero4
model:
  freeze_decoder: false
  loss_mode: task_only
  id_hidden_dim: 1024
  variant: B2
callbacks:
  best_checkpoint:
    dirpath: ${hydra:runtime.output_dir}/checkpoints
""",
        encoding="utf-8",
    )
    manifest = {
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
    paths["split"].write_text(json.dumps(manifest), encoding="utf-8")
    checkpoint = paths["fit"] / "checkpoints" / "best_ckpt" / "epoch_001.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic checkpoint")
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
        "best_model_path": str(checkpoint),
        "best_model_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "config_path": str(paths["config"]),
        "config_sha256": hashlib.sha256(paths["config"].read_bytes()).hexdigest(),
        "split_manifest_path": str(paths["split"]),
        "split_manifest_sha256": hashlib.sha256(paths["split"].read_bytes()).hexdigest(),
    }
    paths["selection"].write_text(json.dumps(receipt), encoding="utf-8")
    return paths, manifest


def _refresh_selection_split_digest(paths: dict[str, Path]) -> None:
    receipt = json.loads(paths["selection"].read_text(encoding="utf-8"))
    receipt["split_manifest_sha256"] = hashlib.sha256(paths["split"].read_bytes()).hexdigest()
    paths["selection"].write_text(json.dumps(receipt), encoding="utf-8")


def test_default_interpreter_is_portable_active_python():
    """The CLI default must not bind a clean clone to one workstation path."""

    assert supervisor.DEFAULT_PYTHON == Path(sys.executable)
    source = Path(supervisor.__file__).read_text(encoding="utf-8")
    assert 'Path("/home/' not in source


def test_only_folds_3_through_14_are_admitted():
    assert supervisor._validate_folds([3, 4, 14]) == (3, 4, 14)
    with pytest.raises(ValueError, match="3..14"):
        supervisor._validate_folds([2])
    with pytest.raises(ValueError, match="3..14"):
        supervisor._validate_folds([15])


def test_b2_train_command_is_joint_decoder_m24_and_not_generic_b3_runner(tmp_path):
    paths = supervisor._paths(tmp_path, 3)
    command = supervisor.train_command(python=Path("/usr/bin/python3"), paths=paths, run_root=tmp_path, fold=3)
    text = " ".join(command)
    assert "experiment=rt_clean_nested_loso_b2_stage_r_zero4" in text
    assert "run_id=rt_clean_nested_loso_m24_b2_d1024_zero4" in text
    assert "data.loso_fold=3" in text and "data.outer_loso_fold=3" in text
    assert "model.id_hidden_dim=1024" in text and "test=false" in text
    assert "trainer.accelerator=gpu" in text
    assert "formal" not in text.lower()


def test_preflight_then_cpu_outer_evaluator_commands_are_explicit(tmp_path):
    paths = supervisor._paths(tmp_path, 4)
    preflight = supervisor.preflight_command(python=Path("/usr/bin/python3"), paths=paths, run_root=tmp_path, fold=4)
    assert str(supervisor.PREFLIGHT) in preflight and "--fold" in preflight and "4" in preflight
    selection = paths["selection"]
    selection.parent.mkdir(parents=True)
    checkpoint = paths["fit"] / "checkpoints" / "best_ckpt" / "epoch_008.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    import hashlib, json
    selection.write_text(json.dumps({
        "schema": "rt_clean_nested_loso_selection_receipt_v1", "status": "PASS_FIT_INNER_SELECTION_ONLY",
        "arm": "zero4", "seed": 42, "selected_by_metric": "val_heldin/r2_mean",
        "selected_metric_scope": "inner_validation_session_only", "formal_heldout_opened": False,
        "outer_target_loaded_during_fit": False, "outer_target_query_labels_read_during_fit": False,
        "best_model_path": str(checkpoint), "best_model_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "config_path": "x", "config_sha256": "x", "split_manifest_path": "x", "split_manifest_sha256": "x",
    }), encoding="utf-8")
    command = supervisor.eval_command(python=Path("/usr/bin/python3"), paths=paths, fold=4)
    text = " ".join(command)
    assert str(supervisor.OUTER_EVALUATOR) in text and "--device cpu" in text
    assert "--outer-fold 4" in text


def test_fit_identity_accepts_real_hydra_local_query_interpolation_and_target_free_split(tmp_path):
    paths, _ = _write_b2_d1024_target_free_fit(tmp_path, fold=3)

    # Real saved configs use `${.calibration_n_trials}`, while also retaining
    # unrelated `${hydra:...}` nodes that cannot be globally resolved here.
    receipt = supervisor._validate_fit_identity(paths, fold=3)

    assert receipt["outer_loso_fold"] == 3


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        (lambda value: value["query"].update(query_start_trial=23), "query.query_start_trial"),
        (lambda value: value["calibration"].update(budget_trials=23), "calibration.budget_trials"),
        (lambda value: value["calibration"].update(trial_index_range=[0, 23]), "calibration.trial_index_range"),
        (lambda value: value.update(target_session_loaded_during_fit=True), "target_session_loaded_during_fit"),
        (lambda value: value["nested_selection"].update(clean=False), "nested_selection.clean"),
        (
            lambda value: value["nested_selection"].update(outer_target_query_labels_read_during_fit=True),
            "nested_selection.outer_target_query_labels_read_during_fit",
        ),
        (lambda value: value["loaded_fit_sessions"].append(value["target_session"]), "loaded_fit_sessions"),
    ],
)
def test_fit_identity_rejects_any_target_free_m24_split_contract_violation(tmp_path, mutation, expected_message):
    paths, manifest = _write_b2_d1024_target_free_fit(tmp_path, fold=3)
    mutation(manifest)
    paths["split"].write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_selection_split_digest(paths)

    with pytest.raises(ValueError, match=expected_message):
        supervisor._validate_fit_identity(paths, fold=3)


def test_plan_is_nonexecuting_and_requires_fresh_cells(monkeypatch, tmp_path):
    monkeypatch.setattr(supervisor, "_require_program", lambda python: None)
    plan = supervisor.plan(run_root=tmp_path / "runs", folds=[3], python=Path("/usr/bin/python3"), gpu=1)
    assert plan["mode"] == "plan_only_no_execution"
    assert plan["formal_heldout_endpoint"] == "absent"
    assert plan["schema"] == "rt_stage_r_b2_d1024_folds03_14_pipeline_plan_v2"
    assert plan["execution_mode_if_continue"] == "continue_one_level_source_fit_cpu_outer_pipeline_v2"
    assert plan["overlap_contract"] == {
        "permitted_overlap": "CPU outer evaluation N with GPU source fit N+1 only",
        "max_concurrent_source_fits": 1,
        "max_concurrent_cpu_outer_evaluators": 1,
        "outer_eval_requires_validated_target_free_selection_config_split": True,
        "outer_eval_N_plus_1_starts_only_after_outer_eval_N_terminalized": True,
        "final_pending_outer_is_explicitly_harvested": True,
    }
    assert (tmp_path / "runs").exists() is False
    assert "outer_eval_cpu_after_selected_fit" in plan["cells"]["fold_03"]


def test_real_outer_executor_is_background_cpu_only_and_future_is_harvested(monkeypatch, tmp_path):
    """Exercise the real executor without launching an evaluator process."""

    worker_started = threading.Event()
    allow_worker_exit = threading.Event()
    observations: dict[str, object] = {}
    main_thread = threading.get_ident()
    paths = supervisor._paths(tmp_path / "runs", 3)
    prepared = supervisor.PreparedOuter(fold=3, paths=paths)

    def fake_eval_command(**kwargs):
        observations["eval_command_fold"] = kwargs["fold"]
        return ["synthetic-cpu-outer-eval"]

    def fake_run(command, *, cwd, env, log_path):
        observations["command"] = command
        observations["cwd"] = cwd
        observations["cuda_visible_devices"] = env["CUDA_VISIBLE_DEVICES"]
        observations["worker_thread"] = threading.get_ident()
        worker_started.set()
        assert allow_worker_exit.wait(timeout=2), "test must explicitly harvest the pending evaluator"
        return 0

    monkeypatch.setattr(supervisor, "eval_command", fake_eval_command)
    monkeypatch.setattr(supervisor, "_run", fake_run)
    monkeypatch.setattr(supervisor, "_validate_outer", lambda paths, *, fold: {"r2_variance_weighted": 0.0})

    cpu_env = {"CUDA_VISIBLE_DEVICES": "", "SYNTHETIC_TEST": "1"}
    with supervisor.ThreadPoolExecutor(max_workers=1, thread_name_prefix="synthetic-outer") as executor:
        pending = supervisor._launch_outer_eval(
            executor=executor, prepared=prepared, python=Path("/usr/bin/python3"), cpu_env=cpu_env
        )
        # `submit` returned on the main thread while the worker remains blocked;
        # this is the exact opportunity used for source fit N+1 in production.
        main_thread_progress = ["submit_returned", "source_fit_next_fold_can_start"]
        assert worker_started.wait(timeout=2)
        assert main_thread_progress[-1] == "source_fit_next_fold_can_start"
        assert pending.future.done() is False
        assert observations["cuda_visible_devices"] == ""
        assert observations["worker_thread"] != main_thread
        assert observations["eval_command_fold"] == 3
        allow_worker_exit.set()
        assert supervisor._harvest_outer(pending) == "passed"

    terminal = supervisor._json(paths["terminal"])
    assert terminal["schema"] == supervisor.TERMINAL_SCHEMA
    assert terminal["status"] == "PASS_CPU_ONE_SHOT_OUTER_EVAL"
    assert observations["command"] == ["synthetic-cpu-outer-eval"]
    assert observations["cwd"] == supervisor.STREAMING_ROOT


def test_continue_pipeline_orders_one_cpu_outer_around_next_gpu_fit_and_harvests_last(monkeypatch, tmp_path):
    """Fit 4 can overlap eval 3, but eval 4 waits for terminal 3."""

    events: list[str] = []
    active_outer = 0
    max_active_outer = 0
    worker_limits: list[int] = []

    class RecordingExecutor:
        def __init__(self, *, max_workers: int, thread_name_prefix: str):
            assert thread_name_prefix == "rt-stage-r-cpu-outer"
            worker_limits.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    def fake_prepare(*, fold, run_root, **kwargs):
        events.append(f"fit{fold}")
        return "source_ready", supervisor.PreparedOuter(fold=fold, paths=supervisor._paths(run_root, fold))

    def fake_launch(*, prepared, **kwargs):
        nonlocal active_outer, max_active_outer
        events.append(f"eval{prepared.fold}")
        active_outer += 1
        max_active_outer = max(max_active_outer, active_outer)
        return supervisor.PendingOuter(fold=prepared.fold, paths=prepared.paths, future=Future())

    def fake_harvest(pending):
        nonlocal active_outer
        assert active_outer == 1
        events.append(f"finalize{pending.fold}")
        active_outer -= 1
        return "passed"

    monkeypatch.setattr(supervisor, "_require_program", lambda python: None)
    monkeypatch.setattr(supervisor, "ThreadPoolExecutor", RecordingExecutor)
    monkeypatch.setattr(supervisor, "_prepare_source_fit", fake_prepare)
    monkeypatch.setattr(supervisor, "_launch_outer_eval", fake_launch)
    monkeypatch.setattr(supervisor, "_harvest_outer", fake_harvest)

    outcomes = supervisor.execute(
        run_root=tmp_path / "runs", folds=[3, 4], python=Path("/usr/bin/python3"), gpu=1,
        failure_policy="continue",
    )

    assert events == ["fit3", "eval3", "fit4", "finalize3", "eval4", "finalize4"]
    assert worker_limits == [1]
    assert max_active_outer == 1 and active_outer == 0
    assert outcomes == {"fold_03": "passed", "fold_04": "passed"}
    summary = (tmp_path / "runs" / "supervisor_summary.json").read_text(encoding="utf-8")
    assert supervisor.SUMMARY_SCHEMA in summary
    assert "continue_one_level_source_fit_cpu_outer_pipeline_v2" in summary


def test_source_selection_audit_failure_never_constructs_or_opens_target_evaluator(monkeypatch, tmp_path):
    """The pipeline capability is withheld before the target-eval command exists."""

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return 0

    monkeypatch.setattr(supervisor, "_run", fake_run)
    monkeypatch.setattr(supervisor, "_validate_preflight", lambda path, *, fold: {})
    monkeypatch.setattr(
        supervisor, "_validate_fit_identity",
        lambda paths, *, fold: (_ for _ in ()).throw(ValueError("selection/config/split audit failed")),
    )
    monkeypatch.setattr(
        supervisor, "eval_command",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("target evaluator must not be constructed")),
    )

    cpu_env = {"CUDA_VISIBLE_DEVICES": ""}
    gpu_env = {"CUDA_VISIBLE_DEVICES": "1"}
    outcome, prepared = supervisor._prepare_source_fit(
        run_root=tmp_path / "runs", fold=3, python=Path("/usr/bin/python3"), gpu=1,
        cpu_env=cpu_env, gpu_env=gpu_env,
    )

    assert outcome == "supervisor_exception" and prepared is None
    assert len(calls) == 2  # target-free preflight and source fit only
    assert all(str(supervisor.OUTER_EVALUATOR) not in command for command in calls)
    terminal = supervisor._json(supervisor._paths(tmp_path / "runs", 3)["terminal"])
    assert terminal["stage"] == "source_audit"
    assert terminal["formal_heldout_opened"] is False


def test_stop_policy_remains_strict_serial_and_never_calls_pipeline(monkeypatch, tmp_path):
    events: list[int] = []

    def fake_serial(*, fold, **kwargs):
        events.append(fold)
        return "passed"

    monkeypatch.setattr(supervisor, "_require_program", lambda python: None)
    monkeypatch.setattr(supervisor, "run_fold", fake_serial)
    monkeypatch.setattr(
        supervisor, "_execute_continue_pipeline",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("stop must not enter pipeline")),
    )

    outcomes = supervisor.execute(
        run_root=tmp_path / "runs", folds=[3, 4], python=Path("/usr/bin/python3"), gpu=1,
        failure_policy="stop",
    )

    assert events == [3, 4]
    assert outcomes == {"fold_03": "passed", "fold_04": "passed"}
    summary = (tmp_path / "runs" / "supervisor_summary.json").read_text(encoding="utf-8")
    assert "stop_strict_serial_no_speculative_source_fit_v2" in summary
