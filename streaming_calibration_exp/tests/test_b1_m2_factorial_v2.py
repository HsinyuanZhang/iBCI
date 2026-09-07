"""CPU-only contract tests for B1's staged matched carrier factorial."""
from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import sys
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from src.data import falcon_datamodule
from src.data.b1_m2_matched_z4_datamodule import (
    B1M2MatchedZ4DataModule,
    _MaskStandardizedT4Dataset,
)
from src.metrics import b1_m2_factorial as core
from scripts.preflight_b1_m2_factorial import build_preflight
from scripts import score_b1_m2_factorial_epochs as b1_scorer
from scripts.score_b1_m2_factorial_epochs import (
    _instantiate_and_strict_load_checkpoint,
    _verify_post_training_binding,
)
from scripts.execute_b1_m2_factorial_cell import build_plan, execute_plan
from scripts.run_b1_m2_factorial import main as legacy_runner_main


def _receipt(cell: core.CellSpec, score: float, *, session: str | None = None) -> dict[str, Any]:
    session = session or f"session-fold-{cell.fold}"
    return {
        "schema_version": core.SCHEMA_VERSION,
        "screen_id": core.SCREEN_ID,
        "stage": cell.stage,
        "fold": cell.fold,
        "seed": cell.seed,
        "carrier": cell.carrier,
        "loss_mode": cell.loss_mode,
        "development_scope_only": True,
        "formal_test_or_external_heldout_opened": False,
        "target_session_backward_updates": False,
        "target_session_decoder_weight_updates": False,
        "target_session_weight_updates_during_b1": False,
        "teacher_provenance": {
            "during_B1_target_weight_updates": False,
            "legacy_frozen_teacher_pretraining_included_B1_validation_session": True,
            "clean_teacher_target_exclusion": False,
            "interpretation": "internal_development_only",
        },
        "query_provenance": {
            "support_direction_labels_used_for_carrier": True,
            "query_behavior_loaded_for_validation_scoring": True,
            "query_behavior_used_for_gradient_updates": False,
            "query_behavior_used_for_carrier_fit": False,
            "query_behavior_used_for_normalizer_fit": False,
            "query_behavior_used_for_checkpoint_selection": False,
            "query_behavior_used_for_stage_f_continuation_gate": cell.stage == "P",
            "external_heldout_opened": False,
        },
        "official_preflight_sha256": "a" * 64,
        "official_preflight": {
            "implementation_bindings": {"fixed.py": "b" * 64},
            "implementation_bindings_sha256": core.sha256_payload({"fixed.py": "b" * 64}),
            "full_preflight_sha256": "a" * 64,
        },
        "source_artifact": {
            "resolved_config_sha256": "c" * 64,
            "science_config_sha256": "d" * 64,
            "source_manifest_sha256": "e" * 64,
            "split_manifest_sha256": "f" * 64,
            "teacher_metadata_sha256": "1" * 64,
            "post_training_path_binding_sha256": "4" * 64,
            "checkpoint_run_dir": "/independent/hydra/log/run",
            "execution_context": {
                "interpreter": "/independent/python",
                "script": "/independent/src/train.py",
                "working_dir": "/independent/streaming_calibration_exp",
            },
            "normalizer_binding": {"feature_group": "t4", "sha256": "2" * 64},
            "split_semantics": {"fold_id": cell.fold},
            "checkpoint_bundle": {
                str(epoch): {"logical_epoch": epoch, "stored_epoch": epoch - 1, "sha256": "3" * 64}
                for epoch in core.EPOCH_WINDOW
            },
        },
        "carrier_provenance": {
            "carrier_fit_executed": True,
            "calibration_direction_labels_read": True,
            "target_session_carrier_fit_executed": True,
            "target_session_direction_labels_used_for_carrier": True,
            "target_session_query_labels_used_for_carrier": False,
            "source_normalizer_fit_only": True,
            "model_visible_carrier": "standardized_t4" if cell.carrier == "t4" else "all_zero_mask_after_standardized_t4",
        },
        "loss": {
            "lambda_y": 1.0,
            "lambda_E": 0.0 if cell.loss_mode == "task_plus_y" else 0.1,
        },
        "epoch_window": list(core.EPOCH_WINDOW),
        "per_epoch_session_r2": {str(epoch): {session: score} for epoch in core.EPOCH_WINDOW},
    }


def _stage_receipts(stage: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for cell in core.cells_for_stage(stage):
        # Interaction = (0.70 - 0.40) - (0.50 - 0.40) = +0.20.
        score = {
            ("t4", "task_plus_y"): 0.70,
            ("z4", "task_plus_y"): 0.40,
            ("t4", "task_plus_y_plus_E"): 0.50,
            ("z4", "task_plus_y_plus_E"): 0.40,
        }[(cell.carrier, cell.loss_mode)]
        rows[cell.key] = _receipt(cell, score)
    return rows


def _sealed_stage(stage: str) -> dict[str, Any]:
    result = core.aggregate_stage(stage, _stage_receipts(stage))
    result["cell_receipt_sha256"] = {cell.key: "5" * 64 for cell in core.cells_for_stage(stage)}
    return result


def _official_preflight(tmp_path: Path) -> tuple[Path, str]:
    root = Path(__file__).resolve().parents[2]
    payload = build_preflight(root)
    path = tmp_path / "official_preflight.json"
    digest = core.write_immutable_json(path, payload)
    return path, digest


def _p_plan(tmp_path: Path):
    preflight, _digest = _official_preflight(tmp_path)
    return build_plan(
        stage="P", fold=0, seed=42, carrier="t4", loss_mode="task_plus_y",
        cuda_visible_devices="0", official_preflight=preflight, stage_p_aggregate=None,
        result_root=tmp_path / "results",
    )


def _write_launch(
    tmp_path: Path,
    *,
    cell: core.CellSpec,
    log_dir: Path,
    artifact_parent: Path,
    prefix: str,
    command: list[str] | None = None,
    context: dict[str, Path] | None = None,
) -> dict[str, Any]:
    context = context or {
        "interpreter": Path("/usr/bin/python3"),
        "script": Path("/independent/src/train.py"),
        "working_dir": Path("/independent/streaming_calibration_exp"),
    }
    context = {key: value.resolve() for key, value in context.items()}
    command = command or [str(context["interpreter"]), str(context["script"]), "train=true", "test=false"]
    paths = {
        "launch": tmp_path / "launch.json",
        "start": tmp_path / "execution.start.json",
        "completion": tmp_path / "execution.completion.json",
        "log": tmp_path / "execution.log",
        "binding": tmp_path / "post.json",
        "score": tmp_path / "score.json",
    }
    core.write_future_launch_receipt(
        paths["launch"], spec=cell, official_preflight_sha256="a" * 64,
        command=command, explicit_log_dir=log_dir, artifact_parent=artifact_parent,
        run_id_prefix=prefix, future_post_training_binding=paths["binding"],
        future_score_receipt=paths["score"],
        cuda_visible_devices="0", execution_start_receipt=paths["start"],
        execution_completion_receipt=paths["completion"], execution_log=paths["log"],
        **context,
    )
    paths.update({"command": command, "context": context})
    return paths


def _complete_launch(paths: dict[str, Any], *, exit_code: int = 0) -> None:
    core.write_execution_start_receipt(
        paths["start"], future_launch_receipt=paths["launch"], cuda_visible_devices="0",
    )
    paths["log"].write_text("subprocess log\n", encoding="utf-8")
    core.write_execution_completion_receipt(
        paths["completion"], future_launch_receipt=paths["launch"],
        execution_start_receipt=paths["start"], invoked_command=paths["command"],
        invoked_working_dir=paths["context"]["working_dir"],
        invoked_environment=core.execution_environment_contract("0"),
        subprocess_started=True, exit_code=exit_code,
    )


def _write_score_path_binding(tmp_path: Path, *, launch: Path, binding: Path) -> Path:
    """Create the minimal immutable bridge needed by scorer's early path check.

    The full binding helper also requires completed training artifacts; this
    fixture intentionally stops before that point so the CLI regression proves
    a wrong ``--out`` cannot reach artifact audit or any runtime path.
    """
    future, future_sha = core.load_verified_immutable_json(launch)
    core.write_immutable_json(
        binding,
        {
            "schema_version": core.SCHEMA_VERSION,
            "receipt_kind": "b1_m2_factorial_post_training_path_binding",
            "screen_id": core.SCREEN_ID,
            "cell": future["cell"],
            "future_launch_receipt_path": str(launch.resolve()),
            "future_launch_receipt_sha256": future_sha,
        },
    )
    return binding


def _remint(path: Path, payload: dict[str, Any]) -> None:
    """Test-only reissue with a valid sidecar, simulating a forged receipt."""
    for target in (path, Path(f"{path}.sha256")):
        target.chmod(0o600)
        target.unlink()
    core.write_immutable_json(path, payload)


def test_scorer_rejects_unreserved_out_before_artifact_audit_or_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct scorer use cannot redirect a cell after its launch is bound.

    Use ``--execute`` deliberately: both the dry artifact audit and the
    heavyweight execution function are patched to fail the test if reached.
    Thus the assertion covers the required pre-Torch/pre-forward boundary,
    not merely the eventual immutable writer's O_EXCL behavior.
    """
    preflight, _preflight_sha = _official_preflight(tmp_path)
    cell = core.cells_for_stage("P")[0]
    paths = _write_launch(
        tmp_path,
        cell=cell,
        log_dir=tmp_path / "hydra-log",
        artifact_parent=tmp_path / "artifact-parent",
        prefix="b1_v2_P_f0_s42_t4_task_plus_y",
    )
    binding = _write_score_path_binding(
        tmp_path, launch=paths["launch"], binding=tmp_path / "post-training-binding.json",
    )
    wrong_out = tmp_path / "redirected-score.json"
    reached: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        reached.append("runtime")
        raise AssertionError("scorer advanced beyond its reserved-output guard")

    monkeypatch.setattr(b1_scorer, "_dry_run_payload", forbidden)
    monkeypatch.setattr(b1_scorer, "_execute", forbidden)
    monkeypatch.setattr(sys, "argv", [
        "score_b1_m2_factorial_epochs.py",
        "--stage", cell.stage,
        "--fold", str(cell.fold),
        "--seed", str(cell.seed),
        "--carrier", cell.carrier,
        "--loss-mode", cell.loss_mode,
        "--official-preflight", str(preflight),
        "--checkpoint-run-dir", str(tmp_path / "never-open-log-dir"),
        "--post-training-binding", str(binding),
        "--artifact", str(tmp_path / "never-open-artifact"),
        "--out", str(wrong_out),
        "--execute",
        "--device", "cpu",
    ])
    with pytest.raises(core.B1ContractError, match="--out must exactly match"):
        b1_scorer.main()
    assert reached == []
    for path in (wrong_out, paths["score"]):
        assert not path.exists()
        assert not Path(f"{path}.sha256").exists()


def test_scorer_reserved_output_guard_rejects_tamper_and_missing_field(tmp_path: Path) -> None:
    cell = core.cells_for_stage("P")[0]
    paths = _write_launch(
        tmp_path,
        cell=cell,
        log_dir=tmp_path / "hydra-log",
        artifact_parent=tmp_path / "artifact-parent",
        prefix="b1_v2_P_f0_s42_t4_task_plus_y",
    )
    future, _future_sha = core.load_verified_immutable_json(paths["launch"])
    future.pop("future_score_receipt")
    _remint(paths["launch"], future)
    missing_field_binding = _write_score_path_binding(
        tmp_path,
        launch=paths["launch"],
        binding=tmp_path / "missing-field-post-training-binding.json",
    )
    with pytest.raises(core.B1ContractError, match="lacks future_score_receipt"):
        b1_scorer._verify_reserved_score_output(missing_field_binding, tmp_path / "score.json")

    tampered_binding = tmp_path / "tampered-post-training-binding.json"
    payload, _digest = core.load_verified_immutable_json(missing_field_binding)
    payload["future_launch_receipt_sha256"] = "0" * 64
    core.write_immutable_json(tampered_binding, payload)
    with pytest.raises(core.B1ContractError, match="future launch SHA drift"):
        b1_scorer._verify_reserved_score_output(tampered_binding, tmp_path / "score.json")


@pytest.mark.parametrize("stale_kind", ("body", "sidecar"))
def test_scorer_reserved_output_guard_rechecks_body_and_sidecar_freshness(
    tmp_path: Path, stale_kind: str,
) -> None:
    """A post-launch collision fails before any scoring work begins."""
    cell = core.cells_for_stage("P")[0]
    paths = _write_launch(
        tmp_path,
        cell=cell,
        log_dir=tmp_path / "hydra-log",
        artifact_parent=tmp_path / "artifact-parent",
        prefix="b1_v2_P_f0_s42_t4_task_plus_y",
    )
    binding = _write_score_path_binding(
        tmp_path, launch=paths["launch"], binding=tmp_path / "post-training-binding.json",
    )
    stale = paths["score"] if stale_kind == "body" else Path(f"{paths['score']}.sha256")
    stale.write_text("stranded\n", encoding="ascii")
    with pytest.raises(core.B1ContractError, match="reserved score output is not fresh"):
        b1_scorer._verify_reserved_score_output(binding, paths["score"])


def test_stage_p_requires_complete_crossed_factorial_and_gates_stage_f() -> None:
    result = core.aggregate_stage("P", _stage_receipts("P"))
    assert result["cell_count"] == 12
    assert result["mean_interaction"] == pytest.approx(0.20)
    assert result["stage_f_predeclared_gate"]["stage_f_authorized_by_evidence"] is True

    broken = _stage_receipts("P")
    broken.pop(next(iter(broken)))
    with pytest.raises(core.B1ContractError, match="receipt lattice mismatch"):
        core.aggregate_stage("P", broken)


def test_stage_p_aggregate_validator_recomputes_lattice_mean_and_gate() -> None:
    sealed = _sealed_stage("P")
    core.validate_stage_p_aggregate(sealed, official_preflight_sha256="a" * 64)

    bad_mean = _sealed_stage("P")
    bad_mean["mean_interaction"] = 0.0
    with pytest.raises(core.B1ContractError, match="mean does not equal"):
        core.validate_stage_p_aggregate(bad_mean, official_preflight_sha256="a" * 64)

    bad_gate = _sealed_stage("P")
    bad_gate["stage_f_predeclared_gate"]["stage_f_authorized_by_evidence"] = False
    with pytest.raises(core.B1ContractError, match="continuation gate drift"):
        core.validate_stage_p_aggregate(bad_gate, official_preflight_sha256="a" * 64)

    bad_lattice = _sealed_stage("P")
    bad_lattice["cell_receipt_sha256"].pop(next(iter(bad_lattice["cell_receipt_sha256"])))
    with pytest.raises(core.B1ContractError, match="cell-receipt lattice"):
        core.validate_stage_p_aggregate(bad_lattice, official_preflight_sha256="a" * 64)


def test_one_cell_executor_default_dry_run_writes_nothing_and_has_exact_train_contract(tmp_path: Path) -> None:
    plan = _p_plan(tmp_path)
    payload = execute_plan(plan, execute=False)
    assert payload["default_dry_run"] is True
    assert payload["gpu_subprocess_started"] is False
    assert payload["command"] == [
        str(Path(__import__("sys").executable).resolve()),
        str((Path(__file__).resolve().parents[1] / "src/train.py").resolve()),
        "experiment=b1_v2_m2_t4_task_plus_y", "seed=42", "data.loso_fold=0",
        "train=true", "test=false", "run_id=b1_v2_P_f0_s42_t4_task_plus_y",
        f"hydra.run.dir={plan.paths.log_dir}", f"paths.artifact_dir={plan.paths.artifact_parent}",
    ]
    assert not plan.paths.launch_receipt.exists()
    assert not plan.paths.execution_start_receipt.exists()


def test_legacy_runner_refuses_launch_and_has_no_prepare_receipt_bypass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight, _digest = _official_preflight(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "run_b1_m2_factorial.py", "--stage", "P", "--official-preflight", str(preflight), "--launch",
    ])
    with pytest.raises(SystemExit, match="Refusing GPU launch"):
        legacy_runner_main()
    monkeypatch.setattr(sys, "argv", [
        "run_b1_m2_factorial.py", "--stage", "P", "--official-preflight", str(preflight),
        "--prepare-launch-receipts",
    ])
    with pytest.raises(SystemExit) as exc:
        legacy_runner_main()
    assert exc.value.code == 2
    assert not (tmp_path / "launch.json").exists()


def test_one_cell_executor_rejects_stage_f_tampered_routing_gate(tmp_path: Path) -> None:
    preflight, preflight_sha = _official_preflight(tmp_path)
    stage_p = _sealed_stage("P")
    stage_p["official_preflight_sha256"] = preflight_sha
    stage_p["stage_f_predeclared_gate"]["stage_f_authorized_by_evidence"] = False
    aggregate = tmp_path / "tampered_stage_p.json"
    core.write_immutable_json(aggregate, stage_p)
    with pytest.raises(core.B1ContractError, match="continuation gate drift"):
        build_plan(
            stage="F", fold=1, seed=42, carrier="t4", loss_mode="task_plus_y",
            cuda_visible_devices="0", official_preflight=preflight, stage_p_aggregate=aggregate,
            result_root=tmp_path / "results",
        )


def test_one_cell_executor_records_failure_with_exact_cwd_env_and_no_binding(tmp_path: Path) -> None:
    plan = _p_plan(tmp_path)
    calls: list[dict[str, Any]] = []

    def failed_run(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return __import__("subprocess").CompletedProcess(command, 17)

    result = execute_plan(plan, execute=True, runner=failed_run)
    assert result["status"] == "subprocess_failed_no_binding_or_scoring"
    assert result["subprocess_exit_code"] == 17
    assert len(calls) == 1
    assert calls[0]["command"] == list(plan.command)
    assert calls[0]["cwd"] == str(plan.working_dir)
    assert calls[0]["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert calls[0]["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert plan.paths.execution_start_receipt.is_file()
    started, _ = core.load_verified_immutable_json(plan.paths.execution_start_receipt)
    assert started["execution_attempted"] is True
    assert started["subprocess_started"] is False
    assert started["subprocess_invocation_pending_at_receipt_write"] is True
    completion, _ = core.load_verified_immutable_json(plan.paths.execution_completion_receipt)
    assert completion["subprocess_started"] is True
    assert completion["subprocess_exit_code"] == 17
    assert completion["training_completed_successfully"] is False
    assert not plan.paths.post_training_binding.exists()


def test_one_cell_executor_fails_closed_for_existing_paths_and_never_overwrites(tmp_path: Path) -> None:
    plan = _p_plan(tmp_path)
    plan.paths.log_dir.mkdir(parents=True)
    with pytest.raises(core.B1ContractError, match="freshness"):
        execute_plan(plan, execute=True, runner=lambda *_args, **_kwargs: pytest.fail("must not spawn"))

    plan = _p_plan(tmp_path / "second")
    def failed_run(command, **_kwargs):
        return __import__("subprocess").CompletedProcess(command, 9)
    execute_plan(plan, execute=True, runner=failed_run)
    with pytest.raises(core.B1ContractError, match="freshness"):
        execute_plan(plan, execute=True, runner=failed_run)


def test_one_cell_executor_rejects_ambiguous_artifact_after_success(tmp_path: Path) -> None:
    plan = _p_plan(tmp_path)

    def successful_train_with_two_artifacts(command, **_kwargs):
        plan.paths.artifact_parent.mkdir(parents=True)
        (plan.paths.artifact_parent / "b1_v2_P_f0_s42_t4_task_plus_y_f0_s42_one").mkdir()
        (plan.paths.artifact_parent / "b1_v2_P_f0_s42_t4_task_plus_y_f0_s42_two").mkdir()
        return __import__("subprocess").CompletedProcess(command, 0)

    with pytest.raises(core.B1ContractError, match="exactly one matching artifact"):
        execute_plan(plan, execute=True, runner=successful_train_with_two_artifacts)
    completion, _ = core.load_verified_immutable_json(plan.paths.execution_completion_receipt)
    assert completion["subprocess_exit_code"] == 0


def test_final_uses_stage_f_only_for_confirmatory_terminal_and_keeps_p_plus_f_descriptive() -> None:
    stage_p = _sealed_stage("P")
    stage_f = _sealed_stage("F")
    final = core.aggregate_final(stage_p, stage_f)
    assert final["inference_unit"] == "crossed 3 predeclared Stage-F LOSO sessions × 3 fixed seeds"
    assert len(final["interaction_by_fold_seed"]) == 9
    assert final["terminal_gate"]["terminal_pass"] is True
    assert final["crossed_bootstrap"]["rng_seed"] == core.CROSSED_BOOTSTRAP_SEED
    assert len(final["session_means"]) == 3
    assert len(final["seed_means"]) == 3
    assert final["crossed_bootstrap"] == core.aggregate_final(stage_p, stage_f)["crossed_bootstrap"]
    assert final["p_plus_f_descriptive_sensitivity_only"]["terminal_gate_applied"] is False
    assert len(final["p_plus_f_descriptive_sensitivity_only"]["interaction_by_fold_seed"]) == 12

    stage_p["stage_f_predeclared_gate"]["stage_f_authorized_by_evidence"] = False
    with pytest.raises(core.B1ContractError, match="not authorized"):
        core.aggregate_final(stage_p, stage_f)


class _TinyT4Dataset:
    window_indices = [("s", 0)]

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        assert index == 0
        side = np.asarray([[1.5, -2.0, 3.0, 4.0]], dtype=np.float32)
        return (np.ones((2, 1)), np.ones((2, 2)), np.ones((3, 2, 1)), "s", side)


def test_z4_wrapper_masks_only_final_standardized_side_tensor() -> None:
    base = _TinyT4Dataset()
    wrapped = _MaskStandardizedT4Dataset(base)
    original = base[0]
    masked = wrapped[0]
    for got, expected in zip(masked[:3], original[:3]):
        assert np.array_equal(got, expected)
    assert masked[3] == original[3]
    assert np.array_equal(masked[4], np.zeros_like(original[4]))
    # The source row/cache is not mutated by the post-standardization null.
    assert np.array_equal(base[0][4], original[4])
    assert wrapped.window_indices == base.window_indices


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_ROOT = _PROJECT_ROOT / "configs"
_B1_M2_DATA_ROOT = _PROJECT_ROOT.parent / "SPINT-main" / "data" / "000953"


def _compose_exact_b1_z4_runtime_config():
    """Compose the production Z4 cell, changing only its local path anchor.

    ``paths.root_dir`` is made absolute solely so this integration smoke has
    the same concrete data root when pytest is invoked from another directory.
    All B1 science fields (M33, LOSO, T4-before-mask, fit-only scope, and
    task-plus-y loss) come directly from the exact production experiment.
    """
    with initialize_config_dir(version_base="1.3", config_dir=str(_CONFIG_ROOT.resolve())):
        return compose(
            config_name="train.yaml",
            overrides=[
                "experiment=b1_v2_m2_z4_task_plus_y",
                f"paths.root_dir={_PROJECT_ROOT.resolve()}",
            ],
        )


@pytest.mark.skipif(
    not _B1_M2_DATA_ROOT.is_dir(),
    reason="requires the local nonsealed M2 held-in NWB fixture",
)
def test_exact_b1_z4_config_fit_setup_uses_path_and_never_loads_heldout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression for the real Z4-v8 setup path, without opening sealed data.

    This intentionally calls ``setup('fit')`` with the exact B1 Z4 experiment
    configuration.  The guard covers every Falcon ``load_nwb`` call and makes
    a held-out open an immediate failure; the experiment itself keeps both
    held-out flags false, so only held-in calibration/minival files are legal.
    """
    cfg = _compose_exact_b1_z4_runtime_config()
    assert cfg.data._target_ == "src.data.b1_m2_matched_z4_datamodule.B1M2MatchedZ4DataModule"
    assert cfg.data.task == "m2"
    assert cfg.data.side_feature_group == "t4"
    assert cfg.data.calibration_n_trials == 33
    assert cfg.data.validation_protocol == "loso"
    assert cfg.data.include_heldout_in_fit is False
    assert cfg.data.include_heldout_in_test is False
    assert cfg.data.b1_z4_mask_after_standardization is True

    datamodule = instantiate(cfg.data)
    assert isinstance(datamodule, B1M2MatchedZ4DataModule)
    # This is the direct regression assertion for the v8 failure.  The base
    # setup calls ``self.hparams.data_dir.rglob(...)`` for held-in files.
    assert isinstance(datamodule.hparams.data_dir, Path)
    assert datamodule._needs_heldout_data("fit") is False

    opened: list[Path] = []
    real_load_nwb = falcon_datamodule.load_nwb

    def _heldin_only_load_nwb(path: Path | str, *args: Any, **kwargs: Any):
        resolved = Path(path)
        opened.append(resolved)
        assert "held-out" not in resolved.as_posix()
        assert "held-in-" in resolved.as_posix()
        return real_load_nwb(path, *args, **kwargs)

    monkeypatch.setattr(falcon_datamodule, "load_nwb", _heldin_only_load_nwb)
    datamodule.setup("fit")

    assert opened, "the smoke must execute real held-in NWB loading"
    assert datamodule.val_heldout_dataset is None
    assert isinstance(datamodule.train_dataset, _MaskStandardizedT4Dataset)
    assert isinstance(datamodule.val_heldin_dataset, _MaskStandardizedT4Dataset)
    for dataset in (datamodule.train_dataset, datamodule.val_heldin_dataset):
        row = dataset[0]
        assert np.asarray(row[4]).ndim == 2
        assert np.asarray(row[4]).shape[-1] == 4
        assert np.array_equal(np.asarray(row[4]), np.zeros_like(np.asarray(row[4])))
    assert datamodule.get_split_manifest()["b1_z4_control"]["formal_or_external_heldout_files_opened"] is False


def test_cpu_preflight_staticly_binds_p_and_predeclared_f_without_runtime_io() -> None:
    root = Path(__file__).resolve().parents[2]
    payload = build_preflight(root)
    assert payload["operations"]["torch_imported"] is False
    assert payload["operations"]["nwb_opened"] is False
    assert len(payload["stage_p_cells"]) == 12
    assert len(payload["stage_f_cells"]) == 36
    assert payload["factorial"]["stage_f"]["folds"] == [1, 2, 3]
    assert payload["operations"]["teacher_checkpoint_bytes_hashed"] is True
    assert payload["operations"]["checkpoint_deserialized"] is False
    for row in payload["stage_p_cells"] + payload["stage_f_cells"]:
        assert row["science_config"]["train"] is True
        assert row["science_config"]["test"] is False
    assert "streaming_calibration_exp/src/train.py" in payload["implementation_bindings"]["files"]
    assert len(payload["implementation_bindings"]["teacher"]["checkpoint"]["sha256"]) == 64
    core.validate_preflight_payload(payload)


def test_aggregate_rejects_tampered_checkpoint_and_preflight_binding() -> None:
    rows = _stage_receipts("P")
    key = next(iter(rows))
    rows[key]["source_artifact"]["checkpoint_bundle"]["5"]["stored_epoch"] = 99
    with pytest.raises(core.B1ContractError, match="checkpoint epoch binding drift"):
        core.aggregate_stage("P", rows)

    rows = _stage_receipts("P")
    key = next(iter(rows))
    rows[key]["official_preflight"]["full_preflight_sha256"] = "z" * 64
    with pytest.raises(core.B1ContractError, match="embedded preflight SHA drift"):
        core.aggregate_stage("P", rows)


def test_final_rejects_session_seed_lattice_drift() -> None:
    p = core.aggregate_stage("P", _stage_receipts("P"))
    f_rows = _stage_receipts("F")
    # The same LOSO fold must identify the same validation session for all seeds.
    one = core.CellSpec("F", 1, 42, "t4", "task_plus_y")
    f_rows[one.key]["per_epoch_session_r2"] = {
        str(epoch): {"different-session": 0.7} for epoch in core.EPOCH_WINDOW
    }
    with pytest.raises(core.B1ContractError, match="disagree on validation session"):
        core.aggregate_stage("F", f_rows)


def test_future_launch_receipt_is_o_excl_and_fresh(tmp_path: Path) -> None:
    cell = core.cells_for_stage("P")[0]
    paths = _write_launch(
        tmp_path, cell=cell, log_dir=tmp_path / "new-log",
        artifact_parent=tmp_path / "artifact-parent", prefix="b1_v2_P_f0_s42_t4_task_plus_y",
    )
    _payload, digest = core.load_verified_immutable_json(paths["launch"])
    assert len(digest) == 64
    with pytest.raises(core.B1ContractError, match="freshness"):
        core.write_future_launch_receipt(
            paths["launch"], spec=cell, official_preflight_sha256="a" * 64,
            command=["/usr/bin/python3", "/independent/src/train.py"], explicit_log_dir=tmp_path / "other-log",
            artifact_parent=tmp_path / "other-artifact", run_id_prefix="b1",
            future_post_training_binding=tmp_path / "other-post-training.json",
            future_score_receipt=tmp_path / "other-score.json",
            interpreter=Path("/usr/bin/python3"), script=Path("/independent/src/train.py"),
            working_dir=Path("/independent/streaming_calibration_exp"),
            cuda_visible_devices="0", execution_start_receipt=tmp_path / "other.start.json",
            execution_completion_receipt=tmp_path / "other.completion.json",
            execution_log=tmp_path / "other.log",
        )


@pytest.mark.parametrize("stale_name", ("post.json", "score.json"))
def test_future_launch_receipt_rejects_stale_immutable_output_sidecars(
    tmp_path: Path, stale_name: str,
) -> None:
    """A stranded immutable sidecar must fail before any GPU subprocess can run."""
    cell = core.cells_for_stage("P")[0]
    post = tmp_path / "post.json"
    score = tmp_path / "score.json"
    Path(f"{tmp_path / stale_name}.sha256").write_text("stale\n", encoding="ascii")
    with pytest.raises(core.B1ContractError, match="freshness"):
        core.write_future_launch_receipt(
            tmp_path / "launch.json", spec=cell, official_preflight_sha256="a" * 64,
            command=["/usr/bin/python3", "/independent/src/train.py"],
            explicit_log_dir=tmp_path / "log", artifact_parent=tmp_path / "artifacts",
            run_id_prefix="b1_v2_P_f0_s42_t4_task_plus_y",
            future_post_training_binding=post, future_score_receipt=score,
            interpreter=Path("/usr/bin/python3"), script=Path("/independent/src/train.py"),
            working_dir=Path("/independent/streaming_calibration_exp"), cuda_visible_devices="0",
            execution_start_receipt=tmp_path / "start.json",
            execution_completion_receipt=tmp_path / "completion.json",
            execution_log=tmp_path / "train.log",
        )


def test_post_training_binding_requires_separate_committed_log_and_artifact_paths(tmp_path: Path) -> None:
    cell = core.cells_for_stage("P")[0]
    log_dir = tmp_path / "hydra-log"
    artifact_parent = tmp_path / "artifact-parent"
    artifact = artifact_parent / "b1_v2_P_f0_s42_t4_task_plus_y_20260813_120000"
    chain = _write_launch(
        tmp_path, cell=cell, log_dir=log_dir, artifact_parent=artifact_parent,
        prefix="b1_v2_P_f0_s42_t4_task_plus_y",
    )
    _complete_launch(chain)
    artifact.mkdir(parents=True)
    for name in ("resolved_config.yaml", "source_manifest.json", "split_manifest.json", "teacher_metadata.json"):
        (artifact / name).write_text("x", encoding="utf-8")
    epoch_dir = log_dir / "checkpoints" / "epoch_ckpts"
    epoch_dir.mkdir(parents=True)
    for epoch in core.EPOCH_WINDOW:
        (epoch_dir / f"epoch_{epoch - 1:03d}.ckpt").write_bytes(f"epoch{epoch}".encode())
    binding = chain["binding"]
    with pytest.raises(core.B1ContractError, match="requires a successful completion"):
        core.write_post_training_binding(
            tmp_path / "legacy-post.json", future_launch_receipt=chain["launch"], artifact=artifact,
            checkpoint_run_dir=log_dir, interpreter=Path("/usr/bin/python3"),
            script=Path("/independent/src/train.py"),
            working_dir=Path("/independent/streaming_calibration_exp"),
            execution_completion_receipt=None,  # type: ignore[arg-type]
        )
    with pytest.raises(core.B1ContractError, match="binding path differs"):
        core.write_post_training_binding(
            tmp_path / "wrong-post.json", future_launch_receipt=chain["launch"], artifact=artifact,
            checkpoint_run_dir=log_dir, interpreter=Path("/usr/bin/python3"),
            script=Path("/independent/src/train.py"),
            working_dir=Path("/independent/streaming_calibration_exp"),
            execution_completion_receipt=chain["completion"],
        )
    digest = core.write_post_training_binding(
        binding, future_launch_receipt=chain["launch"], artifact=artifact, checkpoint_run_dir=log_dir,
        interpreter=Path("/usr/bin/python3"), script=Path("/independent/src/train.py"),
        working_dir=Path("/independent/streaming_calibration_exp"),
        execution_completion_receipt=chain["completion"],
    )
    assert len(digest) == 64
    payload, _ = core.load_verified_immutable_json(binding)
    assert payload["artifact"] == str(artifact.resolve())
    assert payload["checkpoint_run_dir"] == str(log_dir.resolve())


def test_post_training_binding_rejects_matching_artifact_parent_decoy(tmp_path: Path) -> None:
    cell = core.cells_for_stage("P")[0]
    log_dir = tmp_path / "hydra-log"
    artifact_parent = tmp_path / "artifact-parent"
    artifact = artifact_parent / "b1_v2_P_f0_s42_t4_task_plus_y_20260813_120000"
    chain = _write_launch(
        tmp_path, cell=cell, log_dir=log_dir, artifact_parent=artifact_parent,
        prefix="b1_v2_P_f0_s42_t4_task_plus_y",
    )
    _complete_launch(chain)
    artifact.mkdir(parents=True)
    for name in ("resolved_config.yaml", "source_manifest.json", "split_manifest.json", "teacher_metadata.json"):
        (artifact / name).write_text("x", encoding="utf-8")
    epoch_dir = log_dir / "checkpoints" / "epoch_ckpts"
    epoch_dir.mkdir(parents=True)
    for epoch in core.EPOCH_WINDOW:
        (epoch_dir / f"epoch_{epoch - 1:03d}.ckpt").write_bytes(f"epoch{epoch}".encode())
    # A rerun/decoy with the same committed prefix makes the artifact source
    # ambiguous and must fail before an immutable score can bind it.
    (artifact_parent / "b1_v2_P_f0_s42_t4_task_plus_y_decoy").mkdir()
    with pytest.raises(core.B1ContractError, match="exactly one matching"):
        core.write_post_training_binding(
            chain["binding"], future_launch_receipt=chain["launch"],
            artifact=artifact, checkpoint_run_dir=log_dir,
            interpreter=Path("/usr/bin/python3"), script=Path("/independent/src/train.py"),
            working_dir=Path("/independent/streaming_calibration_exp"),
            execution_completion_receipt=chain["completion"],
        )


def test_post_training_binding_rejects_execution_context_substitution(tmp_path: Path) -> None:
    cell = core.cells_for_stage("P")[0]
    log_dir = tmp_path / "hydra-log"
    artifact_parent = tmp_path / "artifact-parent"
    artifact = artifact_parent / "b1_v2_P_f0_s42_t4_task_plus_y_20260813_120000"
    context = dict(
        interpreter=Path("/usr/bin/python3"), script=Path("/independent/src/train.py"),
        working_dir=Path("/independent/streaming_calibration_exp"),
    )
    chain = _write_launch(
        tmp_path, cell=cell, log_dir=log_dir, artifact_parent=artifact_parent,
        prefix="b1_v2_P_f0_s42_t4_task_plus_y", context=context,
    )
    _complete_launch(chain)
    artifact.mkdir(parents=True)
    for name in ("resolved_config.yaml", "source_manifest.json", "split_manifest.json", "teacher_metadata.json"):
        (artifact / name).write_text("x", encoding="utf-8")
    epoch_dir = log_dir / "checkpoints" / "epoch_ckpts"
    epoch_dir.mkdir(parents=True)
    for epoch in core.EPOCH_WINDOW:
        (epoch_dir / f"epoch_{epoch - 1:03d}.ckpt").write_bytes(f"epoch{epoch}".encode())
    with pytest.raises(core.B1ContractError, match="execution context differs"):
        core.write_post_training_binding(
            chain["binding"], future_launch_receipt=chain["launch"], artifact=artifact, checkpoint_run_dir=log_dir,
            interpreter=Path("/usr/bin/python3"), script=Path("/different/src/train.py"),
            working_dir=Path("/independent/streaming_calibration_exp"),
            execution_completion_receipt=chain["completion"],
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("actual_invoked_command", ["/usr/bin/python3", "/independent/src/train.py", "train=false"]),
        ("actual_working_dir", "/different/working-dir"),
        ("actual_cuda_visible_devices", "7"),
        ("cell", "P/f0/s42/z4/task_plus_y"),
    ],
)
def test_execution_chain_rejects_completion_invocation_tamper(
    tmp_path: Path, field: str, replacement: Any,
) -> None:
    cell = core.cells_for_stage("P")[0]
    chain = _write_launch(
        tmp_path, cell=cell, log_dir=tmp_path / "hydra-log",
        artifact_parent=tmp_path / "artifact-parent", prefix="b1_v2_P_f0_s42_t4_task_plus_y",
    )
    _complete_launch(chain)
    completion, _sha = core.load_verified_immutable_json(chain["completion"])
    completion[field] = replacement
    _remint(chain["completion"], completion)
    with pytest.raises(core.B1ContractError, match="completion provenance drift"):
        core.validate_execution_chain(
            future_launch_receipt=chain["launch"], execution_start_receipt=chain["start"],
            execution_completion_receipt=chain["completion"], require_success=True,
        )


def test_execution_chain_rejects_reminted_start_sha_and_legacy_null_completion(tmp_path: Path) -> None:
    cell = core.cells_for_stage("P")[0]
    chain = _write_launch(
        tmp_path, cell=cell, log_dir=tmp_path / "hydra-log",
        artifact_parent=tmp_path / "artifact-parent", prefix="b1_v2_P_f0_s42_t4_task_plus_y",
    )
    _complete_launch(chain)
    start, _sha = core.load_verified_immutable_json(chain["start"])
    start["test_only_remint"] = True
    _remint(chain["start"], start)
    with pytest.raises(core.B1ContractError, match="completion provenance drift"):
        core.validate_execution_chain(
            future_launch_receipt=chain["launch"], execution_start_receipt=chain["start"],
            execution_completion_receipt=chain["completion"], require_success=True,
        )

    second = _write_launch(
        tmp_path / "legacy", cell=cell, log_dir=tmp_path / "legacy-hydra-log",
        artifact_parent=tmp_path / "legacy-artifact-parent", prefix="b1_v2_P_f0_s42_t4_task_plus_y",
    )
    _complete_launch(second)
    launch, _sha = core.load_verified_immutable_json(second["launch"])
    launch["execution_receipts"]["completion"] = None
    _remint(second["launch"], launch)
    with pytest.raises(core.B1ContractError, match="mandatory start/completion/log"):
        core.validate_execution_chain(
            future_launch_receipt=second["launch"], execution_start_receipt=second["start"],
            execution_completion_receipt=second["completion"], require_success=True,
        )


def test_binder_and_scorer_reopen_successful_execution_chain(tmp_path: Path) -> None:
    cell = core.cells_for_stage("P")[0]
    project = Path(__file__).resolve().parents[1]
    context = {
        "interpreter": Path(__import__("sys").executable).resolve(),
        "script": (project / "src/train.py").resolve(),
        "working_dir": project.resolve(),
    }
    log_dir = tmp_path / "hydra-log"
    artifact_parent = tmp_path / "artifact-parent"
    prefix = "b1_v2_P_f0_s42_t4_task_plus_y"
    artifact = artifact_parent / f"{prefix}_20260813"
    chain = _write_launch(
        tmp_path, cell=cell, log_dir=log_dir, artifact_parent=artifact_parent,
        prefix=prefix, context=context,
    )
    _complete_launch(chain)
    artifact.mkdir(parents=True)
    for name in ("resolved_config.yaml", "source_manifest.json", "split_manifest.json", "teacher_metadata.json"):
        (artifact / name).write_text("x", encoding="utf-8")
    epoch_dir = log_dir / "checkpoints" / "epoch_ckpts"
    epoch_dir.mkdir(parents=True)
    for epoch in core.EPOCH_WINDOW:
        (epoch_dir / f"epoch_{epoch - 1:03d}.ckpt").write_bytes(f"epoch{epoch}".encode())
    binding = chain["binding"]
    core.write_post_training_binding(
        binding, future_launch_receipt=chain["launch"], artifact=artifact,
        checkpoint_run_dir=log_dir, execution_completion_receipt=chain["completion"], **context,
    )
    _verify_post_training_binding(
        binding, spec=cell, preflight_sha="a" * 64, artifact=artifact, checkpoint_run_dir=log_dir,
    )
    completion, _sha = core.load_verified_immutable_json(chain["completion"])
    completion["actual_invoked_command"] = [*completion["actual_invoked_command"], "tampered=true"]
    _remint(chain["completion"], completion)
    with pytest.raises(core.B1ContractError):
        _verify_post_training_binding(
            binding, spec=cell, preflight_sha="a" * 64, artifact=artifact, checkpoint_run_dir=log_dir,
        )


def test_runtime_dependency_closure_and_exact_artifact_manifest_fail_closed() -> None:
    root = Path(__file__).resolve().parents[2]
    bindings = core.source_bindings(root)
    core.validate_live_source_bindings(root, bindings)
    runtime = bindings["runtime_dependency_manifest"]
    assert "streaming_calibration_exp/src/data/validation_protocol.py" in runtime
    assert "streaming_calibration_exp/third_party/falcon_challenge/filtering.py" in runtime
    expected_artifact = core.expected_b1_source_manifest_bindings(bindings)
    audit = core.validate_exact_artifact_source_manifest(expected_artifact, bindings)
    assert audit["extra_count"] == 0

    # The generic writer records the whole repository source tree.  A new,
    # unimported experiment or a non-runtime test is evidence, not scientific
    # drift, and must be preserved as an extra rather than reject the cell.
    with_unrelated = {
        **expected_artifact,
        "src/models/a1_unrelated_experiment.py": "1" * 64,
        "tests/test_falcon_sampler.py": "2" * 64,
    }
    audit = core.validate_exact_artifact_source_manifest(with_unrelated, bindings)
    assert audit["extra_count"] == 2
    assert audit["extra_paths"] == [
        "src/models/a1_unrelated_experiment.py", "tests/test_falcon_sampler.py",
    ]

    # These currently existing files are not in B1's recursive import/config
    # closure and therefore cannot invalidate a successor preflight.
    assert "streaming_calibration_exp/src/models/a1_hidden_carrier_module.py" not in runtime
    assert "streaming_calibration_exp/src/data/c2_m2_matched_z4_datamodule.py" not in runtime
    assert "streaming_calibration_exp/tests/test_falcon_sampler.py" not in runtime
    omitted_artifact = dict(expected_artifact)
    omitted_artifact.pop("src/data/validation_protocol.py")
    with pytest.raises(core.B1ContractError, match="relevant artifact source manifest drift"):
        core.validate_exact_artifact_source_manifest(omitted_artifact, bindings)
    modified_artifact = dict(expected_artifact)
    modified_artifact["src/data/validation_protocol.py"] = "0" * 64
    with pytest.raises(core.B1ContractError, match="relevant artifact source manifest drift"):
        core.validate_exact_artifact_source_manifest(modified_artifact, bindings)

    forged = deepcopy(bindings)
    forged["runtime_dependency_manifest"].pop(
        "streaming_calibration_exp/third_party/falcon_challenge/filtering.py"
    )
    forged["runtime_dependency_manifest_sha256"] = core.sha256_payload(
        forged["runtime_dependency_manifest"]
    )
    with pytest.raises(core.B1ContractError):
        core.validate_live_source_bindings(root, forged)

    # A self-consistent but wrong expected digest for a true runtime dependency
    # must still fail against the live closure.
    forged = deepcopy(bindings)
    critical = "streaming_calibration_exp/src/data/validation_protocol.py"
    forged["files"][critical] = "0" * 64
    forged["runtime_dependency_manifest"][critical] = "0" * 64
    forged["artifact_source_manifest_sha256"] = core.sha256_payload({
        path[len("streaming_calibration_exp/"):]: digest
        for path, digest in forged["files"].items()
    })
    forged["runtime_dependency_manifest_sha256"] = core.sha256_payload(
        forged["runtime_dependency_manifest"]
    )
    with pytest.raises(core.B1ContractError, match="recursive runtime/config dependency closure"):
        core.validate_live_source_bindings(root, forged)


def test_real_development_streaming_checkpoint_setup_and_strict_load_cpu_only() -> None:
    """Regression smoke for the scorer's actual checkpoint construction path.

    This uses a non-sealed historical M2 development checkpoint.  It calls no
    data-module setup/NWB reader/forward or trainer, and only proves that the
    scorer creates teacher/student before strict state restoration without
    unexpected keys.
    """
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / (
        "logs/p2_general_carrier_m2_k4/runs/"
        "2026-08-05-23-48-01-226009_rid-None_f0_s42/"
        "checkpoints/best_ckpt/epoch_001.ckpt"
    )
    if not checkpoint.is_file():
        pytest.skip("non-sealed historical M2 development checkpoint unavailable")
    model = _instantiate_and_strict_load_checkpoint(checkpoint, working_dir=root)
    assert model.teacher is not None
    assert model.student is not None


def test_preflight_rejects_tampered_expansion_manifest() -> None:
    root = Path(__file__).resolve().parents[2]
    payload = build_preflight(root)
    payload["stage_f_expansion_manifest"]["predeclared_folds"] = [2, 3, 4]
    with pytest.raises(core.B1ContractError, match="digest drift"):
        core.validate_preflight_payload(payload)
