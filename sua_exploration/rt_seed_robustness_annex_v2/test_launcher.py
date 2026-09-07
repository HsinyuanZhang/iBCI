from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest
import yaml

from . import launcher
from . import supervisor


def _source_receipt(
    *, seed: int, fold: int, arm: str, initial: str = "a" * 64
) -> dict[str, object]:
    return {
        "schema": "rt_seed_robustness_annex_v2_source_initial_state_v1",
        "status": "PASS_SOURCE_INITIAL_STATE_RECORDED",
        "arm": arm,
        "fold": fold,
        "seed": seed,
        "initial_state_hash": initial,
        "initial_state_phase": "before_first_optimizer_step",
        **launcher.provenance_bindings(),
        "accounting": {
            "parameter_count": 1234,
            "macs_per_decode_call": 5678,
            "cached_state_bytes": 128,
        },
    }


def _outer_receipt(*, seed: int, fold: int, arm: str) -> dict[str, object]:
    return {
        "schema": "rt_clean_nested_loso_outer_eval_v1",
        "status": "PASS_ONE_SHOT_OUTER_TARGET_NO_BACKPROP",
        "arm": arm,
        "seed": seed,
        "outer_loso_fold": fold,
        "outer_target_session": "ses-RT-20131009",
        "inner_validation_session": "ses-RT-20131010",
        "r2_variance_weighted": 0.2,
        "query_windows_evaluated": 10,
        "target_backpropagation": False,
        "optimizer_present": False,
        "model_state_unchanged": True,
        "model_state_sha256_before": "b" * 64,
        "model_state_sha256_after": "b" * 64,
        "target_query_labels_used_for_calibration": False,
        "target_query_labels_used_for_normalization": False,
        "target_query_labels_used_for_checkpoint_selection": False,
        "target_query_labels_used_for_scoring_only": True,
        "checkpoint_sha256": "c" * 64,
    }


def _selection_and_manifest(tmp_path: Path, *, seed: int, fold: int, arm: str) -> tuple[Path, Path]:
    manifest = tmp_path / "split_manifest.json"
    binding = next(item for item in launcher.spec.FOLD_BINDINGS if int(item["fold"]) == fold)
    scope = {
        field: binding.get(field, {"fold": fold})
        for field in launcher.spec.SOURCE_SPLIT_SCOPE_FIELDS
    }
    scope.update({"outer_loso_fold": fold, "loso_fold": fold, "arm": arm})
    normalizer = {
        "fit_scope": "inner_train_sessions_only",
        "fit_sessions": ["source-a", "source-b"],
        "excluded_inner_validation_session": binding["inner_validation_session"],
        "excluded_outer_target_session": binding["target_session"],
        "mean": [0.0, 0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0, 1.0],
        "feature_group": arm,
    }
    scope["source_sessions"] = ["source-a", "source-b"]
    scope["outer_source_sessions"] = ["source-a", "source-b"]
    scope["inner_train_sessions"] = ["source-a", "source-b"]
    scope["inner_validation_session"] = binding["inner_validation_session"]
    scope["target_session"] = binding["target_session"]
    manifest.write_text(
        json.dumps({**scope, "source_only_normalizer": normalizer}),
        encoding="utf-8",
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema": "rt_clean_nested_loso_selection_receipt_v1",
                "status": "PASS_FIT_INNER_SELECTION_ONLY",
                "arm": arm,
                "seed": seed,
                "outer_loso_fold": fold,
                "selected_epoch": 10,
                "selected_global_step": 100,
                "split_manifest_sha256": launcher.sha256_file(manifest),
            }
        ),
        encoding="utf-8",
    )
    return selection, manifest


def _paired_receipt_for_source(
    tmp_path: Path,
    *,
    seed: int,
    fold: int,
    arm: str,
    source_path: Path,
    source_payload: dict[str, object],
) -> Path:
    other_arm = "afc4_mb4" if arm == "afc4_vel" else "afc4_vel"
    other_path = tmp_path / f"source-{other_arm}.json"
    other_path.write_text(
        json.dumps(_source_receipt(seed=seed, fold=fold, arm=other_arm)),
        encoding="utf-8",
    )
    pair_path = tmp_path / "paired.json"
    launcher.pair_initial_state_receipts(
        cell_seed=seed,
        cell_fold=fold,
        full_receipt=source_path if arm == "afc4_vel" else other_path,
        mb4_receipt=source_path if arm == "afc4_mb4" else other_path,
        output=pair_path,
    )
    return pair_path


def test_shared_api_is_static_and_common_cli_is_not_called() -> None:
    api = launcher.validate_shared_api()
    assert api["runner_cli_used_by_isolated_route"] is False
    # Current historical CLI omits MB4.  The isolated direct-Hydra route must
    # expose this fact in the receipt rather than silently patching the CLI.
    assert api["runner_cli_supports_afc4_mb4"] is False
    assert api["fail_closed_if_required_direct_api_missing"] is True


def test_strict_arm_commands_use_direct_hydra_and_unique_run_ids(tmp_path: Path) -> None:
    full = launcher.Cell(seed=43, fold=0, arm="afc4_vel")
    mb4 = launcher.Cell(seed=43, fold=0, arm="afc4_mb4")
    artifact_root = tmp_path / "fresh-wave"
    full_cmd = launcher.build_train_command(full, artifact_root=artifact_root, accelerator="cpu")
    mb4_cmd = launcher.build_train_command(mb4, artifact_root=artifact_root, accelerator="cpu")
    assert any("afc4_vel" in part for part in full_cmd)
    assert any("afc4_mb4" in part for part in mb4_cmd)
    assert full_cmd != mb4_cmd
    assert any(part.endswith("streaming_calibration_exp/src/train.py") for part in full_cmd)
    assert not any(part.endswith("run_rt_clean_nested_loso.py") for part in full_cmd)
    assert "test=false" in full_cmd
    assert "ckpt_path=null" in full_cmd
    assert "model.encoder_warmstart_path=null" in full_cmd
    assert "trainer.max_epochs=35" in full_cmd
    assert any(part.startswith("hydra.run.dir=") for part in full_cmd)
    assert "hydra.job.chdir=false" in full_cmd
    assert "data.side_feature_group=afc4_mb4" in mb4_cmd
    assert any(part.startswith("+callbacks.annex_initial_state._target_=") for part in full_cmd)
    assert full_cmd[0] == str(launcher.DEFAULT_PYTHON)
    wrapper = launcher.render_execution_wrapper(full_cmd)
    assert f"cd {launcher.STREAMING_ROOT}" in wrapper
    assert "PYTHONNOUSERSITE=1" in wrapper
    assert "PYTHONPATH=" in wrapper


@pytest.mark.parametrize("arm", ["afc4_vel", "afc4_mb4"])
def test_full_and_mb4_hydra_configs_resolve_to_the_frozen_annex_contract(
    tmp_path: Path, arm: str
) -> None:
    """Compose the actual job config without constructing a Trainer or reading NWB."""

    cell = launcher.Cell(seed=43, fold=0, arm=arm)
    command = launcher.build_train_command(
        cell, artifact_root=tmp_path / f"fresh-{arm}", accelerator="cpu"
    )
    command = command[:2] + ["--cfg", "job", "--resolve"] + command[2:]
    completed = subprocess.run(
        command,
        cwd=launcher.STREAMING_ROOT,
        env=launcher.execution_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    resolved = yaml.safe_load(completed.stdout)
    assert resolved["data"]["side_feature_group"] == arm
    assert resolved["data"]["calibration_n_trials"] == 24
    assert resolved["data"]["query_start_trial"] == 24
    assert resolved["data"]["window_size"] == 50
    assert resolved["data"]["loso_fold"] == 0
    assert resolved["data"]["outer_loso_fold"] == 0
    assert resolved["data"]["sampler_seed"] == 43
    assert resolved["seed"] == 43
    assert resolved["trainer"]["max_epochs"] == 35
    assert resolved["train"] is True
    assert resolved["test"] is False
    assert resolved["model"]["freeze_decoder"] is False


def test_v2_3_receipt_keeps_the_exact_twenty_cell_forward_only_plan() -> None:
    receipt = json.loads(
        (launcher.V2_ROOT / "launch_readiness_supplemental_v2_3.json").read_text(
            encoding="utf-8"
        )
    )
    cells = receipt["cells"]
    assert len(cells) == 20
    assert {
        (row["seed"], row["fold"], row["arm"])
        for row in cells
    } == {
        (seed, fold, arm)
        for seed in (43, 44)
        for fold in (0, 3, 6, 9, 12)
        for arm in ("afc4_vel", "afc4_mb4")
    }
    for row in cells:
        command = row["source_fit_command"]
        assert "experiment=rt_clean_nested_loso_m24" in command
        assert "trainer.max_epochs=35" in command
        assert "test=false" in command
        assert "ckpt_path=null" in command
        assert "model.encoder_warmstart_path=null" in command
    target = receipt["target_forward_only_accounting_summary"]
    assert target["target_backpropagation"] is False
    assert target["target_optimizer_present"] is False
    assert target["target_model_state_equal_required"] is True


def test_execution_environment_disables_ambient_user_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONNOUSERSITE", "0")
    env = launcher.execution_environment()
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PYTHONPATH"] == os.pathsep.join(
        [str(launcher.ROOT.resolve()), str(launcher.STREAMING_ROOT.resolve())]
    )


def test_pinned_interpreter_imports_callback_src_and_conda_torch() -> None:
    code = r'''
import json, os, site, torch
from pathlib import Path
from sua_exploration.rt_seed_robustness_annex_v2.runtime import AnnexInitialStateCallback
import src
print(json.dumps({
  "callback": AnnexInitialStateCallback.__name__,
  "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
  "user_site_enabled": bool(site.ENABLE_USER_SITE),
  "user_site": site.getusersitepackages(),
  "torch_file": torch.__file__,
}))
'''
    completed = subprocess.run(
        [str(launcher.DEFAULT_PYTHON), "-c", code],
        cwd=launcher.STREAMING_ROOT,
        env=launcher.execution_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["callback"] == "AnnexInitialStateCallback"
    assert payload["python_no_user_site"] == "1"
    assert payload["user_site_enabled"] is False
    torch_file = Path(payload["torch_file"]).resolve()
    assert launcher.DEFAULT_PYTHON.parent.parent.resolve() in torch_file.parents
    assert Path(payload["user_site"]).resolve() not in torch_file.parents


def test_two_lane_supervisor_is_deterministic_static_plan(tmp_path: Path) -> None:
    artifact_root = tmp_path / "fresh-wave"
    plan = supervisor.build_wave_plan(seed=43, fold=0, artifact_root=artifact_root)
    assert plan["status"] == "DRY_RUN_PREPARED_NOT_AUTHORIZED"
    assert plan["gpu_launched"] is False
    assert [row["arm"] for row in plan["cells"]] == ["afc4_vel", "afc4_mb4"]
    assert plan["failure_policy"]["no_mtime_or_score_path_guessing"] is True
    assert plan["artifact_root"] == str(artifact_root)
    assert plan["gpu_lanes"]["independent"] is True
    assert plan["execution_environment"]["python_no_user_site"] == "1"
    assert [row["cuda_visible_devices"] for row in plan["cells"]] == ["0", "1"]
    for row in plan["cells"]:
        assert row["score_based_path_selection"] is False
        assert row["mtime_or_glob_path_selection"] is False
        assert row["fit_dir"].endswith(f"s43_f0_{row['arm']}/fit")
        assert "hydra.run.dir=" in " ".join(row["source_fit_command"])
        assert "ckpt_path=null" in row["source_fit_command"]
        assert "model.encoder_warmstart_path=null" in row["source_fit_command"]
        assert row["execution_cwd"] == str(launcher.STREAMING_ROOT.resolve())


def test_invalid_arm_fails_closed() -> None:
    with pytest.raises(launcher.LaunchReadinessError):
        launcher.Cell(seed=43, fold=0, arm="afc4_b4")


def test_readiness_receipt_v2_1_is_static_and_non_authorizing(tmp_path: Path) -> None:
    receipt_path = tmp_path / "readiness.json"
    receipt = launcher.build_readiness_receipt_v2_1(output=receipt_path)
    assert receipt["status"] == "PASS_STATIC_E2E_WAVE_READINESS_NOT_AUTHORIZED"
    assert receipt["schema"] == "rt_seed_robustness_annex_v2_1_launch_readiness_v1"
    assert receipt["gpu_authorized"] is False
    assert receipt["gpu_launched"] is False
    assert receipt["training_started"] is False
    assert receipt["nwb_read"] is False
    assert len(receipt["cells"]) == 20
    assert len(receipt["waves"]) == 10
    assert receipt["interpreter"]["path"] == str(launcher.DEFAULT_PYTHON)
    assert receipt["interpreter"]["runtime"]["callback_import"] == "AnnexInitialStateCallback"
    assert receipt["interpreter"]["runtime"]["cuda"]["runtime_probe_performed"] is False
    assert receipt["interpreter"]["runtime"]["cuda"]["allocation_performed"] is False
    assert receipt["execution_environment"]["python_no_user_site"] == "1"
    assert receipt["interpreter"]["python_no_user_site"] == "1"
    assert receipt["interpreter"]["runtime"]["user_site_enabled"] is False
    assert receipt["interpreter"]["runtime"]["torch_within_pinned_prefix"] is True
    assert receipt["interpreter"]["runtime"]["torch_outside_user_site"] is True
    assert "/.local/" not in receipt["interpreter"]["runtime"]["torch_file"]
    assert receipt["shared_api"]["runner_cli_used_by_isolated_route"] is False
    assert receipt["selection"]["selection_used_scores"] is False
    assert receipt["seed42_endpoint_policy"]["seed42_endpoint_rewritten"] is False
    assert receipt["gpu_lane_policy"]["independent_lanes"] is True
    assert receipt_path.is_file()
    assert (receipt_path.stat().st_mode & 0o777) == 0o444


def test_supplemental_readiness_preserves_prior_seal_and_has_top_level_summaries(
    tmp_path: Path,
) -> None:
    prior = launcher.V2_ROOT / "launch_readiness_receipt.json"
    before = prior.read_bytes()
    before_hash = launcher.sha256_file(prior)
    output = tmp_path / "supplemental.json"
    receipt = launcher.build_readiness_supplemental_v2_1(output=output)
    assert prior.read_bytes() == before
    assert launcher.sha256_file(prior) == before_hash
    assert (prior.stat().st_mode & 0o777) == 0o444
    assert receipt["schema"] == (
        "rt_seed_robustness_annex_v2_1_launch_readiness_supplemental_v1"
    )
    assert receipt["status"] == (
        "PASS_STATIC_E2E_WAVE_READINESS_USER_SITE_ISOLATED_SUPPLEMENTAL_NOT_AUTHORIZED"
    )
    assert receipt["append_only_supplement"]["sha256"] == before_hash
    isolation = receipt["interpreter_isolation_summary"]
    assert isolation["exact_python_path"] == str(launcher.DEFAULT_PYTHON)
    assert isolation["python_no_user_site"] == "1"
    assert isolation["user_site_enabled"] is False
    assert isolation["torch_within_pinned_prefix"] is True
    assert isolation["torch_outside_user_site"] is True
    assert "/.local/" not in isolation["torch_file"]
    target = receipt["target_forward_only_accounting_summary"]
    assert target["target_model_state_equal_required"] is True
    assert target["target_optimizer_present"] is False
    assert target["target_backpropagation"] is False
    assert target["full_mb4_accounting_equal_before_target_eval_required"] is True
    paired = receipt["paired_initial_state_summary"]
    assert paired["validated_before_any_outer_target_eval"] is True
    assert paired["initial_state_hash_equal_required"] is True
    assert paired["accounting_equal_required"] == [
        "parameter_count",
        "macs_per_decode_call",
        "cached_state_bytes",
    ]
    assert receipt["gpu_authorized"] is False
    assert receipt["gpu_launched"] is False
    assert receipt["nwb_read"] is False
    assert (output.stat().st_mode & 0o777) == 0o444


def test_pair_initial_state_equality(tmp_path: Path) -> None:
    full = tmp_path / "full.json"
    mb4 = tmp_path / "mb4.json"
    full.write_text(json.dumps(_source_receipt(seed=43, fold=0, arm="afc4_vel")), encoding="utf-8")
    mb4.write_text(json.dumps(_source_receipt(seed=43, fold=0, arm="afc4_mb4")), encoding="utf-8")
    output = tmp_path / "pair.json"
    result = launcher.pair_initial_state_receipts(
        cell_seed=43,
        cell_fold=0,
        full_receipt=full,
        mb4_receipt=mb4,
        output=output,
    )
    assert result["status"] == "PASS_PAIRED_INITIAL_STATE_EQUAL_NOT_AUTHORIZED"
    assert result["initial_state_hash"] == "a" * 64


def test_pair_mismatch_fails_closed(tmp_path: Path) -> None:
    full = tmp_path / "full.json"
    mb4 = tmp_path / "mb4.json"
    full.write_text(json.dumps(_source_receipt(seed=43, fold=0, arm="afc4_vel")), encoding="utf-8")
    mb4.write_text(json.dumps(_source_receipt(seed=43, fold=0, arm="afc4_mb4", initial="d" * 64)), encoding="utf-8")
    with pytest.raises(launcher.LaunchReadinessError, match="initial_state_hash"):
        launcher.pair_initial_state_receipts(
            cell_seed=43,
            cell_fold=0,
            full_receipt=full,
            mb4_receipt=mb4,
            output=tmp_path / "pair.json",
        )


def test_finalize_normalizes_target_state_fields_without_nwb(tmp_path: Path) -> None:
    seed, fold, arm = 43, 0, "afc4_vel"
    source_path = tmp_path / "source.json"
    source_payload = _source_receipt(seed=seed, fold=fold, arm=arm)
    source_path.write_text(json.dumps(source_payload), encoding="utf-8")
    paired = _paired_receipt_for_source(
        tmp_path,
        seed=seed,
        fold=fold,
        arm=arm,
        source_path=source_path,
        source_payload=source_payload,
    )
    outer_path = tmp_path / "outer.json"
    outer_path.write_text(json.dumps(_outer_receipt(seed=seed, fold=fold, arm=arm)), encoding="utf-8")
    selection, manifest = _selection_and_manifest(tmp_path, seed=seed, fold=fold, arm=arm)
    result = launcher.build_cell_receipt(
        launcher.Cell(seed=seed, fold=fold, arm=arm),
        source_initial_receipt=source_path,
        outer_eval_receipt=outer_path,
        selection_receipt=selection,
        split_manifest=manifest,
        paired_initial_receipt=paired,
        output=tmp_path / "cell.json",
    )
    assert result["target_model_state_unchanged"] is True
    assert result["target_model_state_before_sha256"] == "b" * 64
    assert result["target_model_state_after_sha256"] == "b" * 64
    assert result["target_optimizer_present"] is False
    assert result["target_backpropagation"] is False
    assert result["parameter_count"] == 1234
    assert result["target_forward_only_accounting"]["paired_full_mb4_equal_before_target_eval"] is True
    assert result["paired_initial_state_equal_before_target_eval"] is True


def test_finalize_rejects_legacy_target_optimizer_flag(tmp_path: Path) -> None:
    seed, fold, arm = 43, 0, "afc4_vel"
    source_path = tmp_path / "source.json"
    source_payload = _source_receipt(seed=seed, fold=fold, arm=arm)
    source_path.write_text(json.dumps(source_payload), encoding="utf-8")
    paired = _paired_receipt_for_source(
        tmp_path,
        seed=seed,
        fold=fold,
        arm=arm,
        source_path=source_path,
        source_payload=source_payload,
    )
    bad_outer = _outer_receipt(seed=seed, fold=fold, arm=arm)
    bad_outer["optimizer_present"] = True
    outer_path = tmp_path / "outer.json"
    outer_path.write_text(json.dumps(bad_outer), encoding="utf-8")
    selection, manifest = _selection_and_manifest(tmp_path, seed=seed, fold=fold, arm=arm)
    with pytest.raises(launcher.LaunchReadinessError, match="optimizer"):
        launcher.build_cell_receipt(
            launcher.Cell(seed=seed, fold=fold, arm=arm),
            source_initial_receipt=source_path,
            outer_eval_receipt=outer_path,
            selection_receipt=selection,
            split_manifest=manifest,
            paired_initial_receipt=paired,
            output=tmp_path / "cell.json",
        )


def test_non_pinned_interpreter_and_nonfresh_roots_fail_closed(tmp_path: Path) -> None:
    cell = launcher.Cell(seed=43, fold=0, arm="afc4_vel")
    with pytest.raises(launcher.LaunchReadinessError, match="exactly"):
        launcher.build_train_command(
            cell,
            artifact_root=tmp_path / "fresh",
            python_executable="/usr/bin/python3",
        )
    with pytest.raises(launcher.LaunchReadinessError, match="absolute"):
        launcher.build_train_command(cell, artifact_root=Path("relative-artifacts"))
    with pytest.raises(launcher.LaunchReadinessError, match="fresh"):
        launcher.build_train_command(cell, artifact_root=tmp_path)


def test_wave_execution_gate_fails_before_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = supervisor.build_wave_plan(
        seed=43, fold=0, artifact_root=tmp_path / "fresh-wave"
    )
    monkeypatch.delenv(launcher.EXECUTION_ENABLE_ENV, raising=False)
    monkeypatch.delenv(launcher.EXECUTION_AUTHORIZATION_ENV, raising=False)
    with pytest.raises(launcher.LaunchReadinessError, match="execution disabled"):
        supervisor.run_wave(plan, receipt_path=tmp_path / "must-not-exist.json")
    assert not (tmp_path / "fresh-wave").exists()
    assert not (tmp_path / "must-not-exist.json").exists()


def test_authorized_wave_rejects_tampered_command_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "fresh-wave"
    plan = supervisor.build_wave_plan(seed=43, fold=0, artifact_root=artifact_root)
    plan["cells"][0]["source_fit_command"] = ["/bin/true"]
    monkeypatch.setenv(launcher.EXECUTION_ENABLE_ENV, "1")
    monkeypatch.setenv(
        launcher.EXECUTION_AUTHORIZATION_ENV,
        launcher.EXECUTION_AUTHORIZATION_VALUE,
    )
    with pytest.raises(supervisor.SupervisorError, match="command drift"):
        supervisor.run_wave(plan, receipt_path=artifact_root / "wave_receipt.json")
    assert not artifact_root.exists()


def test_direct_single_arm_execution_is_forbidden(tmp_path: Path) -> None:
    with pytest.raises(launcher.LaunchReadinessError, match="paired supervisor"):
        launcher.main(
            [
                "train",
                "--seed",
                "43",
                "--fold",
                "0",
                "--arm",
                "afc4_vel",
                "--artifact-root",
                str(tmp_path / "fresh-wave"),
                "--execute",
            ]
        )
