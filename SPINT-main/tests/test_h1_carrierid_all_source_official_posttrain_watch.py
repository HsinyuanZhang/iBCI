from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.h1_carrierid_all_source_official_posttrain_watch as watch


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def _execution_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, preexisting: tuple[str, ...] = ()
) -> watch.ExecutionBinding:
    root = (tmp_path / "SPINT-main").resolve()
    art = root / "pilot_artifacts/h1_carrierid_all_source_official_v1"
    logs = art / "logs"
    runs = root / "logs/h1_carrierid_all_source_official/runs"
    scripts = root / "scripts"
    logs.mkdir(parents=True)
    runs.mkdir(parents=True)
    scripts.mkdir(parents=True)
    asset = art / "H1_CARRIERID_ALL_SOURCE_ASSET_PREFLIGHT_v1.json"
    launch = art / "H1_CARRIERID_ALL_SOURCE_LAUNCH_RECEIPT_v1.json"
    execution = art / "H1_CARRIERID_ALL_SOURCE_GPU1_EXECUTION_LAUNCH_v1.json"
    start = logs / "h1_all_source_gpu1_start.txt"
    train_log = logs / "h1_all_source_gpu1.log"
    guard = scripts / "h1_carrierid_all_source_official_gpu1_launch_guard.py"
    guard.write_text("# synthetic guard\n", encoding="utf-8")
    asset.write_text("synthetic asset\n", encoding="utf-8")
    code_sha = {"data": "d" * 64}
    config_sha = "c" * 64
    launch.write_text(json.dumps({
        "candidate": {"config_sha256": config_sha}, "code_sha256": code_sha,
    }), encoding="utf-8")
    for path in (asset, launch):
        path.chmod(0o444)
    for name in preexisting:
        (runs / name).mkdir()

    monkeypatch.setattr(watch, "ROOT", root)
    monkeypatch.setattr(watch, "ART", art)
    monkeypatch.setattr(watch, "ASSET", asset)
    monkeypatch.setattr(watch, "LAUNCH", launch)
    monkeypatch.setattr(watch, "EXECUTION", execution)
    monkeypatch.setattr(watch, "START", start)
    monkeypatch.setattr(watch, "TRAIN_LOG", train_log)
    monkeypatch.setattr(watch, "RUNS", runs)
    nonce = "0123456789abcdef0123456789abcdef"
    start_time_ns = 1_700_000_000_000_000_000
    hc0 = [
        {
            "date": date,
            "checkpoint_sha256": "1" * 64, "config_sha256": "2" * 64,
            "pair_sha256": "3" * 64, "phase2_source_binding_sha256": "4" * 64,
            "phase1_source_manifest_sha256": "5" * 64,
        }
        for date in ("19250113", "19250119")
    ]
    body = {
        "schema": "h1_carrierid_all_public_source_gpu1_execution_launch_v1",
        "status": "PASS_ALL_SOURCE_GPU1_LAUNCH_RESERVED",
        "nonce": nonce, "start_time_ns": start_time_ns,
        "start_time": "2026-08-09T00:00:00+0000",
        "tmux_session": watch.TMUX_SESSION, "guard_session": watch.GUARD_SESSION,
        "workdir": str(root), "expected_command": watch._expected_command(),
        "expected_command_text": watch._expected_command_text(),
        "asset_manifest": {"path": str(asset), "sha256": watch._sha256_file(asset)},
        "prepared_launch_receipt": {
            "path": str(launch), "sha256": watch._sha256_file(launch),
        },
        "prepared_candidate_config_sha256": config_sha,
        "current_code_sha256": code_sha,
        "launch_guard_sha256": watch._sha256_file(guard),
        "preexisting_all_source_run_dirs": list(preexisting),
        "expected_new_run_root": str(runs),
        "h_c0_gpu1_terminal_checkpoints": hc0,
        "scope": {
            "formal_test_labels_opened": 0, "target_optimizer_steps": 0,
            "target_backward_steps": 0, "evalai_submission_authorized": False,
        },
    }
    execution.write_text(json.dumps(body), encoding="utf-8")
    execution.chmod(0o444)
    start.write_text("\n".join((
        f"nonce={nonce}", f"start_time_ns={start_time_ns}",
        f"start_time={body['start_time']}", f"workdir={root}",
        f"tmux_session={watch.TMUX_SESSION}", f"guard_session={watch.GUARD_SESSION}",
        f"command={watch._expected_command_text()}", f"stdout_stderr_log={train_log}",
        f"execution_receipt={execution}",
    )) + "\n", encoding="utf-8")
    start.chmod(0o444)
    train_log.write_text("training completed\n", encoding="utf-8")
    for path, timestamp in ((execution, start_time_ns + 10), (start, start_time_ns + 20),
                            (train_log, start_time_ns + 30)):
        os.utime(path, ns=(timestamp, timestamp))
    return watch._validate_execution_binding()


@pytest.mark.parametrize(
    ("responses", "expected"),
    [
        ([_completed(1)], "absent"),
        ([_completed(0), _completed(0, "0|0\n")], "running"),
        ([_completed(0), _completed(0, "1|0\n")], "exited_zero"),
    ],
)
def test_tmux_state_distinguishes_absent_running_and_clean_dead_pane(
    monkeypatch: pytest.MonkeyPatch, responses: list[SimpleNamespace], expected: str
) -> None:
    queue = list(responses)
    monkeypatch.setattr(watch.subprocess, "run", lambda *_args, **_kwargs: queue.pop(0))
    assert watch._tmux_state() == expected


@pytest.mark.parametrize(
    "responses",
    [
        [_completed(2, stderr="server failure")],
        [_completed(0), _completed(2, stderr="pane failure")],
        [_completed(0), _completed(0, "1|9\n")],
        [_completed(0), _completed(0, "garbage\n")],
    ],
)
def test_tmux_state_fails_closed_on_command_or_pane_failure(
    monkeypatch: pytest.MonkeyPatch, responses: list[SimpleNamespace]
) -> None:
    queue = list(responses)
    monkeypatch.setattr(watch.subprocess, "run", lambda *_args, **_kwargs: queue.pop(0))
    with pytest.raises(RuntimeError):
        watch._tmux_state()


def test_gpu_release_query_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        watch.subprocess, "run", lambda *_args, **_kwargs: _completed(2, stderr="driver unavailable")
    )
    with pytest.raises(RuntimeError, match="nvidia-smi"):
        watch._gpu1_busy()
    monkeypatch.setattr(
        watch.subprocess, "run", lambda *_args, **_kwargs: _completed(0, stdout="not-a-pid\n")
    )
    with pytest.raises(RuntimeError, match="malformed"):
        watch._gpu1_busy()
    monkeypatch.setattr(
        watch.subprocess, "run", lambda *_args, **_kwargs: _completed(0, stdout="1234\n")
    )
    assert watch._gpu1_busy() is True


def test_wait_observes_tmux_exit_then_two_idle_gpu_samples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = _execution_fixture(monkeypatch, tmp_path)
    states = iter(("running", "absent"))
    gpu = iter((True, False, False))
    monkeypatch.setattr(watch, "_tmux_state", lambda: next(states))
    monkeypatch.setattr(watch, "_gpu1_busy", lambda: next(gpu))
    monkeypatch.setattr(watch.time, "sleep", lambda _seconds: None)
    assert watch._wait_for_training_completion(poll_seconds=0) == expected


def test_execution_binding_rejects_start_nonce_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    binding = _execution_fixture(monkeypatch, tmp_path)
    assert binding.nonce == "0123456789abcdef0123456789abcdef"
    watch.START.chmod(0o644)
    text = watch.START.read_text().replace(binding.nonce, "f" * 32, 1)
    watch.START.write_text(text, encoding="utf-8")
    watch.START.chmod(0o444)
    with pytest.raises(RuntimeError, match="nonce"):
        watch._validate_execution_binding()


def _write_allowlist(root: Path) -> None:
    groups = (
        ("sub-HumanPitt-held-in-calib", watch.EXPECTED_HELDIN_SESSIONS),
        ("sub-HumanPitt-held-out-calib", watch.EXPECTED_HELDOUT_SESSIONS),
    )
    for dirname, sessions in groups:
        directory = root / dirname
        directory.mkdir(parents=True)
        for session in sessions:
            (directory / f"{dirname}_{session}.nwb").write_bytes(session.encode())


def test_allowlist_requires_exact_13_plus_14_sessions(tmp_path: Path) -> None:
    _write_allowlist(tmp_path)
    paths, heldin, heldout = watch._allowlist(tmp_path)
    assert len(paths) == 27 and len(heldin) == 13 and len(heldout) == 14
    assert [watch._session_from_path(path) for path in heldin] == list(watch.EXPECTED_HELDIN_SESSIONS)
    missing = heldout[0]
    missing.unlink()
    replacement = missing.parent / "sub-HumanPitt-held-out-calib_ses-19990101T000000.nwb"
    replacement.write_bytes(b"replacement")
    with pytest.raises(RuntimeError, match="exact-session allowlist drift"):
        watch._allowlist(tmp_path)


def _write_run(run_root: Path, name: str, mtime_ns: int) -> Path:
    run = run_root / name
    checkpoint = run / "checkpoints/fixed_epoch50/epoch_049.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(name.encode())
    (run / ".hydra").mkdir()
    (run / ".hydra/config.yaml").write_text("task: h1\n", encoding="utf-8")
    os.utime(checkpoint, ns=(mtime_ns, mtime_ns))
    return checkpoint


def test_checkpoint_is_unique_and_newer_than_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(watch, "RUNS", tmp_path)
    binding = watch.ExecutionBinding("0" * 32, 200, "a" * 64, ("old",))
    # An old run may have a newer mtime; the execution snapshot, not mtime
    # ranking, excludes it from the candidate set.
    _write_run(tmp_path, "old", 500)
    fresh = _write_run(tmp_path, "fresh", 300)
    assert watch._checkpoint(binding) == fresh.resolve()
    _write_run(tmp_path, "second_fresh", 400)
    with pytest.raises(RuntimeError, match="ambiguous post-reservation"):
        watch._checkpoint(binding)


def test_checkpoint_rejects_symlink(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(watch, "RUNS", tmp_path)
    target = _write_run(tmp_path, "real", 100)
    linked_run = tmp_path / "linked"
    (linked_run / "checkpoints/fixed_epoch50").mkdir(parents=True)
    (linked_run / "checkpoints/fixed_epoch50/epoch_049.ckpt").symlink_to(target)
    binding = watch.ExecutionBinding("0" * 32, 200, "a" * 64, ("real",))
    with pytest.raises(RuntimeError, match="symlink"):
        watch._checkpoint(binding)


def test_smoke_contract_is_h1_seven_dimensional() -> None:
    watch._validate_smoke_output(np.zeros((1, 7), dtype=np.float32))
    with pytest.raises(RuntimeError, match=r"\[1,7\]"):
        watch._validate_smoke_output(np.zeros((1, 4), dtype=np.float32))
    bad = np.zeros((1, 7), dtype=np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(RuntimeError, match="finite"):
        watch._validate_smoke_output(bad)


def _resolved_config(root: Path, asset: Path) -> dict:
    return {
        "protocol_id": "h1_carrierid_all_public_source_official_candidate_v1",
        "task_name": "h1_carrierid_all_source_official",
        "seed": 42, "train": True, "test": False, "ckpt_path": None, "logger": False,
        "official_candidate": {
            "asset_manifest_path": str(asset),
            "training_scope": "all_13_public_held_in_calibration_recordings",
            "calibration_support_trials": 4, "formal_test_labels_opened": 0,
            "target_optimizer_steps": 0, "target_backward_steps": 0,
            "evalai_submission_authorized": False,
        },
        "data": {
            "_target_": "src.data.h1_carrierid_all_source_official.H1CarrierIdAllSourceDataModule",
            "task": "h1", "data_dir": str(root / "data/000954"),
            "asset_manifest_path": str(asset), "batch_size": 32, "window_size": 700,
            "calibration_n_trials": 4, "max_trial_length": 1024, "seed": 42,
            "fixed_epochs": 50, "num_workers": 0, "pin_memory": False,
        },
        "model": {
            "_target_": "src.models.h1_carrierid_all_source_official_module.H1CarrierIdAllSourceLitModule",
            "task": "h1", "fixed_seed": 42, "decode_last_timestep_only": True,
            "predict_scaled_behavior": True, "behavior_scaling_factor": 20.0,
            "clean_teacher": True,
            "net": {
                "_target_": "src.models.components.h1_carrierid_spint.H1CarrierIdSpint",
                "carrier_dim": 4, "carrier_trial_length": 1024, "zero_carrier": False,
                "model_dim": 1024, "num_covariates": 7, "window_size": 700,
            },
        },
        "trainer": {
            "accelerator": "gpu", "devices": 1, "min_epochs": 50, "max_epochs": 50,
            "precision": "32-true", "enable_checkpointing": True,
            "limit_val_batches": 0, "num_sanity_val_steps": 0,
        },
        "callbacks": {"fixed_epoch50": {
            "monitor": None, "save_top_k": -1, "save_last": False,
            "every_n_epochs": 50, "auto_insert_metric_name": False,
        }},
    }


def test_resolved_config_rejects_validation_and_output_dimension_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "SPINT-main"
    asset = root / "pilot_artifacts/assets.json"
    (root / "data/000954").mkdir(parents=True)
    asset.parent.mkdir(parents=True)
    asset.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(watch, "ROOT", root)
    monkeypatch.setattr(watch, "ASSET", asset)
    config = _resolved_config(root, asset)
    _data, data_dir = watch._validate_resolved_config(config)
    assert data_dir == (root / "data/000954").resolve()
    config["trainer"]["limit_val_batches"] = 1
    with pytest.raises(RuntimeError, match="limit_val_batches"):
        watch._validate_resolved_config(config)
    config["trainer"]["limit_val_batches"] = 0
    config["model"]["net"]["num_covariates"] = 4
    with pytest.raises(RuntimeError, match="num_covariates"):
        watch._validate_resolved_config(config)


def test_publish_bundle_refuses_overwrite_and_rolls_back_link_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = tmp_path / "stage1"
    second = tmp_path / "stage2"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    final1 = tmp_path / "final1"
    final2 = tmp_path / "final2"
    final2.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="overwrite"):
        watch._publish_bundle(((first, final1), (second, final2)))
    assert not final1.exists() and final2.read_bytes() == b"existing"

    final2.unlink()
    real_link = os.link
    calls = 0

    def fail_second(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic publish failure")
        return real_link(source, target)

    monkeypatch.setattr(watch.os, "link", fail_second)
    with pytest.raises(OSError, match="synthetic"):
        watch._publish_bundle(((first, final1), (second, final2)))
    assert not final1.exists() and not final2.exists()
