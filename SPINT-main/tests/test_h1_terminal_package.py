from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from falcon_challenge.config import FalconConfig, FalconTask

import scripts.h1_terminal_package as package_gate
from src.data.h1_baseline_datamodule import H1_BASELINE_HELDIN_SESSIONS, H1_BASELINE_HELDOUT_SESSIONS


VARIANT = "paper_lr_1e-5"


def _write_config(path: Path) -> None:
    heldin = ", ".join(H1_BASELINE_HELDIN_SESSIONS)
    path.write_text(
        f"""protocol_id: spint_h1_baseline_reproduction_v1
protocol_variant: {VARIANT}
comparison_role: appendix_paper_reference
seed: 42
test: false
clean_teacher_mode: false
optimized_metric: null
data:
  _target_: src.data.h1_baseline_datamodule.H1BaselineDataModule
  heldin_session_names: [{heldin}]
  batch_size: 32
  calibration_n_trials: 2
  random_calibration: true
  window_size: 700
  max_trial_length: 1024
  num_workers: 0
  use_calib_intertrials: false
  trial_feature_type: raw
  interpolate_trials: true
  interpolate_trials_kind: cubic
  smooth_calibration: false
model:
  clean_teacher: true
  scheduler_monitor: val_heldin/r2_mean
  behavior_scaling_factor: 20.0
  optimizer:
    lr: 1.0e-5
trainer:
  max_epochs: 50
  min_epochs: 50
  deterministic: false
callbacks:
  fixed_epoch50:
    monitor: null
    every_n_epochs: 50
""",
        encoding="utf-8",
    )


def _write_checkpoint(path: Path) -> None:
    torch.save(
        {
            "epoch": 49,
            "global_step": 17,
            "hyper_parameters": {"task": "h1", "clean_teacher": True},
            "optimizer_states": [{"param_groups": [{"lr": 1.0e-5}]}],
        },
        path,
    )


def _write_public_calibration_tree(root: Path) -> None:
    groups = (
        ("sub-HumanPitt-held-in-calib", H1_BASELINE_HELDIN_SESSIONS),
        ("sub-HumanPitt-held-out-calib", H1_BASELINE_HELDOUT_SESSIONS),
    )
    for directory, sessions in groups:
        target = root / directory
        target.mkdir(parents=True)
        for session in sessions:
            (target / f"sub-HumanPitt-{directory.removeprefix('sub-HumanPitt-')}_{session}.nwb").write_bytes(b"fixture")


def _run_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    run = tmp_path / "h1_baseline_staged_fixture" / VARIANT
    config = run / ".hydra" / "config.yaml"
    checkpoint = run / "checkpoints" / "fixed_epoch50" / "epoch_049.ckpt"
    config.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    _write_config(config)
    _write_checkpoint(checkpoint)
    return run, config, checkpoint


def test_terminal_package_uses_exact_public_allowlist_and_writes_immutable_receipt(monkeypatch, tmp_path):
    data_dir = tmp_path / "000954"
    _write_public_calibration_tree(data_dir)
    run, config, checkpoint = _run_paths(tmp_path)
    package = tmp_path / "local_data" / "spint_h1_paper_lr_1e-5.pkl"
    receipt_path = tmp_path / "receipt.json"
    captured: dict[str, object] = {}

    def fake_export(**kwargs):
        captured.update(kwargs)
        task_config = FalconConfig(task=FalconTask.h1)
        features = {
            task_config.hash_dataset(Path(path).stem): np.zeros((2, 1024, 176), dtype=np.uint8)
            for path in kwargs["calibration_files"]
        }
        payload = {
            "decoder": {},
            "task": FalconTask.h1,
            "window_size": 700,
            "behavior_scaling_factor": 20.0,
            "interpolate_trials": True,
            "interpolate_trials_kind": "cubic",
            "calib_trial_features": features,
            "smooth_calibration": False,
        }
        with Path(kwargs["save_path"]).open("wb") as handle:
            pickle.dump(payload, handle)
        return payload

    monkeypatch.setattr(package_gate, "export_spint_decoder", fake_export)
    receipt = package_gate.package_terminal_h1(
        data_dir=data_dir,
        run_dir=run,
        checkpoint_path=checkpoint,
        config_path=config,
        expected_variant=VARIANT,
        package_path=package,
        receipt_path=receipt_path,
    )
    assert receipt["status"] == "PASS_H1_TERMINAL_PACKAGE_ALLOWLISTED"
    assert receipt["checkpoint_epoch_zero_based"] == 49
    assert receipt["epochs_completed"] == 50
    assert receipt["comparison_role"] == "appendix_paper_reference"
    assert receipt["input_manifest"]["heldin_calibration_recordings"] == 13
    assert receipt["input_manifest"]["public_heldout_calibration_recordings"] == 14
    assert receipt["input_manifest"]["minival_included"] is False
    assert len(captured["calibration_files"]) == 27
    assert all("minival" not in str(path).lower() for path in captured["calibration_files"])
    assert receipt["payload_audit"]["feature_count"] == 27
    assert receipt["payload_audit"]["neural_calibration_features_only"] is True
    assert receipt["selection_guards"]["minival_used_for_package_selection"] is False
    assert receipt["selection_guards"]["private_test_opened"] is False
    assert receipt_path.stat().st_mode & 0o777 == 0o444
    with pytest.raises(FileExistsError, match="overwrite"):
        package_gate.package_terminal_h1(
            data_dir=data_dir,
            run_dir=run,
            checkpoint_path=checkpoint,
            config_path=config,
            expected_variant=VARIANT,
            package_path=tmp_path / "second.pkl",
            receipt_path=receipt_path,
        )


def test_terminal_package_rejects_unknown_calibration_nwb_before_export(monkeypatch, tmp_path):
    data_dir = tmp_path / "000954"
    _write_public_calibration_tree(data_dir)
    unknown = data_dir / "sub-HumanPitt-held-out-calib" / "sub-HumanPitt-held-out-calib_ses-19990101T000000.nwb"
    unknown.write_bytes(b"unexpected")
    run, config, checkpoint = _run_paths(tmp_path)
    monkeypatch.setattr(package_gate, "export_spint_decoder", lambda **_kwargs: pytest.fail("exporter must not run"))
    with pytest.raises(ValueError, match="allowlist mismatch"):
        package_gate.package_terminal_h1(
            data_dir=data_dir,
            run_dir=run,
            checkpoint_path=checkpoint,
            config_path=config,
            expected_variant=VARIANT,
            package_path=tmp_path / "bad.pkl",
            receipt_path=tmp_path / "bad.json",
        )
