from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

import scripts.h1_baseline_eval as evaluator
from src.data.h1_baseline_datamodule import H1_BASELINE_HELDIN_SESSIONS
from src.models.components.spint import SpintModel
from src.models.falcon_module import FalconLitModule


class _TinyHeldinDataset(Dataset):
    def __init__(self):
        self.rows = []
        self.window_indices = []
        for session_index, session in enumerate(H1_BASELINE_HELDIN_SESSIONS):
            for row in range(2):
                self.window_indices.append((session, row))
                neural = np.zeros((10, 176), dtype=np.float32)
                target = np.full((10, 7), session_index + row + 1, dtype=np.float32)
                target[:, 0] += np.arange(10, dtype=np.float32) * 0.1
                calibration = np.zeros((2, 1024, 176), dtype=np.float32)
                self.rows.append((neural, target, calibration, session))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class _TinySessionBatchSampler:
    def __init__(self, dataset):
        self.session_to_indices = {}
        for index, (session, _row) in enumerate(dataset.window_indices):
            self.session_to_indices.setdefault(session, []).append(index)

    def __iter__(self):
        yield from self.session_to_indices.values()

    def __len__(self):
        return len(self.session_to_indices)


class _FakeBaselineDataModule:
    def __init__(self, **kwargs):
        del kwargs
        self.val_heldin_dataset = _TinyHeldinDataset()
        self.val_heldin_batch_sampler = _TinySessionBatchSampler(self.val_heldin_dataset)

    def setup(self, stage):
        assert stage == "fit"

    def input_manifest(self):
        rows = []
        for role in ("heldin_calib", "heldin_minival"):
            for index, session in enumerate(H1_BASELINE_HELDIN_SESSIONS):
                rows.append({"role": role, "session": session, "path": f"/tmp/{role}_{index}.nwb"})
        return {
            "schema": "spint_h1_baseline_input_manifest_v1",
            "formal_heldout_opened": False,
            "files": rows,
        }


def _config(path: Path, variant: str = "released_code_lr_5e-5") -> None:
    lr = "5.0e-5" if variant.startswith("released") else "1.0e-5"
    role = "implementation_primary" if variant.startswith("released") else "appendix_paper_reference"
    path.write_text(
        f"""protocol_id: spint_h1_baseline_reproduction_v1
protocol_variant: {variant}
comparison_role: {role}
seed: 42
test: false
clean_teacher_mode: false
optimized_metric: null
paths:
  output_dir: ${{hydra:runtime.output_dir}}
  work_dir: ${{hydra:runtime.cwd}}
data:
  _target_: src.data.h1_baseline_datamodule.H1BaselineDataModule
  heldin_session_names: [ses-19250101T000000]
  batch_size: 32
  calibration_n_trials: 2
  random_calibration: true
  window_size: 700
  max_trial_length: 1024
  num_workers: 0
model:
  clean_teacher: true
  scheduler_monitor: val_heldin/r2_mean
  optimizer:
    lr: {lr}
trainer:
  max_epochs: 50
  min_epochs: 50
  deterministic: false
callbacks:
  fixed_epoch50:
    dirpath: ${{paths.output_dir}}/checkpoints/fixed_epoch50
    monitor: null
    every_n_epochs: 50
"""
    )


def _checkpoint(path: Path, optimizer_lr: float = 5.0e-5) -> None:
    net = SpintModel(model_dim=16, num_covariates=7, window_size=10, num_heads=4, num_layers=1, num_id_layers=3)
    model = FalconLitModule(
        task="h1", net=net, decode_last_timestep_only=True,
        predict_scaled_behavior=False, behavior_scaling_factor=20.0,
        optimizer=None, scheduler=None, compile=False,
        clean_teacher=True, scheduler_monitor="val_heldin/r2_mean",
    )
    with torch.no_grad():
        model(torch.zeros(1, 10, 176), calib_trialized_neural_features=torch.zeros(1, 2, 1024, 176))
    torch.save(
        {
            "epoch": 49,
            "global_step": 7,
            "pytorch-lightning_version": "2.4.0",
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "optimizer_states": [{"param_groups": [{"lr": optimizer_lr}]}],
        },
        path,
    )


def test_terminal_finalizer_enters_eval_loop_and_refuses_overwrite(monkeypatch, tmp_path):
    config = tmp_path / "resolved_config.yaml"
    checkpoint = tmp_path / "epoch_049.ckpt"
    output = tmp_path / "terminal_receipt.json"
    _config(config)
    _checkpoint(checkpoint)
    monkeypatch.setattr(evaluator, "H1BaselineDataModule", _FakeBaselineDataModule)
    receipt = evaluator.evaluate_terminal_checkpoint(
        data_dir=tmp_path,
        checkpoint_path=checkpoint,
        config_path=config,
        expected_variant="released_code_lr_5e-5",
        output_path=output,
        device="cpu",
    )
    assert receipt["status"] == "PASS_H1_BASELINE_HELDIN_TERMINAL_EVAL"
    assert receipt["epochs_completed"] == 50
    assert receipt["global_step"] == 7
    assert receipt["checkpoint_optimizer_lr"] == pytest.approx(5.0e-5)
    assert receipt["session_count"] == 13
    assert receipt["aggregate"]["pooled_query_windows"] == 26
    assert receipt["day_group_count"] == 6
    assert receipt["aggregate"]["official_day_mean_r2"] == pytest.approx(
        np.mean([row["r2_variance_weighted"] for row in receipt["per_day"].values()])
    )
    assert receipt["guards"]["optimizer_present"] is False
    assert receipt["guards"]["backpropagation"] is False
    assert receipt["guards"]["model_state_unchanged"] is True
    assert output.stat().st_mode & 0o777 == 0o444
    with pytest.raises(FileExistsError, match="overwrite"):
        evaluator.evaluate_terminal_checkpoint(
            data_dir=tmp_path,
            checkpoint_path=checkpoint,
            config_path=config,
            expected_variant="released_code_lr_5e-5",
            output_path=output,
            device="cpu",
        )


def test_terminal_finalizer_rejects_nonterminal_epoch(monkeypatch, tmp_path):
    config = tmp_path / "resolved_config.yaml"
    checkpoint = tmp_path / "epoch_048.ckpt"
    _config(config)
    _checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["epoch"] = 48
    torch.save(payload, checkpoint)
    monkeypatch.setattr(evaluator, "H1BaselineDataModule", _FakeBaselineDataModule)
    with pytest.raises(ValueError, match="epoch=49"):
        evaluator.evaluate_terminal_checkpoint(
            data_dir=tmp_path,
            checkpoint_path=checkpoint,
            config_path=config,
            expected_variant="released_code_lr_5e-5",
            output_path=tmp_path / "bad.json",
            device="cpu",
        )


def test_terminal_finalizer_rejects_checkpoint_lr_arm_mismatch(monkeypatch, tmp_path):
    config = tmp_path / "paper_config.yaml"
    checkpoint = tmp_path / "released_ckpt.ckpt"
    _config(config, variant="paper_lr_1e-5")
    _checkpoint(checkpoint, optimizer_lr=5.0e-5)
    monkeypatch.setattr(evaluator, "H1BaselineDataModule", _FakeBaselineDataModule)
    with pytest.raises(ValueError, match="optimizer LR"):
        evaluator.evaluate_terminal_checkpoint(
            data_dir=tmp_path,
            checkpoint_path=checkpoint,
            config_path=config,
            expected_variant="paper_lr_1e-5",
            output_path=tmp_path / "mismatch.json",
            device="cpu",
        )
