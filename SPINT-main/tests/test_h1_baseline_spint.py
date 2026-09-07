from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from src.data.h1_baseline_datamodule import (
    H1_BASELINE_HELDIN_SESSIONS,
    H1BaselineDataModule,
    h1_session_date,
)
from src.models.components.spint import SpintModel
from src.models.falcon_module import FalconLitModule
from scripts.h1_baseline_preflight import _validate_resolved_variant_diff


def _make_tree(root: Path) -> None:
    for directory, token in (
        ("sub-HumanPitt-held-in-calib", "held-in-calib"),
        ("sub-HumanPitt-held-in-minival", "held-in-minival"),
    ):
        folder = root / directory
        folder.mkdir(parents=True)
        for session in H1_BASELINE_HELDIN_SESSIONS:
            (folder / f"sub-HumanPitt-{token}_{session}.nwb").write_bytes(b"synthetic")
    heldout = root / "sub-HumanPitt-held-out-calib"
    heldout.mkdir()
    (heldout / "sub-HumanPitt-held-out-calib_ses-19250126T113454.nwb").write_bytes(b"heldout")


def _fake_load(path, task):
    del task
    # Three 700-bin trials make the two-trial contiguous selector observable,
    # while keeping this test independent of real NWB files.
    length = 2100
    neural = np.arange(length * 176, dtype=np.float32).reshape(length, 176)
    covariates = np.ones((length, 7), dtype=np.float32)
    trial_change = np.zeros(length, dtype=bool)
    trial_change[[0, 700, 1400]] = True
    eval_mask = np.ones(length, dtype=bool)
    return neural, covariates, trial_change, eval_mask


def _module(root: Path) -> H1BaselineDataModule:
    return H1BaselineDataModule(
        task="h1",
        data_dir=str(root),
        heldin_session_names=H1_BASELINE_HELDIN_SESSIONS,
        batch_size=2,
        window_size=700,
        calibration_n_trials=2,
        random_calibration=True,
        smooth_calibration=False,
        max_trial_length=1024,
        use_intertrials=True,
        use_calib_intertrials=False,
        trial_feature_type="raw",
        interpolate_trials=True,
        interpolate_trials_kind="cubic",
        num_workers=0,
    )


def test_h1_baseline_composes_named_lr_variants_and_fixed_epoch_contract():
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        paper = compose(config_name="train.yaml", overrides=["experiment=h1_baseline_paper_lr"])
        released = compose(config_name="train.yaml", overrides=["experiment=h1_baseline_released_code_lr"])
    assert paper.data._target_.endswith("h1_baseline_datamodule.H1BaselineDataModule")
    assert paper.data.calibration_n_trials == 2
    assert paper.data.random_calibration is True
    assert paper.data.window_size == released.data.window_size == 700
    assert paper.data.max_trial_length == released.data.max_trial_length == 1024
    assert paper.model.optimizer.lr == 1.0e-5
    assert released.model.optimizer.lr == 5.0e-5
    assert paper.model.optimizer.lr != released.model.optimizer.lr
    assert paper.model.net.model_dim == released.model.net.model_dim == 1024
    assert paper.model.net.num_heads == released.model.net.num_heads == 64
    assert paper.model.net.dynamic_dropout is True
    assert paper.model.clean_teacher is True
    assert paper.model.scheduler_monitor == "val_heldin/r2_mean"
    assert paper.test is False and released.test is False
    assert paper.optimized_metric is None and released.optimized_metric is None
    assert paper.trainer.max_epochs == paper.trainer.min_epochs == 50
    assert released.trainer.max_epochs == released.trainer.min_epochs == 50
    assert paper.trainer.deterministic is False and released.trainer.deterministic is False
    assert paper.callbacks.fixed_epoch50.monitor is None
    assert paper.callbacks.fixed_epoch50.every_n_epochs == 50
    assert "early_stopping" not in paper.callbacks


def test_h1_baseline_enumerates_only_heldin_and_groups_six_dates(monkeypatch, tmp_path):
    _make_tree(tmp_path)
    calls = []

    def load(path, task):
        if "held-out" in str(path):
            raise AssertionError(f"formal held-out path opened: {path}")
        calls.append(Path(path).resolve())
        return _fake_load(path, task)

    monkeypatch.setattr("src.data.h1_baseline_datamodule.load_nwb", load)
    module = _module(tmp_path)
    module.setup("fit")
    assert len(calls) == 26
    assert all("held-out" not in str(path) for path in calls)
    manifest = module.input_manifest()
    assert manifest["formal_heldout_opened"] is False
    assert manifest["discrete_direction_labels_available"] is False
    assert manifest["dense_velocity_covariates_available"] is True
    groups = {h1_session_date(name) for name in H1_BASELINE_HELDIN_SESSIONS}
    assert groups == {
        "19250101", "19250108", "19250113", "19250115", "19250119", "19250120"
    }
    assert all(len([s for s in H1_BASELINE_HELDIN_SESSIONS if h1_session_date(s) == group]) >= 2 for group in groups)
    with pytest.raises(RuntimeError, match="forbids test"):
        module.setup("test")


def test_h1_baseline_random_train_two_contiguous_and_val_first_two(monkeypatch, tmp_path):
    _make_tree(tmp_path)
    monkeypatch.setattr("src.data.h1_baseline_datamodule.load_nwb", _fake_load)
    module = _module(tmp_path)
    module.setup("fit")
    session = H1_BASELINE_HELDIN_SESSIONS[0]
    train_index = next(i for i, (name, _start) in enumerate(module.train_dataset.window_indices) if name == session)
    source = module.train_dataset.calib_trialized_neural_features[session]
    import src.data.falcon_datamodule as original_data

    original = original_data.random.randint
    try:
        original_data.random.randint = lambda low, high: low
        low = module.train_dataset[train_index][2]
        original_data.random.randint = lambda low, high: high
        high = module.train_dataset[train_index][2]
    finally:
        original_data.random.randint = original
    np.testing.assert_allclose(low, source[:2], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(high, source[-2:], rtol=0.0, atol=0.0)
    val_index = next(i for i, (name, _start) in enumerate(module.val_heldin_dataset.window_indices) if name == session)
    np.testing.assert_allclose(module.val_heldin_dataset[val_index][2], source[:2], rtol=0.0, atol=0.0)


def test_h1_baseline_model_is_original_spint_and_heldin_only_metrics():
    net = SpintModel(
        model_dim=32, num_covariates=7, window_size=10, num_heads=4,
        num_layers=1, num_id_layers=3, dynamic_dropout=True,
    )
    module = FalconLitModule(
        task="h1", net=net, decode_last_timestep_only=True,
        predict_scaled_behavior=True, behavior_scaling_factor=20.0,
        optimizer=None, scheduler=None, compile=False,
        clean_teacher=True, scheduler_monitor="val_heldin/r2_mean",
    )
    assert isinstance(module.net, SpintModel)
    assert module.hparams.clean_teacher is True
    assert module.hparams.scheduler_monitor == "val_heldin/r2_mean"
    assert len(module.val_heldout_r2) == 14
    # clean_teacher=True keeps the original model/forward/loss, but suppresses
    # unpopulated held-out metric logging in the epoch hook.
    neural = torch.zeros(2, 10, 176)
    calibration = torch.zeros(2, 2, 1024, 176)
    behavior = torch.zeros(2, 10, 7)
    net.eval()
    with torch.no_grad():
        output = net(neural, calibration)
    assert output.shape == behavior.shape


def test_h1_baseline_resolved_diff_rejects_non_lr_drift():
    shared = {
        "data": {"window_size": 700},
        "model": {"optimizer": {"lr": 1.0e-5}, "net": {"model_dim": 1024}},
        "trainer": {"max_epochs": 50},
        "callbacks": {"fixed_epoch50": {"monitor": None}},
        "seed": 42,
        "protocol_variant": "paper_lr_1e-5",
        "comparison_role": "appendix_paper_reference",
        "task_name": "paper",
        "tags": ["paper"],
    }
    released = {
        **shared,
        "model": {"optimizer": {"lr": 5.0e-5}, "net": {"model_dim": 1024}},
        "protocol_variant": "released_code_lr_5e-5",
        "comparison_role": "implementation_primary",
        "task_name": "released",
        "tags": ["released"],
    }
    evidence = _validate_resolved_variant_diff(shared, released)
    assert evidence["unexpected_differences"] == []
    released["model"]["net"]["model_dim"] = 512
    with pytest.raises(ValueError, match="not LR-only"):
        _validate_resolved_variant_diff(shared, released)
