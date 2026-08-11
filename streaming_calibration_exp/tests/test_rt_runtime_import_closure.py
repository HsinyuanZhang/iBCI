"""Clean-clone import/config closure checks for the RT Stage-R runtime.

The test deliberately uses no NWB, checkpoint, or ignored result artifact. It
checks the same Hydra composition used by the production preflight, then
constructs only the fit objects reached before ``DataModule.setup``. Callback
configuration is checked as a target/field closure and its class is directly
constructed with temporary paths; resolving ``${paths.output_dir}`` outside a
Hydra job is not a production execution mode.
"""
from __future__ import annotations

import importlib
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"


def _compose_stage_r():
    with initialize_config_dir(config_dir=str(CONFIG_ROOT.resolve()), version_base="1.3"):
        return compose(
            config_name="train",
            overrides=[
                "experiment=rt_clean_nested_loso_b2_stage_r_zero4",
                "data.loso_fold=3",
                "data.outer_loso_fold=3",
                "model.id_hidden_dim=1024",
                "seed=42",
                "test=false",
                "trainer.accelerator=cpu",
                "trainer.devices=1",
            ],
        )


def test_rt_runtime_modules_import_without_data_or_ignored_results():
    modules = (
        "src.data.rt_k4_loader",
        "src.data.rt_sparse_endpoint_loader",
        "src.data.rt_nested_loso_datamodule",
        "src.models.streaming_calibration_module",
        "src.callbacks.rt_nested_selection_receipt",
        "src.rt_clean_nested_loso_eval",
    )
    for name in modules:
        imported = importlib.import_module(name)
        assert imported is not None


def test_stage_r_config_composes_and_instantiates_pre_setup_objects(tmp_path: Path):
    cfg = _compose_stage_r()
    assert cfg.data._target_ == "src.data.rt_nested_loso_datamodule.RtNestedLossoDataModule"
    assert cfg.model._target_ == "src.models.streaming_calibration_module.StreamingCalibrationLitModule"
    assert cfg.data.side_feature_group == "zero4"
    assert cfg.data.calibration_n_trials == 24
    assert cfg.data.query_start_trial == 24
    assert cfg.model.variant == "B2"
    assert cfg.model.id_hidden_dim == 1024

    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)
    assert datamodule._feature_group == "zero4"
    assert datamodule._setup_complete is False
    assert datamodule.outer_target_loaded is False
    assert model._variant == "B2"
    assert model._side_dim == 0
    assert model.teacher is None and model.student is None

    callback_cfg = cfg.callbacks.rt_nested_selection_receipt
    callback_target = str(callback_cfg._target_)
    module_name, class_name = callback_target.rsplit(".", 1)
    callback_class = getattr(importlib.import_module(module_name), class_name)
    callback = callback_class(
        output_path=str(tmp_path / "selection.json"),
        split_manifest_path=str(tmp_path / "split_manifest.json"),
        config_path=str(tmp_path / "config.yaml"),
        run_id="synthetic_rt_stage_r",
        arm="zero4",
        outer_loso_fold=3,
        seed=42,
        monitor="val_heldin/r2_mean",
    )
    assert callback.arm == "zero4"
    assert callback.outer_loso_fold == 3


def test_stage_r_config_declares_all_import_targets_in_clone():
    required = (
        PROJECT_ROOT / "configs/train.yaml",
        PROJECT_ROOT / "configs/data/rt_nested_loso_m24.yaml",
        PROJECT_ROOT / "configs/model/streaming_b2.yaml",
        PROJECT_ROOT / "configs/experiment/rt_clean_nested_loso_b2_stage_r_zero4.yaml",
        PROJECT_ROOT / "src/data/rt_nested_loso_datamodule.py",
        PROJECT_ROOT / "src/models/streaming_calibration_module.py",
        PROJECT_ROOT / "src/callbacks/rt_nested_selection_receipt.py",
    )
    assert all(path.is_file() for path in required)
