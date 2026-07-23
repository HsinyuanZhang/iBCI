"""Tests for B0 export and final-eval config wiring."""
from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from omegaconf import OmegaConf

from src.models.streaming_calibration_module import StreamingCalibrationLitModule


def test_eval_b0_trainer_config_instantiates():
    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
        cfg = compose(config_name="eval_b0.yaml")
    trainer = instantiate(cfg.trainer)
    assert isinstance(trainer, Trainer)


def test_final_eval_has_no_training_optimized_metric():
    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
        cfg = compose(config_name="train.yaml", overrides=["experiment=final_heldout_eval"])
    assert cfg.optimized_metric is None
    assert cfg.train is False
    assert cfg.data.include_heldout_in_test is True


@pytest.mark.parametrize(
    ("experiment", "field", "value"),
    [
        ("final_heldout_eval_b2_d128", "id_hidden_dim", 128),
        ("final_heldout_eval_b3_d64", "hidden_dim", 64),
    ],
)
def test_final_eval_model_shape_matches_locked_checkpoint(experiment: str, field: str, value: int):
    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
        cfg = compose(config_name="train.yaml", overrides=[f"experiment={experiment}"])
    assert cfg.model.variant in {"B2", "B3"}
    assert int(cfg.model[field]) == value


@pytest.mark.parametrize("experiment,variant", [("b15_m2_loso", "B15"), ("b16_m2_loso", "B16")])
def test_mua_architecture_reuse_presets_lock_fair_loso_protocol(experiment: str, variant: str):
    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
        cfg = compose(config_name="train.yaml", overrides=[f"experiment={experiment}"])

    assert cfg.model.variant == variant
    assert cfg.model.hidden_dim == 64
    assert cfg.model.freeze_decoder is True
    assert cfg.model.loss_mode == "task_plus_y_plus_E"
    assert cfg.data.task == "m2"
    assert cfg.data.validation_protocol == "loso"
    assert cfg.data.loso_fold == 0
    assert cfg.data.include_heldout_in_fit is False
    assert cfg.data.include_heldout_in_test is True
    assert cfg.seed == 42
    assert cfg.trainer.max_epochs == 20
    assert cfg.callbacks.early_stopping.patience == 5


@pytest.mark.parametrize(
    ("experiment", "variant", "freeze_base", "tune_fusion"),
    [
        ("b3_m2_loso_internal", "B3", False, False),
        ("b16z_m2_loso_internal", "B16Z", True, False),
        ("b16r1_m2_loso_internal", "B16R1", True, False),
        ("b16r1f_m2_loso_internal", "B16R1F", True, False),
        ("b16r8f_m2_loso_internal", "B16R8F", True, False),
        ("b16r8mf_m2_loso_internal", "B16R8MF", True, False),
        ("b16zf_m2_loso_internal", "B16ZF", True, False),
        ("b16zf_sb_m2_loso_internal", "B16ZF", True, False),
        ("b16zf_b_m2_loso_internal", "B16ZF", True, False),
        ("b16zf_tb_m2_loso_internal", "B16ZF", True, False),
        ("b16zf_pc_m2_loso_internal", "B16ZF", True, False),
        ("b16zf_e03_m2_loso_internal", "B16ZF", True, False),
        ("b16zfs_m2_loso_internal", "B16ZFS", True, False),
        ("b16zfd_m2_loso_internal", "B16ZFD", True, False),
        ("b16zfo_m2_loso_internal", "B16ZFO", True, False),
        ("b16g_m2_loso_internal", "B16G", True, False),
        ("b16zf_fusion_m2_loso_internal", "B16ZF", False, True),
        ("b16zf_diff_fusion_m2_loso_internal", "B16ZF", False, True),
    ],
)
def test_b16_internal_presets_never_expose_external_heldout(
    experiment: str, variant: str, freeze_base: bool, tune_fusion: bool
):
    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    overrides = [f"experiment={experiment}"]
    if variant != "B3":
        overrides.append("model.encoder_warmstart_path=/tmp/b3_encoder_state.pt")
    with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
        cfg = compose(config_name="train.yaml", overrides=overrides)

    assert cfg.model.variant == variant
    assert cfg.data.validation_protocol == "loso"
    assert cfg.data.loso_fold == 1
    assert cfg.data.include_heldout_in_fit is False
    assert cfg.data.include_heldout_in_test is False
    assert cfg.callbacks.early_stopping.patience == 5
    if variant != "B3":
        assert bool(cfg.model.freeze_encoder_base) is freeze_base
        assert bool(cfg.model.tune_encoder_fusion) is tune_fusion
        assert OmegaConf.select(cfg, "model.encoder_warmstart_path") == "/tmp/b3_encoder_state.pt"


def test_b16zf_session_balanced_preset_enables_only_training_sampler_controls():
    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=b16zf_sb_m2_loso_internal",
                "model.encoder_warmstart_path=/tmp/b3_encoder_state.pt",
            ],
        )

    assert cfg.model.variant == "B16ZF"
    assert cfg.data.balance_session_batches is True
    assert cfg.data.reshuffle_train_sampler_each_epoch is True
    assert cfg.data.include_heldout_in_fit is False
    assert cfg.data.include_heldout_in_test is False


def test_b16zf_tempered_balance_preset_is_half_strength_without_epoch_reshuffle():
    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=b16zf_tb_m2_loso_internal",
                "model.encoder_warmstart_path=/tmp/b3_encoder_state.pt",
            ],
        )

    assert float(cfg.data.balance_session_batches) == 0.5
    assert cfg.data.reshuffle_train_sampler_each_epoch is False
    assert cfg.data.include_heldout_in_fit is False
    assert cfg.data.include_heldout_in_test is False


def test_b16zf_e03_preset_changes_only_the_identity_loss_strength():
    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=b16zf_e03_m2_loso_internal",
                "model.encoder_warmstart_path=/tmp/b3_encoder_state.pt",
            ],
        )

    assert float(cfg.model.lambda_E) == pytest.approx(0.03)
    assert float(cfg.model.support_prediction_consistency_weight) == 0.0
    assert float(cfg.data.balance_session_batches) == 0.0
    assert cfg.data.reshuffle_train_sampler_each_epoch is False


def test_loso_on_test_epoch_end_logs_one_finite_session():
    module = StreamingCalibrationLitModule(
        task="m2",
        variant="B3",
        teacher_ckpt_path=str(Path(__file__).resolve().parents[1] / "dummy.ckpt"),
        window_size=50,
        optimizer=None,
    )
    session = "ses-2020-10-19-Run1"
    preds = __import__("torch").tensor([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]])
    targets = __import__("torch").tensor([[0.0, 0.1], [0.1, 0.2], [0.2, 0.3]])
    module.test_heldin_r2[session].update(preds, targets)
    logged = []

    def _log(name, value, **kwargs):
        if hasattr(value, "compute"):
            value = value.compute()
        if hasattr(value, "item"):
            value = value.item()
        logged.append((name, float(value)))

    module.log = _log  # type: ignore[method-assign]
    module.on_test_epoch_end()

    session_logs = [name for name, _ in logged if name.endswith("/r2")]
    assert session_logs == [f"test_heldin_{session}/r2"]
    mean_logs = [value for name, value in logged if name == "test_heldin/r2_mean"]
    assert len(mean_logs) == 1
    assert mean_logs[0] > -1e9
