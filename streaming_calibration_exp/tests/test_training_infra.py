"""Gradient and callback instantiation tests."""
from __future__ import annotations

from omegaconf import OmegaConf

import pytest
import torch

from src.models.components.spint import SpintModel
from src.models.components.streaming_encoders import EarlyPoolEncoder
from src.models.components.streaming_spint import StreamingSpintModel
from src.models.streaming_calibration_module import (
    StreamingCalibrationLitModule,
    split_ssc_t4_support_views,
)
from src.utils.instantiators import _collect_target_nodes, instantiate_callbacks


def _init_decoder(decoder: SpintModel) -> None:
    calib = torch.randn(1, 2, 100, 96)
    neural = torch.randn(1, 50, 96)
    decoder.eval()
    with torch.no_grad():
        decoder(neural, calib)


def test_task_only_backward_updates_encoder_not_decoder():
    decoder = SpintModel(
        model_dim=64,
        num_covariates=2,
        window_size=50,
        num_heads=4,
        num_layers=1,
        num_id_layers=3,
    )
    _init_decoder(decoder)
    encoder = EarlyPoolEncoder(100, 50, 32)
    model = StreamingSpintModel(decoder=decoder, id_encoder=encoder)
    model.freeze_decoder()
    model.train()

    neural = torch.randn(1, 50, 96)
    calib = torch.randn(1, 2, 100, 96)
    behavior, _ = model(neural, calib)
    loss = behavior.pow(2).mean()
    loss.backward()

    encoder_grads = [p.grad for p in encoder.parameters() if p.grad is not None]
    assert encoder_grads, "encoder should receive task gradients through frozen decoder"
    assert any(g.abs().sum() > 0 for g in encoder_grads)
    assert all(p.grad is None for p in decoder.parameters())


def test_collect_target_nodes_ignores_polluted_root():
    cfg = OmegaConf.create(
        {
            "_target_": "lightning.pytorch.callbacks.RichModelSummary",
            "best_checkpoint": {"_target_": "lightning.pytorch.callbacks.ModelCheckpoint"},
            "early_stopping": {"_target_": "lightning.pytorch.callbacks.EarlyStopping"},
        }
    )
    nodes = []
    _collect_target_nodes(cfg, nodes)
    targets = {node._target_ for node in nodes}
    assert "lightning.pytorch.callbacks.ModelCheckpoint" in targets
    assert "lightning.pytorch.callbacks.EarlyStopping" in targets
    assert "lightning.pytorch.callbacks.RichModelSummary" not in targets


def test_train_callbacks_compose_to_six_targets():
    from hydra import compose, initialize_config_dir
    from pathlib import Path

    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
        cfg = compose(config_name="train.yaml", overrides=["experiment=b2_d128"])
    nodes = []
    _collect_target_nodes(cfg.callbacks, nodes)
    assert len(nodes) == 6
    targets = {node._target_ for node in nodes}
    assert "lightning.pytorch.callbacks.ModelCheckpoint" in targets
    assert "lightning.pytorch.callbacks.EarlyStopping" in targets
    assert "lightning.pytorch.callbacks.LearningRateMonitor" in targets
    assert "src.callbacks.override_epoch_step.OverrideEpochStepCallback" in targets


def test_drop_early_stopping_is_noop_when_disabled():
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary

    from src.train import _drop_early_stopping

    callbacks = [ModelCheckpoint(), EarlyStopping(monitor="x"), RichModelSummary()]
    result = _drop_early_stopping(callbacks, enabled=False)
    assert result is callbacks
    assert len(result) == 3


def test_drop_early_stopping_removes_early_stopping_when_enabled():
    # M2 fixed-epoch-budget mode (sua_exploration/docs/CURRENT_RESULTS.md section H.2):
    # --no_early_stopping / no_early_stopping=true must drop EarlyStopping so every
    # variant trains exactly trainer.max_epochs, without touching other callbacks.
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary

    from src.train import _drop_early_stopping

    checkpoint_cb = ModelCheckpoint()
    summary_cb = RichModelSummary()
    callbacks = [checkpoint_cb, EarlyStopping(monitor="x"), summary_cb]
    result = _drop_early_stopping(callbacks, enabled=True)
    assert result == [checkpoint_cb, summary_cb]
    assert not any(isinstance(cb, EarlyStopping) for cb in result)


def test_no_early_stopping_defaults_false_in_train_yaml():
    from hydra import compose, initialize_config_dir
    from pathlib import Path

    cfg_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
        cfg = compose(config_name="train.yaml", overrides=["experiment=b2_d128"])
    assert cfg.get("no_early_stopping") is False


def test_support_prediction_consistency_weight_is_nonnegative():
    with pytest.raises(ValueError, match="must be >= 0"):
        StreamingCalibrationLitModule(
            task="m2",
            variant="B16ZF",
            teacher_ckpt_path="unused.ckpt",
            window_size=50,
            optimizer=None,
            support_prediction_consistency_weight=-0.1,
        )


def test_training_step_adds_two_support_prediction_consistency():
    class _SupportSensitiveStudent(torch.nn.Module):
        def forward(self, neural, calib_trials):
            value = calib_trials.mean(dim=(1, 2, 3)).view(-1, 1, 1)
            prediction = value.expand(-1, neural.shape[1], 2)
            identity = torch.zeros(
                neural.shape[0], neural.shape[-1], neural.shape[1]
            )
            return prediction, identity

    module = StreamingCalibrationLitModule(
        task="m2",
        variant="B16ZF",
        teacher_ckpt_path="unused.ckpt",
        window_size=50,
        optimizer=None,
        support_prediction_consistency_weight=1.0,
    )
    module.student = _SupportSensitiveStudent()
    module.log = lambda *args, **kwargs: None
    primary_loss = torch.tensor(1.0, requires_grad=True)
    module.model_step = lambda batch: {
        "loss": primary_loss,
        "behavior_pred": torch.zeros(2, 1, 2),
        "session_name": ["same-session", "same-session"],
    }
    neural = torch.zeros(2, 50, 3)
    behavior = torch.zeros(2, 50, 2)
    calib = torch.stack([torch.zeros(2, 4, 3), torch.ones(2, 4, 3)])

    loss = module.training_step(
        (neural, behavior, calib, ["same-session", "same-session"]), 0
    )

    assert loss > primary_loss


def test_ssc_t4_views_are_disjoint_even_odd_m12_subsets():
    calib = torch.arange(2 * 24 * 3 * 4, dtype=torch.float32).reshape(2, 24, 3, 4)
    even, odd = split_ssc_t4_support_views(calib)
    assert even.shape == odd.shape == (2, 12, 3, 4)
    assert torch.equal(even, calib[:, 0::2])
    assert torch.equal(odd, calib[:, 1::2])
    with pytest.raises(ValueError, match="M24"):
        split_ssc_t4_support_views(calib[:, :23])


def test_ssc_t4_prediction_anchor_detaches_full_path_and_updates_half_paths():
    class _HalfSupportStudent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, neural, calib_trials, side_features):
            value = self.scale * calib_trials.mean(dim=(1, 2, 3)).view(-1, 1, 1)
            prediction = value.expand(-1, neural.shape[1], 2)
            identity = torch.zeros(neural.shape[0], neural.shape[-1], neural.shape[1])
            return prediction, identity

    module = StreamingCalibrationLitModule(
        task="m2", variant="B3S", teacher_ckpt_path="unused.ckpt", window_size=50,
        optimizer=None, side_dim=4, ssc_t4_prediction_consistency_weight=1.0,
    )
    student = _HalfSupportStudent()
    module.student = student
    module.log = lambda *args, **kwargs: None
    full_parameter = torch.nn.Parameter(torch.tensor(1.0))
    primary_loss = full_parameter.square()
    module.model_step = lambda batch: {
        "loss": primary_loss,
        "behavior_pred": full_parameter.expand(2, 1, 2),
        "session_name": ["same-session", "same-session"],
    }
    neural = torch.zeros(2, 50, 3)
    behavior = torch.zeros(2, 50, 2)
    calib = torch.stack([
        torch.arange(24, dtype=torch.float32).view(24, 1, 1).expand(24, 4, 3),
        (10 + torch.arange(24, dtype=torch.float32)).view(24, 1, 1).expand(24, 4, 3),
    ])
    side = torch.zeros(2, 3, 4)
    loss = module.training_step(
        (neural, behavior, calib, ["same-session", "same-session"], side), 0
    )
    loss.backward()
    # The full-M24 prediction is explicitly stop-gradient: it receives only
    # the ordinary full-path primary loss d(x^2)/dx=2 at x=1.
    assert torch.allclose(full_parameter.grad, torch.tensor(2.0))
    # Both M12 activity views use the shared student, so the SSC loss carries
    # a real optimization signal through the half-support paths.
    assert student.scale.grad is not None and student.scale.grad.abs() > 0
