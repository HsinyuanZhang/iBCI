"""CPU selector/checkpoint contracts for the exact-head oracle wrapper."""
from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.models.components.spint import SpintModel
from src.models.components.streaming_spint import StreamingSpintModel
from src.models.components.streaming_spint_head_oracle_adapter import (
    TeacherHeadOracleStreamingSpint,
)
from src.models.head_oracle_module import TeacherHeadOracleLitModule
from src.models.streaming_calibration_module import (
    StreamingCalibrationLitModule,
)


class _IdentityEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward_batch(
        self, calib_trials, side_features=None, electrode_ids=None
    ):
        del side_features, electrode_ids
        return calib_trials.mean(dim=1).permute(0, 2, 1) * self.weight


def _substrate() -> StreamingSpintModel:
    decoder = SpintModel(
        model_dim=16,
        num_covariates=2,
        window_size=6,
        num_heads=4,
        num_layers=1,
        num_id_layers=1,
        dropout_rate=0.0,
        dynamic_dropout=False,
        tf_drop_rate=0.0,
    )
    decoder.fc_id_in(torch.zeros(1, 1, 1, 6))
    return StreamingSpintModel(
        decoder=decoder,
        id_encoder=_IdentityEncoder(),
        decoder_mode="coupled",
    )


def _fake_parent_setup(self, stage):
    del stage
    if self.student is None:
        self.teacher = _substrate().decoder
        self.student = _substrate()
        if self._freeze_decoder:
            self.student.freeze_decoder()


def _module(*, mode="e_t4", seed=None, freeze_decoder=False):
    return TeacherHeadOracleLitModule(
        task="mc_maze",
        teacher_ckpt_path="/not/opened.ckpt",
        variant="B3S",
        window_size=6,
        trial_length=6,
        id_hidden_dim=8,
        hidden_dim=8,
        freeze_decoder=freeze_decoder,
        loss_mode="task_only",
        identity_mode="calibrated",
        side_dim=4,
        optimizer=partial(torch.optim.Adam, lr=1.0e-4),
        scheduler=None,
        compile=False,
        decoder_mode="coupled",
        oracle_key_mode=mode,
        oracle_key_permutation_seed=seed,
    )


def _optimizer_parameter_ids(module) -> set[int]:
    configured = module.configure_optimizers()
    optimizer = (
        configured["optimizer"]
        if isinstance(configured, dict)
        else configured
    )
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def test_setup_selects_oracle_and_excludes_legacy_transformer(
    monkeypatch,
):
    monkeypatch.setattr(
        StreamingCalibrationLitModule, "setup", _fake_parent_setup
    )
    module = _module()
    module.setup("fit")
    assert isinstance(module.student, TeacherHeadOracleStreamingSpint)
    parameter_ids = _optimizer_parameter_ids(module)
    assert id(module.student.head_oracle.q_proj.weight) in parameter_ids
    assert id(module.student.id_encoder.weight) in parameter_ids
    assert all(
        id(parameter) not in parameter_ids
        for parameter in module.student.decoder.transformer.parameters()
    )
    receipt = module.oracle_initialization_receipt
    assert receipt["teacher_head_count"] == 4
    assert receipt["teacher_headwise_softmax_preserved"] is True


def test_frozen_decoder_optimizer_contains_only_identity_encoder(
    monkeypatch,
):
    monkeypatch.setattr(
        StreamingCalibrationLitModule, "setup", _fake_parent_setup
    )
    module = _module(freeze_decoder=True)
    module.setup("fit")
    parameter_ids = _optimizer_parameter_ids(module)
    assert id(module.student.id_encoder.weight) in parameter_ids
    assert all(
        id(parameter) not in parameter_ids
        for parameter in module.student.decoder.parameters()
    )
    assert all(
        id(parameter) not in parameter_ids
        for parameter in module.student.head_oracle.parameters()
    )


def test_parent_model_step_keeps_encoder_t4_aligned_and_permutes_only_e(
    monkeypatch,
):
    monkeypatch.setattr(
        StreamingCalibrationLitModule, "setup", _fake_parent_setup
    )
    module = _module(mode="e_ts4", seed=17)
    module.setup("fit")
    aligned_seen = []
    key_identity_seen = []
    original_encoder = module.student.id_encoder.forward_batch
    original_key_input = module.student._identity_key_input

    def capture_encoder(calib_trials, side_features=None, electrode_ids=None):
        aligned_seen.append(side_features.detach().clone())
        return original_encoder(
            calib_trials,
            side_features=side_features,
            electrode_ids=electrode_ids,
        )

    def capture_key_input(identity):
        key_identity_seen.append(identity.detach().clone())
        return original_key_input(identity)

    module.student.id_encoder.forward_batch = capture_encoder
    module.student._identity_key_input = capture_key_input
    neural = torch.randn(2, 6, 5)
    target = torch.randn(2, 6, 2)
    calib = torch.randn(2, 3, 6, 5)
    t4 = torch.randn(2, 5, 4)
    result = module.model_step(
        (neural, target, calib, ["same", "same"], t4)
    )
    assert torch.equal(aligned_seen[0], t4)
    # The captured identity is aligned; permutation occurs only after capture.
    assert key_identity_seen[0].shape == (2, 5, 6)
    assert result["behavior_pred"].shape == (2, 1, 2)
    assert torch.isfinite(result["loss"])


def test_checkpoint_receipt_validates_restored_active_hash(monkeypatch):
    monkeypatch.setattr(
        StreamingCalibrationLitModule, "setup", _fake_parent_setup
    )
    source = _module()
    source.setup("fit")
    with torch.no_grad():
        source.student.head_oracle.q_proj.weight.add_(0.1)
    checkpoint = {}
    source.on_save_checkpoint(checkpoint)
    state = source.state_dict()

    restored = _module()
    restored.setup("fit")
    restored.on_load_checkpoint(checkpoint)
    restored.load_state_dict(state, strict=True)
    restored.validate_loaded_oracle_checkpoint_receipt()

    tampered = _module()
    tampered.setup("fit")
    tampered.on_load_checkpoint(checkpoint)
    tampered_state = {
        name: tensor.clone() for name, tensor in state.items()
    }
    tampered_state["student.head_oracle.q_proj.weight"].add_(1.0)
    tampered.load_state_dict(tampered_state, strict=True)
    with pytest.raises(ValueError, match="active head-oracle hash"):
        tampered.validate_loaded_oracle_checkpoint_receipt()

    mismatched = _module(mode="e_ts4", seed=17)
    with pytest.raises(ValueError, match="oracle_key_mode"):
        mismatched.on_load_checkpoint(checkpoint)


def test_selector_guards_and_distributed_setup_fail_closed(monkeypatch):
    common = {
        "task": "mc_maze",
        "teacher_ckpt_path": "/not/opened.ckpt",
        "variant": "B3S",
        "window_size": 6,
        "trial_length": 6,
        "side_dim": 4,
        "optimizer": partial(torch.optim.Adam, lr=1.0e-4),
        "decoder_mode": "coupled",
    }
    with pytest.raises(ValueError, match="permutation seed"):
        TeacherHeadOracleLitModule(
            **common, oracle_key_mode="e_ts4"
        )
    with pytest.raises(ValueError, match="side_dim=4"):
        TeacherHeadOracleLitModule(
            **{**common, "side_dim": 0}, oracle_key_mode="e_t4"
        )
    module_for_shape = _module()
    with pytest.raises(ValueError, match="shape"):
        module_for_shape.decoder_key_features(torch.randn(2, 5, 3))
    monkeypatch.setattr(
        StreamingCalibrationLitModule, "setup", _fake_parent_setup
    )
    module = _module()
    module._trainer = SimpleNamespace(world_size=2)
    with pytest.raises(RuntimeError, match="world_size=1"):
        module.setup("fit")
