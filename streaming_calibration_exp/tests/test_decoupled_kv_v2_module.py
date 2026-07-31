"""CPU selector/optimizer contracts for the isolated v2 Lightning wrapper."""
from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from src.models.components.spint import SpintModel
from src.models.components.streaming_spint import StreamingSpintModel
from src.models.components.streaming_spint_v2_adapter import (
    TeacherReadinDecoupledStreamingSpint,
)
from src.models.decoupled_kv_v2_module import (
    TeacherReadinDecoupledLitModule,
)
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


def _module(*, key_mode="e_t4", freeze_decoder=False, seed=None):
    return TeacherReadinDecoupledLitModule(
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
        v2_key_mode=key_mode,
        v2_key_dim=6,
        v2_value_dim=8,
        v2_key_permutation_seed=seed,
    )


def _optimizer_parameter_ids(module) -> set[int]:
    configured = module.configure_optimizers()
    optimizer = configured["optimizer"] if isinstance(configured, dict) else configured
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def test_setup_selects_v2_and_trainable_optimizer_contains_direct_branch(monkeypatch):
    monkeypatch.setattr(
        StreamingCalibrationLitModule, "setup", _fake_parent_setup
    )
    module = _module(freeze_decoder=False)
    module.setup("fit")
    assert isinstance(
        module.student, TeacherReadinDecoupledStreamingSpint
    )
    parameter_ids = _optimizer_parameter_ids(module)
    assert id(module.student.decoupled_v2.direct_key_proj.weight) in parameter_ids
    assert id(module.student.decoupled_v2.query_proj.weight) in parameter_ids
    assert id(module.student.id_encoder.weight) in parameter_ids
    assert all(
        id(parameter) not in parameter_ids
        for parameter in module.student.decoder.transformer.parameters()
    )
    module.train()
    assert module.student.decoder.transformer.training is False
    receipt = module.student.v2_initialization_receipt
    assert receipt["legacy_transformer_active_in_v2"] is False
    assert receipt["legacy_transformer_trainable"] is False
    assert receipt["active_factor_sha256"] == receipt["initial_factor_sha256"]
    assert module.hparams["v2_key_mode"] == "e_t4"
    assert module.hparams["v2_key_dim"] == 6
    assert module.hparams["v2_value_dim"] == 8


def test_frozen_decoder_optimizer_contains_encoder_only(monkeypatch):
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
        for parameter in module.student.decoupled_v2.parameters()
    )


def test_key_feature_modes_are_causal_and_deterministic():
    t4 = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4)
    aligned = _module(key_mode="e_t4").decoder_key_features(t4)
    assert aligned is t4
    shuffled_module = _module(key_mode="e_ts4", seed=17)
    shuffled = shuffled_module.decoder_key_features(t4)
    order = np.random.RandomState(17).permutation(5)
    assert torch.equal(shuffled, t4[:, order])
    assert torch.equal(
        shuffled_module.decoder_key_features(t4), shuffled
    )
    assert _module(key_mode="e_only").decoder_key_features(t4) is None
    assert _module(key_mode="x_only").decoder_key_features(t4) is None


def test_parent_model_step_keeps_encoder_t4_aligned_and_shuffles_only_direct_key(
    monkeypatch,
):
    monkeypatch.setattr(
        StreamingCalibrationLitModule, "setup", _fake_parent_setup
    )
    module = _module(key_mode="e_ts4", seed=17)
    module.setup("fit")
    module.train()
    aligned_seen = []
    direct_seen = []
    original_encoder = module.student.id_encoder.forward_batch
    original_key = module.student.decoupled_v2.derive_static_key

    def capture_encoder(calib_trials, side_features=None, electrode_ids=None):
        aligned_seen.append(side_features.detach().clone())
        return original_encoder(
            calib_trials,
            side_features=side_features,
            electrode_ids=electrode_ids,
        )

    def capture_key(hidden_identity, direct_features):
        direct_seen.append(direct_features.detach().clone())
        return original_key(hidden_identity, direct_features)

    monkeypatch.setattr(
        module.student.id_encoder, "forward_batch", capture_encoder
    )
    monkeypatch.setattr(
        module.student.decoupled_v2, "derive_static_key", capture_key
    )
    neural = torch.randn(2, 6, 5)
    target = torch.randn(2, 6, 2)
    calib = torch.randn(2, 3, 6, 5)
    t4 = torch.randn(2, 5, 4)
    result = module.model_step(
        (neural, target, calib, ["same", "same"], t4)
    )
    order = np.random.RandomState(17).permutation(5)
    assert torch.equal(aligned_seen[0], t4)
    assert torch.equal(direct_seen[0], t4[:, order])
    assert result["behavior_pred"].shape == (2, 1, 2)
    assert torch.isfinite(result["loss"])


def test_v2_selector_guards_fail_closed():
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
        TeacherReadinDecoupledLitModule(
            **common, v2_key_mode="e_ts4"
        )
    with pytest.raises(ValueError, match="side_dim=4"):
        TeacherReadinDecoupledLitModule(
            **{**common, "side_dim": 0}, v2_key_mode="e_t4"
        )
    with pytest.raises(ValueError, match="fresh common-teacher"):
        TeacherReadinDecoupledLitModule(
            **common,
            v2_key_mode="e_t4",
            encoder_warmstart_path="/tmp/forbidden.ckpt",
        )
    with pytest.raises(ValueError, match="variant='B3S'"):
        TeacherReadinDecoupledLitModule(
            **{**common, "variant": "B3TS"},
            v2_key_mode="e_t4",
        )


def test_checkpoint_receipt_tracks_restored_active_not_fresh_svd_factors(
    monkeypatch,
):
    monkeypatch.setattr(
        StreamingCalibrationLitModule, "setup", _fake_parent_setup
    )
    source = _module(freeze_decoder=False)
    source.setup("fit")
    with torch.no_grad():
        source.student.decoupled_v2.direct_key_proj.weight.add_(0.25)
    source_receipt = source.active_v2_checkpoint_receipt()
    assert (
        source_receipt["active_factor_sha256"]
        != source_receipt["initial_factor_sha256"]
    )
    checkpoint = {}
    source.on_save_checkpoint(checkpoint)
    state = source.state_dict()

    restored = _module(freeze_decoder=False)
    restored.setup("fit")
    restored.on_load_checkpoint(checkpoint)
    restored.load_state_dict(state, strict=True)
    restored.validate_loaded_v2_checkpoint_receipt()
    restored_receipt = restored.active_v2_checkpoint_receipt()
    assert (
        restored_receipt["active_factor_sha256"]
        == source_receipt["active_factor_sha256"]
    )
    assert restored._pending_v2_checkpoint_receipt is None

    mismatched = _module(key_mode="e_only")
    with pytest.raises(ValueError, match="v2_key_mode"):
        mismatched.on_load_checkpoint(checkpoint)

    tampered = _module(freeze_decoder=False)
    tampered.setup("fit")
    tampered.on_load_checkpoint(checkpoint)
    tampered_state = {
        name: tensor.clone() for name, tensor in state.items()
    }
    tampered_state[
        "student.decoupled_v2.direct_key_proj.weight"
    ].add_(1.0)
    tampered.load_state_dict(tampered_state, strict=True)
    with pytest.raises(ValueError, match="active v2 factor hash"):
        tampered.validate_loaded_v2_checkpoint_receipt()


def test_distributed_setup_fails_closed_before_cpu_svd(monkeypatch):
    monkeypatch.setattr(
        StreamingCalibrationLitModule, "setup", _fake_parent_setup
    )
    module = _module()
    module._trainer = SimpleNamespace(world_size=2)
    with pytest.raises(RuntimeError, match="world_size=1"):
        module.setup("fit")
