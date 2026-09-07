"""Selector/checkpoint contracts for the factorized T4 logit residual."""
from __future__ import annotations

from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.models.components.spint import SpintModel
from src.models.components.streaming_spint import StreamingSpintModel
from src.models.components.streaming_spint_t4_logit_residual_adapter import (
    CoupledT4LogitResidualStreamingSpint,
)
from src.models.streaming_calibration_module import StreamingCalibrationLitModule
from src.models.t4_logit_residual_module import T4LogitResidualLitModule


class _IdentityEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.observed_t4: list[torch.Tensor] = []

    def forward_batch(self, calib_trials, side_features=None, electrode_ids=None):
        del electrode_ids
        self.observed_t4.append(side_features)
        return calib_trials.mean(dim=1).permute(0, 2, 1) * self.weight


def _substrate() -> StreamingSpintModel:
    torch.manual_seed(191)
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
    return StreamingSpintModel(decoder=decoder, id_encoder=_IdentityEncoder(), decoder_mode="coupled")


def _fake_parent_setup(self, stage):
    del stage
    if self.student is None:
        self.teacher = _substrate().decoder
        self.student = _substrate()


def _module(
    anchor: Path,
    *,
    mode: str = "aligned",
    interaction: str = "attention_logit",
    seed: int | None = None,
) -> T4LogitResidualLitModule:
    return T4LogitResidualLitModule(
        task="mc_maze",
        teacher_ckpt_path="/not/opened.ckpt",
        variant="B3S",
        window_size=6,
        trial_length=6,
        id_hidden_dim=8,
        hidden_dim=8,
        freeze_decoder=False,
        loss_mode="task_only",
        identity_mode="calibrated",
        side_dim=4,
        encoder_warmstart_path=str(anchor),
        optimizer=partial(torch.optim.Adam, lr=1.0e-4),
        scheduler=None,
        compile=False,
        decoder_mode="coupled",
        residual_mode=mode,
        interaction_mode=interaction,
        residual_rank=3,
        residual_permutation_seed=seed,
    )


def _optimizer_ids(module: T4LogitResidualLitModule) -> set[int]:
    configured = module.configure_optimizers()
    optimizer = configured["optimizer"] if isinstance(configured, dict) else configured
    return {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}


def test_setup_restores_substrate_and_optimizer_only_sees_two_factors(monkeypatch, tmp_path):
    anchor = tmp_path / "anchor.ckpt"
    anchor.write_bytes(b"selected-t4")
    monkeypatch.setattr(StreamingCalibrationLitModule, "setup", _fake_parent_setup)
    module = _module(anchor)
    module.setup("fit")
    assert isinstance(module.student, CoupledT4LogitResidualStreamingSpint)
    factor_ids = {id(parameter) for parameter in module.student.t4_logit_residual.parameters()}
    assert _optimizer_ids(module) == factor_ids
    assert len(factor_ids) == 2
    assert all(not parameter.requires_grad for parameter in module.student.decoder.parameters())
    assert all(not parameter.requires_grad for parameter in module.student.id_encoder.parameters())


def test_parent_step_keeps_encoder_t4_aligned_for_shuffled_control(monkeypatch, tmp_path):
    anchor = tmp_path / "anchor.ckpt"
    anchor.write_bytes(b"selected-t4")
    monkeypatch.setattr(StreamingCalibrationLitModule, "setup", _fake_parent_setup)
    module = _module(anchor, mode="shuffled", seed=17)
    module.setup("fit")
    neural = torch.randn(2, 6, 5)
    target = torch.randn(2, 6, 2)
    calib = torch.randn(2, 3, 6, 5)
    t4 = torch.randn(2, 5, 4)
    result = module.model_step((neural, target, calib, ["same", "same"], t4))
    assert module.student.id_encoder.observed_t4[0] is t4
    assert result["behavior_pred"].shape == (2, 1, 2)
    assert torch.isfinite(result["loss"])


def test_checkpoint_receipt_detects_factor_tamper(monkeypatch, tmp_path):
    anchor = tmp_path / "anchor.ckpt"
    anchor.write_bytes(b"selected-t4")
    monkeypatch.setattr(StreamingCalibrationLitModule, "setup", _fake_parent_setup)
    source = _module(anchor)
    source.setup("fit")
    with torch.no_grad():
        source.student.t4_logit_residual.query_factors.add_(0.1)
    checkpoint = {}
    source.on_save_checkpoint(checkpoint)
    state = source.state_dict()

    restored = _module(anchor)
    restored.setup("fit")
    restored.on_load_checkpoint(checkpoint)
    restored.load_state_dict(state, strict=True)
    restored.validate_loaded_t4_logit_residual_receipt()

    tampered = _module(anchor)
    tampered.setup("fit")
    tampered.on_load_checkpoint(checkpoint)
    bad = {name: tensor.clone() for name, tensor in state.items()}
    bad["student.t4_logit_residual.query_factors"].add_(1.0)
    tampered.load_state_dict(bad, strict=True)
    with pytest.raises(ValueError, match="logit-residual hash"):
        tampered.validate_loaded_t4_logit_residual_receipt()


def test_attention_and_additive_controls_have_identical_parameter_count(monkeypatch, tmp_path):
    anchor = tmp_path / "anchor.ckpt"
    anchor.write_bytes(b"selected-t4")
    monkeypatch.setattr(StreamingCalibrationLitModule, "setup", _fake_parent_setup)
    attention = _module(anchor, interaction="attention_logit")
    additive = _module(anchor, interaction="additive_control")
    attention.setup("fit")
    additive.setup("fit")
    assert sum(p.numel() for p in attention.student.parameters() if p.requires_grad) == 18
    assert sum(p.numel() for p in additive.student.parameters() if p.requires_grad) == 18


def test_guards_and_distributed_setup_fail_closed(monkeypatch, tmp_path):
    anchor = tmp_path / "anchor.ckpt"
    anchor.write_bytes(b"selected-t4")
    with pytest.raises(ValueError, match="permutation seed"):
        _module(anchor, mode="shuffled")
    with pytest.raises(ValueError, match="selected full T4"):
        T4LogitResidualLitModule(
            task="mc_maze",
            teacher_ckpt_path="/not/opened.ckpt",
            variant="B3S",
            window_size=6,
            trial_length=6,
            side_dim=4,
            encoder_warmstart_path=None,
            optimizer=partial(torch.optim.Adam, lr=1.0e-4),
            residual_mode="aligned",
            interaction_mode="attention_logit",
        )
    monkeypatch.setattr(StreamingCalibrationLitModule, "setup", _fake_parent_setup)
    module = _module(anchor)
    module._trainer = SimpleNamespace(world_size=2)
    with pytest.raises(RuntimeError, match="world_size=1"):
        module.setup("fit")
