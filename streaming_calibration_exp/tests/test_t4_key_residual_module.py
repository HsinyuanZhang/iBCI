"""CPU selector/checkpoint contracts for the T4 key-residual wrapper."""
from __future__ import annotations

from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.models.components.spint import SpintModel
from src.models.components.streaming_spint import (
    StreamingSpintModel,
)
from src.models.components.streaming_spint_t4_key_residual_adapter import (
    CoupledT4KeyResidualStreamingSpint,
)
from src.models.streaming_calibration_module import (
    StreamingCalibrationLitModule,
)
from src.models.t4_key_residual_module import (
    T4KeyResidualLitModule,
)


class _IdentityEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.observed_t4: list[torch.Tensor] = []

    def forward_batch(
        self,
        calib_trials,
        side_features=None,
        electrode_ids=None,
    ):
        del electrode_ids
        self.observed_t4.append(side_features)
        return (
            calib_trials.mean(dim=1).permute(0, 2, 1)
            * self.weight
        )


def _substrate() -> StreamingSpintModel:
    torch.manual_seed(83)
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


def _module(
    anchor: Path,
    *,
    mode: str = "aligned",
    seed: int | None = None,
    policy: str = "residual_only",
) -> T4KeyResidualLitModule:
    return T4KeyResidualLitModule(
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
        residual_rank=3,
        residual_permutation_seed=seed,
        residual_training_policy=policy,
    )


def _optimizer_parameter_ids(
    module: T4KeyResidualLitModule,
) -> set[int]:
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


def test_setup_selects_adapter_and_residual_only_optimizer(
    monkeypatch, tmp_path
):
    anchor = tmp_path / "anchor.ckpt"
    anchor.write_bytes(b"selected-t4")
    monkeypatch.setattr(
        StreamingCalibrationLitModule,
        "setup",
        _fake_parent_setup,
    )
    module = _module(anchor)
    module.setup("fit")
    assert isinstance(
        module.student,
        CoupledT4KeyResidualStreamingSpint,
    )
    parameter_ids = _optimizer_parameter_ids(module)
    residual_ids = {
        id(parameter)
        for parameter in (
            module.student.t4_key_residual.parameters()
        )
    }
    assert parameter_ids == residual_ids
    assert all(
        not parameter.requires_grad
        for parameter in module.student.decoder.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in (
            module.student.id_encoder.parameters()
        )
    )


def test_predeclared_optimization_adds_only_attention_out(
    monkeypatch, tmp_path
):
    anchor = tmp_path / "anchor.ckpt"
    anchor.write_bytes(b"selected-t4")
    monkeypatch.setattr(
        StreamingCalibrationLitModule,
        "setup",
        _fake_parent_setup,
    )
    module = _module(
        anchor,
        policy="residual_plus_attention_out",
    )
    module.setup("fit")
    parameter_ids = _optimizer_parameter_ids(module)
    residual_ids = {
        id(parameter)
        for parameter in (
            module.student.t4_key_residual.parameters()
        )
    }
    attention_out_ids = {
        id(parameter)
        for parameter in (
            module.student.decoder.transformer.layers[0]
            .cross_attn.out_proj.parameters()
        )
    }
    assert parameter_ids == (
        residual_ids | attention_out_ids
    )


def test_parent_model_step_keeps_encoder_t4_aligned(
    monkeypatch, tmp_path
):
    anchor = tmp_path / "anchor.ckpt"
    anchor.write_bytes(b"selected-t4")
    monkeypatch.setattr(
        StreamingCalibrationLitModule,
        "setup",
        _fake_parent_setup,
    )
    module = _module(
        anchor, mode="shuffled", seed=17
    )
    module.setup("fit")
    neural = torch.randn(2, 6, 5)
    target = torch.randn(2, 6, 2)
    calib = torch.randn(2, 3, 6, 5)
    t4 = torch.randn(2, 5, 4)
    result = module.model_step(
        (neural, target, calib, ["same", "same"], t4)
    )
    assert (
        module.student.id_encoder.observed_t4[0] is t4
    )
    assert result["behavior_pred"].shape == (2, 1, 2)
    assert torch.isfinite(result["loss"])


def test_checkpoint_receipt_detects_factor_and_anchor_tamper(
    monkeypatch, tmp_path
):
    anchor = tmp_path / "anchor.ckpt"
    anchor.write_bytes(b"selected-t4")
    monkeypatch.setattr(
        StreamingCalibrationLitModule,
        "setup",
        _fake_parent_setup,
    )
    source = _module(anchor)
    source.setup("fit")
    with torch.no_grad():
        source.student.t4_key_residual.output_projection.weight.add_(
            0.1
        )
    checkpoint = {}
    source.on_save_checkpoint(checkpoint)
    state = source.state_dict()

    restored = _module(anchor)
    restored.setup("fit")
    restored.on_load_checkpoint(checkpoint)
    restored.load_state_dict(state, strict=True)
    restored.validate_loaded_key_residual_checkpoint_receipt()

    factor_tampered = _module(anchor)
    factor_tampered.setup("fit")
    factor_tampered.on_load_checkpoint(checkpoint)
    tampered_state = {
        name: tensor.clone()
        for name, tensor in state.items()
    }
    tampered_state[
        "student.t4_key_residual.output_projection.weight"
    ].add_(1.0)
    factor_tampered.load_state_dict(
        tampered_state, strict=True
    )
    with pytest.raises(
        ValueError, match="key-residual hash"
    ):
        (
            factor_tampered
            .validate_loaded_key_residual_checkpoint_receipt()
        )

    anchor.write_bytes(b"tampered-anchor")
    anchor_tampered = _module(anchor)
    anchor_tampered.setup("fit")
    anchor_tampered.on_load_checkpoint(checkpoint)
    anchor_tampered.load_state_dict(state, strict=True)
    with pytest.raises(
        ValueError, match="anchor hash"
    ):
        (
            anchor_tampered
            .validate_loaded_key_residual_checkpoint_receipt()
        )


def test_guards_and_distributed_setup_fail_closed(
    monkeypatch, tmp_path
):
    anchor = tmp_path / "anchor.ckpt"
    anchor.write_bytes(b"selected-t4")
    common = {
        "task": "mc_maze",
        "teacher_ckpt_path": "/not/opened.ckpt",
        "variant": "B3S",
        "window_size": 6,
        "trial_length": 6,
        "side_dim": 4,
        "encoder_warmstart_path": str(anchor),
        "optimizer": partial(
            torch.optim.Adam, lr=1.0e-4
        ),
        "decoder_mode": "coupled",
    }
    with pytest.raises(
        ValueError, match="permutation seed"
    ):
        T4KeyResidualLitModule(
            **common, residual_mode="shuffled"
        )
    with pytest.raises(
        ValueError, match="selected full T4"
    ):
        T4KeyResidualLitModule(
            **{
                **common,
                "encoder_warmstart_path": None,
            },
            residual_mode="aligned",
        )
    with pytest.raises(
        ValueError, match="freeze_decoder"
    ):
        T4KeyResidualLitModule(
            **{**common, "freeze_decoder": True},
            residual_mode="aligned",
        )
    module_for_shape = _module(anchor)
    with pytest.raises(ValueError, match="shape"):
        module_for_shape.decoder_key_features(
            torch.randn(2, 5, 3)
        )
    monkeypatch.setattr(
        StreamingCalibrationLitModule,
        "setup",
        _fake_parent_setup,
    )
    module = _module(anchor)
    module._trainer = SimpleNamespace(world_size=2)
    with pytest.raises(
        RuntimeError, match="world_size=1"
    ):
        module.setup("fit")
