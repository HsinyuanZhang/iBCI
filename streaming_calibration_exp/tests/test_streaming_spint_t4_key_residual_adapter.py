"""CPU contracts for the baseline-preserving T4 key-residual adapter."""
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from src.models.components.spint import SpintModel
from src.models.components.streaming_spint import StreamingSpintModel
from src.models.components.streaming_spint_t4_key_residual_adapter import (
    CoupledT4KeyResidualStreamingSpint,
    T4KeyResidualState,
    ZeroInitializedT4KeyResidual,
)


class _IdentityEncoder(nn.Module):
    def __init__(self, window_size: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.window_size = window_size
        self.observed_side_features: list[torch.Tensor | None] = []

    def forward_batch(
        self,
        calib_trials,
        side_features=None,
        electrode_ids=None,
    ):
        del electrode_ids
        self.observed_side_features.append(side_features)
        return (
            calib_trials.mean(dim=1).permute(0, 2, 1)
            * self.scale
        )


def _decoder(
    *,
    model_dim: int = 16,
    window_size: int = 6,
    num_heads: int = 4,
) -> SpintModel:
    torch.manual_seed(31)
    decoder = SpintModel(
        model_dim=model_dim,
        num_covariates=2,
        window_size=window_size,
        num_heads=num_heads,
        num_layers=1,
        num_id_layers=1,
        dropout_rate=0.0,
        dynamic_dropout=False,
        tf_drop_rate=0.0,
    )
    decoder.fc_id_in(
        torch.zeros(1, 1, 1, window_size)
    )
    return decoder


def _model(
    mode: str = "aligned",
) -> CoupledT4KeyResidualStreamingSpint:
    return CoupledT4KeyResidualStreamingSpint(
        decoder=_decoder(),
        id_encoder=_IdentityEncoder(6),
        residual_mode=mode,
        residual_rank=3,
        residual_permutation_seed=(
            41 if mode == "shuffled" else None
        ),
    )


def test_zero_initialized_path_is_bitwise_identical_to_coupled_t4():
    decoder = _decoder()
    residual = CoupledT4KeyResidualStreamingSpint(
        decoder=copy.deepcopy(decoder),
        id_encoder=_IdentityEncoder(6),
        residual_mode="aligned",
        residual_rank=3,
    ).eval()
    coupled = StreamingSpintModel(
        decoder=copy.deepcopy(decoder),
        id_encoder=_IdentityEncoder(6),
        decoder_mode="coupled",
    ).eval()
    neural = torch.randn(3, 6, 5)
    identity = torch.randn(1, 5, 6)
    t4 = torch.randn(1, 5, 4)

    state = residual.derive_t4_key_residual_state(t4)
    assert torch.count_nonzero(
        state.hidden_key_residual
    ).item() == 0
    expected = coupled.decode_with_identity(neural, identity)
    actual = residual.decode_with_t4_key_residual_state(
        neural, identity, state
    )
    assert torch.equal(actual, expected)


def test_cached_and_on_the_fly_residual_paths_match():
    model = _model().eval()
    with torch.no_grad():
        model.t4_key_residual.output_projection.weight.fill_(
            0.05
        )
    neural = torch.randn(3, 6, 5)
    identity = torch.randn(1, 5, 6)
    t4 = torch.randn(1, 5, 4)

    state = model.derive_t4_key_residual_state(t4)
    direct = model.decode_with_t4_key_residual(
        neural, identity, t4
    )
    cached = model.decode_with_t4_key_residual_state(
        neural, identity, state
    )
    assert torch.equal(direct, cached)
    assert state.nbytes == 5 * 16 * 4


def test_shuffled_control_changes_only_direct_residual_t4():
    aligned = _model("aligned").eval()
    shuffled = _model("shuffled").eval()
    shuffled.load_state_dict(
        aligned.state_dict(), strict=False
    )
    with torch.no_grad():
        aligned.t4_key_residual.output_projection.weight.fill_(
            0.05
        )
        shuffled.t4_key_residual.output_projection.weight.fill_(
            0.05
        )
    neural = torch.randn(2, 6, 5)
    calib = torch.randn(2, 3, 6, 5)
    t4 = torch.arange(
        40, dtype=torch.float32
    ).view(2, 5, 4)

    aligned_prediction, aligned_identity = aligned(
        neural, calib_trials=calib, side_features=t4
    )
    shuffled_prediction, shuffled_identity = shuffled(
        neural, calib_trials=calib, side_features=t4
    )
    assert torch.equal(aligned_identity, shuffled_identity)
    assert aligned.id_encoder.observed_side_features == [t4]
    assert shuffled.id_encoder.observed_side_features == [t4]
    assert not torch.equal(
        aligned.derive_t4_key_residual_state(
            t4
        ).hidden_key_residual,
        shuffled.derive_t4_key_residual_state(
            t4
        ).hidden_key_residual,
    )
    assert not torch.equal(
        aligned_prediction, shuffled_prediction
    )


def test_frozen_backbone_pilot_trains_only_residual_factors():
    model = _model().train()
    frozen = model.freeze_backbone_for_residual_pilot()
    model.train()
    assert frozen > 0
    assert model.decoder.training is False
    assert model.id_encoder.training is False
    assert all(
        not parameter.requires_grad
        for parameter in model.decoder.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.id_encoder.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.t4_key_residual.parameters()
    )

    neural = torch.randn(2, 6, 5)
    identity = torch.randn(1, 5, 6)
    t4 = torch.randn(1, 5, 4)
    model.decode_with_t4_key_residual(
        neural, identity, t4
    ).square().mean().backward()
    assert (
        model.t4_key_residual.output_projection.weight.grad
        is not None
    )
    assert all(
        parameter.grad is None
        for parameter in model.decoder.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.id_encoder.parameters()
    )


def test_production_shape_receipts_bind_cost_state_and_initialization():
    decoder = _decoder(
        model_dim=512,
        window_size=50,
        num_heads=64,
    )
    model = CoupledT4KeyResidualStreamingSpint(
        decoder=decoder,
        id_encoder=_IdentityEncoder(50),
        residual_mode="aligned",
        residual_rank=8,
    )
    cost = model.residual_cost_receipt(
        batch_size=1, num_units=64
    )
    assert (
        cost["online_linear_attention_ffn_macs_per_window"]
        == 57_970_688
    )
    assert (
        cost["calibration_only_residual_macs"]
        == 264_192
    )
    assert (
        cost["persistent_additional_state"]["bytes_fp32"]
        == 131_072
    )
    assert (
        cost["online_increment"]["additional_linear_macs"]
        == 0
    )
    receipt = model.key_residual_receipt
    assert receipt[
        "teacher_coupled_activity_identity_readin_preserved"
    ] is True
    assert receipt["teacher_value_path_preserved"] is True
    assert receipt["teacher_head_count"] == 64
    assert receipt[
        "output_projection_zero_initialized"
    ] is True


def test_guards_fail_closed():
    with pytest.raises(ValueError, match="permutation seed"):
        CoupledT4KeyResidualStreamingSpint(
            decoder=_decoder(),
            id_encoder=_IdentityEncoder(6),
            residual_mode="shuffled",
        )
    with pytest.raises(ValueError, match="forbids"):
        CoupledT4KeyResidualStreamingSpint(
            decoder=_decoder(),
            id_encoder=_IdentityEncoder(6),
            residual_mode="aligned",
            residual_permutation_seed=2,
        )
    with pytest.raises(ValueError, match="positive"):
        ZeroInitializedT4KeyResidual(
            t4_dim=4, rank=0, hidden_dim=16
        )
    with pytest.raises(ValueError, match="shape"):
        T4KeyResidualState(torch.zeros(4, 16))
    model = _model()
    with pytest.raises(ValueError, match="requires aligned"):
        model(
            torch.randn(1, 6, 5),
            calib_trials=torch.randn(1, 2, 6, 5),
        )
    with pytest.raises(ValueError, match="owns"):
        model(
            torch.randn(1, 6, 5),
            calib_trials=torch.randn(1, 2, 6, 5),
            side_features=torch.randn(1, 5, 4),
            decoder_key_features=torch.randn(1, 5, 4),
        )
