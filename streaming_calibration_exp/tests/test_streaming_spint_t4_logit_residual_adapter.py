"""CPU contracts for the factorized T4 attention-logit residual."""
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from src.models.components.spint import SpintModel
from src.models.components.streaming_spint import StreamingSpintModel
from src.models.components.streaming_spint_t4_logit_residual_adapter import (
    CoupledT4LogitResidualStreamingSpint,
    T4LogitResidualState,
    ZeroInitializedT4LogitResidual,
)


class _IdentityEncoder(nn.Module):
    def __init__(self, window_size: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.window_size = window_size
        self.observed_side_features: list[torch.Tensor | None] = []

    def forward_batch(self, calib_trials, side_features=None, electrode_ids=None):
        del electrode_ids
        self.observed_side_features.append(side_features)
        return calib_trials.mean(dim=1).permute(0, 2, 1) * self.scale


def _decoder(*, model_dim: int = 16, window_size: int = 6, num_heads: int = 4) -> SpintModel:
    torch.manual_seed(131)
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
    decoder.fc_id_in(torch.zeros(1, 1, 1, window_size))
    return decoder


def _model(
    residual_mode: str = "aligned",
    interaction_mode: str = "attention_logit",
) -> CoupledT4LogitResidualStreamingSpint:
    return CoupledT4LogitResidualStreamingSpint(
        decoder=_decoder(),
        id_encoder=_IdentityEncoder(6),
        residual_mode=residual_mode,
        interaction_mode=interaction_mode,
        residual_rank=3,
        residual_permutation_seed=41 if residual_mode == "shuffled" else None,
    )


@pytest.mark.parametrize("interaction_mode", ["attention_logit", "additive_control"])
def test_zero_init_is_bitwise_equal_to_selected_coupled_path(interaction_mode):
    decoder = _decoder()
    residual = CoupledT4LogitResidualStreamingSpint(
        decoder=copy.deepcopy(decoder),
        id_encoder=_IdentityEncoder(6),
        residual_mode="aligned",
        interaction_mode=interaction_mode,
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
    state = residual.derive_t4_logit_residual_state(t4)
    assert torch.count_nonzero(residual.t4_logit_residual.logit_bias(state.unit_factors)).item() == 0
    expected = coupled.decode_with_identity(neural, identity)
    actual = residual.decode_with_t4_logit_residual_state(neural, identity, state)
    assert torch.equal(actual, expected)


def test_cached_and_on_the_fly_paths_are_bitwise_equal():
    model = _model().eval()
    with torch.no_grad():
        model.t4_logit_residual.query_factors.fill_(0.05)
    neural = torch.randn(3, 6, 5)
    identity = torch.randn(1, 5, 6)
    t4 = torch.randn(1, 5, 4)
    state = model.derive_t4_logit_residual_state(t4)
    direct = model.decode_with_t4_logit_residual(neural, identity, t4)
    cached = model.decode_with_t4_logit_residual_state(neural, identity, state)
    assert torch.equal(direct, cached)
    assert state.nbytes == 5 * 3 * 4


def test_shuffled_control_changes_only_new_residual_attachment():
    aligned = _model("aligned").eval()
    shuffled = _model("shuffled").eval()
    shuffled.load_state_dict(aligned.state_dict(), strict=False)
    with torch.no_grad():
        aligned.t4_logit_residual.query_factors.fill_(0.2)
        shuffled.t4_logit_residual.query_factors.fill_(0.2)
    neural = torch.randn(2, 6, 5)
    calib = torch.randn(2, 3, 6, 5)
    t4 = torch.arange(40, dtype=torch.float32).view(2, 5, 4)
    aligned_prediction, aligned_identity = aligned(neural, calib_trials=calib, side_features=t4)
    shuffled_prediction, shuffled_identity = shuffled(neural, calib_trials=calib, side_features=t4)
    assert torch.equal(aligned_identity, shuffled_identity)
    assert aligned.id_encoder.observed_side_features == [t4]
    assert shuffled.id_encoder.observed_side_features == [t4]
    assert not torch.equal(
        aligned.derive_t4_logit_residual_state(t4).unit_factors,
        shuffled.derive_t4_logit_residual_state(t4).unit_factors,
    )
    assert not torch.equal(aligned_prediction, shuffled_prediction)


def test_joint_unit_permutation_preserves_output():
    model = _model().eval()
    with torch.no_grad():
        model.t4_logit_residual.query_factors.normal_()
    neural = torch.randn(2, 6, 5)
    identity = torch.randn(2, 5, 6)
    t4 = torch.randn(2, 5, 4)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    original = model.decode_with_t4_logit_residual(neural, identity, t4)
    permuted = model.decode_with_t4_logit_residual(
        neural[:, :, permutation], identity[:, permutation], t4[:, permutation]
    )
    assert torch.allclose(original, permuted, atol=1.0e-6, rtol=1.0e-6)


def test_frozen_pilot_trains_only_two_factor_tensors():
    model = _model().train()
    assert model.freeze_backbone_for_residual_pilot() > 0
    model.train()
    assert model.decoder.training is False
    assert model.id_encoder.training is False
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {
        "t4_logit_residual.query_factors",
        "t4_logit_residual.unit_projection.weight",
    }
    neural = torch.randn(2, 6, 5)
    identity = torch.randn(1, 5, 6)
    t4 = torch.randn(1, 5, 4)
    model.decode_with_t4_logit_residual(neural, identity, t4).square().mean().backward()
    assert model.t4_logit_residual.query_factors.grad is not None
    assert torch.count_nonzero(model.t4_logit_residual.query_factors.grad).item() > 0
    assert all(parameter.grad is None for parameter in model.decoder.parameters())
    assert all(parameter.grad is None for parameter in model.id_encoder.parameters())


def test_production_receipt_has_rank_state_not_full_teacher_width_state():
    model = CoupledT4LogitResidualStreamingSpint(
        decoder=_decoder(model_dim=512, window_size=50, num_heads=64),
        id_encoder=_IdentityEncoder(50),
        residual_mode="aligned",
        interaction_mode="attention_logit",
        residual_rank=8,
    )
    cost = model.residual_cost_receipt(batch_size=1, num_units=64)
    assert cost["coupled_reference"]["total"] == 57_970_688
    assert cost["calibration_only_unit_factor_macs"] == 2_048
    assert cost["online_increment"]["query_unit_factor_macs"] == 1_024
    assert cost["persistent_additional_state"]["elements"] == 512
    assert cost["persistent_additional_state"]["bytes_fp32"] == 2_048
    assert cost["persistent_additional_state"]["bytes_int8_without_quantization_metadata"] == 512
    assert cost["trainable_parameter_count"] == 48
    assert cost["neuron_axis_quadratic_term"] is False


def test_guards_fail_closed():
    with pytest.raises(ValueError, match="permutation seed"):
        CoupledT4LogitResidualStreamingSpint(
            decoder=_decoder(),
            id_encoder=_IdentityEncoder(6),
            residual_mode="shuffled",
            interaction_mode="attention_logit",
        )
    with pytest.raises(ValueError, match="unsupported interaction"):
        CoupledT4LogitResidualStreamingSpint(
            decoder=_decoder(),
            id_encoder=_IdentityEncoder(6),
            residual_mode="aligned",
            interaction_mode="bad",
        )
    with pytest.raises(ValueError, match="positive"):
        ZeroInitializedT4LogitResidual(t4_dim=4, rank=0, num_queries=2)
    with pytest.raises(ValueError, match="shape"):
        T4LogitResidualState(torch.zeros(4, 8))
