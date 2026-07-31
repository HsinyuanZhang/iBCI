"""CPU integration contracts for the isolated teacher-readin K/V v2 adapter."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models.components.spint import SpintModel
from src.models.components.streaming_spint_v2_adapter import (
    TeacherReadinDecoupledStreamingSpint,
)


class _IdentityEncoder(nn.Module):
    def __init__(self, window_size: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.window_size = window_size

    def forward_batch(
        self,
        calib_trials,
        side_features=None,
        electrode_ids=None,
    ):
        del side_features, electrode_ids
        # [B,M,T,N] -> [B,N,T], with T == window_size in this fixture.
        return calib_trials.mean(dim=1).permute(0, 2, 1) * self.scale


def _decoder() -> SpintModel:
    torch.manual_seed(9)
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
    # The production decoder is materialized by loading the teacher checkpoint.
    # Materialize this fixture's unrelated LazyLinear ID path as well so the
    # inherited freeze receipt can enumerate every decoder parameter.
    decoder.fc_id_in(torch.zeros(1, 1, 1, 6))
    return decoder


def _model(key_mode: str = "e_t4"):
    return TeacherReadinDecoupledStreamingSpint(
        decoder=_decoder(),
        id_encoder=_IdentityEncoder(window_size=6),
        key_mode=key_mode,
        key_dim=6,
        value_dim=8,
        direct_feature_dim=4,
    )


def test_static_training_and_cached_paths_match_and_use_hidden_readin(monkeypatch):
    model = _model("e_t4")
    model.eval()
    neural = torch.randn(2, 6, 5)
    identity = torch.randn(2, 5, 6)
    direct = torch.randn(2, 5, 4)
    calls = []
    original = model.decoder.fc_in.forward

    def capture(value):
        calls.append(value.detach().clone())
        return original(value)

    monkeypatch.setattr(model.decoder.fc_in, "forward", capture)
    direct_output = model.decode_with_decoupled_identity(
        neural, identity, decoder_key_features=direct
    )
    assert [tuple(value.shape) for value in calls] == [
        (2, 5, 6),  # activity x
        (1, 2, 6),  # learned query rep
        (2, 5, 6),  # identity E
    ]

    calls.clear()
    state = model.derive_decoupled_kv_state(identity, direct)
    assert [tuple(value.shape) for value in calls] == [(2, 5, 6)]
    calls.clear()
    cached_output = model.decode_with_decoupled_kv_state(neural, state)
    assert [tuple(value.shape) for value in calls] == [
        (2, 5, 6),
        (1, 2, 6),
    ]
    assert torch.allclose(direct_output, cached_output, atol=1e-6)


def test_x_only_uses_dynamic_hidden_activity_key_and_has_no_cache(monkeypatch):
    model = _model("x_only")
    model.eval()
    neural = torch.randn(2, 6, 5)
    identity = torch.randn(2, 5, 6)
    key_calls = 0
    direct_calls = 0
    original_key = model.decoupled_v2.key_proj.forward
    original_direct = model.decoupled_v2.direct_key_proj.forward

    def count_key(*args, **kwargs):
        nonlocal key_calls
        key_calls += 1
        return original_key(*args, **kwargs)

    def count_direct(*args, **kwargs):
        nonlocal direct_calls
        direct_calls += 1
        return original_direct(*args, **kwargs)

    monkeypatch.setattr(model.decoupled_v2.key_proj, "forward", count_key)
    monkeypatch.setattr(
        model.decoupled_v2.direct_key_proj, "forward", count_direct
    )
    output = model.decode_with_decoupled_identity(neural, identity)
    assert output.shape == (2, 6, 2)
    assert key_calls == 1
    assert direct_calls == 0
    try:
        model.derive_decoupled_kv_state(identity)
    except RuntimeError as error:
        assert "no static" in str(error)
    else:
        raise AssertionError("x_only unexpectedly created a static state")
    deployment = model.decode_x_only(neural)
    ignored_identity = model.decode_with_decoupled_identity(
        neural, torch.randn(99)
    )
    assert torch.allclose(deployment, ignored_identity, atol=1e-6)
    with pytest.raises(ValueError, match="forbids supplied"):
        model.decode_with_decoupled_identity(
            neural,
            identity,
            decoder_key_features=torch.randn(2, 5, 4),
        )


def test_forward_preserves_encoder_t4_input_and_direct_key_separation():
    model = _model("e_ts4")
    model.eval()
    neural = torch.randn(2, 6, 5)
    calib = torch.randn(2, 3, 6, 5)
    aligned_t4 = torch.randn(2, 5, 4)
    shuffled_direct = aligned_t4.roll(1, dims=1)
    output, identity = model(
        neural,
        calib_trials=calib,
        side_features=aligned_t4,
        decoder_key_features=shuffled_direct,
    )
    expected_identity = model.id_encoder.forward_batch(
        calib, side_features=aligned_t4
    )
    assert torch.equal(identity, expected_identity)
    assert output.shape == (2, 6, 2)


def test_e_only_fails_closed_on_nonzero_or_zero_supplied_direct_features():
    model = _model("e_only")
    model.eval()
    neural = torch.randn(2, 6, 5)
    identity = torch.randn(2, 5, 6)
    output = model.decode_with_decoupled_identity(
        neural, identity, decoder_key_features=None
    )
    assert output.shape == (2, 6, 2)
    with pytest.raises(ValueError, match="e_only forbids"):
        model.decode_with_decoupled_identity(
            neural,
            identity,
            decoder_key_features=torch.zeros(2, 5, 4),
        )


def test_adapter_forward_dispatch_is_explicitly_v2(monkeypatch):
    model = _model("e_t4")
    neural = torch.randn(2, 6, 5)
    calib = torch.randn(2, 3, 6, 5)
    t4 = torch.randn(2, 5, 4)
    calls = 0
    original = model.decode_with_decoupled_identity

    def forbid_coupled(*args, **kwargs):
        raise AssertionError("adapter forwarded through coupled decoder")

    def count_v2(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "decode_with_identity", forbid_coupled)
    monkeypatch.setattr(model, "decode_with_decoupled_identity", count_v2)
    model(
        neural,
        calib_trials=calib,
        side_features=t4,
        decoder_key_features=t4,
    )
    assert calls == 1


def test_cached_decode_never_calls_static_key_projections(monkeypatch):
    model = _model("e_t4")
    model.eval()
    neural = torch.randn(2, 6, 5)
    identity = torch.randn(1, 5, 6)
    direct = torch.randn(1, 5, 4)
    key_calls = 0
    direct_calls = 0
    original_key = model.decoupled_v2.key_proj.forward
    original_direct = model.decoupled_v2.direct_key_proj.forward

    def count_key(*args, **kwargs):
        nonlocal key_calls
        key_calls += 1
        return original_key(*args, **kwargs)

    def count_direct(*args, **kwargs):
        nonlocal direct_calls
        direct_calls += 1
        return original_direct(*args, **kwargs)

    monkeypatch.setattr(model.decoupled_v2.key_proj, "forward", count_key)
    monkeypatch.setattr(
        model.decoupled_v2.direct_key_proj, "forward", count_direct
    )
    state = model.derive_decoupled_kv_state(identity, direct)
    assert (key_calls, direct_calls) == (1, 1)
    model.decode_with_decoupled_kv_state(neural, state)
    assert (key_calls, direct_calls) == (1, 1)


def test_reference_costs_and_initialization_receipt_are_bound_to_adapter():
    model = TeacherReadinDecoupledStreamingSpint(
        decoder=SpintModel(
            model_dim=512,
            num_covariates=2,
            window_size=50,
            num_heads=64,
            num_layers=1,
            num_id_layers=1,
            dropout_rate=0.0,
            dynamic_dropout=False,
            tf_drop_rate=0.0,
        ),
        id_encoder=_IdentityEncoder(window_size=50),
        key_mode="e_t4",
        key_dim=48,
        value_dim=64,
        direct_feature_dim=4,
    )
    receipt = model.v2_cost_receipt(batch_size=1, num_units=64)
    assert receipt["online_macs_per_frame"]["total"] == 25_462_784
    assert receipt["calibration_only_macs"]["total"] == 20_000_768
    assert receipt["persistent_state"]["bytes"] == 12_288
    assert receipt["online_mac_reduction_fraction_vs_coupled"] > 0.56
    assert receipt["persistent_state_nonincreasing_vs_E"] is True
    assert receipt["reference_dtype"] == "float32"
    assert receipt["identity_encoder_compute_included"] is False
    assert (
        receipt["identity_encoder_computed_for_framework_metrics_only"]
        is False
    )
    initialization = model.v2_initialization_receipt
    assert initialization["teacher_head_count"] == 64
    assert initialization["student_softmax_head_count"] == 1
    assert initialization["exact_teacher_mha_equivalence"] is False
    assert initialization["unused_legacy_decoder_parameters_frozen"] is True


def test_gradients_reach_identity_encoder_direct_branch_and_activity():
    model = _model("e_t4")
    neural = torch.randn(2, 6, 5, requires_grad=True)
    calib = torch.randn(2, 3, 6, 5, requires_grad=True)
    t4 = torch.randn(2, 5, 4, requires_grad=True)
    output, _ = model(
        neural,
        calib_trials=calib,
        side_features=t4,
        decoder_key_features=t4,
    )
    output.square().mean().backward()
    assert neural.grad is not None and torch.isfinite(neural.grad).all()
    assert calib.grad is not None and torch.isfinite(calib.grad).all()
    assert t4.grad is not None and torch.isfinite(t4.grad).all()
    assert model.id_encoder.scale.grad is not None
    assert model.decoupled_v2.direct_key_proj.weight.grad is not None


def test_freeze_decoder_includes_v2_but_not_identity_encoder():
    model = _model("e_t4")
    frozen = model.freeze_decoder()
    assert frozen > 0
    assert all(not parameter.requires_grad for parameter in model.decoder.parameters())
    assert all(
        not parameter.requires_grad
        for parameter in model.decoupled_v2.parameters()
    )
    assert model.id_encoder.scale.requires_grad
    model.train()
    assert model.decoder.training is False
    assert model.decoupled_v2.training is False


def test_unused_legacy_decoder_modules_are_excluded_from_end_to_end_optimizer():
    model = _model("e_t4")
    assert all(
        not parameter.requires_grad
        for parameter in model.decoder.transformer.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.decoder.fc_id_in.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model.decoder.fc_id_out.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.decoder.fc_in.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.decoder.fc_out.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.decoupled_v2.parameters()
    )


def test_single_session_static_state_expands_across_online_batch():
    model = _model("e_t4")
    model.eval()
    neural = torch.randn(3, 6, 5)
    identity = torch.randn(1, 5, 6)
    direct = torch.randn(1, 5, 4)
    state = model.derive_decoupled_kv_state(identity, direct)
    cached = model.decode_with_decoupled_kv_state(neural, state)
    expanded = model.decode_with_decoupled_identity(
        neural,
        identity.expand(3, -1, -1),
        decoder_key_features=direct.expand(3, -1, -1),
    )
    assert torch.allclose(cached, expanded, atol=1e-6)
