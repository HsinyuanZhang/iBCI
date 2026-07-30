"""Tests for calibration-derived fixed-token routing."""
from __future__ import annotations

import torch

from src.models.components.spint import SpintModel
from src.models.components.streaming_encoders import EarlyPoolEncoder
from src.models.components.streaming_spint import CalibrationFixedSlotRouter, StreamingSpintModel


def build_model(slot_count: int = 3, routing_mode: str = "soft") -> StreamingSpintModel:
    decoder = SpintModel(
        model_dim=16,
        num_covariates=2,
        window_size=5,
        num_heads=4,
        num_layers=1,
        dropout_rate=0.0,
        tf_drop_rate=0.0,
    )
    encoder = EarlyPoolEncoder(trial_length=4, window_size=5, hidden_dim=8)
    return StreamingSpintModel(
        decoder=decoder,
        id_encoder=encoder,
        fixed_slot_count=slot_count,
        fixed_slot_dim=6,
        fixed_slot_mode=routing_mode,
        fixed_slot_fusion="film",
    )


def test_fixed_slot_router_returns_fixed_decoder_tokens_and_preserves_identity_shape():
    torch.manual_seed(17)
    model = build_model().eval()
    neural = torch.randn(2, 5, 7)
    calibration = torch.randn(2, 4, 4, 7)

    with torch.no_grad():
        behavior, identity = model(neural, calibration)

    assert behavior.shape == (2, 5, 2)
    assert identity.shape == (2, 7, 5)
    assert model.fixed_slot_router is not None


def test_fixed_slot_state_matches_end_to_end_decode_without_recomputing_identity():
    torch.manual_seed(19)
    model = build_model().eval()
    neural = torch.randn(2, 5, 7)
    calibration = torch.randn(2, 4, 4, 7)

    with torch.no_grad():
        identity = model.compute_identity(calibration)
        direct = model.decode_with_identity(neural, identity)
        calibration_state = model.derive_fixed_slot_state(identity, num_neurons=7)
        cached = model.decode_with_fixed_slot_state(neural, calibration_state)

    assert torch.allclose(direct, cached, atol=1.0e-6)
    assert calibration_state["assignment"].shape == (2, 7, 3)
    assert calibration_state["slot_mass"].shape == (2, 3)
    assert calibration_state["gain"].shape == (2, 3, 5)


def test_fixed_slot_state_reuses_calibration_derived_neuron_gate():
    torch.manual_seed(21)
    model = build_model().eval()
    neural = torch.randn(2, 5, 7)
    calibration = torch.randn(2, 4, 4, 7)
    neuron_gate = torch.rand(2, 7, 1)

    with torch.no_grad():
        identity = model.compute_identity(calibration)
        direct = model.decode_with_identity(neural, identity, neuron_gate=neuron_gate)
        calibration_state = model.derive_fixed_slot_state(
            identity,
            num_neurons=7,
            neuron_gate=neuron_gate,
        )
        cached = model.decode_with_fixed_slot_state(neural, calibration_state)

    assert torch.allclose(direct, cached, atol=1.0e-6)
    assert torch.equal(calibration_state["neuron_gate"], neuron_gate)


def test_single_session_fixed_slot_state_broadcasts_across_online_windows():
    torch.manual_seed(22)
    model = build_model().eval()
    calibration = torch.randn(1, 4, 4, 7)
    neural = torch.randn(3, 5, 7)
    neuron_gate = torch.rand(1, 7, 1)

    with torch.no_grad():
        identity = model.compute_identity(calibration)
        state = model.derive_fixed_slot_state(
            identity,
            num_neurons=7,
            neuron_gate=neuron_gate,
        )
        broadcast_output = model.decode_with_fixed_slot_state(neural, state)
        repeated_state = {
            key: value.expand(neural.shape[0], *value.shape[1:])
            for key, value in state.items()
        }
        repeated_output = model.decode_with_fixed_slot_state(neural, repeated_state)

    assert broadcast_output.shape == (3, 5, 2)
    assert torch.allclose(broadcast_output, repeated_output, atol=1.0e-6)


def test_fixed_slot_router_is_invariant_to_a_consistent_unit_permutation():
    torch.manual_seed(23)
    model = build_model().eval()
    neural = torch.randn(1, 5, 7)
    calibration = torch.randn(1, 4, 4, 7)
    permutation = torch.tensor([4, 0, 6, 2, 1, 5, 3])

    with torch.no_grad():
        original, _ = model(neural, calibration)
        permuted, _ = model(neural[..., permutation], calibration[..., permutation])

    assert torch.allclose(original, permuted, atol=1.0e-6)


def test_top1_router_assigns_each_unit_to_exactly_one_slot():
    torch.manual_seed(29)
    router = CalibrationFixedSlotRouter(
        window_size=5,
        slot_count=3,
        router_dim=6,
        routing_mode="top1",
        fusion="film",
        temperature=1.0,
    ).eval()
    identity = torch.randn(2, 7, 5)

    with torch.no_grad():
        calibration_state = router.derive_calibration_state(identity, num_neurons=7)

    assignment = calibration_state["assignment"]
    assert torch.equal(assignment.sum(dim=-1), torch.ones(2, 7))
    assert torch.all((assignment == 0.0) | (assignment == 1.0))


def test_lower_routing_temperature_sharpens_the_same_assignment_logits():
    torch.manual_seed(37)
    warm_router = CalibrationFixedSlotRouter(
        window_size=5,
        slot_count=4,
        router_dim=6,
        routing_mode="soft",
        fusion="film",
        temperature=1.0,
    ).eval()
    sharp_router = CalibrationFixedSlotRouter(
        window_size=5,
        slot_count=4,
        router_dim=6,
        routing_mode="soft",
        fusion="film",
        temperature=0.1,
    ).eval()
    sharp_router.load_state_dict(warm_router.state_dict())
    identity = torch.randn(2, 7, 5)

    with torch.no_grad():
        warm_assignment = warm_router.derive_calibration_state(identity, num_neurons=7)["assignment"]
        sharp_assignment = sharp_router.derive_calibration_state(identity, num_neurons=7)["assignment"]

    warm_entropy = -(warm_assignment * warm_assignment.clamp_min(1.0e-9).log()).sum(dim=-1)
    sharp_entropy = -(sharp_assignment * sharp_assignment.clamp_min(1.0e-9).log()).sum(dim=-1)
    assert float(sharp_entropy.mean()) < float(warm_entropy.mean())
