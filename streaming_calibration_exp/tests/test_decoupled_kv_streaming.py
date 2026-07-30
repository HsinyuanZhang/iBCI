"""CPU contracts for decoupled K/V wiring above the attention core."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from src.models.components.spint import SpintModel
from src.models.components.streaming_spint import StreamingSpintModel
from src.models.streaming_calibration_module import StreamingCalibrationLitModule


def _decoder(model_dim: int = 16, num_heads: int = 2) -> SpintModel:
    return SpintModel(
        model_dim=model_dim,
        num_covariates=2,
        window_size=50,
        num_heads=num_heads,
        num_layers=1,
        dropout_rate=0.0,
        dynamic_dropout=False,
        tf_drop_rate=0.0,
    )


def _streaming(key_mode: str = "e_t4") -> StreamingSpintModel:
    return StreamingSpintModel(
        decoder=_decoder(),
        id_encoder=nn.Identity(),
        decoder_mode="decoupled",
        decoupled_key_mode=key_mode,
        decoupled_key_dim=32,
        decoupled_value_dim=32,
    ).eval()


def _inputs(batch: int = 2, units: int = 7):
    return (
        torch.randn(batch, 50, units),
        torch.randn(batch, units, 50),
        torch.randn(batch, units, 4),
    )


def test_static_key_training_path_matches_exported_cached_state():
    torch.manual_seed(11)
    model = _streaming("e_t4")
    neural, identity, t4 = _inputs()
    with torch.no_grad():
        direct = model.decode_with_decoupled_identity(neural, identity, t4)
        state = model.derive_decoupled_kv_state(identity, t4)
        cached = model.decode_with_decoupled_kv_state(neural, state)
    assert torch.allclose(direct, cached, atol=1e-6)
    assert set(vars(state)) == {"projected_keys"}
    assert state.projected_key.shape == (2, 7, 32)


def test_key_only_t4_change_does_not_change_identity_or_value_input(monkeypatch):
    torch.manual_seed(12)
    model = _streaming("e_t4")
    neural, identity, t4 = _inputs(batch=1)
    shuffled = t4[:, torch.tensor([4, 0, 6, 2, 1, 5, 3])]
    observed_values: list[torch.Tensor] = []
    original = model.decoupled_transformer.layers[0].value_proj.forward

    def capture(value: torch.Tensor) -> torch.Tensor:
        observed_values.append(value.detach().clone())
        return original(value)

    monkeypatch.setattr(
        model.decoupled_transformer.layers[0].value_proj, "forward", capture
    )
    with torch.no_grad():
        first = model.decode_with_decoupled_identity(neural, identity, t4)
        second = model.decode_with_decoupled_identity(neural, identity, shuffled)
    assert not torch.equal(first, second)
    assert len(observed_values) == 2
    assert torch.equal(observed_values[0], observed_values[1])
    assert torch.equal(identity, identity.clone())


def test_x_only_key_removes_identity_from_task_path():
    torch.manual_seed(13)
    model = _streaming("x_only")
    neural, identity, _t4 = _inputs(batch=1)
    with torch.no_grad():
        first = model.decode_with_decoupled_identity(neural, identity)
        second = model.decode_with_decoupled_identity(
            neural, identity + torch.randn_like(identity) * 100.0
        )
    assert torch.equal(first, second)
    with pytest.raises(RuntimeError, match="no static K state"):
        model.derive_decoupled_kv_state(identity)


def test_static_identity_key_backpropagates_to_identity_and_all_projections():
    torch.manual_seed(14)
    model = _streaming("e_t4").train()
    neural, identity, t4 = _inputs(batch=1)
    identity.requires_grad_(True)
    output = model.decode_with_decoupled_identity(neural, identity, t4)
    output.square().mean().backward()
    assert identity.grad is not None and torch.isfinite(identity.grad).all()
    parameters = dict(model.decoupled_transformer.layers[0].named_parameters())
    for name in (
        "query_proj.weight",
        "key_proj.weight",
        "value_proj.weight",
        "out_proj.weight",
    ):
        assert parameters[name].grad is not None
        assert torch.isfinite(parameters[name].grad).all()


def test_decoupled_module_build_guards_reject_fixed_slots_and_non_t4_width():
    with pytest.raises(ValueError, match="fixed_slot_count=0"):
        StreamingSpintModel(
            decoder=_decoder(),
            id_encoder=nn.Identity(),
            fixed_slot_count=4,
            decoder_mode="decoupled",
        )
    with pytest.raises(ValueError, match="four-dimensional T4"):
        StreamingCalibrationLitModule(
            task="mc_maze",
            variant="B3S",
            teacher_ckpt_path="/not/opened.ckpt",
            window_size=50,
            side_dim=6,
            decoder_mode="decoupled",
        )


def test_decoder_key_feature_routing_is_key_only_and_deterministic():
    module = StreamingCalibrationLitModule(
        task="mc_maze",
        variant="B3S",
        teacher_ckpt_path="/not/opened.ckpt",
        window_size=50,
        side_dim=4,
        decoder_mode="decoupled",
        decoupled_key_mode="e_ts4",
        decoupled_key_permutation_seed=17,
    )
    t4 = torch.arange(2 * 7 * 4, dtype=torch.float32).reshape(2, 7, 4)
    original = t4.clone()
    observed = module.decoder_key_features(t4)
    order = torch.as_tensor(np.random.RandomState(17).permutation(7))
    assert torch.equal(observed, t4.index_select(1, order))
    assert torch.equal(t4, original)


def test_all_decoupled_key_modes_have_identical_parameter_initialization():
    states = []
    for mode in ("e_t4", "e_ts4", "e_only", "x_only"):
        torch.manual_seed(23)
        model = _streaming(mode)
        states.append(model.decoupled_transformer.state_dict())
    for name in states[0]:
        assert all(torch.equal(states[0][name], state[name]) for state in states[1:])


def test_legacy_coupled_default_has_no_new_module_or_output_drift():
    torch.manual_seed(29)
    decoder = _decoder().eval()
    model = StreamingSpintModel(decoder=decoder, id_encoder=nn.Identity()).eval()
    neural, identity, _ = _inputs(batch=1)
    with torch.no_grad():
        expected = model.decode_with_identity(neural, identity)
        observed, returned_identity = model(neural, identity=identity)
    assert model.decoder_mode == "coupled"
    assert model.decoupled_transformer is None
    assert torch.equal(observed, expected)
    assert returned_identity is identity


def test_full_decoder_cost_receipt_includes_shared_work_and_large_reduction():
    # Use the actual MC_Maze teacher width/head configuration. At the tiny
    # d_model=16 unit-test fixture, a deliberately fixed Dk=Dv=32 low-rank
    # pilot is wider than the legacy attention and is not an efficiency claim.
    model = StreamingSpintModel(
        decoder=_decoder(model_dim=512, num_heads=64),
        id_encoder=nn.Identity(),
        decoder_mode="decoupled",
        decoupled_key_mode="e_t4",
        decoupled_key_dim=32,
        decoupled_value_dim=32,
        decoupled_num_heads=2,
    )
    receipt = model.decoder_cost_comparison_receipt(
        batch_size=1, num_neurons=64
    )
    coupled = receipt["coupled"]
    decoupled = receipt["decoupled"]
    assert coupled["total"] > decoupled["total"]
    assert decoupled["online_mac_reduction_fraction_vs_coupled"] >= 0.25
    assert decoupled["persistent_state_width"] == 32
    assert decoupled["persistent_state_bytes_fp32"] == 64 * 32 * 4
    assert decoupled["persistent_state_nonincreasing_vs_coupled"] is True
    assert decoupled["no_unit_quadratic_term"] is True

    legacy = StreamingSpintModel(
        decoder=_decoder(), id_encoder=nn.Identity()
    )
    legacy_receipt = legacy.decoder_cost_comparison_receipt(
        batch_size=1, num_neurons=64
    )
    assert legacy_receipt["active_mode"] == "coupled"
    assert "decoupled" not in legacy_receipt
