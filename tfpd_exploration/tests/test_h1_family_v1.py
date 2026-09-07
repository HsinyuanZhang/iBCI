from __future__ import annotations

import torch

from tfpd_exploration.src.h1_family_v1.model import (
    initialization_receipt,
    make_v2_initialized_pair,
    make_v2_unscaled_dot_pair,
    make_v2_unscaled_dot_localbalanced_pair,
    local_fc1_balance_factor,
    route_gate_gradient_l1,
    zero_gate_parity,
)
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank


def _synthetic_bank() -> H1Bank:
    g = torch.Generator().manual_seed(19)
    return H1Bank(
        E0=torch.randn(176, 700, generator=g),
        T=torch.randn(176, 4, generator=g),
        unit_mask=torch.ones(176, dtype=torch.bool),
    )


def test_v2_initialized_common_flat_route_gzero_parity_and_gate_gradient() -> None:
    flat, route = make_v2_initialized_pair(seed=42)
    receipt = initialization_receipt(flat, route)
    assert receipt["shared_parameter_max_abs_diff"] == 0.0
    assert receipt["route_only_parameter_tensors"] > 0
    x = torch.randn(1, 7, 176, generator=torch.Generator().manual_seed(23))
    bank = _synthetic_bank()
    assert zero_gate_parity(flat, route, x, bank)["max_abs_diff"] == 0.0
    route.zero_grad(set_to_none=True)
    route.forward_last(x, bank).square().mean().backward()
    assert route_gate_gradient_l1(route) > 0.0


def test_v2_flat_constructor_is_raw_scale_one_only_for_family_pair() -> None:
    flat, route = make_v2_initialized_pair(seed=42)
    assert flat.activity_scale == 1.0
    assert route.activity_scale == 1.0


def test_unscaled_dot_keeps_v2_weights_and_flat_route_gzero_contract() -> None:
    baseline_flat, _ = make_v2_initialized_pair(seed=42)
    flat, route = make_v2_unscaled_dot_pair(seed=42)
    for name, value in baseline_flat.state_dict().items():
        assert torch.equal(value, flat.state_dict()[name])
    bank = _synthetic_bank()
    x = torch.randn(1, 7, 176, generator=torch.Generator().manual_seed(29))
    assert zero_gate_parity(flat, route, x, bank)["max_abs_diff"] == 0.0
    # This test establishes that the custom attention path is actually active,
    # rather than merely carrying a version label around V2's old forward.
    assert not torch.equal(baseline_flat.forward_last(x, bank), flat.forward_last(x, bank))


def test_localbalanced_init_only_scales_the_shared_local_fc1_block() -> None:
    baseline, _ = make_v2_unscaled_dot_pair(seed=42)
    flat, route = make_v2_unscaled_dot_localbalanced_pair(seed=42)
    factor = local_fc1_balance_factor(flat.cfg)
    base_weight = baseline.frontend.token_mlp.fc1.weight
    assert torch.equal(flat.frontend.token_mlp.fc1.weight[:, 16:], base_weight[:, 16:])
    assert torch.equal(route.frontend.token_mlp.fc1.weight, flat.frontend.token_mlp.fc1.weight)
    assert torch.equal(flat.frontend.token_mlp.fc1.weight[:, :16], base_weight[:, :16] * factor)
    bank = _synthetic_bank()
    x = torch.randn(1, 7, 176, generator=torch.Generator().manual_seed(31))
    assert zero_gate_parity(flat, route, x, bank)["max_abs_diff"] == 0.0
