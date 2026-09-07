from __future__ import annotations

import torch

from tfpd_exploration.src.m2_family_v1 import M2SlotRoutingBias, add_route_bias
from tfpd_exploration.src.m2_family_v1.decoder import ExplicitSlotUnitAttention, make_paired_decoders, shared_parameter_max_abs_diff
from tfpd_exploration.src.m2_dual_track_v1.contracts import SessionBank


def test_zero_gate_is_exact_flat_logits_and_softmax() -> None:
    torch.manual_seed(20260906)
    routing = M2SlotRoutingBias()
    flat = torch.randn(2, 8, 8, 96)
    e0 = torch.randn(2, 96, 50)
    functional = torch.randn(2, 96, 4)
    routed = add_route_bias(flat, routing, e0, functional)
    assert torch.equal(routed, flat)
    assert torch.equal(torch.softmax(routed, dim=-1), torch.softmax(flat, dim=-1))


def test_route_uses_only_calibration_carrier_and_gate_can_learn_at_zero() -> None:
    torch.manual_seed(20260906)
    routing = M2SlotRoutingBias()
    # Make route scores observably nonzero but retain the required zero gate.
    e0 = torch.randn(1, 96, 50)
    functional = torch.randn(1, 96, 4)
    flat = torch.randn(1, 8, 8, 96, requires_grad=True)
    assert torch.count_nonzero(routing(e0, functional)) == 0
    routed = add_route_bias(flat, routing, e0, functional)
    # A non-symmetric weighted loss gives each head a nonzero path to g.
    weights = torch.arange(routed.numel(), dtype=routed.dtype).reshape_as(routed)
    (routed * weights).sum().backward()
    assert routing.g.grad is not None
    assert torch.count_nonzero(routing.g.grad) == routing.cfg.heads


def test_static_carrier_has_no_time_axis_or_spike_input() -> None:
    routing = M2SlotRoutingBias()
    # The public forward accepts only E0 and T.  It broadcasts a frozen carrier
    # over a batch, which is the intended static-per-session semantics.
    e0 = torch.randn(96, 50)
    functional = torch.randn(96, 4)
    assert routing(e0, functional).shape == (1, 8, 8, 96)


def test_explicit_attention_maps_mha_forward_and_backward_with_mask() -> None:
    torch.manual_seed(17)
    mha = torch.nn.MultiheadAttention(256, 8, dropout=0.0, batch_first=True)
    explicit = ExplicitSlotUnitAttention.from_mha(mha, routed=False)
    slots = torch.randn(2, 8, 256, requires_grad=True)
    tokens = torch.randn(2, 96, 256, requires_grad=True)
    mask = torch.zeros(2, 96, dtype=torch.bool)
    mask[:, 70:] = True
    x0, y0 = slots.detach().clone().requires_grad_(), tokens.detach().clone().requires_grad_()
    expected, _ = mha(x0, y0, y0, key_padding_mask=mask, need_weights=False)
    actual = explicit(slots, tokens, mask)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
    expected.square().mean().backward()
    actual.square().mean().backward()
    assert torch.allclose(slots.grad, x0.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(tokens.grad, y0.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(explicit.in_proj_weight.grad, mha.in_proj_weight.grad, atol=1e-5, rtol=1e-5)


def _bank() -> SessionBank:
    import numpy as np
    return SessionBank("cpu-test", (), (), np.empty((0,)), np.empty((0,)), np.empty((0,), dtype=np.int64),
                       torch.randn(96, 50), torch.randn(96, 4), torch.ones(96, dtype=torch.bool), {})


def test_paired_factory_copies_shared_weights_and_zero_gate_is_full_forward_equal() -> None:
    torch.manual_seed(99)
    flat, route = make_paired_decoders(42)
    assert shared_parameter_max_abs_diff(flat, route) == 0.0
    flat.eval(); route.eval()
    x, bank = torch.randn(2, 50, 96), _bank()
    with torch.no_grad():
        left = flat.forward_last(x, bank)
        right = route.forward_last(x, bank)
    assert torch.allclose(left, right, atol=1e-5, rtol=1e-5)


def test_zero_gate_full_route_still_has_gate_gradient() -> None:
    flat, route = make_paired_decoders(43)
    del flat
    route.train()
    loss = route.forward_last(torch.randn(2, 50, 96), _bank()).square().mean()
    loss.backward()
    gradient = route.frontend.attn.routing.g.grad
    assert gradient is not None and torch.count_nonzero(gradient) == 8
