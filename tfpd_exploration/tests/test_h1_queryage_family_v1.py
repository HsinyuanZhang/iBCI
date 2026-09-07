from dataclasses import replace

import pytest
import torch

from tfpd_exploration.src.h1_queryage_family_v1.model import (
    FAMILY_FLAT,
    FAMILY_ROUTE,
    make_queryage_localbalanced_pair,
)
from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2 import QueryTemporalStack
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_config import H1_TEMPORAL
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank, shared_parameter_max_abs_diff


def bank(batch=1):
    generator = torch.Generator().manual_seed(321)
    return H1Bank(torch.randn(batch, 176, 700, generator=generator) * .03,
                  torch.randn(batch, 176, 4, generator=generator) * .1,
                  torch.rand(batch, 176, generator=generator) > .1)


def test_exact_w700_topology_names_and_pair_identity():
    flat, route = make_queryage_localbalanced_pair(seed=42)
    for model, expected in ((flat, FAMILY_FLAT), (route, FAMILY_ROUTE)):
        temporal = model.temporal
        assert model.name == expected and isinstance(temporal, QueryTemporalStack)
        assert (temporal.width, temporal.heads, temporal.window, temporal.age_buckets, len(temporal.blocks)) == (256, 8, 700, 16, 4)
        assert all(block.ffn[0].out_features == 512 for block in temporal.blocks)
        assert not hasattr(temporal, "pe")
        assert model.prediction_divisor == 20.0
        assert model.training_target_space == "runtime_scaled_velocity__target_is_20x_native"
    assert shared_parameter_max_abs_diff(flat, route) == 0.0
    assert torch.equal(route.frontend.attn.g, torch.zeros_like(route.frontend.attn.g))


def test_current_query_forward_mask_semantics_gate_gradient_and_native20():
    torch.set_num_threads(1)
    flat, route = make_queryage_localbalanced_pair(seed=42)
    x = torch.randn(1, 700, 176, generator=torch.Generator().manual_seed(322))
    item = bank()
    keep = item.unit_mask.clone()
    z = flat.encode_frontend(x, item, keep)
    assert flat.temporal(z).shape == (1, 1, 256)
    flat.eval(); route.eval()
    with torch.inference_mode():
        f = flat.forward_last(x, item, keep)
        r = route.forward_last(x, item, keep)
        changed_keep = keep.clone(); changed_keep[:, ::3] = False
        masked = flat.forward_last(x, item, changed_keep)
    torch.testing.assert_close(f, r, atol=1e-5, rtol=1e-5)
    assert not torch.equal(f, masked)
    assert torch.isfinite(f).all() and torch.isfinite(f / 20).all() and f.shape == (1, 7)
    route.train(); route.zero_grad(set_to_none=True)
    route.forward_last(x, item, keep).square().mean().backward()
    assert route.frontend.attn.g.grad is not None and float(route.frontend.attn.g.grad.abs().max()) > 0
    route.eval()
    with torch.inference_mode():
        g0 = route.forward_last(x, item, keep)
        route.frontend.attn.g.fill_(.2)
        g1 = route.forward_last(x, item, keep)
    assert not torch.equal(g0, g1)


def test_plain_state_reload_and_w700_rejection():
    flat, route = make_queryage_localbalanced_pair(seed=42)
    plain = {name: value.detach().clone() for name, value in route.state_dict().items()}
    _flat2, restored = make_queryage_localbalanced_pair(seed=42)
    restored.load_state_dict(plain, strict=True)
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in plain.items())
    with pytest.raises(ValueError, match="W700"):
        make_queryage_localbalanced_pair(cfg=replace(H1_TEMPORAL, window=699))
