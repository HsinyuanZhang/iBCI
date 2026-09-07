"""CPU correctness for H1 temporal FLAT / ROUTE. No C2 decoder-weight init."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_calibration import (
    C2_CKPT_SHA256,
    C2_EPOCH15_PATH,
    load_frozen_c2_materializer,
    sha256_file,
)
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_config import H1_TEMPORAL
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import (
    H1Bank,
    H1TemporalFlatDecoder,
    H1TemporalRouteDecoder,
    copy_flat_into_route,
    count_h1_decoder_parameters,
    prove_zero_gate_equals_flat,
    shared_parameter_max_abs_diff,
)


def _bank(n: int = 176, seed: int = 0) -> H1Bank:
    g = torch.Generator().manual_seed(seed)
    return H1Bank(
        E0=torch.randn(n, H1_TEMPORAL.e0_dim, generator=g),
        T=torch.randn(n, H1_TEMPORAL.hc_dim, generator=g),
        unit_mask=torch.ones(n, dtype=torch.bool),
    )


def test_pe_covers_700() -> None:
    assert H1_TEMPORAL.pe_max_len >= 700
    assert H1_TEMPORAL.window == 700
    model = H1TemporalFlatDecoder(seed=42)
    assert int(model.temporal.pe.size(0)) >= 700
    assert int(model.temporal.pe.size(0)) != 256 or H1_TEMPORAL.pe_max_len >= 700


def test_output_contract() -> None:
    model = H1TemporalFlatDecoder(seed=42)
    y = model.forward_last(torch.randn(2, 32, 176), _bank())
    assert tuple(y.shape) == (2, 7)


def test_factored_token_mlp_matches_concat() -> None:
    model = H1TemporalFlatDecoder(seed=42)
    local = torch.randn(2, 5, 176, 16)
    e0 = torch.randn(176, 700)
    hc = torch.randn(176, 4)
    left = model.frontend.token_mlp(local, e0, hc)
    right = model.frontend.token_mlp.naive_concat(local, e0, hc)
    assert torch.allclose(left, right, atol=1e-5, rtol=1e-4)


def test_zero_gate_equals_flat() -> None:
    flat = H1TemporalFlatDecoder(seed=42)
    route = H1TemporalRouteDecoder(seed=42, flat_template=flat)
    assert float(route.frontend.attn.g.abs().max()) == 0.0
    assert shared_parameter_max_abs_diff(flat, route) == 0.0
    x = torch.randn(2, 24, 176)
    prove_zero_gate_equals_flat(flat, route, x, _bank())


def test_routing_grads_after_real_update() -> None:
    flat = H1TemporalFlatDecoder(seed=42)
    route = H1TemporalRouteDecoder(seed=42, flat_template=flat)
    opt = torch.optim.AdamW(route.parameters(), lr=1e-3)
    bank = _bank()
    x = torch.randn(2, 16, 176)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        pred = route.forward_last(x, bank)
        pred.square().mean().backward()
        opt.step()
    opt.zero_grad(set_to_none=True)
    route.forward_last(x, bank).square().mean().backward()
    g_grad = route.frontend.attn.g.grad
    q_grad = route.frontend.attn.q_cal.grad
    p_grad = route.frontend.attn.route_proj[0].weight.grad
    assert g_grad is not None and float(g_grad.abs().sum()) > 0.0
    assert q_grad is not None and float(q_grad.abs().sum()) > 0.0
    assert p_grad is not None and float(p_grad.abs().sum()) > 0.0


def test_unit_mask_fail_closed() -> None:
    model = H1TemporalFlatDecoder(seed=42).eval()
    bank = _bank()
    keep = torch.zeros(176, dtype=torch.bool)
    keep[3] = True
    x = torch.randn(1, 12, 176)
    with torch.no_grad():
        y = model.forward_last(x, bank, unit_mask=keep)
    assert torch.isfinite(y).all()


def test_causal_future_perturbation() -> None:
    model = H1TemporalFlatDecoder(seed=42).eval()
    bank = _bank()
    x = torch.randn(1, 20, 176)
    x2 = x.clone()
    x2[:, 15:] += 3.0
    with torch.no_grad():
        a = model.forward_scores(x, bank)
        b = model.forward_scores(x2, bank)
    assert torch.allclose(a[:, :15], b[:, :15], atol=1e-5, rtol=1e-4)
    assert not torch.allclose(a[:, 15:], b[:, 15:], atol=1e-5, rtol=1e-4)


def test_tile_parity_cpu() -> None:
    a = H1TemporalFlatDecoder(seed=42).train()
    b = H1TemporalFlatDecoder(seed=42).train()
    b.load_state_dict(a.state_dict())
    b.set_memory_policy(7, False)
    bank = _bank()
    keep = torch.ones(1, 176, dtype=torch.bool)
    x = torch.randn(1, 21, 176, requires_grad=True)
    x2 = x.detach().clone().requires_grad_(True)
    y = torch.randn(1, 7)
    pred_a = a.forward_last(x, bank, dropout_keep=keep)
    pred_b = b.forward_last(x2, bank, dropout_keep=keep)
    torch.nn.functional.mse_loss(pred_a, y).backward()
    torch.nn.functional.mse_loss(pred_b, y).backward()
    assert torch.allclose(pred_a.detach(), pred_b.detach(), atol=1e-5, rtol=1e-4)
    assert torch.allclose(x.grad, x2.grad, atol=2e-4, rtol=1e-4)


def test_c2_sha_and_materializer_does_not_expose_decoder_init() -> None:
    assert sha256_file(C2_EPOCH15_PATH) == C2_CKPT_SHA256
    mat = load_frozen_c2_materializer()
    activity = torch.randn(1, 3, 1024, 176)
    carrier = torch.randn(1, 176, 4)
    e0, hc = mat.materialize_bank(activity, carrier)
    assert tuple(e0.shape) == (176, 700)
    assert tuple(hc.shape) == (176, 4)
    assert mat.e0_is_fused_identity
    for parameter in list(mat.pre_pool.parameters()) + list(mat.post_pool.parameters()):
        assert parameter.requires_grad is False


def test_new_decoder_not_initialized_from_c2_transformer() -> None:
    payload = torch.load(C2_EPOCH15_PATH, map_location="cpu", weights_only=False)
    c2 = payload["state_dict"]
    flat = H1TemporalFlatDecoder(seed=42)
    names = set(flat.state_dict())
    overlap = names.intersection(c2)
    assert overlap == set()
    digest = hashlib.sha256()
    for name, tensor in sorted(flat.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    # Just prove the model exists independently; hash is not the C2 state hash.
    assert digest.hexdigest() != payload["metadata"]["state_sha256"]


def test_route_has_more_params() -> None:
    flat = H1TemporalFlatDecoder(seed=42)
    route = H1TemporalRouteDecoder(seed=42, flat_template=flat)
    assert count_h1_decoder_parameters(route)["routing"] > 0
    assert count_h1_decoder_parameters(route)["decoder_total"] > count_h1_decoder_parameters(flat)["decoder_total"]


def test_copy_then_route_rng_does_not_move_shared() -> None:
    flat = H1TemporalFlatDecoder(seed=42)
    route = H1TemporalRouteDecoder(seed=99)
    copy_flat_into_route(flat, route)
    before = {k: v.detach().clone() for k, v in route.state_dict().items() if "attn.q_cal" not in k and "attn.g" not in k and "route_proj" not in k}
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import initialize_route_only

    initialize_route_only(route, 42)
    after = route.state_dict()
    for key, tensor in before.items():
        assert torch.equal(tensor, after[key])
    assert float(route.frontend.attn.g.abs().max()) == 0.0
