"""CPU correctness for M1 small-B FLAT / ROUTE. No S-Fix decoder-weight init."""

from __future__ import annotations

import torch

from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_calibration import (
    S_FIX_PATH,
    S_FIX_SHA256,
    sha256_file,
    sfix_decoder_keys,
)
from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_config import M1_TEMPORAL
from tfpd_exploration.src.two_mainlines_long_v1.decoder.m1_temporal import (
    M1Bank,
    M1TemporalFlatDecoder,
    M1TemporalRouteDecoder,
    count_m1_decoder_parameters,
    prove_zero_gate_equals_flat,
    shared_parameter_max_abs_diff,
)


def _bank(n: int = 64, seed: int = 0) -> M1Bank:
    g = torch.Generator().manual_seed(seed)
    return M1Bank(
        E0=torch.randn(n, M1_TEMPORAL.e0_dim, generator=g),
        T=torch.randn(n, M1_TEMPORAL.hc_dim, generator=g),
        unit_mask=torch.ones(n, dtype=torch.bool),
    )


def test_contract_is_m1_not_h1_or_m2() -> None:
    assert M1_TEMPORAL.window == 100
    assert M1_TEMPORAL.out_dim == 16
    assert M1_TEMPORAL.e0_dim == 100
    assert M1_TEMPORAL.n_units == 64
    assert M1_TEMPORAL.pe_max_len >= 100
    assert M1_TEMPORAL.prediction_divisor == 1.0


def test_pe_covers_100_not_256_or_700() -> None:
    model = M1TemporalFlatDecoder(seed=42)
    assert int(model.temporal.pe.size(0)) == 100
    assert int(model.cfg.out_dim) == 16


def test_output_is_16d_emg() -> None:
    y = M1TemporalFlatDecoder(seed=42).forward_last(torch.randn(2, 100, 64), _bank())
    assert tuple(y.shape) == (2, 16)


def test_zero_gate_equals_flat() -> None:
    flat = M1TemporalFlatDecoder(seed=42)
    route = M1TemporalRouteDecoder(seed=42, flat_template=flat)
    assert float(route.frontend.attn.g.abs().max()) == 0.0
    assert shared_parameter_max_abs_diff(flat, route) == 0.0
    prove_zero_gate_equals_flat(flat, route, torch.randn(2, 32, 64), _bank())


def test_routing_grads_after_two_updates() -> None:
    flat = M1TemporalFlatDecoder(seed=42)
    route = M1TemporalRouteDecoder(seed=42, flat_template=flat)
    opt = torch.optim.AdamW(route.parameters(), lr=1e-3)
    bank = _bank()
    x = torch.randn(2, 16, 64)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        route.forward_last(x, bank).square().mean().backward()
        opt.step()
    opt.zero_grad(set_to_none=True)
    route.forward_last(x, bank).square().mean().backward()
    assert float(route.frontend.attn.g.grad.abs().sum()) > 0.0
    assert float(route.frontend.attn.q_cal.grad.abs().sum()) > 0.0


def test_causal_future_perturbation() -> None:
    model = M1TemporalFlatDecoder(seed=42).eval()
    bank = _bank()
    x = torch.randn(1, 20, 64)
    x2 = x.clone()
    x2[:, 15:] += 3.0
    with torch.no_grad():
        a = model.forward_scores(x, bank)
        b = model.forward_scores(x2, bank)
    assert torch.allclose(a[:, :15], b[:, :15], atol=1e-5, rtol=1e-4)
    assert not torch.allclose(a[:, 15:], b[:, 15:], atol=1e-5, rtol=1e-4)


def test_sfix_sha_and_new_decoder_does_not_load_sfix_decoder() -> None:
    assert sha256_file(S_FIX_PATH) == S_FIX_SHA256
    flat = M1TemporalFlatDecoder(seed=42)
    overlap = set(flat.state_dict()).intersection(sfix_decoder_keys())
    assert overlap == set()


def test_route_has_more_params() -> None:
    flat = M1TemporalFlatDecoder(seed=42)
    route = M1TemporalRouteDecoder(seed=42, flat_template=flat)
    assert count_m1_decoder_parameters(route)["routing"] > 0
    assert count_m1_decoder_parameters(route)["decoder_total"] > count_m1_decoder_parameters(flat)["decoder_total"]
