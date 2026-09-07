from types import SimpleNamespace
import torch
import pytest

from tfpd_exploration.src.m2_family_v1.decoder import make_paired_decoders
from tfpd_exploration.src.family_runtime_v1.m2_family_spatial import M2FamilyLiftedSpatial


@pytest.mark.parametrize("batch", [1, 7])
@pytest.mark.parametrize("routed,g", [(False, 0.0), (True, 0.0), (True, 0.2)])
def test_actual_family_lift_matches_native_and_never_calls_native_frontend(batch, routed, g):
    torch.manual_seed(91 + batch)
    flat, route = make_paired_decoders(42); model = route if routed else flat
    model.eval()
    with torch.no_grad():
        if routed: model.frontend.attn.routing.g.fill_(g)
    bank = SimpleNamespace(E0=torch.randn(batch, 96, 50), T=torch.randn(batch, 96, 4),
                           unit_mask=torch.rand(batch, 96) > .15)
    bank.unit_mask[:, 0] = True
    raw = torch.randn(batch, 50, 96)
    native = model.frontend(raw, bank, bank.unit_mask)
    lift = M2FamilyLiftedSpatial(model, bank)
    derived = lift.derived_tensors()
    assert all(torch.is_tensor(t) for t in derived) and len(derived) == (6 if routed else 5)
    original = model.frontend.forward
    def forbidden(*args, **kwargs): raise AssertionError("lift used native frontend fallback")
    model.frontend.forward = forbidden
    try:
        full = lift.full(raw)
        repair = lift.repair(raw)
    finally:
        model.frontend.forward = original
    assert torch.allclose(full, native, atol=1e-5, rtol=1e-5)
    assert repair.shape == (batch, 5, native.shape[-1])
    assert torch.allclose(repair, native[:, [0, 1, 2, 3, 49]], atol=1e-5, rtol=1e-5)
