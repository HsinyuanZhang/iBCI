"""CPU synthetic parity contracts for the opt-in static-affine M2 adapter."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tfpd_exploration.src.family_runtime_v1.m2_family_causal import M2FamilyCausalRuntime, M2RuntimeBank
from tfpd_exploration.src.family_runtime_v1.m2_family_static_affine import (
    EQUIVALENCE_ATOL, EQUIVALENCE_RTOL, STATUS, M2FamilyStaticAffineRuntime,
)
from tfpd_exploration.src.m2_family_v1.decoder import make_paired_decoders


def fixture(arm: str, batch: int, per_lane: bool):
    model = make_paired_decoders(42)[arm == "route"].eval()
    with torch.no_grad():
        if arm == "route":
            model.frontend.attn.routing.g.fill_(.2)
    generator = torch.Generator().manual_seed(810 + batch + per_lane)
    leading = batch if per_lane else 1
    e0 = torch.randn(leading, 96, 50, generator=generator) * .2
    t4 = torch.randn(leading, 96, 4, generator=generator) * .1
    mask = torch.rand(leading, 96, generator=generator) > .1
    if not per_lane:
        e0, t4, mask = e0[0], t4[0], mask[0]
    return model, M2RuntimeBank(e0, t4, mask)


@torch.no_grad()
@pytest.mark.parametrize("arm,batch,per_lane", [("flat", 1, False), ("route", 1, False), ("flat", 7, True), ("route", 7, True)])
def test_startup_rollover_and_full_frontend_parity(arm, batch, per_lane):
    torch.set_num_threads(1)
    model, bank = fixture(arm, batch, per_lane)
    baseline = M2FamilyCausalRuntime(model, bank, batch_size=batch)
    experimental = M2FamilyStaticAffineRuntime(model, bank, batch_size=batch)
    source = np.random.default_rng(811).poisson(.3, (53, batch, 96)).astype(np.float32)
    for tick, value in enumerate(source):
        expected = baseline.predict(value)
        actual = experimental.predict(value)
        assert actual.flags.owndata and actual.flags.c_contiguous and actual.dtype == np.float32
        assert torch.equal(experimental.raw, baseline.raw)
        np.testing.assert_allclose(actual, expected, atol=EQUIVALENCE_ATOL, rtol=EQUIVALENCE_RTOL)
        np.testing.assert_allclose(experimental.frontend.numpy(), baseline.frontend.numpy(), atol=EQUIVALENCE_ATOL, rtol=EQUIVALENCE_RTOL)
        if tick in (0, 1, 3, 4, 48, 49, 50, 52):
            direct = (model.forward_last(experimental.raw, bank, bank.unit_mask) / 5).numpy()
            np.testing.assert_allclose(actual, direct, atol=EQUIVALENCE_ATOL, rtol=EQUIVALENCE_RTOL)


@torch.no_grad()
@pytest.mark.parametrize("per_lane", [False, True])
def test_b7_reset_reduced_active_batch_padding_and_on_done(per_lane):
    torch.set_num_threads(1)
    model, bank = fixture("route", 7, per_lane)
    runtime = M2FamilyStaticAffineRuntime(model, bank, batch_size=7)
    partial = M2RuntimeBank(bank.E0[:1].clone() if bank.E0.ndim == 3 else bank.E0.clone(),
                            bank.T[:1].clone() if bank.T.ndim == 3 else bank.T.clone(),
                            bank.unit_mask[:1].clone() if bank.unit_mask.ndim == 2 else bank.unit_mask.clone())
    runtime.reset(partial, batch=1)
    value = np.ones((1, 96), np.float32)
    output = runtime.predict(value)
    assert output.shape == (7, 2) and np.all(output[1:] == 0)
    retained = runtime.raw.clone(); runtime.on_done(np.ones(1, dtype=bool))
    assert torch.equal(retained, runtime.raw)
    assert runtime.static_affine_bytes == runtime.spatial.static_affine.numel() * 4
    assert runtime.state_bytes()["derived_static"] >= runtime.static_affine_bytes


@torch.no_grad()
def test_bank_model_route_and_derived_static_mutations_refresh_or_refuse():
    torch.set_num_threads(1)
    model, bank = fixture("route", 7, True)
    runtime = M2FamilyStaticAffineRuntime(model, bank, batch_size=7)
    value = np.ones((7, 96), np.float32)
    runtime.predict(value)
    mutations = (
        lambda: bank.E0.add_(.01), lambda: bank.T.mul_(.9),
        lambda: bank.unit_mask[:, 0].logical_not_(),
        lambda: model.frontend.token_mlp[0].weight.add_(.001),
        lambda: model.frontend.token_mlp[0].bias.add_(.001),
        lambda: model.frontend.attn.routing.g.add_(.1),
    )
    for mutate in mutations:
        mutate()
        output = runtime.predict(value)
        direct = (model.forward_last(runtime.raw, bank, bank.unit_mask) / 5).numpy()
        np.testing.assert_allclose(output, direct, atol=EQUIVALENCE_ATOL, rtol=EQUIVALENCE_RTOL)
    raw = runtime.raw.clone()
    runtime.spatial.static_affine.add_(.01)
    with pytest.raises(RuntimeError, match="derived experimental"):
        runtime.predict(value)
    assert torch.equal(raw, runtime.raw)


def test_input_errors_metadata_and_static_state_accounting():
    torch.set_num_threads(1)
    model, bank = fixture("flat", 1, False)
    runtime = M2FamilyStaticAffineRuntime(model, bank)
    assert runtime.status == STATUS == "EXPERIMENTAL_NOT_PROMOTED"
    assert runtime.equivalence["kind"] == "real_arithmetic_equivalent_fp32_tolerance_not_bit_exact"
    initial = runtime.raw.clone()
    for bad in (np.zeros((1, 96), np.float64), np.full((1, 96), np.nan, np.float32), np.zeros((2, 96), np.float32)):
        with pytest.raises(ValueError):
            runtime.predict(bad)
        assert torch.equal(initial, runtime.raw)
    sizes = runtime.state_bytes()
    assert runtime.static_affine_bytes > 0
    assert sizes["derived_static"] >= runtime.static_affine_bytes
