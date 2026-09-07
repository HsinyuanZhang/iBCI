import numpy as np
import pytest
import torch

from tfpd_exploration.src.family_runtime_v1.m2_family_causal import M2RuntimeBank
from tfpd_exploration.src.family_runtime_v1.m2_family_queryage import M2FamilyQueryAgeRuntime
from tfpd_exploration.src.m2_family_v1.decoder import make_paired_decoders
from tfpd_exploration.src.m2_queryage_family_v1.model import make_paired_queryage_decoders


def fixture(arm: str, batch: int, *, heterogeneous: bool):
    model = make_paired_queryage_decoders(42)[arm == "route"].eval()
    with torch.no_grad():
        if arm == "route":
            model.frontend.attn.routing.g.fill_(.2)
    generator = torch.Generator().manual_seed(117 + batch)
    if heterogeneous:
        e0 = torch.randn(batch, 96, 50, generator=generator) * .2
        functional = torch.randn(batch, 96, 4, generator=generator) * .1
        mask = torch.rand(batch, 96, generator=generator) > .15
    else:
        e0 = torch.randn(96, 50, generator=generator) * .2
        functional = torch.randn(96, 4, generator=generator) * .1
        mask = torch.rand(96, generator=generator) > .15
    return model, M2RuntimeBank(e0, functional, mask)


@torch.no_grad()
def direct(model, bank, raw):
    return (model.forward_last(raw, bank, bank.unit_mask) / 5).numpy()


@pytest.mark.parametrize("arm,batch,heterogeneous", [("flat", 1, False), ("route", 1, True), ("route", 7, True)])
def test_queryage_fullwindow_native_equivalence_startup_wrap_and_ownership(arm, batch, heterogeneous):
    torch.set_num_threads(1)
    model, bank = fixture(arm, batch, heterogeneous=heterogeneous)
    runtime = M2FamilyQueryAgeRuntime(model, bank, batch_size=batch)
    raw = torch.zeros(batch, 50, 96)
    values = np.random.default_rng(118).poisson(.25, (110, batch, 96)).astype(np.float32)
    for step, value in enumerate(values):
        raw = torch.cat((raw[:, 1:], torch.from_numpy(value)[:, None]), dim=1)
        got = runtime.predict(value)
        assert got.flags.owndata and got.flags.c_contiguous and got.shape == (batch, 2)
        if step in (0, 1, 49, 50, 59, 79, 109):
            np.testing.assert_allclose(got, direct(model, bank, raw), atol=1e-5, rtol=1e-5)
    # Caller cannot alias a returned public prediction.
    got[0, 0] = 99
    assert runtime.predict(values[-1])[0, 0] != 99
    assert runtime.state_bytes()["raw_and_frontend"] == 50 * (96 + 256) * 4 * batch


def test_partial_active_done_reset_mutation_and_invalid_observation():
    torch.set_num_threads(1)
    model, bank = fixture("route", 7, heterogeneous=True)
    runtime = M2FamilyQueryAgeRuntime(model, bank, batch_size=7)
    partial = M2RuntimeBank(bank.E0[:1].clone(), bank.T[:1].clone(), bank.unit_mask[:1].clone())
    runtime.reset(partial, batch=1)
    before = runtime.raw.clone()
    runtime.on_done(np.array([True]))
    assert torch.equal(before, runtime.raw)
    value = np.ones((1, 96), dtype=np.float32)
    got = runtime.predict(value)
    assert np.all(got[1:] == 0)
    for invalid in (np.full((1, 96), np.nan, dtype=np.float32), np.zeros((1, 96), dtype=np.uint8), np.zeros((2, 96), dtype=np.float32)):
        retained = runtime.raw.clone()
        with pytest.raises(ValueError):
            runtime.predict(invalid)
        assert torch.equal(retained, runtime.raw)
    with torch.no_grad():
        partial.E0.add_(.01); partial.T.mul_(.9); partial.unit_mask[:, 0].logical_not_()
        model.frontend.attn.in_proj_bias.add_(.001); model.frontend.attn.routing.g.add_(.1)
    runtime._audit()
    torch.testing.assert_close(runtime.frontend, runtime._native_frontend(runtime.raw, partial), atol=1e-5, rtol=1e-5)
    runtime.spatial.derived_tensors()[0].add_(.01)
    with pytest.raises(RuntimeError, match="derived"):
        runtime.predict(value)


def test_rejects_old_causal_model_and_training_mode():
    torch.set_num_threads(1)
    old = make_paired_decoders(42)[0].eval()
    _new, bank = fixture("flat", 1, heterogeneous=False)
    with pytest.raises(TypeError, match="QueryAge"):
        M2FamilyQueryAgeRuntime(old, bank)
    model, bank = fixture("flat", 1, heterogeneous=False)
    runtime = M2FamilyQueryAgeRuntime(model, bank)
    model.train()
    with pytest.raises(RuntimeError, match="eval"):
        runtime.predict(np.zeros((1, 96), dtype=np.float32))
