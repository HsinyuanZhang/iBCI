import numpy as np
import pytest
import torch

from tfpd_exploration.src.family_runtime_v1.m2_family_causal import M2FamilyCausalRuntime, M2RuntimeBank
from tfpd_exploration.src.m2_family_v1.decoder import make_paired_decoders


def fixture(arm, batch):
    model = make_paired_decoders(42)[arm == "route"].eval()
    with torch.no_grad():
        if arm == "route":
            model.frontend.attn.routing.g.fill_(.2)
    torch.manual_seed(939)
    bank = M2RuntimeBank(torch.randn(batch, 96, 50) * .2,
                         torch.randn(batch, 96, 4) * .1, torch.rand(batch, 96) > .1)
    return model, bank


@torch.no_grad()
def direct(model, bank, raw):
    return (model.forward_last(raw, bank, bank.unit_mask) / 5).numpy()


@pytest.mark.parametrize("arm,batch", [("flat", 1), ("route", 1), ("route", 7)])
def test_actual_family_stream_rollover_mutation_and_inactive_padding(arm, batch):
    torch.set_num_threads(1)
    model, bank = fixture(arm, batch)
    runtime = M2FamilyCausalRuntime(model, bank, batch_size=batch)
    raw = torch.zeros(batch, 50, 96)
    source = np.random.default_rng(940).poisson(.3, (53, batch, 96)).astype(np.float32)
    for index, value in enumerate(source):
        raw = torch.cat((raw[:, 1:], torch.from_numpy(value)[:, None]), dim=1)
        got = runtime.predict(value)
        assert got.flags.owndata and got.flags.c_contiguous
        assert torch.equal(runtime.raw, raw)
        if index in (0, 1, 3, 4, 48, 49, 50, 52):
            np.testing.assert_allclose(got, direct(model, bank, raw), atol=1e-5, rtol=1e-5)
        if index == 20:
            with torch.no_grad():
                bank.E0.add_(.01)
                bank.T.mul_(.9)
                bank.unit_mask[:, 0].logical_not_()
                model.frontend.attn.in_proj_bias.add_(.001)
                model.temporal.pe.add_(.0001)
                if arm == "route":
                    model.frontend.attn.routing.g.add_(.1)
            runtime._audit()
            torch.testing.assert_close(runtime.frontend, runtime._native_frontend(raw, bank), atol=1e-5, rtol=1e-5)
    partial = M2RuntimeBank(bank.E0[:1].clone(), bank.T[:1].clone(), bank.unit_mask[:1].clone())
    runtime.reset(partial, batch=1)
    runtime.observe(source[0, :1])
    retained = runtime.raw.clone()
    runtime.on_done(np.ones(1, dtype=bool))
    assert torch.equal(retained, runtime.raw)
    got = runtime.predict(source[1, :1])
    np.testing.assert_allclose(got[:1], direct(model, partial, runtime.raw), atol=1e-5, rtol=1e-5)
    if batch > 1:
        assert np.all(got[1:] == 0)
    assert runtime.state_bytes()["raw_and_frontend"] == 50 * (96 + 256) * 4


def test_rejected_inputs_reset_and_derived_mutation_do_not_advance():
    torch.set_num_threads(1)
    model, bank = fixture("route", 1)
    runtime = M2FamilyCausalRuntime(model, bank)
    raw = runtime.raw.clone()
    for value in (np.full((1, 96), np.nan, np.float32), np.zeros((1, 96), np.uint8)):
        with pytest.raises(ValueError):
            runtime.predict(value)
        assert torch.equal(raw, runtime.raw)
    invalid = M2RuntimeBank(bank.E0, bank.T, torch.zeros_like(bank.unit_mask))
    with pytest.raises(ValueError, match="at least one"):
        runtime.reset(invalid)
    assert runtime.bank is bank and torch.equal(raw, runtime.raw)
    runtime.spatial.derived_tensors()[0].add_(.01)
    with pytest.raises(RuntimeError, match="derived"):
        runtime.predict(np.zeros((1, 96), np.float32))
    assert torch.equal(raw, runtime.raw)


def test_last_query_does_not_run_full_final_block(monkeypatch):
    torch.set_num_threads(1)
    model, bank = fixture("flat", 1)
    runtime = M2FamilyCausalRuntime(model, bank)
    def forbidden(*args, **kwargs):
        raise AssertionError("unused final full block evaluated")
    monkeypatch.setattr(model.temporal.blocks[-1], "forward", forbidden)
    runtime.predict(np.zeros((1, 96), np.float32))
    model.train()
    with pytest.raises(RuntimeError, match="eval"):
        runtime.predict(np.zeros((1, 96), np.float32))
