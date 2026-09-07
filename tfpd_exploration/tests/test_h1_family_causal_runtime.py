import copy

import numpy as np
import pytest
import torch

from tfpd_exploration.src.family_runtime_v1.h1_causal import H1CausalRuntime, W
from tfpd_exploration.src.h1_family_v1.model import make_v2_unscaled_dot_localbalanced_pair
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank


def make_model_bank(arm, batch):
    flat, route = make_v2_unscaled_dot_localbalanced_pair(seed=42)
    model = flat if arm == "flat" else route
    model.eval()
    with torch.no_grad():
        if arm == "route":
            model.frontend.attn.g.fill_(.2)
    torch.manual_seed(912)
    bank = H1Bank(torch.randn(batch, 176, 700) * .2,
                  torch.randn(batch, 176, 4) * .1,
                  torch.rand(batch, 176) > .1)
    return model, bank


def direct(model, bank, raw):
    with torch.no_grad():
        return (model.forward_last(raw, bank) / 20).numpy()


@pytest.mark.parametrize("arm,batch", [("flat", 1), ("route", 1), ("route", 2)])
def test_actual_w700_startup_mutation_refresh_and_reset(arm, batch):
    torch.set_num_threads(1)
    model, bank = make_model_bank(arm, batch)
    engine = H1CausalRuntime(model, bank, batch_size=batch)
    raw = torch.zeros(batch, W, 176)
    source = np.random.default_rng(93).poisson(.3, (8, batch, 176)).astype(np.float32)
    for index, value in enumerate(source):
        raw = torch.cat((raw[:, 1:], torch.from_numpy(value)[:, None]), dim=1)
        if index == 2:
            engine.observe(value)
        else:
            got = engine.predict(value)
            np.testing.assert_allclose(got, direct(model, bank, raw), atol=1e-5, rtol=1e-5)
            assert got.flags.owndata and got.flags.c_contiguous
        assert torch.equal(raw, engine.raw)
        retained = engine.raw.clone()
        engine.on_done(np.ones(batch, dtype=bool))
        assert torch.equal(retained, engine.raw)
    with torch.no_grad():
        bank.E0.add_(.03)
        bank.T.add_(.01)
        bank.unit_mask[:, 0].logical_not_()
        model.frontend.attn.k_proj.weight.add_(.0001)
        if arm == "route":
            model.frontend.attn.g.add_(.1)
    engine._audit()
    torch.testing.assert_close(engine.frontend, model.encode_frontend(raw, bank), atol=1e-5, rtol=1e-5)
    raw = torch.cat((raw[:, 1:], torch.from_numpy(source[0])[:, None]), dim=1)
    np.testing.assert_allclose(engine.predict(source[0]), direct(model, bank, raw), atol=1e-5, rtol=1e-5)
    new_bank = H1Bank(bank.E0[:1].clone(), bank.T[:1].clone(), bank.unit_mask[:1].clone())
    engine.reset(new_bank, batch=1)
    raw = torch.zeros(1, W, 176); raw[:, -1] = torch.from_numpy(source[0, :1])
    got = engine.predict(source[0, :1])
    np.testing.assert_allclose(got[:1], direct(model, new_bank, raw), atol=1e-5, rtol=1e-5)
    if batch > 1:
        assert np.all(got[1:] == 0)
    assert engine.state_bytes().raw_and_frontend == (W * (176 + 256) * 4)


def test_actual_w700_full_rollover_nonzero_route():
    torch.set_num_threads(1)
    model, bank = make_model_bank("route", 1)
    engine = H1CausalRuntime(model, bank, batch_size=1)
    raw = torch.zeros(1, W, 176)
    source = np.random.default_rng(94).poisson(.3, (W + 3, 1, 176)).astype(np.float32)
    for index, value in enumerate(source):
        raw = torch.cat((raw[:, 1:], torch.from_numpy(value)[:, None]), dim=1)
        got = engine.predict(value)
        assert torch.equal(engine.raw, raw)
        assert np.isfinite(got).all()
        if index in (0, 1, 3, 4, 698, 699, 700, 702):
            np.testing.assert_allclose(got, direct(model, bank, raw), atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(engine.frontend, model.encode_frontend(raw, bank), atol=1e-5, rtol=1e-5)


def test_h1_temporal_last_query_does_not_call_final_block_forward(monkeypatch):
    torch.set_num_threads(1)
    model, bank = make_model_bank("flat", 1)
    engine = H1CausalRuntime(model, bank)
    def forbidden(*args, **kwargs):
        raise AssertionError("final full-block forward called")
    monkeypatch.setattr(model.temporal.blocks[-1], "forward", forbidden)
    engine.predict(np.zeros((1, 176), dtype=np.float32))
    before = engine.raw.clone()
    with pytest.raises(ValueError):
        engine.predict(np.full((1, 176), np.nan, dtype=np.float32))
    assert torch.equal(before, engine.raw)
    model.train()
    with pytest.raises(RuntimeError, match="eval"):
        engine.predict(np.zeros((1, 176), dtype=np.float32))
    model.eval()
    with torch.no_grad():
        engine.spatial.qwk.add_(1)
    with pytest.raises(RuntimeError, match="derived"):
        engine.predict(np.zeros((1, 176), dtype=np.float32))
