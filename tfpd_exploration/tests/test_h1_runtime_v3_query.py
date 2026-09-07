"""Exact static carrier caching remains tied to live bank/model state."""
from __future__ import annotations

import numpy as np
import torch

from tfpd_exploration.src.h1_optimized_v6.model import make_matched_pair
from tfpd_exploration.src.h1_runtime_v3 import StaticSignedQueryStream
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank


def fixture():
    torch.set_num_threads(1)
    model = make_matched_pair()[1].eval()
    g = torch.Generator().manual_seed(4321)
    bank = H1Bank(torch.randn(176, 700, generator=g), torch.randn(176, 4, generator=g),
                  torch.ones(176, dtype=torch.bool))
    engine = StaticSignedQueryStream(model, bank, task="h1", session_id="source", unit_ids=range(176))
    return model, bank, engine


def oracle(model, bank, raw):
    with torch.no_grad():
        return (model.forward_last(raw, bank)/20.).numpy()


def test_startup_rollover_and_static_memory_accounting():
    model, bank, engine = fixture()
    g = torch.Generator().manual_seed(11)
    raw = torch.zeros(1, 700, 176)
    with torch.no_grad():
        np.testing.assert_allclose(engine.current_prediction().numpy(), oracle(model, bank, raw), atol=1e-5, rtol=1e-5)
        for step in range(12):
            x = torch.randn(1, 176, generator=g)
            raw = torch.cat((raw[:, 1:], x[:, None]), dim=1)
            np.testing.assert_allclose(engine.predict(x.numpy()), oracle(model, bank, raw), atol=1e-5, rtol=1e-5)
        raw = torch.randn(1, 700, 176, generator=g)
        engine.reset(history=raw)
        for step in range(6):
            x = torch.randn(1, 176, generator=g)
            raw = torch.cat((raw[:, 1:], x[:, None]), dim=1)
            np.testing.assert_allclose(engine.predict(x.numpy()), oracle(model, bank, raw), atol=1e-5, rtol=1e-5)
    assert engine.static_carrier_bytes == (176*256+176)*4
    assert engine.state_bytes > engine.static_carrier_bytes


def test_numeric_mask_bank_and_model_mutations_rebuild_static_weights():
    model, bank, engine = fixture()
    raw = torch.randn(1, 700, 176, generator=torch.Generator().manual_seed(13))
    engine.reset(history=raw)
    previous = engine.static_rebuild_count
    with torch.no_grad():
        for change in (lambda: bank.unit_mask.__setitem__(0, False),
                       lambda: bank.T[1].add_(0.5),
                       lambda: model.frontend.phi.weight.add_(0.001)):
            change()
            got = engine.current_prediction().numpy()
            np.testing.assert_allclose(got, oracle(model, bank, raw), atol=1e-5, rtol=1e-5)
            assert engine.static_rebuild_count == previous + 1
            previous += 1
