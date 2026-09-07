"""Exact five-token repair, including public API and lifecycle boundaries."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tfpd_exploration.src.family_runtime_v1 import FiveTokenCurrentQueryStream, FiveTokenM2Decoder
from tfpd_exploration.src.family_runtime_v1.repair import repaired_local_features
from tfpd_exploration.tests.test_m1_runtime_v3 import _fixture, _reference_last
from tfpd_exploration.tests.test_m2_runtime_v3 import TAGS
from tfpd_exploration.src.m1_runtime_v3 import HeterogeneousCurrentQueryStream
from tfpd_exploration.src.m2_runtime_v3.runtime import RuntimeV3Decoder
from tfpd_exploration.src.family_runtime_v1.m1_lifted import LiftedFiveTokenCurrentQueryStream


M2_PAYLOAD = "tfpd_exploration/submissions/evalai_m2_small_trf_ext6_epochpick_v1/artifacts/m2_small_trf_s42_ema_e08_ext6.pkl"


@pytest.mark.parametrize("batch", [1, 4])
def test_m1_five_token_exact_full_window_and_inactive_lanes(batch):
    torch.set_num_threads(1)
    package, model, bank = _fixture(batch)
    reference = HeterogeneousCurrentQueryStream(model, bank)
    candidate = FiveTokenCurrentQueryStream(model, bank)
    values = torch.randn(112, batch, 64, generator=torch.Generator().manual_seed(9502))
    with torch.no_grad():
        for index, value in enumerate(values):
            active = max(1, batch - 1) if index in (2, 3, 99, 100) else batch
            frozen = candidate.raw[active:].clone()
            actual = candidate.predict(value[:active].numpy(), active=active)
            expected = reference.predict(value[:active].numpy(), active=active)
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
            assert torch.equal(candidate.raw[active:], frozen)
            if index in (0, 2, 3, 4, 98, 99, 100, 111):
                full = _reference_last(package, model, candidate, bank)[:active].numpy()
                np.testing.assert_allclose(actual, full, atol=1e-5, rtol=1e-5)
        raw = candidate.raw.clone()
        candidate.on_done([True] * batch)
        assert torch.equal(raw, candidate.raw)
        assert candidate.state_bytes == reference.state_bytes
        bank.unit_mask[0, 0].logical_not_()
        with pytest.raises(RuntimeError, match="fail closed"):
            candidate.predict(values[0].numpy())
        candidate.refresh_state()
        assert np.isfinite(candidate.predict(values[0].numpy())).all()


def test_repair_local_exact_matches_selected_full_convolution():
    torch.set_num_threads(1)
    _, model, _ = _fixture(4)
    raw = torch.randn(4, 100, 64, generator=torch.Generator().manual_seed(9503))
    full = model.frontend.local_conv(raw)[:, (0, 1, 2, 3, 99)]
    got = repaired_local_features(model.frontend.local_conv, raw)
    torch.testing.assert_close(got, full, atol=1e-6, rtol=1e-6)


def test_lifted_attention_m1_public_startup_and_window_wrap():
    torch.set_num_threads(1)
    package, model, bank = _fixture(4)
    candidate = LiftedFiveTokenCurrentQueryStream(model, bank)
    baseline = FiveTokenCurrentQueryStream(model, bank)
    rng = np.random.default_rng(9831)
    with torch.no_grad():
        for step in range(120):
            active = 2 if step in (1, 4, 99, 101) else 4
            value = rng.poisson(.4, size=(active, 64)).astype(np.float32)
            actual = candidate.predict(value, active=active)
            expected = baseline.predict(value, active=active)
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
            if step in (0, 4, 99, 101, 119):
                full = _reference_last(package, model, candidate, bank)[:active].numpy()
                np.testing.assert_allclose(actual, full, rtol=1e-5, atol=1e-5)
        assert candidate.state_bytes > baseline.state_bytes
        bank.E0.add_(.01)
        with pytest.raises(RuntimeError, match="fail closed"):
            candidate.predict(value)
        candidate.refresh_state()
        baseline.refresh_state()
        np.testing.assert_allclose(candidate.predict(value), baseline.predict(value), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("batch", [1, 7])
def test_m2_current_e8_five_token_exact_boundary_and_mutation(batch):
    torch.set_num_threads(1)
    reference = RuntimeV3Decoder(M2_PAYLOAD, batch_size=batch)
    candidate = FiveTokenM2Decoder(M2_PAYLOAD, batch_size=batch)
    reference.reset(TAGS[:batch])
    candidate.reset(TAGS[:batch])
    values = np.random.default_rng(9504).poisson(.3, size=(62, batch, 96)).astype(np.float32)
    with torch.no_grad():
        for index, value in enumerate(values):
            if index == 52:
                for decoder in (reference, candidate):
                    decoder._engine.bank.unit_mask[0, 0].logical_not_()
            actual = candidate.predict(value)
            expected = reference.predict(value)
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
            if index in (0, 2, 3, 4, 48, 49, 50, 52, 61):
                engine = candidate._engine
                full = candidate.model.forward_last(engine.raw, engine.bank, engine.bank.unit_mask).numpy() / 5
                np.testing.assert_allclose(actual, full, atol=1e-5, rtol=1e-5)
        assert candidate.state_bytes() == reference.state_bytes()
        candidate.reset(TAGS[:1])
        assert candidate.predict(values[0, :1]).shape == (batch, 2)
