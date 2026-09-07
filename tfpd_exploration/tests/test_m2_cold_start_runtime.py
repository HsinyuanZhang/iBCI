import numpy as np
import pytest
import torch

from tfpd_exploration.src.family_runtime_v1.m2_cold_start import ColdStartLinearConvM2Decoder
from tfpd_exploration.src.family_runtime_v1.m2_linear_conv import LinearConvFiveTokenM2Decoder, _LinearConvFiveTokenExactE
from tfpd_exploration.tests.test_m2_grouped_value_runtime import P, oracle
from tfpd_exploration.tests.test_m2_runtime_v3 import TAGS


@pytest.mark.parametrize("batch", [1, 7])
def test_cold_start_full_zero_frontend_and_every_public_bin_through_rollover(batch):
    torch.set_num_threads(1)
    reference = LinearConvFiveTokenM2Decoder(P, batch_size=batch)
    candidate = ColdStartLinearConvM2Decoder(P, batch_size=batch)
    reference.reset(TAGS[:batch]); candidate.reset(TAGS[:batch])
    torch.testing.assert_close(candidate._engine.frontend, reference._engine.frontend, atol=1e-5, rtol=1e-5)
    assert torch.equal(candidate._engine.raw, reference._engine.raw)
    assert candidate.state_bytes() == reference.state_bytes()
    data = np.random.default_rng(620).poisson(.3, (57, batch, 96)).astype("float32")
    with torch.no_grad():
        for index, value in enumerate(data):
            got, expected = candidate.predict(value), reference.predict(value)
            np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)
            if index in (0, 1, 3, 4, 49, 50, 56):
                np.testing.assert_allclose(got, oracle(candidate), atol=1e-5, rtol=1e-5)
        candidate._engine.bank.E0.add_(.01)
        candidate.model.frontend.local_conv.conv.bias.add_(.03)
        candidate._engine.bank.unit_mask[:, 0].logical_not_()
        got = candidate.predict(data[0])
        np.testing.assert_allclose(got, oracle(candidate), atol=1e-5, rtol=1e-5)
        candidate.reset(TAGS[-1:])
        got = candidate.predict(data[0, :1])
        np.testing.assert_allclose(got[:1], oracle(candidate), atol=1e-5, rtol=1e-5)
        if batch > 1:
            assert np.all(got[1:] == 0)


def test_cold_reset_never_evaluates_discarded_temporal_result(monkeypatch):
    torch.set_num_threads(1)
    candidate = ColdStartLinearConvM2Decoder(P, batch_size=7)
    def forbidden(*args, **kwargs):
        raise AssertionError("reset called temporal forward")
    monkeypatch.setattr(_LinearConvFiveTokenExactE, "_last", forbidden)
    candidate.reset(TAGS[:7])
    assert candidate._engine.frontend.shape == (7, 50, 256)
    with pytest.raises(Exception, match="unknown"):
        candidate.reset(["not_a_frozen_tag"])
