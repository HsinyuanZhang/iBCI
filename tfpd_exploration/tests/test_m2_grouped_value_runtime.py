import numpy as np
import pytest
import torch

from tfpd_exploration.src.family_runtime_v1.grouped_value import grouped_value_projection
from tfpd_exploration.src.family_runtime_v1.m2_grouped import GroupedValueFiveTokenM2Decoder
from tfpd_exploration.src.family_runtime_v1.m2_lifted import LiftedFiveTokenM2Decoder
from tfpd_exploration.tests.test_m2_runtime_v3 import TAGS

P = "tfpd_exploration/submissions/evalai_m2_small_trf_ext6_epochpick_v1/artifacts/m2_small_trf_s42_ema_e08_ext6.pkl"


def oracle(decoder):
    engine = decoder._engine
    return decoder.model.forward_last(engine.raw, engine.bank, engine.bank.unit_mask).numpy() / 5.0


def test_grouped_value_projection_matches_broadcast_matmul_fp32():
    generator = torch.Generator().manual_seed(91)
    attended = torch.randn(7, 8, 8, 256, generator=generator)
    weight = torch.randn(8, 256, 32, generator=generator)
    expected = torch.matmul(attended, weight.unsqueeze(0))
    actual = grouped_value_projection(attended, weight)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("batch", [1, 7])
def test_grouped_decoder_b1_b7_startup_w50_reset_direct_and_lifted_parity(batch):
    torch.set_num_threads(1)
    old = LiftedFiveTokenM2Decoder(P, batch_size=batch)
    grouped = GroupedValueFiveTokenM2Decoder(P, batch_size=batch)
    old.reset(TAGS[:batch])
    grouped.reset(TAGS[:batch])
    values = np.random.default_rng(9261).poisson(.3, (55, batch, 96)).astype("float32")
    for index, value in enumerate(values):
        old_value = old.predict(value)
        grouped_output = grouped.predict(value)
        np.testing.assert_allclose(grouped_output, old_value, atol=1e-5, rtol=1e-5)
        if index in (0, 4, 49, 50, 54):
            np.testing.assert_allclose(grouped_output[:batch], oracle(grouped), atol=1e-5, rtol=1e-5)
    grouped.reset(TAGS[:1])
    assert grouped.predict(values[0, :1]).shape == (batch, 2)


def test_grouped_decoder_mutation_refresh_derived_guard_and_state_bytes():
    old = LiftedFiveTokenM2Decoder(P, batch_size=1)
    grouped = GroupedValueFiveTokenM2Decoder(P, batch_size=1)
    old.reset(TAGS[:1])
    grouped.reset(TAGS[:1])
    value = np.zeros((1, 96), dtype=np.float32)
    old.predict(value)
    grouped.predict(value)
    assert grouped.state_bytes() == old.state_bytes()
    with torch.no_grad():
        grouped.model.frontend.mha.in_proj_weight[:512].add_(.001)
        grouped.model.frontend.slots.add_(.002)
        grouped.model.frontend.slot_norm.weight.mul_(.999)
    got = grouped.predict(value)
    np.testing.assert_allclose(got, oracle(grouped), atol=1e-5, rtol=1e-5)
    grouped._engine.bank.E0.add_(.01)
    grouped._engine.bank.unit_mask[0, 0].logical_not_()
    got = grouped.predict(value)
    np.testing.assert_allclose(got, oracle(grouped), atol=1e-5, rtol=1e-5)
    grouped._engine._lifted.qwk.add_(.01)
    with pytest.raises(Exception, match="derived"):
        grouped.predict(value)
