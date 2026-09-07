import numpy as np
import pytest
import torch

from tfpd_exploration.src.family_runtime_v1.linear_conv import repaired_local_features_linear
from tfpd_exploration.src.family_runtime_v1.m2_grouped import GroupedValueFiveTokenM2Decoder
from tfpd_exploration.src.family_runtime_v1.m2_linear_conv import LinearConvFiveTokenM2Decoder
from tfpd_exploration.src.family_runtime_v1.repair import repaired_local_features
from tfpd_exploration.tests.test_m2_grouped_value_runtime import P, oracle
from tfpd_exploration.tests.test_m2_runtime_v3 import TAGS


@pytest.mark.parametrize("batch,width", [(1, 9), (7, 50), (4, 100)])
def test_linear_local_repair_matches_full_conv_and_stitched(batch, width):
    torch.set_num_threads(1)
    model = LinearConvFiveTokenM2Decoder(P, batch_size=1).model
    conv = model.frontend.local_conv
    values = torch.randn(batch, width, 96, generator=torch.Generator().manual_seed(810))
    with torch.no_grad():
        actual = repaired_local_features_linear(conv, values)
        full = conv(values)
        expected = torch.cat((full[:, :4], full[:, -1:]), dim=1)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(actual, repaired_local_features(conv, values), atol=1e-5, rtol=1e-5)
    conv.conv.stride = (2,)
    with pytest.raises(ValueError, match="stride"):
        repaired_local_features_linear(conv, values)


@pytest.mark.parametrize("batch", [1, 7])
def test_linear_public_startup_boundary_reset_mutation_and_owning_output(batch):
    torch.set_num_threads(1)
    old = GroupedValueFiveTokenM2Decoder(P, batch_size=batch)
    candidate = LinearConvFiveTokenM2Decoder(P, batch_size=batch)
    old.reset(TAGS[:batch]); candidate.reset(TAGS[:batch])
    values = np.random.default_rng(7102).poisson(.3, (56, batch, 96)).astype("float32")
    with torch.no_grad():
        for index, value in enumerate(values):
            got = candidate.predict(value)
            expected = old.predict(value)
            np.testing.assert_allclose(got, expected, atol=1e-5, rtol=1e-5)
            assert got.flags.owndata and got.dtype == np.float32 and got.shape == (batch, 2)
            if index in (0, 1, 3, 4, 49, 50, 55):
                np.testing.assert_allclose(got, oracle(candidate), atol=1e-5, rtol=1e-5)
        assert candidate.state_bytes() == old.state_bytes()
        candidate.model.frontend.local_conv.conv.weight.add_(.02)
        candidate.model.frontend.local_conv.conv.bias.add_(.01)
        candidate.model.frontend.mha.in_proj_weight[:512].add_(.001)
        got = candidate.predict(values[0])
        np.testing.assert_allclose(got, oracle(candidate), atol=1e-5, rtol=1e-5)
        candidate._engine.bank.T.add_(.02)
        candidate._engine.bank.unit_mask[:, 0].logical_not_()
        got = candidate.predict(values[1])
        np.testing.assert_allclose(got, oracle(candidate), atol=1e-5, rtol=1e-5)
        candidate.reset(TAGS[:1])
        got = candidate.predict(values[0, :1])
        assert got.shape == (batch, 2)
        np.testing.assert_allclose(got[:1], oracle(candidate), atol=1e-5, rtol=1e-5)
        if batch > 1:
            assert np.all(got[1:] == 0)
