import torch

from tfpd_exploration.src.h1_optimized_v2.model import H1CurrentQueryDecoder, make_matched_pair
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank


def test_activity_scale_is_positive_and_unit_permutation_invariant():
    torch.manual_seed(3)
    model = H1CurrentQueryDecoder(activity_scale=32.0).eval()
    x = torch.randn(1, 700, 176)
    bank = H1Bank(torch.randn(176, 700), torch.randn(176, 4), torch.ones(176, dtype=torch.bool))
    order = torch.randperm(176)
    with torch.no_grad():
        left = model.forward_last(x, bank)
        right = model.forward_last(x[:, :, order], H1Bank(bank.E0[order], bank.T[order], bank.unit_mask[order]))
    torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-5)


def test_activity_scale_rejects_nonpositive_value():
    for value in (0.0, -1.0):
        try:
            H1CurrentQueryDecoder(activity_scale=value)
        except ValueError:
            pass
        else:
            raise AssertionError("nonpositive activity scale must be rejected")


def test_matched_pair_shares_frontend_and_readout_initialization():
    full, query = make_matched_pair(activity_scale=32.0)
    for left, right in ((full.frontend, query.frontend), (full.final_norm, query.final_norm), (full.readout, query.readout)):
        for key, value in left.state_dict().items():
            torch.testing.assert_close(value, right.state_dict()[key], rtol=0, atol=0)
    for block in query.temporal.blocks:
        torch.testing.assert_close(block.age_bias, torch.zeros_like(block.age_bias), rtol=0, atol=0)
