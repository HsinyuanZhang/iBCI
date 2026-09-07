import pytest
import torch
from tfpd_exploration.src.h1_family_v1.model import make_v2_unscaled_dot_localbalanced_pair
from tfpd_exploration.src.family_runtime_v1.h1_lifted_frontend import H1LiftedSpatialFrontend
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank


@pytest.mark.parametrize("batch", [1, 2])
def test_actual_common_pair_lifted_full_repair_and_nonzero_route(batch, monkeypatch):
    torch.set_num_threads(1)
    flat, route = make_v2_unscaled_dot_localbalanced_pair(seed=42)
    flat.eval(); route.eval()
    torch.manual_seed(901)
    mask = torch.rand(batch, 176) > .15
    bank = H1Bank(torch.randn(batch, 176, 700), torch.randn(batch, 176, 4), mask)
    raw = torch.randn(batch, 96, 176)
    for model, gate in ((flat, None), (route, 0.), (route, .2)):
        if gate is not None:
            with torch.no_grad():
                model.frontend.attn.g.fill_(gate)
        lifted = H1LiftedSpatialFrontend(model, bank)
        with torch.no_grad():
            expected = model.encode_frontend(raw, bank)
        torch.testing.assert_close(lifted.full(raw), expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(lifted.repair(raw), expected[:, [0, 1, 2, 3, 95]], atol=1e-5, rtol=1e-5)
        short = raw[:, :9]
        torch.testing.assert_close(lifted.full(short), model.encode_frontend(short, bank), atol=1e-5, rtol=1e-5)
        def forbidden(*args, **kwargs):
            raise AssertionError("native full attention called by lifted path")
        with monkeypatch.context() as patch:
            patch.setattr(model.frontend, "_set_from_local", forbidden)
            lifted.repair(raw)


def test_unbatched_bank_and_all_masked_matches_native():
    torch.set_num_threads(1)
    flat, _ = make_v2_unscaled_dot_localbalanced_pair(seed=42); flat.eval()
    bank = H1Bank(torch.randn(176, 700), torch.randn(176, 4), torch.zeros(176, dtype=torch.bool))
    raw = torch.randn(2, 9, 176)
    lifted = H1LiftedSpatialFrontend(flat, bank)
    torch.testing.assert_close(lifted.full(raw), flat.encode_frontend(raw, bank), atol=1e-5, rtol=1e-5)
