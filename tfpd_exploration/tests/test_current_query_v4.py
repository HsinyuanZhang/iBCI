"""Read-time priors have explicit new semantics but preserve finite cache parity."""
import pytest
import torch

from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2 import QueryMemoryCache
from tfpd_exploration.src.two_mainlines_long_v1.current_query_v3 import LogAgeQueryTemporalStack
from tfpd_exploration.src.two_mainlines_long_v1.current_query_v4 import RecencyPriorQueryTemporalStack


CFG = dict(width=16, heads=4, layers=2, ffn=24, window=31, age_buckets=8, seed=42)
TAUS = (2., 8., 32., float("inf"))


def test_parameters_preserved_and_old_state_cannot_silently_load():
    old = LogAgeQueryTemporalStack(**CFG)
    new = RecencyPriorQueryTemporalStack(**CFG, head_taus=TAUS)
    for key, value in old.named_parameters():
        assert torch.equal(value, dict(new.named_parameters())[key])
    assert int(new.temporal_contract_version) == 4
    with pytest.raises(RuntimeError, match="Missing key"):
        new.load_state_dict(old.state_dict(), strict=True)
    restored = RecencyPriorQueryTemporalStack(**CFG, head_taus=TAUS)
    restored.load_state_dict(new.state_dict(), strict=True)
    for key, value in new.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


def test_prior_direction_normalized_local_mass_and_unbiased_global_head():
    model = RecencyPriorQueryTemporalStack(**CFG, head_taus=TAUS)
    prior = model.blocks[0].recency_prior_by_age
    assert torch.equal(prior[:, 0], torch.zeros(4))
    assert torch.equal(prior[-1], torch.zeros(31))
    assert bool((prior[:-1, 1:] < prior[:-1, :-1]).all())
    mass = prior.softmax(-1)
    assert float(mass[0, :5].sum()) > .9
    torch.testing.assert_close(mass[-1], torch.full((31,), 1 / 31))


def test_cached_realigned_original_memory_and_rollover_match_recompute():
    torch.manual_seed(9)
    model = RecencyPriorQueryTemporalStack(**CFG, head_taus=TAUS).eval()
    cache = QueryMemoryCache(model)
    z = torch.randn(2, 31, 16)
    with torch.no_grad():
        cache.reset(z)
        for _ in range(40):
            z = torch.cat((z[:, 1:], torch.randn(2, 1, 16)), dim=1)
            # Boundary-repaired tokens also change after the shift.
            z[:, :4] = torch.randn_like(z[:, :4])
            cache.advance(z, torch.tensor([0, 1, 2, 3, 30]))
            torch.testing.assert_close(cache.predict(), model(z), atol=1e-5, rtol=1e-5)


def test_invalid_prior_contract_is_rejected():
    for taus in ((1., 2.), (1., 2., 3., 4.), (0., 2., 4., float("inf")), (float("nan"), 2., 4., float("inf"))):
        with pytest.raises(ValueError):
            RecencyPriorQueryTemporalStack(**CFG, head_taus=taus)
