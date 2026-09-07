"""Log-age memory stays exact-cacheable; no learned R2 is inferred here."""
import pytest
import torch

from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2.core import QueryMemoryCache, QueryTemporalStack
from tfpd_exploration.src.two_mainlines_long_v1.current_query_v3.core import LogAgeQueryTemporalStack, log_age_indices


def pair():
    args = dict(width=16, heads=4, layers=2, ffn=24, window=700, age_buckets=16, seed=42)
    return QueryTemporalStack(**args).eval(), LogAgeQueryTemporalStack(**args).eval()


def test_initial_parameters_identical_but_new_contract_is_strict():
    old, new = pair()
    for name, value in old.named_parameters():
        torch.testing.assert_close(value, dict(new.named_parameters())[name], atol=0, rtol=0)
    z = torch.randn(2, 700, 16)
    torch.testing.assert_close(old(z), new(z), atol=0, rtol=0)
    with pytest.raises(RuntimeError, match="Missing key"):
        new.load_state_dict(old.state_dict(), strict=True)
    indices = log_age_indices(700, 16)
    assert indices.shape == (700,) and indices[0] == 0 and indices[-1] == 15
    assert indices[1] != indices[43]
    assert torch.all(indices[1:] >= indices[:-1])


def test_short_lag_exchangeability_is_removed_by_log_lookup():
    old, new = pair()
    with torch.no_grad():
        for stack in (old, new):
            for block in stack.blocks:
                block.age_bias.copy_(torch.arange(16)[None].expand(4, -1) * .35)
    torch.manual_seed(61)
    z = torch.randn(1, 700, 16)
    swapped = z.clone()
    swapped[:, [656, 698]] = z[:, [698, 656]]  # Ages43 and1; never exchange the current query.
    torch.testing.assert_close(old(z), old(swapped), atol=1e-6, rtol=1e-6)
    assert float((new(z) - new(swapped)).abs().max()) > 1e-5


def test_log_memory_rollover_matches_fresh_projection():
    _, model = pair()
    z = torch.randn(1, 700, 16)
    memory = QueryMemoryCache(model)
    memory.reset(z)
    with torch.no_grad():
        for block in model.blocks:
            block.age_bias.normal_(std=.2)
        for _ in range(5):
            z = torch.cat((z[:, 1:], torch.randn(1, 1, 16)), dim=1)
            z[:, :4] = torch.randn(1, 4, 16)
            memory.advance(z, (0, 1, 2, 3, 699))
            torch.testing.assert_close(memory.predict(), model(z), atol=1e-5, rtol=1e-5)
