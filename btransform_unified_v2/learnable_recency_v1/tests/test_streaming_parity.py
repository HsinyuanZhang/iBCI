from __future__ import annotations

import pytest
import torch

from learnable_recency_v1.temporal import LearnableRecencyTemporal

from helpers import TIERS, make_pair, scramble_new_params, stream


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("context_bins,time", [(50, 19), (300, 21)])
def test_local_dense_step_parity_random_weights(tier: str, context_bins: int, time: int) -> None:
    _fixed, model = make_pair(context_bins, tier, seed=11, width=16, heads=4)
    generator = torch.Generator().manual_seed(11 * time + hash(tier) % 97)
    scramble_new_params(model, generator)
    model.train()
    z = torch.randn(3, time, 16)
    valid = torch.zeros(3, time, dtype=torch.bool)
    valid[0, 3:] = True
    valid[1, :] = True
    valid[2, 1:] = True
    local = model(z, valid, backend="local")
    dense = model(z, valid, backend="dense")
    stepped = stream(model, z, valid)
    torch.testing.assert_close(local, dense, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(local, stepped, atol=1e-5, rtol=1e-5)
    assert torch.equal(local[:, :3][0], torch.zeros_like(local[0, :3]))


@pytest.mark.parametrize("tier", TIERS)
def test_all_invalid_and_left_padding_m2(tier: str) -> None:
    _fixed, model = make_pair(50, tier, seed=13, width=16, heads=4)
    scramble_new_params(model, torch.Generator().manual_seed(13))
    model.eval()
    z = torch.randn(2, 14, 16)
    empty = torch.zeros(2, 14, dtype=torch.bool)
    torch.testing.assert_close(model(z, empty), torch.zeros(2, 14, 16))
    torch.testing.assert_close(stream(model, z, empty), torch.zeros(2, 14, 16))
    mask = torch.zeros(2, 14, dtype=torch.bool)
    mask[:, 4:] = True
    local = model(z, mask)
    torch.testing.assert_close(local[:, :4], torch.zeros_like(local[:, :4]))
    torch.testing.assert_close(local, stream(model, z, mask), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(local, model(z, mask, backend="dense"), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("tier", TIERS)
def test_h1_window_wrap_parity(tier: str) -> None:
    _fixed, model = make_pair(300, tier, seed=17, width=16, heads=4)
    scramble_new_params(model, torch.Generator().manual_seed(17))
    model.eval()
    z = torch.randn(2, 80, 16)
    mask = torch.ones(2, 80, dtype=torch.bool)
    mask[1, :6] = False
    local = model(z, mask)
    torch.testing.assert_close(local, model(z, mask, backend="dense"), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(local, stream(model, z, mask), atol=1e-5, rtol=1e-5)


def test_select_and_reset_rows_keep_increment_cache() -> None:
    _fixed, model = make_pair(50, "fox_gate", seed=19, width=16, heads=4)
    scramble_new_params(model, torch.Generator().manual_seed(19))
    model.eval()
    z = torch.randn(2, 8, 16)
    state = model.init_state(2, z.device, z.dtype)
    for time in range(5):
        _, state = model.step(z[:, time], state)
    selected = model.select_rows(state, [1, 0])
    assert len(selected.increments[0][0]) == len(selected.keys[0][0])
    reset = model.reset_rows(selected, [0])
    assert reset.increments[0][0] == []
    assert reset.keys[0][0] == []
    hidden, reset = model.step(z[:, 5], reset)
    assert hidden.shape == (2, 16)
    assert torch.isfinite(hidden).all()
