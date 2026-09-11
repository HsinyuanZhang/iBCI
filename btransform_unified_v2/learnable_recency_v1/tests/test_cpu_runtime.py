from __future__ import annotations

import pytest
import torch

from learnable_recency_v1.cpu_temporal import CpuLearnableRecencyRuntime

from helpers import TIERS, make_pair, scramble_new_params


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("context_bins", [50, 300])
def test_cpu_runtime_matches_step(tier: str, context_bins: int) -> None:
    _fixed, model = make_pair(context_bins, tier, seed=29, width=16, heads=4)
    scramble_new_params(model, torch.Generator().manual_seed(29))
    model.eval()
    runtime = CpuLearnableRecencyRuntime(model, batch_size=2)
    state = model.init_state(2, "cpu", torch.float32)
    steps = 2 * max(model.config.windows) + 3 if context_bins == 50 else 20
    for time in range(steps):
        z = torch.randn(2, 16)
        valid = torch.tensor([time != 2, time > 1])
        got = runtime.step(z, valid)
        want, state = model.step(z, state, valid)
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_cpu_runtime_reset_and_reorder() -> None:
    _fixed, model = make_pair(50, "cable", seed=31, width=16, heads=4)
    scramble_new_params(model, torch.Generator().manual_seed(31))
    model.eval()
    runtime = CpuLearnableRecencyRuntime(model, 2)
    state = model.init_state(2, "cpu", torch.float32)
    for _ in range(6):
        z = torch.randn(2, 16)
        got = runtime.step(z)
        want, state = model.step(z, state)
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
    runtime.reorder([1, 0])
    state = model.select_rows(state, [1, 0])
    runtime.reset_rows([1])
    state = model.reset_rows(state, [1])
    z = torch.randn(2, 16)
    got = runtime.step(z)
    want, state = model.step(z, state)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
