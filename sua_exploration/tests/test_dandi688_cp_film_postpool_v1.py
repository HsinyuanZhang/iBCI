from __future__ import annotations

import pytest

from mc_maze.dandi688_cp_film_postpool_v1 import core, plan


def _row(mean, positives=4):
    return {"mean_delta": mean, "positive_sessions": positives}


def test_decision_requires_native_and_empty_gain():
    result = core.decide(_row(0.01), _row(0.02), _row(0.002), _row(0.001))
    assert result["cp10_profile_utility"] is True
    assert result["cp30_profile_utility"] is False
    assert result["seed_expansion_authorized"] is True


def test_averaged_state_is_float64_mean_cast_back():
    torch = pytest.importorskip("torch")
    states = [{"x": torch.tensor([float(index)], dtype=torch.float32)} for index in range(1, 5)]
    out = core.averaged_state(states)
    assert out["x"].dtype == torch.float32
    assert out["x"].item() == 2.5


def test_arm_parameter_count_and_zero_head():
    torch = pytest.importorskip("torch")
    post = torch.nn.Sequential(
        torch.nn.Linear(68, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64),
        torch.nn.ReLU(), torch.nn.Linear(64, 50),
    )
    arm = core.build_arm(post)
    assert sum(parameter.numel() for parameter in arm.parameters()) == plan.TRAINABLE_PARAMETERS
    assert torch.count_nonzero(arm.film[2].weight).item() == 0
    assert torch.count_nonzero(arm.film[2].bias).item() == 0
