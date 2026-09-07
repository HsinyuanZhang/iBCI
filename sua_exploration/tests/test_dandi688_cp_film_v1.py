from __future__ import annotations

import numpy as np
import pytest

from mc_maze.dandi688_cp_film_v1 import core, data, plan


def test_profile_is_finite_masked_and_nondegenerate():
    rng = np.random.default_rng(4)
    neural = rng.poisson(0.2, size=(800, 17)).astype(np.float32)
    velocity = rng.normal(size=(800, 2)).astype(np.float32)
    profile, evidence = core.profile_from_bins(neural, velocity)
    assert profile.shape == (17, 4)
    assert np.isfinite(profile).all()
    assert np.count_nonzero(profile[:, 2:]) == 0
    assert evidence["low_state_bins"] >= 200
    assert evidence["high_state_bins"] >= 200


def test_zero_init_film_is_exact_native():
    torch = pytest.importorskip("torch")
    base = torch.nn.Module()
    base.post_pool = torch.nn.Sequential(torch.nn.Linear(68, 64), torch.nn.ReLU(), torch.nn.Linear(64, 50))
    mean = torch.randn(1, 19, 64)
    carrier = torch.randn(1, 19, 4)
    profile = torch.randn(1, 19, 4)
    film = core.build_film()
    native = base.post_pool(torch.cat((mean, carrier), dim=-1))
    observed = core.film_identity(base, mean, carrier, profile, film)
    assert torch.equal(native, observed)
    assert sum(parameter.numel() for parameter in film.parameters()) == 1224
    assert not bool(torch.signbit(film[2].weight).any())
    assert not bool(torch.signbit(film[2].bias).any())


def test_profile_shuffle_is_deterministic_nonidentity():
    profile = np.arange(80, dtype=np.float32).reshape(20, 4)
    a, pa = data.shuffled_profile(profile, "session", 42)
    b, pb = data.shuffled_profile(profile, "session", 42)
    assert np.array_equal(a, b) and np.array_equal(pa, pb)
    assert not np.array_equal(pa, np.arange(20))
    assert np.array_equal(a, profile[pa])


def test_decision_requires_controls_for_positive_claim():
    base = {"mean_delta": 0.012, "median_delta": 0.01, "worst_delta": -0.001, "positive_sessions": 5, "session_count": 6, "per_session_delta": {}}
    weak = {**base, "mean_delta": 0.001}
    result = core.decide({"CP10@M10": base, "CP30@M30": weak, "SHUFFLE10@M10": {**weak, "mean_delta": -0.001}, "EMPTY@ZERO": {**weak, "mean_delta": 0.0}})
    assert result["seed_expansion_authorized"]
    assert result["cp10_solid"]
    assert not result["evalai_push"]


def test_frozen_constants_are_cross_dataset_early_pool_contract():
    assert plan.PROFILE_MASK == (1.0, 1.0, 0.0, 0.0)
    assert plan.CONTEXT_DIM == 8 and plan.FILM_RANK == 8
    assert plan.ACTIVITY_SUPPORT == 30 and plan.QUERY_START == 50
    assert plan.ARMS == ("CP10", "CP30", "SHUFFLE10", "EMPTY")

