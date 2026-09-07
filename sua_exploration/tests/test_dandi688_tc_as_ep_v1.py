from __future__ import annotations

import random

import numpy as np
import pytest

from sua_exploration.mc_maze.dandi688_tc_as_ep_v1 import plan
from sua_exploration.mc_maze.dandi688_tc_as_ep_v1.selector import (
    ActivityAuthority,
    _extremum_index,
    jaccard,
    point_biserial_and_smd_full,
    robust_activity_z,
    select_cov_anti10,
    select_cov_fixed,
    select_cov_random10,
    select_cov_top10,
)


def authority() -> ActivityAuthority:
    rng = np.random.default_rng(20260904)
    directions = np.tile(np.arange(8, dtype=np.int64), 7)[:50]
    # Make coordinates nondegenerate while retaining repeated directions.
    rates = rng.gamma(shape=2.0, scale=4.0, size=(50, 12)).astype(np.float64)
    durations = (3.0 + rng.uniform(0.0, 2.0, size=50)).astype(np.float64)
    return ActivityAuthority(
        session_id="synthetic",
        rates_hz=rates,
        durations_s=durations,
        directions=directions,
        chronological_indices=np.arange(50, dtype=np.int64),
    ).validated()


def assert_covered(result, source: ActivityAuthority) -> None:
    assert len(result.indices) == 10
    assert list(result.indices) == sorted(result.indices)
    assert len(set(result.indices)) == 10
    counts = np.bincount(source.directions[list(result.indices)], minlength=8)
    assert np.all(counts >= 1)
    assert np.all(counts <= 2)


def test_all_support_laws_are_covered_sorted_and_deterministic() -> None:
    source = authority()
    results = [
        select_cov_top10(source),
        select_cov_anti10(source),
        select_cov_fixed(source, late=False),
        select_cov_fixed(source, late=True),
        select_cov_random10(
            source, training_seed=42, epoch=3, sample_or_window_id="window-17"
        ),
    ]
    for result in results:
        assert_covered(result, source)
    assert select_cov_top10(source).receipt() == select_cov_top10(source).receipt()
    assert select_cov_anti10(source).receipt() == select_cov_anti10(source).receipt()


def test_invalid_direction_stays_in_normalizer_but_never_support() -> None:
    source = authority()
    directions = source.directions.copy()
    directions[0] = -1
    # Direction zero still has later candidates.
    changed = ActivityAuthority(
        session_id=source.session_id,
        rates_hz=source.rates_hz,
        durations_s=source.durations_s,
        directions=directions,
        chronological_indices=source.chronological_indices,
    ).validated()
    _, receipt = robust_activity_z(changed)
    assert receipt["normalizer_pool"].endswith("including_direction_minus_one")
    for result in (
        select_cov_top10(changed),
        select_cov_anti10(changed),
        select_cov_fixed(changed, late=False),
        select_cov_fixed(changed, late=True),
    ):
        assert 0 not in result.indices
        assert_covered(result, changed)


def test_global_extremum_tie_rule_is_traversal_independent() -> None:
    candidates = [9, 2, 5]
    values = [1.0 + 5e-13, 1.0, 1.0 + 2e-13]
    assert _extremum_index(candidates, values, maximize=False) == 2
    assert _extremum_index(candidates[::-1], values[::-1], maximize=False) == 2
    max_values = [3.0, 3.0 - 5e-13, 3.0 - 2e-13]
    assert _extremum_index(candidates, max_values, maximize=True) == 2
    assert _extremum_index(candidates[::-1], max_values[::-1], maximize=True) == 2


def test_stateless_random_support_has_entropy_without_global_rng_drift() -> None:
    source = authority()
    np.random.seed(123)
    random.seed(456)
    np_before = np.random.get_state()
    py_before = random.getstate()
    try:
        import torch

        torch.manual_seed(789)
        torch_before = torch.random.get_rng_state().clone()
    except ModuleNotFoundError:
        torch = None
        torch_before = None

    selections = {
        select_cov_random10(
            source,
            training_seed=42,
            epoch=1,
            sample_or_window_id=f"window-{index}",
        ).indices
        for index in range(64)
    }
    assert len(selections) > 1
    for indices in selections:
        assert_covered(type("R", (), {"indices": indices})(), source)

    np_after = np.random.get_state()
    py_after = random.getstate()
    assert np_before[0] == np_after[0]
    assert np.array_equal(np_before[1], np_after[1])
    assert np_before[2:] == np_after[2:]
    assert py_before == py_after
    if torch_before is not None:
        assert torch is not None
        assert torch.equal(torch_before, torch.random.get_rng_state())


def test_zero_mad_coordinates_are_exact_zero() -> None:
    source = authority()
    rates = source.rates_hz.copy()
    rates[:, 0] = 7.0
    changed = ActivityAuthority(
        session_id=source.session_id,
        rates_hz=rates,
        durations_s=source.durations_s,
        directions=source.directions,
        chronological_indices=source.chronological_indices,
    ).validated()
    z, receipt = robust_activity_z(changed)
    assert np.array_equal(z[:, 0], np.zeros(50, dtype=np.float64))
    assert receipt["zero_mad_coordinate_count"] >= 1


def test_activity_feature_projects_out_linear_log_duration_component() -> None:
    source = authority()
    _z, receipt = robust_activity_z(source)
    assert receipt["feature"].startswith("duration_residualized_")
    q = np.log(source.durations_s)
    q -= q.mean()
    features = np.log1p(source.rates_hz)
    features0 = features - features.mean(axis=0, keepdims=True)
    slope = (q[:, None].T @ features0 / np.sum(q * q)).reshape(-1)
    residual = features - q[:, None] * slope[None, :]
    assert np.max(np.abs(q @ residual)) < 1e-12


def test_point_biserial_and_full_sd_smd_identity() -> None:
    durations = np.linspace(2.5, 5.0, 50, dtype=np.float64)
    selected = tuple(range(0, 50, 5))
    pb, smd = point_biserial_and_smd_full(durations, selected)
    assert smd == pytest.approx(pb / np.sqrt(0.2 * 0.8), abs=1e-15)


def test_jaccard_exact_two_replacements_is_two_thirds() -> None:
    left = tuple(range(10))
    right = tuple(range(8)) + (10, 11)
    assert jaccard(left, right) == pytest.approx(2.0 / 3.0)


def test_authority_rejects_negative_rates_and_missing_direction() -> None:
    source = authority()
    rates = source.rates_hz.copy()
    rates[0, 0] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        ActivityAuthority(
            source.session_id,
            rates,
            source.durations_s,
            source.directions,
            source.chronological_indices,
        ).validated()
    directions = source.directions.copy()
    directions[directions == 7] = 6
    with pytest.raises(ValueError, match="all eight"):
        ActivityAuthority(
            source.session_id,
            source.rates_hz,
            source.durations_s,
            directions,
            source.chronological_indices,
        ).validated()
