from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sua_exploration.mc_maze import h1_event_conditioning_sweep as screen
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as v2


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "SPINT-main/data/000954"


def _event(index: int, trial: int, *, displacement_scale: float = 1.0) -> v1.MovementEvent:
    displacement = np.zeros(7, dtype=np.float64)
    displacement[index % 7] = displacement_scale * (1.0 + index / 10.0)
    log_rates = np.full(v1.EXPECTED_NEURONS, index / 100.0, dtype=np.float64)
    return v1.MovementEvent(
        row_id=index,
        tag=v1.MOVEMENT_TAGS[index % len(v1.MOVEMENT_TAGS)],
        trial_value=float(trial + 1),
        trial_index=trial,
        start_time=float(index),
        stop_time=float(index) + 0.5,
        duration_seconds=0.5,
        eval_bins=25,
        displacement=displacement,
        log_rates=log_rates,
    )


def _session(events: tuple[v1.MovementEvent, ...]) -> v1.EventSession:
    trials = tuple(sorted({event.trial_value for event in events}))
    return v1.EventSession(
        session_name="ses-19250101T111740",
        date="19250101",
        path=Path("synthetic.nwb"),
        input_sha256="0" * 64,
        trial_values=trials,
        eval_bins_per_trial=(25,) * len(trials),
        events=events,
        exclusion_counts={},
        position_description="synthetic",
        position_unit="m",
        position_conversion=1.0,
        position_offset=0.0,
    )


def _load_real_sessions() -> dict[str, v1.EventSession]:
    paths = v1.index_heldin_calib(DATA_ROOT)
    return {name: v1.load_event_session(paths[name]) for name in v1.H1_HELDIN_SESSIONS}


def test_maxspread_selects_spread_points_and_minspread_selects_cluster() -> None:
    cluster = np.zeros((5, 2), dtype=np.float64)
    outliers = np.asarray([[100.0, 0.0], [0.0, 100.0], [-100.0, 0.0]], dtype=np.float64)
    z = np.vstack((cluster, outliers))
    spread = screen.select_maxspread(z, 3)
    compact = screen.select_minspread(z, 3)
    assert set(spread.tolist()) == {5, 6, 7}
    assert set(compact.tolist()) == {0, 1, 2}


def test_greedy_selectors_are_deterministic_and_return_k_distinct_indices() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=(20, 4))
    for selector in (screen.select_maxspread, screen.select_minspread):
        first = selector(z, 10)
        second = selector(z, 10)
        np.testing.assert_array_equal(first, second)
        assert first.size == 10
        assert len(set(first.tolist())) == 10


def test_random_subsets_are_reproducible_and_vary_by_replicate() -> None:
    z = np.arange(40, dtype=np.float64).reshape(10, 4)
    first = screen.select_random(z, 5, session="ses-test", budget=4, replicate=0)
    again = screen.select_random(z, 5, session="ses-test", budget=4, replicate=0)
    other = screen.select_random(z, 5, session="ses-test", budget=4, replicate=1)
    np.testing.assert_array_equal(first, again)
    assert first.size == 5
    assert not np.array_equal(first, other)


def test_rank_correlation_matches_hand_computed_value_with_ties() -> None:
    x = np.asarray([1.0, 2.0, 2.0, 4.0], dtype=np.float64)
    y = np.asarray([1.0, 3.0, 5.0, 7.0], dtype=np.float64)
    observed = screen.spearman_correlation(x, y)
    expected = screen.pearson_correlation(screen._average_ranks(x), screen._average_ranks(y))
    assert observed == pytest.approx(expected, abs=1.0e-12)
    pooled = screen.rank_correlations(x, y)
    assert pooled["spearman"] == pytest.approx(observed, abs=1.0e-12)


def test_rank_deficient_subset_returns_undefined_status() -> None:
    sessions = _load_real_sessions()
    name = v1.H1_HELDIN_SESSIONS[0]
    session = sessions[name]
    basis = v2.fit_source_all_event_basis(sessions, outer_date=v1.session_date(name))
    reference = np.zeros((v1.EXPECTED_NEURONS, v2.CARRIER_DIM), dtype=np.float64)
    cell = screen.evaluate_subset_cell(
        session,
        basis,
        k=10,
        rule="first",
        indices=np.zeros(10, dtype=np.int64),
        reference_carrier=reference,
    )
    assert cell["status"] == "undefined_rank_deficient"


def test_full_support_reference_matches_forward_transfer() -> None:
    sessions = _load_real_sessions()
    bases = {date: v2.fit_source_all_event_basis(sessions, outer_date=date) for date in v1.H1_DATES}
    full_support_rows = {
        name: screen.evaluate_full_support(sessions[name], bases[v1.session_date(name)])
        for name in v1.H1_HELDIN_SESSIONS
    }
    reproduction = screen.verify_full_support_reproduction(sessions, bases, full_support_rows)
    assert reproduction["passed"] is True
    assert reproduction["maximum_absolute_difference"] <= screen.HSE5_TOLERANCE


def test_run_screen_is_deterministic() -> None:
    sessions = _load_real_sessions()
    first = screen.run_screen(sessions)
    second = screen.run_screen(sessions)
    assert v1.canonical_json_bytes(first) == v1.canonical_json_bytes(second)
