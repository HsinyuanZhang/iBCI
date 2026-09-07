from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from sua_exploration.mc_maze import h1_event_budget_sweep as sweep
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as v2


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "SPINT-main/data/000954"


def _event(index: int, trial: int, *, rates: float | None = None) -> v1.MovementEvent:
    displacement = np.zeros(7, dtype=np.float64)
    displacement[index % 7] = 1.0 + index / 10.0
    log_rates = np.full(v1.EXPECTED_NEURONS, rates if rates is not None else index / 100.0, dtype=np.float64)
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


def test_truncation_uses_first_k_chronological_events() -> None:
    events = tuple(_event(index, trial=index // 3) for index in range(24))
    session = _session(events)
    support = v2.select_trial_range(session, start_index=0, budget=4)
    subset = support[:10]
    assert [event.row_id for event in subset] == [event.row_id for event in support[:10]]
    assert subset == support[:10]
    assert tuple(event.row_id for event in subset) != tuple(event.row_id for event in reversed(subset))


def test_insufficient_support_events_returns_undefined_status() -> None:
    sessions = _load_real_sessions()
    session = sessions["ses-19250101T111740"]
    basis = v2.fit_source_all_event_basis(sessions, outer_date="19250108")
    support = v2.select_trial_range(session, start_index=0, budget=4)
    requested = len(support) + 1
    reference = np.zeros((v1.EXPECTED_NEURONS, v2.CARRIER_DIM), dtype=np.float64)
    cell = sweep.evaluate_session_at_k(session, basis, requested, reference_carrier=reference)
    assert cell["status"] == "undefined_insufficient_events"
    assert cell["support_events_available"] == len(support)


def test_shuffle_modes_are_fixed_point_free_and_rule_driven() -> None:
    events = tuple(_event(index, trial=index // 4) for index in range(12))
    assert sweep.shuffle_mode(events) == "within_trial"
    order, _ = v1.within_trial_label_shuffle(events, session="ses-test", budget=4)
    assert not np.any(order == np.arange(len(events)))

    sparse_events = tuple(_event(index, trial=index if index < 8 else 1) for index in range(10))
    assert sweep.shuffle_mode(sparse_events) == "global_cyclic"
    order, manifest = sweep.global_cyclic_shuffle_order(
        len(sparse_events), session="ses-test", budget=4, k=10,
    )
    assert manifest["shift"] >= 1
    assert not np.any(order == np.arange(len(sparse_events)))


def test_all_cell_carrier_fidelity_is_one() -> None:
    sessions = _load_real_sessions()
    body = sweep.run_screen(sessions)
    for name in v1.H1_HELDIN_SESSIONS:
        cell = body["sessions"][name]["all"]
        assert cell["status"] == "defined"
        assert cell["carrier_fidelity"] == pytest.approx(1.0, abs=1.0e-12)


def test_all_cell_reproduces_forward_transfer() -> None:
    sessions = _load_real_sessions()
    body = sweep.run_screen(sessions)
    bases = {date: v2.fit_source_all_event_basis(sessions, outer_date=date) for date in v1.H1_DATES}
    all_cells = {name: body["sessions"][name]["all"] for name in v1.H1_HELDIN_SESSIONS}
    reproduction = sweep.reproduces_hse5(sessions, bases, all_cells)
    assert reproduction["passed"] is True
    assert reproduction["maximum_absolute_difference"] <= sweep.HSE5_TOLERANCE


def test_run_screen_is_deterministic() -> None:
    sessions = _load_real_sessions()
    first = sweep.run_screen(sessions)
    second = sweep.run_screen(sessions)
    assert v1.canonical_json_bytes(first) == v1.canonical_json_bytes(second)
