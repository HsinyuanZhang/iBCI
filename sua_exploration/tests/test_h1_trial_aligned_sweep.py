from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1
from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as v2
from sua_exploration.mc_maze import h1_trial_aligned_sweep as sweep


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


def test_support_and_eval_select_exact_events() -> None:
    events = tuple(_event(index, trial=index // 3) for index in range(24))
    session = _session(events)
    support_m3 = sweep.support_events(session, 3)
    eval_e4 = sweep.eval_events(session, 4)
    assert all(event.trial_index < 3 for event in support_m3)
    assert all(event.trial_index >= 4 for event in eval_e4)
    assert support_m3 == tuple(event for event in session.events if event.trial_index < 3)
    assert eval_e4 == tuple(event for event in session.events if event.trial_index >= 4)


def test_m1_yields_undefined_status() -> None:
    events = tuple(_event(index, trial=index // 4) for index in range(24))
    session = _session(events)
    sessions = {name: session for name in v1.H1_HELDIN_SESSIONS}
    basis = v2.fit_source_all_event_basis(sessions, outer_date="19250108")
    reference = np.zeros((v1.EXPECTED_NEURONS, v2.CARRIER_DIM), dtype=np.float64)
    cell = sweep.evaluate_cell(session, basis, 1, 4, reference_carrier=reference)
    assert cell["status"] == "undefined_insufficient_support_events"
    assert cell["support_events"] < sweep.MIN_SUPPORT_EVENTS


def test_decomposition_identity_on_synthetic_values() -> None:
    cells = {
        sweep.cell_key(3, 3): {"status": "defined", "median_delta_intercept": 0.30},
        sweep.cell_key(3, 4): {"status": "defined", "median_delta_intercept": 0.20},
        sweep.cell_key(4, 4): {"status": "defined", "median_delta_intercept": 0.05},
    }
    result = sweep.session_decomposition(cells)
    assert result["identity_holds"] is True
    assert result["identity_residual"] <= sweep.DECOMPOSITION_TOLERANCE
    assert result["total"] == pytest.approx(result["eval_effect"] + result["support_effect"])


def test_sealed_cells_reproduce_forward_transfer() -> None:
    sessions = _load_real_sessions()
    body = sweep.run_screen(sessions)
    reproduction = body["integrity_checks"]["sealed_forward_transfer_reproduction"]
    assert reproduction["passed"] is True
    assert reproduction["maximum_absolute_difference"] <= sweep.SEALED_TOLERANCE


def test_trial_contiguity_holds_on_real_data() -> None:
    sessions = _load_real_sessions()
    body = sweep.run_screen(sessions)
    contiguity = body["integrity_checks"]["trial_contiguity"]
    assert contiguity["all_sessions"] is True
    assert all(contiguity["per_session"].values())


def test_run_screen_is_deterministic() -> None:
    sessions = _load_real_sessions()
    first = sweep.run_screen(sessions)
    second = sweep.run_screen(sessions)
    assert v1.canonical_json_bytes(first) == v1.canonical_json_bytes(second)
