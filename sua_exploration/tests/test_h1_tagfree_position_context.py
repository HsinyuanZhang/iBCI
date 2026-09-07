from __future__ import annotations

import copy
from typing import Any

import numpy as np

from sua_exploration.mc_maze import h1_event_carrier_design_screen as sealed
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1
from sua_exploration.mc_maze import h1_tagfree_position_context as pctx


def context(index: int, trial: int, *, tag: str | None = None) -> pctx.ContextEvent:
    delta = np.arange(7, dtype=np.float64) / 10.0 + index
    chosen_tag = tag or v1.MOVEMENT_TAGS[index % len(v1.MOVEMENT_TAGS)]
    base = v1.MovementEvent(
        row_id=index,
        tag=chosen_tag,
        trial_value=float(trial + 1),
        trial_index=trial,
        start_time=float(index),
        stop_time=float(index) + 1.0,
        duration_seconds=1.0,
        eval_bins=50,
        displacement=delta,
        log_rates=np.full(v1.EXPECTED_NEURONS, index / 10.0, dtype=np.float64),
    )
    start = np.arange(7, dtype=np.float64) + trial
    return pctx.ContextEvent(base=base, start_state=start, midpoint_state=start + delta / 2.0)


def test_reference_families_match_sealed_raw_features() -> None:
    events = tuple(context(index, 0) for index in range(8))
    for family in ("delta", "context"):
        ours = pctx.raw_features(events, family)
        sealed_values = sealed.raw_features(events, family)
        np.testing.assert_array_equal(ours, sealed_values)


def test_tag_free_families_ignore_tag_overrides() -> None:
    events = tuple(context(index, trial % 3, tag=v1.MOVEMENT_TAGS[index % len(v1.MOVEMENT_TAGS)])
                   for trial in range(3) for index in range(8))
    alternate = tuple(v1.MOVEMENT_TAGS[(index + 3) % len(v1.MOVEMENT_TAGS)] for index in range(len(events)))
    for family in ("poscontext", "startstop", "deltastart"):
        baseline = pctx.raw_features(events, family)
        swapped = pctx.raw_features(events, family, tag_overrides=alternate)
        np.testing.assert_array_equal(baseline, swapped)


def _orthonormal_basis(matrix: np.ndarray) -> np.ndarray:
    q, _r = np.linalg.qr(matrix, mode="reduced")
    return q


def test_startstop_and_deltastart_span_poscontext_column_space() -> None:
    rng = np.random.default_rng(0)
    events = []
    for index in range(24):
        delta = rng.normal(size=7)
        start = rng.normal(size=7)
        base = v1.MovementEvent(
            row_id=index,
            tag=v1.MOVEMENT_TAGS[index % len(v1.MOVEMENT_TAGS)],
            trial_value=float(index // 8 + 1),
            trial_index=index // 8,
            start_time=float(index),
            stop_time=float(index) + 1.0,
            duration_seconds=1.0,
            eval_bins=50,
            displacement=delta,
            log_rates=np.full(v1.EXPECTED_NEURONS, 0.0, dtype=np.float64),
        )
        events.append(pctx.ContextEvent(base=base, start_state=start, midpoint_state=start + delta / 2.0))
    poscontext = pctx.raw_features(tuple(events), "poscontext")
    startstop = pctx.raw_features(tuple(events), "startstop")
    deltastart = pctx.raw_features(tuple(events), "deltastart")
    q_ref = _orthonormal_basis(poscontext)
    for matrix in (startstop, deltastart):
        q_other = _orthonormal_basis(matrix)
        overlap = q_ref.T @ q_other
        np.testing.assert_allclose(overlap @ overlap.T, np.eye(q_ref.shape[1]), atol=1.0e-10)


def test_lodo_basis_excludes_evaluated_session_date() -> None:
    sessions = {
        name: pctx.ContextSession(
            base=v1.EventSession(
                session_name=name,
                date=v1.session_date(name),
                path=__import__("pathlib").Path("/tmp/placeholder.nwb"),
                input_sha256="0" * 64,
                trial_values=(1.0, 2.0, 3.0, 4.0, 5.0),
                eval_bins_per_trial=(10, 10, 10, 10, 10),
                events=tuple(),
                exclusion_counts={},
                position_description="tx,ty,tz,rx,g1,g2,g3",
                position_unit="m",
                position_conversion=1.0,
                position_offset=0.0,
            ),
            events=tuple(context(index, 0) for index in range(4)),
        )
        for name in v1.H1_HELDIN_SESSIONS
    }
    for name in v1.H1_HELDIN_SESSIONS:
        outer_date = v1.session_date(name)
        source_names, _events = pctx._source_rows(sessions, outer_date)
        assert name not in source_names
        assert outer_date not in {v1.session_date(source_name) for source_name in source_names}


def test_feature_dimensionality() -> None:
    events = tuple(context(index, 0) for index in range(4))
    assert pctx.raw_features(events, "delta").shape == (4, 7)
    assert pctx.raw_features(events, "context").shape == (4, 22)
    assert pctx.raw_features(events, "poscontext").shape == (4, 14)
    assert pctx.raw_features(events, "startstop").shape == (4, 14)
    assert pctx.raw_features(events, "deltastart").shape == (4, 14)


def _passing_aggregate() -> dict[str, Any]:
    sessions = {name: {"median_delta_label_shuffle": 0.01, "median_delta_intercept": 0.01} for name in v1.H1_HELDIN_SESSIONS}
    return {
        "correct_minus_pca_delta_q4": {
            "defined_sessions": 13,
            "mean": 0.02,
            "median": 0.015,
            "positive": 13,
            "leave_largest_absolute_out_mean": 0.015,
        },
        "correct_minus_label_shuffle": v1.paired_summary([(name, 0.01) for name in v1.H1_HELDIN_SESSIONS]),
        "correct_minus_intercept": v1.paired_summary([(name, 0.01) for name in v1.H1_HELDIN_SESSIONS]),
    }


def test_gate_clauses_fail_independently() -> None:
    base = _passing_aggregate()
    candidate = pctx.CANDIDATES[2]

    def gate(aggregate: dict[str, Any], budget: int) -> bool:
        return pctx.evaluate_budget_gate(aggregate, budget=budget, candidate=candidate)["passes_primary_gate"]

    assert gate(base, 3)
    low_mean = copy.deepcopy(base)
    low_mean["correct_minus_pca_delta_q4"]["mean"] = 0.001
    assert not gate(low_mean, 3)

    low_positive = copy.deepcopy(base)
    low_positive["correct_minus_pca_delta_q4"]["positive"] = 10
    assert not gate(low_positive, 3)

    low_loo = copy.deepcopy(base)
    low_loo["correct_minus_pca_delta_q4"]["leave_largest_absolute_out_mean"] = -0.001
    assert not gate(low_loo, 3)

    bad_label = copy.deepcopy(base)
    bad_label["correct_minus_label_shuffle"] = v1.paired_summary([(name, -0.01) for name in v1.H1_HELDIN_SESSIONS])
    assert not gate(bad_label, 3)

    bad_intercept = copy.deepcopy(base)
    bad_intercept["correct_minus_intercept"] = v1.paired_summary([(name, -0.01) for name in v1.H1_HELDIN_SESSIONS])
    assert not gate(bad_intercept, 3)


def _synthetic_sessions() -> dict[str, pctx.ContextSession]:
    rng = np.random.default_rng(0)
    sessions: dict[str, pctx.ContextSession] = {}
    for session_index, name in enumerate(v1.H1_HELDIN_SESSIONS):
        events: list[pctx.ContextEvent] = []
        for trial in range(5):
            for event_index in range(8):
                offset = session_index * 100 + trial * 10 + event_index
                delta = rng.normal(size=7) + offset
                start = rng.normal(size=7) + trial
                log_rates = rng.normal(size=v1.EXPECTED_NEURONS) + offset
                base = v1.MovementEvent(
                    row_id=offset,
                    tag=v1.MOVEMENT_TAGS[event_index % len(v1.MOVEMENT_TAGS)],
                    trial_value=float(trial + 1),
                    trial_index=trial,
                    start_time=float(offset),
                    stop_time=float(offset) + 1.0,
                    duration_seconds=1.0,
                    eval_bins=50,
                    displacement=delta,
                    log_rates=log_rates,
                )
                events.append(pctx.ContextEvent(base=base, start_state=start, midpoint_state=start + delta / 2.0))
        sessions[name] = pctx.ContextSession(
            base=v1.EventSession(
                session_name=name,
                date=v1.session_date(name),
                path=__import__("pathlib").Path(f"/tmp/{name}.nwb"),
                input_sha256=f"{session_index:064x}",
                trial_values=(1.0, 2.0, 3.0, 4.0, 5.0),
                eval_bins_per_trial=(10, 10, 10, 10, 10),
                events=tuple(),
                exclusion_counts={},
                position_description="tx,ty,tz,rx,g1,g2,g3",
                position_unit="m",
                position_conversion=1.0,
                position_offset=0.0,
            ),
            events=tuple(events),
        )
    return sessions


def test_deterministic_canonical_json() -> None:
    sessions = _synthetic_sessions()
    first = v1.canonical_json_bytes(pctx.run_screen(sessions))
    second = v1.canonical_json_bytes(pctx.run_screen(sessions))
    assert first == second
