from __future__ import annotations

import numpy as np

from sua_exploration.mc_maze import h1_event_carrier_design_screen as design
from sua_exploration.mc_maze import h1_event_carrier_semantic_v4 as semantic
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1


def _context_event(index: int, duration: float, velocity: np.ndarray) -> design.ContextEvent:
    displacement = np.asarray(velocity, dtype=np.float64) * float(duration)
    base = v1.MovementEvent(
        row_id=index,
        tag=v1.MOVEMENT_TAGS[index % len(v1.MOVEMENT_TAGS)],
        trial_value=1.0,
        trial_index=0,
        start_time=float(index),
        stop_time=float(index) + float(duration),
        duration_seconds=float(duration),
        eval_bins=max(v1.MIN_EVAL_BINS, int(round(duration / v1.BIN_SECONDS))),
        displacement=displacement,
        log_rates=np.zeros(v1.EXPECTED_NEURONS, dtype=np.float64),
    )
    return design.ContextEvent(
        base=base,
        start_state=np.zeros(v1.POSITION_DIM, dtype=np.float64),
        midpoint_state=displacement / 2.0,
    )


def test_v4_candidate_matrix_is_fixed_and_five_wide() -> None:
    assert [candidate.name for candidate in semantic.CANDIDATES] == [
        "pca_delta_q4",
        "pca_mean_velocity_q4",
        "ser_mean_velocity_q4",
        "ser_direction_speed_q4",
    ]
    assert all(candidate.carrier_dim == 5 for candidate in semantic.CANDIDATES)
    assert semantic.RANK == 4


def test_mean_velocity_uses_only_endpoint_displacement_and_duration() -> None:
    velocity = np.asarray([1.0, -2.0, 0.5, 0.25, -0.75, 1.5, 2.5], dtype=np.float64)
    events = (_context_event(0, 0.5, velocity), _context_event(1, 1.75, velocity))
    observed = semantic.semantic_raw_features(events, "mean_velocity")
    np.testing.assert_allclose(observed, np.stack((velocity, velocity)), rtol=0.0, atol=1.0e-12)
    displacement = semantic.semantic_raw_features(events, "delta")
    assert not np.allclose(displacement[0], displacement[1])


def test_direction_speed_uses_source_frozen_coordinate_normalizer() -> None:
    events = (
        _context_event(0, 1.0, np.asarray([1, 2, 3, 4, 5, 6, 7], dtype=np.float64)),
        _context_event(1, 1.0, np.asarray([-2, 1, 4, 1, 3, 8, 2], dtype=np.float64)),
    )
    mean = np.zeros(v1.POSITION_DIM, dtype=np.float64)
    scale = np.arange(1, v1.POSITION_DIM + 1, dtype=np.float64)
    output = semantic.semantic_raw_features(
        events, "direction_speed", velocity_mean=mean, velocity_scale=scale,
    )
    assert output.shape == (2, 8)
    np.testing.assert_allclose(np.linalg.norm(output[:, :7], axis=1), 1.0, rtol=0.0, atol=1.0e-12)
    assert np.all(output[:, 7] > 0)


def test_semantic_gate_does_not_lower_material_threshold() -> None:
    candidate = semantic.CANDIDATES[1]
    summary = {
        "defined_sessions": 13,
        "mean": 0.0199,
        "median": 0.02,
        "positive": 13,
        "leave_largest_absolute_out_mean": 0.02,
    }
    aggregate = {
        "correct_minus_hse5": summary,
        "correct_minus_label_shuffle": {**summary, "mean": 0.03},
        "correct_minus_intercept": {**summary, "mean": 0.03},
    }
    assert semantic.budget_gate(candidate, aggregate)["passed"] is False
    aggregate["correct_minus_hse5"] = {**summary, "mean": 0.0201}
    assert semantic.budget_gate(candidate, aggregate)["passed"] is True
