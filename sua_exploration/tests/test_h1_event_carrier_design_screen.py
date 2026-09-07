from __future__ import annotations

import numpy as np

from sua_exploration.mc_maze import h1_event_carrier_design_screen as s
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1


def context(index: int, trial: int) -> s.ContextEvent:
    delta = np.arange(7, dtype=np.float64) / 10.0 + index
    base = v1.MovementEvent(
        row_id=index,
        tag=v1.MOVEMENT_TAGS[index % len(v1.MOVEMENT_TAGS)],
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
    return s.ContextEvent(base=base, start_state=start, midpoint_state=start + delta / 2.0)


def test_feature_families_have_frozen_dimensions() -> None:
    events = tuple(context(index, 0) for index in range(8))
    assert s.raw_features(events, "delta").shape == (8, 7)
    assert s.raw_features(events, "context").shape == (8, 22)
    assert s.raw_features(events, "tag_delta").shape == (8, 64)
    assert s.raw_features(events, "tag_context").shape == (8, 120)


def test_tag_shuffle_changes_every_tag_within_each_trial() -> None:
    events = tuple(context(trial * 8 + index, trial) for trial in range(3) for index in range(8))
    order, manifest = s.within_trial_tag_shuffle(events, session=v1.H1_HELDIN_SESSIONS[0], budget=3)
    assert manifest["fixed_points"] == 0
    assert sorted(order.tolist()) == list(range(24))
    assert all(events[index].base.tag != events[int(order[index])].base.tag for index in range(24))
    assert all(events[index].base.trial_index == events[int(order[index])].base.trial_index for index in range(24))


def test_target_closed_form_recovers_unregularized_intercept_and_weights() -> None:
    rng = np.random.default_rng(42)
    latent = rng.normal(size=(32, 4)) * 1.0e8
    weights = rng.normal(size=(4, v1.EXPECTED_NEURONS))
    intercept = rng.normal(size=(v1.EXPECTED_NEURONS,))
    response = latent @ weights + intercept
    old = s.TARGET_RIDGE_LAMBDA
    # Reproduce the algebra locally with an effectively zero penalty by using
    # very large latent magnitudes; exact column ordering is the invariant.
    carrier = s.fit_target_carrier(
        latent,
        response,
        durations=np.ones(32),
        weighted=False,
    )
    assert old == 3.0
    assert carrier.shape == (v1.EXPECTED_NEURONS, 5)
    np.testing.assert_allclose(carrier[:, :4], weights.T, atol=1.0e-12, rtol=1.0e-12)
    np.testing.assert_allclose(carrier[:, -1], intercept, atol=1.0e-6, rtol=0.0)


def test_candidate_matrix_keeps_deployable_arms_five_wide() -> None:
    primary = [candidate for candidate in s.CANDIDATES if not candidate.diagnostic_only]
    diagnostics = [candidate for candidate in s.CANDIDATES if candidate.diagnostic_only]
    assert all(candidate.carrier_dim == 5 for candidate in primary)
    assert all(candidate.carrier_dim == 8 for candidate in diagnostics)
