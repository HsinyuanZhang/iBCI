from __future__ import annotations

import numpy as np

from sua_exploration.mc_maze import h1_event_carrier_design_screen as v1screen
from sua_exploration.mc_maze import h1_event_carrier_nested_context_v2 as v2
from sua_exploration.mc_maze import h1_sparse_event_endpoint as event_v1


def event(index: int, trial: int) -> v1screen.ContextEvent:
    delta = np.linspace(0.0, 1.0, 7) + index
    base = event_v1.MovementEvent(
        row_id=index,
        tag=event_v1.MOVEMENT_TAGS[index % 8],
        trial_value=float(trial + 1),
        trial_index=trial,
        start_time=float(index),
        stop_time=float(index) + 1.0,
        duration_seconds=1.0,
        eval_bins=50,
        displacement=delta,
        log_rates=np.full(176, index, dtype=np.float64),
    )
    start = np.arange(7, dtype=np.float64) + trial
    return v1screen.ContextEvent(base=base, start_state=start, midpoint_state=start + delta / 2.0)


def test_nested_matrix_is_fixed_and_unique() -> None:
    names = [configuration.name for configuration in v2.CONFIGURATIONS]
    assert len(names) == len(set(names)) == 11
    assert names[0] == "context_mid_anchor"
    assert all(configuration.source_ridge > 0 and configuration.target_ridge > 0 for configuration in v2.CONFIGURATIONS)


def test_context_variants_are_sparse_endpoint_reparameterizations() -> None:
    events = tuple(event(index, 0) for index in range(8))
    assert v2.raw_features(events, "context_mid").shape == (8, 22)
    assert v2.raw_features(events, "context_start").shape == (8, 22)
    assert v2.raw_features(events, "context_stop").shape == (8, 22)
    assert v2.raw_features(events, "tag_delta").shape == (8, 64)
    np.testing.assert_allclose(
        v2.raw_features(events, "context_mid"),
        v1screen.raw_features(events, "context"),
    )


def test_tag_override_does_not_change_endpoint_coordinates() -> None:
    events = tuple(event(index, 0) for index in range(8))
    correct = v2.raw_features(events, "context_mid")
    wrong_tags = tuple(reversed(event_v1.MOVEMENT_TAGS))
    wrong = v2.raw_features(events, "context_mid", tag_overrides=wrong_tags)
    np.testing.assert_array_equal(correct[:, :14], wrong[:, :14])
    assert not np.array_equal(correct[:, 14:], wrong[:, 14:])
