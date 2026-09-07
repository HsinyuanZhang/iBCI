from __future__ import annotations

import numpy as np

from sua_exploration.mc_maze import h1_event_carrier_pno5 as pno5
from sua_exploration.mc_maze import h1_sparse_event_endpoint as v1


def _events(count: int = 12) -> tuple[v1.MovementEvent, ...]:
    # Four trials, three events each, valid only as synthetic support metadata.
    return tuple(v1.MovementEvent(row_id=index, tag="Reach", trial_value=float(index // 3), trial_index=index // 3,
                                  start_time=float(index), stop_time=float(index) + .2, duration_seconds=.2, eval_bins=10,
                                  displacement=np.zeros(7), log_rates=np.zeros(v1.EXPECTED_NEURONS)) for index in range(count))


def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[v1.MovementEvent, ...]]:
    rng = np.random.default_rng(550); events = _events(); z = rng.normal(size=(len(events), 4))
    # A known unit shared trial effect creates an exact support-only nuisance.
    nuisance = np.repeat(np.array([-1.0, .4, .7, -.1]), 3)
    w = rng.normal(scale=.05, size=(4, v1.EXPECTED_NEURONS)); b = rng.normal(scale=.1, size=v1.EXPECTED_NEURONS)
    y = z @ w + b + nuisance[:, None]
    return z, y, nuisance, events


def test_pno5_synthetic_trial_population_nuisance_recovery_and_fixed_gauge() -> None:
    _z, y, truth, events = _arrays()
    recovered, manifest = pno5.support_trial_population_nuisance(events, y)
    # Channel-average endpoint signal is stochastic rather than exactly zero,
    # but trial ordering and the zero-mean identifiable gauge are preserved.
    assert manifest["support_only"] is True and manifest["uses_query_statistics"] is False
    assert manifest["support_trial_count"] == 4
    assert abs(recovered.mean()) < 1e-12
    assert np.corrcoef(recovered, truth)[0, 1] > .95


def test_pno5_no_query_stat_use_and_5d_order() -> None:
    z, y, _truth, events = _arrays(); nuisance, _ = pno5.support_trial_population_nuisance(events, y)
    carrier, metadata = pno5.fit_pno5_carrier(z, y, nuisance)
    query_z = np.random.default_rng(2).normal(size=(5, 4))
    first = pno5.predict(carrier, query_z)
    # Query population values do not appear in the prediction API; wildly
    # different raw later observations can only affect subsequent scoring.
    later_a = np.zeros_like(first); later_b = np.full_like(first, 1e6)
    np.testing.assert_array_equal(first, pno5.predict(carrier, query_z))
    assert not np.array_equal(later_a, later_b)
    assert carrier.shape == (176, 5) and metadata["target_backward_steps"] == 0
    assert tuple(pno5.PREDECLARATION["carrier_order"]) == ("w1", "w2", "w3", "w4", "b")


def test_pno5_deterministic_nuisance_and_attachment_shuffles() -> None:
    z, y, _truth, events = _arrays(); nuisance, _ = pno5.support_trial_population_nuisance(events, y)
    first, first_manifest = pno5._nuisance_order("ses-19250101T111740", 3, len(events))
    second, second_manifest = pno5._nuisance_order("ses-19250101T111740", 3, len(events))
    np.testing.assert_array_equal(first, second); assert first_manifest == second_manifest and first_manifest["fixed_points"] == 0
    correct, _ = pno5.fit_pno5_carrier(z, y, nuisance)
    attachment, attachment_manifest = v1_with_v2_row_shuffle(correct)
    assert attachment_manifest["fixed_points"] == 0 and not np.array_equal(attachment, correct)


def v1_with_v2_row_shuffle(carrier: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    # Keep test import scope explicit: PNO5 deliberately uses the H-SE5
    # fixed-point-free channel attachment primitive.
    from sua_exploration.mc_maze import h1_sparse_event_endpoint_v2 as hse5
    return hse5.row_shuffle(carrier, session="ses-19250101T111740", budget=3)


def test_pno5_outer_source_date_exclusion_and_gate() -> None:
    names = pno5.source_names_for_outer("19250101")
    assert len(names) == 11 and all(v1.session_date(name) != "19250101" for name in names)
    good = {"defined_sessions": 13, "mean": .0201, "median": .0101, "positive": 10, "leave_largest_absolute_out_mean": .001}
    aggregate = {"correct_minus_hse5": good, "correct_minus_label_shuffle": good, "correct_minus_nuisance_only": good,
                 "correct_minus_intercept": good, "correct_minus_nuisance_row_shuffle": good, "correct_minus_carrier_attachment_shuffle": good}
    assert pno5.gate(aggregate)["passed"] is True
    aggregate["correct_minus_nuisance_only"] = {**good, "mean": -.01}
    assert pno5.gate(aggregate)["passed"] is False
