from __future__ import annotations

import pytest

from h1_causal_representative_activity_v1.plan import ARM_ORDER, DATE_ORDER, decision, dry_plan, selection_for_arm


def test_selection_is_causal_support_anchored_and_exact_cap():
    for output in (4, 5, 29, 30, 31, 80):
        for arm in ARM_ORDER:
            selected = selection_for_arm(arm, output_trial_index=output)
            assert selected[:4] == (0, 1, 2, 3)
            assert len(set(selected)) == len(selected)
            assert max(selected) < output if selected else True
            if arm != "CAUSAL_ALL_PAST":
                assert len(selected) <= 30


def test_coverage_spans_history_while_fifo_tracks_recent_trials():
    coverage = selection_for_arm("CAUSAL_COVERAGE_CAP30", output_trial_index=100)
    fifo = selection_for_arm("CAUSAL_FIFO_CAP30", output_trial_index=100)
    assert coverage[4] == 4 and coverage[-1] == 99
    assert fifo[4:] == tuple(range(74, 100))
    assert coverage != fifo


def _rows(delta: float, positives: int = 5):
    rows = []
    for index, date in enumerate(DATE_ORDER):
        value = delta if index < positives else -0.001
        baseline = 0.4
        rows.append({"outer_date": date, "predecessor": {"CAUSAL_GROWING_CAP30": {"equal_recording_mean_r2": baseline}}, "new_arms": {
            "CAUSAL_FIFO_CAP30": {"equal_recording_mean_r2": baseline},
            "CAUSAL_COVERAGE_CAP30": {"equal_recording_mean_r2": baseline + value},
            "CAUSAL_ALL_PAST": {"equal_recording_mean_r2": baseline + 0.02},
        }})
    return rows


def test_decision_requires_mean_and_four_dates():
    assert decision(_rows(0.02))["pass"] is True
    assert decision(_rows(0.005))["pass"] is False
    assert decision(_rows(0.02, positives=3))["pass"] is False


def test_decision_rejects_arm_reorder():
    rows = _rows(0.02)
    rows[0]["new_arms"] = dict(reversed(tuple(rows[0]["new_arms"].items())))
    with pytest.raises(ValueError, match="arm order"):
        decision(rows)


def test_dry_plan_is_inert_and_predeclares_governing_arm():
    payload = dry_plan()
    assert payload["status"] == "DRY_NO_TARGET_NO_CHECKPOINT_NO_CUDA_NO_WRITE"
    assert payload["governing_arm"] == "CAUSAL_COVERAGE_CAP30"
    assert payload["target_updates"] == 0
