from __future__ import annotations

import numpy as np
import pytest

from tfpd_exploration.src.m2_t4_activity_budget_screen_v1 import core
from tfpd_exploration.src.m2_t4_activity_budget_screen_v1 import physical


def test_cell_order_and_semantics() -> None:
    assert [cell.name for cell in core.CELL_SPECS] == [
        "ridge_static_m30",
        "ridge_static_m10",
        "ridge_activity30_m10",
        "ridge_static_m4",
        "ridge_activity30_m4",
    ]
    assert [(cell.carrier_budget, cell.activity_budget) for cell in core.CELL_SPECS] == [
        (30, 30), (10, 10), (10, 30), (4, 4), (4, 30)
    ]


def test_selected_support_and_first30_activity_split() -> None:
    values = np.arange(35 * 100 * 96, dtype=np.float32).reshape(35, 100, 96)
    selected = (1, 7, 16, 29)
    static = core.select_activity_rows(
        values, selected_indices=selected, activity_budget=4
    )
    activity = core.select_activity_rows(
        values, selected_indices=selected, activity_budget=30
    )
    assert static.shape == (4, 100, 96)
    assert activity.shape == (30, 100, 96)
    assert np.array_equal(static, values[list(selected)])
    assert np.array_equal(activity, values[:30])
    with pytest.raises(core.ScreenError):
        core.select_activity_rows(values, selected_indices=range(10), activity_budget=4)


def test_post30_selection_requires_full_window_start_after_boundary() -> None:
    trial_starts = np.arange(40, dtype=np.int64) * 100 + 49
    windows = np.asarray([3000, 3048, 3049, 3050, 3200], dtype=np.int64)
    selected = core.select_common_post30_window_starts(windows, trial_starts)
    assert selected.tolist() == [3049, 3050, 3200]
    with pytest.raises(core.ScreenError):
        core.select_common_post30_window_starts([1, 1], trial_starts)


def test_variance_weighted_r2_matches_manual_pooled_sse_sst() -> None:
    target = np.asarray([[0.0, 2.0], [1.0, 4.0], [2.0, 8.0]], dtype=np.float32)
    prediction = target + np.asarray([[0.1, -0.2], [0.0, 0.1], [-0.1, 0.2]], dtype=np.float32)
    sse = np.square(target.astype(np.float64) - prediction.astype(np.float64)).sum()
    sst = np.square(target.astype(np.float64) - target.mean(axis=0, keepdims=True)).sum()
    assert core.variance_weighted_r2(target, prediction) == pytest.approx(1.0 - sse / sst)


def test_paired_summary_is_equal_session_not_pooled() -> None:
    candidate = {"b": 0.4, "a": 0.1, "c": 0.9}
    reference = {"a": 0.0, "b": 0.5, "c": 0.4}
    contrast = core.paired_contrast(candidate, reference)
    assert contrast["positive_sessions"] == 2
    assert contrast["candidate_minus_reference_mean"] == pytest.approx((0.1 - 0.1 + 0.5) / 3)
    summary = core.summarize_sessions(candidate)
    assert summary["equal_session_mean"] == pytest.approx((0.1 + 0.4 + 0.9) / 3)


def test_ridge_helper_accepts_doptimal_m4_indices() -> None:
    class Dataset:
        pass

    dataset = Dataset()
    session = "synthetic"
    theta = np.asarray([0.0, np.pi / 2, np.pi, -np.pi / 2], dtype=np.float32)
    scale = np.linspace(0.1, 1.0, 96, dtype=np.float32)
    rates = (
        2.0
        + np.cos(theta)[:, None] * scale[None, :]
        + np.sin(theta)[:, None] * (0.5 * scale[None, :])
    ).astype(np.float32)
    dataset.calib_trial_spike_sums = {session: rates * 100.0}
    dataset.calib_trial_lengths = {session: np.full(4, 100, dtype=np.int64)}
    dataset.calib_trial_target_angles = {session: theta}
    dataset.side_feature_mean = np.zeros(4, dtype=np.float32)
    dataset.side_feature_std = np.ones(4, dtype=np.float32)
    side, evidence = physical._ridge_side(
        dataset, session, np.arange(4, dtype=np.int64)
    )
    assert side.shape == (96, 4)
    assert evidence["budget"] == 4
    assert evidence["usable_directional_trials"] == 4
