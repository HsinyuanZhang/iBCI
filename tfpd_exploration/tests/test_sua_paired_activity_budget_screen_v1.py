from __future__ import annotations

import numpy as np
import pytest

from tfpd_exploration.src.sua_paired_activity_budget_screen_v1 import core, plan


def test_matrix_is_exactly_three_new_cells_plus_one_reference() -> None:
    assert plan.CELL_ORDER == (
        "ols_m10_activity30_reference",
        "ols_m10_activity10",
        "ridge_m4_activity4",
        "ridge_m4_activity30",
    )
    assert plan.EXPECTED_REFERENCE_ROWS == 90
    assert plan.EXPECTED_NEW_ROWS == 270
    assert plan.EXPECTED_ALIGNED_ROWS == 360
    assert plan.EAGER_CACHED_PARITY_ATOL == 1.0e-5


def test_doptimal_m4_is_rank_three_and_inside_first30() -> None:
    theta = np.tile(np.arange(8, dtype=np.float64) * (np.pi / 4.0), 4)[:30]
    selected = core.select_m4_support(theta)
    assert selected.shape == (4,)
    assert np.all(np.diff(selected) > 0)
    assert int(selected.max()) < 30
    design = np.column_stack((np.cos(theta[selected]), np.sin(theta[selected]), np.ones(4)))
    assert np.linalg.matrix_rank(design) == 3


def test_activity_split_uses_selected_rows_or_exact_first30() -> None:
    calibration = np.arange(30 * 100 * 7, dtype=np.float32).reshape(30, 100, 7)
    selected = np.asarray([1, 8, 17, 25], dtype=np.int64)
    m4 = core.select_activity(calibration, selected_support=selected, activity_budget=4)
    m30 = core.select_activity(calibration, selected_support=selected, activity_budget=30)
    assert np.array_equal(m4, calibration[selected])
    assert np.array_equal(m30, calibration)
    with pytest.raises(core.ScreenError):
        core.select_activity(calibration, selected_support=selected, activity_budget=10)


def test_variance_weighted_r2_matches_pooled_sse_sst() -> None:
    target = np.asarray([[0.0, 1.0], [1.0, 4.0], [3.0, 9.0]], dtype=np.float32)
    prediction = target + np.asarray([[0.1, -0.2], [0.0, 0.1], [-0.1, 0.3]], dtype=np.float32)
    expected = 1.0 - np.square(target - prediction).sum() / np.square(
        target - target.mean(axis=0, keepdims=True)
    ).sum()
    assert core.variance_weighted_r2(target, prediction) == pytest.approx(float(expected))


def _values(offset: float) -> dict[str, dict[int, float]]:
    return {
        f"session_{index:02d}": {seed: offset + index / 100.0 + seed / 10000.0 for seed in plan.SEEDS}
        for index in range(plan.EXPECTED_SESSIONS)
    }


def test_summary_and_pairing_use_sessions_and_seeds_equally() -> None:
    reference = _values(0.0)
    candidate = _values(0.05)
    summary = core.summarize_rows(candidate)
    assert summary["session_count"] == 15
    assert summary["seed_count"] == 3
    contrast = core.paired_contrast(candidate, reference)
    assert contrast["equal_session_seed_mean"] == pytest.approx(0.05)
    assert contrast["positive_sessions_after_seed_average"] == 15
    assert contrast["positive_seed_means"] == 3
    assert contrast["hierarchical_bootstrap_95"]["lower_95"] == pytest.approx(0.05)
    assert contrast["hierarchical_bootstrap_95"]["upper_95"] == pytest.approx(0.05)


def test_array_hash_binds_shape_and_dtype() -> None:
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert core.array_sha256(values) != core.array_sha256(values.reshape(4, 3))
    assert core.array_sha256(values) != core.array_sha256(values.astype(np.float64))
