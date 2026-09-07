"""Unit contracts for native-MUA M1 categorical calibration profile (D4)."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from src.data.falcon_d4_features import (
    calibration_obj_id_labels,
    d4_from_trial_sums,
    deterministic_d4_row_permutation,
    fit_train_d4_stats,
    normalize_d4,
    validate_trial_label_alignment,
)


def _synthetic_support() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Rates (not counts) deliberately vary within labels and have unequal exposure.
    rates = np.asarray([[2.0, 10.0], [4.0, 6.0], [3.0, 5.0], [7.0, 1.0], [9.0, 4.0], [5.0, 8.0]])
    lengths = np.asarray([2, 9, 3, 5, 2, 11])
    labels = np.asarray([1, 1, 2, 3, 4, 4])
    return rates * lengths[:, None], lengths, labels


def test_d4_is_hand_computed_per_trial_exposure_corrected_category_mean() -> None:
    sums, lengths, labels = _synthetic_support()
    actual = d4_from_trial_sums(sums, lengths, labels, source="synthetic")
    # category 1 averages trial rates [2,10] and [4,6], rather than raw counts;
    # category 4 likewise averages [9,4] and [5,8].
    expected = np.asarray([[3.0, 3.0, 7.0, 7.0], [8.0, 5.0, 1.0, 6.0]], dtype=np.float32)
    np.testing.assert_allclose(actual, expected)


def test_d4_unequal_exposure_cannot_be_replaced_by_pooled_count_rate() -> None:
    sums, lengths, labels = _synthetic_support()
    actual = d4_from_trial_sums(sums, lengths, labels, source="unequal-exposure")
    pooled_label_1 = sums[labels == 1].sum(axis=0) / lengths[labels == 1].sum()
    assert not np.allclose(actual[:, 0], pooled_label_1)
    np.testing.assert_allclose(actual[:, 0], [3.0, 8.0])


@pytest.mark.parametrize("labels", [np.asarray([1, 1, 2, 3]), np.asarray([1, 2, 3, 5]), np.asarray([1, 2, 3, np.nan])])
def test_d4_fails_closed_on_missing_or_unknown_labels(labels: np.ndarray) -> None:
    sums = np.ones((len(labels), 2))
    lengths = np.ones(len(labels))
    with pytest.raises(ValueError, match="D4.*(levels|integers|outside)"):
        d4_from_trial_sums(sums, lengths, labels, source="bad-labels")


def test_d4_rejects_non_m1_task_before_reading_any_nwb() -> None:
    with pytest.raises(ValueError, match="M1 only"):
        calibration_obj_id_labels(Path("must-not-be-opened.nwb"), "m2")


def test_d4_trial_boundary_label_alignment_fails_closed() -> None:
    with pytest.raises(ValueError, match="cannot be aligned"):
        validate_trial_label_alignment(
            np.asarray([True, False, True, False]), np.asarray([1, 2, 3, 4]), source="mismatch"
        )


def test_d4_train_only_normalization_and_chronological_first_ten() -> None:
    sums, lengths, labels = _synthetic_support()
    # Repeat to make ten chronological support trials; first ten have each level.
    sums10 = np.concatenate([sums, sums[:4]], axis=0)
    lengths10 = np.concatenate([lengths, lengths[:4]])
    labels10 = np.concatenate([labels, labels[:4]])
    # An eleventh poisoned trial must not affect frozen chronological [0:10].
    poisoned_sums = np.concatenate([sums10, np.asarray([[9_999.0, 8_888.0]])])
    poisoned_lengths = np.concatenate([lengths10, np.asarray([1])])
    poisoned_labels = np.concatenate([labels10, np.asarray([4])])
    shifted = sums10 + lengths10[:, None] * np.asarray([10.0, -3.0])
    mean, std = fit_train_d4_stats(
        {"train-a": poisoned_sums}, {"train-a": poisoned_lengths},
        {"train-a": poisoned_labels}, ["train-a"], calibration_n_trials=10,
    )
    mean_with_unlisted, std_with_unlisted = fit_train_d4_stats(
        {"train-a": poisoned_sums, "train-b": shifted}, {"train-a": poisoned_lengths, "train-b": lengths10},
        {"train-a": poisoned_labels, "train-b": labels10}, ["train-a"], calibration_n_trials=10,
    )
    np.testing.assert_allclose(mean_with_unlisted, mean)
    np.testing.assert_allclose(std_with_unlisted, std)
    raw_train = d4_from_trial_sums(sums10, lengths10, labels10, source="train")
    np.testing.assert_allclose(mean, raw_train.mean(axis=0))
    np.testing.assert_allclose(std, np.where(raw_train.std(axis=0) <= 1.0e-6, 1.0, raw_train.std(axis=0)))
    np.testing.assert_allclose(normalize_d4(raw_train, mean, std).mean(axis=0), np.zeros(4), atol=1e-6)
    with pytest.raises(ValueError, match="frozen"):
        fit_train_d4_stats({"train-a": sums10}, {"train-a": lengths10}, {"train-a": labels10}, ["train-a"], calibration_n_trials=40)
    with pytest.raises(ValueError, match="forbids"):
        fit_train_d4_stats({"train-a": sums10}, {"train-a": lengths10}, {"train-a": labels10}, ["train-a"], calibration_n_trials=10, all_support_windows=True)


def test_ds4_is_deterministic_nonidentity_complete_row_permutation() -> None:
    values = np.arange(24, dtype=np.float32).reshape(6, 4)
    first = deterministic_d4_row_permutation(6, session_name="20120924", seed=42)
    second = deterministic_d4_row_permutation(6, session_name="20120924", seed=42)
    np.testing.assert_array_equal(first, second)
    assert sorted(first.tolist()) == list(range(6))
    assert not np.array_equal(first, np.arange(6))
    shuffled = values[first]
    np.testing.assert_array_equal(np.sort(shuffled, axis=0), np.sort(values, axis=0))
    assert not np.array_equal(shuffled, values)


def test_d4_has_no_spike_time_searchsorted_path() -> None:
    source = inspect.getsource(d4_from_trial_sums)
    assert "searchsorted" not in source
