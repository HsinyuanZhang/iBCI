"""Synthetic contracts for the RT-specific AFC4 event and null semantics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.falcon_datamodule import (
    SessionBatchSampler,
    mask_normalized_k4_components,
)
from src.data.falcon_k4_features import k4_from_raw_calibration
from src.data.rt_k4_loader import build_rt_reach_segments, summarize_rt_trial_budget


def test_rt_go_cues_are_fail_closed_and_whole_bin_qualified() -> None:
    starts = np.asarray([0.0, 1.0, 2.0])
    stops = np.asarray([1.0, 2.0, 3.0])
    cues = np.asarray(
        [
            [0.11, 0.45, 0.70, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
            [2.04, 2.33, 2.68, np.nan],
        ]
    )
    trial_change, segment_ids, eval_mask, audit = build_rt_reach_segments(
        n_bins=150,
        trial_start_times=starts,
        trial_stop_times=stops,
        go_cue_time_array=cues,
        num_targets=np.asarray([3, 4, 3]),
        velocity_bin_valid=np.ones(150, dtype=bool),
    )

    # The first cue at 110 ms maps to bin 6 (120 ms); bins touching the cue
    # are not silently claimed by the reach.
    assert trial_change[[0, 50, 100]].all()
    assert np.all(segment_ids[:6] == -1)
    assert not eval_mask[:6].any()
    assert audit["complete_cue_trials"] == 2
    assert audit["accepted_reach_segments"] == 6
    assert audit["excluded_trials"] == 1
    assert audit["trial_exclusion_reasons"] == {"nonfinite_required_go_cue": 1}
    assert np.all(segment_ids[eval_mask] >= 0)

    support = summarize_rt_trial_budget(audit, budget_trials=3)
    assert support["budget_trials"] == 3
    assert support["complete_cue_trials_within_budget"] == 2
    assert support["accepted_reach_segments_within_budget"] == 6
    assert support["excluded_trials_within_budget"] == 1


def _segmented_raw_support() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """M24 support with two accepted 200-ms reaches per raw trial."""
    rng = np.random.RandomState(7)
    trials, bins_per_trial, channels = 24, 20, 3
    total = trials * bins_per_trial
    neural = np.zeros((total, channels), dtype=np.float64)
    covariates = np.zeros((total, 2), dtype=np.float64)
    trial_change = np.zeros(total, dtype=bool)
    segment_ids = np.full(total, -1, dtype=np.int64)
    weights = np.asarray([[1.5, -0.7], [0.2, 1.1], [-0.8, 0.5]])
    baseline = np.asarray([7.0, 4.0, 5.5])
    for trial in range(trials):
        base = trial * bins_per_trial
        trial_change[base] = True
        for reach in range(2):
            left = base + reach * 10
            right = left + 10
            segment_ids[left:right] = trial * 2 + reach
            velocity = rng.uniform(0.25, 2.0, size=2)
            covariates[left:right] = velocity
            # Five raw 20-ms bins make one rate measurement.  Each accepted
            # reach admits exactly its first block; the block at +100 ms would
            # cross the reach boundary after the +40-ms behaviour lead.
            rate = baseline + weights @ velocity
            neural[left:left + 5] = rate * 0.02
    return neural, covariates, trial_change, segment_ids


def test_rt_k4_segments_block_crossings_and_label_null_changes_pairing() -> None:
    neural, covariates, trial_change, segment_ids = _segmented_raw_support()
    aligned, aligned_audit = k4_from_raw_calibration(
        neural,
        covariates,
        trial_change,
        calibration_n_trials=24,
        segment_ids=segment_ids,
    )
    shuffled, null_audit = k4_from_raw_calibration(
        neural,
        covariates,
        trial_change,
        calibration_n_trials=24,
        segment_ids=segment_ids,
        label_shuffle=True,
        label_shuffle_seed=42,
        label_shuffle_session_name="synthetic-rt",
    )

    assert aligned_audit.active_blocks == 48
    assert aligned_audit.extra is not None
    assert aligned_audit.extra["candidate_trial_bounded_blocks"] == 72
    assert aligned_audit.extra["segment_qualified_blocks"] == 48
    assert aligned_audit.extra["segment_constrained"] is True
    assert null_audit.extra is not None
    assert null_audit.extra["label_shuffle"] is True
    assert null_audit.extra["label_changed_blocks"] == 48
    assert np.max(np.abs(aligned - shuffled)) > 1.0e-4


def test_rt_afc4_component_ablations_mask_only_normalized_coordinates() -> None:
    """Component arms share the full AFC4 normalizer and preserve unit order."""
    raw = np.asarray(
        [[12.0, -3.0, 5.0, 20.0], [8.0, 7.0, 2.0, 10.0], [4.0, 1.0, 9.0, 30.0]],
        dtype=np.float32,
    )
    mean = np.asarray([2.0, -1.0, 1.0, 10.0], dtype=np.float32)
    std = np.asarray([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    normalized = (raw - mean) / std

    mb4 = mask_normalized_k4_components(normalized, "afc4_mb4")
    b4 = mask_normalized_k4_components(normalized, "afc4_b4")
    w4 = mask_normalized_k4_components(normalized, "afc4_w4")

    np.testing.assert_array_equal(mb4[:, :2], np.zeros((3, 2), dtype=np.float32))
    np.testing.assert_array_equal(mb4[:, 2:], normalized[:, 2:])
    np.testing.assert_array_equal(b4[:, :3], np.zeros((3, 3), dtype=np.float32))
    np.testing.assert_array_equal(b4[:, 3], normalized[:, 3])
    np.testing.assert_array_equal(w4[:, :2], normalized[:, :2])
    np.testing.assert_array_equal(w4[:, 2:], np.zeros((3, 2), dtype=np.float32))


def test_rt_afc4_component_masks_preserve_variable_unit_axis_and_legacy_groups() -> None:
    for units in (1, 57, 88):
        normalized = np.arange(units * 4, dtype=np.float32).reshape(units, 4)
        for group in ("afc4_mb4", "afc4_b4", "afc4_w4"):
            assert mask_normalized_k4_components(normalized, group).shape == (units, 4)
        # Existing M2 aliases and the completed RT arms must remain bitwise
        # identity transforms; only the new opt-in groups are altered.
        for group in ("k4", "ks4", "k4ls", "afc4_vel", "afc4_rs", "afc4_ls"):
            np.testing.assert_array_equal(
                mask_normalized_k4_components(normalized, group), normalized
            )


@dataclass
class _Dataset:
    window_indices: list[tuple[str, int]]


def test_exact_session_window_budget_is_equal_without_window_recycling() -> None:
    dataset = _Dataset(
        [("a", index) for index in range(12)]
        + [("b", index) for index in range(18)]
        + [("c", index) for index in range(24)]
    )
    sampler = SessionBatchSampler(
        dataset,
        batch_size=3,
        shuffle=True,
        seed=42,
        window_budget_per_session=6,
        require_full_window_budget=True,
        reshuffle_each_epoch=True,
    )
    assert sampler.session_batch_counts == {"a": 2, "b": 2, "c": 2}
    assert len(sampler) == 6
    first_epoch = list(sampler)
    assert all(len(batch) == 3 for batch in first_epoch)
    assert all(
        len({dataset.window_indices[index][0] for index in batch}) == 1
        for batch in first_epoch
    )
