"""Focused contracts for the frozen M2 K4/KS4 Gate-B estimator."""
from __future__ import annotations

from pathlib import Path
from collections import OrderedDict

import numpy as np
import pytest

from src.data.falcon_datamodule import FalconDataModule, FalconDataset
from src.data.falcon_k4_features import (
    K4_CALIBRATION_TRIALS,
    deterministic_k4_row_permutation,
    fit_train_k4_stats,
    k4_from_raw_calibration,
)


def _synthetic_raw_m2() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """33 raw trials with one valid 100-ms block each and known K4 weights."""
    rng = np.random.RandomState(2)
    trials, bins_per_trial, channels = K4_CALIBRATION_TRIALS, 10, 3
    total = trials * bins_per_trial
    trial_change = np.zeros(total, dtype=bool)
    trial_change[::bins_per_trial] = True
    y_trial = rng.uniform(0.25, 2.0, size=(trials, 2))
    covariates = np.repeat(y_trial, bins_per_trial, axis=0)
    weights = np.asarray([[2.0, -1.0], [-0.5, 3.0], [1.5, 0.25]])
    baseline = np.asarray([8.0, 5.0, 2.0])
    rates = y_trial @ weights.T + baseline
    # A 100-ms block has five 20-ms bins, so each raw bin receives rate*0.02.
    neural = np.repeat((rates * 0.02), bins_per_trial, axis=0)
    return neural, covariates, trial_change, weights, baseline


def test_k4_recovers_known_movement_aligned_weights_without_trial_average() -> None:
    neural, covariates, trial_change, weights, baseline = _synthetic_raw_m2()
    features, audit = k4_from_raw_calibration(neural, covariates, trial_change)
    np.testing.assert_allclose(features[:, :2], weights, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(features[:, 2], np.linalg.norm(weights, axis=1), rtol=1e-5)
    np.testing.assert_allclose(features[:, 3], baseline, rtol=1e-5, atol=1e-5)
    assert audit.active_blocks == K4_CALIBRATION_TRIALS
    assert audit.max_trial_length_used is False
    assert audit.design_rank == 3
    assert np.isfinite(audit.design_condition)
    m24_features, m24_audit = k4_from_raw_calibration(
        neural, covariates, trial_change, calibration_n_trials=24
    )
    np.testing.assert_allclose(m24_features[:, :2], weights, rtol=1e-5, atol=1e-5)
    assert m24_audit.calibration_trials == 24


def test_default_m2_k4_path_is_bitwise_identical_to_legacy_trial_only_estimator() -> None:
    """RT segment/null options must not alter the pre-existing M2 default path."""
    neural, covariates, trial_change, _weights, _baseline = _synthetic_raw_m2()
    actual, audit = k4_from_raw_calibration(neural, covariates, trial_change)

    # This is the original trial-bounded implementation written out locally,
    # rather than a call through the extended estimator.  It fixes the order of
    # raw blocks, active filtering, least squares and descriptor assembly.
    starts = np.flatnonzero(trial_change)
    ends = np.r_[starts[1:], len(trial_change)]
    active = ~np.all(np.abs(covariates) < 1.0e-3, axis=1)
    rates, behavior = [], []
    for start, end in zip(starts[:K4_CALIBRATION_TRIALS], ends[:K4_CALIBRATION_TRIALS]):
        for left in range(int(start), int(end) - 5 - 2 + 1, 5):
            right = left + 5
            y_left, y_right = left + 2, left + 2 + 5
            if not (active[left:right].all() and active[y_left:y_right].all()):
                continue
            rates.append(neural[left:right].sum(axis=0) / 0.1)
            behavior.append(covariates[y_left:y_right].mean(axis=0))
    design = np.column_stack([np.ones(len(behavior)), np.asarray(behavior)])
    coefficients, *_ = np.linalg.lstsq(design, np.asarray(rates), rcond=None)
    expected = np.column_stack(
        [
            coefficients[1:, :].T[:, 0],
            coefficients[1:, :].T[:, 1],
            np.linalg.norm(coefficients[1:, :].T, axis=1),
            coefficients[0],
        ]
    ).astype(np.float32)

    np.testing.assert_array_equal(actual, expected)
    assert set(audit.as_dict()) == {
        "calibration_trials",
        "active_blocks",
        "raw_bin_ms",
        "block_width_bins",
        "behavior_lead_bins",
        "active_rule",
        "max_trial_length_used",
        "design_rank",
        "design_condition",
    }


def test_k4_refuses_nonfrozen_support_and_stationary_calibration() -> None:
    neural, covariates, trial_change, _weights, _baseline = _synthetic_raw_m2()
    with pytest.raises(ValueError, match="supports only"):
        k4_from_raw_calibration(neural, covariates, trial_change, calibration_n_trials=20)
    with pytest.raises(ValueError, match="fewer than three"):
        k4_from_raw_calibration(neural, np.zeros_like(covariates), trial_change)
    rank_one = covariates.copy()
    rank_one[:, 1] = rank_one[:, 0]
    with pytest.raises(ValueError, match="rank=3"):
        k4_from_raw_calibration(neural, rank_one, trial_change)


def test_ks4_permutation_is_deterministic_full_row_nonidentity() -> None:
    first = deterministic_k4_row_permutation(12, session_name="session-a", seed=42)
    second = deterministic_k4_row_permutation(12, session_name="session-a", seed=42)
    np.testing.assert_array_equal(first, second)
    assert sorted(first.tolist()) == list(range(12))
    assert not np.array_equal(first, np.arange(12))


def test_train_k4_statistics_require_only_named_train_sessions() -> None:
    values = {"a": np.ones((2, 4), dtype=np.float32), "b": np.full((3, 4), 3.0, dtype=np.float32)}
    mean, std = fit_train_k4_stats(values, ["a", "b"])
    np.testing.assert_allclose(mean, 2.2)
    assert np.all(std > 0)
    with pytest.raises(ValueError, match="Missing"):
        fit_train_k4_stats(values, ["a", "missing"])


def test_k4_datamodule_uses_heldin_raw_calibration_only() -> None:
    data_dir = Path(__file__).resolve().parents[2] / "SPINT-main" / "data" / "000953"
    if not data_dir.is_dir():
        pytest.skip(f"FALCON data unavailable: {data_dir}")
    dm = FalconDataModule(
        task="m2", data_dir=str(data_dir), heldin_session_names=[""], batch_size=2,
        window_size=50, calibration_n_trials=33, random_calibration=False,
        smooth_calibration=False, max_trial_length=100, use_intertrials=True,
        use_calib_intertrials=False, interpolate_trials=True, interpolate_trials_kind="cubic",
        validation_protocol="loso", loso_fold=1, include_heldout_in_fit=False,
        include_heldout_in_test=False, num_workers=0, pin_memory=False,
        side_feature_group="k4", side_feature_shuffle_seed=42,
    )
    dm.setup("fit")
    assert dm.val_heldout_dataset is None
    batch = next(iter(dm.val_dataloader()))
    assert len(batch) == 5
    assert tuple(batch[4].shape) == (2, 96, 4)
    assert dm.native_k4_normalization["feature_group"] == "k4"
    manifest = dm.get_split_manifest()
    assert manifest["heldout_evaluated_in_fit"] is False
    assert manifest["native_k4_normalization"]["train_sessions"] == dm.train_session_names
    assert manifest["k4_estimator"]["raw_full_trial_no_interpolation_no_max_trial_length_cap"] is True


def test_k4_datamodule_refuses_heldout_or_nonraw_config() -> None:
    for kwargs, match in [
        ({"include_heldout_in_test": True}, "only in setup\\(stage='test'\\)"),
        ({"smooth_calibration": True}, "requires smooth_calibration=false"),
        ({"standardize_covariates": True}, "requires standardize_covariates=false"),
        ({"use_intertrials": False}, "requires use_intertrials=true"),
        ({"remove_calib_still_times": True}, "requires remove_calib_still_times=false"),
    ]:
        config = {
            "task": "m2", "data_dir": "unused", "side_feature_group": "k4",
            "calibration_n_trials": 33, "random_calibration": False,
            "smooth_calibration": False, "standardize_covariates": False,
            "use_intertrials": True, "remove_calib_still_times": False,
        }
        config.update(kwargs)
        dm = FalconDataModule(**config)
        with pytest.raises(ValueError, match=match):
            dm.setup("fit")


def test_query_start_trial_excludes_entire_temporal_history_window() -> None:
    """Disjoint local replay must not let a query history touch support bins."""
    trial_change = np.zeros(60, dtype=bool)
    trial_change[[0, 20, 40]] = True
    session = {
        "neural": np.zeros((60, 3), dtype=np.float32),
        "covariates": np.ones((60, 2), dtype=np.float32),
        "eval_mask": np.ones(60, dtype=bool),
        "trial_change": trial_change,
    }
    dataset = FalconDataset(
        OrderedDict({"s": session}), OrderedDict({"s": session}),
        window_size=5, calibration_n_trials=1, random_calibration=False,
        smooth_calibration=False, query_start_trial=1,
    )
    # Four bins of history padding precede raw bin zero, so trial 1 starts at 24.  A
    # window ending after that boundary is not sufficient: every window must
    # itself start at/after it.
    audit = dataset.query_window_audit["s"]
    assert audit["minimum_window_start_padded_bin"] == 24
    assert audit["raw_query_start_bin"] == 20
    assert audit["total_trials"] == 3
    assert audit["support_trials"] == audit["query_start_trial"] == 1
    assert audit["query_trials"] == 2
    assert audit["window_size"] == 5
    assert audit["full_window_disjoint"] is True
    assert audit["eligible_windows"] > 0
    assert all(start >= 24 for _session, start in dataset.window_indices)
