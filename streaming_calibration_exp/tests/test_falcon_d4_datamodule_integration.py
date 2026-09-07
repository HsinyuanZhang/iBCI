"""Integration contract for D4/DS4 through the native FALCON dataset path."""
from __future__ import annotations

from collections import Counter

import numpy as np

from src.data.falcon_datamodule import FalconDataset


def _session() -> dict[str, np.ndarray]:
    # Ten two-bin trials, all four required labels, and three channels so a
    # DS4 permutation can be checked without depending on an NWB fixture.
    trial_change = np.zeros(20, dtype=bool)
    trial_change[::2] = True
    neural = np.arange(60, dtype=np.float32).reshape(20, 3) + 1.0
    return {
        "neural": neural,
        "covariates": np.zeros((20, 2), dtype=np.float32),
        "trial_change": trial_change,
        "eval_mask": np.ones(20, dtype=bool),
        "trial_obj_ids": np.asarray([1, 2, 3, 4, 1, 2, 3, 4, 1, 2], dtype=np.int64),
    }


def _dataset(group: str) -> FalconDataset:
    session = _session()
    dataset = FalconDataset(
        sessions_dict={"synthetic": session},
        calib_sessions_dict={"synthetic": session},
        window_size=1,
        calibration_n_trials=10,
        random_calibration=False,
        smooth_calibration=False,
        max_trial_length=2,
        use_calib_intertrials=True,
        side_feature_group=group,
        side_feature_shuffle_seed=42,
    )
    dataset.set_native_d4_normalization(np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32))
    return dataset


def test_dataset_wires_d4_and_ds4_after_train_only_normalization() -> None:
    d4 = _dataset("d4")
    ds4 = _dataset("ds4")
    d4_side = np.asarray(d4[0][-1])
    ds4_side = np.asarray(ds4[0][-1])

    assert d4_side.shape == (3, 4)
    assert np.isfinite(d4_side).all()
    assert not np.array_equal(d4_side, ds4_side)
    assert Counter(map(tuple, d4_side.tolist())) == Counter(map(tuple, ds4_side.tolist()))
    # Dataset-level enforcement prevents random support or a different M from
    # silently entering the frozen chronological M=10 profile.
    with np.testing.assert_raises_regex(ValueError, "chronological calibration trials\\[0:10\\]"):
        d4._native_d4_side_features("synthetic", 1, 10)
    with np.testing.assert_raises_regex(ValueError, "chronological calibration trials\\[0:10\\]"):
        d4._native_d4_side_features("synthetic", 0, 9)
