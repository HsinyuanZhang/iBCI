"""Fail-closed tests for the M1 held-in-calib post-support replay plumbing."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "SPINT-main/data"

requires_falcon_data = pytest.mark.skipif(
    not (DATA / "000941").is_dir(),
    reason="local FALCON M1 dandiset is not present",
)


def _import_datamodule():
    import sys

    sys.path.insert(0, str(ROOT / "streaming_calibration_exp"))
    from src.data.falcon_datamodule import FalconDataModule

    return FalconDataModule


def _module(**overrides):
    FalconDataModule = _import_datamodule()
    base = dict(
        task="m1",
        data_dir=str(DATA / "000941") + "/",
        window_size=100,
        calibration_n_trials=10,
        random_calibration=False,
        smooth_calibration=False,
        max_trial_length=1024,
        use_intertrials=True,
        use_calib_intertrials=False,
        trial_feature_type="raw",
        interpolate_trials=True,
        interpolate_trials_kind="cubic",
        pad_value=-1.0,
        validation_protocol="loso",
        include_heldout_in_fit=False,
        include_heldout_in_test=False,
        num_workers=0,
        side_feature_group="none",
        heldin_query_start_trial=10,
    )
    base.update(overrides)
    dm = FalconDataModule(**base)
    trainer = MagicMock()
    trainer.world_size = 1
    dm.trainer = trainer
    return dm


@pytest.mark.parametrize(
    "override",
    [
        {"task": "m2"},
        {"validation_protocol": "minival"},
        {"calibration_n_trials": 24},
        {"task": "m2", "heldin_query_start_trial": 10},
        {"random_calibration": True},
        {"include_heldout_in_fit": True},
    ],
)
def test_heldin_query_start_rejects_every_non_correction_path(override):
    with pytest.raises(ValueError, match="heldin_query_start_trial/heldin_query_end_trial are reserved"):
        _module(**override).setup(stage="fit")


@requires_falcon_data
def test_default_path_uses_minival_and_matches_historical_window_counts():
    dm = _module(heldin_query_start_trial=0, loso_fold=1)
    dm.setup(stage="fit")
    assert len(dm.val_heldin_dataset) in {255, 256}
    session = dm.val_heldin_session_names[0]
    audit = dm.val_heldin_dataset.query_window_audit[session]
    assert audit["query_start_trial"] == 0
    assert audit["support_trials"] == 0
    assert audit["full_window_disjoint"] is False
    assert audit["total_trials"] == 2


@requires_falcon_data
@pytest.mark.parametrize("fold, expected_query_trials", [(1, 399), (2, 366)])
def test_guarded_protocol_scores_hundreds_of_windows_from_calib(fold, expected_query_trials):
    dm = _module(loso_fold=fold)
    dm.setup(stage="fit")
    session = dm.val_heldin_session_names[0]
    audit = dm.val_heldin_dataset.query_window_audit[session]
    assert len(dm.val_heldin_dataset) > 300
    assert audit["query_trials"] == expected_query_trials
    assert audit["query_start_trial"] == 10
    assert audit["support_trials"] == 10
    assert audit["full_window_disjoint"] is True
    assert audit["minimum_window_start_padded_bin"] == audit["raw_query_start_bin"] + 99


@requires_falcon_data
def test_guarded_protocol_respects_support_query_boundary():
    dm = _module(loso_fold=1)
    dm.setup(stage="fit")
    session = dm.val_heldin_session_names[0]
    audit = dm.val_heldin_dataset.query_window_audit[session]
    assert audit["minimum_window_start_padded_bin"] >= audit["raw_query_start_bin"] + dm.hparams.window_size - 1


@requires_falcon_data
@pytest.mark.parametrize("fold", [1, 2])
def test_left_out_session_is_absent_from_training(fold):
    dm = _module(loso_fold=fold)
    dm.setup(stage="fit")
    left_out = dm.val_heldin_session_names
    assert len(left_out) == 1
    assert left_out[0] not in dm.train_session_names


@requires_falcon_data
def test_default_heldin_query_start_is_zero():
    FalconDataModule = _import_datamodule()
    dm = FalconDataModule(task="m1", data_dir="/unused")
    assert dm.hparams.heldin_query_start_trial == 0
