"""Fail-closed tests for bounded M1 held-in query windows (clean selection protocol)."""
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

SUPPORT_END = 10
SELECTION_END = 210
REPORT_START = 210


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
        heldin_query_start_trial=SUPPORT_END,
        heldin_query_end_trial=SELECTION_END,
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
        {"random_calibration": True},
        {"include_heldout_in_fit": True},
        {"heldin_query_end_trial": 5},
        {"heldin_query_start_trial": 210, "heldin_query_end_trial": 210},
        {"heldin_query_start_trial": 11, "heldin_query_end_trial": None},
        {"heldin_query_start_trial": 50, "heldin_query_end_trial": 210},
        {"heldin_query_start_trial": 10, "heldin_query_end_trial": 100},
    ],
)
def test_bounded_heldin_query_rejects_invalid_paths(override):
    with pytest.raises(ValueError):
        _module(**override).setup(stage="fit")


def test_heldin_query_start_without_m1_protocol_still_rejects():
    with pytest.raises(ValueError, match="heldin_query_start_trial/heldin_query_end_trial are reserved"):
        _module(task="m2", heldin_query_start_trial=11, heldin_query_end_trial=None).setup(stage="fit")


def test_only_predeclared_m1_window_pairs_are_allowed():
    """Arbitrary offsets under an otherwise valid M1 LOSO config must fail closed."""
    with pytest.raises(ValueError, match=r"allowed window pairs"):
        _module(heldin_query_start_trial=11, heldin_query_end_trial=None).setup(stage="fit")


@requires_falcon_data
def test_default_path_uses_minival_and_matches_historical_window_counts():
    dm = _module(heldin_query_start_trial=0, heldin_query_end_trial=None, loso_fold=1)
    dm.setup(stage="fit")
    assert len(dm.val_heldin_dataset) in {255, 256}
    session = dm.val_heldin_session_names[0]
    audit = dm.val_heldin_dataset.query_window_audit[session]
    assert audit["query_start_trial"] == 0
    assert audit.get("query_end_trial") is None
    assert audit["support_trials"] == 0
    assert audit["full_window_disjoint"] is False
    assert audit["total_trials"] == 2


@requires_falcon_data
@pytest.mark.parametrize(
    "fold, expected_selection_trials, expected_report_trials",
    [
        (1, 200, 199),
        (2, 200, 166),
    ],
)
def test_selection_and_report_window_trial_counts(fold, expected_selection_trials, expected_report_trials):
    dm_sel = _module(loso_fold=fold)
    dm_sel.setup(stage="fit")
    session = dm_sel.val_heldin_session_names[0]
    sel_audit = dm_sel.val_heldin_dataset.query_window_audit[session]
    assert sel_audit["query_start_trial"] == SUPPORT_END
    assert sel_audit["query_end_trial"] == SELECTION_END
    assert sel_audit["query_trials"] == expected_selection_trials
    assert sel_audit["full_window_disjoint"] is True

    dm_rep = _module(
        loso_fold=fold,
        heldin_query_start_trial=REPORT_START,
        heldin_query_end_trial=None,
    )
    dm_rep.setup(stage="fit")
    rep_audit = dm_rep.val_heldin_dataset.query_window_audit[session]
    assert rep_audit["query_start_trial"] == REPORT_START
    assert rep_audit["query_end_trial"] is None
    assert rep_audit["query_trials"] == expected_report_trials
    assert rep_audit["full_window_disjoint"] is True


@requires_falcon_data
@pytest.mark.parametrize("fold", [1, 2])
def test_bounded_window_does_not_straddle_boundaries(fold):
    dm = _module(loso_fold=fold)
    dm.setup(stage="fit")
    session = dm.val_heldin_session_names[0]
    audit = dm.val_heldin_dataset.query_window_audit[session]
    window_size = dm.hparams.window_size
    starts = dm.val_heldin_dataset.trial_start_indices[session]
    lower = int(starts[SUPPORT_END])
    upper_exclusive = int(starts[SELECTION_END])
    assert audit["minimum_window_start_padded_bin"] >= lower
    assert audit["maximum_window_start_padded_bin"] <= upper_exclusive - window_size
    assert audit["minimum_window_start_padded_bin"] == audit["raw_query_start_bin"] + window_size - 1

    for _, start_idx in dm.val_heldin_dataset.window_indices:
        assert start_idx >= lower
        assert start_idx + window_size <= upper_exclusive


@requires_falcon_data
@pytest.mark.parametrize("fold", [1, 2])
def test_report_window_does_not_straddle_lower_boundary(fold):
    dm = _module(loso_fold=fold, heldin_query_start_trial=REPORT_START, heldin_query_end_trial=None)
    dm.setup(stage="fit")
    session = dm.val_heldin_session_names[0]
    audit = dm.val_heldin_dataset.query_window_audit[session]
    window_size = dm.hparams.window_size
    starts = dm.val_heldin_dataset.trial_start_indices[session]
    lower = int(starts[REPORT_START])
    assert audit["minimum_window_start_padded_bin"] >= lower
    for _, start_idx in dm.val_heldin_dataset.window_indices:
        assert start_idx >= lower


@requires_falcon_data
def test_split_manifest_records_bounded_window_audit():
    dm = _module(loso_fold=1)
    dm.setup(stage="fit")
    manifest = dm.get_split_manifest()
    assert manifest["heldin_query_start_trial"] == SUPPORT_END
    assert manifest["heldin_query_end_trial"] == SELECTION_END
    assert "heldin_query_window_audit" in manifest


@requires_falcon_data
def test_default_heldin_query_params_are_zero_and_none():
    FalconDataModule = _import_datamodule()
    dm = FalconDataModule(task="m1", data_dir="/unused")
    assert dm.hparams.heldin_query_start_trial == 0
    assert dm.hparams.heldin_query_end_trial is None
