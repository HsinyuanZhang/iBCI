from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.falcon_m1_all_source_b3_datamodule import (  # noqa: E402
    ALLOWED_SIDE_FEATURE_GROUPS,
    M1_ALL_SOURCE_SESSIONS,
    M1AllSourceB3DataModule,
)


def _hparams(**changes):
    values = dict(
        task="m1",
        calibration_n_trials=10,
        random_calibration=False,
        include_heldout_in_fit=False,
        include_heldout_in_test=False,
        query_start_trial=0,
        heldin_query_start_trial=0,
        heldin_query_end_trial=None,
        validation_protocol="all_source",
        smooth_calibration=False,
        side_feature_group="none",
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_all_source_b3_session_allowlist_and_side_groups():
    assert M1_ALL_SOURCE_SESSIONS == (
        "ses-20120924",
        "ses-20120926",
        "ses-20120927",
        "ses-20120928",
    )
    assert ALLOWED_SIDE_FEATURE_GROUPS == ("none", "t4", "zero4")
    with pytest.raises(ValueError, match="exactly"):
        M1AllSourceB3DataModule(
            task="m1", data_dir=".", source_session_names=["ses-20120924"],
            calibration_n_trials=10, random_calibration=False,
            include_heldout_in_fit=False, include_heldout_in_test=False,
            query_start_trial=0, heldin_query_start_trial=0,
            heldin_query_end_trial=None, side_feature_group="none",
        )
    with pytest.raises(ValueError, match="side_feature_group"):
        M1AllSourceB3DataModule(
            task="m1", data_dir=".",
            calibration_n_trials=10, random_calibration=False,
            include_heldout_in_fit=False, include_heldout_in_test=False,
            query_start_trial=0, heldin_query_start_trial=0,
            heldin_query_end_trial=None, side_feature_group="k4",
        )


def test_all_source_b3_accepts_t4_and_rejects_query():
    ok = M1AllSourceB3DataModule(
        task="m1", data_dir=".",
        calibration_n_trials=10, random_calibration=False,
        include_heldout_in_fit=False, include_heldout_in_test=False,
        query_start_trial=0, heldin_query_start_trial=0,
        heldin_query_end_trial=None, side_feature_group="t4",
        validation_protocol="all_source",
    )
    assert str(ok.hparams.side_feature_group).lower() == "t4"
    instance = object.__new__(M1AllSourceB3DataModule)
    instance._hparams = _hparams(side_feature_group="t4")
    instance._assert_fit_stage("fit")
    for changes, message in (
        ({"query_start_trial": 1}, "query"),
        ({"heldin_query_start_trial": 10}, "query"),
        ({"include_heldout_in_test": True}, "held-out"),
        ({"calibration_n_trials": 24}, "M10"),
        ({"side_feature_group": "d4"}, "side_feature_group"),
    ):
        bad = object.__new__(M1AllSourceB3DataModule)
        bad._hparams = _hparams(**changes)
        with pytest.raises(ValueError, match=message):
            bad._assert_fit_stage("fit")
