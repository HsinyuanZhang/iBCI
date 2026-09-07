"""Fail-closed tests for the M2/M33 zero-query correction exception."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from src.data.falcon_datamodule import FalconDataModule, FalconDataset


def _module(**overrides):
    base = dict(
        task="m2", data_dir="/definitely/not/read/for/rejected/configs", window_size=50,
        calibration_n_trials=33, random_calibration=False, include_heldout_in_fit=False,
        include_heldout_in_test=True, query_start_trial=33, allow_empty_heldout_query=True,
    )
    base.update(overrides)
    dm = FalconDataModule(**base)
    trainer = MagicMock()
    trainer.world_size = 1
    dm.trainer = trainer
    return dm


@pytest.mark.parametrize(
    "override, stage",
    [
        ({"task": "m1"}, "test"),
        ({"calibration_n_trials": 24, "query_start_trial": 24}, "test"),
        ({"include_heldout_in_test": False}, "test"),
        ({"include_heldout_in_fit": True}, "test"),
        ({}, "fit"),
    ],
)
def test_empty_query_exception_rejects_every_non_correction_path(override, stage):
    with pytest.raises(ValueError, match="allow_empty_heldout_query is reserved"):
        _module(**override).setup(stage=stage)


def test_empty_query_session_is_audited_but_contributes_no_window():
    # 33 one-bin trials give the M33 correction precisely zero query trials.
    trial_change = np.ones(33, dtype=bool)
    session = {
        "neural": np.zeros((33, 2), dtype=np.float32),
        "covariates": np.zeros((33, 2), dtype=np.float32),
        "eval_mask": np.ones(33, dtype=bool),
        "trial_change": trial_change,
    }
    dataset = FalconDataset(
        {"zero_query": session}, {"zero_query": session}, window_size=50,
        calibration_n_trials=33, random_calibration=False, smooth_calibration=False,
        max_trial_length=100, query_start_trial=33,
        allow_empty_query_sessions=True,
    )
    audit = dataset.query_window_audit["zero_query"]
    assert len(dataset) == 0
    assert audit["total_trials"] == 33
    assert audit["query_trials"] == audit["eligible_windows"] == 0
    assert audit["ineligible_reason"] == "zero_query_trials_after_chronological_support"
    assert audit["full_window_disjoint"] is True


def test_default_exception_flag_is_false():
    dm = FalconDataModule(task="m2", data_dir="/unused")
    assert dm.hparams.allow_empty_heldout_query is False
