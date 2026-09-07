"""Deployment-only variable-budget calibration primitives for H1 CarrierID.

The historical all-source data module is bound by the immutable v1 recovery
receipt and therefore remains unchanged.  This module is an append-only v5
deployment boundary: it accepts the exact public calibration allowlist,
retains the actual TrialNum values, and permits a support budget of three or
four trials without padding or duplication.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from falcon_challenge.config import FalconTask
from src.data.h1_carrierid_all_source_official import assert_deployment_calibration_path
from src.data.h1_m4_eb_pilot import (
    DEPLOYMENT_MIN_TRIALS,
    EXPECTED_NEURONS,
    H1PilotRecord,
    VELOCITY_DIM,
    _trial_blocks,
    session_date,
    session_from_path,
    sha256_file,
)


def load_deployment_calibration_record_v5(path: str | Path) -> H1PilotRecord:
    """Load one explicit public calibration record with at least three trials.

    This deliberately mirrors the historical loader's byte semantics while
    living outside its v1 code-hash boundary.  The caller chooses exactly
    three or four values from ``record.trial_values`` for deployment; no
    synthetic fourth trial is introduced here.
    """

    resolved = assert_deployment_calibration_path(path)
    from falcon_challenge.dataloaders import load_nwb
    from pynwb import NWBHDF5IO

    neural, velocity, trial_change, eval_mask = load_nwb(resolved, FalconTask.h1)
    with NWBHDF5IO(str(resolved), "r", load_namespaces=True) as handle:
        nwb = handle.read()
        if "TrialNum" not in nwb.acquisition:
            raise ValueError(f"{resolved}: calibration TrialNum is missing")
        trial_num = np.asarray(nwb.acquisition["TrialNum"].data[:], dtype=np.float64)
    spikes64 = np.asarray(neural, dtype=np.float64)
    velocity64 = np.asarray(velocity, dtype=np.float64)
    spikes, targets = spikes64.astype(np.float32), velocity64.astype(np.float32)
    changes, mask = np.asarray(trial_change, bool).reshape(-1), np.asarray(eval_mask, bool).reshape(-1)
    if spikes.ndim != 2 or spikes.shape[1] != EXPECTED_NEURONS:
        raise ValueError(f"invalid H1 calibration neural shape {spikes.shape}")
    if targets.ndim != 2 or targets.shape[1] != VELOCITY_DIM:
        raise ValueError(f"invalid H1 calibration target shape {targets.shape}")
    if len({spikes.shape[0], targets.shape[0], changes.size, mask.size, trial_num.size}) != 1:
        raise ValueError("deployment calibration arrays are misaligned")
    ordered = trial_num[mask & np.isfinite(trial_num)]
    if ordered.size == 0 or not np.all(np.diff(ordered) >= 0.0):
        raise ValueError("calibration TrialNum is not chronological")
    values: list[float] = []
    for value in ordered.tolist():
        value = float(value)
        if not values or value != values[-1]:
            values.append(value)
    if len(values) < DEPLOYMENT_MIN_TRIALS:
        raise ValueError(
            f"deployment calibration has fewer than {DEPLOYMENT_MIN_TRIALS} trials: {resolved}"
        )
    trials = tuple(_trial_blocks(value, spikes64, velocity64, mask, trial_num) for value in values)
    return H1PilotRecord(
        session_from_path(resolved), session_date(session_from_path(resolved)), resolved,
        sha256_file(resolved), spikes, targets, changes, mask, trial_num, tuple(values), trials,
    )
