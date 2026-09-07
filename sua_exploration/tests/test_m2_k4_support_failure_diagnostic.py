"""Synthetic contracts for the bounded support-only K4 failure diagnostic."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "k4_failure_diagnostic", ROOT / "sua_exploration/scripts/diagnose_m2_m24_k4_support_failure.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_support_blocks_never_cross_trial_or_use_unrequested_trials() -> None:
    rng = np.random.RandomState(2)
    trials, bins, channels = 24, 15, 4
    velocity = rng.normal(size=(trials * bins, 2))
    velocity[np.arange(trials) * bins] = 0.0  # one inactive candidate per trial
    neural = rng.poisson(1.5, size=(trials * bins, channels)).astype(float)
    change = np.zeros(trials * bins, dtype=bool); change[::bins] = True
    rate, y = MODULE.blocks(neural, velocity, change, lo=0, hi=12)
    assert rate.shape[1] == channels and y.shape[1] == 2
    assert len(rate) < 12 * 2  # the shifted behaviour block rejects inactive boundaries
    with np.testing.assert_raises(ValueError):
        MODULE.blocks(neural, velocity, change, lo=0, hi=25)


def test_ols_fit_reports_finite_support_only_uncertainty_descriptors() -> None:
    rng = np.random.RandomState(7)
    y = rng.normal(size=(80, 2))
    weights = rng.normal(size=(5, 2)); intercept = rng.normal(size=5)
    rate = y @ weights.T + intercept + rng.normal(scale=0.05, size=(80, 5))
    record = MODULE.fit(rate, y)
    assert record["feature"].shape == (5, 4)
    assert record["active_blocks"] == 80
    assert np.isfinite(record["condition"])
    assert record["weight_snr_median"] > 1.0


def test_script_declares_support_row_scope_without_claiming_the_file_was_sealed() -> None:
    text = (ROOT / "sua_exploration/scripts/diagnose_m2_m24_k4_support_failure.py").read_text()
    assert "query_rows_used_in_descriptors\": False" in text
    assert "heldout_calibration_files_previously_opened_in_frozen_replay\": True" in text
    assert "frozen query deltas are appended after fitting only" in text
    assert "refusing to overwrite" in text
