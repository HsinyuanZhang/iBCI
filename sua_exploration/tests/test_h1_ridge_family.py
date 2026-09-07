"""Tests for the H1 ridge-family fairness experiment."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "SPINT-main") not in sys.path:
    sys.path.insert(0, str(ROOT / "SPINT-main"))

from sua_exploration.mc_maze import h1_ridge_family as family


DATA_DIR = ROOT / "SPINT-main/data/000954"
PYTHON = "/home/xinyuan/Work_host/SPINT/.venv/bin/python" if False else None


def _load_splits() -> dict[str, family.SessionSplit]:
    from src.data.h1_m4_eb_pilot import H1_M4_FOLD0_TARGET, load_target_records

    records = load_target_records(DATA_DIR)
    return {name: family.build_session_split(records[name]) for name in H1_M4_FOLD0_TARGET}


@pytest.fixture(scope="module")
def splits() -> dict[str, family.SessionSplit]:
    return _load_splits()


def test_sealed_arm_reproduces_reference(splits: dict[str, family.SessionSplit]) -> None:
    result = family.run_arm("ridge_v2r2_sealed", splits)
    pooled = float(result["pooled"]["r2"])
    assert pooled == pytest.approx(family.SEALED_POOLED_R2, abs=1.0e-9)
    for session_name, expected in family.SEALED_PER_SESSION_R2.items():
        assert float(result["per_session"][session_name]["r2"]) == pytest.approx(expected, abs=1.0e-9)


def test_pca_basis_uses_calibration_only(splits: dict[str, family.SessionSplit]) -> None:
    split = next(iter(splits.values()))
    pca_before = family.fit_channel_pca(split.pca_calibration_bins, 16)
    perturbed = copy.deepcopy(split)
    perturbed_neural = perturbed.neural.copy()
    perturbed_neural[perturbed.query_output_bins] += 1000.0
    object.__setattr__(perturbed, "neural", perturbed_neural)
    pca_after = family.fit_channel_pca(perturbed.pca_calibration_bins, 16)
    assert pca_before.digest() == pca_after.digest()


def test_lambda_selection_never_reads_query_windows(splits: dict[str, family.SessionSplit]) -> None:
    split = next(iter(splits.values()))
    features = family.build_feature_matrix(
        split,
        split.calibration_target_bins,
        history_bins=family.HISTORY_BINS_DEFAULT,
    )
    baseline_lambda, _ = family.select_lambda(features, split.calibration_truth, split.calibration_trial_ids)

    perturbed = copy.deepcopy(split)
    perturbed_neural = perturbed.neural.copy()
    perturbed_neural[perturbed.query_output_bins] = np.nan
    object.__setattr__(perturbed, "neural", perturbed_neural)
    perturbed_features = family.build_feature_matrix(
        perturbed,
        perturbed.calibration_target_bins,
        history_bins=family.HISTORY_BINS_DEFAULT,
    )
    perturbed_lambda, _ = family.select_lambda(
        perturbed_features,
        perturbed.calibration_truth,
        perturbed.calibration_trial_ids,
    )
    assert perturbed_lambda == baseline_lambda
    assert float(baseline_lambda) > 0.0
    assert np.array_equal(features, perturbed_features)


def test_observations_per_parameter_arithmetic_on_synthetic() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 12))
    y = rng.normal(size=(40, family.VELOCITY_DIM))
    trial_ids = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
    fit = family.fit_normalized_ridge(x, y, normalized_lambda=1.0)
    assert family.observations_per_parameter(40, 12) == pytest.approx(40 / 12)
    assert family.fitted_parameter_count(12) == 12 * family.VELOCITY_DIM
    assert int(fit["feature_dim"]) == 12
    assert int(fit["calibration_rows"]) == 40


def test_determinism_across_two_runs(splits: dict[str, family.SessionSplit]) -> None:
    first = family.run_arm("ridge_lambda_cv", splits)
    second = family.run_arm("ridge_lambda_cv", splits)
    assert first == second


def test_runner_smoke_dry_compute(splits: dict[str, family.SessionSplit]) -> None:
    sealed = family.run_arm("ridge_v2r2_sealed", splits)
    assert sealed["pooled"]["supervision_coordinates"] == 42616
    assert sealed["pooled"]["query_windows"] == 8965
