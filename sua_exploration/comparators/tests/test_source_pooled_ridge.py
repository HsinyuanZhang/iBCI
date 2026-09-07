"""Tests for source-pooled ridge fairness experiment."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sua_exploration.mc_maze import priority_a2_normalized_ridge_v2 as ridge
from sua_exploration.comparators.core import source_pooled_ridge as spr


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"


def _synthetic() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260812)
    features = rng.normal(size=(29, 6))
    targets = features @ np.asarray([[0.4, -0.1], [0.2, 0.3], [-0.3, 0.2], [0.1, -0.4], [0.5, 0.1], [-0.2, 0.6]]) + [1.0, -2.0]
    weights = rng.uniform(0.2, 2.5, size=29)
    probe = rng.normal(size=(7, 6))
    return features, targets, weights, probe


def test_gamma_zero_reproduces_scratch_exactly() -> None:
    x, y, w, probe = _synthetic()
    prior = np.ones((x.shape[1], 2), dtype=np.float64)
    scratch = ridge.fit_normalized_weighted_ridge(x, y, w, normalized_lambda=1.0)
    anchored = spr.fit_prior_anchored_normalized_weighted_ridge(
        x, y, w, normalized_lambda=1.0, gamma=0.0, beta_prior=prior
    )
    np.testing.assert_allclose(scratch.coefficients, anchored.coefficients, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(scratch.intercept, anchored.intercept, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        ridge.predict_normalized_weighted_ridge(probe, scratch),
        ridge.predict_normalized_weighted_ridge(probe, anchored),
        rtol=0.0,
        atol=0.0,
    )


def test_intercept_unpenalized_and_not_shrunk_toward_prior() -> None:
    x, y, w, _probe = _synthetic()
    prior = np.full((x.shape[1], 2), 999.0)
    readout = spr.fit_prior_anchored_normalized_weighted_ridge(
        x, y, w, normalized_lambda=1.0, gamma=10.0, beta_prior=prior
    )
    ybar = (w[:, None] * y).sum(axis=0) / w.sum()
    np.testing.assert_allclose(readout.intercept, ybar, rtol=0.0, atol=2e-12)
    residual = (w[:, None] * (y - ridge.predict_normalized_weighted_ridge(x, readout))).sum(axis=0)
    np.testing.assert_allclose(residual, 0.0, rtol=0.0, atol=2e-9)


def test_correspondence_detector_definable_and_undefinable() -> None:
    stable = [np.array([10, 11, 12], dtype=np.int64) for _ in range(4)]
    mapping, identical, count = spr.detect_correspondence(stable, identifiers_are_positional_only=False)
    assert mapping is True
    assert identical is True
    assert count == 3
    positional = [np.array([0, 1, 2], dtype=np.int64) for _ in range(4)]
    mapping, identical, count = spr.detect_correspondence(positional, identifiers_are_positional_only=True)
    assert mapping is False
    assert identical is True
    assert count == 0
    mismatched = [np.array([1, 2, 3], dtype=np.int64), np.array([1, 2, 4], dtype=np.int64)]
    mapping, identical, count = spr.detect_correspondence(mismatched, identifiers_are_positional_only=False)
    assert mapping is False
    assert identical is False
    assert count == 0


def test_lambda_gamma_selection_never_reads_target_query() -> None:
    x, y, w, _probe = _synthetic()

    class _Session:
        def __init__(self, asset_id: str, query_marker: float) -> None:
            self.asset_id = asset_id
            self.session_id = asset_id
            self.support_starts = np.arange(3, dtype=np.int64)
            self.support_features = x
            self.dense_targets = y
            self.direction_targets = y
            self.uniform_weights = w
            self.query_starts = np.arange(3, dtype=np.int64)
            self.query_target = np.full((3, 2), query_marker, dtype=np.float32)
            self.channel_ids = np.arange(x.shape[1], dtype=np.int64)

            def query_features_fn(starts: np.ndarray) -> np.ndarray:
                return x[: starts.size]

            self.query_features_fn = query_features_fn

    sessions = [_Session("a", 1.0), _Session("b", 2.0), _Session("c", 3.0)]
    observed_markers: list[float] = []

    def _record_source_query_score(prediction: np.ndarray, query_target: np.ndarray) -> float:
        observed_markers.append(float(query_target[0, 0]))
        return float(np.mean(prediction))

    _selected_lambda, _selected_gamma, provenance = spr.select_lambda_gamma_source_loso(
        sessions,
        arm="ridge_direction_scratch",
        budget=15,
        recompute_r2=_record_source_query_score,
    )
    assert provenance["target_query_consumed"] is False
    assert set(observed_markers).issubset({1.0, 2.0, 3.0})
    assert 99.0 not in observed_markers


def test_pooled_prior_equals_mean_of_source_betas() -> None:
    x, y, w, _probe = _synthetic()
    coefficients = [
        ridge.fit_normalized_weighted_ridge(x, y, w, normalized_lambda=1.0).coefficients,
        ridge.fit_normalized_weighted_ridge(x, y * 0.5 + 0.1, w, normalized_lambda=1.0).coefficients,
        ridge.fit_normalized_weighted_ridge(x, y * -0.25 + 0.2, w, normalized_lambda=1.0).coefficients,
    ]
    pooled = spr.pool_beta_prior(coefficients)
    expected = np.mean(np.stack(coefficients, axis=0), axis=0)
    np.testing.assert_allclose(pooled, expected, rtol=0.0, atol=0.0)


def test_determinism_across_two_runs() -> None:
    x, y, w, probe = _synthetic()
    prior = ridge.fit_normalized_weighted_ridge(x, y, w, normalized_lambda=1.0).coefficients
    first = spr.fit_prior_anchored_normalized_weighted_ridge(
        x, y, w, normalized_lambda=1.0, gamma=0.1, beta_prior=prior
    )
    second = spr.fit_prior_anchored_normalized_weighted_ridge(
        x, y, w, normalized_lambda=1.0, gamma=0.1, beta_prior=prior
    )
    np.testing.assert_allclose(first.coefficients, second.coefficients, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        ridge.predict_normalized_weighted_ridge(probe, first),
        ridge.predict_normalized_weighted_ridge(probe, second),
        rtol=0.0,
        atol=0.0,
    )


def test_runner_contract_without_data() -> None:
    import os
    import subprocess

    env = {**os.environ, "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(REPO_ROOT)}
    completed = subprocess.run(
        [PYTHON, str(REPO_ROOT / "sua_exploration/scripts/run_source_pooled_ridge.py")],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["run_started"] is False
    assert "numerical_contract" in payload
