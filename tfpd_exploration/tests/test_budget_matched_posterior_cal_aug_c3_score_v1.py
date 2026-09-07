from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from budget_matched_posterior_cal_aug_c3_v1 import score
from budget_matched_posterior_cal_aug_c3_v1.features import ReliabilityNormalizer
from budget_matched_posterior_cal_aug_v1.posterior import (
    fit_equal_budget_normalizer,
    fit_posterior_mean,
    fit_source_prior,
)


def _synthetic():
    theta = np.asarray([0.0, np.pi / 2, np.pi, 3 * np.pi / 2], dtype=np.float64)
    rates = np.stack([
        2.0 + 0.5 * np.cos(theta) + 0.2 * np.sin(theta),
        1.0 - 0.3 * np.cos(theta) + 0.4 * np.sin(theta),
    ])
    # Add deterministic residuals while retaining rank three / positive DOF.
    rates[:, 3] += np.asarray([0.11, -0.07])
    prior = fit_source_prior([(rates, theta)])
    fit = fit_posterior_mean(rates, theta, prior_variance=prior.variance)
    normalizer = fit_equal_budget_normalizer({4: fit.raw_t4, 10: fit.raw_t4 + 0.1, 30: fit.raw_t4 + 0.2})
    qnorm = ReliabilityNormalizer(
        mean=0.0,
        std=1.0,
        row_count=6,
        rows_sha256="0" * 64,
        per_budget_row_count={4: 2, 10: 2, 30: 2},
        per_budget_rows_sha256={4: "1" * 64, 10: "2" * 64, 30: "3" * 64},
    )

    class Inputs:
        pass

    inputs = Inputs()
    inputs.n_units = 2
    inputs.selected_by_budget = {4: np.arange(4)}
    inputs.theta = theta
    inputs.rates = rates.T
    return inputs, prior, normalizer, qnorm


def test_posterior_side_uses_exact_selected_support():
    inputs, prior, normalizer, qnorm = _synthetic()
    result = score.posterior_side_for_inputs(
        inputs, 4, prior=prior, normalizer=normalizer, q_normalizer=qnorm
    )
    expected = fit_posterior_mean(inputs.rates.T, inputs.theta, prior_variance=prior.variance)
    assert result["fit"].raw_t4_sha256 == expected.raw_t4_sha256
    assert result["side4"].shape == (2, 4)
    assert result["q"].shape == (2,)
    changed = inputs.rates.copy()
    changed[0, 0] += 1.0
    inputs.rates = changed
    mutated = score.posterior_side_for_inputs(
        inputs, 4, prior=prior, normalizer=normalizer, q_normalizer=qnorm
    )
    assert mutated["fit"].raw_t4_sha256 != result["fit"].raw_t4_sha256


def test_q_shuffle_is_deterministic_nonidentity_permutation():
    a = score.q_shuffle_permutation("external", "session-a", 4, 61)
    b = score.q_shuffle_permutation("external", "session-a", 4, 61)
    c = score.q_shuffle_permutation("external", "session-b", 4, 61)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, np.arange(61))
    assert np.array_equal(np.sort(a), np.arange(61))
    assert not np.array_equal(a, c)


def test_decision_readout_keeps_e02_and_e03_identification_separate():
    cells = {}
    for key in ("external:m4", "external:m10", "external:m30", "within:m30"):
        cells[key] = {
            "equal_session_mean_delta": 0.0,
            "positive_sessions": 0,
            "worst_session_delta": 0.0,
        }
    c2 = json.loads(json.dumps(cells))
    c2["external:m4"] = {
        "equal_session_mean_delta": 0.021, "positive_sessions": 10,
        "worst_session_delta": -0.01,
    }
    c2["external:m30"] = {
        "equal_session_mean_delta": -0.005, "positive_sessions": 5,
        "worst_session_delta": -0.01,
    }
    c2["within:m30"] = {
        "equal_session_mean_delta": 0.0, "positive_sessions": 3,
        "worst_session_delta": 0.0,
    }
    positive = json.loads(json.dumps(cells))
    positive["external:m4"] = {
        "equal_session_mean_delta": 0.01, "positive_sessions": 9,
        "worst_session_delta": -0.01,
    }
    positive["external:m30"] = {
        "equal_session_mean_delta": 0.0, "positive_sessions": 8,
        "worst_session_delta": 0.0,
    }
    summary = {
        "c2_minus_c1": c2,
        "real_minus_c2": positive,
        "real_minus_constant": positive,
        "real_minus_q_shuffle": positive,
    }
    result = score.decision_readout(summary)
    assert result["c2_promotion"]["passed"] is True
    assert result["e03_performance_identification"]["passed"] is True
    assert result["e03_performance_identification"]["claim_complete"] is False
    assert result["old_c2_ridge_input_score_disposition"].startswith("DIAGNOSTIC_ONLY")


def test_closure_separates_review_files():
    repo = Path(__file__).resolve().parents[2]
    execution = score.execution_closure(repo)
    review = score.review_closure(repo)
    assert set(execution["files"]) == set(score.EXECUTION_PATHS)
    assert set(review["files"]) == set(score.REVIEW_PATHS)
    assert not set(execution["files"]) & set(review["files"])
    changed = dict(review)
    changed["files"] = dict(review["files"])
    first = next(iter(changed["files"]))
    changed["files"][first] = {"sha256": "f" * 64, "bytes": 1}
    changed["closure_sha256"] = "e" * 64
    disposition = score.review_drift(review, changed)
    assert disposition["numerical_acceptance_affected"] is False
    assert disposition["status"] == "ACCEPTED_NON_NUMERIC_DRIFT"
