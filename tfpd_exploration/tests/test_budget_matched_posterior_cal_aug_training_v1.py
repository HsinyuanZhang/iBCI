from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tfpd_exploration/src"))

from budget_matched_posterior_cal_aug_v1.posterior import (  # noqa: E402
    fit_equal_budget_normalizer,
    fit_posterior_mean,
    fit_source_prior,
)
from budget_matched_posterior_cal_aug_v1.training import (  # noqa: E402
    BudgetMatchedPosteriorDataset,
    BudgetTaggedBatchSampler,
    build_session_posterior_features,
    posterior_normalizer_from_payload,
    source_prior_from_payload,
    tagged_batch_digest,
)


def _arrays(offset: float = 0.0):
    theta = np.resize(np.asarray([0.0, np.pi / 2, np.pi, -np.pi / 2]), 30)
    design = np.stack((np.ones(30), np.cos(theta), np.sin(theta)), axis=1)
    beta = np.asarray([[2.0, 1.0, -0.5], [1.0, -0.2, 0.8]])
    rates = beta @ design.T + offset + np.linspace(-0.05, 0.05, 30)[None, :]
    return rates, theta


def _authority():
    sessions = [_arrays(index * 0.01) for index in range(3)]
    prior = fit_source_prior(sessions)
    rows = {4: [], 10: [], 30: []}
    for rates, theta in sessions:
        for budget in rows:
            rows[budget].append(
                fit_posterior_mean(
                    rates[:, :budget], theta[:budget], prior_variance=prior.variance
                ).raw_t4
            )
    normalizer = fit_equal_budget_normalizer(
        {budget: np.concatenate(parts, axis=0) for budget, parts in rows.items()}
    )
    return sessions, prior, normalizer


def test_session_features_are_budget_specific_and_single_cast_float32() -> None:
    sessions, prior, normalizer = _authority()
    rates, theta = sessions[0]
    feature = build_session_posterior_features(
        session="s0",
        trial_rates=rates,
        theta_radians=theta,
        source_prior=prior,
        posterior_normalizer=normalizer,
    )
    assert set(feature.normalized_side_by_budget) == {4, 10, 30}
    assert all(value.dtype == np.float32 for value in feature.normalized_side_by_budget.values())
    assert len(set(feature.raw_t4_sha256_by_budget.values())) == 3
    assert len(set(feature.normalized_side_sha256_by_budget.values())) == 3


def test_session_features_use_v3_finite_label_subset_without_changing_m4_m10() -> None:
    sessions, prior, normalizer = _authority()
    rates, theta = sessions[0]
    complete = build_session_posterior_features(
        session="complete", trial_rates=rates, theta_radians=theta,
        source_prior=prior, posterior_normalizer=normalizer,
    )
    missing = theta.copy()
    missing[27] = np.nan
    masked = build_session_posterior_features(
        session="masked", trial_rates=rates, theta_radians=missing,
        source_prior=prior, posterior_normalizer=normalizer,
    )
    for budget in (4, 10):
        assert np.array_equal(
            complete.normalized_side_by_budget[budget],
            masked.normalized_side_by_budget[budget],
        )
        assert complete.raw_t4_sha256_by_budget[budget] == masked.raw_t4_sha256_by_budget[budget]
    assert complete.raw_t4_sha256_by_budget[30] != masked.raw_t4_sha256_by_budget[30]


def test_source_authority_types_round_trip_and_reject_semantic_drift() -> None:
    _sessions, prior, normalizer = _authority()
    prior_payload = {**prior.payload_without_sha(), "body_sha256": prior.body_sha256}
    normalizer_payload = {
        **normalizer.payload_without_sha(), "body_sha256": normalizer.body_sha256
    }
    assert source_prior_from_payload(prior_payload) == prior
    restored = posterior_normalizer_from_payload(normalizer_payload)
    assert restored.body_sha256 == normalizer.body_sha256
    assert np.array_equal(restored.mean, normalizer.mean)
    altered = dict(prior_payload)
    altered["shared_unchanged_across_budgets"] = [4, 10, 30]
    with pytest.raises(ValueError, match="sharing"):
        source_prior_from_payload(altered)


def test_tagged_sampler_assigns_one_exact_cycle_budget_to_whole_batch() -> None:
    base = [[9, 8], [7], [6, 5, 4], [3]]
    sampler = BudgetTaggedBatchSampler(base, start_batch=1)
    realized = list(sampler)
    assert [[tag[1] for tag in batch] for batch in realized] == [
        [10, 10], [4], [30, 30, 30], [10]
    ]
    assert [row["budget"] for row in sampler.tagged_batches] == [10, 4, 30, 10]
    assert tagged_batch_digest(sampler.tagged_batches) == tagged_batch_digest(
        sampler.tagged_batches
    )


def test_dataset_rewrites_only_calibration_and_side_for_same_session_budget() -> None:
    torch = pytest.importorskip("torch")
    sessions, prior, normalizer = _authority()
    features = {
        "s0": build_session_posterior_features(
            session="s0",
            trial_rates=sessions[0][0],
            theta_radians=sessions[0][1],
            source_prior=prior,
            posterior_normalizer=normalizer,
        )
    }
    neural = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    target = torch.ones(10, 2)
    calibration = torch.arange(30 * 5 * 2, dtype=torch.float32).reshape(30, 5, 2)
    canonical_side = torch.zeros(2, 4)
    tail = {"preserve": True}
    base = [(neural, target, calibration, "s0", canonical_side, tail)]
    view = BudgetMatchedPosteriorDataset(base, features)

    sample = view[(0, 4, 0)]
    assert sample[0] is neural
    assert sample[1] is target
    assert sample[2].shape == (4, 5, 2)
    assert torch.equal(sample[2], calibration[:4])
    assert sample[3] == "s0"
    assert torch.equal(
        sample[4], torch.from_numpy(features["s0"].normalized_side_by_budget[4])
    )
    assert sample[5] is tail
    assert calibration.shape[0] == 30
    assert torch.equal(canonical_side, torch.zeros_like(canonical_side))


def test_dataset_rejects_untagged_unknown_session_and_unit_axis_drift() -> None:
    torch = pytest.importorskip("torch")
    sessions, prior, normalizer = _authority()
    feature = build_session_posterior_features(
        session="s0",
        trial_rates=sessions[0][0],
        theta_radians=sessions[0][1],
        source_prior=prior,
        posterior_normalizer=normalizer,
    )
    good = (torch.zeros(2), torch.zeros(2), torch.zeros(30, 5, 2), "s0", torch.zeros(2, 4))
    view = BudgetMatchedPosteriorDataset([good], {"s0": feature})
    with pytest.raises(ValueError, match="requires an"):
        view[0]
    bad_session = BudgetMatchedPosteriorDataset(
        [(good[0], good[1], good[2], "other", good[4])], {"s0": feature}
    )
    with pytest.raises(ValueError, match="not authorized"):
        bad_session[(0, 4, 0)]
    bad_units = BudgetMatchedPosteriorDataset(
        [(good[0], good[1], torch.zeros(30, 5, 3), "s0", torch.zeros(2, 4))],
        {"s0": feature},
    )
    with pytest.raises(ValueError, match="unit axis"):
        bad_units[(0, 4, 0)]


def test_sampler_rejects_any_relaxed_or_reordered_cycle() -> None:
    with pytest.raises(ValueError, match="exactly"):
        BudgetTaggedBatchSampler([[0]], cycle=(4, 10, 30))
