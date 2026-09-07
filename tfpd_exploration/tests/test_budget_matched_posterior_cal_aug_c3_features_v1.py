from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest

from budget_matched_posterior_cal_aug_c3_v1.features import (
    BudgetMatchedReliabilityDataset,
    C3Arm,
    fit_source_reliability_normalizer,
    widen_cell_d_for_reliability,
)
from budget_matched_posterior_cal_aug_v1.posterior import PosteriorContractError, array_sha256
from budget_matched_posterior_cal_aug_v1.training import SessionPosteriorFeatures


def _feature(name: str, offset: float) -> SessionPosteriorFeatures:
    side = {
        budget: np.ascontiguousarray(
            np.arange(12, dtype=np.float32).reshape(3, 4) + offset + budget,
            dtype=np.float32,
        )
        for budget in (30, 10, 4)
    }
    q = {
        budget: np.ascontiguousarray(
            np.asarray([offset + budget, offset + budget + 1, offset + budget + 2], np.float64)
        )
        for budget in (30, 10, 4)
    }
    return SessionPosteriorFeatures(
        session=name,
        unit_count=3,
        normalized_side_by_budget=side,
        raw_t4_sha256_by_budget={budget: f"raw-{budget}-{name}" for budget in (30, 10, 4)},
        normalized_side_sha256_by_budget={budget: array_sha256(side[budget]) for budget in (30, 10, 4)},
        angular_reliability_by_budget=q,
        posterior_normalizer_sha256="normalizer",
        source_prior_sha256="prior",
    )


def test_q_normalizer_is_equal_budget_source_only_and_constant_is_zero() -> None:
    features = OrderedDict((name, _feature(name, offset)) for name, offset in (("s0", 0.0), ("s1", 10.0)))
    normalizer = fit_source_reliability_normalizer(features, source_roster=("s0", "s1"))
    expected = np.concatenate([
        features[name].angular_reliability_by_budget[budget]
        for budget in (4, 10, 30)
        for name in ("s0", "s1")
    ])
    assert normalizer.mean == float(expected.mean())
    assert normalizer.std == float(expected.std(ddof=0))
    assert normalizer.per_budget_row_count == {4: 6, 10: 6, 30: 6}
    assert normalizer.payload()["constant_normalized_value"] == 0.0


class _BaseDataset:
    window_indices = [("s0", 0)]

    def __init__(self):
        import torch

        self.row = (
            torch.zeros(50, 3),
            torch.zeros(50, 2),
            torch.zeros(30, 100, 3),
            "s0",
            torch.zeros(3, 4),
        )

    def __len__(self):
        return 1

    def __getitem__(self, index):
        assert index == 0
        return self.row


def test_real_and_constant_views_change_only_fifth_column() -> None:
    features = {"s0": _feature("s0", 0.0)}
    normalizer = fit_source_reliability_normalizer(features, source_roster=("s0",))
    real = BudgetMatchedReliabilityDataset(_BaseDataset(), features, normalizer, arm="real")[(0, 4, 0)]
    constant = BudgetMatchedReliabilityDataset(_BaseDataset(), features, normalizer, arm="constant")[(0, 4, 0)]
    assert tuple(real[2].shape) == (4, 100, 3)
    assert tuple(real[4].shape) == (3, 5)
    assert np.array_equal(real[4][:, :4].numpy(), constant[4][:, :4].numpy())
    assert np.array_equal(constant[4][:, 4].numpy(), np.zeros(3, np.float32))
    assert not np.array_equal(real[4][:, 4].numpy(), constant[4][:, 4].numpy())


def test_unknown_arm_and_roster_drift_fail_closed() -> None:
    features = {"s0": _feature("s0", 0.0)}
    normalizer = fit_source_reliability_normalizer(features, source_roster=("s0",))
    with pytest.raises(PosteriorContractError, match="unknown C3 arm"):
        BudgetMatchedReliabilityDataset(_BaseDataset(), features, normalizer, arm="wrong")
    with pytest.raises(PosteriorContractError, match="roster/features mismatch"):
        fit_source_reliability_normalizer(features, source_roster=("s1",))


def test_model_widening_copies_every_old_weight_and_zeros_only_q_column() -> None:
    import torch
    from tfpd_lane.pop_robust import build_population_robustness_model

    model = build_population_robustness_model(seed=42, cell="D")
    from src.models.components.streaming_encoders import build_encoder

    old_encoder = model.id_encoder
    old = {key: value.detach().clone() for key, value in old_encoder.state_dict().items()}
    widen_cell_d_for_reliability(model, build_encoder)
    new = model.id_encoder.state_dict()
    assert model.id_encoder.side_dim == 5
    for key in old:
        if key == "post_pool.0.weight":
            assert torch.equal(new[key][:, : old[key].shape[1]], old[key])
            assert torch.count_nonzero(new[key][:, -1]).item() == 0
        else:
            assert torch.equal(new[key], old[key])
