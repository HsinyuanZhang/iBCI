"""No-data, no-CUDA unit tests for the M2 prerequisite audit laws."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT / "tfpd_exploration", ROOT / "tfpd_exploration" / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.cdm_p1_m2_v1 import audit, plan


def test_rank_auc_chance_and_perfect() -> None:
    statistic = [0.1, 0.2, 0.3, 0.4]
    assert audit.rank_auc(statistic, [False, False, True, True]) == pytest.approx(1.0)
    assert audit.rank_auc(statistic, [True, True, False, False]) == pytest.approx(0.0)
    assert audit.rank_auc(statistic, [True, False, False, True]) == pytest.approx(0.5)


def test_rank_auc_handles_ties() -> None:
    statistic = [1.0, 1.0, 1.0, 1.0]
    assert audit.rank_auc(statistic, [True, False, False, True]) == pytest.approx(0.5)


def test_rank_auc_rejects_single_class() -> None:
    with pytest.raises(audit.AuditError):
        audit.rank_auc([1.0, 2.0, 3.0], [True, True, True])


def test_boundary_statistics_shapes_and_causality() -> None:
    rng = np.random.default_rng(7)
    neural = rng.poisson(2.0, size=(50, 4)).astype(np.float64)
    statistics = audit.boundary_statistics(neural)
    assert set(statistics) == set(plan.BOUNDARY_STATISTICS)
    for value in statistics.values():
        assert value.shape == (50,)
        assert np.isfinite(value).all()
    # The first bin of each difference statistic uses only bin 0 (causal).
    assert statistics["abs_delta_population_count"][0] == 0.0
    assert statistics["abs_delta_population_vector"][0] == 0.0


def test_support_anchor_parity_is_bitwise_against_fit_ridge_t4() -> None:
    from src.calibration_budget_comparators_v1 import fit_ridge_t4

    rng = np.random.default_rng(11)
    angles = np.asarray([k * math.pi / 4.0 - 3.0 * math.pi / 4.0 for k in range(8)] +
                        [0.0, math.pi / 2.0])
    rates = rng.gamma(2.0, 3.0, size=(angles.size, 12))
    sealed, _evidence = fit_ridge_t4(
        rates, angles, normalized_lambda=plan.RIDGE_NORMALIZED_LAMBDA,
    )
    parity = audit.support_anchor_parity(
        rates, angles,
        normalized_lambda=plan.RIDGE_NORMALIZED_LAMBDA, sealed_raw_t4=sealed,
    )
    assert parity["rebuilt_bitwise_equal"] is True
    assert parity["max_abs_difference"] == 0.0
    assert parity["a0_eigenvalues_min"] > 0.0


def test_anchor_rebuild_uses_sqrt_not_hypot() -> None:
    block = np.asarray([[3.0e150], [4.0e150], [1.0]])
    rows = audit.anchor_rebuild_t4(block)
    # np.sqrt(a*a + c*c) overflows exactly like the sealed M2 law would not
    # produce; with hypot it would be finite -- the mirror must follow the
    # sealed arithmetic, which for moderate values is plain sqrt.
    moderate = np.asarray([[3.0], [4.0], [1.0]])
    assert audit.anchor_rebuild_t4(moderate)[0, 2] == pytest.approx(5.0)
    assert rows.shape == (1, 4)


def test_boundary_verdict_fail_closed_paths() -> None:
    within = {
        "s1": {"population_count": 0.51, "abs_delta_population_count": 0.50, "abs_delta_population_vector": 0.49},
        "s2": {"population_count": 0.53, "abs_delta_population_count": 0.52, "abs_delta_population_vector": 0.48},
    }
    verdict = audit.boundary_verdict(
        contract_markers_present=True, on_done_served_to_decoder=False,
        source_auc_by_session=within, auc_floor=plan.BOUNDARY_AUC_PASS_MIN,
    )
    assert verdict["verdict"] == "NOT_EVALUABLE_OFFICIAL_CONTRACT"
    assert verdict["failing_source_sessions"] == ["s1", "s2"]

    strong = {
        "s1": {"population_count": 0.9, "abs_delta_population_count": 0.7, "abs_delta_population_vector": 0.8},
    }
    assert audit.boundary_verdict(
        contract_markers_present=True, on_done_served_to_decoder=False,
        source_auc_by_session=strong, auc_floor=0.65,
    )["verdict"] == "RECONSTRUCTIBLE_CAUSAL_LAW"

    assert audit.boundary_verdict(
        contract_markers_present=True, on_done_served_to_decoder=True,
        source_auc_by_session=strong, auc_floor=0.65,
    )["verdict"] == "AVAILABLE_UNDER_OFFICIAL_CONTRACT"

    with pytest.raises(audit.AuditError):
        audit.boundary_verdict(
            contract_markers_present=False, on_done_served_to_decoder=False,
            source_auc_by_session=strong, auc_floor=0.65,
        )


def test_plan_constants_are_frozen() -> None:
    assert plan.RIDGE_NORMALIZED_LAMBDA == 0.1
    assert plan.CHECKPOINT_SHA256.startswith("25d7bc72")
    assert plan.BOUNDARY_AUC_PASS_MIN > 0.5
    assert len(plan.BOUNDARY_STATISTICS) == 3
    assert plan.EXPECTED_WITHIN_SESSIONS == 7
    assert plan.EXPECTED_EXTERNAL_SESSIONS == 6
