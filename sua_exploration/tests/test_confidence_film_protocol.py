from __future__ import annotations

import numpy as np
import pytest

from mc_maze.confidence_film_protocol import (
    ACTIVITY_CALIBRATION_N,
    COMMON_EVALUATION_START,
    T4_BUDGETS,
    ConfidenceFiLMProtocol,
    make_protocol,
    validate_protocol,
)
from scripts.aggregate_sua_confidence_film_t4_budget import summarize


def test_fixed_budget_protocol_supports_only_the_predeclared_t4_sweep():
    assert [make_protocol(budget).t4_budget for budget in T4_BUDGETS] == [10, 15, 20, 30, 50]
    protocol = make_protocol(50)
    assert protocol.activity_calibration_n == ACTIVITY_CALIBRATION_N == 30
    assert protocol.evaluation_calibration_n == 30
    assert protocol.common_evaluation_start == COMMON_EVALUATION_START == 50
    assert protocol.formal_test_evaluated is False


def test_protocol_rejects_budget_and_no_test_contract_drift():
    with pytest.raises(ValueError, match="T4 budget"):
        make_protocol(25)
    with pytest.raises(ValueError, match="train/validation-only"):
        validate_protocol(
            ConfidenceFiLMProtocol(30, 50, 30, 50, (27, 6, 6), 100, True)
        )
    with pytest.raises(ValueError, match="common evaluation start"):
        validate_protocol(
            ConfidenceFiLMProtocol(30, 50, 30, 30, (27, 6, 6), 100, False)
        )


def test_stage0_and_formal_effectiveness_gates_are_not_conflated():
    sessions = [f"session_{index}" for index in range(6)]
    one_seed = summarize(
        np.full((1, 6), 0.54),
        np.full((1, 6), 0.50),
        seeds=(42,),
        sessions=sessions,
    )
    assert one_seed["passes_stage0_descriptive_gates"] is True
    assert one_seed["passes_formal_effectiveness_gates"] is False
    assert one_seed["formal_effectiveness_gates"]["at_least_three_predeclared_seeds"] is False

    three_seeds = summarize(
        np.full((3, 6), 0.54),
        np.full((3, 6), 0.50),
        seeds=(42, 43, 44),
        sessions=sessions,
    )
    assert three_seeds["passes_stage0_descriptive_gates"] is True
    assert three_seeds["passes_formal_effectiveness_gates"] is True
