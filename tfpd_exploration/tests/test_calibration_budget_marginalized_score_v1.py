from __future__ import annotations

import json
from pathlib import Path

from src import calibration_budget_marginalized_score_v1 as score


def _contrast(budget: int, delta: float, positive: int) -> dict[str, object]:
    return {
        "surface": "external", "budget": budget, "cbm_regime": "total_calibration_limited",
        "cbm_estimator": "ordinary_ols", "cbm_support": "chronological",
        "baseline_system": "cell_d_ols", "baseline_regime": "total_calibration_limited",
        "mean_delta_r2": delta, "positive_sessions": positive,
    }


def test_gate_requires_short_budget_gain_breadth_and_m30_safety() -> None:
    passed = score._gate([_contrast(4, 0.05, 10), _contrast(10, 0.0, 8), _contrast(30, -0.03, 8)])
    assert passed["decision"] == "PASS"
    failed = score._gate([_contrast(4, 0.049, 15), _contrast(10, 0.1, 15), _contrast(30, 0.0, 15)])
    assert failed["decision"] == "STOP"


def test_dry_plan_has_both_information_regimes() -> None:
    plan = score.dry_plan()
    assert plan["budgets"] == [4, 10, 30]
    assert set(plan["regimes"]) == {"label_limited_m30_activity", "total_calibration_limited"}
    assert plan["estimators"] == ["ordinary_ols", "fixed_ridge_0p1"]
    assert plan["supports"] == ["chronological", "doptimal_first30"]
    assert plan["cell_count"] == 40
    assert plan["target_optimizer_steps"] == 0


def test_contrasts_keep_cbm_estimator_identity() -> None:
    cbm = [{
        "surface": "within", "budget": 4, "regime": "total_calibration_limited",
        "support": "doptimal_first30", "estimator": "fixed_ridge_0p1",
        "sessions": [{"session": "s1", "r2": 0.2}, {"session": "s2", "r2": 0.4}],
    }]
    base = [{
        "surface": "within", "budget": 4, "system": "cell_d_ols",
        "regime": "total_calibration_limited",
        "sessions": [{"session": "s1", "r2": 0.1}, {"session": "s2", "r2": 0.3}],
    }]
    rows = score._contrasts(cbm, base)
    assert len(rows) == 1
    assert rows[0]["cbm_estimator"] == "fixed_ridge_0p1"
    assert rows[0]["cbm_support"] == "doptimal_first30"
    assert abs(rows[0]["mean_delta_r2"] - 0.1) < 1.0e-12


def test_performance_gate_is_recipe_matched_and_requires_all_checks() -> None:
    rows = [
        {"surface": "external", "budget": 4, "mean_delta_r2": 0.05,
         "positive_sessions": 10},
        {"surface": "external", "budget": 10, "mean_delta_r2": 0.0,
         "positive_sessions": 8},
        {"surface": "external", "budget": 30, "mean_delta_r2": -0.03,
         "positive_sessions": 8},
    ]
    assert score._performance_gate(rows)["decision"] == "PASS"
    rows[0] = {**rows[0], "positive_sessions": 9}
    assert score._performance_gate(rows)["decision"] == "STOP"


def test_short_budget_system_comparisons_are_descriptive_and_paired() -> None:
    cbm = [{
        "surface": "external", "budget": 4, "regime": "total_calibration_limited",
        "support": "doptimal_first30", "estimator": "fixed_ridge_0p1",
        "sessions": [{"session": "s1", "r2": 0.3}, {"session": "s2", "r2": 0.2}],
    }]
    baselines = [
        {"surface": "external", "budget": 4, "system": "original_spint_b0_epoch12",
         "sessions": [{"session": "s1", "r2": 0.0}, {"session": "s2", "r2": -0.1}]},
        {"surface": "external", "budget": 4, "system": "arm_a_total_calibration",
         "sessions": [{"session": "s1", "r2": 0.1}, {"session": "s2", "r2": 0.1}]},
    ]
    rows = score._short_budget_system_contrasts(cbm, baselines)
    assert len(rows) == 2
    assert {row["baseline_system"] for row in rows} == {
        "original_spint_b0_epoch12", "arm_a_total_calibration",
    }
    assert all(row["comparison_role"] ==
               "descriptive_system_comparison_not_treatment_attribution" for row in rows)
    assert rows[0]["positive_sessions"] == 2


def test_matched_training_effects_accept_actual_factorial_estimator_vocabulary() -> None:
    root = Path(__file__).resolve().parents[2]
    factorial = json.loads((
        root / "tfpd_exploration/results/calibration_budget_protocol_factorial_v2/receipt.json"
    ).read_bytes())["cells"]
    phase1 = json.loads((
        root / "tfpd_exploration/results/calibration_budget_comparators_v1/receipt.json"
    ).read_bytes())["cells"]
    own = []
    recipe = {4: "doptimal_first30", 10: "chronological", 30: "chronological"}
    for surface in ("within", "external"):
        for budget, support in recipe.items():
            if budget in (4, 10):
                base = next(cell for cell in factorial if cell["surface"] == surface
                            and cell["budget"] == budget and cell["support"] == support
                            and cell["estimator"] == "ridge_fixed_0p1"
                            and cell["regime"] == "total_selected_calibration")
            else:
                base = next(cell for cell in phase1 if cell["surface"] == surface
                            and cell["budget"] == 30
                            and cell["system"] == "cell_d_ridge_t4_fixed_0p1"
                            and cell["regime"] == "label_limited_m30_activity")
            own.append({
                "surface": surface, "budget": budget, "support": support,
                "estimator": "fixed_ridge_0p1", "regime": "total_calibration_limited",
                "sessions": [{**row, "r2": float(row["r2"]) + 0.01} for row in base["sessions"]],
            })
    rows = score._matched_training_effects(own, factorial, phase1)
    assert len(rows) == 6
    assert all(abs(float(row["mean_delta_r2"]) - 0.01) < 1.0e-12 for row in rows)


def test_actual_failed_v1_predecessor_is_bound() -> None:
    root = Path(__file__).resolve().parents[2]
    predecessor = score._validate_failed_v1(root)
    assert predecessor["attempt_sha256"] == score.FAILED_V1_ATTEMPT_SHA256
    assert predecessor["failure_sha256"] == score.FAILED_V1_FAILURE_SHA256
    assert predecessor["scientific_result_published"] is False
