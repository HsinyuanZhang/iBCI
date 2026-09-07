from src import original_spint_short_budget_v1 as short


def test_dry_plan_is_a_real_short_budget_curve() -> None:
    plan = short.dry_plan()
    assert plan["budgets"] == [4, 10, 30]
    assert plan["systems"] == ["original_spint_b0_epoch12", "arm_a_total_calibration"]
    assert plan["fixed_query_after_trial30"] is True
    assert plan["target_optimizer_steps"] == 0

