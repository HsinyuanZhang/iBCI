from src import ridge_t4_lambda_curve_v1 as curve


def test_dry_plan_freezes_grid_and_external_selection_boundary() -> None:
    plan = curve.dry_plan()
    assert plan["budgets"] == [4, 10]
    assert plan["normalized_lambda_grid"] == list(curve.LAMBDAS)
    assert plan["external_not_used_for_selection"] is True
    assert plan["target_optimizer_steps"] == 0


def test_selector_uses_within_only_and_locks_matching_external_row() -> None:
    cells = []
    for budget in curve.BUDGETS:
        for regime, _ in curve.REGIMES:
            for surface in ("within", "external"):
                for value in curve.LAMBDAS:
                    score = -abs(value - 0.1) if surface == "within" else value
                    cells.append({"surface": surface, "budget": budget, "regime": regime,
                                  "normalized_lambda": value,
                                  "summary": {"equal_session_mean_r2": score}})
    selected = curve._select_within(cells)
    assert len(selected) == 4
    assert all(row["normalized_lambda"] == 0.1 for row in selected)
    assert all(row["external_summary"]["equal_session_mean_r2"] == 0.1 for row in selected)

