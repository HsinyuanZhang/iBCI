import numpy as np
import torch

from src import calibration_budget_protocol_factorial_v1 as route


def test_dry_plan_is_complete_causal_factorial() -> None:
    plan = route.dry_plan()
    assert plan["budgets"] == [4, 10]
    assert plan["supports"] == ["chronological", "doptimal_first30"]
    assert plan["estimators"] == ["ols", "ridge_fixed_0p1"]
    assert plan["regimes"] == ["label_limited_m30_activity", "total_selected_calibration"]
    assert plan["query_after_trial30"] is True
    assert plan["target_optimizer_steps"] == 0


def test_paired_contrasts_use_chronological_ols_label_base() -> None:
    cells = []
    for surface, count in (("within", 6), ("external", 15)):
        for budget in route.BUDGETS:
            for support in route.SUPPORTS:
                for estimator in route.ESTIMATORS:
                    for regime in route.REGIMES:
                        value = 0.0
                        if support == "doptimal_first30":
                            value += 0.02
                        if estimator == "ridge_fixed_0p1":
                            value += 0.03
                        if regime == "total_selected_calibration":
                            value -= 0.01
                        rows = [{"session": f"s{i:02d}", "r2": value} for i in range(count)]
                        cells.append({
                            "surface": surface, "budget": budget, "support": support,
                            "estimator": estimator, "regime": regime, "sessions": rows,
                        })
    contrasts = route._paired_contrasts(cells)
    assert len(contrasts) == 28
    best = next(row for row in contrasts if row["surface"] == "external"
                and row["budget"] == 4
                and row["contrast"].startswith("doptimal_first30__ridge_fixed_0p1__label"))
    assert abs(best["mean_delta_r2"] - 0.05) < 1e-12
    assert best["positive_sessions"] == 15


def test_float64_ols_is_cast_to_cell_d_float32_before_normalization() -> None:
    raw = np.arange(20, dtype=np.float64).reshape(5, 4)
    side = route.normalize_raw_t4(
        raw, mean=torch.zeros(4, dtype=torch.float32),
        std=torch.ones(4, dtype=torch.float32), torch_module=torch,
    )
    assert side.dtype == torch.float32
    assert side.shape == (1, 5, 4)
    assert torch.equal(side[0], torch.from_numpy(raw.astype(np.float32)))
