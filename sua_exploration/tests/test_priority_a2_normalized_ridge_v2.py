"""Focused numerical-contract tests for corrected Priority A2 v2 ridge."""
from __future__ import annotations

import json
import numpy as np
import pytest

from sua_exploration.mc_maze import priority_a2_normalized_ridge_v2 as ridge
from sua_exploration.scripts import run_priority_a2_same_target_density_v2 as a2b_runner
from sua_exploration.scripts import run_priority_a2_weighting_control_v2 as a2a_runner


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(90210)
    features = rng.normal(size=(37, 4))
    targets = features @ np.asarray([[0.7, -0.2], [0.1, 0.3], [-0.5, 0.4], [0.25, -0.7]]) + [2.0, -3.0]
    weights = rng.uniform(0.1, 4.0, size=37)
    probe = rng.normal(size=(11, 4))
    return features, targets, weights, probe


def _independent_normalized_unweighted_prediction(x: np.ndarray, y: np.ndarray, probe: np.ndarray, lam: float) -> np.ndarray:
    """Reference equation kept local so a shared helper cannot self-certify."""
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < ridge.FEATURE_STD_EPS] = 1.0
    z = (x - mean) / scale
    ymean = y.mean(axis=0)
    beta = np.linalg.solve((z.T @ z) / len(x) + lam * np.eye(x.shape[1]), (z.T @ (y - ymean)) / len(x))
    return ((probe - mean) / scale) @ beta + ymean


def test_uniform_weights_reproduce_sealed_normalized_unweighted_equation() -> None:
    x, y, _weights, probe = _data()
    actual = ridge.predict_normalized_weighted_ridge(
        probe, ridge.fit_normalized_weighted_ridge(x, y, np.ones(x.shape[0]), normalized_lambda=1.0)
    )
    expected = _independent_normalized_unweighted_prediction(x, y, probe, 1.0)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-12)


def test_global_weight_scaling_leaves_predictions_unchanged() -> None:
    x, y, weights, probe = _data()
    first = ridge.predict_normalized_weighted_ridge(probe, ridge.fit_normalized_weighted_ridge(x, y, weights))
    scaled = ridge.predict_normalized_weighted_ridge(probe, ridge.fit_normalized_weighted_ridge(x, y, weights * 1.0e7))
    np.testing.assert_allclose(first, scaled, rtol=0.0, atol=3e-12)


def _independent_weighted_primal(x: np.ndarray, y: np.ndarray, w: np.ndarray, probe: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Direct primal normal equation; intentionally not a core helper."""
    total = float(w.sum())
    mean = (w[:, None] * x).sum(0) / total
    ymean = (w[:, None] * y).sum(0) / total
    centered = x - mean
    scale = np.sqrt((w[:, None] * centered**2).sum(0) / total)
    scale[scale < ridge.FEATURE_STD_EPS] = 1.0
    z = centered / scale
    beta = np.linalg.solve((z * w[:, None]).T @ z / total + np.eye(x.shape[1]), (z * w[:, None]).T @ (y - ymean) / total)
    return beta, ((probe - mean) / scale) @ beta + ymean


def test_adaptive_primal_and_dual_are_equivalent_to_independent_primal_equation() -> None:
    rng = np.random.default_rng(441)
    # rows >= features selects the core primal form.
    x_primal, y_primal, w_primal = rng.normal(size=(19, 5)), rng.normal(size=(19, 2)), rng.uniform(0.2, 2.0, 19)
    probe_primal = rng.normal(size=(4, 5))
    primal = ridge.fit_normalized_weighted_ridge(x_primal, y_primal, w_primal)
    beta_ref, prediction_ref = _independent_weighted_primal(x_primal, y_primal, w_primal, probe_primal)
    assert primal.solver_form == "primal"
    np.testing.assert_allclose(primal.coefficients, beta_ref, rtol=0.0, atol=3e-12)
    np.testing.assert_allclose(ridge.predict_normalized_weighted_ridge(probe_primal, primal), prediction_ref, rtol=0.0, atol=3e-12)
    # rows < features selects dual, but must yield the same coefficients/predictions
    # as the independent p×p primal equation.
    x_dual, y_dual, w_dual = rng.normal(size=(7, 13)), rng.normal(size=(7, 2)), rng.uniform(0.2, 2.0, 7)
    probe_dual = rng.normal(size=(4, 13))
    dual = ridge.fit_normalized_weighted_ridge(x_dual, y_dual, w_dual)
    beta_ref, prediction_ref = _independent_weighted_primal(x_dual, y_dual, w_dual, probe_dual)
    assert dual.solver_form == "dual"
    np.testing.assert_allclose(dual.coefficients, beta_ref, rtol=0.0, atol=4e-12)
    np.testing.assert_allclose(ridge.predict_normalized_weighted_ridge(probe_dual, dual), prediction_ref, rtol=0.0, atol=4e-12)


def test_explicit_intercept_satisfies_weighted_first_order_condition() -> None:
    x, y, weights, _probe = _data()
    readout = ridge.fit_normalized_weighted_ridge(x, y, weights)
    residual = (weights[:, None] * (y - ridge.predict_normalized_weighted_ridge(x, readout))).sum(axis=0)
    np.testing.assert_allclose(residual, 0.0, rtol=0.0, atol=2e-9)
    constant_x = np.full((8, 3), 9.0)
    constant_y = np.tile(np.asarray([[3.0, -4.0]]), (8, 1))
    constant = ridge.fit_normalized_weighted_ridge(constant_x, constant_y, np.arange(1, 9, dtype=float))
    np.testing.assert_allclose(ridge.predict_normalized_weighted_ridge(constant_x, constant), constant_y, atol=1e-12)


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "not-a-number"])
def test_missing_or_nonfinite_target_dir_fails_closed(bad: object) -> None:
    trials = [{"start": 0, "stop": 100, "target_dir": 0.1}, {"start": 100, "stop": 200, "target_dir": bad}]
    with pytest.raises(ridge.PriorityA2NumericalError, match="target_dir"):
        ridge.require_target_directions(trials)


@pytest.mark.parametrize("bad_weights", [np.asarray([1.0, 1.0, 0.0]), np.asarray([1.0, -1.0, 2.0]), np.asarray([1.0, np.nan, 2.0])])
def test_shape_finite_and_positive_inputs_fail_closed(bad_weights: np.ndarray) -> None:
    x = np.ones((3, 2))
    y = np.ones((3, 2))
    with pytest.raises(ridge.PriorityA2NumericalError):
        ridge.fit_normalized_weighted_ridge(x, y, bad_weights)
    with pytest.raises(ridge.PriorityA2NumericalError):
        ridge.fit_normalized_weighted_ridge(x, np.ones((3, 1)), np.ones(3))
    with pytest.raises(ridge.PriorityA2NumericalError):
        ridge.fit_normalized_weighted_ridge(np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]]), y, np.ones(3))


def test_old_unnormalized_formula_is_rejected_by_sealed_reference() -> None:
    """This is the exact A2a bug: data terms are sums but lambda remains 1."""
    x, y, _weights, probe = _data()
    mean, scale = x.mean(0), x.std(0)
    scale[scale < ridge.FEATURE_STD_EPS] = 1.0
    z, yc = (x - mean) / scale, y - y.mean(0)
    old_beta = np.linalg.solve(z.T @ z + np.eye(x.shape[1]), z.T @ yc)
    old_prediction = ((probe - mean) / scale) @ old_beta + y.mean(0)
    sealed = _independent_normalized_unweighted_prediction(x, y, probe, 1.0)
    assert float(np.max(np.abs(old_prediction - sealed))) > 1.0e-3


def test_nested_masks_are_stable_target_blind_and_equal_trial_weighted() -> None:
    owner = np.repeat(np.arange(3, dtype=np.int64), [18, 19, 20])
    masks_a = ridge.nested_density_masks(owner, 3, session_or_asset="asset-A", mask_seed=42)
    masks_b = ridge.nested_density_masks(owner, 3, session_or_asset="asset-A", mask_seed=42)
    for k in (1, 2, 4, 8, 16):
        np.testing.assert_array_equal(masks_a[k], masks_b[k])
    for low, high in zip((1, 2, 4, 8), (2, 4, 8, 16)):
        assert np.all(~masks_a[low] | masks_a[high])
    selected_owner = owner[masks_a[4]]
    weights = ridge.equal_trial_weights(selected_owner, 3)
    totals = np.bincount(selected_owner, weights=weights, minlength=3)
    np.testing.assert_allclose(totals, totals[0], atol=1e-12)


def test_predata_numerical_contract_passes() -> None:
    report = ridge.numerical_contract_self_test()
    assert report["uniform_reference_max_abs_error"] <= 1e-12
    assert report["weight_scaling_max_abs_error"] <= 2e-12


def test_a2b_requires_a_verified_a2a_receipt_before_any_loader(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The prerequisite check runs before the imports that can open V9/NWB data."""
    missing = tmp_path / "missing_a2a.json"
    monkeypatch.setattr(a2b_runner, "A2A_RECEIPT_PATH", missing)
    monkeypatch.setattr(a2b_runner, "RECEIPT_PATH", tmp_path / "a2b.json")
    with pytest.raises(FileNotFoundError, match="requires an existing verified A2a"):
        a2b_runner.run()


def test_runner_default_mode_only_self_tests_and_does_not_write_receipt(monkeypatch: pytest.MonkeyPatch, tmp_path, capsys) -> None:
    target = tmp_path / "would_be_a2b_receipt.json"
    monkeypatch.setattr(a2b_runner, "RECEIPT_PATH", target)
    monkeypatch.setattr("sys.argv", ["run_priority_a2_same_target_density_v2.py"])
    a2b_runner.main()
    assert '"full_run_started": false' in capsys.readouterr().out
    assert not target.exists()


def test_a2a_sealed_dense_score_loader_has_all_six_view_budget_references() -> None:
    """Local workspace artifacts contain the full 15×2×3 authoritative map."""
    values, hashes = a2a_runner._sealed_dense_scores()
    assert len(values) == 90
    assert {budget for budget, _asset, _view in values} == {15, 30, 50}
    assert {view for _budget, _asset, view in values} == {"sua", "pseudo_mua"}
    assert set(hashes) == {"ridge50_m50_aggregate_sha256", "ridge_budget_m15_m30_receipt_sha256"}


def _write_valid_a2a_receipt(path) -> dict[str, object]:
    assets = [f"asset-{index:02d}" for index in range(15)]
    cells = []
    for asset in assets:
        for view in ("sua", "pseudo_mua"):
            for budget in (15, 30, 50):
                cells.append({"asset_id": asset, "view": view, "budget": budget, "support_query_overlap_count": 0,
                              "arms": {"dense_uniform": {}, "dense_equal_trial": {}, "direction_uniform": {}, "direction_equal_trial": {}},
                              "sealed_uniform_dense_reproduction": {"absolute_error": 0.0, "atol": 5e-5}})
    payload: dict[str, object] = {
        "schema": "priority_a2a_weighting_control_v2", "status": "COMPLETED_CPU_ONLY",
        "input_bindings": {"runner_sha256": a2b_runner.sha256_file(a2b_runner.SCRIPT_DIR / "run_priority_a2_weighting_control_v2.py"),
                           "ridge_core_sha256": a2b_runner.sha256_file(a2b_runner.REPO_ROOT / "sua_exploration/mc_maze/priority_a2_normalized_ridge_v2.py")},
        "numerical_contract": {"uniform_reference_max_abs_error": 0.0, "weight_scaling_max_abs_error": 0.0, "intercept_foc_max_abs_error": 0.0},
        "cells": cells,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_a2b_a2a_receipt_validation_accepts_only_the_exact_grid(tmp_path) -> None:
    receipt = tmp_path / "a2a.json"
    _write_valid_a2a_receipt(receipt)
    binding = a2b_runner.verify_a2a_v2_receipt(receipt)
    assert binding["asset_ids"] == [f"asset-{index:02d}" for index in range(15)]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["cells"][0]["sealed_uniform_dense_reproduction"].update({"atol": 1.0}), "reproduction gate"),
        (lambda payload: payload["input_bindings"].update({"ridge_core_sha256": "0" * 64}), "core SHA"),
        (lambda payload: payload["cells"].__setitem__(1, dict(payload["cells"][0])), "duplicate"),
        (lambda payload: payload["cells"][0].update({"support_query_overlap_count": 1}), "nonzero support/query overlap"),
        (lambda payload: payload["cells"][0]["arms"].update({"post_hoc_arm": {}}), "exactly the frozen 2x2 arms"),
    ],
)
def test_a2b_rejects_adversarial_a2a_receipt(tmp_path, mutation, message: str) -> None:
    receipt = tmp_path / "forged_a2a.json"
    payload = _write_valid_a2a_receipt(receipt)
    mutation(payload)
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ridge.PriorityA2NumericalError, match=message):
        a2b_runner.verify_a2a_v2_receipt(receipt)
