from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "sua_exploration/scripts/aggregate_m2_carrier_stability_factorial_v2.py"
SPEC = importlib.util.spec_from_file_location("aggregate_factorial_v2", PATH)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MOD)


def test_strict_maps_nonfinite_to_null():
    assert MOD.strict({"x": np.nan, "y": [np.inf, 2.]}) == {"x": None, "y": [None, 2.0]}


def test_median_ignores_null_and_nonfinite():
    assert MOD.median([None, np.nan, .2, .8]) == .5


def test_random_accounting_separates_coverage_budget_from_matched_pairs():
    arm = lambda covered, paired, valid: {"balanced_coverage": {"first": {"defined": covered}, "second": {"defined": covered}},
                                          "balanced_ols": {"defined": paired}, "random_equal_size_ols": {"defined": paired, "n_requested": 50, "n_valid": valid}}
    result = MOD.random_accounting([({}, arm(True, True, 50)), ({}, arm(True, False, 0)), ({}, arm(False, False, 0))])
    assert result == {"coverage_eligible_requested": 100, "fit_failure_arms": 1, "matched_pair_arms": 1, "matched_pair_requested": 50, "matched_pair_valid": 50}
