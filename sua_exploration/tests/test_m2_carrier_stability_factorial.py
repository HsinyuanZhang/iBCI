"""Pure-function contracts for the held-in K4/T4 factorial audit."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "sua_exploration/scripts/audit_m2_carrier_stability_factorial.py"
SPEC = importlib.util.spec_from_file_location("carrier_factorial", PATH)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MOD)


def test_k4_splits_are_disjoint_and_prefix_bounded():
    blocks = {"trial": np.repeat(np.arange(12), 2), "block": np.arange(24)}
    for left, right in MOD.k4_split_masks(blocks, 10).values():
        assert not np.any(left & right)
        assert not np.any((left | right) & (blocks["trial"] >= 10))


def test_component_metrics_keep_loso_standardizer_explicit():
    first = MOD.CarrierFit(weights=np.array([[1.,2.],[2.,1.],[3.,1.]]), intercept=np.array([1.,2.,3.]), alpha=0., lag_bins=0)
    second = MOD.CarrierFit(weights=np.array([[2.,1.],[1.,2.],[3.,2.]]), intercept=np.array([2.,2.,4.]), alpha=0., lag_bins=0)
    out = MOD.k4_component_metrics(first, second, (np.array([1.,1.]), np.array([2.,2.])))
    assert {"wx_pearson", "wy_pearson", "w_norm_pearson", "b_pearson", "loso_standardized_norm_b_flattened_cosine"} <= set(out)


def test_t4_rank_deficient_half_is_undefined_not_pseudoinverted():
    arrays = {"sums": np.ones((4,3)), "lengths": np.ones(4), "angles": np.zeros(4)}
    feature, coverage = MOD.t4_fit(arrays, np.ones(4, bool), "synthetic")
    assert feature is None and coverage["undefined_reason"] == "direction_design_rank_not_3"


def test_t4_mask_uses_the_first_m_prefix_of_a_max_m_cache():
    arrays = {"sums": np.ones((33,3)), "lengths": np.ones(33), "angles": np.zeros(33)}
    feature, coverage = MOD.t4_fit(arrays, np.array([True, True, True, True]), "synthetic")
    assert feature is None and coverage["n_trials"] == 4


def test_balanced_indices_refuse_an_empty_direction_bin():
    blocks = {"velocity": np.tile(np.array([[1., 0.]]), (12,1))}
    selected, coverage = MOD.balanced_indices(blocks, np.ones(12, bool))
    assert selected is None and coverage["defined"] is False


def test_balanced_empty_bin_keeps_all_block_arms_present():
    blocks = {"velocity": np.tile(np.array([[1., 0.]]), (20,1)), "rate": np.ones((20,3)), "trial": np.repeat(np.arange(10),2), "block": np.arange(20)}
    out = MOD.balanced_arm(blocks, np.arange(20) % 2 == 0, np.arange(20) % 2 == 1, (np.array([1.,1.]), np.array([1.,1.])))
    assert out["balanced_ols"]["defined"] is False
    assert "all_block_ols" in out and "all_block_ridge" in out


def test_strict_json_converts_nonfinite_values_to_null():
    payload = MOD.strict_json({"x": np.nan, "y": np.inf, "z": np.array([1., -np.inf])})
    assert payload == {"x": None, "y": None, "z": [1.0, None]}


def test_nested_lag_keeps_invalid_leads_when_common_intersection_is_empty():
    blocks = {"velocity": np.ones((9,2)), "rate": np.ones((9,3)), "trial": np.arange(9), "block": np.arange(9)}
    cache = {(f"s{i}", lead): blocks for i in range(7) for lead in MOD.LEAD_GRID_BINS}
    lead, audit = MOD.select_lag_nested(cache, "s0", 8)
    assert lead is None and audit["undefined_reason"] == "no_lead_has_all_six_reference_sessions_valid"
    assert set(audit["selection_curve"]) == {str(x) for x in MOD.LEAD_GRID_BINS}
    assert all(value["valid_count"] == 0 for value in audit["selection_curve"].values())
