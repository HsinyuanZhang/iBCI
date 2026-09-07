"""Focused contracts for the CPU-only K4/Gate-A reliability curve."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "sua_exploration/scripts/audit_m2_k4_split_half_curve.py"
SPEC = importlib.util.spec_from_file_location("k4_split_half_curve", PATH)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_reliability_metrics_keep_gate_a_pearson_distinct_from_cosine() -> None:
    left = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]])
    right = np.array([[2.0, 0.0], [0.0, 1.0], [1.0, 4.0]])
    result = MOD.reliability_metrics(left, right)
    assert set(result) >= {"flattened_w_pearson", "flattened_w_cosine", "per_electrode_2d_cosine_median", "per_electrode_2d_cosine_median_excluding_lowest_magnitude_quartile"}
    assert np.isfinite(result["flattened_w_pearson"])
    assert np.isfinite(result["flattened_w_cosine"])
    assert result["flattened_w_pearson"] != result["flattened_w_cosine"]


def test_threshold_minimum_requires_both_metrics_and_requested_cohort_size() -> None:
    good = {"flattened_w_pearson": 0.6, "flattened_w_cosine": 0.7}
    weak = {"flattened_w_pearson": 0.6, "flattened_w_cosine": 0.4}
    by_m = {8: [good] * 6 + [weak], 12: [good] * 7}
    assert MOD.threshold_minimum(by_m, required=6) == 8
    assert MOD.threshold_minimum(by_m, required=7) == 12


def test_script_documents_heldin_only_and_gate_a_exact_fit() -> None:
    text = PATH.read_text(encoding="utf-8")
    assert "held-in-calib" in text
    assert "fit_encoding" in text
    assert "flattened-W Pearson" in text
    assert "held-out calibration/query files" in text
