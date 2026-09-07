"""Synthetic, no-data contracts for the overlap-gate decoder calibration (CPU-only).

These tests exercise the pure-math machinery (exact Spearman/permutation test,
monotonicity check, gain arithmetic, sealed-constant self-consistency, receipt
sanitation) without touching NWB data or checkpoints, matching this repo's
existing convention for CPU-only contract tests (see test_h1_lag_screen.py).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from src.data import h1_overlap_gate_decoder_calibration as cal


# --------------------------------------------------------------------------- #
# Exact Spearman / permutation machinery.
# --------------------------------------------------------------------------- #
def test_ranks_no_ties():
    assert cal._ranks([3.0, 1.0, 2.0]) == [3.0, 1.0, 2.0]


def test_ranks_with_ties_average():
    # two tied smallest values share ranks 1,2 -> average 1.5
    assert cal._ranks([5.0, 5.0, 1.0]) == [2.5, 2.5, 1.0]


def test_spearman_perfect_positive_monotone():
    result = cal.exact_spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
    assert result["rho"] == pytest.approx(1.0)
    assert result["n"] == 3
    assert result["n_permutations_enumerated"] == 6


def test_spearman_perfect_negative_monotone():
    result = cal.exact_spearman([1.0, 2.0, 3.0], [30.0, 20.0, 10.0])
    assert result["rho"] == pytest.approx(-1.0)


def test_spearman_n3_nonextreme_exact_p_is_one():
    # With n=3 and no ties, every non-extreme ordering has |rho|=0.5, so all
    # six permutations are at least as extreme and the exact two-sided p is
    # one.  Perfect orderings are different: only rho=+/-1 are as extreme.
    for y in ([3.0, 1.0, 2.0], [2.0, 3.0, 1.0]):
        result = cal.exact_spearman([1.0, 2.0, 3.0], y)
        assert result["exact_two_sided_p_value"] == pytest.approx(1.0)


def test_spearman_n3_perfect_exact_p_is_one_third():
    result = cal.exact_spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert result["exact_two_sided_p_value"] == pytest.approx(2.0 / 6.0)


def test_spearman_matches_known_n4_case():
    # x ranks [1,2,3,4], y ranks [2,4,1,3] -> rho = 0.0 (hand-derived)
    result = cal.exact_spearman([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 1.0, 3.0])
    assert result["rho"] == pytest.approx(0.0)
    assert result["n_permutations_enumerated"] == 24


def test_spearman_rejects_mismatched_length():
    with pytest.raises(cal.OverlapGateCalibrationError):
        cal.exact_spearman([1.0, 2.0], [1.0, 2.0, 3.0])


def test_spearman_rejects_n_too_large():
    with pytest.raises(cal.OverlapGateCalibrationError):
        cal.exact_spearman(list(range(10)), list(range(10)))


# --------------------------------------------------------------------------- #
# Monotonicity check: exact definition (rho attains an extreme value), not a
# heuristic threshold.
# --------------------------------------------------------------------------- #
def test_monotone_increasing_detected():
    out = cal.check_monotone([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    assert out["is_monotone_increasing"]
    assert not out["is_monotone_decreasing"]
    assert out["is_monotone"]


def test_monotone_decreasing_detected():
    out = cal.check_monotone([1.0, 2.0, 3.0], [0.3, 0.2, 0.1])
    assert out["is_monotone_decreasing"]
    assert out["is_monotone"]


def test_non_monotone_up_down_up_detected():
    # This is exactly the shape of this module's real primary-set finding:
    # sorted by x, y goes up, then down, then up (or down/up/down).
    out = cal.check_monotone([1.0, 2.0, 3.0], [0.3, -0.1, 0.2])
    assert not out["is_monotone_increasing"]
    assert not out["is_monotone_decreasing"]
    assert not out["is_monotone"]


def test_non_monotone_ties_are_not_forced_monotone_by_le_ge():
    # A flat then rising sequence is monotone-non-decreasing by the <= rule;
    # verify the boundary behaves as documented (ties count as monotone).
    out = cal.check_monotone([1.0, 2.0, 3.0], [0.1, 0.1, 0.2])
    assert out["is_monotone_increasing"]


# --------------------------------------------------------------------------- #
# Sealed-constant arithmetic: gains must equal exactly what CURRENT_RESULTS.md
# and the terminal-eval receipts report (to full float precision, since these
# are frozen numbers, not re-derived estimates).
# --------------------------------------------------------------------------- #
def test_sealed_decoder_r2_gains_match_cited_values():
    gain_h_c = cal.SEALED_DECODER_R2["h_c"] - cal.SEALED_DECODER_R2["h_c0"]
    gain_h_rs = cal.SEALED_DECODER_R2["h_rs"] - cal.SEALED_DECODER_R2["h_c0"]
    gain_h_ls = cal.SEALED_DECODER_R2["h_ls"] - cal.SEALED_DECODER_R2["h_c0"]
    # CURRENT_RESULTS.md 2026-08-07 21:10 HKT / 23:28 HKT tables (rounded to 1e-6).
    assert gain_h_c == pytest.approx(0.0388952, abs=1e-6)
    assert gain_h_rs == pytest.approx(-0.0272239, abs=1e-6)
    assert gain_h_ls == pytest.approx(0.0132797, abs=1e-6)


def test_checkpoint_path_keys_match_sealed_r2_keys():
    assert set(cal.CHECKPOINT_PATHS) == set(cal.SEALED_DECODER_R2)
    assert set(cal.CHECKPOINT_FILE_SHA256) == set(cal.SEALED_DECODER_R2)


def test_all_checkpoint_sha256_are_64_char_hex():
    for key, digest in cal.CHECKPOINT_FILE_SHA256.items():
        assert len(digest) == 64, key
        int(digest, 16)  # raises if not hex


# --------------------------------------------------------------------------- #
# Pooled/per-column residual aggregation, on synthetic carriers/activity.
# --------------------------------------------------------------------------- #
def test_pooled_and_per_column_residual_synthetic():
    rng = np.random.default_rng(0)
    # The production overlap statistic is intentionally bound to the fixed
    # H1 array width.  Synthetic contracts therefore use that same width.
    n_neurons = 176
    activity_dim = 8
    columns = ["c0", "c1", "c2", "c3"]
    carriers = {}
    activity = {}
    for name in cal.H1_M4_FOLD0_SOURCE:
        act = rng.normal(size=(n_neurons, activity_dim))
        # carrier perfectly linear in activity -> residual should be ~0
        beta = rng.normal(size=(activity_dim, 4))
        carrier = act @ beta
        carriers[name] = carrier
        activity[name] = act
    out = cal.pooled_and_per_column_residual(carriers, activity, columns)
    assert out["pooled_median"] < 1e-8
    for c in columns:
        assert out["per_column_median"][c] < 1e-8


def test_pooled_and_per_column_residual_independent_carrier_is_near_one():
    rng = np.random.default_rng(1)
    n_neurons = 176
    activity_dim = 8
    columns = ["c0", "c1", "c2", "c3"]
    carriers = {}
    activity = {}
    for name in cal.H1_M4_FOLD0_SOURCE:
        activity[name] = rng.normal(size=(n_neurons, activity_dim))
        carriers[name] = rng.normal(size=(n_neurons, 4))  # independent of activity
    out = cal.pooled_and_per_column_residual(carriers, activity, columns)
    assert out["pooled_median"] > 0.5


# --------------------------------------------------------------------------- #
# Receipt canonicalization: refuses to overwrite, deterministic bytes -> sha256.
# --------------------------------------------------------------------------- #
def test_receipt_write_refuses_overwrite(tmp_path):
    from src.data.h1_content_lever_screen import write_receipt

    target = tmp_path / "receipt.json"
    info1 = write_receipt(target, {"a": 1})
    assert target.exists()
    with pytest.raises(FileExistsError):
        write_receipt(target, {"a": 1})
    # sha256 matches the actual bytes on disk
    assert info1["receipt_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# NOT_RECONSTRUCTED_ARMS inventory: must be present and honestly flagged.
# --------------------------------------------------------------------------- #
def test_not_reconstructed_inventory_present_and_flagged():
    inv = cal.NOT_RECONSTRUCTED_ARMS
    assert "sua_t4_family" in inv
    assert "rt_family" in inv
    for family in inv.values():
        assert family["activity_path_reconstructable"] is False
        assert len(family["arms"]) > 0


def test_candidate_residuals_cited_are_the_three_named_in_the_task():
    assert set(cal.CANDIDATE_RESIDUALS_CITED) == {
        "l_a_pooled_median_W_column",
        "l_c_pooled_median_W_column",
        "population_structure_carrier",
    }
    assert cal.CANDIDATE_RESIDUALS_CITED["l_a_pooled_median_W_column"] == pytest.approx(0.364)
    assert cal.CANDIDATE_RESIDUALS_CITED["l_c_pooled_median_W_column"] == pytest.approx(0.580)
    assert cal.CANDIDATE_RESIDUALS_CITED["population_structure_carrier"] == pytest.approx(0.190)
