"""Unit tests for K4 component decomposition diagnostic helpers."""
from __future__ import annotations

import numpy as np
import pytest

from carrier_perf.k4_component_decomposition import (
    build_conclusions,
    channel_matching_accuracy,
    hybrid_margin_within_noise,
    per_channel_w_halfsplit_cosine_median,
    residual_r_squared,
)


def test_channel_matching_identical_descriptors_top1_is_one():
    rng = np.random.RandomState(0)
    desc = rng.randn(20, 4)
    out = channel_matching_accuracy(desc, desc.copy())
    assert out["top1_accuracy"] == pytest.approx(1.0)
    assert out["mean_reciprocal_rank"] == pytest.approx(1.0)
    assert out["chance_top1_accuracy"] == pytest.approx(1.0 / 20)


def test_channel_matching_shuffled_near_chance():
    rng = np.random.RandomState(1)
    n = 64
    desc_a = rng.randn(n, 4)
    perm = rng.permutation(n)
    desc_b = desc_a[perm]
    out = channel_matching_accuracy(desc_a, desc_b)
    chance = 1.0 / n
    assert out["top1_accuracy"] < chance + 0.15
    assert out["chance_top1_accuracy"] == pytest.approx(chance)


def test_residual_r_squared_exact_linear_dependence_is_near_zero():
    rng = np.random.RandomState(2)
    n = 40
    t4 = rng.randn(n, 4)
    coef = np.array([0.5, -1.2, 0.3, 2.0, 0.7])
    design = np.column_stack([np.ones(n), t4])
    k4_dim = design @ coef
    resid = residual_r_squared(k4_dim, t4)
    assert resid == pytest.approx(0.0, abs=1e-10)


def test_residual_r_squared_independent_dim_is_high():
    rng = np.random.RandomState(3)
    n = 50
    t4 = rng.randn(n, 4)
    independent = rng.randn(n)
    resid = residual_r_squared(independent, t4)
    assert resid > 0.8


def test_per_channel_w_halfsplit_cosine_identical_halves_is_one():
    rng = np.random.RandomState(4)
    w = rng.randn(16, 4).astype(np.float32)
    cos = per_channel_w_halfsplit_cosine_median(w, w.copy())
    assert cos == pytest.approx(1.0)


def test_aggregation_sensitivity_or_vs_majority_disagree():
    """OR and majority rules disagree => headline_is_aggregation_sensitive is true."""
    session_records = []
    for session_idx in range(7):
        incremental = {}
        for dim in ("w_x", "w_y", "w_norm", "baseline_rate"):
            if dim == "w_x":
                verdict = "novel_and_reliable" if session_idx == 0 else "redundant_with_t4"
                resid, reliability = 0.9, 0.9
            else:
                verdict = "redundant_with_t4"
                resid, reliability = 0.1, 0.9
            incremental[dim] = {
                "verdict": verdict,
                "residual_r_squared": resid,
                "split_half_reliability": reliability,
            }
        session_records.append({"incremental_over_t4": incremental})

    ident = {
        "hybrid_t4w_k4b": {"top1_accuracy_mean": 0.05},
        "t4_full": {"top1_accuracy_mean": 0.06},
        "k4_full": {"top1_accuracy_mean": 0.06},
    }
    conclusions = build_conclusions(session_records, ident)

    assert conclusions["any_session_or"]["recommendation"] == "fusion_worthwhile"
    assert conclusions["aggregation_sensitivity"]["majority"]["recommendation"] == "neither"
    assert conclusions["headline_is_aggregation_sensitive"] is True
    assert conclusions["aggregation_sensitivity"]["any_session_or"]["per_k4_dim_verdict"]["w_x"] == (
        "novel_and_reliable"
    )
    assert (
        conclusions["aggregation_sensitivity"]["majority"]["per_k4_dim_verdict"]["w_x"]
        == "redundant_with_t4"
    )


def test_hybrid_margin_within_noise_true_when_gap_below_se():
    diffs = [0.001, -0.001, 0.0005, -0.0005, 0.0, 0.0002]
    mean_diff = sum(diffs) / len(diffs)
    var = sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1)
    se = (var ** 0.5) / (len(diffs) ** 0.5)
    assert abs(mean_diff) < se
    assert hybrid_margin_within_noise(mean_diff, se) is True


def test_hybrid_margin_within_noise_false_when_gap_exceeds_se():
    diffs = [0.05] * 10
    mean_diff = 0.05
    se = 0.0  # zero variance => SE is 0, margin is not within noise
    assert hybrid_margin_within_noise(mean_diff, se) is False
    diffs_varied = [0.04, 0.05, 0.06, 0.05, 0.05, 0.04, 0.06, 0.05]
    mean_diff = sum(diffs_varied) / len(diffs_varied)
    var = sum((d - mean_diff) ** 2 for d in diffs_varied) / (len(diffs_varied) - 1)
    se = (var ** 0.5) / (len(diffs_varied) ** 0.5)
    assert mean_diff > se
    assert hybrid_margin_within_noise(mean_diff, se) is False
