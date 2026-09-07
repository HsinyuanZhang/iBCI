"""Unit tests for remaining CPU experiment helpers."""
from __future__ import annotations

import math

import numpy as np
import pytest

from carrier_perf.p1_closeout import bootstrap_mean_ci
from carrier_perf.p1_estimators import TuningFit
from carrier_perf.p2_rt_gocue_audit import MIN_USABLE_FRACTION
from carrier_perf.p4_harmonic_strata import (
    circular_dispersion,
    fit_harmonic,
    higher_harmonic_ratio,
    _monotonic_nondecreasing,
)
from carrier_perf.p5b_confidence_gate import gate_fits, shrink_fits


def test_circular_dispersion_aligned_is_near_zero():
    phis = np.zeros(5)
    w = np.ones(5)
    assert circular_dispersion(phis, w) == pytest.approx(0.0, abs=1e-12)


def test_circular_dispersion_opposite_is_high():
    phis = np.array([0.0, math.pi])
    w = np.array([1.0, 1.0])
    assert circular_dispersion(phis, w) == pytest.approx(1.0, abs=1e-12)


def test_harmonic_fit_recovers_known_second_order():
    rng = np.random.RandomState(0)
    angles = rng.uniform(-math.pi, math.pi, 80)
    rates = (
        2.0
        + 1.0 * np.cos(angles)
        - 0.5 * np.sin(angles)
        + 0.8 * np.cos(2 * angles)
        + 0.3 * np.sin(2 * angles)
    )
    fit = fit_harmonic(angles, rates)
    assert fit.status == "ok"
    assert fit.a == pytest.approx(1.0, abs=1e-6)
    assert fit.a2 == pytest.approx(0.8, abs=1e-6)
    assert higher_harmonic_ratio(fit.a, fit.c, fit.a2, fit.c2) > 0.2


def test_monotonic_helper():
    assert _monotonic_nondecreasing([0.1, 0.2, 0.3])
    assert not _monotonic_nondecreasing([0.1, 0.3, 0.2])


def test_bootstrap_ci_covers_mean():
    vals = [0.1, 0.2, 0.15, 0.12, 0.18]
    ci = bootstrap_mean_ci(vals, n_boot=500, seed=1)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
    assert ci["n"] == 5


def test_p5b_gate_zeros_high_residual():
    fits = [
        TuningFit(1, 0, 1, 0, "B", 10, 3, status="ok"),
        TuningFit(1, 0, 1, 0, "B", 10, 3, status="ok"),
    ]
    vars_ = np.array([0.1, 10.0])
    gated = gate_fits(fits, vars_)
    assert gated[0].status == "ok"
    assert gated[1].status == "degenerate_zeros"


def test_p5b_shrink_reduces_modulation():
    fit = TuningFit(a=2.0, c=0.0, m=2.0, b=1.0, estimator="B", n_rows=10, design_rank=3, status="ok")
    shrunk = shrink_fits([fit], np.array([4.0]), strength=3.0)[0]
    assert shrunk.m < fit.m
    assert shrunk.b == fit.b


def test_rt_floor_predeclared():
    assert MIN_USABLE_FRACTION == 0.50
