"""Additional CPU tests for Stage A/B helpers, P3 receipt, and P5a production policy."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from carrier_perf.p1_estimators import TuningFit
from carrier_perf.p1_stage_a_audit import _pv_decode_r2, _rate_mse
from carrier_perf.p1_stage_b_audit import fit_estimator_c_kernel_smoothed
from carrier_perf.p3_baseline_receipt import build_p3_stage_a_receipt
from carrier_perf.p5a_degeneracy import apply_degeneracy_policy, decide_direction_degeneracy


def test_rate_mse_prefers_correct_fit():
    angles = np.linspace(-math.pi, math.pi, 16, endpoint=False)
    true = TuningFit(a=1.0, c=0.0, m=1.0, b=2.0, estimator="B", n_rows=16, design_rank=3, status="ok")
    wrong = TuningFit(a=0.0, c=1.0, m=1.0, b=2.0, estimator="B", n_rows=16, design_rank=3, status="ok")
    rates = (true.b + true.a * np.cos(angles) + true.c * np.sin(angles))[None, :]
    assert _rate_mse([true], angles, rates) < _rate_mse([wrong], angles, rates)


def test_pv_decode_r2_recovers_aligned_velocity():
    rng = np.random.RandomState(0)
    n_units, n_s, n_e = 8, 20, 20
    fits = [
        TuningFit(
            a=float(np.cos(i)),
            c=float(np.sin(i)),
            m=1.0,
            b=0.0,
            estimator="B",
            n_rows=n_s,
            design_rank=3,
            status="ok",
        )
        for i in np.linspace(0, 2 * math.pi, n_units, endpoint=False)
    ]
    dirs = np.stack([[f.a, f.c] for f in fits], axis=0)
    support_rates = rng.randn(n_units, n_s)
    eval_rates = rng.randn(n_units, n_e)
    # Construct velocity as affine of population vector so R2 ~ 1.
    pv_s = (support_rates.T @ dirs) / n_units
    pv_e = (eval_rates.T @ dirs) / n_units
    A = np.array([[1.2, 0.1], [-0.05, 0.9]])
    b = np.array([0.3, -0.2])
    support_vel = pv_s @ A + b
    eval_vel = pv_e @ A + b
    r2 = _pv_decode_r2(fits, support_rates, support_vel, eval_rates, eval_vel)
    assert r2 == pytest.approx(1.0, abs=1e-6)


def test_estimator_c_smoothes_without_bin_snap():
    rng = np.random.RandomState(1)
    angles = rng.uniform(-math.pi, math.pi, size=40)
    rates = 3.0 + 1.5 * np.cos(angles) - 0.7 * np.sin(angles) + rng.normal(0, 0.05, size=40)
    fit = fit_estimator_c_kernel_smoothed(angles, rates)
    assert fit.status == "ok"
    assert fit.estimator == "C_kernel_smooth"
    assert fit.a == pytest.approx(1.5, abs=0.3)
    assert fit.c == pytest.approx(-0.7, abs=0.3)


def test_p3_receipt_matches_sealed_aggregates():
    from pathlib import Path

    spint = Path(__file__).resolve().parents[3]
    receipt = build_p3_stage_a_receipt(spint_root=spint)
    assert receipt["status"] == "completed_cpu_only"
    assert receipt["structural_claim_rs4_and_ls4_below_z4"] is True
    assert receipt["exact_means_match_sealed_aggregates"] is True


def test_p5a_raise_is_default_fail_closed():
    decision = decide_direction_degeneracy(present_directions=1, num_channels=10)
    assert decision.action == "raise"
    with pytest.raises(ValueError, match="refusing silent"):
        apply_degeneracy_policy(decision)


def test_production_degeneracy_raise(monkeypatch):
    import sys
    from pathlib import Path

    sua = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(sua))
    monkeypatch.delenv("SUA_DIRECTION_DEGENERACY_POLICY", raising=False)
    from mc_maze.unit_side_features import enforce_direction_degeneracy_policy

    with pytest.raises(ValueError, match="P5a fail-closed"):
        enforce_direction_degeneracy_policy(
            present_directions=1,
            num_channels=5,
            session_name="fake-rt",
        )


def test_production_degeneracy_warn_and_fill(monkeypatch):
    import sys
    from pathlib import Path

    sua = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(sua))
    from mc_maze.unit_side_features import enforce_direction_degeneracy_policy

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        n = enforce_direction_degeneracy_policy(
            present_directions=0,
            num_channels=7,
            session_name="fake-co",
            policy="warn_and_fill",
        )
    assert n == 7
    assert any("warn_and_fill" in str(w.message) for w in caught)
