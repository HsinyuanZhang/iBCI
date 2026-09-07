"""Focused tests for the §5.5 zero-GPU pre-gate (Deliverable A).

Mirrors the frozen contract one property per test:

- construction (ii) generator validity (positivity discipline, shapes,
  support-only carrier fit);
- causality of the shared activity / lag-stack pipeline;
- THE core discipline: the true target y_t never enters any static or
  Fisher weight — query-bin targets can be replaced by garbage without
  changing those predictions (the leaked alignment oracle, by contrast,
  must change);
- static weights are non-negative and deterministic (frozen after fit);
- gate arithmetic: construction deltas and pass flags follow the frozen
  rules, and the two constructions can pass and fail independently.
"""

from __future__ import annotations

import copy

import pytest
import torch

from src.tfpd.synth import generate_session
from src.tfpd_lane import pregate as pg


def test_state_matched_generator_validity():
    session = pg.generate_state_matched_session(seed=7, length=128, num_units=32)
    assert session.counts.shape == (128, 32)
    assert session.behaviour.shape == (128, 2)
    assert session.carrier.shape == (32, 4)
    assert torch.isfinite(session.counts).all() and (session.counts >= 0).all()
    assert torch.isfinite(session.carrier).all()
    assert int(session.support_mask.sum()) == int(128 * pg.SUPPORT_FRACTION)
    assert bool(session.support_mask[0]) and not bool(session.support_mask[-1])


def test_state_matched_population_actually_sparse():
    """Narrow tuning + high baselines: each unit is strongly elevated
    somewhere in the session, but at any typical bin only a small matched
    minority is elevated (the construction's defining property)."""
    session = pg.generate_state_matched_session(seed=11, length=256, num_units=64)
    counts = session.counts.float()
    per_unit_median = counts.median(dim=0).values
    # every unit carries signal somewhere (all units informative)...
    assert (counts.max(dim=0).values - per_unit_median).min() > 1.0
    # ...but at a typical bin only a small state-matched minority is elevated
    elevated_fraction = ((counts - per_unit_median.unsqueeze(0)) > 2.0).float().mean()
    assert elevated_fraction < 0.25


def test_trailing_mean_causality():
    x = torch.rand(64, 16)
    a = pg.trailing_mean(x, window=8)
    perturbed = x.clone()
    perturbed[40:] += 3.0
    a2 = pg.trailing_mean(perturbed, window=8)
    assert torch.equal(a[:32], a2[:32])
    assert not torch.equal(a[-1], a2[-1])


def test_lag_stack_causality():
    x = torch.rand(32, 3)
    f = pg.lag_stack(x, lags=4)
    perturbed = x.clone()
    perturbed[20:] += 5.0
    f2 = pg.lag_stack(perturbed, lags=4)
    assert torch.equal(f[:16], f2[:16])


def test_static_weights_nonnegative_and_deterministic():
    session = generate_session(seed=3, length=256, num_units=32)
    counts = session.counts.float()
    behaviour = session.behaviour.float()
    carrier = session.carrier.float()
    support_rows, _ = pg._row_masks(session)
    w1 = pg.fit_static_weights(counts, carrier, behaviour, support_rows)
    w2 = pg.fit_static_weights(counts, carrier, behaviour, support_rows)
    assert (w1 >= 0).all()
    assert torch.equal(w1, w2)  # frozen: refit on identical support is identical


def test_no_true_target_in_static_or_fisher_weights():
    """Core §5.5 discipline: replacing the true QUERY-bin behaviour with
    garbage leaves the static and Fisher predictions bit-identical (their
    weights never read the query target; support bins are legitimate
    calibration); the alignment oracle must change (it leaks)."""
    session = generate_session(seed=5, length=256, num_units=48)
    support_end = int(session.support_mask.sum().item())
    corrupted = copy.deepcopy(session)
    torch.manual_seed(0)
    corrupted.behaviour[support_end:] = torch.rand_like(corrupted.behaviour[support_end:]) * 100.0 - 50.0

    clean = pg.run_session(session)
    dirty = pg.run_session(corrupted)

    torch.testing.assert_close(clean.predictions["static"], dirty.predictions["static"])
    torch.testing.assert_close(clean.predictions["fisher"], dirty.predictions["fisher"])
    assert not torch.allclose(clean.predictions["align"], dirty.predictions["align"])


def test_run_session_output_structure():
    session = generate_session(seed=9, length=256, num_units=32)
    result = pg.run_session(session)
    assert set(result.query_r2) == set(pg.ESTIMATORS)
    assert set(result.support_r2) == set(pg.ESTIMATORS)
    for name in pg.ESTIMATORS:
        assert isinstance(result.query_r2[name], float)
        assert result.predictions[name].shape == (256, 2)


def test_construction_gate_arithmetic():
    result = pg.run_construction("isotropic_cosine", seed=13, num_sessions=2)
    means = result["query_r2_mean"]
    assert result["deltas"]["fisher_minus_static"] == pytest.approx(
        means["fisher"] - means["static"], abs=1e-9
    )
    expected = (
        result["deltas"]["fisher_minus_static"] >= pg.DELTA_MIN
        and result["deltas"]["align_minus_static"] < pg.DELTA_MIN
    )
    assert result["passed"] is expected

    result_ii = pg.run_construction("state_matched", seed=13, num_sessions=2)
    expected_ii = result_ii["deltas"]["align_minus_static"] >= pg.DELTA_MIN
    assert result_ii["passed"] is expected_ii


def test_gate_non_vacuous_both_directions():
    """The frozen rules admit pass and fail outcomes for each construction
    (checked on synthetic delta scenarios, not on measured data)."""
    for fisher_delta, align_delta, expected in [
        (0.10, 0.01, True),  # fisher clears margin, align weak -> pass
        (0.00, 0.01, False),  # fisher ties static -> fail
        (0.10, 0.20, False),  # align oracle strong -> fail
    ]:
        means = {
            "static": 0.5,
            "fisher": 0.5 + fisher_delta,
            "align": 0.5 + align_delta,
        }
        deltas = {
            "fisher_minus_static": means["fisher"] - means["static"],
            "align_minus_static": means["align"] - means["static"],
        }
        passed = deltas["fisher_minus_static"] >= pg.DELTA_MIN and deltas["align_minus_static"] < pg.DELTA_MIN
        assert passed is expected

    for align_delta, expected in [(0.08, True), (0.02, False)]:
        passed = align_delta >= pg.DELTA_MIN
        assert passed is expected


def test_run_pregate_end_to_end_smoke():
    result = pg.run_pregate(seed=21, num_sessions=2)
    assert result["schema"] == "tfpd_pregate_timevarying_v1"
    assert set(result["constructions"]) == {"isotropic_cosine", "state_matched"}
    assert result["status"] in ("PASS", "FAIL")
    assert "leaked upper bound" in result["note"]
