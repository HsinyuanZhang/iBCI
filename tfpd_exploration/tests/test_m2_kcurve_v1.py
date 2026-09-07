"""Synthetic no-data/no-CUDA tests for the M2 labeled-pair k-curve V1.

The review-critical properties:

1. the first-30 candidate law is the sealed isfinite mask (fail-closed);
2. the D-opt-prefix law: the support at k is the FIRST k of one frozen greedy
   order (nested, deterministic), the k=4 prefix IS the sealed M4 D-opt law,
   and k outside [4, usable] fails closed;
3. the usable-cap law: the grid is capped at min(per-session usable), the
   endpoint is exactly that minimum, capping is disclosed;
4. the act30 activity law: the direct first-30 spelling is k-independent and
   bitwise-equal to the sealed ``select_activity_rows`` on the
   guard-admissible cardinalities (the sealed guard rejects k=5);
5. the k-row CDM binding laws: the generalized binding reproduces the sealed
   ``fit_initial_carrier`` carrier bit-for-bit at k=4 and preserves the
   support + FIFO == 30 stack invariant with the FIFO rolling at capacity;
6. the k-curve table law: paired per-session values at every k, means,
   deficits, fail-closed on session-set drift or a missing endpoint;
7. the saturation law: k* is the smallest k within 0.005 of the endpoint in
   the deficit sense (a mean above the endpoint qualifies), the endpoint
   always qualifies, the verdict string is exactly ``SATURATES_AT_<k*>``;
8. the monotonicity law: adjacent deltas, every dip disclosed;
9. the anchor matchers and the cross-family support agreement bind;
10. the plan binds the sealed foundations and the pre-registered grid.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "tfpd_exploration", ROOT / "tfpd_exploration" / "src",
             ROOT / "sua_exploration"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.m2_kcurve_v1 import laws, plan  # noqa: E402
from src.m2_t4_activity_budget_screen_v1.core import (  # noqa: E402
    ScreenError,
    select_activity_rows,
)
# the B3S rows must be typed with the SAME causal_dual_memory_cell_d_v1
# module object the k-row binding consumes (the tfpd_exploration.src
# spelling): cdm.ActivityMemory narrows by isinstance and the two import
# spellings of one file are distinct module objects (the sealed G-package
# comment's law).
from tfpd_exploration.src.causal_dual_memory_cell_d_v1 import (  # noqa: E402
    core as cdm_core,
)
from tfpd_exploration.src.pseudo_mua_precision_cdm_v2_screen_v1 import (  # noqa: E402
    core as pseudo_core,
)


def _angles(count: int = 30, *, seed: int = 5) -> np.ndarray:
    """The roster's alternating directional pattern: odd positions labeled."""
    rng = np.random.default_rng(seed)
    values = rng.uniform(-math.pi, math.pi, size=count)
    values[0::2] = np.nan
    return np.ascontiguousarray(values, dtype=np.float64)


def _curve(seed: int = 3, *, ks=(4, 5, 6, 8, 10, 12, 15), sessions=("sesA", "sesB", "sesC")):
    """A rising synthetic k-curve with a known endpoint."""
    rng = np.random.default_rng(seed)
    base = {name: rng.uniform(0.2, 0.4) for name in sessions}
    per_k = {}
    for k in ks:
        gain = 0.9 * (1.0 - math.exp(-(k - 4) / 3.0))
        per_k[int(k)] = {name: base[name] + gain + 0.001 * ((hash(name) + k) % 3) for name in sessions}
    return per_k


# ---------------------------------------------------------------------------
# 1. the first-30 candidate law.
# ---------------------------------------------------------------------------


def test_first30_candidates_is_the_sealed_isfinite_mask() -> None:
    theta = _angles(seed=11)
    assert laws.first30_candidates(theta).tolist() == list(range(1, 30, 2))
    sparse = np.full(30, np.nan)
    sparse[[2, 4, 5, 9, 28]] = [0.1, 0.9, -0.7, 2.2, -2.9]
    assert laws.first30_candidates(sparse).tolist() == [2, 4, 5, 9, 28]
    # labels beyond position 29 never enter the pool
    extended = np.concatenate([sparse, np.linspace(-3.0, 3.0, 10)])
    assert laws.first30_candidates(extended).tolist() == [2, 4, 5, 9, 28]


def test_first30_candidates_fails_closed() -> None:
    with pytest.raises(laws.KCurveLawError):
        laws.first30_candidates(np.asarray([0.1, 0.2, 0.3]))  # no first-30 container
    broken = np.full(30, np.nan)
    broken[[0, 1, 2]] = [0.1, 0.2, 0.3]
    with pytest.raises(laws.KCurveLawError):
        laws.first30_candidates(broken)  # fewer than four directional rows


def test_usable_count_matches_candidates() -> None:
    theta = _angles(seed=13)
    assert laws.usable_count(theta) == laws.first30_candidates(theta).size == 15


# ---------------------------------------------------------------------------
# 2. the D-opt-prefix law.
# ---------------------------------------------------------------------------


def test_greedy_order_is_a_deterministic_permutation_of_candidates() -> None:
    theta = _angles(seed=17)
    order = laws.dopt_greedy_order(theta)
    assert sorted(order.tolist()) == laws.first30_candidates(theta).tolist()
    assert np.array_equal(order, laws.dopt_greedy_order(theta))


def test_prefix_supports_are_nested_first_k_of_one_order() -> None:
    for seed in (5, 11, 23, 31, 47):
        theta = _angles(seed=seed)
        nesting = laws.prefix_nesting(theta, (4, 5, 6, 8, 10, 12, 15))
        assert nesting["prefix_exact"] and nesting["prefix_nested"]
        assert nesting["k4_prefix_is_sealed_m4_law"] is True
        order = np.asarray(nesting["greedy_order_positions"])
        for k in (4, 5, 6, 8, 10, 12, 15):
            assert np.array_equal(
                laws.kcurve_support(theta, k), np.sort(order[:k]))


def test_k4_prefix_is_the_sealed_m4_law() -> None:
    from src.m2_chrono4_strict_v1 import laws as chrono_laws

    for seed in (7, 19, 41):
        theta = _angles(seed=seed)
        assert np.array_equal(
            laws.kcurve_support(theta, 4), chrono_laws.dopt_support_indices(theta))


def test_kcurve_support_fails_closed_outside_the_cap() -> None:
    theta = _angles(seed=29)  # 15 usable candidates
    with pytest.raises(laws.KCurveLawError):
        laws.kcurve_support(theta, 3)  # below the M4 anchor cardinality
    with pytest.raises(laws.KCurveLawError):
        laws.kcurve_support(theta, 16)  # above the session's usable count
    sparse = np.full(30, np.nan)
    sparse[[1, 3, 5, 7, 9, 11]] = np.linspace(-2.8, 2.8, 6)
    with pytest.raises(laws.KCurveLawError):
        laws.kcurve_support(sparse, 7)  # only six usable candidates


def test_endpoint_support_is_every_usable_row() -> None:
    theta = _angles(seed=37)
    assert np.array_equal(laws.endpoint_support(theta), laws.first30_candidates(theta))
    assert int(laws.endpoint_support(theta).max()) < 30


# ---------------------------------------------------------------------------
# 3. the usable-cap law.
# ---------------------------------------------------------------------------


def test_usable_cap_full_grid_when_every_session_qualifies() -> None:
    census = {"within|sesA": 15, "within|sesB": 15, "external|sesC": 15}
    cap = laws.usable_cap(census)
    assert cap["min_usable"] == 15
    assert cap["effective_grid"] == [4, 5, 6, 8, 10, 12, 15]
    assert cap["capped"] is False and cap["dropped_by_cap"] == []
    assert cap["endpoint_k"] == 15 and cap["endpoint_is_all_usable"] is True


def test_usable_cap_caps_and_discloses() -> None:
    census = {"within|sesA": 15, "within|sesB": 12, "external|sesC": 15}
    cap = laws.usable_cap(census)
    assert cap["min_usable"] == 12
    assert cap["effective_grid"] == [4, 5, 6, 8, 10, 12]
    assert cap["capped"] is True and cap["dropped_by_cap"] == [15]
    assert cap["endpoint_k"] == 12 and cap["endpoint_is_all_usable"] is True


def test_usable_cap_fails_closed() -> None:
    with pytest.raises(laws.KCurveLawError):
        laws.usable_cap({})
    with pytest.raises(laws.KCurveLawError):
        laws.usable_cap({"sesA": 3})  # cannot bind even the M4 anchor


# ---------------------------------------------------------------------------
# 4. the act30 activity law.
# ---------------------------------------------------------------------------


def _calibration(trials: int = 32) -> np.ndarray:
    """The sealed M2 calibration topology: [trials, 100 bins, 96 channels]."""
    rng = np.random.default_rng(9)
    return np.ascontiguousarray(rng.random((trials, 100, 96)), dtype=np.float32)


def test_act30_activity_is_the_full_first30_block_and_k_independent() -> None:
    calibration = _calibration()
    activity = laws.act30_activity_rows(calibration)
    assert activity.shape == (30, 100, 96)
    assert np.array_equal(activity, calibration[:30])
    # the activity never moves with k
    for k in (4, 5, 8, 15):
        assert np.array_equal(laws.act30_activity_rows(calibration), activity)


def test_act30_parity_with_sealed_helper_on_admissible_cardinalities() -> None:
    calibration = _calibration()
    for support in ([1, 5, 9, 13], [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]):
        parity = laws.act30_parity_with_sealed_helper(calibration, support)
        assert parity["bitwise_equal"] is True
    # the sealed helper's own guard rejects a k=5 support: the direct
    # spelling is the same arithmetic unguarded (the disclosed deviation)
    with pytest.raises(ScreenError):
        select_activity_rows(
            calibration, selected_indices=np.asarray([1, 3, 5, 7, 9]),
            activity_budget=30,
        )


def test_sealed_act30_helper_returns_the_first30_block() -> None:
    # the sealed branch the direct spelling mirrors, on its own terms
    calibration = _calibration()
    sealed = select_activity_rows(
        calibration, selected_indices=np.arange(30, dtype=np.int64),
        activity_budget=30,
    )
    assert np.array_equal(sealed, calibration[:30])


def test_act30_fails_closed_on_short_blocks() -> None:
    with pytest.raises(laws.KCurveLawError):
        laws.act30_activity_rows(_calibration(29))


# ---------------------------------------------------------------------------
# 5. the k-row CDM binding laws.
# ---------------------------------------------------------------------------


def _synthetic_support() -> dict[str, object]:
    rng = np.random.default_rng(21)
    channels = np.arange(6, dtype=np.int64)
    digest = cdm_core.channel_order_digest(channels)
    theta30 = _angles(seed=21)
    rates30 = np.ascontiguousarray(rng.random((30, 6)) * 40.0 + 5.0)
    support_b3s = tuple(
        cdm_core.B3SInterpolatedSpikeCountTrial(
            activity=np.ascontiguousarray(rng.random((100, 6)), dtype=np.float32),
            session_id="sesA", trial_id=f"sesA:trial:{index}",
            channel_order_sha256=digest,
        )
        for index in range(30)
    )
    return {
        "support_rates_hz30": rates30,
        "theta30": theta30,
        "support_b3s": support_b3s,
        "channels": channels,
    }


def _krow_binding(support: dict[str, object], indices: np.ndarray):
    from src.m2_kcurve_v1 import physical

    return physical._krow_carrier_and_activity(support, indices)


def test_krow_binding_reproduces_the_sealed_carrier_bitwise_at_k4() -> None:
    support = _synthetic_support()
    indices = laws.kcurve_support(support["theta30"], 4)
    carrier, activity = _krow_binding(support, indices)
    sealed_carrier, _posterior = pseudo_core.fit_initial_carrier(
        support_rates=np.ascontiguousarray(support["support_rates_hz30"][indices]),
        direction_indices=pseudo_core.canonical_direction_indices(
            np.asarray(support["theta30"])[indices]),
        channel_ids=support["channels"],
        valid_mask=np.ones(6, dtype=np.bool_),
    )
    assert np.array_equal(
        np.ascontiguousarray(carrier, dtype=np.float64),
        np.ascontiguousarray(np.asarray(sealed_carrier.active_t4), dtype=np.float64))
    assert activity.stack().shape == (4, 100, 6)


def test_krow_binding_preserves_the_30_stack_ceiling_and_rolls() -> None:
    support = _synthetic_support()
    channels = support["channels"]
    digest = cdm_core.channel_order_digest(channels)
    for k in (4, 5, 8, 15):
        indices = laws.kcurve_support(support["theta30"], k)
        _carrier, activity = _krow_binding(support, indices)
        assert activity.fifo_capacity == 30 - k
        assert activity.stack().shape[0] == k
        query = cdm_core.B3SInterpolatedSpikeCountTrial(
            activity=np.ascontiguousarray(
                np.random.default_rng(k).random((100, 6)), dtype=np.float32),
            session_id="sesA", trial_id=f"sesA:query:{k}",
            channel_order_sha256=digest,
        )
        grown = activity.after_completed_trial(query)
        assert grown.stack().shape[0] == k + 1
        rolled = grown
        for step in range(30 - k):
            rolled = rolled.after_completed_trial(cdm_core.B3SInterpolatedSpikeCountTrial(
                activity=np.ascontiguousarray(
                    np.random.default_rng(100 + step).random((100, 6)), dtype=np.float32),
                session_id="sesA", trial_id=f"sesA:query:{k}:{step}",
                channel_order_sha256=digest,
            ))
        assert rolled.stack().shape[0] == 30  # the frozen ceiling binds


# ---------------------------------------------------------------------------
# 6. the k-curve table law.
# ---------------------------------------------------------------------------


def test_kcurve_table_pairs_sessions_and_computes_deficits() -> None:
    per_k = _curve()
    table = laws.kcurve_table(per_k, endpoint_k=15)
    assert table["k_order"] == [4, 5, 6, 8, 10, 12, 15]
    assert table["endpoint_k"] == 15
    for k in per_k:
        assert table["equal_session_means"][str(k)] == pytest.approx(
            sum(per_k[k].values()) / len(per_k[k]), rel=1e-12)
        assert set(table["per_session_r2"][str(k)]) == set(per_k[k])
    assert table["deficit_to_endpoint"]["15"] == pytest.approx(0.0, abs=1e-15)
    assert table["deficit_to_endpoint"]["4"] == pytest.approx(
        table["endpoint_mean"] - table["equal_session_means"]["4"])


def test_kcurve_table_fails_closed_on_pairing_drift() -> None:
    per_k = _curve()
    drifted = dict(per_k)
    drifted[5] = {key: value for key, value in per_k[5].items() if key != "sesC"}
    with pytest.raises(laws.KCurveLawError):
        laws.kcurve_table(drifted, endpoint_k=15)
    with pytest.raises(laws.KCurveLawError):
        laws.kcurve_table(per_k, endpoint_k=11)  # endpoint not in the curve
    with pytest.raises(laws.KCurveLawError):
        laws.kcurve_table({}, endpoint_k=15)


# ---------------------------------------------------------------------------
# 7. the saturation law.
# ---------------------------------------------------------------------------


def test_saturation_smallest_k_within_tolerance() -> None:
    per_k = _curve(seed=3)
    result = laws.saturation_point(per_k, endpoint_k=15)
    endpoint = result["endpoint_mean"]
    expected = next(
        k for k in (4, 5, 6, 8, 10, 12, 15)
        if endpoint - result["equal_session_means"][str(k)] <= 0.005)
    assert result["k_star"] == expected
    assert result["verdict"] == f"SATURATES_AT_{expected}"
    assert result["k_star"] in result["qualifying_ks"]


def test_saturation_deficit_sense_allows_above_endpoint() -> None:
    per_k = _curve(seed=3)
    # lift k=5 far ABOVE the endpoint: negative deficit qualifies
    per_k[5] = {name: value + 0.02 for name, value in per_k[15].items()}
    result = laws.saturation_point(per_k, endpoint_k=15)
    assert result["k_star"] == 5
    assert result["k_star_deficit"] < 0.0


def test_saturation_boundary_semantics() -> None:
    # a deficit comfortably below the tolerance qualifies, one comfortably
    # above does not; the exact 0.005 boundary is float-representation
    # dependent and therefore carries the 1e-12 disclosure band instead
    per_k = {4: {"sesA": 0.60, "sesB": 0.60}, 15: {"sesA": 0.6049, "sesB": 0.6049}}
    inside = laws.saturation_point(per_k, endpoint_k=15)
    assert inside["k_star"] == 4
    per_k[15] = {"sesA": 0.6051, "sesB": 0.6051}
    outside = laws.saturation_point(per_k, endpoint_k=15)
    assert outside["k_star"] == 15
    assert "boundary_bands" in outside and "any_within_epsilon_band_of_boundary" in outside


def test_saturation_endpoint_always_qualifies_and_fails_closed() -> None:
    per_k = {4: {"sesA": 0.1, "sesB": 0.1}, 15: {"sesA": 0.9, "sesB": 0.9}}
    result = laws.saturation_point(per_k, endpoint_k=15)
    assert result["k_star"] == 15 and result["k_star_deficit"] == pytest.approx(0.0)
    with pytest.raises(laws.KCurveLawError):
        laws.saturation_point({}, endpoint_k=15)


# ---------------------------------------------------------------------------
# 8. the monotonicity law.
# ---------------------------------------------------------------------------


def test_monotonicity_reports_dips() -> None:
    per_k = _curve(seed=3)
    clean = laws.monotonicity(per_k, endpoint_k=15)
    assert clean["mean_monotone_nondecreasing"] is True
    assert clean["dips"] == []
    assert len(clean["adjacent_deltas"]) == 6
    dipped = dict(per_k)
    # push k=8 below k=6: the 6 -> 8 adjacent delta must turn negative
    dipped[8] = {name: value - 0.01 for name, value in per_k[6].items()}
    dirty = laws.monotonicity(dipped, endpoint_k=15)
    assert dirty["mean_monotone_nondecreasing"] is False
    assert [item["from_k"] for item in dirty["dips"]] == [6]
    assert dirty["dips"][0]["to_k"] == 8
    assert dirty["dips"][0]["delta"] < 0.0
    # the per-session disclosure counts the sessions whose own R2 dipped
    assert all(count >= 1 for count in dirty["per_session_dip_counts"].values())


# ---------------------------------------------------------------------------
# 9. the anchor matchers and the cross-family agreement.
# ---------------------------------------------------------------------------


def test_anchor_matchers_bind() -> None:
    sealed = {
        "window_count": 12, "ordered_window_starts_sha256": "abc",
        "query_starts_sha256": "abc", "target_sha256": "def", "r2": 0.5,
    }
    cell = {
        "window_count": 12, "ordered_window_starts_sha256": "abc",
        "target_sha256": "def", "r2": 0.5 + 9e-6,
    }
    anchored = laws.fidelity_anchor(cell, sealed, r2_tolerance=1e-5)
    assert anchored["exact_match"] is True
    assert anchored["r2_within_tolerance"] is True
    assert laws.fidelity_anchor({**cell, "r2": 0.5 + 2e-5}, sealed,
                                r2_tolerance=1e-5)["exact_match"] is False
    assert laws.fidelity_anchor({**cell, "target_sha256": "xyz"}, sealed,
                                r2_tolerance=1e-5)["exact_match"] is False
    proof = laws.dopt_proof([1, 5, 9, 13], [1, 5, 9, 13])
    assert proof["exact_match"] is True
    assert laws.dopt_proof([1, 5, 9, 13], [1, 5, 9, 17])["exact_match"] is False


def test_cross_family_support_agreement() -> None:
    theta = _angles(seed=53)
    entry = laws.prefix_nesting(theta, (4, 5, 15))
    assert laws.cross_family_support_agreement(
        {"within_post30|sesA": entry}, {"within_post30|sesA": entry})
    drifted = {**entry, "greedy_order_positions": list(reversed(entry["greedy_order_positions"]))}
    with pytest.raises(laws.KCurveLawError):
        laws.cross_family_support_agreement(
            {"within_post30|sesA": entry}, {"within_post30|sesA": drifted})
    with pytest.raises(laws.KCurveLawError):
        laws.cross_family_support_agreement(
            {"within_post30|sesA": entry}, {"external_post30_local|sesB": entry})


# ---------------------------------------------------------------------------
# 10. the plan bindings.
# ---------------------------------------------------------------------------


def test_plan_binds_the_pre_registered_grid_and_readouts() -> None:
    assert plan.K_GRID_REQUESTED == (4, 5, 6, 8, 10, 12, 15)
    assert plan.K_ANCHOR_LOW == 4
    assert plan.K_ALL_USABLE_EXPECTED == 15
    assert plan.SURFACES == ("within_post30", "external_post30_local")
    assert plan.READOUT_LAW["saturation"]["tolerance"] == 0.005
    assert plan.READOUT_LAW["verdicts"]["format"] == "SATURATES_AT_<k*>"
    assert plan.RIDGE_NORMALIZED_LAMBDA == 0.1
    assert plan.EXPECTED_WITHIN_SESSIONS == 7
    assert plan.EXPECTED_EXTERNAL_SESSIONS == 6
    assert plan.ACTIVITY_STACK_LIMIT == 30
    assert plan.ENVIRONMENT_LAW["cuda_visible_devices"] == ""
    assert plan.ENVIRONMENT_LAW["torch_num_threads"] == 4


def test_plan_pins_the_sealed_foundations() -> None:
    import re

    hex64 = re.compile(r"^[0-9a-f]{64}$")
    for name, value in (
        ("budget_screen", plan.BUDGET_SCREEN_SCORE_SHA256),
        ("comparator", plan.COMPARATOR_SCORE_SHA256),
        ("cdm_screen", plan.CDM_SCREEN_SCORE_SHA256),
        ("g_terminal", plan.G_TERMINAL_SHA256),
        ("t4_checkpoint", plan.T4_CHECKPOINT_SHA256),
        ("normalization", plan.NORMALIZATION_SHA256),
    ):
        assert hex64.match(value), name
    anchors = plan.ANCHORS["sealed_rows_read_from_receipts"]
    assert anchors["static_k4"]["cell"] == "ridge_activity30_m4"
    assert anchors["static_k4"]["expected_means"]["within_post30"] == pytest.approx(0.677220, abs=1e-6)
    assert anchors["static_k4"]["expected_means"]["external_official_query"] == pytest.approx(0.290992, abs=1e-6)
    assert anchors["static_kall"]["cell"] == "ridge_static_m30"
    assert anchors["static_kall"]["expected_means"]["within_post30"] == pytest.approx(0.689261, abs=1e-6)
    assert anchors["static_kall"]["expected_means"]["external_official_query"] == pytest.approx(0.295220, abs=1e-6)
    assert anchors["cdm_k4"]["cell"] == "m4_activity_only"
    assert anchors["cdm_k4"]["expected_means"]["external_post30_local"] == pytest.approx(0.299057, abs=1e-6)
    assert anchors["cdm_k4"]["expected_means"]["within_post30"] == pytest.approx(0.654086, abs=1e-6)
    assert plan.ANCHORS["direct_surface_anchors"]["r2_tolerance"] == 1e-5
    assert plan.ANCHORS["executor_fidelity"]["r2_tolerance"] == 1e-5
    for item in plan.OWNED_PATHS:
        assert item.startswith("tfpd_exploration/")
    assert "tfpd_exploration/src/m2_kcurve_v1/plan.py" in plan.OWNED_PATHS
