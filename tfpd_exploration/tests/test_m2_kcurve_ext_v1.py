"""Synthetic no-data/no-CUDA tests for the M2 k-curve EXTENSION V1.

The review-critical properties:

1. the first-B candidate law: the sealed isfinite mask within the leading
   block, B=30 equal to the sealed k-curve first-30 pool, B=10 equal to the
   reblock10 first-10 mask, fail-closed outside [4, 30] and on < 4 usable;
2. the per-B D-opt prefix law: the support at k is the FIRST k of one frozen
   per-(session,B) greedy order, nested, the B=30 order IS the sealed
   k-curve law, k outside [4, usable_B] fails closed;
3. the B-column grid law: {4, min(5,usable_B), 8, min(10,usable_B),
   all-usable_B} resolved against the min-usable cap, capping disclosed,
   the endpoint IS the cap, cap < 4 fails closed;
4. the decline-slope law (endpoint mean minus k=4 mean);
5. the Q1 verdict rule (both branches, the tolerance boundary);
6. the matched-k law and the Q2 pairing roster;
7. the Q3 best-cell law (argmax with the deterministic tie-break);
8. the exact-CPU anchor matcher (pure-data exact, tolerance, prediction
   digest; like-for-like starts-field selection);
9. the cross-family support agreement matcher;
10. the plan binds the sealed foundations, the pre-registered grid, the
    verdict strings and the environment law.
"""

from __future__ import annotations

import json
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

from src.m2_kcurve_ext_v1 import laws, plan  # noqa: E402
from src.m2_kcurve_v1 import laws as kcurve_laws  # noqa: E402


def _angles(count: int = 30, *, seed: int = 5) -> np.ndarray:
    """The roster's alternating directional pattern: odd positions labeled."""
    rng = np.random.default_rng(seed)
    values = rng.uniform(-math.pi, math.pi, size=count)
    values[0::2] = np.nan
    return np.ascontiguousarray(values, dtype=np.float64)


def _cell_row(r2: float, *, starts: str = "starts", target: str = "target",
              prediction: str = "pred", windows: int = 10, **extra) -> dict:
    return {
        "r2": r2, "ordered_window_starts_sha256": starts,
        "query_starts_sha256": starts, "target_sha256": target,
        "prediction_sha256": prediction, "window_count": windows, **extra,
    }


def _sessions(seed: int = 2, names=("sesA", "sesB", "sesC")) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    return {name: float(rng.uniform(0.2, 0.4)) for name in names}


# ---------------------------------------------------------------------------
# 1. the first-B candidate law.
# ---------------------------------------------------------------------------


def test_blockB_candidates_at_every_B() -> None:
    theta = _angles(seed=11)
    assert laws.blockB_candidates(theta, 30).tolist() == list(range(1, 30, 2))
    assert laws.blockB_candidates(theta, 10).tolist() == [1, 3, 5, 7, 9]
    assert laws.blockB_candidates(theta, 20).tolist() == list(range(1, 20, 2))
    sparse = np.full(30, np.nan)
    sparse[[2, 4, 5, 9, 22, 28]] = [0.1, 0.9, -0.7, 2.2, -2.9, 1.1]
    assert laws.blockB_candidates(sparse, 10).tolist() == [2, 4, 5, 9]
    assert laws.blockB_candidates(sparse, 20).tolist() == [2, 4, 5, 9]
    assert laws.blockB_candidates(sparse, 30).tolist() == [2, 4, 5, 9, 22, 28]


def test_blockB_candidates_b30_is_the_sealed_first30_law() -> None:
    for seed in (3, 7, 21):
        theta = _angles(seed=seed)
        assert np.array_equal(laws.blockB_candidates(theta, 30),
                              kcurve_laws.first30_candidates(theta))


def test_blockB_candidates_fails_closed() -> None:
    theta = _angles(seed=13)
    with pytest.raises(laws.KCurveExtLawError):
        laws.blockB_candidates(theta, 3)  # B below the M4 floor
    with pytest.raises(laws.KCurveExtLawError):
        laws.blockB_candidates(theta, 31)  # B beyond the sealed first-30 pool
    with pytest.raises(laws.KCurveExtLawError):
        laws.blockB_candidates(np.asarray([0.1, 0.2, 0.3]), 10)  # no container
    broken = np.full(30, np.nan)
    broken[[0, 1, 2]] = [0.1, 0.2, 0.3]
    with pytest.raises(laws.KCurveExtLawError):
        laws.blockB_candidates(broken, 10)  # fewer than four directional rows


def test_usable_count_B_matches_candidates() -> None:
    theta = _angles(seed=17)
    assert laws.usable_count_B(theta, 10) == 5
    assert laws.usable_count_B(theta, 20) == 10
    assert laws.usable_count_B(theta, 30) == 15 == kcurve_laws.usable_count(theta)


# ---------------------------------------------------------------------------
# 2. the per-B D-opt prefix law.
# ---------------------------------------------------------------------------


def test_dopt_order_B_is_a_deterministic_permutation() -> None:
    theta = _angles(seed=19)
    for B in plan.B_AXIS:
        order = laws.dopt_greedy_order_B(theta, B)
        assert sorted(order.tolist()) == laws.blockB_candidates(theta, B).tolist()
        assert np.array_equal(order, laws.dopt_greedy_order_B(theta, B))
        assert int(order.max()) < B


def test_dopt_order_B30_is_the_sealed_kcurve_order() -> None:
    for seed in (3, 7, 21):
        theta = _angles(seed=seed)
        assert np.array_equal(laws.dopt_greedy_order_B(theta, 30),
                              kcurve_laws.dopt_greedy_order(theta))


def test_support_at_is_the_first_k_prefix_and_nested() -> None:
    theta = _angles(seed=23)
    for B in plan.B_AXIS:
        order = laws.dopt_greedy_order_B(theta, B)
        usable = int(order.size)
        ks = sorted({4, min(5, usable), 8 if 8 <= usable else usable, usable})
        supports = {k: laws.support_at(theta, k, B) for k in ks}
        for k in ks:
            assert np.array_equal(supports[k], np.sort(order[:k]))
            assert supports[k].size == k
            assert int(supports[k].max()) < B
        for a, b in zip(sorted(ks)[:-1], sorted(ks)[1:]):
            assert bool(np.isin(supports[a], supports[b]).all())


def test_support_at_fails_closed() -> None:
    theta = _angles(seed=29)
    with pytest.raises(laws.KCurveExtLawError):
        laws.support_at(theta, 3, 30)  # below the M4 floor
    with pytest.raises(laws.KCurveExtLawError):
        laws.support_at(theta, 6, 10)  # beyond usable within B=10
    with pytest.raises(laws.KCurveExtLawError):
        laws.support_at(theta, 16, 30)  # beyond usable within B=30


def test_prefix_payload_b30_identity_flags() -> None:
    theta = _angles(seed=31)
    payload = laws.prefix_payload(theta, 30, [4, 5, 8, 10, 15])
    assert payload["b30_order_is_sealed_kcurve_law"] is True
    assert payload["b30_usable_matches_sealed"] is True
    assert payload["prefix_exact"] and payload["prefix_nested"]
    assert payload["supports"]["15"] == laws.blockB_candidates(theta, 30).tolist()
    payload10 = laws.prefix_payload(theta, 10, [4, 5])
    assert payload10["usable_within_b"] == 5
    assert "b30_order_is_sealed_kcurve_law" not in payload10


# ---------------------------------------------------------------------------
# 3. the B-column grid law.
# ---------------------------------------------------------------------------


def test_requested_template_resolution() -> None:
    assert laws.requested_template(15) == [4, 5, 8, 10, 15]
    assert laws.requested_template(10) == [4, 5, 8, 10]
    assert laws.requested_template(5) == [4, 5]
    assert laws.requested_template(4) == [4]
    assert laws.requested_template(6) == [4, 5, 6]


def test_requested_template_fails_closed_below_the_floor() -> None:
    with pytest.raises(laws.KCurveExtLawError):
        laws.requested_template(3)


def test_b_grid_caps_at_min_usable_and_discloses() -> None:
    census = {"a|ses1": 5, "a|ses2": 6, "b|ses1": 5, "b|ses2": 5}
    grid = laws.b_grid(census, 10)
    assert grid["min_usable"] == 5
    assert grid["effective_grid"] == [4, 5]
    assert grid["endpoint_k"] == 5
    assert grid["dropped_by_cap"] == [8]
    assert grid["capped"] is True
    full = laws.b_grid({"a|ses1": 15, "a|ses2": 15}, 30)
    assert full["effective_grid"] == [4, 5, 8, 10, 15]
    assert full["capped"] is False
    assert full["dropped_by_cap"] == []


def test_b_grid_fails_closed_on_a_short_roster() -> None:
    with pytest.raises(laws.KCurveExtLawError):
        laws.b_grid({"a|ses1": 3}, 10)
    with pytest.raises(laws.KCurveExtLawError):
        laws.b_grid({}, 10)


# ---------------------------------------------------------------------------
# 4-5. the decline slope and the Q1 verdict rule.
# ---------------------------------------------------------------------------


def test_decline_slope_is_endpoint_minus_k4() -> None:
    per_k = {4: {"s": 0.30}, 15: {"s": 0.24}}
    assert laws.decline_slope(per_k, k_low=4, k_high=15) == pytest.approx(-0.06)
    rising = {4: {"s": 0.20}, 15: {"s": 0.25}}
    assert laws.decline_slope(rising, k_low=4, k_high=15) == pytest.approx(0.05)
    with pytest.raises(laws.KCurveExtLawError):
        laws.decline_slope({4: {"s": 0.3}}, k_low=4, k_high=15)


def test_q1_verdict_confirmed_branch() -> None:
    verdict = laws.q1_verdict(slope_fifo30=-0.054, slope_uncapped=-0.006,
                              slope_static=-0.007, tolerance=0.005)
    assert verdict["verdict"] == "CAPACITY_COMPETITION_CONFIRMED"
    assert verdict["confirmed"] is True
    assert verdict["threshold_static_minus_tolerance"] == pytest.approx(-0.012)
    assert verdict["decline_removed_fraction"] == pytest.approx(1.0)


def test_q1_verdict_dominant_branch() -> None:
    # the sealed-curve regime: the uncapped decline equals the capped decline
    verdict = laws.q1_verdict(slope_fifo30=-0.054, slope_uncapped=-0.054,
                              slope_static=-0.007, tolerance=0.005)
    assert verdict["verdict"] == "CARRIER_OVERFIT_DOMINANT"
    assert verdict["confirmed"] is False
    assert verdict["decline_removed_fraction"] == pytest.approx(0.0)


def test_q1_verdict_boundary_band() -> None:
    exactly_at = laws.q1_verdict(slope_fifo30=-0.054, slope_uncapped=-0.012,
                                 slope_static=-0.007, tolerance=0.005)
    assert exactly_at["verdict"] == "CAPACITY_COMPETITION_CONFIRMED"
    just_below = laws.q1_verdict(slope_fifo30=-0.054, slope_uncapped=-0.0120001,
                                 slope_static=-0.007, tolerance=0.005)
    assert just_below["verdict"] == "CARRIER_OVERFIT_DOMINANT"


def test_q1_verdict_degenerate_static_reference() -> None:
    verdict = laws.q1_verdict(slope_fifo30=0.0, slope_uncapped=0.0, slope_static=0.0)
    assert verdict["decline_removed_fraction"] is None
    assert verdict["verdict"] == "CAPACITY_COMPETITION_CONFIRMED"


# ---------------------------------------------------------------------------
# 6. the matched-k law (Q2 pairing).
# ---------------------------------------------------------------------------


def test_matched_ks_pairs_the_columns() -> None:
    assert laws.matched_ks([4, 5], [4, 5, 8, 10, 15]) == [4, 5]
    assert laws.matched_ks([4, 5, 8, 10], [4, 5, 8, 10, 15]) == [4, 5, 8, 10]
    assert laws.matched_ks([4], [4]) == [4]
    with pytest.raises(laws.KCurveExtLawError):
        laws.matched_ks([4, 5], [8, 10])


def test_law_paired_delta_reuses_the_sealed_law() -> None:
    candidate = {"ses1": 0.30, "ses2": 0.20}
    reference = {"ses1": 0.28, "ses2": 0.24}
    delta = laws.law_paired_delta(candidate, reference)
    assert delta["equal_session_mean_delta"] == pytest.approx(-0.01)
    assert delta["per_session_delta"] == {"ses1": pytest.approx(0.02),
                                          "ses2": pytest.approx(-0.04)}
    assert delta["positive_sessions"] == 1


# ---------------------------------------------------------------------------
# 7. the Q3 best-cell law.
# ---------------------------------------------------------------------------


def test_best_cell_picks_the_max() -> None:
    table = {"FIFO30|B30|K4": 0.299, "UNCAPPED|B30|K4": 0.301,
             "FIFO30|B10|K4": 0.28, "UNCAPPED|B20|K10": 0.25}
    best = laws.best_cell(table)
    assert best["law"] == "UNCAPPED" and best["B"] == 30 and best["k"] == 4
    assert best["value"] == pytest.approx(0.301)
    assert [item["key"] for item in best["ranking"]][0] == "UNCAPPED|B30|K4"
    assert len(best["ranking"]) == 4


def test_best_cell_tie_break_is_deterministic() -> None:
    table = {"UNCAPPED|B30|K4": 0.30, "FIFO30|B30|K4": 0.30,
             "FIFO30|B10|K4": 0.30, "FIFO30|B10|K5": 0.30}
    best = laws.best_cell(table)
    assert best["law"] == "FIFO30" and best["B"] == 10 and best["k"] == 4
    with pytest.raises(laws.KCurveExtLawError):
        laws.best_cell({"MYSTERY|B30|K4": 0.3})
    with pytest.raises(laws.KCurveExtLawError):
        laws.best_cell({})


    table2 = {"FIFO30|B30|K4": 0.30, "FIFO30|B10|K4": 0.31}
    assert laws.best_cell(table2)["B"] == 10


# ---------------------------------------------------------------------------
# 8. the exact-CPU anchor matcher.
# ---------------------------------------------------------------------------


def test_exact_cpu_anchor_requires_every_field() -> None:
    sealed = _cell_row(0.25)
    assert laws.exact_cpu_anchor(_cell_row(0.25), sealed)["exact_match"] is True
    drifted_r2 = laws.exact_cpu_anchor(_cell_row(0.25 + 1e-8), sealed)
    assert drifted_r2["exact_match"] is False
    assert drifted_r2["field_matches"]["r2_within_tolerance"] is False
    assert laws.exact_cpu_anchor(_cell_row(0.25, target="other"), sealed)[
        "exact_match"] is False
    assert laws.exact_cpu_anchor(_cell_row(0.25, prediction="other"), sealed)[
        "exact_match"] is False
    assert laws.exact_cpu_anchor(_cell_row(0.25, windows=11), sealed)[
        "exact_match"] is False


def test_exact_cpu_anchor_matches_on_the_common_starts_field() -> None:
    sealed_memory_scan_row = {
        "r2": 0.25, "query_starts_sha256": "abc", "target_sha256": "t",
        "prediction_sha256": "p", "window_count": 10,
    }
    cell = _cell_row(0.25, starts="abc", target="t", prediction="p")
    matched = laws.exact_cpu_anchor(cell, sealed_memory_scan_row)
    assert matched["starts_digest_field"] == "query_starts_sha256"
    assert matched["exact_match"] is True
    mismatched = laws.exact_cpu_anchor(_cell_row(0.25, starts="zzz"),
                                       sealed_memory_scan_row)
    assert mismatched["exact_match"] is False


# ---------------------------------------------------------------------------
# 9. the cross-family support agreement matcher.
# ---------------------------------------------------------------------------


def test_support_agreement_binds_the_two_spellings() -> None:
    left = {"greedy_order_positions": [3, 1, 5], "supports": {"4": [1, 3, 5, 7]},
            "usable_within_b": 5}
    right = {"greedy_order_positions": [3, 1, 5], "supports": {"4": [1, 3, 5, 7]},
             "usable_within_b": 5}
    assert laws.support_agreement(left, right)["agree"] is True
    right["supports"]["4"] = [1, 3, 5, 9]
    assert laws.support_agreement(left, right)["agree"] is False
    right2 = dict(right, greedy_order_positions=[1, 3, 5])
    assert laws.support_agreement(left, right2)["agree"] is False


# ---------------------------------------------------------------------------
# 10. the plan bindings.
# ---------------------------------------------------------------------------


def test_plan_binds_the_grid_and_laws() -> None:
    assert plan.B_AXIS == (10, 20, 30)
    assert plan.B_REFERENCE == 30
    assert plan.LAWS == ("FIFO30", "UNCAPPED")
    assert plan.K_GRID_TEMPLATE == ("4", "min(5,usable_B)", "8",
                                    "min(10,usable_B)", "all-usable_B")
    assert plan.SURFACES == ("within_post30", "external_post30_local")
    assert plan.VERDICTS == ("CAPACITY_COMPETITION_CONFIRMED",
                             "CARRIER_OVERFIT_DOMINANT")
    assert plan.READOUT_LAW["q1_law_contrast"]["tolerance"] == 0.005
    assert plan.ANCHORS["exact_cpu_matcher"]["r2_tolerance"] == 1.0e-9
    assert plan.ENVIRONMENT_LAW["cuda_visible_devices"] == ""
    assert plan.ENVIRONMENT_LAW["torch_num_threads"] == 4
    assert plan.ENVIRONMENT_LAW["dataloader_workers"] == 0


def test_plan_pins_the_sealed_foundations() -> None:
    assert plan.KCURVE_REPLAY_SHA256 == (
        "f3c1a55fb6207162dcfa4761a83fe2904e5b3d434b368168f47c88a13bff1e12")
    assert plan.MEMORY_SCAN_REPLAY_SHA256 == (
        "428be13ebbb38b8f85db36a65a6e4e4ac8f40896d11c187850974892fe20f284")
    assert plan.REBLOCK10_REPLAY_SHA256 == (
        "2ae7be22265203a888ed5e826254bce39873d725c3ae520483017c127bed2495")
    for relative in (plan.KCURVE_REPLAY_RELATIVE, plan.MEMORY_SCAN_REPLAY_RELATIVE,
                     plan.REBLOCK10_REPLAY_RELATIVE):
        assert (ROOT / relative).exists(), relative
    for relative in plan.OWNED_PATHS:
        assert (ROOT / relative).exists(), relative


def test_memory_law_constants_bind_the_sealed_scan() -> None:
    # the UNCAPPED law's provenance is the sealed scan, unchanged
    from src.m2_memory_law_scan_v1 import memory as scan_memory

    assert scan_memory.FrozenB3SUniformPool.family == "uniform_uncapped"
    assert plan.MEMORY_LAWS["UNCAPPED"]["origin"] == plan.MEMORY_SCAN_PACKAGE
    assert "FrozenB3SUniformPool" in plan.MEMORY_LAWS["UNCAPPED"]["law"]
    assert plan.MEMORY_LAWS["FIFO30"]["origin"] == plan.KCURVE_PACKAGE
    assert "cdm_k_cell" in plan.MEMORY_LAWS["FIFO30"]["law"]


def test_q1_rule_text_matches_the_law() -> None:
    text = plan.READOUT_LAW["q1_law_contrast"]["verdict_rule"]
    assert "CAPACITY_COMPETITION_CONFIRMED" in text
    assert "CARRIER_OVERFIT_DOMINANT" in text
    assert "slope_UNCAPPED >= slope_STATIC - 0.005" in text


# ---------------------------------------------------------------------------
# 11. the receipt-path integration tests (stop-loss directive 2026-09-02:
#     every aborted attempt died AFTER the grid scored, in summary
#     composition or in a receipt-spelling binding -- so the complete
#     post-materialization pipeline is exercised here on scratch dirs, no
#     data, no CUDA).
# ---------------------------------------------------------------------------


def _scratch_structures() -> dict[str, object]:
    """Synthetic but shape-accurate Phase-A..D products for the tail test."""
    sessions = {"within_post30": ["w1", "w2"],
                "external_post30_local": ["e1", "e2", "e3"]}
    grids = {10: {"effective_grid": [4, 5], "endpoint_k": 5},
             20: {"effective_grid": [4, 5, 8, 10], "endpoint_k": 10},
             30: {"effective_grid": [4, 5, 8, 10, 15], "endpoint_k": 15}}
    static_grid = {30: [4, 5, 8, 10, 15], 10: [4], 20: [4]}

    def r2v(surface: str, B: int, k: int, session: str, law: str) -> float:
        base = 0.30 if surface == "external_post30_local" else 0.65
        jitter = {"w1": 0.001, "w2": -0.001, "e1": 0.002,
                  "e2": -0.002, "e3": 0.0005}[session]
        return base + 0.001 * k - 0.01 * (B == 10) + 0.002 * (law == "UNCAPPED") + jitter

    cdm_rows = {law: {s: {sess: {
        f"CDM_{law}_B{B}_K{k}": {
            "r2": r2v(s, B, k, sess, law), "window_count": 10,
            "ordered_window_starts_sha256": "x", "target_sha256": "y",
            "prediction_sha256": f"{law}{B}{k}",
            "fifo_advance_events": 5 if s.startswith("ext") else 200}
        for B in grids for k in grids[B]["effective_grid"]}
        for sess in names} for s, names in sessions.items()} for law in plan.LAWS}
    static_rows = {s: {sess: {
        f"STATIC_B{B}_K{k}": {"r2": r2v(s, B, k, sess, "FIFO30")}
        for B in static_grid for k in static_grid[B]}
        for sess in names} for s, names in sessions.items()}
    law_equivalence = {f"{s}|{sess}|B{B}K{k}": {
        "capacity": 30 - k,
        "eviction_free": (5 <= 30 - k) if s.startswith("ext") else False,
        "identical": {"prediction_sha256": True, "r2_bitwise": True},
        "r2_delta": 0.0}
        for s, names in sessions.items() for sess in names
        for B in grids for k in grids[B]["effective_grid"]}
    capacity_census = {f"{s}|{sess}": {
        "advance_events": 5 if s.startswith("ext") else 200,
        "fifo_capacity_at_k": {}, "eviction_free_at_every_grid_k": s.startswith("ext")}
        for s, names in sessions.items() for sess in names}
    supports = {sp: {"surface|sess": {B: {"greedy_order_positions": [1],
                                          "supports": {}, "usable_within_b": 5}
                                      for B in grids}} for sp in plan.FAMILIES}
    return {
        "grids": grids, "static_grid": static_grid, "cdm_rows": cdm_rows,
        "static_rows": static_rows, "law_equivalence": law_equivalence,
        "capacity_census": capacity_census, "supports": supports,
        "support_proofs": {"k": {"ok": True, "b10_k4_support_matches_reblock10": True}},
        "cross_family": {"k": {"agree": True}},
        "fifo30_anchors": {"k": {"exact_match": True, "r2_delta": 0.0}},
        "static_anchors": {"k": {"exact_match": True, "r2_delta": 0.0}},
        "uncapped_anchors": {"k": {"exact_match": True, "r2_delta": 0.0}},
        "krow_parity": {"k": {"carrier_bitwise_equal": True,
                              "initial_activity_stack_bitwise_equal": True}},
        "pairing": {"k": {"exact_match": True}},
    }


def _run_tail(base: dict[str, object], scratch: Path) -> tuple[dict, dict]:
    """Execute the REAL tail source of physical.execute on synthetic inputs."""
    import textwrap
    import types

    from src.m2_kcurve_ext_v1 import physical

    source = Path(physical.__file__).read_text(encoding="utf-8")
    start = source.index(
        "    boolean_flags = [value for value in anchors_summary.values()")
    tail = source[start:]
    argnames = ["laws", "plan", "kcurve_laws", "np", "json", "_require",
                "_publish", "time", "resource", "torch", "batch_size",
                "attempt_digest", "root", "started", "grids", "static_grid",
                "cdm_rows", "static_rows", "support_proofs", "cross_family",
                "fifo30_anchors", "static_anchors", "uncapped_anchors",
                "krow_parity", "pairing", "law_equivalence", "capacity_census",
                "supports", "anchors_summary"]

    published: dict[str, dict] = {}

    def publish(path: Path, payload: dict) -> str:
        published[str(path)] = json.loads(json.dumps(payload, allow_nan=False))
        return "digest_%d" % len(published)

    class FakeTime:
        def monotonic(self) -> float:
            return 100.0

    class FakeResource:
        RUSAGE_SELF = 0

        def getrusage(self, _which: int) -> types.SimpleNamespace:
            return types.SimpleNamespace(ru_maxrss=1)

    fake_torch = types.SimpleNamespace(__version__="test",
                                       get_num_threads=lambda: 4)

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)

    anchors_summary = {
        "b30_support_identity_all_exact": True,
        "b10_k4_vs_reblock10_all_exact": True,
        "cross_family_support_agreement": True,
        "fifo30_b30_all_exact": True,
        "static_b30_all_exact": True,
        "uncapped_b30_k4_all_exact": True,
        "krow_binding_parity_all": True,
        "pure_data_pairing_all_exact": True,
        "external_law_equivalence_holds": True,
        "fifo30_max_abs_r2_delta": 0.0,
        "static_max_abs_r2_delta": 0.0,
        "uncapped_max_abs_r2_delta": 0.0,
    }
    kwargs = dict(
        laws=laws, plan=plan, kcurve_laws=kcurve_laws, np=np, json=json,
        _require=require, _publish=publish, time=FakeTime(),
        resource=FakeResource(), torch=fake_torch, batch_size=1024,
        attempt_digest="ad", root=scratch, started=0.0,
        anchors_summary=anchors_summary, **base)
    namespace: dict[str, object] = {}
    wrapper = "def _execute_tail(" + ", ".join(argnames) + "):\n" + tail
    exec(compile(wrapper, "tail_under_test.py", "exec"), namespace)  # noqa: S102
    result = namespace["_execute_tail"](**kwargs)
    return published, result


def test_tail_composition_publishes_both_receipts(tmp_path: Path) -> None:
    published, result = _run_tail(_scratch_structures(), tmp_path)
    assert sorted(published) == [
        str(tmp_path / "replay.json"), str(tmp_path / "terminal.json")]
    terminal = published[str(tmp_path / "terminal.json")]
    replay = published[str(tmp_path / "replay.json")]
    assert terminal["status"] == "TERMINAL"
    assert replay["status"] == "REPLAY_COMPLETE"
    assert terminal["replay_sha256"] == "digest_1"  # the stub's first publish
    assert replay["attempt_sha256"] == terminal["attempt_sha256"] == "ad"
    assert result["status"] == "TERMINAL"
    assert result["stop_conditions_fired"] == []
    # the verdict, matched-k roster and best-cell law all composed
    assert terminal["q1"]["verdict"] in plan.VERDICTS
    assert sorted(terminal["q2"]["mean_deltas"]) == [
        "external_post30_local|B10_minus_B30|k4",
        "external_post30_local|B10_minus_B30|k5",
        "external_post30_local|B20_minus_B30|k10",
        "external_post30_local|B20_minus_B30|k4",
        "external_post30_local|B20_minus_B30|k5",
        "external_post30_local|B20_minus_B30|k8",
        "within_post30|B10_minus_B30|k4",
        "within_post30|B10_minus_B30|k5",
        "within_post30|B20_minus_B30|k10",
        "within_post30|B20_minus_B30|k4",
        "within_post30|B20_minus_B30|k5",
        "within_post30|B20_minus_B30|k8"]
    assert terminal["q3"]["best_external_cell"]["law"] in plan.LAWS
    assert len(terminal["q3"]["ranking"]) == 2 * (2 + 4 + 5)
    # the mean tables cover every measured cell and nothing else
    assert sorted(terminal["equal_session_means"]["cdm"]["FIFO30"][
        "external_post30_local"]) == [
        "B10K4", "B10K5", "B20K10", "B20K4", "B20K5", "B20K8",
        "B30K10", "B30K15", "B30K4", "B30K5", "B30K8"]
    assert sorted(terminal["equal_session_means"]["static"][
        "within_post30"]) == [
        "B10K4", "B20K4", "B30K10", "B30K15", "B30K4", "B30K5", "B30K8"]
    # json round-trip with allow_nan=False already succeeded inside publish


def test_tail_fail_closed_on_a_failed_anchor(tmp_path: Path) -> None:
    base = _scratch_structures()
    published, _result = _run_tail(base, tmp_path)
    assert published  # the all-pass path publishes
    scratch2 = tmp_path / "fail"
    # a single failed anchor flag must stop publication (fail-closed law)
    import pytest as _pytest

    class FailingPublish:
        def __call__(self, path: Path, payload: dict) -> str:
            raise AssertionError("publication must not happen on a failed anchor")

    source_base = dict(base)
    # emulate the failed-anchor stop condition by rerunning the tail with a
    # poisoned anchors summary; the tail must raise before any publish
    from src.m2_kcurve_ext_v1 import physical

    source = Path(physical.__file__).read_text(encoding="utf-8")
    start = source.index(
        "    boolean_flags = [value for value in anchors_summary.values()")
    tail = source[start:]
    args = ["laws", "plan", "kcurve_laws", "np", "json", "_require",
            "_publish", "time", "resource", "torch", "batch_size",
            "attempt_digest", "root", "started", "grids", "static_grid",
            "cdm_rows", "static_rows", "support_proofs", "cross_family",
            "fifo30_anchors", "static_anchors", "uncapped_anchors",
            "krow_parity", "pairing", "law_equivalence", "capacity_census",
            "supports", "anchors_summary"]
    bad_summary = {
        "b30_support_identity_all_exact": True,
        "b10_k4_vs_reblock10_all_exact": False,  # the attempt-4 failure mode
        "cross_family_support_agreement": True,
        "fifo30_b30_all_exact": True,
        "static_b30_all_exact": True,
        "uncapped_b30_k4_all_exact": True,
        "krow_binding_parity_all": True,
        "pure_data_pairing_all_exact": True,
        "external_law_equivalence_holds": True,
        "fifo30_max_abs_r2_delta": 0.0,
        "static_max_abs_r2_delta": 0.0,
        "uncapped_max_abs_r2_delta": 0.0,
    }
    import types as _types

    class FakeTime:
        def monotonic(self) -> float:
            return 100.0

    class FakeResource:
        RUSAGE_SELF = 0

        def getrusage(self, _which: int) -> _types.SimpleNamespace:
            return _types.SimpleNamespace(ru_maxrss=1)

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)

    kwargs = dict(
        laws=laws, plan=plan, kcurve_laws=kcurve_laws, np=np, json=json,
        _require=require,
        _publish=lambda path, payload: (_ for _ in ()).throw(
            AssertionError("publication must not happen on a failed anchor")),
        time=FakeTime(), resource=FakeResource(),
        torch=_types.SimpleNamespace(__version__="t", get_num_threads=lambda: 4),
        batch_size=1024, attempt_digest="ad", root=scratch2, started=0.0,
        anchors_summary=bad_summary, **source_base)
    namespace: dict[str, object] = {}
    wrapper = "def _execute_tail(" + ", ".join(args) + "):\n" + tail
    exec(compile(wrapper, "tail_fail.py", "exec"), namespace)  # noqa: S102
    with _pytest.raises(AssertionError, match="stop conditions fired"):
        namespace["_execute_tail"](**kwargs)


def test_publish_receipt_law_on_scratch(tmp_path: Path) -> None:
    import hashlib

    from src.m2_kcurve_ext_v1 import physical

    target = tmp_path / "scratch.json"
    digest = physical._publish(target, {"a": 1, "b": [True, None]})
    body = target.read_bytes()
    assert hashlib.sha256(body).hexdigest() == digest
    assert (target.stat().st_mode & 0o777) == 0o444
    sidecar = tmp_path / "scratch.json.sha256"
    assert sidecar.read_text(encoding="ascii").strip() == f"{digest}  scratch.json"
    assert (sidecar.stat().st_mode & 0o777) == 0o444
    # atomicity: a second publish to the same path must fail (O_EXCL)
    with pytest.raises(OSError):
        physical._publish(target, {"a": 2})


def test_reblock10_surface_spellings_resolve_for_every_session() -> None:
    """The attempt-4 failure mode: the sealed reblock10 receipt spells the
    static family's external surface ``external_post10_query`` while the CDM
    family spells ``external_post10_local``; the loader must resolve BOTH."""
    import hashlib

    from src.m2_kcurve_ext_v1 import physical

    path = ROOT / plan.REBLOCK10_REPLAY_RELATIVE
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == plan.REBLOCK10_REPLAY_SHA256, "the sealed reblock10 receipt drifted"
    payload = json.loads(path.read_text(encoding="utf-8"))
    loader = physical._load_sealed_reblock10(ROOT)
    for spelling, surface_map in loader["surface_maps"].items():
        for surface in plan.SURFACES:
            reblock_surface = surface_map[surface]
            sessions = [key.split("|", 1)[1] for key in
                        payload["supports"][spelling]
                        if key.startswith(reblock_surface + "|")]
            expected = (plan.EXPECTED_WITHIN_SESSIONS if surface == "within_post30"
                        else plan.EXPECTED_EXTERNAL_SESSIONS)
            assert len(sessions) == expected, (
                f"{spelling}/{reblock_surface}: session coverage drift")
            for session in sessions:
                entry = loader["supports"].get((spelling, reblock_surface, session))
                assert entry is not None and "k4_support_positions" in entry, (
                    f"missing B10 anchor at {spelling}/{surface}/{session}")
