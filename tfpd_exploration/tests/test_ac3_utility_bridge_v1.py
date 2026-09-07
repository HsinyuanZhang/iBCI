"""Synthetic-fixture tests for the AC3-U M4-only utility bridge (V1).

No data root, no checkpoint, no CUDA: every test runs on constructed numpy
blocks, the real imported frozen ``core`` evidence objects, and the real frozen
``policy.build_construction_predictions``.  The review-critical properties:

1. the §4 rotation law -- norm preservation, direction alignment including
   wrap-around, the predeclared undefined-theta fallback, validity-evidence
   identity and dtype cast-back;
2. seam purity -- frozen construction names reach ONLY the original function
   and return its own object unchanged; rotating names pre-rotate and delegate
   ``raw``; install/restore leaves the frozen module byte-identical in
   behaviour;
3. the deterministic cursor mapping -- wrapper call n maps to bound trial n,
   the call count is asserted, and the predeclared drift guard separates a
   correct mapping from a wrong one;
4. §8 gate boundary semantics -- a value exactly at +0.01 passes, a value
   1e-13 below fails, and the program epsilon surfaces as a disclosed band;
5. receipt schema sanity and the digest-verification helpers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT / "tfpd_exploration", ROOT / "tfpd_exploration" / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.causal_dual_memory_cell_d_v1 import core as cdm_core
from src.learned_gate_p2prime_v1 import policy as p2policy

from src.ac3_utility_bridge_v1 import directions, gates, plan, rotation, wrapper


# ---------------------------------------------------------------------------
# Shared synthetic fixture: real frozen evidence objects, constructed blocks.
# ---------------------------------------------------------------------------


def _validity(bins, trial_id="synthetic-trial", session_id="synthetic-session"):
    return cdm_core.VelocityValidityEvidence(
        valid_mask=np.ones(bins, dtype=bool),
        session_id=session_id,
        trial_id=trial_id,
        prediction_interval_start_bin=0,
        prediction_interval_stop_bin=bins,
    )


def _views(rng, bins, direction, spread=0.35, dtype=np.float64):
    """Four group views whose net displacements all point near ``direction``."""
    views = []
    for group in range(4):
        bias = rng.normal(0.0, spread)
        angle = direction + bias
        speed = rng.uniform(4.0, 9.0)
        rows = np.stack([
            speed * math.cos(angle) + rng.normal(0.0, 0.2, size=bins),
            speed * math.sin(angle) + rng.normal(0.0, 0.2, size=bins),
        ], axis=1)
        views.append(cdm_core.CompletedVelocityPrediction(
            np.asarray(rows, dtype=dtype), _validity(bins),
        ))
    return views


def _frozen_cache(views_blocks):
    """The materialize CSR layout: [P, 2, 4] plus per-trial starts/counts."""
    chunks = [np.stack([np.asarray(item.velocity, dtype=np.float64) for item in block], axis=2)
              for block in views_blocks]
    counts = [chunk.shape[0] for chunk in chunks]
    starts, offset = [], 0
    for count in counts:
        starts.append(offset)
        offset += count
    return {
        "velocity_flat": np.concatenate(chunks, axis=0),
        "row_starts": np.asarray(starts, dtype=np.int64),
        "row_counts": np.asarray(counts, dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# 1. The rotation law.
# ---------------------------------------------------------------------------


def test_rotation_preserves_every_norm_and_the_mask():
    rng = np.random.default_rng(3)
    views = _views(rng, bins=17, direction=0.7)
    theta = 2.9
    rotated, record = rotation.rotate_group_predictions(views, theta, row_id="UGE")
    assert record["applied"] is True
    assert len(rotated) == 4
    report = rotation.norm_preservation_report(
        [item.velocity for item in views], [item.velocity for item in rotated],
    )
    assert report["max_relative_speed_gap"] <= 1.0e-12
    assert report["max_relative_displacement_norm_gap"] <= 1.0e-12
    for left, right in zip(views, rotated):
        assert left.velocity.shape == right.velocity.shape
        assert np.array_equal(left.validity.valid_mask, right.validity.valid_mask)
        assert int(left.validity.valid_mask.sum()) == int(right.validity.valid_mask.sum())


def test_rotation_aligns_the_net_displacement_with_theta_including_wraparound():
    rng = np.random.default_rng(5)
    for direction in (0.0, 0.7, -2.9, 3.14159, -3.14159, 2.0):
        views = _views(rng, bins=11, direction=direction)
        # A target angle on the far side of the wrap boundary.
        target = rotation.wrap_rad(direction + math.pi - 0.05)
        rotated, record = rotation.rotate_group_predictions(views, target, row_id="UGE")
        assert record["applied"] is True
        for item in rotated:
            vector = rotation.net_displacement(item.velocity)
            got = math.atan2(float(vector[1]), float(vector[0]))
            assert abs(rotation.wrap_rad(got - target)) < 1.0e-9


def test_rotation_passes_through_undefined_theta_without_touching_the_views():
    rng = np.random.default_rng(7)
    views = _views(rng, bins=9, direction=0.2)
    for undefined in (float("nan"), None):
        rotated, record = rotation.rotate_group_predictions(views, undefined, row_id="UGE")
        assert record["applied"] is False
        assert record["reason"] == "undefined_theta_hat_predeclared_fallback"
        assert tuple(rotated) == tuple(views)
        for left, right in zip(views, rotated):
            assert left is right


def test_rotation_reuses_the_same_validity_evidence_object():
    rng = np.random.default_rng(11)
    views = _views(rng, bins=13, direction=-1.1)
    rotated, _record = rotation.rotate_group_predictions(views, 0.4, row_id="UGE")
    for left, right in zip(views, rotated):
        assert right.validity is left.validity


def test_rotation_casts_back_to_the_incoming_dtype():
    rng = np.random.default_rng(13)
    views = _views(rng, bins=8, direction=0.9, dtype=np.float64)
    theta = -2.0
    rotated, record = rotation.rotate_group_predictions(views, theta, row_id="UGE")
    for item in rotated:
        assert item.velocity.dtype == np.float64
    rotated32 = rotation.rotate_trajectory(views[0].velocity.astype(np.float32), 1.1)
    assert rotated32.dtype == np.float32
    assert record["norm_report"]["max_relative_speed_gap"] <= plan.NORM_PRESERVATION_TOLERANCE


def test_rotation_keeps_the_frozen_acceptance_pattern_on_magnitude_gates():
    rng = np.random.default_rng(17)
    config = cdm_core.CDMDConfig(
        support_budget_m=4, minimum_movement_bins=1, minimum_displacement=1.0e-9,
        minimum_mean_speed=1.0e-9, max_canonical_distance_rad=math.pi / 4.0,
    )
    views = _views(rng, bins=12, direction=0.3)
    rotated, _record = rotation.rotate_group_predictions(views, 1.0, row_id="UGE")
    for left, right in zip(views, rotated):
        before = cdm_core.pseudo_direction_from_velocity(
            left.velocity, left.validity.valid_mask, config=config,
        )
        after = cdm_core.pseudo_direction_from_velocity(
            right.velocity, right.validity.valid_mask, config=config,
        )
        assert before.accepted == after.accepted
        assert before.movement_bins == after.movement_bins
        assert abs(before.displacement_norm - after.displacement_norm) < 1.0e-9
        assert abs(before.mean_speed - after.mean_speed) < 1.0e-9


def test_wrap_rad_range_and_rotation_delta():
    assert rotation.wrap_rad(math.pi) == -math.pi
    assert rotation.wrap_rad(0.0) == 0.0
    for value in (-4.0 * math.pi, -2.0 * math.pi, 2.0 * math.pi, 4.0 * math.pi):
        assert abs(rotation.wrap_rad(value)) < 1.0e-15
    for value in (-3.0, -0.4, 0.4, 3.0):
        assert -math.pi <= rotation.wrap_rad(value) < math.pi
    delta = rotation.rotation_delta(0.25, np.asarray([-1.0, -1.0]))
    assert abs(delta - rotation.wrap_rad(0.25 - math.atan2(-1.0, -1.0))) < 1.0e-15
    with pytest.raises(rotation.AC3URotationError):
        rotation.rotation_delta(float("nan"), np.asarray([1.0, 0.0]))


# ---------------------------------------------------------------------------
# 2. Seam purity.
# ---------------------------------------------------------------------------


class _RecordingOriginal:
    """Stand-in frozen function: records calls, returns a sentinel object."""

    def __init__(self):
        self.calls = []
        self.sentinel = ("sentinel-views", {"construction": "recorded"})

    def __call__(self, *, group_predictions, construction, behavior_rows=None,
                 behavior_mean=(), behavior_std=(), alpha=0.25):
        self.calls.append({
            "group_predictions": tuple(group_predictions),
            "construction": construction,
            "behavior_rows": behavior_rows,
            "behavior_mean": tuple(behavior_mean),
            "behavior_std": tuple(behavior_std),
            "alpha": alpha,
        })
        return self.sentinel


class _PolicyDouble:
    def __init__(self, original):
        self.build_construction_predictions = original


def test_frozen_constructions_hit_only_the_original_and_return_its_object():
    original = _RecordingOriginal()
    seam = wrapper.ConstructionSeam(original=original, policy_module=_PolicyDouble(original))
    seam.install()
    try:
        for construction in plan.FROZEN_CONSTRUCTIONS:
            views = ("view-a", "view-b", "view-c", "view-d")
            result = seam(
                group_predictions=views, construction=construction,
                behavior_rows=None, behavior_mean=(0.0, 0.0), behavior_std=(1.0, 1.0),
            )
            assert result is original.sentinel
            assert original.calls[-1]["construction"] == construction
            assert original.calls[-1]["group_predictions"] == views
            assert original.calls[-1]["behavior_rows"] is None
    finally:
        seam.restore()
    assert len(original.calls) == len(plan.FROZEN_CONSTRUCTIONS)
    assert seam.passthrough_calls == len(plan.FROZEN_CONSTRUCTIONS)
    assert seam.rotated_calls == 0


def test_unknown_construction_is_rejected_by_the_seam():
    original = _RecordingOriginal()
    module = _PolicyDouble(original)
    seam = wrapper.ConstructionSeam(original=original, policy_module=module)
    seam.install()
    try:
        with pytest.raises(wrapper.AC3USeamError):
            seam(group_predictions=(1, 2, 3, 4), construction="not_a_construction")
    finally:
        seam.restore()


def test_rotating_construction_pre_rotates_and_delegates_raw():
    rng = np.random.default_rng(19)
    views = _views(rng, bins=10, direction=0.4)
    original = _RecordingOriginal()
    module = _PolicyDouble(original)
    seam = wrapper.ConstructionSeam(original=original, policy_module=module)
    seam.install()
    cache = _frozen_cache([views])
    try:
        seam.begin_session(wrapper.SessionBinding(
            row_id="UGE", construction="group_ensemble", session="synthetic-session",
            trial_ids=["synthetic-session:trial:0"], theta=np.asarray([1.2]),
            frozen_velocity=cache["velocity_flat"],
            row_starts=cache["row_starts"], row_counts=cache["row_counts"],
        ))
        seam(group_predictions=views, construction="group_ensemble")
        record = seam.end_session()
        assert record["rotated_calls"] == 1
        assert record["groups_rotated"] == 4
        assert record["undefined_theta_fallback_trials"] == 0
        assert record["validity_objects_reused"] == 4
        assert record["validity_objects_rebuilt"] == 0
        assert len(original.calls) == 1
        delegated = original.calls[0]
        assert delegated["construction"] == "raw"
        for incoming, start in zip(delegated["group_predictions"], views):
            assert incoming is not start
            vector = rotation.net_displacement(incoming.velocity)
            assert abs(rotation.wrap_rad(
                math.atan2(float(vector[1]), float(vector[0])) - 1.2)) < 1.0e-9
            assert incoming.validity is start.validity
    finally:
        seam.restore()


def test_seam_install_and_restore_leave_the_frozen_module_untouched():
    seam = wrapper.build_seam()
    original = seam.original
    assert original is p2policy.build_construction_predictions
    seam.install()
    try:
        assert p2policy.build_construction_predictions is seam
    finally:
        seam.restore()
    assert p2policy.build_construction_predictions is original


def test_real_frozen_raw_construction_is_bitwise_identical_through_the_seam():
    rng = np.random.default_rng(23)
    views = _views(rng, bins=9, direction=-0.6)
    module = _PolicyDouble(p2policy.build_construction_predictions)
    seam = wrapper.ConstructionSeam(
        original=p2policy.build_construction_predictions, policy_module=module,
    )
    seam.install()
    try:
        through_seam, meta_seam = seam(group_predictions=views, construction="raw")
    finally:
        seam.restore()
    direct, meta_direct = p2policy.build_construction_predictions(
        group_predictions=views, construction="raw",
    )
    assert meta_seam == meta_direct
    for left, right in zip(through_seam, direct):
        assert np.array_equal(left.velocity, right.velocity)
        assert left.validity is right.validity
    assert through_seam[0].velocity.dtype == direct[0].velocity.dtype


# ---------------------------------------------------------------------------
# 3. The deterministic cursor mapping and the drift guard.
# ---------------------------------------------------------------------------


def _bound_session(seam, blocks, theta, row_id="UGE", construction="group_ensemble",
                   session="synthetic-session", permuted=False):
    cache = _frozen_cache(blocks)
    trial_ids = [f"{session}:trial:{index}" for index in range(len(blocks))]
    order = list(range(len(blocks)))
    if permuted:
        order = order[1:] + order[:1]
    seam.begin_session(wrapper.SessionBinding(
        row_id=row_id, construction=construction, session=session,
        trial_ids=[trial_ids[index] for index in order],
        theta=np.asarray([theta[index] for index in order]),
        frozen_velocity=cache["velocity_flat"],
        row_starts=cache["row_starts"][order], row_counts=cache["row_counts"][order],
    ))
    return cache


def test_cursor_maps_call_n_to_bound_trial_n_in_order():
    rng = np.random.default_rng(29)
    blocks = [_views(rng, bins=7, direction=0.3 + 0.2 * index) for index in range(4)]
    theta = [0.4, -1.2, 2.0, 2.8]
    original = _RecordingOriginal()
    seam = wrapper.ConstructionSeam(original=original, policy_module=_PolicyDouble(original))
    seam.install()
    try:
        _bound_session(seam, blocks, theta)
        for index, block in enumerate(blocks):
            seam(group_predictions=block, construction="group_ensemble")
            delegated = original.calls[-1]
            vector = rotation.net_displacement(delegated["group_predictions"][0].velocity)
            assert abs(rotation.wrap_rad(
                math.atan2(float(vector[1]), float(vector[0])) - theta[index])) < 1.0e-9
        record = seam.end_session()
        assert record["wrapper_calls"] == 4 == record["expected_trials"]
        import hashlib

        assert record["trial_id_sequence_sha256"] == hashlib.sha256(
            json.dumps([f"synthetic-session:trial:{i}" for i in range(4)],
                       separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        assert record["frozen_cache_drift"]["guard_passed"] is True
    finally:
        seam.restore()


def test_end_session_rejects_a_call_count_that_disagrees_with_the_roster():
    rng = np.random.default_rng(31)
    blocks = [_views(rng, bins=6, direction=0.5) for _ in range(3)]
    original = _RecordingOriginal()
    seam = wrapper.ConstructionSeam(original=original, policy_module=_PolicyDouble(original))
    seam.install()
    try:
        _bound_session(seam, blocks, [0.1, 0.2, 0.3])
        seam(group_predictions=blocks[0], construction="group_ensemble")
        with pytest.raises(wrapper.AC3USeamError):
            seam.end_session()
    finally:
        seam.restore()


def test_drift_guard_fires_on_a_wrong_mapping_and_only_then():
    rng = np.random.default_rng(37)
    angles = (0.0, 1.2, 2.4, -2.6, -1.3, 3.0)
    blocks = [_views(rng, bins=8, direction=float(angle)) for angle in angles]
    theta = [float(rotation.wrap_rad(angle + 0.05)) for angle in angles]
    original = _RecordingOriginal()
    module = _PolicyDouble(original)
    seam = wrapper.ConstructionSeam(original=original, policy_module=module)
    seam.install()
    try:
        _bound_session(seam, blocks, theta)
        for block in blocks:
            seam(group_predictions=block, construction="group_ensemble")
        record = seam.end_session()
        assert record["frozen_cache_drift"]["guard_passed"] is True
        assert record["frozen_cache_drift"]["median_rad"] < 0.2

        # A one-step rotation of the bound roster is exactly the wrong-mapping
        # failure mode the guard exists to catch: unrelated trials get paired and
        # the median gap rises toward the uniform value pi/2.
        _bound_session(seam, blocks, theta, permuted=True)
        for block in blocks:
            seam(group_predictions=block, construction="group_ensemble")
        with pytest.raises(wrapper.AC3USeamError) as caught:
            seam.end_session()
        assert "drift guard fired" in str(caught.value)
    finally:
        seam.restore()


def test_drift_guard_threshold_is_well_below_the_uniform_median():
    uniform = np.random.default_rng(53).uniform(-math.pi, math.pi, size=200_000)
    median = float(np.median(np.abs(uniform)))
    assert median == pytest.approx(math.pi / 2, abs=1.0e-2)
    assert plan.DRIFT_GUARD["session_median_max_rad"] < median - 0.5


def test_session_binding_rejects_shape_and_roster_drift():
    rng = np.random.default_rng(41)
    blocks = [_views(rng, bins=5, direction=0.2)]
    cache = _frozen_cache(blocks)
    with pytest.raises(wrapper.AC3USeamError):
        wrapper.SessionBinding(
            row_id="UGE", construction="group_ensemble", session="s", trial_ids=["s:trial:0"],
            theta=np.asarray([0.1, 0.2]), frozen_velocity=cache["velocity_flat"],
            row_starts=cache["row_starts"], row_counts=cache["row_counts"],
        )
    with pytest.raises(wrapper.AC3USeamError):
        wrapper.SessionBinding(
            row_id="UGE", construction="group_ensemble", session="s",
            trial_ids=["s:trial:0", "s:trial:1"],
            theta=np.asarray([0.1, 0.2]), frozen_velocity=cache["velocity_flat"],
            row_starts=cache["row_starts"], row_counts=cache["row_counts"],
        )
    with pytest.raises(wrapper.AC3USeamError):
        wrapper.SessionBinding(
            row_id="UGE", construction="group_ensemble", session="s", trial_ids=["s:trial:0"],
            theta=np.asarray([0.1]), frozen_velocity=cache["velocity_flat"][:, :1, :],
            row_starts=cache["row_starts"], row_counts=cache["row_counts"],
        )


# ---------------------------------------------------------------------------
# 4. Gate boundary semantics and dispositions.
# ---------------------------------------------------------------------------


def test_margin_boundary_is_exact_greater_or_equal():
    assert gates.margin_verdict(0.01)["meets_margin"] is True
    assert gates.margin_verdict(0.01 - 1.0e-13)["meets_margin"] is False
    assert gates.margin_verdict(0.01 + 1.0e-13)["meets_margin"] is True
    assert gates.margin_verdict(0.0)["meets_margin"] is False
    assert gates.margin_verdict(-0.01)["meets_margin"] is False


def test_margin_epsilon_is_a_disclosed_band_not_a_widening():
    just_inside = gates.margin_verdict(0.01 - 1.0e-13)
    assert just_inside["meets_margin"] is False
    assert just_inside["within_epsilon_band_of_boundary"] is True
    clearly_below = gates.margin_verdict(0.009)
    assert clearly_below["meets_margin"] is False
    assert clearly_below["within_epsilon_band_of_boundary"] is False
    at_margin = gates.margin_verdict(0.01)
    assert at_margin["meets_margin"] is True
    assert at_margin["within_epsilon_band_of_boundary"] is False
    assert just_inside["boundary_epsilon"] == plan.GATE_BOUNDARY_EPSILON == 1.0e-12


def _governing(u0, uge, u2):
    sessions = list(plan.SESSIONS)
    return gates.evaluate_gates(
        governing_r2={
            "U0": dict(zip(sessions, u0)),
            "UGE": dict(zip(sessions, uge)),
            "U2": dict(zip(sessions, u2)),
        },
        sessions=sessions,
    )


def test_gate_advances_only_on_both_conditions():
    u0 = [0.40] * 6
    uge = [0.40 + 0.02] * 6
    result = _governing(u0, uge, uge)
    assert result["UGE_gate"]["passed"] is True
    assert result["disposition"] == plan.DISPOSITION_ADVANCE
    assert result["equal_session_mean_r2"]["UGE"] == pytest.approx(0.42)


def test_gate_fails_when_breadth_is_three_of_six_even_with_a_large_mean():
    u0 = [0.40] * 6
    uge = [0.40, 0.40, 0.40, 0.40 + 0.06, 0.40 + 0.06, 0.40 + 0.06]
    result = _governing(u0, uge, uge)
    assert result["UGE_gate"]["breadth_passed"] is False
    assert result["UGE_gate"]["passed"] is False
    assert result["disposition"] == plan.DISPOSITION_NULL


def test_gate_boundary_value_passes_and_just_below_fails():
    u0 = [0.40] * 6
    at_margin = _governing(u0, [0.41] * 6, [0.41] * 6)
    assert at_margin["UGE_gate"]["passed"] is True
    just_below = _governing(u0, [0.41 - 1.0e-13] * 6, [0.41 - 1.0e-13] * 6)
    assert just_below["UGE_gate"]["passed"] is False
    assert just_below["disposition"] == plan.DISPOSITION_NULL


def test_u2_rescue_requires_the_margin_and_the_breadth_over_u0():
    u0 = [0.40] * 6
    uge = [0.40] * 6
    u2 = [0.40 + 0.02] * 6
    result = _governing(u0, uge, u2)
    assert result["UGE_gate"]["passed"] is False
    assert result["U2_rescue"]["passed"] is True
    assert result["disposition"] == plan.DISPOSITION_ADVANCE_U2_RESCUE
    # A large U2 - UGE margin alone cannot rescue a failed breadth over U0.
    u2_narrow = [0.40, 0.40, 0.40, 0.40 + 0.06, 0.40 + 0.06, 0.40 + 0.06]
    rescue = _governing(u0, uge, u2_narrow)
    assert rescue["U2_rescue"]["over_UGE_margin"]["meets_margin"] is True
    assert rescue["U2_rescue"]["passed"] is False
    assert rescue["disposition"] == plan.DISPOSITION_NULL


def test_null_disposition_and_the_five_session_sensitivity_block():
    u0 = [0.40] * 6
    uge = [0.40] * 6
    u2 = [0.40] * 6
    result = _governing(u0, uge, u2)
    assert result["disposition"] == plan.DISPOSITION_NULL
    sensitivity = result["five_session_sensitivity_non_governing"]
    assert sensitivity["non_governing"] is True
    assert sensitivity["excluded_session"] == plan.HIGH_ERROR_SESSION
    assert plan.HIGH_ERROR_SESSION not in sensitivity["sessions"]
    assert len(sensitivity["sessions"]) == 5
    assert result["high_error_session_included_in_governing"] is True
    assert result["no_m10_m30_claim"]


def test_paired_deltas_require_matching_rosters():
    left = {"a": 0.5, "b": 0.4}
    right = {"a": 0.3, "b": 0.5}
    stats = gates.paired_deltas(left, right)
    assert stats["per_session"] == {"a": pytest.approx(0.2), "b": pytest.approx(-0.1)}
    assert stats["positive_sessions"] == 1
    assert stats["equal_session_mean_delta"] == pytest.approx(0.05)
    # The mean difference is the difference of the two row means (the frozen
    # policy._mean_delta arithmetic), not the mean of the per-session deltas.
    assert stats["equal_session_mean_delta"] == gates.equal_session_mean([0.5, 0.4]) - gates.equal_session_mean(
        [0.3, 0.5]
    )
    with pytest.raises(gates.AC3UGateError):
        gates.paired_deltas(left, {"a": 0.3})


# ---------------------------------------------------------------------------
# 5. Receipt schema sanity and the digest-verification helpers.
# ---------------------------------------------------------------------------


def test_pre_registration_payload_is_pinned():
    payload = plan.gate_spec_payload()
    assert payload["row_order"] == ["U0", "UGE", "U2"]
    assert payload["rotating_constructions"] == ["group_ensemble", "r2_head"]
    assert payload["frozen_constructions"] == list(p2policy.CONSTRUCTIONS)
    assert set(payload["frozen_constructions"]).isdisjoint(set(payload["rotating_constructions"]))
    assert payload["rows"]["UGE"]["direction_row"] == "R0.5"
    assert payload["rows"]["U2"]["direction_row"] == "R2"
    assert payload["rows"]["U0"]["construction"] == "raw"
    assert payload["budget"] == 4 and payload["surface"] == "within"
    assert payload["horizon_H"] == 5
    assert payload["sessions"] == list(plan.SESSIONS)
    assert len(payload["sessions"]) == 6
    assert payload["gates"]["boundary_epsilon"] == 1.0e-12
    assert payload["drift_guard"]["session_median_max_rad"] == 0.9
    assert payload["drift_guard"]["session_median_max_rad"] < math.pi / 2
    assert payload["scoring"]["governing"]["name"] == "raw_no_output_filter"
    assert payload["scoring"]["anchor"]["name"] == "filtered_causal_ema_a0.25"
    assert payload["pre_registered_expectation"]["true_direction_ceiling_over_raw"] == 0.0046
    assert payload["gpu_index"] == 1


def test_sidecar_verification_helper_accepts_and_rejects(tmp_path):
    body = json.dumps({"a": 1}, sort_keys=True).encode("utf-8") + b"\n"
    import hashlib

    target = tmp_path / "receipt.json"
    target.write_bytes(body)
    (tmp_path / "receipt.json.sha256").write_text(
        f"{hashlib.sha256(body).hexdigest()}  receipt.json\n", encoding="ascii",
    )
    assert directions._verify_sidecar(target) == hashlib.sha256(body).hexdigest()
    (tmp_path / "receipt.json.sha256").write_text("0" * 64 + "  receipt.json\n", encoding="ascii")
    with pytest.raises(directions.AC3UDirectionsError):
        directions._verify_sidecar(target)
    (tmp_path / "receipt.json.sha256").unlink()
    with pytest.raises(directions.AC3UDirectionsError):
        directions._verify_sidecar(target)


def test_theta_arrays_round_trip_and_tamper_detection():
    values = np.asarray([0.1, float("nan"), -2.5, 3.14159])
    payload = {"theta_arrays": {"R0.5": {"array_digest": None, "theta": [float(v) for v in values]}}}
    from src.ac3_action_continuity_v1 import matrix as mtx

    payload["theta_arrays"]["R0.5"]["array_digest"] = mtx.array_digest(values)
    reloaded = directions.load_theta_arrays(payload)
    assert np.array_equal(reloaded["R0.5"], values, equal_nan=True)
    payload["theta_arrays"]["R0.5"]["theta"][0] = 0.2
    with pytest.raises(directions.AC3UDirectionsError):
        directions.load_theta_arrays(payload)


def test_session_binding_blocks_are_contiguous_and_covered():
    sessions = list(plan.SESSIONS)
    trial_ids = []
    for session in sessions:
        for index in range(3):
            trial_ids.append(f"{session}:trial:{index}")
    binding = directions.session_binding({"sessions": sessions, "trial_ids": trial_ids})
    assert binding["n_trials"] == 18
    offset = 0
    for session in sessions:
        block = binding["sessions"][session]
        assert block["start"] == offset
        assert block["n_trials"] == 3
        assert block["stop"] == offset + 3
        assert all(item.startswith(f"{session}:trial:") for item in block["trial_ids"])
        offset += 3
    with pytest.raises(directions.AC3UDirectionsError):
        directions.session_binding({"sessions": ["other"], "trial_ids": trial_ids})


def test_anchor_report_fields_and_bit_exactness():
    sealed = {
        "matrix_r2": 0.5, "house_raw_r2": 0.4, "prediction_sha256_raw": "a",
        "filtered_prediction_sha256": "b", "n_windows": 10, "oracle_accept_count": 3,
        "oracle_decision_count": 9, "carrier_transitions_committed": 3,
        "activity_transitions_committed": 10, "carrier_rejection_counts": {"x": 4},
        "initial_carrier_sha256": "c", "initial_activity_sha256": "d",
        "final_carrier_sha256": "e", "oracle_level": "COHERENT_GREEDY_ORACLE",
        "horizon_H": 5, "construction": "raw", "output_filter": "causal_ema_a0.25",
        "oracle_decisions": [
            {"trial_id": "t0", "proposal_accepted_by_b8": True, "action": "accept",
             "u_j": 0.001, "horizon_trials": 5},
        ],
    }
    replayed = dict(sealed)
    report = replay_module_anchor(sealed)
    assert report["bit_exact"] is True
    drifted = dict(sealed)
    drifted["oracle_accept_count"] = 4
    report = replay_module_anchor(sealed, drifted)
    assert report["bit_exact"] is False
    assert report["field_matches"]["oracle_accept_count"] is False
    assert report["differences"]["oracle_accept_count"] == {"replayed": 4, "sealed": 3}
    drifted_decisions = dict(sealed)
    drifted_decisions["oracle_decisions"] = [
        {"trial_id": "t0", "proposal_accepted_by_b8": True, "action": "accept",
         "u_j": 0.0010000000000001, "horizon_trials": 5},
    ]
    report = replay_module_anchor(sealed, drifted_decisions)
    assert report["all_fields_exact"] is True
    assert report["oracle_decision_records_exact"] is False
    assert report["bit_exact"] is False


def replay_module_anchor(sealed, replayed=None):
    from src.ac3_utility_bridge_v1 import replay as replay_module

    return replay_module._anchor_report(replayed if replayed is not None else sealed, sealed)


def test_terminal_gate_block_schema():
    u0 = [0.40] * 6
    result = _governing(u0, u0, u0)
    for key in (
        "scoring", "governing", "sessions", "equal_session_mean_r2", "UGE_minus_U0",
        "U2_minus_UGE", "U2_minus_U0", "UGE_gate", "U2_rescue", "disposition",
        "five_session_sensitivity_non_governing", "no_m10_m30_claim",
        "pre_registered_expectation",
    ):
        assert key in result
    assert result["scoring"] == "raw_no_output_filter"
    assert result["governing"] is True
    assert set(result["equal_session_mean_r2"]) == {"U0", "UGE", "U2"}
    for row in ("U0", "UGE", "U2"):
        assert isinstance(result["equal_session_mean_r2"][row], float)
    assert set(result["UGE_minus_U0"]["per_session"]) == set(plan.SESSIONS)
    assert result["UGE_minus_U0"]["n_sessions"] == 6


def test_rotation_law_is_rejected_on_a_norm_violation():
    class _Bad:
        def __init__(self, velocity, validity):
            self.velocity = velocity
            self.validity = validity

    rng = np.random.default_rng(43)
    views = _views(rng, bins=6, direction=0.2)
    bad = [
        _Bad(np.asarray(item.velocity, dtype=np.float64) * (1.0 + 1.0e-3), item.validity)
        for item in views
    ]
    report = rotation.norm_preservation_report(
        [item.velocity for item in views], [item.velocity for item in bad],
    )
    assert report["max_relative_speed_gap"] > plan.NORM_PRESERVATION_TOLERANCE
    with pytest.raises(rotation.AC3URotationError):
        rotation.assert_norm_preservation(
            [item.velocity for item in views], [item.velocity for item in bad],
        )


# ---------------------------------------------------------------------------
# 6. Replay guards and the terminal body (no data, no CUDA).
# ---------------------------------------------------------------------------


def test_run_replay_refuses_without_its_prerequisites(tmp_path):
    from src.ac3_utility_bridge_v1 import replay as replay_module

    base = ROOT / "tfpd_exploration"
    empty = tmp_path / "bridge"
    empty.mkdir()
    with pytest.raises(replay_module.AC3UReplayError):
        replay_module.run_replay(base, gpu_index=1, output_root=empty)
    (empty / "attempt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(replay_module.AC3UReplayError):
        replay_module.run_replay(base, gpu_index=1, output_root=empty)
    terminal = tmp_path / "closed"
    terminal.mkdir()
    (terminal / "terminal.json").write_text("{}", encoding="utf-8")
    with pytest.raises(replay_module.AC3UReplayError):
        replay_module.run_replay(base, gpu_index=1, output_root=terminal)


def _synthetic_replay_payload():
    sessions = list(plan.SESSIONS)
    governing = {
        "U0": {session: 0.40 for session in sessions},
        "UGE": {session: 0.40 + 0.02 for session in sessions},
        "U2": {session: 0.40 + 0.03 for session in sessions},
    }
    anchor = {
        "U0": {session: 0.45 for session in sessions},
        "UGE": {session: 0.45 + 0.001 for session in sessions},
        "U2": {session: 0.45 + 0.002 for session in sessions},
    }
    return {
        "rows": {
            row: {
                "construction": plan.ROWS[row]["construction"],
                "direction_row": plan.ROWS[row]["direction_row"],
                "sessions": [
                    {
                        "session": session,
                        "matrix_r2_raw_governing": governing[row][session],
                        "matrix_r2_filtered_anchor": anchor[row][session],
                        "house_raw_r2": 0.5,
                        "prediction_sha256_raw": f"{row}-{index}",
                        "oracle_accept_count": 3,
                        "oracle_decision_count": 9,
                        "carrier_transitions_committed": 3,
                    }
                    for index, session in enumerate(sessions)
                ],
                "governing_raw_matrix_r2": governing[row],
                "anchor_filtered_matrix_r2": anchor[row],
            }
            for row in plan.ROW_ORDER
        },
        "sessions": sessions,
        "u0_anchor_all_sessions_bit_exact": True,
        "model_state_digest_before_sha256": "x" * 64,
        "model_state_digest_after_sha256": "x" * 64,
        "model_state_digest_unchanged": True,
    }


def test_build_terminal_carries_both_scoring_domains():
    from src.ac3_utility_bridge_v1 import replay as replay_module

    body = replay_module.build_terminal(replay_payload=_synthetic_replay_payload())
    gates_block = body["gates"]
    assert gates_block["scoring"] == "raw_no_output_filter"
    assert gates_block["disposition"] == plan.DISPOSITION_ADVANCE
    assert gates_block["equal_session_mean_r2"]["UGE"] == pytest.approx(0.42)
    reference = body["anchor_domain_reference_non_governing"]
    assert reference["non_governing"] is True
    assert reference["scoring"] == "filtered_causal_ema_a0.25"
    assert reference["equal_session_mean_r2"]["UGE"] == pytest.approx(0.451)
    # The two scoring domains are evaluated independently: here the anchor
    # (filtered) domain misses the +0.01 margin while the governing raw domain
    # clears it, and only the governing block sets the disposition.
    assert reference["disposition_if_it_governed"] == plan.DISPOSITION_NULL
    for row in plan.ROW_ORDER:
        assert len(body["per_session_rows"][row]) == 6
        for item in body["per_session_rows"][row]:
            assert set(item) == {
                "session", "matrix_r2_raw_governing", "matrix_r2_filtered_anchor",
                "house_raw_r2", "prediction_sha256_raw", "oracle_accept_count",
                "oracle_decision_count", "carrier_transitions_committed",
            }


def test_build_terminal_rejects_a_drifted_roster():
    from src.ac3_utility_bridge_v1 import replay as replay_module

    payload = _synthetic_replay_payload()
    payload["sessions"] = list(plan.SESSIONS)[:-1]
    with pytest.raises(replay_module.AC3UReplayError):
        replay_module.build_terminal(replay_payload=payload)
