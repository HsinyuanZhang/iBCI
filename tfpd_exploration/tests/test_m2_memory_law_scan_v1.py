"""Synthetic no-data/no-CUDA tests for the M2 memory-law scan laws.

The review-critical properties:

1. the EMA recurrence equals its closed form and carries the documented
   weights (effective pool, decaying support-mean initialization);
2. the UNCAPPED streaming pool reproduces the frozen B3S stack-mean
   accumulation bitwise (the frozen encoder's own push_trial/finalize
   arithmetic), with no eviction, including pools larger than the sealed
   30-trial stack;
3. the sealed CAP30 topology is what the plan documents (FIFO of 30 - M
   retaining the most recent completed trials);
4. the anchor logic is exact on pure-data fields and tolerance-bounded on R2;
5. the gate margins honor the 1e-12 epsilon band and the verdict law selects
   exactly as pre-registered;
6. the drift reading's Spearman statistic handles ties.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "tfpd_exploration", ROOT / "tfpd_exploration" / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import src as tfpd_src_package  # noqa: E402

_STREAMING_SRC = ROOT / "streaming_calibration_exp" / "src"
if _STREAMING_SRC.is_dir() and str(_STREAMING_SRC) not in tfpd_src_package.__path__:
    tfpd_src_package.__path__.append(str(_STREAMING_SRC))

from src.models.components.streaming_encoders import (  # noqa: E402
    SideFeatureEarlyPoolEncoder,
)

from src.causal_dual_memory_cell_d_v1 import core as cdm_core  # noqa: E402
from src.cdm_p1_m2_local_v1 import plan as g_plan  # noqa: E402
from src.m2_memory_law_scan_v1 import gates, memory as memory_law, physical, plan  # noqa: E402

from tfpd_exploration.src.m2_precision_cdm_v2_screen_v1 import plan as cdm_plan  # noqa: E402


DEVICE = torch.device("cpu")


def _tiny_encoder(*, neurons: int = 6, trial_length: int = 20, seed: int = 7):
    torch.manual_seed(seed)
    encoder = SideFeatureEarlyPoolEncoder(
        trial_length=trial_length, window_size=7, hidden_dim=8, side_dim=4,
    )
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder, neurons


def _random_activities(count: int, *, neurons: int = 6, trial_length: int = 20,
                       seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.gamma(2.0, 1.5, size=(count, trial_length, neurons)).astype(np.float32)


def _side_tensor(neurons: int, seed: int = 5):
    rng = np.random.default_rng(seed)
    side = rng.normal(size=(neurons, 4)).astype(np.float32)
    return torch.from_numpy(side).unsqueeze(0)


def _phi_numpy(encoder: SideFeatureEarlyPoolEncoder, activity: np.ndarray) -> np.ndarray:
    """Independent pre-pool feature path (the module itself, not the pool)."""
    trial = torch.from_numpy(np.ascontiguousarray(activity, dtype=np.float32))
    with torch.inference_mode():
        phi = encoder.pre_pool(trial.unsqueeze(0).permute(0, 2, 1))
    return phi[0].numpy().astype(np.float32, copy=False)


def _stack_identity(encoder: SideFeatureEarlyPoolEncoder, stack: np.ndarray,
                    side_tensor) -> torch.Tensor:
    with torch.inference_mode():
        return encoder.forward_batch(
            torch.from_numpy(np.ascontiguousarray(stack, dtype=np.float32)).unsqueeze(0),
            side_features=side_tensor,
        )


# ---------------------------------------------------------------------------
# 1. plan constants bind the sealed foundation.
# ---------------------------------------------------------------------------


def test_plan_constants_bind_the_sealed_foundation() -> None:
    assert plan.ACTIVITY_STACK_LIMIT == cdm_plan.ACTIVITY_STACK_LIMIT == g_plan.ACTIVITY_HORIZON == 30
    assert plan.SURFACES == cdm_plan.SURFACES == ("within_post30", "external_post30_local")
    assert plan.CDM_SCREEN_SCORE_SHA256 == g_plan.CDM_SCREEN_SCORE_SHA256
    assert cdm_core.activity_fifo_capacity_for_support_budget(4) == 30 - 4
    assert cdm_core.activity_fifo_capacity_for_support_budget(10) == 30 - 10
    assert plan.POLICY_ORDER[0] == plan.BASELINE_POLICY == "UNIFORM_CAP30"
    assert plan.EMA_ALPHAS == {"EMA_A090": 0.90, "EMA_A095": 0.95, "EMA_A080": 0.80}
    assert plan.GATES["primary"]["delta_floor"] == 0.01
    assert plan.GATES["primary"]["breadth_min"] == 5
    assert plan.GATES["primary"]["breadth_denominator"] == 7
    assert plan.GATES["safety"]["floor"] == -0.02
    assert plan.GATES["safety"]["budgets"] == [4, 10]
    assert plan.GATES["boundary_epsilon"] == 1.0e-12
    assert plan.ENVIRONMENT_LAW["cuda_visible_devices"] == ""
    assert plan.ENVIRONMENT_LAW["torch_num_threads"] == 4


# ---------------------------------------------------------------------------
# 2. the EMA arithmetic.
# ---------------------------------------------------------------------------


def test_ema_closed_form_weights_and_effective_pool() -> None:
    rng = np.random.default_rng(3)
    support = rng.normal(size=(4, 5))
    completed = list(rng.normal(size=(9, 5)))
    for name, alpha in plan.EMA_ALPHAS.items():
        state = memory_law.ema_closed_form(support, completed, alpha)
        k = len(completed)
        manual = (alpha ** k) * support.mean(axis=0)
        for index, feature in enumerate(completed):
            manual = manual + (1.0 - alpha) * (alpha ** (k - 1 - index)) * feature
        assert np.allclose(state, manual, rtol=0.0, atol=1.0e-12)
        # weights on the completed trials plus the decaying init sum to one
        total = alpha ** k + (1.0 - alpha) * sum(
            alpha ** (k - 1 - index) for index in range(k))
        assert abs(total - 1.0) < 1.0e-12
        assert 1.0 / (1.0 - alpha) == pytest.approx(
            plan.POLICIES[name]["effective_pool"], abs=1e-9)
        # the float32 law pair
        alpha32, beta32 = memory_law.float32_pair(alpha)
        assert alpha32.dtype == np.float32 and beta32.dtype == np.float32
        assert beta32 == np.float32(np.float32(1.0) - np.float32(alpha))


def test_ema_pool_identity_and_recurrence() -> None:
    encoder, neurons = _tiny_encoder()
    activities = _random_activities(12, neurons=neurons)
    side = _side_tensor(neurons)
    support = list(activities[:4])
    completed = list(activities[4:])
    for alpha in (0.90, 0.95, 0.80):
        pool = memory_law.FrozenB3SEmaPool(
            id_encoder=encoder, support_activities=support, channels=neurons,
            alpha=alpha, device=DEVICE, dtype=torch.float32, torch=torch,
        )
        uniform = memory_law.FrozenB3SUniformPool(
            id_encoder=encoder, support_activities=support, channels=neurons,
            device=DEVICE, dtype=torch.float32, torch=torch,
        )
        # E0 IS the support mean: before any commit the two laws agree bitwise.
        with torch.inference_mode():
            ema_identity = pool.identity(side)
            uniform_identity = uniform.identity(side)
        assert torch.equal(ema_identity, uniform_identity)
        assert pool.support_count == 4 and pool.pool_count == 0
        for activity in completed:
            pool.commit(activity)
            uniform.commit(activity)
        assert pool.pool_count == len(completed)  # count separate from the mean law
        support_phi = np.stack([_phi_numpy(encoder, item) for item in support])
        completed_phi = [_phi_numpy(encoder, item) for item in completed]
        reference = memory_law.ema_closed_form(
            support_phi, completed_phi, alpha, dtype=np.float32,
        )
        assert np.allclose(pool.pooled_features_numpy(), reference, rtol=0.0, atol=1.0e-5)
        law = pool.law_payload()
        assert law["alpha"] == pytest.approx(alpha, abs=1e-7)
        # the effective pool is computed from the operative float32 alpha
        assert law["effective_pool"] == pytest.approx(1.0 / (1.0 - alpha), abs=1e-4)
        assert law["completed_count"] == len(completed)
        assert law["initial_support_mean_weight_after_k_trials"] == pytest.approx(
            alpha ** len(completed), abs=1e-6)


# ---------------------------------------------------------------------------
# 3. the uncapped accumulation.
# ---------------------------------------------------------------------------


def test_uniform_uncapped_matches_frozen_stack_mean_bitwise() -> None:
    encoder, neurons = _tiny_encoder()
    activities = _random_activities(45, neurons=neurons)  # 4 support + 41 completed
    side = _side_tensor(neurons)
    pool = memory_law.FrozenB3SUniformPool(
        id_encoder=encoder, support_activities=list(activities[:4]),
        channels=neurons, device=DEVICE, dtype=torch.float32, torch=torch,
    )
    for index in range(4, activities.shape[0]):
        # the pool holds support + activities[4:index] here (pre-commit)
        completed = activities[:index]
        with torch.inference_mode():
            streaming_identity = pool.identity(side)
        stack_identity = _stack_identity(encoder, completed, side)
        assert torch.equal(streaming_identity, stack_identity), (
            f"streaming law drifted from the frozen stack mean at pool size {index}"
        )
        pool.commit(activities[index])
    # no eviction, ever: the pool grows past the sealed 30-trial stack
    assert pool.pool_count == activities.shape[0] == 45
    assert pool.law_payload()["eviction"] == "none_ever"


def test_uncapped_accumulation_equals_feature_mean() -> None:
    encoder, neurons = _tiny_encoder()
    activities = _random_activities(18, neurons=neurons, seed=23)
    pool = memory_law.FrozenB3SUniformPool(
        id_encoder=encoder, support_activities=list(activities[:10]),
        channels=neurons, device=DEVICE, dtype=torch.float32, torch=torch,
    )
    for activity in activities[10:]:
        pool.commit(activity)
    phis = np.stack([_phi_numpy(encoder, item) for item in activities])
    assert pool.pool_count == 18
    assert np.allclose(pool.pooled_features_numpy(), phis.mean(axis=0), rtol=0.0, atol=1.0e-5)


# ---------------------------------------------------------------------------
# 4. the sealed CAP30 topology.
# ---------------------------------------------------------------------------


def test_sealed_fifo_cap30_topology() -> None:
    channels = np.arange(5, dtype=np.int64)
    channel_sha = cdm_core.channel_order_digest(channels)

    def trial(index: int) -> cdm_core.B3SInterpolatedSpikeCountTrial:
        rng = np.random.default_rng(100 + index)
        return cdm_core.B3SInterpolatedSpikeCountTrial(
            activity=rng.gamma(2.0, 1.0, size=(100, 5)).astype(np.float32),
            session_id="synthetic", trial_id=f"synthetic:trial:{index}",
            channel_order_sha256=channel_sha,
        )

    for budget in (4, 10):
        capacity = plan.ACTIVITY_STACK_LIMIT - budget
        memory = cdm_core.ActivityMemory.initialize(
            tuple(trial(index) for index in range(budget)),
            channel_ids=channels, fifo_capacity=capacity,
        )
        for index in range(budget, budget + 40):
            memory = memory.after_completed_trial(trial(index))
        assert memory.query_count == capacity
        retained = [item.trial_id for item in memory.query_trials]
        expected = [f"synthetic:trial:{index}"
                    for index in range(budget + 40 - capacity, budget + 40)]
        assert retained == expected
        stack = memory.stack()
        assert stack.shape[0] == plan.ACTIVITY_STACK_LIMIT


# ---------------------------------------------------------------------------
# 5. the anchor logic.
# ---------------------------------------------------------------------------


def test_anchor_matches_boundaries() -> None:
    sealed = {
        "query_starts_sha256": "a" * 64, "target_sha256": "b" * 64,
        "window_count": 1234, "r2": 0.5, "prediction_sha256": "c" * 64,
    }
    # 2**-10 is exactly representable, so the delta is exactly 9.765625e-4
    cell = {
        "query_starts_sha256": "a" * 64, "target_sha256": "b" * 64,
        "window_count": 1234, "r2": 0.5 + 2.0 ** -10, "prediction_sha256": "d" * 64,
    }
    result = physical.anchor_matches(cell, sealed, r2_tolerance=1.0e-3)
    assert result["exact_match"] is True
    assert result["field_matches"]["r2_within_tolerance"] is True
    assert result["prediction_sha256_match_across_devices"] is False
    assert result["r2_delta"] == pytest.approx(2.0 ** -10, abs=1e-18)
    # comfortably past the tolerance fails the anchor
    cell["r2"] = 0.5 + 2.0 ** -9
    assert physical.anchor_matches(cell, sealed, r2_tolerance=1.0e-3)["exact_match"] is False
    # any pure-data field mismatch fails hard
    cell["r2"] = 0.5
    cell["window_count"] = 1235
    broken = physical.anchor_matches(cell, sealed, r2_tolerance=1.0e-3)
    assert broken["exact_match"] is False
    assert broken["field_matches"]["window_count"] is False


# ---------------------------------------------------------------------------
# 6. the gate boundaries (1e-12) and the verdict law.
# ---------------------------------------------------------------------------


EPSILON = plan.GATES["boundary_epsilon"]


def test_margin_epsilon_band_law() -> None:
    at = gates._margin(0.01, 0.01, epsilon=EPSILON)
    assert at["meets_margin"] is True and at["within_epsilon_band_of_boundary"] is True
    inside = gates._margin(0.01 - 5.0e-13, 0.01, epsilon=EPSILON)
    assert inside["meets_margin"] is True and inside["within_epsilon_band_of_boundary"] is True
    outside = gates._margin(0.01 - 5.0e-12, 0.01, epsilon=EPSILON)
    assert outside["meets_margin"] is False and outside["within_epsilon_band_of_boundary"] is False
    clear = gates._margin(0.02, 0.01, epsilon=EPSILON)
    assert clear["meets_margin"] is True and clear["within_epsilon_band_of_boundary"] is False
    safety_at = gates._margin(-0.02, -0.02, epsilon=EPSILON)
    assert safety_at["meets_margin"] is True
    safety_inside = gates._margin(-0.02 - 5.0e-13, -0.02, epsilon=EPSILON)
    assert safety_inside["meets_margin"] is True
    safety_outside = gates._margin(-0.02 - 5.0e-12, -0.02, epsilon=EPSILON)
    assert safety_outside["meets_margin"] is False


SESSIONS = [f"s{index}" for index in range(7)]


def _gate_inputs(mean_delta: float, positive: int):
    baseline = {name: 0.50 for name in SESSIONS}
    candidate = dict(baseline)
    for name in SESSIONS[:positive]:
        candidate[name] = baseline[name] + mean_delta * len(SESSIONS) / positive
    external = {name: 0.30 for name in SESSIONS}
    external_candidate = dict(external)
    return candidate, baseline, external_candidate, external


def test_policy_gates_and_verdict_selection() -> None:
    # a clear EMA pass: +0.02 mean delta, 6/7 breadth, safe external
    candidate, baseline, ext_candidate, ext_baseline = _gate_inputs(0.02, 6)
    winner = gates.evaluate_policy_gates(
        policy="EMA_A090", within_m4=candidate, baseline_within_m4=baseline,
        external_by_budget={4: ext_candidate, 10: ext_candidate},
        baseline_external_by_budget={4: ext_baseline, 10: ext_baseline},
        gates_law=plan.GATES,
    )
    assert winner["primary"]["passed"] is True
    assert winner["safety"]["passed"] is True
    assert winner["selected"] is True
    assert winner["primary"]["delta"]["positive_sessions"] == 6
    assert winner["primary"]["delta"]["equal_session_mean_delta"] == pytest.approx(0.02, abs=1e-12)

    # breadth 4/7 fails the primary gate even with a large mean delta
    narrow, narrow_base, _, _ = _gate_inputs(0.05, 4)
    fails_breadth = gates.evaluate_policy_gates(
        policy="EMA_A090", within_m4=narrow, baseline_within_m4=narrow_base,
        external_by_budget={4: ext_candidate, 10: ext_candidate},
        baseline_external_by_budget={4: ext_baseline, 10: ext_baseline},
        gates_law=plan.GATES,
    )
    assert fails_breadth["primary"]["passed"] is False
    assert fails_breadth["disposition"] == "CHALLENGER_FAILS_THE_PRIMARY_GATE"

    # a miss inside the 1e-12 band of the +0.01 floor never flips the verdict
    edge_candidate = {name: 0.50 for name in SESSIONS}
    edge_base = {name: 0.50 - 0.01 + 5.0e-13 for name in SESSIONS}
    edge = gates.evaluate_policy_gates(
        policy="EMA_A080", within_m4=edge_candidate, baseline_within_m4=edge_base,
        external_by_budget={4: ext_candidate, 10: ext_candidate},
        baseline_external_by_budget={4: ext_baseline, 10: ext_baseline},
        gates_law=plan.GATES,
    )
    assert edge["primary"]["delta_margin"]["within_epsilon_band_of_boundary"] is True
    assert edge["primary"]["delta_margin"]["meets_margin"] is True

    # safety failure at one budget excludes a primary passer
    unsafe_external = {name: 0.30 - 0.03 for name in SESSIONS}
    unsafe = gates.evaluate_policy_gates(
        policy="EMA_A095", within_m4=candidate, baseline_within_m4=baseline,
        external_by_budget={4: ext_candidate, 10: unsafe_external},
        baseline_external_by_budget={4: ext_baseline, 10: ext_baseline},
        gates_law=plan.GATES,
    )
    assert unsafe["primary"]["passed"] is True and unsafe["safety"]["passed"] is False
    assert unsafe["disposition"] == "PRIMARY_PASS_SAFETY_FAIL_EXCLUDED"

    # verdict law: retained, EMA wins, UNCAPPED wins, tie-break, safety exclusion
    def result(policy, selected, primary_pass, safety_pass, delta):
        return {
            "policy": policy, "selected": selected,
            "primary": {"passed": primary_pass,
                        "delta": {"equal_session_mean_delta": delta}},
            "safety": {"passed": safety_pass},
        }

    order = [item for item in plan.POLICY_ORDER if item != plan.BASELINE_POLICY]
    retained = gates.select_verdict(
        policy_results={name: result(name, False, False, True, 0.0) for name in order},
        policy_order=order,
    )
    assert retained["verdict"] == "MEMORY_LAW_UNIFORM_RETAINED"
    assert retained["winner"] is None

    ema_wins = gates.select_verdict(
        policy_results={
            "UNIFORM_UNCAPPED": result("UNIFORM_UNCAPPED", True, True, True, 0.012),
            "EMA_A090": result("EMA_A090", True, True, True, 0.02),
            "EMA_A095": result("EMA_A095", False, True, False, 0.05),
            "EMA_A080": result("EMA_A080", False, False, True, -0.01),
        },
        policy_order=order,
    )
    assert ema_wins["verdict"] == "EMA_RECENCY_WINS(EMA_A090)"
    assert ema_wins["primary_pass_safety_fail_excluded"] == ["EMA_A095"]

    uncapped_wins = gates.select_verdict(
        policy_results={
            "UNIFORM_UNCAPPED": result("UNIFORM_UNCAPPED", True, True, True, 0.03),
            "EMA_A090": result("EMA_A090", True, True, True, 0.02),
            "EMA_A095": result("EMA_A095", False, False, True, 0.0),
            "EMA_A080": result("EMA_A080", False, False, True, 0.0),
        },
        policy_order=order,
    )
    assert uncapped_wins["verdict"] == "UNCAPPED_WINS"

    tied = gates.select_verdict(
        policy_results={
            "UNIFORM_UNCAPPED": result("UNIFORM_UNCAPPED", True, True, True, 0.02),
            "EMA_A090": result("EMA_A090", True, True, True, 0.02),
            "EMA_A095": result("EMA_A095", False, False, True, 0.0),
            "EMA_A080": result("EMA_A080", False, False, True, 0.0),
        },
        policy_order=order,
    )
    # POLICY_ORDER tie-break: UNIFORM_UNCAPPED is enumerated before the EMAs
    assert tied["verdict"] == "UNCAPPED_WINS"

    only_excluded = gates.select_verdict(
        policy_results={
            "UNIFORM_UNCAPPED": result("UNIFORM_UNCAPPED", False, False, True, 0.0),
            "EMA_A090": result("EMA_A090", False, True, False, 0.05),
            "EMA_A095": result("EMA_A095", False, False, True, 0.0),
            "EMA_A080": result("EMA_A080", False, False, True, 0.0),
        },
        policy_order=order,
    )
    assert only_excluded["verdict"] == "MEMORY_LAW_UNIFORM_RETAINED"
    assert only_excluded["primary_pass_safety_fail_excluded"] == ["EMA_A090"]


def test_paired_delta_and_session_sd() -> None:
    baseline = {"a": 0.5, "b": 0.4, "c": 0.6}
    candidate = {"a": 0.52, "b": 0.38, "c": 0.66}
    delta = gates.paired_delta(candidate, baseline)
    assert delta["per_session_delta"] == {"a": pytest.approx(0.02), "b": pytest.approx(-0.02), "c": pytest.approx(0.06)}
    assert delta["equal_session_mean_delta"] == pytest.approx(0.02)
    assert delta["positive_sessions"] == 2
    assert delta["session_count"] == 3
    values = {"a": 0.2, "b": 0.4, "c": 0.6}
    assert gates.equal_session_mean(values) == pytest.approx(0.4)
    assert gates.session_sd(values) == pytest.approx(
        float(np.sqrt(((np.array([0.2, 0.4, 0.6]) - 0.4) ** 2).mean())), abs=1e-15)
    with pytest.raises(gates.MemoryLawGateError):
        gates.paired_delta({"a": 1.0}, {"b": 1.0})


# ---------------------------------------------------------------------------
# 7. the drift reading.
# ---------------------------------------------------------------------------


def test_spearman_and_drift_reading() -> None:
    assert gates.spearman_rho([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert gates.spearman_rho([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert gates.spearman_rho([1, 1, 2], [1, 2, 2]) == pytest.approx(0.5)
    counts = {f"s{index}": 100 * (index + 1) for index in range(7)}
    deltas_increasing = {
        name: 0.001 * index for index, name in enumerate(sorted(counts, key=counts.get))
    }
    # alternating deltas are exactly orthogonal to the trial-count ranks
    deltas_alternating = {
        f"s{index}": 0.001 if index % 2 == 0 else -0.001 for index in range(7)
    }
    reading = gates.drift_reading(
        session_completed_trials=counts,
        within_m4_deltas_by_policy={"EMA_A090": deltas_increasing,
                                    "EMA_A080": deltas_alternating},
    )
    assert reading["per_policy"]["EMA_A090"]["spearman_rho_delta_vs_completed_trials"] == pytest.approx(1.0)
    assert reading["per_policy"]["EMA_A080"]["spearman_rho_delta_vs_completed_trials"] == pytest.approx(0.0)
    median = reading["median_completed_trials"]
    assert median == 400.0
    assert len(reading["long_half_sessions"]) == 3 and len(reading["short_half_sessions"]) == 4
    # long half = s4, s5, s6 with deltas [+v, -v, +v] -> mean v/3
    assert reading["per_policy"]["EMA_A080"]["long_half_mean_delta"] == pytest.approx(0.001 / 3.0)
