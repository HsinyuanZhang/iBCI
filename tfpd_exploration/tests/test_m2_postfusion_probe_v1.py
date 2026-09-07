"""Synthetic no-data/no-CUDA tests for the M2 post-fusion identity probe laws.

The review-critical properties:

1. the plan constants bind the sealed champion shape (D-opt-4 support, CAP30
   FIFO capacity 26, the sealed G00m/k4 anchor means, the 1e-5 anchor
   tolerance and the 1e-12 gate epsilon);
2. the per-trial identity law: ``per_trial_identity`` IS
   ``post_pool(concat(pre_pool(trial), side))`` bitwise (the frozen encoder's
   own single-trial finalize arithmetic) and equals a single-trial stack
   through ``forward_batch`` bitwise;
3. the running-mean accumulation: the uncapped pool's incremental running
   mean equals the from-scratch sequential float32 sum bitwise at every
   commit; the capped pool evicts only COMPLETED identities (support is never
   evicted), holds the capacity, and stays within float32 drift of the
   from-scratch mean over its members;
4. the EMA pool in identity space: E0 IS the uniform support-identity mean
   bitwise and the recurrence is the float32 (alpha, 1-alpha) law;
5. the Jensen-gap premise: mean-after-MLP differs from MLP-after-mean on a
   nonlinear frozen encoder (the shortcut law does not exist);
6. the anchor logic is exact on pure-data fields and 1e-5-tolerance-bounded
   on R2;
7. the gate margins honor the 1e-12 epsilon band and the three-way verdict
   law selects exactly as pre-registered (PROMISING / NULL / HARMFUL, and the
   mean-pass-breadth-fail disclosure).
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

from src.cdm_p1_m2_local_v1 import plan as g_plan  # noqa: E402
from src.m2_postfusion_probe_v1 import gates, memory as identity_pools, physical, plan  # noqa: E402

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


def _identity(encoder: SideFeatureEarlyPoolEncoder, activity: np.ndarray,
              side_tensor) -> torch.Tensor:
    """The independent per-trial identity authority: modules, not the pool."""
    with torch.inference_mode():
        return identity_pools.per_trial_identity(
            encoder, torch, activity, side_tensor,
            channels=activity.shape[1], device=DEVICE, dtype=torch.float32,
        )


def _pooled_identity(encoder: SideFeatureEarlyPoolEncoder, stack: np.ndarray,
                     side_tensor) -> torch.Tensor:
    with torch.inference_mode():
        return encoder.forward_batch(
            torch.from_numpy(np.ascontiguousarray(stack, dtype=np.float32)).unsqueeze(0),
            side_features=side_tensor,
        )


# ---------------------------------------------------------------------------
# 1. plan constants bind the sealed champion shape.
# ---------------------------------------------------------------------------


def test_plan_constants_bind_the_sealed_foundation() -> None:
    assert plan.BUDGETS == (4,) and plan.M4 == 4
    assert plan.ACTIVITY_STACK_LIMIT == cdm_plan.ACTIVITY_STACK_LIMIT == g_plan.ACTIVITY_HORIZON == 30
    assert plan.FIFO_CAPACITY == 26 == plan.ACTIVITY_STACK_LIMIT - plan.M4
    assert plan.SURFACES == ("external_post30_local", "within_post30")
    assert plan.EXPECTED_SESSIONS_BY_SURFACE == {"external_post30_local": 6, "within_post30": 7}
    assert plan.CELL_ORDER == (
        "POOLED", "POSTFUSION_MEAN", "POSTFUSION_ACCUM_UNCAPPED", "POSTFUSION_EMA_A090")
    assert plan.BASELINE_CELL == "POOLED" and plan.GATED_CELL == "POSTFUSION_MEAN"
    assert plan.EMA_ALPHA == 0.90
    assert plan.G00M_ANCHOR_MEANS == {
        "external_post30_local": 0.2990573453320357,
        "within_post30": 0.6540862067123892,
    }
    assert plan.ANCHORS["pooled_vs_sealed_g00m_k4_rows"]["r2_tolerance"] == 1.0e-5
    assert plan.ANCHORS["pooled_vs_sealed_g00m_k4_rows"]["mean_tolerance"] == 1.0e-5
    assert plan.GATES["primary"]["delta_floor"] == 0.005
    assert plan.GATES["primary"]["breadth_min"] == 4
    assert plan.GATES["primary"]["breadth_denominator"] == 6
    assert plan.GATES["harmful"]["floor"] == -0.005
    assert plan.GATES["boundary_epsilon"] == 1.0e-12
    assert plan.ENVIRONMENT_LAW["cuda_visible_devices"] == ""
    assert plan.ENVIRONMENT_LAW["torch_num_threads"] == 4
    assert plan.ENVIRONMENT_LAW["dataloader_workers"] == 0
    assert plan.TRIAL_LENGTH == 100 and plan.ID_HIDDEN_DIM == 64
    assert plan.SIDE_DIM == 4 and plan.WINDOW_BINS == 50
    assert plan.CELLS["POSTFUSION_MEAN"]["mean_placement"] == "after_the_fusion_mlp"
    assert plan.CELLS["POOLED"]["mean_placement"] == "before_the_fusion_mlp"


def test_identity_path_structure_law_binds_the_frozen_encoder() -> None:
    # a plan-scale encoder (the structural law is mirrored at launch on the
    # real frozen checkpoint through the same function)
    torch.manual_seed(7)
    encoder = SideFeatureEarlyPoolEncoder(
        trial_length=plan.TRIAL_LENGTH, window_size=plan.WINDOW_BINS,
        hidden_dim=plan.ID_HIDDEN_DIM, side_dim=plan.SIDE_DIM,
    )
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    structure = physical._verify_identity_path_structure(
        student=type("S", (), {"id_encoder": encoder}), torch=torch)
    assert structure["variant"] == "B3S"
    assert structure["pre_pool"] == "Linear(100->64)+ReLU"
    assert structure["post_pool"].startswith("68->64->64->50")
    linears = [layer for layer in encoder.post_pool if isinstance(layer, torch.nn.Linear)]
    assert len(linears) == 3
    assert linears[0].in_features == plan.ID_HIDDEN_DIM + plan.SIDE_DIM == 68
    assert linears[-1].out_features == plan.WINDOW_BINS == 50
    # a hidden-dim drift must fail the binding
    encoder.hidden_dim = encoder.hidden_dim + 1
    try:
        with pytest.raises(physical.PostFusionProbeError):
            physical._verify_identity_path_structure(
                student=type("S", (), {"id_encoder": encoder}), torch=torch)
    finally:
        encoder.hidden_dim = encoder.hidden_dim - 1


# ---------------------------------------------------------------------------
# 2. the per-trial identity law.
# ---------------------------------------------------------------------------


def test_per_trial_identity_is_post_pool_of_concat_bitwise() -> None:
    encoder, neurons = _tiny_encoder()
    activities = _random_activities(5, neurons=neurons)
    side = _side_tensor(neurons)
    for activity in activities:
        mine = _identity(encoder, activity, side)
        with torch.inference_mode():
            trial = torch.from_numpy(np.ascontiguousarray(activity, dtype=np.float32))
            phi = encoder.pre_pool(trial.unsqueeze(0).permute(0, 2, 1))
            manual = encoder.post_pool(torch.cat([phi, side], dim=-1))
        assert torch.equal(mine, manual)
        # a single-trial stack through forward_batch is the same arithmetic
        stack = _pooled_identity(encoder, activity[None, ...], side)
        assert torch.equal(mine, stack)
        assert tuple(mine.shape) == (1, neurons, encoder.window_size)


# ---------------------------------------------------------------------------
# 3. the running-mean accumulation.
# ---------------------------------------------------------------------------


def test_uncapped_running_mean_is_bitwise_sequential_accumulation() -> None:
    encoder, neurons = _tiny_encoder()
    activities = _random_activities(40, neurons=neurons)  # 4 support + 36 completed
    side = _side_tensor(neurons)
    support = [_identity(encoder, item, side) for item in activities[:4]]
    completed = [_identity(encoder, item, side) for item in activities[4:]]
    pool = identity_pools.PostFusionUniformIdentityPool(support_identities=support)
    members = list(support)
    with torch.inference_mode():
        for index, identity in enumerate(completed):
            deployed = pool.deployed()
            reference = identity_pools.sequential_mean(members)
            assert torch.equal(deployed, reference), (
                f"running law drifted from the from-scratch sum at commit {index}"
            )
            pool.commit(identity)
            members.append(identity)
        assert pool.pool_count == 40 == len(members)
        law = pool.law_payload()
    assert law["policy"] == "UNIFORM_UNCAPPED"
    assert law["eviction"] == "none, ever"
    assert law["support_count"] == 4 and law["completed_count"] == 36
    assert law["evictions"] == 0


def test_capped_pool_evicts_only_completed_identities() -> None:
    encoder, neurons = _tiny_encoder()
    activities = _random_activities(20, neurons=neurons, seed=23)
    side = _side_tensor(neurons)
    support = [_identity(encoder, item, side) for item in activities[:4]]
    completed = [_identity(encoder, item, side) for item in activities[4:]]
    capacity = 7  # 4 support + 3 completed (the CAP30 law at miniature scale)
    pool = identity_pools.PostFusionUniformIdentityPool(
        support_identities=support, capacity=capacity)
    with torch.inference_mode():
        for identity in completed:
            if pool.pool_count < capacity:
                # pre-eviction: the running law IS the sequential order, so
                # the deployed mean is bitwise the from-scratch mean (the
                # first-scored agreement anchor's no-eviction branch)
                assert torch.equal(pool.deployed(), pool.from_scratch_mean())
            pool.commit(identity)
        assert pool.pool_count == capacity
        assert pool.completed_count == len(completed)
        assert pool.evictions == len(completed) - (capacity - 4)
        # support identities are never evicted
        members = pool.member_identities()
        assert all(torch.equal(a, b) for a, b in zip(members[:4], support))
        # the retained completed identities are the MOST RECENT three
        assert all(torch.equal(a, b) for a, b in zip(members[4:], completed[-3:]))
        # the incremental running mean stays within float32 drift of the
        # from-scratch mean over the same members
        drift = float(torch.max(torch.abs(
            pool.deployed() - pool.from_scratch_mean())).item())
        assert drift < 1.0e-4
    law = pool.law_payload()
    assert law["policy"] == "UNIFORM_CAP30"
    assert law["capacity"] == 7 and law["evictions"] == 13


def test_pool_construction_law_failures() -> None:
    encoder, neurons = _tiny_encoder()
    activities = _random_activities(3, neurons=neurons)
    side = _side_tensor(neurons)
    ids = [_identity(encoder, item, side) for item in activities]
    with pytest.raises(identity_pools.PostFusionMemoryError):
        identity_pools.PostFusionUniformIdentityPool(support_identities=[])
    with pytest.raises(identity_pools.PostFusionMemoryError):
        # capacity must exceed the support count (support is never evicted)
        identity_pools.PostFusionUniformIdentityPool(support_identities=ids, capacity=3)
    with pytest.raises(identity_pools.PostFusionMemoryError):
        identity_pools.PostFusionEmaIdentityPool(support_identities=ids, alpha=1.0)


# ---------------------------------------------------------------------------
# 4. the EMA pool in identity space.
# ---------------------------------------------------------------------------


def test_ema_pool_initializes_at_support_mean_and_recurses() -> None:
    encoder, neurons = _tiny_encoder()
    activities = _random_activities(14, neurons=neurons, seed=31)
    side = _side_tensor(neurons)
    support = [_identity(encoder, item, side) for item in activities[:4]]
    completed = [_identity(encoder, item, side) for item in activities[4:]]
    uniform = identity_pools.PostFusionUniformIdentityPool(support_identities=support)
    pool = identity_pools.PostFusionEmaIdentityPool(
        support_identities=support, alpha=0.90)
    alpha32, beta32 = identity_pools.float32_pair(0.90)
    with torch.inference_mode():
        # E0 IS the support-identity mean: bitwise agreement before commits
        assert torch.equal(pool.deployed(), uniform.deployed())
        state = identity_pools.sequential_mean(list(support))
        for identity in completed:
            pool.commit(identity)
            state = alpha32 * state + beta32 * identity
        assert torch.equal(pool.deployed(), state)
    law = pool.law_payload()
    assert law["alpha"] == pytest.approx(0.90, abs=1e-7)
    assert law["beta"] == pytest.approx(0.10, abs=1e-7)
    assert law["effective_pool"] == pytest.approx(10.0, abs=1e-5)
    assert law["completed_count"] == 10
    assert law["initial_support_mean_weight_after_k_trials"] == pytest.approx(
        0.90 ** 10, abs=1e-6)


# ---------------------------------------------------------------------------
# 5. the Jensen-gap premise (the shortcut law does not exist).
# ---------------------------------------------------------------------------


def test_mean_after_mlp_differs_from_mlp_after_mean() -> None:
    encoder, neurons = _tiny_encoder(seed=13)
    activities = _random_activities(6, neurons=neurons, seed=17)
    side = _side_tensor(neurons)
    ids = [_identity(encoder, item, side) for item in activities]
    with torch.inference_mode():
        mean_after = identity_pools.sequential_mean(ids)
        mlp_after = _pooled_identity(encoder, activities, side)
    gap = (mean_after - mlp_after).abs()
    assert float(gap.max().item()) > 1.0e-3, "a nonlinear MLP must not commute with the mean"
    assert torch.isfinite(gap).all()


# ---------------------------------------------------------------------------
# 6. the anchor logic (pure-data fields exact, R2 within 1e-5).
# ---------------------------------------------------------------------------


def test_anchor_matches_boundaries() -> None:
    sealed = {
        "query_starts_sha256": "a" * 64, "target_sha256": "b" * 64,
        "window_count": 1234, "r2": 0.5, "prediction_sha256": "c" * 64,
    }
    # 2**-17 (~7.6e-6) is exactly representable at 0.5's scale, so the delta
    # is exact and sits inside the 1e-5 tolerance
    cell = {
        "query_starts_sha256": "a" * 64, "target_sha256": "b" * 64,
        "window_count": 1234, "r2": 0.5 + 2.0 ** -17, "prediction_sha256": "d" * 64,
    }
    result = physical.anchor_matches(cell, sealed, r2_tolerance=1.0e-5)
    assert result["exact_match"] is True
    assert result["field_matches"]["r2_within_tolerance"] is True
    assert result["prediction_sha256_match_across_devices"] is False
    assert result["r2_delta"] == pytest.approx(2.0 ** -17, abs=1e-18)
    # 2**-16 (~1.53e-5) is past the tolerance
    cell["r2"] = 0.5 + 2.0 ** -16
    assert physical.anchor_matches(cell, sealed, r2_tolerance=1.0e-5)["exact_match"] is False
    # any pure-data field mismatch fails hard
    cell["r2"] = 0.5
    cell["window_count"] = 1235
    broken = physical.anchor_matches(cell, sealed, r2_tolerance=1.0e-5)
    assert broken["exact_match"] is False
    assert broken["field_matches"]["window_count"] is False


# ---------------------------------------------------------------------------
# 7. the gate boundaries (1e-12) and the three-way verdict law.
# ---------------------------------------------------------------------------


EPSILON = plan.GATES["boundary_epsilon"]
SESSIONS = [f"s{index}" for index in range(6)]


def _delta(mean_delta: float, positive: int) -> dict:
    per_session = {name: 0.0 for name in SESSIONS}
    if positive > 0:
        for name in SESSIONS[:positive]:
            per_session[name] = mean_delta * len(SESSIONS) / positive
    values = [per_session[name] for name in SESSIONS]
    return {
        "per_session_delta": per_session,
        "equal_session_mean_delta": float(sum(values) / len(values)),
        "positive_sessions": int(sum(1 for value in values if value > 0.0)),
        "session_count": len(SESSIONS),
    }


def _uniform_delta(value: float) -> dict:
    per_session = {name: float(value) for name in SESSIONS}
    return {
        "per_session_delta": per_session,
        "equal_session_mean_delta": float(value),
        "positive_sessions": int(sum(1 for item in per_session.values() if item > 0.0)),
        "session_count": len(SESSIONS),
    }


def test_margin_epsilon_band_law() -> None:
    at = gates._margin(0.005, 0.005, epsilon=EPSILON)
    assert at["meets_margin"] is True and at["within_epsilon_band_of_boundary"] is True
    inside = gates._margin(0.005 - 5.0e-13, 0.005, epsilon=EPSILON)
    assert inside["meets_margin"] is True and inside["within_epsilon_band_of_boundary"] is True
    outside = gates._margin(0.005 - 5.0e-12, 0.005, epsilon=EPSILON)
    assert outside["meets_margin"] is False and outside["within_epsilon_band_of_boundary"] is False
    clear = gates._margin(0.02, 0.005, epsilon=EPSILON)
    assert clear["meets_margin"] is True and clear["within_epsilon_band_of_boundary"] is False


def test_verdict_three_way_law() -> None:
    # PROMISING: exactly at the floor with exactly 4/6 breadth
    result = gates.evaluate_verdict(
        external_delta=_delta(0.005, 4), gates_law=plan.GATES)
    assert result["verdict"] == "POSTFUSION_PROMISING"
    assert result["mean_pass_breadth_fail"] is False
    # a miss inside the 1e-12 band of the floor never flips the verdict
    band = gates.evaluate_verdict(
        external_delta=_delta(0.005 - 5.0e-13, 4), gates_law=plan.GATES)
    assert band["verdict"] == "POSTFUSION_PROMISING"
    assert band["delta_margin"]["within_epsilon_band_of_boundary"] is True
    # NULL strictly inside the band
    null = gates.evaluate_verdict(
        external_delta=_delta(0.0049, 6), gates_law=plan.GATES)
    assert null["verdict"] == "POSTFUSION_NULL"
    # HARMFUL at the floor
    harmful = gates.evaluate_verdict(
        external_delta=_uniform_delta(-0.005), gates_law=plan.GATES)
    assert harmful["verdict"] == "POSTFUSION_HARMFUL"
    assert harmful["harmful_margin"]["within_epsilon_band_of_boundary"] is True
    # HARMFUL strictly past the floor
    assert gates.evaluate_verdict(
        external_delta=_uniform_delta(-0.006), gates_law=plan.GATES)["verdict"] == "POSTFUSION_HARMFUL"
    # delta floor met but breadth 3/6 fails: NULL with the disclosure set
    narrow = gates.evaluate_verdict(
        external_delta=_delta(0.008, 3), gates_law=plan.GATES)
    assert narrow["verdict"] == "POSTFUSION_NULL"
    assert narrow["mean_pass_breadth_fail"] is True
    # breadth is an integer margin: 3 never meets 4 even inside the band
    assert narrow["breadth_margin"]["meets_margin"] is False


def test_paired_delta_and_session_sd() -> None:
    baseline = {"a": 0.5, "b": 0.4, "c": 0.6}
    candidate = {"a": 0.52, "b": 0.38, "c": 0.66}
    delta = gates.paired_delta(candidate, baseline)
    assert delta["per_session_delta"] == {
        "a": pytest.approx(0.02), "b": pytest.approx(-0.02), "c": pytest.approx(0.06)}
    assert delta["equal_session_mean_delta"] == pytest.approx(0.02)
    assert delta["positive_sessions"] == 2
    assert delta["session_count"] == 3
    values = {"a": 0.2, "b": 0.4, "c": 0.6}
    assert gates.equal_session_mean(values) == pytest.approx(0.4)
    assert gates.session_sd(values) == pytest.approx(
        float(np.sqrt(((np.array([0.2, 0.4, 0.6]) - 0.4) ** 2).mean())), abs=1e-15)
    with pytest.raises(gates.PostFusionProbeGateError):
        gates.paired_delta({"a": 1.0}, {"b": 1.0})
