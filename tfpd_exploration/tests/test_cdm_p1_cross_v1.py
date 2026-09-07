"""Synthetic no-data/no-CUDA tests for the CDM x P1 factorial, Part A (V1).

Everything runs on constructed numpy blocks, real frozen
``src.causal_dual_memory_cell_d_v1.core`` capability objects, the frozen P2'
filters, the REAL Stage-O/Stage-P modules this route orchestrates, a CPU-only
torch toy of the weight-swap law, and the frozen receipts already inside the
repository (no SUBC/SUBM data roots, no CUDA).  The review-critical properties:

1. the pre-registration: the four cells, the sealed P1 hyperparameters (pinned
   AND equal to the sealed Stage-P terminal receipt), the weight-swap law and
   the work-order section 2 gate boundaries;
2. the sealed bindings: the Stage-P promotion anchor and the C1 SWA artifact /
   deployment binding, fail-closed on drift;
3. the weight-swap law: strict load, artifact/state-digest fail-closed, BOTH
   runtime model seams swapped, restore with digest re-verification;
4. the cell laws on a toy runtime: F00/F10 are the Stage-O O0 arm VERBATIM,
   F01/F11 are the Stage-P P1 rollout VERBATIM under the sealed selection, the
   initial carrier is weight-invariant, and the per-arm M30 no-op holds;
5. the section 2 gates: epsilon band semantics, per-cell promotion, the F11
   additivity clause, the within/M30 floors, stop conditions, paired contrasts;
6. the stage guards and the terminal builder.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Mapping, Optional

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT / "tfpd_exploration", ROOT / "tfpd_exploration" / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.causal_dual_memory_cell_d_score_v1 import plan as v1plan
from src.causal_dual_memory_cell_d_v1 import core as cdm_core
from src.learned_gate_p2prime_v1 import filters as p2filters

from src.support_anchored_t4_stage_o_v1 import replay as stage_o_replay
from src.support_anchored_t4_stage_p_v1 import replay as stage_p_replay

from src.cdm_p1_cross_v1 import gates, plan, replay, weights

CANON = cdm_core.CANONICAL_DIRECTIONS_RAD
N_UNITS = 8
SESSION_ID = "toy-session"
DT = 0.02
DEFAULT_BIAS = (0.02, -0.03, 0.01, -0.01)
DEFAULT_SPEED = (9.0, 12.0, 6.0, 15.0)


# ---------------------------------------------------------------------------
# 1. Pre-registration and the sealed hyperparameter pin.
# ---------------------------------------------------------------------------


def test_pre_registration_payload_is_pinned():
    payload = plan.pre_registration_payload()
    assert payload["cell_order"] == ["F00", "F10", "F01", "F11"]
    assert payload["baseline_cell"] == "F00"
    assert payload["candidate_cells"] == ["F10", "F01", "F11"]
    assert payload["cell_by_arm"] == {"sealed": ["F00", "F01"], "c1": ["F10", "F11"]}
    assert payload["carrier_law_by_cell"] == {
        "F00": "frozen_t4", "F10": "frozen_t4", "F01": "p1_online", "F11": "p1_online",
    }
    assert payload["budgets"] == [4, 10, 30]
    assert payload["surfaces"] == ["within", "external"]
    assert payload["hyperparameters"]["m4"]["rho_M"] == 0.5
    assert payload["hyperparameters"]["m4"]["c_M"] == 286.77449403760136
    assert payload["hyperparameters"]["m10"]["rho_M"] == 1.0
    assert payload["hyperparameters"]["m10"]["c_M"] == 426.5520051748518
    assert payload["hyperparameters"]["m30"]["alpha_M"] == 0.0
    assert payload["hyperparameters"]["m30"]["c_M"] is None
    assert payload["hyperparameters"]["m30"]["thresholds"] == payload["hyperparameters"]["m10"]["thresholds"]
    assert payload["gates"]["boundary_epsilon"] == 1.0e-12
    assert payload["gates"]["per_cell_promotion"]["driving_budget"] == 4
    assert payload["gates"]["per_cell_promotion"]["cells"] == ["F10", "F01", "F11"]
    assert payload["gates"]["additivity"]["slack"] == 0.0
    assert payload["gates"]["safety"]["within_every_budget_floor"] == -0.02
    assert payload["gates"]["safety"]["m30_floor"] == -0.02
    assert payload["hyperparameter_law"]["re_selection"].startswith("FORBIDDEN")
    assert payload["weight_swap_law"]["training_under_swap"] == "FORBIDDEN"
    plan.validate_pre_registration(payload)
    drifted = json.loads(json.dumps(payload))
    drifted["hyperparameters"]["m4"]["alpha_M"] = 0.25
    with pytest.raises(ValueError):
        plan.validate_pre_registration(drifted)
    drifted = json.loads(json.dumps(payload))
    drifted["cells"]["F10"]["weights"] = "sealed"
    with pytest.raises(ValueError):
        plan.validate_pre_registration(drifted)
    drifted = json.loads(json.dumps(payload))
    drifted["gates"]["additivity"]["slack"] = 0.005
    with pytest.raises(ValueError):
        plan.validate_pre_registration(drifted)
    drifted = json.loads(json.dumps(payload))
    drifted["hyperparameters"]["m30"]["thresholds"] = dict(
        plan.SEALED_P1_HYPERPARAMETERS[4]["thresholds"]
    )
    with pytest.raises(ValueError):
        plan.validate_pre_registration(drifted)


def test_cell_hyperparameters_carry_the_sealed_selection():
    for budget in plan.BUDGETS:
        hp = replay.cell_hyperparameters(budget)
        pinned = plan.SEALED_P1_HYPERPARAMETERS[budget]
        assert hp.rho_M == pinned["rho_M"]
        assert hp.alpha_M == pinned["alpha_M"]
        assert hp.c_M == pinned["c_M"]
        assert hp.thresholds.payload() == pinned["thresholds"]
        assert hp.thresholds.tau_d == math.pi / 4.0
    assert replay.cell_hyperparameters(30).alpha_M == 0.0


# ---------------------------------------------------------------------------
# 2. Sealed bindings (frozen receipts inside the repository only).
# ---------------------------------------------------------------------------


def test_stage_p_go_anchor_verification():
    anchor = replay.verify_stage_p_go_anchor(ROOT)
    assert anchor["verified"] is True
    assert anchor["decision"] == plan.STAGE_P_GO_ANCHOR["decision"]
    assert anchor["driving_cell"] == "P1@m4"
    assert anchor["m4_external_p1_minus_p0"] == plan.STAGE_P_GO_ANCHOR["m4_external_p1_minus_p0"]


def test_sealed_selection_matches_the_sealed_stage_p_terminal():
    binding = replay.verify_sealed_selection_from_terminal(ROOT)
    assert binding["values_match_plan"] is True
    for budget in (4, 10):
        assert binding["checked"][f"m{budget}"]["sealed"] == dict(
            plan.SEALED_P1_HYPERPARAMETERS[budget]
        )


def test_c1_bindings_verification():
    binding = replay.verify_c1_bindings(ROOT)
    assert binding["verified"] is True
    assert binding["artifact_sha256"] == plan.C1_SWA_SHA256
    assert binding["deployment_arm_state_sha256"] == plan.C1_ARM_STATE_SHA256
    assert binding["strict_load"] is True


def test_c1_binding_fails_closed_on_deployment_drift(tmp_path):
    deployment = {
        "status": plan.C1_DEPLOYMENT_STATUS,
        "arms": {"c1": {"bindings": {
            "artifact_sha256": "0" * 64, "arm_state_sha256": plan.C1_ARM_STATE_SHA256,
            "strict_load": True,
        }}},
    }
    target = tmp_path / plan.C1_DEPLOYMENT_TERMINAL_RELATIVE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(deployment), encoding="utf-8")
    with pytest.raises(replay.CrossReplayError):
        replay.verify_c1_bindings(tmp_path)


# ---------------------------------------------------------------------------
# 3. The weight-swap law on a CPU-only torch toy.
# ---------------------------------------------------------------------------


class _ToyRecorder:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, key):
        return 0 if key == "uniform_calls" else []


class _ToyPopRobust:
    """The toy pop_robust seam: builder + dropout recorder."""

    @staticmethod
    def build_population_robustness_model(seed=42, cell="D"):
        assert seed == 42 and cell == "D"
        return _ToySwapModel()

    @staticmethod
    def dynamic_dropout_recorder():
        return _ToyRecorder()


class _ToySwapModel:
    """A minimal model with the call/state surface the swap law consumes."""

    def __init__(self):
        self.training = True
        self.state = {"w": np.ones((3, 3), dtype=np.float64), "b": np.arange(4, dtype=np.float64)}

    def load_state_dict(self, state, strict=True):
        assert strict is True
        assert set(state) == set(self.state), "strict-load key drift"
        self.state = {key: np.array(value, dtype=np.float64, copy=True) for key, value in state.items()}

    def to(self, device):
        return self

    def eval(self):
        self.training = False
        return self

    def state_dict(self):
        return {key: value for key, value in self.state.items()}

    def named_parameters(self):
        return iter(())

    def __call__(self, neural, *, calib_trials=None, side_features=None):
        return neural, None


class _ToyArmCommon:
    """Digest law mirroring tfpd_lane.arm_common over the toy state dicts."""

    @staticmethod
    def state_sha256(model) -> str:
        digest = hashlib.sha256()
        state = model.state_dict()
        for key in sorted(state):
            digest.update(key.encode("utf-8"))
            array = np.ascontiguousarray(state[key])
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(str(tuple(array.shape)).encode("utf-8"))
            digest.update(array.tobytes())
        return digest.hexdigest()


class _ToyExecutorState:
    def __init__(self, model):
        self.model = model


class _ToySwapRuntimeState:
    def __init__(self, model):
        self.modules = {
            "torch": torch, "pop_robust": _ToyPopRobust(), "arm_common": _ToyArmCommon(),
        }
        self.device = None
        self.model = model
        self.executor_state = _ToyExecutorState(model)


class _ToySwapRuntime:
    def __init__(self):
        self.state = _ToySwapRuntimeState(_ToySwapModel())

    def _require_state(self):
        return self.state


def _fresh_eval_model(state: Mapping[str, Any]) -> _ToySwapModel:
    model = _ToySwapModel()
    model.load_state_dict(state)
    model.eval()
    return model


def _write_toy_swa(path: Path, state: Mapping[str, Any]):
    torch.save({"state_dict": dict(state), "swa_manifest": {}}, path)


def test_swap_binds_both_seams_and_verifies_the_sealed_digest(tmp_path):
    runtime = _ToySwapRuntime()
    sealed = runtime.state.model
    sealed_digest = _ToyArmCommon.state_sha256(sealed)
    artifact = tmp_path / "swa_final4.pt"
    state = {"w": np.full((3, 3), 2.0), "b": np.arange(4, dtype=np.float64)}
    _write_toy_swa(artifact, state)
    expected_digest = _ToyArmCommon.state_sha256(_fresh_eval_model(state))
    assert expected_digest != sealed_digest
    binding = weights.swap_runtime_weights(
        runtime, swa_path=artifact,
        expected_artifact_sha256=weights.artifact_sha256(artifact),
        expected_state_sha256=expected_digest,
    )
    assert binding["strict_load"] is True
    assert binding["arm_state_sha256"] == expected_digest
    assert binding["sealed_state_sha256_before_swap"] == sealed_digest
    assert binding["sealed_model"] is sealed  # the retained sealed model object
    assert runtime.state.model is not sealed
    assert runtime.state.executor_state.model is runtime.state.model
    assert runtime.state.model.training is False
    assert binding["purity_proof"]["repeated_fixed_forward_bitwise_equal"] is True
    assert binding["purity_proof"]["model_state_unchanged"] is True
    restore = weights.restore_runtime_weights(
        runtime, sealed_model=binding["sealed_model"],
        expected_sealed_state_sha256=sealed_digest,
    )
    assert restore["sealed_model_restored"] is True
    assert runtime.state.model is sealed
    assert runtime.state.executor_state.model is sealed
    assert weights.verify_model_digest(
        runtime, expected=sealed_digest, label="toy",
    )["matches_expected"] is True


def test_swap_fails_closed_on_artifact_and_state_drift(tmp_path):
    runtime = _ToySwapRuntime()
    artifact = tmp_path / "swa_final4.pt"
    state = {"w": np.full((3, 3), 2.0), "b": np.arange(4, dtype=np.float64)}
    _write_toy_swa(artifact, state)
    with pytest.raises(weights.WeightSwapError):
        weights.swap_runtime_weights(
            runtime, swa_path=artifact,
            expected_artifact_sha256="0" * 64, expected_state_sha256="1" * 64,
        )
    with pytest.raises(weights.WeightSwapError):
        weights.swap_runtime_weights(
            runtime, swa_path=artifact,
            expected_artifact_sha256=weights.artifact_sha256(artifact),
            expected_state_sha256="1" * 64,
        )
    # Nothing was swapped by the failed attempts.
    assert runtime.state.model.training is True
    assert runtime.state.executor_state.model is runtime.state.model


def test_verify_model_digest_catches_seam_divergence():
    runtime = _ToySwapRuntime()
    digest = _ToyArmCommon.state_sha256(runtime.state.model)
    diverged = _ToySwapModel()
    diverged.load_state_dict({"w": np.zeros((3, 3)), "b": np.arange(4, dtype=np.float64)})
    runtime.state.executor_state.model = diverged
    with pytest.raises(weights.WeightSwapError):
        weights.verify_model_digest(runtime, expected=digest, label="diverged")
    runtime.state.executor_state.model = runtime.state.model
    with pytest.raises(weights.WeightSwapError):
        weights.verify_model_digest(runtime, expected="2" * 64, label="wrong")


# ---------------------------------------------------------------------------
# 4. The cell laws on a toy runtime (the frozen Stage-O/Stage-P modules).
# ---------------------------------------------------------------------------


def _channel_digest():
    return cdm_core.channel_order_digest(np.arange(N_UNITS, dtype=np.int64))


def _b3s(rng, trial_id, channel_digest):
    activity = (rng.random((100, N_UNITS), dtype=np.float32) * 3.0)
    return cdm_core.B3SInterpolatedSpikeCountTrial(
        activity=activity, session_id=SESSION_ID, trial_id=trial_id,
        channel_order_sha256=channel_digest,
    )


def _native(rng, trial_id, channel_digest, start, length):
    counts = rng.poisson(2.0, size=(length, N_UNITS)).astype(np.float64)
    return cdm_core.NativeRewardedTrialSpikeCounts(
        counts=counts, session_id=SESSION_ID, trial_id=trial_id,
        channel_order_sha256=channel_digest,
        rewarded_interval_start_bin=start, rewarded_interval_stop_bin=start + length,
    )


def _behavior_rows(windows: int, direction_index: int) -> np.ndarray:
    theta = CANON[int(direction_index)]
    physical = np.stack(
        [np.full(windows, 9.0 * math.cos(theta)), np.full(windows, 9.0 * math.sin(theta))],
        axis=1,
    )
    mean = np.asarray(v1plan.SEALED_BEHAVIOR_MEAN, dtype=np.float64)
    std = np.asarray(v1plan.SEALED_BEHAVIOR_STD, dtype=np.float64)
    return ((physical - mean[None, :]) / std[None, :]).astype(np.float32)


@dataclass
class ToyTrial:
    trial_id: str
    chronology_position: int
    b3s_activity: Any
    native_counts: Any
    velocity_validity: Any
    neural_windows: Any
    endpoint_bins: Any
    group_bias: tuple = DEFAULT_BIAS
    group_speed: tuple = DEFAULT_SPEED


@dataclass
class ToySession:
    surface: str
    session: str
    behavior: np.ndarray
    channel_ids: np.ndarray
    trials_by_id: Mapping[str, ToyTrial]
    query_trial_ids: Mapping[int, tuple]
    support_trial_ids: Mapping[int, tuple]
    support_rates: Mapping[str, np.ndarray]
    support_direction_indices: Mapping[str, int]


def _tuning_rates(direction_index: int, rng) -> np.ndarray:
    theta = CANON[int(direction_index)]
    preferred = np.asarray(CANON, dtype=np.float64)[np.arange(N_UNITS) % len(CANON)]
    gain = 40.0 + 8.0 * np.arange(N_UNITS) / N_UNITS
    return 45.0 + gain * np.cos(theta - preferred) + rng.normal(0.0, 1.0, size=N_UNITS)


def build_toy_session(seed: int = 7, *, n_query: int = 6, surface: str = "within") -> ToySession:
    rng = np.random.default_rng(seed)
    channel_digest = _channel_digest()
    channel_ids = np.arange(N_UNITS, dtype=np.int64)
    support_directions = [0, 2, 5, 0, 1, 3, 6, 4, 7, 2, 0, 5, 1, 4, 6, 3, 7, 0, 2, 5, 1, 3, 4, 6, 7, 0, 2, 5, 1, 3]
    query_directions = [1, 3, 6, 4, 7, 2, 1, 0][:n_query]
    trials: dict[str, ToyTrial] = {}
    behavior = np.zeros((40 * 60, 2), dtype=np.float32)
    support_rates: dict[str, np.ndarray] = {}
    support_direction_indices: dict[str, int] = {}
    position = 0

    def _make(trial_id, direction, windows, length):
        nonlocal position
        start = position * 40
        endpoints = np.arange(start, start + windows, dtype=np.int64)
        behavior[endpoints] = _behavior_rows(windows, direction)
        validity = cdm_core.VelocityValidityEvidence(
            valid_mask=np.ones(windows, dtype=np.bool_), session_id=SESSION_ID,
            trial_id=trial_id, prediction_interval_start_bin=int(endpoints[0]),
            prediction_interval_stop_bin=int(endpoints[-1]) + 1,
        )
        trial = ToyTrial(
            trial_id=trial_id, chronology_position=position,
            b3s_activity=_b3s(rng, trial_id, channel_digest),
            native_counts=_native(rng, trial_id, channel_digest, start, length),
            velocity_validity=validity,
            neural_windows=rng.random((windows, 50, N_UNITS), dtype=np.float32),
            endpoint_bins=endpoints,
        )
        position += 1
        return trial

    for index in range(30):
        trial_id = f"{SESSION_ID}:trial:{index}"
        trials[trial_id] = _make(trial_id, support_directions[index], 30, 40)
        support_rates[trial_id] = _tuning_rates(support_directions[index], rng)
        support_direction_indices[trial_id] = int(support_directions[index])
    for offset, direction in enumerate(query_directions):
        trial_id = f"{SESSION_ID}:trial:{30 + offset}"
        trials[trial_id] = _make(trial_id, direction, 25, 36)
    support_ids = {budget: tuple(f"{SESSION_ID}:trial:{index}" for index in range(budget))
                   for budget in (4, 10, 30)}
    query_ids = {budget: tuple(f"{SESSION_ID}:trial:{index}" for index in range(30, 30 + n_query))
                 for budget in (4, 10, 30)}
    return ToySession(
        surface=surface, session=SESSION_ID, behavior=behavior, channel_ids=channel_ids,
        trials_by_id=trials, query_trial_ids=query_ids, support_trial_ids=support_ids,
        support_rates=support_rates, support_direction_indices=support_direction_indices,
    )


@dataclass
class ToyTransition:
    trial_id: str
    activity_transition_committed: bool
    activity_fifo_changed: bool
    carrier_transition_committed: bool
    activity_rejection_reason: Optional[str]
    carrier_rejection_reason: Optional[str]
    state_before_sha256: str
    state_after_sha256: str
    activity_before_sha256: str
    activity_after_sha256: str
    carrier_before_sha256: str
    carrier_after_sha256: str


class ToyActivityOnlyMemory(cdm_core.IndependentActivityCausalDualMemory):
    def observe_completed_trial(self, *, b3s_trial_activity, carrier_trial_counts, complementary_predictions):
        del carrier_trial_counts, complementary_predictions
        reason = self.state.activity.validate_complete_trial(b3s_trial_activity)
        if reason is not None:
            return self._activity_invalid_pending(reason)
        candidate, changed = self._activity_candidate(b3s_trial_activity)
        return self._carrier_rejected_pending(
            activity_candidate=candidate, activity_fifo_changed=changed,
            reason=cdm_core.UpdateRejectionReason.INSUFFICIENT_EVIDENCE,
        )


class ToyRuntime:
    """The toy runtime surface the frozen rollouts consume, with a weight knob."""

    def __init__(self, weight_scale: float = 1.0) -> None:
        self._memory_mode = "cdm"
        self.weight_scale = float(weight_scale)
        self.forward_log: list[tuple[str, str, np.ndarray]] = []
        self.completed_trials = 0
        self._session: Optional[ToySession] = None

    def _make_memory(self, *, activity, carrier):
        if self._memory_mode == "activity_only":
            return ToyActivityOnlyMemory(activity=activity, carrier=carrier)
        return cdm_core.IndependentActivityCausalDualMemory(activity=activity, carrier=carrier)

    def _initial_memory(self, *, session, budget):
        support_ids = tuple(session.support_trial_ids[budget])
        rates = np.ascontiguousarray(
            np.stack([session.support_rates[item] for item in support_ids]), dtype=np.float64,
        )
        directions = np.ascontiguousarray(
            np.asarray([session.support_direction_indices[item] for item in support_ids], dtype=np.int64),
        )
        initial = cdm_core.fit_carriers_from_trial_table(
            rates, directions, mode=cdm_core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
        )
        config = cdm_core.CDMDConfig(
            support_budget_m=budget, active_fit_mode=cdm_core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
        )
        carrier = cdm_core.CarrierMemory.from_support_trials(
            initial_raw_t4=initial, channel_ids=session.channel_ids,
            support_trial_rates=rates, support_direction_indices=directions,
            config=config, valid_mask=np.ones(N_UNITS, dtype=np.bool_),
        )
        activity = cdm_core.ActivityMemory.initialize(
            [session.trials_by_id[item].b3s_activity for item in support_ids],
            channel_ids=session.channel_ids, fifo_capacity=int(config.activity_fifo_capacity),
        )
        memory = self._make_memory(activity=activity, carrier=carrier)
        evidence = {
            "budget": budget,
            "support_trial_ids": list(support_ids),
            "initial_carrier_sha256": cdm_core.array_digest(initial),
            "initial_activity_sha256": memory.state.activity.digest,
            "group_assignment_sha256": carrier.groups.digest,
        }
        return memory, evidence

    def _governing_forward(self, *, trial, inputs):
        side = np.asarray(inputs.active_t4, dtype=np.float64)
        stack = np.asarray(inputs.activity_trials, dtype=np.float64)
        neural = np.asarray(trial.neural_windows, dtype=np.float64)
        context = stack.mean(axis=(0, 1))
        drive = neural.mean(axis=1) + 0.1 * context[None, :]
        velocity = np.column_stack((drive @ side[:, 0], drive @ side[:, 1]))
        velocity = self.weight_scale * velocity
        self.forward_log.append((trial.trial_id, inputs.state_digest, velocity.copy()))
        return velocity

    def _truth_angle(self, session, trial) -> float:
        restored, _padded = p2filters.true_physical_velocity_one_trial(
            np.asarray(session.behavior[trial.endpoint_bins]),
            behavior_mean=v1plan.SEALED_BEHAVIOR_MEAN, behavior_std=v1plan.SEALED_BEHAVIOR_STD,
        )
        displacement = restored.sum(axis=0) * DT
        return float(math.atan2(displacement[1], displacement[0]))

    def _group_predictions_dispatch(self, *, trial, memory):
        del memory
        session = self._session
        truth = self._truth_angle(session, trial)
        windows = int(np.asarray(trial.endpoint_bins).size)
        views = []
        for group in range(cdm_core.GROUP_COUNT):
            theta = truth + float(trial.group_bias[group])
            speed = float(trial.group_speed[group]) * self.weight_scale
            velocity = np.column_stack((
                np.full(windows, speed * math.cos(theta)),
                np.full(windows, speed * math.sin(theta)),
            ))
            views.append(cdm_core.CompletedVelocityPrediction(velocity, trial.velocity_validity))
        evidence = [{"forward_chunk_count": 1, "held_group": group} for group in range(cdm_core.GROUP_COUNT)]
        return tuple(views), evidence

    def _commit_dispatch(self, *, memory, pending, trial_id):
        outcome = memory.commit_independent(pending)
        return ToyTransition(
            trial_id=trial_id,
            activity_transition_committed=outcome.activity_transition_committed,
            activity_fifo_changed=outcome.activity_fifo_changed,
            carrier_transition_committed=outcome.carrier_transition_committed,
            activity_rejection_reason=None if outcome.activity_rejection_reason is None
            else outcome.activity_rejection_reason.value,
            carrier_rejection_reason=None if outcome.carrier_rejection_reason is None
            else outcome.carrier_rejection_reason.value,
            state_before_sha256=outcome.state_before_sha256,
            state_after_sha256=outcome.state_after_sha256,
            activity_before_sha256=outcome.activity_before_sha256,
            activity_after_sha256=outcome.activity_after_sha256,
            carrier_before_sha256=outcome.carrier_before_sha256,
            carrier_after_sha256=outcome.carrier_after_sha256,
        )

    def _session_target_views(self, session, budget):
        targets, masks = [], []
        for trial_id in session.query_trial_ids[budget]:
            trial = session.trials_by_id[trial_id]
            targets.append(np.ascontiguousarray(session.behavior[trial.endpoint_bins], dtype=np.float32))
            masks.append(np.ones(trial.endpoint_bins.size, dtype=np.uint8))
        return targets, masks, 1.0

    def _matrix_r2(self, per_trial_prediction, targets, masks):
        joined = np.ascontiguousarray(np.concatenate([
            np.asarray(item, dtype=np.float64)[m.astype(bool)]
            for item, m in zip(per_trial_prediction, masks)
        ], axis=0))
        target = np.ascontiguousarray(np.concatenate([
            np.asarray(item, dtype=np.float64)[m.astype(bool)]
            for item, m in zip(targets, masks)
        ], axis=0))
        sse = float(np.sum((joined - target) ** 2))
        sst = float(np.sum((target - target.mean(axis=0)) ** 2)) or 1.0
        return 1.0 - sse / sst, joined

    def _house_raw_r2(self, per_trial_raw, targets, masks):
        value, _joined = self._matrix_r2(per_trial_raw, targets, masks)
        return float(value)

    @staticmethod
    def _raw_prediction_digest(per_trial_raw, masks):
        joined = np.ascontiguousarray(np.concatenate([
            np.asarray(item, dtype=np.float32)[m.astype(bool)]
            for item, m in zip(per_trial_raw, masks)
        ], axis=0))
        return hashlib.sha256(joined.tobytes()).hexdigest()

    def _bump_trial_counters(self):
        self.completed_trials += 1

    def _rollout_activity(self, *, session, budget):
        self._memory_mode = "activity_only"
        memory, initial = self._initial_memory(session=session, budget=budget)
        targets, masks, _sst = self._session_target_views(session, budget)
        per_trial_raw: list[np.ndarray] = []
        transitions: list[ToyTransition] = []
        for trial_id in session.query_trial_ids[budget]:
            trial = session.trials_by_id[trial_id]
            inputs = memory.read_prediction_inputs()
            per_trial_raw.append(self._governing_forward(trial=trial, inputs=inputs))
            pending = memory.observe_completed_trial(
                b3s_trial_activity=trial.b3s_activity, carrier_trial_counts=trial.native_counts,
                complementary_predictions=(None,) * cdm_core.GROUP_COUNT,
            )
            transitions.append(self._commit_dispatch(memory=memory, pending=pending, trial_id=trial_id))
            self._bump_trial_counters()
        assert all(not item.carrier_transition_committed for item in transitions)
        per_trial_filtered = [p2filters.apply_output_filter_one_trial(item) for item in per_trial_raw]
        raw_r2 = self._house_raw_r2(per_trial_raw, targets, masks)
        filtered_r2, _joined = self._matrix_r2(per_trial_filtered, targets, masks)
        rows = {
            "A0": {"house_raw_r2": raw_r2, "matrix_r2": raw_r2,
                   "prediction_sha256_raw": self._raw_prediction_digest(per_trial_raw, masks)},
            "A1": {"house_raw_r2": raw_r2, "matrix_r2": float(filtered_r2),
                   "prediction_sha256_raw": self._raw_prediction_digest(per_trial_raw, masks)},
        }
        return {
            "rows": rows, "transitions": transitions, "initial": initial,
            "per_trial_raw": per_trial_raw, "per_trial_filtered": per_trial_filtered,
            "leakage_flags": dict(plan.LEAKAGE_FLAGS_BY_CELL["F00"]),
            "per_trial_receipts": [
                {"trial_id": item.trial_id, "fs": None, "sb": item.state_before_sha256,
                 "sa": item.state_after_sha256} for item in transitions
            ],
        }


def _attach(runtime: ToyRuntime, session: ToySession) -> ToyRuntime:
    runtime._session = session
    runtime._memory_mode = "cdm"
    return runtime


def test_frozen_t4_cells_are_the_stage_o_o0_arm_verbatim():
    session = build_toy_session()
    direct = stage_o_replay.rollout_o0(_attach(ToyRuntime(), session), session=session, budget=4)
    mine = replay.run_frozen_t4_rollout(_attach(ToyRuntime(), session), session=session, budget=4)
    assert mine["row"] == "O0"
    assert mine["rows"]["O0"]["matrix_r2"] == direct["rows"]["O0"]["matrix_r2"]
    assert mine["rows"]["O0_raw"]["house_raw_r2"] == direct["rows"]["O0_raw"]["house_raw_r2"]
    for left, right in zip(mine["per_trial_raw"], direct["per_trial_raw"]):
        assert np.array_equal(np.asarray(left), np.asarray(right))


def test_p1_cells_are_the_stage_p_p1_rollout_verbatim_under_the_sealed_selection():
    session = build_toy_session()
    hp = replay.cell_hyperparameters(4)
    direct = stage_p_replay.rollout_p(
        _attach(ToyRuntime(), session), session=session, budget=4,
        spec=stage_p_replay.ROW_SPECS["P1"], hp=hp,
    )
    mine = replay.run_p1_rollout(_attach(ToyRuntime(), session), session=session, budget=4, hp=hp)
    assert mine["row"] == "P1"
    assert mine["hyperparameters"] == direct["hyperparameters"]
    assert mine["bank_digest_chain"] == direct["bank_digest_chain"]
    assert mine["committed_rows"] == direct["committed_rows"]
    for left, right in zip(mine["per_trial_raw"], direct["per_trial_raw"]):
        assert np.array_equal(np.asarray(left), np.asarray(right))
    assert stage_o_replay.verify_causality_receipts(mine["receipts"]) is True


def test_session_rows_carry_the_factorial_receipt_fields():
    session = build_toy_session()
    runtime = _attach(ToyRuntime(), session)
    targets, masks, _sst = runtime._session_target_views(session, 4)
    hp = replay.cell_hyperparameters(4)
    rollout_f00 = replay.run_frozen_t4_rollout(runtime, session=session, budget=4)
    f00 = replay.build_session_row(
        runtime, cell="F00", session=session, budget=4, targets=targets, masks=masks,
        rollout=rollout_f00, hp=None, model_state_sha256="sealed-digest",
    )
    rollout_f01 = replay.run_p1_rollout(runtime, session=session, budget=4, hp=hp)
    f01 = replay.build_session_row(
        runtime, cell="F01", session=session, budget=4, targets=targets, masks=masks,
        rollout=rollout_f01, hp=hp, model_state_sha256="sealed-digest",
    )
    c1_runtime = _attach(ToyRuntime(weight_scale=1.5), session)
    rollout_f11 = replay.run_p1_rollout(c1_runtime, session=session, budget=4, hp=hp)
    f11 = replay.build_session_row(
        c1_runtime, cell="F11", session=session, budget=4, targets=targets, masks=masks,
        rollout=rollout_f11, hp=hp, model_state_sha256="c1-digest",
    )
    for row, cell, arm in ((f00, "F00", "sealed"), (f01, "F01", "sealed"), (f11, "F11", "c1")):
        assert row["schema"] == "cdm_p1_cross_v1_row_session_v1"
        assert row["cell"] == cell
        assert row["weights"] == arm
        assert row["model_state_sha256"] == ("sealed-digest" if arm == "sealed" else "c1-digest")
        assert row["leakage_flags"] == dict(plan.LEAKAGE_FLAGS_BY_CELL[cell])
        assert row["causality_state_chain_verified"] is True
    # The sealed selection is recorded VERBATIM.  The toy geometry is far from
    # the DANDI calibration, so the sealed c_M honestly rejects every toy block
    # through the trust region: the row reports those rejections as nulls.
    assert f01["hyperparameters"] == hp.payload()
    assert f11["hyperparameters"] == hp.payload()
    assert f01["carrier_transitions_committed"] == 0
    assert f01["carrier_rejection_counts"]["support_trust_region_exceeded"] > 0
    assert "movement" in f01 and "movement" in f11
    assert "movement" not in f00
    # The weight factor alone moves the raw predictions (same carrier law).
    for left, right in zip(rollout_f11["per_trial_raw"], rollout_f01["per_trial_raw"]):
        if not np.array_equal(np.asarray(left), np.asarray(right)):
            break
    else:
        pytest.fail("the weight factor never moved the toy raw predictions")
    # The initial carrier is weight-invariant by construction.
    assert replay.anchor_initial_carrier(row=f11, baseline_row=f00, label="toy")["exact_match"] is True
    drifted = dict(f11)
    drifted["initial_carrier_sha256"] = "drift"
    assert replay.anchor_initial_carrier(row=drifted, baseline_row=f00, label="toy")[
        "exact_match"
    ] is False


def test_carrier_factor_moves_predictions_under_a_toy_calibrated_trust_region():
    """The carrier factor's movement path, exercised with a TOY-calibrated c_M.

    The sealed c_M belongs to the DANDI geometry; this toy diagnostic only
    proves the orchestration moves predictions when the gate commits.
    """
    from src.support_anchored_t4_stage_o_v1 import trust_region

    session = build_toy_session()
    runtime = _attach(ToyRuntime(), session)
    targets, masks, _sst = runtime._session_target_views(session, 4)
    sealed_hp = replay.cell_hyperparameters(4)
    unbounded = stage_p_replay.rollout_p(
        _attach(ToyRuntime(), session), session=session, budget=4,
        spec=stage_p_replay.ROW_SPECS["P1"],
        hp=stage_p_replay.CellHyperparameters(
            rho_M=sealed_hp.rho_M, alpha_M=sealed_hp.alpha_M, c_M=None,
            thresholds=sealed_hp.thresholds,
        ),
    )
    assert unbounded["committed_rows"] > 0
    toy_c_M = trust_region.calibrate_c_M(
        [float(item) for item in unbounded["proposed_d2_values"]],
    )
    toy_hp = stage_p_replay.CellHyperparameters(
        rho_M=sealed_hp.rho_M, alpha_M=sealed_hp.alpha_M, c_M=toy_c_M,
        thresholds=sealed_hp.thresholds,
    )
    rollout_f00 = replay.run_frozen_t4_rollout(runtime, session=session, budget=4)
    rollout_f01 = replay.run_p1_rollout(runtime, session=session, budget=4, hp=toy_hp)
    f01 = replay.build_session_row(
        runtime, cell="F01", session=session, budget=4, targets=targets, masks=masks,
        rollout=rollout_f01, hp=toy_hp, model_state_sha256="sealed-digest",
    )
    f11_runtime = _attach(ToyRuntime(weight_scale=1.5), session)
    rollout_f11 = replay.run_p1_rollout(f11_runtime, session=session, budget=4, hp=toy_hp)
    f11 = replay.build_session_row(
        f11_runtime, cell="F11", session=session, budget=4, targets=targets, masks=masks,
        rollout=rollout_f11, hp=toy_hp, model_state_sha256="c1-digest",
    )
    assert f01["carrier_transitions_committed"] > 0
    assert f01["movement"]["n_committed"] > 0
    assert f01["final_carrier_sha256"] != f01["initial_carrier_sha256"]
    for left, right in zip(rollout_f01["per_trial_raw"], rollout_f00["per_trial_raw"]):
        if not np.array_equal(np.asarray(left), np.asarray(right)):
            break
    else:
        pytest.fail("the P1 carrier factor never moved the toy raw predictions")
    # The full stack differs from both single-factor cells.
    for reference in (rollout_f01["per_trial_raw"], rollout_f00["per_trial_raw"]):
        for left, right in zip(rollout_f11["per_trial_raw"], reference):
            if not np.array_equal(np.asarray(left), np.asarray(right)):
                break
        else:
            pytest.fail("the full stack never diverged from a single-factor cell")
    assert replay.anchor_initial_carrier(row=f11, baseline_row=f01, label="toy")[
        "exact_match"
    ] is True


def test_m30_noop_law_against_the_same_weight_arm():
    session = build_toy_session(seed=11)
    f00_m30 = replay.run_frozen_t4_rollout(
        _attach(ToyRuntime(), session), session=session, budget=30,
    )
    f01_m30 = replay.run_p1_rollout(
        _attach(ToyRuntime(), session), session=session, budget=30,
        hp=replay.cell_hyperparameters(30),
    )
    replay.verify_m30_noop(
        rollout=f01_m30, reference_raw=f00_m30["per_trial_raw"], cell="F01", session=SESSION_ID,
    )
    c1_runtime = _attach(ToyRuntime(weight_scale=1.7), session)
    f10_m30 = replay.run_frozen_t4_rollout(c1_runtime, session=session, budget=30)
    f11_m30 = replay.run_p1_rollout(
        _attach(ToyRuntime(weight_scale=1.7), session), session=session, budget=30,
        hp=replay.cell_hyperparameters(30),
    )
    replay.verify_m30_noop(
        rollout=f11_m30, reference_raw=f10_m30["per_trial_raw"], cell="F11", session=SESSION_ID,
    )
    # The no-op is per arm: F11@M30 differs from F01@M30 under the weight factor.
    assert not np.array_equal(
        np.asarray(f11_m30["per_trial_raw"][0]), np.asarray(f01_m30["per_trial_raw"][0]),
    )
    mutated = [np.asarray(item).copy() for item in f10_m30["per_trial_raw"]]
    mutated[0][0, 0] += 1.0
    with pytest.raises(replay.CrossReplayError):
        replay.verify_m30_noop(
            rollout=f11_m30, reference_raw=mutated, cell="F11", session=SESSION_ID,
        )
    committed = [dict(item) for item in f11_m30["receipts"]]
    committed[0]["acc"] = True
    with pytest.raises(replay.CrossReplayError):
        replay.verify_m30_noop(
            rollout={"per_trial_raw": f11_m30["per_trial_raw"], "receipts": committed},
            reference_raw=f10_m30["per_trial_raw"], cell="F11", session=SESSION_ID,
        )


def test_sealed_stage_p_row_anchor_semantics():
    session = build_toy_session()
    runtime = _attach(ToyRuntime(), session)
    targets, masks, _sst = runtime._session_target_views(session, 4)
    hp = replay.cell_hyperparameters(4)
    row = replay.build_session_row(
        runtime, cell="F01", session=session, budget=4, targets=targets, masks=masks,
        rollout=replay.run_p1_rollout(runtime, session=session, budget=4, hp=hp),
        hp=hp, model_state_sha256="sealed-digest",
    )
    sealed = {field: row[field] for field in (
        "house_raw_r2", "prediction_sha256_raw", "matrix_r2",
        "filtered_prediction_sha256", "n_windows",
    )}
    anchor = replay.anchor_row_vs_sealed_stage_p(row=row, sealed_row=sealed, label="toy")
    assert anchor["exact_match"] is True
    for field in ("matrix_r2", "prediction_sha256_raw"):
        drifted = dict(sealed)
        drifted[field] = 0.5 if isinstance(sealed[field], float) else "drift"
        assert replay.anchor_row_vs_sealed_stage_p(
            row=row, sealed_row=drifted, label="toy",
        )["exact_match"] is False


def test_run_cross_replay_refuses_without_its_prerequisites(tmp_path):
    empty = tmp_path / "part-a"
    empty.mkdir()
    with pytest.raises(replay.CrossReplayError):
        replay.run_cross_replay(ROOT, gpu_index=1, output_root=empty)
    with pytest.raises(replay.CrossReplayError):
        replay.run_cross_replay(ROOT, gpu_index=0, output_root=empty)
    (empty / "attempt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(replay.CrossReplayError):
        replay.run_cross_replay(ROOT, gpu_index=1, output_root=empty)
    closed = tmp_path / "closed"
    closed.mkdir()
    (closed / "attempt.json").write_text("{}", encoding="utf-8")
    (closed / "replay.json").write_text("{}", encoding="utf-8")
    with pytest.raises(replay.CrossReplayError):
        replay.run_cross_replay(ROOT, gpu_index=1, output_root=closed)
    terminalized = tmp_path / "terminalized"
    terminalized.mkdir()
    (terminalized / "attempt.json").write_text("{}", encoding="utf-8")
    (terminalized / "terminal.json").write_text("{}", encoding="utf-8")
    with pytest.raises(replay.CrossReplayError):
        replay.run_cross_replay(ROOT, gpu_index=1, output_root=terminalized)


# ---------------------------------------------------------------------------
# 5. The work-order section 2 gates.
# ---------------------------------------------------------------------------


def _gate_view(
    *,
    f00: float = 0.40,
    f10_delta: float = 0.0, f01_delta: float = 0.0, f11_delta: float = 0.0,
    positive: int = 15, within_delta: float = 0.0, m30_delta: float = 0.0,
    n_external: int = 15, n_within: int = 6,
) -> dict:
    deltas = {"F00": 0.0, "F10": f10_delta, "F01": f01_delta, "F11": f11_delta}

    def external_sessions(delta: float) -> dict[str, float]:
        negative = -0.05
        if positive >= n_external or delta == 0.0:
            return {f"e{i}": delta for i in range(n_external)}
        bumped = (n_external * delta - (n_external - positive) * negative) / positive
        return {f"e{i}": (bumped if i < positive else negative) for i in range(n_external)}

    view: dict[str, Any] = {}
    for budget in plan.BUDGETS:
        external: dict[str, Any] = {}
        for cell, delta in deltas.items():
            if cell == "F00":
                values = {f"e{i}": 0.0 for i in range(n_external)}
            elif budget == 30:
                values = {f"e{i}": m30_delta for i in range(n_external)}
            elif budget == plan.DRIVING_BUDGET:
                values = external_sessions(delta)
            else:
                values = {f"e{i}": delta for i in range(n_external)}
            external[cell] = {
                "mean_r2": f00 + sum(values.values()) / n_external,
                "per_session": {key: f00 + value for key, value in values.items()},
            }
        within: dict[str, Any] = {}
        for cell in deltas:
            bump = 0.0 if cell == "F00" else within_delta
            within[cell] = {
                "mean_r2": f00 + bump,
                "per_session": {f"w{i}": f00 + bump for i in range(n_within)},
            }
        view[f"m{budget}"] = {"external": external, "within": within}
    return view


_SAFETY = {key: True for key in gates.SAFETY_CONDITIONS}


def test_margin_boundary_is_exact_and_epsilon_is_only_a_band():
    verdict = gates.margin_verdict(0.01, 0.01)
    assert verdict["meets_margin"] is True
    assert verdict["within_epsilon_band_of_boundary"] is False
    below = gates.margin_verdict(0.01 - 1.0e-13, 0.01)
    assert below["meets_margin"] is False
    assert below["within_epsilon_band_of_boundary"] is True
    assert below["boundary_epsilon"] == 1.0e-12
    assert gates.margin_verdict(0.001, 0.01)["within_epsilon_band_of_boundary"] is False


def test_per_cell_promotion_gate_requires_delta_and_breadth():
    view = _gate_view(f10_delta=0.05, f01_delta=0.05, f11_delta=0.05, positive=12)
    gate = gates.evaluate_cross_gate(view, safety=_SAFETY)
    for cell in plan.CANDIDATE_CELLS:
        assert gate["per_cell"][cell]["promoted"] is True
        assert gate["per_cell"][cell]["disposition"] == plan.DISPOSITION_CELL_PROMOTED
        assert gate["per_cell"][cell]["external_m4_margin"]["meets_margin"] is True
    narrow = _gate_view(f10_delta=0.05, f01_delta=0.05, f11_delta=0.05, positive=9)
    gate = gates.evaluate_cross_gate(narrow, safety=_SAFETY)
    for cell in plan.CANDIDATE_CELLS:
        assert gate["per_cell"][cell]["promoted"] is False
        assert gate["per_cell"][cell]["breadth_margin"]["meets_margin"] is False
    small = _gate_view(f10_delta=0.0099, f01_delta=0.0099, f11_delta=0.0099, positive=15)
    gate = gates.evaluate_cross_gate(small, safety=_SAFETY)
    for cell in plan.CANDIDATE_CELLS:
        assert gate["per_cell"][cell]["promoted"] is False
        assert gate["per_cell"][cell]["external_m4_margin"]["meets_margin"] is False


def test_safety_floors_block_promotion():
    view = _gate_view(f10_delta=0.05, f01_delta=0.05, f11_delta=0.05,
                      within_delta=-0.03, positive=15)
    gate = gates.evaluate_cross_gate(view, safety=_SAFETY)
    for cell in plan.CANDIDATE_CELLS:
        assert gate["per_cell"][cell]["within_bounds"]["all_meet_margin"] is False
        assert gate["per_cell"][cell]["promoted"] is False
    m30 = _gate_view(f10_delta=0.05, f01_delta=0.05, f11_delta=0.05,
                     m30_delta=-0.021, positive=15)
    gate = gates.evaluate_cross_gate(m30, safety=_SAFETY)
    for cell in plan.CANDIDATE_CELLS:
        assert gate["per_cell"][cell]["external_m30_safety"]["meets_margin"] is False
        assert gate["per_cell"][cell]["promoted"] is False
    edge = _gate_view(f10_delta=0.05, f01_delta=0.05, f11_delta=0.05,
                      within_delta=-0.0199, m30_delta=-0.0199, positive=15)
    gate = gates.evaluate_cross_gate(edge, safety=_SAFETY)
    assert all(gate["per_cell"][cell]["promoted"] for cell in plan.CANDIDATE_CELLS)
    unsafe = dict(_SAFETY)
    unsafe["weight_swap_binding_verified"] = False
    gate = gates.evaluate_cross_gate(edge, safety=unsafe)
    assert gate["safety_all_pass"] is False
    with pytest.raises(gates.CrossGateError):
        gates.evaluate_cross_gate(edge, safety={"unexpected": True})


def test_additivity_gate_binds_external_m4_exactly():
    held = _gate_view(f10_delta=0.02, f01_delta=0.03, f11_delta=0.035, positive=15)
    gate = gates.evaluate_additivity_gate(held)
    assert gate["held"] is True
    assert gate["reference_max_f10_f01"] == pytest.approx(0.43)
    assert gate["disposition"] == plan.DISPOSITION_ADDITIVITY_HELD
    broken = _gate_view(f10_delta=0.02, f01_delta=0.03, f11_delta=0.025, positive=15)
    gate = gates.evaluate_additivity_gate(broken)
    assert gate["held"] is False
    assert gate["disposition"] == plan.DISPOSITION_ADDITIVITY_VIOLATED
    tied = _gate_view(f10_delta=0.02, f01_delta=0.03, f11_delta=0.03, positive=15)
    assert gates.evaluate_additivity_gate(tied)["held"] is True  # exact >= boundary
    inside_band = _gate_view(f10_delta=0.02, f01_delta=0.03, f11_delta=0.03 - 1.0e-13, positive=15)
    verdict = gates.evaluate_additivity_gate(inside_band)["margin"]
    assert verdict["meets_margin"] is False
    assert verdict["within_epsilon_band_of_boundary"] is True


def test_stop_conditions_and_paired_contrasts():
    view = _gate_view(f10_delta=0.05, f01_delta=0.05, f11_delta=0.045, positive=9)
    gate = gates.evaluate_cross_gate(view, safety=_SAFETY)
    stops = gates.stop_conditions(gate)
    assert stops["conditions"]["no_candidate_beats_f00"]["fired"] is True
    assert stops["conditions"]["additivity_violated"]["fired"] is True
    assert stops["any_fired"] is True
    contrasts = gates.paired_contrast_tables(view)
    assert set(contrasts) == {
        f"{cell}_minus_F00_m{budget}_{surface}"
        for cell in plan.CANDIDATE_CELLS for budget in plan.BUDGETS
        for surface in plan.SURFACES
    }
    entry = contrasts["F11_minus_F00_m4_external"]
    assert entry["n_sessions"] == 15
    assert entry["equal_session_mean_delta"] == pytest.approx(0.045)
    roster_drift = _gate_view()
    del roster_drift["m4"]["external"]["F11"]["per_session"]["e0"]
    with pytest.raises(gates.CrossGateError):
        gates.paired_contrast_tables(roster_drift)


# ---------------------------------------------------------------------------
# 6. The terminal builder on a synthetic replay payload.
# ---------------------------------------------------------------------------


def test_terminal_builder_end_to_end_on_a_synthetic_matrix():
    within_roster = [f"w{i}" for i in range(6)]
    external_roster = [f"e{i}" for i in range(15)]
    matrix: dict[str, Any] = {}
    for budget in plan.BUDGETS:
        matrix[f"m{budget}"] = {}
        for surface, roster in (("within", within_roster), ("external", external_roster)):
            cells = {}
            for cell, delta in (("F00", 0.0), ("F10", 0.04 if budget == 4 else -0.002),
                                ("F01", 0.03 if budget == 4 else -0.001),
                                ("F11", 0.05 if budget == 4 else -0.0015)):
                per_session = {}
                for index, session in enumerate(roster):
                    bump = delta
                    if surface == "external" and budget == 4 and cell != "F00":
                        bump = delta + 0.01 if index < 11 else -0.05
                    per_session[session] = 0.40 + bump
                cells[cell] = {"mean_r2": sum(per_session.values()) / len(roster),
                               "per_session": per_session}
            matrix[f"m{budget}"][surface] = cells
    replay_payload = {
        "matrix": matrix,
        "target_optimizer_backward_update": 0,
        "target_backward_calls": 0,
        "target_parameter_update_calls": 0,
        "normalizer_update_calls": 0,
        "decoder_training": False,
        "model_or_checkpoint_updated": False,
        "causality_state_chains_all_rows": True,
        "trust_region_no_committed_drift": True,
        "hyperparameter_binding": {
            "values_match_plan": True,
            "m30": {"matches_pin": True},
        },
        "anchors": {
            "f00_vs_sealed_activity_only_all_exact": True,
            "f00_vs_sealed_stage_p_all_exact": True,
            "f01_vs_sealed_stage_p_all_exact": True,
            "initial_carrier_invariance_all_exact": True,
            "m30_noop_all_cells": True,
            "stage_p_go": {"decision": plan.STAGE_P_GO_ANCHOR["decision"]},
            "c1_weight_binding": {"verified": True},
        },
        "weight_swap": {
            "c1_swap": {
                "strict_load": True, "arm_state_sha256": plan.C1_ARM_STATE_SHA256,
                "artifact_sha256": plan.C1_SWA_SHA256,
            },
            "sealed_restore": None,
        },
        "model_state_digests": {"sealed_model_restored_and_verified": True},
    }
    body = replay.build_terminal(replay_payload=replay_payload)
    gate = body["gate"]
    assert gate["safety_all_pass"] is True
    for cell in plan.CANDIDATE_CELLS:
        assert gate["per_cell"][cell]["promoted"] is True
    assert gate["additivity"]["held"] is True
    assert body["stop_conditions"]["any_fired"] is False
    assert body["paired_contrasts"]["F10_minus_F00_m4_external"]["n_total"] == 15
    assert body["paired_contrasts"]["F10_minus_F00_m4_external"]["n_positive"] == 11
    assert body["equal_session_means"]["m4"]["external"]["F00"] == pytest.approx(0.40)
    # A failed anchor must flip the safety block, not the arithmetic.
    drifted = json.loads(json.dumps(replay_payload))
    drifted["anchors"]["f01_vs_sealed_stage_p_all_exact"] = False
    body = replay.build_terminal(replay_payload=drifted)
    assert body["gate"]["safety_all_pass"] is False
    assert body["gate"]["safety"]["f01_bit_anchor"] is False
    with pytest.raises(gates.CrossGateError):
        replay.build_terminal(replay_payload=dict(replay_payload, matrix={"m4": matrix["m4"]}))
