"""Focused no-data/no-CUDA gates for CDM-D independent-activity Source V3."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import pytest
import torch


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "tfpd_exploration", ROOT / "tfpd_exploration/src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from src.causal_dual_memory_cell_d_v1 import core  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_execute as v1  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_execute_physical as v1_physical  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_execute_physical_v2 as v2_physical  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_execute_physical_v3 as physical_v3  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_execute_v2 as v2  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_execute_v3 as v3  # noqa: E402


SESSION = "independent-activity-synthetic-session"


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _table(*, units: int = 8) -> dict[str, np.ndarray]:
    directions = np.asarray((0, 2, 4, 3, 3, 3, 5, 7, 1, 6, 3), dtype=np.int64)
    theta = np.asarray([core.CANONICAL_DIRECTIONS_RAD[index] for index in directions], dtype=np.float64)
    a = np.linspace(0.7, 1.4, units, dtype=np.float64)
    c = np.linspace(-0.8, 0.5, units, dtype=np.float64)
    b = np.linspace(4.0, 5.0, units, dtype=np.float64)
    rates = b[None, :] + np.cos(theta)[:, None] * a[None, :] + np.sin(theta)[:, None] * c[None, :]
    channels = np.arange(1000, 1000 + units, dtype=np.int64)
    return {
        "directions": directions,
        "rates": rates,
        "raw_t4": np.column_stack((a, c, np.hypot(a, c), b)).astype(np.float64),
        "channels": channels,
        "channel_digest": np.asarray([core.channel_order_digest(channels)]),
    }


def _digest(table: Mapping[str, np.ndarray]) -> str:
    return str(table["channel_digest"][0])


def _config(budget: int) -> core.CDMDConfig:
    return core.CDMDConfig(
        support_budget_m=budget,
        dt=1.0,
        minimum_movement_bins=2,
        minimum_displacement=0.01,
        minimum_mean_speed=0.01,
        max_canonical_distance_rad=math.pi / 8.0,
        max_group_direction_disagreement_rad=math.pi / 8.0,
        minimum_accepted_evidence=3,
        active_fit_mode=core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL,
    )


def _b3s(table: Mapping[str, np.ndarray], rates: np.ndarray, trial_id: str, *, session: str = SESSION,
          channel_digest: str | None = None) -> core.B3SInterpolatedSpikeCountTrial:
    return core.B3SInterpolatedSpikeCountTrial(
        np.repeat(np.asarray(rates, dtype=np.float64)[None, :], 100, axis=0),
        session,
        trial_id,
        _digest(table) if channel_digest is None else channel_digest,
    )


def _native(table: Mapping[str, np.ndarray], rates: np.ndarray, trial_id: str, *, session: str = SESSION,
            channel_digest: str | None = None) -> core.NativeRewardedTrialSpikeCounts:
    return core.NativeRewardedTrialSpikeCounts(
        np.repeat((np.asarray(rates, dtype=np.float64) * 0.020)[None, :], 7, axis=0),
        session,
        trial_id,
        _digest(table) if channel_digest is None else channel_digest,
        10,
        17,
    )


def _validity(trial_id: str, *, mask: np.ndarray | None = None, session: str = SESSION) -> core.VelocityValidityEvidence:
    return core.VelocityValidityEvidence(
        np.ones(6, dtype=np.bool_) if mask is None else np.asarray(mask, dtype=np.bool_),
        session,
        trial_id,
        20,
        26,
    )


def _predictions(trial_id: str, *, direction: int = 3, mask: np.ndarray | None = None,
                 session: str = SESSION) -> tuple[core.CompletedVelocityPrediction, ...]:
    validity = _validity(trial_id, mask=mask, session=session)
    theta = core.CANONICAL_DIRECTIONS_RAD[direction]
    velocity = np.repeat([[math.cos(theta), math.sin(theta)]], validity.valid_mask.size, axis=0)
    return tuple(core.CompletedVelocityPrediction(velocity, validity) for _ in range(core.GROUP_COUNT))


def _memory(budget: int) -> tuple[core.IndependentActivityCausalDualMemory, dict[str, np.ndarray]]:
    table = _table()
    config = _config(budget)
    support_indices = np.resize(np.arange(table["rates"].shape[0], dtype=np.int64), budget)
    carrier = core.CarrierMemory.from_support_trials(
        initial_raw_t4=table["raw_t4"],
        channel_ids=table["channels"],
        support_trial_rates=table["rates"][support_indices],
        support_direction_indices=table["directions"][support_indices],
        config=config,
    )
    activity = core.ActivityMemory.initialize(
        tuple(
            _b3s(table, table["rates"][index], f"support-{position}")
            for position, index in enumerate(support_indices)
        ),
        channel_ids=table["channels"],
        fifo_capacity=config.activity_fifo_capacity,
    )
    return core.IndependentActivityCausalDualMemory(activity=activity, carrier=carrier), table


def _observe(
    memory: core.IndependentActivityCausalDualMemory,
    table: Mapping[str, np.ndarray],
    trial_id: str,
    *,
    b3s: object | None = None,
    native: object | None = None,
    predictions: object | None = None,
) -> core.IndependentActivityPendingTrialUpdate:
    rates = table["rates"][3]
    return memory.observe_completed_trial(
        b3s_trial_activity=_b3s(table, rates, trial_id) if b3s is None else b3s,
        carrier_trial_counts=_native(table, rates, trial_id) if native is None else native,
        complementary_predictions=_predictions(trial_id) if predictions is None else predictions,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("budget,capacity", ((4, 26), (10, 20)))
def test_rejected_carrier_advances_only_valid_b3s_activity_fifo(
    budget: int, capacity: int,
) -> None:
    memory, table = _memory(budget)
    before_carrier = memory.state.carrier.digest
    before_state = memory.state.digest
    pending = _observe(
        memory,
        table,
        "reject-low-motion",
        predictions=_predictions("reject-low-motion", mask=np.zeros(6, dtype=np.bool_)),
    )
    assert pending.activity_transition_ready is True
    assert pending.carrier_transition_accepted is False
    assert pending.carrier_rejection_reason is core.UpdateRejectionReason.MOVEMENT_TOO_SHORT
    outcome = memory.commit_independent(pending)
    assert outcome.activity_transition_committed is True
    assert outcome.activity_fifo_changed is True
    assert outcome.carrier_transition_committed is False
    assert outcome.carrier_before_sha256 == outcome.carrier_after_sha256 == before_carrier
    assert outcome.state_before_sha256 == before_state != outcome.state_after_sha256
    assert memory.state.activity.query_count == 1
    assert memory.state.activity.fifo_capacity == capacity
    core.validate_independent_activity_outcome_payload(outcome.payload())


def test_accepted_carrier_advances_both_memories() -> None:
    memory, table = _memory(4)
    pending = _observe(memory, table, "accepted")
    assert pending.activity_transition_ready is True
    assert pending.carrier_transition_accepted is True
    outcome = memory.commit_independent(pending)
    assert outcome.activity_transition_committed is True
    assert outcome.activity_fifo_changed is True
    assert outcome.carrier_transition_committed is True
    assert outcome.carrier_rejection_reason is None
    assert outcome.carrier_before_sha256 != outcome.carrier_after_sha256
    assert memory.state.activity.query_count == 1


def test_invalid_b3s_has_no_transition_but_bad_carrier_view_cannot_suppress_valid_activity() -> None:
    memory, table = _memory(4)
    before = memory.state.digest
    invalid = _observe(memory, table, "invalid-b3s", b3s=object())
    invalid_outcome = memory.commit_independent(invalid)
    assert invalid_outcome.activity_transition_committed is False
    assert invalid_outcome.activity_rejection_reason is core.UpdateRejectionReason.TRIAL_CAPABILITY
    assert invalid_outcome.state_before_sha256 == invalid_outcome.state_after_sha256 == before
    valid_b3s = _b3s(table, table["rates"][3], "bad-native")
    pending = _observe(memory, table, "bad-native", b3s=valid_b3s, native=object())
    outcome = memory.commit_independent(pending)
    assert outcome.activity_transition_committed is True
    assert outcome.carrier_transition_committed is False
    assert outcome.carrier_rejection_reason is core.UpdateRejectionReason.TRIAL_CAPABILITY
    assert outcome.carrier_before_sha256 == outcome.carrier_after_sha256
    assert memory.state.activity.query_count == 1
    wrong_session = _b3s(table, table["rates"][3], "wrong-session", session="other")
    no_transition = _observe(memory, table, "wrong-session", b3s=wrong_session)
    assert memory.commit_independent(no_transition).activity_transition_committed is False


def test_stale_pending_fails_closed_and_current_trial_uses_pretransition_state() -> None:
    memory, table = _memory(4)
    pre = memory.read_prediction_inputs()
    first = _observe(memory, table, "first")
    assert first.fallback.state_digest == pre.state_digest
    second = _observe(memory, table, "second")
    memory.commit_independent(second)
    with pytest.raises(core.CDMDStage0Error, match="stale_pending_update"):
        memory.commit_independent(first)
    assert memory.read_prediction_inputs().state_digest != pre.state_digest
    assert first.fallback.activity_trials.shape[0] == 4
    assert memory.read_prediction_inputs().activity_trials.shape[0] == 5


def test_m30_has_valid_transition_count_but_literal_empty_activity_fifo() -> None:
    memory, table = _memory(30)
    before_activity = memory.state.activity.digest
    before_carrier = memory.state.carrier.digest
    pending = _observe(
        memory, table, "m30-rejected",
        predictions=_predictions("m30-rejected", mask=np.zeros(6, dtype=np.bool_)),
    )
    outcome = memory.commit_independent(pending)
    assert outcome.activity_transition_committed is True
    assert outcome.activity_fifo_changed is False
    assert outcome.activity_before_sha256 == outcome.activity_after_sha256 == before_activity
    assert outcome.carrier_before_sha256 == outcome.carrier_after_sha256 == before_carrier
    assert memory.state.activity.fifo_capacity == 0
    assert memory.state.activity.query_count == 0
    assert memory.state.committed_query_trials == 1


@pytest.mark.parametrize("budget,capacity,steps", ((4, 26, 27), (10, 20, 21), (30, 0, 3)))
def test_exact_fifo_capacities_and_eviction(budget: int, capacity: int, steps: int) -> None:
    memory, table = _memory(budget)
    for position in range(steps):
        trial_id = f"rejected-{position}"
        outcome = memory.commit_independent(_observe(
            memory,
            table,
            trial_id,
            predictions=_predictions(trial_id, mask=np.zeros(6, dtype=np.bool_)),
        ))
        assert outcome.activity_transition_committed is True
        assert outcome.carrier_transition_committed is False
    assert memory.state.activity.fifo_capacity == capacity
    assert memory.state.activity.query_count == capacity
    assert memory.state.committed_query_trials == steps
    if capacity:
        assert memory.state.activity.query_trials[0].trial_id == f"rejected-{steps - capacity}"


def test_host_python_numpy_torch_rng_states_are_unchanged() -> None:
    memory, table = _memory(4)
    random.seed(77)
    np.random.seed(77)
    torch.manual_seed(77)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    pending = _observe(
        memory, table, "rng-rejected",
        predictions=_predictions("rng-rejected", mask=np.zeros(6, dtype=np.bool_)),
    )
    memory.commit_independent(pending)
    assert random.getstate() == python_before
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_before[0]
    assert np.array_equal(after_numpy[1], numpy_before[1]) and after_numpy[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert torch.cuda.is_initialized() is False


def test_frozen_v2_machine_retains_accepted_only_behavior_and_v3_labels_it_superseded() -> None:
    table = _table()
    config = _config(4)
    support_indices = np.arange(4, dtype=np.int64)
    carrier = core.CarrierMemory.from_support_trials(
        initial_raw_t4=table["raw_t4"], channel_ids=table["channels"],
        support_trial_rates=table["rates"][support_indices],
        support_direction_indices=table["directions"][support_indices], config=config,
    )
    activity = core.ActivityMemory.initialize(
        tuple(_b3s(table, table["rates"][index], f"old-support-{index}") for index in support_indices),
        channel_ids=table["channels"], fifo_capacity=config.activity_fifo_capacity,
    )
    old = core.CausalDualMemory(activity=activity, carrier=carrier)
    before_activity = old.state.activity.digest
    pending = old.observe_completed_trial(
        b3s_trial_activity=_b3s(table, table["rates"][3], "old-reject"),
        carrier_trial_counts=_native(table, table["rates"][3], "old-reject"),
        complementary_predictions=_predictions("old-reject", mask=np.zeros(6, dtype=np.bool_)),
    )
    outcome = old.commit(pending)
    assert outcome.committed is False
    assert old.state.activity.digest == before_activity
    assert v3.V2_SUPERSEDED_EVIDENCE["status"] == "SUPERSEDED_NOT_REWRITTEN"
    assert v3.V2_SUPERSEDED_EVIDENCE["reason"] == "carrier_rejection_suppressed_valid_unlabeled_b3s_activity_transition"


def test_receipt_validator_rejects_collapsed_transition_booleans_and_digest_forgery() -> None:
    memory, table = _memory(4)
    rejected = memory.commit_independent(_observe(
        memory, table, "receipt-reject",
        predictions=_predictions("receipt-reject", mask=np.zeros(6, dtype=np.bool_)),
    ))
    payload = rejected.payload()
    core.validate_independent_activity_outcome_payload(payload)
    collapsed = dict(payload)
    collapsed["carrier_transition_committed"] = collapsed["activity_transition_committed"]
    with pytest.raises(core.CDMDStage0Error, match="carrier"):
        core.validate_independent_activity_outcome_payload(collapsed)
    forged = dict(payload)
    forged["activity_fifo_changed"] = False
    with pytest.raises(core.CDMDStage0Error, match="FIFO"):
        core.validate_independent_activity_outcome_payload(forged)


def _committing_trace_pool(*, budget: int, count: int, prefix: str) -> tuple[list[str], list[dict[str, object]]]:
    ids = [f"{prefix}-{index}" for index in range(count)]
    rows: list[dict[str, object]] = []
    for index, trial_id in enumerate(ids):
        # Adjacent state/activity/carrier digests are deliberately chained;
        # carrier remains exact on all rejected rows while activity advances.
        rows.append(_committing_trace(
            budget=budget,
            trial_id=trial_id,
            state_before=f"{index % 10}", state_after=f"{(index + 1) % 10}",
            activity_before=f"{(index + 2) % 10}", activity_after=f"{(index + 3) % 10}",
            carrier_before="a", carrier_after="a",
        ))
    return ids, rows


@pytest.mark.parametrize("budget,count", ((4, 26), (10, 20)))
def test_full_gate_trace_exact_cardinality_order_and_digest_chain(
    budget: int, count: int,
) -> None:
    ids, trace = _committing_trace_pool(budget=budget, count=count, prefix=f"m{budget}")
    v3._validate_transition_trace(
        trace, budget=budget, expected_trial_ids=ids, require_offline_m30=False,
    )
    with pytest.raises(v3.SourceExecutionV3Error, match="length"):
        v3._validate_transition_trace(
            trace[:-1], budget=budget, expected_trial_ids=ids, require_offline_m30=False,
        )
    reordered = [dict(row) for row in trace]
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(v3.SourceExecutionV3Error, match="trial-id/order"):
        v3._validate_transition_trace(
            reordered, budget=budget, expected_trial_ids=ids, require_offline_m30=False,
        )
    chain_break = [dict(row) for row in trace]
    chain_break[1]["state_before_sha256"] = "f" * 64
    with pytest.raises(v3.SourceExecutionV3Error, match="digest chain"):
        v3._validate_transition_trace(
            chain_break, budget=budget, expected_trial_ids=ids, require_offline_m30=False,
        )


def test_full_m30_trace_binds_all_thirty_offline_fixed_pool_ids() -> None:
    ids = [f"m30-{index}" for index in range(30)]
    trace = [_offline_m30_trace(trial_id) for trial_id in ids]
    v3._validate_transition_trace(
        trace, budget=30, expected_trial_ids=ids, require_offline_m30=True,
    )
    duplicate = [dict(row) for row in trace]
    duplicate[-1]["trial_id"] = ids[0]
    with pytest.raises(v3.SourceExecutionV3Error, match="trial-id/order"):
        v3._validate_transition_trace(
            duplicate, budget=30, expected_trial_ids=ids, require_offline_m30=True,
        )


def test_v3_smoke_is_literal_m10_positive_capacity_not_v1_m30_offline_pool() -> None:
    assert v3.SOURCE_SMOKE_SPEC.smoke_budget == 10
    assert v3.SOURCE_SMOKE_SPEC.smoke_audit_positions == (10, 11)
    assert v3.SOURCE_SMOKE_SPEC.payload()["smoke"] == {
        "session": v1.SOURCE_SMOKE_SESSION,
        "budget": 10,
        "support_positions": list(range(10)),
        "audit_positions": [10, 11],
        "group_count": 4,
        "required_activity_transition_count": 2,
        "max_endpoints_per_forward_chunk": 128,
    }
    with pytest.raises(v3.SourceExecutionV3Error, match="M10"):
        v3.SourceExecutionV3Spec(
            "source_smoke", "tfpd_exploration/results/forged_m30", v1.SOURCE_SMOKE_SESSION, 30, (30, 31),
        )


def _profile() -> dict[str, object]:
    return dict(v1.COMPATIBLE_DEVICE_PROFILES["gpu1"])


def _roster() -> tuple[str, ...]:
    return (v1.SOURCE_SMOKE_SESSION, *(f"sub-C_ses-CO-{index:08d}" for index in range(26)))


def _synthetic_closure() -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_closure_v3",
        "inherited_explicit_path_count": 0,
        "v2_superseded_evidence": dict(v3.V2_SUPERSEDED_EVIDENCE),
        "paths": [],
        "closure_sha256": "d" * 64,
    }


def _synthetic_identity(spec: v3.SourceExecutionV3Spec) -> v3.SourceExecutionV3Identity:
    closure = _synthetic_closure()
    return v3.SourceExecutionV3Identity(
        spec=spec,
        closure=closure,
        strict_train_roster=_roster(),
        fixed_assets={asset.label: asset.payload() for asset in v1.FIXED_ASSETS},
        normalizers=v1.SEALED_NORMALIZERS,
        selected_device=_profile(),
    )


def _runtime_attestation() -> dict[str, object]:
    return {
        **_profile(),
        "visible_devices": 1,
        "attested": True,
        "torch_cuda_matmul_allow_tf32": False,
        "torch_cudnn_allow_tf32": False,
    }


def _sealed_swa_proof() -> dict[str, object]:
    return {
        "schema": v1.SEALED_SWA_LOAD_PROOF_SCHEMA,
        "sealed_terminal_sha256": v1.SEALED_CELL_D_TERMINAL_SHA256,
        "sealed_swa_sha256": v1.SEALED_CELL_D_SWA_SHA256,
        "fresh_strict_load": True,
        "recomputed_state_dict_sha256": "c" * 64,
        "initialized_trainable_parameters": v1.SEALED_CELL_D_INITIALIZED_TRAINABLE_PARAMETERS,
        "uninitialized_lazy_keys": list(v1.SEALED_CELL_D_UNINITIALIZED_LAZY_KEYS),
        "model_eval": True,
        "no_grad": True,
        "finite_forward": True,
        "repeated_fixed_forward_bitwise_equal": True,
        "model_state_unchanged": True,
        "dynamic_dropout_calls": 0,
    }


def _theta_proof() -> dict[str, object]:
    raw = np.asarray([[0.5 + index, -0.25 - index, 0.1 + index * 0.01, 2.0 + index]
                      for index in range(7)], dtype=np.float32)
    theta = np.arctan2(raw[:, 1].astype(np.float64), raw[:, 0].astype(np.float64)).astype(np.float64)
    valid = (raw[:, 2].astype(np.float64) > 1.0e-6).astype(np.bool_)
    _mask, proof = v2.validate_theta_raw_t4_authority(
        raw,
        sealed_raw_t4_sha256=v2.theta_raw_t4_float32_bytes_sha256(raw),
        sealed_theta_float64=theta,
        sealed_valid_mask=valid,
        n_units=raw.shape[0],
        channel_ids=np.arange(raw.shape[0], dtype=np.int64),
        session=v1.SOURCE_SMOKE_SESSION,
        modulation_eps=1.0e-6,
    )
    return proof


def _offline_m30_trace(trial_id: str, *, state: str = "1", activity: str = "2", carrier: str = "3") -> dict[str, object]:
    return {
        "trial_id": trial_id,
        "schema": "causal_dual_memory_independent_activity_offline_m30_v3",
        "budget": 30,
        "source_audit_offline_no_deployment_commit": True,
        "activity_fifo_capacity": 0,
        "activity_query_count_before": 0,
        "activity_query_count_after": 0,
        "state_before_sha256": state * 64,
        "state_after_sha256": state * 64,
        "activity_before_sha256": activity * 64,
        "activity_after_sha256": activity * 64,
        "carrier_before_sha256": carrier * 64,
        "carrier_after_sha256": carrier * 64,
    }


def _committing_trace(
    *, budget: int, trial_id: str, state_before: str, state_after: str,
    activity_before: str, activity_after: str, carrier_before: str,
    carrier_after: str, carrier_committed: bool = False,
) -> dict[str, object]:
    """Minimal exact V3 outcome payload plus route-owned trace fields."""

    return {
        "trial_id": trial_id,
        "budget": budget,
        "schema": "causal_dual_memory_independent_activity_outcome_v3",
        "activity_transition_committed": True,
        "activity_fifo_changed": True,
        "carrier_transition_committed": carrier_committed,
        "carrier_rejection_reason_or_null": None if carrier_committed else "movement_too_short",
        "activity_rejection_reason_or_null": None,
        "state_before_sha256": state_before * 64,
        "state_after_sha256": state_after * 64,
        "activity_before_sha256": activity_before * 64,
        "activity_after_sha256": activity_after * 64,
        "carrier_before_sha256": carrier_before * 64,
        "carrier_after_sha256": carrier_after * 64,
    }


class _MockV3Backend:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.runtime = object()

    def preflight(self, *, root, identity, flags):
        self.events.append("preflight")
        return {"source_resolved_or_opened": False, "checkpoint_opened": False, "cuda_initialized": False}

    def prepare(self, *, root, identity, flags):
        self.events.append("prepare")
        flags.source_resolved = flags.source_opened = flags.checkpoint_opened = flags.cuda_initialized = True
        return self.runtime

    def source_authority(self, runtime, *, identity, flags):
        self.events.append("source_authority")
        proof = _theta_proof()
        topology = {
            "session_id": v1.SOURCE_SMOKE_SESSION,
            "source_descriptor_sha256": "a" * 64,
            "raw_t4_channel_order_sha256": proof["canonical_unit_order_sha256"],
            "theta_valid_mask_sha256": proof["valid_mask_sha256"],
            "total_unit_count": proof["raw_t4_shape"][0],
            "valid_unit_count": proof["raw_t4_shape"][0],
            "invalid_unit_count": 0,
            "valid_mask_sha256": proof["valid_mask_sha256"],
            "invalid_assignment_is_minus_one": True,
            "theta_authority_binds_only_validity_not_budget_groups": True,
        }
        return {
            "schema": "causal_dual_memory_cell_d_source_execution_authority_v1",
            "cell": v1.CELL,
            "identity_sha256": identity.sha256,
            "strict_train_roster": list(identity.strict_train_roster),
            "strict_train_roster_sha256": v1.roster_sha256(identity.strict_train_roster),
            "normalizers": identity.normalizers.payload(),
            "fixed_assets": {asset.label: asset.payload() for asset in v1.FIXED_ASSETS},
            "runtime_environment": _runtime_attestation(),
            "sealed_swa_load_proof": _sealed_swa_proof(),
            "physical_session_count": 1,
            "strict_manifest_bound_without_nonphysical_resolution": True,
            "sessions": [topology],
            "v2_theta_raw_proofs": [proof],
            "source_only": True,
            "access": {
                "source_opened": True,
                "within_opened": False,
                "external_opened": False,
                "formal_opened": False,
                "target_opened": False,
                "optimizer_steps": 0,
                "backward_calls": 0,
                "parameter_updates": 0,
                "normalizer_refit": False,
            },
        }

    def run_smoke(self, runtime, *, identity, flags):
        self.events.append("smoke")
        flags.model_forward_calls += 8
        trace = [
            _committing_trace(
                budget=10, trial_id="smoke-10", state_before="1", state_after="2",
                activity_before="3", activity_after="4", carrier_before="5", carrier_after="5",
            ),
            _committing_trace(
                budget=10, trial_id="smoke-11", state_before="2", state_after="6",
                activity_before="4", activity_after="7", carrier_before="5", carrier_after="8",
                carrier_committed=True,
            ),
        ]
        return {
            "schema": "causal_dual_memory_cell_d_source_execution_smoke_v3",
            "session": v1.SOURCE_SMOKE_SESSION,
            "budget": 10,
            "support_positions": list(range(10)),
            "audit_positions": [10, 11],
            "support_trial_ids": [f"support-{index}" for index in range(10)],
            "audit_trial_ids": ["smoke-10", "smoke-11"],
            "group_count": 4,
            "all_four_groups_finalized": True,
            "b8_threshold_applied": False,
            "required_activity_transition_count": 2,
            "activity_fifo_capacity": 20,
            "activity_transition_committed_count": 2,
            "carrier_transition_committed_count": 1,
            "carrier_transition_rejected_count": 1,
            "budget_initial_carrier_recipe": "fixed_ridge_by_trial",
            "budget_initial_carrier_support_rows": 10,
            "raw_m30_t4_used_as_initializer": False,
            "initial_support_rate_domain": v1.INITIAL_SUPPORT_RATE_DOMAIN,
            "online_update_rate_domain": v1.ONLINE_UPDATE_RATE_DOMAIN,
            "initial_support_rates_sha256": "4" * 64,
            "initial_support_exposure_seconds_sha256": "5" * 64,
            "budget_initial_carrier_parity": {"mode": "fixed_ridge_by_trial"},
            "budget_initial_carrier_sha256": "6" * 64,
            "budget_groups_sha256": "7" * 64,
            "budget_group_assignment_sha256": "8" * 64,
            "budget_group_valid_mask_sha256": "9" * 64,
            "v3_independent_activity_transitions": trace,
            "source_only": True,
        }

    def run_budget(self, runtime, *, budget, identity, flags):
        raise AssertionError("smoke mock must not run source-gate budgets")

    def revalidate(self, *, root, identity, flags):
        self.events.append("revalidate")

    def resources(self, runtime, *, flags):
        self.events.append("resources")
        return {
            "endpoint_chunks_per_s": 2.0,
            "trials_per_s": 1.0,
            "wall_seconds": 1.0,
            "endpoint_chunks_completed": 2,
            "completed_trials": 1,
            "rss_bytes": 1,
            "current_cuda_allocated_bytes": 4,
            "current_cuda_reserved_bytes": 8,
            "peak_cuda_allocated_bytes": 4,
            "peak_cuda_reserved_bytes": 8,
            "selected_device": _profile(),
        }

    def close(self, runtime):
        self.events.append("close")


def test_v3_explicit_closure_identity_and_no_glob_drift_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    assert v3.WORKORDER_SHA256 == "c53511d170069185bae2a7f639466646917f5a0ef84da47fa73f14e102288478"
    closure = v3.execution_closure_payload(ROOT)
    paths = [row["path"] for row in closure["paths"]]
    assert len(paths) == len(set(paths)) == 53
    assert paths[-5:] == [
        v3.WORKORDER_RELATIVE,
        "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_v3.py",
        "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical_v3.py",
        "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_gate_v3.py",
        "tfpd_exploration/tests/test_causal_dual_memory_cell_d_source_execution_v3.py",
    ]
    assert closure["v2_superseded_evidence"] == v3.V2_SUPERSEDED_EVIDENCE
    identity = _synthetic_identity(v3.SOURCE_SMOKE_SPEC)
    monkeypatch.setattr(v3, "execution_closure_payload", lambda root: {**_synthetic_closure(), "closure_sha256": "e" * 64})
    with pytest.raises(v3.SourceExecutionV3Error, match="closure"):
        v3.validate_identity_current(ROOT, identity)


def test_v3_lifecycle_publishes_split_transition_receipts_only_after_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "tfpd_exploration/results").mkdir(parents=True)
    identity = _synthetic_identity(v3.SOURCE_SMOKE_SPEC)
    monkeypatch.setattr(v3, "execution_closure_payload", lambda root: _synthetic_closure())
    capability = v3._issue_root_reviewed_capability(
        tmp_path, identity, source_data_root=None, seal=v3._ROOT_REVIEW_SEAL,
    )
    backend = _MockV3Backend()
    result = v3.execute_authorized(
        tmp_path,
        identity=identity,
        capability=capability,
        backend=backend,
        environ={"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
    )
    assert result["status"] == "SMOKE_COMPLETED"
    assert backend.events == ["preflight", "prepare", "source_authority", "smoke", "revalidate", "resources", "close"]
    root = tmp_path / v3.SOURCE_SMOKE_ROOT_RELATIVE
    assert {path.name for path in root.iterdir()} == {
        "attempt.json", "attempt.json.sha256", "launch.json", "launch.json.sha256",
        "source_authority.json", "source_authority.json.sha256", "smoke.json", "smoke.json.sha256",
        "terminal.json", "terminal.json.sha256",
    }
    terminal = json.loads((root / "terminal.json").read_bytes())
    assert terminal["launch_closure_sha256"] == terminal["final_closure_sha256"] == "d" * 64
    smoke = json.loads((root / "smoke.json").read_bytes())
    assert smoke["v3_binding"]["independent_activity_contract"] == v3.INDEPENDENT_ACTIVITY_CONTRACT
    assert smoke["budget"] == 10
    assert smoke["support_positions"] == list(range(10))
    assert smoke["audit_positions"] == [10, 11]
    assert smoke["activity_transition_committed_count"] == 2
    assert [row["trial_id"] for row in smoke["v3_independent_activity_transitions"]] == [
        "smoke-10", "smoke-11",
    ]
    assert smoke["v3_independent_activity_transitions"][0]["carrier_transition_committed"] is False
    assert smoke["v3_independent_activity_transitions"][1]["carrier_transition_committed"] is True


def test_v3_physical_factory_is_a_narrow_inherited_parser_model_seam_without_data_or_cuda(tmp_path: Path) -> None:
    source_cap = v1._issue_root_reviewed_source_data_capability(
        canonical_root=tmp_path / "strict_source",
        strict_train_roster=_roster(),
        seal=v1._ROOT_REVIEW_SEAL,
    )
    backend = physical_v3.build_reviewed_physical_backend(
        root=ROOT, source_data=source_cap, selected_device=_profile(),
    )
    assert isinstance(backend, physical_v3.PhysicalSourceExecutionBackendV3)
    assert isinstance(backend.provider, v2_physical.V2ThetaStrict27SessionProvider)  # type: ignore[name-defined]
    assert isinstance(backend.executor, physical_v3.IndependentActivityCellDFourGroupExecutor)
    seam = inspect.getsource(physical_v3.IndependentActivityCellDFourGroupExecutor.begin_session_budget)
    assert "IndependentActivityCausalDualMemory" in seam
    assert "V2ThetaStrict27SessionProvider" not in seam
    assert not (tmp_path / "strict_source").exists()
    assert torch.cuda.is_initialized() is False


def test_static_cli_and_public_import_are_torch_free_and_fail_closed() -> None:
    script = ROOT / "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_gate_v3.py"
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    dry = subprocess.run([sys.executable, "-I", str(script), "--dry-run"], cwd=ROOT, text=True,
                         capture_output=True, check=True, env=environment)
    payload = json.loads(dry.stdout)
    assert payload["workorder_sha256"] == v3.WORKORDER_SHA256
    assert payload["opens_source"] is payload["loads_checkpoint"] is payload["initializes_cuda"] is False
    blocked = subprocess.run([sys.executable, "-I", str(script), "--execute", "--source-smoke"], cwd=ROOT,
                             text=True, capture_output=True, env=environment)
    assert blocked.returncode != 0 and "root-reviewed in-process capability" in blocked.stderr
    probe = (
        "import pathlib,sys; root=pathlib.Path.cwd()/'tfpd_exploration'; "
        "sys.path.insert(0,str(root)); import src.causal_dual_memory_cell_d_v1.source_execute_v3; "
        "print('torch' in sys.modules)"
    )
    imported = subprocess.run([sys.executable, "-I", "-c", probe], cwd=ROOT, text=True,
                              capture_output=True, check=True, env=environment)
    assert imported.stdout.strip() == "False"
    assert "import torch" not in script.read_text(encoding="utf-8")
