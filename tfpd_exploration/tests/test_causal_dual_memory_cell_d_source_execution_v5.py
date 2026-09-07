"""Focused no-data/no-CUDA tests for CDM-D Source Execution V5."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import pytest


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "tfpd_exploration", ROOT / "tfpd_exploration/src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from src.causal_dual_memory_cell_d_v1 import core  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_adapter  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_audit  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_execute as v1  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_execute_physical as physical_v1  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_execute_physical_v3 as physical_v3  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_execute_physical_v4 as physical_v4  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_execute_physical_v5 as physical_v5  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_execute_v2 as v2  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_execute_v3 as v3  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_execute_v5 as v5  # noqa: E402


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _profile() -> dict[str, object]:
    return dict(v1.COMPATIBLE_DEVICE_PROFILES["gpu1"])


def _roster() -> tuple[str, ...]:
    return (v1.SOURCE_SMOKE_SESSION, *(f"sub-C_ses-CO-{index:08d}" for index in range(26)))


def _write_pair(directory: Path, name: str, payload: Mapping[str, object]) -> str:
    body = v5._json_bytes(dict(payload))
    digest = _sha(body)
    (directory / name).write_bytes(body)
    (directory / f"{name}.sha256").write_bytes(f"{digest}  {name}\n".encode("ascii"))
    os.chmod(directory / name, 0o444)
    os.chmod(directory / f"{name}.sha256", 0o444)
    return digest


def _commit_transition(
    *,
    budget: int,
    trial_id: str,
    ordinal: int,
    carrier_committed: bool = False,
) -> dict[str, object]:
    before_state = f"{ordinal + 1:064x}"
    after_state = f"{ordinal + 2:064x}"
    before_activity = f"{ordinal + 101:064x}"
    after_activity = f"{ordinal + 102:064x}"
    before_carrier = f"{ordinal + 201:064x}"
    after_carrier = f"{ordinal + 202:064x}" if carrier_committed else before_carrier
    return {
        "trial_id": trial_id,
        "budget": budget,
        "schema": "causal_dual_memory_independent_activity_outcome_v3",
        "activity_transition_committed": True,
        "activity_fifo_changed": True,
        "carrier_transition_committed": carrier_committed,
        "carrier_rejection_reason_or_null": None if carrier_committed else core.UpdateRejectionReason.MOVEMENT_TOO_SHORT.value,
        "activity_rejection_reason_or_null": None,
        "state_before_sha256": before_state,
        "state_after_sha256": after_state,
        "activity_before_sha256": before_activity,
        "activity_after_sha256": after_activity,
        "carrier_before_sha256": before_carrier,
        "carrier_after_sha256": after_carrier,
    }


def _chain_transitions(*, budget: int, ids: tuple[str, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    state, activity, carrier = "1" * 64, "2" * 64, "3" * 64
    for index, trial_id in enumerate(ids):
        next_state = f"{1000 + index:064x}"
        next_activity = f"{2000 + index:064x}"
        rows.append({
            "trial_id": trial_id,
            "budget": budget,
            "schema": "causal_dual_memory_independent_activity_outcome_v3",
            "activity_transition_committed": True,
            "activity_fifo_changed": True,
            "carrier_transition_committed": False,
            "carrier_rejection_reason_or_null": core.UpdateRejectionReason.MOVEMENT_TOO_SHORT.value,
            "activity_rejection_reason_or_null": None,
            "state_before_sha256": state,
            "state_after_sha256": next_state,
            "activity_before_sha256": activity,
            "activity_after_sha256": next_activity,
            "carrier_before_sha256": carrier,
            "carrier_after_sha256": carrier,
        })
        state, activity = next_state, next_activity
    return rows


def _offline_m30_trace(*, ids: tuple[str, ...]) -> list[dict[str, object]]:
    return [{
        "trial_id": trial_id,
        "budget": 30,
        "schema": "causal_dual_memory_independent_activity_offline_m30_v3",
        "source_audit_offline_no_deployment_commit": True,
        "pending_activity_transition_ready": True,
        "pending_carrier_transition_accepted": False,
        "pending_activity_fifo_would_change": False,
        "pending_carrier_rejection_reason_or_null": core.UpdateRejectionReason.MOVEMENT_TOO_SHORT.value,
        "activity_fifo_capacity": 0,
        "activity_query_count_before": 0,
        "activity_query_count_after": 0,
        "state_before_sha256": "a" * 64,
        "state_after_sha256": "a" * 64,
        "activity_before_sha256": "b" * 64,
        "activity_after_sha256": "b" * 64,
        "carrier_before_sha256": "c" * 64,
        "carrier_after_sha256": "c" * 64,
    } for trial_id in ids]


def _write_v4_predecessors(
    root: Path,
    *,
    mutate: str | None = None,
) -> tuple[v5.V4SmokePredecessorExpectation, v5.V4GateFailurePredecessorExpectation]:
    """Create schema-valid held V4 graphs for isolated no-data lifecycle tests."""

    smoke_relative = "tfpd_exploration/results/synthetic_v4_smoke"
    smoke_directory = root / smoke_relative
    smoke_directory.mkdir(parents=True)
    smoke_identity = {"closure": {"closure_sha256": "a" * 64}, "synthetic": "v4-smoke"}
    smoke_identity_sha = _sha(v5._json_bytes(smoke_identity))
    smoke_binding = {"identity_sha256": smoke_identity_sha, "v4_closure_sha256": "a" * 64}
    smoke_attempt = {
        "schema": "causal_dual_memory_cell_d_source_execution_attempt_v4",
        "status": "ATTEMPT_RESERVED", "identity": smoke_identity, "source_only": True,
        "source_resolved_or_opened": False, "checkpoint_opened": False, "cuda_initialized": False,
    }
    smoke_attempt_sha = _write_pair(smoke_directory, "attempt.json", smoke_attempt)
    smoke_launch = {
        "schema": "causal_dual_memory_cell_d_source_execution_launch_v4", "status": "LAUNCHED",
        "identity": smoke_identity, "attempt_sha256": smoke_attempt_sha,
        "launch_closure_sha256": "a" * 64,
    }
    smoke_launch_sha = _write_pair(smoke_directory, "launch.json", smoke_launch)
    smoke_authority = {
        "schema": "causal_dual_memory_cell_d_source_execution_authority_v1",
        "identity_sha256": smoke_identity_sha, "source_only": True, "v4_binding": smoke_binding,
        "access": {
            "source_opened": True, "within_opened": False, "external_opened": False,
            "formal_opened": False, "target_opened": False, "backward_calls": 0,
            "optimizer_steps": 0, "parameter_updates": 0,
        },
    }
    smoke_authority_sha = _write_pair(smoke_directory, "source_authority.json", smoke_authority)
    smoke_ids = ("smoke-10", "smoke-11")
    smoke_trace = _chain_transitions(budget=10, ids=smoke_ids)
    smoke_payload = {
        "schema": "causal_dual_memory_cell_d_source_execution_smoke_v3",
        "session": v1.SOURCE_SMOKE_SESSION, "budget": 10,
        "support_positions": list(range(10)), "audit_positions": [10, 11],
        "audit_trial_ids": list(smoke_ids), "activity_transition_committed_count": 2,
        "carrier_transition_committed_count": 0, "carrier_transition_rejected_count": 2,
        "v3_independent_activity_transitions": smoke_trace, "source_only": True,
        "v4_binding": smoke_binding,
    }
    smoke_sha = _write_pair(smoke_directory, "smoke.json", smoke_payload)
    smoke_terminal = {
        "schema": "causal_dual_memory_cell_d_source_execution_terminal_v4", "status": "SMOKE_COMPLETED",
        "identity": smoke_identity, "attempt_sha256": smoke_attempt_sha, "launch_sha256": smoke_launch_sha,
        "source_authority_sha256": smoke_authority_sha, "evidence_sha256s": {"smoke.json": smoke_sha},
        "launch_closure_sha256": "a" * 64, "final_closure_sha256": "a" * 64,
        "source_only": True, "target_optimizer_backward_update": 0,
    }
    smoke_terminal_sha = _write_pair(smoke_directory, "terminal.json", smoke_terminal)
    smoke_expectation = v5.V4SmokePredecessorExpectation(
        smoke_relative, smoke_attempt_sha, smoke_launch_sha, smoke_authority_sha, smoke_sha, smoke_terminal_sha,
        smoke_identity_sha, "a" * 64,
    )

    gate_relative = "tfpd_exploration/results/synthetic_v4_gate_failure"
    gate_directory = root / gate_relative
    gate_directory.mkdir(parents=True)
    gate_identity = {"closure": {"closure_sha256": "b" * 64}, "synthetic": "v4-gate"}
    gate_identity_sha = _sha(v5._json_bytes(gate_identity))
    gate_binding = {"identity_sha256": gate_identity_sha, "v4_closure_sha256": "b" * 64}
    gate_attempt = {
        "schema": "causal_dual_memory_cell_d_source_execution_attempt_v4",
        "status": "ATTEMPT_RESERVED", "identity": gate_identity, "source_only": True,
        "source_resolved_or_opened": False, "checkpoint_opened": False, "cuda_initialized": False,
    }
    gate_attempt_sha = _write_pair(gate_directory, "attempt.json", gate_attempt)
    gate_launch = {
        "schema": "causal_dual_memory_cell_d_source_execution_launch_v4", "status": "LAUNCHED",
        "identity": gate_identity, "attempt_sha256": gate_attempt_sha,
        "launch_closure_sha256": "b" * 64,
    }
    gate_launch_sha = _write_pair(gate_directory, "launch.json", gate_launch)
    gate_authority = {
        "schema": "causal_dual_memory_cell_d_source_execution_authority_v1",
        "identity_sha256": gate_identity_sha, "source_only": True, "v4_binding": gate_binding,
        "access": {
            "source_opened": True, "within_opened": False, "external_opened": False,
            "formal_opened": False, "target_opened": False, "backward_calls": 0,
            "optimizer_steps": 0, "parameter_updates": 0,
        },
    }
    gate_authority_sha = _write_pair(gate_directory, "source_authority.json", gate_authority)
    gate_failure = {
        "schema": "causal_dual_memory_cell_d_source_execution_failure_v4", "status": "FAILED",
        "identity": gate_identity, "attempt_sha256": gate_attempt_sha, "launch_sha256": gate_launch_sha,
        "source_authority_sha256": gate_authority_sha, "stage": "budget_m30",
        "error_class": "SourceExecutionPhysicalV3Error", "error_sha256": "e" * 64,
        "terminal_published": False,
        "flags": {
            "source_opened": True, "checkpoint_opened": True, "cuda_initialized": True,
            "model_forward_calls": 8, "backward_calls": 0, "optimizer_steps": 0,
            "parameter_updates": 0, "within_opened": False, "external_opened": False,
            "formal_opened": False, "target_opened": False,
        },
    }
    gate_failure_sha = _write_pair(gate_directory, "failure.json", gate_failure)
    gate_expectation = v5.V4GateFailurePredecessorExpectation(
        gate_relative, gate_attempt_sha, gate_launch_sha, gate_authority_sha, gate_failure_sha,
        gate_identity_sha, "b" * 64, "e" * 64,
    )

    if mutate == "extra":
        (smoke_directory / "extra.json").write_text("{}", encoding="utf-8")
        os.chmod(smoke_directory / "extra.json", 0o444)
    elif mutate == "sidecar":
        os.chmod(gate_directory / "failure.json.sha256", 0o644)
        (gate_directory / "failure.json.sha256").write_text(f"{gate_failure_sha}  forged.json\n", encoding="ascii")
        os.chmod(gate_directory / "failure.json.sha256", 0o444)
    elif mutate == "mode":
        os.chmod(smoke_directory / "launch.json", 0o644)
    elif mutate == "semantic":
        os.chmod(gate_directory / "failure.json", 0o644)
        os.chmod(gate_directory / "failure.json.sha256", 0o644)
        gate_failure["stage"] = "budget_m10"
        gate_failure_sha = _write_pair(gate_directory, "failure.json", gate_failure)
        gate_expectation = replace(gate_expectation, failure_sha256=gate_failure_sha)
    return smoke_expectation, gate_expectation


def _stage_current_closure(stage: Path) -> None:
    for relative in (*v5._V5_INHERITED_PATHS, *v5._V5_OWNED_PATHS):
        source = ROOT / relative
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        os.chmod(target, 0o644)
    (stage / "tfpd_exploration/results").mkdir(parents=True, exist_ok=True)


def _identity(
    stage: Path,
    smoke: v5.V4SmokePredecessorExpectation,
    gate: v5.V4GateFailurePredecessorExpectation,
) -> v5.SourceExecutionV5Identity:
    return v5.SourceExecutionV5Identity(
        spec=v5.SourceExecutionV5Spec("source_gate", "tfpd_exploration/results/v5_synthetic_gate"),
        closure=v5.execution_closure_payload(stage, accepted_smoke=smoke, failed_gate=gate),
        strict_train_roster=_roster(),
        fixed_assets={asset.label: asset.payload() for asset in v1.FIXED_ASSETS},
        normalizers=v1.SEALED_NORMALIZERS,
        selected_device=_profile(),
        accepted_v4_smoke=smoke,
        failed_v4_gate=gate,
    )


def _runtime_attestation() -> dict[str, object]:
    return {
        **_profile(), "visible_devices": 1, "attested": True,
        "torch_cuda_matmul_allow_tf32": False, "torch_cudnn_allow_tf32": False,
    }


def _sealed_swa_proof() -> dict[str, object]:
    return {
        "schema": v1.SEALED_SWA_LOAD_PROOF_SCHEMA,
        "sealed_terminal_sha256": v1.SEALED_CELL_D_TERMINAL_SHA256,
        "sealed_swa_sha256": v1.SEALED_CELL_D_SWA_SHA256,
        "fresh_strict_load": True, "recomputed_state_dict_sha256": "c" * 64,
        "initialized_trainable_parameters": v1.SEALED_CELL_D_INITIALIZED_TRAINABLE_PARAMETERS,
        "uninitialized_lazy_keys": list(v1.SEALED_CELL_D_UNINITIALIZED_LAZY_KEYS),
        "model_eval": True, "no_grad": True, "finite_forward": True,
        "repeated_fixed_forward_bitwise_equal": True, "model_state_unchanged": True,
        "dynamic_dropout_calls": 0,
    }


def _theta_proof(session: str) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_theta_raw_proof_v2", "session": session,
        "hash_law": v2.THETA_RAW_HASH_LAW, "hash_law_expression": v2.THETA_RAW_HASH_LAW_TEXT,
        "raw_t4_shape": [7, 4], "raw_t4_dtype": "float32", "raw_t4_all_finite": True,
        "raw_t4_sha256": "1" * 64, "sealed_raw_t4_sha256": "1" * 64,
        "theta_float64_sha256": "2" * 64, "sealed_theta_float64_sha256": "2" * 64,
        "theta_atan2_float64_bitwise_equal": True,
        "valid_mask_sha256": "3" * 64, "sealed_valid_mask_sha256": "3" * 64,
        "validity_raw_m_gt_modulation_eps_bitwise_equal": True,
        "known_invalid_unit_count": v1.KNOWN_THETA_INVALID_UNIT_COUNTS.get(session, 0),
        "canonical_unit_order_exact": True, "canonical_unit_order_sha256": "4" * 64,
    }


def _topology(session: str) -> dict[str, object]:
    invalid = v1.KNOWN_THETA_INVALID_UNIT_COUNTS.get(session, 0)
    return {
        "session_id": session, "source_descriptor_sha256": "a" * 64,
        "raw_t4_channel_order_sha256": "4" * 64, "theta_valid_mask_sha256": "3" * 64,
        "total_unit_count": 7, "valid_unit_count": 7 - invalid, "invalid_unit_count": invalid,
        "valid_mask_sha256": "3" * 64, "invalid_assignment_is_minus_one": True,
        "theta_authority_binds_only_validity_not_budget_groups": True,
    }


def _raw_physical_evidence(*, budget: int, ordinal: int) -> dict[str, object]:
    if budget == 30:
        transition = _offline_m30_trace(ids=("unused",))[0]
        transition.pop("trial_id")
        transition.pop("budget")
        return {
            "all_four_groups_finalized": True, "group_label_broadcast": False,
            "all_group_inputs_physically_sliced": True, "held_group_count": 4,
            "label_join_before_outcomes": False,
            "deployment_activity_memory_before_sha256": "a" * 64,
            "deployment_activity_memory_after_sha256": "a" * 64,
            "carrier_state_before_sha256": "c" * 64, "carrier_state_after_sha256": "c" * 64,
            "accepted_group_update_count": 0, "activity_fifo_capacity": 0,
            "activity_query_count_before": 0, "activity_query_count_after": 0,
            "m30_audit_enters_deployment_activity_memory": False,
            "independent_activity_transition_contract": dict(v3.INDEPENDENT_ACTIVITY_CONTRACT),
            "independent_activity_transition": transition,
        }
    transition = _commit_transition(budget=budget, trial_id="unused", ordinal=ordinal)
    transition.pop("trial_id")
    transition.pop("budget")
    return {
        "all_four_groups_finalized": True, "group_label_broadcast": False,
        "all_group_inputs_physically_sliced": True, "held_group_count": 4,
        "label_join_before_outcomes": False,
        "deployment_activity_memory_before_sha256": transition["activity_before_sha256"],
        "deployment_activity_memory_after_sha256": transition["activity_after_sha256"],
        "carrier_state_before_sha256": transition["carrier_before_sha256"],
        "carrier_state_after_sha256": transition["carrier_after_sha256"],
        "accepted_group_update_count": 0, "activity_fifo_capacity": 30 - budget,
        "activity_query_count_before": ordinal, "activity_query_count_after": ordinal + 1,
        "m30_audit_enters_deployment_activity_memory": True,
        "independent_activity_transition_contract": dict(v3.INDEPENDENT_ACTIVITY_CONTRACT),
        "independent_activity_transition": transition,
    }


def _raw_event(*, session: str, trial_id: str, budget: int, ordinal: int) -> physical_v1.FinalizedFourGroupPseudo:
    return physical_v1.FinalizedFourGroupPseudo(
        session, trial_id, (None, None, None, None),
        (core.UpdateRejectionReason.MOVEMENT_TOO_SHORT,) * 4,
        f"{ordinal + 500:064x}", _raw_physical_evidence(budget=budget, ordinal=ordinal),
    )


_Event = tuple[str, str, int | None, str]


class _SyntheticProvider:
    def __init__(self, *, event_log: list[_Event] | None = None) -> None:
        self.truth_calls: list[tuple[str, str]] = []
        self.event_log = event_log

    def join_audit_true_direction(self, *, session_id: str, trial_id: str) -> source_adapter.AuditOnlyTrueDirection:
        self.truth_calls.append((session_id, trial_id))
        if self.event_log is not None:
            self.event_log.append(("truth_join", session_id, None, trial_id))
        return source_adapter.AuditOnlyTrueDirection(session_id, trial_id, 0.0)


class _SyntheticV5Executor(physical_v5.V5OneShotFinalizedRowExecutor):
    def __init__(self, *, event_log: list[_Event] | None = None) -> None:
        super().__init__(root=ROOT)
        self.forward_calls: list[tuple[str, int, str]] = []
        self.event_log = event_log

    def execute_completed_trial(self, *, material, budget, trial_id, support_trial_ids, flags):
        self.forward_calls.append((material.session_id, budget, trial_id))
        if self.event_log is not None:
            self.event_log.append(("raw_executor", material.session_id, budget, trial_id))
        raw = _raw_event(
            session=material.session_id, trial_id=trial_id, budget=budget, ordinal=len(self.forward_calls),
        )
        return self._capture_raw_event(raw, material=material, budget=budget, trial_id=trial_id)

    def _capture_raw_event(self, raw, *, material, budget, trial_id):
        if self.event_log is not None:
            self.event_log.append(("capture", material.session_id, budget, trial_id))
        return super()._capture_raw_event(raw, material=material, budget=budget, trial_id=trial_id)

    def consume_raw_event(self, *, session_id, budget, trial_id):
        if self.event_log is not None:
            self.event_log.append(("consume", session_id, budget, trial_id))
        return super().consume_raw_event(session_id=session_id, budget=budget, trial_id=trial_id)


class _SyntheticFullGateBackend(physical_v5.PhysicalSourceExecutionBackendV5):
    """Exercise inherited V3/V1 full-gate MRO without source/NWB tensors."""

    def _session_b8(self, runtime, *, budget, session, flags):
        material = runtime.materials[session]
        row = self._finalized_row(
            runtime=runtime, material=material, budget=budget,
            trial_id=f"{session}:query:m{budget}", support_trial_ids=(), flags=flags,
        )
        assert type(row) is source_audit.GroupedPseudoAuditRow
        return {"session": session, "grouped_type": type(row).__name__}


def _bare_backend(
    *, provider: object, executor: object,
    backend_type: type[physical_v5.PhysicalSourceExecutionBackendV5] = physical_v5.PhysicalSourceExecutionBackendV5,
) -> physical_v5.PhysicalSourceExecutionBackendV5:
    backend = object.__new__(backend_type)
    backend.provider = provider
    backend.executor = executor
    backend._v3_transitions = {}
    backend._runtime = None
    backend._closed = False
    return backend


def test_workorder_closure_and_static_v5_contract_are_explicit() -> None:
    closure = v5.execution_closure_payload(ROOT)
    paths = [row["path"] for row in closure["paths"]]
    assert v5.WORKORDER_SHA256 == "503fa0276b3bb32fd31b9da51dca961d1214dfef0a0d7c40cd5e8bec3e87cb9b"
    assert len(paths) == len(set(paths)) == len(v5._V5_INHERITED_PATHS) + len(v5._V5_OWNED_PATHS)
    assert paths[-5:] == list(v5._V5_OWNED_PATHS)
    assert v5.SOURCE_GATE_SPEC.kind == "source_gate"
    assert "glob" not in inspect.getsource(v5.execution_closure_payload).lower()
    assert v5.V4_ACCEPTED_SMOKE.payload()["exact_leaf_count"] == 10
    assert v5.V4_FAILED_GATE.payload()["exact_leaf_count"] == 8


@pytest.mark.parametrize("mutate", (None, "extra", "sidecar", "mode", "semantic"))
def test_held_v4_smoke_and_gate_predecessors_are_exact_before_v5(tmp_path: Path, mutate: str | None) -> None:
    smoke, gate = _write_v4_predecessors(tmp_path, mutate=mutate)
    if mutate is None:
        proof = v5.validate_v4_predecessors(tmp_path, accepted_smoke=smoke, failed_gate=gate)
        assert proof["accepted_v4_smoke"] == smoke.payload()
        assert proof["failed_v4_gate"] == gate.payload()
        return
    with pytest.raises(v5.SourceExecutionV5Error):
        v5.validate_v4_predecessors(tmp_path, accepted_smoke=smoke, failed_gate=gate)


def test_v4_full_gate_mro_reproduces_historical_post_conversion_type_error() -> None:
    provider = _SyntheticProvider()
    executor = SimpleNamespace(
        execute_completed_trial=lambda **kwargs: _raw_event(
            session=kwargs["material"].session_id, trial_id=kwargs["trial_id"], budget=kwargs["budget"], ordinal=1,
        ),
    )
    backend = object.__new__(physical_v4.PhysicalSourceExecutionBackendV4)
    backend.provider = provider
    backend.executor = executor
    backend._v3_transitions = {}
    material = SimpleNamespace(session_id=v1.SOURCE_SMOKE_SESSION)
    with pytest.raises(physical_v3.SourceExecutionPhysicalV3Error, match="finalized row type drift"):
        physical_v4.PhysicalSourceExecutionBackendV4._finalized_row(
            backend, runtime=SimpleNamespace(executor=executor, provider=provider), material=material,
            budget=10, trial_id="historical-row", support_trial_ids=(), flags=v1.RuntimeFlags(),
        )
    assert provider.truth_calls == [(v1.SOURCE_SMOKE_SESSION, "historical-row")]


def test_v5_actual_full_gate_mro_captures_raw_once_joins_once_and_returns_grouped_row() -> None:
    event_log: list[_Event] = []
    provider = _SyntheticProvider(event_log=event_log)
    executor = _SyntheticV5Executor(event_log=event_log)
    backend = _bare_backend(provider=provider, executor=executor, backend_type=_SyntheticFullGateBackend)
    roster = _roster()
    materials = {session: SimpleNamespace(session_id=session) for session in roster}
    runtime = SimpleNamespace(
        physical_roster=roster, manifest_roster=roster, materials=materials,
        provider=provider, executor=executor, closed=False,
    )
    backend._runtime = runtime
    flags = v1.RuntimeFlags()
    for budget in (30, 10, 4):
        rows = backend.run_budget(runtime, budget=budget, identity=object(), flags=flags)
        assert rows[0]["grouped_type"] == "GroupedPseudoAuditRow"
        trace = rows[0]["v3_independent_activity_transitions"]
        assert len(trace) == 1 and trace[0]["budget"] == budget
        if budget == 30:
            assert trace[0]["schema"] == "causal_dual_memory_independent_activity_offline_m30_v3"
            assert trace[0]["state_before_sha256"] == trace[0]["state_after_sha256"]
        else:
            assert trace[0]["activity_transition_committed"] is True
            assert trace[0]["carrier_transition_committed"] is False
    assert len(executor.forward_calls) == 3 * len(roster)
    assert len(provider.truth_calls) == 3 * len(roster)
    expected_event_log: list[_Event] = []
    for budget in (30, 10, 4):
        for session in roster:
            trial_id = f"{session}:query:m{budget}"
            expected_event_log.extend((
                ("raw_executor", session, budget, trial_id),
                ("capture", session, budget, trial_id),
                ("truth_join", session, None, trial_id),
                ("consume", session, budget, trial_id),
            ))
    assert event_log == expected_event_log
    assert len(event_log) == len(set(event_log)) == 4 * 3 * len(roster)
    executor.assert_no_unconsumed_raw_events()
    method = inspect.getsource(physical_v5.PhysicalSourceExecutionBackendV5._finalized_row)
    assert "PhysicalSourceExecutionBackend._finalized_row" in method and "super()._finalized_row" not in method
    assert "GroupedPseudoAuditRow" in inspect.getsource(physical_v5.PhysicalSourceExecutionBackendV5._validate_and_record_raw_event)


def test_v5_raw_event_one_shot_and_return_type_boundaries_fail_closed() -> None:
    provider = _SyntheticProvider()
    executor = _SyntheticV5Executor()
    backend = _bare_backend(provider=provider, executor=executor)
    material = SimpleNamespace(session_id=v1.SOURCE_SMOKE_SESSION)
    raw = _raw_event(session=material.session_id, trial_id="one-shot", budget=10, ordinal=1)
    executor._capture_raw_event(raw, material=material, budget=10, trial_id="one-shot")
    with pytest.raises(physical_v5.SourceExecutionPhysicalV5Error, match="stale leftover"):
        executor.assert_no_unconsumed_raw_events(budget=10)
    consumed = executor.consume_raw_event(session_id=material.session_id, budget=10, trial_id="one-shot")
    assert consumed is raw
    with pytest.raises(physical_v5.SourceExecutionPhysicalV5Error, match="consumed twice"):
        executor.consume_raw_event(session_id=material.session_id, budget=10, trial_id="one-shot")
    with pytest.raises(physical_v5.SourceExecutionPhysicalV5Error, match="duplicate raw"):
        executor._capture_raw_event(raw, material=material, budget=10, trial_id="one-shot")
    grouped = source_audit.GroupedPseudoAuditRow(
        budget=10, session_id=material.session_id, trial_id="one-shot",
        true_direction=source_adapter.AuditOnlyTrueDirection(material.session_id, "one-shot", 0.0),
        pseudo_direction_indices=raw.pseudo_direction_indices, rejection_reasons=raw.rejection_reasons,
        prefix_digest=raw.prefix_digest,
    )
    with pytest.raises(physical_v5.SourceExecutionPhysicalV5Error, match="return type"):
        backend._validate_and_record_raw_event(
            raw=raw, grouped=object(), material=material, budget=10, trial_id="one-shot",
        )
    with pytest.raises(physical_v5.SourceExecutionPhysicalV5Error, match="raw finalized-event type"):
        backend._validate_and_record_raw_event(
            raw=object(), grouped=grouped, material=material, budget=10, trial_id="one-shot",
        )


def test_v5_consume_never_captured_fresh_key_fails_closed_as_missing_or_stale() -> None:
    executor = _SyntheticV5Executor()
    with pytest.raises(physical_v5.SourceExecutionPhysicalV5Error, match="consumer missing or stale"):
        executor.consume_raw_event(
            session_id=v1.SOURCE_SMOKE_SESSION, budget=10, trial_id="never-captured",
        )


@pytest.mark.parametrize(
    ("raw_session", "raw_trial_id"),
    (
        (_roster()[1], "expected-trial"),
        (v1.SOURCE_SMOKE_SESSION, "different-trial"),
    ),
    ids=("session-mismatch", "trial-mismatch"),
)
def test_v5_capture_exact_raw_event_with_mismatched_identity_fails_closed(
    raw_session: str,
    raw_trial_id: str,
) -> None:
    executor = _SyntheticV5Executor()
    material = SimpleNamespace(session_id=v1.SOURCE_SMOKE_SESSION)
    raw = _raw_event(session=raw_session, trial_id=raw_trial_id, budget=10, ordinal=1)
    with pytest.raises(physical_v5.SourceExecutionPhysicalV5Error, match="type/identity drift"):
        executor._capture_raw_event(
            raw, material=material, budget=10, trial_id="expected-trial",
        )


class _SuccessfulFullGateBackend:
    """Complete source-free full-gate lifecycle backend with exact V3 traces."""

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
        sessions = [_topology(session) for session in identity.strict_train_roster]
        return {
            "schema": "causal_dual_memory_cell_d_source_execution_authority_v1", "cell": v1.CELL,
            "identity_sha256": identity.sha256, "strict_train_roster": list(identity.strict_train_roster),
            "strict_train_roster_sha256": v1.roster_sha256(identity.strict_train_roster),
            "normalizers": identity.normalizers.payload(),
            "fixed_assets": {asset.label: asset.payload() for asset in v1.FIXED_ASSETS},
            "runtime_environment": _runtime_attestation(), "sealed_swa_load_proof": _sealed_swa_proof(),
            "physical_session_count": len(sessions), "strict_manifest_bound_without_nonphysical_resolution": True,
            "sessions": sessions, "v2_theta_raw_proofs": [_theta_proof(session) for session in identity.strict_train_roster],
            "source_only": True,
            "access": {
                "source_opened": True, "within_opened": False, "external_opened": False,
                "formal_opened": False, "target_opened": False, "optimizer_steps": 0,
                "backward_calls": 0, "parameter_updates": 0, "normalizer_refit": False,
            },
        }

    def run_budget(self, runtime, *, budget, identity, flags):
        self.events.append(f"budget:{budget}")
        pool_count = {30: 30, 10: 20, 4: 26}[budget]
        rows: list[dict[str, object]] = []
        for session in identity.strict_train_roster:
            pool = tuple(f"{session}:m{budget}:{index}" for index in range(pool_count))
            trace = _offline_m30_trace(ids=pool) if budget == 30 else _chain_transitions(budget=budget, ids=pool)
            topology = _topology(session)
            rows.append({
                "schema": "causal_dual_memory_cell_d_source_execution_session_b8_v1", "budget": budget,
                "session": session, "status": "COMPLETE_FIXED_POOL", "pass": True,
                "unit_topology": {
                    "total_unit_count": topology["total_unit_count"], "valid_unit_count": topology["valid_unit_count"],
                    "invalid_unit_count": topology["invalid_unit_count"], "valid_mask_sha256": topology["valid_mask_sha256"],
                },
                "source_authority_unit_topology": topology, "fixed_pool_trial_ids": list(pool),
                "support_trial_ids": [f"{session}:support:{index}" for index in range(budget)],
                "v3_independent_activity_transitions": trace,
                "budget_initial_carrier_recipe": "fixed_ridge_by_trial",
                "budget_initial_carrier_support_rows": budget, "raw_m30_t4_used_as_initializer": False,
                "initial_support_rate_domain": v1.INITIAL_SUPPORT_RATE_DOMAIN,
                "online_update_rate_domain": v1.ONLINE_UPDATE_RATE_DOMAIN,
                "initial_support_rates_sha256": "4" * 64, "initial_support_exposure_seconds_sha256": "5" * 64,
                "budget_initial_carrier_parity": {"mode": "fixed_ridge_by_trial"},
                "budget_initial_carrier_sha256": "6" * 64, "budget_groups_sha256": "7" * 64,
                "budget_group_assignment_sha256": "8" * 64, "budget_group_valid_mask_sha256": "9" * 64,
                "source_only": True, "target_optimizer_backward_update": 0,
            })
        flags.model_forward_calls += len(rows) * pool_count * 8
        return tuple(rows)

    def revalidate(self, *, root, identity, flags):
        self.events.append("revalidate")

    def resources(self, runtime, *, flags):
        self.events.append("resources")
        return {
            "endpoint_chunks_per_s": 2.0, "trials_per_s": 1.0, "wall_seconds": 1.0,
            "endpoint_chunks_completed": 2, "completed_trials": 1, "rss_bytes": 1,
            "current_cuda_allocated_bytes": 4, "current_cuda_reserved_bytes": 8,
            "peak_cuda_allocated_bytes": 4, "peak_cuda_reserved_bytes": 8,
            "selected_device": _profile(),
        }

    def close(self, runtime):
        self.events.append("close")


class _FailingPrepareBackend(_SuccessfulFullGateBackend):
    def prepare(self, *, root, identity, flags):
        self.events.append("prepare")
        raise RuntimeError("synthetic V5 prepare failure")


def test_v5_complete_full_gate_lifecycle_publishes_all_budget_pairs_then_terminal(tmp_path: Path) -> None:
    _stage_current_closure(tmp_path)
    smoke, gate = _write_v4_predecessors(tmp_path)
    identity = _identity(tmp_path, smoke, gate)
    environment = {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    capability = v5._issue_root_reviewed_capability(
        tmp_path, identity, source_data_root=None, seal=v5._ROOT_REVIEW_SEAL, environ=environment,
    )
    backend = _SuccessfulFullGateBackend()
    result = v5.execute_authorized(tmp_path, identity=identity, capability=capability, backend=backend, environ=environment)
    assert result["status"] == "PASS_SOURCE_CONSTRUCTIBLE"
    assert backend.events == [
        "preflight", "prepare", "source_authority", "budget:30", "budget:10", "budget:4",
        "revalidate", "resources", "close",
    ]
    output = tmp_path / identity.spec.root_relative
    bodies = sorted(path.name for path in output.iterdir() if path.suffix == ".json")
    assert len(bodies) == 3 + 3 * (len(identity.strict_train_roster) + 1) + 1
    assert "failure.json" not in bodies and "terminal.json" in bodies
    terminal = json.loads((output / "terminal.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "PASS_SOURCE_CONSTRUCTIBLE"
    assert terminal["launch_closure_sha256"] == terminal["final_closure_sha256"] == identity.closure["closure_sha256"]
    assert terminal["v5_binding"] == v5._v5_binding(identity)
    for name in bodies:
        body = (output / name).read_bytes()
        assert (output / f"{name}.sha256").read_bytes() == f"{_sha(body)}  {name}\n".encode("ascii")


def test_v5_failure_is_honest_and_predecessors_environment_closure_are_rechecked(tmp_path: Path) -> None:
    _stage_current_closure(tmp_path)
    smoke, gate = _write_v4_predecessors(tmp_path)
    identity = _identity(tmp_path, smoke, gate)
    environment = {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    capability = v5._issue_root_reviewed_capability(
        tmp_path, identity, source_data_root=None, seal=v5._ROOT_REVIEW_SEAL, environ=environment,
    )
    with pytest.raises(RuntimeError, match="synthetic V5 prepare failure"):
        v5.execute_authorized(
            tmp_path, identity=identity, capability=capability, backend=_FailingPrepareBackend(), environ=environment,
        )
    output = tmp_path / identity.spec.root_relative
    assert {path.name for path in output.iterdir()} == {
        "attempt.json", "attempt.json.sha256", "launch.json", "launch.json.sha256",
        "failure.json", "failure.json.sha256",
    }
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["stage"] == "prepare" and failure["source_authority_sha256"] is None
    assert failure["v5_binding"] == v5._v5_binding(identity)
    with pytest.raises(v5.SourceExecutionV5Error, match="CUDA_VISIBLE_DEVICES"):
        v5._issue_root_reviewed_capability(
            tmp_path, identity, source_data_root=None, seal=v5._ROOT_REVIEW_SEAL,
            environ={"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
        )


def test_v5_factory_is_exact_and_only_replaces_finalized_row_seam(tmp_path: Path) -> None:
    source_capability = v1._issue_root_reviewed_source_data_capability(
        canonical_root=tmp_path / "strict_source", strict_train_roster=_roster(), seal=v1._ROOT_REVIEW_SEAL,
    )
    backend = physical_v5.build_reviewed_physical_backend(
        root=ROOT, source_data=source_capability, selected_device=_profile(),
    )
    assert type(backend.provider) is physical_v4.V4ThetaStrict27SessionProvider
    assert type(backend.executor) is physical_v5.V5OneShotFinalizedRowExecutor
    assert isinstance(backend, physical_v4.PhysicalSourceExecutionBackendV4)
    assert isinstance(backend.executor, physical_v4.V4IndependentActivityCellDFourGroupExecutor)
    assert "monkeypatch" not in inspect.getsource(physical_v5).lower()
    assert not (tmp_path / "strict_source").exists()


def test_v5_static_cli_is_torch_free_dry_and_public_execution_fails_closed() -> None:
    script = ROOT / "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_gate_v5.py"
    environment = {
        **os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
    }
    dry = subprocess.run(
        [sys.executable, "-I", str(script), "--dry-run"], cwd=ROOT, text=True,
        capture_output=True, check=True, env=environment,
    )
    payload = json.loads(dry.stdout)
    assert payload["workorder_sha256"] == v5.WORKORDER_SHA256
    assert payload["opens_source"] is payload["loads_checkpoint"] is payload["initializes_cuda"] is False
    blocked = subprocess.run(
        [sys.executable, "-I", str(script), "--execute", "--source-gate"], cwd=ROOT,
        text=True, capture_output=True, env=environment,
    )
    assert blocked.returncode != 0 and "root-reviewed in-process V5 capability" in blocked.stderr
    probe = (
        "import pathlib,sys; root=pathlib.Path.cwd()/'tfpd_exploration'; "
        "sys.path.insert(0,str(root)); import src.causal_dual_memory_cell_d_v1.source_execute_v5; "
        "print('torch' in sys.modules)"
    )
    imported = subprocess.run(
        [sys.executable, "-I", "-c", probe], cwd=ROOT, text=True,
        capture_output=True, check=True, env=environment,
    )
    assert imported.stdout.strip() == "False"
    assert "import torch" not in script.read_text(encoding="utf-8")
