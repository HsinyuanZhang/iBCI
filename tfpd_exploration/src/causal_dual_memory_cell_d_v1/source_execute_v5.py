"""CDM-D Source Execution V5 finalized-row successor.

V5 is a fresh, source-only gate lifecycle.  It preserves V3/V4 science and
physical loading while binding both immutable V4 graphs that establish why a
new root is necessary: the accepted M10 smoke and the failed full gate.  Its
only physical change lives in :mod:`source_execute_physical_v5`: retain the
raw K=4 event exactly once, let the unbound V1 base build the audit/B8 row,
then append V3's independent-activity trace from the retained raw event.

Importing this module is static.  It neither imports Torch nor opens source,
checkpoint, predecessor-result, or CUDA resources.  Those actions remain
behind an opaque in-process root capability.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import source_execute as v1
from . import source_execute_v2 as v2
from . import source_execute_v3 as v3
from . import source_execute_v4 as v4


CELL = v1.CELL
V5_ROUTE = "causal_dual_memory_cell_d_source_execution_v5_finalized_row_successor"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CDM_D_SOURCE_EXECUTION_V5_FINALIZED_ROW_SUCCESSOR_20260825.md"
WORKORDER_SHA256 = "503fa0276b3bb32fd31b9da51dca961d1214dfef0a0d7c40cd5e8bec3e87cb9b"

SOURCE_GATE_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_source_gate_v5"
V4_SMOKE_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_source_smoke_v4"
V4_GATE_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_source_gate_v4"
V4_CLOSURE_SHA256 = "7854b667abfdec7cb004eff64e52c9f662e5eed293d7f0faef9d4fc45a094200"
V3_INDEPENDENT_ACTIVITY_CONTRACT = dict(v3.INDEPENDENT_ACTIVITY_CONTRACT)


class SourceExecutionV5Error(v1.SourceExecutionError):
    """Fail closed for V5 lineage, closure, or lifecycle drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceExecutionV5Error(message)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: object, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be an exact lowercase SHA-256",
    )
    return value


@dataclass(frozen=True)
class SourceExecutionV5Spec:
    """The single V5 full-gate run specification.

    V5 deliberately has no smoke root: the accepted V4 M10 smoke already
    exercised the unchanged executor/state-machine path.  The V5 repair is a
    full-gate finalized-row composition boundary.
    """

    kind: str
    root_relative: str
    smoke_session: str | None = None
    smoke_budget: int | None = None
    smoke_audit_positions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _require(self.kind == "source_gate", "V5 permits only the full strict-27 source gate")
        try:
            v1._safe_relative(self.root_relative)
        except v1.SourceExecutionError as error:
            raise SourceExecutionV5Error(str(error)) from error
        _require(
            self.smoke_session is None and self.smoke_budget is None and self.smoke_audit_positions == (),
            "V5 full gate may not carry a redundant smoke specification",
        )

    def payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "root_relative": self.root_relative,
            "source_only": True,
            "target_optimizer_steps": 0,
            "within_external_formal_target_forbidden": True,
            "fail_fast_budget_order": list(v1.FAIL_FAST_BUDGET_ORDER),
            "breadth_min_passing_sessions": v1.BREADTH_MIN_PASSING_SESSIONS,
            "smoke": None,
        }


SOURCE_GATE_SPEC = SourceExecutionV5Spec("source_gate", SOURCE_GATE_ROOT_RELATIVE)


@dataclass(frozen=True)
class V4SmokePredecessorExpectation:
    """The accepted V4 smoke graph held before any V5 action."""

    root_relative: str
    attempt_sha256: str
    launch_sha256: str
    source_authority_sha256: str
    smoke_sha256: str
    terminal_sha256: str
    identity_sha256: str
    v4_closure_sha256: str

    def __post_init__(self) -> None:
        try:
            v1._safe_relative(self.root_relative)
        except v1.SourceExecutionError as error:
            raise SourceExecutionV5Error(str(error)) from error
        for label, value in (
            ("V4 smoke attempt", self.attempt_sha256),
            ("V4 smoke launch", self.launch_sha256),
            ("V4 smoke authority", self.source_authority_sha256),
            ("V4 smoke evidence", self.smoke_sha256),
            ("V4 smoke terminal", self.terminal_sha256),
            ("V4 smoke identity", self.identity_sha256),
            ("V4 smoke closure", self.v4_closure_sha256),
        ):
            _sha(value, label)

    @property
    def body_sha256s(self) -> dict[str, str]:
        return {
            "attempt.json": self.attempt_sha256,
            "launch.json": self.launch_sha256,
            "source_authority.json": self.source_authority_sha256,
            "smoke.json": self.smoke_sha256,
            "terminal.json": self.terminal_sha256,
        }

    def payload(self) -> dict[str, object]:
        return {
            "schema": "causal_dual_memory_cell_d_source_execution_v4_smoke_predecessor_v5",
            "root_relative": self.root_relative,
            "leaves": [
                {"name": name, "sha256": digest, "mode": 0o444}
                for name, digest in self.body_sha256s.items()
            ],
            "exact_leaf_count": 10,
            "identity_sha256": self.identity_sha256,
            "v4_implementation_closure_sha256": self.v4_closure_sha256,
            "smoke_contract": {
                "status": "SMOKE_COMPLETED",
                "budget": 10,
                "support_positions": list(range(10)),
                "audit_positions": [10, 11],
                "activity_transition_committed_count": 2,
                "carrier_transition_committed_count": 0,
                "carrier_transition_rejected_count": 2,
                "source_only": True,
                "target_optimizer_backward_update": 0,
            },
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(_json_bytes(self.payload()))


@dataclass(frozen=True)
class V4GateFailurePredecessorExpectation:
    """The immutable V4 gate failure that V5 repairs without retrying."""

    root_relative: str
    attempt_sha256: str
    launch_sha256: str
    source_authority_sha256: str
    failure_sha256: str
    identity_sha256: str
    v4_closure_sha256: str
    error_sha256: str

    def __post_init__(self) -> None:
        try:
            v1._safe_relative(self.root_relative)
        except v1.SourceExecutionError as error:
            raise SourceExecutionV5Error(str(error)) from error
        for label, value in (
            ("V4 gate attempt", self.attempt_sha256),
            ("V4 gate launch", self.launch_sha256),
            ("V4 gate authority", self.source_authority_sha256),
            ("V4 gate failure", self.failure_sha256),
            ("V4 gate identity", self.identity_sha256),
            ("V4 gate closure", self.v4_closure_sha256),
            ("V4 gate error", self.error_sha256),
        ):
            _sha(value, label)

    @property
    def body_sha256s(self) -> dict[str, str]:
        return {
            "attempt.json": self.attempt_sha256,
            "launch.json": self.launch_sha256,
            "source_authority.json": self.source_authority_sha256,
            "failure.json": self.failure_sha256,
        }

    def payload(self) -> dict[str, object]:
        return {
            "schema": "causal_dual_memory_cell_d_source_execution_v4_gate_failure_predecessor_v5",
            "root_relative": self.root_relative,
            "leaves": [
                {"name": name, "sha256": digest, "mode": 0o444}
                for name, digest in self.body_sha256s.items()
            ],
            "exact_leaf_count": 8,
            "identity_sha256": self.identity_sha256,
            "v4_implementation_closure_sha256": self.v4_closure_sha256,
            "failure_contract": {
                "status": "FAILED",
                "stage": "budget_m30",
                "error_class": "SourceExecutionPhysicalV3Error",
                "error_sha256": self.error_sha256,
                "source_opened": True,
                "checkpoint_opened": True,
                "cuda_initialized": True,
                "model_forward_calls": 8,
                "backward_calls": 0,
                "optimizer_steps": 0,
                "parameter_updates": 0,
                "target_optimizer_backward_update": 0,
                "terminal_published": False,
            },
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(_json_bytes(self.payload()))


V4_ACCEPTED_SMOKE = V4SmokePredecessorExpectation(
    root_relative=V4_SMOKE_ROOT_RELATIVE,
    attempt_sha256="c9dfd75b173d92c95d85e77327a2f02e1190b91afb594aada245a7ed691c8ca9",
    launch_sha256="c8c7c1e72b5904286ee5a3c139e5b44d952af153ee29aa4b23f2f28019074736",
    source_authority_sha256="2d473889d0d206e43834f34ef814dd0511d34352e6252d9c67549e425c23d42e",
    smoke_sha256="bef1637ea02b54f6f51ea5c0516b3322eeb2d9e29664702e874bd69591a3c02f",
    terminal_sha256="305499831e169670e3ff0988a52ae8d3425960daf250e028c7182e5e3a8c73ca",
    identity_sha256="38291a4af71d0ef03b9060a59f49ae1df194375d64d8bbb3d6c14a01907e6f50",
    v4_closure_sha256=V4_CLOSURE_SHA256,
)

V4_FAILED_GATE = V4GateFailurePredecessorExpectation(
    root_relative=V4_GATE_ROOT_RELATIVE,
    attempt_sha256="8adff90f9686c6aaa9e1ba7306fee3e417009e0e74b6e09ca82182031cdb8da9",
    launch_sha256="45de7f3d0df03f33aef240fb8214559c434108cb0d0f2f9f96047d323d8c03a2",
    source_authority_sha256="7264c4c5a57f7edc8fecd1f7144f4603cdb4f92a745fc8cfccf46afbcb5e8324",
    failure_sha256="7105726660dbc481d791df7a2e63fa8870c8dd66dc28be02f841dbb3fa05720e",
    identity_sha256="4c2f6fa1a1359b0f393340f37de90786de3fd72ab538cefdb45dfcf69e95cbe4",
    v4_closure_sha256=V4_CLOSURE_SHA256,
    error_sha256="8e918cf7fad3d75ce559664025520b92f12bb3591698d8937ab2602c315dbb04",
)


# This intentionally remains a literal, no-glob closure.  V5 executes V4's
# closure-loader seams, V3's independent state machine, and V1's unbound
# finalized-row implementation, so every inherited runtime byte is explicit.
_V5_INHERITED_PATHS: tuple[str, ...] = (
    "tfpd_exploration/docs/WORKORDER_CDM_D_STAGE0_20260825.md",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/__init__.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/plan.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/core.py",
    "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_stage0.py",
    "tfpd_exploration/tests/test_causal_dual_memory_cell_d_stage0.py",
    "sua_exploration/mc_maze/d_optimal_calibration_design.py",
    "tfpd_exploration/src/calibration_budget_comparators_v1.py",
    "tfpd_exploration/docs/WORKORDER_CDM_D_SOURCE_AUDIT_20260825.md",
    "sua_exploration/mc_maze/pseudo_label_carrier_gate.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_adapter.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_audit.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/physical.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/lifecycle.py",
    "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_audit.py",
    "tfpd_exploration/tests/test_causal_dual_memory_cell_d_source_audit.py",
    "tfpd_exploration/docs/WORKORDER_CDM_D_SOURCE_EXECUTION_V1_20260825.md",
    "tfpd_exploration/src/__init__.py",
    "tfpd_exploration/src/cell_d_equal_session_v1.py",
    "tfpd_exploration/src/posterior_marginalized_cell_d_v1/__init__.py",
    "tfpd_exploration/src/posterior_marginalized_cell_d_v1/plan.py",
    "tfpd_exploration/src/posterior_carrier_v1/__init__.py",
    "tfpd_exploration/src/posterior_carrier_v1/plan.py",
    "tfpd_exploration/src/posterior_carrier_v1/core.py",
    "tfpd_exploration/src/posterior_carrier_v1/phase_b.py",
    "tfpd_exploration/src/posterior_carrier_v1/source_adapter.py",
    "tfpd_exploration/src/posterior_carrier_v1/source_adapter_v2.py",
    "tfpd_exploration/src/tfpd_lane/arm_common.py",
    "tfpd_exploration/src/tfpd_lane/pop_robust.py",
    "sua_exploration/mc_maze/__init__.py",
    "sua_exploration/mc_maze/a2_matched_subject_shift_v2_core.py",
    "sua_exploration/mc_maze/datamodule.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "sua_exploration/mc_maze/unit_side_features.py",
    "streaming_calibration_exp/src/__init__.py",
    "streaming_calibration_exp/src/models/__init__.py",
    "streaming_calibration_exp/src/models/components/__init__.py",
    "streaming_calibration_exp/src/models/components/spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/rt_ld_gain.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical.py",
    "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_gate.py",
    "tfpd_exploration/tests/test_causal_dual_memory_cell_d_source_execution.py",
    "tfpd_exploration/docs/WORKORDER_CDM_D_SOURCE_EXECUTION_V2_20260825.md",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_v2.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical_v2.py",
    "tfpd_exploration/docs/WORKORDER_CDM_D_INDEPENDENT_ACTIVITY_SUCCESSOR_V3_20260825.md",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_v3.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical_v3.py",
    "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_gate_v3.py",
    "tfpd_exploration/tests/test_causal_dual_memory_cell_d_source_execution_v3.py",
    "tfpd_exploration/docs/WORKORDER_CDM_D_SOURCE_EXECUTION_V4_LOADER_SUCCESSOR_20260825.md",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_v4.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical_v4.py",
    "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_gate_v4.py",
    "tfpd_exploration/tests/test_causal_dual_memory_cell_d_source_execution_v4.py",
)

_V5_OWNED_PATHS: tuple[str, ...] = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_v5.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical_v5.py",
    "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_gate_v5.py",
    "tfpd_exploration/tests/test_causal_dual_memory_cell_d_source_execution_v5.py",
)


def _regular_sha256(root: Path, relative: str) -> str:
    try:
        return v1._regular_sha256(Path(root), relative)
    except v1.SourceExecutionError as error:
        raise SourceExecutionV5Error(str(error)) from error


def execution_closure_payload(
    root: Path,
    *,
    accepted_smoke: V4SmokePredecessorExpectation = V4_ACCEPTED_SMOKE,
    failed_gate: V4GateFailurePredecessorExpectation = V4_FAILED_GATE,
) -> dict[str, object]:
    """Rebuild the exact V5 closure from its literal path topology."""

    _require(isinstance(accepted_smoke, V4SmokePredecessorExpectation), "V5 smoke expectation type drift")
    _require(isinstance(failed_gate, V4GateFailurePredecessorExpectation), "V5 gate expectation type drift")
    paths: list[str] = []
    for relative in (*_V5_INHERITED_PATHS, *_V5_OWNED_PATHS):
        if relative not in paths:
            paths.append(relative)
    rows = [{"path": relative, "sha256": _regular_sha256(Path(root), relative)} for relative in paths]
    _require(
        next((row["sha256"] for row in rows if row["path"] == WORKORDER_RELATIVE), None) == WORKORDER_SHA256,
        "V5 workorder SHA drift",
    )
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_closure_v5",
        "v4_accepted_smoke_predecessor_sha256": accepted_smoke.sha256,
        "v4_failed_gate_predecessor_sha256": failed_gate.sha256,
        "v4_implementation_closure_sha256": V4_CLOSURE_SHA256,
        "v3_independent_activity_contract": dict(V3_INDEPENDENT_ACTIVITY_CONTRACT),
        "paths": rows,
        "closure_sha256": sha256_bytes(_json_bytes(rows)),
    }


def _json_mapping(body: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(body)
    except json.JSONDecodeError as error:
        raise SourceExecutionV5Error(f"V4 predecessor {label} is not JSON") from error
    _require(isinstance(value, Mapping), f"V4 predecessor {label} JSON root drift")
    return dict(value)


def _read_held_pairs(
    root: Path,
    *,
    root_relative: str,
    body_sha256s: Mapping[str, str],
    label: str,
) -> tuple[dict[str, dict[str, object]], tuple[int, int]]:
    """Read one exact predecessor graph through one held directory FD."""

    relative = v1._safe_relative(root_relative)
    try:
        fd, held_identity = v1._open_held_directory(Path(root), relative)
    except v1.SourceExecutionError as error:
        raise SourceExecutionV5Error(str(error)) from error
    try:
        expected = {name: _sha(digest, f"{label} expected {name} SHA") for name, digest in body_sha256s.items()}
        _require(
            all(type(name) is str and Path(name).name == name and name.endswith(".json") for name in expected),
            f"{label} predecessor body-name topology drift",
        )
        try:
            names = set(os.listdir(fd))
        except OSError as error:
            raise SourceExecutionV5Error(f"{label} predecessor directory enumeration failed") from error
        expected_names = set(expected) | {f"{name}.sha256" for name in expected}
        _require(names == expected_names, f"{label} predecessor exact pair topology drift")
        payloads: dict[str, dict[str, object]] = {}
        for name, digest in expected.items():
            try:
                body = v1._read_held_leaf(fd, name, mode=0o444)
                sidecar = v1._read_held_leaf(fd, f"{name}.sha256", mode=0o444)
            except v1.SourceExecutionError as error:
                raise SourceExecutionV5Error(str(error)) from error
            _require(sha256_bytes(body) == digest, f"{label} predecessor {name} body SHA drift")
            _require(sidecar == f"{digest}  {name}\n".encode("ascii"),
                     f"{label} predecessor {name} canonical sidecar drift")
            payloads[name] = _json_mapping(body, f"{label} {name}")
        named_after = os.lstat(Path(root).absolute() / relative)
        held_after = os.fstat(fd)
        _require(
            stat.S_ISDIR(named_after.st_mode)
            and not stat.S_ISLNK(named_after.st_mode)
            and (int(named_after.st_dev), int(named_after.st_ino)) == held_identity
            and (int(held_after.st_dev), int(held_after.st_ino)) == held_identity,
            f"{label} predecessor named/held directory identity drift",
        )
        return payloads, held_identity
    finally:
        os.close(fd)


def _identity_from_predecessor(
    payloads: Mapping[str, Mapping[str, object]],
    *,
    expected_identity_sha256: str,
    expected_closure_sha256: str,
    label: str,
) -> dict[str, object]:
    attempt = payloads.get("attempt.json")
    _require(isinstance(attempt, Mapping) and isinstance(attempt.get("identity"), Mapping),
             f"{label} predecessor attempt identity missing")
    identity = dict(attempt["identity"])
    _require(sha256_bytes(_json_bytes(identity)) == expected_identity_sha256,
             f"{label} predecessor identity SHA drift")
    closure = identity.get("closure")
    _require(isinstance(closure, Mapping) and closure.get("closure_sha256") == expected_closure_sha256,
             f"{label} predecessor identity closure drift")
    for name, payload in payloads.items():
        # V4 attempt/launch/terminal bodies carry the complete identity;
        # source-authority and evidence bodies intentionally carry only the
        # V4 durable binding so they never duplicate a mutable identity blob.
        if name in {"attempt.json", "launch.json", "terminal.json", "failure.json"}:
            _require(payload.get("identity") == identity, f"{label} predecessor {name} identity graph drift")
            continue
        binding = payload.get("v4_binding")
        _require(
            isinstance(binding, Mapping)
            and binding.get("identity_sha256") == expected_identity_sha256
            and binding.get("v4_closure_sha256") == expected_closure_sha256,
            f"{label} predecessor {name} durable identity binding drift",
        )
    return identity


def _validate_v4_smoke_semantics(
    payloads: Mapping[str, Mapping[str, object]], expectation: V4SmokePredecessorExpectation,
) -> None:
    identity = _identity_from_predecessor(
        payloads,
        expected_identity_sha256=expectation.identity_sha256,
        expected_closure_sha256=expectation.v4_closure_sha256,
        label="V4 accepted smoke",
    )
    attempt = payloads["attempt.json"]
    launch = payloads["launch.json"]
    authority = payloads["source_authority.json"]
    smoke = payloads["smoke.json"]
    terminal = payloads["terminal.json"]
    _require(
        attempt.get("schema") == "causal_dual_memory_cell_d_source_execution_attempt_v4"
        and attempt.get("status") == "ATTEMPT_RESERVED"
        and attempt.get("source_only") is True
        and attempt.get("source_resolved_or_opened") is False
        and attempt.get("checkpoint_opened") is False
        and attempt.get("cuda_initialized") is False,
        "V4 accepted smoke attempt schema/boundary drift",
    )
    _require(
        launch.get("schema") == "causal_dual_memory_cell_d_source_execution_launch_v4"
        and launch.get("status") == "LAUNCHED"
        and launch.get("attempt_sha256") == expectation.attempt_sha256
        and launch.get("launch_closure_sha256") == expectation.v4_closure_sha256,
        "V4 accepted smoke launch graph drift",
    )
    access = authority.get("access")
    _require(
        authority.get("schema") == "causal_dual_memory_cell_d_source_execution_authority_v1"
        and authority.get("identity_sha256") == expectation.identity_sha256
        and authority.get("source_only") is True
        and isinstance(access, Mapping)
        and access.get("source_opened") is True
        and access.get("within_opened") is False
        and access.get("external_opened") is False
        and access.get("formal_opened") is False
        and access.get("target_opened") is False
        and access.get("backward_calls") == access.get("optimizer_steps") == access.get("parameter_updates") == 0,
        "V4 accepted smoke source-authority boundary drift",
    )
    _require(
        smoke.get("schema") == "causal_dual_memory_cell_d_source_execution_smoke_v3"
        and smoke.get("session") == v1.SOURCE_SMOKE_SESSION
        and smoke.get("budget") == 10
        and smoke.get("support_positions") == list(range(10))
        and smoke.get("audit_positions") == [10, 11]
        and smoke.get("activity_transition_committed_count") == 2
        and smoke.get("carrier_transition_committed_count") == 0
        and smoke.get("carrier_transition_rejected_count") == 2
        and smoke.get("source_only") is True,
        "V4 accepted smoke M10 contract drift",
    )
    audit_ids = smoke.get("audit_trial_ids")
    _require(isinstance(audit_ids, list) and len(audit_ids) == 2 and all(isinstance(item, str) for item in audit_ids),
             "V4 accepted smoke audit-ID topology drift")
    try:
        v3._validate_transition_trace(
            smoke.get("v3_independent_activity_transitions"), budget=10,
            expected_trial_ids=tuple(audit_ids), require_offline_m30=False, smoke=True,
        )
    except v3.SourceExecutionV3Error as error:
        raise SourceExecutionV5Error(str(error)) from error
    trace = smoke["v3_independent_activity_transitions"]
    _require(
        all(
            row.get("activity_transition_committed") is True
            and row.get("carrier_transition_committed") is False
            and row.get("activity_before_sha256") != row.get("activity_after_sha256")
            and row.get("carrier_before_sha256") == row.get("carrier_after_sha256")
            for row in trace
        ),
        "V4 accepted smoke activity/carrier separation drift",
    )
    _require(
        terminal.get("schema") == "causal_dual_memory_cell_d_source_execution_terminal_v4"
        and terminal.get("status") == "SMOKE_COMPLETED"
        and terminal.get("attempt_sha256") == expectation.attempt_sha256
        and terminal.get("launch_sha256") == expectation.launch_sha256
        and terminal.get("source_authority_sha256") == expectation.source_authority_sha256
        and terminal.get("evidence_sha256s") == {"smoke.json": expectation.smoke_sha256}
        and terminal.get("launch_closure_sha256") == expectation.v4_closure_sha256
        and terminal.get("final_closure_sha256") == expectation.v4_closure_sha256
        and terminal.get("source_only") is True
        and terminal.get("target_optimizer_backward_update") == 0
        and identity.get("closure", {}).get("closure_sha256") == expectation.v4_closure_sha256,
        "V4 accepted smoke terminal graph drift",
    )


def _validate_v4_gate_failure_semantics(
    payloads: Mapping[str, Mapping[str, object]], expectation: V4GateFailurePredecessorExpectation,
) -> None:
    _identity_from_predecessor(
        payloads,
        expected_identity_sha256=expectation.identity_sha256,
        expected_closure_sha256=expectation.v4_closure_sha256,
        label="V4 failed gate",
    )
    attempt = payloads["attempt.json"]
    launch = payloads["launch.json"]
    authority = payloads["source_authority.json"]
    failure = payloads["failure.json"]
    _require(
        attempt.get("schema") == "causal_dual_memory_cell_d_source_execution_attempt_v4"
        and attempt.get("status") == "ATTEMPT_RESERVED"
        and attempt.get("source_only") is True
        and attempt.get("source_resolved_or_opened") is False
        and attempt.get("checkpoint_opened") is False
        and attempt.get("cuda_initialized") is False,
        "V4 failed gate attempt schema/boundary drift",
    )
    _require(
        launch.get("schema") == "causal_dual_memory_cell_d_source_execution_launch_v4"
        and launch.get("status") == "LAUNCHED"
        and launch.get("attempt_sha256") == expectation.attempt_sha256
        and launch.get("launch_closure_sha256") == expectation.v4_closure_sha256,
        "V4 failed gate launch graph drift",
    )
    access = authority.get("access")
    _require(
        authority.get("schema") == "causal_dual_memory_cell_d_source_execution_authority_v1"
        and authority.get("identity_sha256") == expectation.identity_sha256
        and authority.get("source_only") is True
        and isinstance(access, Mapping)
        and access.get("source_opened") is True
        and access.get("within_opened") is False
        and access.get("external_opened") is False
        and access.get("formal_opened") is False
        and access.get("target_opened") is False
        and access.get("backward_calls") == access.get("optimizer_steps") == access.get("parameter_updates") == 0,
        "V4 failed gate source-authority boundary drift",
    )
    flags = failure.get("flags")
    _require(
        failure.get("schema") == "causal_dual_memory_cell_d_source_execution_failure_v4"
        and failure.get("status") == "FAILED"
        and failure.get("attempt_sha256") == expectation.attempt_sha256
        and failure.get("launch_sha256") == expectation.launch_sha256
        and failure.get("source_authority_sha256") == expectation.source_authority_sha256
        and failure.get("stage") == "budget_m30"
        and failure.get("error_class") == "SourceExecutionPhysicalV3Error"
        and failure.get("error_sha256") == expectation.error_sha256
        and failure.get("terminal_published") is False
        and isinstance(flags, Mapping)
        and flags.get("source_opened") is True
        and flags.get("checkpoint_opened") is True
        and flags.get("cuda_initialized") is True
        and flags.get("model_forward_calls") == 8
        and flags.get("backward_calls") == flags.get("optimizer_steps") == flags.get("parameter_updates") == 0
        and flags.get("within_opened") is False
        and flags.get("external_opened") is False
        and flags.get("formal_opened") is False
        and flags.get("target_opened") is False,
        "V4 failed gate finalized-row failure boundary drift",
    )


def validate_v4_predecessors(
    root: Path,
    *,
    accepted_smoke: V4SmokePredecessorExpectation = V4_ACCEPTED_SMOKE,
    failed_gate: V4GateFailurePredecessorExpectation = V4_FAILED_GATE,
) -> dict[str, object]:
    """Hold and exact-validate both V4 graphs before V5 can write anything."""

    _require(isinstance(accepted_smoke, V4SmokePredecessorExpectation), "V5 smoke expectation type drift")
    _require(isinstance(failed_gate, V4GateFailurePredecessorExpectation), "V5 gate expectation type drift")
    smoke_payloads, smoke_held = _read_held_pairs(
        Path(root), root_relative=accepted_smoke.root_relative,
        body_sha256s=accepted_smoke.body_sha256s, label="V4 accepted smoke",
    )
    _validate_v4_smoke_semantics(smoke_payloads, accepted_smoke)
    gate_payloads, gate_held = _read_held_pairs(
        Path(root), root_relative=failed_gate.root_relative,
        body_sha256s=failed_gate.body_sha256s, label="V4 failed gate",
    )
    _validate_v4_gate_failure_semantics(gate_payloads, failed_gate)
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_v4_predecessors_validated_v5",
        "accepted_v4_smoke": accepted_smoke.payload(),
        "failed_v4_gate": failed_gate.payload(),
        "accepted_v4_smoke_held_directory": {"device": smoke_held[0], "inode": smoke_held[1]},
        "failed_v4_gate_held_directory": {"device": gate_held[0], "inode": gate_held[1]},
    }


@dataclass(frozen=True)
class SourceExecutionV5Identity:
    """Typed V5 identity with V1-compatible physical fields."""

    spec: SourceExecutionV5Spec
    closure: Mapping[str, object]
    strict_train_roster: tuple[str, ...]
    fixed_assets: Mapping[str, Mapping[str, object]]
    normalizers: v1.SourceNormalizers
    selected_device: Mapping[str, object]
    accepted_v4_smoke: V4SmokePredecessorExpectation = field(default_factory=lambda: V4_ACCEPTED_SMOKE)
    failed_v4_gate: V4GateFailurePredecessorExpectation = field(default_factory=lambda: V4_FAILED_GATE)
    independent_activity_contract: Mapping[str, object] = field(default_factory=lambda: dict(V3_INDEPENDENT_ACTIVITY_CONTRACT))

    def __post_init__(self) -> None:
        _require(isinstance(self.spec, SourceExecutionV5Spec), "V5 identity run-spec type drift")
        roster = tuple(self.strict_train_roster)
        _require(len(roster) == v1.STRICT_SOURCE_COUNT and len(set(roster)) == len(roster),
                 "V5 identity strict roster drift")
        _require(
            isinstance(self.closure, Mapping)
            and self.closure.get("schema") == "causal_dual_memory_cell_d_source_execution_closure_v5"
            and _sha(self.closure.get("closure_sha256"), "V5 identity closure") == self.closure.get("closure_sha256")
            and self.closure.get("v4_accepted_smoke_predecessor_sha256") == self.accepted_v4_smoke.sha256
            and self.closure.get("v4_failed_gate_predecessor_sha256") == self.failed_v4_gate.sha256
            and self.closure.get("v4_implementation_closure_sha256") == V4_CLOSURE_SHA256,
            "V5 identity closure/predecessor topology drift",
        )
        _require(dict(self.independent_activity_contract) == V3_INDEPENDENT_ACTIVITY_CONTRACT,
                 "V5 identity must preserve V3 independent-activity contract")
        _require(isinstance(self.normalizers, v1.SourceNormalizers) and self.normalizers == v1.SEALED_NORMALIZERS,
                 "V5 identity sealed ordinary normalizer drift")
        _require(v1.validate_compatible_device_profile(self.selected_device) == dict(self.selected_device),
                 "V5 selected device profile drift")
        _require(tuple(self.fixed_assets) == tuple(asset.label for asset in v1.FIXED_ASSETS),
                 "V5 fixed-asset topology drift")
        for asset in v1.FIXED_ASSETS:
            _require(isinstance(self.fixed_assets[asset.label], Mapping)
                     and dict(self.fixed_assets[asset.label]) == asset.payload(),
                     f"V5 fixed asset drift: {asset.label}")
        object.__setattr__(self, "strict_train_roster", roster)

    def payload(self) -> dict[str, object]:
        return {
            "schema": "causal_dual_memory_cell_d_source_execution_identity_v5",
            "cell": CELL,
            "run_spec": self.spec.payload(),
            "closure": dict(self.closure),
            "strict_train_roster": list(self.strict_train_roster),
            "strict_train_roster_sha256": v1.roster_sha256(self.strict_train_roster),
            "fixed_assets": {label: dict(value) for label, value in self.fixed_assets.items()},
            "normalizers": self.normalizers.payload(),
            "selected_device": dict(self.selected_device),
            "physical_backend_protocol": "v1_base_finalized_row_v5_raw_event_capture",
            "independent_activity_contract": dict(self.independent_activity_contract),
            "accepted_v4_smoke": self.accepted_v4_smoke.payload(),
            "failed_v4_gate": self.failed_v4_gate.payload(),
            "source_only": True,
            "within_external_formal_target_forbidden": True,
            "target_optimizer_backward_update": 0,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(_json_bytes(self.payload()))


def build_identity(
    root: Path,
    *,
    spec: SourceExecutionV5Spec = SOURCE_GATE_SPEC,
    selected_device: Mapping[str, object],
    fixed_assets: Mapping[str, v1.BoundAsset] | None = None,
) -> SourceExecutionV5Identity:
    """Descriptor-build identity without resolving source data or V4 roots."""

    _require(isinstance(spec, SourceExecutionV5Spec), "V5 identity requires typed V5 gate spec")
    assets = dict(v1.descriptor_read_fixed_assets(Path(root)) if fixed_assets is None else fixed_assets)
    _require(tuple(assets) == tuple(asset.label for asset in v1.FIXED_ASSETS),
             "V5 fixed-asset preflight topology drift")
    manifest = assets.get("strict_manifest")
    _require(isinstance(manifest, v1.BoundAsset), "V5 strict manifest fixed asset drift")
    return SourceExecutionV5Identity(
        spec=spec,
        closure=execution_closure_payload(Path(root)),
        strict_train_roster=v1.parse_strict_train_roster(manifest.body),
        fixed_assets={label: bound.asset.payload() for label, bound in assets.items()},
        normalizers=v1.SEALED_NORMALIZERS,
        selected_device=v1.validate_compatible_device_profile(selected_device),
    )


def validate_identity_current(root: Path, identity: SourceExecutionV5Identity) -> None:
    _require(isinstance(identity, SourceExecutionV5Identity), "V5 execution identity must be typed")
    _require(
        identity.closure == execution_closure_payload(
            Path(root), accepted_smoke=identity.accepted_v4_smoke, failed_gate=identity.failed_v4_gate,
        ),
        "V5 implementation closure drift",
    )
    _require(identity.normalizers == v1.SEALED_NORMALIZERS,
             "V5 sealed normalizer authority drift")
    _require(v1.validate_compatible_device_profile(identity.selected_device) == dict(identity.selected_device),
             "V5 selected device profile drift")


class _V5RootReviewSeal:
    pass


_ROOT_REVIEW_SEAL = _V5RootReviewSeal()


@dataclass(frozen=True)
class SourceExecutionV5Capability:
    identity_sha256: str
    source_data_root: v1.StrictSourceDataRootCapability | None = field(repr=False, compare=False)
    predecessor_sha256: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _sha(self.identity_sha256, "V5 capability identity SHA")
        _sha(self.predecessor_sha256, "V5 capability predecessor SHA")
        _require(
            self._seal is _ROOT_REVIEW_SEAL
            and (self.source_data_root is None or isinstance(self.source_data_root, v1.StrictSourceDataRootCapability)),
            "V5 capability provenance drift",
        )


def assert_prospective_root_fresh(root: Path, spec: SourceExecutionV5Spec) -> None:
    _require(isinstance(spec, SourceExecutionV5Spec), "V5 freshness requires typed V5 spec")
    candidate = Path(root).absolute() / v1._safe_relative(spec.root_relative)
    try:
        os.lstat(candidate)
    except FileNotFoundError:
        return
    except OSError as error:
        raise SourceExecutionV5Error("V5 prospective root cannot be inspected safely") from error
    raise SourceExecutionV5Error("V5 prospective source gate root already exists")


def _issue_root_reviewed_capability(
    root: Path,
    identity: SourceExecutionV5Identity,
    *,
    source_data_root: v1.StrictSourceDataRootCapability | None,
    seal: object,
    environ: Mapping[str, str] | None = None,
) -> SourceExecutionV5Capability:
    """Root-only V5 issuance: held V4 graphs, closure/env, then freshness."""

    _require(seal is _ROOT_REVIEW_SEAL, "only the V5 root reviewer may issue execution capability")
    predecessors = validate_v4_predecessors(
        Path(root), accepted_smoke=identity.accepted_v4_smoke, failed_gate=identity.failed_v4_gate,
    )
    validate_identity_current(Path(root), identity)
    try:
        v1.validate_selected_device_environment(identity.selected_device, environ)
    except v1.SourceExecutionError as error:
        raise SourceExecutionV5Error(str(error)) from error
    assert_prospective_root_fresh(Path(root), identity.spec)
    if source_data_root is not None:
        _require(source_data_root.strict_train_roster_sha256 == v1.roster_sha256(identity.strict_train_roster),
                 "V5 source-data capability roster drift")
    return SourceExecutionV5Capability(
        identity.sha256, source_data_root, sha256_bytes(_json_bytes(predecessors)), seal,
    )


def require_execution_capability(
    capability: object,
    identity: SourceExecutionV5Identity,
) -> SourceExecutionV5Capability:
    _require(
        isinstance(capability, SourceExecutionV5Capability)
        and capability._seal is _ROOT_REVIEW_SEAL
        and capability.identity_sha256 == identity.sha256,
        "V5 execution requires an exact in-process root-reviewed capability",
    )
    if capability.source_data_root is not None:
        _require(capability.source_data_root.strict_train_roster_sha256 == v1.roster_sha256(identity.strict_train_roster),
                 "V5 execution source-data capability roster drift")
    return capability


def _v5_binding(identity: SourceExecutionV5Identity) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_v5_binding",
        "identity_sha256": identity.sha256,
        "v5_closure_sha256": identity.closure["closure_sha256"],
        "accepted_v4_smoke": identity.accepted_v4_smoke.payload(),
        "failed_v4_gate": identity.failed_v4_gate.payload(),
        "v4_implementation_closure_sha256": V4_CLOSURE_SHA256,
        "v3_independent_activity_contract": dict(V3_INDEPENDENT_ACTIVITY_CONTRACT),
    }


def _validate_v5_binding(value: object, *, identity: SourceExecutionV5Identity) -> None:
    _require(isinstance(value, Mapping) and dict(value) == _v5_binding(identity),
             "V5 durable closure/predecessor binding drift")


def _enrich_payload(value: Mapping[str, object], *, identity: SourceExecutionV5Identity) -> dict[str, object]:
    result = dict(value)
    result["v5_binding"] = _v5_binding(identity)
    result["independent_activity_contract"] = dict(V3_INDEPENDENT_ACTIVITY_CONTRACT)
    return result


def _validate_source_authority_v5(
    value: Mapping[str, object], *, identity: SourceExecutionV5Identity, flags: v1.RuntimeFlags,
) -> None:
    try:
        v1._validate_source_authority(value, identity, flags)
        v4._validate_theta_proofs(value, identity=identity)  # V4 verifier is field-compatible, not type-coupled.
    except (v1.SourceExecutionError, v4.SourceExecutionV4Error) as error:
        raise SourceExecutionV5Error(str(error)) from error
    _validate_v5_binding(value.get("v5_binding"), identity=identity)
    _require(value.get("independent_activity_contract") == V3_INDEPENDENT_ACTIVITY_CONTRACT,
             "V5 source authority independent-activity contract drift")


def _validate_v5_evidence(
    value: Mapping[str, object], *, identity: SourceExecutionV5Identity, budget: int | None, require_trace: bool,
) -> None:
    _validate_v5_binding(value.get("v5_binding"), identity=identity)
    _require(value.get("independent_activity_contract") == V3_INDEPENDENT_ACTIVITY_CONTRACT,
             "V5 evidence independent-activity contract drift")
    if not require_trace:
        return
    _require(budget in (4, 10, 30), "V5 trace budget drift")
    if value.get("status") == "STOP_MISSING_REQUIRED_CHRONOLOGY":
        _require(value.get("v3_independent_activity_transitions") == [],
                 "V5 missing chronology may not fabricate an independent trace")
        return
    fixed_pool = value.get("fixed_pool_trial_ids")
    _require(isinstance(fixed_pool, list), "V5 evidence requires exact parent fixed-pool IDs")
    try:
        v3._validate_transition_trace(
            value.get("v3_independent_activity_transitions"), budget=int(budget),
            expected_trial_ids=tuple(fixed_pool), require_offline_m30=int(budget) == 30, smoke=False,
        )
    except v3.SourceExecutionV3Error as error:
        raise SourceExecutionV5Error(str(error)) from error


def _attempt_payload(identity: SourceExecutionV5Identity) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_attempt_v5",
        "cell": CELL,
        "status": "ATTEMPT_RESERVED",
        "identity": identity.payload(),
        "v5_binding": _v5_binding(identity),
        "source_only": True,
        "source_resolved_or_opened": False,
        "checkpoint_opened": False,
        "cuda_initialized": False,
        "target_optimizer_backward_update": 0,
        "within_external_formal_target_forbidden": True,
    }


def _launch_payload(
    identity: SourceExecutionV5Identity, attempt_sha256: str, preflight: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_launch_v5",
        "cell": CELL,
        "status": "LAUNCHED",
        "identity": identity.payload(),
        "attempt_sha256": _sha(attempt_sha256, "V5 launch attempt SHA"),
        "preflight": dict(preflight),
        "v5_binding": _v5_binding(identity),
        "launch_closure_sha256": identity.closure["closure_sha256"],
        "source_only": True,
        "target_optimizer_backward_update": 0,
    }


def _terminal_payload(
    identity: SourceExecutionV5Identity,
    *,
    attempt_sha256: str,
    launch_sha256: str,
    source_authority_sha256: str,
    evidence_sha256s: Mapping[str, str],
    status: str,
    resources: Mapping[str, object],
) -> dict[str, object]:
    try:
        v1._validate_resources(resources)
    except v1.SourceExecutionError as error:
        raise SourceExecutionV5Error(str(error)) from error
    _require(status in {"PASS_SOURCE_CONSTRUCTIBLE", "STOP_SOURCE_B8_CONSTRUCTIBILITY"},
             "V5 terminal status drift")
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_terminal_v5",
        "cell": CELL,
        "status": status,
        "identity": identity.payload(),
        "attempt_sha256": _sha(attempt_sha256, "V5 terminal attempt SHA"),
        "launch_sha256": _sha(launch_sha256, "V5 terminal launch SHA"),
        "source_authority_sha256": _sha(source_authority_sha256, "V5 terminal source authority SHA"),
        "evidence_sha256s": {name: _sha(digest, f"V5 terminal {name} SHA")
                               for name, digest in evidence_sha256s.items()},
        "resources": dict(resources),
        "v5_binding": _v5_binding(identity),
        "launch_closure_sha256": identity.closure["closure_sha256"],
        "final_closure_sha256": identity.closure["closure_sha256"],
        "source_only": True,
        "target_optimizer_backward_update": 0,
    }


def _failure_payload(
    identity: SourceExecutionV5Identity,
    *,
    attempt_sha256: str,
    launch_sha256: str | None,
    source_authority_sha256: str | None,
    flags: v1.RuntimeFlags,
    error: BaseException,
) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_failure_v5",
        "cell": CELL,
        "status": "FAILED",
        "identity": identity.payload(),
        "attempt_sha256": _sha(attempt_sha256, "V5 failure attempt SHA"),
        "launch_sha256": None if launch_sha256 is None else _sha(launch_sha256, "V5 failure launch SHA"),
        "source_authority_sha256": None if source_authority_sha256 is None else _sha(
            source_authority_sha256, "V5 failure authority SHA",
        ),
        "stage": flags.stage,
        "error_class": type(error).__name__,
        "error_sha256": sha256_bytes(repr(error).encode("utf-8")),
        "flags": flags.payload(),
        "v5_binding": _v5_binding(identity),
        "terminal_published": False,
        "source_only": True,
    }


def _validate_pre_execution(
    root: Path,
    identity: SourceExecutionV5Identity,
    capability: object,
    environ: Mapping[str, str] | None,
) -> SourceExecutionV5Capability:
    approved = require_execution_capability(capability, identity)
    predecessors = validate_v4_predecessors(
        Path(root), accepted_smoke=identity.accepted_v4_smoke, failed_gate=identity.failed_v4_gate,
    )
    _require(approved.predecessor_sha256 == sha256_bytes(_json_bytes(predecessors)),
             "V5 capability/predecessor held-graph drift")
    validate_identity_current(Path(root), identity)
    assert_prospective_root_fresh(Path(root), identity.spec)
    try:
        v1.validate_selected_device_environment(identity.selected_device, environ)
    except v1.SourceExecutionError as error:
        raise SourceExecutionV5Error(str(error)) from error
    return approved


def _validate_live_authorities(
    root: Path,
    identity: SourceExecutionV5Identity,
    capability: object,
    environ: Mapping[str, str] | None,
) -> SourceExecutionV5Capability:
    """Recheck immutable authorities after V5 owns its fresh artifact root."""

    approved = require_execution_capability(capability, identity)
    predecessors = validate_v4_predecessors(
        Path(root), accepted_smoke=identity.accepted_v4_smoke, failed_gate=identity.failed_v4_gate,
    )
    _require(approved.predecessor_sha256 == sha256_bytes(_json_bytes(predecessors)),
             "V5 live capability/predecessor held-graph drift")
    validate_identity_current(Path(root), identity)
    try:
        v1.validate_selected_device_environment(identity.selected_device, environ)
    except v1.SourceExecutionError as error:
        raise SourceExecutionV5Error(str(error)) from error
    return approved


def _validate_live_final_revalidation(
    root: Path,
    identity: SourceExecutionV5Identity,
    capability: object,
    artifact: v1.ArtifactRoot,
    environ: Mapping[str, str] | None,
    *,
    expected_published_sha256s: Mapping[str, str],
) -> SourceExecutionV5Capability:
    """Bind current authorities and the V5-owned published pair bytes exactly."""

    approved = _validate_live_authorities(Path(root), identity, capability, environ)
    _require(isinstance(expected_published_sha256s, Mapping),
             "V5 live expected-published SHA mapping must be a mapping")
    expected_pairs: dict[str, str] = {}
    for name, digest in expected_published_sha256s.items():
        _require(type(name) is str and Path(name).name == name and name.endswith(".json"),
                 "V5 live expected-published body name drift")
        _require(name not in expected_pairs, "V5 live expected-published duplicate body name")
        expected_pairs[name] = _sha(digest, f"V5 live expected {name} SHA")
    expected_directory = Path(root).absolute() / v1._safe_relative(identity.spec.root_relative)
    _require(artifact.directory == expected_directory, "V5 held artifact directory/spec root drift")
    try:
        artifact._revalidate()
        names = set(os.listdir(artifact.fd))
    except v1.SourceExecutionError as error:
        raise SourceExecutionV5Error(str(error)) from error
    except OSError as error:
        raise SourceExecutionV5Error("V5 held artifact enumeration failed") from error
    _require(set(artifact.published) == set(expected_pairs), "V5 live artifact publication topology drift")
    expected_names = set(expected_pairs) | {f"{name}.sha256" for name in expected_pairs}
    _require(names == expected_names, "V5 live artifact pair topology drift before terminal publication")
    for name in sorted(expected_pairs):
        try:
            payload = artifact.read_json_pair(name, expected_sha256=expected_pairs[name])
        except v1.SourceExecutionError as error:
            raise SourceExecutionV5Error(str(error)) from error
        _require(isinstance(payload, Mapping), "V5 live artifact JSON pair mapping drift")
    return approved


def execute_authorized(
    root: Path,
    *,
    identity: SourceExecutionV5Identity,
    capability: object,
    backend: v1.SourceExecutionBackend,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Execute the one V5 gate only after all no-write gates have passed."""

    _validate_pre_execution(Path(root), identity, capability, environ)
    flags = v1.RuntimeFlags(stage="preflight")
    preflight = backend.preflight(root=Path(root), identity=identity, flags=flags)
    _require(
        isinstance(preflight, Mapping)
        and preflight.get("source_resolved_or_opened") is False
        and preflight.get("checkpoint_opened") is False
        and preflight.get("cuda_initialized") is False,
        "V5 source-free preflight boundary drift",
    )
    artifact = v1.reserve_artifact_root(Path(root), spec=identity.spec, roster=identity.strict_train_roster)
    runtime: Any | None = None
    attempt_sha: str | None = None
    launch_sha: str | None = None
    authority_sha: str | None = None
    try:
        flags.stage = "attempt"
        attempt_sha = artifact.publish_json_pair("attempt.json", _attempt_payload(identity))
        flags.stage = "launch"
        launch_sha = artifact.publish_json_pair("launch.json", _launch_payload(identity, attempt_sha, preflight))
        flags.stage = "prepare"
        runtime = backend.prepare(root=Path(root), identity=identity, flags=flags)
        flags.stage = "source_authority"
        authority = _enrich_payload(backend.source_authority(runtime, identity=identity, flags=flags), identity=identity)
        _validate_source_authority_v5(authority, identity=identity, flags=flags)
        authority_sha = artifact.publish_json_pair("source_authority.json", authority)
        evidence_sha: dict[str, str] = {}
        final_status = "PASS_SOURCE_CONSTRUCTIBLE"
        for budget in v1.FAIL_FAST_BUDGET_ORDER:
            flags.stage = f"budget_m{budget}"
            rows = tuple(
                _enrich_payload(row, identity=identity)
                for row in backend.run_budget(runtime, budget=budget, identity=identity, flags=flags)
            )
            for session, row in zip(identity.strict_train_roster, rows, strict=True):
                try:
                    v1._validate_session_evidence(row, budget=budget, expected_session=session)
                except v1.SourceExecutionError as error:
                    raise SourceExecutionV5Error(str(error)) from error
                _validate_v5_evidence(row, identity=identity, budget=budget, require_trace=True)
                evidence_sha[f"budget_m{budget}__{session}.json"] = artifact.publish_json_pair(
                    f"budget_m{budget}__{session}.json", row,
                )
            try:
                aggregate = v1._aggregate_budget(rows, budget=budget, roster=identity.strict_train_roster)
            except v1.SourceExecutionError as error:
                raise SourceExecutionV5Error(str(error)) from error
            aggregate = _enrich_payload(aggregate, identity=identity)
            _validate_v5_evidence(aggregate, identity=identity, budget=None, require_trace=False)
            evidence_sha[f"budget_m{budget}_aggregate.json"] = artifact.publish_json_pair(
                f"budget_m{budget}_aggregate.json", aggregate,
            )
            if aggregate["breadth_pass"] is False:
                final_status = "STOP_SOURCE_B8_CONSTRUCTIBILITY"
                break
        flags.stage = "final_revalidation"
        _require(
            attempt_sha is not None and launch_sha is not None and authority_sha is not None,
            "V5 final revalidation requires mandatory publication SHA values",
        )
        _validate_live_final_revalidation(
            Path(root), identity, capability, artifact, environ,
            expected_published_sha256s={
                "attempt.json": attempt_sha,
                "launch.json": launch_sha,
                "source_authority.json": authority_sha,
                **evidence_sha,
            },
        )
        backend.revalidate(root=Path(root), identity=identity, flags=flags)
        resources = dict(backend.resources(runtime, flags=flags))
        flags.stage = "terminal"
        terminal_sha = artifact.publish_json_pair(
            "terminal.json",
            _terminal_payload(
                identity, attempt_sha256=attempt_sha, launch_sha256=launch_sha,
                source_authority_sha256=authority_sha, evidence_sha256s=evidence_sha,
                status=final_status, resources=resources,
            ),
        )
        return {
            "attempt_sha256": attempt_sha,
            "launch_sha256": launch_sha,
            "source_authority_sha256": authority_sha,
            "terminal_sha256": terminal_sha,
            "status": final_status,
            "flags": flags.payload(),
        }
    except BaseException as error:
        if attempt_sha is not None and "terminal.json" not in artifact.published:
            # The same immutable predecessor/closure/environment gates guard
            # failure publication; expected own-root existence is intentionally
            # not treated as a collision after reservation.
            _validate_live_authorities(Path(root), identity, capability, environ)
            artifact.publish_json_pair(
                "failure.json",
                _failure_payload(
                    identity, attempt_sha256=attempt_sha, launch_sha256=launch_sha,
                    source_authority_sha256=authority_sha, flags=flags, error=error,
                ),
            )
        raise
    finally:
        try:
            backend.close(runtime)
        finally:
            artifact.close()


def execute_reviewed_physical(
    root: Path,
    *,
    identity: SourceExecutionV5Identity,
    capability: object,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """The sole V5 physical entry point; the public CLI cannot reach it."""

    approved = _validate_pre_execution(Path(root), identity, capability, environ)
    source_data = approved.source_data_root
    _require(
        source_data is not None
        and source_data.strict_train_roster_sha256 == v1.roster_sha256(identity.strict_train_roster),
        "reviewed V5 physical route needs exact strict source-data capability",
    )
    from .source_execute_physical_v5 import build_reviewed_physical_backend

    backend = build_reviewed_physical_backend(
        root=Path(root), source_data=source_data, selected_device=identity.selected_device,
    )
    return execute_authorized(Path(root), identity=identity, capability=approved, backend=backend, environ=environ)


def dry_plan(root: Path | None = None) -> dict[str, object]:
    """Static public V5 plan; no source/checkpoint/CUDA/write action."""

    result: dict[str, object] = {
        "cell": CELL,
        "phase": "source_execution_v5_finalized_row_successor",
        "workorder_sha256": WORKORDER_SHA256,
        "accepted_v4_smoke": V4_ACCEPTED_SMOKE.payload(),
        "failed_v4_gate": V4_FAILED_GATE.payload(),
        "v3_scientific_contract": dict(V3_INDEPENDENT_ACTIVITY_CONTRACT),
        "source_gate_root_relative": SOURCE_GATE_ROOT_RELATIVE,
        "execution_authorized": False,
        "opens_source": False,
        "loads_checkpoint": False,
        "initializes_cuda": False,
        "creates_root": False,
        "scores": False,
        "launches": False,
    }
    if root is not None:
        result["closure"] = execution_closure_payload(Path(root))
    return result
