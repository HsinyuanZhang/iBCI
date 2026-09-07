"""CS-WG M1 V4 source-only GPU-smoke successor.

This additive route leaves V1/V2/V3 code and their immutable receipt roots
untouched.  It binds the accepted V3 common-stratum source audit through one
held no-follow descriptor chain, then delegates the physical 100-step loop to
the frozen V1 runner.  Importing this module is stdlib-only: no parser, Torch,
CUDA, source body, result root, or artifact root is touched until an explicit
root-reviewed execution capability reaches the deferred execution function.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from . import source_audit_v2 as v2
from . import source_audit_v3 as v3
from . import source_lifecycle as v1


class SourceSmokeV4Error(RuntimeError):
    """Fail closed for V4 predecessor, authority, or smoke lifecycle drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceSmokeV4Error(message)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64
             and all(character in "0123456789abcdef" for character in value),
             f"CS-WG V4 {label} must be a lowercase SHA-256")
    return value


def _safe_relative(value: object) -> str:
    _require(isinstance(value, str) and value, "CS-WG V4 relative path is absent")
    path = Path(value)
    _require(not path.is_absolute() and ".." not in path.parts and path.name not in {"", ".", ".."},
             "CS-WG V4 relative path is unsafe")
    return path.as_posix()


CELL = "CROSS_SESSION_WORST_GROUP_SPINT_M1_V1"
PHASE = "m1_source_smoke_v4_common_stratum_successor"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CS_WG_M1_SOURCE_SMOKE_V4_COMMON_STRATUM_SUCCESSOR_20260826.md"
WORKORDER_SHA256 = "c3ae0d1f5ea1f033419891f7a5217dc3cdc0011284b545d2ae5d150886b0fbe6"
V4_ROOT_RELATIVE = "tfpd_exploration/results/cross_session_worst_group_m1_source_smoke_v4"

V3_ROOT_RELATIVE = "tfpd_exploration/results/cross_session_worst_group_m1_source_audit_v3"
V3_ATTEMPT_SHA256 = "f63aec4a528495d937a514a6b2194f0240bfb97f48e95ad82b44a28cbb2e4287"
V3_LAUNCH_SHA256 = "04243c940627c4d9e745a3834f9cf58b3cce2bfeffba9ac0684b570e723f4c9c"
V3_SOURCE_AUTHORITY_SHA256 = "0f5c9e47113ca579c56d46282f2f4cc60e27bcbc452fd0b3cab89df352bacd0b"
V3_AUDIT_SHA256 = "ce64d34a2a4ddbf4f6825a7dfbec81eb05a8c4a5f93e1dd577ee597e7b2f16d9"
V3_TERMINAL_SHA256 = "2a27b02db39a4826f37b93dbcbe9b8c227fefa3b3c8c154136960bdf5f6e9230"
V3_CLOSURE_SHA256 = "d52168e567188b8ede816f4764cf829ecd920b2540c323fee14569ec7503fa1e"
V3_IDENTITY_SHA256 = "90892d8022be400b1b9e9046f003e86f81e9c48ae9ee892c4a93b99547c489ff"
V3_FAILURE_CHAIN_SHA256 = "47d588d2762276c49d9ff8ae1c63d9185fc5b8ebf1ede88480df5c3659328112"
V3_FALLBACK_SHA256 = "2e763763bf55276bfef175815ec7b6cb454393a87278513f86f8d99677618089"
V3_ELIGIBLE_COMMON_MIN2_COUNT = 43
V3_RETENTION = (
    ("20120926", 54_467, 54_476),
    ("20120927", 49_189, 49_228),
    ("20120928", 54_766, 54_783),
)

_OWNED_PATHS = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/cross_session_worst_group_v1/source_smoke_v4.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_smoke_physical_v4.py",
    "tfpd_exploration/scripts/run_cross_session_worst_group_m1_source_smoke_v4.py",
    "tfpd_exploration/tests/test_cross_session_worst_group_m1_source_smoke_v4.py",
)


def _read_regular_no_follow(root: Path, relative: str) -> str:
    path = Path(root).absolute() / _safe_relative(relative)
    try:
        info = os.lstat(path)
    except OSError as error:
        raise SourceSmokeV4Error(f"CS-WG V4 closure leaf inaccessible: {relative}") from error
    _require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode),
             f"CS-WG V4 closure leaf is not a regular non-symlink: {relative}")
    with open(path, "rb") as handle:
        return _sha(handle.read())


def implementation_closure(root: Path) -> dict[str, object]:
    """Rebuild V4's explicit no-glob execution closure.

    The accepted V3 closure remains an immutable predecessor.  V4 binds it
    verbatim and adds only its own five leaves, so a live stage cannot silently
    borrow an unlisted copy of the source reader, V3 fallback, or V1 runner.
    """
    try:
        inherited = v3.implementation_closure(Path(root))
    except v3.SourceAuditV3Error as error:
        raise SourceSmokeV4Error("CS-WG V4 inherited V3 closure drift") from error
    _require(inherited.get("closure_sha256") == V3_CLOSURE_SHA256,
             "CS-WG V4 accepted V3 closure literal/current-byte drift")
    rows_value = inherited.get("paths")
    _require(isinstance(rows_value, list) and rows_value, "CS-WG V4 inherited V3 closure topology drift")
    rows = [dict(item) for item in rows_value]
    existing = {item.get("path") for item in rows}
    for relative in _OWNED_PATHS:
        _require(relative not in existing, "CS-WG V4 closure path duplication")
        rows.append({"path": relative, "sha256": _read_regular_no_follow(Path(root), relative)})
    _require(rows[len(rows_value)]["sha256"] == WORKORDER_SHA256,
             "CS-WG V4 workorder literal/body drift")
    body = {
        "schema": "cross_session_worst_group_m1_source_smoke_v4_closure_v1",
        "accepted_v3_closure_sha256": V3_CLOSURE_SHA256,
        "accepted_v3_identity_sha256": V3_IDENTITY_SHA256,
        "paths": rows,
    }
    return {**body, "closure_sha256": _sha(_json_bytes(body))}


def validate_current_closure(root: Path, closure: Mapping[str, object]) -> None:
    _require(isinstance(closure, Mapping) and dict(closure) == implementation_closure(Path(root)),
             "CS-WG V4 closure/current-byte drift")


@dataclass(frozen=True)
class V3CompletedAuditGraphExpectation:
    """Exact accepted V3 completed graph, including its ten immutable leaves."""

    root_relative: str = V3_ROOT_RELATIVE
    attempt_sha256: str = V3_ATTEMPT_SHA256
    launch_sha256: str = V3_LAUNCH_SHA256
    source_authority_sha256: str = V3_SOURCE_AUTHORITY_SHA256
    audit_sha256: str = V3_AUDIT_SHA256
    terminal_sha256: str = V3_TERMINAL_SHA256

    def __post_init__(self) -> None:
        _require(_safe_relative(self.root_relative) == V3_ROOT_RELATIVE
                 and all(_require_sha(item, "V3 completed graph") for item in (
                     self.attempt_sha256, self.launch_sha256, self.source_authority_sha256,
                     self.audit_sha256, self.terminal_sha256,
                 )), "CS-WG V4 V3 completed-graph expectation drift")

    def pairs(self) -> tuple[tuple[str, str], ...]:
        return (
            ("attempt.json", self.attempt_sha256),
            ("launch.json", self.launch_sha256),
            ("source_authority.json", self.source_authority_sha256),
            ("audit.json", self.audit_sha256),
            ("terminal.json", self.terminal_sha256),
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_source_audit_v3_completed_graph_expectation_v4",
            "root_relative": self.root_relative,
            "pairs": [{"name": name, "sha256": digest} for name, digest in self.pairs()],
            "exact_leaf_count": 10,
            "leaf_mode": "0444",
            "canonical_basename_sidecars": True,
            "failure_absent": True,
            "accepted_v3_closure_sha256": V3_CLOSURE_SHA256,
            "accepted_v3_identity_sha256": V3_IDENTITY_SHA256,
            "accepted_v1_v2_chain_sha256": V3_FAILURE_CHAIN_SHA256,
            "accepted_fallback_sha256": V3_FALLBACK_SHA256,
        }


LIVE_V3_COMPLETED_EXPECTATION = V3CompletedAuditGraphExpectation()


def _flags_exact(value: Mapping[str, object], *, source_only: bool = True) -> bool:
    return ((not source_only or value.get("source_only") is True)
            and value.get("target_optimizer_backward_update") == 0
            and all(value.get(name) is expected for name, expected in v1.FORBIDDEN_SURFACE_FLAGS.items()))


@dataclass(frozen=True)
class HeldV3CompletedAuditGraph:
    """Exact body/sidecar graph held during validation, without source data."""

    expectation: V3CompletedAuditGraphExpectation
    root_identity: tuple[int, int]
    named_chain_identities: tuple[tuple[str, int, int], ...]
    attempt: Mapping[str, object] = field(repr=False, compare=False)
    launch: Mapping[str, object] = field(repr=False, compare=False)
    source_authority: Mapping[str, object] = field(repr=False, compare=False)
    audit: Mapping[str, object] = field(repr=False, compare=False)
    terminal: Mapping[str, object] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require(isinstance(self.expectation, V3CompletedAuditGraphExpectation)
                 and len(self.root_identity) == 2 and all(type(item) is int and item >= 0
                                                           for item in self.root_identity)
                 and self.named_chain_identities
                 and all(isinstance(item, Mapping) for item in (
                     self.attempt, self.launch, self.source_authority, self.audit, self.terminal,
                 )), "CS-WG V4 held V3 graph type drift")
        for name in ("attempt", "launch", "source_authority", "audit", "terminal"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_source_audit_v3_completed_graph_binding_v4",
            "expectation": self.expectation.payload(),
            "root_identity": list(self.root_identity),
            "named_chain_identities": [list(item) for item in self.named_chain_identities],
            "attempt_sha256": self.expectation.attempt_sha256,
            "launch_sha256": self.expectation.launch_sha256,
            "source_authority_sha256": self.expectation.source_authority_sha256,
            "audit_sha256": self.expectation.audit_sha256,
            "terminal_sha256": self.expectation.terminal_sha256,
            "failure_absent": True,
            "source_only": True,
            "target_optimizer_backward_update": 0,
            **v1.FORBIDDEN_SURFACE_FLAGS,
        }

    @property
    def sha256(self) -> str:
        return _sha(_json_bytes(self.payload()))


def _expected_v3_identity(root: Path) -> v3.SourceAuditV3Identity:
    try:
        identity = v3.build_source_audit_v3_identity(Path(root))
        v3.validate_source_audit_v3_identity_current(Path(root), identity)
    except v3.SourceAuditV3Error as error:
        raise SourceSmokeV4Error("CS-WG V4 accepted V3 identity/closure drift") from error
    _require(identity.sha256 == V3_IDENTITY_SHA256
             and identity.closure.get("closure_sha256") == V3_CLOSURE_SHA256,
             "CS-WG V4 accepted V3 identity literal drift")
    return identity


def _validate_v3_completed_semantics(
    *,
    attempt: Mapping[str, object],
    launch: Mapping[str, object],
    authority: Mapping[str, object],
    audit: Mapping[str, object],
    terminal: Mapping[str, object],
    expectation: V3CompletedAuditGraphExpectation,
    identity: v3.SourceAuditV3Identity,
) -> None:
    """Validate V3's successful source-only graph without re-reading source data."""
    identity_payload = identity.payload()
    _require(attempt.get("schema") == "cross_session_worst_group_m1_source_audit_attempt_v3"
             and attempt.get("cell") == CELL and attempt.get("status") == "ATTEMPT_RESERVED"
             and attempt.get("identity") == identity_payload
             and attempt.get("v1_v2_failed_predecessor_chain_sha256") is None
             and isinstance(attempt.get("v1_v2_failed_predecessor_chain"), Mapping)
             and _sha(_json_bytes(dict(attempt["v1_v2_failed_predecessor_chain"]))) == V3_FAILURE_CHAIN_SHA256
             and attempt.get("source_resolved_or_opened") is False
             and attempt.get("model_constructed") is False
             and attempt.get("cuda_initialized") is False
             and attempt.get("optimizer_steps_completed") == 0 and _flags_exact(attempt),
             "CS-WG V4 V3 attempt semantics drift")
    _require(launch.get("schema") == "cross_session_worst_group_m1_source_audit_launch_v3"
             and launch.get("cell") == CELL and launch.get("status") == "LAUNCHED"
             and launch.get("identity") == identity_payload
             and launch.get("v1_v2_failed_predecessor_chain_sha256") == V3_FAILURE_CHAIN_SHA256
             and launch.get("attempt_sha256") == expectation.attempt_sha256
             and launch.get("source_resolved_or_opened") is False
             and launch.get("model_constructed") is False
             and launch.get("cuda_initialized") is False
             and launch.get("optimizer_steps_completed") == 0 and _flags_exact(launch),
             "CS-WG V4 V3 launch semantics drift")
    fallback = authority.get("deterministic_common_stratum_fallback")
    inherited = authority.get("inherited_v1_source_authority")
    _require(authority.get("schema") == "cross_session_worst_group_m1_source_audit_authority_v3"
             and authority.get("identity") == identity_payload
             and authority.get("v1_v2_failed_predecessor_chain_sha256") == V3_FAILURE_CHAIN_SHA256
             and isinstance(inherited, Mapping)
             and authority.get("inherited_v1_source_authority_sha256") == _sha(_json_bytes(dict(inherited)))
             and isinstance(fallback, Mapping)
             and authority.get("deterministic_common_stratum_fallback_sha256") == V3_FALLBACK_SHA256
             and _sha(_json_bytes(dict(fallback))) == V3_FALLBACK_SHA256
             and inherited.get("deterministic_common_stratum_fallback") == fallback
             and inherited.get("deterministic_common_stratum_fallback_sha256") == V3_FALLBACK_SHA256
             and authority.get("model_constructed") is False
             and authority.get("cuda_initialized") is False
             and authority.get("optimizer_steps_completed") == 0 and _flags_exact(authority),
             "CS-WG V4 V3 source-authority/fallback semantics drift")
    _require(fallback.get("mode") == v3.COMMON_STRATUM_FALLBACK_MODE
             and fallback.get("eligible_common_min2_count") == V3_ELIGIBLE_COMMON_MIN2_COUNT
             and fallback.get("constructible_step_zero_b32") is True
             and fallback.get("source_sessions") == [item[0] for item in V3_RETENTION]
             and isinstance(fallback.get("per_session_retention"), Mapping),
             "CS-WG V4 V3 fallback topology drift")
    retention = fallback["per_session_retention"]
    for session, retained, original in V3_RETENTION:
        row = retention.get(session)
        _require(isinstance(row, Mapping)
                 and row.get("retained_eligible_window_count") == retained
                 and row.get("original_window_count") == original,
                 "CS-WG V4 V3 fallback retention literal drift")
    step_zero = authority.get("common_stratum_step_zero_evidence")
    _require(isinstance(step_zero, Mapping)
             and isinstance(step_zero.get("quota"), Mapping)
             and step_zero["quota"].get("counts") == [10, 11, 11]
             and step_zero.get("row_count") == 32
             and step_zero.get("no_duplicate_sample_ids_within_session_microbatch") is True
             and audit.get("schema") == "cross_session_worst_group_m1_source_audit_result_v3"
             and audit.get("identity_sha256") == identity.sha256
             and audit.get("source_authority_sha256") == expectation.source_authority_sha256
             and audit.get("v1_v2_failed_predecessor_chain_sha256") == V3_FAILURE_CHAIN_SHA256
             and audit.get("deterministic_common_stratum_fallback") == fallback
             and audit.get("deterministic_common_stratum_fallback_sha256") == V3_FALLBACK_SHA256
             and audit.get("common_stratum_step_zero_evidence") == step_zero
             and audit.get("model_constructed") is False
             and audit.get("cuda_initialized") is False
             and audit.get("optimizer_steps_completed") == 0 and _flags_exact(audit),
             "CS-WG V4 V3 audit semantics drift")
    _require(terminal.get("schema") == "cross_session_worst_group_m1_source_audit_terminal_v3"
             and terminal.get("cell") == CELL
             and terminal.get("status") == "PASS_SOURCE_AUDIT_COMMON_STRATUM_CONSTRUCTIBLE"
             and terminal.get("identity") == identity_payload
             and terminal.get("v1_v2_failed_predecessor_chain_sha256") == V3_FAILURE_CHAIN_SHA256
             and terminal.get("attempt_sha256") == expectation.attempt_sha256
             and terminal.get("launch_sha256") == expectation.launch_sha256
             and terminal.get("source_authority_sha256") == expectation.source_authority_sha256
             and terminal.get("audit_sha256") == expectation.audit_sha256
             and terminal.get("deterministic_common_stratum_fallback_sha256") == V3_FALLBACK_SHA256
             and terminal.get("launch_closure_sha256") == V3_CLOSURE_SHA256
             and terminal.get("final_closure_sha256") == V3_CLOSURE_SHA256
             and terminal.get("model_constructed") is False
             and terminal.get("cuda_initialized") is False
             and terminal.get("optimizer_steps_completed") == 0 and _flags_exact(terminal),
             "CS-WG V4 V3 terminal semantics drift")


def _validate_held_v3_completed_graph(
    root: Path,
    expectation: V3CompletedAuditGraphExpectation,
    identity: v3.SourceAuditV3Identity,
    *,
    semantic_validator: Any = _validate_v3_completed_semantics,
) -> HeldV3CompletedAuditGraph:
    """Hold one V3 result-root FD while validating exact body/sidecar topology."""
    try:
        base_fd, opened, named = v2._open_held_result_directory(Path(root), expectation.root_relative)
    except v2.SourceAuditV2Error as error:
        raise SourceSmokeV4Error("CS-WG V4 V3 predecessor directory validation failed") from error
    result_fd = opened[-1]
    try:
        expected_names = tuple(sorted(
            item for body, _digest in expectation.pairs() for item in (body, f"{body}.sha256")
        ))
        _require(tuple(sorted(os.listdir(result_fd))) == expected_names,
                 "CS-WG V4 V3 predecessor topology/failure/extra leaf drift")
        values: dict[str, dict[str, object]] = {}
        for name, expected_sha in expectation.pairs():
            try:
                body = v2._read_held_leaf(result_fd, name)
                sidecar = v2._read_held_leaf(result_fd, f"{name}.sha256")
            except v2.SourceAuditV2Error as error:
                raise SourceSmokeV4Error("CS-WG V4 V3 predecessor leaf mode/type drift") from error
            _require(_sha(body) == expected_sha
                     and sidecar == f"{expected_sha}  {name}\n".encode("ascii"),
                     "CS-WG V4 V3 predecessor body/sidecar digest drift")
            try:
                parsed = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise SourceSmokeV4Error("CS-WG V4 V3 predecessor JSON decode drift") from error
            _require(isinstance(parsed, dict), "CS-WG V4 V3 predecessor JSON object drift")
            values[name] = parsed
        _require(tuple(sorted(os.listdir(result_fd))) == expected_names,
                 "CS-WG V4 V3 predecessor topology changed during held reads")
        _require(callable(semantic_validator), "CS-WG V4 V3 predecessor semantic validator is absent")
        semantic_validator(
            attempt=values["attempt.json"], launch=values["launch.json"],
            authority=values["source_authority.json"], audit=values["audit.json"],
            terminal=values["terminal.json"], expectation=expectation, identity=identity,
        )
        try:
            named_after = v2._named_chain_identities(Path(root), expectation.root_relative)
        except v2.SourceAuditV2Error as error:
            raise SourceSmokeV4Error("CS-WG V4 V3 predecessor named revalidation failed") from error
        _require(named_after == named, "CS-WG V4 V3 predecessor named identity drift")
        held = os.fstat(result_fd)
        return HeldV3CompletedAuditGraph(
            expectation=expectation,
            root_identity=(int(held.st_dev), int(held.st_ino)),
            named_chain_identities=named,
            attempt=values["attempt.json"], launch=values["launch.json"],
            source_authority=values["source_authority.json"], audit=values["audit.json"],
            terminal=values["terminal.json"],
        )
    finally:
        v2._close_chain(base_fd, opened)


def validate_completed_v3_source_audit_graph(root: Path) -> HeldV3CompletedAuditGraph:
    """Production V3 predecessor loader; never called by the dry CLI."""
    identity = _expected_v3_identity(Path(root))
    return _validate_held_v3_completed_graph(Path(root), LIVE_V3_COMPLETED_EXPECTATION, identity)


@dataclass(frozen=True)
class SourceSmokeV4Spec:
    """Fresh V4 root with an embedded, unmodified V1 smoke run-spec."""

    root_relative: str = V4_ROOT_RELATIVE

    def __post_init__(self) -> None:
        _require(_safe_relative(self.root_relative) == V4_ROOT_RELATIVE,
                 "CS-WG V4 smoke root literal drift")

    def payload(self) -> dict[str, object]:
        inherited = v1.source_smoke_spec()
        return {
            "schema": "cross_session_worst_group_m1_source_smoke_spec_v4",
            "root_relative": self.root_relative,
            "inherited_v1_smoke_spec": inherited.payload(),
            "accepted_v3_common_stratum_source_audit": LIVE_V3_COMPLETED_EXPECTATION.payload(),
            "optimizer_steps": v1.SMOKE_STEPS,
            "common_stratum_fallback_mode": v3.COMMON_STRATUM_FALLBACK_MODE,
            "source_only": True,
            "target_optimizer_backward_update": 0,
            **v1.FORBIDDEN_SURFACE_FLAGS,
        }

    @property
    def sha256(self) -> str:
        return _sha(_json_bytes(self.payload()))


@dataclass(frozen=True)
class SourceSmokeV4Identity:
    """V4 identity binds a fresh root to the exact frozen V1 smoke graph."""

    spec: SourceSmokeV4Spec
    inherited_v1_smoke_identity: v1.SourceExecutionIdentity
    accepted_v3_identity: v3.SourceAuditV3Identity
    closure: Mapping[str, object]

    def __post_init__(self) -> None:
        _require(isinstance(self.spec, SourceSmokeV4Spec)
                 and isinstance(self.inherited_v1_smoke_identity, v1.SourceExecutionIdentity)
                 and self.inherited_v1_smoke_identity.spec == v1.source_smoke_spec()
                 and isinstance(self.accepted_v3_identity, v3.SourceAuditV3Identity)
                 and self.accepted_v3_identity.sha256 == V3_IDENTITY_SHA256
                 and self.accepted_v3_identity.closure.get("closure_sha256") == V3_CLOSURE_SHA256
                 and isinstance(self.closure, Mapping) and self.closure.get("closure_sha256") is not None,
                 "CS-WG V4 smoke identity topology drift")
        object.__setattr__(self, "closure", MappingProxyType(dict(self.closure)))

    @property
    def device(self) -> v1.DeviceProfile:
        return self.inherited_v1_smoke_identity.device

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_source_smoke_identity_v4",
            "cell": CELL,
            "phase": PHASE,
            "spec": self.spec.payload(),
            "inherited_v1_smoke_identity": self.inherited_v1_smoke_identity.payload(),
            "accepted_v3_source_audit_identity": self.accepted_v3_identity.payload(),
            "accepted_v3_source_audit_identity_sha256": V3_IDENTITY_SHA256,
            "accepted_v3_closure_sha256": V3_CLOSURE_SHA256,
            "accepted_v3_completed_graph": LIVE_V3_COMPLETED_EXPECTATION.payload(),
            "closure": dict(self.closure),
            "source_only": True,
            "no_amp_tf32_compile": True,
            "target_optimizer_backward_update": 0,
            **v1.FORBIDDEN_SURFACE_FLAGS,
        }

    @property
    def sha256(self) -> str:
        return _sha(_json_bytes(self.payload()))


def build_source_smoke_v4_identity(root: Path, *, device: v1.DeviceProfile) -> SourceSmokeV4Identity:
    """Build code/metadata identity only; it never opens a V3 result or NWB."""
    try:
        inherited = v1.build_identity(Path(root), spec=v1.source_smoke_spec(), device=device)
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV4Error("CS-WG V4 inherited V1 smoke identity drift") from error
    return SourceSmokeV4Identity(
        SourceSmokeV4Spec(), inherited, _expected_v3_identity(Path(root)), implementation_closure(Path(root)),
    )


def validate_source_smoke_v4_identity_current(root: Path, identity: SourceSmokeV4Identity) -> None:
    _require(isinstance(identity, SourceSmokeV4Identity), "CS-WG V4 identity must be typed")
    try:
        v1.validate_identity_current(Path(root), identity.inherited_v1_smoke_identity)
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV4Error("CS-WG V4 inherited V1 smoke identity current-byte drift") from error
    current_v3 = _expected_v3_identity(Path(root))
    _require(current_v3.payload() == identity.accepted_v3_identity.payload(),
             "CS-WG V4 accepted V3 identity payload drift")
    validate_current_closure(Path(root), identity.closure)


class _V4RootReviewSeal:
    pass


_V4_ROOT_REVIEW_SEAL = _V4RootReviewSeal()


@dataclass(frozen=True)
class SourceSmokeV4Capability:
    identity_sha256: str
    v3_completed_graph_sha256: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_sha(self.identity_sha256, "V4 capability identity")
        _require_sha(self.v3_completed_graph_sha256, "V4 capability V3 graph")
        _require(self._seal is _V4_ROOT_REVIEW_SEAL,
                 "CS-WG V4 smoke requires an in-process root-reviewed capability")


def _assert_v4_root_fresh(root: Path, spec: SourceSmokeV4Spec) -> None:
    candidate = Path(root).absolute() / _safe_relative(spec.root_relative)
    try:
        os.lstat(candidate)
    except FileNotFoundError:
        return
    except OSError as error:
        raise SourceSmokeV4Error("CS-WG V4 prospective smoke root cannot be safely inspected") from error
    raise SourceSmokeV4Error("CS-WG V4 canonical smoke root already exists")


def _issue_root_reviewed_source_smoke_v4_capability(
    root: Path,
    identity: SourceSmokeV4Identity,
    *,
    environ: Mapping[str, str] | None,
    review_seal: object,
    predecessor_loader: Any,
) -> SourceSmokeV4Capability:
    _require(review_seal is _V4_ROOT_REVIEW_SEAL,
             "only the root reviewer may issue CS-WG V4 smoke capability")
    validate_source_smoke_v4_identity_current(Path(root), identity)
    graph = predecessor_loader(Path(root))
    _require(isinstance(graph, HeldV3CompletedAuditGraph),
             "CS-WG V4 predecessor loader returned wrong completed-graph type")
    try:
        v1.validate_device_environment(identity.device, environ)
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV4Error("CS-WG V4 selected device/thread environment drift") from error
    _assert_v4_root_fresh(Path(root), identity.spec)
    return SourceSmokeV4Capability(identity.sha256, graph.sha256, _V4_ROOT_REVIEW_SEAL)


def issue_root_reviewed_source_smoke_v4_capability(
    root: Path, identity: SourceSmokeV4Identity, *, environ: Mapping[str, str] | None,
    review_seal: object,
) -> SourceSmokeV4Capability:
    """Root-only issuer; it loads the accepted V3 graph before V4 reservation."""
    return _issue_root_reviewed_source_smoke_v4_capability(
        Path(root), identity, environ=environ, review_seal=review_seal,
        predecessor_loader=validate_completed_v3_source_audit_graph,
    )


def _require_v4_capability(capability: object, identity: SourceSmokeV4Identity) -> SourceSmokeV4Capability:
    _require(isinstance(capability, SourceSmokeV4Capability)
             and capability._seal is _V4_ROOT_REVIEW_SEAL
             and capability.identity_sha256 == identity.sha256,
             "CS-WG V4 smoke requires exact root-reviewed capability")
    return capability


def _v4_attempt_payload(identity: SourceSmokeV4Identity, predecessor: HeldV3CompletedAuditGraph) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_source_smoke_attempt_v4",
        "cell": CELL,
        "status": "ATTEMPT_RESERVED",
        "identity": identity.payload(),
        "accepted_v3_completed_graph": predecessor.payload(),
        "source_resolved_or_opened": False,
        "model_constructed": False,
        "cuda_initialized": False,
        "optimizer_steps_completed": 0,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _validate_launch_backend(value: object) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG V4 physical launch must be a mapping")
    item = dict(value)
    _require(item.get("schema") == "cross_session_worst_group_m1_source_smoke_physical_launch_v4"
             and item.get("provider") == "V4CachedCommonStratumSourceProvider"
             and item.get("runner") == "TorchCSWGSmokeRunner"
             and item.get("inherited_v3_fallback") == v3.COMMON_STRATUM_FALLBACK_MODE
             and item.get("source_opened") is False
             and item.get("model_constructed") is False
             and item.get("cuda_initialized") is False
             and item.get("optimizer_steps_completed") == 0
             and item.get("source_only") is True,
             "CS-WG V4 physical launch semantics drift")
    return item


def _v4_launch_payload(
    identity: SourceSmokeV4Identity, predecessor: HeldV3CompletedAuditGraph,
    attempt_sha256: str, backend: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_source_smoke_launch_v4",
        "cell": CELL,
        "status": "LAUNCHED",
        "identity": identity.payload(),
        "accepted_v3_completed_graph_sha256": predecessor.sha256,
        "attempt_sha256": _require_sha(attempt_sha256, "V4 launch attempt"),
        "backend": _validate_launch_backend(backend),
        "source_resolved_or_opened": False,
        "model_constructed": False,
        "cuda_initialized": False,
        "optimizer_steps_completed": 0,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _prepared_v1_authority(prepared: object) -> dict[str, object]:
    builder = getattr(prepared, "inherited_v1_authority_fragment", None)
    _require(callable(builder), "CS-WG V4 prepared smoke fold lacks inherited V1 authority seam")
    value = builder()
    _require(isinstance(value, Mapping), "CS-WG V4 prepared inherited authority must be a mapping")
    return dict(value)


def _prepared_fallback(prepared: object) -> tuple[dict[str, object], str, dict[str, object]]:
    payload = getattr(prepared, "fallback_payload", None)
    digest = getattr(prepared, "fallback_sha256", None)
    step_zero = getattr(prepared, "step_zero_common_stratum_evidence", None)
    _require(isinstance(payload, Mapping) and _require_sha(digest, "prepared V3 fallback")
             and isinstance(step_zero, Mapping),
             "CS-WG V4 prepared fallback evidence seam drift")
    return dict(payload), digest, dict(step_zero)


def _prepared_cache_evidence(prepared: object) -> dict[str, object]:
    value = getattr(prepared, "digest_cache_payload", None)
    _require(callable(value), "CS-WG V4 prepared digest-cache evidence seam drift")
    payload = value()
    _require(isinstance(payload, Mapping), "CS-WG V4 digest-cache evidence must be a mapping")
    return dict(payload)


def _v4_source_authority_payload(
    identity: SourceSmokeV4Identity, predecessor: HeldV3CompletedAuditGraph, prepared: object,
    *, accepted_fallback_sha256: str = V3_FALLBACK_SHA256,
) -> dict[str, object]:
    inherited = v1.source_authority_payload(identity.inherited_v1_smoke_identity, _prepared_v1_authority(prepared))
    try:
        v1._validate_source_authority(inherited, identity.inherited_v1_smoke_identity)
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV4Error("CS-WG V4 inherited V1 source-authority drift") from error
    fallback, fallback_sha, step_zero = _prepared_fallback(prepared)
    cache = _prepared_cache_evidence(prepared)
    _require(fallback_sha == accepted_fallback_sha256
             and _sha(_json_bytes(fallback)) == accepted_fallback_sha256,
             "CS-WG V4 prepared fallback does not exact-match accepted V3")
    return {
        "schema": "cross_session_worst_group_m1_source_smoke_authority_v4",
        "identity": identity.payload(),
        "accepted_v3_completed_graph_sha256": predecessor.sha256,
        "accepted_v3_source_authority_sha256": predecessor.expectation.source_authority_sha256,
        "inherited_v1_source_authority": inherited,
        "inherited_v1_source_authority_sha256": _sha(_json_bytes(inherited)),
        "deterministic_common_stratum_fallback": fallback,
        "deterministic_common_stratum_fallback_sha256": fallback_sha,
        "common_stratum_step_zero_evidence": step_zero,
        "route_local_input_digest_cache": cache,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _validate_v4_source_authority(
    value: object, identity: SourceSmokeV4Identity, predecessor: HeldV3CompletedAuditGraph, prepared: object,
    *, accepted_fallback_sha256: str = V3_FALLBACK_SHA256,
) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG V4 source authority must be a mapping")
    item = dict(value)
    inherited = item.get("inherited_v1_source_authority")
    fallback, fallback_sha, step_zero = _prepared_fallback(prepared)
    cache = _prepared_cache_evidence(prepared)
    _require(item.get("schema") == "cross_session_worst_group_m1_source_smoke_authority_v4"
             and item.get("identity") == identity.payload()
             and item.get("accepted_v3_completed_graph_sha256") == predecessor.sha256
             and item.get("accepted_v3_source_authority_sha256") == predecessor.expectation.source_authority_sha256
             and isinstance(inherited, Mapping)
             and item.get("inherited_v1_source_authority_sha256") == _sha(_json_bytes(dict(inherited)))
             and item.get("deterministic_common_stratum_fallback") == fallback
             and item.get("deterministic_common_stratum_fallback_sha256") == fallback_sha == accepted_fallback_sha256
             and item.get("common_stratum_step_zero_evidence") == step_zero
             and item.get("route_local_input_digest_cache") == cache
             and _flags_exact(item), "CS-WG V4 source authority cross-binding drift")
    try:
        v1._validate_source_authority(dict(inherited), identity.inherited_v1_smoke_identity)
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV4Error("CS-WG V4 inherited V1 authority semantic drift") from error
    _require(inherited.get("deterministic_common_stratum_fallback") == fallback
             and inherited.get("deterministic_common_stratum_fallback_sha256") == fallback_sha
             and inherited.get("common_stratum_step_zero_evidence") == step_zero,
             "CS-WG V4 inherited authority fallback drift")
    _validate_digest_cache_payload(cache, identity)
    return item


def _validate_digest_cache_payload(value: Mapping[str, object], identity: SourceSmokeV4Identity) -> None:
    sessions = list(identity.inherited_v1_smoke_identity.spec.stage0_spec.source_sessions)
    calibration = value.get("session_calibration_sha256")
    rows = value.get("rows_per_session")
    _require(value.get("schema") == "cross_session_worst_group_m1_input_digest_cache_v4"
             and value.get("source_sessions") == sessions
             and isinstance(calibration, Mapping) and tuple(calibration) == tuple(sessions)
             and isinstance(rows, Mapping) and tuple(rows) == tuple(sessions)
             and all(_require_sha(calibration[session], "cached session calibration")
                     and type(rows[session]) is int and rows[session] > 0 for session in sessions)
             and value.get("semantic_equivalence_to_core_source_episode_row_input_digests") is True
             and value.get("uses_validated_session_calibration_sha256") is True
             and value.get("calibration_digest_recomputed_per_row") is False
             and value.get("calibration_digest_cache_entries") == len(sessions)
             and value.get("all_b32_rows_still_stack_per_row_calibration") is True
             and _require_sha(value.get("all_cached_row_input_digests_sha256"), "cached row input digests")
             and _require_sha(value.get("cache_payload_sha256"), "cache payload"),
             "CS-WG V4 route-local input-digest cache evidence drift")
    without_self = dict(value)
    self_sha = without_self.pop("cache_payload_sha256")
    _require(self_sha == _sha(_json_bytes(without_self)),
             "CS-WG V4 route-local input-digest cache canonical digest drift")


def _v4_smoke_payload(
    identity: SourceSmokeV4Identity, predecessor: HeldV3CompletedAuditGraph,
    authority_sha256: str, authority: Mapping[str, object], inherited_smoke: Mapping[str, object],
) -> dict[str, object]:
    inner = dict(inherited_smoke)
    inner.pop("_checkpoint_bodies", None)
    inner.setdefault("schema", "cross_session_worst_group_m1_source_smoke_v1")
    inner.setdefault("identity_sha256", identity.inherited_v1_smoke_identity.sha256)
    inner.setdefault("source_authority_sha256", _sha(_json_bytes(dict(authority["inherited_v1_source_authority"]))))
    try:
        v1._validate_smoke_result(inner, identity.inherited_v1_smoke_identity)
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV4Error("CS-WG V4 inherited physical smoke evidence drift") from error
    return {
        "schema": "cross_session_worst_group_m1_source_smoke_result_v4",
        "identity_sha256": identity.sha256,
        "source_authority_sha256": _require_sha(authority_sha256, "V4 smoke authority"),
        "accepted_v3_completed_graph_sha256": predecessor.sha256,
        "inherited_v1_smoke": inner,
        "inherited_v1_smoke_sha256": _sha(_json_bytes(inner)),
        "optimizer_steps": inner["optimizer_steps"],
        "one_concatenated_forward_per_step": inner["one_concatenated_forward_per_step"],
        "step_zero_calibration_ownership": authority["common_stratum_step_zero_evidence"],
        "route_local_input_digest_cache": authority["route_local_input_digest_cache"],
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _validate_v4_smoke(
    value: object, identity: SourceSmokeV4Identity, predecessor: HeldV3CompletedAuditGraph,
    authority: Mapping[str, object], authority_sha256: str,
) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG V4 smoke result must be a mapping")
    item = dict(value)
    inherited = item.get("inherited_v1_smoke")
    _require(isinstance(inherited, Mapping)
             and item.get("schema") == "cross_session_worst_group_m1_source_smoke_result_v4"
             and item.get("identity_sha256") == identity.sha256
             and item.get("source_authority_sha256") == authority_sha256
             and item.get("accepted_v3_completed_graph_sha256") == predecessor.sha256
             and item.get("inherited_v1_smoke_sha256") == _sha(_json_bytes(dict(inherited)))
             and item.get("optimizer_steps") == v1.SMOKE_STEPS
             and item.get("one_concatenated_forward_per_step") is True
             and item.get("step_zero_calibration_ownership")
                 == authority.get("common_stratum_step_zero_evidence")
             and item.get("route_local_input_digest_cache") == authority.get("route_local_input_digest_cache")
             and _flags_exact(item), "CS-WG V4 smoke wrapper provenance drift")
    try:
        v1._validate_smoke_result(dict(inherited), identity.inherited_v1_smoke_identity)
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV4Error("CS-WG V4 inherited smoke semantic drift") from error
    return item


def _v4_checkpoint_manifest(
    identity: SourceSmokeV4Identity, *, inherited_manifest: Mapping[str, object],
) -> dict[str, object]:
    try:
        v1._validate_checkpoint_manifest(dict(inherited_manifest), identity.inherited_v1_smoke_identity)
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV4Error("CS-WG V4 inherited checkpoint manifest drift") from error
    return {
        "schema": "cross_session_worst_group_m1_source_smoke_checkpoint_manifest_v4",
        "identity_sha256": identity.sha256,
        "inherited_v1_checkpoint_manifest": dict(inherited_manifest),
        "inherited_v1_checkpoint_manifest_sha256": _sha(_json_bytes(dict(inherited_manifest))),
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _validate_v4_checkpoint_manifest(value: object, identity: SourceSmokeV4Identity) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG V4 checkpoint manifest must be a mapping")
    item = dict(value)
    inherited = item.get("inherited_v1_checkpoint_manifest")
    _require(item.get("schema") == "cross_session_worst_group_m1_source_smoke_checkpoint_manifest_v4"
             and item.get("identity_sha256") == identity.sha256
             and isinstance(inherited, Mapping)
             and item.get("inherited_v1_checkpoint_manifest_sha256") == _sha(_json_bytes(dict(inherited)))
             and _flags_exact(item), "CS-WG V4 checkpoint-manifest wrapper drift")
    try:
        v1._validate_checkpoint_manifest(dict(inherited), identity.inherited_v1_smoke_identity)
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV4Error("CS-WG V4 inherited checkpoint manifest semantics drift") from error
    return item


def _publish_v4_checkpoints(
    artifact: v1.ImmutableArtifactRoot,
    identity: SourceSmokeV4Identity,
    bodies: object,
    *,
    inherited_smoke: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    """Publish each V1 checkpoint body once, then publish only V4's wrapper.

    The inherited V1 lifecycle normally publishes its own
    ``checkpoint_manifest.json``.  V4 deliberately owns that filename so it
    can bind the V3 predecessor and its V4 identity.  Calling V1's publishing
    helper here would therefore make an otherwise valid immutable lifecycle
    collide with the V4 wrapper.  This narrow helper preserves V1's exact body
    filenames and manifest schema in-memory, validates that schema through
    V1's validator, and publishes a single V4 manifest that embeds it.
    """
    _require(isinstance(bodies, Mapping) and tuple(sorted(bodies)) == (
        "best_source_train_loss", "last",
    ), "CS-WG V4 smoke must return exact best/last checkpoint bodies")
    roles: dict[str, dict[str, str]] = {}
    for role in ("best_source_train_loss", "last"):
        body = bodies[role]
        _require(isinstance(body, bytes) and body,
                 f"CS-WG V4 checkpoint {role} bytes drift")
        filename = f"checkpoint_{role}.pt"
        state_field = (
            "best_checkpoint_state_sha256"
            if role == "best_source_train_loss"
            else "final_model_state_sha256"
        )
        roles[role] = {
            "filename": filename,
            "sha256": artifact.publish_bytes(filename, body),
            "state_sha256": _require_sha(inherited_smoke.get(state_field), f"V4 smoke {role} state"),
        }
    inherited_manifest = {
        "schema": "cross_session_worst_group_m1_source_checkpoint_manifest_v1",
        "identity_sha256": identity.inherited_v1_smoke_identity.sha256,
        "checkpoints": roles,
        "monitor": "source_train_loss_only",
        "early_stopping": False,
        "validation_or_target_selection": False,
        "source_only": True,
    }
    try:
        v1._validate_checkpoint_manifest(inherited_manifest, identity.inherited_v1_smoke_identity)
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV4Error("CS-WG V4 inherited checkpoint body/schema drift") from error
    wrapper = _v4_checkpoint_manifest(identity, inherited_manifest=inherited_manifest)
    return wrapper, artifact.publish_json("checkpoint_manifest.json", wrapper)


def _v4_terminal_payload(
    identity: SourceSmokeV4Identity, predecessor: HeldV3CompletedAuditGraph, *, attempt_sha256: str,
    launch_sha256: str, authority_sha256: str, smoke_sha256: str, checkpoints_sha256: str,
) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_source_smoke_terminal_v4",
        "cell": CELL,
        "status": "PASS_SOURCE_SMOKE_COMMON_STRATUM_CONSTRUCTIBLE",
        "identity": identity.payload(),
        "accepted_v3_completed_graph_sha256": predecessor.sha256,
        "attempt_sha256": _require_sha(attempt_sha256, "V4 terminal attempt"),
        "launch_sha256": _require_sha(launch_sha256, "V4 terminal launch"),
        "source_authority_sha256": _require_sha(authority_sha256, "V4 terminal source authority"),
        "smoke_sha256": _require_sha(smoke_sha256, "V4 terminal smoke"),
        "checkpoint_manifest_sha256": _require_sha(checkpoints_sha256, "V4 terminal checkpoint manifest"),
        "launch_closure_sha256": identity.closure["closure_sha256"],
        "final_closure_sha256": identity.closure["closure_sha256"],
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _v4_failure_payload(
    identity: SourceSmokeV4Identity, predecessor: HeldV3CompletedAuditGraph, *, attempt_sha256: str,
    launch_sha256: str | None, authority_sha256: str | None, progress: v1.LifecycleProgress,
    error: BaseException,
) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_source_smoke_failure_v4",
        "cell": CELL,
        "status": "FAILED",
        "identity": identity.payload(),
        "accepted_v3_completed_graph_sha256": predecessor.sha256,
        "attempt_sha256": _require_sha(attempt_sha256, "V4 failure attempt"),
        "launch_sha256": None if launch_sha256 is None else _require_sha(launch_sha256, "V4 failure launch"),
        "source_authority_sha256": None if authority_sha256 is None else _require_sha(
            authority_sha256, "V4 failure source authority",
        ),
        "progress": progress.payload(),
        "error_class": type(error).__name__,
        "error_sha256": _sha(repr(error).encode("utf-8")),
        "terminal_published": False,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


@dataclass(frozen=True)
class SourceSmokeV4LifecycleResult:
    root_identity: tuple[int, int]
    attempt_sha256: str
    launch_sha256: str | None
    source_authority_sha256: str | None
    smoke_sha256: str | None
    checkpoint_manifest_sha256: str | None
    terminal_sha256: str | None
    failure_sha256: str | None


class DeferredSourceSmokeV4Backend(Protocol):
    def launch_payload(self, identity: SourceSmokeV4Identity) -> Mapping[str, object]: ...

    def prepare_source(self, identity: SourceSmokeV4Identity) -> object: ...

    def run_smoke(self, identity: SourceSmokeV4Identity) -> Mapping[str, object]: ...

    def checkpoint_bodies(self) -> Mapping[str, bytes]: ...

    def progress(self) -> v1.LifecycleProgress: ...

    def close(self) -> None: ...


def _revalidate_v4_published_graph(
    artifact: v1.ImmutableArtifactRoot, identity: SourceSmokeV4Identity, predecessor: HeldV3CompletedAuditGraph,
    prepared: object, *, attempt_sha256: str, launch_sha256: str, authority_sha256: str,
    smoke_sha256: str, checkpoints_sha256: str, accepted_fallback_sha256: str = V3_FALLBACK_SHA256,
) -> None:
    expected = (
        "attempt.json", "attempt.json.sha256", "launch.json", "launch.json.sha256",
        "source_authority.json", "source_authority.json.sha256", "smoke.json", "smoke.json.sha256",
        "checkpoint_best_source_train_loss.pt", "checkpoint_best_source_train_loss.pt.sha256",
        "checkpoint_last.pt", "checkpoint_last.pt.sha256", "checkpoint_manifest.json", "checkpoint_manifest.json.sha256",
    )
    artifact.validate_live(expected_names=expected)
    attempt = artifact.read_json_pair("attempt.json", expected_sha256=attempt_sha256)
    launch = artifact.read_json_pair("launch.json", expected_sha256=launch_sha256)
    authority = artifact.read_json_pair("source_authority.json", expected_sha256=authority_sha256)
    smoke = artifact.read_json_pair("smoke.json", expected_sha256=smoke_sha256)
    manifest = artifact.read_json_pair("checkpoint_manifest.json", expected_sha256=checkpoints_sha256)
    _require(attempt == _v4_attempt_payload(identity, predecessor)
             and launch == _v4_launch_payload(identity, predecessor, attempt_sha256, launch["backend"]),
             "CS-WG V4 published attempt/launch graph drift")
    _validate_v4_source_authority(
        authority, identity, predecessor, prepared, accepted_fallback_sha256=accepted_fallback_sha256,
    )
    _validate_v4_smoke(smoke, identity, predecessor, authority, authority_sha256)
    manifest_value = _validate_v4_checkpoint_manifest(manifest, identity)
    inherited_manifest = manifest_value["inherited_v1_checkpoint_manifest"]
    assert isinstance(inherited_manifest, Mapping)
    for role in ("best_source_train_loss", "last"):
        entry = inherited_manifest["checkpoints"][role]
        assert isinstance(entry, Mapping)
        artifact.read_bytes_pair(str(entry["filename"]), expected_sha256=str(entry["sha256"]))


def _execute_reviewed_source_smoke_v4(
    root: Path, *, identity: SourceSmokeV4Identity, capability: object, backend: DeferredSourceSmokeV4Backend,
    environ: Mapping[str, str] | None, predecessor_loader: Any,
    accepted_fallback_sha256: str = V3_FALLBACK_SHA256,
) -> SourceSmokeV4LifecycleResult:
    cap = _require_v4_capability(capability, identity)
    validate_source_smoke_v4_identity_current(Path(root), identity)
    predecessor = predecessor_loader(Path(root))
    _require(isinstance(predecessor, HeldV3CompletedAuditGraph)
             and predecessor.sha256 == cap.v3_completed_graph_sha256,
             "CS-WG V4 predecessor/capability binding drift")
    try:
        v1.validate_device_environment(identity.device, environ)
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV4Error("CS-WG V4 device environment drift before reserve") from error
    _assert_v4_root_fresh(Path(root), identity.spec)
    artifact = v1.ImmutableArtifactRoot.reserve(Path(root), identity.spec)  # type: ignore[arg-type]
    attempt_sha256 = artifact.publish_json("attempt.json", _v4_attempt_payload(identity, predecessor))
    launch_sha256: str | None = None
    authority_sha256: str | None = None
    smoke_sha256: str | None = None
    checkpoints_sha256: str | None = None
    terminal_sha256: str | None = None
    failure_sha256: str | None = None
    try:
        launch_sha256 = artifact.publish_json(
            "launch.json", _v4_launch_payload(identity, predecessor, attempt_sha256, backend.launch_payload(identity)),
        )
        prepared = backend.prepare_source(identity)
        authority = _v4_source_authority_payload(
            identity, predecessor, prepared, accepted_fallback_sha256=accepted_fallback_sha256,
        )
        _validate_v4_source_authority(
            authority, identity, predecessor, prepared, accepted_fallback_sha256=accepted_fallback_sha256,
        )
        authority_sha256 = artifact.publish_json("source_authority.json", authority)
        inherited_smoke = dict(backend.run_smoke(identity))
        checkpoint_bodies = inherited_smoke.pop("_checkpoint_bodies", None)
        if checkpoint_bodies is None:
            checkpoint_bodies = backend.checkpoint_bodies()
        smoke = _v4_smoke_payload(identity, predecessor, authority_sha256, authority, inherited_smoke)
        _validate_v4_smoke(smoke, identity, predecessor, authority, authority_sha256)
        smoke_sha256 = artifact.publish_json("smoke.json", smoke)
        _manifest, checkpoints_sha256 = _publish_v4_checkpoints(
            artifact, identity, checkpoint_bodies,
            inherited_smoke=dict(smoke["inherited_v1_smoke"]),
        )
        # Final checks deliberately use live closure/environment and exact V3
        # graph, but never re-assert prospective absence after own reservation.
        validate_source_smoke_v4_identity_current(Path(root), identity)
        predecessor_now = predecessor_loader(Path(root))
        _require(isinstance(predecessor_now, HeldV3CompletedAuditGraph)
                 and predecessor_now.sha256 == predecessor.sha256 == cap.v3_completed_graph_sha256,
                 "CS-WG V4 accepted V3 predecessor drifted during smoke")
        try:
            v1.validate_device_environment(identity.device, environ)
        except v1.SourceLifecycleError as error:
            raise SourceSmokeV4Error("CS-WG V4 device environment drift before terminal") from error
        _revalidate_v4_published_graph(
            artifact, identity, predecessor, prepared,
            attempt_sha256=attempt_sha256, launch_sha256=launch_sha256,
            authority_sha256=authority_sha256, smoke_sha256=smoke_sha256,
            checkpoints_sha256=checkpoints_sha256, accepted_fallback_sha256=accepted_fallback_sha256,
        )
        terminal = _v4_terminal_payload(
            identity, predecessor, attempt_sha256=attempt_sha256, launch_sha256=launch_sha256,
            authority_sha256=authority_sha256, smoke_sha256=smoke_sha256,
            checkpoints_sha256=checkpoints_sha256,
        )
        terminal_sha256 = artifact.publish_json("terminal.json", terminal)
        artifact.validate_live(expected_names=(
            "attempt.json", "attempt.json.sha256", "launch.json", "launch.json.sha256",
            "source_authority.json", "source_authority.json.sha256", "smoke.json", "smoke.json.sha256",
            "checkpoint_best_source_train_loss.pt", "checkpoint_best_source_train_loss.pt.sha256",
            "checkpoint_last.pt", "checkpoint_last.pt.sha256", "checkpoint_manifest.json", "checkpoint_manifest.json.sha256",
            "terminal.json", "terminal.json.sha256",
        ))
        _require(artifact.read_json_pair("terminal.json", expected_sha256=terminal_sha256) == terminal,
                 "CS-WG V4 published terminal graph drift")
        return SourceSmokeV4LifecycleResult(
            artifact.root_identity, attempt_sha256, launch_sha256, authority_sha256,
            smoke_sha256, checkpoints_sha256, terminal_sha256, None,
        )
    except BaseException as error:
        try:
            progress = backend.progress()
            _require(isinstance(progress, v1.LifecycleProgress), "CS-WG V4 backend progress type drift")
        except BaseException:
            progress = v1.LifecycleProgress()
        failure_sha256 = artifact.publish_json(
            "failure.json",
            _v4_failure_payload(
                identity, predecessor, attempt_sha256=attempt_sha256, launch_sha256=launch_sha256,
                authority_sha256=authority_sha256, progress=progress, error=error,
            ),
        )
        return SourceSmokeV4LifecycleResult(
            artifact.root_identity, attempt_sha256, launch_sha256, authority_sha256,
            smoke_sha256, checkpoints_sha256, terminal_sha256, failure_sha256,
        )
    finally:
        try:
            backend.close()
        finally:
            artifact.close()


def execute_reviewed_source_smoke_v4(
    root: Path, *, identity: SourceSmokeV4Identity, capability: object, backend: DeferredSourceSmokeV4Backend,
    environ: Mapping[str, str] | None,
) -> SourceSmokeV4LifecycleResult:
    """Root-only physical V4 smoke; the public CLI never reaches this function."""
    return _execute_reviewed_source_smoke_v4(
        Path(root), identity=identity, capability=capability, backend=backend, environ=environ,
        predecessor_loader=validate_completed_v3_source_audit_graph,
    )


def dry_plan(root: Path | None = None) -> dict[str, object]:
    """Static declaration only.  ``root`` is deliberately ignored."""
    del root
    return {
        "cell": CELL,
        "phase": PHASE,
        "workorder_sha256": WORKORDER_SHA256,
        "fresh_v4_root_relative": V4_ROOT_RELATIVE,
        "accepted_v3_completed_graph": LIVE_V3_COMPLETED_EXPECTATION.payload(),
        "accepted_v3_closure_sha256": V3_CLOSURE_SHA256,
        "accepted_v3_identity_sha256": V3_IDENTITY_SHA256,
        "source_only_100_step_smoke": {
            "optimizer_steps": v1.SMOKE_STEPS,
            "one_concatenated_mixed_session_forward_per_step": True,
            "total_batch_size": 32,
            "lambda": 1.0,
            "tau": 0.01,
            "amp": False,
            "tf32": False,
            "compile": False,
        },
        "route_local_digest_cache": {
            "semantic_equivalence_required": True,
            "session_calibration_digest_reused_per_row": True,
            "per_row_b32_calibration_stack_preserved": True,
        },
        "opens_source_or_target": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "execution_authorized": False,
    }


__all__ = (
    "CELL", "PHASE", "WORKORDER_RELATIVE", "WORKORDER_SHA256", "V4_ROOT_RELATIVE",
    "V3_ROOT_RELATIVE", "V3_CLOSURE_SHA256", "V3_IDENTITY_SHA256", "V3_FAILURE_CHAIN_SHA256",
    "V3_FALLBACK_SHA256", "LIVE_V3_COMPLETED_EXPECTATION", "SourceSmokeV4Error",
    "V3CompletedAuditGraphExpectation", "HeldV3CompletedAuditGraph", "SourceSmokeV4Spec",
    "SourceSmokeV4Identity", "SourceSmokeV4Capability", "SourceSmokeV4LifecycleResult",
    "implementation_closure", "validate_current_closure", "build_source_smoke_v4_identity",
    "validate_source_smoke_v4_identity_current", "validate_completed_v3_source_audit_graph",
    "issue_root_reviewed_source_smoke_v4_capability", "execute_reviewed_source_smoke_v4", "dry_plan",
)
