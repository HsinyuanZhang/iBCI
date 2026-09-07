"""CS-WG M1 V6 audit-spec source-smoke successor.

V6 is deliberately a narrow composition over the accepted V3 common-stratum
audit and the unchanged V1 optimizer loop.  V5 failed before a source
authority because it passed a smoke run-spec to V3's audit-only preparer.  No
historical result is rebuilt as current code: V3 and V5 are held immutable
receipt graphs, while V6 owns a new closure, identity, capability, and root.
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
from . import source_lifecycle as v1
from . import source_smoke_v5 as v5


class SourceSmokeV6Error(RuntimeError):
    """Fail closed for a V6 predecessor, lifecycle, or audit-spec drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceSmokeV6Error(message)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64
             and all(character in "0123456789abcdef" for character in value),
             f"CS-WG V6 {label} must be a lowercase SHA-256")
    return value


def _safe_relative(value: object) -> str:
    _require(isinstance(value, str) and value, "CS-WG V6 relative path is absent")
    path = Path(value)
    _require(not path.is_absolute() and ".." not in path.parts and path.name not in {"", ".", ".."},
             "CS-WG V6 relative path is unsafe")
    return path.as_posix()


def _directory_identity(info: os.stat_result) -> tuple[int, int, int]:
    """Stable identity for a held directory without treating mtime as authority."""
    return (int(info.st_dev), int(info.st_ino), int(stat.S_IFMT(info.st_mode)))


def _leaf_snapshot(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    """Identity plus mutation-sensitive metadata for one regular closure leaf."""
    return (
        int(info.st_dev), int(info.st_ino), int(stat.S_IFMT(info.st_mode)),
        int(info.st_size), int(info.st_mtime_ns), int(info.st_ctime_ns),
    )


def _stat_at_no_follow(directory_fd: int, name: str, *, label: str) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as error:
        raise SourceSmokeV6Error(f"CS-WG V6 closure {label} inaccessible") from error


CELL = v5.CELL
PHASE = "m1_source_smoke_v6_audit_spec_successor"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CS_WG_M1_SOURCE_SMOKE_V6_AUDIT_SPEC_SUCCESSOR_20260826.md"
WORKORDER_SHA256 = "3baf485008655aa5dc9e6a24d2a8e1805b791e1c78e377b1eeb45b3eceea1a3c"
V6_ROOT_RELATIVE = "tfpd_exploration/results/cross_session_worst_group_m1_source_smoke_v6"

# These V3 values are historical evidence, not a request to rebuild V3's
# current closure.  ``v5`` already owns the descriptor-safe exact 10-leaf
# expectation and semantic validator used below.
V3_CLOSURE_SHA256 = v5.V3_CLOSURE_SHA256
V3_IDENTITY_SHA256 = v5.V3_IDENTITY_SHA256
V3_FALLBACK_SHA256 = v5.V3_FALLBACK_SHA256
V3_COMPLETED_EXPECTATION = v5.V3_COMPLETED_EXPECTATION

V5_ROOT_RELATIVE = "tfpd_exploration/results/cross_session_worst_group_m1_source_smoke_v5"
V5_ATTEMPT_SHA256 = "ae6bf550dfff4179d51dbb62616a7fa97d2bfb33d1e7de5ab8c4917096afb9c4"
V5_LAUNCH_SHA256 = "03a33580948f9c95dd8f022ea1947ed9f696f551974fe546c5edb03c94d3756f"
V5_FAILURE_SHA256 = "0cbd5c0551fa654bf7d9b2b45dfe334d7023bcfb142087fefd297ed85199ce11"
V5_FAILURE_CLASS = "SourceAuditV3Error"
V5_FAILURE_PREIMAGE = "SourceAuditV3Error('CS-WG V3 source selection/outer-target ordering drift')"
V5_FAILURE_ERROR_SHA256 = "840a7a668324571626ba202fd63a3f9590252701be5b91707f2a0b11d2d3d9db"

_RUNTIME_DEPENDENCY_PATHS = (
    "tfpd_exploration/docs/WORKORDER_CS_WG_M1_SOURCE_AUDIT_V2_NAMESPACE_SUCCESSOR_20260826.md",
    "tfpd_exploration/docs/WORKORDER_CS_WG_M1_SOURCE_AUDIT_V3_COMMON_STRATUM_SUCCESSOR_20260826.md",
    "tfpd_exploration/docs/WORKORDER_CS_WG_M1_SOURCE_SMOKE_V4_COMMON_STRATUM_SUCCESSOR_20260826.md",
    "tfpd_exploration/docs/WORKORDER_CS_WG_M1_SOURCE_SMOKE_V5_GRANULAR_DERIVATIVE_SUCCESSOR_20260826.md",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_audit_v2.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_audit_v3.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_smoke_v4.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_smoke_physical_v4.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_smoke_v5.py",
)
_OWNED_PATHS = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/cross_session_worst_group_v1/source_smoke_v6.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_smoke_physical_v6.py",
    "tfpd_exploration/scripts/run_cross_session_worst_group_m1_source_smoke_v6.py",
    "tfpd_exploration/tests/test_cross_session_worst_group_m1_source_smoke_v6.py",
)


def _read_regular_no_follow(root: Path, relative: str) -> str:
    """Hash one closure leaf through a held, no-follow descriptor chain.

    A closure hash is authorization material, so a pre-open ``lstat`` followed
    by pathname ``open`` is insufficient: a named leaf could be swapped (or
    made a symlink) between those operations.  We hold every directory on the
    relative path, open every component with ``O_NOFOLLOW``, compare the
    opened FD to its no-follow named entry, read the leaf through that same FD,
    and then prove that the named root/directory/leaf chain still names the
    exact held objects.  No mode policy is inferred here; the body SHA is the
    closure authority and immutable receipt leaves have their own mode checks.
    """
    root_path = Path(root).absolute()
    safe_relative = _safe_relative(relative)
    components = Path(safe_relative).parts
    _require(components, "CS-WG V6 closure relative path is absent")
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    _require(isinstance(no_follow, int) and no_follow != 0
             and isinstance(directory, int) and directory != 0,
             "CS-WG V6 closure no-follow directory descriptors are unavailable")
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | no_follow | directory
    leaf_flags = os.O_RDONLY | os.O_CLOEXEC | no_follow
    try:
        root_named_before = os.lstat(root_path)
    except OSError as error:
        raise SourceSmokeV6Error("CS-WG V6 closure root inaccessible") from error
    _require(stat.S_ISDIR(root_named_before.st_mode) and not stat.S_ISLNK(root_named_before.st_mode),
             "CS-WG V6 closure root is not a directory non-symlink")
    try:
        root_fd = os.open(os.fspath(root_path), directory_flags)
    except OSError as error:
        raise SourceSmokeV6Error("CS-WG V6 closure root no-follow open failed") from error
    held_fds = [root_fd]
    directory_links: list[tuple[int, str, int, tuple[int, int, int]]] = []
    leaf_fd: int | None = None
    try:
        root_opened = os.fstat(root_fd)
        _require(stat.S_ISDIR(root_opened.st_mode)
                 and _directory_identity(root_opened) == _directory_identity(root_named_before),
                 "CS-WG V6 closure root changed during no-follow open")
        current_fd = root_fd
        for component in components[:-1]:
            named_before = _stat_at_no_follow(current_fd, component, label="directory entry")
            _require(stat.S_ISDIR(named_before.st_mode) and not stat.S_ISLNK(named_before.st_mode),
                     f"CS-WG V6 closure directory is not a directory non-symlink: {relative}")
            try:
                child_fd = os.open(component, directory_flags, dir_fd=current_fd)
            except OSError as error:
                raise SourceSmokeV6Error(f"CS-WG V6 closure directory no-follow open failed: {relative}") from error
            held_fds.append(child_fd)
            child_opened = os.fstat(child_fd)
            _require(stat.S_ISDIR(child_opened.st_mode)
                     and _directory_identity(child_opened) == _directory_identity(named_before),
                     f"CS-WG V6 closure directory changed during no-follow open: {relative}")
            directory_links.append((current_fd, component, child_fd, _directory_identity(child_opened)))
            current_fd = child_fd

        leaf_name = components[-1]
        leaf_named_before = _stat_at_no_follow(current_fd, leaf_name, label="leaf entry")
        _require(stat.S_ISREG(leaf_named_before.st_mode) and not stat.S_ISLNK(leaf_named_before.st_mode),
                 f"CS-WG V6 closure leaf is not a regular non-symlink: {relative}")
        try:
            leaf_fd = os.open(leaf_name, leaf_flags, dir_fd=current_fd)
        except OSError as error:
            raise SourceSmokeV6Error(f"CS-WG V6 closure leaf no-follow open failed: {relative}") from error
        leaf_opened = os.fstat(leaf_fd)
        _require(stat.S_ISREG(leaf_opened.st_mode)
                 and _leaf_snapshot(leaf_opened) == _leaf_snapshot(leaf_named_before),
                 f"CS-WG V6 closure leaf changed during no-follow open: {relative}")
        digest = hashlib.sha256()
        while chunk := os.read(leaf_fd, 1 << 20):
            digest.update(chunk)
        leaf_after_read = os.fstat(leaf_fd)
        _require(_leaf_snapshot(leaf_after_read) == _leaf_snapshot(leaf_opened),
                 f"CS-WG V6 closure leaf changed while read: {relative}")
        leaf_named_after = _stat_at_no_follow(current_fd, leaf_name, label="leaf revalidation")
        _require(stat.S_ISREG(leaf_named_after.st_mode)
                 and _leaf_snapshot(leaf_named_after) == _leaf_snapshot(leaf_opened),
                 f"CS-WG V6 closure leaf named identity changed after read: {relative}")
        for parent_fd, component, child_fd, expected_identity in directory_links:
            named_after = _stat_at_no_follow(parent_fd, component, label="directory revalidation")
            _require(stat.S_ISDIR(named_after.st_mode)
                     and _directory_identity(named_after) == expected_identity
                     and _directory_identity(os.fstat(child_fd)) == expected_identity,
                     f"CS-WG V6 closure directory named identity changed after read: {relative}")
        try:
            root_named_after = os.lstat(root_path)
        except OSError as error:
            raise SourceSmokeV6Error("CS-WG V6 closure root revalidation failed") from error
        _require(stat.S_ISDIR(root_named_after.st_mode)
                 and _directory_identity(root_named_after) == _directory_identity(root_opened),
                 "CS-WG V6 closure root named identity changed after read")
        return digest.hexdigest()
    finally:
        if leaf_fd is not None:
            os.close(leaf_fd)
        for held_fd in reversed(held_fds):
            os.close(held_fd)


def implementation_closure(root: Path) -> dict[str, object]:
    """Explicit V6 closure; never invokes a V2/V3/V4/V5 closure builder.

    V6 may import their individual closure-bound modules, but historical route
    closures cannot be reinterpreted as current-byte authority after successors
    have changed source.  The inherited V1 closure is the live base-loop seam;
    every additional eager local module is bound directly below.
    """
    try:
        inherited = v1.implementation_closure(Path(root))
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV6Error("CS-WG V6 current V1 source closure drift") from error
    inherited_rows = inherited.get("paths")
    _require(isinstance(inherited_rows, list) and inherited_rows,
             "CS-WG V6 inherited V1 closure topology drift")
    rows = [dict(item) for item in inherited_rows]
    known = {item.get("path") for item in rows}
    for relative in (*_RUNTIME_DEPENDENCY_PATHS, *_OWNED_PATHS):
        _require(relative not in known, "CS-WG V6 closure path duplication")
        rows.append({"path": relative, "sha256": _read_regular_no_follow(Path(root), relative)})
        known.add(relative)
    workorder = next(item for item in rows if item["path"] == WORKORDER_RELATIVE)
    _require(workorder["sha256"] == WORKORDER_SHA256, "CS-WG V6 workorder literal/body drift")
    body = {
        "schema": "cross_session_worst_group_m1_source_smoke_v6_closure_v1",
        "current_v1_source_closure_sha256": inherited.get("closure_sha256"),
        "historical_v3_closure_sha256": V3_CLOSURE_SHA256,
        "historical_v3_identity_sha256": V3_IDENTITY_SHA256,
        "historical_v5_failed_graph": V5_FAILED_EXPECTATION.payload(),
        "inherits_v5_derivative_gate_sha256": _sha(_json_bytes(v5.derivative_gate_contract_payload())),
        "paths": rows,
    }
    return {**body, "closure_sha256": _sha(_json_bytes(body))}


def validate_current_closure(root: Path, value: Mapping[str, object]) -> None:
    _require(isinstance(value, Mapping) and dict(value) == implementation_closure(Path(root)),
             "CS-WG V6 successor closure/current-byte drift")


V5_FAILED_EXPECTATION = v5.HistoricalGraphExpectation(
    V5_ROOT_RELATIVE,
    (("attempt.json", V5_ATTEMPT_SHA256), ("launch.json", V5_LAUNCH_SHA256),
     ("failure.json", V5_FAILURE_SHA256)),
    6,
    "failed V5 audit-spec graph",
)


def _flags_exact(value: Mapping[str, object]) -> bool:
    return (value.get("source_only") is True
            and value.get("target_optimizer_backward_update") == 0
            and all(value.get(name) is expected for name, expected in v1.FORBIDDEN_SURFACE_FLAGS.items()))


def _identity_payload(value: Mapping[str, object], *, label: str) -> Mapping[str, object]:
    identity = value.get("identity")
    _require(isinstance(identity, Mapping), f"CS-WG V6 {label} identity absent")
    return identity


def _identity_sha(value: Mapping[str, object], *, label: str) -> str:
    return _sha(_json_bytes(dict(_identity_payload(value, label=label))))


def _validate_historical_v5_graph(graph: v5.HeldHistoricalGraph) -> None:
    """Validate V5's real failed-before-authority receipt shape.

    Exact body SHA checks are performed by the held loader.  This schema check
    prevents the fixed bytes from being mistakenly treated as a completed or
    GPU-using predecessor, without requiring V5's old closure to equal current
    code bytes.
    """
    _require(graph.expectation == V5_FAILED_EXPECTATION, "CS-WG V6 V5 expectation drift")
    attempt = graph.body("attempt.json")
    launch = graph.body("launch.json")
    failure = graph.body("failure.json")
    identity = _identity_payload(attempt, label="V5 attempt")
    closure = identity.get("closure")
    _require(isinstance(closure, Mapping) and _require_sha(closure.get("closure_sha256"), "V5 historical closure")
             and identity.get("schema") == "cross_session_worst_group_m1_source_smoke_identity_v5"
             and identity.get("cell") == CELL
             and identity.get("phase") == "m1_source_smoke_v5_granular_derivative_successor"
             and identity.get("accepted_v3_closure_sha256_historical") == V3_CLOSURE_SHA256
             and identity.get("accepted_v3_identity_sha256_historical") == V3_IDENTITY_SHA256
             and identity.get("accepted_v3_completed_graph") == V3_COMPLETED_EXPECTATION.payload()
             and identity.get("failed_v4_graph") == v5.V4_FAILED_EXPECTATION.payload(),
             "CS-WG V6 V5 historical identity/lineage drift")
    _require(attempt.get("schema") == "cross_session_worst_group_m1_source_smoke_attempt_v5"
             and attempt.get("cell") == CELL and attempt.get("status") == "ATTEMPT_RESERVED"
             and attempt.get("source_resolved_or_opened") is False
             and attempt.get("model_constructed") is False
             and attempt.get("cuda_initialized") is False
             and attempt.get("optimizer_steps_completed") == 0
             and _flags_exact(attempt),
             "CS-WG V6 V5 attempt boundary/schema drift")
    _require(launch.get("schema") == "cross_session_worst_group_m1_source_smoke_launch_v5"
             and launch.get("cell") == CELL and launch.get("status") == "LAUNCHED"
             and launch.get("attempt_sha256") == V5_ATTEMPT_SHA256
             and launch.get("source_resolved_or_opened") is False
             and launch.get("model_constructed") is False
             and launch.get("cuda_initialized") is False
             and launch.get("optimizer_steps_completed") == 0
             and _flags_exact(launch),
             "CS-WG V6 V5 launch boundary/schema drift")
    _require(failure.get("schema") == "cross_session_worst_group_m1_source_smoke_failure_v5"
             and failure.get("cell") == CELL and failure.get("status") == "FAILED"
             and failure.get("attempt_sha256") == V5_ATTEMPT_SHA256
             and failure.get("launch_sha256") == V5_LAUNCH_SHA256
             and failure.get("source_authority_sha256") is None
             and failure.get("failure_stage") == "physical_or_lifecycle"
             and failure.get("failed_predicate") == "backend_or_lifecycle_exception"
             and failure.get("error_class") == V5_FAILURE_CLASS
             and failure.get("error_sha256") == V5_FAILURE_ERROR_SHA256
             and failure.get("terminal_published") is False
             and _flags_exact(failure),
             "CS-WG V6 V5 failure semantics drift")
    progress = failure.get("progress")
    _require(isinstance(progress, Mapping)
             and progress.get("model_constructed") is False
             and progress.get("cuda_initialized") is False
             and progress.get("optimizer_steps_completed") == 0,
             "CS-WG V6 V5 failure zero-CUDA/progress drift")
    for label, value in (("launch", launch), ("failure", failure)):
        _require(_identity_payload(value, label=f"V5 {label}") == identity,
                 "CS-WG V6 V5 identity propagation drift")
    _require(_sha(V5_FAILURE_PREIMAGE.encode("utf-8")) == V5_FAILURE_ERROR_SHA256,
             "CS-WG V6 V5 failure-preimage literal drift")


def validate_v6_predecessors(root: Path) -> tuple[v5.HeldHistoricalGraph, v5.HeldHistoricalGraph]:
    """Read completed V3 and failed V5 graph through independent held FDs."""
    v3_graph = v5._read_held_historical_graph(
        Path(root), V3_COMPLETED_EXPECTATION, semantic_validator=v5._validate_historical_v3_graph,
    )
    v5_graph = v5._read_held_historical_graph(
        Path(root), V5_FAILED_EXPECTATION, semantic_validator=_validate_historical_v5_graph,
    )
    return v3_graph, v5_graph


@dataclass(frozen=True)
class HistoricalV1AuditIdentity:
    """The only historical identity field V6 needs for the audit builder."""

    spec: v1.SourceRouteSpec

    def __post_init__(self) -> None:
        _require(isinstance(self.spec, v1.SourceRouteSpec) and self.spec == v1.source_audit_spec(),
                 "CS-WG V6 historical V3 audit-spec identity drift")


@dataclass(frozen=True)
class HistoricalV3AcceptedIdentity:
    """Typed, receipt-derived V3 audit identity; no V3 current closure rebuild."""

    inherited_v1_identity: HistoricalV1AuditIdentity
    historical_identity_sha256: str = V3_IDENTITY_SHA256
    historical_closure_sha256: str = V3_CLOSURE_SHA256

    def __post_init__(self) -> None:
        _require(isinstance(self.inherited_v1_identity, HistoricalV1AuditIdentity)
                 and self.historical_identity_sha256 == V3_IDENTITY_SHA256
                 and self.historical_closure_sha256 == V3_CLOSURE_SHA256,
                 "CS-WG V6 accepted V3 typed identity drift")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_historical_v3_audit_identity_v6",
            "inherited_v1_audit_spec": self.inherited_v1_identity.spec.payload(),
            "historical_v3_identity_sha256": self.historical_identity_sha256,
            "historical_v3_closure_sha256": self.historical_closure_sha256,
            "source_only": True,
        }


def _historical_v3_audit_spec_from_identity_payload(
    identity: Mapping[str, object], *, expected_identity_sha256: str,
    expected_closure_sha256: str,
) -> v1.SourceRouteSpec:
    """Validate the receipt's nested audit spec without rebuilding old code."""
    inherited_v2 = identity.get("inherited_v2_identity")
    _require(isinstance(inherited_v2, Mapping), "CS-WG V6 V3 inherited V2 identity absent")
    inherited_v1 = inherited_v2.get("inherited_v1_identity")
    _require(isinstance(inherited_v1, Mapping)
             and inherited_v1.get("schema") == "cross_session_worst_group_m1_source_audit_identity_v1",
             "CS-WG V6 V3 inherited V1 audit identity schema drift")
    expected_spec = v1.source_audit_spec().payload()
    _require(inherited_v1.get("spec") == expected_spec,
             "CS-WG V6 V3 accepted audit-spec payload drift")
    _require(_require_sha(expected_identity_sha256, "V3 expected identity")
             and _require_sha(expected_closure_sha256, "V3 expected closure")
             and _sha(_json_bytes(dict(identity))) == expected_identity_sha256
             and isinstance(identity.get("closure"), Mapping)
             and identity["closure"].get("closure_sha256") == expected_closure_sha256,
             "CS-WG V6 V3 accepted identity digest/closure drift")
    return v1.source_audit_spec()


def historical_v3_accepted_identity(graph: v5.HeldHistoricalGraph) -> HistoricalV3AcceptedIdentity:
    """Rehydrate only the audited V1 audit spec from immutable V3 receipts."""
    _require(graph.expectation == V3_COMPLETED_EXPECTATION,
             "CS-WG V6 historical V3 graph expectation drift")
    identity = _identity_payload(graph.body("attempt.json"), label="V3 attempt")
    spec = _historical_v3_audit_spec_from_identity_payload(
        identity, expected_identity_sha256=V3_IDENTITY_SHA256, expected_closure_sha256=V3_CLOSURE_SHA256,
    )
    return HistoricalV3AcceptedIdentity(
        HistoricalV1AuditIdentity(spec),
    )


@dataclass(frozen=True)
class SourceSmokeV6Spec:
    root_relative: str = V6_ROOT_RELATIVE

    def __post_init__(self) -> None:
        _require(_safe_relative(self.root_relative) == V6_ROOT_RELATIVE,
                 "CS-WG V6 smoke root literal drift")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_source_smoke_spec_v6",
            "root_relative": self.root_relative,
            "inherited_v1_smoke_spec": v1.source_smoke_spec().payload(),
            "accepted_v3_audit_spec": v1.source_audit_spec().payload(),
            "audit_then_rebind_to_smoke": True,
            "optimizer_steps": v1.SMOKE_STEPS,
            "derivative_numeric_gate": v5.derivative_gate_contract_payload(),
            "source_only": True,
            "target_optimizer_backward_update": 0,
            **v1.FORBIDDEN_SURFACE_FLAGS,
        }

    @property
    def sha256(self) -> str:
        return _sha(_json_bytes(self.payload()))


@dataclass(frozen=True)
class SourceSmokeV6Identity:
    spec: SourceSmokeV6Spec
    inherited_v1_smoke_identity: v1.SourceExecutionIdentity
    accepted_v3_identity: HistoricalV3AcceptedIdentity
    closure: Mapping[str, object]

    def __post_init__(self) -> None:
        _require(isinstance(self.spec, SourceSmokeV6Spec)
                 and isinstance(self.inherited_v1_smoke_identity, v1.SourceExecutionIdentity)
                 and self.inherited_v1_smoke_identity.spec == v1.source_smoke_spec()
                 and isinstance(self.accepted_v3_identity, HistoricalV3AcceptedIdentity)
                 and isinstance(self.closure, Mapping)
                 and _require_sha(self.closure.get("closure_sha256"), "closure"),
                 "CS-WG V6 identity topology drift")
        object.__setattr__(self, "closure", MappingProxyType(dict(self.closure)))

    @property
    def device(self) -> v1.DeviceProfile:
        return self.inherited_v1_smoke_identity.device

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_source_smoke_identity_v6",
            "cell": CELL,
            "phase": PHASE,
            "spec": self.spec.payload(),
            "inherited_v1_smoke_identity": self.inherited_v1_smoke_identity.payload(),
            "accepted_v3_identity": self.accepted_v3_identity.payload(),
            "accepted_v3_completed_graph": V3_COMPLETED_EXPECTATION.payload(),
            "failed_v5_graph": V5_FAILED_EXPECTATION.payload(),
            "failed_v5_error_preimage_sha256": V5_FAILURE_ERROR_SHA256,
            "closure": dict(self.closure),
            "source_only": True,
            "no_amp_tf32_compile": True,
            "target_optimizer_backward_update": 0,
            **v1.FORBIDDEN_SURFACE_FLAGS,
        }

    @property
    def sha256(self) -> str:
        return _sha(_json_bytes(self.payload()))


def build_source_smoke_v6_identity(root: Path, *, device: v1.DeviceProfile) -> SourceSmokeV6Identity:
    try:
        inherited = v1.build_identity(Path(root), spec=v1.source_smoke_spec(), device=device)
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV6Error("CS-WG V6 inherited V1 identity drift") from error
    # The static typed form is cross-checked against actual held V3 receipts
    # before a capability or root may exist.
    accepted = HistoricalV3AcceptedIdentity(
        HistoricalV1AuditIdentity(v1.source_audit_spec()),
    )
    return SourceSmokeV6Identity(SourceSmokeV6Spec(), inherited, accepted, implementation_closure(Path(root)))


def validate_source_smoke_v6_identity_current(root: Path, identity: SourceSmokeV6Identity) -> None:
    _require(isinstance(identity, SourceSmokeV6Identity), "CS-WG V6 identity must be typed")
    try:
        v1.validate_identity_current(Path(root), identity.inherited_v1_smoke_identity)
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV6Error("CS-WG V6 inherited V1 identity current-byte drift") from error
    _require(identity.accepted_v3_identity.inherited_v1_identity.spec == v1.source_audit_spec(),
             "CS-WG V6 audit-spec identity drift")
    validate_current_closure(Path(root), identity.closure)


class _V6RootReviewSeal:
    pass


_V6_ROOT_REVIEW_SEAL = _V6RootReviewSeal()


@dataclass(frozen=True)
class SourceSmokeV6Capability:
    identity_sha256: str
    v3_completed_graph_sha256: str
    v5_failed_graph_sha256: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require(all(_require_sha(value, "capability graph") for value in (
            self.identity_sha256, self.v3_completed_graph_sha256, self.v5_failed_graph_sha256,
        )) and self._seal is _V6_ROOT_REVIEW_SEAL,
                 "CS-WG V6 requires an in-process root-reviewed capability")


def _assert_v6_root_fresh(root: Path, spec: SourceSmokeV6Spec) -> None:
    candidate = Path(root).absolute() / _safe_relative(spec.root_relative)
    try:
        os.lstat(candidate)
    except FileNotFoundError:
        return
    except OSError as error:
        raise SourceSmokeV6Error("CS-WG V6 prospective smoke root cannot be safely inspected") from error
    raise SourceSmokeV6Error("CS-WG V6 canonical smoke root already exists")


def _typed_v3_identity_matches(identity: SourceSmokeV6Identity, graph: v5.HeldHistoricalGraph) -> None:
    observed = historical_v3_accepted_identity(graph)
    _require(observed.inherited_v1_identity.spec == identity.accepted_v3_identity.inherited_v1_identity.spec,
             "CS-WG V6 typed V3 audit-spec/rebind binding drift")


def _issue_root_reviewed_source_smoke_v6_capability(
    root: Path, identity: SourceSmokeV6Identity, *, environ: Mapping[str, str] | None,
    review_seal: object, predecessor_loader: Any,
) -> SourceSmokeV6Capability:
    _require(review_seal is _V6_ROOT_REVIEW_SEAL,
             "only the root reviewer may issue CS-WG V6 smoke capability")
    validate_source_smoke_v6_identity_current(Path(root), identity)
    predecessors = predecessor_loader(Path(root))
    _require(isinstance(predecessors, tuple) and len(predecessors) == 2
             and all(isinstance(item, v5.HeldHistoricalGraph) for item in predecessors),
             "CS-WG V6 predecessor loader type drift")
    v3_graph, v5_graph = predecessors
    _require(v3_graph.expectation == V3_COMPLETED_EXPECTATION and v5_graph.expectation == V5_FAILED_EXPECTATION,
             "CS-WG V6 predecessor expectation drift")
    _typed_v3_identity_matches(identity, v3_graph)
    try:
        v1.validate_device_environment(identity.device, environ)
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV6Error("CS-WG V6 selected device/thread environment drift") from error
    _assert_v6_root_fresh(Path(root), identity.spec)
    return SourceSmokeV6Capability(identity.sha256, v3_graph.sha256, v5_graph.sha256, _V6_ROOT_REVIEW_SEAL)


def issue_root_reviewed_source_smoke_v6_capability(
    root: Path, identity: SourceSmokeV6Identity, *, environ: Mapping[str, str] | None, review_seal: object,
) -> SourceSmokeV6Capability:
    return _issue_root_reviewed_source_smoke_v6_capability(
        Path(root), identity, environ=environ, review_seal=review_seal, predecessor_loader=validate_v6_predecessors,
    )


def _require_v6_capability(capability: object, identity: SourceSmokeV6Identity) -> SourceSmokeV6Capability:
    _require(isinstance(capability, SourceSmokeV6Capability) and capability._seal is _V6_ROOT_REVIEW_SEAL
             and capability.identity_sha256 == identity.sha256,
             "CS-WG V6 smoke requires exact root-reviewed capability")
    return capability


def _prepared_authority_fragment(prepared: object) -> dict[str, object]:
    return v5._prepared_authority_fragment(prepared)


def _prepared_cache(prepared: object) -> dict[str, object]:
    return v5._prepared_cache(prepared)


def _prepared_fallback(prepared: object) -> tuple[dict[str, object], str, dict[str, object]]:
    return v5._prepared_fallback(prepared)


def _validate_rebound_prepared(identity: SourceSmokeV6Identity, prepared: object) -> None:
    audit = getattr(prepared, "audit_prepared", None)
    smoke = getattr(prepared, "inherited_smoke_prepared", None)
    _require(audit is not None and smoke is not None
             and getattr(audit, "spec", None) == identity.accepted_v3_identity.inherited_v1_identity.spec
             and getattr(smoke, "spec", None) == identity.inherited_v1_smoke_identity.spec,
             "CS-WG V6 audit-spec-to-smoke prepared rebind drift")


def _v6_attempt_payload(identity: SourceSmokeV6Identity, v3_graph: v5.HeldHistoricalGraph,
                        v5_graph: v5.HeldHistoricalGraph) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_source_smoke_attempt_v6",
        "cell": CELL,
        "status": "ATTEMPT_RESERVED",
        "identity": identity.payload(),
        "accepted_v3_completed_graph": v3_graph.payload(),
        "failed_v5_graph": v5_graph.payload(),
        "source_resolved_or_opened": False,
        "model_constructed": False,
        "cuda_initialized": False,
        "optimizer_steps_completed": 0,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _validate_launch_backend(value: object) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG V6 physical launch must be a mapping")
    item = dict(value)
    _require(item.get("schema") == "cross_session_worst_group_m1_source_smoke_physical_launch_v6"
             and item.get("provider") == "V6AuditSpecReboundCommonStratumSourceProvider"
             and item.get("runner") == "TorchCSWGSmokeRunner"
             and item.get("derivative_observer") == "V5DerivativeEvidenceCollector"
             and item.get("audit_spec_then_rebind_to_smoke") is True
             and item.get("source_opened") is False and item.get("model_constructed") is False
             and item.get("cuda_initialized") is False and item.get("optimizer_steps_completed") == 0
             and item.get("source_only") is True,
             "CS-WG V6 physical launch semantics drift")
    return item


def _v6_launch_payload(identity: SourceSmokeV6Identity, v3_graph: v5.HeldHistoricalGraph,
                       v5_graph: v5.HeldHistoricalGraph, attempt_sha256: str,
                       backend: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_source_smoke_launch_v6",
        "cell": CELL,
        "status": "LAUNCHED",
        "identity": identity.payload(),
        "accepted_v3_completed_graph_sha256": v3_graph.sha256,
        "failed_v5_graph_sha256": v5_graph.sha256,
        "attempt_sha256": _require_sha(attempt_sha256, "launch attempt"),
        "backend": _validate_launch_backend(backend),
        "source_resolved_or_opened": False,
        "model_constructed": False,
        "cuda_initialized": False,
        "optimizer_steps_completed": 0,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _v6_source_authority_payload(identity: SourceSmokeV6Identity, v3_graph: v5.HeldHistoricalGraph,
                                 v5_graph: v5.HeldHistoricalGraph, prepared: object,
                                 *, accepted_fallback_sha256: str = V3_FALLBACK_SHA256) -> dict[str, object]:
    _validate_rebound_prepared(identity, prepared)
    inherited = v1.source_authority_payload(identity.inherited_v1_smoke_identity, _prepared_authority_fragment(prepared))
    try:
        v1._validate_source_authority(inherited, identity.inherited_v1_smoke_identity)
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV6Error("CS-WG V6 inherited V1 source authority drift") from error
    fallback, fallback_sha, step_zero = _prepared_fallback(prepared)
    _require(_require_sha(accepted_fallback_sha256, "accepted V3 fallback")
             and fallback_sha == accepted_fallback_sha256 and _sha(_json_bytes(fallback)) == accepted_fallback_sha256
             and fallback == v3_graph.body("source_authority.json").get("deterministic_common_stratum_fallback"),
             "CS-WG V6 prepared common-stratum fallback differs from accepted V3")
    return {
        "schema": "cross_session_worst_group_m1_source_smoke_authority_v6",
        "identity": identity.payload(),
        "accepted_v3_completed_graph_sha256": v3_graph.sha256,
        "failed_v5_graph_sha256": v5_graph.sha256,
        "audit_spec_to_smoke_rebind": {
            "historical_v3_audit_spec": identity.accepted_v3_identity.inherited_v1_identity.spec.payload(),
            "inherited_v1_smoke_spec": identity.inherited_v1_smoke_identity.spec.payload(),
            "v3_audit_builder_received_audit_spec": True,
            "v1_runner_received_rebound_smoke_spec": True,
        },
        "inherited_v1_source_authority": inherited,
        "inherited_v1_source_authority_sha256": _sha(_json_bytes(inherited)),
        "deterministic_common_stratum_fallback": fallback,
        "deterministic_common_stratum_fallback_sha256": fallback_sha,
        "common_stratum_step_zero_evidence": step_zero,
        "route_local_input_digest_cache": _prepared_cache(prepared),
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _validate_v6_source_authority(value: object, identity: SourceSmokeV6Identity, v3_graph: v5.HeldHistoricalGraph,
                                  v5_graph: v5.HeldHistoricalGraph, prepared: object,
                                  *, accepted_fallback_sha256: str = V3_FALLBACK_SHA256) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG V6 source authority must be a mapping")
    item = dict(value)
    _validate_rebound_prepared(identity, prepared)
    fallback, fallback_sha, step_zero = _prepared_fallback(prepared)
    inherited = item.get("inherited_v1_source_authority")
    expected_rebind = {
        "historical_v3_audit_spec": identity.accepted_v3_identity.inherited_v1_identity.spec.payload(),
        "inherited_v1_smoke_spec": identity.inherited_v1_smoke_identity.spec.payload(),
        "v3_audit_builder_received_audit_spec": True,
        "v1_runner_received_rebound_smoke_spec": True,
    }
    _require(item.get("schema") == "cross_session_worst_group_m1_source_smoke_authority_v6"
             and item.get("identity") == identity.payload()
             and item.get("accepted_v3_completed_graph_sha256") == v3_graph.sha256
             and item.get("failed_v5_graph_sha256") == v5_graph.sha256
             and item.get("audit_spec_to_smoke_rebind") == expected_rebind
             and isinstance(inherited, Mapping)
             and item.get("inherited_v1_source_authority_sha256") == _sha(_json_bytes(dict(inherited)))
             and item.get("deterministic_common_stratum_fallback") == fallback
             and _require_sha(accepted_fallback_sha256, "accepted V3 fallback")
             and item.get("deterministic_common_stratum_fallback_sha256") == fallback_sha == accepted_fallback_sha256
             and item.get("common_stratum_step_zero_evidence") == step_zero
             and item.get("route_local_input_digest_cache") == _prepared_cache(prepared)
             and _flags_exact(item), "CS-WG V6 source authority cross-binding drift")
    try:
        v1._validate_source_authority(dict(inherited), identity.inherited_v1_smoke_identity)
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV6Error("CS-WG V6 inherited V1 source authority semantic drift") from error
    return item


def _v6_smoke_payload(identity: SourceSmokeV6Identity, v3_graph: v5.HeldHistoricalGraph,
                      v5_graph: v5.HeldHistoricalGraph, authority_sha256: str,
                      authority: Mapping[str, object], raw_inner: Mapping[str, object],
                      derivative_evidence: object) -> dict[str, object]:
    inner = dict(raw_inner)
    inner.pop("_checkpoint_bodies", None)
    v5._validate_inner_non_derivative_smoke(inner, identity.inherited_v1_smoke_identity)
    evidence = v5._validate_derivative_evidence(
        derivative_evidence, expected_sessions=identity.inherited_v1_smoke_identity.spec.stage0_spec.source_sessions,
    )
    return {
        "schema": "cross_session_worst_group_m1_source_smoke_result_v6",
        "identity_sha256": identity.sha256,
        "source_authority_sha256": _require_sha(authority_sha256, "smoke authority"),
        "accepted_v3_completed_graph_sha256": v3_graph.sha256,
        "failed_v5_graph_sha256": v5_graph.sha256,
        "inherited_v1_smoke": inner,
        "inherited_v1_smoke_sha256": _sha(_json_bytes(inner)),
        "derivative_numeric_evidence": evidence,
        "v5_derivative_numeric_gate_retained_exactly": True,
        "non_derivative_v1_predicates_validated": True,
        "optimizer_steps": inner["optimizer_steps"],
        "one_concatenated_forward_per_step": inner["one_concatenated_forward_per_step"],
        "step_zero_calibration_ownership": authority["common_stratum_step_zero_evidence"],
        "route_local_input_digest_cache": authority["route_local_input_digest_cache"],
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _validate_v6_smoke(value: object, identity: SourceSmokeV6Identity, v3_graph: v5.HeldHistoricalGraph,
                       v5_graph: v5.HeldHistoricalGraph, authority: Mapping[str, object],
                       authority_sha256: str) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG V6 smoke result must be a mapping")
    item = dict(value)
    inherited = item.get("inherited_v1_smoke")
    _require(item.get("schema") == "cross_session_worst_group_m1_source_smoke_result_v6"
             and item.get("identity_sha256") == identity.sha256
             and item.get("source_authority_sha256") == authority_sha256
             and item.get("accepted_v3_completed_graph_sha256") == v3_graph.sha256
             and item.get("failed_v5_graph_sha256") == v5_graph.sha256
             and isinstance(inherited, Mapping)
             and item.get("inherited_v1_smoke_sha256") == _sha(_json_bytes(dict(inherited)))
             and item.get("v5_derivative_numeric_gate_retained_exactly") is True
             and item.get("non_derivative_v1_predicates_validated") is True
             and item.get("optimizer_steps") == v1.SMOKE_STEPS
             and item.get("one_concatenated_forward_per_step") is True
             and item.get("step_zero_calibration_ownership") == authority.get("common_stratum_step_zero_evidence")
             and item.get("route_local_input_digest_cache") == authority.get("route_local_input_digest_cache")
             and _flags_exact(item), "CS-WG V6 smoke wrapper provenance drift")
    v5._validate_inner_non_derivative_smoke(dict(inherited), identity.inherited_v1_smoke_identity)
    v5._validate_derivative_evidence(
        item.get("derivative_numeric_evidence"),
        expected_sessions=identity.inherited_v1_smoke_identity.spec.stage0_spec.source_sessions,
    )
    return item


def _publish_v6_checkpoints(artifact: v1.ImmutableArtifactRoot, identity: SourceSmokeV6Identity,
                            bodies: object, *, inherited_smoke: Mapping[str, object]) -> tuple[dict[str, object], str]:
    _require(isinstance(bodies, Mapping) and tuple(sorted(bodies)) == ("best_source_train_loss", "last"),
             "CS-WG V6 smoke must return exact best/last checkpoint bodies")
    checkpoints: dict[str, dict[str, str]] = {}
    for role in ("best_source_train_loss", "last"):
        body = bodies[role]
        _require(isinstance(body, bytes) and body, f"CS-WG V6 checkpoint {role} body drift")
        state_key = "best_checkpoint_state_sha256" if role == "best_source_train_loss" else "final_model_state_sha256"
        checkpoints[role] = {
            "filename": f"checkpoint_{role}.pt",
            "sha256": artifact.publish_bytes(f"checkpoint_{role}.pt", body),
            "state_sha256": _require_sha(inherited_smoke.get(state_key), f"{role} state"),
        }
    inherited_manifest = {
        "schema": "cross_session_worst_group_m1_source_checkpoint_manifest_v1",
        "identity_sha256": identity.inherited_v1_smoke_identity.sha256,
        "checkpoints": checkpoints,
        "monitor": "source_train_loss_only",
        "early_stopping": False,
        "validation_or_target_selection": False,
        "source_only": True,
    }
    try:
        v1._validate_checkpoint_manifest(inherited_manifest, identity.inherited_v1_smoke_identity)
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV6Error("CS-WG V6 inherited checkpoint manifest drift") from error
    wrapper = {
        "schema": "cross_session_worst_group_m1_source_smoke_checkpoint_manifest_v6",
        "identity_sha256": identity.sha256,
        "inherited_v1_checkpoint_manifest": inherited_manifest,
        "inherited_v1_checkpoint_manifest_sha256": _sha(_json_bytes(inherited_manifest)),
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }
    return wrapper, artifact.publish_json("checkpoint_manifest.json", wrapper)


def _validate_v6_checkpoint_manifest(value: object, identity: SourceSmokeV6Identity) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG V6 checkpoint manifest must be a mapping")
    item = dict(value)
    inherited = item.get("inherited_v1_checkpoint_manifest")
    _require(item.get("schema") == "cross_session_worst_group_m1_source_smoke_checkpoint_manifest_v6"
             and item.get("identity_sha256") == identity.sha256 and isinstance(inherited, Mapping)
             and item.get("inherited_v1_checkpoint_manifest_sha256") == _sha(_json_bytes(dict(inherited)))
             and _flags_exact(item), "CS-WG V6 checkpoint wrapper drift")
    try:
        v1._validate_checkpoint_manifest(dict(inherited), identity.inherited_v1_smoke_identity)
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV6Error("CS-WG V6 inherited checkpoint manifest semantic drift") from error
    return item


def _v6_terminal_payload(identity: SourceSmokeV6Identity, v3_graph: v5.HeldHistoricalGraph,
                         v5_graph: v5.HeldHistoricalGraph, *, attempt_sha256: str,
                         launch_sha256: str, authority_sha256: str, smoke_sha256: str,
                         checkpoints_sha256: str) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_source_smoke_terminal_v6",
        "cell": CELL,
        "status": "PASS_SOURCE_SMOKE_COMMON_STRATUM_CONSTRUCTIBLE_AUDIT_SPEC_REBOUND",
        "identity": identity.payload(),
        "accepted_v3_completed_graph_sha256": v3_graph.sha256,
        "failed_v5_graph_sha256": v5_graph.sha256,
        "attempt_sha256": _require_sha(attempt_sha256, "terminal attempt"),
        "launch_sha256": _require_sha(launch_sha256, "terminal launch"),
        "source_authority_sha256": _require_sha(authority_sha256, "terminal authority"),
        "smoke_sha256": _require_sha(smoke_sha256, "terminal smoke"),
        "checkpoint_manifest_sha256": _require_sha(checkpoints_sha256, "terminal checkpoints"),
        "launch_closure_sha256": identity.closure["closure_sha256"],
        "final_closure_sha256": identity.closure["closure_sha256"],
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _failure_details(error: BaseException) -> tuple[str, str, dict[str, object]]:
    stage, predicate, observed = v5._failure_details(error)
    _require(stage in {"v5_smoke_validation", "derivative_observer", "physical_or_lifecycle"}
             and predicate in {
                 "main_schema_scalars", "gradient_coverage", "rng", "resources", "derivative_numeric_gate",
                 "derivative_observer_exception", "backend_or_lifecycle_exception",
             }, "CS-WG V6 inherited V5 failure predicate drift")
    return stage, predicate, observed


def _v6_failure_payload(identity: SourceSmokeV6Identity, v3_graph: v5.HeldHistoricalGraph,
                        v5_graph: v5.HeldHistoricalGraph, *, attempt_sha256: str,
                        launch_sha256: str | None, authority_sha256: str | None,
                        progress: v1.LifecycleProgress, error: BaseException) -> dict[str, object]:
    stage, predicate, observed = _failure_details(error)
    return {
        "schema": "cross_session_worst_group_m1_source_smoke_failure_v6",
        "cell": CELL,
        "status": "FAILED",
        "identity": identity.payload(),
        "accepted_v3_completed_graph_sha256": v3_graph.sha256,
        "failed_v5_graph_sha256": v5_graph.sha256,
        "attempt_sha256": _require_sha(attempt_sha256, "failure attempt"),
        "launch_sha256": None if launch_sha256 is None else _require_sha(launch_sha256, "failure launch"),
        "source_authority_sha256": None if authority_sha256 is None else _require_sha(authority_sha256, "failure authority"),
        "progress": progress.payload(),
        "failure_stage": stage,
        "failed_predicate": predicate,
        "observed_summary": observed,
        "error_class": type(error).__name__,
        "error_sha256": _sha(repr(error).encode("utf-8")),
        "terminal_published": False,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


@dataclass(frozen=True)
class SourceSmokeV6LifecycleResult:
    root_identity: tuple[int, int]
    attempt_sha256: str
    launch_sha256: str | None
    source_authority_sha256: str | None
    smoke_sha256: str | None
    checkpoint_manifest_sha256: str | None
    terminal_sha256: str | None
    failure_sha256: str | None


class DeferredSourceSmokeV6Backend(Protocol):
    def launch_payload(self, identity: SourceSmokeV6Identity) -> Mapping[str, object]: ...
    def prepare_source(self, identity: SourceSmokeV6Identity) -> object: ...
    def run_smoke(self, identity: SourceSmokeV6Identity) -> Mapping[str, object]: ...
    def checkpoint_bodies(self) -> Mapping[str, bytes]: ...
    def progress(self) -> v1.LifecycleProgress: ...
    def close(self) -> None: ...


def _revalidate_v6_published_graph(artifact: v1.ImmutableArtifactRoot, identity: SourceSmokeV6Identity,
                                  v3_graph: v5.HeldHistoricalGraph, v5_graph: v5.HeldHistoricalGraph,
                                  prepared: object, *, attempt_sha256: str, launch_sha256: str,
                                  authority_sha256: str, smoke_sha256: str,
                                  checkpoints_sha256: str,
                                  accepted_fallback_sha256: str = V3_FALLBACK_SHA256) -> None:
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
    _require(attempt == _v6_attempt_payload(identity, v3_graph, v5_graph)
             and launch == _v6_launch_payload(identity, v3_graph, v5_graph, attempt_sha256, launch["backend"]),
             "CS-WG V6 published attempt/launch graph drift")
    _validate_v6_source_authority(
        authority, identity, v3_graph, v5_graph, prepared,
        accepted_fallback_sha256=accepted_fallback_sha256,
    )
    _validate_v6_smoke(smoke, identity, v3_graph, v5_graph, authority, authority_sha256)
    inherited = _validate_v6_checkpoint_manifest(manifest, identity)["inherited_v1_checkpoint_manifest"]
    assert isinstance(inherited, Mapping)
    for role in ("best_source_train_loss", "last"):
        entry = inherited["checkpoints"][role]
        assert isinstance(entry, Mapping)
        artifact.read_bytes_pair(str(entry["filename"]), expected_sha256=str(entry["sha256"]))


def _execute_reviewed_source_smoke_v6(
    root: Path, *, identity: SourceSmokeV6Identity, capability: object, backend: DeferredSourceSmokeV6Backend,
    environ: Mapping[str, str] | None, predecessor_loader: Any,
    typed_v3_identity_validator: Any = _typed_v3_identity_matches,
    accepted_fallback_sha256: str = V3_FALLBACK_SHA256,
) -> SourceSmokeV6LifecycleResult:
    cap = _require_v6_capability(capability, identity)
    validate_source_smoke_v6_identity_current(Path(root), identity)
    predecessors = predecessor_loader(Path(root))
    _require(isinstance(predecessors, tuple) and len(predecessors) == 2
             and all(isinstance(item, v5.HeldHistoricalGraph) for item in predecessors),
             "CS-WG V6 predecessor loader type drift")
    v3_graph, v5_graph = predecessors
    _require(v3_graph.expectation == V3_COMPLETED_EXPECTATION and v5_graph.expectation == V5_FAILED_EXPECTATION
             and v3_graph.sha256 == cap.v3_completed_graph_sha256
             and v5_graph.sha256 == cap.v5_failed_graph_sha256,
             "CS-WG V6 predecessor/capability binding drift")
    _require(callable(typed_v3_identity_validator), "CS-WG V6 typed V3 identity validator is absent")
    typed_v3_identity_validator(identity, v3_graph)
    try:
        v1.validate_device_environment(identity.device, environ)
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV6Error("CS-WG V6 device environment drift before reserve") from error
    _assert_v6_root_fresh(Path(root), identity.spec)
    artifact = v1.ImmutableArtifactRoot.reserve(Path(root), identity.spec)  # type: ignore[arg-type]
    attempt_sha256 = artifact.publish_json("attempt.json", _v6_attempt_payload(identity, v3_graph, v5_graph))
    launch_sha256: str | None = None
    authority_sha256: str | None = None
    smoke_sha256: str | None = None
    checkpoints_sha256: str | None = None
    terminal_sha256: str | None = None
    try:
        launch_sha256 = artifact.publish_json(
            "launch.json", _v6_launch_payload(identity, v3_graph, v5_graph, attempt_sha256, backend.launch_payload(identity)),
        )
        prepared = backend.prepare_source(identity)
        authority = _v6_source_authority_payload(
            identity, v3_graph, v5_graph, prepared, accepted_fallback_sha256=accepted_fallback_sha256,
        )
        _validate_v6_source_authority(
            authority, identity, v3_graph, v5_graph, prepared,
            accepted_fallback_sha256=accepted_fallback_sha256,
        )
        authority_sha256 = artifact.publish_json("source_authority.json", authority)
        raw = dict(backend.run_smoke(identity))
        checkpoint_bodies = raw.pop("_checkpoint_bodies", None)
        derivative_evidence = raw.pop("_v5_derivative_evidence", raw.pop("_v6_derivative_evidence", None))
        if checkpoint_bodies is None:
            checkpoint_bodies = backend.checkpoint_bodies()
        smoke = _v6_smoke_payload(identity, v3_graph, v5_graph, authority_sha256, authority, raw, derivative_evidence)
        _validate_v6_smoke(smoke, identity, v3_graph, v5_graph, authority, authority_sha256)
        smoke_sha256 = artifact.publish_json("smoke.json", smoke)
        _manifest, checkpoints_sha256 = _publish_v6_checkpoints(
            artifact, identity, checkpoint_bodies, inherited_smoke=dict(smoke["inherited_v1_smoke"]),
        )
        validate_source_smoke_v6_identity_current(Path(root), identity)
        predecessors_now = predecessor_loader(Path(root))
        _require(isinstance(predecessors_now, tuple) and len(predecessors_now) == 2
                 and predecessors_now[0].sha256 == v3_graph.sha256 == cap.v3_completed_graph_sha256
                 and predecessors_now[1].sha256 == v5_graph.sha256 == cap.v5_failed_graph_sha256,
                 "CS-WG V6 immutable predecessor drifted during smoke")
        typed_v3_identity_validator(identity, predecessors_now[0])
        try:
            v1.validate_device_environment(identity.device, environ)
        except v1.SourceLifecycleError as error:
            raise SourceSmokeV6Error("CS-WG V6 device environment drift before terminal") from error
        _revalidate_v6_published_graph(
            artifact, identity, v3_graph, v5_graph, prepared,
            attempt_sha256=attempt_sha256, launch_sha256=launch_sha256, authority_sha256=authority_sha256,
            smoke_sha256=smoke_sha256, checkpoints_sha256=checkpoints_sha256,
            accepted_fallback_sha256=accepted_fallback_sha256,
        )
        terminal = _v6_terminal_payload(
            identity, v3_graph, v5_graph, attempt_sha256=attempt_sha256, launch_sha256=launch_sha256,
            authority_sha256=authority_sha256, smoke_sha256=smoke_sha256, checkpoints_sha256=checkpoints_sha256,
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
                 "CS-WG V6 published terminal graph drift")
        return SourceSmokeV6LifecycleResult(
            artifact.root_identity, attempt_sha256, launch_sha256, authority_sha256,
            smoke_sha256, checkpoints_sha256, terminal_sha256, None,
        )
    except BaseException as error:
        try:
            progress = backend.progress()
            _require(isinstance(progress, v1.LifecycleProgress), "CS-WG V6 backend progress type drift")
        except BaseException:
            progress = v1.LifecycleProgress()
        failure_sha256 = artifact.publish_json(
            "failure.json",
            _v6_failure_payload(
                identity, v3_graph, v5_graph, attempt_sha256=attempt_sha256, launch_sha256=launch_sha256,
                authority_sha256=authority_sha256, progress=progress, error=error,
            ),
        )
        return SourceSmokeV6LifecycleResult(
            artifact.root_identity, attempt_sha256, launch_sha256, authority_sha256,
            smoke_sha256, checkpoints_sha256, terminal_sha256, failure_sha256,
        )
    finally:
        try:
            backend.close()
        finally:
            artifact.close()


def execute_reviewed_source_smoke_v6(
    root: Path, *, identity: SourceSmokeV6Identity, capability: object, backend: DeferredSourceSmokeV6Backend,
    environ: Mapping[str, str] | None,
) -> SourceSmokeV6LifecycleResult:
    """Root-only V6 physical route; the public CLI remains dry and inert."""
    return _execute_reviewed_source_smoke_v6(
        Path(root), identity=identity, capability=capability, backend=backend, environ=environ,
        predecessor_loader=validate_v6_predecessors,
    )


def dry_plan(root: Path | None = None) -> dict[str, object]:
    """Static declaration only; this function does not inspect ``root``."""
    del root
    return {
        "cell": CELL,
        "phase": PHASE,
        "workorder_sha256": WORKORDER_SHA256,
        "fresh_v6_root_relative": V6_ROOT_RELATIVE,
        "accepted_v3_completed_graph": V3_COMPLETED_EXPECTATION.payload(),
        "failed_v5_graph": {
            **V5_FAILED_EXPECTATION.payload(),
            "error_class": V5_FAILURE_CLASS,
            "error_sha256": V5_FAILURE_ERROR_SHA256,
            "error_preimage": V5_FAILURE_PREIMAGE,
        },
        "audit_spec_to_smoke_rebind": {
            "v3_builder_spec": v1.source_audit_spec().payload(),
            "v1_runner_spec": v1.source_smoke_spec().payload(),
            "direct_smoke_spec_to_v3_builder_forbidden": True,
        },
        "source_only_smoke": {
            "optimizer_steps": v1.SMOKE_STEPS,
            "one_concatenated_mixed_session_forward_per_step": True,
            "total_batch_size": v1.plan.TOTAL_BATCH_SIZE,
            "derivative_numeric_gate": v5.derivative_gate_contract_payload(),
            "amp": False, "tf32": False, "compile": False,
        },
        "opens_source_or_target": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "execution_authorized": False,
    }


__all__ = (
    "CELL", "PHASE", "WORKORDER_RELATIVE", "WORKORDER_SHA256", "V6_ROOT_RELATIVE",
    "V3_COMPLETED_EXPECTATION", "V5_FAILED_EXPECTATION", "SourceSmokeV6Error",
    "HistoricalV1AuditIdentity", "HistoricalV3AcceptedIdentity", "SourceSmokeV6Spec", "SourceSmokeV6Identity",
    "SourceSmokeV6Capability", "SourceSmokeV6LifecycleResult", "implementation_closure", "validate_current_closure",
    "historical_v3_accepted_identity", "validate_v6_predecessors", "build_source_smoke_v6_identity",
    "validate_source_smoke_v6_identity_current", "issue_root_reviewed_source_smoke_v6_capability",
    "execute_reviewed_source_smoke_v6", "dry_plan",
)
