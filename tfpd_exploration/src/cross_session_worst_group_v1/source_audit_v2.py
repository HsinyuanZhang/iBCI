"""CS-WG M1 source-audit V2 namespace successor.

This module is deliberately additive.  It composes the frozen V1 source-only
audit backend and validators, adds a fresh V2 identity/capability lifecycle,
and treats the V1 namespace failure as immutable predecessor evidence.  An
import of this module opens neither a result root nor an NWB and imports no
Torch or CUDA runtime.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
import stat
import sys
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from . import source_lifecycle as v1


class SourceAuditV2Error(RuntimeError):
    """Fail closed for V2 predecessor, namespace, identity, or lifecycle drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceAuditV2Error(message)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64
             and all(character in "0123456789abcdef" for character in value),
             f"CS-WG V2 {label} must be a lowercase SHA-256")
    return value


def _safe_relative(value: object) -> str:
    _require(isinstance(value, str) and value, "CS-WG V2 relative path is absent")
    path = Path(value)
    _require(not path.is_absolute() and ".." not in path.parts and path.name not in {"", ".", ".."},
             "CS-WG V2 relative path is unsafe")
    return path.as_posix()


CELL = "CROSS_SESSION_WORST_GROUP_SPINT_M1_V1"
PHASE = "m1_source_audit_v2_namespace_successor"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CS_WG_M1_SOURCE_AUDIT_V2_NAMESPACE_SUCCESSOR_20260826.md"
WORKORDER_SHA256 = "dc439c5a5692e20e5e6cb3f37ee5fcb7137a35b0cde5f5aaedc90379c3e49607"
V2_ROOT_RELATIVE = "tfpd_exploration/results/cross_session_worst_group_m1_source_audit_v2"
V1_REPAIRED_CLOSURE_SHA256 = "b5d4fdf5505fd53f160d56160daa642e36e8077171f654d9c5cdff6ee8fb7a28"
V1_METADATA_MANIFEST_SHA256 = "4afcfdabe53fe936287d5b4dbc241804904897d8e7de3bcb7b091ed2cde16ff6"

V1_FAILED_ROOT_RELATIVE = "tfpd_exploration/results/cross_session_worst_group_m1_source_audit_v1"
V1_ATTEMPT_SHA256 = "5d0cd206644d29b4ac81139d9a7cbe4aaedc404935a1029597ef1a0f3f9a1e75"
V1_LAUNCH_SHA256 = "895fb18d342a51a91e97e5a89ef13e53d662bd556fc2b0881ca1e7be5a70a99d"
V1_FAILURE_SHA256 = "f5ec355bd26d4bb13f48117e122a2f2b4c5eb3dac2e18cc9b357dfa10554436b"
V1_FAILURE_ERROR_CLASS = "SourceReaderError"
V1_FAILURE_ERROR_SHA256 = "7730762e166a649a38f201a078398d00337352875f4fa2169150fac19c517e65"

_OWNED_PATHS = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/cross_session_worst_group_v1/source_audit_v2.py",
    "tfpd_exploration/scripts/run_cross_session_worst_group_m1_source_audit_v2.py",
    "tfpd_exploration/tests/test_cross_session_worst_group_m1_source_audit_v2.py",
)
_V2_MODULE_NAME = "tfpd_exploration.src.cross_session_worst_group_v1.source_audit_v2"
_V1_PHYSICAL_MODULE = "tfpd_exploration.src.cross_session_worst_group_v1.source_physical"
_V1_READER_MODULE = "tfpd_exploration.src.cross_session_worst_group_v1.source_reader"


def _read_regular_no_follow(root: Path, relative: str) -> str:
    path = Path(root).absolute() / _safe_relative(relative)
    try:
        info = os.lstat(path)
    except OSError as error:
        raise SourceAuditV2Error(f"CS-WG V2 closure leaf inaccessible: {relative}") from error
    _require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode),
             f"CS-WG V2 closure leaf is not a regular non-symlink: {relative}")
    with open(path, "rb") as handle:
        return _sha(handle.read())


def implementation_closure(root: Path) -> dict[str, object]:
    """Bind V1's repaired closure and every V2 executable leaf explicitly."""
    inherited = v1.implementation_closure(Path(root))
    _require(inherited.get("closure_sha256") == V1_REPAIRED_CLOSURE_SHA256,
             "CS-WG V2 inherited repaired V1 closure drift")
    metadata = v1.load_m1_metadata_manifest_authority(Path(root))
    _require(metadata.get("body_sha256") == V1_METADATA_MANIFEST_SHA256,
             "CS-WG V2 sealed M1 metadata manifest drift")
    inherited_rows = inherited.get("paths")
    _require(isinstance(inherited_rows, list) and inherited_rows,
             "CS-WG V2 inherited closure path topology drift")
    rows = [dict(row) for row in inherited_rows]
    known = {row.get("path") for row in rows}
    for relative in _OWNED_PATHS:
        _require(relative not in known, "CS-WG V2 closure path duplication")
        rows.append({"path": relative, "sha256": _read_regular_no_follow(Path(root), relative)})
    _require(rows[len(inherited_rows)]["sha256"] == WORKORDER_SHA256,
             "CS-WG V2 workorder literal/body drift")
    binding = {
        "schema": "cross_session_worst_group_m1_source_audit_v2_closure_v1",
        "inherited_v1_closure_sha256": V1_REPAIRED_CLOSURE_SHA256,
        "sealed_metadata_manifest_sha256": V1_METADATA_MANIFEST_SHA256,
        "paths": rows,
    }
    return {
        **binding,
        "closure_sha256": _sha(_json_bytes(binding)),
    }


def validate_current_closure(root: Path, closure: Mapping[str, object]) -> None:
    _require(isinstance(closure, Mapping), "CS-WG V2 closure must be a mapping")
    current = implementation_closure(Path(root))
    _require(dict(closure) == current, "CS-WG V2 closure/current-byte drift")


@dataclass(frozen=True)
class V1FailedGraphExpectation:
    """Exact immutable V1 failure topology expected by the production route."""

    root_relative: str
    attempt_sha256: str
    launch_sha256: str
    failure_sha256: str
    error_class: str
    error_sha256: str

    def __post_init__(self) -> None:
        _require(_safe_relative(self.root_relative) == self.root_relative
                 and all(_require_sha(value, "V1 failure graph")
                         for value in (self.attempt_sha256, self.launch_sha256, self.failure_sha256,
                                       self.error_sha256))
                 and isinstance(self.error_class, str) and self.error_class,
                 "CS-WG V2 V1 failure expectation drift")

    def pairs(self) -> tuple[tuple[str, str], ...]:
        return (
            ("attempt.json", self.attempt_sha256),
            ("launch.json", self.launch_sha256),
            ("failure.json", self.failure_sha256),
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_source_audit_v1_failed_graph_expectation_v2",
            "root_relative": self.root_relative,
            "pairs": [{"name": name, "sha256": digest} for name, digest in self.pairs()],
            "error_class": self.error_class,
            "error_sha256": self.error_sha256,
            "exact_leaf_count": 6,
            "leaf_mode": "0444",
            "canonical_basename_sidecars": True,
            "source_authority_absent": True,
            "audit_terminal_absent": True,
        }


LIVE_V1_FAILED_EXPECTATION = V1FailedGraphExpectation(
    root_relative=V1_FAILED_ROOT_RELATIVE,
    attempt_sha256=V1_ATTEMPT_SHA256,
    launch_sha256=V1_LAUNCH_SHA256,
    failure_sha256=V1_FAILURE_SHA256,
    error_class=V1_FAILURE_ERROR_CLASS,
    error_sha256=V1_FAILURE_ERROR_SHA256,
)


def _identity_tuple(info: os.stat_result) -> tuple[int, int]:
    return int(info.st_dev), int(info.st_ino)


def _open_dir(path: Path, *, label: str) -> int:
    try:
        return os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise SourceAuditV2Error(f"CS-WG V2 {label} cannot be opened no-follow") from error


def _open_held_result_directory(root: Path, relative: str) -> tuple[int, list[int], tuple[tuple[str, int, int], ...]]:
    """Open a no-follow directory chain and bind named-to-held identities."""
    base = Path(root).absolute()
    try:
        base_info = os.lstat(base)
    except OSError as error:
        raise SourceAuditV2Error("CS-WG V2 predecessor code root is inaccessible") from error
    _require(stat.S_ISDIR(base_info.st_mode) and not stat.S_ISLNK(base_info.st_mode),
             "CS-WG V2 predecessor code root is not a regular directory")
    base_fd = _open_dir(base, label="predecessor code root")
    opened: list[int] = []
    identities: list[tuple[str, int, int]] = [(".", *_identity_tuple(base_info))]
    current_fd = base_fd
    try:
        observed_base = os.fstat(base_fd)
        _require(_identity_tuple(observed_base) == _identity_tuple(base_info)
                 and stat.S_ISDIR(observed_base.st_mode) and not stat.S_ISLNK(observed_base.st_mode),
                 "CS-WG V2 predecessor code root changed between lstat/open")
        pieces = tuple(Path(_safe_relative(relative)).parts)
        for position, piece in enumerate(pieces):
            named = os.stat(piece, dir_fd=current_fd, follow_symlinks=False)
            _require(stat.S_ISDIR(named.st_mode) and not stat.S_ISLNK(named.st_mode),
                     "CS-WG V2 predecessor directory component is unsafe")
            try:
                child_fd = os.open(piece, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                                   | getattr(os, "O_NOFOLLOW", 0), dir_fd=current_fd)
            except OSError as error:
                raise SourceAuditV2Error("CS-WG V2 predecessor directory open drift") from error
            opened.append(child_fd)
            held = os.fstat(child_fd)
            _require(_identity_tuple(held) == _identity_tuple(named)
                     and stat.S_ISDIR(held.st_mode) and not stat.S_ISLNK(held.st_mode),
                     "CS-WG V2 predecessor directory changed between stat/open")
            current_fd = child_fd
            identities.append(("/".join(pieces[:position + 1]), *_identity_tuple(named)))
        final = os.fstat(current_fd)
        _require(stat.S_IMODE(final.st_mode) == 0o755,
                 "CS-WG V2 predecessor root mode drift")
        return base_fd, opened, tuple(identities)
    except BaseException:
        for descriptor in reversed(opened):
            os.close(descriptor)
        os.close(base_fd)
        raise


def _close_chain(base_fd: int, opened: list[int]) -> None:
    for descriptor in reversed(opened):
        os.close(descriptor)
    os.close(base_fd)


def _read_held_leaf(directory_fd: int, name: str) -> bytes:
    try:
        descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
    except OSError as error:
        raise SourceAuditV2Error(f"CS-WG V2 predecessor leaf unavailable: {name}") from error
    try:
        info = os.fstat(descriptor)
        _require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444
                 and info.st_nlink == 1,
                 "CS-WG V2 predecessor leaf type/mode/hardlink drift")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _named_chain_identities(root: Path, relative: str) -> tuple[tuple[str, int, int], ...]:
    """Fresh no-follow reopen used after held predecessor reads."""
    base_fd, opened, identities = _open_held_result_directory(Path(root), relative)
    try:
        return identities
    finally:
        _close_chain(base_fd, opened)


def _forbidden_flags_exact(value: Mapping[str, object], *, require_source_only: bool = True) -> bool:
    return ((not require_source_only or value.get("source_only") is True)
            and value.get("target_optimizer_backward_update") == 0
            and all(value.get(name) is expected for name, expected in v1.FORBIDDEN_SURFACE_FLAGS.items()))


def _validate_v1_failure_semantics(
    attempt: Mapping[str, object], launch: Mapping[str, object], failure: Mapping[str, object],
    expectation: V1FailedGraphExpectation,
) -> None:
    """Validate the live-shaped namespace failure without rebuilding V1 state."""
    identity = attempt.get("identity")
    _require(isinstance(identity, Mapping)
             and attempt.get("schema") == "cross_session_worst_group_m1_source_audit_attempt_v1"
             and attempt.get("cell") == CELL and attempt.get("status") == "ATTEMPT_RESERVED"
             and attempt.get("source_resolved_or_opened") is False
             and attempt.get("model_constructed") is False
             and attempt.get("cuda_initialized") is False
             and attempt.get("optimizer_steps_completed") == 0
             and _forbidden_flags_exact(attempt),
             "CS-WG V2 V1 attempt semantic drift")
    identity_value = dict(identity)
    spec = identity_value.get("spec")
    closure = identity_value.get("closure")
    _require(identity_value.get("schema") == "cross_session_worst_group_m1_source_audit_identity_v1"
             and isinstance(spec, Mapping) and spec.get("run_kind") == "audit"
             and spec.get("root_relative") == v1.SOURCE_AUDIT_ROOT_RELATIVE
             and isinstance(closure, Mapping) and closure.get("closure_sha256") == V1_REPAIRED_CLOSURE_SHA256
             and identity_value.get("m1_metadata_manifest") == v1.m1_metadata_manifest_binding_payload()
             and identity_value.get("model_constructed") is False
             and identity_value.get("cuda_initialized") is False
             and identity_value.get("optimizer_steps") == 0
             and identity_value.get("source_only") is True,
             "CS-WG V2 V1 attempt identity/closure/manifest drift")
    backend = launch.get("backend")
    _require(launch.get("schema") == "cross_session_worst_group_m1_source_audit_launch_v1"
             and launch.get("cell") == CELL and launch.get("status") == "LAUNCHED"
             and launch.get("identity") == identity_value
             and launch.get("attempt_sha256") == expectation.attempt_sha256
             and isinstance(backend, Mapping)
             and backend.get("schema") == "cross_session_worst_group_m1_source_audit_physical_launch_v1"
             and backend.get("provider") == "StrictM1SourceProvider"
             and backend.get("source_opened") is False
             and backend.get("model_constructed") is False
             and backend.get("cuda_initialized") is False
             and backend.get("optimizer_steps_completed") == 0
             and backend.get("source_only") is True
             and launch.get("source_resolved_or_opened") is False
             and launch.get("model_constructed") is False
             and launch.get("cuda_initialized") is False
             and launch.get("optimizer_steps_completed") == 0
             and _forbidden_flags_exact(launch),
             "CS-WG V2 V1 launch semantic drift")
    progress = failure.get("progress")
    _require(failure.get("schema") == "cross_session_worst_group_m1_source_audit_failure_v1"
             and failure.get("cell") == CELL and failure.get("status") == "FAILED"
             and failure.get("identity") == identity_value
             and failure.get("attempt_sha256") == expectation.attempt_sha256
             and failure.get("launch_sha256") == expectation.launch_sha256
             and failure.get("source_authority_sha256") is None
             and isinstance(progress, Mapping)
             and type(progress.get("source_resolved_or_opened")) is bool
             and progress.get("model_constructed") is False
             and progress.get("cuda_initialized") is False
             and progress.get("optimizer_steps_completed") == 0
             and progress.get("source_authority_published") is False
             and _forbidden_flags_exact(progress, require_source_only=False)
             and failure.get("error_class") == expectation.error_class
             and failure.get("error_sha256") == expectation.error_sha256
             and failure.get("terminal_published") is False
             and _forbidden_flags_exact(failure),
             "CS-WG V2 V1 failure semantic/source-model-CUDA boundary drift")


@dataclass(frozen=True)
class HeldV1FailedAuditGraph:
    """Validated immutable V1 failure evidence with held directory identity."""

    expectation: V1FailedGraphExpectation
    root_identity: tuple[int, int]
    named_chain_identities: tuple[tuple[str, int, int], ...]
    attempt: Mapping[str, object] = field(repr=False, compare=False)
    launch: Mapping[str, object] = field(repr=False, compare=False)
    failure: Mapping[str, object] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require(isinstance(self.expectation, V1FailedGraphExpectation)
                 and len(self.root_identity) == 2
                 and all(type(value) is int and value >= 0 for value in self.root_identity)
                 and self.named_chain_identities
                 and all(isinstance(value, Mapping) for value in (self.attempt, self.launch, self.failure)),
                 "CS-WG V2 held V1 predecessor graph type drift")
        object.__setattr__(self, "attempt", MappingProxyType(dict(self.attempt)))
        object.__setattr__(self, "launch", MappingProxyType(dict(self.launch)))
        object.__setattr__(self, "failure", MappingProxyType(dict(self.failure)))

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_source_audit_v1_failed_graph_binding_v2",
            "expectation": self.expectation.payload(),
            "root_identity": list(self.root_identity),
            "named_chain_identities": [list(item) for item in self.named_chain_identities],
            "attempt_sha256": self.expectation.attempt_sha256,
            "launch_sha256": self.expectation.launch_sha256,
            "failure_sha256": self.expectation.failure_sha256,
            "source_authority_absent": True,
            "audit_terminal_absent": True,
            "source_only": True,
            "target_optimizer_backward_update": 0,
            **v1.FORBIDDEN_SURFACE_FLAGS,
        }

    @property
    def sha256(self) -> str:
        return _sha(_json_bytes(self.payload()))


def _validate_held_v1_failed_audit_graph(
    root: Path, expectation: V1FailedGraphExpectation,
) -> HeldV1FailedAuditGraph:
    """Read the complete predecessor under one held no-follow result FD."""
    base_fd, opened, identities = _open_held_result_directory(Path(root), expectation.root_relative)
    result_fd = opened[-1]
    try:
        expected_names = tuple(
            sorted(name for body, _digest in expectation.pairs() for name in (body, f"{body}.sha256")),
        )
        names = tuple(sorted(os.listdir(result_fd)))
        _require(names == expected_names,
                 "CS-WG V2 V1 failed predecessor topology/partial-extra leaf drift")
        values: dict[str, dict[str, object]] = {}
        for body_name, expected_sha in expectation.pairs():
            body = _read_held_leaf(result_fd, body_name)
            sidecar = _read_held_leaf(result_fd, f"{body_name}.sha256")
            _require(_sha(body) == expected_sha
                     and sidecar == f"{expected_sha}  {body_name}\n".encode("ascii"),
                     "CS-WG V2 V1 failed predecessor body/sidecar digest drift")
            try:
                parsed = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise SourceAuditV2Error("CS-WG V2 V1 failed predecessor JSON decode drift") from error
            _require(isinstance(parsed, dict), "CS-WG V2 V1 predecessor body must be an object")
            values[body_name] = parsed
        _require(tuple(sorted(os.listdir(result_fd))) == expected_names,
                 "CS-WG V2 V1 failed predecessor topology changed during held reads")
        held_info = os.fstat(result_fd)
        _validate_v1_failure_semantics(
            values["attempt.json"], values["launch.json"], values["failure.json"], expectation,
        )
        # A second independent no-follow named walk makes replacement of the
        # predecessor directory visible even when the original held FD remains
        # valid.  It never opens an NWB or a V2 output root.
        _require(_named_chain_identities(Path(root), expectation.root_relative) == identities,
                 "CS-WG V2 V1 failed predecessor named directory identity drift")
        return HeldV1FailedAuditGraph(
            expectation=expectation,
            root_identity=_identity_tuple(held_info),
            named_chain_identities=identities,
            attempt=values["attempt.json"],
            launch=values["launch.json"],
            failure=values["failure.json"],
        )
    finally:
        _close_chain(base_fd, opened)


def validate_v1_failed_audit_graph(root: Path) -> HeldV1FailedAuditGraph:
    """Validate exactly the immutable real V1 failure graph; no test override."""
    return _validate_held_v1_failed_audit_graph(Path(root), LIVE_V1_FAILED_EXPECTATION)


@dataclass(frozen=True)
class SourceAuditV2Spec:
    """Fresh V2 root while retaining the exact V1 CPU/source-only fold."""

    root_relative: str = V2_ROOT_RELATIVE

    def __post_init__(self) -> None:
        _require(_safe_relative(self.root_relative) == V2_ROOT_RELATIVE,
                 "CS-WG V2 source-audit root literal drift")

    def payload(self) -> dict[str, object]:
        inherited = v1.source_audit_spec()
        return {
            "schema": "cross_session_worst_group_m1_source_audit_v2_spec_v1",
            "root_relative": self.root_relative,
            "inherited_v1_source_audit_spec": inherited.payload(),
            "source_only": True,
            "model_constructed": False,
            "cuda_initialized": False,
            "optimizer_steps": 0,
            "current_gpu_smoke_capability_issuable": False,
            "target_optimizer_backward_update": 0,
            **v1.FORBIDDEN_SURFACE_FLAGS,
        }


@dataclass(frozen=True)
class SourceAuditV2Identity:
    spec: SourceAuditV2Spec
    inherited_v1_identity: v1.SourceAuditIdentity
    closure: Mapping[str, object]

    def __post_init__(self) -> None:
        _require(isinstance(self.spec, SourceAuditV2Spec)
                 and isinstance(self.inherited_v1_identity, v1.SourceAuditIdentity)
                 and isinstance(self.closure, Mapping)
                 and self.closure.get("closure_sha256") is not None,
                 "CS-WG V2 source-audit identity drift")
        _require(self.inherited_v1_identity.closure.get("closure_sha256") == V1_REPAIRED_CLOSURE_SHA256,
                 "CS-WG V2 inherited V1 identity closure drift")
        object.__setattr__(self, "closure", MappingProxyType(dict(self.closure)))

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_source_audit_identity_v2",
            "cell": CELL,
            "phase": PHASE,
            "spec": self.spec.payload(),
            "inherited_v1_identity": self.inherited_v1_identity.payload(),
            "inherited_v1_repaired_closure_sha256": V1_REPAIRED_CLOSURE_SHA256,
            "sealed_metadata_manifest": v1.m1_metadata_manifest_binding_payload(),
            "v1_failed_predecessor_expectation": LIVE_V1_FAILED_EXPECTATION.payload(),
            "closure": dict(self.closure),
            "source_only": True,
            "model_constructed": False,
            "cuda_initialized": False,
            "optimizer_steps": 0,
            "current_gpu_smoke_capability_issuable": False,
            "target_optimizer_backward_update": 0,
            **v1.FORBIDDEN_SURFACE_FLAGS,
        }

    @property
    def sha256(self) -> str:
        return _sha(_json_bytes(self.payload()))


def _module_path_within(module: object, base: Path) -> bool:
    candidate = getattr(module, "__file__", None)
    if not isinstance(candidate, str):
        return False
    try:
        Path(candidate).resolve(strict=True).relative_to(Path(base).resolve(strict=True))
    except (OSError, ValueError):
        return False
    return True


def bootstrap_reviewed_v1_route(root: Path) -> object:
    """Import V1 only under its fully-qualified route package name.

    It intentionally does not remove or replace any ``sys.modules`` entry.
    If a launcher has already occupied top-level ``src`` with the project
    package, the V2 route fails before source preparation rather than trying to
    repair Python's module cache in place.
    """
    _require(__name__ == _V2_MODULE_NAME,
             "CS-WG V2 must itself be imported through tfpd_exploration.src, never top-level src")
    code_root = Path(root).absolute()
    route_root = code_root / "tfpd_exploration" / "src" / "cross_session_worst_group_v1"
    _require(route_root.is_dir() and not route_root.is_symlink(),
             "CS-WG V2 reviewed route package path drift")
    streaming_src_root = code_root / "streaming_calibration_exp" / "src"
    current_src = sys.modules.get("src")
    if current_src is not None:
        _require(_module_path_within(current_src, streaming_src_root),
                 "CS-WG V2 refuses an existing non-streaming top-level src")
    root_string = str(code_root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    module = importlib.import_module(_V1_PHYSICAL_MODULE)
    _require(_module_path_within(module, route_root),
             "CS-WG V2 reviewed V1 physical module path drift")
    after_src = sys.modules.get("src")
    _require(after_src is current_src,
             "CS-WG V2 bootstrap unexpectedly changed top-level src ownership")
    return module


def reach_deferred_parser_seam(root: Path) -> Mapping[str, object]:
    """Test-only reviewed-bootstrap probe; it does not open source or CUDA."""
    bootstrap_reviewed_v1_route(Path(root))
    reader = importlib.import_module(_V1_READER_MODULE)
    loader = getattr(reader, "load_native_m1_runtime", None)
    _require(callable(loader), "CS-WG V2 deferred parser loader export drift")
    runtime = loader(Path(root))
    module = sys.modules.get("src")
    _require(module is not None
             and _module_path_within(module, Path(root).absolute() / "streaming_calibration_exp" / "src"),
             "CS-WG V2 deferred parser did not acquire historical streaming src ownership")
    return {
        "schema": "cross_session_worst_group_m1_source_audit_v2_deferred_parser_seam_v1",
        "route_module": _V1_PHYSICAL_MODULE,
        "parser_module_path": getattr(runtime, "parser_module_path", None),
        "top_level_src_owner": str(getattr(module, "__file__", "")),
        "opens_nwb": False,
        "initializes_cuda": False,
    }


def build_source_audit_v2_identity(root: Path) -> SourceAuditV2Identity:
    """Construct only code/metadata identity; it never reads V1 result or source."""
    inherited = v1.build_source_audit_identity(Path(root))
    _require(inherited.closure.get("closure_sha256") == V1_REPAIRED_CLOSURE_SHA256,
             "CS-WG V2 inherited V1 source-audit closure drift")
    _require(v1.load_m1_metadata_manifest_authority(Path(root)).get("body_sha256")
             == V1_METADATA_MANIFEST_SHA256,
             "CS-WG V2 sealed metadata authority drift")
    return SourceAuditV2Identity(SourceAuditV2Spec(), inherited, implementation_closure(Path(root)))


def validate_source_audit_v2_identity_current(root: Path, identity: SourceAuditV2Identity) -> None:
    _require(isinstance(identity, SourceAuditV2Identity), "CS-WG V2 identity must be typed")
    v1.validate_source_audit_identity_current(Path(root), identity.inherited_v1_identity)
    _require(identity.inherited_v1_identity.closure.get("closure_sha256") == V1_REPAIRED_CLOSURE_SHA256,
             "CS-WG V2 inherited V1 current closure drift")
    _require(v1.load_m1_metadata_manifest_authority(Path(root)).get("body_sha256")
             == V1_METADATA_MANIFEST_SHA256,
             "CS-WG V2 current metadata manifest drift")
    validate_current_closure(Path(root), identity.closure)


class _V2RootReviewSeal:
    pass


_V2_ROOT_REVIEW_SEAL = _V2RootReviewSeal()


@dataclass(frozen=True)
class SourceAuditV2Capability:
    identity_sha256: str
    predecessor_binding_sha256: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_sha(self.identity_sha256, "V2 capability identity")
        _require_sha(self.predecessor_binding_sha256, "V2 capability predecessor")
        _require(self._seal is _V2_ROOT_REVIEW_SEAL,
                 "CS-WG V2 source audit requires an in-process root-reviewed capability")


def _assert_v2_root_fresh(root: Path, spec: SourceAuditV2Spec) -> None:
    candidate = Path(root).absolute() / _safe_relative(spec.root_relative)
    try:
        os.lstat(candidate)
    except FileNotFoundError:
        return
    except OSError as error:
        raise SourceAuditV2Error("CS-WG V2 prospective root cannot be safely inspected") from error
    raise SourceAuditV2Error("CS-WG V2 canonical source-audit root already exists")


def _issue_source_audit_v2_capability(
    root: Path,
    identity: SourceAuditV2Identity,
    *,
    review_seal: object,
    predecessor_loader: Any,
    route_bootstrapper: Any,
) -> SourceAuditV2Capability:
    _require(review_seal is _V2_ROOT_REVIEW_SEAL,
             "only the root reviewer may issue CS-WG V2 source-audit capability")
    validate_source_audit_v2_identity_current(Path(root), identity)
    predecessor = predecessor_loader(Path(root))
    _require(isinstance(predecessor, HeldV1FailedAuditGraph),
             "CS-WG V2 predecessor loader returned wrong type")
    # Predecessor validation intentionally precedes any prospective V2 root
    # inspection/reservation *or* reviewed physical bootstrap.  A tampered V1
    # graph cannot mint a fresh V2 capability, import its physical route, or
    # create a V2 directory.
    _require(callable(route_bootstrapper), "CS-WG V2 reviewed namespace bootstrap is absent")
    route_bootstrapper(Path(root))
    _assert_v2_root_fresh(Path(root), identity.spec)
    return SourceAuditV2Capability(identity.sha256, predecessor.sha256, _V2_ROOT_REVIEW_SEAL)


def issue_root_reviewed_source_audit_v2_capability(
    root: Path, identity: SourceAuditV2Identity, *, review_seal: object,
) -> SourceAuditV2Capability:
    return _issue_source_audit_v2_capability(
        Path(root), identity, review_seal=review_seal,
        predecessor_loader=validate_v1_failed_audit_graph,
        route_bootstrapper=bootstrap_reviewed_v1_route,
    )


def _require_v2_capability(capability: object, identity: SourceAuditV2Identity) -> SourceAuditV2Capability:
    _require(isinstance(capability, SourceAuditV2Capability)
             and capability._seal is _V2_ROOT_REVIEW_SEAL
             and capability.identity_sha256 == identity.sha256,
             "CS-WG V2 source audit requires exact root-reviewed capability")
    return capability


class DeferredSourceAuditV2Backend(Protocol):
    """The inherited V1 audit backend surface, kept model/CUDA-free."""

    def launch_payload(self, identity: v1.SourceAuditIdentity) -> Mapping[str, object]: ...

    def prepare_source(self, identity: v1.SourceAuditIdentity) -> Mapping[str, object]: ...

    def run_source_audit(self, identity: v1.SourceAuditIdentity) -> Mapping[str, object]: ...

    def progress(self) -> v1.LifecycleProgress: ...

    def close(self) -> None: ...


def build_reviewed_v2_source_audit_backend(*, root: Path, source_root: Path) -> DeferredSourceAuditV2Backend:
    """Construct the inherited deferred physical audit through the safe bootstrap.

    The inherited V1 factory only constructs a metadata-bound deferred reader;
    it cannot resolve the lexical ``source_root`` until V2 has durably
    published its attempt and calls ``prepare_source``.
    """
    module = bootstrap_reviewed_v1_route(Path(root))
    factory = getattr(module, "build_route_owned_source_audit_backend", None)
    _require(callable(factory), "CS-WG V2 inherited route-owned audit factory drift")
    backend = factory(root=Path(root), source_root=Path(source_root))
    _require(all(callable(getattr(backend, name, None))
                 for name in ("launch_payload", "prepare_source", "run_source_audit", "progress", "close")),
             "CS-WG V2 inherited audit backend protocol drift")
    return backend


def _v2_attempt_payload(identity: SourceAuditV2Identity, predecessor: HeldV1FailedAuditGraph) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_source_audit_attempt_v2",
        "cell": CELL,
        "status": "ATTEMPT_RESERVED",
        "identity": identity.payload(),
        "v1_failed_predecessor": predecessor.payload(),
        "source_resolved_or_opened": False,
        "model_constructed": False,
        "cuda_initialized": False,
        "optimizer_steps_completed": 0,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _validate_inherited_launch_backend(value: object) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG V2 inherited launch backend must be a mapping")
    backend = dict(value)
    _require(backend.get("schema") == "cross_session_worst_group_m1_source_audit_physical_launch_v1"
             and backend.get("provider") == "StrictM1SourceProvider"
             and backend.get("source_opened") is False
             and backend.get("model_constructed") is False
             and backend.get("cuda_initialized") is False
             and backend.get("optimizer_steps_completed") == 0
             and backend.get("source_only") is True,
             "CS-WG V2 inherited launch backend boundary drift")
    return backend


def _v2_launch_payload(
    identity: SourceAuditV2Identity, predecessor: HeldV1FailedAuditGraph, attempt_sha256: str,
    backend: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_source_audit_launch_v2",
        "cell": CELL,
        "status": "LAUNCHED",
        "identity": identity.payload(),
        "v1_failed_predecessor_sha256": predecessor.sha256,
        "attempt_sha256": _require_sha(attempt_sha256, "V2 launch attempt"),
        "inherited_v1_backend": _validate_inherited_launch_backend(backend),
        "source_resolved_or_opened": False,
        "model_constructed": False,
        "cuda_initialized": False,
        "optimizer_steps_completed": 0,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _v2_source_authority_payload(
    identity: SourceAuditV2Identity, predecessor: HeldV1FailedAuditGraph,
    prepared: Mapping[str, object],
) -> dict[str, object]:
    inherited = v1.source_authority_payload(identity.inherited_v1_identity, prepared)
    v1._validate_source_authority(inherited, identity.inherited_v1_identity)
    return {
        "schema": "cross_session_worst_group_m1_source_audit_authority_v2",
        "identity": identity.payload(),
        "v1_failed_predecessor_sha256": predecessor.sha256,
        "inherited_v1_source_authority": inherited,
        "inherited_v1_source_authority_sha256": _sha(_json_bytes(inherited)),
        "model_constructed": False,
        "cuda_initialized": False,
        "optimizer_steps_completed": 0,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _validate_v2_source_authority(value: object, identity: SourceAuditV2Identity, predecessor: HeldV1FailedAuditGraph) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG V2 source authority must be a mapping")
    item = dict(value)
    inherited = item.get("inherited_v1_source_authority")
    _require(item.get("schema") == "cross_session_worst_group_m1_source_audit_authority_v2"
             and item.get("identity") == identity.payload()
             and item.get("v1_failed_predecessor_sha256") == predecessor.sha256
             and isinstance(inherited, Mapping)
             and item.get("inherited_v1_source_authority_sha256") == _sha(_json_bytes(dict(inherited)))
             and item.get("model_constructed") is False
             and item.get("cuda_initialized") is False
             and item.get("optimizer_steps_completed") == 0
             and _forbidden_flags_exact(item),
             "CS-WG V2 source authority topology/provenance drift")
    v1._validate_source_authority(dict(inherited), identity.inherited_v1_identity)
    return item


def _v2_audit_payload(
    identity: SourceAuditV2Identity, predecessor: HeldV1FailedAuditGraph,
    authority: Mapping[str, object], authority_sha256: str, inherited_audit: Mapping[str, object],
) -> dict[str, object]:
    inner_authority = authority["inherited_v1_source_authority"]
    _require(isinstance(inner_authority, Mapping), "CS-WG V2 inherited source authority is absent")
    inner_sha = _sha(_json_bytes(dict(inner_authority)))
    inner = dict(inherited_audit)
    inner.setdefault("schema", "cross_session_worst_group_m1_source_audit_v1")
    inner.setdefault("identity_sha256", identity.inherited_v1_identity.sha256)
    inner.setdefault("source_authority_sha256", inner_sha)
    v1._validate_source_audit_result(
        inner, identity.inherited_v1_identity, dict(inner_authority), authority_sha256=inner_sha,
    )
    return {
        "schema": "cross_session_worst_group_m1_source_audit_result_v2",
        "identity_sha256": identity.sha256,
        "source_authority_sha256": _require_sha(authority_sha256, "V2 audit authority"),
        "v1_failed_predecessor_sha256": predecessor.sha256,
        "inherited_v1_source_authority_sha256": inner_sha,
        "inherited_v1_audit": inner,
        "model_constructed": False,
        "cuda_initialized": False,
        "optimizer_steps_completed": 0,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _validate_v2_audit_result(
    value: object, identity: SourceAuditV2Identity, predecessor: HeldV1FailedAuditGraph,
    authority: Mapping[str, object], authority_sha256: str,
) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG V2 audit result must be a mapping")
    item = dict(value)
    inherited_authority = authority.get("inherited_v1_source_authority")
    inner = item.get("inherited_v1_audit")
    _require(isinstance(inherited_authority, Mapping) and isinstance(inner, Mapping)
             and item.get("schema") == "cross_session_worst_group_m1_source_audit_result_v2"
             and item.get("identity_sha256") == identity.sha256
             and item.get("source_authority_sha256") == authority_sha256
             and item.get("v1_failed_predecessor_sha256") == predecessor.sha256
             and item.get("inherited_v1_source_authority_sha256") == _sha(_json_bytes(dict(inherited_authority)))
             and item.get("model_constructed") is False
             and item.get("cuda_initialized") is False
             and item.get("optimizer_steps_completed") == 0
             and _forbidden_flags_exact(item),
             "CS-WG V2 audit result topology/provenance drift")
    v1._validate_source_audit_result(
        dict(inner), identity.inherited_v1_identity, dict(inherited_authority),
        authority_sha256=_sha(_json_bytes(dict(inherited_authority))),
    )
    return item


def _v2_terminal_payload(
    identity: SourceAuditV2Identity, predecessor: HeldV1FailedAuditGraph, *, attempt_sha256: str,
    launch_sha256: str, authority_sha256: str, audit_sha256: str,
) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_source_audit_terminal_v2",
        "cell": CELL,
        "status": "PASS_SOURCE_AUDIT_CONSTRUCTIBLE",
        "identity": identity.payload(),
        "v1_failed_predecessor_sha256": predecessor.sha256,
        "attempt_sha256": _require_sha(attempt_sha256, "V2 terminal attempt"),
        "launch_sha256": _require_sha(launch_sha256, "V2 terminal launch"),
        "source_authority_sha256": _require_sha(authority_sha256, "V2 terminal source authority"),
        "audit_sha256": _require_sha(audit_sha256, "V2 terminal audit"),
        "launch_closure_sha256": identity.closure["closure_sha256"],
        "final_closure_sha256": identity.closure["closure_sha256"],
        "model_constructed": False,
        "cuda_initialized": False,
        "optimizer_steps_completed": 0,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _v2_failure_payload(
    identity: SourceAuditV2Identity, predecessor: HeldV1FailedAuditGraph, *, attempt_sha256: str,
    launch_sha256: str | None, authority_sha256: str | None, progress: v1.LifecycleProgress,
    error: BaseException,
) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_source_audit_failure_v2",
        "cell": CELL,
        "status": "FAILED",
        "identity": identity.payload(),
        "v1_failed_predecessor_sha256": predecessor.sha256,
        "attempt_sha256": _require_sha(attempt_sha256, "V2 failure attempt"),
        "launch_sha256": None if launch_sha256 is None else _require_sha(launch_sha256, "V2 failure launch"),
        "source_authority_sha256": None if authority_sha256 is None else _require_sha(
            authority_sha256, "V2 failure source authority",
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
class SourceAuditV2LifecycleResult:
    root_identity: tuple[int, int]
    attempt_sha256: str
    launch_sha256: str | None
    source_authority_sha256: str | None
    audit_sha256: str | None
    terminal_sha256: str | None
    failure_sha256: str | None


def _execute_reviewed_source_audit_v2(
    root: Path,
    *, identity: SourceAuditV2Identity, capability: object, backend: DeferredSourceAuditV2Backend,
    predecessor_loader: Any, route_bootstrapper: Any,
) -> SourceAuditV2LifecycleResult:
    cap = _require_v2_capability(capability, identity)
    validate_source_audit_v2_identity_current(Path(root), identity)
    predecessor = predecessor_loader(Path(root))
    _require(isinstance(predecessor, HeldV1FailedAuditGraph)
             and predecessor.sha256 == cap.predecessor_binding_sha256,
             "CS-WG V2 predecessor/capability binding drift")
    _require(callable(route_bootstrapper), "CS-WG V2 reviewed namespace bootstrap is absent")
    route_bootstrapper(Path(root))
    _assert_v2_root_fresh(Path(root), identity.spec)
    # The V1 primitive only reads ``root_relative`` and uses its held artifact
    # implementation; V2's typed spec gives it a distinct fresh root.
    artifact = v1.ImmutableArtifactRoot.reserve(Path(root), identity.spec)  # type: ignore[arg-type]
    attempt_sha256 = artifact.publish_json("attempt.json", _v2_attempt_payload(identity, predecessor))
    launch_sha256: str | None = None
    authority_sha256: str | None = None
    audit_sha256: str | None = None
    terminal_sha256: str | None = None
    failure_sha256: str | None = None
    try:
        launch_sha256 = artifact.publish_json(
            "launch.json",
            _v2_launch_payload(identity, predecessor, attempt_sha256,
                               backend.launch_payload(identity.inherited_v1_identity)),
        )
        authority = _v2_source_authority_payload(
            identity, predecessor, backend.prepare_source(identity.inherited_v1_identity),
        )
        _validate_v2_source_authority(authority, identity, predecessor)
        authority_sha256 = artifact.publish_json("source_authority.json", authority)
        audit = _v2_audit_payload(
            identity, predecessor, authority, authority_sha256,
            backend.run_source_audit(identity.inherited_v1_identity),
        )
        _validate_v2_audit_result(audit, identity, predecessor, authority, authority_sha256)
        audit_sha256 = artifact.publish_json("audit.json", audit)
        validate_source_audit_v2_identity_current(Path(root), identity)
        predecessor_now = predecessor_loader(Path(root))
        _require(isinstance(predecessor_now, HeldV1FailedAuditGraph)
                 and predecessor_now.sha256 == predecessor.sha256,
                 "CS-WG V2 V1 predecessor drifted during source audit")
        expected_before_terminal = (
            "attempt.json", "attempt.json.sha256", "launch.json", "launch.json.sha256",
            "source_authority.json", "source_authority.json.sha256", "audit.json", "audit.json.sha256",
        )
        artifact.validate_live(expected_names=expected_before_terminal)
        attempt = artifact.read_json_pair("attempt.json", expected_sha256=attempt_sha256)
        launch = artifact.read_json_pair("launch.json", expected_sha256=launch_sha256)
        authority_reloaded = artifact.read_json_pair("source_authority.json", expected_sha256=authority_sha256)
        audit_reloaded = artifact.read_json_pair("audit.json", expected_sha256=audit_sha256)
        _require(attempt == _v2_attempt_payload(identity, predecessor)
                 and launch == _v2_launch_payload(
                     identity, predecessor, attempt_sha256, launch["inherited_v1_backend"],
                 ),
                 "CS-WG V2 published attempt/launch graph drift")
        _validate_v2_source_authority(authority_reloaded, identity, predecessor)
        _validate_v2_audit_result(audit_reloaded, identity, predecessor, authority_reloaded, authority_sha256)
        terminal = _v2_terminal_payload(
            identity, predecessor, attempt_sha256=attempt_sha256, launch_sha256=launch_sha256,
            authority_sha256=authority_sha256, audit_sha256=audit_sha256,
        )
        terminal_sha256 = artifact.publish_json("terminal.json", terminal)
        artifact.validate_live(expected_names=(*expected_before_terminal, "terminal.json", "terminal.json.sha256"))
        _require(artifact.read_json_pair("terminal.json", expected_sha256=terminal_sha256) == terminal,
                 "CS-WG V2 published terminal graph drift")
        return SourceAuditV2LifecycleResult(
            artifact.root_identity, attempt_sha256, launch_sha256, authority_sha256,
            audit_sha256, terminal_sha256, None,
        )
    except BaseException as error:
        try:
            progress = backend.progress()
            _require(isinstance(progress, v1.LifecycleProgress),
                     "CS-WG V2 backend returned invalid failure progress")
        except BaseException:
            progress = v1.LifecycleProgress()
        failure_sha256 = artifact.publish_json(
            "failure.json",
            _v2_failure_payload(
                identity, predecessor, attempt_sha256=attempt_sha256, launch_sha256=launch_sha256,
                authority_sha256=authority_sha256, progress=progress, error=error,
            ),
        )
        return SourceAuditV2LifecycleResult(
            artifact.root_identity, attempt_sha256, launch_sha256, authority_sha256,
            audit_sha256, terminal_sha256, failure_sha256,
        )
    finally:
        try:
            backend.close()
        finally:
            artifact.close()


def execute_reviewed_source_audit_v2(
    root: Path, *, identity: SourceAuditV2Identity, capability: object, backend: DeferredSourceAuditV2Backend,
) -> SourceAuditV2LifecycleResult:
    """Root-only V2 source audit; it contains no GPU or smoke route."""
    return _execute_reviewed_source_audit_v2(
        Path(root), identity=identity, capability=capability, backend=backend,
        predecessor_loader=validate_v1_failed_audit_graph,
        route_bootstrapper=bootstrap_reviewed_v1_route,
    )


def dry_plan(root: Path | None = None) -> dict[str, object]:
    """Static public declaration.  It imports no Torch and touches no root/data."""
    result: dict[str, object] = {
        "cell": CELL,
        "phase": PHASE,
        "workorder_sha256": WORKORDER_SHA256,
        "v2_root_relative": V2_ROOT_RELATIVE,
        "inherited_v1_repaired_closure_sha256": V1_REPAIRED_CLOSURE_SHA256,
        "sealed_metadata_manifest_sha256": V1_METADATA_MANIFEST_SHA256,
        "v1_failed_predecessor": LIVE_V1_FAILED_EXPECTATION.payload(),
        "reviewed_namespace": {
            "route_import": _V1_PHYSICAL_MODULE,
            "top_level_src_project_owner_forbidden": True,
            "top_level_src_streaming_owner_allowed_only_at_deferred_parser_seam": True,
            "sys_modules_mutation": False,
        },
        "current_gpu_smoke_capability_issuable": False,
        "future_gpu_smoke_requires_successor_identity": True,
        "opens_source_or_target": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "launches": False,
        "execution_authorized": False,
        "capability_requirement": "opaque in-process root-reviewed V2 source-audit capability",
    }
    if root is not None:
        result["closure"] = implementation_closure(Path(root))
    return result


__all__ = (
    "SourceAuditV2Error",
    "HeldV1FailedAuditGraph",
    "LIVE_V1_FAILED_EXPECTATION",
    "SourceAuditV2Capability",
    "SourceAuditV2Identity",
    "SourceAuditV2LifecycleResult",
    "SourceAuditV2Spec",
    "bootstrap_reviewed_v1_route",
    "build_reviewed_v2_source_audit_backend",
    "build_source_audit_v2_identity",
    "dry_plan",
    "execute_reviewed_source_audit_v2",
    "implementation_closure",
    "issue_root_reviewed_source_audit_v2_capability",
    "reach_deferred_parser_seam",
    "validate_v1_failed_audit_graph",
)
