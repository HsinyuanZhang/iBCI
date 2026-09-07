"""CS-WG M1 source-audit V3 deterministic common-stratum successor.

The frozen V1/V2 routes remain untouched.  V3 composes their descriptor-safe
source reader and artifact primitives, but replaces the one pre-model point at
which V1 required all source stratum sets to be identical.  The replacement is
the Stage-0-authorized deterministic ``common ∩ min-count-2`` fallback.

Importing this module is deliberately code/metadata-only: it imports neither
Torch nor the native M1 parser, opens no NWB/result root, and initializes no
CUDA.  The parser remains behind the reviewed fully-qualified bootstrap and
is reached only after a future durable attempt publication.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from . import core
from . import plan
from . import source_audit_v2 as v2
from . import source_lifecycle as v1


class SourceAuditV3Error(RuntimeError):
    """Fail closed for V3 predecessor, fallback, namespace, or lifecycle drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceAuditV3Error(message)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64
             and all(character in "0123456789abcdef" for character in value),
             f"CS-WG V3 {label} must be a lowercase SHA-256")
    return value


def _safe_relative(value: object) -> str:
    _require(isinstance(value, str) and value, "CS-WG V3 relative path is absent")
    path = Path(value)
    _require(not path.is_absolute() and ".." not in path.parts and path.name not in {"", ".", ".."},
             "CS-WG V3 relative path is unsafe")
    return path.as_posix()


CELL = "CROSS_SESSION_WORST_GROUP_SPINT_M1_V1"
PHASE = "m1_source_audit_v3_common_stratum_successor"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CS_WG_M1_SOURCE_AUDIT_V3_COMMON_STRATUM_SUCCESSOR_20260826.md"
WORKORDER_SHA256 = "ebc536ff0a2e1c64e0a6e33b8b6ca0f744f79b08db2b7c56d45275481ea94d14"
V3_ROOT_RELATIVE = "tfpd_exploration/results/cross_session_worst_group_m1_source_audit_v3"

V2_REPAIRED_CLOSURE_SHA256 = "ca77c59103830fcb38e6d51af49b355485b6b916a3b4402e26d4b20e21588fa0"
V2_IDENTITY_SHA256 = "e9565bce5d0f2e4e683b599f5e0e0e65a089eae4d0308afd7aa2d58170a6b581"
V1_REPAIRED_CLOSURE_SHA256 = v2.V1_REPAIRED_CLOSURE_SHA256
M1_METADATA_MANIFEST_SHA256 = v2.V1_METADATA_MANIFEST_SHA256

V2_FAILED_ROOT_RELATIVE = "tfpd_exploration/results/cross_session_worst_group_m1_source_audit_v2"
V2_ATTEMPT_SHA256 = "163e73fc5c793b009e7ec86f4af7cc1841bec0bbf5865629c7013a3df468350b"
V2_LAUNCH_SHA256 = "9af2bb8c400dad927b0d3dbf711d9d24242c0e4bfb55b8163a96e0a5e4ea4fcc"
V2_FAILURE_SHA256 = "945a7b5f843e14a6cb4008309c9df4ab510b5f60dbda4a33ba54e6dc563889cb"
V2_FAILURE_ERROR_CLASS = "SourcePhysicalError"
V2_FAILURE_ERROR_SHA256 = "52dde6c1a3dbbe88f47a5f32e281da20c1b3aeeb32166a14cb1c7fe36108e543"

COMMON_STRATUM_FALLBACK_MODE = "DETERMINISTIC_COMMON_STRATUM_MIN2_PRUNE_V1"
MINIMUM_ROWS_PER_ELIGIBLE_STRATUM = 2
MINIMUM_ELIGIBLE_COMMON_STRATA = 10

_OWNED_PATHS = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/cross_session_worst_group_v1/source_audit_v3.py",
    "tfpd_exploration/scripts/run_cross_session_worst_group_m1_source_audit_v3.py",
    "tfpd_exploration/tests/test_cross_session_worst_group_m1_source_audit_v3.py",
)
_V3_MODULE_NAME = "tfpd_exploration.src.cross_session_worst_group_v1.source_audit_v3"
_V1_PHYSICAL_MODULE = "tfpd_exploration.src.cross_session_worst_group_v1.source_physical"


def _read_regular_no_follow(root: Path, relative: str) -> str:
    path = Path(root).absolute() / _safe_relative(relative)
    try:
        info = os.lstat(path)
    except OSError as error:
        raise SourceAuditV3Error(f"CS-WG V3 closure leaf inaccessible: {relative}") from error
    _require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode),
             f"CS-WG V3 closure leaf is not a regular non-symlink: {relative}")
    with open(path, "rb") as handle:
        return _sha(handle.read())


def implementation_closure(root: Path) -> dict[str, object]:
    """Explicitly bind the frozen V2 closure plus the four V3 leaves."""
    try:
        inherited = v2.implementation_closure(Path(root))
    except v2.SourceAuditV2Error as error:
        raise SourceAuditV3Error("CS-WG V3 inherited V2 closure drift") from error
    _require(inherited.get("closure_sha256") == V2_REPAIRED_CLOSURE_SHA256,
             "CS-WG V3 inherited V2 closure literal/current-byte drift")
    _require(v1.load_m1_metadata_manifest_authority(Path(root)).get("body_sha256")
             == M1_METADATA_MANIFEST_SHA256,
             "CS-WG V3 sealed metadata manifest drift")
    inherited_rows = inherited.get("paths")
    _require(isinstance(inherited_rows, list) and inherited_rows,
             "CS-WG V3 inherited V2 closure topology drift")
    rows = [dict(row) for row in inherited_rows]
    known = {row.get("path") for row in rows}
    for relative in _OWNED_PATHS:
        _require(relative not in known, "CS-WG V3 closure path duplication")
        rows.append({"path": relative, "sha256": _read_regular_no_follow(Path(root), relative)})
    _require(rows[len(inherited_rows)]["sha256"] == WORKORDER_SHA256,
             "CS-WG V3 workorder literal/body drift")
    payload = {
        "schema": "cross_session_worst_group_m1_source_audit_v3_closure_v1",
        "inherited_v2_closure_sha256": V2_REPAIRED_CLOSURE_SHA256,
        "inherited_v1_closure_sha256": V1_REPAIRED_CLOSURE_SHA256,
        "sealed_metadata_manifest_sha256": M1_METADATA_MANIFEST_SHA256,
        "paths": rows,
    }
    return {**payload, "closure_sha256": _sha(_json_bytes(payload))}


def validate_current_closure(root: Path, closure: Mapping[str, object]) -> None:
    _require(isinstance(closure, Mapping) and dict(closure) == implementation_closure(Path(root)),
             "CS-WG V3 closure/current-byte drift")


@dataclass(frozen=True)
class V2FailedGraphExpectation:
    """Exact immutable V2 failure topology required by the V3 route."""

    root_relative: str
    attempt_sha256: str
    launch_sha256: str
    failure_sha256: str
    error_class: str
    error_sha256: str

    def __post_init__(self) -> None:
        _require(_safe_relative(self.root_relative) == self.root_relative
                 and all(_require_sha(value, "V2 failed graph")
                         for value in (self.attempt_sha256, self.launch_sha256,
                                       self.failure_sha256, self.error_sha256))
                 and isinstance(self.error_class, str) and self.error_class,
                 "CS-WG V3 V2 predecessor expectation drift")

    def pairs(self) -> tuple[tuple[str, str], ...]:
        return (
            ("attempt.json", self.attempt_sha256),
            ("launch.json", self.launch_sha256),
            ("failure.json", self.failure_sha256),
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_source_audit_v2_failed_graph_expectation_v3",
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


LIVE_V2_FAILED_EXPECTATION = V2FailedGraphExpectation(
    root_relative=V2_FAILED_ROOT_RELATIVE,
    attempt_sha256=V2_ATTEMPT_SHA256,
    launch_sha256=V2_LAUNCH_SHA256,
    failure_sha256=V2_FAILURE_SHA256,
    error_class=V2_FAILURE_ERROR_CLASS,
    error_sha256=V2_FAILURE_ERROR_SHA256,
)


def _flags_exact(value: Mapping[str, object], *, require_source_only: bool = True) -> bool:
    return ((not require_source_only or value.get("source_only") is True)
            and value.get("target_optimizer_backward_update") == 0
            and all(value.get(name) is expected for name, expected in v1.FORBIDDEN_SURFACE_FLAGS.items()))


def _expected_v2_identity(root: Path) -> v2.SourceAuditV2Identity:
    try:
        identity = v2.build_source_audit_v2_identity(Path(root))
    except v2.SourceAuditV2Error as error:
        raise SourceAuditV3Error("CS-WG V3 inherited V2 identity rebuild drift") from error
    _require(identity.sha256 == V2_IDENTITY_SHA256
             and identity.closure.get("closure_sha256") == V2_REPAIRED_CLOSURE_SHA256,
             "CS-WG V3 inherited V2 identity/closure literal drift")
    return identity


@dataclass(frozen=True)
class HeldV2FailedAuditGraph:
    """Held V2 namespace failure evidence cross-bound to a held V1 graph."""

    expectation: V2FailedGraphExpectation
    root_identity: tuple[int, int]
    named_chain_identities: tuple[tuple[str, int, int], ...]
    v1_predecessor_sha256: str
    attempt: Mapping[str, object] = field(repr=False, compare=False)
    launch: Mapping[str, object] = field(repr=False, compare=False)
    failure: Mapping[str, object] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require(isinstance(self.expectation, V2FailedGraphExpectation)
                 and len(self.root_identity) == 2
                 and all(type(item) is int and item >= 0 for item in self.root_identity)
                 and self.named_chain_identities
                 and _require_sha(self.v1_predecessor_sha256, "held V2/V1 binding")
                 and all(isinstance(item, Mapping) for item in (self.attempt, self.launch, self.failure)),
                 "CS-WG V3 held V2 predecessor graph type drift")
        object.__setattr__(self, "attempt", MappingProxyType(dict(self.attempt)))
        object.__setattr__(self, "launch", MappingProxyType(dict(self.launch)))
        object.__setattr__(self, "failure", MappingProxyType(dict(self.failure)))

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_source_audit_v2_failed_graph_binding_v3",
            "expectation": self.expectation.payload(),
            "root_identity": list(self.root_identity),
            "named_chain_identities": [list(item) for item in self.named_chain_identities],
            "v1_predecessor_sha256": self.v1_predecessor_sha256,
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


def _validate_v2_failure_semantics(
    attempt: Mapping[str, object], launch: Mapping[str, object], failure: Mapping[str, object],
    *, expected: V2FailedGraphExpectation, v1_graph: v2.HeldV1FailedAuditGraph,
    identity: v2.SourceAuditV2Identity,
) -> None:
    identity_payload = identity.payload()
    _require(attempt.get("schema") == "cross_session_worst_group_m1_source_audit_attempt_v2"
             and attempt.get("cell") == CELL and attempt.get("status") == "ATTEMPT_RESERVED"
             and attempt.get("identity") == identity_payload
             and attempt.get("v1_failed_predecessor") == v1_graph.payload()
             and attempt.get("source_resolved_or_opened") is False
             and attempt.get("model_constructed") is False
             and attempt.get("cuda_initialized") is False
             and attempt.get("optimizer_steps_completed") == 0
             and _flags_exact(attempt),
             "CS-WG V3 V2 attempt/identity/V1-link semantic drift")
    backend = launch.get("inherited_v1_backend")
    _require(launch.get("schema") == "cross_session_worst_group_m1_source_audit_launch_v2"
             and launch.get("cell") == CELL and launch.get("status") == "LAUNCHED"
             and launch.get("identity") == identity_payload
             and launch.get("v1_failed_predecessor_sha256") == v1_graph.sha256
             and launch.get("attempt_sha256") == expected.attempt_sha256
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
             and _flags_exact(launch),
             "CS-WG V3 V2 launch/V1-link semantic drift")
    progress = failure.get("progress")
    _require(failure.get("schema") == "cross_session_worst_group_m1_source_audit_failure_v2"
             and failure.get("cell") == CELL and failure.get("status") == "FAILED"
             and failure.get("identity") == identity_payload
             and failure.get("v1_failed_predecessor_sha256") == v1_graph.sha256
             and failure.get("attempt_sha256") == expected.attempt_sha256
             and failure.get("launch_sha256") == expected.launch_sha256
             and failure.get("source_authority_sha256") is None
             and isinstance(progress, Mapping)
             and progress.get("source_resolved_or_opened") is True
             and progress.get("model_constructed") is False
             and progress.get("cuda_initialized") is False
             and progress.get("optimizer_steps_completed") == 0
             and progress.get("source_authority_published") is False
             and _flags_exact(progress, require_source_only=False)
             and failure.get("error_class") == expected.error_class
             and failure.get("error_sha256") == expected.error_sha256
             and failure.get("terminal_published") is False
             and _flags_exact(failure),
             "CS-WG V3 V2 failure source/model/CUDA/optimizer boundary drift")


def _validate_held_v2_failed_audit_graph(
    root: Path, expectation: V2FailedGraphExpectation, v1_graph: v2.HeldV1FailedAuditGraph,
    v2_identity: v2.SourceAuditV2Identity,
) -> HeldV2FailedAuditGraph:
    """Validate the complete V2 failure graph under one held no-follow FD."""
    try:
        base_fd, opened, identities = v2._open_held_result_directory(Path(root), expectation.root_relative)
    except v2.SourceAuditV2Error as error:
        raise SourceAuditV3Error("CS-WG V3 V2 predecessor directory validation failed") from error
    result_fd = opened[-1]
    try:
        expected_names = tuple(sorted(
            (name for body, _digest in expectation.pairs() for name in (body, f"{body}.sha256")),
        ))
        _require(tuple(sorted(os.listdir(result_fd))) == expected_names,
                 "CS-WG V3 V2 failed predecessor topology/partial-extra leaf drift")
        values: dict[str, dict[str, object]] = {}
        for body_name, expected_sha in expectation.pairs():
            try:
                body = v2._read_held_leaf(result_fd, body_name)
                sidecar = v2._read_held_leaf(result_fd, f"{body_name}.sha256")
            except v2.SourceAuditV2Error as error:
                raise SourceAuditV3Error("CS-WG V3 V2 predecessor leaf mode/type drift") from error
            _require(_sha(body) == expected_sha
                     and sidecar == f"{expected_sha}  {body_name}\n".encode("ascii"),
                     "CS-WG V3 V2 predecessor body/sidecar digest drift")
            try:
                parsed = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise SourceAuditV3Error("CS-WG V3 V2 predecessor JSON decode drift") from error
            _require(isinstance(parsed, dict), "CS-WG V3 V2 predecessor JSON object drift")
            values[body_name] = parsed
        _require(tuple(sorted(os.listdir(result_fd))) == expected_names,
                 "CS-WG V3 V2 predecessor topology changed during held reads")
        _validate_v2_failure_semantics(
            values["attempt.json"], values["launch.json"], values["failure.json"],
            expected=expectation, v1_graph=v1_graph, identity=v2_identity,
        )
        try:
            named = v2._named_chain_identities(Path(root), expectation.root_relative)
        except v2.SourceAuditV2Error as error:
            raise SourceAuditV3Error("CS-WG V3 V2 predecessor named revalidation failed") from error
        _require(named == identities, "CS-WG V3 V2 predecessor named directory identity drift")
        held = os.fstat(result_fd)
        return HeldV2FailedAuditGraph(
            expectation=expectation,
            root_identity=(int(held.st_dev), int(held.st_ino)),
            named_chain_identities=identities,
            v1_predecessor_sha256=v1_graph.sha256,
            attempt=values["attempt.json"],
            launch=values["launch.json"],
            failure=values["failure.json"],
        )
    finally:
        v2._close_chain(base_fd, opened)


@dataclass(frozen=True)
class HeldFailurePredecessorChain:
    """Exact V1 and V2 immutable failures held and cross-bound as one chain."""

    v1_graph: v2.HeldV1FailedAuditGraph
    v2_graph: HeldV2FailedAuditGraph

    def __post_init__(self) -> None:
        _require(isinstance(self.v1_graph, v2.HeldV1FailedAuditGraph)
                 and isinstance(self.v2_graph, HeldV2FailedAuditGraph)
                 and self.v2_graph.v1_predecessor_sha256 == self.v1_graph.sha256,
                 "CS-WG V3 held predecessor-chain link drift")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_source_audit_v1_v2_failed_predecessor_chain_v3",
            "v1": self.v1_graph.payload(),
            "v2": self.v2_graph.payload(),
            "v1_sha256": self.v1_graph.sha256,
            "v2_sha256": self.v2_graph.sha256,
            "source_only": True,
            "target_optimizer_backward_update": 0,
            **v1.FORBIDDEN_SURFACE_FLAGS,
        }

    @property
    def sha256(self) -> str:
        return _sha(_json_bytes(self.payload()))


def validate_v1_v2_failed_predecessor_chain(root: Path) -> HeldFailurePredecessorChain:
    """Production loader for the exact immutable V1 then V2 failed graphs."""
    try:
        held_v1 = v2.validate_v1_failed_audit_graph(Path(root))
    except v2.SourceAuditV2Error as error:
        raise SourceAuditV3Error("CS-WG V3 V1 predecessor validation failed") from error
    identity = _expected_v2_identity(Path(root))
    held_v2 = _validate_held_v2_failed_audit_graph(
        Path(root), LIVE_V2_FAILED_EXPECTATION, held_v1, identity,
    )
    return HeldFailurePredecessorChain(held_v1, held_v2)


def _stratum_payloads(items: Sequence[core.TaskStratum]) -> list[dict[str, object]]:
    return [item.payload() for item in tuple(sorted(items))]


def _rows_digest(rows: Sequence[core.SourceEpisodeRow]) -> str:
    return _sha(_json_bytes([
        {
            "session_id": row.session_id,
            "sample_index": row.sample_index,
            "sample_id": row.sample_id,
            "stratum": row.stratum.payload(),
            "input_digests": row.input_digests(),
            "target_sha256": core.array_digest(row.raw_final_target),
        }
        for row in rows
    ]))


def _pool_row_digest(pool: core.SessionStratumPool) -> str:
    return _rows_digest(pool.rows)


def _task_stratum_from_payload(value: object, *, label: str) -> core.TaskStratum:
    """Rehydrate one receipt stratum without accepting an arbitrary mapping."""
    _require(isinstance(value, Mapping)
             and set(value) == {
                 "active_flag", "raw_output_norm_quantile", "dominant_coordinate",
             }
             and type(value.get("active_flag")) is bool
             and type(value.get("raw_output_norm_quantile")) is int
             and type(value.get("dominant_coordinate")) is int,
             f"CS-WG V3 {label} task-stratum payload drift")
    try:
        return core.TaskStratum(
            value["active_flag"], value["raw_output_norm_quantile"], value["dominant_coordinate"],
        )
    except core.CSWGCoreError as error:
        raise SourceAuditV3Error(f"CS-WG V3 {label} task-stratum topology drift") from error


def _validate_fallback_against_inherited_authority(
    fallback: "CommonStratumFallbackEvidence", inherited: Mapping[str, object],
) -> None:
    """Bind every fallback count to the original, unpruned source assignment.

    The V1 authority deliberately persists all physical source rows and their
    source-only assignments.  V3 must not merely report a plausible common
    intersection beside that authority: its count tables have to be exactly
    reconstructible from those original assignments before the route-owned
    training pools are pruned.
    """
    _require(isinstance(fallback, CommonStratumFallbackEvidence)
             and isinstance(inherited, Mapping),
             "CS-WG V3 fallback/inherited authority type drift")
    sessions = fallback.source_sessions
    label_authority = inherited.get("source_label_authority")
    assignments = inherited.get("assigned_strata")
    materials = inherited.get("source_session_materials")
    source_label_digests = label_authority.get("source_label_digests") if isinstance(label_authority, Mapping) else None
    _require(isinstance(label_authority, Mapping)
             and inherited.get("source_label_authority_sha256")
                 == fallback.source_label_authority_sha256
             and fallback.source_label_authority_sha256 == _sha(_json_bytes(dict(label_authority)))
             and label_authority.get("source_sessions") == list(sessions)
             and isinstance(source_label_digests, Mapping)
             and tuple(source_label_digests) == sessions
             and isinstance(assignments, Mapping) and tuple(assignments) == sessions
             and isinstance(materials, list) and len(materials) == len(sessions),
             "CS-WG V3 fallback/source-only authority cross-binding drift")
    for session, material in zip(sessions, materials, strict=True):
        assignment = assignments[session]
        _require(isinstance(assignment, Mapping)
                 and assignment.get("session") == session
                 and assignment.get("authority_sha256") == fallback.source_label_authority_sha256
                 and assignment.get("source_only") is True
                 and isinstance(assignment.get("sample_indices"), list)
                 and isinstance(assignment.get("strata"), list)
                 and len(assignment["sample_indices"]) == len(assignment["strata"])
                 and isinstance(material, Mapping)
                 and material.get("source_label_digest")
                     == source_label_digests.get(session)
                 and material.get("valid_final_bin_count") == len(assignment["strata"]),
                 "CS-WG V3 fallback original source assignment topology drift")
        indices = assignment["sample_indices"]
        _require(all(type(index) is int and index >= 0 for index in indices)
                 and len(set(indices)) == len(indices),
                 "CS-WG V3 fallback original source sample-index drift")
        observed = Counter(
            _task_stratum_from_payload(item, label=f"{session} assigned")
            for item in assignment["strata"]
        )
        _require(tuple(sorted(observed.items())) == fallback.original_counts[session],
                 "CS-WG V3 fallback original count table/assignment drift")


@dataclass(frozen=True)
class CommonStratumFallbackEvidence:
    """Immutable source-only evidence for the declared min2 common fallback."""

    source_sessions: tuple[str, ...]
    source_label_authority_sha256: str
    original_counts: Mapping[str, tuple[tuple[core.TaskStratum, int], ...]]
    original_pool_row_sha256: Mapping[str, str]
    raw_common: tuple[core.TaskStratum, ...]
    eligible_common_min2: tuple[core.TaskStratum, ...]
    retained_counts: Mapping[str, int]
    dropped_noncommon_counts: Mapping[str, int]
    dropped_sparse_common_counts: Mapping[str, int]
    retained_pool_row_sha256: Mapping[str, str]
    mode: str = COMMON_STRATUM_FALLBACK_MODE
    minimum_per_session_count: int = MINIMUM_ROWS_PER_ELIGIBLE_STRATUM
    minimum_eligible_strata: int = MINIMUM_ELIGIBLE_COMMON_STRATA

    def __post_init__(self) -> None:
        sessions = tuple(self.source_sessions)
        counts = {session: tuple(value) for session, value in self.original_counts.items()}
        original_digest = dict(self.original_pool_row_sha256)
        retained = {session: int(value) for session, value in self.retained_counts.items()}
        dropped_noncommon = {session: int(value) for session, value in self.dropped_noncommon_counts.items()}
        dropped_sparse = {session: int(value) for session, value in self.dropped_sparse_common_counts.items()}
        retained_digest = dict(self.retained_pool_row_sha256)
        _require(len(sessions) == 3 and len(set(sessions)) == 3
                 and all(session in plan.HELD_IN_SOURCE_SESSIONS for session in sessions)
                 and tuple(counts) == sessions == tuple(original_digest) == tuple(retained)
                 and tuple(dropped_noncommon) == sessions == tuple(dropped_sparse) == tuple(retained_digest)
                 and self.mode == COMMON_STRATUM_FALLBACK_MODE
                 and self.minimum_per_session_count == MINIMUM_ROWS_PER_ELIGIBLE_STRATUM
                 and self.minimum_eligible_strata == MINIMUM_ELIGIBLE_COMMON_STRATA
                 and tuple(sorted(self.raw_common)) == self.raw_common
                 and tuple(sorted(self.eligible_common_min2)) == self.eligible_common_min2
                 and len(set(self.raw_common)) == len(self.raw_common)
                 and len(set(self.eligible_common_min2)) == len(self.eligible_common_min2)
                 and set(self.eligible_common_min2).issubset(set(self.raw_common))
                 and _require_sha(self.source_label_authority_sha256, "pooled source stratum authority")
                 and all(_require_sha(value, "original pool row") for value in original_digest.values())
                 and all(_require_sha(value, "retained pool row") for value in retained_digest.values()),
                 "CS-WG V3 common-stratum fallback evidence topology drift")
        original_sets: dict[str, set[core.TaskStratum]] = {}
        for session in sessions:
            table = counts[session]
            _require(table and len({item[0] for item in table}) == len(table)
                     and tuple(item[0] for item in table) == tuple(sorted(item[0] for item in table))
                     and all(isinstance(stratum, core.TaskStratum) and type(count) is int and count > 0
                             for stratum, count in table),
                     "CS-WG V3 original source stratum count-table drift")
            original_sets[session] = {stratum for stratum, _count in table}
        union = set().union(*(original_sets[session] for session in sessions))
        raw_common = set.intersection(*(original_sets[session] for session in sessions))
        expected_eligible = {
            stratum for stratum in raw_common
            if all(dict(counts[session])[stratum] >= MINIMUM_ROWS_PER_ELIGIBLE_STRATUM for session in sessions)
        }
        _require(tuple(sorted(raw_common)) == self.raw_common
                 and tuple(sorted(expected_eligible)) == self.eligible_common_min2,
                 "CS-WG V3 common/min2 stratum set drift")
        for session in sessions:
            table_map = dict(counts[session])
            total = sum(table_map.values())
            retained_expected = sum(table_map[stratum] for stratum in expected_eligible)
            noncommon_expected = sum(count for stratum, count in table_map.items() if stratum not in raw_common)
            sparse_expected = sum(
                count for stratum, count in table_map.items()
                if stratum in raw_common and stratum not in expected_eligible
            )
            _require(retained[session] == retained_expected
                     and dropped_noncommon[session] == noncommon_expected
                     and dropped_sparse[session] == sparse_expected
                     and total == retained_expected + noncommon_expected + sparse_expected,
                     "CS-WG V3 fallback retained/dropped count conservation drift")
        object.__setattr__(self, "source_sessions", sessions)
        object.__setattr__(self, "original_counts", MappingProxyType(counts))
        object.__setattr__(self, "original_pool_row_sha256", MappingProxyType(original_digest))
        object.__setattr__(self, "retained_counts", MappingProxyType(retained))
        object.__setattr__(self, "dropped_noncommon_counts", MappingProxyType(dropped_noncommon))
        object.__setattr__(self, "dropped_sparse_common_counts", MappingProxyType(dropped_sparse))
        object.__setattr__(self, "retained_pool_row_sha256", MappingProxyType(retained_digest))

    @property
    def constructible(self) -> bool:
        return len(self.eligible_common_min2) >= self.minimum_eligible_strata

    def payload(self) -> dict[str, object]:
        union = tuple(sorted(set().union(*(
            {stratum for stratum, _count in self.original_counts[session]}
            for session in self.source_sessions
        ))))
        original_sets = {
            session: tuple(stratum for stratum, _count in self.original_counts[session])
            for session in self.source_sessions
        }
        retention: dict[str, dict[str, object]] = {}
        for session in self.source_sessions:
            original_count = sum(count for _stratum, count in self.original_counts[session])
            retained_count = self.retained_counts[session]
            retention[session] = {
                "original_window_count": original_count,
                "retained_eligible_window_count": retained_count,
                "retained_fraction": retained_count / original_count,
                "dropped_noncommon_window_count": self.dropped_noncommon_counts[session],
                "dropped_sparse_common_window_count": self.dropped_sparse_common_counts[session],
                "original_pool_row_sha256": self.original_pool_row_sha256[session],
                "retained_pool_row_sha256": self.retained_pool_row_sha256[session],
            }
        return {
            "schema": "cross_session_worst_group_m1_common_stratum_fallback_v3",
            "mode": self.mode,
            "source_sessions": list(self.source_sessions),
            "source_label_authority_sha256": self.source_label_authority_sha256,
            "minimum_per_session_count": self.minimum_per_session_count,
            "minimum_eligible_strata": self.minimum_eligible_strata,
            "original_stratum_counts": {
                session: [{"stratum": stratum.payload(), "count": count}
                          for stratum, count in self.original_counts[session]]
                for session in self.source_sessions
            },
            "original_stratum_sets": {
                session: _stratum_payloads(original_sets[session]) for session in self.source_sessions
            },
            "union": _stratum_payloads(union),
            "missing_vs_union": {
                session: _stratum_payloads(tuple(sorted(set(union) - set(original_sets[session]))))
                for session in self.source_sessions
            },
            "raw_common_intersection": _stratum_payloads(self.raw_common),
            "eligible_common_min2": _stratum_payloads(self.eligible_common_min2),
            "eligible_common_min2_count": len(self.eligible_common_min2),
            "exact_identical_eligible_training_pool_set": True,
            "training_pool_strata": {
                session: _stratum_payloads(self.eligible_common_min2)
                for session in self.source_sessions
            },
            "per_session_retention": retention,
            "step_zero_max_rows_per_eligible_stratum_per_session": 2,
            "constructible_step_zero_b32": self.constructible,
            "source_only": True,
            "target_labels_used": False,
            "target_optimizer_backward_update": 0,
            **v1.FORBIDDEN_SURFACE_FLAGS,
        }

    @property
    def sha256(self) -> str:
        return _sha(_json_bytes(self.payload()))


class CommonStratumConstructibilityError(SourceAuditV3Error):
    """Typed pre-model failure retaining complete source-only stratum evidence."""

    def __init__(self, evidence: CommonStratumFallbackEvidence) -> None:
        self.evidence = evidence
        super().__init__(
            "CS-WG V3 requires at least ten eligible min2 common task strata; "
            f"observed {len(evidence.eligible_common_min2)}",
        )

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_common_stratum_constructibility_failure_v3",
            "stage": "common_stratum_constructibility",
            "reason": "eligible_common_min2_below_required_10",
            "eligible_common_min2_count": len(self.evidence.eligible_common_min2),
            "minimum_eligible_strata": self.evidence.minimum_eligible_strata,
            "fallback_topology": self.evidence.payload(),
            "model_constructed": False,
            "cuda_initialized": False,
            "optimizer_steps_completed": 0,
            "source_only": True,
            "target_optimizer_backward_update": 0,
            **v1.FORBIDDEN_SURFACE_FLAGS,
        }


def derive_common_stratum_fallback(
    pools: Mapping[str, core.SessionStratumPool], *, source_sessions: Sequence[str],
) -> tuple[CommonStratumFallbackEvidence, Mapping[str, core.SessionStratumPool]]:
    """Derive the only allowed deterministic fallback without fitting on target labels."""
    sessions = tuple(source_sessions)
    _require(len(sessions) == 3 and tuple(pools) == sessions
             and all(isinstance(pools[session], core.SessionStratumPool)
                     and pools[session].session_id == session for session in sessions)
             and len({pools[session].authority_sha256 for session in sessions}) == 1,
             "CS-WG V3 fallback requires exact ordered three-source pools/authority")
    original_counts: dict[str, tuple[tuple[core.TaskStratum, int], ...]] = {}
    original_digests: dict[str, str] = {}
    sets: dict[str, set[core.TaskStratum]] = {}
    for session in sessions:
        counts = Counter(row.stratum for row in pools[session].rows)
        table = tuple(sorted((stratum, int(count)) for stratum, count in counts.items()))
        _require(table, "CS-WG V3 source pool has no original task strata")
        original_counts[session] = table
        original_digests[session] = _pool_row_digest(pools[session])
        sets[session] = set(counts)
    raw_common = set.intersection(*(sets[session] for session in sessions))
    eligible = {
        stratum for stratum in raw_common
        if all(dict(original_counts[session])[stratum] >= MINIMUM_ROWS_PER_ELIGIBLE_STRATUM
               for session in sessions)
    }
    retained_rows_by_session: dict[str, tuple[core.SourceEpisodeRow, ...]] = {}
    retained_counts: dict[str, int] = {}
    noncommon_counts: dict[str, int] = {}
    sparse_counts: dict[str, int] = {}
    retained_digests: dict[str, str] = {}
    for session in sessions:
        original = pools[session]
        retained_rows = tuple(row for row in original.rows if row.stratum in eligible)
        retained_rows_by_session[session] = retained_rows
        retained_counts[session] = len(retained_rows)
        table = dict(original_counts[session])
        noncommon_counts[session] = sum(count for stratum, count in table.items() if stratum not in raw_common)
        sparse_counts[session] = sum(
            count for stratum, count in table.items()
            if stratum in raw_common and stratum not in eligible
        )
        retained_digests[session] = _rows_digest(retained_rows)
    evidence = CommonStratumFallbackEvidence(
        source_sessions=sessions,
        source_label_authority_sha256=pools[sessions[0]].authority_sha256,
        original_counts=original_counts,
        original_pool_row_sha256=original_digests,
        raw_common=tuple(sorted(raw_common)),
        eligible_common_min2=tuple(sorted(eligible)),
        retained_counts=retained_counts,
        dropped_noncommon_counts=noncommon_counts,
        dropped_sparse_common_counts=sparse_counts,
        retained_pool_row_sha256=retained_digests,
    )
    if not evidence.constructible:
        raise CommonStratumConstructibilityError(evidence)
    pruned: dict[str, core.SessionStratumPool] = {}
    for session in sessions:
        original = pools[session]
        pruned_pool = core.SessionStratumPool(
            session, original.authority_sha256, retained_rows_by_session[session],
        )
        _require(pruned_pool.strata == evidence.eligible_common_min2,
                 "CS-WG V3 pruned training pool stratum topology drift")
        pruned[session] = pruned_pool
    _require(all(pruned[session].strata == evidence.eligible_common_min2 for session in sessions),
             "CS-WG V3 training pools do not expose exact eligible common set")
    return evidence, MappingProxyType(pruned)


@dataclass(frozen=True)
class CommonStratumPreparedAudit:
    """V1-compatible pruned prepared fold plus immutable original-pool evidence."""

    inherited_prepared: Any = field(repr=False, compare=False)
    fallback: CommonStratumFallbackEvidence

    def __post_init__(self) -> None:
        prepared = self.inherited_prepared
        spec = getattr(prepared, "spec", None)
        stage0_spec = getattr(spec, "stage0_spec", None)
        prepared_sessions = tuple(getattr(stage0_spec, "source_sessions", ()))
        pools = getattr(prepared, "pools", {})
        sessions = self.fallback.source_sessions
        _require(isinstance(self.fallback, CommonStratumFallbackEvidence)
                 and self.fallback.constructible
                 and prepared_sessions == sessions
                 and isinstance(pools, Mapping) and tuple(pools) == sessions
                 and all(pools[session].strata == self.fallback.eligible_common_min2 for session in sessions),
                 "CS-WG V3 common-stratum prepared fold topology drift")

    @property
    def spec(self) -> v1.SourceRouteSpec:
        return self.inherited_prepared.spec

    def episode(self, step_index: int) -> core.BalancedEpisode:
        return self.inherited_prepared.episode(step_index)

    def step_zero_common_stratum_evidence(self) -> dict[str, object]:
        """Durably describe the actual 11/11/10 constructibility witness.

        ``eligible_common_min2`` is a sufficient mathematical condition, but
        the source authority also records the concrete deterministic episode
        that the inherited B32 builder will hand to its future one-forward
        route.  This prevents a receipt from claiming constructibility without
        binding the quota, rows, and per-stratum maximum used at step zero.
        """
        episode = self.episode(0)
        expected_quota = plan.outer_fold_episode_quota(self.fallback.source_sessions, step_index=0)
        _require(episode.quota.payload() == expected_quota.payload()
                 and sorted(len(batch.rows) for batch in episode.microbatches) == [10, 11, 11]
                 and len(episode.all_rows) == plan.TOTAL_BATCH_SIZE
                 and all(row.stratum in self.fallback.eligible_common_min2 for row in episode.all_rows),
                 "CS-WG V3 step-zero common-stratum quota/eligibility drift")
        per_session: list[dict[str, object]] = []
        for batch in episode.microbatches:
            counts = Counter(row.stratum for row in batch.rows)
            _require(len({row.sample_id for row in batch.rows}) == len(batch.rows)
                     and max(counts.values()) <= MINIMUM_ROWS_PER_ELIGIBLE_STRATUM,
                     "CS-WG V3 step-zero common-stratum duplicate/cap drift")
            per_session.append({
                "session_id": batch.session_id,
                "row_count": len(batch.rows),
                "sample_ids_sha256": _sha(_json_bytes([row.sample_id for row in batch.rows])),
                "per_stratum_counts": [
                    {"stratum": stratum.payload(), "count": counts[stratum]}
                    for stratum in episode.represented_strata
                ],
            })
        return {
            "schema": "cross_session_worst_group_m1_common_stratum_step_zero_v3",
            "quota": expected_quota.payload(),
            "episode_sha256": episode.digest,
            "represented_strata": _stratum_payloads(episode.represented_strata),
            "eligible_common_min2_sha256": _sha(_json_bytes(
                _stratum_payloads(self.fallback.eligible_common_min2),
            )),
            "row_count": len(episode.all_rows),
            "per_session": per_session,
            "max_rows_per_represented_stratum_per_session": MINIMUM_ROWS_PER_ELIGIBLE_STRATUM,
            "no_duplicate_sample_ids_within_session_microbatch": True,
            "uses_only_eligible_common_min2": True,
        }

    def authority_fragment(self) -> dict[str, object]:
        fragment = dict(self.inherited_prepared.authority_fragment())
        fragment.update({
            "deterministic_common_stratum_fallback": self.fallback.payload(),
            "deterministic_common_stratum_fallback_sha256": self.fallback.sha256,
            "common_stratum_step_zero_evidence": self.step_zero_common_stratum_evidence(),
            "original_source_rows_retained_in_authority_evidence": True,
            "training_pool_exposes_only_eligible_common_min2": True,
        })
        return fragment

    def audit_fragment(self, identity: v1.SourceAuditIdentity) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_source_audit_v1",
            "identity_sha256": identity.sha256,
            "valid_source_windows": {
                session: self.inherited_prepared.materials[session].valid_source_windows
                for session in self.fallback.source_sessions
            },
            "common_strata": [item.payload() for item in self.fallback.eligible_common_min2],
            "step_zero_calibration_ownership": self.inherited_prepared.step_zero_calibration_ownership(),
            "paired_cswg_and_matched_erm_steps_per_epoch": self.inherited_prepared.paired_steps_per_epoch,
            "constructible_step_zero_b32": True,
            "deterministic_common_stratum_fallback": self.fallback.payload(),
            "deterministic_common_stratum_fallback_sha256": self.fallback.sha256,
            "common_stratum_step_zero_evidence": self.step_zero_common_stratum_evidence(),
            "model_constructed": False,
            "cuda_initialized": False,
            "optimizer_steps_completed": 0,
            "source_only": True,
            "target_optimizer_backward_update": 0,
            **v1.FORBIDDEN_SURFACE_FLAGS,
        }


def _validate_step_zero_common_stratum_evidence(
    value: object, prepared: CommonStratumPreparedAudit,
) -> dict[str, object]:
    _require(isinstance(value, Mapping) and isinstance(prepared, CommonStratumPreparedAudit),
             "CS-WG V3 step-zero evidence/prepared type drift")
    expected = prepared.step_zero_common_stratum_evidence()
    _require(dict(value) == expected,
             "CS-WG V3 step-zero 11/11/10 constructibility evidence drift")
    return expected


def build_common_stratum_prepared_audit(
    *, physical_module: Any, spec: v1.SourceRouteSpec, descriptors: Sequence[Any], materials: Mapping[str, Any],
) -> CommonStratumPreparedAudit:
    """Build the route-owned pruned fold after exactly one physical source read."""
    sessions = spec.stage0_spec.source_sessions
    descriptor_tuple = tuple(descriptors)
    material_map = dict(materials)
    _require(isinstance(spec, v1.SourceRouteSpec) and spec.run_kind == "audit"
             and tuple(item.session_id for item in descriptor_tuple) == sessions
             and tuple(material_map) == sessions
             and spec.stage0_spec.outer_target_session not in material_map
             and all(material_map[session].descriptor == descriptor
                     for session, descriptor in zip(sessions, descriptor_tuple, strict=True)),
             "CS-WG V3 source selection/outer-target ordering drift")
    labels = {session: material_map[session].labels for session in sessions}
    authority = core.fit_run_spec_source_stratum_authority(spec.stage0_spec, labels)
    assignments = {
        session: core.assign_source_task_strata(labels[session], authority)
        for session in sessions
    }
    original_pools: dict[str, core.SessionStratumPool] = {}
    for session in sessions:
        material = material_map[session]
        assignment = assignments[session]
        bound = {
            index: material.rows_by_sample_index[index].bind_stratum(stratum)
            for index, stratum in zip(assignment.sample_indices, assignment.strata, strict=True)
        }
        original_pools[session] = core.build_session_stratum_pool(assignment, rows_by_sample_index=bound)
    fallback, pruned = derive_common_stratum_fallback(original_pools, source_sessions=sessions)
    quota = plan.outer_fold_episode_quota(sessions, step_index=0)
    episode = core.build_balanced_episode(pruned, quota=quota)
    _require(all(len({row.sample_id for row in batch.rows}) == len(batch.rows)
                 and all(row.stratum in fallback.eligible_common_min2 for row in batch.rows)
                 for batch in episode.microbatches),
             "CS-WG V3 step-zero fallback episode duplicate/eligible-stratum drift")
    compatibility = core.derive_concat_compatibility_authority(episode)
    paired_steps = v1.paired_epoch_step_count(
        spec, {session: material_map[session].valid_source_windows for session in sessions},
    )
    prepared_type = getattr(physical_module, "PreparedSourceFold", None)
    _require(callable(prepared_type), "CS-WG V3 inherited PreparedSourceFold seam drift")
    inherited = prepared_type(
        spec=spec,
        descriptors=descriptor_tuple,
        materials=material_map,
        stratum_authority=authority,
        assigned=assignments,
        pools=pruned,
        compatibility=compatibility,
        paired_steps_per_epoch=paired_steps,
    )
    return CommonStratumPreparedAudit(inherited, fallback)


@dataclass
class CommonStratumSourceProvider:
    """Strict V1 descriptor/reader composition with only fold construction replaced."""

    physical_module: Any = field(repr=False, compare=False)
    manifest: Any = field(repr=False, compare=False)
    reader: Any = field(repr=False, compare=False)
    read_events: list[str] = field(default_factory=list, init=False)
    _source_opened: bool = field(default=False, init=False, repr=False)

    def _resolve_descriptors(self, spec: v1.SourceRouteSpec) -> tuple[Any, ...]:
        frozen_type = getattr(self.physical_module, "FrozenM1SourceManifest", None)
        if isinstance(frozen_type, type) and isinstance(self.manifest, frozen_type):
            resolver = getattr(self.manifest, "select_exact_sources", None)
        else:
            resolver = getattr(self.manifest, "resolve_exact_sources", None)
        _require(callable(resolver), "CS-WG V3 strict descriptor resolver seam drift")
        descriptors = tuple(resolver(spec))
        _require(tuple(item.session_id for item in descriptors) == spec.stage0_spec.source_sessions
                 and spec.stage0_spec.outer_target_session not in {item.session_id for item in descriptors},
                 "CS-WG V3 strict selected source descriptor order/target drift")
        return descriptors

    def prepare(self, spec: v1.SourceRouteSpec) -> CommonStratumPreparedAudit:
        descriptors = self._resolve_descriptors(spec)
        materials: dict[str, Any] = {}
        for descriptor in descriptors:
            self._source_opened = True
            material = self.reader.read_source_session(descriptor)
            _require(getattr(material, "descriptor", None) == descriptor,
                     "CS-WG V3 native reader descriptor/material identity drift")
            self.read_events.append(descriptor.session_id)
            materials[descriptor.session_id] = material
        _require(tuple(self.read_events[-len(descriptors):]) == tuple(item.session_id for item in descriptors),
                 "CS-WG V3 physical source reader order drift")
        return build_common_stratum_prepared_audit(
            physical_module=self.physical_module, spec=spec, descriptors=descriptors, materials=materials,
        )

    def progress(self) -> v1.LifecycleProgress:
        return v1.LifecycleProgress(source_resolved_or_opened=self._source_opened)


@dataclass
class PhysicalCommonStratumAuditBackend:
    """CPU/source-only physical backend; it has no model, CUDA, or optimizer seam."""

    provider: CommonStratumSourceProvider
    _prepared: CommonStratumPreparedAudit | None = field(default=None, init=False, repr=False)
    _progress: v1.LifecycleProgress = field(default_factory=v1.LifecycleProgress, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def launch_payload(self, identity: v1.SourceAuditIdentity) -> Mapping[str, object]:
        _require(isinstance(identity, v1.SourceAuditIdentity) and not self._closed,
                 "CS-WG V3 physical audit launch identity drift")
        return {
            "schema": "cross_session_worst_group_m1_source_audit_physical_launch_v3",
            "provider": "V3CommonStratumSourceProvider",
            "inherited_reader_provider": "StrictM1SourceProvider",
            "fallback_mode": COMMON_STRATUM_FALLBACK_MODE,
            "source_opened": False,
            "model_constructed": False,
            "cuda_initialized": False,
            "optimizer_steps_completed": 0,
            "source_only": True,
        }

    def prepare_source(self, identity: v1.SourceAuditIdentity) -> CommonStratumPreparedAudit:
        _require(isinstance(identity, v1.SourceAuditIdentity) and not self._closed and self._prepared is None,
                 "CS-WG V3 physical source preparation lifecycle drift")
        try:
            prepared = self.provider.prepare(identity.spec)
        except BaseException:
            self._progress = self.provider.progress()
            raise
        self._prepared = prepared
        self._progress = v1.LifecycleProgress(source_resolved_or_opened=True)
        return prepared

    def run_source_audit(self, identity: v1.SourceAuditIdentity) -> Mapping[str, object]:
        _require(isinstance(identity, v1.SourceAuditIdentity) and not self._closed
                 and isinstance(self._prepared, CommonStratumPreparedAudit)
                 and self._prepared.spec == identity.spec,
                 "CS-WG V3 physical audit needs exact prepared common-stratum fold")
        self._progress = v1.LifecycleProgress(
            source_resolved_or_opened=True, source_authority_published=True,
        )
        return self._prepared.audit_fragment(identity)

    def progress(self) -> v1.LifecycleProgress:
        return self._progress

    def close(self) -> None:
        self._closed = True


def bootstrap_reviewed_v3_route(root: Path) -> object:
    """Use V2's frozen qualified bootstrap without top-level ``src`` mutation."""
    _require(__name__ == _V3_MODULE_NAME,
             "CS-WG V3 must itself be imported through tfpd_exploration.src, never top-level src")
    try:
        module = v2.bootstrap_reviewed_v1_route(Path(root))
    except v2.SourceAuditV2Error as error:
        raise SourceAuditV3Error("CS-WG V3 reviewed namespace bootstrap failed") from error
    _require(getattr(module, "__name__", None) == _V1_PHYSICAL_MODULE,
             "CS-WG V3 reviewed physical module name drift")
    return module


def reach_deferred_parser_seam(root: Path) -> Mapping[str, object]:
    """Clean-subprocess test hook; no source body or CUDA action is allowed."""
    bootstrap_reviewed_v3_route(Path(root))
    try:
        result = v2.reach_deferred_parser_seam(Path(root))
    except v2.SourceAuditV2Error as error:
        raise SourceAuditV3Error("CS-WG V3 deferred parser seam failed") from error
    _require(result.get("opens_nwb") is False and result.get("initializes_cuda") is False,
             "CS-WG V3 deferred parser seam source/CUDA boundary drift")
    return {
        "schema": "cross_session_worst_group_m1_source_audit_v3_deferred_parser_seam_v1",
        "inherits": dict(result),
        "opens_nwb": False,
        "initializes_cuda": False,
    }


def build_reviewed_v3_source_audit_backend(*, root: Path, source_root: Path) -> PhysicalCommonStratumAuditBackend:
    """Construct the V3 fallback provider without resolving a source descriptor/body."""
    module = bootstrap_reviewed_v3_route(Path(root))
    factory = getattr(module, "build_route_owned_source_audit_backend", None)
    _require(callable(factory), "CS-WG V3 inherited route-owned audit factory drift")
    inherited = factory(root=Path(root), source_root=Path(source_root))
    base_provider = getattr(inherited, "provider", None)
    _require(base_provider is not None
             and type(base_provider).__name__ == "StrictM1SourceProvider"
             and hasattr(base_provider, "manifest") and hasattr(base_provider, "reader"),
             "CS-WG V3 inherited strict provider seam drift")
    return PhysicalCommonStratumAuditBackend(
        CommonStratumSourceProvider(module, base_provider.manifest, base_provider.reader),
    )


@dataclass(frozen=True)
class SourceAuditV3Spec:
    """Fresh V3 CPU/source-only audit root, never the prospective GPU smoke root."""

    root_relative: str = V3_ROOT_RELATIVE

    def __post_init__(self) -> None:
        _require(_safe_relative(self.root_relative) == V3_ROOT_RELATIVE,
                 "CS-WG V3 source-audit root literal drift")

    def payload(self) -> dict[str, object]:
        inherited = v1.source_audit_spec()
        return {
            "schema": "cross_session_worst_group_m1_source_audit_v3_spec_v1",
            "root_relative": self.root_relative,
            "inherited_v1_source_audit_spec": inherited.payload(),
            "fallback_mode": COMMON_STRATUM_FALLBACK_MODE,
            "minimum_per_session_count": MINIMUM_ROWS_PER_ELIGIBLE_STRATUM,
            "minimum_eligible_strata": MINIMUM_ELIGIBLE_COMMON_STRATA,
            "source_only": True,
            "model_constructed": False,
            "cuda_initialized": False,
            "optimizer_steps": 0,
            "current_gpu_smoke_capability_issuable": False,
            "target_optimizer_backward_update": 0,
            **v1.FORBIDDEN_SURFACE_FLAGS,
        }


@dataclass(frozen=True)
class SourceAuditV3Identity:
    """Typed V3 authority retaining exact V1 and V2 code/receipt lineage."""

    spec: SourceAuditV3Spec
    inherited_v2_identity: v2.SourceAuditV2Identity
    closure: Mapping[str, object]

    def __post_init__(self) -> None:
        _require(isinstance(self.spec, SourceAuditV3Spec)
                 and isinstance(self.inherited_v2_identity, v2.SourceAuditV2Identity)
                 and self.inherited_v2_identity.sha256 == V2_IDENTITY_SHA256
                 and self.inherited_v2_identity.closure.get("closure_sha256") == V2_REPAIRED_CLOSURE_SHA256
                 and isinstance(self.closure, Mapping) and self.closure.get("closure_sha256") is not None,
                 "CS-WG V3 identity inherited V2/closure drift")
        object.__setattr__(self, "closure", MappingProxyType(dict(self.closure)))

    @property
    def inherited_v1_identity(self) -> v1.SourceAuditIdentity:
        return self.inherited_v2_identity.inherited_v1_identity

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_source_audit_identity_v3",
            "cell": CELL,
            "phase": PHASE,
            "spec": self.spec.payload(),
            "inherited_v2_identity": self.inherited_v2_identity.payload(),
            "inherited_v2_identity_sha256": V2_IDENTITY_SHA256,
            "inherited_v2_closure_sha256": V2_REPAIRED_CLOSURE_SHA256,
            "sealed_metadata_manifest": v1.m1_metadata_manifest_binding_payload(),
            "v2_failed_predecessor_expectation": LIVE_V2_FAILED_EXPECTATION.payload(),
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


def build_source_audit_v3_identity(root: Path) -> SourceAuditV3Identity:
    """Build only code/metadata identity; no immutable result or source body is read."""
    inherited = _expected_v2_identity(Path(root))
    return SourceAuditV3Identity(SourceAuditV3Spec(), inherited, implementation_closure(Path(root)))


def validate_source_audit_v3_identity_current(root: Path, identity: SourceAuditV3Identity) -> None:
    _require(isinstance(identity, SourceAuditV3Identity), "CS-WG V3 identity must be typed")
    try:
        v2.validate_source_audit_v2_identity_current(Path(root), identity.inherited_v2_identity)
    except v2.SourceAuditV2Error as error:
        raise SourceAuditV3Error("CS-WG V3 inherited V2 identity current-byte drift") from error
    _require(identity.inherited_v2_identity.sha256 == V2_IDENTITY_SHA256
             and v1.load_m1_metadata_manifest_authority(Path(root)).get("body_sha256")
                 == M1_METADATA_MANIFEST_SHA256,
             "CS-WG V3 inherited identity/metadata literal drift")
    validate_current_closure(Path(root), identity.closure)


class _V3RootReviewSeal:
    pass


_V3_ROOT_REVIEW_SEAL = _V3RootReviewSeal()


@dataclass(frozen=True)
class SourceAuditV3Capability:
    identity_sha256: str
    predecessor_chain_sha256: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_sha(self.identity_sha256, "V3 capability identity")
        _require_sha(self.predecessor_chain_sha256, "V3 capability predecessor chain")
        _require(self._seal is _V3_ROOT_REVIEW_SEAL,
                 "CS-WG V3 source audit requires an in-process root-reviewed capability")


def _assert_v3_root_fresh(root: Path, spec: SourceAuditV3Spec) -> None:
    candidate = Path(root).absolute() / _safe_relative(spec.root_relative)
    try:
        os.lstat(candidate)
    except FileNotFoundError:
        return
    except OSError as error:
        raise SourceAuditV3Error("CS-WG V3 prospective root cannot be safely inspected") from error
    raise SourceAuditV3Error("CS-WG V3 canonical source-audit root already exists")


def _issue_source_audit_v3_capability(
    root: Path, identity: SourceAuditV3Identity, *, review_seal: object,
    predecessor_loader: Any, route_bootstrapper: Any,
) -> SourceAuditV3Capability:
    _require(review_seal is _V3_ROOT_REVIEW_SEAL,
             "only the root reviewer may issue CS-WG V3 source-audit capability")
    validate_source_audit_v3_identity_current(Path(root), identity)
    predecessor = predecessor_loader(Path(root))
    _require(isinstance(predecessor, HeldFailurePredecessorChain),
             "CS-WG V3 predecessor loader returned wrong type")
    # The full immutable V1→V2 chain is validated before bootstrap, fresh-root
    # inspection, capability return, or source preparation.
    _require(callable(route_bootstrapper), "CS-WG V3 reviewed namespace bootstrap is absent")
    route_bootstrapper(Path(root))
    _assert_v3_root_fresh(Path(root), identity.spec)
    return SourceAuditV3Capability(identity.sha256, predecessor.sha256, _V3_ROOT_REVIEW_SEAL)


def issue_root_reviewed_source_audit_v3_capability(
    root: Path, identity: SourceAuditV3Identity, *, review_seal: object,
) -> SourceAuditV3Capability:
    return _issue_source_audit_v3_capability(
        Path(root), identity, review_seal=review_seal,
        predecessor_loader=validate_v1_v2_failed_predecessor_chain,
        route_bootstrapper=bootstrap_reviewed_v3_route,
    )


def _require_v3_capability(capability: object, identity: SourceAuditV3Identity) -> SourceAuditV3Capability:
    _require(isinstance(capability, SourceAuditV3Capability)
             and capability._seal is _V3_ROOT_REVIEW_SEAL
             and capability.identity_sha256 == identity.sha256,
             "CS-WG V3 source audit requires exact root-reviewed capability")
    return capability


def _validate_physical_v3_launch(value: object) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG V3 physical launch must be a mapping")
    launch = dict(value)
    _require(launch.get("schema") == "cross_session_worst_group_m1_source_audit_physical_launch_v3"
             and launch.get("provider") == "V3CommonStratumSourceProvider"
             and launch.get("inherited_reader_provider") == "StrictM1SourceProvider"
             and launch.get("fallback_mode") == COMMON_STRATUM_FALLBACK_MODE
             and launch.get("source_opened") is False
             and launch.get("model_constructed") is False
             and launch.get("cuda_initialized") is False
             and launch.get("optimizer_steps_completed") == 0
             and launch.get("source_only") is True,
             "CS-WG V3 physical launch source/model/CUDA boundary drift")
    return launch


def _v3_attempt_payload(identity: SourceAuditV3Identity, chain: HeldFailurePredecessorChain) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_source_audit_attempt_v3",
        "cell": CELL,
        "status": "ATTEMPT_RESERVED",
        "identity": identity.payload(),
        "v1_v2_failed_predecessor_chain": chain.payload(),
        "source_resolved_or_opened": False,
        "model_constructed": False,
        "cuda_initialized": False,
        "optimizer_steps_completed": 0,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _v3_launch_payload(
    identity: SourceAuditV3Identity, chain: HeldFailurePredecessorChain, attempt_sha256: str,
    backend: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_source_audit_launch_v3",
        "cell": CELL,
        "status": "LAUNCHED",
        "identity": identity.payload(),
        "v1_v2_failed_predecessor_chain_sha256": chain.sha256,
        "attempt_sha256": _require_sha(attempt_sha256, "V3 launch attempt"),
        "backend": _validate_physical_v3_launch(backend),
        "source_resolved_or_opened": False,
        "model_constructed": False,
        "cuda_initialized": False,
        "optimizer_steps_completed": 0,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _v3_source_authority_payload(
    identity: SourceAuditV3Identity, chain: HeldFailurePredecessorChain,
    prepared: CommonStratumPreparedAudit,
) -> dict[str, object]:
    _require(isinstance(prepared, CommonStratumPreparedAudit)
             and prepared.spec == identity.inherited_v1_identity.spec,
             "CS-WG V3 source authority prepared-fold identity drift")
    inherited = v1.source_authority_payload(identity.inherited_v1_identity, prepared.authority_fragment())
    v1._validate_source_authority(inherited, identity.inherited_v1_identity)
    fallback = prepared.fallback.payload()
    step_zero = prepared.step_zero_common_stratum_evidence()
    _require(inherited.get("deterministic_common_stratum_fallback") == fallback
             and inherited.get("deterministic_common_stratum_fallback_sha256") == prepared.fallback.sha256
             and inherited.get("common_stratum_step_zero_evidence") == step_zero,
             "CS-WG V3 source authority fallback cross-binding drift")
    _require(inherited.get("source_label_authority_sha256")
             == prepared.fallback.source_label_authority_sha256,
             "CS-WG V3 source authority pooled-label/fallback binding drift")
    _validate_fallback_against_inherited_authority(prepared.fallback, inherited)
    return {
        "schema": "cross_session_worst_group_m1_source_audit_authority_v3",
        "identity": identity.payload(),
        "v1_v2_failed_predecessor_chain_sha256": chain.sha256,
        "inherited_v1_source_authority": inherited,
        "inherited_v1_source_authority_sha256": _sha(_json_bytes(inherited)),
        "deterministic_common_stratum_fallback": fallback,
        "deterministic_common_stratum_fallback_sha256": prepared.fallback.sha256,
        "common_stratum_step_zero_evidence": step_zero,
        "model_constructed": False,
        "cuda_initialized": False,
        "optimizer_steps_completed": 0,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _validate_v3_source_authority(
    value: object, identity: SourceAuditV3Identity, chain: HeldFailurePredecessorChain,
    prepared: CommonStratumPreparedAudit,
) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG V3 source authority must be a mapping")
    item = dict(value)
    inherited = item.get("inherited_v1_source_authority")
    _require(isinstance(prepared, CommonStratumPreparedAudit),
             "CS-WG V3 source authority prepared-fold type drift")
    fallback = prepared.fallback
    expected_fallback = fallback.payload()
    expected_step_zero = prepared.step_zero_common_stratum_evidence()
    _require(item.get("schema") == "cross_session_worst_group_m1_source_audit_authority_v3"
             and item.get("identity") == identity.payload()
             and item.get("v1_v2_failed_predecessor_chain_sha256") == chain.sha256
             and isinstance(inherited, Mapping)
             and item.get("inherited_v1_source_authority_sha256") == _sha(_json_bytes(dict(inherited)))
             and item.get("deterministic_common_stratum_fallback") == expected_fallback
             and item.get("deterministic_common_stratum_fallback_sha256") == fallback.sha256
             and inherited.get("deterministic_common_stratum_fallback") == expected_fallback
             and inherited.get("deterministic_common_stratum_fallback_sha256") == fallback.sha256
             and item.get("common_stratum_step_zero_evidence") == expected_step_zero
             and inherited.get("common_stratum_step_zero_evidence") == expected_step_zero
             and inherited.get("source_label_authority_sha256") == fallback.source_label_authority_sha256
             and item.get("model_constructed") is False
             and item.get("cuda_initialized") is False
             and item.get("optimizer_steps_completed") == 0
             and _flags_exact(item),
             "CS-WG V3 source authority fallback/provenance drift")
    v1._validate_source_authority(dict(inherited), identity.inherited_v1_identity)
    _validate_fallback_against_inherited_authority(fallback, inherited)
    _validate_step_zero_common_stratum_evidence(item.get("common_stratum_step_zero_evidence"), prepared)
    return item


def _v3_audit_payload(
    identity: SourceAuditV3Identity, chain: HeldFailurePredecessorChain, authority: Mapping[str, object],
    authority_sha256: str, inherited_audit: Mapping[str, object], prepared: CommonStratumPreparedAudit,
) -> dict[str, object]:
    inherited_authority = authority.get("inherited_v1_source_authority")
    _require(isinstance(inherited_authority, Mapping), "CS-WG V3 inherited source authority absent")
    _require(isinstance(prepared, CommonStratumPreparedAudit),
             "CS-WG V3 audit prepared-fold type drift")
    fallback = prepared.fallback
    inner_authority_sha = _sha(_json_bytes(dict(inherited_authority)))
    inner = dict(inherited_audit)
    inner.setdefault("schema", "cross_session_worst_group_m1_source_audit_v1")
    inner.setdefault("identity_sha256", identity.inherited_v1_identity.sha256)
    inner.setdefault("source_authority_sha256", inner_authority_sha)
    _require(inner.get("deterministic_common_stratum_fallback") == fallback.payload()
             and inner.get("deterministic_common_stratum_fallback_sha256") == fallback.sha256
             and inner.get("common_stratum_step_zero_evidence")
                 == prepared.step_zero_common_stratum_evidence(),
             "CS-WG V3 source audit fallback evidence drift")
    v1._validate_source_audit_result(
        inner, identity.inherited_v1_identity, dict(inherited_authority), authority_sha256=inner_authority_sha,
    )
    return {
        "schema": "cross_session_worst_group_m1_source_audit_result_v3",
        "identity_sha256": identity.sha256,
        "source_authority_sha256": _require_sha(authority_sha256, "V3 audit authority"),
        "v1_v2_failed_predecessor_chain_sha256": chain.sha256,
        "inherited_v1_source_authority_sha256": inner_authority_sha,
        "inherited_v1_audit": inner,
        "deterministic_common_stratum_fallback": fallback.payload(),
        "deterministic_common_stratum_fallback_sha256": fallback.sha256,
        "common_stratum_step_zero_evidence": prepared.step_zero_common_stratum_evidence(),
        "model_constructed": False,
        "cuda_initialized": False,
        "optimizer_steps_completed": 0,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _validate_v3_audit_result(
    value: object, identity: SourceAuditV3Identity, chain: HeldFailurePredecessorChain,
    authority: Mapping[str, object], authority_sha256: str, prepared: CommonStratumPreparedAudit,
) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG V3 audit result must be a mapping")
    item = dict(value)
    inherited_authority = authority.get("inherited_v1_source_authority")
    inner = item.get("inherited_v1_audit")
    _require(isinstance(prepared, CommonStratumPreparedAudit),
             "CS-WG V3 audit prepared-fold type drift")
    fallback = prepared.fallback
    expected_fallback = fallback.payload()
    expected_step_zero = prepared.step_zero_common_stratum_evidence()
    _require(isinstance(inherited_authority, Mapping) and isinstance(inner, Mapping)
             and item.get("schema") == "cross_session_worst_group_m1_source_audit_result_v3"
             and item.get("identity_sha256") == identity.sha256
             and item.get("source_authority_sha256") == authority_sha256
             and item.get("v1_v2_failed_predecessor_chain_sha256") == chain.sha256
             and item.get("inherited_v1_source_authority_sha256") == _sha(_json_bytes(dict(inherited_authority)))
             and item.get("deterministic_common_stratum_fallback") == expected_fallback
             and item.get("deterministic_common_stratum_fallback_sha256") == fallback.sha256
             and inner.get("deterministic_common_stratum_fallback") == expected_fallback
             and inner.get("deterministic_common_stratum_fallback_sha256") == fallback.sha256
             and item.get("common_stratum_step_zero_evidence") == expected_step_zero
             and inner.get("common_stratum_step_zero_evidence") == expected_step_zero
             and item.get("model_constructed") is False
             and item.get("cuda_initialized") is False
             and item.get("optimizer_steps_completed") == 0
             and _flags_exact(item),
             "CS-WG V3 audit fallback/provenance drift")
    v1._validate_source_audit_result(
        dict(inner), identity.inherited_v1_identity, dict(inherited_authority),
        authority_sha256=_sha(_json_bytes(dict(inherited_authority))),
    )
    _validate_step_zero_common_stratum_evidence(item.get("common_stratum_step_zero_evidence"), prepared)
    return item


def _v3_terminal_payload(
    identity: SourceAuditV3Identity, chain: HeldFailurePredecessorChain, *, attempt_sha256: str,
    launch_sha256: str, authority_sha256: str, audit_sha256: str, fallback: CommonStratumFallbackEvidence,
) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_source_audit_terminal_v3",
        "cell": CELL,
        "status": "PASS_SOURCE_AUDIT_COMMON_STRATUM_CONSTRUCTIBLE",
        "identity": identity.payload(),
        "v1_v2_failed_predecessor_chain_sha256": chain.sha256,
        "attempt_sha256": _require_sha(attempt_sha256, "V3 terminal attempt"),
        "launch_sha256": _require_sha(launch_sha256, "V3 terminal launch"),
        "source_authority_sha256": _require_sha(authority_sha256, "V3 terminal source authority"),
        "audit_sha256": _require_sha(audit_sha256, "V3 terminal audit"),
        "deterministic_common_stratum_fallback_sha256": fallback.sha256,
        "launch_closure_sha256": identity.closure["closure_sha256"],
        "final_closure_sha256": identity.closure["closure_sha256"],
        "model_constructed": False,
        "cuda_initialized": False,
        "optimizer_steps_completed": 0,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _v3_failure_payload(
    identity: SourceAuditV3Identity, chain: HeldFailurePredecessorChain, *, attempt_sha256: str,
    launch_sha256: str | None, authority_sha256: str | None, progress: v1.LifecycleProgress,
    error: BaseException,
) -> dict[str, object]:
    typed = error.payload() if isinstance(error, CommonStratumConstructibilityError) else None
    return {
        "schema": "cross_session_worst_group_m1_source_audit_failure_v3",
        "cell": CELL,
        "status": "FAILED",
        "identity": identity.payload(),
        "v1_v2_failed_predecessor_chain_sha256": chain.sha256,
        "attempt_sha256": _require_sha(attempt_sha256, "V3 failure attempt"),
        "launch_sha256": None if launch_sha256 is None else _require_sha(launch_sha256, "V3 failure launch"),
        "source_authority_sha256": None if authority_sha256 is None else _require_sha(
            authority_sha256, "V3 failure source authority",
        ),
        "progress": progress.payload(),
        "failure_stage": "common_stratum_constructibility" if typed is not None else "source_audit_runtime",
        "typed_failure_topology": typed,
        "error_class": type(error).__name__,
        "error_sha256": _sha(repr(error).encode("utf-8")),
        "terminal_published": False,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


@dataclass(frozen=True)
class SourceAuditV3LifecycleResult:
    root_identity: tuple[int, int]
    attempt_sha256: str
    launch_sha256: str | None
    source_authority_sha256: str | None
    audit_sha256: str | None
    terminal_sha256: str | None
    failure_sha256: str | None


class DeferredSourceAuditV3Backend(Protocol):
    def launch_payload(self, identity: v1.SourceAuditIdentity) -> Mapping[str, object]: ...

    def prepare_source(self, identity: v1.SourceAuditIdentity) -> CommonStratumPreparedAudit: ...

    def run_source_audit(self, identity: v1.SourceAuditIdentity) -> Mapping[str, object]: ...

    def progress(self) -> v1.LifecycleProgress: ...

    def close(self) -> None: ...


def _execute_reviewed_source_audit_v3(
    root: Path, *, identity: SourceAuditV3Identity, capability: object,
    backend: DeferredSourceAuditV3Backend, predecessor_loader: Any, route_bootstrapper: Any,
) -> SourceAuditV3LifecycleResult:
    cap = _require_v3_capability(capability, identity)
    validate_source_audit_v3_identity_current(Path(root), identity)
    chain = predecessor_loader(Path(root))
    _require(isinstance(chain, HeldFailurePredecessorChain)
             and chain.sha256 == cap.predecessor_chain_sha256,
             "CS-WG V3 predecessor/capability binding drift")
    _require(callable(route_bootstrapper), "CS-WG V3 reviewed namespace bootstrap absent")
    route_bootstrapper(Path(root))
    _assert_v3_root_fresh(Path(root), identity.spec)
    artifact = v1.ImmutableArtifactRoot.reserve(Path(root), identity.spec)  # type: ignore[arg-type]
    attempt_sha256 = artifact.publish_json("attempt.json", _v3_attempt_payload(identity, chain))
    launch_sha256: str | None = None
    authority_sha256: str | None = None
    audit_sha256: str | None = None
    terminal_sha256: str | None = None
    failure_sha256: str | None = None
    try:
        launch_sha256 = artifact.publish_json(
            "launch.json",
            _v3_launch_payload(identity, chain, attempt_sha256, backend.launch_payload(identity.inherited_v1_identity)),
        )
        prepared = backend.prepare_source(identity.inherited_v1_identity)
        _require(isinstance(prepared, CommonStratumPreparedAudit),
                 "CS-WG V3 backend returned wrong prepared common-stratum type")
        authority = _v3_source_authority_payload(identity, chain, prepared)
        _validate_v3_source_authority(authority, identity, chain, prepared)
        authority_sha256 = artifact.publish_json("source_authority.json", authority)
        audit = _v3_audit_payload(
            identity, chain, authority, authority_sha256,
            backend.run_source_audit(identity.inherited_v1_identity), prepared,
        )
        _validate_v3_audit_result(audit, identity, chain, authority, authority_sha256, prepared)
        audit_sha256 = artifact.publish_json("audit.json", audit)
        validate_source_audit_v3_identity_current(Path(root), identity)
        chain_now = predecessor_loader(Path(root))
        _require(isinstance(chain_now, HeldFailurePredecessorChain) and chain_now.sha256 == chain.sha256,
                 "CS-WG V3 immutable predecessor chain drifted during source audit")
        expected_before_terminal = (
            "attempt.json", "attempt.json.sha256", "launch.json", "launch.json.sha256",
            "source_authority.json", "source_authority.json.sha256", "audit.json", "audit.json.sha256",
        )
        artifact.validate_live(expected_names=expected_before_terminal)
        attempt = artifact.read_json_pair("attempt.json", expected_sha256=attempt_sha256)
        launch = artifact.read_json_pair("launch.json", expected_sha256=launch_sha256)
        authority_reloaded = artifact.read_json_pair("source_authority.json", expected_sha256=authority_sha256)
        audit_reloaded = artifact.read_json_pair("audit.json", expected_sha256=audit_sha256)
        _require(attempt == _v3_attempt_payload(identity, chain)
                 and launch == _v3_launch_payload(identity, chain, attempt_sha256, launch["backend"]),
                 "CS-WG V3 published attempt/launch graph drift")
        _validate_v3_source_authority(authority_reloaded, identity, chain, prepared)
        _validate_v3_audit_result(
            audit_reloaded, identity, chain, authority_reloaded, authority_sha256, prepared,
        )
        terminal = _v3_terminal_payload(
            identity, chain, attempt_sha256=attempt_sha256, launch_sha256=launch_sha256,
            authority_sha256=authority_sha256, audit_sha256=audit_sha256, fallback=prepared.fallback,
        )
        terminal_sha256 = artifact.publish_json("terminal.json", terminal)
        artifact.validate_live(expected_names=(*expected_before_terminal, "terminal.json", "terminal.json.sha256"))
        _require(artifact.read_json_pair("terminal.json", expected_sha256=terminal_sha256) == terminal,
                 "CS-WG V3 published terminal graph drift")
        return SourceAuditV3LifecycleResult(
            artifact.root_identity, attempt_sha256, launch_sha256, authority_sha256,
            audit_sha256, terminal_sha256, None,
        )
    except BaseException as error:
        try:
            progress = backend.progress()
            _require(isinstance(progress, v1.LifecycleProgress),
                     "CS-WG V3 backend returned invalid failure progress")
        except BaseException:
            progress = v1.LifecycleProgress()
        failure_sha256 = artifact.publish_json(
            "failure.json",
            _v3_failure_payload(
                identity, chain, attempt_sha256=attempt_sha256, launch_sha256=launch_sha256,
                authority_sha256=authority_sha256, progress=progress, error=error,
            ),
        )
        return SourceAuditV3LifecycleResult(
            artifact.root_identity, attempt_sha256, launch_sha256, authority_sha256,
            audit_sha256, terminal_sha256, failure_sha256,
        )
    finally:
        try:
            backend.close()
        finally:
            artifact.close()


def execute_reviewed_source_audit_v3(
    root: Path, *, identity: SourceAuditV3Identity, capability: object, backend: DeferredSourceAuditV3Backend,
) -> SourceAuditV3LifecycleResult:
    """Root-only V3 source audit.  It has no GPU smoke or model route."""
    return _execute_reviewed_source_audit_v3(
        Path(root), identity=identity, capability=capability, backend=backend,
        predecessor_loader=validate_v1_v2_failed_predecessor_chain,
        route_bootstrapper=bootstrap_reviewed_v3_route,
    )


def dry_plan(root: Path | None = None) -> dict[str, object]:
    """Static public declaration; it reads no result root, source, parser, Torch, or CUDA."""
    result: dict[str, object] = {
        "cell": CELL,
        "phase": PHASE,
        "workorder_sha256": WORKORDER_SHA256,
        "fresh_v3_root_relative": V3_ROOT_RELATIVE,
        "inherited_v2_closure_sha256": V2_REPAIRED_CLOSURE_SHA256,
        "inherited_v2_identity_sha256": V2_IDENTITY_SHA256,
        "sealed_metadata_manifest_sha256": M1_METADATA_MANIFEST_SHA256,
        "immutable_v2_failed_predecessor": LIVE_V2_FAILED_EXPECTATION.payload(),
        "fallback": {
            "mode": COMMON_STRATUM_FALLBACK_MODE,
            "minimum_per_session_count": MINIMUM_ROWS_PER_ELIGIBLE_STRATUM,
            "minimum_eligible_strata": MINIMUM_ELIGIBLE_COMMON_STRATA,
            "pooled_source_quantile_fit_unchanged": True,
            "target_labels_used": False,
        },
        "reviewed_namespace": {
            "route_import": _V1_PHYSICAL_MODULE,
            "top_level_project_src_forbidden": True,
            "top_level_streaming_src_reserved_for_deferred_parser": True,
            "sys_modules_mutation": False,
        },
        "current_gpu_smoke_capability_issuable": False,
        "opens_source_or_target": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "launches": False,
        "execution_authorized": False,
        "capability_requirement": "opaque in-process root-reviewed V3 source-audit capability",
    }
    if root is not None:
        result["closure"] = implementation_closure(Path(root))
    return result


__all__ = (
    "COMMON_STRATUM_FALLBACK_MODE",
    "CommonStratumConstructibilityError",
    "CommonStratumFallbackEvidence",
    "CommonStratumPreparedAudit",
    "CommonStratumSourceProvider",
    "HeldFailurePredecessorChain",
    "HeldV2FailedAuditGraph",
    "LIVE_V2_FAILED_EXPECTATION",
    "PhysicalCommonStratumAuditBackend",
    "SourceAuditV3Capability",
    "SourceAuditV3Error",
    "SourceAuditV3Identity",
    "SourceAuditV3LifecycleResult",
    "SourceAuditV3Spec",
    "V2FailedGraphExpectation",
    "bootstrap_reviewed_v3_route",
    "build_common_stratum_prepared_audit",
    "build_reviewed_v3_source_audit_backend",
    "build_source_audit_v3_identity",
    "derive_common_stratum_fallback",
    "dry_plan",
    "execute_reviewed_source_audit_v3",
    "implementation_closure",
    "issue_root_reviewed_source_audit_v3_capability",
    "reach_deferred_parser_seam",
    "validate_v1_v2_failed_predecessor_chain",
)
