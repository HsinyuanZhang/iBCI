"""CS-WG M1 V5 granular-derivative source-smoke successor.

This is an additive successor for the immutable V4 post-run validation
failure.  Its public import is stdlib-only.  It binds the completed V3 audit
and failed V4 smoke graph by held no-follow descriptor reads, then delegates
the physical 100-step optimizer loop to the unchanged V1 runner through an
optional read-only derivative observer.

The V3/V4 closures below are historical receipt evidence.  They are never
reconstructed as current closures: the explicitly authorised V5 observer seam
changes current source bytes, so V5 builds and records its own successor
closure instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

from . import source_audit_v2 as v2
from . import source_lifecycle as v1


class SourceSmokeV5Error(RuntimeError):
    """Fail closed for V5 predecessor, derivative, or lifecycle drift."""


class V5ValidationError(SourceSmokeV5Error):
    """A receipt-safe, typed V5 validation failure."""

    def __init__(self, failed_predicate: str, message: str, observed_summary: Mapping[str, object] | None = None) -> None:
        super().__init__(message)
        self.failure_stage = "v5_smoke_validation"
        self.failed_predicate = failed_predicate
        self.observed_summary = {} if observed_summary is None else dict(observed_summary)


class DerivativeObservationError(SourceSmokeV5Error):
    """Raised by the successor observer before the next optimizer boundary."""

    def __init__(self, message: str, *, completed_steps: int) -> None:
        super().__init__(message)
        self.failure_stage = "derivative_observer"
        self.failed_predicate = "derivative_observer_exception"
        self.observed_summary = {"completed_optimizer_steps": completed_steps}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceSmokeV5Error(message)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64
             and all(character in "0123456789abcdef" for character in value),
             f"CS-WG V5 {label} must be a lowercase SHA-256")
    return value


def _safe_relative(value: object) -> str:
    _require(isinstance(value, str) and value, "CS-WG V5 relative path is absent")
    path = Path(value)
    _require(not path.is_absolute() and ".." not in path.parts and path.name not in {"", ".", ".."},
             "CS-WG V5 relative path is unsafe")
    return path.as_posix()


CELL = "CROSS_SESSION_WORST_GROUP_SPINT_M1_V1"
PHASE = "m1_source_smoke_v5_granular_derivative_successor"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CS_WG_M1_SOURCE_SMOKE_V5_GRANULAR_DERIVATIVE_SUCCESSOR_20260826.md"
WORKORDER_SHA256 = "65dec719e8931270ae1d9b9bdc11c550a8f68198b6457e327ac90431ae1a85d4"
V5_ROOT_RELATIVE = "tfpd_exploration/results/cross_session_worst_group_m1_source_smoke_v5"

V3_ROOT_RELATIVE = "tfpd_exploration/results/cross_session_worst_group_m1_source_audit_v3"
V3_ATTEMPT_SHA256 = "f63aec4a528495d937a514a6b2194f0240bfb97f48e95ad82b44a28cbb2e4287"
V3_LAUNCH_SHA256 = "04243c940627c4d9e745a3834f9cf58b3cce2bfeffba9ac0684b570e723f4c9c"
V3_SOURCE_AUTHORITY_SHA256 = "0f5c9e47113ca579c56d46282f2f4cc60e27bcbc452fd0b3cab89df352bacd0b"
V3_AUDIT_SHA256 = "ce64d34a2a4ddbf4f6825a7dfbec81eb05a8c4a5f93e1dd577ee597e7b2f16d9"
V3_TERMINAL_SHA256 = "2a27b02db39a4826f37b93dbcbe9b8c227fefa3b3c8c154136960bdf5f6e9230"
V3_CLOSURE_SHA256 = "d52168e567188b8ede816f4764cf829ecd920b2540c323fee14569ec7503fa1e"
V3_IDENTITY_SHA256 = "90892d8022be400b1b9e9046f003e86f81e9c48ae9ee892c4a93b99547c489ff"
V3_FALLBACK_SHA256 = "2e763763bf55276bfef175815ec7b6cb454393a87278513f86f8d99677618089"

V4_ROOT_RELATIVE = "tfpd_exploration/results/cross_session_worst_group_m1_source_smoke_v4"
V4_ATTEMPT_SHA256 = "0d4179d77cd8aa66bdad70d1164682dba38b3d960fab3d524cf5f213e4601b27"
V4_LAUNCH_SHA256 = "422dec3d014bf3fa0fed716b8634cc6e8da6076f36f0f303f657a388f7b417ad"
V4_SOURCE_AUTHORITY_SHA256 = "40336ed6ea1f6dd2b6511eee4d22f5cf76c4af29510f8c258819bd396c78b4fe"
V4_FAILURE_SHA256 = "1a8237db7ddf34535bebd83cc773da7efde6b5aa97517a09638b391d6eae91d3"
V4_FAILURE_ERROR_SHA256 = "bfeff083be483479e40fb82078a3b8b1e5755b0f04ec0f341cc840af892cf813"
V4_HISTORICAL_CLOSURE_SHA256 = "46cf82d19bb80f79cbb4df3a8ea34a4778af4cf8bc5f1f8674e4fe2ba729a676"

# The numerical envelope is pre-registered in the workorder and must never
# be inferred from a live result.
DERIVATIVE_DOMAIN_MIN = 0.0
DERIVATIVE_DOMAIN_MAX = 10.0
DERIVATIVE_TAU = 0.01
DERIVATIVE_SUM_ABS_TOLERANCE = 2.0e-6
DERIVATIVE_REFERENCE_ABS_TOLERANCE = 2.0e-5
DERIVATIVE_MIN_TOLERANCE = -2.0e-5
DERIVATIVE_DIGEST_DOMAIN = b"CSWG-V5-derivative-observation-v1\x00"

_RUNTIME_DEPENDENCY_PATHS = (
    "tfpd_exploration/docs/WORKORDER_CS_WG_M1_SOURCE_SMOKE_V4_COMMON_STRATUM_SUCCESSOR_20260826.md",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_audit_v2.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_audit_v3.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_smoke_v4.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_smoke_physical_v4.py",
)
_OWNED_PATHS = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/cross_session_worst_group_v1/source_smoke_v5.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_smoke_physical_v5.py",
    "tfpd_exploration/scripts/run_cross_session_worst_group_m1_source_smoke_v5.py",
    "tfpd_exploration/tests/test_cross_session_worst_group_m1_source_smoke_v5.py",
)


def _read_regular_no_follow(root: Path, relative: str) -> str:
    path = Path(root).absolute() / _safe_relative(relative)
    try:
        info = os.lstat(path)
    except OSError as error:
        raise SourceSmokeV5Error(f"CS-WG V5 closure leaf inaccessible: {relative}") from error
    _require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode),
             f"CS-WG V5 closure leaf is not a regular non-symlink: {relative}")
    with open(path, "rb") as handle:
        return _sha(handle.read())


def implementation_closure(root: Path) -> dict[str, object]:
    """Build V5's own explicit closure without asserting old closure freshness."""
    try:
        inherited = v1.implementation_closure(Path(root))
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV5Error("CS-WG V5 current V1 source closure drift") from error
    inherited_rows = inherited.get("paths")
    _require(isinstance(inherited_rows, list) and inherited_rows,
             "CS-WG V5 inherited closure topology drift")
    rows = [dict(item) for item in inherited_rows]
    existing = {item.get("path") for item in rows}
    for relative in (*_RUNTIME_DEPENDENCY_PATHS, *_OWNED_PATHS):
        _require(relative not in existing, "CS-WG V5 closure path duplication")
        rows.append({"path": relative, "sha256": _read_regular_no_follow(Path(root), relative)})
        existing.add(relative)
    workorder_row = next(item for item in rows if item["path"] == WORKORDER_RELATIVE)
    _require(workorder_row["sha256"] == WORKORDER_SHA256,
             "CS-WG V5 workorder literal/body drift")
    body = {
        "schema": "cross_session_worst_group_m1_source_smoke_v5_closure_v1",
        "current_v1_source_closure_sha256": inherited.get("closure_sha256"),
        "historical_v3_closure_sha256": V3_CLOSURE_SHA256,
        "historical_v4_closure_sha256": V4_HISTORICAL_CLOSURE_SHA256,
        "paths": rows,
    }
    return {**body, "closure_sha256": _sha(_json_bytes(body))}


def validate_current_closure(root: Path, value: Mapping[str, object]) -> None:
    _require(isinstance(value, Mapping) and dict(value) == implementation_closure(Path(root)),
             "CS-WG V5 successor closure/current-byte drift")


def _flags_exact(value: Mapping[str, object]) -> bool:
    return (value.get("source_only") is True
            and value.get("target_optimizer_backward_update") == 0
            and all(value.get(name) is expected for name, expected in v1.FORBIDDEN_SURFACE_FLAGS.items()))


@dataclass(frozen=True)
class HistoricalGraphExpectation:
    root_relative: str
    pairs_value: tuple[tuple[str, str], ...]
    exact_leaf_count: int
    label: str

    def __post_init__(self) -> None:
        _require(_safe_relative(self.root_relative) == self.root_relative
                 and self.exact_leaf_count == len(self.pairs_value) * 2
                 and all(isinstance(name, str) and name.endswith(".json") and _require_sha(digest, self.label)
                         for name, digest in self.pairs_value),
                 "CS-WG V5 historical graph expectation drift")

    def pairs(self) -> tuple[tuple[str, str], ...]:
        return self.pairs_value

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_historical_graph_expectation_v5",
            "label": self.label,
            "root_relative": self.root_relative,
            "pairs": [{"name": name, "sha256": digest} for name, digest in self.pairs_value],
            "exact_leaf_count": self.exact_leaf_count,
            "leaf_mode": "0444",
            "canonical_basename_sidecars": True,
        }


V3_COMPLETED_EXPECTATION = HistoricalGraphExpectation(
    V3_ROOT_RELATIVE,
    (("attempt.json", V3_ATTEMPT_SHA256), ("launch.json", V3_LAUNCH_SHA256),
     ("source_authority.json", V3_SOURCE_AUTHORITY_SHA256), ("audit.json", V3_AUDIT_SHA256),
     ("terminal.json", V3_TERMINAL_SHA256)),
    10, "accepted V3 graph",
)
V4_FAILED_EXPECTATION = HistoricalGraphExpectation(
    V4_ROOT_RELATIVE,
    (("attempt.json", V4_ATTEMPT_SHA256), ("launch.json", V4_LAUNCH_SHA256),
     ("source_authority.json", V4_SOURCE_AUTHORITY_SHA256), ("failure.json", V4_FAILURE_SHA256)),
    8, "failed V4 graph",
)


@dataclass(frozen=True)
class HeldHistoricalGraph:
    expectation: HistoricalGraphExpectation
    root_identity: tuple[int, int]
    named_chain_identities: tuple[tuple[str, int, int], ...]
    bodies: Mapping[str, Mapping[str, object]] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        values = {str(name): MappingProxyType(dict(value)) for name, value in dict(self.bodies).items()}
        _require(tuple(sorted(values)) == tuple(sorted(name for name, _ in self.expectation.pairs()))
                 and len(self.root_identity) == 2 and all(type(item) is int and item >= 0 for item in self.root_identity)
                 and self.named_chain_identities,
                 "CS-WG V5 held historical graph topology drift")
        object.__setattr__(self, "bodies", MappingProxyType(values))

    def body(self, name: str) -> Mapping[str, object]:
        value = self.bodies.get(name)
        _require(isinstance(value, Mapping), "CS-WG V5 held historical body absent")
        return value

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_held_historical_graph_binding_v5",
            "expectation": self.expectation.payload(),
            "root_identity": list(self.root_identity),
            "named_chain_identities": [list(item) for item in self.named_chain_identities],
            "body_sha256": {name: digest for name, digest in self.expectation.pairs()},
        }

    @property
    def sha256(self) -> str:
        return _sha(_json_bytes(self.payload()))


def _identity_payload(value: Mapping[str, object], *, label: str) -> Mapping[str, object]:
    identity = value.get("identity")
    _require(isinstance(identity, Mapping), f"CS-WG V5 {label} identity absent")
    return identity


def _identity_hash(value: Mapping[str, object], *, label: str) -> str:
    return _sha(_json_bytes(dict(_identity_payload(value, label=label))))


def _validate_v3_identity_propagation(
    *, identity: Mapping[str, object], audit: Mapping[str, object],
    nested_identity_receipts: Mapping[str, Mapping[str, object]], expected_identity_sha256: str,
) -> None:
    """Validate the real V3 split identity schema without inventing fields.

    V3's attempt, launch, source-authority, and terminal receipts carry the
    complete nested identity object.  Its audit result deliberately carries
    only ``identity_sha256``.  The split is part of the immutable producer
    schema: accepting an audit-side nested identity would be a different
    receipt topology, while accepting a missing/substituted nested identity in
    any of the other three bodies would sever the graph.
    """
    _require(_require_sha(expected_identity_sha256, "V3 expected identity")
             and _sha(_json_bytes(dict(identity))) == expected_identity_sha256,
             "CS-WG V5 V3 historical identity digest drift")
    _require(audit.get("identity_sha256") == expected_identity_sha256 and "identity" not in audit,
             "CS-WG V5 V3 audit identity-shape/digest drift")
    for label, receipt in nested_identity_receipts.items():
        try:
            observed = _identity_payload(receipt, label=f"V3 {label}")
        except SourceSmokeV5Error as error:
            raise SourceSmokeV5Error("CS-WG V5 V3 nested identity propagation drift") from error
        _require(observed == identity, "CS-WG V5 V3 nested identity propagation drift")


def _validate_historical_v3_graph(graph: HeldHistoricalGraph) -> None:
    _require(graph.expectation == V3_COMPLETED_EXPECTATION, "CS-WG V5 V3 graph expectation drift")
    attempt = graph.body("attempt.json")
    launch = graph.body("launch.json")
    authority = graph.body("source_authority.json")
    audit = graph.body("audit.json")
    terminal = graph.body("terminal.json")
    identity = _identity_payload(attempt, label="V3 attempt")
    _require(isinstance(identity.get("closure"), Mapping)
             and identity["closure"].get("closure_sha256") == V3_CLOSURE_SHA256,
             "CS-WG V5 V3 historical identity/closure drift")
    _require(attempt.get("schema") == "cross_session_worst_group_m1_source_audit_attempt_v3"
             and attempt.get("status") == "ATTEMPT_RESERVED" and attempt.get("cell") == CELL
             and launch.get("schema") == "cross_session_worst_group_m1_source_audit_launch_v3"
             and launch.get("status") == "LAUNCHED" and launch.get("attempt_sha256") == V3_ATTEMPT_SHA256
             and authority.get("schema") == "cross_session_worst_group_m1_source_audit_authority_v3"
             and audit.get("schema") == "cross_session_worst_group_m1_source_audit_result_v3"
             and audit.get("identity_sha256") == V3_IDENTITY_SHA256
             and audit.get("source_authority_sha256") == V3_SOURCE_AUTHORITY_SHA256
             and terminal.get("schema") == "cross_session_worst_group_m1_source_audit_terminal_v3"
             and terminal.get("status") == "PASS_SOURCE_AUDIT_COMMON_STRATUM_CONSTRUCTIBLE"
             and terminal.get("attempt_sha256") == V3_ATTEMPT_SHA256
             and terminal.get("launch_sha256") == V3_LAUNCH_SHA256
             and terminal.get("source_authority_sha256") == V3_SOURCE_AUTHORITY_SHA256
             and terminal.get("audit_sha256") == V3_AUDIT_SHA256
             and terminal.get("launch_closure_sha256") == V3_CLOSURE_SHA256
             and terminal.get("final_closure_sha256") == V3_CLOSURE_SHA256,
             "CS-WG V5 V3 completed graph semantics drift")
    _validate_v3_identity_propagation(
        identity=identity,
        audit=audit,
        nested_identity_receipts={"launch": launch, "authority": authority, "terminal": terminal},
        expected_identity_sha256=V3_IDENTITY_SHA256,
    )
    fallback = authority.get("deterministic_common_stratum_fallback")
    _require(isinstance(fallback, Mapping)
             and authority.get("deterministic_common_stratum_fallback_sha256") == V3_FALLBACK_SHA256
             and _sha(_json_bytes(dict(fallback))) == V3_FALLBACK_SHA256
             and audit.get("deterministic_common_stratum_fallback_sha256") == V3_FALLBACK_SHA256
             and terminal.get("deterministic_common_stratum_fallback_sha256") == V3_FALLBACK_SHA256,
             "CS-WG V5 V3 fallback binding drift")
    for value, label in ((launch, "launch"), (authority, "authority"), (terminal, "terminal")):
        _require(_flags_exact(value), "CS-WG V5 V3 nested identity/source boundary drift")
    _require(_flags_exact(audit), "CS-WG V5 V3 audit source boundary drift")
    _require(_flags_exact(attempt) and attempt.get("source_resolved_or_opened") is False
             and attempt.get("model_constructed") is False and attempt.get("cuda_initialized") is False,
             "CS-WG V5 V3 attempt boundary drift")


def _validate_historical_v4_graph(graph: HeldHistoricalGraph) -> None:
    _require(graph.expectation == V4_FAILED_EXPECTATION, "CS-WG V5 V4 graph expectation drift")
    attempt = graph.body("attempt.json")
    launch = graph.body("launch.json")
    authority = graph.body("source_authority.json")
    failure = graph.body("failure.json")
    identity = _identity_payload(attempt, label="V4 attempt")
    closure = identity.get("closure")
    _require(isinstance(closure, Mapping) and closure.get("closure_sha256") == V4_HISTORICAL_CLOSURE_SHA256
             and identity.get("accepted_v3_closure_sha256") == V3_CLOSURE_SHA256
             and identity.get("accepted_v3_source_audit_identity_sha256") == V3_IDENTITY_SHA256,
             "CS-WG V5 V4 historical identity/closure drift")
    _require(attempt.get("schema") == "cross_session_worst_group_m1_source_smoke_attempt_v4"
             and attempt.get("status") == "ATTEMPT_RESERVED" and attempt.get("cell") == CELL
             and launch.get("schema") == "cross_session_worst_group_m1_source_smoke_launch_v4"
             and launch.get("status") == "LAUNCHED" and launch.get("attempt_sha256") == V4_ATTEMPT_SHA256
             and authority.get("schema") == "cross_session_worst_group_m1_source_smoke_authority_v4"
             and failure.get("schema") == "cross_session_worst_group_m1_source_smoke_failure_v4"
             and failure.get("status") == "FAILED"
             and failure.get("attempt_sha256") == V4_ATTEMPT_SHA256
             and failure.get("launch_sha256") == V4_LAUNCH_SHA256
             and failure.get("source_authority_sha256") == V4_SOURCE_AUTHORITY_SHA256
             and failure.get("error_class") == "SourceSmokeV4Error"
             and failure.get("error_sha256") == V4_FAILURE_ERROR_SHA256
             and failure.get("terminal_published") is False,
             "CS-WG V5 V4 failed graph semantics drift")
    progress = failure.get("progress")
    _require(isinstance(progress, Mapping) and progress.get("optimizer_steps_completed") == v1.SMOKE_STEPS
             and progress.get("source_resolved_or_opened") is True
             and progress.get("source_authority_published") is True,
             "CS-WG V5 V4 failed-run progress drift")
    for value, label in ((launch, "launch"), (authority, "authority"), (failure, "failure")):
        _require(_identity_payload(value, label=f"V4 {label}") == identity and _flags_exact(value),
                 "CS-WG V5 V4 identity/source boundary drift")
    _require(_flags_exact(attempt) and attempt.get("source_resolved_or_opened") is False
             and attempt.get("model_constructed") is False and attempt.get("cuda_initialized") is False,
             "CS-WG V5 V4 attempt boundary drift")


def _read_held_historical_graph(
    root: Path, expectation: HistoricalGraphExpectation, *, semantic_validator: Any | None = None,
) -> HeldHistoricalGraph:
    """Read exact immutable pairs through a single held no-follow root FD."""
    try:
        base_fd, opened, named = v2._open_held_result_directory(Path(root), expectation.root_relative)
    except v2.SourceAuditV2Error as error:
        raise SourceSmokeV5Error("CS-WG V5 historical predecessor directory validation failed") from error
    result_fd = opened[-1]
    try:
        expected_names = tuple(sorted(name for body, _ in expectation.pairs() for name in (body, f"{body}.sha256")))
        _require(tuple(sorted(os.listdir(result_fd))) == expected_names,
                 "CS-WG V5 historical predecessor topology/extra/failure drift")
        bodies: dict[str, dict[str, object]] = {}
        for name, expected_sha in expectation.pairs():
            try:
                body = v2._read_held_leaf(result_fd, name)
                sidecar = v2._read_held_leaf(result_fd, f"{name}.sha256")
            except v2.SourceAuditV2Error as error:
                raise SourceSmokeV5Error("CS-WG V5 historical predecessor leaf mode/type drift") from error
            _require(_sha(body) == expected_sha and sidecar == f"{expected_sha}  {name}\n".encode("ascii"),
                     "CS-WG V5 historical predecessor body/sidecar digest drift")
            try:
                payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise SourceSmokeV5Error("CS-WG V5 historical predecessor JSON decode drift") from error
            _require(isinstance(payload, dict), "CS-WG V5 historical predecessor JSON object drift")
            bodies[name] = payload
        _require(tuple(sorted(os.listdir(result_fd))) == expected_names,
                 "CS-WG V5 historical predecessor topology changed during held read")
        held = os.fstat(result_fd)
        graph = HeldHistoricalGraph(
            expectation=expectation,
            root_identity=(int(held.st_dev), int(held.st_ino)),
            named_chain_identities=named,
            bodies=bodies,
        )
        if semantic_validator is not None:
            _require(callable(semantic_validator), "CS-WG V5 historical semantic validator is absent")
            semantic_validator(graph)
        elif expectation == V3_COMPLETED_EXPECTATION:
            _validate_historical_v3_graph(graph)
        elif expectation == V4_FAILED_EXPECTATION:
            _validate_historical_v4_graph(graph)
        else:
            raise SourceSmokeV5Error("CS-WG V5 historical predecessor expectation is unrecognised")
        try:
            named_after = v2._named_chain_identities(Path(root), expectation.root_relative)
        except v2.SourceAuditV2Error as error:
            raise SourceSmokeV5Error("CS-WG V5 historical predecessor named revalidation failed") from error
        _require(named_after == named, "CS-WG V5 historical predecessor named identity drift")
        return graph
    finally:
        v2._close_chain(base_fd, opened)


def validate_v5_predecessors(root: Path) -> tuple[HeldHistoricalGraph, HeldHistoricalGraph]:
    """Load V3 success and V4 failure in fixed order; no source/data is read."""
    return (
        _read_held_historical_graph(Path(root), V3_COMPLETED_EXPECTATION),
        _read_held_historical_graph(Path(root), V4_FAILED_EXPECTATION),
    )


@dataclass(frozen=True)
class SourceSmokeV5Spec:
    root_relative: str = V5_ROOT_RELATIVE

    def __post_init__(self) -> None:
        _require(_safe_relative(self.root_relative) == V5_ROOT_RELATIVE,
                 "CS-WG V5 smoke root literal drift")

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_source_smoke_spec_v5",
            "root_relative": self.root_relative,
            "inherited_v1_smoke_spec": v1.source_smoke_spec().payload(),
            "optimizer_steps": v1.SMOKE_STEPS,
            "derivative_numeric_gate": derivative_gate_contract_payload(),
            "source_only": True,
            "target_optimizer_backward_update": 0,
            **v1.FORBIDDEN_SURFACE_FLAGS,
        }

    @property
    def sha256(self) -> str:
        return _sha(_json_bytes(self.payload()))


def derivative_gate_contract_payload() -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_derivative_numeric_gate_v5",
        "loss_domain": [DERIVATIVE_DOMAIN_MIN, DERIVATIVE_DOMAIN_MAX],
        "tau": DERIVATIVE_TAU,
        "reference": "float64_stable_softmax(session_mse/tau)",
        "raw_sum_abs_error_max": DERIVATIVE_SUM_ABS_TOLERANCE,
        "raw_reference_abs_error_max": DERIVATIVE_REFERENCE_ABS_TOLERANCE,
        "raw_minimum": DERIVATIVE_MIN_TOLERANCE,
        "raw_fp32_autograd_observed": True,
        "live_result_tuning_forbidden": True,
    }


@dataclass(frozen=True)
class SourceSmokeV5Identity:
    spec: SourceSmokeV5Spec
    inherited_v1_smoke_identity: v1.SourceExecutionIdentity
    closure: Mapping[str, object]

    def __post_init__(self) -> None:
        _require(isinstance(self.spec, SourceSmokeV5Spec)
                 and isinstance(self.inherited_v1_smoke_identity, v1.SourceExecutionIdentity)
                 and self.inherited_v1_smoke_identity.spec == v1.source_smoke_spec()
                 and isinstance(self.closure, Mapping) and _require_sha(self.closure.get("closure_sha256"), "closure"),
                 "CS-WG V5 identity topology drift")
        object.__setattr__(self, "closure", MappingProxyType(dict(self.closure)))

    @property
    def device(self) -> v1.DeviceProfile:
        return self.inherited_v1_smoke_identity.device

    def payload(self) -> dict[str, object]:
        return {
            "schema": "cross_session_worst_group_m1_source_smoke_identity_v5",
            "cell": CELL,
            "phase": PHASE,
            "spec": self.spec.payload(),
            "inherited_v1_smoke_identity": self.inherited_v1_smoke_identity.payload(),
            "accepted_v3_completed_graph": V3_COMPLETED_EXPECTATION.payload(),
            "accepted_v3_closure_sha256_historical": V3_CLOSURE_SHA256,
            "accepted_v3_identity_sha256_historical": V3_IDENTITY_SHA256,
            "failed_v4_graph": V4_FAILED_EXPECTATION.payload(),
            "failed_v4_closure_sha256_historical": V4_HISTORICAL_CLOSURE_SHA256,
            "closure": dict(self.closure),
            "source_only": True,
            "no_amp_tf32_compile": True,
            "target_optimizer_backward_update": 0,
            **v1.FORBIDDEN_SURFACE_FLAGS,
        }

    @property
    def sha256(self) -> str:
        return _sha(_json_bytes(self.payload()))


def build_source_smoke_v5_identity(root: Path, *, device: v1.DeviceProfile) -> SourceSmokeV5Identity:
    try:
        inherited = v1.build_identity(Path(root), spec=v1.source_smoke_spec(), device=device)
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV5Error("CS-WG V5 inherited V1 identity drift") from error
    return SourceSmokeV5Identity(SourceSmokeV5Spec(), inherited, implementation_closure(Path(root)))


def validate_source_smoke_v5_identity_current(root: Path, identity: SourceSmokeV5Identity) -> None:
    _require(isinstance(identity, SourceSmokeV5Identity), "CS-WG V5 identity must be typed")
    try:
        v1.validate_identity_current(Path(root), identity.inherited_v1_smoke_identity)
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV5Error("CS-WG V5 inherited V1 identity current-byte drift") from error
    validate_current_closure(Path(root), identity.closure)


class _V5RootReviewSeal:
    pass


_V5_ROOT_REVIEW_SEAL = _V5RootReviewSeal()


@dataclass(frozen=True)
class SourceSmokeV5Capability:
    identity_sha256: str
    v3_completed_graph_sha256: str
    v4_failed_graph_sha256: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _require(all(_require_sha(value, "capability graph") for value in (
            self.identity_sha256, self.v3_completed_graph_sha256, self.v4_failed_graph_sha256,
        )) and self._seal is _V5_ROOT_REVIEW_SEAL,
                 "CS-WG V5 requires an in-process root-reviewed capability")


def _assert_v5_root_fresh(root: Path, spec: SourceSmokeV5Spec) -> None:
    candidate = Path(root).absolute() / _safe_relative(spec.root_relative)
    try:
        os.lstat(candidate)
    except FileNotFoundError:
        return
    except OSError as error:
        raise SourceSmokeV5Error("CS-WG V5 prospective smoke root cannot be safely inspected") from error
    raise SourceSmokeV5Error("CS-WG V5 canonical smoke root already exists")


def _issue_root_reviewed_source_smoke_v5_capability(
    root: Path, identity: SourceSmokeV5Identity, *, environ: Mapping[str, str] | None,
    review_seal: object, predecessor_loader: Any,
) -> SourceSmokeV5Capability:
    _require(review_seal is _V5_ROOT_REVIEW_SEAL,
             "only the root reviewer may issue CS-WG V5 smoke capability")
    validate_source_smoke_v5_identity_current(Path(root), identity)
    predecessors = predecessor_loader(Path(root))
    _require(isinstance(predecessors, tuple) and len(predecessors) == 2
             and all(isinstance(item, HeldHistoricalGraph) for item in predecessors),
             "CS-WG V5 predecessor loader type drift")
    v3_graph, v4_graph = predecessors
    _require(v3_graph.expectation == V3_COMPLETED_EXPECTATION and v4_graph.expectation == V4_FAILED_EXPECTATION,
             "CS-WG V5 predecessor expectation drift")
    try:
        v1.validate_device_environment(identity.device, environ)
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV5Error("CS-WG V5 selected device/thread environment drift") from error
    _assert_v5_root_fresh(Path(root), identity.spec)
    return SourceSmokeV5Capability(identity.sha256, v3_graph.sha256, v4_graph.sha256, _V5_ROOT_REVIEW_SEAL)


def issue_root_reviewed_source_smoke_v5_capability(
    root: Path, identity: SourceSmokeV5Identity, *, environ: Mapping[str, str] | None, review_seal: object,
) -> SourceSmokeV5Capability:
    return _issue_root_reviewed_source_smoke_v5_capability(
        Path(root), identity, environ=environ, review_seal=review_seal, predecessor_loader=validate_v5_predecessors,
    )


def _require_v5_capability(capability: object, identity: SourceSmokeV5Identity) -> SourceSmokeV5Capability:
    _require(isinstance(capability, SourceSmokeV5Capability) and capability._seal is _V5_ROOT_REVIEW_SEAL
             and capability.identity_sha256 == identity.sha256,
             "CS-WG V5 smoke requires exact root-reviewed capability")
    return capability


@dataclass
class DerivativeEvidenceCollector:
    """Compact V5 observer state; it never retains a Torch/GPU tensor."""

    expected_sessions: tuple[str, str, str]
    _rows: list[dict[str, object]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        _require(tuple(sorted(self.expected_sessions)) == self.expected_sessions and len(set(self.expected_sessions)) == 3,
                 "CS-WG V5 derivative collector session topology drift")

    def observe(self, payload: Mapping[str, object]) -> None:
        try:
            item = dict(payload)
            step = item.get("step_index")
            sessions = item.get("session_ids")
            losses = item.get("session_loss_values_fp32")
            raw = item.get("raw_autograd_weight_values_fp32")
            if not (item.get("schema") == "cross_session_worst_group_m1_derivative_observation_v1"
                    and type(step) is int and step == len(self._rows)
                    and tuple(sessions) == self.expected_sessions
                    and isinstance(losses, tuple) and isinstance(raw, tuple)
                    and len(losses) == len(raw) == 3
                    and item.get("tensor_count") == 0 and item.get("graph_retained") is False
                    and all(type(value) is float for value in (*losses, *raw))):
                raise ValueError("observation schema/scalars")
            self._rows.append({"step_index": step, "losses": list(losses), "raw": list(raw)})
        except BaseException as error:
            if isinstance(error, DerivativeObservationError):
                raise
            raise DerivativeObservationError(
                "CS-WG V5 derivative observer payload drift", completed_steps=len(self._rows),
            ) from error

    @staticmethod
    def _stable_reference(losses: Sequence[float]) -> tuple[float, float, float]:
        scaled = [float(value) / DERIVATIVE_TAU for value in losses]
        maximum = max(scaled)
        exponentials = [math.exp(value - maximum) for value in scaled]
        denominator = math.fsum(exponentials)
        return tuple(value / denominator for value in exponentials)  # type: ignore[return-value]

    def validate(self, *, expected_steps: int) -> dict[str, object]:
        _require(type(expected_steps) is int and expected_steps == v1.SMOKE_STEPS,
                 "CS-WG V5 derivative expected-step literal drift")
        rows = [dict(row) for row in self._rows]
        if len(rows) != expected_steps:
            raise V5ValidationError("derivative_numeric_gate", "CS-WG V5 derivative observation count drift", {
                "observed_steps": len(rows), "expected_steps": expected_steps,
            })
        raw_min = math.inf
        raw_max = -math.inf
        max_sum_error = 0.0
        max_reference_error = 0.0
        for index, row in enumerate(rows):
            losses = row["losses"]
            raw = row["raw"]
            if row.get("step_index") != index or not isinstance(losses, list) or not isinstance(raw, list):
                raise V5ValidationError("derivative_numeric_gate", "CS-WG V5 derivative row order/schema drift", {
                    "row_index": index,
                })
            if (len(losses) != 3 or len(raw) != 3 or not all(type(value) is float for value in (*losses, *raw))):
                raise V5ValidationError("derivative_numeric_gate", "CS-WG V5 derivative row scalar topology drift", {
                    "row_index": index,
                })
            if not all(math.isfinite(value) and DERIVATIVE_DOMAIN_MIN <= value <= DERIVATIVE_DOMAIN_MAX for value in losses):
                raise V5ValidationError("derivative_numeric_gate", "CS-WG V5 session loss domain/nonfinite drift", {
                    "row_index": index, "losses": list(losses),
                })
            if not all(math.isfinite(value) for value in raw):
                raise V5ValidationError("derivative_numeric_gate", "CS-WG V5 raw autograd derivative nonfinite", {
                    "row_index": index,
                })
            reference = self._stable_reference(losses)
            sum_error = abs(math.fsum(raw) - 1.0)
            reference_error = max(abs(item - expected) for item, expected in zip(raw, reference, strict=True))
            raw_min = min(raw_min, *raw)
            raw_max = max(raw_max, *raw)
            max_sum_error = max(max_sum_error, sum_error)
            max_reference_error = max(max_reference_error, reference_error)
            if (sum_error > DERIVATIVE_SUM_ABS_TOLERANCE
                    or reference_error > DERIVATIVE_REFERENCE_ABS_TOLERANCE
                    or min(raw) < DERIVATIVE_MIN_TOLERANCE):
                raise V5ValidationError("derivative_numeric_gate", "CS-WG V5 raw/reference derivative gate drift", {
                    "row_index": index,
                    "raw_min": min(raw),
                    "raw_sum_abs_error": sum_error,
                    "raw_reference_max_abs_error": reference_error,
                    "losses": list(losses),
                })
        canonical_rows = _json_bytes(rows)
        digest = _sha(DERIVATIVE_DIGEST_DOMAIN + canonical_rows)
        return {
            "schema": "cross_session_worst_group_m1_derivative_numeric_evidence_v5",
            "session_ids": list(self.expected_sessions),
            "steps_observed": len(rows),
            "gate": derivative_gate_contract_payload(),
            "raw_observation_domain_separated_sha256": digest,
            "raw_global_min": raw_min,
            "raw_global_max": raw_max,
            "max_sum_abs_error": max_sum_error,
            "max_reference_abs_error": max_reference_error,
            "all_rows_pass": True,
            "retained_gpu_tensors": 0,
            "second_forward_or_backward": False,
        }


def _validate_derivative_evidence(value: object, *, expected_sessions: Sequence[str]) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG V5 derivative evidence must be a mapping")
    item = dict(value)
    gate = derivative_gate_contract_payload()
    floats = ("raw_global_min", "raw_global_max", "max_sum_abs_error", "max_reference_abs_error")
    _require(item.get("schema") == "cross_session_worst_group_m1_derivative_numeric_evidence_v5"
             and item.get("session_ids") == list(expected_sessions)
             and item.get("steps_observed") == v1.SMOKE_STEPS
             and item.get("gate") == gate
             and _require_sha(item.get("raw_observation_domain_separated_sha256"), "derivative observation")
             and all(type(item.get(name)) in {int, float} and math.isfinite(float(item[name])) for name in floats)
             and item.get("all_rows_pass") is True and item.get("retained_gpu_tensors") == 0
             and item.get("second_forward_or_backward") is False
             and float(item["raw_global_min"]) >= DERIVATIVE_MIN_TOLERANCE
             and float(item["max_sum_abs_error"]) <= DERIVATIVE_SUM_ABS_TOLERANCE
             and float(item["max_reference_abs_error"]) <= DERIVATIVE_REFERENCE_ABS_TOLERANCE,
             "CS-WG V5 derivative evidence schema/gate drift")
    return item


def _validate_main_schema_scalars(inner: Mapping[str, object], inherited: v1.SourceExecutionIdentity) -> None:
    item = dict(inner)
    expected_shape = [v1.plan.TOTAL_BATCH_SIZE, v1.plan.M1_WINDOW_SIZE, v1.plan.M1_RAW_BEHAVIOR_OUTPUTS]
    _require(item.get("schema") == "cross_session_worst_group_m1_source_smoke_v1"
             and item.get("identity_sha256") == inherited.sha256
             and item.get("optimizer_steps") == v1.SMOKE_STEPS
             and item.get("total_windows_per_step") == v1.plan.TOTAL_BATCH_SIZE
             and item.get("one_concatenated_forward_per_step") is True
             and item.get("calibration_shape_per_row") == list(v1.plan.M1_CALIBRATION_SHAPE_PER_ROW)
             and item.get("model_parameter_count") == v1.plan.M1_LIVE_PARAMETERS_AFTER_LAZY1024
             and item.get("model_output_shape") == expected_shape
             and all(item.get(name) is True for name in (
                 "finite_objective", "finite_model", "finite_gradients", "finite_adam_state", "model_state_changed",
                 "checkpoint_reload_strict", "best_checkpoint_reload_strict", "last_checkpoint_reload_strict",
                 "dynamic_dropout_preserved",
             ))
             and isinstance(item.get("session_objective_derivatives_nonnegative"), bool)
             and _require_sha(item.get("initial_model_state_sha256"), "initial model state")
             and _require_sha(item.get("final_model_state_sha256"), "final model state")
             and item.get("initial_model_state_sha256") != item.get("final_model_state_sha256")
             and _require_sha(item.get("best_checkpoint_state_sha256"), "best checkpoint state")
             and _flags_exact(item),
             "CS-WG V5 inherited V1 main schema/scalars drift")


def _validate_gradient_coverage(inner: Mapping[str, object]) -> None:
    coverage = inner.get("gradient_coverage")
    _require(isinstance(coverage, Mapping)
             and type(coverage.get("trainable_parameter_count")) is int and coverage["trainable_parameter_count"] > 0
             and _require_sha(coverage.get("trainable_parameter_names_sha256"), "trainable parameter names")
             and coverage.get("observed_gradient_count") == coverage.get("trainable_parameter_count")
             and coverage.get("observed_gradient_names_sha256") == coverage.get("trainable_parameter_names_sha256")
             and coverage.get("missing_trainable_names") == [] and coverage.get("excluded_trainable_names") == [],
             "CS-WG V5 inherited V1 gradient coverage drift")


def _validate_rng(inner: Mapping[str, object]) -> None:
    rng = inner.get("rng")
    _require(isinstance(rng, Mapping)
             and rng.get("schema") == "cross_session_worst_group_m1_rng_policy_v1"
             and rng.get("seed") == v1.SEED
             and rng.get("domains") == ["python", "numpy", "torch_cpu", "torch_cuda_selected"]
             and rng.get("scheduler_uses_host_rng") is False
             and rng.get("model_initialization_and_dynamic_dropout_seeded") is True
             and _require_sha(rng.get("pre_run_state_digest"), "RNG pre-run state")
             and rng.get("post_run_state_restored") is True,
             "CS-WG V5 inherited V1 RNG receipt drift")


def _validate_inner_non_derivative_smoke(inner: Mapping[str, object], inherited: v1.SourceExecutionIdentity) -> None:
    safe_main = {
        "inner_smoke_sha256": _sha(_json_bytes(dict(inner))),
        "optimizer_steps": inner.get("optimizer_steps"),
        "legacy_derivative_nonnegative": inner.get("session_objective_derivatives_nonnegative"),
    }
    try:
        _validate_main_schema_scalars(inner, inherited)
    except SourceSmokeV5Error as error:
        raise V5ValidationError("main_schema_scalars", str(error), safe_main) from error
    try:
        _validate_gradient_coverage(inner)
    except SourceSmokeV5Error as error:
        coverage = inner.get("gradient_coverage")
        raise V5ValidationError("gradient_coverage", str(error), {
            "inner_smoke_sha256": safe_main["inner_smoke_sha256"],
            "gradient_coverage": dict(coverage) if isinstance(coverage, Mapping) else None,
        }) from error
    try:
        _validate_rng(inner)
    except SourceSmokeV5Error as error:
        rng = inner.get("rng")
        raise V5ValidationError("rng", str(error), {
            "inner_smoke_sha256": safe_main["inner_smoke_sha256"],
            "rng": dict(rng) if isinstance(rng, Mapping) else None,
        }) from error
    try:
        v1._validate_resources(inner.get("resources"))
    except v1.SourceLifecycleError as error:
        resources = inner.get("resources")
        raise V5ValidationError("resources", "CS-WG V5 inherited V1 resource gate drift", {
            "inner_smoke_sha256": safe_main["inner_smoke_sha256"],
            "resources": dict(resources) if isinstance(resources, Mapping) else None,
        }) from error


def _v5_attempt_payload(identity: SourceSmokeV5Identity, v3_graph: HeldHistoricalGraph, v4_graph: HeldHistoricalGraph) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_source_smoke_attempt_v5",
        "cell": CELL,
        "status": "ATTEMPT_RESERVED",
        "identity": identity.payload(),
        "accepted_v3_completed_graph": v3_graph.payload(),
        "failed_v4_graph": v4_graph.payload(),
        "source_resolved_or_opened": False,
        "model_constructed": False,
        "cuda_initialized": False,
        "optimizer_steps_completed": 0,
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _validate_launch_backend(value: object) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG V5 physical launch must be a mapping")
    item = dict(value)
    _require(item.get("schema") == "cross_session_worst_group_m1_source_smoke_physical_launch_v5"
             and item.get("provider") == "V5CachedCommonStratumSourceProvider"
             and item.get("runner") == "TorchCSWGSmokeRunner"
             and item.get("derivative_observer") == "V5DerivativeEvidenceCollector"
             and item.get("source_opened") is False and item.get("model_constructed") is False
             and item.get("cuda_initialized") is False and item.get("optimizer_steps_completed") == 0
             and item.get("source_only") is True,
             "CS-WG V5 physical launch semantics drift")
    return item


def _v5_launch_payload(identity: SourceSmokeV5Identity, v3_graph: HeldHistoricalGraph, v4_graph: HeldHistoricalGraph,
                       attempt_sha256: str, backend: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_source_smoke_launch_v5",
        "cell": CELL,
        "status": "LAUNCHED",
        "identity": identity.payload(),
        "accepted_v3_completed_graph_sha256": v3_graph.sha256,
        "failed_v4_graph_sha256": v4_graph.sha256,
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


def _prepared_authority_fragment(prepared: object) -> dict[str, object]:
    method = getattr(prepared, "inherited_v1_authority_fragment", None)
    _require(callable(method), "CS-WG V5 prepared V4-cache authority seam drift")
    value = method()
    _require(isinstance(value, Mapping), "CS-WG V5 prepared authority fragment must be a mapping")
    return dict(value)


def _prepared_cache(prepared: object) -> dict[str, object]:
    method = getattr(prepared, "digest_cache_payload", None)
    _require(callable(method), "CS-WG V5 prepared cache evidence seam drift")
    value = method()
    _require(isinstance(value, Mapping), "CS-WG V5 cache evidence must be a mapping")
    return dict(value)


def _prepared_fallback(prepared: object) -> tuple[dict[str, object], str, dict[str, object]]:
    fallback = getattr(prepared, "fallback_payload", None)
    digest = getattr(prepared, "fallback_sha256", None)
    step_zero = getattr(prepared, "step_zero_common_stratum_evidence", None)
    _require(isinstance(fallback, Mapping) and _require_sha(digest, "prepared V3 fallback")
             and isinstance(step_zero, Mapping), "CS-WG V5 prepared fallback seam drift")
    return dict(fallback), digest, dict(step_zero)


def _v5_source_authority_payload(identity: SourceSmokeV5Identity, v3_graph: HeldHistoricalGraph,
                                 v4_graph: HeldHistoricalGraph, prepared: object,
                                 *, accepted_fallback_sha256: str = V3_FALLBACK_SHA256) -> dict[str, object]:
    inherited = v1.source_authority_payload(identity.inherited_v1_smoke_identity, _prepared_authority_fragment(prepared))
    try:
        v1._validate_source_authority(inherited, identity.inherited_v1_smoke_identity)
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV5Error("CS-WG V5 inherited V1 source authority drift") from error
    fallback, fallback_sha, step_zero = _prepared_fallback(prepared)
    _require(_require_sha(accepted_fallback_sha256, "accepted V3 fallback")
             and fallback_sha == accepted_fallback_sha256 and _sha(_json_bytes(fallback)) == accepted_fallback_sha256
             and fallback == v3_graph.body("source_authority.json").get("deterministic_common_stratum_fallback"),
             "CS-WG V5 prepared common-stratum fallback differs from accepted V3")
    return {
        "schema": "cross_session_worst_group_m1_source_smoke_authority_v5",
        "identity": identity.payload(),
        "accepted_v3_completed_graph_sha256": v3_graph.sha256,
        "failed_v4_graph_sha256": v4_graph.sha256,
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


def _validate_v5_source_authority(value: object, identity: SourceSmokeV5Identity, v3_graph: HeldHistoricalGraph,
                                  v4_graph: HeldHistoricalGraph, prepared: object,
                                  *, accepted_fallback_sha256: str = V3_FALLBACK_SHA256) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG V5 source authority must be a mapping")
    item = dict(value)
    inherited = item.get("inherited_v1_source_authority")
    fallback, fallback_sha, step_zero = _prepared_fallback(prepared)
    _require(item.get("schema") == "cross_session_worst_group_m1_source_smoke_authority_v5"
             and item.get("identity") == identity.payload()
             and item.get("accepted_v3_completed_graph_sha256") == v3_graph.sha256
             and item.get("failed_v4_graph_sha256") == v4_graph.sha256
             and isinstance(inherited, Mapping)
             and item.get("inherited_v1_source_authority_sha256") == _sha(_json_bytes(dict(inherited)))
             and item.get("deterministic_common_stratum_fallback") == fallback
             and _require_sha(accepted_fallback_sha256, "accepted V3 fallback")
             and item.get("deterministic_common_stratum_fallback_sha256") == fallback_sha == accepted_fallback_sha256
             and item.get("common_stratum_step_zero_evidence") == step_zero
             and item.get("route_local_input_digest_cache") == _prepared_cache(prepared)
             and _flags_exact(item), "CS-WG V5 source authority cross-binding drift")
    try:
        v1._validate_source_authority(dict(inherited), identity.inherited_v1_smoke_identity)
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV5Error("CS-WG V5 inherited V1 source authority semantic drift") from error
    return item


def _v5_smoke_payload(identity: SourceSmokeV5Identity, v3_graph: HeldHistoricalGraph, v4_graph: HeldHistoricalGraph,
                      authority_sha256: str, authority: Mapping[str, object], raw_inner: Mapping[str, object],
                      derivative_evidence: object) -> dict[str, object]:
    inner = dict(raw_inner)
    inner.pop("_checkpoint_bodies", None)
    _validate_inner_non_derivative_smoke(inner, identity.inherited_v1_smoke_identity)
    evidence = _validate_derivative_evidence(
        derivative_evidence, expected_sessions=identity.inherited_v1_smoke_identity.spec.stage0_spec.source_sessions,
    )
    return {
        "schema": "cross_session_worst_group_m1_source_smoke_result_v5",
        "identity_sha256": identity.sha256,
        "source_authority_sha256": _require_sha(authority_sha256, "smoke authority"),
        "accepted_v3_completed_graph_sha256": v3_graph.sha256,
        "failed_v4_graph_sha256": v4_graph.sha256,
        "inherited_v1_smoke": inner,
        "inherited_v1_smoke_sha256": _sha(_json_bytes(inner)),
        "derivative_numeric_evidence": evidence,
        "non_derivative_v1_predicates_validated": True,
        "optimizer_steps": inner["optimizer_steps"],
        "one_concatenated_forward_per_step": inner["one_concatenated_forward_per_step"],
        "step_zero_calibration_ownership": authority["common_stratum_step_zero_evidence"],
        "route_local_input_digest_cache": authority["route_local_input_digest_cache"],
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _validate_v5_smoke(value: object, identity: SourceSmokeV5Identity, v3_graph: HeldHistoricalGraph,
                       v4_graph: HeldHistoricalGraph, authority: Mapping[str, object], authority_sha256: str) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG V5 smoke result must be a mapping")
    item = dict(value)
    inherited = item.get("inherited_v1_smoke")
    _require(item.get("schema") == "cross_session_worst_group_m1_source_smoke_result_v5"
             and item.get("identity_sha256") == identity.sha256
             and item.get("source_authority_sha256") == authority_sha256
             and item.get("accepted_v3_completed_graph_sha256") == v3_graph.sha256
             and item.get("failed_v4_graph_sha256") == v4_graph.sha256
             and isinstance(inherited, Mapping)
             and item.get("inherited_v1_smoke_sha256") == _sha(_json_bytes(dict(inherited)))
             and item.get("non_derivative_v1_predicates_validated") is True
             and item.get("optimizer_steps") == v1.SMOKE_STEPS
             and item.get("one_concatenated_forward_per_step") is True
             and item.get("step_zero_calibration_ownership") == authority.get("common_stratum_step_zero_evidence")
             and item.get("route_local_input_digest_cache") == authority.get("route_local_input_digest_cache")
             and _flags_exact(item), "CS-WG V5 smoke wrapper provenance drift")
    _validate_inner_non_derivative_smoke(dict(inherited), identity.inherited_v1_smoke_identity)
    _validate_derivative_evidence(
        item.get("derivative_numeric_evidence"),
        expected_sessions=identity.inherited_v1_smoke_identity.spec.stage0_spec.source_sessions,
    )
    return item


def _publish_v5_checkpoints(artifact: v1.ImmutableArtifactRoot, identity: SourceSmokeV5Identity,
                            bodies: object, *, inherited_smoke: Mapping[str, object]) -> tuple[dict[str, object], str]:
    _require(isinstance(bodies, Mapping) and tuple(sorted(bodies)) == ("best_source_train_loss", "last"),
             "CS-WG V5 smoke must return exact best/last checkpoint bodies")
    checkpoints: dict[str, dict[str, str]] = {}
    for role in ("best_source_train_loss", "last"):
        body = bodies[role]
        _require(isinstance(body, bytes) and body, f"CS-WG V5 checkpoint {role} body drift")
        state_field = "best_checkpoint_state_sha256" if role == "best_source_train_loss" else "final_model_state_sha256"
        checkpoints[role] = {
            "filename": f"checkpoint_{role}.pt",
            "sha256": artifact.publish_bytes(f"checkpoint_{role}.pt", body),
            "state_sha256": _require_sha(inherited_smoke.get(state_field), f"{role} state"),
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
        raise SourceSmokeV5Error("CS-WG V5 inherited checkpoint manifest drift") from error
    wrapper = {
        "schema": "cross_session_worst_group_m1_source_smoke_checkpoint_manifest_v5",
        "identity_sha256": identity.sha256,
        "inherited_v1_checkpoint_manifest": inherited_manifest,
        "inherited_v1_checkpoint_manifest_sha256": _sha(_json_bytes(inherited_manifest)),
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }
    return wrapper, artifact.publish_json("checkpoint_manifest.json", wrapper)


def _validate_v5_checkpoint_manifest(value: object, identity: SourceSmokeV5Identity) -> dict[str, object]:
    _require(isinstance(value, Mapping), "CS-WG V5 checkpoint manifest must be a mapping")
    item = dict(value)
    inherited = item.get("inherited_v1_checkpoint_manifest")
    _require(item.get("schema") == "cross_session_worst_group_m1_source_smoke_checkpoint_manifest_v5"
             and item.get("identity_sha256") == identity.sha256 and isinstance(inherited, Mapping)
             and item.get("inherited_v1_checkpoint_manifest_sha256") == _sha(_json_bytes(dict(inherited)))
             and _flags_exact(item), "CS-WG V5 checkpoint wrapper drift")
    try:
        v1._validate_checkpoint_manifest(dict(inherited), identity.inherited_v1_smoke_identity)
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV5Error("CS-WG V5 inherited checkpoint manifest semantics drift") from error
    return item


def _v5_terminal_payload(identity: SourceSmokeV5Identity, v3_graph: HeldHistoricalGraph, v4_graph: HeldHistoricalGraph,
                         *, attempt_sha256: str, launch_sha256: str, authority_sha256: str, smoke_sha256: str,
                         checkpoints_sha256: str) -> dict[str, object]:
    return {
        "schema": "cross_session_worst_group_m1_source_smoke_terminal_v5",
        "cell": CELL,
        "status": "PASS_SOURCE_SMOKE_COMMON_STRATUM_CONSTRUCTIBLE_GRANULAR_DERIVATIVE",
        "identity": identity.payload(),
        "accepted_v3_completed_graph_sha256": v3_graph.sha256,
        "failed_v4_graph_sha256": v4_graph.sha256,
        "attempt_sha256": _require_sha(attempt_sha256, "terminal attempt"),
        "launch_sha256": _require_sha(launch_sha256, "terminal launch"),
        "source_authority_sha256": _require_sha(authority_sha256, "terminal authority"),
        "smoke_sha256": _require_sha(smoke_sha256, "terminal smoke"),
        "checkpoint_manifest_sha256": _require_sha(checkpoints_sha256, "terminal checkpoint manifest"),
        "launch_closure_sha256": identity.closure["closure_sha256"],
        "final_closure_sha256": identity.closure["closure_sha256"],
        "source_only": True,
        "target_optimizer_backward_update": 0,
        **v1.FORBIDDEN_SURFACE_FLAGS,
    }


def _failure_details(error: BaseException) -> tuple[str, str, dict[str, object]]:
    if isinstance(error, V5ValidationError):
        return error.failure_stage, error.failed_predicate, dict(error.observed_summary)
    if isinstance(error, DerivativeObservationError):
        return error.failure_stage, error.failed_predicate, dict(error.observed_summary)
    return "physical_or_lifecycle", "backend_or_lifecycle_exception", {
        "error_class": type(error).__name__,
    }


def _v5_failure_payload(identity: SourceSmokeV5Identity, v3_graph: HeldHistoricalGraph, v4_graph: HeldHistoricalGraph,
                        *, attempt_sha256: str, launch_sha256: str | None, authority_sha256: str | None,
                        progress: v1.LifecycleProgress, error: BaseException) -> dict[str, object]:
    stage, predicate, observed = _failure_details(error)
    _require(stage in {"v5_smoke_validation", "derivative_observer", "physical_or_lifecycle"}
             and predicate in {
                 "main_schema_scalars", "gradient_coverage", "rng", "resources", "derivative_numeric_gate",
                 "derivative_observer_exception", "backend_or_lifecycle_exception",
             }, "CS-WG V5 failure predicate topology drift")
    return {
        "schema": "cross_session_worst_group_m1_source_smoke_failure_v5",
        "cell": CELL,
        "status": "FAILED",
        "identity": identity.payload(),
        "accepted_v3_completed_graph_sha256": v3_graph.sha256,
        "failed_v4_graph_sha256": v4_graph.sha256,
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
class SourceSmokeV5LifecycleResult:
    root_identity: tuple[int, int]
    attempt_sha256: str
    launch_sha256: str | None
    source_authority_sha256: str | None
    smoke_sha256: str | None
    checkpoint_manifest_sha256: str | None
    terminal_sha256: str | None
    failure_sha256: str | None


class DeferredSourceSmokeV5Backend(Protocol):
    def launch_payload(self, identity: SourceSmokeV5Identity) -> Mapping[str, object]: ...
    def prepare_source(self, identity: SourceSmokeV5Identity) -> object: ...
    def run_smoke(self, identity: SourceSmokeV5Identity) -> Mapping[str, object]: ...
    def checkpoint_bodies(self) -> Mapping[str, bytes]: ...
    def progress(self) -> v1.LifecycleProgress: ...
    def close(self) -> None: ...


def _revalidate_v5_published_graph(artifact: v1.ImmutableArtifactRoot, identity: SourceSmokeV5Identity,
                                  v3_graph: HeldHistoricalGraph, v4_graph: HeldHistoricalGraph, prepared: object,
                                  *, attempt_sha256: str, launch_sha256: str, authority_sha256: str,
                                  smoke_sha256: str, checkpoints_sha256: str,
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
    _require(attempt == _v5_attempt_payload(identity, v3_graph, v4_graph)
             and launch == _v5_launch_payload(identity, v3_graph, v4_graph, attempt_sha256, launch["backend"]),
             "CS-WG V5 published attempt/launch graph drift")
    _validate_v5_source_authority(
        authority, identity, v3_graph, v4_graph, prepared,
        accepted_fallback_sha256=accepted_fallback_sha256,
    )
    _validate_v5_smoke(smoke, identity, v3_graph, v4_graph, authority, authority_sha256)
    manifest_value = _validate_v5_checkpoint_manifest(manifest, identity)
    inherited = manifest_value["inherited_v1_checkpoint_manifest"]
    assert isinstance(inherited, Mapping)
    for role in ("best_source_train_loss", "last"):
        entry = inherited["checkpoints"][role]
        assert isinstance(entry, Mapping)
        artifact.read_bytes_pair(str(entry["filename"]), expected_sha256=str(entry["sha256"]))


def _execute_reviewed_source_smoke_v5(
    root: Path, *, identity: SourceSmokeV5Identity, capability: object, backend: DeferredSourceSmokeV5Backend,
    environ: Mapping[str, str] | None, predecessor_loader: Any,
    accepted_fallback_sha256: str = V3_FALLBACK_SHA256,
) -> SourceSmokeV5LifecycleResult:
    cap = _require_v5_capability(capability, identity)
    validate_source_smoke_v5_identity_current(Path(root), identity)
    predecessors = predecessor_loader(Path(root))
    _require(isinstance(predecessors, tuple) and len(predecessors) == 2
             and all(isinstance(item, HeldHistoricalGraph) for item in predecessors),
             "CS-WG V5 predecessor loader type drift")
    v3_graph, v4_graph = predecessors
    _require(v3_graph.sha256 == cap.v3_completed_graph_sha256 and v4_graph.sha256 == cap.v4_failed_graph_sha256,
             "CS-WG V5 predecessor/capability binding drift")
    try:
        v1.validate_device_environment(identity.device, environ)
    except v1.SourceLifecycleError as error:
        raise SourceSmokeV5Error("CS-WG V5 device environment drift before reserve") from error
    _assert_v5_root_fresh(Path(root), identity.spec)
    artifact = v1.ImmutableArtifactRoot.reserve(Path(root), identity.spec)  # type: ignore[arg-type]
    attempt_sha256 = artifact.publish_json("attempt.json", _v5_attempt_payload(identity, v3_graph, v4_graph))
    launch_sha256: str | None = None
    authority_sha256: str | None = None
    smoke_sha256: str | None = None
    checkpoints_sha256: str | None = None
    terminal_sha256: str | None = None
    try:
        launch_sha256 = artifact.publish_json(
            "launch.json", _v5_launch_payload(identity, v3_graph, v4_graph, attempt_sha256, backend.launch_payload(identity)),
        )
        prepared = backend.prepare_source(identity)
        authority = _v5_source_authority_payload(
            identity, v3_graph, v4_graph, prepared, accepted_fallback_sha256=accepted_fallback_sha256,
        )
        _validate_v5_source_authority(
            authority, identity, v3_graph, v4_graph, prepared,
            accepted_fallback_sha256=accepted_fallback_sha256,
        )
        authority_sha256 = artifact.publish_json("source_authority.json", authority)
        raw = dict(backend.run_smoke(identity))
        checkpoint_bodies = raw.pop("_checkpoint_bodies", None)
        derivative_evidence = raw.pop("_v5_derivative_evidence", None)
        if checkpoint_bodies is None:
            checkpoint_bodies = backend.checkpoint_bodies()
        smoke = _v5_smoke_payload(
            identity, v3_graph, v4_graph, authority_sha256, authority, raw, derivative_evidence,
        )
        _validate_v5_smoke(smoke, identity, v3_graph, v4_graph, authority, authority_sha256)
        smoke_sha256 = artifact.publish_json("smoke.json", smoke)
        _manifest, checkpoints_sha256 = _publish_v5_checkpoints(
            artifact, identity, checkpoint_bodies, inherited_smoke=dict(smoke["inherited_v1_smoke"]),
        )
        validate_source_smoke_v5_identity_current(Path(root), identity)
        predecessors_now = predecessor_loader(Path(root))
        _require(isinstance(predecessors_now, tuple) and len(predecessors_now) == 2
                 and predecessors_now[0].sha256 == v3_graph.sha256 == cap.v3_completed_graph_sha256
                 and predecessors_now[1].sha256 == v4_graph.sha256 == cap.v4_failed_graph_sha256,
                 "CS-WG V5 immutable predecessor drifted during smoke")
        try:
            v1.validate_device_environment(identity.device, environ)
        except v1.SourceLifecycleError as error:
            raise SourceSmokeV5Error("CS-WG V5 device environment drift before terminal") from error
        _revalidate_v5_published_graph(
            artifact, identity, v3_graph, v4_graph, prepared,
            attempt_sha256=attempt_sha256, launch_sha256=launch_sha256, authority_sha256=authority_sha256,
            smoke_sha256=smoke_sha256, checkpoints_sha256=checkpoints_sha256,
            accepted_fallback_sha256=accepted_fallback_sha256,
        )
        terminal = _v5_terminal_payload(
            identity, v3_graph, v4_graph, attempt_sha256=attempt_sha256, launch_sha256=launch_sha256,
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
                 "CS-WG V5 published terminal graph drift")
        return SourceSmokeV5LifecycleResult(
            artifact.root_identity, attempt_sha256, launch_sha256, authority_sha256,
            smoke_sha256, checkpoints_sha256, terminal_sha256, None,
        )
    except BaseException as error:
        try:
            progress = backend.progress()
            _require(isinstance(progress, v1.LifecycleProgress), "CS-WG V5 backend progress type drift")
        except BaseException:
            progress = v1.LifecycleProgress()
        failure_sha256 = artifact.publish_json(
            "failure.json",
            _v5_failure_payload(
                identity, v3_graph, v4_graph, attempt_sha256=attempt_sha256, launch_sha256=launch_sha256,
                authority_sha256=authority_sha256, progress=progress, error=error,
            ),
        )
        return SourceSmokeV5LifecycleResult(
            artifact.root_identity, attempt_sha256, launch_sha256, authority_sha256,
            smoke_sha256, checkpoints_sha256, terminal_sha256, failure_sha256,
        )
    finally:
        try:
            backend.close()
        finally:
            artifact.close()


def execute_reviewed_source_smoke_v5(
    root: Path, *, identity: SourceSmokeV5Identity, capability: object, backend: DeferredSourceSmokeV5Backend,
    environ: Mapping[str, str] | None,
) -> SourceSmokeV5LifecycleResult:
    """Root-only physical V5 smoke; the public CLI cannot call this."""
    return _execute_reviewed_source_smoke_v5(
        Path(root), identity=identity, capability=capability, backend=backend, environ=environ,
        predecessor_loader=validate_v5_predecessors,
    )


def dry_plan(root: Path | None = None) -> dict[str, object]:
    """Static declaration only; ``root`` is deliberately not inspected."""
    del root
    return {
        "cell": CELL,
        "phase": PHASE,
        "workorder_sha256": WORKORDER_SHA256,
        "fresh_v5_root_relative": V5_ROOT_RELATIVE,
        "accepted_v3_completed_graph": V3_COMPLETED_EXPECTATION.payload(),
        "failed_v4_graph": {**V4_FAILED_EXPECTATION.payload(), "error_sha256": V4_FAILURE_ERROR_SHA256},
        "historical_closures_not_rebuilt_as_current": {
            "v3": V3_CLOSURE_SHA256, "v4": V4_HISTORICAL_CLOSURE_SHA256,
        },
        "source_only_smoke": {
            "optimizer_steps": v1.SMOKE_STEPS,
            "one_concatenated_mixed_session_forward_per_step": True,
            "total_batch_size": v1.plan.TOTAL_BATCH_SIZE,
            "derivative_numeric_gate": derivative_gate_contract_payload(),
            "amp": False, "tf32": False, "compile": False,
        },
        "opens_source_or_target": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "execution_authorized": False,
    }


__all__ = (
    "CELL", "PHASE", "WORKORDER_RELATIVE", "WORKORDER_SHA256", "V5_ROOT_RELATIVE",
    "V3_COMPLETED_EXPECTATION", "V4_FAILED_EXPECTATION", "SourceSmokeV5Error", "V5ValidationError",
    "DerivativeObservationError", "HistoricalGraphExpectation", "HeldHistoricalGraph", "SourceSmokeV5Spec",
    "SourceSmokeV5Identity", "SourceSmokeV5Capability", "SourceSmokeV5LifecycleResult", "DerivativeEvidenceCollector",
    "derivative_gate_contract_payload", "implementation_closure", "validate_current_closure",
    "build_source_smoke_v5_identity", "validate_source_smoke_v5_identity_current", "validate_v5_predecessors",
    "issue_root_reviewed_source_smoke_v5_capability", "execute_reviewed_source_smoke_v5", "dry_plan",
)
