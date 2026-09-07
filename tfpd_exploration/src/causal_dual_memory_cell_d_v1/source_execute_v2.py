"""Additive CDM-D Source Execution V2 byte-domain successor.

This module deliberately composes the frozen V1 source-execution lifecycle.
It changes one and only one physical authority comparison: the theta
authority's raw-T4 SHA is ``SHA256(contiguous float32 raw bytes)`` rather
than V1's incompatible float64 tensor-digest domain.  The public CLI imports
only this standard-library-only module; source files, Torch, checkpoints,
and CUDA remain behind the reviewed in-process capability boundary.
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


CELL = v1.CELL
V2_ROUTE = "causal_dual_memory_cell_d_source_execution_v2"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CDM_D_SOURCE_EXECUTION_V2_20260825.md"
WORKORDER_SHA256 = "b71e8a7796db228f04c73efee841793bf0998413bbb436ecdc36d7c30d6e75ce"

# This is the independently accepted V1 implementation closure, not a
# self-consistency label.  V2 reconstructs it before capability issue and
# again before terminal publication.
V1_IMPLEMENTATION_CLOSURE_SHA256 = "77f9495780605d05be177500ff0ef61b3bd35a427ddd12af4fbab97b3486428b"

SOURCE_SMOKE_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_source_smoke_v2"
SOURCE_GATE_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_source_gate_v2"
V1_FAILED_SMOKE_ROOT_RELATIVE = "tfpd_exploration/results/causal_dual_memory_cell_d_source_smoke_v1"

THETA_RAW_HASH_LAW = "theta_float32_contiguous_raw_bytes_v1"
THETA_RAW_HASH_LAW_TEXT = "SHA256(np.ascontiguousarray(raw_t4,dtype=np.float32).tobytes())"


class SourceExecutionV2Error(v1.SourceExecutionError):
    """Fail closed for V2 lineage, byte-domain, and lifecycle drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceExecutionV2Error(message)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64
             and all(character in "0123456789abcdef" for character in value),
             f"{label} must be an exact lowercase SHA-256")
    return value


def theta_raw_t4_float32_bytes_sha256(raw_t4: Any) -> str:
    """Return the *sealed theta-authority* raw-T4 hash domain exactly.

    NumPy is deliberately local to this execution-only helper.  It neither
    imports Torch nor changes the source adapter's raw numeric tensor.  The
    caller retains that ordinary raw tensor after this one authority check.
    """
    import numpy as np

    raw = np.ascontiguousarray(np.asarray(raw_t4, dtype=np.float32))
    return hashlib.sha256(raw.tobytes()).hexdigest()


def validate_theta_raw_t4_authority(
    raw_t4: Any,
    *,
    sealed_raw_t4_sha256: str,
    sealed_theta_float64: Any,
    sealed_valid_mask: Any,
    n_units: int,
    channel_ids: Any,
    session: str,
    modulation_eps: float,
) -> tuple[Any, dict[str, object]]:
    """Pure NumPy proof for V2's sealed theta raw-T4 authority contract.

    This is intentionally usable in no-data synthetic tests.  The physical
    provider supplies arrays decoded from the sealed theta artifact and then
    replaces the two local byte-only topology digests with the inherited
    source-adapter digest domain for its durable source-authority row.
    """
    import numpy as np

    _require(type(n_units) is int and n_units > 0, "V2 theta authority unit count drift")
    _sha(sealed_raw_t4_sha256, "V2 sealed raw T4 SHA")
    observed_float64 = np.asarray(raw_t4, dtype=np.float64)
    observed_float32 = np.ascontiguousarray(np.asarray(raw_t4, dtype=np.float32))
    theta = np.ascontiguousarray(np.asarray(sealed_theta_float64, dtype=np.float64))
    valid = np.ascontiguousarray(np.asarray(sealed_valid_mask, dtype=np.bool_))
    channels = np.ascontiguousarray(np.asarray(channel_ids, dtype=np.int64))
    _require(
        observed_float64.ndim == 2 and observed_float64.shape == (n_units, 4)
        and observed_float32.shape == (n_units, 4)
        and bool(np.isfinite(observed_float64).all()) and bool(np.isfinite(observed_float32).all())
        and theta.shape == valid.shape == channels.shape == (n_units,),
        "V2 raw T4 exact [n_units,4]/finite topology drift",
    )
    raw_sha = theta_raw_t4_float32_bytes_sha256(observed_float32)
    _require(raw_sha == sealed_raw_t4_sha256,
             "V2 theta authority contiguous-float32 raw T4 bytes drift")
    recomputed_theta = np.ascontiguousarray(
        np.arctan2(observed_float64[:, 1], observed_float64[:, 0]), dtype=np.float64,
    )
    _require(recomputed_theta.tobytes() == theta.tobytes(),
             "V2 theta atan2(c,a) bitwise authority drift")
    recomputed_valid = np.ascontiguousarray(
        observed_float64[:, 2] > float(modulation_eps), dtype=np.bool_,
    )
    _require(recomputed_valid.tobytes() == valid.tobytes(),
             "V2 theta raw_m > MODULATION_EPS bitwise authority drift")
    canonical_channels = np.arange(n_units, dtype=np.int64)
    _require(np.array_equal(channels, canonical_channels),
             "V2 raw-T4 canonical unit-order proof drift")
    expected_invalid = v1.KNOWN_THETA_INVALID_UNIT_COUNTS.get(session, 0)
    _require(int((~recomputed_valid).sum()) == expected_invalid,
             "V2 theta exact known undefined-unit count drift")
    proof = {
        "schema": "causal_dual_memory_cell_d_theta_raw_proof_v2",
        "session": session,
        "hash_law": THETA_RAW_HASH_LAW,
        "hash_law_expression": THETA_RAW_HASH_LAW_TEXT,
        "raw_t4_shape": [n_units, 4],
        "raw_t4_dtype": "float32",
        "raw_t4_all_finite": True,
        "raw_t4_sha256": raw_sha,
        "sealed_raw_t4_sha256": sealed_raw_t4_sha256,
        "theta_float64_sha256": sha256_bytes(recomputed_theta.tobytes()),
        "sealed_theta_float64_sha256": sha256_bytes(theta.tobytes()),
        "theta_atan2_float64_bitwise_equal": True,
        "valid_mask_sha256": sha256_bytes(recomputed_valid.tobytes()),
        "sealed_valid_mask_sha256": sha256_bytes(valid.tobytes()),
        "validity_raw_m_gt_modulation_eps_bitwise_equal": True,
        "known_invalid_unit_count": expected_invalid,
        "canonical_unit_order_exact": True,
        "canonical_unit_order_sha256": sha256_bytes(canonical_channels.tobytes()),
    }
    return recomputed_valid.copy(), proof


@dataclass(frozen=True)
class V1FailedPredecessorExpectation:
    """Immutable V1 failure graph that licenses, but is never overwritten by, V2."""

    root_relative: str
    attempt_sha256: str
    launch_sha256: str
    failure_sha256: str
    v1_closure_sha256: str

    def __post_init__(self) -> None:
        v1._safe_relative(self.root_relative)
        for label, value in (
            ("V1 predecessor attempt", self.attempt_sha256),
            ("V1 predecessor launch", self.launch_sha256),
            ("V1 predecessor failure", self.failure_sha256),
            ("V1 predecessor closure", self.v1_closure_sha256),
        ):
            _sha(value, label)

    def payload(self) -> dict[str, object]:
        return {
            "schema": "causal_dual_memory_cell_d_source_execution_v1_failed_predecessor_v2",
            "root_relative": self.root_relative,
            "leaves": [
                {"name": "attempt.json", "sha256": self.attempt_sha256, "mode": 0o444},
                {"name": "launch.json", "sha256": self.launch_sha256, "mode": 0o444},
                {"name": "failure.json", "sha256": self.failure_sha256, "mode": 0o444},
            ],
            "exact_leaf_count": 6,
            "v1_implementation_closure_sha256": self.v1_closure_sha256,
            "failure_contract": {
                "status": "FAILED",
                "stage": "prepare",
                "source_resolved": True,
                "source_opened": False,
                "checkpoint_opened": False,
                "cuda_initialized": False,
                "model_forward_calls": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
                "parameter_updates": 0,
                "within_external_formal_target_opened": False,
                "terminal_published": False,
            },
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(_json_bytes(self.payload()))


V1_FAILED_PREDECESSOR = V1FailedPredecessorExpectation(
    root_relative=V1_FAILED_SMOKE_ROOT_RELATIVE,
    attempt_sha256="13825a85ec3937653a27245b55db7dd1d1df024822e3b33dafe71633aa31b29f",
    launch_sha256="b76b1d721e460297b45c7986615fd01aef8c429f20946891a7120c2120e4f963",
    failure_sha256="a283cca52947fe54d681ae4444bd7885515a62bdf20adb2350362629b14f16ea",
    v1_closure_sha256=V1_IMPLEMENTATION_CLOSURE_SHA256,
)


SOURCE_SMOKE_SPEC = v1.SourceExecutionSpec(
    "source_smoke", SOURCE_SMOKE_ROOT_RELATIVE, v1.SOURCE_SMOKE_SESSION, 30, (30, 31),
)
SOURCE_GATE_SPEC = v1.SourceExecutionSpec("source_gate", SOURCE_GATE_ROOT_RELATIVE, None, None, ())


_V2_OWNED_PATHS: tuple[str, ...] = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_v2.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical_v2.py",
    "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_gate_v2.py",
    "tfpd_exploration/tests/test_causal_dual_memory_cell_d_source_execution_v2.py",
)


def _regular_sha256(root: Path, relative: str) -> str:
    """Use V1's direct non-symlink closure read without a glob or import scan."""
    try:
        return v1._regular_sha256(Path(root), relative)
    except v1.SourceExecutionError as error:
        raise SourceExecutionV2Error(str(error)) from error


def execution_closure_payload(root: Path) -> dict[str, object]:
    """Bind the frozen V1 dependency closure plus the four V2-owned leaves."""
    inherited = v1.execution_closure_payload(Path(root))
    _require(
        inherited.get("closure_sha256") == V1_IMPLEMENTATION_CLOSURE_SHA256,
        "accepted V1 implementation closure drift",
    )
    inherited_rows = inherited.get("paths")
    _require(isinstance(inherited_rows, list)
             and all(isinstance(row, Mapping) and isinstance(row.get("path"), str)
                     and _sha(row.get("sha256"), "inherited closure row SHA") == row.get("sha256")
                     for row in inherited_rows),
             "accepted V1 closure path topology drift")
    paths: list[str] = []
    for relative in [*(str(row["path"]) for row in inherited_rows), *_V2_OWNED_PATHS]:
        if relative not in paths:
            paths.append(relative)
    rows = [{"path": relative, "sha256": _regular_sha256(Path(root), relative)} for relative in paths]
    expected_workorder = next((row["sha256"] for row in rows if row["path"] == WORKORDER_RELATIVE), None)
    _require(expected_workorder == WORKORDER_SHA256, "Source Execution V2 workorder SHA drift")
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_closure_v2",
        "v1_implementation_closure_sha256": V1_IMPLEMENTATION_CLOSURE_SHA256,
        "v1_inherited_path_count": len(inherited_rows),
        "paths": rows,
        "closure_sha256": sha256_bytes(_json_bytes(rows)),
    }


@dataclass(frozen=True)
class SourceExecutionV2Identity:
    """V2 identity wrapping a byte-exact frozen V1 execution identity."""

    inherited_v1_identity: v1.SourceExecutionIdentity
    closure: Mapping[str, object]
    predecessor: V1FailedPredecessorExpectation = field(default_factory=lambda: V1_FAILED_PREDECESSOR)
    theta_raw_hash_law: str = THETA_RAW_HASH_LAW

    def __post_init__(self) -> None:
        _require(isinstance(self.inherited_v1_identity, v1.SourceExecutionIdentity),
                 "V2 identity must wrap a typed V1 identity")
        _require(isinstance(self.predecessor, V1FailedPredecessorExpectation),
                 "V2 predecessor expectation type drift")
        _require(self.predecessor == V1_FAILED_PREDECESSOR,
                 "V2 predecessor graph must be the exact immutable V1 failure")
        _require(self.theta_raw_hash_law == THETA_RAW_HASH_LAW,
                 "V2 theta raw-hash law drift")
        _require(isinstance(self.closure, Mapping)
                 and self.closure.get("schema") == "causal_dual_memory_cell_d_source_execution_closure_v2"
                 and self.closure.get("v1_implementation_closure_sha256") == V1_IMPLEMENTATION_CLOSURE_SHA256
                 and _sha(self.closure.get("closure_sha256"), "V2 identity closure")
                 == self.closure.get("closure_sha256"),
                 "V2 identity closure topology drift")
        _require(self.inherited_v1_identity.closure.get("closure_sha256") == V1_IMPLEMENTATION_CLOSURE_SHA256,
                 "V2 inherited V1 identity closure drift")

    @property
    def spec(self) -> v1.SourceExecutionSpec:
        return self.inherited_v1_identity.spec

    @property
    def strict_train_roster(self) -> tuple[str, ...]:
        return self.inherited_v1_identity.strict_train_roster

    @property
    def selected_device(self) -> Mapping[str, object]:
        return self.inherited_v1_identity.selected_device

    def payload(self) -> dict[str, object]:
        return {
            "schema": "causal_dual_memory_cell_d_source_execution_identity_v2",
            "cell": CELL,
            "run_spec": self.spec.payload(),
            "closure": dict(self.closure),
            "inherited_v1_identity": self.inherited_v1_identity.payload(),
            "v1_failed_predecessor": self.predecessor.payload(),
            "v1_failed_predecessor_sha256": self.predecessor.sha256,
            "theta_raw_t4_hash_law": self.theta_raw_hash_law,
            "theta_raw_t4_hash_law_expression": THETA_RAW_HASH_LAW_TEXT,
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
    spec: v1.SourceExecutionSpec,
    selected_device: Mapping[str, object],
    fixed_assets: Mapping[str, v1.BoundAsset] | None = None,
) -> SourceExecutionV2Identity:
    """Build V2 identity source-free by reusing V1's fixed-asset preflight."""
    inherited = v1.build_identity(
        Path(root), spec=spec, selected_device=selected_device, fixed_assets=fixed_assets,
    )
    _require(inherited.closure.get("closure_sha256") == V1_IMPLEMENTATION_CLOSURE_SHA256,
             "V2 cannot inherit an unaccepted V1 implementation closure")
    return SourceExecutionV2Identity(inherited, execution_closure_payload(Path(root)))


def validate_identity_current(root: Path, identity: SourceExecutionV2Identity) -> None:
    _require(isinstance(identity, SourceExecutionV2Identity), "V2 execution identity must be typed")
    try:
        v1.validate_identity_current(Path(root), identity.inherited_v1_identity)
    except v1.SourceExecutionError as error:
        raise SourceExecutionV2Error(str(error)) from error
    _require(identity.closure == execution_closure_payload(Path(root)),
             "V2 implementation closure drift")
    _require(identity.inherited_v1_identity.closure.get("closure_sha256") == V1_IMPLEMENTATION_CLOSURE_SHA256,
             "V2 inherited V1 closure drift")


@dataclass(frozen=True)
class V1FailedPredecessorBinding:
    """A same-held-directory-FD verification record for the immutable V1 graph."""

    expectation: V1FailedPredecessorExpectation
    directory_device: int
    directory_inode: int

    def payload(self) -> dict[str, object]:
        return {
            "schema": "causal_dual_memory_cell_d_source_execution_v1_failed_predecessor_binding_v2",
            "expectation": self.expectation.payload(),
            "expectation_sha256": self.expectation.sha256,
            "directory_device": self.directory_device,
            "directory_inode": self.directory_inode,
            "single_held_nofollow_directory_fd": True,
            "all_six_leaves_revalidated_before_v2": True,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(_json_bytes(self.payload()))


def _read_exact_predecessor_leaf(fd: int, name: str) -> bytes:
    try:
        return v1._read_held_leaf(fd, name, mode=0o444)
    except v1.SourceExecutionError as error:
        raise SourceExecutionV2Error(str(error)) from error


def _load_json(body: bytes, label: str) -> dict[str, object]:
    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as error:
        raise SourceExecutionV2Error(f"{label} is not JSON") from error
    _require(isinstance(decoded, Mapping), f"{label} JSON root drift")
    return dict(decoded)


def _validate_v1_predecessor_semantics(
    attempt: Mapping[str, object], launch: Mapping[str, object], failure: Mapping[str, object],
    expectation: V1FailedPredecessorExpectation,
) -> None:
    identity = attempt.get("identity")
    _require(
        attempt.get("schema") == "causal_dual_memory_cell_d_source_execution_attempt_v1"
        and attempt.get("cell") == CELL
        and attempt.get("status") == "ATTEMPT_RESERVED"
        and isinstance(identity, Mapping)
        and identity.get("schema") == "causal_dual_memory_cell_d_source_execution_identity_v1"
        and isinstance(identity.get("closure"), Mapping)
        and identity["closure"].get("closure_sha256") == expectation.v1_closure_sha256
        and isinstance(identity.get("run_spec"), Mapping)
        and identity["run_spec"].get("kind") == "source_smoke"
        and identity["run_spec"].get("root_relative") == expectation.root_relative
        and attempt.get("source_only") is True
        and attempt.get("source_resolved_or_opened") is False
        and attempt.get("checkpoint_opened") is False
        and attempt.get("cuda_initialized") is False
        and attempt.get("target_optimizer_backward_update") == 0
        and attempt.get("within_external_formal_target_forbidden") is True,
        "V1 failed predecessor attempt semantic drift",
    )
    _require(
        launch.get("schema") == "causal_dual_memory_cell_d_source_execution_launch_v1"
        and launch.get("cell") == CELL
        and launch.get("status") == "LAUNCHED"
        and launch.get("identity") == identity
        and launch.get("attempt_sha256") == expectation.attempt_sha256
        and isinstance(launch.get("preflight"), Mapping)
        and launch["preflight"].get("source_resolved_or_opened") is False
        and launch["preflight"].get("checkpoint_opened") is False
        and launch["preflight"].get("cuda_initialized") is False
        and launch.get("source_only") is True
        and launch.get("target_optimizer_backward_update") == 0,
        "V1 failed predecessor launch semantic drift",
    )
    flags = failure.get("flags")
    _require(
        failure.get("schema") == "causal_dual_memory_cell_d_source_execution_failure_v1"
        and failure.get("cell") == CELL
        and failure.get("status") == "FAILED"
        and failure.get("identity") == identity
        and failure.get("attempt_sha256") == expectation.attempt_sha256
        and failure.get("launch_sha256") == expectation.launch_sha256
        and failure.get("source_authority_sha256") is None
        and failure.get("stage") == "prepare"
        and isinstance(flags, Mapping)
        and flags.get("stage") == "prepare"
        and flags.get("source_resolved") is True
        and flags.get("source_opened") is False
        and flags.get("checkpoint_opened") is False
        and flags.get("cuda_initialized") is False
        and flags.get("model_forward_calls") == 0
        and flags.get("backward_calls") == 0
        and flags.get("optimizer_steps") == 0
        and flags.get("parameter_updates") == 0
        and flags.get("within_opened") is False
        and flags.get("external_opened") is False
        and flags.get("formal_opened") is False
        and flags.get("target_opened") is False
        and flags.get("normalizer_refit") is False
        and flags.get("oom_retry_attempted") is False
        and failure.get("terminal_published") is False
        and failure.get("source_only") is True,
        "V1 failed predecessor failure semantic drift",
    )


def validate_v1_failed_predecessor(
    root: Path,
    *,
    expectation: V1FailedPredecessorExpectation | None = None,
) -> V1FailedPredecessorBinding:
    """Validate V1's exact six-leaf failed graph under one held directory FD.

    ``expectation`` is injectable solely for isolated synthetic tests.  Every
    production capability and execution call uses the immutable module literal
    above, never a caller-supplied predecessor graph.
    """
    expected = V1_FAILED_PREDECESSOR if expectation is None else expectation
    _require(isinstance(expected, V1FailedPredecessorExpectation),
             "V1 predecessor expectation must be typed")
    expectation = expected
    relative = v1._safe_relative(expectation.root_relative)
    directory = Path(root).absolute() / relative
    try:
        named = os.lstat(directory)
        _require(stat.S_ISDIR(named.st_mode) and not stat.S_ISLNK(named.st_mode),
                 "V1 failed predecessor root is not a regular non-symlink directory")
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise SourceExecutionV2Error("V1 failed predecessor root cannot be opened descriptor-safely") from error
    identity = (int(named.st_dev), int(named.st_ino))
    try:
        held = os.fstat(fd)
        _require(stat.S_ISDIR(held.st_mode) and (int(held.st_dev), int(held.st_ino)) == identity,
                 "V1 failed predecessor root identity drift before held read")
        expected_sha = {
            "attempt.json": expectation.attempt_sha256,
            "launch.json": expectation.launch_sha256,
            "failure.json": expectation.failure_sha256,
        }
        expected_names = set(expected_sha)
        expected_names.update(f"{name}.sha256" for name in tuple(expected_sha))
        observed_names = set(os.listdir(fd))
        _require(observed_names == expected_names,
                 "V1 failed predecessor six-leaf topology drift")
        bodies: dict[str, bytes] = {}
        for name, digest in expected_sha.items():
            body = _read_exact_predecessor_leaf(fd, name)
            _require(sha256_bytes(body) == digest, f"V1 failed predecessor {name} body SHA drift")
            _require(
                _read_exact_predecessor_leaf(fd, f"{name}.sha256")
                == f"{digest}  {name}\n".encode("ascii"),
                f"V1 failed predecessor {name} canonical sidecar drift",
            )
            bodies[name] = body
        named_after = os.lstat(directory)
        held_after = os.fstat(fd)
        _require(
            stat.S_ISDIR(named_after.st_mode) and not stat.S_ISLNK(named_after.st_mode)
            and (int(named_after.st_dev), int(named_after.st_ino)) == identity
            and (int(held_after.st_dev), int(held_after.st_ino)) == identity,
            "V1 failed predecessor root identity drift during held read",
        )
        _validate_v1_predecessor_semantics(
            _load_json(bodies["attempt.json"], "V1 predecessor attempt"),
            _load_json(bodies["launch.json"], "V1 predecessor launch"),
            _load_json(bodies["failure.json"], "V1 predecessor failure"),
            expectation,
        )
        return V1FailedPredecessorBinding(expectation, *identity)
    finally:
        os.close(fd)


def assert_prospective_root_fresh(root: Path, spec: v1.SourceExecutionSpec) -> None:
    """Spec-scoped no-write freshness check; it never creates the candidate."""
    _require(isinstance(spec, v1.SourceExecutionSpec), "V2 freshness requires a typed V1 run spec")
    candidate = Path(root).absolute() / v1._safe_relative(spec.root_relative)
    try:
        os.lstat(candidate)
    except FileNotFoundError:
        return
    except OSError as error:
        raise SourceExecutionV2Error("V2 prospective result root cannot be inspected safely") from error
    raise SourceExecutionV2Error("V2 prospective source-execution root already exists")


class _V2RootReviewSeal:
    pass


_ROOT_REVIEW_SEAL = _V2RootReviewSeal()


@dataclass(frozen=True)
class SourceExecutionV2Capability:
    """Opaque root-reviewed V2 capability coupled to V1's source-data cap."""

    identity_sha256: str
    inherited_v1_capability: v1.SourceExecutionCapability = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        _sha(self.identity_sha256, "V2 execution capability identity SHA")
        _require(self._seal is _ROOT_REVIEW_SEAL
                 and isinstance(self.inherited_v1_capability, v1.SourceExecutionCapability),
                 "V2 execution capability is not root-reviewed")


def _issue_root_reviewed_capability(
    root: Path,
    identity: SourceExecutionV2Identity,
    *,
    source_data_root: v1.StrictSourceDataRootCapability | None,
    seal: object,
) -> SourceExecutionV2Capability:
    """Root-only capability minting with all predecessor gates before a root exists."""
    _require(seal is _ROOT_REVIEW_SEAL, "only the root reviewer may issue a V2 execution capability")
    validate_identity_current(Path(root), identity)
    validate_v1_failed_predecessor(Path(root))
    assert_prospective_root_fresh(Path(root), identity.spec)
    inherited = v1._issue_root_reviewed_capability(
        identity.inherited_v1_identity,
        seal=v1._ROOT_REVIEW_SEAL,
        source_data_root=source_data_root,
    )
    return SourceExecutionV2Capability(identity.sha256, inherited, seal)


def require_execution_capability(
    capability: object, identity: SourceExecutionV2Identity,
) -> SourceExecutionV2Capability:
    _require(isinstance(capability, SourceExecutionV2Capability)
             and capability._seal is _ROOT_REVIEW_SEAL
             and capability.identity_sha256 == identity.sha256,
             "V2 execution requires an exact in-process root-reviewed capability")
    try:
        v1.require_execution_capability(capability.inherited_v1_capability, identity.inherited_v1_identity)
    except v1.SourceExecutionError as error:
        raise SourceExecutionV2Error(str(error)) from error
    return capability


def _v2_binding(identity: SourceExecutionV2Identity, predecessor: V1FailedPredecessorBinding) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_v2_binding",
        "identity_sha256": identity.sha256,
        "v2_closure_sha256": identity.closure["closure_sha256"],
        "v1_implementation_closure_sha256": V1_IMPLEMENTATION_CLOSURE_SHA256,
        "v1_failed_predecessor_expectation_sha256": identity.predecessor.sha256,
        "v1_failed_predecessor_binding_sha256": predecessor.sha256,
        "theta_raw_t4_hash_law": THETA_RAW_HASH_LAW,
        "theta_raw_t4_hash_law_expression": THETA_RAW_HASH_LAW_TEXT,
    }


def _validate_v2_binding(
    value: object, *, identity: SourceExecutionV2Identity, predecessor: V1FailedPredecessorBinding,
) -> None:
    _require(
        isinstance(value, Mapping) and dict(value) == _v2_binding(identity, predecessor),
        "V2 durable predecessor/closure/hash-law binding drift",
    )


def _attempt_payload(identity: SourceExecutionV2Identity, predecessor: V1FailedPredecessorBinding) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_attempt_v2",
        "cell": CELL,
        "status": "ATTEMPT_RESERVED",
        "identity": identity.payload(),
        "v1_failed_predecessor_binding": predecessor.payload(),
        "v2_binding": _v2_binding(identity, predecessor),
        "source_only": True,
        "source_resolved_or_opened": False,
        "checkpoint_opened": False,
        "cuda_initialized": False,
        "target_optimizer_backward_update": 0,
        "within_external_formal_target_forbidden": True,
    }


def _launch_payload(
    identity: SourceExecutionV2Identity,
    predecessor: V1FailedPredecessorBinding,
    attempt_sha256: str,
    preflight: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_launch_v2",
        "cell": CELL,
        "status": "LAUNCHED",
        "identity": identity.payload(),
        "attempt_sha256": _sha(attempt_sha256, "V2 launch attempt SHA"),
        "preflight": dict(preflight),
        "v1_failed_predecessor_binding": predecessor.payload(),
        "v2_binding": _v2_binding(identity, predecessor),
        "launch_closure_sha256": identity.closure["closure_sha256"],
        "source_only": True,
        "target_optimizer_backward_update": 0,
    }


def _validate_theta_raw_proof(
    row: object, *, expected_session: str,
) -> None:
    _require(isinstance(row, Mapping), "V2 theta raw proof must be a mapping")
    shape = row.get("raw_t4_shape")
    _require(
        row.get("schema") == "causal_dual_memory_cell_d_theta_raw_proof_v2"
        and row.get("session") == expected_session
        and row.get("hash_law") == THETA_RAW_HASH_LAW
        and row.get("hash_law_expression") == THETA_RAW_HASH_LAW_TEXT
        and isinstance(shape, list) and len(shape) == 2 and type(shape[0]) is int and shape[0] > 0
        and shape[1] == 4
        and row.get("raw_t4_dtype") == "float32"
        and row.get("raw_t4_all_finite") is True
        and row.get("theta_atan2_float64_bitwise_equal") is True
        and row.get("validity_raw_m_gt_modulation_eps_bitwise_equal") is True
        and row.get("canonical_unit_order_exact") is True
        and type(row.get("known_invalid_unit_count")) is int
        and row["known_invalid_unit_count"] == v1.KNOWN_THETA_INVALID_UNIT_COUNTS.get(expected_session, 0),
        "V2 theta raw proof semantic topology drift",
    )
    for key in (
        "raw_t4_sha256", "sealed_raw_t4_sha256", "theta_float64_sha256",
        "sealed_theta_float64_sha256", "valid_mask_sha256", "sealed_valid_mask_sha256",
        "canonical_unit_order_sha256",
    ):
        _require(_sha(row.get(key), f"V2 theta raw proof {key}") == row.get(key),
                 "V2 theta raw proof digest drift")
    _require(
        row["raw_t4_sha256"] == row["sealed_raw_t4_sha256"]
        and row["theta_float64_sha256"] == row["sealed_theta_float64_sha256"]
        and row["valid_mask_sha256"] == row["sealed_valid_mask_sha256"],
        "V2 theta raw proof value/authority mismatch",
    )


def _enrich_payload(
    value: Mapping[str, object], *, identity: SourceExecutionV2Identity,
    predecessor: V1FailedPredecessorBinding,
) -> dict[str, object]:
    result = dict(value)
    result["v2_binding"] = _v2_binding(identity, predecessor)
    result["v1_failed_predecessor_binding"] = predecessor.payload()
    return result


def _validate_source_authority_v2(
    value: Mapping[str, object], *, identity: SourceExecutionV2Identity,
    predecessor: V1FailedPredecessorBinding, flags: v1.RuntimeFlags,
) -> None:
    try:
        v1._validate_source_authority(value, identity.inherited_v1_identity, flags)
    except v1.SourceExecutionError as error:
        raise SourceExecutionV2Error(str(error)) from error
    _validate_v2_binding(value.get("v2_binding"), identity=identity, predecessor=predecessor)
    _require(value.get("v1_failed_predecessor_binding") == predecessor.payload(),
             "V2 source authority predecessor binding drift")
    proofs = value.get("v2_theta_raw_proofs")
    expected_sessions = (
        (v1.SOURCE_SMOKE_SESSION,)
        if identity.spec.kind == "source_smoke" else tuple(identity.strict_train_roster)
    )
    _require(isinstance(proofs, list) and tuple(
        item.get("session") if isinstance(item, Mapping) else None for item in proofs
    ) == expected_sessions, "V2 theta raw proof session topology drift")
    sessions = value.get("sessions")
    _require(isinstance(sessions, list) and len(sessions) == len(expected_sessions),
             "V2 theta proof/source-authority session topology drift")
    for session, proof, source_row in zip(expected_sessions, proofs, sessions, strict=True):
        _validate_theta_raw_proof(proof, expected_session=session)
        _require(
            isinstance(source_row, Mapping)
            and proof["raw_t4_shape"] == [source_row.get("total_unit_count"), 4]
            and proof["known_invalid_unit_count"] == source_row.get("invalid_unit_count")
            and proof["valid_mask_sha256"] == source_row.get("valid_mask_sha256")
            and proof["valid_mask_sha256"] == source_row.get("theta_valid_mask_sha256")
            and proof["canonical_unit_order_sha256"] == source_row.get("raw_t4_channel_order_sha256"),
            "V2 theta proof/source-authority topology or unit-order cross-binding drift",
        )


def _validate_evidence_v2(
    value: Mapping[str, object], *, identity: SourceExecutionV2Identity,
    predecessor: V1FailedPredecessorBinding,
) -> None:
    _validate_v2_binding(value.get("v2_binding"), identity=identity, predecessor=predecessor)
    _require(value.get("v1_failed_predecessor_binding") == predecessor.payload(),
             "V2 evidence predecessor binding drift")


def _terminal_payload(
    identity: SourceExecutionV2Identity,
    *,
    predecessor: V1FailedPredecessorBinding,
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
        raise SourceExecutionV2Error(str(error)) from error
    _require(status in {"PASS_SOURCE_CONSTRUCTIBLE", "STOP_SOURCE_B8_CONSTRUCTIBILITY", "SMOKE_COMPLETED"},
             "V2 terminal status drift")
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_terminal_v2",
        "cell": CELL,
        "status": status,
        "identity": identity.payload(),
        "attempt_sha256": _sha(attempt_sha256, "V2 terminal attempt SHA"),
        "launch_sha256": _sha(launch_sha256, "V2 terminal launch SHA"),
        "source_authority_sha256": _sha(source_authority_sha256, "V2 terminal source authority SHA"),
        "evidence_sha256s": {name: _sha(digest, f"V2 terminal {name} SHA")
                               for name, digest in evidence_sha256s.items()},
        "resources": dict(resources),
        "v1_failed_predecessor_binding": predecessor.payload(),
        "v2_binding": _v2_binding(identity, predecessor),
        "launch_closure_sha256": identity.closure["closure_sha256"],
        "final_closure_sha256": identity.closure["closure_sha256"],
        "source_only": True,
        "target_optimizer_backward_update": 0,
    }


def _failure_payload(
    identity: SourceExecutionV2Identity,
    *,
    predecessor: V1FailedPredecessorBinding,
    attempt_sha256: str,
    launch_sha256: str | None,
    source_authority_sha256: str | None,
    flags: v1.RuntimeFlags,
    error: BaseException,
) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_failure_v2",
        "cell": CELL,
        "status": "FAILED",
        "identity": identity.payload(),
        "attempt_sha256": _sha(attempt_sha256, "V2 failure attempt SHA"),
        "launch_sha256": None if launch_sha256 is None else _sha(launch_sha256, "V2 failure launch SHA"),
        "source_authority_sha256": None if source_authority_sha256 is None else _sha(
            source_authority_sha256, "V2 failure source-authority SHA",
        ),
        "stage": flags.stage,
        "error_class": type(error).__name__,
        "error_sha256": sha256_bytes(repr(error).encode("utf-8")),
        "flags": flags.payload(),
        "v1_failed_predecessor_binding": predecessor.payload(),
        "v2_binding": _v2_binding(identity, predecessor),
        "terminal_published": False,
        "source_only": True,
    }


def _validate_pre_execution(
    root: Path, identity: SourceExecutionV2Identity, capability: object,
    environ: Mapping[str, str] | None,
) -> tuple[SourceExecutionV2Capability, V1FailedPredecessorBinding]:
    approved = require_execution_capability(capability, identity)
    validate_identity_current(Path(root), identity)
    predecessor = validate_v1_failed_predecessor(Path(root))
    assert_prospective_root_fresh(Path(root), identity.spec)
    try:
        v1.validate_selected_device_environment(identity.selected_device, environ)
    except v1.SourceExecutionError as error:
        raise SourceExecutionV2Error(str(error)) from error
    return approved, predecessor


def execute_authorized(
    root: Path,
    *,
    identity: SourceExecutionV2Identity,
    capability: object,
    backend: v1.SourceExecutionBackend,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Thin V2 receipt wrapper over V1 validators, artifacts, and backend API.

    The physical parser/executor is not copied: ``backend`` remains the V1
    protocol and receives the wrapped immutable V1 identity.  This wrapper is
    limited to the V2 durable lineage/hash-law envelope that V1 could not have
    known when its immutable result failed.
    """
    _approved, predecessor = _validate_pre_execution(Path(root), identity, capability, environ)
    flags = v1.RuntimeFlags(stage="preflight")
    preflight = backend.preflight(root=Path(root), identity=identity.inherited_v1_identity, flags=flags)
    _require(isinstance(preflight, Mapping)
             and preflight.get("source_resolved_or_opened") is False
             and preflight.get("checkpoint_opened") is False
             and preflight.get("cuda_initialized") is False,
             "V2 source-free preflight boundary drift")
    artifact = v1.reserve_artifact_root(
        Path(root), spec=identity.spec, roster=identity.strict_train_roster,
    )
    runtime: Any | None = None
    attempt_sha: str | None = None
    launch_sha: str | None = None
    authority_sha: str | None = None
    try:
        flags.stage = "attempt"
        attempt_sha = artifact.publish_json_pair("attempt.json", _attempt_payload(identity, predecessor))
        flags.stage = "launch"
        launch_sha = artifact.publish_json_pair(
            "launch.json", _launch_payload(identity, predecessor, attempt_sha, preflight),
        )
        flags.stage = "prepare"
        runtime = backend.prepare(root=Path(root), identity=identity.inherited_v1_identity, flags=flags)
        flags.stage = "source_authority"
        authority = _enrich_payload(
            backend.source_authority(runtime, identity=identity.inherited_v1_identity, flags=flags),
            identity=identity, predecessor=predecessor,
        )
        _validate_source_authority_v2(authority, identity=identity, predecessor=predecessor, flags=flags)
        authority_sha = artifact.publish_json_pair("source_authority.json", authority)
        evidence_sha: dict[str, str] = {}
        if identity.spec.kind == "source_smoke":
            flags.stage = "smoke"
            smoke = _enrich_payload(
                backend.run_smoke(runtime, identity=identity.inherited_v1_identity, flags=flags),
                identity=identity, predecessor=predecessor,
            )
            try:
                v1._validate_smoke(smoke, identity.inherited_v1_identity)
            except v1.SourceExecutionError as error:
                raise SourceExecutionV2Error(str(error)) from error
            _validate_evidence_v2(smoke, identity=identity, predecessor=predecessor)
            evidence_sha["smoke.json"] = artifact.publish_json_pair("smoke.json", smoke)
            final_status = "SMOKE_COMPLETED"
        else:
            final_status = "PASS_SOURCE_CONSTRUCTIBLE"
            for budget in v1.FAIL_FAST_BUDGET_ORDER:
                flags.stage = f"budget_m{budget}"
                raw_rows = tuple(backend.run_budget(
                    runtime, budget=budget, identity=identity.inherited_v1_identity, flags=flags,
                ))
                rows = tuple(_enrich_payload(row, identity=identity, predecessor=predecessor)
                             for row in raw_rows)
                try:
                    aggregate = v1._aggregate_budget(rows, budget=budget, roster=identity.strict_train_roster)
                except v1.SourceExecutionError as error:
                    raise SourceExecutionV2Error(str(error)) from error
                for session, row in zip(identity.strict_train_roster, rows, strict=True):
                    _validate_evidence_v2(row, identity=identity, predecessor=predecessor)
                    name = f"budget_m{budget}__{session}.json"
                    evidence_sha[name] = artifact.publish_json_pair(name, row)
                aggregate = _enrich_payload(aggregate, identity=identity, predecessor=predecessor)
                _validate_evidence_v2(aggregate, identity=identity, predecessor=predecessor)
                aggregate_name = f"budget_m{budget}_aggregate.json"
                evidence_sha[aggregate_name] = artifact.publish_json_pair(aggregate_name, aggregate)
                if aggregate["breadth_pass"] is False:
                    final_status = "STOP_SOURCE_B8_CONSTRUCTIBILITY"
                    break
        flags.stage = "final_revalidation"
        validate_identity_current(Path(root), identity)
        # Re-read the frozen V1 failure graph after physical forwards, too;
        # its immutable lineage must remain exactly what authorized V2.
        final_predecessor = validate_v1_failed_predecessor(Path(root))
        _require(final_predecessor.payload() == predecessor.payload(),
                 "V1 predecessor identity changed during V2 execution")
        backend.revalidate(root=Path(root), identity=identity.inherited_v1_identity, flags=flags)
        resources = dict(backend.resources(runtime, flags=flags))
        flags.stage = "terminal"
        terminal = _terminal_payload(
            identity, predecessor=predecessor, attempt_sha256=attempt_sha, launch_sha256=launch_sha,
            source_authority_sha256=authority_sha, evidence_sha256s=evidence_sha,
            status=final_status, resources=resources,
        )
        terminal_sha = artifact.publish_json_pair("terminal.json", terminal)
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
            try:
                artifact.publish_json_pair(
                    "failure.json",
                    _failure_payload(
                        identity, predecessor=predecessor, attempt_sha256=attempt_sha,
                        launch_sha256=launch_sha, source_authority_sha256=authority_sha,
                        flags=flags, error=error,
                    ),
                )
            except BaseException:
                pass
        raise
    finally:
        try:
            backend.close(runtime)
        finally:
            artifact.close()


def execute_reviewed_physical(
    root: Path,
    *,
    identity: SourceExecutionV2Identity,
    capability: object,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Only reviewed V2 physical launch entry; public CLI cannot reach it."""
    approved, _predecessor = _validate_pre_execution(Path(root), identity, capability, environ)
    source_data = approved.inherited_v1_capability.source_data_root
    _require(source_data is not None
             and source_data.strict_train_roster_sha256 == v1.roster_sha256(identity.strict_train_roster),
             "reviewed V2 physical route needs exact inherited strict source-data capability")
    from .source_execute_physical_v2 import build_reviewed_physical_backend

    backend = build_reviewed_physical_backend(
        root=Path(root), source_data=source_data, selected_device=identity.selected_device,
    )
    return execute_authorized(
        Path(root), identity=identity, capability=approved, backend=backend, environ=environ,
    )


def dry_plan(root: Path | None = None) -> dict[str, object]:
    """Static, side-effect-free public plan; root is optional for code audit only."""
    result: dict[str, object] = {
        "cell": CELL,
        "phase": "source_execution_v2_theta_raw_bytes_successor",
        "workorder_sha256": WORKORDER_SHA256,
        "accepted_v1_implementation_closure_sha256": V1_IMPLEMENTATION_CLOSURE_SHA256,
        "v1_failed_predecessor": V1_FAILED_PREDECESSOR.payload(),
        "theta_raw_t4_hash_law": THETA_RAW_HASH_LAW,
        "source_smoke_root_relative": SOURCE_SMOKE_ROOT_RELATIVE,
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
