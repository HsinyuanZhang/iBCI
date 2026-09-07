"""Fail-closed additive Phase-E V2 wrapper for TF-SR seed 42.

V1 is immutable and remains the only implementation of the physical matched
scorer.  V2 changes a failed deployment binding only: it records and
revalidates V1's failed attempt, requires the canonical sub-C/sub-M roots,
and emits a new authority/score lineage.  There is deliberately no copied
model, parser, metric, or target-data implementation in this module.

The module has standard-library imports only at import time.  The frozen V1
scorer is loaded through the same static-package pattern used by its public
CLI; its Torch-bearing imports remain lazy behind V1's physical backend.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Protocol, Sequence


CELL = "TFSR_B3ST4_DDROP_SEED42"
PHASE = "TFSR_PHASE_E_MATCHED_SCORE_V2"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_TFSR_PHASE_E_MATCHED_SCORE_V2_20260823.md"
WORKORDER_SHA256 = "610cfa1c92e94f61f50b3f11ca5084902fee754de3f308c84e544455588bf8a9"
AUTHORITY_ROOT_RELATIVE = "tfpd_exploration/results/tfsr_b3st4_ddrop_seed42_score_authority_v2"
SCORE_ROOT_RELATIVE = "tfpd_exploration/results/tfsr_b3st4_ddrop_seed42_matched_score_v2"
AUTHORITY_TOPOLOGY = ("official_preflight.json", "root_authorization.json")
SCORE_TOPOLOGY = ("attempt.json", "input_authority.json", "score.json", "terminal.json", "failure.json")

# These are paths, not convenience prefixes.  The corrected roots are part of
# V2's signed deployment contract and must compare byte-for-byte before a V2
# result root is reserved or a data-path factory is constructed.
SUBC_DATA_ROOT = "/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-C"
SUBM_DATA_ROOT = "/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-M"

V1_ATTEMPT_SHA256 = "dc5255069e1ede930b511df5280a5e596c017c8d61ca34ca59880557c0d3be3a"
V1_FAILURE_SHA256 = "c4054cf120a631332f01cf019a4d51b08a0f0ff09d882c89e2ac28d5761c6404"

_V1_MODULE_NAME = "_tfsr_phase_e_v1_core_for_v2"
_V1: ModuleType | None = None


def _repository_root_from_module() -> Path:
    # .../tfpd_exploration/src/tfsr_b3st4_ddrop_seed42_score_v2/score_v2.py
    return Path(__file__).resolve().parents[3]


def _v1() -> ModuleType:
    """Load frozen V1 as a route-local static package without its __init__.

    This is import machinery, not a monkeypatch: no V1 function, module
    global, scorer contract, or backend method is replaced.  It avoids the
    Torch-bearing V1 package initializer on V2's dry route exactly as V1's
    own reviewed CLI does.
    """
    global _V1
    if _V1 is not None:
        return _V1
    existing = sys.modules.get(_V1_MODULE_NAME + ".score")
    if isinstance(existing, ModuleType):
        _V1 = existing
        return existing
    root = _repository_root_from_module()
    package = ModuleType(_V1_MODULE_NAME)
    package.__path__ = [str(root / "tfpd_exploration/src/tfsr_b3st4_ddrop_v1")]
    sys.modules[_V1_MODULE_NAME] = package
    source = root / "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/score.py"
    spec = importlib.util.spec_from_file_location(_V1_MODULE_NAME + ".score", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to statically load frozen Phase-E V1 scorer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _V1 = module
    return module


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True).encode("utf-8") + b"\n"


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise V2Error(f"{label} must be lowercase SHA-256")
    return value


class V2Error(RuntimeError):
    """A V2 provenance, deployment, or lifecycle boundary violation."""


def _safe_relative(value: str) -> str:
    if not isinstance(value, str) or not value or value.startswith("/"):
        raise V2Error("relative path is invalid")
    path = Path(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise V2Error("relative path traversal")
    return value


def _directory_identity(path: Path) -> tuple[int, int]:
    try:
        info = os.lstat(path)
    except OSError as error:
        raise V2Error("required directory is absent") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise V2Error("required directory is not a canonical directory")
    return (info.st_dev, info.st_ino)


def _read_all(fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        block = os.read(fd, 1 << 20)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


def _read_regular_from_fd(directory_fd: int, name: str, *, expected_mode: int = 0o444) -> bytes:
    if not isinstance(name, str) or "/" in name or name in {"", ".", ".."}:
        raise V2Error("immutable leaf name drift")
    try:
        leaf = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    except OSError as error:
        raise V2Error("immutable predecessor leaf missing") from error
    try:
        info = os.fstat(leaf)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != expected_mode:
            raise V2Error("immutable predecessor leaf type/mode drift")
        return _read_all(leaf)
    finally:
        os.close(leaf)


def _read_exact_failed_v1_pairs(directory: Path, *, attempt_sha256: str, failure_sha256: str) -> tuple[Mapping[str, Any], Mapping[str, Any], tuple[int, int]]:
    """Descriptor-read the entire failed V1 score topology under one FD."""
    expected = {"attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256"}
    initial = _directory_identity(directory)
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != initial:
            raise V2Error("V1 failed-score root changed while opening")
        names = set(os.listdir(fd))
        if names != expected:
            raise V2Error("V1 failed-score topology must contain exactly immutable attempt/failure pairs")
        bodies: dict[str, bytes] = {}
        for name, digest in (("attempt.json", attempt_sha256), ("failure.json", failure_sha256)):
            body = _read_regular_from_fd(fd, name)
            if _sha(body) != digest:
                raise V2Error("V1 predecessor body SHA drift")
            sidecar = _read_regular_from_fd(fd, name + ".sha256")
            if sidecar != f"{digest}  {name}\n".encode("ascii"):
                raise V2Error("V1 predecessor sidecar drift")
            bodies[name] = body
        try:
            attempt = json.loads(bodies["attempt.json"])
            failure = json.loads(bodies["failure.json"])
        except (TypeError, json.JSONDecodeError) as error:
            raise V2Error("V1 predecessor JSON malformed") from error
        if not isinstance(attempt, Mapping) or not isinstance(failure, Mapping):
            raise V2Error("V1 predecessor JSON root drift")
        final = _directory_identity(directory)
        if final != initial:
            raise V2Error("V1 failed-score root identity changed during read")
        return attempt, failure, initial
    finally:
        os.close(fd)


@dataclass(frozen=True)
class V1FailureExpectation:
    attempt_sha256: str = V1_ATTEMPT_SHA256
    failure_sha256: str = V1_FAILURE_SHA256

    def __post_init__(self) -> None:
        _require_sha(self.attempt_sha256, "V1 attempt")
        _require_sha(self.failure_sha256, "V1 failure")

    def payload(self) -> dict[str, str]:
        return {"attempt_sha256": self.attempt_sha256, "failure_sha256": self.failure_sha256}


@dataclass(frozen=True)
class V1FailureLineage:
    """A fully validated V1 authority plus its immutable failed lifecycle."""

    v1_authorization: Mapping[str, object]
    v1_identity: Mapping[str, object]
    failed_score_root_relative: str
    expectation: V1FailureExpectation
    failed_score_root_identity: tuple[int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.v1_authorization, Mapping) or not isinstance(self.v1_identity, Mapping):
            raise ValueError("V1 lineage authority/identity must be mappings")
        if len(self.failed_score_root_identity) != 2 or any(type(item) is not int or item < 0 for item in self.failed_score_root_identity):
            raise ValueError("V1 failed-score root identity drift")
        _safe_relative(self.failed_score_root_relative)

    def payload(self) -> dict[str, object]:
        return {
            "v1_authorization": json.loads(json.dumps(self.v1_authorization, sort_keys=True)),
            "v1_identity": json.loads(json.dumps(self.v1_identity, sort_keys=True)),
            "failed_score_root_relative": self.failed_score_root_relative,
            "v1_failed_pairs": self.expectation.payload(),
            "failed_score_root_identity": list(self.failed_score_root_identity),
        }


def _expected_v1_identity(v1: ModuleType, fixed: Mapping[str, Any], training: Any, authorization: Any) -> Any:
    return v1.ScoreIdentity(
        fixed_authorities={name: material.binding() for name, material in fixed.items()},
        training_terminal_sha256=training.terminal_sha256,
        training_swa_sha256=training.swa_sha256,
        training_swa_state_digest=training.swa_state_digest,
        launch_closure=authorization.expected_closure,
        phase_e_authorization=authorization.payload(),
    )


def _validate_v1_failure_semantics(v1: ModuleType, attempt: Mapping[str, Any], failure: Mapping[str, Any], identity: Any) -> None:
    try:
        v1.validate_attempt_payload(attempt, identity)
        v1.validate_failure_payload(failure)
    except Exception as error:
        raise V2Error("V1 failed lifecycle schema/identity drift") from error
    if failure.get("stage") != "resolve_inputs":
        raise V2Error("V1 predecessor is not the expected resolve-inputs failure")
    if failure.get("resolved") != {"source": False, "within": True, "external": True, "formal": False}:
        raise V2Error("V1 failure input-resolution boundary drift")
    if failure.get("opened") != {"source": False, "within": False, "external": False, "formal": False}:
        raise V2Error("V1 failure opened-data boundary drift")
    if failure.get("backward_calls") != 0 or failure.get("optimizer_calls") != 0:
        raise V2Error("V1 failed attempt crossed update boundary")
    forward = failure.get("forward_calls")
    try:
        checked = v1.ScoreFlags.validate_forward_calls(forward)
    except Exception as error:
        raise V2Error("V1 failure forward accounting drift") from error
    if any(count != 0 for surfaces in checked.values() for modes in surfaces.values() for count in modes.values()):
        raise V2Error("V1 failed attempt made a forward call")
    if failure.get("terminal_published") is not False:
        raise V2Error("V1 failure illegally published terminal")


def validate_v1_failed_lineage(
    root: Path,
    *,
    fixed_authorities: Mapping[str, Any] | None = None,
    training: Any | None = None,
    authorization: Any | None = None,
    expectation: V1FailureExpectation = V1FailureExpectation(),
) -> V1FailureLineage:
    """Reload V1 authority and failed pairs before any V2 reservation.

    The optional arguments are an injection seam for no-data tests only.  The
    production path obtains every one from frozen V1 validators; it never
    trusts a caller-supplied V1 identity or failure payload.
    """
    v1 = _v1()
    fixed = fixed_authorities if fixed_authorities is not None else v1.verify_fixed_authorities(root)
    train = training if training is not None else v1.validate_phase_d_training_terminal(root)
    auth = authorization if authorization is not None else v1.verify_phase_e_authorization(root, train, fixed)
    identity = _expected_v1_identity(v1, fixed, train, auth)
    directory = root.absolute() / v1.SCORE_ROOT_RELATIVE
    attempt, failure, directory_identity = _read_exact_failed_v1_pairs(
        directory,
        attempt_sha256=expectation.attempt_sha256,
        failure_sha256=expectation.failure_sha256,
    )
    _validate_v1_failure_semantics(v1, attempt, failure, identity)
    return V1FailureLineage(
        v1_authorization=auth.payload(), v1_identity=identity.payload(),
        failed_score_root_relative=v1.SCORE_ROOT_RELATIVE, expectation=expectation,
        failed_score_root_identity=directory_identity,
    )


def _validate_production_lineage(lineage: V1FailureLineage) -> None:
    if lineage.failed_score_root_relative != _v1().SCORE_ROOT_RELATIVE:
        raise V2Error("V2 lineage references a noncanonical V1 score root")
    if lineage.expectation != V1FailureExpectation():
        raise V2Error("V2 lineage does not bind the exact V1 failed pairs")


class RootPublicationCapability:
    __slots__ = ("_seal",)

    def __init__(self, seal: object) -> None:
        if seal is not _ROOT_SEAL:
            raise TypeError("V2 root publication capability is internal")
        self._seal = seal


class ExecutionCapability:
    __slots__ = ("authorization", "_seal")

    def __init__(self, authorization: "V2Authorization", seal: object) -> None:
        if seal is not _EXECUTION_SEAL:
            raise TypeError("V2 execution capability is internal")
        self.authorization = authorization
        self._seal = seal


_ROOT_SEAL = object()
_EXECUTION_SEAL = object()


def issue_root_publication_capability() -> RootPublicationCapability:
    """Root-reviewed preflight publication token; it does not authorize score."""
    return RootPublicationCapability(_ROOT_SEAL)


def _issue_execution_capability(authorization: "V2Authorization") -> ExecutionCapability:
    return ExecutionCapability(authorization, _EXECUTION_SEAL)


def _require_execution_capability(capability: object, identity: "V2ScoreIdentity") -> None:
    if not isinstance(capability, ExecutionCapability) or capability._seal is not _EXECUTION_SEAL:
        raise V2Error("V2 score requires a verified in-process execution capability")
    if capability.authorization.payload() != identity.v2_authorization:
        raise V2Error("V2 execution capability/identity mismatch")


def _v2_closure_paths() -> tuple[str, ...]:
    v1_paths = tuple(_v1().PHASE_E_CLOSURE)
    own = (
        WORKORDER_RELATIVE,
        "tfpd_exploration/src/tfsr_b3st4_ddrop_seed42_score_v2/__init__.py",
        "tfpd_exploration/src/tfsr_b3st4_ddrop_seed42_score_v2/score_v2.py",
        "tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_seed42_score_v2.py",
        "tfpd_exploration/tests/test_tfsr_b3st4_ddrop_seed42_score_v2.py",
    )
    paths = own + v1_paths
    if len(paths) != len(set(paths)):
        raise V2Error("V2 explicit closure contains duplicate paths")
    return paths


V2_CLOSURE = _v2_closure_paths()


def phase_e_v2_closure(root: Path) -> dict[str, object]:
    v1 = _v1()
    by_path: dict[str, str] = {}
    for relative in V2_CLOSURE:
        body, _ = v1._canonical_regular_bytes(root, relative, expected_mode=0o664)
        by_path[relative] = v1._sha(body)
    if by_path.get(WORKORDER_RELATIVE) != WORKORDER_SHA256:
        raise V2Error("V2 workorder SHA drift")
    # V1 itself is closure-bound as a complete physical dependency.  The
    # nested map makes a missing V1 runtime leaf visible in V2 provenance.
    v1_closure = v1.phase_e_closure(root)
    payload = {"files": by_path, "v1_phase_e_closure": v1_closure}
    return {
        "paths": list(V2_CLOSURE),
        "sha256_by_path": by_path,
        "v1_phase_e_closure": v1_closure,
        "closure_sha256": _sha(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")),
    }


def _validate_v2_closure(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"paths", "sha256_by_path", "v1_phase_e_closure", "closure_sha256"}:
        raise V2Error("V2 closure schema drift")
    if value.get("paths") != list(V2_CLOSURE) or not isinstance(value.get("sha256_by_path"), Mapping):
        raise V2Error("V2 closure path topology drift")
    by_path = value["sha256_by_path"]
    if set(by_path) != set(V2_CLOSURE) or any(not isinstance(digest, str) or len(digest) != 64 for digest in by_path.values()):
        raise V2Error("V2 closure leaf digest drift")
    if by_path.get(WORKORDER_RELATIVE) != WORKORDER_SHA256:
        raise V2Error("V2 workorder closure binding drift")
    v1 = _v1()
    try:
        checked_v1 = v1._validate_phase_e_closure_payload(value.get("v1_phase_e_closure"))
    except Exception as error:
        raise V2Error("nested V1 Phase-E closure invalid") from error
    payload = {"files": dict(by_path), "v1_phase_e_closure": checked_v1}
    expected = _sha(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    if value.get("closure_sha256") != expected:
        raise V2Error("V2 closure aggregate drift")
    return {
        "paths": list(V2_CLOSURE), "sha256_by_path": dict(by_path),
        "v1_phase_e_closure": checked_v1, "closure_sha256": expected,
    }


def _correct_data_roots(environment: Mapping[str, str] | None = None) -> tuple[Path, Path]:
    env = os.environ if environment is None else environment
    subc = env.get("SUBC_DATA_ROOT")
    subm = env.get("SUBM_DATA_ROOT")
    if subc != SUBC_DATA_ROOT or subm != SUBM_DATA_ROOT:
        raise V2Error("V2 requires exact canonical SUBC_DATA_ROOT and SUBM_DATA_ROOT before reservation")
    return Path(SUBC_DATA_ROOT), Path(SUBM_DATA_ROOT)


@dataclass(frozen=True)
class V2ScoreIdentity:
    v1_identity: Mapping[str, object]
    v1_failed_lineage: Mapping[str, object]
    v2_authorization: Mapping[str, object]
    launch_closure: Mapping[str, object]

    def __post_init__(self) -> None:
        if not all(isinstance(value, Mapping) for value in (
            self.v1_identity, self.v1_failed_lineage, self.v2_authorization, self.launch_closure,
        )):
            raise ValueError("V2 identity fields must be mappings")

    def payload(self) -> dict[str, object]:
        return {
            "v1_identity": json.loads(json.dumps(self.v1_identity, sort_keys=True)),
            "v1_failed_lineage": json.loads(json.dumps(self.v1_failed_lineage, sort_keys=True)),
            "v2_authorization": json.loads(json.dumps(self.v2_authorization, sort_keys=True)),
            "launch_closure": json.loads(json.dumps(self.launch_closure, sort_keys=True)),
        }


@dataclass(frozen=True)
class V2Authorization:
    preflight_sha256: str
    root_authorization_sha256: str
    preflight: Mapping[str, object]
    root_authorization: Mapping[str, object]
    expected_closure: Mapping[str, object]

    def __post_init__(self) -> None:
        _require_sha(self.preflight_sha256, "V2 preflight")
        _require_sha(self.root_authorization_sha256, "V2 root authorization")

    def payload(self) -> dict[str, object]:
        return {
            "preflight_sha256": self.preflight_sha256,
            "root_authorization_sha256": self.root_authorization_sha256,
            "expected_closure": json.loads(json.dumps(self.expected_closure, sort_keys=True)),
        }


def _v2_authority_parent(root: Path) -> tuple[Path, str]:
    path = root.absolute() / _safe_relative(AUTHORITY_ROOT_RELATIVE)
    return path.parent, path.name


def _v2_score_parent(root: Path) -> tuple[Path, str]:
    path = root.absolute() / _safe_relative(SCORE_ROOT_RELATIVE)
    return path.parent, path.name


def reserve_v2_authority_artifact(
    root: Path,
    capability: RootPublicationCapability,
    *,
    lineage_loader: Callable[[Path], V1FailureLineage] = validate_v1_failed_lineage,
) -> Any:
    """Verify V1 before the first V2 authority-root inode is created."""
    if not isinstance(capability, RootPublicationCapability) or capability._seal is not _ROOT_SEAL:
        raise V2Error("only root may reserve V2 authority root")
    lineage = lineage_loader(root)
    _validate_production_lineage(lineage)
    parent, name = _v2_authority_parent(root)
    return _v1().reserve_artifact_root(parent, name, AUTHORITY_TOPOLOGY)


def build_target_free_preflight_v2(
    root: Path,
    *,
    lineage_loader: Callable[[Path], V1FailureLineage] = validate_v1_failed_lineage,
    closure_loader: Callable[[Path], Mapping[str, object]] = phase_e_v2_closure,
) -> dict[str, object]:
    """Build a V2 authority payload without resolving any evaluation path."""
    lineage = lineage_loader(root)
    _validate_production_lineage(lineage)
    closure = _validate_v2_closure(closure_loader(root))
    v1 = _v1()
    return {
        "schema": "tfsr_phase_e_v2_target_free_preflight_v1",
        "status": "PREFLIGHT_ACCEPTED",
        "cell": CELL,
        "phase": PHASE,
        "v1_score_spec": v1.PUBLIC_SPEC.payload(),
        "v1_failure_lineage": lineage.payload(),
        "v2_closure": closure,
        "authority_root_relative": AUTHORITY_ROOT_RELATIVE,
        "score_root_relative": SCORE_ROOT_RELATIVE,
        "deployment_data_roots": {"SUBC_DATA_ROOT": SUBC_DATA_ROOT, "SUBM_DATA_ROOT": SUBM_DATA_ROOT},
        "semantic_change": "correct_canonical_data_root_binding_only",
        "target_free": True,
    }


def validate_target_free_preflight_v2(
    value: Mapping[str, Any],
    *,
    lineage: V1FailureLineage,
    closure: Mapping[str, object],
) -> dict[str, object]:
    expected = {
        "schema", "status", "cell", "phase", "v1_score_spec", "v1_failure_lineage", "v2_closure",
        "authority_root_relative", "score_root_relative", "deployment_data_roots", "semantic_change", "target_free",
    }
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "tfsr_phase_e_v2_target_free_preflight_v1"
            or value.get("status") != "PREFLIGHT_ACCEPTED" or value.get("cell") != CELL
            or value.get("phase") != PHASE or value.get("v1_score_spec") != _v1().PUBLIC_SPEC.payload()
            or value.get("v1_failure_lineage") != lineage.payload()
            or value.get("v2_closure") != _validate_v2_closure(closure)
            or value.get("authority_root_relative") != AUTHORITY_ROOT_RELATIVE
            or value.get("score_root_relative") != SCORE_ROOT_RELATIVE
            or value.get("deployment_data_roots") != {"SUBC_DATA_ROOT": SUBC_DATA_ROOT, "SUBM_DATA_ROOT": SUBM_DATA_ROOT}
            or value.get("semantic_change") != "correct_canonical_data_root_binding_only"
            or value.get("target_free") is not True):
        raise V2Error("V2 target-free preflight schema/binding drift")
    return dict(value)


def build_root_authorization_v2(*, preflight_sha256: str, preflight: Mapping[str, object]) -> dict[str, object]:
    _require_sha(preflight_sha256, "V2 preflight")
    if not isinstance(preflight, Mapping):
        raise V2Error("V2 authorization requires mapping preflight")
    return {
        "schema": "tfsr_phase_e_v2_root_authorization_v1",
        "status": "ROOT_AUTHORIZED",
        "cell": CELL,
        "phase": PHASE,
        "official_preflight_sha256": preflight_sha256,
        "authority_root_relative": AUTHORITY_ROOT_RELATIVE,
        "score_root_relative": SCORE_ROOT_RELATIVE,
        "v2_closure": preflight.get("v2_closure"),
        "v1_failure_lineage": preflight.get("v1_failure_lineage"),
        "deployment_data_roots": preflight.get("deployment_data_roots"),
        "target_free_preflight_required": True,
    }


def validate_root_authorization_v2(
    value: Mapping[str, Any], *, preflight_sha256: str, preflight: Mapping[str, object],
) -> dict[str, object]:
    _require_sha(preflight_sha256, "V2 preflight")
    expected = {
        "schema", "status", "cell", "phase", "official_preflight_sha256", "authority_root_relative",
        "score_root_relative", "v2_closure", "v1_failure_lineage", "deployment_data_roots",
        "target_free_preflight_required",
    }
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "tfsr_phase_e_v2_root_authorization_v1"
            or value.get("status") != "ROOT_AUTHORIZED" or value.get("cell") != CELL
            or value.get("phase") != PHASE or value.get("official_preflight_sha256") != preflight_sha256
            or value.get("authority_root_relative") != AUTHORITY_ROOT_RELATIVE
            or value.get("score_root_relative") != SCORE_ROOT_RELATIVE
            or value.get("v2_closure") != preflight.get("v2_closure")
            or value.get("v1_failure_lineage") != preflight.get("v1_failure_lineage")
            or value.get("deployment_data_roots") != preflight.get("deployment_data_roots")
            or value.get("target_free_preflight_required") is not True):
        raise V2Error("V2 root authorization binding drift")
    return dict(value)


def publish_target_free_preflight_v2(
    artifact: Any,
    capability: RootPublicationCapability,
    payload: Mapping[str, object],
    *,
    lineage: V1FailureLineage,
    closure: Mapping[str, object],
) -> str:
    if not isinstance(capability, RootPublicationCapability) or capability._seal is not _ROOT_SEAL:
        raise V2Error("only root may publish V2 preflight")
    checked = validate_target_free_preflight_v2(payload, lineage=lineage, closure=closure)
    return artifact.publish_json("official_preflight.json", checked)


def publish_root_authorization_v2(
    artifact: Any,
    capability: RootPublicationCapability,
    payload: Mapping[str, object],
    *,
    lineage: V1FailureLineage,
    closure: Mapping[str, object],
) -> str:
    if not isinstance(capability, RootPublicationCapability) or capability._seal is not _ROOT_SEAL:
        raise V2Error("only root may publish V2 authorization")
    body = artifact.reload_pair("official_preflight.json")
    digest = _sha(body)
    try:
        preflight = json.loads(body)
    except (TypeError, json.JSONDecodeError) as error:
        raise V2Error("durable V2 preflight JSON malformed") from error
    if not isinstance(preflight, Mapping):
        raise V2Error("durable V2 preflight JSON root drift")
    checked = validate_target_free_preflight_v2(preflight, lineage=lineage, closure=closure)
    authorization = validate_root_authorization_v2(payload, preflight_sha256=digest, preflight=checked)
    return artifact.publish_json("root_authorization.json", authorization)


def _read_v2_authority_pair(directory: Path, name: str) -> tuple[Mapping[str, Any], str]:
    body, _ = _v1()._canonical_regular_bytes(directory, name, expected_mode=0o444)
    digest = _sha(body)
    sidecar, _ = _v1()._canonical_regular_bytes(directory, name + ".sha256", expected_mode=0o444)
    if sidecar != f"{digest}  {name}\n".encode("ascii"):
        raise V2Error("V2 authority sidecar drift")
    try:
        value = json.loads(body)
    except (TypeError, json.JSONDecodeError) as error:
        raise V2Error("V2 authority JSON malformed") from error
    if not isinstance(value, Mapping):
        raise V2Error("V2 authority JSON root drift")
    return value, digest


def verify_v2_authorization(
    root: Path,
    *,
    lineage_loader: Callable[[Path], V1FailureLineage] = validate_v1_failed_lineage,
    closure_loader: Callable[[Path], Mapping[str, object]] = phase_e_v2_closure,
) -> tuple[V2Authorization, V1FailureLineage]:
    """Reload both V2 authority leaves and repeat V1 lineage verification."""
    lineage = lineage_loader(root)
    _validate_production_lineage(lineage)
    closure = _validate_v2_closure(closure_loader(root))
    directory = root.absolute() / AUTHORITY_ROOT_RELATIVE
    preflight, preflight_sha = _read_v2_authority_pair(directory, "official_preflight.json")
    checked_preflight = validate_target_free_preflight_v2(preflight, lineage=lineage, closure=closure)
    authorization, authorization_sha = _read_v2_authority_pair(directory, "root_authorization.json")
    checked_authorization = validate_root_authorization_v2(
        authorization, preflight_sha256=preflight_sha, preflight=checked_preflight,
    )
    return V2Authorization(
        preflight_sha256=preflight_sha, root_authorization_sha256=authorization_sha,
        preflight=checked_preflight, root_authorization=checked_authorization,
        expected_closure=closure,
    ), lineage


def _v2_attempt_payload(identity: V2ScoreIdentity) -> dict[str, object]:
    return {
        "schema": "tfsr_phase_e_v2_score_attempt_v1", "cell": CELL, "phase": PHASE,
        "identity": identity.payload(), "topology": list(SCORE_TOPOLOGY),
        "resolved": {"source": False, "within": False, "external": False, "formal": False},
        "opened": {"source": False, "within": False, "external": False, "formal": False},
        "backward_calls": 0, "optimizer_calls": 0,
    }


def validate_v2_attempt_payload(value: Mapping[str, Any], identity: V2ScoreIdentity) -> None:
    expected = {"schema", "cell", "phase", "identity", "topology", "resolved", "opened", "backward_calls", "optimizer_calls"}
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "tfsr_phase_e_v2_score_attempt_v1"
            or value.get("cell") != CELL or value.get("phase") != PHASE
            or value.get("identity") != identity.payload() or value.get("topology") != list(SCORE_TOPOLOGY)
            or value.get("resolved") != {"source": False, "within": False, "external": False, "formal": False}
            or value.get("opened") != {"source": False, "within": False, "external": False, "formal": False}
            or value.get("backward_calls") != 0 or value.get("optimizer_calls") != 0):
        raise V2Error("V2 attempt schema/boundary drift")


def _v2_input_payload(evidence: Any, identity: V2ScoreIdentity, v1_identity: Any) -> dict[str, object]:
    v1 = _v1()
    inner = v1._input_authority_payload(evidence, v1_identity)
    v1.validate_input_authority_payload(inner, v1_identity)
    return {
        "schema": "tfsr_phase_e_v2_input_authority_v1", "cell": CELL, "phase": PHASE,
        "identity": identity.payload(), "v1_input_authority": inner,
        "v1_input_authority_sha256": _sha(v1._json_bytes(inner)),
    }


def validate_v2_input_payload(value: Mapping[str, Any], identity: V2ScoreIdentity, v1_identity: Any) -> None:
    expected = {"schema", "cell", "phase", "identity", "v1_input_authority", "v1_input_authority_sha256"}
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "tfsr_phase_e_v2_input_authority_v1"
            or value.get("cell") != CELL or value.get("phase") != PHASE
            or value.get("identity") != identity.payload() or not isinstance(value.get("v1_input_authority"), Mapping)):
        raise V2Error("V2 input authority schema/identity drift")
    inner = value["v1_input_authority"]
    try:
        _v1().validate_input_authority_payload(inner, v1_identity)
    except Exception as error:
        raise V2Error("V2 nested V1 input authority drift") from error
    if value.get("v1_input_authority_sha256") != _sha(_v1()._json_bytes(inner)):
        raise V2Error("V2 nested V1 input authority digest drift")


def _v2_score_payload(identity: V2ScoreIdentity, input_sha: str, inner: Mapping[str, Any]) -> dict[str, object]:
    return {
        "schema": "tfsr_phase_e_v2_score_v1", "cell": CELL, "phase": PHASE,
        "identity": identity.payload(), "input_authority_sha256": input_sha,
        "v1_score": json.loads(json.dumps(inner, sort_keys=True)),
        "v1_score_sha256": _sha(_v1()._json_bytes(inner)),
        "semantic_change": "correct_canonical_data_root_binding_only",
    }


def validate_v2_score_payload(value: Mapping[str, Any], *, identity: V2ScoreIdentity, input_sha: str, v1_identity: Any) -> None:
    expected = {"schema", "cell", "phase", "identity", "input_authority_sha256", "v1_score", "v1_score_sha256", "semantic_change"}
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "tfsr_phase_e_v2_score_v1" or value.get("cell") != CELL
            or value.get("phase") != PHASE or value.get("identity") != identity.payload()
            or value.get("input_authority_sha256") != input_sha
            or value.get("semantic_change") != "correct_canonical_data_root_binding_only"
            or not isinstance(value.get("v1_score"), Mapping)):
        raise V2Error("V2 score schema/identity drift")
    inner = value["v1_score"]
    try:
        _v1().validate_score_payload(inner, identity=v1_identity, input_authority_sha256=input_sha)
    except Exception as error:
        raise V2Error("nested V1 score semantics drift") from error
    if value.get("v1_score_sha256") != _sha(_v1()._json_bytes(inner)):
        raise V2Error("nested V1 score digest drift")


def _v2_terminal_payload(identity: V2ScoreIdentity, *, attempt_sha: str, input_sha: str, score_sha: str, score: Mapping[str, Any], final_closure: Mapping[str, object]) -> dict[str, object]:
    inner = score["v1_score"]
    v1 = _v1()
    verdict = v1.decide_verdict(
        external=inner["tfsr_minus_cell_d"]["external"], within=inner["tfsr_minus_cell_d"]["within"],
    )
    return {
        "schema": "tfsr_phase_e_v2_score_terminal_v1", "status": "SCORE_COMPLETE", "cell": CELL, "phase": PHASE,
        "identity": identity.payload(), "attempt_sha256": attempt_sha, "input_authority_sha256": input_sha,
        "score_sha256": score_sha, "launch_closure": identity.launch_closure, "final_closure": dict(final_closure),
        "verdict": verdict, "semantic_change": "correct_canonical_data_root_binding_only",
        "boundaries": {"target_optimizer_steps": 0, "backward_calls": 0, "formal_resolved": False,
                       "formal_opened": False, "terminal_after_revalidation": True},
    }


def validate_v2_terminal_payload(value: Mapping[str, Any], *, identity: V2ScoreIdentity, score: Mapping[str, Any], expected_score_sha: str | None = None) -> None:
    expected = {"schema", "status", "cell", "phase", "identity", "attempt_sha256", "input_authority_sha256", "score_sha256",
                "launch_closure", "final_closure", "verdict", "semantic_change", "boundaries"}
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "tfsr_phase_e_v2_score_terminal_v1" or value.get("status") != "SCORE_COMPLETE"
            or value.get("cell") != CELL or value.get("phase") != PHASE or value.get("identity") != identity.payload()
            or value.get("launch_closure") != identity.launch_closure or value.get("final_closure") != identity.launch_closure
            or value.get("semantic_change") != "correct_canonical_data_root_binding_only"
            or value.get("boundaries") != {"target_optimizer_steps": 0, "backward_calls": 0, "formal_resolved": False,
                                             "formal_opened": False, "terminal_after_revalidation": True}):
        raise V2Error("V2 terminal schema/closure drift")
    for key in ("attempt_sha256", "input_authority_sha256", "score_sha256"):
        _require_sha(value.get(key), key)
    if expected_score_sha is not None and value.get("score_sha256") != expected_score_sha:
        raise V2Error("V2 terminal score binding drift")
    inner = score.get("v1_score") if isinstance(score, Mapping) else None
    if not isinstance(inner, Mapping):
        raise V2Error("V2 terminal missing nested score")
    verdict = _v1().decide_verdict(external=inner["tfsr_minus_cell_d"]["external"], within=inner["tfsr_minus_cell_d"]["within"])
    if value.get("verdict") != verdict:
        raise V2Error("V2 terminal verdict drift")


def _v2_failure_payload(identity: V2ScoreIdentity, attempt_sha: str | None, flags: Any) -> dict[str, object]:
    return {
        "schema": "tfsr_phase_e_v2_score_failure_v1", "cell": CELL, "phase": PHASE,
        "identity": identity.payload(), "attempt_sha256": attempt_sha, "stage": flags.stage,
        **flags.payload(), "traceback_sha256": _sha(traceback.format_exc().encode("utf-8")),
    }


def validate_v2_failure_payload(value: Mapping[str, Any], identity: V2ScoreIdentity) -> None:
    expected = {"schema", "cell", "phase", "identity", "attempt_sha256", "stage", "resolved", "opened", "forward_calls",
                "backward_calls", "optimizer_calls", "terminal_published", "traceback_sha256"}
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "tfsr_phase_e_v2_score_failure_v1" or value.get("cell") != CELL
            or value.get("phase") != PHASE or value.get("identity") != identity.payload()
            or value.get("terminal_published") is not False or not isinstance(value.get("stage"), str)):
        raise V2Error("V2 failure schema drift")
    if value.get("attempt_sha256") is not None:
        _require_sha(value.get("attempt_sha256"), "V2 failure attempt")
    try:
        _v1().ScoreFlags.validate_forward_calls(value.get("forward_calls"))
    except Exception as error:
        raise V2Error("V2 failure forward accounting drift") from error
    if type(value.get("backward_calls")) is not int or type(value.get("optimizer_calls")) is not int:
        raise V2Error("V2 failure update counters drift")
    _require_sha(value.get("traceback_sha256"), "V2 failure traceback")


class ScoreBackend(Protocol):
    def resolve_inputs(self, **kwargs: Any) -> Any: ...
    def score_cell_d(self, **kwargs: Any) -> Any: ...
    def score_tfsr(self, **kwargs: Any) -> Any: ...
    def reverify_after_forwards(self, **kwargs: Any) -> None: ...
    def resource_disclosure(self) -> Mapping[str, object]: ...
    def close(self) -> None: ...


class PhysicalMatchedScoreBackendV2(_v1().PhysicalMatchedScoreBackend):
    """Exact inheritance seam: V2 changes no physical score operation."""

    pass


def _publish_v2_failure(artifact: Any, identity: V2ScoreIdentity, attempt_sha: str | None, flags: Any) -> None:
    if flags.terminal_published or artifact.has_name("terminal.json") or artifact.has_name("failure.json"):
        return
    payload = _v2_failure_payload(identity, attempt_sha, flags)
    validate_v2_failure_payload(payload, identity)
    digest = artifact.publish_json("failure.json", payload)
    validate_v2_failure_payload(artifact.reload_json("failure.json", digest), identity)


def run_score_lifecycle_v2(
    *,
    artifact: Any,
    identity: V2ScoreIdentity,
    execution_capability: ExecutionCapability,
    v1_identity: Any,
    within_roster: tuple[str, ...],
    external_sessions: tuple[str, ...],
    within_roster_factory: Callable[[], tuple[Any, ...]],
    external_roster_factory: Callable[[], tuple[Any, ...]],
    sealed_cell_d_table: Mapping[str, Any],
    a2_pooled: Mapping[str, Mapping[str, float]],
    backend: ScoreBackend,
    final_reverify: Callable[[], Mapping[str, object]],
) -> Mapping[str, Any]:
    """Small V2 receipt envelope over the exact frozen V1 score operations."""
    v1 = _v1()
    _require_execution_capability(execution_capability, identity)
    flags = v1.ScoreFlags()
    attempt_sha: str | None = None
    try:
        flags.stage = "attempt"
        attempt = _v2_attempt_payload(identity)
        validate_v2_attempt_payload(attempt, identity)
        attempt_sha = artifact.publish_json("attempt.json", attempt)
        validate_v2_attempt_payload(artifact.reload_json("attempt.json", attempt_sha), identity)

        flags.stage = "resolve_inputs"
        within_assets = within_roster_factory()
        if tuple(item.session for item in within_assets) != tuple(sorted(within_roster)):
            raise V2Error("V2 post-attempt within roster binding drift")
        flags.within_resolved = True
        external_raw = external_roster_factory()
        if tuple(item.session for item in external_raw) != external_sessions:
            raise V2Error("V2 post-attempt external roster binding drift")
        flags.external_resolved = True
        external_assets = tuple(v1._as_evaluation_asset(item) for item in external_raw)
        evidence = backend.resolve_inputs(spec=v1.PUBLIC_SPEC, within_roster=within_roster, within_assets=within_assets,
                                          external_roster=external_assets, flags=flags)
        v1.validate_input_authority_evidence(evidence, within_roster=within_roster, within_assets=within_assets,
                                             external_roster=external_assets)
        input_payload = _v2_input_payload(evidence, identity, v1_identity)
        validate_v2_input_payload(input_payload, identity, v1_identity)
        input_sha = artifact.publish_json("input_authority.json", input_payload)
        validate_v2_input_payload(artifact.reload_json("input_authority.json", input_sha), identity, v1_identity)

        cell_d: list[Any] = []
        for surface, roster in (("within", within_roster), ("external", tuple(item.session for item in external_assets))):
            flags.stage = f"cell_d_{surface}"
            item = backend.score_cell_d(surface=surface, input_authority_sha256=input_sha, flags=flags)
            v1._validate_mode_against_roster(item, system="cell_d", surface=surface, mode="aligned",
                                             expected_sessions=roster, input_authority_sha256=input_sha)
            v1.assert_cell_d_parity(item, sealed_cell_d_table)
            cell_d.append(item)

        tfsr: list[Any] = []
        for surface, roster in (("within", within_roster), ("external", tuple(item.session for item in external_assets))):
            for mode in ("aligned", "zero", "wrong_pair"):
                flags.stage = f"tfsr_{surface}_{mode}"
                item = backend.score_tfsr(surface=surface, mode=mode, input_authority_sha256=input_sha, flags=flags)
                v1._validate_mode_against_roster(item, system="tfsr", surface=surface, mode=mode,
                                                 expected_sessions=roster, input_authority_sha256=input_sha)
                tfsr.append(item)
        if flags.formal_resolved or flags.formal_opened or flags.backward_calls or flags.optimizer_calls:
            raise V2Error("V2 forward path crossed formal/update boundary")
        flags.stage = "post_forward_reverify"
        backend.reverify_after_forwards(flags=flags)
        # Preserve V1's double held-descriptor revalidation exactly.  It is
        # intentionally redundant at the byte level but part of the frozen
        # physical scorer's accepted failure discipline.
        backend.reverify_after_forwards(flags=flags)
        # The caller's verifier must re-read both V1 authority/failure lineage
        # and V2 authorization, not merely compare a retained in-memory map.
        final_closure = final_reverify()
        if final_closure != identity.launch_closure:
            raise V2Error("V2 launch/final closure drift")

        flags.stage = "score"
        inner = v1.build_score_payload(identity=v1_identity, input_authority_sha256=input_sha, cell_d=cell_d,
                                       tfsr=tfsr, a2_pooled=a2_pooled, resources=backend.resource_disclosure(), flags=flags)
        v1.validate_score_payload(inner, identity=v1_identity, input_authority_sha256=input_sha)
        score = _v2_score_payload(identity, input_sha, inner)
        validate_v2_score_payload(score, identity=identity, input_sha=input_sha, v1_identity=v1_identity)
        score_body = _json_bytes(score)
        score_sha = _sha(score_body)
        flags.stage = "terminal_revalidation"
        # Pair publication is all-or-nothing: an exception cannot leave an
        # unterminalized scientific V2 score body.
        validate_v2_attempt_payload(artifact.reload_json("attempt.json", attempt_sha), identity)
        validate_v2_input_payload(artifact.reload_json("input_authority.json", input_sha), identity, v1_identity)
        terminal = _v2_terminal_payload(identity, attempt_sha=attempt_sha, input_sha=input_sha,
                                        score_sha=score_sha, score=score, final_closure=final_closure)
        validate_v2_terminal_payload(terminal, identity=identity, score=score, expected_score_sha=score_sha)
        terminal_body = _json_bytes(terminal)

        def validate_group(bodies: Mapping[str, bytes], digests: Mapping[str, str]) -> None:
            if (bodies.get("score.json") != score_body or bodies.get("terminal.json") != terminal_body
                    or digests.get("score.json") != score_sha):
                raise V2Error("V2 score/terminal publication binding drift")
            loaded_score = json.loads(artifact.reload_pair("score.json", score_sha))
            loaded_terminal = json.loads(artifact.reload_pair("terminal.json", _sha(terminal_body)))
            if not isinstance(loaded_score, Mapping) or not isinstance(loaded_terminal, Mapping):
                raise V2Error("V2 group JSON root drift")
            validate_v2_score_payload(loaded_score, identity=identity, input_sha=input_sha, v1_identity=v1_identity)
            validate_v2_terminal_payload(loaded_terminal, identity=identity, score=loaded_score, expected_score_sha=score_sha)

        hashes = artifact.publish_group({"score.json": score_body, "terminal.json": terminal_body}, post_publish=validate_group)
        flags.terminal_published = True
        final_score = artifact.reload_json("score.json", hashes["score.json"])
        final_terminal = artifact.reload_json("terminal.json", hashes["terminal.json"])
        validate_v2_score_payload(final_score, identity=identity, input_sha=input_sha, v1_identity=v1_identity)
        validate_v2_terminal_payload(final_terminal, identity=identity, score=final_score, expected_score_sha=hashes["score.json"])
        if artifact.has_name("failure.json"):
            raise V2Error("V2 terminal cannot coexist with failure")
        return final_terminal
    except BaseException:
        if attempt_sha is not None:
            try:
                _publish_v2_failure(artifact, identity, attempt_sha, flags)
            except BaseException:
                pass
        raise
    finally:
        backend.close()


def _reserve_v2_score_artifact(root: Path, *, lineage: V1FailureLineage) -> Any:
    _validate_production_lineage(lineage)
    parent, name = _v2_score_parent(root)
    return _v1().reserve_artifact_root(parent, name, SCORE_TOPOLOGY)


@dataclass(frozen=True)
class _ExecutionPreparation:
    fixed: Mapping[str, Any]
    training: Any
    v1_authorization: Any
    lineage: V1FailureLineage
    v2_authorization: V2Authorization
    subc_root: Path
    subm_root: Path


def _prepare_authorized_execution(
    root: Path,
    *,
    environment: Mapping[str, str] | None = None,
    fixed_loader: Callable[[Path], Mapping[str, Any]] | None = None,
    training_loader: Callable[[Path], Any] | None = None,
    v1_authorization_loader: Callable[[Path, Any, Any], Any] | None = None,
    lineage_loader: Callable[[Path, Mapping[str, Any], Any, Any], V1FailureLineage] | None = None,
    v2_authorization_loader: Callable[[Path, V1FailureLineage], tuple[V2Authorization, V1FailureLineage]] | None = None,
) -> _ExecutionPreparation:
    """Prepare V2 in its required access order, with a narrow test seam.

    The corrected environment values are deliberately checked first.  Thus a
    repeat of V1's parent-directory mistake cannot reserve a V2 score root,
    call a target-data factory, or even invoke the metadata loaders supplied
    here.  Production leaves every loader at ``None`` and reaches only frozen
    V1/V2 validators.
    """
    subc_root, subm_root = _correct_data_roots(environment)
    v1 = _v1()
    fixed = fixed_loader(root) if fixed_loader is not None else v1.verify_fixed_authorities(root)
    training = training_loader(root) if training_loader is not None else v1.validate_phase_d_training_terminal(root)
    v1_authorization = (
        v1_authorization_loader(root, training, fixed)
        if v1_authorization_loader is not None
        else v1.verify_phase_e_authorization(root, training, fixed)
    )
    lineage = (
        lineage_loader(root, fixed, training, v1_authorization)
        if lineage_loader is not None
        else validate_v1_failed_lineage(root, fixed_authorities=fixed, training=training, authorization=v1_authorization)
    )
    _validate_production_lineage(lineage)
    v2_authorization, verified_lineage = (
        v2_authorization_loader(root, lineage)
        if v2_authorization_loader is not None
        else verify_v2_authorization(root, lineage_loader=lambda _root: lineage)
    )
    if verified_lineage.payload() != lineage.payload():
        raise V2Error("V2 authorization V1 lineage mismatch")
    return _ExecutionPreparation(
        fixed=fixed, training=training, v1_authorization=v1_authorization, lineage=lineage,
        v2_authorization=v2_authorization, subc_root=subc_root, subm_root=subm_root,
    )


def execute_authorized_v2(root: Path) -> Mapping[str, Any]:
    """Physical V2 route; unreachable from CLI without its exact two flags."""
    v1 = _v1()
    prepared = _prepare_authorized_execution(root)
    fixed, training = prepared.fixed, prepared.training
    v1_authorization, lineage, v2_authorization = prepared.v1_authorization, prepared.lineage, prepared.v2_authorization
    v1_identity = _expected_v1_identity(v1, fixed, training, v1_authorization)
    identity = V2ScoreIdentity(v1_identity=v1_identity.payload(), v1_failed_lineage=lineage.payload(),
                               v2_authorization=v2_authorization.payload(), launch_closure=v2_authorization.expected_closure)
    capability = _issue_execution_capability(v2_authorization)
    artifact = _reserve_v2_score_artifact(root, lineage=lineage)
    within_roster = v1.extract_within_roster(fixed["strict_manifest"].value or {})
    external_sessions = v1.extract_external_sessions(fixed["external_asset_ledger"].value or {})
    a2_pooled = v1.extract_a2_pooled_tables(fixed["a2_matched_reference"].value or {}, within_sessions=within_roster,
                                             external_sessions=external_sessions)

    def within_factory() -> tuple[Any, ...]:
        return v1.join_within_assets(v1_authorization.preflight, prepared.subc_root)

    def external_factory() -> tuple[Any, ...]:
        return v1.join_external_assets(fixed["external_asset_ledger"].value or {}, fixed["external_scope"].value or {}, prepared.subm_root)

    def final_reverify() -> Mapping[str, object]:
        final_fixed = v1.verify_fixed_authorities(root)
        final_training = v1.validate_phase_d_training_terminal(root)
        final_v1_auth = v1.verify_phase_e_authorization(root, final_training, final_fixed)
        final_lineage = validate_v1_failed_lineage(root, fixed_authorities=final_fixed, training=final_training,
                                                   authorization=final_v1_auth)
        if final_lineage.payload() != lineage.payload():
            raise V2Error("V1 predecessor lineage changed after forwards")
        final_v2_auth, _ = verify_v2_authorization(root, lineage_loader=lambda _root: final_lineage)
        if final_v2_auth.payload() != v2_authorization.payload():
            raise V2Error("V2 authorization changed after forwards")
        return _validate_v2_closure(phase_e_v2_closure(root))

    backend = PhysicalMatchedScoreBackendV2(root=root, fixed_authorities=fixed, training=training,
                                            authorization=v1_authorization)
    return run_score_lifecycle_v2(
        artifact=artifact, identity=identity, execution_capability=capability, v1_identity=v1_identity,
        within_roster=within_roster, external_sessions=external_sessions,
        within_roster_factory=within_factory, external_roster_factory=external_factory,
        sealed_cell_d_table=fixed["cell_d_table"].value or {}, a2_pooled=a2_pooled,
        backend=backend, final_reverify=final_reverify,
    )


def dry_plan() -> dict[str, object]:
    """Pure declarative V2 plan; no repository read, Torch, or data access."""
    return {
        "cell": CELL, "phase": PHASE,
        "status": "NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_SCORE",
        "authorization": "none",
        "execution_flags_required_together": ["--execute", "--i-have-phase-e-root-authorization"],
        "authority_root": AUTHORITY_ROOT_RELATIVE, "score_root": SCORE_ROOT_RELATIVE,
        "predecessor": {"v1_attempt_sha256": V1_ATTEMPT_SHA256, "v1_failure_sha256": V1_FAILURE_SHA256,
                          "semantic_change": "correct_canonical_data_root_binding_only"},
        "correct_roots": {"SUBC_DATA_ROOT": SUBC_DATA_ROOT, "SUBM_DATA_ROOT": SUBM_DATA_ROOT},
        "physical_backend": "PhysicalMatchedScoreBackendV2 inherits frozen V1 without overrides",
        "metric_and_matrix": "frozen V1 last-bin/equal-session Cell-D plus TF-SR aligned/zero/wrong-pair",
    }
