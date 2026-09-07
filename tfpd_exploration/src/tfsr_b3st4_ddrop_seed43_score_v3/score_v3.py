"""Thin seed43 Phase-E V3 successor for the exact physical ``ScoreSpec``.

V2 correctly failed closed before reading an evaluation asset: its lifecycle
passed a value-equivalent ``ScoreSpec`` from a separately static-loaded module
to the inherited physical backend.  The backend intentionally compares to its
own frozen public object.  This additive route keeps every scientific and
physical implementation in the reviewed V1/V2 chain and changes only that
object-identity hand-off.

There are no Torch imports at module import time.  The physical backend stays
behind the V1 composition seam and is reachable only from the future reviewed
execution route.
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
from typing import Any, Callable, Mapping, Protocol


CELL = "TFSR_B3ST4_DDROP_SEED43"
PHASE = "TFSR_PHASE_E_SEED43_REPLICATION_ADDENDUM_V3"
SEED = 43
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_TFSR_SEED43_PHASE_E_V3_20260823.md"
# Replaced with the exact body digest as the final frozen edit in this route.
WORKORDER_SHA256 = "a1cbeb19b2b6265f72a3167a08d6cc34fabb4136171bcb16fdf04a125123d019"
SOURCE_RELATIVE = "tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_score_v3/score_v3.py"
INIT_RELATIVE = "tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_score_v3/__init__.py"
CLI_RELATIVE = "tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_seed43_score_v3.py"
TEST_RELATIVE = "tfpd_exploration/tests/test_tfsr_b3st4_ddrop_seed43_score_v3.py"

AUTHORITY_ROOT_RELATIVE = "tfpd_exploration/results/tfsr_b3st4_ddrop_seed43_score_authority_v3"
SCORE_ROOT_RELATIVE = "tfpd_exploration/results/tfsr_b3st4_ddrop_seed43_matched_score_v3"
AUTHORITY_TOPOLOGY = ("official_preflight.json", "root_authorization.json")
SCORE_TOPOLOGY = ("attempt.json", "input_replay.json", "score.json", "terminal.json", "failure.json")
PUBLIC_FLAGS = frozenset(("--execute", "--i-have-seed43-phase-e-v3-authorization"))

SUBC_DATA_ROOT = "/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-C"
SUBM_DATA_ROOT = "/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-M"

V2_AUTHORITY_PREFLIGHT_SHA256 = "b870bf0ea9f01bcd64441f22612e3bb9cfbb0a18eb909ac44b95bc366c6b0f07"
V2_AUTHORITY_ROOT_AUTHORIZATION_SHA256 = "4c999af634eb0caa718cc377b5bb165c10fd8cd876ab90d7892541905491adc4"
V2_FAILED_ATTEMPT_SHA256 = "ac9f4a54ce8d3a719791b401caecd4f35f05bb8124374a90d181f865989cc29f"
V2_FAILED_FAILURE_SHA256 = "e228395570a22904d99c678fc523fa01e14f78a4d5528735ad5e1a6652262458"

_V2_NAME = "_tfsr_seed43_phase_e_v2_for_v3"
_V2: ModuleType | None = None


class FailClosedError(RuntimeError):
    """A V3 receipt, closure, capability, or exact-specification violation."""


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_static(*, package_name: str, directory: Path, leaf: str) -> ModuleType:
    name = package_name + "." + leaf.removesuffix(".py")
    present = sys.modules.get(name)
    if isinstance(present, ModuleType):
        return present
    package = ModuleType(package_name)
    package.__path__ = [str(directory)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(name, directory / leaf)
    if spec is None or spec.loader is None:
        raise RuntimeError("static V3 predecessor loader failure")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _v2() -> ModuleType:
    global _V2
    if _V2 is None:
        _V2 = _load_static(
            package_name=_V2_NAME,
            directory=_repository_root() / "tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_score_v2",
            leaf="score_v2.py",
        )
    return _V2


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")).encode("utf-8") + b"\n"


def _copy(value: object) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _require_sha(value: object, label: str) -> str:
    if not _is_sha(value):
        raise FailClosedError(f"{label} must be an exact lowercase SHA-256")
    return str(value)


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise FailClosedError("route path must be a canonical safe relative path")
    return value


def _directory_identity(path: Path) -> tuple[int, int]:
    try:
        info = os.lstat(path)
    except OSError as error:
        raise FailClosedError("required immutable directory is absent") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise FailClosedError("required immutable directory identity drift")
    return (info.st_dev, info.st_ino)


def _read_all(fd: int) -> bytes:
    blocks: list[bytes] = []
    while True:
        block = os.read(fd, 1 << 20)
        if not block:
            return b"".join(blocks)
        blocks.append(block)


def _read_0444(directory_fd: int, name: str) -> bytes:
    if not isinstance(name, str) or "/" in name or name in {"", ".", ".."}:
        raise FailClosedError("immutable predecessor leaf name drift")
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    except OSError as error:
        raise FailClosedError("immutable predecessor leaf missing") from error
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444:
            raise FailClosedError("immutable predecessor leaf type/mode drift")
        return _read_all(fd)
    finally:
        os.close(fd)


def _read_exact_pairs(directory: Path, expected: Mapping[str, str]) -> tuple[dict[str, Mapping[str, object]], tuple[int, int]]:
    """Read only exact immutable receipt pairs through one held directory FD."""
    initial = _directory_identity(directory)
    expected_names = {name for body in expected for name in (body, body + ".sha256")}
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != initial or set(os.listdir(fd)) != expected_names:
            raise FailClosedError("immutable predecessor topology/identity drift")
        result: dict[str, Mapping[str, object]] = {}
        for name, digest in expected.items():
            body = _read_0444(fd, name)
            if _sha(body) != digest:
                raise FailClosedError("immutable predecessor body SHA drift")
            if _read_0444(fd, name + ".sha256") != f"{digest}  {name}\n".encode("ascii"):
                raise FailClosedError("immutable predecessor canonical sidecar drift")
            try:
                payload = json.loads(body)
            except (TypeError, json.JSONDecodeError) as error:
                raise FailClosedError("immutable predecessor JSON malformed") from error
            if not isinstance(payload, Mapping):
                raise FailClosedError("immutable predecessor JSON root drift")
            result[name] = payload
        if _directory_identity(directory) != initial:
            raise FailClosedError("immutable predecessor directory changed during read")
        return result, initial
    finally:
        os.close(fd)


def physical_seed42_score_module() -> ModuleType:
    """Return the module that owns the inherited backend's frozen spec.

    This is deliberately distinct from the static V1 module nested inside the
    seed42 V2 wrapper.  The latter has an equal *payload* but a foreign class.
    """
    # A root-reviewed direct API call must have the same deterministic import
    # context as the public CLI.  These are fixed repository locations, not
    # ambient-path discovery; this function is never reached by dry mode.
    root = _repository_root()
    for path in (root / "tfpd_exploration" / "src", root / "tfpd_exploration"):
        rendered = str(path)
        if rendered not in sys.path:
            sys.path.insert(0, rendered)
    return _v2()._seed43_v1()._seed42_score()


def expected_physical_score_spec() -> Any:
    physical = physical_seed42_score_module()
    value = physical.PUBLIC_SPEC
    static = _v2()._seed42_v2()._v1().PUBLIC_SPEC
    if value.payload() != static.payload():
        raise FailClosedError("physical/static public ScoreSpec payload drift")
    # The regression being repaired is meaningful only if these truly arise
    # from different static import lineages.  Do not accidentally erase it by
    # coercing one object into the other.
    if value is static or value == static:
        raise FailClosedError("V3 expected distinct physical/static ScoreSpec lineage")
    return value


@dataclass(frozen=True)
class FailedV2Predecessor:
    """Descriptor-validated V2 authority plus its immutable pre-input failure."""

    upstream: Any
    training: Any
    authorization: Any
    v2_identity: Any
    authority_root_identity: tuple[int, int]
    failed_score_root_identity: tuple[int, int]
    authority_preflight_sha256: str = V2_AUTHORITY_PREFLIGHT_SHA256
    authority_root_authorization_sha256: str = V2_AUTHORITY_ROOT_AUTHORIZATION_SHA256
    failed_attempt_sha256: str = V2_FAILED_ATTEMPT_SHA256
    failed_failure_sha256: str = V2_FAILED_FAILURE_SHA256

    def __post_init__(self) -> None:
        for label, value in (
            ("V2 authority preflight", self.authority_preflight_sha256),
            ("V2 authority authorization", self.authority_root_authorization_sha256),
            ("V2 failed attempt", self.failed_attempt_sha256),
            ("V2 failed failure", self.failed_failure_sha256),
        ):
            _require_sha(value, label)
        for identity in (self.authority_root_identity, self.failed_score_root_identity):
            if len(identity) != 2 or any(type(item) is not int or item < 0 for item in identity):
                raise ValueError("immutable V2 directory identity drift")

    def binding_payload(self) -> dict[str, object]:
        return {
            "schema": "tfsr_seed43_phase_e_v3_failed_v2_predecessor_v1",
            "v2_authority_root_relative": _v2().AUTHORITY_ROOT_RELATIVE,
            "v2_authority_root_identity": list(self.authority_root_identity),
            "v2_authority_pairs": {
                "official_preflight_sha256": self.authority_preflight_sha256,
                "root_authorization_sha256": self.authority_root_authorization_sha256,
            },
            "v2_failed_score_root_relative": _v2().SCORE_ROOT_RELATIVE,
            "v2_failed_score_root_identity": list(self.failed_score_root_identity),
            "v2_failed_pairs": {
                "attempt_sha256": self.failed_attempt_sha256,
                "failure_sha256": self.failed_failure_sha256,
            },
            "v2_failure_stage": "resolve_identical_seed42_v2_inputs_after_attempt",
            "v2_failure_has_no_input_terminal_or_forwards": True,
            "v2_authorization": _copy(self.authorization.payload()),
        }


def _validate_v2_failure_facts(value: Mapping[str, object]) -> None:
    if (
        value.get("schema") != "tfsr_seed43_phase_e_v2_failure_v1"
        or value.get("stage") != "resolve_identical_seed42_v2_inputs_after_attempt"
        or value.get("input_replay_sha256") is not None
        or value.get("terminal_published") is not False
        or value.get("backward_calls") != 0
        or value.get("optimizer_calls") != 0
        or value.get("cell_d_rerun") is not False
        or value.get("resolved") != {"source": False, "within": True, "external": True, "formal": False}
        or value.get("opened") != {"source": False, "within": False, "external": False, "formal": False}
        or value.get("tfsr_forward_calls") != {
            "within": {"aligned": 0, "zero": 0, "wrong_pair": 0},
            "external": {"aligned": 0, "zero": 0, "wrong_pair": 0},
        }
    ):
        raise FailClosedError("V2 failed lifecycle boundary/stage drift")


def load_failed_v2_predecessor(
    root: Path,
    *,
    upstream_loader: Callable[[Path], Any] | None = None,
    training_loader: Callable[[Path], Any] | None = None,
    authorization_loader: Callable[[Path, Any, Any], Any] | None = None,
) -> FailedV2Predecessor:
    """Validate V2 completed authority then failed attempt/failure before V3 exists."""
    v2 = _v2()
    s43 = v2._seed43_v1()
    upstream = (upstream_loader or v2.load_completed_seed42_v2)(root)
    training = (training_loader or s43.validate_seed43_training_terminal)(root)
    authorization = (
        authorization_loader(root, upstream, training)
        if authorization_loader is not None
        else v2.verify_seed43_v2_authorization(root, upstream=upstream, training=training)
    )
    identity = v2.Seed43V2Identity(
        training=training, upstream=upstream, closure=authorization.closure, authorization=authorization.payload()
    )
    authority_payloads, authority_identity = _read_exact_pairs(
        root.absolute() / v2.AUTHORITY_ROOT_RELATIVE,
        {
            "official_preflight.json": V2_AUTHORITY_PREFLIGHT_SHA256,
            "root_authorization.json": V2_AUTHORITY_ROOT_AUTHORIZATION_SHA256,
        },
    )
    # Verify the durable semantic pair a second time even when a caller
    # injected an authorization loader in a no-data test.
    v2.validate_target_free_preflight(
        authority_payloads["official_preflight.json"], upstream=upstream, training=training, closure=authorization.closure
    )
    v2.validate_root_authorization(
        authority_payloads["root_authorization.json"],
        preflight_sha256=V2_AUTHORITY_PREFLIGHT_SHA256,
        preflight=authority_payloads["official_preflight.json"],
    )
    failed_payloads, failed_identity = _read_exact_pairs(
        root.absolute() / v2.SCORE_ROOT_RELATIVE,
        {"attempt.json": V2_FAILED_ATTEMPT_SHA256, "failure.json": V2_FAILED_FAILURE_SHA256},
    )
    v2.validate_attempt_payload(failed_payloads["attempt.json"], identity)
    v2.validate_failure_payload(
        failed_payloads["failure.json"], identity=identity, attempt_sha=V2_FAILED_ATTEMPT_SHA256, input_sha=None
    )
    _validate_v2_failure_facts(failed_payloads["failure.json"])
    return FailedV2Predecessor(
        upstream=upstream,
        training=training,
        authorization=authorization,
        v2_identity=identity,
        authority_root_identity=authority_identity,
        failed_score_root_identity=failed_identity,
    )


def _closure_paths() -> tuple[str, ...]:
    v2 = _v2()
    return tuple(dict.fromkeys((
        WORKORDER_RELATIVE, INIT_RELATIVE, SOURCE_RELATIVE, CLI_RELATIVE, TEST_RELATIVE,
        *tuple(v2.V2_CLOSURE),
    )))


V3_CLOSURE = _closure_paths()


def phase_e_seed43_v3_closure(root: Path) -> dict[str, object]:
    v2 = _v2(); reader = v2._seed42_v2()._v1()
    hashes: dict[str, str] = {}
    for relative in V3_CLOSURE:
        body, _ = reader._canonical_regular_bytes(root.absolute(), relative, expected_mode=0o664)
        hashes[relative] = _sha(body)
    if hashes.get(WORKORDER_RELATIVE) != WORKORDER_SHA256:
        raise FailClosedError("seed43 V3 workorder SHA drift")
    nested = v2.phase_e_seed43_v2_closure(root)
    encoded = json.dumps(
        {"paths": list(V3_CLOSURE), "sha256_by_path": hashes, "seed43_v2_closure": nested},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return {
        "paths": list(V3_CLOSURE), "sha256_by_path": hashes,
        "seed43_v2_closure": nested, "closure_sha256": _sha(encoded),
    }


def validate_phase_e_seed43_v3_closure(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"paths", "sha256_by_path", "seed43_v2_closure", "closure_sha256"}:
        raise FailClosedError("seed43 V3 closure schema drift")
    hashes = value.get("sha256_by_path")
    if value.get("paths") != list(V3_CLOSURE) or not isinstance(hashes, Mapping) or set(hashes) != set(V3_CLOSURE):
        raise FailClosedError("seed43 V3 closure topology drift")
    if any(not _is_sha(hashes.get(path)) for path in V3_CLOSURE) or hashes.get(WORKORDER_RELATIVE) != WORKORDER_SHA256:
        raise FailClosedError("seed43 V3 closure leaf drift")
    try:
        nested = _v2().validate_phase_e_seed43_v2_closure(value.get("seed43_v2_closure"))
    except Exception as error:
        raise FailClosedError("seed43 V3 inherited V2 closure drift") from error
    encoded = json.dumps(
        {"paths": list(V3_CLOSURE), "sha256_by_path": dict(hashes), "seed43_v2_closure": nested},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    if value.get("closure_sha256") != _sha(encoded):
        raise FailClosedError("seed43 V3 closure aggregate drift")
    return {
        "paths": list(V3_CLOSURE), "sha256_by_path": {path: str(hashes[path]) for path in V3_CLOSURE},
        "seed43_v2_closure": nested, "closure_sha256": str(value["closure_sha256"]),
    }


@dataclass(frozen=True)
class Seed43V3Authorization:
    preflight_sha256: str
    root_authorization_sha256: str
    preflight: Mapping[str, object]
    root_authorization: Mapping[str, object]
    closure: Mapping[str, object]

    def __post_init__(self) -> None:
        _require_sha(self.preflight_sha256, "seed43 V3 preflight")
        _require_sha(self.root_authorization_sha256, "seed43 V3 root authorization")

    def payload(self) -> dict[str, object]:
        return {
            "preflight_sha256": self.preflight_sha256,
            "root_authorization_sha256": self.root_authorization_sha256,
            "closure": _copy(self.closure),
        }


@dataclass(frozen=True)
class Seed43V3Identity:
    predecessor: FailedV2Predecessor
    closure: Mapping[str, object]
    authorization: Mapping[str, object]

    @property
    def training(self) -> Any:
        return self.predecessor.training

    @property
    def upstream(self) -> Any:
        return self.predecessor.upstream

    @property
    def v2_identity(self) -> Any:
        return self.predecessor.v2_identity

    def payload(self) -> dict[str, object]:
        return {
            "seed43_training": self.training.payload(),
            "upstream_seed42_v2": self.upstream.binding_payload(),
            "failed_v2_predecessor": self.predecessor.binding_payload(),
            "closure": _copy(self.closure),
            "authorization": _copy(self.authorization),
        }


class RootPublicationCapability:
    __slots__ = ("_seal",)
    def __init__(self, seal: object) -> None:
        if seal is not _ROOT_SEAL:
            raise TypeError("seed43 V3 root publication capability is internal")
        self._seal = seal


class ExecutionCapability:
    __slots__ = ("authorization", "_seal")
    def __init__(self, authorization: Seed43V3Authorization, seal: object) -> None:
        if seal is not _EXECUTION_SEAL:
            raise TypeError("seed43 V3 execution capability is internal")
        self.authorization = authorization
        self._seal = seal


_ROOT_SEAL = object()
_EXECUTION_SEAL = object()


def issue_root_publication_capability() -> RootPublicationCapability:
    return RootPublicationCapability(_ROOT_SEAL)


def _issue_execution_capability(authorization: Seed43V3Authorization) -> ExecutionCapability:
    return ExecutionCapability(authorization, _EXECUTION_SEAL)


def _require_execution_capability(capability: object, identity: Seed43V3Identity) -> None:
    if not isinstance(capability, ExecutionCapability) or capability._seal is not _EXECUTION_SEAL:
        raise FailClosedError("seed43 V3 requires root-reviewed in-process execution capability")
    if capability.authorization.payload() != identity.authorization:
        raise FailClosedError("seed43 V3 capability/identity authorization drift")


def _authority_parent(root: Path) -> tuple[Path, str]:
    path = root.absolute() / _safe_relative(AUTHORITY_ROOT_RELATIVE)
    return path.parent, path.name


def _score_parent(root: Path) -> tuple[Path, str]:
    path = root.absolute() / _safe_relative(SCORE_ROOT_RELATIVE)
    return path.parent, path.name


def _correct_data_roots(environment: Mapping[str, str] | None = None) -> tuple[Path, Path]:
    env = os.environ if environment is None else environment
    if env.get("SUBC_DATA_ROOT") != SUBC_DATA_ROOT or env.get("SUBM_DATA_ROOT") != SUBM_DATA_ROOT:
        raise FailClosedError("seed43 V3 requires exact canonical SUBC_DATA_ROOT and SUBM_DATA_ROOT before reservation")
    return Path(SUBC_DATA_ROOT), Path(SUBM_DATA_ROOT)


def build_target_free_preflight(root: Path, *, predecessor: FailedV2Predecessor, closure: Mapping[str, object]) -> dict[str, object]:
    checked = validate_phase_e_seed43_v3_closure(closure)
    payload = expected_physical_score_spec().payload()
    return {
        "schema": "tfsr_seed43_phase_e_v3_preflight_v1", "status": "PREFLIGHT_ACCEPTED", "cell": CELL,
        "phase": PHASE, "seed": SEED, "physical_score_spec_payload": payload,
        "physical_score_spec_owner": "src.tfsr_b3st4_ddrop_v1.score.PUBLIC_SPEC",
        "seed43_training": predecessor.training.payload(), "upstream_seed42_v2": predecessor.upstream.binding_payload(),
        "failed_v2_predecessor": predecessor.binding_payload(), "closure": checked,
        "authority_root_relative": AUTHORITY_ROOT_RELATIVE, "score_root_relative": SCORE_ROOT_RELATIVE,
        "deployment_data_roots": {"SUBC_DATA_ROOT": SUBC_DATA_ROOT, "SUBM_DATA_ROOT": SUBM_DATA_ROOT},
        "target_free": True, "formal_sessions_inert": list(_v2()._seed42_v2()._v1().FORMAL_TEST_SESSION_NAMES),
        "science": {"matrix": "seed43_v1_TFSR_six_cells", "metric": "last_bin_variance_weighted_r2_equal_session",
                    "cell_d_replay": "nested_completed_seed42_v2_v1_evidence_only", "gpu_profile": "inherited_seed43_v1_GPU1"},
        "repair": {"v2_failure": "foreign_static_ScoreSpec", "v3_change": "exact_backend_owned_PUBLIC_SPEC_object_only"},
    }


def validate_target_free_preflight(value: Mapping[str, object], *, predecessor: FailedV2Predecessor,
                                   closure: Mapping[str, object]) -> dict[str, object]:
    checked = validate_phase_e_seed43_v3_closure(closure)
    expected = {
        "schema", "status", "cell", "phase", "seed", "physical_score_spec_payload", "physical_score_spec_owner",
        "seed43_training", "upstream_seed42_v2", "failed_v2_predecessor", "closure", "authority_root_relative",
        "score_root_relative", "deployment_data_roots", "target_free", "formal_sessions_inert", "science", "repair",
    }
    if (
        not isinstance(value, Mapping) or set(value) != expected
        or value.get("schema") != "tfsr_seed43_phase_e_v3_preflight_v1" or value.get("status") != "PREFLIGHT_ACCEPTED"
        or value.get("cell") != CELL or value.get("phase") != PHASE or value.get("seed") != SEED
        or value.get("physical_score_spec_payload") != expected_physical_score_spec().payload()
        or value.get("physical_score_spec_owner") != "src.tfsr_b3st4_ddrop_v1.score.PUBLIC_SPEC"
        or value.get("seed43_training") != predecessor.training.payload()
        or value.get("upstream_seed42_v2") != predecessor.upstream.binding_payload()
        or value.get("failed_v2_predecessor") != predecessor.binding_payload() or value.get("closure") != checked
        or value.get("authority_root_relative") != AUTHORITY_ROOT_RELATIVE or value.get("score_root_relative") != SCORE_ROOT_RELATIVE
        or value.get("deployment_data_roots") != {"SUBC_DATA_ROOT": SUBC_DATA_ROOT, "SUBM_DATA_ROOT": SUBM_DATA_ROOT}
        or value.get("target_free") is not True
        or value.get("formal_sessions_inert") != list(_v2()._seed42_v2()._v1().FORMAL_TEST_SESSION_NAMES)
        or value.get("science") != {"matrix": "seed43_v1_TFSR_six_cells", "metric": "last_bin_variance_weighted_r2_equal_session",
                                    "cell_d_replay": "nested_completed_seed42_v2_v1_evidence_only", "gpu_profile": "inherited_seed43_v1_GPU1"}
        or value.get("repair") != {"v2_failure": "foreign_static_ScoreSpec", "v3_change": "exact_backend_owned_PUBLIC_SPEC_object_only"}
    ):
        raise FailClosedError("seed43 V3 preflight schema/binding drift")
    return _copy(value)


def build_root_authorization(*, preflight_sha256: str, preflight: Mapping[str, object]) -> dict[str, object]:
    _require_sha(preflight_sha256, "seed43 V3 preflight")
    return {
        "schema": "tfsr_seed43_phase_e_v3_root_authorization_v1", "status": "ROOT_AUTHORIZED", "cell": CELL,
        "phase": PHASE, "official_preflight_sha256": preflight_sha256,
        "authority_root_relative": AUTHORITY_ROOT_RELATIVE, "score_root_relative": SCORE_ROOT_RELATIVE,
        "closure": preflight.get("closure"), "failed_v2_predecessor": preflight.get("failed_v2_predecessor"),
        "seed43_terminal_required": True, "target_free_preflight_required": True,
    }


def validate_root_authorization(value: Mapping[str, object], *, preflight_sha256: str,
                                preflight: Mapping[str, object]) -> dict[str, object]:
    _require_sha(preflight_sha256, "seed43 V3 preflight")
    expected = {
        "schema", "status", "cell", "phase", "official_preflight_sha256", "authority_root_relative",
        "score_root_relative", "closure", "failed_v2_predecessor", "seed43_terminal_required",
        "target_free_preflight_required",
    }
    if (
        not isinstance(value, Mapping) or set(value) != expected
        or value.get("schema") != "tfsr_seed43_phase_e_v3_root_authorization_v1"
        or value.get("status") != "ROOT_AUTHORIZED" or value.get("cell") != CELL or value.get("phase") != PHASE
        or value.get("official_preflight_sha256") != preflight_sha256
        or value.get("authority_root_relative") != AUTHORITY_ROOT_RELATIVE or value.get("score_root_relative") != SCORE_ROOT_RELATIVE
        or value.get("closure") != preflight.get("closure")
        or value.get("failed_v2_predecessor") != preflight.get("failed_v2_predecessor")
        or value.get("seed43_terminal_required") is not True or value.get("target_free_preflight_required") is not True
    ):
        raise FailClosedError("seed43 V3 root authorization binding drift")
    return _copy(value)


def reserve_authority_artifact(root: Path, capability: RootPublicationCapability, *,
                               predecessor_loader: Callable[[Path], FailedV2Predecessor] = load_failed_v2_predecessor) -> Any:
    if not isinstance(capability, RootPublicationCapability) or capability._seal is not _ROOT_SEAL:
        raise FailClosedError("only root may reserve seed43 V3 authority root")
    predecessor_loader(root)
    parent, name = _authority_parent(root)
    return _v2()._seed42_v2()._v1().reserve_artifact_root(parent, name, AUTHORITY_TOPOLOGY)


def publish_target_free_preflight(artifact: Any, capability: RootPublicationCapability, payload: Mapping[str, object], *,
                                  predecessor: FailedV2Predecessor, closure: Mapping[str, object]) -> str:
    if not isinstance(capability, RootPublicationCapability) or capability._seal is not _ROOT_SEAL:
        raise FailClosedError("only root may publish seed43 V3 preflight")
    return artifact.publish_json("official_preflight.json", validate_target_free_preflight(payload, predecessor=predecessor, closure=closure))


def publish_root_authorization(artifact: Any, capability: RootPublicationCapability, payload: Mapping[str, object], *,
                               predecessor: FailedV2Predecessor, closure: Mapping[str, object]) -> str:
    if not isinstance(capability, RootPublicationCapability) or capability._seal is not _ROOT_SEAL:
        raise FailClosedError("only root may publish seed43 V3 authorization")
    body = artifact.reload_pair("official_preflight.json")
    try:
        preflight = json.loads(body)
    except (TypeError, json.JSONDecodeError) as error:
        raise FailClosedError("durable V3 preflight JSON malformed") from error
    if not isinstance(preflight, Mapping):
        raise FailClosedError("durable V3 preflight JSON root drift")
    checked = validate_target_free_preflight(preflight, predecessor=predecessor, closure=closure)
    return artifact.publish_json("root_authorization.json", validate_root_authorization(
        payload, preflight_sha256=_sha(body), preflight=checked,
    ))


def _read_authority_pair(directory: Path, name: str) -> tuple[Mapping[str, object], str]:
    reader = _v2()._seed42_v2()._v1()
    body, _ = reader._canonical_regular_bytes(directory, name, expected_mode=0o444)
    digest = _sha(body)
    sidecar, _ = reader._canonical_regular_bytes(directory, name + ".sha256", expected_mode=0o444)
    if sidecar != f"{digest}  {name}\n".encode("ascii"):
        raise FailClosedError("seed43 V3 authority sidecar drift")
    try:
        value = json.loads(body)
    except (TypeError, json.JSONDecodeError) as error:
        raise FailClosedError("seed43 V3 authority JSON malformed") from error
    if not isinstance(value, Mapping):
        raise FailClosedError("seed43 V3 authority JSON root drift")
    return value, digest


def verify_seed43_v3_authorization(root: Path, *, predecessor: FailedV2Predecessor,
                                   closure_loader: Callable[[Path], Mapping[str, object]] = phase_e_seed43_v3_closure) -> Seed43V3Authorization:
    closure = validate_phase_e_seed43_v3_closure(closure_loader(root))
    directory = root.absolute() / AUTHORITY_ROOT_RELATIVE
    preflight, preflight_sha = _read_authority_pair(directory, "official_preflight.json")
    checked = validate_target_free_preflight(preflight, predecessor=predecessor, closure=closure)
    authorization, authorization_sha = _read_authority_pair(directory, "root_authorization.json")
    checked_auth = validate_root_authorization(authorization, preflight_sha256=preflight_sha, preflight=checked)
    return Seed43V3Authorization(preflight_sha, authorization_sha, checked, checked_auth, closure)


def _v3_attempt_payload(identity: Seed43V3Identity) -> dict[str, object]:
    return {
        "schema": "tfsr_seed43_phase_e_v3_attempt_v1", "cell": CELL, "phase": PHASE, "seed": SEED,
        "identity": identity.payload(), "topology": list(SCORE_TOPOLOGY),
        "resolved": {"source": False, "within": False, "external": False, "formal": False},
        "opened": {"source": False, "within": False, "external": False, "formal": False},
        "backward_calls": 0, "optimizer_calls": 0, "cell_d_rerun": False,
    }


def validate_attempt_payload(value: Mapping[str, object], identity: Seed43V3Identity) -> None:
    expected = {"schema", "cell", "phase", "seed", "identity", "topology", "resolved", "opened", "backward_calls", "optimizer_calls", "cell_d_rerun"}
    if (
        not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != "tfsr_seed43_phase_e_v3_attempt_v1"
        or value.get("cell") != CELL or value.get("phase") != PHASE or value.get("seed") != SEED
        or value.get("identity") != identity.payload() or value.get("topology") != list(SCORE_TOPOLOGY)
        or value.get("resolved") != {"source": False, "within": False, "external": False, "formal": False}
        or value.get("opened") != {"source": False, "within": False, "external": False, "formal": False}
        or value.get("backward_calls") != 0 or value.get("optimizer_calls") != 0 or value.get("cell_d_rerun") is not False
    ):
        raise FailClosedError("seed43 V3 attempt schema/boundary drift")


def _input_replay_payload(identity: Seed43V3Identity, v2_core: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": "tfsr_seed43_phase_e_v3_input_replay_v1", "cell": CELL, "phase": PHASE,
        "identity": identity.payload(), "v2_core_input_replay": _copy(v2_core),
        "v2_core_input_replay_sha256": _sha(_json_bytes(v2_core)),
    }


def validate_input_replay_payload(value: Mapping[str, object], *, identity: Seed43V3Identity) -> None:
    expected = {"schema", "cell", "phase", "identity", "v2_core_input_replay", "v2_core_input_replay_sha256"}
    if (
        not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != "tfsr_seed43_phase_e_v3_input_replay_v1"
        or value.get("cell") != CELL or value.get("phase") != PHASE or value.get("identity") != identity.payload()
        or not isinstance(value.get("v2_core_input_replay"), Mapping)
    ):
        raise FailClosedError("seed43 V3 input replay schema/binding drift")
    core = value["v2_core_input_replay"]
    if value.get("v2_core_input_replay_sha256") != _sha(_json_bytes(core)):
        raise FailClosedError("seed43 V3 nested input replay SHA drift")
    _v2().validate_input_replay_payload(core, identity=identity.v2_identity)


def _score_payload(identity: Seed43V3Identity, input_sha: str, v2_core: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": "tfsr_seed43_phase_e_v3_score_v1", "status": "REPLICATION_SCORE_COMPLETE", "cell": CELL,
        "phase": PHASE, "seed": SEED, "identity": identity.payload(), "input_replay_sha256": input_sha,
        "v2_core_score": _copy(v2_core), "v2_core_score_sha256": _sha(_json_bytes(v2_core)),
        "repair": "exact_physical_backend_PUBLIC_SPEC_object", "seed42_gate_not_revised": True,
    }


def validate_score_payload(value: Mapping[str, object], *, identity: Seed43V3Identity, input_sha: str) -> None:
    expected = {"schema", "status", "cell", "phase", "seed", "identity", "input_replay_sha256", "v2_core_score", "v2_core_score_sha256", "repair", "seed42_gate_not_revised"}
    if (
        not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != "tfsr_seed43_phase_e_v3_score_v1"
        or value.get("status") != "REPLICATION_SCORE_COMPLETE" or value.get("cell") != CELL or value.get("phase") != PHASE
        or value.get("seed") != SEED or value.get("identity") != identity.payload() or value.get("input_replay_sha256") != input_sha
        or not isinstance(value.get("v2_core_score"), Mapping)
        or value.get("repair") != "exact_physical_backend_PUBLIC_SPEC_object" or value.get("seed42_gate_not_revised") is not True
    ):
        raise FailClosedError("seed43 V3 score schema/binding drift")
    core = value["v2_core_score"]
    if value.get("v2_core_score_sha256") != _sha(_json_bytes(core)):
        raise FailClosedError("seed43 V3 nested score SHA drift")
    _v2().validate_score_payload(core, identity=identity.v2_identity, input_sha=input_sha)


def _terminal_payload(identity: Seed43V3Identity, *, attempt_sha: str, input_sha: str, score_sha: str,
                      final_closure: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": "tfsr_seed43_phase_e_v3_terminal_v1", "status": "REPLICATION_SCORE_COMPLETE", "cell": CELL,
        "phase": PHASE, "seed": SEED, "identity": identity.payload(), "attempt_sha256": attempt_sha,
        "input_replay_sha256": input_sha, "score_sha256": score_sha, "launch_closure": _copy(identity.closure),
        "final_closure": _copy(final_closure), "repair": "exact_physical_backend_PUBLIC_SPEC_object",
        "boundaries": {"target_optimizer_steps": 0, "backward_calls": 0, "optimizer_calls": 0,
                       "formal_resolved": False, "formal_opened": False, "cell_d_rerun": False,
                       "terminal_after_revalidation": True},
    }


def validate_terminal_payload(value: Mapping[str, object], *, identity: Seed43V3Identity,
                              score: Mapping[str, object], expected_score_sha: str | None = None) -> None:
    expected = {"schema", "status", "cell", "phase", "seed", "identity", "attempt_sha256", "input_replay_sha256", "score_sha256", "launch_closure", "final_closure", "repair", "boundaries"}
    if (
        not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != "tfsr_seed43_phase_e_v3_terminal_v1"
        or value.get("status") != "REPLICATION_SCORE_COMPLETE" or value.get("cell") != CELL or value.get("phase") != PHASE
        or value.get("seed") != SEED or value.get("identity") != identity.payload()
        or value.get("launch_closure") != identity.closure or value.get("final_closure") != identity.closure
        or value.get("repair") != "exact_physical_backend_PUBLIC_SPEC_object"
    ):
        raise FailClosedError("seed43 V3 terminal schema/binding drift")
    for field in ("attempt_sha256", "input_replay_sha256", "score_sha256"):
        _require_sha(value.get(field), field)
    if expected_score_sha is not None and value.get("score_sha256") != expected_score_sha:
        raise FailClosedError("seed43 V3 terminal score SHA drift")
    if value.get("boundaries") != {"target_optimizer_steps": 0, "backward_calls": 0, "optimizer_calls": 0,
                                    "formal_resolved": False, "formal_opened": False, "cell_d_rerun": False,
                                    "terminal_after_revalidation": True}:
        raise FailClosedError("seed43 V3 terminal boundary drift")
    validate_score_payload(score, identity=identity, input_sha=str(value["input_replay_sha256"]))


def _failure_payload(identity: Seed43V3Identity, *, attempt_sha: str, input_sha: str | None, flags: Any) -> dict[str, object]:
    return {
        "schema": "tfsr_seed43_phase_e_v3_failure_v1", "cell": CELL, "phase": PHASE, "seed": SEED,
        "identity": identity.payload(), "attempt_sha256": attempt_sha, "input_replay_sha256": input_sha,
        "stage": flags.stage, **flags.payload(), "cell_d_rerun": False,
        "traceback_sha256": _sha(traceback.format_exc().encode("utf-8")),
    }


def validate_failure_payload(value: Mapping[str, object], *, identity: Seed43V3Identity, attempt_sha: str,
                             input_sha: str | None) -> None:
    expected = {"schema", "cell", "phase", "seed", "identity", "attempt_sha256", "input_replay_sha256", "stage", "resolved", "opened", "tfsr_forward_calls", "backward_calls", "optimizer_calls", "terminal_published", "cell_d_rerun", "traceback_sha256"}
    if (
        not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != "tfsr_seed43_phase_e_v3_failure_v1"
        or value.get("cell") != CELL or value.get("phase") != PHASE or value.get("seed") != SEED
        or value.get("identity") != identity.payload() or value.get("attempt_sha256") != attempt_sha
        or value.get("input_replay_sha256") != input_sha or value.get("terminal_published") is not False
        or value.get("cell_d_rerun") is not False
    ):
        raise FailClosedError("seed43 V3 failure schema drift")
    _require_sha(value.get("traceback_sha256"), "seed43 V3 failure trace")
    flags = _v2()._seed43_v1().Score43Flags(
        stage=str(value.get("stage")),
        source_resolved=bool(value.get("resolved", {}).get("source")) if isinstance(value.get("resolved"), Mapping) else False,
        within_resolved=bool(value.get("resolved", {}).get("within")) if isinstance(value.get("resolved"), Mapping) else False,
        external_resolved=bool(value.get("resolved", {}).get("external")) if isinstance(value.get("resolved"), Mapping) else False,
        formal_resolved=bool(value.get("resolved", {}).get("formal")) if isinstance(value.get("resolved"), Mapping) else False,
        source_opened=bool(value.get("opened", {}).get("source")) if isinstance(value.get("opened"), Mapping) else False,
        within_opened=bool(value.get("opened", {}).get("within")) if isinstance(value.get("opened"), Mapping) else False,
        external_opened=bool(value.get("opened", {}).get("external")) if isinstance(value.get("opened"), Mapping) else False,
        formal_opened=bool(value.get("opened", {}).get("formal")) if isinstance(value.get("opened"), Mapping) else False,
        forward_calls=_copy(value.get("tfsr_forward_calls")) if isinstance(value.get("tfsr_forward_calls"), Mapping) else {},
        backward_calls=value.get("backward_calls"), optimizer_calls=value.get("optimizer_calls"), terminal_published=False,
    )
    if flags.payload() != {"stage": value["stage"], "resolved": value["resolved"], "opened": value["opened"],
                           "tfsr_forward_calls": value["tfsr_forward_calls"], "backward_calls": value["backward_calls"],
                           "optimizer_calls": value["optimizer_calls"], "terminal_published": False}:
        raise FailClosedError("seed43 V3 failure flags drift")


def _publish_failure(artifact: Any, identity: Seed43V3Identity, *, attempt_sha: str, input_sha: str | None, flags: Any) -> None:
    if flags.terminal_published or artifact.has_name("terminal.json") or artifact.has_name("failure.json"):
        return
    payload = _failure_payload(identity, attempt_sha=attempt_sha, input_sha=input_sha, flags=flags)
    validate_failure_payload(payload, identity=identity, attempt_sha=attempt_sha, input_sha=input_sha)
    digest = artifact.publish_json("failure.json", payload)
    validate_failure_payload(artifact.reload_json("failure.json", digest), identity=identity, attempt_sha=attempt_sha, input_sha=input_sha)


class Backend(Protocol):
    def resolve_inputs(self, **kwargs: Any) -> Any: ...
    def score_tfsr(self, **kwargs: Any) -> Any: ...
    def build_parity_manifest(self, **kwargs: Any) -> Mapping[str, object]: ...
    def reverify_after_forwards(self, **kwargs: Any) -> None: ...
    def resource_disclosure(self) -> Mapping[str, object]: ...
    def close(self) -> None: ...


def build_physical_backend(*, root: Path, fixed_authorities: Mapping[str, Any], upstream_training: Any,
                           upstream_authorization: Any, seed43_training: Any) -> Any:
    """The sole physical seam: exact frozen V1 seed43 backend composition."""
    return _v2().build_physical_backend(
        root=root, fixed_authorities=fixed_authorities, upstream_training=upstream_training,
        upstream_authorization=upstream_authorization, seed43_training=seed43_training,
    )


def run_seed43_v3_lifecycle(
    *, artifact: Any, identity: Seed43V3Identity, execution_capability: ExecutionCapability,
    within_roster: tuple[str, ...], external_sessions: tuple[str, ...],
    within_assets_factory: Callable[[], tuple[Any, ...]], external_assets_factory: Callable[[], tuple[Any, ...]],
    backend: Backend, final_reverify: Callable[[], Mapping[str, object]],
    physical_score_module: ModuleType | None = None,
) -> Mapping[str, object]:
    """Run the unchanged six-cell core after the V3 exact-spec hand-off."""
    _require_execution_capability(execution_capability, identity)
    v2 = _v2(); s43 = v2._seed43_v1()
    score_module = physical_seed42_score_module() if physical_score_module is None else physical_score_module
    expected_spec = expected_physical_score_spec() if physical_score_module is None else score_module.PUBLIC_SPEC
    if expected_spec is not score_module.PUBLIC_SPEC:
        raise FailClosedError("V3 lifecycle did not receive backend-owned exact ScoreSpec object")
    flags = s43.Score43Flags()
    attempt_sha: str | None = None
    input_sha: str | None = None
    try:
        flags.stage = "attempt"
        attempt = _v3_attempt_payload(identity); validate_attempt_payload(attempt, identity)
        attempt_sha = artifact.publish_json("attempt.json", attempt)
        validate_attempt_payload(artifact.reload_json("attempt.json", attempt_sha), identity)
        flags.stage = "resolve_identical_seed42_v2_inputs_after_attempt"
        within_assets = within_assets_factory(); flags.within_resolved = True
        raw_external_assets = external_assets_factory(); flags.external_resolved = True
        if tuple(item.session for item in within_assets) != tuple(sorted(within_roster)):
            raise FailClosedError("seed43 V3 within roster drift after attempt")
        if tuple(item.session for item in raw_external_assets) != external_sessions:
            raise FailClosedError("seed43 V3 external roster drift after attempt")
        external_assets = tuple(score_module._as_evaluation_asset(item) for item in raw_external_assets)
        # This is the only V3 operational change: spec is the *same object*
        # exported by the module that owns the inherited physical backend.
        evidence = backend.resolve_inputs(spec=expected_spec, within_roster=within_roster, within_assets=within_assets,
                                          external_roster=external_assets, flags=flags)
        score_module.validate_input_authority_evidence(
            evidence, within_roster=within_roster, within_assets=within_assets, external_roster=external_assets
        )
        core_replay = s43._validate_input_replay(evidence, upstream=identity.upstream, flags=flags, s42=score_module)
        v2_replay = v2._input_replay_payload(identity.v2_identity, core_replay)
        v2.validate_input_replay_payload(v2_replay, identity=identity.v2_identity)
        payload = _input_replay_payload(identity, v2_replay); validate_input_replay_payload(payload, identity=identity)
        input_sha = artifact.publish_json("input_replay.json", payload)
        validate_input_replay_payload(artifact.reload_json("input_replay.json", input_sha), identity=identity)
        modes: list[Any] = []
        for surface, roster in (("within", within_roster), ("external", external_sessions)):
            for mode in ("aligned", "zero", "wrong_pair"):
                flags.stage = f"seed43_tfsr_{surface}_{mode}"
                item = backend.score_tfsr(surface=surface, mode=mode, input_authority_sha256=input_sha, flags=flags)
                score_module._validate_mode_against_roster(
                    item, system="tfsr", surface=surface, mode=mode, expected_sessions=roster,
                    input_authority_sha256=input_sha,
                )
                modes.append(item)
        if flags.formal_resolved or flags.formal_opened or flags.backward_calls or flags.optimizer_calls:
            raise FailClosedError("seed43 V3 forward path crossed forbidden boundary")
        flags.stage = "post_forward_reverify"; backend.reverify_after_forwards(flags=flags)
        parity = backend.build_parity_manifest(flags=flags); s43.validate_parity_manifest(parity, flags)
        flags.stage = "score"
        core_score = s43.build_score_payload(
            identity=v2._core_identity(identity.v2_identity), input_replay_sha256=input_sha,
            tfsr_evidence=modes, parity_manifest=parity, resources=backend.resource_disclosure(), flags=flags,
        )
        s43.validate_score_payload(core_score, identity=v2._core_identity(identity.v2_identity), input_replay_sha256=input_sha)
        v2_score = v2._score_payload(identity.v2_identity, input_sha, core_score)
        v2.validate_score_payload(v2_score, identity=identity.v2_identity, input_sha=input_sha)
        score = _score_payload(identity, input_sha, v2_score); validate_score_payload(score, identity=identity, input_sha=input_sha)
        score_body = _json_bytes(score); score_sha = _sha(score_body)
        flags.stage = "terminal_revalidation"; final_closure = final_reverify()
        if validate_phase_e_seed43_v3_closure(final_closure) != identity.closure:
            raise FailClosedError("seed43 V3 launch/final closure drift")
        terminal = _terminal_payload(identity, attempt_sha=attempt_sha, input_sha=input_sha, score_sha=score_sha,
                                     final_closure=final_closure)
        validate_terminal_payload(terminal, identity=identity, score=score, expected_score_sha=score_sha)
        terminal_body = _json_bytes(terminal)
        def validate_group(bodies: Mapping[str, bytes], digests: Mapping[str, str]) -> None:
            if bodies.get("score.json") != score_body or bodies.get("terminal.json") != terminal_body or digests.get("score.json") != score_sha:
                raise FailClosedError("seed43 V3 atomic score/terminal binding drift")
            loaded_score = json.loads(artifact.reload_pair("score.json", score_sha))
            loaded_terminal = json.loads(artifact.reload_pair("terminal.json", _sha(terminal_body)))
            if not isinstance(loaded_score, Mapping) or not isinstance(loaded_terminal, Mapping):
                raise FailClosedError("seed43 V3 atomic JSON root drift")
            validate_score_payload(loaded_score, identity=identity, input_sha=input_sha)
            validate_terminal_payload(loaded_terminal, identity=identity, score=loaded_score, expected_score_sha=score_sha)
        hashes = artifact.publish_group({"score.json": score_body, "terminal.json": terminal_body}, post_publish=validate_group)
        flags.terminal_published = True
        final_score = artifact.reload_json("score.json", hashes["score.json"])
        final_terminal = artifact.reload_json("terminal.json", hashes["terminal.json"])
        validate_score_payload(final_score, identity=identity, input_sha=input_sha)
        validate_terminal_payload(final_terminal, identity=identity, score=final_score, expected_score_sha=hashes["score.json"])
        if artifact.has_name("failure.json"):
            raise FailClosedError("seed43 V3 terminal cannot coexist with failure")
        return final_terminal
    except BaseException:
        if attempt_sha is not None:
            try:
                _publish_failure(artifact, identity, attempt_sha=attempt_sha, input_sha=input_sha, flags=flags)
            except BaseException:
                pass
        raise
    finally:
        backend.close()


def _prepare_execution(root: Path, *, environment: Mapping[str, str] | None = None,
                       predecessor_loader: Callable[[Path], FailedV2Predecessor] = load_failed_v2_predecessor,
                       authorization_loader: Callable[[Path, FailedV2Predecessor], Seed43V3Authorization] | None = None) -> tuple[Path, Path, FailedV2Predecessor, Seed43V3Authorization]:
    subc, subm = _correct_data_roots(environment)
    predecessor = predecessor_loader(root)
    authorization = (authorization_loader(root, predecessor) if authorization_loader is not None
                     else verify_seed43_v3_authorization(root, predecessor=predecessor))
    return subc, subm, predecessor, authorization


def execute_authorized_v3(root: Path) -> Mapping[str, object]:
    """Future GPU1 route; never called by the public dry CLI or tests."""
    subc, subm, predecessor, authorization = _prepare_execution(root)
    identity = Seed43V3Identity(predecessor=predecessor, closure=authorization.closure, authorization=authorization.payload())
    capability = _issue_execution_capability(authorization)
    score_module = physical_seed42_score_module()
    fixed = score_module.verify_fixed_authorities(root)
    source_training = score_module.validate_phase_d_training_terminal(root)
    source_authorization = score_module.verify_phase_e_authorization(root, source_training, fixed)
    nested = predecessor.upstream.nested_v1_identity
    if (nested.get("training_terminal_sha256") != source_training.terminal_sha256
            or nested.get("training_swa_sha256") != source_training.swa_sha256
            or nested.get("training_swa_state_digest") != source_training.swa_state_digest):
        raise FailClosedError("seed43 V3 nested seed42 lineage differs from inherited physical input backend")
    parent, name = _score_parent(root)
    artifact = score_module.reserve_artifact_root(parent, name, SCORE_TOPOLOGY)
    within = score_module.extract_within_roster(fixed["strict_manifest"].value or {})
    external = score_module.extract_external_sessions(fixed["external_asset_ledger"].value or {})
    backend = build_physical_backend(
        root=root, fixed_authorities=fixed, upstream_training=source_training,
        upstream_authorization=source_authorization, seed43_training=predecessor.training,
    )
    def within_factory() -> tuple[Any, ...]:
        return score_module.join_within_assets(source_authorization.preflight, subc)
    def external_factory() -> tuple[Any, ...]:
        return score_module.join_external_assets(
            fixed["external_asset_ledger"].value or {}, fixed["external_scope"].value or {}, subm
        )
    def final_reverify() -> Mapping[str, object]:
        final_predecessor = load_failed_v2_predecessor(root)
        final_authorization = verify_seed43_v3_authorization(root, predecessor=final_predecessor)
        if (final_predecessor.binding_payload() != predecessor.binding_payload()
                or final_authorization.payload() != authorization.payload()):
            raise FailClosedError("seed43 V3 predecessor/authorization changed after forwards")
        return phase_e_seed43_v3_closure(root)
    return run_seed43_v3_lifecycle(
        artifact=artifact, identity=identity, execution_capability=capability,
        within_roster=within, external_sessions=external,
        within_assets_factory=within_factory, external_assets_factory=external_factory,
        backend=backend, final_reverify=final_reverify, physical_score_module=score_module,
    )


def dry_plan() -> dict[str, object]:
    return {
        "cell": CELL, "phase": PHASE, "seed": SEED,
        "status": "NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_SCORE",
        "authorization": "none", "execution_flags_required_together": sorted(PUBLIC_FLAGS),
        "authority_root": AUTHORITY_ROOT_RELATIVE, "score_root": SCORE_ROOT_RELATIVE,
        "predecessor": {
            "v2_authority_pairs": {"official_preflight_sha256": V2_AUTHORITY_PREFLIGHT_SHA256,
                                    "root_authorization_sha256": V2_AUTHORITY_ROOT_AUTHORIZATION_SHA256},
            "v2_failed_pairs": {"attempt_sha256": V2_FAILED_ATTEMPT_SHA256,
                                "failure_sha256": V2_FAILED_FAILURE_SHA256},
        },
        "repair": "pass_exact_backend_owned_PUBLIC_SPEC_object; payload_equality_is_not_substituted",
        "semantics": "inherited_seed43_v1_GPU1_six_cells_last_bin_equal_session_metric; no_Cell-D_rerun",
        "correct_roots": {"SUBC_DATA_ROOT": SUBC_DATA_ROOT, "SUBM_DATA_ROOT": SUBM_DATA_ROOT},
    }
