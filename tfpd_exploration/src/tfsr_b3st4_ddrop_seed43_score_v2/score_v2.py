"""Seed43 Phase-E V2: reuse completed seed42 V2 evidence without copying V1.

This module is deliberately a thin receipt/provenance wrapper.  It composes
the frozen seed43 V1 physical backend and its score helpers, while replacing
only the failed V1 upstream-terminal assumption with the descriptor-validated
seed42 V2 outer graph and its validated nested V1 input/score evidence.
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


CELL = "TFSR_B3ST4_DDROP_SEED43"
PHASE = "TFSR_PHASE_E_SEED43_REPLICATION_ADDENDUM_V2"
SEED = 43
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_TFSR_SEED43_PHASE_E_V2_20260823.md"
WORKORDER_SHA256 = "6035edd5c3d3e47d06148ff8928780a16f1d4584e43ce857750393636562e252"
SOURCE_RELATIVE = "tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_score_v2/score_v2.py"
INIT_RELATIVE = "tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_score_v2/__init__.py"
CLI_RELATIVE = "tfpd_exploration/scripts/run_tfsr_b3st4_ddrop_seed43_score_v2.py"
TEST_RELATIVE = "tfpd_exploration/tests/test_tfsr_b3st4_ddrop_seed43_score_v2.py"

AUTHORITY_ROOT_RELATIVE = "tfpd_exploration/results/tfsr_b3st4_ddrop_seed43_score_authority_v2"
SCORE_ROOT_RELATIVE = "tfpd_exploration/results/tfsr_b3st4_ddrop_seed43_matched_score_v2"
AUTHORITY_TOPOLOGY = ("official_preflight.json", "root_authorization.json")
SCORE_TOPOLOGY = ("attempt.json", "input_replay.json", "score.json", "terminal.json", "failure.json")
PUBLIC_FLAGS = frozenset(("--execute", "--i-have-seed43-phase-e-v2-authorization"))

SUBC_DATA_ROOT = "/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-C"
SUBM_DATA_ROOT = "/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-M"

SEED42_V2_ATTEMPT_SHA256 = "d1e97ba3c3d0dae077b4482b244d282cfe57722fdbccaeaf9816b8c56937ea03"
SEED42_V2_INPUT_SHA256 = "8d00e184c3ccd0cfde152a567c779e86f2724198388ff8751878e342a47b963d"
SEED42_V2_SCORE_SHA256 = "8551ddf2fe437c9bd37e4767991bec5f8d9328778ad4c883afcbac3fd185a9dd"
SEED42_V2_TERMINAL_SHA256 = "3007383ef0e4d2b8cc78aea262729fca876261a8bf25c300212e6cc573a2b579"

_S42_V2_NAME = "_tfsr_seed42_phase_e_v2_for_seed43_v2"
_S43_V1_NAME = "_tfsr_seed43_phase_e_v1_for_seed43_v2"
_S42_V2: ModuleType | None = None
_S43_V1: ModuleType | None = None


class FailClosedError(RuntimeError):
    """A V2 lineage, boundary, or receipt-integrity violation."""


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_static(*, package_name: str, directory: Path, leaf: str) -> ModuleType:
    present = sys.modules.get(package_name + "." + leaf.removesuffix(".py"))
    if isinstance(present, ModuleType):
        return present
    package = ModuleType(package_name)
    package.__path__ = [str(directory)]
    sys.modules[package_name] = package
    name = package_name + "." + leaf.removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, directory / leaf)
    if spec is None or spec.loader is None:
        raise RuntimeError("static successor module loader failure")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _seed42_v2() -> ModuleType:
    global _S42_V2
    if _S42_V2 is None:
        root = _repository_root()
        _S42_V2 = _load_static(
            package_name=_S42_V2_NAME,
            directory=root / "tfpd_exploration/src/tfsr_b3st4_ddrop_seed42_score_v2",
            leaf="score_v2.py",
        )
    return _S42_V2


def _seed43_v1() -> ModuleType:
    global _S43_V1
    if _S43_V1 is None:
        root = _repository_root()
        _S43_V1 = _load_static(
            package_name=_S43_V1_NAME,
            directory=root / "tfpd_exploration/src/tfsr_b3st4_ddrop_seed43_v1",
            leaf="score_43.py",
        )
    return _S43_V1


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")).encode("utf-8") + b"\n"


def _json_copy(value: object) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(item in "0123456789abcdef" for item in value)


def _require_sha(value: object, label: str) -> str:
    if not _is_sha(value):
        raise FailClosedError(f"{label} must be an exact lowercase SHA-256")
    return str(value)


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise FailClosedError("path must be canonical safe relative")
    return value


def _directory_identity(path: Path) -> tuple[int, int]:
    try:
        info = os.lstat(path)
    except OSError as error:
        raise FailClosedError("required immutable directory absent") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise FailClosedError("required immutable directory is noncanonical")
    return (info.st_dev, info.st_ino)


def _read_all(fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        item = os.read(fd, 1024 * 1024)
        if not item:
            return b"".join(chunks)
        chunks.append(item)


def _read_0444_fd(directory_fd: int, name: str) -> bytes:
    if not isinstance(name, str) or "/" in name or name in {"", ".", ".."}:
        raise FailClosedError("immutable score leaf name drift")
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    except OSError as error:
        raise FailClosedError("immutable score leaf missing") from error
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444:
            raise FailClosedError("immutable score leaf type/mode drift")
        return _read_all(fd)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class Seed42V2Expectation:
    attempt_sha256: str = SEED42_V2_ATTEMPT_SHA256
    input_sha256: str = SEED42_V2_INPUT_SHA256
    score_sha256: str = SEED42_V2_SCORE_SHA256
    terminal_sha256: str = SEED42_V2_TERMINAL_SHA256

    def __post_init__(self) -> None:
        for label, value in (("seed42 V2 attempt", self.attempt_sha256), ("seed42 V2 input", self.input_sha256),
                             ("seed42 V2 score", self.score_sha256), ("seed42 V2 terminal", self.terminal_sha256)):
            _require_sha(value, label)

    def payload(self) -> dict[str, str]:
        return {
            "attempt_sha256": self.attempt_sha256,
            "input_authority_sha256": self.input_sha256,
            "score_sha256": self.score_sha256,
            "terminal_sha256": self.terminal_sha256,
        }


@dataclass(frozen=True)
class Seed42V2Evidence:
    """Outer V2 graph plus validated nested V1 evidence used by seed43 core."""

    expectation: Seed42V2Expectation
    score_root_relative: str
    score_root_identity: tuple[int, int]
    v2_identity: Mapping[str, object]
    v2_closure: Mapping[str, object]
    attempt_payload: Mapping[str, object]
    outer_input_payload: Mapping[str, object]
    outer_score_payload: Mapping[str, object]
    outer_terminal_payload: Mapping[str, object]
    nested_v1_identity: Mapping[str, object]
    nested_v1_input_payload: Mapping[str, object]
    nested_v1_input_sha256: str
    nested_v1_score_payload: Mapping[str, object]
    nested_v1_score_sha256: str
    verdict: str

    def __post_init__(self) -> None:
        if self.verdict != "STOP":
            raise ValueError("seed42 V2 terminal must retain its exact STOP verdict")
        if len(self.score_root_identity) != 2 or any(type(item) is not int or item < 0 for item in self.score_root_identity):
            raise ValueError("seed42 V2 score-root identity drift")
        _safe_relative(self.score_root_relative)
        for value in (self.nested_v1_input_sha256, self.nested_v1_score_sha256):
            _require_sha(value, "nested V1 evidence")

    # Duck-typed interface intentionally consumed by the frozen seed43 V1
    # helpers.  Its binding makes the outer V2 and nested V1 identities both
    # visible; it is not mislabeled as a historical V1 terminal.
    @property
    def input_authority_sha256(self) -> str:
        return self.nested_v1_input_sha256

    @property
    def score_sha256(self) -> str:
        return self.nested_v1_score_sha256

    @property
    def terminal_sha256(self) -> str:
        return self.expectation.terminal_sha256

    @property
    def input_payload(self) -> Mapping[str, object]:
        return self.nested_v1_input_payload

    @property
    def score_payload(self) -> Mapping[str, object]:
        return self.nested_v1_score_payload

    @property
    def terminal_payload(self) -> Mapping[str, object]:
        return self.outer_terminal_payload

    @property
    def build_label(self) -> str:
        return "seed42_phase_e_v2_wrapped_v1_evidence"

    def binding_payload(self) -> dict[str, object]:
        return {
            "schema": "tfsr_seed43_upstream_seed42_phase_e_v2_binding_v1",
            "seed42_v2_score_root_relative": self.score_root_relative,
            "seed42_v2_score_root_identity": list(self.score_root_identity),
            "seed42_v2_pairs": self.expectation.payload(),
            "seed42_v2_closure_sha256": self.v2_closure.get("closure_sha256"),
            "nested_v1_identity": _json_copy(self.nested_v1_identity),
            "nested_v1_input_authority_sha256": self.nested_v1_input_sha256,
            "nested_v1_score_sha256": self.nested_v1_score_sha256,
            "seed42_v2_verdict": self.verdict,
            "cell_d_replay_required": True,
            "upstream_kind": "completed_seed42_phase_e_v2_wrapper",
        }


# Dataclasses cannot use a property with the same field name.  Keep the public
# V1-shaped attribute under a method-independent alias in the factory below.


def _v2_identity_from_terminal(s42v2: ModuleType, terminal: Mapping[str, object]) -> tuple[Any, Any]:
    raw = terminal.get("identity")
    if not isinstance(raw, Mapping) or set(raw) != {"v1_identity", "v1_failed_lineage", "v2_authorization", "launch_closure"}:
        raise FailClosedError("seed42 V2 terminal identity schema drift")
    try:
        identity = s42v2.V2ScoreIdentity(
            v1_identity=raw["v1_identity"], v1_failed_lineage=raw["v1_failed_lineage"],
            v2_authorization=raw["v2_authorization"], launch_closure=raw["launch_closure"],
        )
        v1 = s42v2._v1()
        nested = raw["v1_identity"]
        v1_identity = v1.ScoreIdentity(
            fixed_authorities=nested["fixed_authorities"], training_terminal_sha256=nested["training_terminal_sha256"],
            training_swa_sha256=nested["training_swa_sha256"],
            training_swa_state_digest=nested["training_swa_state_digest"], launch_closure=nested["launch_closure"],
            phase_e_authorization=nested["phase_e_authorization"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FailClosedError("seed42 V2 nested identity cannot be reconstructed") from error
    return identity, v1_identity


def validate_seed42_v2_payload_graph(
    s42v2: ModuleType,
    *,
    attempt_payload: Mapping[str, object],
    input_payload: Mapping[str, object],
    score_payload: Mapping[str, object],
    terminal_payload: Mapping[str, object],
    expectation: Seed42V2Expectation,
    score_root_relative: str,
    score_root_identity: tuple[int, int],
    current_closure: Mapping[str, object],
) -> Seed42V2Evidence:
    """Validate V2 outer receipts and their V1 nested scientific evidence."""
    for payload, digest, label in (
        (attempt_payload, expectation.attempt_sha256, "attempt"),
        (input_payload, expectation.input_sha256, "input"),
        (score_payload, expectation.score_sha256, "score"),
        (terminal_payload, expectation.terminal_sha256, "terminal"),
    ):
        if _sha(_json_bytes(payload)) != digest:
            raise FailClosedError(f"seed42 V2 {label} payload/body SHA drift")
    identity, v1_identity = _v2_identity_from_terminal(s42v2, terminal_payload)
    try:
        s42v2.validate_v2_attempt_payload(attempt_payload, identity)
        s42v2.validate_v2_input_payload(input_payload, identity, v1_identity)
        s42v2.validate_v2_score_payload(score_payload, identity=identity, input_sha=expectation.input_sha256,
                                         v1_identity=v1_identity)
        s42v2.validate_v2_terminal_payload(terminal_payload, identity=identity, score=score_payload,
                                            expected_score_sha=expectation.score_sha256)
    except Exception as error:
        raise FailClosedError("seed42 V2 outer graph schema/binding drift") from error
    if terminal_payload.get("attempt_sha256") != expectation.attempt_sha256 or terminal_payload.get("input_authority_sha256") != expectation.input_sha256:
        raise FailClosedError("seed42 V2 terminal predecessor receipt binding drift")
    if terminal_payload.get("verdict") != "STOP":
        raise FailClosedError("seed42 V2 terminal verdict drift")
    if identity.launch_closure != current_closure or terminal_payload.get("launch_closure") != current_closure:
        raise FailClosedError("seed42 V2 current closure differs from terminal")
    if terminal_payload.get("final_closure") != current_closure:
        raise FailClosedError("seed42 V2 launch/final closure drift")
    nested_input = input_payload.get("v1_input_authority")
    nested_score = score_payload.get("v1_score")
    nested_input_sha = input_payload.get("v1_input_authority_sha256")
    nested_score_sha = score_payload.get("v1_score_sha256")
    if not isinstance(nested_input, Mapping) or not isinstance(nested_score, Mapping):
        raise FailClosedError("seed42 V2 nested V1 evidence missing")
    v1 = s42v2._v1()
    if (_sha(v1._json_bytes(nested_input)) != nested_input_sha
            or _sha(v1._json_bytes(nested_score)) != nested_score_sha):
        raise FailClosedError("seed42 V2 nested V1 evidence SHA drift")
    try:
        v1.validate_input_authority_payload(nested_input, v1_identity)
        v1.validate_score_payload(nested_score, identity=v1_identity, input_authority_sha256=expectation.input_sha256)
    except Exception as error:
        raise FailClosedError("seed42 V2 nested V1 evidence schema drift") from error
    # Avoid the historical V1 terminal fiction: V2 has no V1 terminal to
    # reconstruct.  The evidence keeps actual V2 terminal bytes separately.
    evidence = Seed42V2Evidence(
        expectation=expectation, score_root_relative=score_root_relative, score_root_identity=score_root_identity,
        v2_identity=identity.payload(), v2_closure=_json_copy(current_closure),
        attempt_payload=_json_copy(attempt_payload), outer_input_payload=_json_copy(input_payload),
        outer_score_payload=_json_copy(score_payload), outer_terminal_payload=_json_copy(terminal_payload),
        nested_v1_identity=_json_copy(v1_identity.payload()), nested_v1_input_payload=_json_copy(nested_input),
        nested_v1_input_sha256=str(nested_input_sha), nested_v1_score_payload=_json_copy(nested_score),
        nested_v1_score_sha256=str(nested_score_sha), verdict="STOP",
    )
    return evidence


def _read_seed42_v2_pairs(root: Path, *, expectation: Seed42V2Expectation) -> tuple[dict[str, Mapping[str, object]], tuple[int, int]]:
    s42v2 = _seed42_v2()
    directory = root.absolute() / s42v2.SCORE_ROOT_RELATIVE
    initial = _directory_identity(directory)
    expected_leaves = {
        "attempt.json", "attempt.json.sha256", "input_authority.json", "input_authority.json.sha256",
        "score.json", "score.json.sha256", "terminal.json", "terminal.json.sha256",
    }
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != initial:
            raise FailClosedError("seed42 V2 score-root identity changed while opening")
        if set(os.listdir(fd)) != expected_leaves:
            raise FailClosedError("seed42 V2 completed topology must be exactly four receipt pairs")
        result: dict[str, Mapping[str, object]] = {}
        for leaf, digest in (("attempt.json", expectation.attempt_sha256), ("input_authority.json", expectation.input_sha256),
                             ("score.json", expectation.score_sha256), ("terminal.json", expectation.terminal_sha256)):
            body = _read_0444_fd(fd, leaf)
            if _sha(body) != digest:
                raise FailClosedError("seed42 V2 immutable pair body SHA drift")
            if _read_0444_fd(fd, leaf + ".sha256") != f"{digest}  {leaf}\n".encode("ascii"):
                raise FailClosedError("seed42 V2 immutable pair sidecar drift")
            try:
                value = json.loads(body)
            except (TypeError, json.JSONDecodeError) as error:
                raise FailClosedError("seed42 V2 immutable pair JSON malformed") from error
            if not isinstance(value, Mapping):
                raise FailClosedError("seed42 V2 immutable pair JSON root drift")
            result[leaf] = value
        if _directory_identity(directory) != initial:
            raise FailClosedError("seed42 V2 score-root identity changed during read")
        return result, initial
    finally:
        os.close(fd)


def load_completed_seed42_v2(root: Path, *, expectation: Seed42V2Expectation = Seed42V2Expectation()) -> Seed42V2Evidence:
    """Descriptor-load exact V2 completed graph before any seed43 route exists."""
    pairs, identity = _read_seed42_v2_pairs(root, expectation=expectation)
    s42v2 = _seed42_v2()
    closure = s42v2.phase_e_v2_closure(root)
    return validate_seed42_v2_payload_graph(
        s42v2, attempt_payload=pairs["attempt.json"], input_payload=pairs["input_authority.json"],
        score_payload=pairs["score.json"], terminal_payload=pairs["terminal.json"], expectation=expectation,
        score_root_relative=s42v2.SCORE_ROOT_RELATIVE, score_root_identity=identity, current_closure=closure,
    )


class RootPublicationCapability:
    __slots__ = ("_seal",)
    def __init__(self, seal: object) -> None:
        if seal is not _ROOT_SEAL:
            raise TypeError("seed43 V2 root publication capability is internal")
        self._seal = seal


class ExecutionCapability:
    __slots__ = ("authorization", "_seal")
    def __init__(self, authorization: "Seed43V2Authorization", seal: object) -> None:
        if seal is not _EXECUTION_SEAL:
            raise TypeError("seed43 V2 execution capability is internal")
        self.authorization = authorization
        self._seal = seal


_ROOT_SEAL = object()
_EXECUTION_SEAL = object()


def issue_root_publication_capability() -> RootPublicationCapability:
    return RootPublicationCapability(_ROOT_SEAL)


def _issue_execution_capability(authorization: "Seed43V2Authorization") -> ExecutionCapability:
    return ExecutionCapability(authorization, _EXECUTION_SEAL)


def _require_execution_capability(capability: object, identity: "Seed43V2Identity") -> None:
    if not isinstance(capability, ExecutionCapability) or capability._seal is not _EXECUTION_SEAL:
        raise FailClosedError("seed43 V2 requires reviewed in-process execution capability")
    if capability.authorization.payload() != identity.authorization:
        raise FailClosedError("seed43 V2 capability/identity authorization drift")


def _closure_paths() -> tuple[str, ...]:
    s42v2 = _seed42_v2()
    s43 = _seed43_v1()
    v1 = s42v2._v1()
    inherited_seed43_chain = (
        s43.WORKORDER_RELATIVE, s43.SOURCE_RELATIVE, s43.CLI_RELATIVE, s43.TEST_RELATIVE,
        s43.SEED43_INIT_RELATIVE, s43.SEED43_CONTRACT_RELATIVE, s43.SEED43_TRAIN_RELATIVE,
        s43.SEED43_ACCELERATED_RELATIVE, s43.FROZEN_TRAIN42_RELATIVE,
        "tfpd_exploration/src/tfsr_b3st4_ddrop_v1/throughput_benchmark_v2.py",
        "tfpd_exploration/src/tfpd_lane/arm_common.py", *tuple(v1.PHASE_E_CLOSURE),
    )
    paths = (WORKORDER_RELATIVE, INIT_RELATIVE, SOURCE_RELATIVE, CLI_RELATIVE, TEST_RELATIVE,
             *inherited_seed43_chain, *tuple(s42v2.V2_CLOSURE))
    return tuple(dict.fromkeys(paths))


V2_CLOSURE = _closure_paths()


def phase_e_seed43_v2_closure(root: Path) -> dict[str, object]:
    s42v2 = _seed42_v2(); v1 = s42v2._v1()
    hashes: dict[str, str] = {}
    for relative in V2_CLOSURE:
        body, _ = v1._canonical_regular_bytes(root.absolute(), relative, expected_mode=0o664)
        hashes[relative] = _sha(body)
    if hashes.get(WORKORDER_RELATIVE) != WORKORDER_SHA256:
        raise FailClosedError("seed43 V2 workorder SHA drift")
    upstream_closure = s42v2.phase_e_v2_closure(root)
    encoded = json.dumps({"paths": list(V2_CLOSURE), "sha256_by_path": hashes,
                          "seed42_v2_closure": upstream_closure}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"paths": list(V2_CLOSURE), "sha256_by_path": hashes, "seed42_v2_closure": upstream_closure,
            "closure_sha256": _sha(encoded)}


def validate_phase_e_seed43_v2_closure(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"paths", "sha256_by_path", "seed42_v2_closure", "closure_sha256"}:
        raise FailClosedError("seed43 V2 closure schema drift")
    hashes = value.get("sha256_by_path")
    if value.get("paths") != list(V2_CLOSURE) or not isinstance(hashes, Mapping) or set(hashes) != set(V2_CLOSURE):
        raise FailClosedError("seed43 V2 closure topology drift")
    if any(not _is_sha(hashes.get(path)) for path in V2_CLOSURE) or hashes.get(WORKORDER_RELATIVE) != WORKORDER_SHA256:
        raise FailClosedError("seed43 V2 closure leaf drift")
    s42v2 = _seed42_v2()
    try:
        upstream = s42v2._validate_v2_closure(value.get("seed42_v2_closure"))
    except Exception as error:
        raise FailClosedError("seed43 V2 nested seed42 closure drift") from error
    encoded = json.dumps({"paths": list(V2_CLOSURE), "sha256_by_path": dict(hashes),
                          "seed42_v2_closure": upstream}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if value.get("closure_sha256") != _sha(encoded):
        raise FailClosedError("seed43 V2 closure aggregate drift")
    return {"paths": list(V2_CLOSURE), "sha256_by_path": {path: str(hashes[path]) for path in V2_CLOSURE},
            "seed42_v2_closure": upstream, "closure_sha256": str(value["closure_sha256"])}


def _correct_data_roots(environment: Mapping[str, str] | None = None) -> tuple[Path, Path]:
    env = os.environ if environment is None else environment
    if env.get("SUBC_DATA_ROOT") != SUBC_DATA_ROOT or env.get("SUBM_DATA_ROOT") != SUBM_DATA_ROOT:
        raise FailClosedError("seed43 V2 requires exact canonical SUBC_DATA_ROOT and SUBM_DATA_ROOT before reservation")
    return Path(SUBC_DATA_ROOT), Path(SUBM_DATA_ROOT)


@dataclass(frozen=True)
class Seed43V2Identity:
    training: Any
    upstream: Seed42V2Evidence
    closure: Mapping[str, object]
    authorization: Mapping[str, object]

    def payload(self) -> dict[str, object]:
        return {
            "seed43_training": self.training.payload(), "upstream_seed42_v2": self.upstream.binding_payload(),
            "closure": _json_copy(self.closure), "authorization": _json_copy(self.authorization),
        }


@dataclass(frozen=True)
class Seed43V2Authorization:
    preflight_sha256: str
    root_authorization_sha256: str
    preflight: Mapping[str, object]
    root_authorization: Mapping[str, object]
    closure: Mapping[str, object]

    def __post_init__(self) -> None:
        _require_sha(self.preflight_sha256, "seed43 V2 preflight")
        _require_sha(self.root_authorization_sha256, "seed43 V2 root authorization")

    def payload(self) -> dict[str, object]:
        return {"preflight_sha256": self.preflight_sha256, "root_authorization_sha256": self.root_authorization_sha256,
                "closure": _json_copy(self.closure)}


def _superseded_v1_code_evidence(closure: Mapping[str, object]) -> dict[str, object]:
    s43 = _seed43_v1()
    hashes = closure.get("sha256_by_path") if isinstance(closure, Mapping) else None
    if not isinstance(hashes, Mapping):
        raise FailClosedError("seed43 V2 closure lacks code hashes")
    return {
        "role": "superseded_code_evidence_only_no_result_required",
        "source_relative": s43.SOURCE_RELATIVE,
        "source_sha256": hashes.get(s43.SOURCE_RELATIVE),
        "cli_relative": s43.CLI_RELATIVE,
        "test_relative": s43.TEST_RELATIVE,
        "historical_score_root_relative": s43.SEED43_SCORE_ROOT_RELATIVE,
    }


def _authority_parent(root: Path) -> tuple[Path, str]:
    path = root.absolute() / _safe_relative(AUTHORITY_ROOT_RELATIVE)
    return path.parent, path.name


def _score_parent(root: Path) -> tuple[Path, str]:
    path = root.absolute() / _safe_relative(SCORE_ROOT_RELATIVE)
    return path.parent, path.name


def reserve_authority_artifact(
    root: Path, capability: RootPublicationCapability, *,
    upstream_loader: Callable[[Path], Seed42V2Evidence] = load_completed_seed42_v2,
    training_loader: Callable[[Path], Any] | None = None,
) -> Any:
    if not isinstance(capability, RootPublicationCapability) or capability._seal is not _ROOT_SEAL:
        raise FailClosedError("only root may reserve seed43 V2 authority root")
    upstream_loader(root)
    (training_loader or _seed43_v1().validate_seed43_training_terminal)(root)
    parent, name = _authority_parent(root)
    return _seed42_v2()._v1().reserve_artifact_root(parent, name, AUTHORITY_TOPOLOGY)


def build_target_free_preflight(
    root: Path, *, upstream: Seed42V2Evidence, training: Any, closure: Mapping[str, object],
) -> dict[str, object]:
    checked_closure = validate_phase_e_seed43_v2_closure(closure)
    s43 = _seed43_v1()
    if not isinstance(training, s43.Seed43TrainingEvidence):
        raise FailClosedError("seed43 V2 requires native completed seed43 training evidence")
    return {
        "schema": "tfsr_seed43_phase_e_v2_preflight_v1", "status": "PREFLIGHT_ACCEPTED", "cell": CELL,
        "phase": PHASE, "seed": SEED, "score_spec": _json_copy(s43.PUBLIC_SCORE_SPEC),
        "seed43_training": training.payload(), "upstream_seed42_v2": upstream.binding_payload(),
        "closure": checked_closure, "authority_root_relative": AUTHORITY_ROOT_RELATIVE,
        "score_root_relative": SCORE_ROOT_RELATIVE,
        "deployment_data_roots": {"SUBC_DATA_ROOT": SUBC_DATA_ROOT, "SUBM_DATA_ROOT": SUBM_DATA_ROOT},
        "target_free": True, "formal_sessions_inert": list(_seed42_v2()._v1().FORMAL_TEST_SESSION_NAMES),
        "metric_from_seed43_v1_core": {"governing": "last_bin_variance_weighted_r2_equal_session",
                                        "cell_d_replay": "nested_seed42_v2_v1_evidence_only_no_rerun",
                                        "modes": ["aligned", "zero", "wrong_pair"]},
        "superseded_seed43_v1_code_evidence": _superseded_v1_code_evidence(checked_closure),
    }


def validate_target_free_preflight(
    value: Mapping[str, object], *, upstream: Seed42V2Evidence, training: Any, closure: Mapping[str, object],
) -> dict[str, object]:
    s43 = _seed43_v1(); checked_closure = validate_phase_e_seed43_v2_closure(closure)
    expected = {"schema", "status", "cell", "phase", "seed", "score_spec", "seed43_training", "upstream_seed42_v2",
                "closure", "authority_root_relative", "score_root_relative", "deployment_data_roots", "target_free",
                "formal_sessions_inert", "metric_from_seed43_v1_core", "superseded_seed43_v1_code_evidence"}
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "tfsr_seed43_phase_e_v2_preflight_v1" or value.get("status") != "PREFLIGHT_ACCEPTED"
            or value.get("cell") != CELL or value.get("phase") != PHASE or value.get("seed") != SEED
            or value.get("score_spec") != s43.PUBLIC_SCORE_SPEC or value.get("seed43_training") != training.payload()
            or value.get("upstream_seed42_v2") != upstream.binding_payload() or value.get("closure") != checked_closure
            or value.get("authority_root_relative") != AUTHORITY_ROOT_RELATIVE or value.get("score_root_relative") != SCORE_ROOT_RELATIVE
            or value.get("deployment_data_roots") != {"SUBC_DATA_ROOT": SUBC_DATA_ROOT, "SUBM_DATA_ROOT": SUBM_DATA_ROOT}
            or value.get("target_free") is not True
            or value.get("formal_sessions_inert") != list(_seed42_v2()._v1().FORMAL_TEST_SESSION_NAMES)
            or value.get("metric_from_seed43_v1_core") != {"governing": "last_bin_variance_weighted_r2_equal_session",
                                                              "cell_d_replay": "nested_seed42_v2_v1_evidence_only_no_rerun",
                                                              "modes": ["aligned", "zero", "wrong_pair"]}
            or value.get("superseded_seed43_v1_code_evidence") != _superseded_v1_code_evidence(checked_closure)):
        raise FailClosedError("seed43 V2 preflight schema/binding drift")
    return _json_copy(value)


def build_root_authorization(*, preflight_sha256: str, preflight: Mapping[str, object]) -> dict[str, object]:
    _require_sha(preflight_sha256, "seed43 V2 preflight")
    return {
        "schema": "tfsr_seed43_phase_e_v2_root_authorization_v1", "status": "ROOT_AUTHORIZED", "cell": CELL,
        "phase": PHASE, "official_preflight_sha256": preflight_sha256,
        "authority_root_relative": AUTHORITY_ROOT_RELATIVE, "score_root_relative": SCORE_ROOT_RELATIVE,
        "closure": preflight.get("closure"), "upstream_seed42_v2": preflight.get("upstream_seed42_v2"),
        "seed43_terminal_required": True, "target_free_preflight_required": True,
    }


def validate_root_authorization(value: Mapping[str, object], *, preflight_sha256: str, preflight: Mapping[str, object]) -> dict[str, object]:
    _require_sha(preflight_sha256, "seed43 V2 preflight")
    expected = {"schema", "status", "cell", "phase", "official_preflight_sha256", "authority_root_relative",
                "score_root_relative", "closure", "upstream_seed42_v2", "seed43_terminal_required",
                "target_free_preflight_required"}
    if (not isinstance(value, Mapping) or set(value) != expected
            or value.get("schema") != "tfsr_seed43_phase_e_v2_root_authorization_v1"
            or value.get("status") != "ROOT_AUTHORIZED" or value.get("cell") != CELL or value.get("phase") != PHASE
            or value.get("official_preflight_sha256") != preflight_sha256
            or value.get("authority_root_relative") != AUTHORITY_ROOT_RELATIVE or value.get("score_root_relative") != SCORE_ROOT_RELATIVE
            or value.get("closure") != preflight.get("closure") or value.get("upstream_seed42_v2") != preflight.get("upstream_seed42_v2")
            or value.get("seed43_terminal_required") is not True or value.get("target_free_preflight_required") is not True):
        raise FailClosedError("seed43 V2 root authorization binding drift")
    return _json_copy(value)


def publish_target_free_preflight(
    artifact: Any, capability: RootPublicationCapability, payload: Mapping[str, object], *,
    upstream: Seed42V2Evidence, training: Any, closure: Mapping[str, object],
) -> str:
    if not isinstance(capability, RootPublicationCapability) or capability._seal is not _ROOT_SEAL:
        raise FailClosedError("only root may publish seed43 V2 preflight")
    return artifact.publish_json("official_preflight.json", validate_target_free_preflight(
        payload, upstream=upstream, training=training, closure=closure,
    ))


def publish_root_authorization(
    artifact: Any, capability: RootPublicationCapability, payload: Mapping[str, object], *,
    upstream: Seed42V2Evidence, training: Any, closure: Mapping[str, object],
) -> str:
    if not isinstance(capability, RootPublicationCapability) or capability._seal is not _ROOT_SEAL:
        raise FailClosedError("only root may publish seed43 V2 authorization")
    body = artifact.reload_pair("official_preflight.json")
    try:
        preflight = json.loads(body)
    except (TypeError, json.JSONDecodeError) as error:
        raise FailClosedError("durable seed43 V2 preflight JSON malformed") from error
    if not isinstance(preflight, Mapping):
        raise FailClosedError("durable seed43 V2 preflight JSON root drift")
    checked = validate_target_free_preflight(preflight, upstream=upstream, training=training, closure=closure)
    return artifact.publish_json("root_authorization.json", validate_root_authorization(
        payload, preflight_sha256=_sha(body), preflight=checked,
    ))


def _read_authority_pair(directory: Path, name: str) -> tuple[Mapping[str, object], str]:
    v1 = _seed42_v2()._v1()
    body, _ = v1._canonical_regular_bytes(directory, name, expected_mode=0o444)
    digest = _sha(body)
    sidecar, _ = v1._canonical_regular_bytes(directory, name + ".sha256", expected_mode=0o444)
    if sidecar != f"{digest}  {name}\n".encode("ascii"):
        raise FailClosedError("seed43 V2 authority sidecar drift")
    try:
        value = json.loads(body)
    except (TypeError, json.JSONDecodeError) as error:
        raise FailClosedError("seed43 V2 authority JSON malformed") from error
    if not isinstance(value, Mapping):
        raise FailClosedError("seed43 V2 authority JSON root drift")
    return value, digest


def verify_seed43_v2_authorization(
    root: Path, *, upstream: Seed42V2Evidence, training: Any,
    closure_loader: Callable[[Path], Mapping[str, object]] = phase_e_seed43_v2_closure,
) -> Seed43V2Authorization:
    closure = validate_phase_e_seed43_v2_closure(closure_loader(root))
    directory = root.absolute() / AUTHORITY_ROOT_RELATIVE
    preflight, preflight_sha = _read_authority_pair(directory, "official_preflight.json")
    checked = validate_target_free_preflight(preflight, upstream=upstream, training=training, closure=closure)
    authorization, auth_sha = _read_authority_pair(directory, "root_authorization.json")
    checked_auth = validate_root_authorization(authorization, preflight_sha256=preflight_sha, preflight=checked)
    return Seed43V2Authorization(preflight_sha, auth_sha, checked, checked_auth, closure)


def _core_identity(identity: Seed43V2Identity) -> Any:
    """Use the existing seed43 V1 score algorithms via a typed duck adapter."""
    s43 = _seed43_v1()
    return s43.Seed43ScoreIdentity(training=identity.training, upstream=identity.upstream,
                                   closure=identity.closure, authorization=identity.authorization)


def build_physical_backend(
    *, root: Path, fixed_authorities: Mapping[str, Any], upstream_training: Any,
    upstream_authorization: Any, seed43_training: Any,
) -> Any:
    """Narrow composition seam: exact V1 physical backend, no override."""
    return _seed43_v1().build_seed43_physical_backend(
        root=root, fixed_authorities=fixed_authorities, upstream_training=upstream_training,
        upstream_authorization=upstream_authorization, seed43_training=seed43_training,
    )


def _v2_attempt_payload(identity: Seed43V2Identity) -> dict[str, object]:
    return {"schema": "tfsr_seed43_phase_e_v2_attempt_v1", "cell": CELL, "phase": PHASE, "seed": SEED,
            "identity": identity.payload(), "topology": list(SCORE_TOPOLOGY),
            "resolved": {"source": False, "within": False, "external": False, "formal": False},
            "opened": {"source": False, "within": False, "external": False, "formal": False},
            "backward_calls": 0, "optimizer_calls": 0, "cell_d_rerun": False}


def validate_attempt_payload(value: Mapping[str, object], identity: Seed43V2Identity) -> None:
    expected = {"schema", "cell", "phase", "seed", "identity", "topology", "resolved", "opened",
                "backward_calls", "optimizer_calls", "cell_d_rerun"}
    if (not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != "tfsr_seed43_phase_e_v2_attempt_v1"
            or value.get("cell") != CELL or value.get("phase") != PHASE or value.get("seed") != SEED
            or value.get("identity") != identity.payload() or value.get("topology") != list(SCORE_TOPOLOGY)
            or value.get("resolved") != {"source": False, "within": False, "external": False, "formal": False}
            or value.get("opened") != {"source": False, "within": False, "external": False, "formal": False}
            or value.get("backward_calls") != 0 or value.get("optimizer_calls") != 0 or value.get("cell_d_rerun") is not False):
        raise FailClosedError("seed43 V2 attempt schema/boundary drift")


def _input_replay_payload(identity: Seed43V2Identity, core_replay: Mapping[str, object]) -> dict[str, object]:
    return {"schema": "tfsr_seed43_phase_e_v2_input_replay_v1", "cell": CELL, "phase": PHASE,
            "identity": identity.payload(), "upstream_seed42_v2": identity.upstream.binding_payload(),
            "seed43_v1_core_input_replay": _json_copy(core_replay),
            "seed43_v1_core_input_replay_sha256": _sha(_json_bytes(core_replay))}


def validate_input_replay_payload(value: Mapping[str, object], *, identity: Seed43V2Identity) -> None:
    expected = {"schema", "cell", "phase", "identity", "upstream_seed42_v2", "seed43_v1_core_input_replay",
                "seed43_v1_core_input_replay_sha256"}
    if (not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != "tfsr_seed43_phase_e_v2_input_replay_v1"
            or value.get("cell") != CELL or value.get("phase") != PHASE or value.get("identity") != identity.payload()
            or value.get("upstream_seed42_v2") != identity.upstream.binding_payload()
            or not isinstance(value.get("seed43_v1_core_input_replay"), Mapping)):
        raise FailClosedError("seed43 V2 input replay schema/identity drift")
    core = value["seed43_v1_core_input_replay"]
    if value.get("seed43_v1_core_input_replay_sha256") != _sha(_json_bytes(core)):
        raise FailClosedError("seed43 V2 input replay nested SHA drift")
    _seed43_v1().validate_input_replay_payload(core, upstream=identity.upstream)


def _score_payload(identity: Seed43V2Identity, input_sha: str, core_score: Mapping[str, object]) -> dict[str, object]:
    return {"schema": "tfsr_seed43_phase_e_v2_score_v1", "status": "REPLICATION_SCORE_COMPLETE", "cell": CELL,
            "phase": PHASE, "seed": SEED, "identity": identity.payload(), "input_replay_sha256": input_sha,
            "upstream_seed42_v2": identity.upstream.binding_payload(), "seed43_v1_core_score": _json_copy(core_score),
            "seed43_v1_core_score_sha256": _sha(_json_bytes(core_score)), "seed42_gate_not_revised": True,
            "superseded_seed43_v1_result_required": False}


def validate_score_payload(value: Mapping[str, object], *, identity: Seed43V2Identity, input_sha: str) -> None:
    expected = {"schema", "status", "cell", "phase", "seed", "identity", "input_replay_sha256", "upstream_seed42_v2",
                "seed43_v1_core_score", "seed43_v1_core_score_sha256", "seed42_gate_not_revised",
                "superseded_seed43_v1_result_required"}
    if (not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != "tfsr_seed43_phase_e_v2_score_v1"
            or value.get("status") != "REPLICATION_SCORE_COMPLETE" or value.get("cell") != CELL or value.get("phase") != PHASE
            or value.get("seed") != SEED or value.get("identity") != identity.payload() or value.get("input_replay_sha256") != input_sha
            or value.get("upstream_seed42_v2") != identity.upstream.binding_payload() or value.get("seed42_gate_not_revised") is not True
            or value.get("superseded_seed43_v1_result_required") is not False or not isinstance(value.get("seed43_v1_core_score"), Mapping)):
        raise FailClosedError("seed43 V2 score schema/binding drift")
    core = value["seed43_v1_core_score"]
    if value.get("seed43_v1_core_score_sha256") != _sha(_json_bytes(core)):
        raise FailClosedError("seed43 V2 nested core-score SHA drift")
    _seed43_v1().validate_score_payload(core, identity=_core_identity(identity), input_replay_sha256=input_sha)


def _terminal_payload(identity: Seed43V2Identity, *, attempt_sha: str, input_sha: str, score_sha: str,
                      score: Mapping[str, object], final_closure: Mapping[str, object]) -> dict[str, object]:
    core = score["seed43_v1_core_score"]
    return {"schema": "tfsr_seed43_phase_e_v2_terminal_v1", "status": "REPLICATION_SCORE_COMPLETE", "cell": CELL,
            "phase": PHASE, "seed": SEED, "identity": identity.payload(), "attempt_sha256": attempt_sha,
            "input_replay_sha256": input_sha, "score_sha256": score_sha, "launch_closure": _json_copy(identity.closure),
            "final_closure": _json_copy(final_closure), "seed42_v2_verdict": identity.upstream.verdict,
            "seed43_verdict": core.get("verdict"), "seed42_gate_not_revised": True,
            "superseded_seed43_v1_result_required": False,
            "boundaries": {"target_optimizer_steps": 0, "backward_calls": 0, "optimizer_calls": 0,
                           "formal_resolved": False, "formal_opened": False, "cell_d_rerun": False,
                           "terminal_after_revalidation": True}}


def validate_terminal_payload(value: Mapping[str, object], *, identity: Seed43V2Identity, score: Mapping[str, object],
                              expected_score_sha: str | None = None) -> None:
    expected = {"schema", "status", "cell", "phase", "seed", "identity", "attempt_sha256", "input_replay_sha256",
                "score_sha256", "launch_closure", "final_closure", "seed42_v2_verdict", "seed43_verdict",
                "seed42_gate_not_revised", "superseded_seed43_v1_result_required", "boundaries"}
    if (not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != "tfsr_seed43_phase_e_v2_terminal_v1"
            or value.get("status") != "REPLICATION_SCORE_COMPLETE" or value.get("cell") != CELL or value.get("phase") != PHASE
            or value.get("seed") != SEED or value.get("identity") != identity.payload()
            or value.get("launch_closure") != identity.closure or value.get("final_closure") != identity.closure
            or value.get("seed42_v2_verdict") != "STOP" or value.get("seed42_gate_not_revised") is not True
            or value.get("superseded_seed43_v1_result_required") is not False):
        raise FailClosedError("seed43 V2 terminal schema/binding drift")
    for field in ("attempt_sha256", "input_replay_sha256", "score_sha256"):
        _require_sha(value.get(field), field)
    if expected_score_sha is not None and value.get("score_sha256") != expected_score_sha:
        raise FailClosedError("seed43 V2 terminal score SHA drift")
    core = score.get("seed43_v1_core_score")
    if not isinstance(core, Mapping) or value.get("seed43_verdict") != core.get("verdict"):
        raise FailClosedError("seed43 V2 terminal core verdict drift")
    if value.get("boundaries") != {"target_optimizer_steps": 0, "backward_calls": 0, "optimizer_calls": 0,
                                    "formal_resolved": False, "formal_opened": False, "cell_d_rerun": False,
                                    "terminal_after_revalidation": True}:
        raise FailClosedError("seed43 V2 terminal boundary drift")


def _failure_payload(identity: Seed43V2Identity, attempt_sha: str, input_sha: str | None, flags: Any) -> dict[str, object]:
    return {"schema": "tfsr_seed43_phase_e_v2_failure_v1", "cell": CELL, "phase": PHASE, "seed": SEED,
            "identity": identity.payload(), "attempt_sha256": attempt_sha, "input_replay_sha256": input_sha,
            "stage": flags.stage, **flags.payload(), "cell_d_rerun": False,
            "traceback_sha256": _sha(traceback.format_exc().encode("utf-8"))}


def validate_failure_payload(value: Mapping[str, object], *, identity: Seed43V2Identity, attempt_sha: str,
                             input_sha: str | None) -> None:
    expected = {"schema", "cell", "phase", "seed", "identity", "attempt_sha256", "input_replay_sha256", "stage",
                "resolved", "opened", "tfsr_forward_calls", "backward_calls", "optimizer_calls", "terminal_published",
                "cell_d_rerun", "traceback_sha256"}
    if (not isinstance(value, Mapping) or set(value) != expected or value.get("schema") != "tfsr_seed43_phase_e_v2_failure_v1"
            or value.get("cell") != CELL or value.get("phase") != PHASE or value.get("seed") != SEED
            or value.get("identity") != identity.payload() or value.get("attempt_sha256") != attempt_sha
            or value.get("input_replay_sha256") != input_sha or value.get("terminal_published") is not False
            or value.get("cell_d_rerun") is not False):
        raise FailClosedError("seed43 V2 failure schema drift")
    _require_sha(value.get("traceback_sha256"), "seed43 V2 failure trace")
    flags = _seed43_v1().Score43Flags(
        stage=str(value.get("stage")),
        source_resolved=bool(value.get("resolved", {}).get("source")) if isinstance(value.get("resolved"), Mapping) else False,
        within_resolved=bool(value.get("resolved", {}).get("within")) if isinstance(value.get("resolved"), Mapping) else False,
        external_resolved=bool(value.get("resolved", {}).get("external")) if isinstance(value.get("resolved"), Mapping) else False,
        formal_resolved=bool(value.get("resolved", {}).get("formal")) if isinstance(value.get("resolved"), Mapping) else False,
        source_opened=bool(value.get("opened", {}).get("source")) if isinstance(value.get("opened"), Mapping) else False,
        within_opened=bool(value.get("opened", {}).get("within")) if isinstance(value.get("opened"), Mapping) else False,
        external_opened=bool(value.get("opened", {}).get("external")) if isinstance(value.get("opened"), Mapping) else False,
        formal_opened=bool(value.get("opened", {}).get("formal")) if isinstance(value.get("opened"), Mapping) else False,
        forward_calls=_json_copy(value.get("tfsr_forward_calls")) if isinstance(value.get("tfsr_forward_calls"), Mapping) else {},
        backward_calls=value.get("backward_calls"), optimizer_calls=value.get("optimizer_calls"), terminal_published=False,
    )
    if flags.payload() != {"stage": value["stage"], "resolved": value["resolved"], "opened": value["opened"],
                           "tfsr_forward_calls": value["tfsr_forward_calls"], "backward_calls": value["backward_calls"],
                           "optimizer_calls": value["optimizer_calls"], "terminal_published": False}:
        raise FailClosedError("seed43 V2 failure flags drift")


def _publish_failure(artifact: Any, identity: Seed43V2Identity, attempt_sha: str, input_sha: str | None, flags: Any) -> None:
    if flags.terminal_published or artifact.has_name("terminal.json") or artifact.has_name("failure.json"):
        return
    payload = _failure_payload(identity, attempt_sha, input_sha, flags)
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


def run_seed43_v2_lifecycle(
    *, artifact: Any, identity: Seed43V2Identity, execution_capability: ExecutionCapability,
    within_roster: tuple[str, ...], external_sessions: tuple[str, ...],
    within_assets_factory: Callable[[], tuple[Any, ...]], external_assets_factory: Callable[[], tuple[Any, ...]],
    backend: Backend, final_reverify: Callable[[], Mapping[str, object]],
) -> Mapping[str, object]:
    """Small V2 envelope over frozen seed43 V1 input/mode/score helpers."""
    _require_execution_capability(execution_capability, identity)
    s43 = _seed43_v1(); s42v1 = _seed42_v2()._v1(); core_identity = _core_identity(identity)
    flags = s43.Score43Flags(); attempt_sha: str | None = None; input_sha: str | None = None
    try:
        flags.stage = "attempt"
        attempt = _v2_attempt_payload(identity); validate_attempt_payload(attempt, identity)
        attempt_sha = artifact.publish_json("attempt.json", attempt)
        validate_attempt_payload(artifact.reload_json("attempt.json", attempt_sha), identity)
        flags.stage = "resolve_identical_seed42_v2_inputs_after_attempt"
        within_assets = within_assets_factory(); flags.within_resolved = True
        # The frozen external factory correctly returns UUID-ledger
        # ``ExternalAssetBinding`` objects.  The shared no-cache parser and
        # input-authority validator consume their surface-labelled
        # ``EvaluationAssetBinding`` form, exactly as the accepted seed42 V2
        # lifecycle does.  Converting here is an interface normalization, not
        # a new target lookup or a second asset authority.
        raw_external_assets = external_assets_factory(); flags.external_resolved = True
        if tuple(item.session for item in within_assets) != tuple(sorted(within_roster)):
            raise FailClosedError("seed43 V2 within roster drift after attempt")
        if tuple(item.session for item in raw_external_assets) != external_sessions:
            raise FailClosedError("seed43 V2 external roster drift after attempt")
        external_assets = tuple(s42v1._as_evaluation_asset(item) for item in raw_external_assets)
        evidence = backend.resolve_inputs(spec=s42v1.PUBLIC_SPEC, within_roster=within_roster, within_assets=within_assets,
                                          external_roster=external_assets, flags=flags)
        s42v1.validate_input_authority_evidence(evidence, within_roster=within_roster, within_assets=within_assets,
                                                external_roster=external_assets)
        core_replay = s43._validate_input_replay(evidence, upstream=identity.upstream, flags=flags, s42=s42v1)
        payload = _input_replay_payload(identity, core_replay); validate_input_replay_payload(payload, identity=identity)
        input_sha = artifact.publish_json("input_replay.json", payload)
        validate_input_replay_payload(artifact.reload_json("input_replay.json", input_sha), identity=identity)
        modes: list[Any] = []
        for surface, roster in (("within", within_roster), ("external", external_sessions)):
            for mode in ("aligned", "zero", "wrong_pair"):
                flags.stage = f"seed43_tfsr_{surface}_{mode}"
                item = backend.score_tfsr(surface=surface, mode=mode, input_authority_sha256=input_sha, flags=flags)
                s42v1._validate_mode_against_roster(item, system="tfsr", surface=surface, mode=mode,
                                                    expected_sessions=roster, input_authority_sha256=input_sha)
                modes.append(item)
        if flags.formal_resolved or flags.formal_opened or flags.backward_calls or flags.optimizer_calls:
            raise FailClosedError("seed43 V2 forward path crossed forbidden boundary")
        flags.stage = "post_forward_reverify"; backend.reverify_after_forwards(flags=flags)
        parity = backend.build_parity_manifest(flags=flags); s43.validate_parity_manifest(parity, flags)
        flags.stage = "score"
        core_score = s43.build_score_payload(identity=core_identity, input_replay_sha256=input_sha, tfsr_evidence=modes,
                                             parity_manifest=parity, resources=backend.resource_disclosure(), flags=flags)
        s43.validate_score_payload(core_score, identity=core_identity, input_replay_sha256=input_sha)
        score = _score_payload(identity, input_sha, core_score); validate_score_payload(score, identity=identity, input_sha=input_sha)
        score_body = _json_bytes(score); score_sha = _sha(score_body)
        flags.stage = "terminal_revalidation"; final_closure = final_reverify()
        if validate_phase_e_seed43_v2_closure(final_closure) != identity.closure:
            raise FailClosedError("seed43 V2 launch/final closure drift")
        terminal = _terminal_payload(identity, attempt_sha=attempt_sha, input_sha=input_sha, score_sha=score_sha,
                                     score=score, final_closure=final_closure)
        validate_terminal_payload(terminal, identity=identity, score=score, expected_score_sha=score_sha)
        terminal_body = _json_bytes(terminal)
        def validate_group(bodies: Mapping[str, bytes], digests: Mapping[str, str]) -> None:
            if (bodies.get("score.json") != score_body or bodies.get("terminal.json") != terminal_body
                    or digests.get("score.json") != score_sha):
                raise FailClosedError("seed43 V2 score/terminal atomic binding drift")
            loaded_score = json.loads(artifact.reload_pair("score.json", score_sha))
            loaded_terminal = json.loads(artifact.reload_pair("terminal.json", _sha(terminal_body)))
            if not isinstance(loaded_score, Mapping) or not isinstance(loaded_terminal, Mapping):
                raise FailClosedError("seed43 V2 atomic group JSON drift")
            validate_score_payload(loaded_score, identity=identity, input_sha=input_sha)
            validate_terminal_payload(loaded_terminal, identity=identity, score=loaded_score, expected_score_sha=score_sha)
        hashes = artifact.publish_group({"score.json": score_body, "terminal.json": terminal_body}, post_publish=validate_group)
        flags.terminal_published = True
        final_score = artifact.reload_json("score.json", hashes["score.json"])
        final_terminal = artifact.reload_json("terminal.json", hashes["terminal.json"])
        validate_score_payload(final_score, identity=identity, input_sha=input_sha)
        validate_terminal_payload(final_terminal, identity=identity, score=final_score, expected_score_sha=hashes["score.json"])
        if artifact.has_name("failure.json"):
            raise FailClosedError("seed43 V2 terminal cannot coexist with failure")
        return final_terminal
    except BaseException:
        if attempt_sha is not None:
            try: _publish_failure(artifact, identity, attempt_sha, input_sha, flags)
            except BaseException: pass
        raise
    finally:
        backend.close()


def _prepare_execution(
    root: Path, *, environment: Mapping[str, str] | None = None,
    upstream_loader: Callable[[Path], Seed42V2Evidence] = load_completed_seed42_v2,
    training_loader: Callable[[Path], Any] | None = None,
    authorization_loader: Callable[[Path, Seed42V2Evidence, Any], Seed43V2Authorization] | None = None,
) -> tuple[Path, Path, Seed42V2Evidence, Any, Seed43V2Authorization]:
    subc, subm = _correct_data_roots(environment)
    upstream = upstream_loader(root)
    training = (training_loader or _seed43_v1().validate_seed43_training_terminal)(root)
    authorization = (authorization_loader(root, upstream, training) if authorization_loader is not None
                     else verify_seed43_v2_authorization(root, upstream=upstream, training=training))
    return subc, subm, upstream, training, authorization


def execute_authorized_v2(root: Path) -> Mapping[str, object]:
    """Future physical route; seed43 terminal and V2 authority are mandatory."""
    subc, subm, upstream, training, authorization = _prepare_execution(root)
    identity = Seed43V2Identity(training=training, upstream=upstream, closure=authorization.closure,
                                 authorization=authorization.payload())
    capability = _issue_execution_capability(authorization)
    s43 = _seed43_v1()
    # Actual V1 backend imports happen only beyond the completed terminal and
    # reloaded V2 authorization gate.  No V1 seed43 result is loaded.
    s42 = s43._seed42_score()
    fixed = s42.verify_fixed_authorities(root); source_training = s42.validate_phase_d_training_terminal(root)
    source_authorization = s42.verify_phase_e_authorization(root, source_training, fixed)
    nested = upstream.nested_v1_identity
    if (nested.get("training_terminal_sha256") != source_training.terminal_sha256
            or nested.get("training_swa_sha256") != source_training.swa_sha256
            or nested.get("training_swa_state_digest") != source_training.swa_state_digest):
        raise FailClosedError("seed43 V2 nested seed42 V1 lineage differs from physical input backend")
    parent, name = _score_parent(root)
    artifact = s42.reserve_artifact_root(parent, name, SCORE_TOPOLOGY)
    within = s42.extract_within_roster(fixed["strict_manifest"].value or {})
    external = s42.extract_external_sessions(fixed["external_asset_ledger"].value or {})
    backend = build_physical_backend(root=root, fixed_authorities=fixed, upstream_training=source_training,
                                     upstream_authorization=source_authorization, seed43_training=training)
    def within_factory() -> tuple[Any, ...]: return s42.join_within_assets(source_authorization.preflight, subc)
    def external_factory() -> tuple[Any, ...]:
        return s42.join_external_assets(fixed["external_asset_ledger"].value or {}, fixed["external_scope"].value or {}, subm)
    def final_reverify() -> Mapping[str, object]:
        final_upstream = load_completed_seed42_v2(root)
        final_training = s43.validate_seed43_training_terminal(root)
        final_auth = verify_seed43_v2_authorization(root, upstream=final_upstream, training=final_training)
        if (final_upstream.binding_payload() != upstream.binding_payload() or final_training.payload() != training.payload()
                or final_auth.payload() != authorization.payload()):
            raise FailClosedError("seed43 V2 upstream/training/authorization changed after forwards")
        return phase_e_seed43_v2_closure(root)
    return run_seed43_v2_lifecycle(
        artifact=artifact, identity=identity, execution_capability=capability, within_roster=within, external_sessions=external,
        within_assets_factory=within_factory, external_assets_factory=external_factory, backend=backend, final_reverify=final_reverify,
    )


def dry_plan() -> dict[str, object]:
    return {"cell": CELL, "phase": PHASE, "seed": SEED,
            "status": "NO_DATA_NO_TARGET_NO_FORMAL_NO_GPU_NO_WRITE_NO_SCORE",
            "authorization": "none", "execution_flags_required_together": sorted(PUBLIC_FLAGS),
            "authority_root": AUTHORITY_ROOT_RELATIVE, "score_root": SCORE_ROOT_RELATIVE,
            "upstream_seed42_v2": Seed42V2Expectation().payload(),
            "seed43_training_requirement": "completed_terminal_swa_final4_required; current_live_training_is_not_authority",
            "semantics": "frozen_seed43_v1_core_six_cell_matrix_and_parity; nested_seed42_v2_Cell-D_replay",
            "correct_roots": {"SUBC_DATA_ROOT": SUBC_DATA_ROOT, "SUBM_DATA_ROOT": SUBM_DATA_ROOT},
            "superseded_seed43_v1": "code_evidence_only_no_result_required"}
