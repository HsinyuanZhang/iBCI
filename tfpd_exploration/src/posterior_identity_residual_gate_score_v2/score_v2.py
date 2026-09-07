"""Minimal immutable-successor lifecycle for the PIRG score-v2 repair.

V1 stopped honestly before evaluation because its generic posterior-carrier
tensor digest cannot byte-view a rank-zero alpha scalar.  This module never
changes V1.  It validates that failed V1 attempt/failure pair before reserving
the fresh V2 root, then reuses the frozen V1 score contracts and only permits
the dedicated V2 physical subclass to repair scalar state hashing.

The public CLI imports :func:`dry_plan` only.  This module is standard-library
plus the V1 dry contract: it opens no data, checkpoint, CUDA device, or remote
host at import time.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from src.posterior_identity_residual_gate_v1 import score as v1
from src.posterior_identity_residual_gate_v1 import train as v1_train


class PIRGScoreV2Error(v1.PIRGScoreError):
    """Fail closed for the additive V2 scalar-digest successor."""


RESULT_ROOT_RELATIVE = "tfpd_exploration/results/posterior_identity_residual_gate_score_v2"
V1_RESULT_ROOT_RELATIVE = v1.RESULT_ROOT_RELATIVE
V1_ATTEMPT_SHA256 = "8c92296fdc7b8f0b8f6361211a13d059daa9756ce7839a5a86bb238ecca832e9"
V1_FAILURE_SHA256 = "d0c595267cd789aa4e546c8e8d0af1413dc3fa8d4b051008859b056b1ab6231b"
V1_FAILURE_ERROR_SHA256 = "fe31620340c5baccbba3a17f15ba29d8347e7645db7c50de7dcbf580d81e9565"

V2_CLOSURE_PATHS = tuple(dict.fromkeys((
    *v1_train.SCORE_IMPLEMENTATION_CLOSURE,
    "tfpd_exploration/src/posterior_identity_residual_gate_score_v2/__init__.py",
    "tfpd_exploration/src/posterior_identity_residual_gate_score_v2/score_v2.py",
    "tfpd_exploration/src/posterior_identity_residual_gate_score_v2/score_physical_v2.py",
    "tfpd_exploration/scripts/run_posterior_identity_residual_gate_score_v2.py",
    "tfpd_exploration/tests/test_posterior_identity_residual_gate_score_v2.py",
)))


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise PIRGScoreV2Error(f"{label} must be an exact lowercase SHA-256")
    return value


@dataclass(frozen=True)
class V1FailedPredecessor:
    """Exact immutable V1 failure evidence required before V2 reservation."""

    root_relative: str = V1_RESULT_ROOT_RELATIVE
    attempt_sha256: str = V1_ATTEMPT_SHA256
    failure_sha256: str = V1_FAILURE_SHA256
    error_sha256: str = V1_FAILURE_ERROR_SHA256

    def payload(self) -> dict[str, object]:
        if self.root_relative != V1_RESULT_ROOT_RELATIVE:
            raise PIRGScoreV2Error("PIRG V1 predecessor root drift")
        return {
            "root_relative": self.root_relative,
            "attempt_sha256": _sha(self.attempt_sha256, "PIRG V1 attempt SHA"),
            "failure_sha256": _sha(self.failure_sha256, "PIRG V1 failure SHA"),
            "error_sha256": _sha(self.error_sha256, "PIRG V1 failure error SHA"),
            "expected_topology": [
                "attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256",
            ],
            "failure_stage": "prepare",
            "cuda_initialized": True,
            "within_opened": False,
            "external_opened": False,
            "formal_opened": False,
            "h1_opened": False,
            "target_optimizer_steps": 0,
            "target_backward_calls": 0,
            "target_update_calls": 0,
            "terminal_published": False,
        }


V1_FAILURE_PREDECESSOR = V1FailedPredecessor()


class _HeldDirectory:
    """One O_NOFOLLOW directory FD for the complete V1 predecessor read."""

    def __init__(self, *, root: Path, fd: int, identity: tuple[int, int]) -> None:
        self.root, self.fd, self.identity = root, fd, identity

    @classmethod
    def open(cls, root: Path) -> "_HeldDirectory":
        path = Path(root).absolute()
        before = os.lstat(path)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise PIRGScoreV2Error("PIRG V1 predecessor root is not a regular directory")
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        opened = os.fstat(fd)
        identity = (int(opened.st_dev), int(opened.st_ino))
        if identity != (int(before.st_dev), int(before.st_ino)):
            os.close(fd)
            raise PIRGScoreV2Error("PIRG V1 predecessor root descriptor identity drift")
        return cls(root=path, fd=fd, identity=identity)

    def names(self) -> set[str]:
        return set(os.listdir(self.fd))

    def read_regular(self, name: str) -> tuple[bytes, int]:
        if not isinstance(name, str) or not name or "/" in name:
            raise PIRGScoreV2Error("PIRG V1 predecessor leaf name drift")
        fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=self.fd)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or (stat.S_IMODE(info.st_mode) != 0o444):
                raise PIRGScoreV2Error("PIRG V1 predecessor leaf type/mode drift")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(fd, 1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks), int(info.st_size)
        finally:
            os.close(fd)

    def reverify(self) -> None:
        opened = os.fstat(self.fd)
        after = os.lstat(self.root)
        expected = self.identity
        if (int(opened.st_dev), int(opened.st_ino)) != expected or (int(after.st_dev), int(after.st_ino)) != expected:
            raise PIRGScoreV2Error("PIRG V1 predecessor root changed during held-FD read")

    def close(self) -> None:
        os.close(self.fd)


def _read_pair(*, held: _HeldDirectory, name: str, expected_sha256: str) -> tuple[bytes, Mapping[str, object]]:
    body, _ = held.read_regular(name)
    sidecar, _ = held.read_regular(f"{name}.sha256")
    expected = _sha(expected_sha256, f"PIRG V1 predecessor {name} SHA")
    if _digest(body) != expected or sidecar != f"{expected}  {name}\n".encode("ascii"):
        raise PIRGScoreV2Error("PIRG V1 predecessor immutable body/sidecar drift")
    try:
        value = json.loads(body)
    except (TypeError, json.JSONDecodeError) as error:
        raise PIRGScoreV2Error("PIRG V1 predecessor JSON decode drift") from error
    if not isinstance(value, Mapping):
        raise PIRGScoreV2Error("PIRG V1 predecessor JSON root drift")
    return body, value


def validate_v1_failed_predecessor(
    root: Path,
    *,
    expected_identity: v1.PIRGScoreIdentity | None = None,
    predecessor: V1FailedPredecessor | None = None,
) -> v1.PIRGScoreIdentity:
    """Descriptor-validate V1's exact terminal-less failure graph.

    ``predecessor`` is injectable solely for CPU synthetic tests.  Production
    authorization always calls this function with the frozen literal default.
    """
    predecessor = V1_FAILURE_PREDECESSOR if predecessor is None else predecessor
    contract = predecessor.payload()
    directory = _HeldDirectory.open(Path(root).absolute() / str(contract["root_relative"]))
    try:
        if directory.names() != set(contract["expected_topology"]):
            raise PIRGScoreV2Error("PIRG V1 predecessor topology must contain only attempt/failure pairs")
        _attempt_body, attempt = _read_pair(
            held=directory, name="attempt.json", expected_sha256=str(contract["attempt_sha256"]),
        )
        _failure_body, failure = _read_pair(
            held=directory, name="failure.json", expected_sha256=str(contract["failure_sha256"]),
        )
        directory.reverify()
    finally:
        directory.close()
    try:
        base_identity_payload = v1.validate_score_identity(attempt.get("identity"))
        base_identity = v1.PIRGScoreIdentity(
            closure=v1.ScoreClosure(base_identity_payload["closure"]),
            source_terminal_sha256=base_identity_payload["source_terminal_sha256"],
            final_alpha_sha256=base_identity_payload["final_alpha_sha256"],
            source_authority_sha256=base_identity_payload["source_authority_sha256"],
        )
    except Exception as error:
        raise PIRGScoreV2Error("PIRG V1 attempt identity is not a frozen V1 score identity") from error
    if expected_identity is not None and base_identity.payload() != v1.validate_score_identity(expected_identity):
        raise PIRGScoreV2Error("PIRG V1 predecessor/selected V1 identity drift")
    expected_attempt = {
        "schema": "posterior_identity_residual_gate_score_attempt_v1",
        "identity": base_identity.payload(),
        "status": "ATTEMPT_RESERVED_BEFORE_EVALUATION_INPUTS",
        "within_opened": False,
        "external_opened": False,
        "formal_opened": False,
        "h1_opened": False,
        "target_optimizer_steps": 0,
        "target_backward_calls": 0,
        "target_update_calls": 0,
    }
    if dict(attempt) != expected_attempt:
        raise PIRGScoreV2Error("PIRG V1 attempt semantic boundary drift")
    required_failure = {
        "schema", "identity", "attempt_sha256", "input_authority_sha256", "stage", "progress",
        "terminal_published", "error_class", "error_sha256", "traceback_sha256",
    }
    progress = failure.get("progress")
    if not isinstance(progress, Mapping):
        raise PIRGScoreV2Error("PIRG V1 failure progress schema drift")
    expected_progress = {
        "within_opened": False,
        "external_opened": False,
        "cuda_initialized": True,
        "target_optimizer_steps": 0,
        "target_backward_calls": 0,
        "target_update_calls": 0,
        "formal_opened": False,
        "h1_opened": False,
    }
    if (
        set(failure) != required_failure
        or failure.get("schema") != "posterior_identity_residual_gate_score_failure_v1"
        or failure.get("identity") != base_identity.payload()
        or failure.get("attempt_sha256") != contract["attempt_sha256"]
        or failure.get("input_authority_sha256") is not None
        or failure.get("stage") != "prepare"
        or dict(progress) != expected_progress
        or failure.get("terminal_published") is not False
        or failure.get("error_sha256") != contract["error_sha256"]
        or not isinstance(failure.get("error_class"), str)
        or not isinstance(failure.get("traceback_sha256"), str)
    ):
        raise PIRGScoreV2Error("PIRG V1 scalar-digest failure semantics drift")
    _sha(failure["traceback_sha256"], "PIRG V1 failure traceback SHA")
    return base_identity


@dataclass(frozen=True)
class ScoreV2Closure:
    value: Mapping[str, object]

    def payload(self) -> dict[str, object]:
        if not isinstance(self.value, Mapping) or set(self.value) != {"paths", "sha256_by_path", "closure_sha256"}:
            raise PIRGScoreV2Error("PIRG V2 closure schema drift")
        paths, hashes = self.value.get("paths"), self.value.get("sha256_by_path")
        if not isinstance(paths, list) or tuple(paths) != V2_CLOSURE_PATHS or not isinstance(hashes, Mapping) or set(hashes) != set(V2_CLOSURE_PATHS):
            raise PIRGScoreV2Error("PIRG V2 closure topology drift")
        body = {
            "paths": list(V2_CLOSURE_PATHS),
            "sha256_by_path": {path: _sha(hashes[path], f"PIRG V2 closure {path}") for path in V2_CLOSURE_PATHS},
        }
        if self.value.get("closure_sha256") != _digest(_json(body)):
            raise PIRGScoreV2Error("PIRG V2 closure digest drift")
        return {**body, "closure_sha256": str(self.value["closure_sha256"])}


def score_v2_implementation_closure(root: Path) -> ScoreV2Closure:
    """Descriptor-hash the explicit V1+V2 score closure without data access."""
    base = Path(root).absolute()
    hashes: dict[str, str] = {}
    for relative in V2_CLOSURE_PATHS:
        path = base / relative
        before = os.lstat(path)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise PIRGScoreV2Error(f"PIRG V2 closure leaf missing or aliased: {relative}")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(fd)
            if (opened.st_dev, opened.st_ino, opened.st_size) != (before.st_dev, before.st_ino, before.st_size):
                raise PIRGScoreV2Error("PIRG V2 closure descriptor identity drift")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(fd, 1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            os.close(fd)
        after = os.lstat(path)
        if (after.st_dev, after.st_ino, after.st_size) != (before.st_dev, before.st_ino, before.st_size):
            raise PIRGScoreV2Error("PIRG V2 closure changed during descriptor read")
        hashes[relative] = _digest(b"".join(chunks))
    body = {"paths": list(V2_CLOSURE_PATHS), "sha256_by_path": hashes}
    return ScoreV2Closure({**body, "closure_sha256": _digest(_json(body))})


@dataclass(frozen=True)
class PIRGScoreV2Identity:
    """Wrap exact V1 science identity with V1 failure and V2 code closure."""

    v1_identity: v1.PIRGScoreIdentity
    closure: ScoreV2Closure

    def payload(self) -> dict[str, object]:
        base_identity = v1.validate_score_identity(self.v1_identity)
        return {
            "schema": "posterior_identity_residual_gate_score_identity_v2",
            "cell": "POSTERIOR_IDENTITY_RESIDUAL_GATE_V1",
            "classification": "NON_GOVERNING_PIRG_PERFORMANCE_SCREEN__SCALAR_DIGEST_REPAIR",
            "result_root_relative": RESULT_ROOT_RELATIVE,
            "v1_score_identity": base_identity,
            "v1_score_closure": base_identity["closure"],
            "v2_score_closure": self.closure.payload(),
            "v1_failed_predecessor": V1_FAILURE_PREDECESSOR.payload(),
            "only_physical_override": "wrapper_alpha_0d_digest__dtype_shape_bytes__flatten_before_uint8_view",
            "parser_model_input_forward_cell_scorer_final_reverify_inherited_from_v1": True,
        }


def validate_score_v2_identity(value: PIRGScoreV2Identity | Mapping[str, object]) -> dict[str, object]:
    payload = value.payload() if isinstance(value, PIRGScoreV2Identity) else dict(value)
    required = {
        "schema", "cell", "classification", "result_root_relative", "v1_score_identity", "v1_score_closure",
        "v2_score_closure", "v1_failed_predecessor", "only_physical_override",
        "parser_model_input_forward_cell_scorer_final_reverify_inherited_from_v1",
    }
    if set(payload) != required:
        raise PIRGScoreV2Error("PIRG V2 identity schema drift")
    v1_payload = v1.validate_score_identity(payload.get("v1_score_identity"))
    closure = ScoreV2Closure(payload.get("v2_score_closure")).payload()
    rebuilt = PIRGScoreV2Identity(
        v1_identity=v1.PIRGScoreIdentity(
            closure=v1.ScoreClosure(v1_payload["closure"]),
            source_terminal_sha256=v1_payload["source_terminal_sha256"],
            final_alpha_sha256=v1_payload["final_alpha_sha256"],
            source_authority_sha256=v1_payload["source_authority_sha256"],
        ),
        closure=ScoreV2Closure(closure),
    ).payload()
    if dict(payload) != rebuilt:
        raise PIRGScoreV2Error("PIRG V2 identity/predecessor/closure binding drift")
    return rebuilt


_CAPABILITY_SEAL = object()


@dataclass(frozen=True)
class PIRGScoreV2Capability:
    identity_sha256: str
    _seal: object = field(default=_CAPABILITY_SEAL, repr=False, compare=False)


def _issue_root_review_capability(identity: PIRGScoreV2Identity) -> PIRGScoreV2Capability:
    return PIRGScoreV2Capability(_digest(_json(validate_score_v2_identity(identity))))


def _require_capability(value: object, *, identity: PIRGScoreV2Identity) -> None:
    if (
        not isinstance(value, PIRGScoreV2Capability)
        or value._seal is not _CAPABILITY_SEAL
        or value.identity_sha256 != _digest(_json(validate_score_v2_identity(identity)))
    ):
        raise PIRGScoreV2Error("PIRG score V2 requires an in-process root-reviewed capability")


def verify_v2_authorization(root: Path, *, identity: PIRGScoreV2Identity, capability: object) -> v1.PIRGScoreIdentity:
    """Complete all V1/V2/predecessor checks before a V2 root may exist."""
    _require_capability(capability, identity=identity)
    stable = validate_score_v2_identity(identity)
    v1_payload = stable["v1_score_identity"]
    base_identity = v1.PIRGScoreIdentity(
        closure=v1.ScoreClosure(v1_payload["closure"]),
        source_terminal_sha256=v1_payload["source_terminal_sha256"],
        final_alpha_sha256=v1_payload["final_alpha_sha256"],
        source_authority_sha256=v1_payload["source_authority_sha256"],
    )
    if v1.score_implementation_closure(root).payload() != base_identity.closure.payload():
        raise PIRGScoreV2Error("PIRG V2 current V1 closure drift before predecessor/data access")
    if score_v2_implementation_closure(root).payload() != identity.closure.payload():
        raise PIRGScoreV2Error("PIRG V2 current successor closure drift before predecessor/data access")
    validate_v1_failed_predecessor(root, expected_identity=base_identity)
    return base_identity


def _attempt_payload(*, identity: PIRGScoreV2Identity) -> dict[str, object]:
    return {
        "schema": "posterior_identity_residual_gate_score_attempt_v2",
        "identity": validate_score_v2_identity(identity),
        "status": "ATTEMPT_RESERVED_AFTER_V1_FAILURE_VALIDATION_BEFORE_EVALUATION_INPUTS",
        "within_opened": False,
        "external_opened": False,
        "formal_opened": False,
        "h1_opened": False,
        "target_optimizer_steps": 0,
        "target_backward_calls": 0,
        "target_update_calls": 0,
    }


def build_v2_score_payload(*, identity: PIRGScoreV2Identity, input_payload: Mapping[str, object],
                           evidence: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Wrap the unchanged V1 scientific matrix in V2 scalar-repair provenance."""
    base_identity = identity.v1_identity
    v1_payload = v1.build_score_payload(identity=base_identity, input_payload=input_payload, evidence=evidence)
    v1.validate_score_payload(v1_payload, identity=base_identity, input_payload=input_payload)
    return {
        "schema": "posterior_identity_residual_gate_score_v2",
        "classification": "NON_GOVERNING_PIRG_PERFORMANCE_SCREEN__SCALAR_DIGEST_REPAIR",
        "identity": validate_score_v2_identity(identity),
        "v1_failed_predecessor": V1_FAILURE_PREDECESSOR.payload(),
        "input_authority_sha256": _digest(_json(input_payload)),
        "v1_score_payload": v1_payload,
        "v1_score_payload_sha256": _digest(_json(v1_payload)),
        "scalar_digest_repair": {
            "base_cell_d_digest": "arm_common.state_sha256_unchanged",
            "alpha_digest": "dtype+original_shape+normalized_bytes__0d_reshape_flat_before_uint8_view",
            "only_physical_override": True,
        },
    }


def validate_v2_score_payload(payload: Mapping[str, object], *, identity: PIRGScoreV2Identity,
                              input_payload: Mapping[str, object]) -> dict[str, object]:
    required = {
        "schema", "classification", "identity", "v1_failed_predecessor", "input_authority_sha256",
        "v1_score_payload", "v1_score_payload_sha256", "scalar_digest_repair",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise PIRGScoreV2Error("PIRG V2 score payload schema drift")
    v1.validate_score_payload(payload.get("v1_score_payload"), identity=identity.v1_identity, input_payload=input_payload)
    rebuilt = build_v2_score_payload(
        identity=identity, input_payload=input_payload,
        evidence=payload["v1_score_payload"]["matrix"],
    )
    if dict(payload) != rebuilt:
        raise PIRGScoreV2Error("PIRG V2 score payload semantic/provenance drift")
    return rebuilt


def _terminal_payload(*, identity: PIRGScoreV2Identity, attempt_sha256: str, input_authority_sha256: str,
                      score_sha256: str, final_v1_closure: Mapping[str, object], progress: Mapping[str, object]) -> dict[str, object]:
    if dict(final_v1_closure) != identity.v1_identity.closure.payload():
        raise PIRGScoreV2Error("PIRG V2 terminal V1 launch/final closure drift")
    if v1._validated_complete_progress(progress) != dict(progress):
        raise PIRGScoreV2Error("PIRG V2 terminal runtime boundary drift")
    return {
        "schema": "posterior_identity_residual_gate_score_terminal_v2",
        "status": "PIRG_QUICK_SCORE_COMPLETE__NON_GOVERNING__SCALAR_DIGEST_REPAIR",
        "identity": validate_score_v2_identity(identity),
        "v1_failed_predecessor": V1_FAILURE_PREDECESSOR.payload(),
        "attempt_sha256": _sha(attempt_sha256, "PIRG V2 terminal attempt SHA"),
        "input_authority_sha256": _sha(input_authority_sha256, "PIRG V2 terminal input SHA"),
        "score_sha256": _sha(score_sha256, "PIRG V2 terminal score SHA"),
        "launch_v1_score_closure": identity.v1_identity.closure.payload(),
        "final_v1_score_closure": dict(final_v1_closure),
        "launch_v2_score_closure": identity.closure.payload(),
        "final_v2_score_closure": identity.closure.payload(),
        "execution_progress": dict(progress),
        "formal_verdict": False,
        "score_terminal_transactional_group": True,
    }


def validate_v2_terminal_payload(payload: Mapping[str, object], *, identity: PIRGScoreV2Identity,
                                 score_sha256: str | None = None) -> dict[str, object]:
    required = {
        "schema", "status", "identity", "v1_failed_predecessor", "attempt_sha256", "input_authority_sha256",
        "score_sha256", "launch_v1_score_closure", "final_v1_score_closure", "launch_v2_score_closure",
        "final_v2_score_closure", "execution_progress", "formal_verdict", "score_terminal_transactional_group",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise PIRGScoreV2Error("PIRG V2 terminal schema drift")
    rebuilt = _terminal_payload(
        identity=identity, attempt_sha256=payload.get("attempt_sha256"),
        input_authority_sha256=payload.get("input_authority_sha256"), score_sha256=payload.get("score_sha256"),
        final_v1_closure=payload.get("final_v1_score_closure"), progress=payload.get("execution_progress"),
    )
    if score_sha256 is not None and rebuilt["score_sha256"] != _sha(score_sha256, "expected PIRG V2 score SHA"):
        raise PIRGScoreV2Error("PIRG V2 terminal score SHA drift")
    if dict(payload) != rebuilt:
        raise PIRGScoreV2Error("PIRG V2 terminal semantic binding drift")
    return rebuilt


class PIRGScoreV2Backend(Protocol):
    def prepare(self, *, identity: v1.PIRGScoreIdentity) -> None: ...
    def resolve_inputs(self, *, identity: v1.PIRGScoreIdentity) -> v1.InputAuthority: ...
    def score_cell(self, *, cell: v1.ScoreCell, input_payload: Mapping[str, object]) -> v1.CellEvidence: ...
    def final_reverify(self, *, identity: v1.PIRGScoreIdentity) -> v1.ScoreClosure: ...
    def progress(self) -> Mapping[str, object]: ...
    def close(self) -> None: ...


def run_v2_score_lifecycle(*, root: Path, identity: PIRGScoreV2Identity, backend: PIRGScoreV2Backend,
                           execution_capability: object) -> Mapping[str, str]:
    """Run only after reviewed capability; V1 failure is checked before reserve."""
    base_identity = verify_v2_authorization(root, identity=identity, capability=execution_capability)
    artifact = v1._ArtifactRoot(Path(root).absolute() / RESULT_ROOT_RELATIVE)
    attempt_sha: str | None = None
    input_sha: str | None = None
    terminal_written = False
    stage = "attempt"
    try:
        artifact.reserve()
        attempt_sha = artifact.publish_json("attempt.json", _attempt_payload(identity=identity))
        stage = "prepare"
        backend.prepare(identity=base_identity)
        stage = "input_authority"
        input_payload = backend.resolve_inputs(identity=base_identity).payload(identity=base_identity)
        v1.validate_input_authority(input_payload, identity=base_identity)
        input_sha = artifact.publish_json("input_authority.json", input_payload)
        stage = "forwards"
        evidence: list[Mapping[str, object]] = []
        for cell in v1.score_matrix():
            item = backend.score_cell(cell=cell, input_payload=input_payload)
            if item.cell != cell:
                raise PIRGScoreV2Error("PIRG V2 backend returned a wrong V1 score cell")
            evidence.append(item.payload(identity=base_identity, input_authority_sha256=input_sha))
        stage = "final_reverify"
        final_v1 = backend.final_reverify(identity=base_identity).payload()
        if final_v1 != base_identity.closure.payload():
            raise PIRGScoreV2Error("PIRG V2 V1 final closure drift")
        # Re-read the immutable failed predecessor and both live closures after
        # all forwards, before scientific score publication.
        verify_v2_authorization(root, identity=identity, capability=execution_capability)
        payload = build_v2_score_payload(identity=identity, input_payload=input_payload, evidence=evidence)
        validate_v2_score_payload(payload, identity=identity, input_payload=input_payload)
        score_sha = _digest(_json(payload))
        terminal = _terminal_payload(
            identity=identity, attempt_sha256=attempt_sha, input_authority_sha256=input_sha,
            score_sha256=score_sha, final_v1_closure=final_v1, progress=backend.progress(),
        )
        validate_v2_terminal_payload(terminal, identity=identity, score_sha256=score_sha)
        result = artifact.publish_group({"score.json": payload, "terminal.json": terminal})
        terminal_written = True
        return {"attempt_sha256": attempt_sha, "input_authority_sha256": input_sha, **result}
    except BaseException as error:
        if attempt_sha is not None and not terminal_written:
            failure = {
                "schema": "posterior_identity_residual_gate_score_failure_v2",
                "identity": validate_score_v2_identity(identity),
                "v1_failed_predecessor": V1_FAILURE_PREDECESSOR.payload(),
                "attempt_sha256": attempt_sha,
                "input_authority_sha256": input_sha,
                "stage": stage,
                "progress": dict(backend.progress()),
                "terminal_published": False,
                "error_class": type(error).__name__,
                "error_sha256": _digest(repr(error).encode("utf-8")),
                "traceback_sha256": _digest("".join(traceback.format_exception(error)).encode("utf-8")),
            }
            try:
                artifact.publish_json("failure.json", failure)
            except BaseException:
                pass
        raise
    finally:
        try:
            backend.close()
        finally:
            artifact.close()


def dry_plan() -> dict[str, object]:
    """Public dry plan: no Torch, data, output, or capability creation."""
    return {
        "status": "DRY_NO_TORCH_NO_DATA_NO_CHECKPOINT_NO_CUDA_NO_GPU_NO_WRITE_NO_LAUNCH",
        "route": "PIRG_QUICK_SCORE_V2_SCALAR_DIGEST_REPAIR",
        "result_root_relative": RESULT_ROOT_RELATIVE,
        "v1_failed_predecessor": V1_FAILURE_PREDECESSOR.payload(),
        "only_physical_override": "wrapper_alpha_0d_digest__dtype_shape_bytes__flatten_before_uint8_view",
        "inherits": ["parser", "model_loader", "input_authority", "forwards", "cell_scorer", "final_held_revalidation"],
        "execution": "requires future in-process root-reviewed capability; not available from this CLI",
    }
