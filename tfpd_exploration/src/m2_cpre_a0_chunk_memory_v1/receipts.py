"""Immutable receipt primitives for the M2 C-Pre/A0 route.

They are intentionally stdlib-only and descriptor-based.  Importing this file
does not inspect a result root; callers receive paths only after an opaque
review-issued capability has been consumed.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class ReceiptError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptError(message)


def _nofollow() -> int:
    value = getattr(os, "O_NOFOLLOW", None)
    _require(isinstance(value, int) and value != 0,
             "platform lacks O_NOFOLLOW; descriptor receipt route fails closed")
    return value


def canonical_body(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _readonly_regular(info: os.stat_result, *, name: str) -> None:
    _require(stat.S_ISREG(info.st_mode), f"{name} is not a regular file")
    _require(stat.S_IMODE(info.st_mode) == 0o444, f"{name} mode must be 0444")
    _require(info.st_nlink == 1, f"{name} link count must be one")


@dataclass(frozen=True)
class Descriptor:
    name: str
    sha256: str
    payload: Mapping[str, Any]
    body_bytes: int


def descriptor_read(root: Path, name: str, *, expected_sha256: str | None = None) -> Descriptor:
    """Read one canonical body + sidecar via held root FD and O_NOFOLLOW."""
    _require("/" not in name and name.endswith(".json"), "descriptor name must be a leaf json file")
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | _nofollow()
    root_fd = os.open(root, flags)
    try:
        root_info = os.fstat(root_fd)
        _require(stat.S_ISDIR(root_info.st_mode), "descriptor root is not a directory")
        leaf_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | _nofollow()
        body_fd = os.open(name, leaf_flags, dir_fd=root_fd)
        try:
            body_info = os.fstat(body_fd)
            _readonly_regular(body_info, name=name)
            chunks: list[bytes] = []
            while True:
                block = os.read(body_fd, 1 << 20)
                if not block:
                    break
                chunks.append(block)
            body = b"".join(chunks)
        finally:
            os.close(body_fd)
        digest = sha256_bytes(body)
        _require(expected_sha256 is None or digest == expected_sha256, f"{name} body SHA drift")
        side_name = name + ".sha256"
        side_fd = os.open(side_name, leaf_flags, dir_fd=root_fd)
        try:
            side_info = os.fstat(side_fd)
            _readonly_regular(side_info, name=side_name)
            side = os.read(side_fd, 4096).decode("ascii")
        finally:
            os.close(side_fd)
        _require(side == f"{digest}  {name}\n", f"{name} sidecar drift")
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReceiptError(f"invalid JSON body for {name}") from exc
        _require(isinstance(payload, dict), f"{name} body must be a JSON object")
        return Descriptor(name=name, sha256=digest, payload=payload, body_bytes=len(body))
    finally:
        os.close(root_fd)


def verify_topology(root: Path, *, bodies: Sequence[str]) -> dict[str, Descriptor]:
    expected = set(bodies) | {name + ".sha256" for name in bodies}
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | _nofollow())
    try:
        observed = set(os.listdir(root_fd))
    finally:
        os.close(root_fd)
    _require(observed == expected, f"immutable receipt topology drift: {sorted(observed ^ expected)}")
    return {name: descriptor_read(root, name) for name in bodies}


def _write_new(fd_root: int, name: str, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | _nofollow()
    fd = os.open(name, flags, 0o444, dir_fd=fd_root)
    try:
        os.write(fd, content)
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        _readonly_regular(os.fstat(fd), name=name)
    finally:
        os.close(fd)


def publish_pair(root: Path, name: str, payload: Mapping[str, Any]) -> Descriptor:
    """Publish a new immutable body/sidecar pair; existing leaves fail closed."""
    _require("/" not in name and name.endswith(".json"), "receipt name must be a leaf json file")
    body = canonical_body(payload)
    digest = sha256_bytes(body)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | _nofollow())
    try:
        _write_new(root_fd, name, body)
        _write_new(root_fd, name + ".sha256", f"{digest}  {name}\n".encode("ascii"))
        os.fsync(root_fd)
    finally:
        os.close(root_fd)
    return descriptor_read(root, name, expected_sha256=digest)


def publish_failure_after_attempt(*, root: Path, attempt_sha256: str,
                                  schema: str, stage: str, error: str,
                                  progress: Mapping[str, Any]) -> Descriptor:
    """Publish the only legal fail-closed topology: attempt + failure.

    The helper refuses to erase or hide a partially published canonical body.
    A caller that already wrote a non-attempt artifact must stop for audit
    rather than manufacture a misleading failure receipt.
    """
    verify_topology(root, bodies=("attempt.json",))
    attempt = descriptor_read(root, "attempt.json")
    _require(attempt.sha256 == attempt_sha256, "failure attempt link drift")
    failure = publish_pair(root, "failure.json", {
        "schema": schema, "status": "FAIL_CLOSED", "attempt_sha256": attempt_sha256,
        "stage": str(stage), "error": str(error), "progress": dict(progress),
    })
    verify_topology(root, bodies=("attempt.json", "failure.json"))
    return failure


def publish_failure_preserving_prefix(*, root: Path, prefix_bodies: Sequence[str],
                                      attempt_sha256: str, schema: str, stage: str,
                                      error: str, progress: Mapping[str, Any]) -> Descriptor:
    """A0 failure publication when immutable launch/runtime evidence exists.

    Unlike C-Pre's pre-metadata failure shape, an A0 failure may legitimately
    retain the exact prefix that proves where target/model/CUDA access stopped.
    No prefix receipt is ever removed or replaced.
    """
    _require("terminal.json" not in os.listdir(root) and "failure.json" not in os.listdir(root),
             "cannot publish failure after terminal/failure")
    descriptors = verify_topology(root, bodies=prefix_bodies)
    _require(descriptors["attempt.json"].sha256 == attempt_sha256, "failure attempt link drift")
    failure = publish_pair(root, "failure.json", {
        "schema": schema, "status": "FAIL_CLOSED", "attempt_sha256": attempt_sha256,
        "stage": str(stage), "error": str(error), "progress": dict(progress),
        "retained_immutable_prefix": {name: descriptors[name].sha256 for name in prefix_bodies},
    })
    verify_topology(root, bodies=tuple(prefix_bodies) + ("failure.json",))
    return failure


def verify_terminal_xor(root: Path, *, success_bodies: Sequence[str]) -> dict[str, Descriptor]:
    names = set(os.listdir(root))
    has_terminal = "terminal.json" in names
    has_failure = "failure.json" in names
    _require(has_terminal != has_failure, "receipt root must publish terminal XOR failure")
    if has_terminal:
        return verify_topology(root, bodies=tuple(success_bodies))
    # A0 may have immutably published launch/input/replay evidence before a
    # runtime failure.  Preserve that honest prefix; never pretend its root is
    # the C-Pre two-leaf topology.  The failure body itself commits the exact
    # retained descriptor map, which is rechecked here before accepting it.
    failure = descriptor_read(root, "failure.json")
    retained = failure.payload.get("retained_immutable_prefix")
    if retained is None:
        return verify_topology(root, bodies=("attempt.json", "failure.json"))
    _require(isinstance(retained, Mapping) and bool(retained) and "attempt.json" in retained,
             "failure retained immutable prefix topology drift")
    # The caller defines its successful flat receipt topology.  A failure may
    # retain only a complete initial prefix of that topology (never terminal),
    # allowing C-Pre's post-metadata failure and A0's post-launch/replay
    # failure to be verified by one exact codec.
    allowed = tuple(name for name in success_bodies if name != "terminal.json")
    _require(set(retained).issubset(set(allowed)), "failure prefix contains a noncanonical receipt")
    prefix = tuple(name for name in allowed if name in retained)
    _require(prefix == allowed[:len(prefix)], "failure retained receipts are not a canonical stage prefix")
    bodies = prefix + ("failure.json",)
    descriptors = verify_topology(root, bodies=bodies)
    for name, digest in retained.items():
        _require(isinstance(digest, str) and descriptors[name].sha256 == digest,
                 "failure retained immutable prefix SHA drift")
    _require(descriptors["attempt.json"].sha256 == failure.payload.get("attempt_sha256"),
             "failure retained attempt link drift")
    return descriptors


def semantic_predecessor_validator(payloads: Mapping[str, Mapping[str, Any]]) -> None:
    """Exact reviewed structure of the three bound local predecessor terminals."""
    memory = payloads["memory_law"]
    chrono = payloads["chrono4"]
    reblock = payloads["reblock10"]
    _require(memory.get("schema") == "m2_memory_law_scan_v1_terminal_v1"
             and memory.get("status") == "TERMINAL"
             and memory.get("official_contract_claimed") is False,
             "memory-law terminal schema/status/local-contract drift")
    verdict = memory.get("verdict")
    _require(isinstance(verdict, Mapping) and verdict.get("winner") == "UNIFORM_UNCAPPED"
             and verdict.get("verdict") == "UNCAPPED_WINS"
             and verdict.get("eligible_policies") == ["UNIFORM_UNCAPPED"],
             "memory-law selected winner drift")
    policy = memory.get("gates", {}).get("policy_results", {}).get("UNIFORM_UNCAPPED", {})
    _require(policy.get("disposition") == "SELECTED_ELIGIBLE"
             and policy.get("primary_passed") is True and policy.get("safety_passed") is True,
             "memory-law selected policy gate drift")
    _require(chrono.get("schema") == "m2_chrono4_strict_v1_terminal_v1"
             and chrono.get("status") == "TERMINAL" and chrono.get("official_contract_claimed") is False
             and chrono.get("headline", {}).get("verdict") == "SELECTION_CONTRIBUTION_LARGE",
             "chrono4 terminal structure/local conclusion drift")
    chrono_delta = chrono.get("selection_contribution_dopt_minus_chrono4", {}).get("cdm_family", {}).get(
        "external_post30_local", {})
    _require(float(chrono_delta.get("equal_session_mean_delta", float("nan"))) > 0.0
             and int(chrono_delta.get("positive_sessions_dopt_better", -1)) >= 4,
             "chrono4 must remain negative evidence for replacing D-opt")
    _require(reblock.get("schema") == "m2_reblock10_v1_terminal_v1"
             and reblock.get("status") == "TERMINAL" and reblock.get("official_contract_claimed") is False
             and reblock.get("primary_verdict") == "REBLOCK_CDM_NOT_WORTH"
             and reblock.get("primary_gate_cdm_k4_external", {}).get("verdict") == "REBLOCK_CDM_NOT_WORTH",
             "reblock10 terminal structure/negative conclusion drift")
    reblock_gate = reblock["primary_gate_cdm_k4_external"]
    _require(float(reblock_gate.get("delta", float("nan"))) < 0.0
             and int(reblock_gate.get("positive_sessions", -1)) == 2
             and int(reblock_gate.get("breadth_denominator", -1)) == 6,
             "reblock10 negative primary evidence drift")
