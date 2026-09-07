"""Held-FD verifier for the immutable AOF-S V1 pre-payload failure graph."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

from . import plan


class BindingError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise BindingError(message)


def _read_leaf(directory_fd: int, name: str) -> tuple[bytes, str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    _need(hasattr(os, "O_NOFOLLOW"), "V2 requires O_NOFOLLOW")
    fd = os.open(name, flags, dir_fd=directory_fd)
    try:
        info = os.fstat(fd)
        _need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444 and info.st_nlink == 1,
              f"V1 predecessor immutable leaf mode/link drift: {name}")
        body = b""
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            body += block
    finally:
        os.close(fd)
    return body, hashlib.sha256(body).hexdigest()


def validate_v1_failure_graph(repo_root: Path = plan.REPO_ROOT) -> dict[str, Any]:
    """Validate exactly three V1 bodies plus sidecars; no result discovery."""
    root = repo_root / plan.V1_FAILURE_ROOT_RELATIVE
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    _need(hasattr(os, "O_NOFOLLOW"), "V2 requires O_NOFOLLOW")
    fd = os.open(root, flags)
    try:
        info = os.fstat(fd)
        expected = set(plan.V1_FAILURE_BODIES) | {name + ".sha256" for name in plan.V1_FAILURE_BODIES}
        _need(set(os.listdir(fd)) == expected, "V1 predecessor exact six-leaf topology drift")
        bodies: dict[str, dict[str, Any]] = {}
        for name, expected_sha in plan.V1_FAILURE_BODIES.items():
            body, digest = _read_leaf(fd, name)
            side, _side_digest = _read_leaf(fd, name + ".sha256")
            _need(digest == expected_sha and side == f"{digest}  {name}\n".encode("ascii"),
                  f"V1 predecessor body/sidecar digest drift: {name}")
            bodies[name] = json.loads(body.decode("utf-8"))
    finally:
        os.close(fd)
    attempt, predecessor, failure = (bodies[name] for name in ("attempt.json", "predecessor_authority.json", "failure.json"))
    _need(attempt.get("schema") == "m2_aof_scalar_static_package_v1_attempt"
          and attempt.get("closure_sha256") == plan.V1_FAILURE_CLOSURE_SHA256,
          "V1 attempt schema/closure drift")
    _need(failure.get("schema") == "m2_aof_scalar_static_package_v1_failure"
          and failure.get("exception_class") == "ExportError"
          and failure.get("published_prefix") == ["attempt.json", "predecessor_authority.json"],
          "V1 failure semantic/prefix drift")
    _need("aofm" in predecessor and "official_581361" in predecessor,
          "V1 predecessor authority semantic drift")
    return {
        "root_relative": plan.V1_FAILURE_ROOT_RELATIVE,
        "root_device": info.st_dev,
        "root_inode": info.st_ino,
        "bodies": dict(plan.V1_FAILURE_BODIES),
        "closure_sha256": plan.V1_FAILURE_CLOSURE_SHA256,
        "failure_stage": "post_predecessor_pre_payload_artifact_docker",
        "receipt_codec_recovery": "side_evidence.selected_indices+selected_indices_sha256",
    }

