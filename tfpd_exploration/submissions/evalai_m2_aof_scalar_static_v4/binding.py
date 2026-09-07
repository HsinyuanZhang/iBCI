"""Held-FD verification of the exact V3 failure and V2 sealed payload graphs."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping

from . import plan


class BindingError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise BindingError(message)


def _fd_read(directory_fd: int, name: str) -> tuple[bytes, str]:
    _need(hasattr(os, "O_NOFOLLOW"), "AOF-S V4 requires O_NOFOLLOW")
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    try:
        info = os.fstat(fd)
        _need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444 and info.st_nlink == 1,
              f"AOF-S V4 immutable leaf mode/link drift: {name}")
        chunks: list[bytes] = []
        while block := os.read(fd, 1 << 20):
            chunks.append(block)
        body = b"".join(chunks)
    finally:
        os.close(fd)
    return body, hashlib.sha256(body).hexdigest()


def _graph(root: Path, expected_bodies: Mapping[str, str], label: str) -> tuple[dict[str, bytes], dict[str, int]]:
    _need(hasattr(os, "O_NOFOLLOW"), "AOF-S V4 requires O_NOFOLLOW")
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        identity = os.fstat(fd)
        expected = set(expected_bodies) | {name + ".sha256" for name in expected_bodies}
        _need(set(os.listdir(fd)) == expected, f"{label} exact topology drift")
        bodies: dict[str, bytes] = {}
        for name, wanted in expected_bodies.items():
            body, digest = _fd_read(fd, name)
            side, _ = _fd_read(fd, name + ".sha256")
            _need(digest == wanted and side == f"{digest}  {name}\n".encode("ascii"),
                  f"{label} body/sidecar drift: {name}")
            bodies[name] = body
    finally:
        os.close(fd)
    return bodies, {"device": identity.st_dev, "inode": identity.st_ino}


def validate_v3_failure_graph(repo_root: Path = plan.REPO_ROOT) -> dict[str, Any]:
    """Bind exactly four V3 bodies/eight leaves; reject its invalid image as evidence."""
    bodies, identity = _graph(repo_root / plan.V3_FAILURE_ROOT_RELATIVE, plan.V3_FAILURE_BODIES, "V3 failure")
    attempt = json.loads(bodies["attempt.json"])
    predecessor = json.loads(bodies["predecessor_authority.json"])
    input_authority = json.loads(bodies["input_authority.json"])
    failure = json.loads(bodies["failure.json"])
    _need(attempt.get("schema") == "m2_aof_scalar_static_package_v3_attempt"
          and attempt.get("status") == "LOCAL_BUILD_V3_RESERVED_NO_NETWORK"
          and attempt.get("closure_sha256") == plan.V3_FAILURE_CLOSURE_SHA256,
          "V3 attempt schema/status/closure drift")
    _need(set(predecessor) == {"v2_failure", "v2_sealed_artifact"}
          and predecessor["v2_sealed_artifact"].get("bodies") == plan.V2_ARTIFACT_BODIES,
          "V3 predecessor payload lineage drift")
    _need(input_authority.get("scientific_rebuild") is False
          and input_authority.get("reused_v2_payload", {}).get("sha256")
          == "d13dffb80227440ea4ab403011e5028d6fe9ab02d8c0b3bcdd2093bf8beea43b",
          "V3 input authority reuse semantics drift")
    _need(failure == {
        "exception_class": "DockerArgvError",
        "exception_message": "AOF-S V3 Docker command failed: Traceback (most recent call last):\n  File \"/workspace/aofs_static_decoder.py\", line 20, in <module>\n    _REPO_ROOT = Path(__file__).resolve().parents[3]\n  File \"/opt/conda/envs/spint/lib/python3.10/pathlib.py\", line 530, in __getitem__\n    raise IndexError(idx)\nIndexError: 3\n",
        "network_submission": False,
        "published_prefix": ["attempt.json", "predecessor_authority.json", "input_authority.json"],
        "schema": "m2_aof_scalar_static_package_v3_failure", "status": "LOCAL_BUILD_FAILED",
    }, "V3 layout-failure semantic/prefix drift")
    return {"root_relative": plan.V3_FAILURE_ROOT_RELATIVE, "root": identity,
            "bodies": dict(plan.V3_FAILURE_BODIES), "closure_sha256": plan.V3_FAILURE_CLOSURE_SHA256,
            "failure_stage": "post_input_docker_container_layout"}


def validate_v2_sealed_artifact(repo_root: Path = plan.REPO_ROOT) -> dict[str, Any]:
    bodies, identity = _graph(repo_root / plan.V2_ARTIFACT_ROOT_RELATIVE, plan.V2_ARTIFACT_BODIES, "V2 artifact")
    receipt = json.loads(bodies[plan.RECEIPT_NAME])
    _need(receipt.get("payload_sha256") == plan.V2_ARTIFACT_BODIES[plan.PAYLOAD_NAME]
          and receipt.get("recovery_profile") == "aofs_static_v2_side_evidence_receipt_recovery",
          "V2 receipt/payload link or recovery semantics drift")
    return {"root_relative": plan.V2_ARTIFACT_ROOT_RELATIVE, "root": identity,
            "bodies": dict(plan.V2_ARTIFACT_BODIES), "receipt": receipt,
            "payload_path": str(repo_root / plan.V2_ARTIFACT_ROOT_RELATIVE / plan.PAYLOAD_NAME)}
