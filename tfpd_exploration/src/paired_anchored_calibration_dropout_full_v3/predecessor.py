"""Held-FD validator for the immutable V2 pre-update P0 failure."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

from . import plan


class PredecessorError(RuntimeError):
    pass


LEAVES = tuple(sorted(
    leaf for body in plan.V2_FAILURE_SHAS for leaf in (body, body + ".sha256")
))
_V2_CLOSURE = "74b9495597fa8ca1b2a6731d78c2fb10108e47e2fc4cc00540b34d68c377763c"
_V1_PREDECESSOR = {
    "failure_sha256": "65b125b982506c82a9daa9eb32d88fed1afca7355155abeab2c224f31b4c3d08",
    "relative": "tfpd_exploration/results/paired_anchored_calibration_dropout_full_v1/p0_fullfull_seed42",
    "topology": [
        "attempt.json", "attempt.json.sha256", "failure.json", "failure.json.sha256",
        "launch.json", "launch.json.sha256", "source_authority.json", "source_authority.json.sha256",
    ],
}


def _read_all(fd: int) -> bytes:
    chunks: list[bytes] = []
    while chunk := os.read(fd, 65_536):
        chunks.append(chunk)
    return b"".join(chunks)


def _read_regular(dfd: int, name: str, expected: str) -> dict:
    st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
    if not stat.S_ISREG(st.st_mode) or stat.S_IMODE(st.st_mode) != 0o444:
        raise PredecessorError(f"invalid mode/type {name}")
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dfd)
    try:
        body = _read_all(fd)
    finally:
        os.close(fd)
    if hashlib.sha256(body).hexdigest() != expected:
        raise PredecessorError(f"body sha {name}")
    side_name = name + ".sha256"
    st = os.stat(side_name, dir_fd=dfd, follow_symlinks=False)
    if not stat.S_ISREG(st.st_mode) or stat.S_IMODE(st.st_mode) != 0o444:
        raise PredecessorError(f"invalid side mode/type {name}")
    fd = os.open(side_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dfd)
    try:
        side = _read_all(fd).decode("utf-8")
    finally:
        os.close(fd)
    if side != f"{expected}  {name}\n":
        raise PredecessorError(f"sidecar {name}")
    return json.loads(body)


def validate_v2_failure(root: Path, relative: str = plan.V2_FAILURE_RELATIVE) -> dict:
    """Exact V2 failure graph validation before V3 mutation and at publication."""
    try:
        dfd = os.open(root / relative, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise PredecessorError(str(exc)) from exc
    try:
        if set(os.listdir(dfd)) != set(LEAVES):
            raise PredecessorError("topology")
        bodies = {
            name: _read_regular(dfd, name, expected)
            for name, expected in plan.V2_FAILURE_SHAS.items()
        }
        attempt = bodies["attempt.json"]
        launch = bodies["launch.json"]
        authority = bodies["source_authority.json"]
        failure = bodies["failure.json"]
        if attempt.get("schema") != "pacd_matched_full_training_v2_attempt" or attempt.get("arm") != "p0":
            raise PredecessorError("attempt semantics")
        if attempt.get("source_closure", {}).get("closure_sha256") != _V2_CLOSURE:
            raise PredecessorError("attempt closure")
        if launch.get("schema") != "pacd_matched_full_training_v2_launch" or launch.get("target_access") is not False:
            raise PredecessorError("launch semantics")
        if launch.get("attempt", {}).get("sha256") != plan.V2_FAILURE_SHAS["attempt.json"] or launch.get("source_authority", {}).get("sha256") != plan.V2_FAILURE_SHAS["source_authority.json"]:
            raise PredecessorError("launch descriptor links")
        if authority.get("target_access") is not False or authority.get("attempt_sha256") != plan.V2_FAILURE_SHAS["attempt.json"]:
            raise PredecessorError("source authority semantics")
        if failure.get("schema") != "pacd_matched_full_training_v2_failure" or failure.get("status") != "CELL_FAILED":
            raise PredecessorError("failure status")
        error = failure.get("failure", {})
        if error.get("kind") != "PACDError" or error.get("detail") != "encoder gradient is zero":
            raise PredecessorError("failure cause")
        progress = failure.get("progress", {})
        if any(progress.get(key) != expected for key, expected in (
            ("epochs_published", 0), ("checkpoints_published", 0), ("swa_published", False),
        )):
            raise PredecessorError("failure progress")
        if failure.get("target_access") is not False or failure.get("source_closure", {}).get("launch", {}).get("closure_sha256") != _V2_CLOSURE or failure.get("source_closure", {}).get("final", {}).get("closure_sha256") != _V2_CLOSURE:
            raise PredecessorError("failure closure/target")
        if failure.get("predecessor") != _V1_PREDECESSOR:
            raise PredecessorError("V1 predecessor binding")
        return {
            "relative": relative,
            "failure_sha256": plan.V2_FAILURE_SHAS["failure.json"],
            "topology": list(LEAVES),
            "v1_predecessor": _V1_PREDECESSOR,
        }
    finally:
        os.close(dfd)
