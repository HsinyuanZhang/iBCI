"""Held-FD validator for the immutable V1 import-recovery predecessor."""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

from . import plan


class BindingError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BindingError(message)


def _required_flag(name: str) -> int:
    value = getattr(os, name, None)
    _require(isinstance(value, int) and value != 0, f"platform lacks required {name}")
    return value


def _read_regular(dirfd: int, name: str) -> bytes:
    fd = os.open(name, os.O_RDONLY | _required_flag("O_NOFOLLOW"), dir_fd=dirfd)
    try:
        info = os.fstat(fd)
        _require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444 and info.st_nlink == 1,
                 f"V1 predecessor leaf mode/type drift: {name}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(fd)


def validate_v1_failure_graph(repo_root: Path) -> dict[str, Any]:
    """Exact six-leaf, no-follow V1 failure witness.

    It opens only the named historical root; it neither imports Torch nor reads
    a checkpoint/data/result successor.
    """
    root = Path(repo_root).absolute()
    base = root / plan.V1_FAILURE_ROOT_RELATIVE
    before = os.lstat(base)
    _require(stat.S_ISDIR(before.st_mode) and not stat.S_ISLNK(before.st_mode), "V1 failure root type drift")
    expected_bodies = set(plan.V1_FAILURE_BODIES)
    expected = expected_bodies | {name + ".sha256" for name in expected_bodies}
    flags = os.O_RDONLY | _required_flag("O_DIRECTORY") | _required_flag("O_NOFOLLOW")
    dirfd = os.open(base, flags)
    try:
        opened = os.fstat(dirfd)
        _require((opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino), "V1 failure root swapped before read")
        _require(set(os.listdir(dirfd)) == expected, "V1 failure graph has missing/extra leaves")
        payloads: dict[str, dict[str, Any]] = {}
        digests: dict[str, str] = {}
        for name, expected_digest in plan.V1_FAILURE_BODIES.items():
            body = _read_regular(dirfd, name)
            digest = plan.sha256_bytes(body)
            _require(digest == expected_digest, f"V1 predecessor body SHA drift: {name}")
            side = _read_regular(dirfd, name + ".sha256")
            _require(side.decode("ascii") == f"{digest}  {name}\n", f"V1 predecessor sidecar drift: {name}")
            try:
                payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise BindingError(f"V1 predecessor JSON drift: {name}") from error
            _require(isinstance(payload, dict), f"V1 predecessor JSON object drift: {name}")
            payloads[name] = payload
            digests[name] = digest
        after = os.fstat(dirfd)
        _require((after.st_dev, after.st_ino) == (before.st_dev, before.st_ino), "V1 failure root swapped during read")
    finally:
        os.close(dirfd)

    attempt, launch, failure = (payloads[name] for name in ("attempt.json", "launch.json", "failure.json"))
    _require(attempt.get("schema") == "m2_postfusion_checkpoint_score_v1_attempt_v1"
             and attempt.get("status") == "ATTEMPT_RESERVED", "V1 attempt semantics drift")
    _require(attempt.get("target_or_checkpoint_opened") is False and attempt.get("cuda_initialized") is False,
             "V1 attempt pre-runtime facts drift")
    _require(launch.get("schema") == "m2_postfusion_checkpoint_score_v1_launch_v1"
             and launch.get("cuda_visible_devices") == "" and launch.get("cuda_initialized") is False,
             "V1 launch CPU semantics drift")
    _require(failure.get("schema") == "m2_postfusion_checkpoint_score_v1_failure_v1"
             and failure.get("status") == "FAILED" and failure.get("terminal_xor_failure") is True,
             "V1 failure terminal semantics drift")
    _require(failure.get("attempt_sha256") == digests["attempt.json"]
             and failure.get("error_sha256") == plan.V1_FAILURE_ERROR_SHA256,
             "V1 failure linkage/error drift")
    progress = failure.get("progress")
    _require(isinstance(progress, dict), "V1 failure progress absent")
    expected_progress = {
        "checkpoint_strict_load_attempted": True, "checkpoint_strict_loaded": True,
        "cuda_initialized": False, "pooled_descriptor_attempted": True,
        "pooled_descriptor_opened": True, "rows": 0,
        "screen_descriptor_attempted": True, "screen_descriptor_opened": True,
        "target_materialization_attempted": True, "target_materialized": False,
        "target_or_checkpoint_opened": True,
    }
    _require(all(progress.get(key) == value for key, value in expected_progress.items()),
             "V1 failure progress semantics drift")
    _require(not (base / "terminal.json").exists(), "V1 failure root cannot contain terminal")
    return {"root_relative": plan.V1_FAILURE_ROOT_RELATIVE, "root_identity": [before.st_dev, before.st_ino],
            "body_sha256": digests, "closure_sha256": attempt.get("closure_sha256"),
            "error_sha256": plan.V1_FAILURE_ERROR_SHA256}
