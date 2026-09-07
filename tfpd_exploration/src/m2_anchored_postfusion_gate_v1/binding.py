"""Held-FD codec for the immutable operator-corrected APFG predecessor."""
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


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise BindingError(message)


def _flag(name: str) -> int:
    value = getattr(os, name, None)
    _require(isinstance(value, int) and value != 0, f"platform lacks {name}")
    return value


def _read(dirfd: int, name: str) -> bytes:
    fd = os.open(name, os.O_RDONLY | _flag("O_NOFOLLOW"), dir_fd=dirfd)
    try:
        info = os.fstat(fd)
        _require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444 and info.st_nlink == 1,
                 f"corrected predecessor leaf type/mode/link drift: {name}")
        blocks: list[bytes] = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                return b"".join(blocks)
            blocks.append(block)
    finally:
        os.close(fd)


def _json(body: bytes, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BindingError(f"corrected predecessor JSON drift: {name}") from error
    _require(isinstance(payload, dict), f"corrected predecessor body is not object: {name}")
    return payload


def validate_operator_corrected_graph(repo_root: Path) -> dict[str, object]:
    """Descriptor-read the exact ten-leaf completed graph, fail closed on drift."""
    root = Path(repo_root).absolute()
    graph = root / plan.CORRECTED_ROOT_RELATIVE
    before = os.lstat(graph)
    _require(stat.S_ISDIR(before.st_mode) and not stat.S_ISLNK(before.st_mode), "corrected predecessor root type drift")
    expected_names = set(plan.CORRECTED_BODIES)
    expected = expected_names | {name + ".sha256" for name in expected_names}
    directory = os.open(graph, os.O_RDONLY | _flag("O_DIRECTORY") | _flag("O_NOFOLLOW"))
    try:
        opened = os.fstat(directory)
        _require((opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino), "corrected root swapped before read")
        _require(set(os.listdir(directory)) == expected, "corrected predecessor leaf topology drift")
        payloads: dict[str, dict[str, Any]] = {}
        digests: dict[str, str] = {}
        for name, literal in plan.CORRECTED_BODIES.items():
            body = _read(directory, name)
            digest = hashlib.sha256(body).hexdigest()
            _require(digest == literal, f"corrected predecessor body SHA drift: {name}")
            sidecar = _read(directory, name + ".sha256")
            _require(sidecar == f"{digest}  {name}\n".encode("ascii"), f"corrected predecessor sidecar drift: {name}")
            payloads[name] = _json(body, name)
            digests[name] = digest
        after = os.fstat(directory)
        _require((after.st_dev, after.st_ino) == (before.st_dev, before.st_ino), "corrected root swapped during read")
    finally:
        os.close(directory)
    attempt, launch, authority, score, terminal = (payloads[name] for name in plan.CORRECTED_BODIES)
    _require(terminal.get("schema") == "m2_postfusion_operator_corrected_score_v1_terminal_v1"
             and terminal.get("status") == "TERMINAL" and terminal.get("row_count") == 78
             and terminal.get("cuda_initialized") is False and terminal.get("pfmean_v2_control_exact") is True,
             "corrected predecessor terminal semantics drift")
    _require(attempt.get("closure_sha256") == plan.CORRECTED_CLOSURE_SHA256,
             "corrected predecessor attempt closure drift")
    _require(terminal.get("attempt_sha256") == digests["attempt.json"]
             and terminal.get("launch_sha256") == digests["launch.json"]
             and terminal.get("input_authority_sha256") == digests["input_authority.json"]
             and terminal.get("score_sha256") == digests["score.json"],
             "corrected predecessor terminal linkage drift")
    _require(isinstance(score.get("rows"), list) and len(score["rows"]) == 78, "corrected predecessor score topology drift")
    _require(authority.get("cuda_initialized") is False and attempt.get("cuda_initialized") is False
             and launch.get("cuda_initialized") is False, "corrected predecessor CUDA semantics drift")
    return {"root_relative": plan.CORRECTED_ROOT_RELATIVE, "root_identity": [before.st_dev, before.st_ino],
            "body_sha256": digests, "closure_sha256": plan.CORRECTED_CLOSURE_SHA256,
            "terminal": terminal}


__all__ = ("BindingError", "validate_operator_corrected_graph")
