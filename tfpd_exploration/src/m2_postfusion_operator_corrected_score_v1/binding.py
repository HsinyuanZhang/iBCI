"""Held-FD validator for the immutable successful literal V2 score."""
from __future__ import annotations

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
                 f"V2 leaf type/mode/link drift: {name}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(fd)


def _payload(body: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BindingError(f"V2 JSON drift: {name}") from error
    _require(isinstance(value, dict), f"V2 payload is not object: {name}")
    return value


def _sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_v2_success_graph(repo_root: Path) -> dict[str, Any]:
    """Read only the named ten-leaf V2 graph with a held directory FD.

    This is deliberately independent from the V1 score binding.  The successor
    cannot silently accept a V2-like receipt with a different terminal schema,
    a missing literal PF-MEAN control, or an extra failure body.
    """
    root = Path(repo_root).absolute()
    base = root / plan.V2_ROOT_RELATIVE
    before = os.lstat(base)
    _require(stat.S_ISDIR(before.st_mode) and not stat.S_ISLNK(before.st_mode), "V2 root type drift")
    names = tuple(plan.V2_BODIES)
    expected = set(names) | {name + ".sha256" for name in names}
    fd = os.open(base, os.O_RDONLY | _flag("O_DIRECTORY") | _flag("O_NOFOLLOW"))
    try:
        opened = os.fstat(fd)
        _require((opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino), "V2 root swap before read")
        _require(set(os.listdir(fd)) == expected, "V2 graph missing/extra leaf")
        payloads: dict[str, dict[str, Any]] = {}
        digests: dict[str, str] = {}
        for name, expected_sha in plan.V2_BODIES.items():
            body = _read(fd, name)
            digest = plan.sha256_bytes(body)
            _require(digest == expected_sha, f"V2 body SHA drift: {name}")
            side = _read(fd, name + ".sha256")
            _require(side.decode("ascii") == f"{digest}  {name}\n", f"V2 sidecar drift: {name}")
            payloads[name] = _payload(body, name)
            digests[name] = digest
        after = os.fstat(fd)
        _require((after.st_dev, after.st_ino) == (before.st_dev, before.st_ino), "V2 root swap during read")
    finally:
        os.close(fd)
    attempt = payloads["attempt.json"]
    launch = payloads["launch.json"]
    authority = payloads["input_authority.json"]
    score = payloads["score.json"]
    terminal = payloads["terminal.json"]
    _require(attempt.get("schema") == "m2_postfusion_checkpoint_score_v2_attempt_v1"
             and attempt.get("status") == "ATTEMPT_RESERVED" and attempt.get("cuda_initialized") is False,
             "V2 attempt semantics drift")
    _require(launch.get("schema") == "m2_postfusion_checkpoint_score_v2_launch_v1"
             and launch.get("cuda_visible_devices") == "" and launch.get("cuda_initialized") is False,
             "V2 launch CPU semantics drift")
    _require(authority.get("schema") == "m2_postfusion_checkpoint_score_v2_input_authority_v1"
             and authority.get("record_count") == 13 and isinstance(authority.get("records"), dict)
             and len(authority["records"]) == 13, "V2 input authority semantics drift")
    _require(score.get("schema") == "m2_postfusion_checkpoint_score_v2_score_v1"
             and isinstance(score.get("rows"), list) and len(score["rows"]) == plan.EXPECTED_ROWS,
             "V2 score topology drift")
    _require(terminal.get("schema") == "m2_postfusion_checkpoint_score_v2_terminal_v1"
             and terminal.get("status") == "TERMINAL" and terminal.get("terminal_xor_failure") is True
             and terminal.get("attempt_sha256") == digests["attempt.json"]
             and terminal.get("launch_sha256") == digests["launch.json"]
             and terminal.get("input_authority_sha256") == digests["input_authority.json"]
             and terminal.get("score_sha256") == digests["score.json"]
             and terminal.get("current_closure_sha256") == plan.V2_CLOSURE_SHA256,
             "V2 terminal/linkage/closure drift")
    keys = [(row.get("surface"), row.get("session"), row.get("arm"), row.get("memory_law")) for row in score["rows"]]
    _require(len(set(keys)) == plan.EXPECTED_ROWS, "V2 score duplicate row")
    control: dict[str, dict[str, Any]] = {}
    for row in score["rows"]:
        if row.get("arm") == "PF-MEAN":
            key = f"{row.get('surface')}|{row.get('session')}|{row.get('memory_law')}"
            _require(key not in control and _sha(row.get("prediction_sha256")) and _sha(row.get("target_sha256")),
                     "V2 PF-MEAN control row drift")
            control[key] = dict(row)
    _require(len(control) == 26, "V2 must provide exact 26-row PF-MEAN control")
    return {"root_relative": plan.V2_ROOT_RELATIVE, "root_identity": [before.st_dev, before.st_ino],
            "body_sha256": digests, "closure_sha256": plan.V2_CLOSURE_SHA256,
            "input_authority": authority, "score": score, "pfmean_control": control}

