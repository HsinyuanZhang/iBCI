"""Held receipt topology validator; live issuance remains fail-closed."""
from __future__ import annotations

import json
import math
import os
import stat
from pathlib import Path
from typing import Mapping

from . import plan


class BindingError(RuntimeError):
    pass


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise BindingError(message)


def _descriptor_with_body(dirfd: int, name: str, *, parse_json: bool = True) -> tuple[dict[str, object] | None, str, bytes]:
    def read_leaf(leaf: str) -> tuple[bytes, os.stat_result]:
        fd = os.open(leaf, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=dirfd)
        try:
            info = os.fstat(fd); body = b""
            while True:
                block = os.read(fd, 1 << 20)
                if not block: break
                body += block
            return body, info
        finally: os.close(fd)
    body, info = read_leaf(name)
    _require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444 and info.st_nlink == 1,
             f"producer body mode/type drift: {name}")
    digest = plan.sha256_bytes(body)
    side_body, side = read_leaf(name + ".sha256")
    _require(stat.S_ISREG(side.st_mode) and stat.S_IMODE(side.st_mode) == 0o444 and side.st_nlink == 1,
             f"producer sidecar mode/type drift: {name}")
    _require(side_body.decode("ascii") == f"{digest}  {name}\n", f"producer sidecar drift: {name}")
    return (json.loads(body.decode("utf-8")) if parse_json else None), digest, body


def descriptor(dirfd: int, name: str, *, parse_json: bool = True) -> tuple[dict[str, object] | None, str]:
    """Descriptor-read one immutable body/sidecar pair from a held parent FD."""
    payload, digest, _body = _descriptor_with_body(dirfd, name, parse_json=parse_json)
    return payload, digest


def validate_screen_graph(root: Path, literals: Mapping[str, object], *, capture_checkpoint_bytes: bool = False) -> dict[str, object]:
    """Validate exactly the successful producer leaves needed for scoring."""
    base = Path(root).absolute(); before = os.lstat(base)
    _require(stat.S_ISDIR(before.st_mode) and not stat.S_ISLNK(before.st_mode), "producer root type drift")
    expected = {"attempt.json", "launch.json", "source_authority.json", "screen.json", "terminal.json",
                "source_best_pf_mean.pt", "source_best_pf_r1.pt", "source_best_pf_r50.pt"}
    observed = {item.name for item in base.iterdir()}
    _require(observed == expected | {name + ".sha256" for name in expected}, "producer graph has missing/extra leaves")
    payloads: dict[str, dict[str, object]] = {}; digests: dict[str, str] = {}; checkpoint_bytes: dict[str, bytes] = {}
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(base, flags)
    try:
      _require((os.fstat(fd).st_dev, os.fstat(fd).st_ino) == (before.st_dev, before.st_ino), "producer root swap")
      for name in ("attempt.json", "launch.json", "source_authority.json", "screen.json", "terminal.json"):
        payload, digests[name], _body = _descriptor_with_body(fd, name)
        _require(isinstance(payload, dict), f"producer JSON body drift: {name}"); payloads[name] = payload
      for name in sorted(expected - set(payloads)):
        _payload, digests[name], body = _descriptor_with_body(fd, name, parse_json=False)
        if capture_checkpoint_bytes:
            arm = next(arm for arm in plan.ARMS
                       if name == f"source_best_{arm.lower().replace('-', '_')}.pt")
            checkpoint_bytes[arm] = body
      _require((os.fstat(fd).st_dev, os.fstat(fd).st_ino) == (before.st_dev, before.st_ino), "producer root swapped during descriptor read")
    finally: os.close(fd)
    terminal = payloads["terminal.json"]; screen = payloads["screen.json"]
    _require(terminal.get("status") == "TERMINAL" and terminal.get("terminal_xor_failure") is True,
             "producer terminal semantics drift")
    _require(not (base / "failure.json").exists(), "successful producer cannot contain failure")
    _require(screen.get("matched_prefusion_control_trained") is False, "producer matched-control disclosure drift")
    checkpoints = screen.get("source_best_checkpoints")
    _require(isinstance(checkpoints, dict) and set(checkpoints) == set(plan.ARMS), "three producer checkpoints absent")
    for arm in plan.ARMS:
        entry = checkpoints[arm]
        _require(isinstance(entry, dict) and isinstance(entry.get("epoch"), int) and isinstance(entry.get("student_state_sha256"), str),
                 f"{arm}: checkpoint descriptor drift")
        filename = f"source_best_{arm.lower().replace('-', '_')}.pt"
        _require(entry.get("filename") == filename and entry.get("sha256") == digests[filename],
                 f"{arm}: checkpoint body linkage drift")
    for name, digest in digests.items(): _require(literals.get("sha256", {}).get(name) == digest, f"producer literal drift: {name}")
    _require(str(base).endswith(str(literals.get("screen_root_relative"))), "producer root literal drift")
    witness: dict[str, object] = {"payloads": payloads, "digests": digests,
                                  "root_identity": [before.st_dev, before.st_ino]}
    if capture_checkpoint_bytes:
        _require(set(checkpoint_bytes) == set(plan.ARMS), "held checkpoint byte capture incomplete")
        witness["checkpoint_bytes"] = checkpoint_bytes
    return witness


def require_live_literals() -> Mapping[str, object]:
    _require(plan.LIVE_PRODUCER_LITERALS is not None, "producer literals are deferred until successful terminal audit")
    return plan.LIVE_PRODUCER_LITERALS


def _required_flag(name: str) -> int:
    value = getattr(os, name, None)
    _require(isinstance(value, int) and value != 0, f"platform lacks required {name}")
    return value


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(letter in "0123456789abcdef" for letter in value)


def _held_leaf(dirfd: int, name: str) -> tuple[bytes, os.stat_result]:
    fd = os.open(name, os.O_RDONLY | _required_flag("O_NOFOLLOW"), dir_fd=dirfd)
    try:
        info = os.fstat(fd)
        _require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444 and info.st_nlink == 1,
                 f"POOLED predecessor leaf mode/type drift: {name}")
        blocks: list[bytes] = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            blocks.append(block)
        return b"".join(blocks), info
    finally:
        os.close(fd)


def _validate_pooled_payload(payload: object) -> dict[str, dict[str, object]]:
    """Validate the exact historical score codec and return its 13 rows.

    The fixed row stream check is intentional: a row map alone would accept a
    reordered historical receipt, losing the sealed cell/session chronology
    that the predecessor made reviewable.
    """
    _require(isinstance(payload, dict), "POOLED predecessor JSON is not an object")
    _require(payload.get("schema") == plan.POOLED_COMPARATOR_SCHEMA and payload.get("status") == "TERMINAL",
             "POOLED predecessor schema/status drift")
    _require(payload.get("row_count") == 130 and tuple(payload.get("cell_order", ())) == plan.POOLED_COMPARATOR_CELL_ORDER,
             "POOLED predecessor row-count/cell-order drift")
    _require(payload.get("protocol") == plan.POOLED_COMPARATOR_PROTOCOL,
             "POOLED predecessor protocol drift")
    rows = payload.get("rows")
    _require(isinstance(rows, list) and len(rows) == 130, "POOLED predecessor rows drift")
    expected_stream = [
        (surface, session, cell)
        for surface in ("within_post30", "external_post30_local")
        for session in plan.POOLED_SESSION_ORDER[surface]
        for cell in plan.POOLED_COMPARATOR_CELL_ORDER
    ]
    _require(len(expected_stream) == 130, "internal POOLED expected stream drift")
    result: dict[str, dict[str, object]] = {}
    required = {"cell", "system", "surface", "session_id", "budget", "r2", "window_count",
                "query_starts_sha256", "target_sha256", "prediction_sha256", "parameter_updates",
                "target_gradients", "target_backward", "target_state_uses", "seed"}
    for index, (row, expected) in enumerate(zip(rows, expected_stream, strict=True)):
        _require(isinstance(row, dict) and required <= set(row), f"POOLED row {index}: required evidence drift")
        surface, session, cell = expected
        _require((row.get("surface"), row.get("session_id"), row.get("cell")) == expected,
                 f"POOLED row {index}: canonical row order drift")
        _require(int(row["seed"]) == 42 and int(row["budget"]) == (4 if cell.startswith("m4_") else 10 if cell.startswith("m10_") else 30),
                 f"POOLED row {index}: budget/seed drift")
        _require(int(row["parameter_updates"]) == 0 and int(row["target_gradients"]) == 0 and int(row["target_backward"]) == 0
                 and int(row["target_state_uses"]) == 0, f"POOLED row {index}: forbidden update drift")
        _require(int(row["window_count"]) > 0 and math.isfinite(float(row["r2"]))
                 and all(_is_sha256(row[field]) for field in ("query_starts_sha256", "target_sha256", "prediction_sha256")),
                 f"POOLED row {index}: governed evidence drift")
        reference = rows[index - (index % len(plan.POOLED_COMPARATOR_CELL_ORDER))]
        _require(isinstance(reference, dict)
                 and row["query_starts_sha256"] == reference.get("query_starts_sha256")
                 and row["target_sha256"] == reference.get("target_sha256")
                 and int(row["window_count"]) == int(reference.get("window_count", -1)),
                 f"POOLED row {index}: governed evidence disagrees within session cell group")
        if cell == "m4_activity_only":
            _require(row.get("system") == "activity_only" and int(row["budget"]) == plan.BUDGET,
                     f"POOLED row {index}: selected m4 activity-only codec drift")
            key = f"{surface}|{session}"
            _require(key not in result, "POOLED selected-session duplicate")
            result[key] = dict(row)
    _require(len(result) == 13
             and {key.split("|", 1)[0] for key in result} == set(plan.SURFACES)
             and sum(key.startswith("external_post30_local|") for key in result) == 6
             and sum(key.startswith("within_post30|") for key in result) == 7,
             "POOLED selected m4 activity-only roster drift")
    return result


def validate_pooled_comparator_score(root: Path) -> dict[str, object]:
    """Held-FD, no-follow validation of the immutable POOLED k4 comparator.

    Only the exact two-leaf predecessor is opened.  It contains no checkpoint
    tensor and this function has no Torch/CUDA/data dependency.
    """
    base = Path(root).absolute()
    before = os.lstat(base)
    _require(stat.S_ISDIR(before.st_mode) and not stat.S_ISLNK(before.st_mode),
             "POOLED predecessor root type drift")
    flags = os.O_RDONLY | _required_flag("O_DIRECTORY") | _required_flag("O_NOFOLLOW")
    dirfd = os.open(base, flags)
    try:
        identity = os.fstat(dirfd)
        _require((identity.st_dev, identity.st_ino) == (before.st_dev, before.st_ino),
                 "POOLED predecessor root swapped before held read")
        _require(set(os.listdir(dirfd)) == {plan.POOLED_COMPARATOR_BODY,
                                             plan.POOLED_COMPARATOR_BODY + ".sha256"},
                 "POOLED predecessor has missing/extra leaves")
        body, _info = _held_leaf(dirfd, plan.POOLED_COMPARATOR_BODY)
        digest = plan.sha256_bytes(body)
        _require(digest == plan.POOLED_COMPARATOR_SHA256, "POOLED predecessor body SHA drift")
        side, _side_info = _held_leaf(dirfd, plan.POOLED_COMPARATOR_BODY + ".sha256")
        _require(side.decode("ascii") == f"{digest}  {plan.POOLED_COMPARATOR_BODY}\n",
                 "POOLED predecessor sidecar drift")
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BindingError("POOLED predecessor JSON decode drift") from error
        selected = _validate_pooled_payload(payload)
        after = os.fstat(dirfd)
        _require((after.st_dev, after.st_ino) == (before.st_dev, before.st_ino),
                 "POOLED predecessor root swapped during held read")
    finally:
        os.close(dirfd)
    return {"root_identity": [before.st_dev, before.st_ino], "body_sha256": digest,
            "score": payload, "comparators": selected}
