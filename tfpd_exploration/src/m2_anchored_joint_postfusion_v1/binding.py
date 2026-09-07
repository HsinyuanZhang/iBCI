"""Descriptor-first AJPF predecessor admission; no Torch/data imports."""
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


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise BindingError(message)


def _read(fd: int, name: str) -> tuple[bytes, str]:
    _need(hasattr(os, "O_NOFOLLOW"), "AJPF requires O_NOFOLLOW")
    leaf = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    try:
        info = os.fstat(leaf)
        _need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444 and info.st_nlink == 1,
              f"AJPF immutable descriptor mode/link drift: {name}")
        data = b"".join(iter(lambda: os.read(leaf, 1 << 20), b""))
    finally:
        os.close(leaf)
    return data, hashlib.sha256(data).hexdigest()


def held_graph(root: Path, expected: Mapping[str, str], *, label: str, exact_topology: bool = False) -> tuple[dict[str, Any], dict[str, int]]:
    _need(hasattr(os, "O_NOFOLLOW"), "AJPF requires O_NOFOLLOW")
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        identity = os.fstat(directory)
        required = set(expected) | {name + ".sha256" for name in expected}
        actual = set(os.listdir(directory))
        _need(actual == required if exact_topology else actual >= required, f"AJPF {label} topology drift")
        result: dict[str, Any] = {}
        for name, digest in expected.items():
            body, observed = _read(directory, name)
            side, _ = _read(directory, name + ".sha256")
            _need(observed == digest and side == f"{observed}  {name}\n".encode("ascii"),
                  f"AJPF {label} body/sidecar drift: {name}")
            result[name] = json.loads(body.decode("utf-8"))
    finally:
        os.close(directory)
    return result, {"device": identity.st_dev, "inode": identity.st_ino}


def validate_apfg_source(root: Path) -> dict[str, Any]:
    bodies, identity = held_graph(root, plan.APFG_SOURCE_BODIES, label="APFG source")
    source = bodies["source_authority.json"]
    _need(source.get("activity_authority") == plan.ACTIVITY_AUTHORITY
          and source.get("m30_causal_coordinate_count") == plan.SOURCE_COORDINATES
          and source.get("ordered_coordinate_sha256") == plan.SOURCE_COORDINATE_SHA256
          and source.get("ordered_batch_sha256") == plan.SOURCE_BATCH_SHA256,
          "AJPF APFG source semantic drift")
    return {"root": identity, "bodies": dict(plan.APFG_SOURCE_BODIES), "source_authority": source}


def validate_pooled_score(root: Path) -> dict[str, Any]:
    bodies, identity = held_graph(root, {"score.json": plan.POOLED_SCORE_SHA256}, label="POOLED score", exact_topology=True)
    score = bodies["score.json"]
    _need(score.get("row_count") == 130 and score.get("status") in {"TERMINAL", "COMPLETE", "SCORED"},
          "AJPF POOLED score schema/status drift")
    return {"root": identity, "score_sha256": plan.POOLED_SCORE_SHA256, "score": score}
