#!/usr/bin/env python3
"""Clean-process production quarantine for external sub-M V8.

Run only as ``python -I <this-script>``.  This script does not import the V8
core module or any project module.  Before handling a mode it validates the
canonical local V8 core source and the canonical V8 anchor by fixed absolute
path, owner, mode, byte count, and SHA-256 using dirfd/openat/O_NOFOLLOW.

The pinned anchor is intentionally blocked.  Therefore this process can only
report status or fail closed; it cannot receive roots, keys, policies,
checkpoints, paths, payloads, grants, or formal-scoring authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any


WORKSPACE = Path("/home/xinyuan/Work_host/SPINT")
SELF = WORKSPACE / "sua_exploration/scripts/run_dandi688_subm_co_three_arm_score_only_v8_production.py"
CORE = WORKSPACE / "sua_exploration/mc_maze/subm_co_three_arm_score_only_v8.py"
ANCHOR = WORKSPACE / "sua_exploration/configs/dandi_000688_subm_v8_pinned_production_anchor.json"
EXPECTED_UID = 1002
EXPECTED_MODE = 0o444
EXPECTED_CORE_SHA256 = "8de87dae2f18ca89efcc17d6d6dbb1ef429011fc7be28adc000fff19febedec6"
EXPECTED_CORE_BYTES = 10_932
EXPECTED_ANCHOR_SHA256 = "5a87ec84429d39986b0e6cfa96b6bc6bc34243b8a026d82194799745e0f18cfc"
EXPECTED_ANCHOR_BYTES = 215
EXPECTED_ANCHOR_PAYLOAD = {
    "anchor_id": "subm-v8-production-formal-quarantine",
    "kind": "dandi_000688_subm_v8_pinned_production_anchor",
    "schema": "dandi_000688_subm_v8_pinned_production_anchor",
    "status": "BLOCKED_NO_ACTIVE_FORMAL_ROOT_CHAIN_V8",
}
BLOCKED_STATUS = "BLOCKED_MISSING_ZERO4_TERMINALS"


class GuardError(RuntimeError):
    pass


class GuardBlocked(GuardError):
    pass


def _open_absolute_regular(path: Path, *, expected_sha256: str, expected_bytes: int) -> bytes:
    """Read an exact owned immutable regular file through a dirfd path chain."""
    if not path.is_absolute():
        raise GuardError("fixed V8 path is not absolute")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    root_fd = os.open(path.anchor, directory_flags)
    current_fd = root_fd
    descriptor = -1
    try:
        for part in path.parts[1:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=current_fd)
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = next_fd
        descriptor = os.open(path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0), dir_fd=current_fd)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise GuardError("V8 fixed path is not a regular file")
        if before.st_uid != EXPECTED_UID or stat.S_IMODE(before.st_mode) != EXPECTED_MODE or before.st_size != expected_bytes:
            raise GuardError("V8 fixed file owner/mode/bytes drift")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 64 * 1024)
            if not block:
                break
            chunks.append(block)
            if sum(len(item) for item in chunks) > expected_bytes:
                raise GuardError("V8 fixed file exceeds byte cap")
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise GuardError("V8 fixed file changed during read")
        leaf = os.stat(path.name, dir_fd=current_fd, follow_symlinks=False)
        if stat.S_ISLNK(leaf.st_mode) or (leaf.st_dev, leaf.st_ino) != (after.st_dev, after.st_ino):
            raise GuardError("V8 fixed file path identity drift")
        raw = b"".join(chunks)
    except OSError as exc:
        raise GuardError("cannot secure-open V8 fixed file") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)
    if len(raw) != expected_bytes or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise GuardError("V8 fixed file SHA-256/bytes drift")
    return raw


def verify_local_quarantine() -> dict[str, Any]:
    """Validate core+anchor without importing either as executable authority."""
    if not sys.flags.isolated:
        raise GuardError("V8 production guard requires Python isolated mode: python -I")
    self_path = Path(os.path.abspath(__file__))
    if self_path != SELF or self_path.is_symlink():
        raise GuardError("V8 production guard must run from its canonical path")
    self_stat = self_path.stat()
    if not stat.S_ISREG(self_stat.st_mode) or self_stat.st_uid != EXPECTED_UID or stat.S_IMODE(self_stat.st_mode) != EXPECTED_MODE:
        raise GuardError("V8 production guard owner/mode drift")
    core_raw = _open_absolute_regular(CORE, expected_sha256=EXPECTED_CORE_SHA256, expected_bytes=EXPECTED_CORE_BYTES)
    anchor_raw = _open_absolute_regular(ANCHOR, expected_sha256=EXPECTED_ANCHOR_SHA256, expected_bytes=EXPECTED_ANCHOR_BYTES)
    try:
        anchor = json.loads(anchor_raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GuardError("V8 canonical anchor malformed") from exc
    canonical = (json.dumps(anchor, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode()
    if not isinstance(anchor, dict) or anchor != EXPECTED_ANCHOR_PAYLOAD or canonical != anchor_raw:
        raise GuardError("V8 canonical anchor schema/status drift")
    return {
        "status": BLOCKED_STATUS,
        "core": {"sha256": hashlib.sha256(core_raw).hexdigest(), "bytes": len(core_raw), "mode": "0444", "uid": EXPECTED_UID},
        "anchor": {"sha256": hashlib.sha256(anchor_raw).hexdigest(), "bytes": len(anchor_raw), "mode": "0444", "uid": EXPECTED_UID},
        "formal_root_chain_active": False,
        "external_subm_checkpoint_torch_gpu_r2_policy_grant_allowed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("status", "formal"), default="status")
    args = parser.parse_args(argv)
    try:
        result = verify_local_quarantine()
        if args.mode == "status":
            print(json.dumps(result, sort_keys=True, indent=2)); return 0
        raise GuardBlocked(BLOCKED_STATUS)
    except GuardBlocked as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr); return 2
    except GuardError as exc:
        print(f"INTEGRITY_FAIL_CLOSED: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
