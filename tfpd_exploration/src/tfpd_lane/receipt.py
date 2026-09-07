"""Transactional receipt utilities for the tfpd_lane deliverables.

Replicates the sealed Stage-0 receipt discipline
(`scripts/run_stage0_synth.py` §4) for the new lane runners:

- CPU/env hard gate (no user-site, empty CUDA_VISIBLE_DEVICES, no CUDA);
- SHA-256 source closure over the runner's bound files, hashed at launch and
  re-hashed before the receipt is written (drift = fail-closed);
- transactional write: private temp file, fsync, chmod 0444-style read-only,
  then `os.link` with O_EXCL semantics so an existing receipt can never be
  overwritten;
- sidecar ``.sha256`` written by the same procedure;
- failure receipts are minted by the same path (no partial pass exists).

This module is new file surface only; it does not touch `src/tfpd/` or the
sealed Stage-0 runner.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import sys
import tempfile
from pathlib import Path

_READ_ONLY_MODE = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_closure(root: Path, patterns: tuple[str, ...]) -> dict:
    """SHA-256 closure over every file matching `patterns`, relative to `root`."""
    files: dict[str, dict] = {}
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            files[str(path.relative_to(root))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    closure = hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()
    return {"files": files, "closure_sha256": closure}


def enforce_environment() -> dict:
    problems = []
    if not sys.flags.no_user_site:
        problems.append("PYTHONNOUSERSITE is not set (user-site shadowing risk)")
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        problems.append("CUDA_VISIBLE_DEVICES is not the empty string")
    if problems:
        print("environment hard gate failed: " + "; ".join(problems), file=sys.stderr)
        raise SystemExit(3)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "no_user_site": bool(sys.flags.no_user_site),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def write_receipt_transactionally(path: Path, payload: dict) -> None:
    """fsync temp file, then link into place with O_EXCL; no overwrite path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".receipt-", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, _READ_ONLY_MODE)
        # O_EXCL link: fails if the receipt already exists — immutable roots.
        os.link(tmp_path, path)
        os.unlink(tmp_path)
        sidecar = path.with_suffix(path.suffix + ".sha256")
        sidecar_body = (sha256_file(path) + "  " + path.name + "\n").encode("utf-8")
        sfd, s_tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".sidecar-", suffix=".tmp")
        s_path = Path(s_tmp)
        with os.fdopen(sfd, "wb") as handle:
            handle.write(sidecar_body)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(s_path, _READ_ONLY_MODE)
        os.link(s_path, sidecar)
        os.unlink(s_path)
    except FileExistsError:
        tmp_path.unlink(missing_ok=True)
        print(f"refusing to overwrite existing receipt {path}", file=sys.stderr)
        raise SystemExit(2)
    finally:
        tmp_path.unlink(missing_ok=True)
