"""Immutable attempt/terminal/failure leaves shared by executable route stages."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def publish_immutable_json(root: Path, name: str, payload: object) -> str:
    root = Path(root)
    target, sidecar = root / name, root / f"{name}.sha256"
    if target.exists() or sidecar.exists():
        raise FileExistsError(f"immutable leaf already exists: {target}")
    body = canonical_bytes(payload)
    digest = hashlib.sha256(body).hexdigest()
    for destination, content in ((target, body), (sidecar, f"{digest}  {name}\n".encode("ascii"))):
        with tempfile.NamedTemporaryFile(dir=root, prefix=f".{destination.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, destination)
    return digest


def begin_attempt(root: Path, payload: dict[str, Any]) -> str:
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"refusing to reuse lifecycle root {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(mode=0o755)
    payload = dict(payload)
    payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    payload.setdefault("status", "STARTED")
    return publish_immutable_json(root, "attempt.json", payload)


def close_terminal(root: Path, *, attempt_sha256: str, payload: dict[str, Any]) -> str:
    body = dict(payload)
    body.update({"status": "PASS", "attempt_sha256": attempt_sha256})
    return publish_immutable_json(root, "terminal.json", body)


def close_failure(root: Path, *, attempt_sha256: str, error: BaseException) -> str:
    prefix = sorted(path.name for path in Path(root).iterdir())
    return publish_immutable_json(root, "failure.json", {"status": "FAIL", "attempt_sha256": attempt_sha256, "published_prefix": prefix, "exception_type": type(error).__name__, "message": str(error)})
