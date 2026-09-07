"""Lane receipt conventions: immutable 0444 body + canonical sidecar, fresh root.

Mirrors the P2' publisher: ``O_EXCL`` creation, 0444, fsync, and a
``<name>.sha256`` sidecar carrying the body digest.  Every receipt body also
carries the standard boundary rows (inference only, zero target
optimizer/backward/update, wall time).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


class ReceiptError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptError(message)


def json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")).encode("utf-8") + b"\n"


def publish(path: Path, payload: Mapping[str, Any]) -> str:
    """Atomically publish an immutable 0444 receipt body plus sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json_bytes(payload)
    digest = hashlib.sha256(body).hexdigest()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        os.write(descriptor, body)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    sidecar = path.with_name(path.name + ".sha256")
    descriptor = os.open(sidecar, flags, 0o444)
    try:
        os.write(descriptor, f"{digest}  {path.name}\n".encode("ascii"))
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return digest


def read_receipt(path: Path) -> dict[str, Any]:
    body = json.loads(Path(path).read_bytes())
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    sidecar = Path(path).with_name(path.name + ".sha256")
    _require(sidecar.exists(), f"receipt sidecar missing: {sidecar}")
    _require(
        sidecar.read_text(encoding="ascii").strip() == f"{digest}  {path.name}",
        f"receipt sidecar drift: {path}",
    )
    return body


def require_fresh(path: Path) -> None:
    _require(not path.exists(), f"receipt already exists (fresh root required): {path}")


def boundary_rows(*, wall_seconds: float, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    rows: dict[str, Any] = {
        "inference_only": True,
        "training_authorized": False,
        "target_optimizer_backward_update": 0,
        "target_backward_calls": 0,
        "target_update_calls": 0,
        "model_or_checkpoint_updated": False,
        "rng_in_evaluation": False,
        "wall_seconds": float(wall_seconds),
    }
    if extra:
        rows.update(dict(extra))
    return rows
