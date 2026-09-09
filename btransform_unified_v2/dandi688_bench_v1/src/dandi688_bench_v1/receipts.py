"""Sealed JSON receipts (same law as btransform_unified_v1.receipts).

A sealed receipt is written atomically (temp file in the same directory,
flush+fsync, ``os.replace``), made read-only (0444), and pinned by a
``<name>.sha256`` sidecar holding the SHA-256 of the sealed bytes.
:func:`read_sealed` re-hashes the file and refuses to parse on any mismatch
with the sidecar (tamper-evidence).

Only ever write under ``btransform_unified_v2/dandi688_bench_v1/``; never
touch historical roots or rift_v1 contract files.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from . import plan

SEALED_MODE = 0o444


class ReceiptSealError(RuntimeError):
    """Raised on missing/tampered sealed receipts."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sidecar_path(path: Path | str) -> Path:
    p = Path(path)
    return p.with_name(p.name + ".sha256")


def seal_json(path: Path | str, payload: Any) -> str:
    """Atomically seal ``payload`` as JSON at ``path``; return the sha256 hex.

    The file is rendered deterministically (sorted keys), replaced atomically
    (replacing even a previous 0444 seal), chmod 0444, and pinned by a
    ``.sha256`` sidecar."""
    p = Path(path)
    plan.require(p.parent != Path(""), "seal target needs a directory")
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    data = text.encode("utf-8")
    digest = _sha256_bytes(data)
    tmp = p.with_name(f".{p.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(tmp, SEALED_MODE)
    os.replace(tmp, p)  # atomic; replaces even a previous 0444 seal
    sidecar = sidecar_path(p)
    sidecar.write_text(f"{digest}  {p.name}\n", encoding="utf-8")
    return digest


def read_sealed(path: Path | str) -> Any:
    """Verify the sidecar hash, then parse and return the sealed payload.

    Raises ReceiptSealError when the file or sidecar is missing, or when the
    file hash does not match the sidecar (tampered)."""
    p = Path(path)
    if not p.is_file():
        raise ReceiptSealError(f"sealed receipt missing: {p}")
    sidecar = sidecar_path(p)
    if not sidecar.is_file():
        raise ReceiptSealError(f"sealed receipt sidecar missing: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").split()[0].strip()
    actual = _sha256_bytes(p.read_bytes())
    if expected != actual:
        raise ReceiptSealError(
            f"sealed receipt hash mismatch for {p}: sidecar {expected} != file {actual} (tampered?)"
        )
    return json.loads(p.read_text(encoding="utf-8"))


__all__ = ["seal_json", "read_sealed", "sidecar_path", "SEALED_MODE", "ReceiptSealError"]
