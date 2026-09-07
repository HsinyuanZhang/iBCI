"""Crash-durable immutable JSON artifacts for the A1 development screen.

Every accepted A1 artifact is a non-symlink regular file with mode 0444 and
an independently written ``.sha256`` sidecar.  Writers use ``O_EXCL`` and
fsync both files and their parent directory.  Readers reject either half of
an incomplete pair, altered bytes, permissive modes, and symlinks.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping


class A1ArtifactError(ValueError):
    """An immutable A1 artifact is absent, mutable, or inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise A1ArtifactError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        (
            json.dumps(
                dict(payload),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()


def sidecar_path(path: Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".sha256")


def _is_immutable_regular(path: Path) -> bool:
    try:
        metadata = Path(path).lstat()
    except OSError:
        return False
    return stat.S_ISREG(metadata.st_mode) and stat.S_IMODE(metadata.st_mode) == 0o444


def write_immutable_json(
    path: Path, payload: Mapping[str, Any]
) -> tuple[Path, Path, str]:
    requested = Path(path).expanduser()
    require(not requested.is_symlink(), f"receipt path may not be a symlink: {requested}")
    body = requested.resolve()
    sidecar = sidecar_path(body)
    if os.path.lexists(body) or os.path.lexists(sidecar):
        raise FileExistsError(f"refusing immutable overwrite: {body}")
    body.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json_bytes(payload)
    digest = hashlib.sha256(raw).hexdigest()

    descriptor = os.open(body, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(body, 0o444)
    require(_is_immutable_regular(body), f"immutable body mode failed: {body}")

    sidecar_raw = f"{digest}  {body.name}\n".encode("ascii")
    descriptor = os.open(sidecar, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(sidecar_raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(sidecar, 0o444)
    require(_is_immutable_regular(sidecar), f"immutable sidecar mode failed: {sidecar}")

    directory_fd = os.open(body.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return body, sidecar, digest


def load_verified_immutable_json(
    path: Path, *, label: str = "immutable JSON", expected_sha256: str | None = None
) -> tuple[dict[str, Any], str]:
    requested = Path(path).expanduser()
    require(not requested.is_symlink(), f"{label} path may not be a symlink: {requested}")
    body = requested.resolve()
    sidecar = sidecar_path(body)
    require(_is_immutable_regular(body), f"{label} body is not mode 0444: {body}")
    require(_is_immutable_regular(sidecar), f"{label} sidecar is not mode 0444: {sidecar}")
    digest = sha256_file(body)
    require(
        sidecar.read_text(encoding="ascii") == f"{digest}  {body.name}\n",
        f"{label} SHA sidecar/body mismatch: {body}",
    )
    if expected_sha256 is not None:
        require(digest == expected_sha256, f"{label} frozen SHA drift: {body}")
    payload = json.loads(body.read_text(encoding="utf-8"))
    require(type(payload) is dict, f"{label} must contain a JSON object")
    return payload, digest
