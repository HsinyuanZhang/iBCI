"""Write JSON/markdown bodies plus sha256 sidecars under the Stage0 root."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from . import plan


class ReceiptError(RuntimeError):
    """Fail closed for result-root writes."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptError(message)


def ensure_stage0_root(repo_root: Path) -> Path:
    root = Path(repo_root) / plan.STAGE0_ROOT_RELATIVE
    root.mkdir(parents=True, exist_ok=True)
    return root


def ensure_revision_root(repo_root: Path) -> Path:
    root = Path(repo_root) / plan.P_REVISION_ROOT_RELATIVE
    root.mkdir(parents=True, exist_ok=True)
    stage0 = Path(repo_root) / plan.STAGE0_ROOT_RELATIVE
    _require(stage0.is_dir(), "Stage0 root missing; do not mint a revision without it")
    _require(root.resolve() != stage0.resolve(), "revision root must not overwrite Stage0")
    return root


def write_json(path: Path, payload: Mapping[str, object]) -> str:
    path = Path(path)
    body = json.dumps(dict(payload), sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    encoded = body.encode("utf-8")
    path.write_bytes(encoded)
    digest = plan.sha256_bytes(encoded)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "  " + path.name + "\n", encoding="utf-8")
    return digest


def write_text(path: Path, text: str) -> str:
    path = Path(path)
    encoded = text.encode("utf-8")
    path.write_bytes(encoded)
    digest = plan.sha256_bytes(encoded)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "  " + path.name + "\n", encoding="utf-8")
    return digest


def file_record(repo_root: Path, relative: str, *, role: str, verify: str | None = None) -> dict[str, object]:
    path = Path(repo_root) / relative
    plan.reject_forbidden_path(path)
    exists = path.is_file()
    digest = plan.sha256_file(path) if exists else None
    if exists and verify is not None and digest != verify:
        raise ReceiptError(f"sha mismatch {relative}: {digest} != {verify}")
    return {
        "relative": relative,
        "exists": exists,
        "sha256": digest,
        "bytes": int(path.stat().st_size) if exists else None,
        "role": role,
        "sha_verified": bool(exists and verify is not None and digest == verify),
    }
