"""Exact, minimal, route-owned Docker context staging for AOF-S V6."""
from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path
from typing import Any

from . import plan


class ContextError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ContextError(message)


def _sources(repo_root: Path) -> dict[str, tuple[Path, str]]:
    pairs = {
        "Dockerfile": (repo_root / plan.PACKAGE_RELATIVE / "Dockerfile", None),
        "aofs_static_decoder.py": (repo_root / "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/aofs_static_decoder.py", plan.V1_RUNTIME_BODIES["tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/aofs_static_decoder.py"]),
        "laws.py": (repo_root / "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/laws.py", plan.V1_RUNTIME_BODIES["tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/laws.py"]),
        "decode.py": (repo_root / "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/decode.py", plan.V1_RUNTIME_BODIES["tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/decode.py"]),
        "third_party/__init__.py": (repo_root / "SPINT-main/third_party/__init__.py", plan.THIRD_PARTY_BODIES["SPINT-main/third_party/__init__.py"]),
        "third_party/falcon_challenge/__init__.py": (repo_root / "SPINT-main/third_party/falcon_challenge/__init__.py", plan.THIRD_PARTY_BODIES["SPINT-main/third_party/falcon_challenge/__init__.py"]),
        "third_party/falcon_challenge/filtering.py": (repo_root / "SPINT-main/third_party/falcon_challenge/filtering.py", plan.THIRD_PARTY_BODIES["SPINT-main/third_party/falcon_challenge/filtering.py"]),
        "decoder.pkl": (repo_root / plan.V2_ARTIFACT_ROOT_RELATIVE / plan.PAYLOAD_NAME, plan.V2_ARTIFACT_BODIES[plan.PAYLOAD_NAME]),
    }
    result: dict[str, tuple[Path, str]] = {}
    for destination, (source, wanted) in pairs.items():
        _need(source.is_file() and not source.is_symlink(), f"AOF-S V6 staged source missing/symlink: {source}")
        digest = plan.sha256_file(source)
        _need(wanted is None or digest == wanted, f"AOF-S V6 staged source digest drift: {source}")
        result[destination] = (source, digest)
    return result


def _actual_files(context: Path) -> set[str]:
    result: set[str] = set()
    for path in context.rglob("*"):
        info = path.stat(follow_symlinks=False)
        _need(not path.is_symlink(), f"AOF-S V6 staged context symlink: {path.relative_to(context)}")
        if stat.S_ISREG(info.st_mode):
            result.add(path.relative_to(context).as_posix())
        else:
            _need(stat.S_ISDIR(info.st_mode), f"AOF-S V6 staged context nonregular leaf: {path.relative_to(context)}")
    return result


def validate_staged_context(*, repo_root: Path, artifact_root: Path) -> dict[str, Any]:
    context = artifact_root / plan.STAGED_CONTEXT_NAME
    _need(context.is_dir() and not context.is_symlink(), "AOF-S V6 staged context absent/symlink")
    sources = _sources(repo_root)
    _need(_actual_files(context) == set(sources), "AOF-S V6 staged context exact topology drift")
    bodies: dict[str, str] = {}
    for relative, (_, digest) in sources.items():
        path = context / relative
        info = path.stat(follow_symlinks=False)
        _need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444 and info.st_nlink == 1,
              f"AOF-S V6 staged context mode/link drift: {relative}")
        _need(plan.sha256_file(path) == digest, f"AOF-S V6 staged context digest drift: {relative}")
        bodies[relative] = digest
    return {"schema": "m2_aof_scalar_static_v6_staged_docker_context", "root": str(context), "bodies": bodies}


def stage_docker_context(*, repo_root: Path, artifact_root: Path) -> dict[str, Any]:
    """Copy only reviewed runtime leaves into a fresh V6-owned context."""
    _need(artifact_root.is_dir() and not artifact_root.is_symlink(), "AOF-S V6 artifact root invalid before context stage")
    context = artifact_root / plan.STAGED_CONTEXT_NAME
    _need(not os.path.lexists(context), "AOF-S V6 staged context must be fresh")
    sources = _sources(repo_root)
    context.mkdir(mode=0o755)
    for relative, (source, digest) in sources.items():
        target = context / relative
        target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        shutil.copyfile(source, target, follow_symlinks=False)
        os.chmod(target, 0o444)
        _need(plan.sha256_file(target) == digest, f"AOF-S V6 staged copy digest drift: {relative}")
    return validate_staged_context(repo_root=repo_root, artifact_root=artifact_root)
