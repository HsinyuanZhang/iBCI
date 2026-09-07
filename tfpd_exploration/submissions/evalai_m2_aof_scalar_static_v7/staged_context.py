"""V7 exact eight-file staged Docker context; only its Dockerfile differs from V6."""
from __future__ import annotations
import os, shutil, stat
from pathlib import Path
from typing import Any
from . import plan
class ContextError(RuntimeError): pass
def _need(v: bool, m: str) -> None:
    if not v: raise ContextError(m)
def _sources(root: Path) -> dict[str, tuple[Path, str]]:
    pairs = {"Dockerfile": (root / plan.PACKAGE_RELATIVE / "Dockerfile", None), "aofs_static_decoder.py": (root / "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/aofs_static_decoder.py", plan.V1_RUNTIME_BODIES["tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/aofs_static_decoder.py"]), "laws.py": (root / "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/laws.py", plan.V1_RUNTIME_BODIES["tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/laws.py"]), "decode.py": (root / "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/decode.py", plan.V1_RUNTIME_BODIES["tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/decode.py"]), "third_party/__init__.py": (root / "SPINT-main/third_party/__init__.py", plan.THIRD_PARTY_BODIES["SPINT-main/third_party/__init__.py"]), "third_party/falcon_challenge/__init__.py": (root / "SPINT-main/third_party/falcon_challenge/__init__.py", plan.THIRD_PARTY_BODIES["SPINT-main/third_party/falcon_challenge/__init__.py"]), "third_party/falcon_challenge/filtering.py": (root / "SPINT-main/third_party/falcon_challenge/filtering.py", plan.THIRD_PARTY_BODIES["SPINT-main/third_party/falcon_challenge/filtering.py"]), "decoder.pkl": (root / plan.V2_ARTIFACT_ROOT_RELATIVE / plan.PAYLOAD_NAME, plan.V2_ARTIFACT_BODIES[plan.PAYLOAD_NAME])}
    out = {}
    for dest, (src, expected) in pairs.items():
        _need(src.is_file() and not src.is_symlink(), f"V7 staged source missing/symlink: {src}"); digest = plan.sha256_file(src); _need(expected is None or digest == expected, f"V7 staged source digest drift: {src}"); out[dest] = (src, digest)
    return out
def _files(context: Path) -> set[str]:
    out = set()
    for p in context.rglob("*"):
        info = p.stat(follow_symlinks=False); _need(not p.is_symlink(), f"V7 staged symlink: {p}")
        if stat.S_ISREG(info.st_mode): out.add(p.relative_to(context).as_posix())
        else: _need(stat.S_ISDIR(info.st_mode), f"V7 staged nonregular leaf: {p}")
    return out
def validate_staged_context(*, repo_root: Path, artifact_root: Path) -> dict[str, Any]:
    context = artifact_root / plan.STAGED_CONTEXT_NAME; _need(context.is_dir() and not context.is_symlink(), "V7 staged context absent/symlink"); sources = _sources(repo_root); _need(_files(context) == set(sources), "V7 staged exact topology drift")
    bodies = {}
    for name, (_, digest) in sources.items():
        p = context / name; info = p.stat(follow_symlinks=False); _need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444 and info.st_nlink == 1, f"V7 staged mode/link drift: {name}"); _need(plan.sha256_file(p) == digest, f"V7 staged digest drift: {name}"); bodies[name] = digest
    return {"schema": "m2_aof_scalar_static_v7_staged_docker_context", "root": str(context), "bodies": bodies}
def stage_docker_context(*, repo_root: Path, artifact_root: Path) -> dict[str, Any]:
    _need(artifact_root.is_dir() and not artifact_root.is_symlink(), "V7 artifact invalid before context stage"); context = artifact_root / plan.STAGED_CONTEXT_NAME; _need(not os.path.lexists(context), "V7 staged context must be fresh"); sources = _sources(repo_root); context.mkdir(mode=0o755)
    for dest, (src, digest) in sources.items():
        target = context / dest; target.parent.mkdir(mode=0o755, parents=True, exist_ok=True); shutil.copyfile(src, target, follow_symlinks=False); os.chmod(target, 0o444); _need(plan.sha256_file(target) == digest, f"V7 staged copy digest drift: {dest}")
    return validate_staged_context(repo_root=repo_root, artifact_root=artifact_root)
