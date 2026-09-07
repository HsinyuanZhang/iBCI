"""AOF-S V6 local admission: shared lifecycle with dependency-layout-only repair."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v1 import driver as _lifecycle
from . import binding, docker_argv, plan, staged_context
from .profile import V6_PROFILE


class RecoveryError(RuntimeError):
    pass


_TOKEN = object()
_CONSUMED: set[tuple[int, int, str]] = set()
_PRE_BUILD_PREFIX = {"attempt.json", "attempt.json.sha256", "predecessor_authority.json", "predecessor_authority.json.sha256"}


@dataclass(frozen=True)
class _Capability:
    token: object; result_root: Path; parent_device: int; parent_inode: int; closure: dict[str, str]; closure_sha256: str
    v5_failure: dict[str, Any]; v2_artifact: dict[str, Any]


def _need(condition: bool, message: str) -> None:
    if not condition: raise RecoveryError(message)


def closure_map(repo_root: Path = plan.REPO_ROOT) -> dict[str, str]:
    plan.validate_static_authority(repo_root)
    result: dict[str, str] = {}
    for relative in plan.STATIC_CLOSURE_RELATIVES:
        leaf = repo_root / relative
        _need(leaf.is_file() and not leaf.is_symlink(), f"AOF-S V6 closure missing/symlink leaf: {relative}")
        result[relative] = plan.sha256_file(leaf)
    return result


def closure_sha256(mapping: Mapping[str, str]) -> str:
    return hashlib.sha256(json.dumps(dict(mapping), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _reviewed(repo_root: Path, mapping: Mapping[str, str], digest: str) -> dict[str, str]:
    measured = closure_map(repo_root)
    _need(dict(mapping) == measured and closure_sha256(measured) == digest, "AOF-S V6 externally reviewed closure map/digest mismatch")
    return measured


def _issue(*, repo_root: Path, result_root: Path, reviewed: Mapping[str, str], digest: str) -> _Capability:
    measured = _reviewed(repo_root, reviewed, digest)
    artifact = repo_root / plan.ARTIFACT_ROOT_RELATIVE
    _need(not os.path.lexists(result_root) and not os.path.lexists(artifact) and not os.path.lexists(artifact.parent),
          "AOF-S V6 fresh result/artifact parent/root required")
    parent = result_root.parent
    _need(parent.is_dir() and not parent.is_symlink(), "AOF-S V6 result parent invalid")
    info = parent.stat(follow_symlinks=False)
    return _Capability(_TOKEN, result_root, info.st_dev, info.st_ino, measured, digest,
                       binding.validate_v5_failure_graph(repo_root), binding.validate_v2_sealed_artifact(repo_root))


def _consume(cap: _Capability, repo_root: Path) -> None:
    _need(isinstance(cap, _Capability) and cap.token is _TOKEN, "AOF-S V6 opaque capability required")
    key = (cap.parent_device, cap.parent_inode, str(cap.result_root))
    _need(key not in _CONSUMED, "AOF-S V6 capability already consumed")
    info = cap.result_root.parent.stat(follow_symlinks=False)
    _need((info.st_dev, info.st_ino) == (cap.parent_device, cap.parent_inode) and not os.path.lexists(cap.result_root),
          "AOF-S V6 result root/parent drift")
    _need(_reviewed(repo_root, cap.closure, cap.closure_sha256) == cap.closure, "AOF-S V6 closure drift before attempt")
    _need(binding.validate_v5_failure_graph(repo_root) == cap.v5_failure, "AOF-S V6 V5 predecessor drift")
    _need(binding.validate_v2_sealed_artifact(repo_root) == cap.v2_artifact, "AOF-S V6 V2 payload drift")
    _CONSUMED.add(key)


def _final(cap: _Capability, root: Path) -> None:
    _need(_reviewed(root, cap.closure, cap.closure_sha256) == cap.closure, "AOF-S V6 closure drift before terminal")
    _need(binding.validate_v5_failure_graph(root) == cap.v5_failure, "AOF-S V6 V5 predecessor drift before terminal")
    _need(binding.validate_v2_sealed_artifact(root) == cap.v2_artifact, "AOF-S V6 V2 payload drift before terminal")


def _create_route_artifact_root(*, repo_root: Path, result_root: Path) -> Path:
    _need(result_root.is_dir() and not result_root.is_symlink() and set(os.listdir(result_root)) == _PRE_BUILD_PREFIX,
          "AOF-S V6 artifact parent may be created only after attempt/predecessor prefix")
    package, artifact = repo_root / plan.PACKAGE_RELATIVE, repo_root / plan.ARTIFACT_ROOT_RELATIVE
    parent = artifact.parent
    _need(package.is_dir() and not package.is_symlink() and parent.parent == package, "AOF-S V6 route-owned artifact package path invalid")
    _need(not os.path.lexists(parent) and not os.path.lexists(artifact), "AOF-S V6 artifact parent/root must remain fresh")
    parent.mkdir(mode=0o755); artifact.mkdir(mode=0o755)
    _need(parent.is_dir() and not parent.is_symlink() and artifact.is_dir() and not artifact.is_symlink() and set(os.listdir(parent)) == {artifact.name},
          "AOF-S V6 artifact parent/root topology drift after creation")
    return artifact


def _profile() -> _lifecycle.LifecycleProfile:
    return _lifecycle.LifecycleProfile(
        attempt_schema="m2_aof_scalar_static_package_v6_attempt", attempt_status="LOCAL_BUILD_V6_RESERVED_NO_NETWORK",
        terminal_schema="m2_aof_scalar_static_package_v6_terminal", terminal_status="LOCAL_BUILD_V6_VALIDATED_NOT_SUBMITTED",
        failure_schema="m2_aof_scalar_static_package_v6_failure", artifact_root_relative=plan.ARTIFACT_ROOT_RELATIVE,
        payload_name=plan.PAYLOAD_NAME, payload_path=lambda root: root / plan.V2_ARTIFACT_ROOT_RELATIVE / plan.PAYLOAD_NAME,
        attempt_authority=lambda cap: {"closure": cap.closure, "closure_sha256": cap.closure_sha256, "recovery_profile": V6_PROFILE.name,
                                       "v5_failure_predecessor": cap.v5_failure},
        predecessor_authority=lambda cap: {"v5_failure": cap.v5_failure, "v2_sealed_artifact": {"root_relative": cap.v2_artifact["root_relative"],
            "root": cap.v2_artifact["root"], "bodies": cap.v2_artifact["bodies"]}},
        input_authority=lambda build: {"reused_v2_payload": build["payload"], "v2_receipt_sha256": build["receipt_sha256"],
                                       "session_records": build["session_records"], "scientific_rebuild": False},
        final_revalidate=lambda root, cap: _final(cap, root))


def _reuse_build(*, repo_root: Path, capability: _Capability) -> Callable[[Path], Mapping[str, Any]]:
    def build(path: Path) -> Mapping[str, Any]:
        _need(path == repo_root / plan.V2_ARTIFACT_ROOT_RELATIVE / plan.PAYLOAD_NAME, "AOF-S V6 only reuses exact V2 payload path")
        artifact, receipt = _create_route_artifact_root(repo_root=repo_root, result_root=capability.result_root), capability.v2_artifact["receipt"]
        context = staged_context.stage_docker_context(repo_root=repo_root, artifact_root=artifact)
        body = {"schema": "m2_aof_scalar_static_package_v6_reused_v2_payload_authority", "v2_artifact_root": capability.v2_artifact["root_relative"],
                "payload": capability.v2_artifact["bodies"][plan.PAYLOAD_NAME], "receipt": capability.v2_artifact["bodies"][plan.RECEIPT_NAME], "scientific_rebuild": False}
        authority = artifact / "reused_v2_payload_authority.json"
        authority.write_text(json.dumps(body, sort_keys=True, indent=2) + "\n", encoding="utf-8"); os.chmod(authority, 0o444)
        return {"payload_sha256": capability.v2_artifact["bodies"][plan.PAYLOAD_NAME], "receipt_sha256": capability.v2_artifact["bodies"][plan.RECEIPT_NAME],
                "payload": _lifecycle._seal_artifact(authority), "session_records": receipt["session_records"], "docker_context": context,
                "v2_artifact": {"root_relative": capability.v2_artifact["root_relative"], "bodies": capability.v2_artifact["bodies"]}}
    return build


def execute_production_local(*, reviewed_closure: Mapping[str, str], reviewed_closure_sha256: str, repo_root: Path = plan.REPO_ROOT) -> dict[str, Any]:
    cap = _issue(repo_root=repo_root, result_root=repo_root / plan.RESULT_ROOT_RELATIVE, reviewed=reviewed_closure, digest=reviewed_closure_sha256)
    from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v1.validate_local import run_production_validation
    artifact = repo_root / plan.ARTIFACT_ROOT_RELATIVE
    return _lifecycle._execute_with_stages(cap, repo_root=repo_root, build_payload=_reuse_build(repo_root=repo_root, capability=cap),
        validate_payload=lambda payload: run_production_validation(payload, artifact),
        docker_preflight=lambda: {"container_minival": docker_argv.build_and_validate_container(repo_root=repo_root, artifact_root=artifact),
                                  "client": "route_owned_subprocess_argv", "network": "none", "pull": False}, profile=_profile(), consume=_consume)


def _execute_synthetic_for_test(*, repo_root: Path, result_root: Path, build: Callable[[Path], Mapping[str, Any]], validate: Callable[[Path], Mapping[str, Any]], docker: Callable[[], Mapping[str, Any]]) -> dict[str, Any]:
    mapping = closure_map(repo_root); cap = _issue(repo_root=repo_root, result_root=result_root, reviewed=mapping, digest=closure_sha256(mapping))
    return _lifecycle._execute_with_stages(cap, repo_root=repo_root, build_payload=build, validate_payload=validate, docker_preflight=docker, profile=_profile(), consume=_consume)
