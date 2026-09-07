"""No-data V2 recovery admission; delegates science/payload build to V1."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v1 import driver as v1_lifecycle
from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v1.export_aofs_static_payload import build_payload

from . import binding, plan
from .profile import AofsRecoveryProfile, V2_RECOVERY_PROFILE


class RecoveryError(RuntimeError):
    pass


_TOKEN = object()
_CONSUMED: set[tuple[int, int, str]] = set()


@dataclass(frozen=True)
class _Capability:
    token: object
    result_root: Path
    parent_device: int
    parent_inode: int
    closure: dict[str, str]
    closure_sha256: str
    v1_failure: dict[str, Any]


def closure_map(repo_root: Path = plan.REPO_ROOT) -> dict[str, str]:
    plan.validate_static_authority(repo_root)
    result: dict[str, str] = {}
    for relative in plan.STATIC_CLOSURE_RELATIVES:
        path = repo_root / relative
        if not path.is_file() or path.is_symlink():
            raise RecoveryError(f"V2 closure missing/symlink leaf: {relative}")
        result[relative] = plan.sha256_file(path)
    return result


def closure_sha256(mapping: Mapping[str, str]) -> str:
    return hashlib.sha256(json.dumps(dict(mapping), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _require_reviewed_closure(repo_root: Path, reviewed: Mapping[str, str], digest: str) -> dict[str, str]:
    measured = closure_map(repo_root)
    if dict(reviewed) != measured or closure_sha256(measured) != digest:
        raise RecoveryError("AOF-S V2 externally reviewed closure map/digest mismatch")
    return measured


def _issue_capability(*, repo_root: Path, result_root: Path, reviewed: Mapping[str, str], digest: str) -> _Capability:
    measured = _require_reviewed_closure(repo_root, reviewed, digest)
    artifact = repo_root / plan.ARTIFACT_ROOT_RELATIVE
    if os.path.lexists(result_root) or os.path.lexists(artifact):
        raise RecoveryError("AOF-S V2 fresh result/artifact root required")
    parent = result_root.parent
    if not parent.is_dir() or parent.is_symlink():
        raise RecoveryError("AOF-S V2 result parent invalid")
    info = parent.stat(follow_symlinks=False)
    return _Capability(_TOKEN, result_root, info.st_dev, info.st_ino, measured, digest,
                       binding.validate_v1_failure_graph(repo_root))


def _consume(capability: _Capability, repo_root: Path) -> None:
    if not isinstance(capability, _Capability) or capability.token is not _TOKEN:
        raise RecoveryError("AOF-S V2 opaque capability required")
    key = (capability.parent_device, capability.parent_inode, str(capability.result_root))
    if key in _CONSUMED:
        raise RecoveryError("AOF-S V2 capability already consumed")
    info = capability.result_root.parent.stat(follow_symlinks=False)
    if (info.st_dev, info.st_ino) != (capability.parent_device, capability.parent_inode) or os.path.lexists(capability.result_root):
        raise RecoveryError("AOF-S V2 result root/parent drift")
    _require_reviewed_closure(repo_root, capability.closure, capability.closure_sha256)
    if binding.validate_v1_failure_graph(repo_root) != capability.v1_failure:
        raise RecoveryError("AOF-S V2 V1 failure predecessor drift")
    _CONSUMED.add(key)


def _lifecycle_profile() -> v1_lifecycle.LifecycleProfile:
    return v1_lifecycle.LifecycleProfile(
        attempt_schema="m2_aof_scalar_static_package_v2_attempt",
        attempt_status="LOCAL_BUILD_V2_RESERVED_NO_NETWORK",
        terminal_schema="m2_aof_scalar_static_package_v2_terminal",
        terminal_status="LOCAL_BUILD_V2_VALIDATED_NOT_SUBMITTED",
        failure_schema="m2_aof_scalar_static_package_v2_failure",
        artifact_root_relative=plan.ARTIFACT_ROOT_RELATIVE,
        payload_name="t4_m2_seed42_dopt4_act30_aofs_identity.pkl",
        attempt_authority=lambda capability: {
            "closure": capability.closure, "closure_sha256": capability.closure_sha256,
            "v1_failure_predecessor": capability.v1_failure,
            "receipt_codec": V2_RECOVERY_PROFILE.receipt_codec.name,
            "recovery_profile": V2_RECOVERY_PROFILE.name,
        },
        predecessor_authority=lambda capability: {"v1_failure": capability.v1_failure},
        input_authority=lambda build: {
            "build_input_authority": build.get("session_records"),
            "v1_failure_predecessor": build.get("v1_failure_predecessor"),
            "receipt_codec": V2_RECOVERY_PROFILE.receipt_codec.name,
            "build_network": False,
        },
        final_revalidate=lambda repo_root, capability: _consume_final(capability, repo_root),
    )


def _consume_final(capability: _Capability, repo_root: Path) -> None:
    _require_reviewed_closure(repo_root, capability.closure, capability.closure_sha256)
    if binding.validate_v1_failure_graph(repo_root) != capability.v1_failure:
        raise RecoveryError("AOF-S V2 V1 failure predecessor drift before terminal")


def build_payload_v2(output: Path, *, profile: AofsRecoveryProfile = V2_RECOVERY_PROFILE) -> dict:
    """Only V2 entry to the V1 physical builder, with a frozen codec choice."""
    if profile is not V2_RECOVERY_PROFILE:
        raise RecoveryError("unrecognized mutable AOF-S recovery profile")
    witness = binding.validate_v1_failure_graph()
    receipt = dict(build_payload(output, receipt_codec=profile.receipt_codec))
    receipt["v1_failure_predecessor"] = witness
    receipt["recovery_profile"] = profile.name
    return receipt


def execute_production_local(
    *,
    reviewed_closure: Mapping[str, str],
    reviewed_closure_sha256: str,
    repo_root: Path = plan.REPO_ROOT,
) -> dict[str, Any]:
    """V2 uses V1's sole profiled lifecycle; callback seams are test-private."""
    capability = _issue_capability(
        repo_root=repo_root,
        result_root=repo_root / plan.RESULT_ROOT_RELATIVE,
        reviewed=reviewed_closure,
        digest=reviewed_closure_sha256,
    )
    from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v1 import docker_local
    from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v1.export_aofs_static_payload import RECEIPT_NAME
    from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v1.validate_local import run_production_validation
    artifact_root = repo_root / plan.ARTIFACT_ROOT_RELATIVE

    def route_build(path: Path) -> Mapping[str, Any]:
        receipt = build_payload_v2(path)
        payload = v1_lifecycle._seal_artifact(path)
        if payload["sha256"] != receipt.get("payload_sha256"):
            raise RecoveryError("AOF-S V2 payload body/receipt SHA drift")
        receipt_path = artifact_root / RECEIPT_NAME
        receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.chmod(receipt_path, 0o444)
        return {**receipt, "payload": payload, "payload_receipt": v1_lifecycle._seal_artifact(receipt_path)}

    def route_docker() -> Mapping[str, Any]:
        import docker
        client = docker.from_env()
        built = docker_local.build_network_none(
            client, repo_root / "tfpd_exploration/submissions", docker_local.BASE_IMAGE_ID,
            "spint-m2:aof-scalar-static-v2-local-v1", dockerfile="evalai_m2_aof_scalar_static_v2/Dockerfile",
        )
        return {**built, "container_minival": docker_local.run_container_minival(
            client, built_image=built, repo_root=repo_root, artifact_root=artifact_root)}

    return v1_lifecycle._execute_with_stages(
        capability, repo_root=repo_root, build_payload=route_build,
        validate_payload=lambda path: run_production_validation(path, artifact_root), docker_preflight=route_docker,
        profile=_lifecycle_profile(), consume=_consume,
    )


def _execute_synthetic_for_test(
    *,
    repo_root: Path,
    result_root: Path,
    build: Callable[[Path], Mapping[str, Any]],
    validate: Callable[[Path], Mapping[str, Any]],
    docker: Callable[[], Mapping[str, Any]],
) -> dict[str, Any]:
    """Private typed test seam exercising the same shared lifecycle exactly."""
    measured = closure_map(repo_root)
    capability = _issue_capability(
        repo_root=repo_root, result_root=result_root, reviewed=measured, digest=closure_sha256(measured),
    )
    return v1_lifecycle._execute_with_stages(
        capability, repo_root=repo_root, build_payload=build, validate_payload=validate,
        docker_preflight=docker, profile=_lifecycle_profile(), consume=_consume,
    )
