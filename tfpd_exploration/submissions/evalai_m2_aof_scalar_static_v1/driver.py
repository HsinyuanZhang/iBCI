"""Immutable, local-only lifecycle for the AOF-S package build.

The public CLI deliberately does not mint this capability.  An operator may
use this module only after a separate local-build review; neither this module
nor any called stage can submit, register, push, login, pull, or otherwise
mutate a network service.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from . import laws, plan


class LifecycleError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise LifecycleError(message)


def _sha256(path: Path) -> str:
    return plan.sha256_file(path)


def closure_map(repo_root: Path = plan.REPO_ROOT) -> dict[str, str]:
    """Compute the explicit closure.  Missing leaves fail closed, never skip."""
    mapping: dict[str, str] = {}
    for relative in plan.STATIC_CLOSURE_RELATIVES:
        path = repo_root / relative
        _need(path.is_file() and not path.is_symlink(), f"missing/symlinked AOF-S closure leaf: {relative}")
        mapping[relative] = _sha256(path)
    return mapping


def closure_sha256(mapping: Mapping[str, str]) -> str:
    return hashlib.sha256(json.dumps(dict(mapping), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write_pair(root: Path, name: str, payload: Mapping[str, Any]) -> str:
    """Publish one immutable body+basename-bound sidecar; never overwrite."""
    body_path = root / name
    side_path = root / f"{name}.sha256"
    _need(not os.path.lexists(body_path) and not os.path.lexists(side_path), f"immutable AOF-S leaf already exists: {name}")
    body = (json.dumps(dict(payload), sort_keys=True, indent=2) + "\n").encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    temp_fd, temp_name = tempfile.mkstemp(prefix=f".{name}.", dir=root)
    try:
        with os.fdopen(temp_fd, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_name, 0o444)
        os.replace(temp_name, body_path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    side_fd, side_name = tempfile.mkstemp(prefix=f".{name}.sha256.", dir=root)
    try:
        side = f"{digest}  {name}\n".encode("ascii")
        with os.fdopen(side_fd, "wb") as handle:
            handle.write(side)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(side_name, 0o444)
        os.replace(side_name, side_path)
    finally:
        if os.path.exists(side_name):
            os.unlink(side_name)
    return digest


def _validate_prefix(root: Path, bodies: tuple[str, ...]) -> None:
    actual = {entry.name for entry in root.iterdir()}
    expected = set(bodies) | {name + ".sha256" for name in bodies}
    _need(actual == expected, f"AOF-S result topology drift: {sorted(actual ^ expected)}")
    for name in bodies:
        for path in (root / name, root / f"{name}.sha256"):
            info = path.stat(follow_symlinks=False)
            _need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444 and info.st_nlink == 1,
                  f"AOF-S immutable leaf mode/link drift: {path.name}")
        body = (root / name).read_bytes()
        digest = hashlib.sha256(body).hexdigest()
        _need((root / f"{name}.sha256").read_bytes() == f"{digest}  {name}\n".encode("ascii"),
              f"AOF-S immutable sidecar drift: {name}")


def _seal_artifact(path: Path) -> dict[str, Any]:
    """Seal a generated local artifact and its basename-bound sidecar."""
    _need(path.is_file() and not path.is_symlink(), f"AOF-S artifact missing/symlinked: {path}")
    info = path.stat(follow_symlinks=False)
    _need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444 and info.st_nlink == 1,
          f"AOF-S artifact mode/link drift: {path.name}")
    digest = _sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    _need(not os.path.lexists(sidecar), f"AOF-S artifact sidecar already exists: {sidecar.name}")
    with sidecar.open("x", encoding="ascii") as handle:
        handle.write(f"{digest}  {path.name}\n")
    sidecar.chmod(0o444)
    return {"path": str(path), "sha256": digest, "sidecar": str(sidecar), "mode": "0444", "nlink": 1}


def _validate_container_minival(evidence: Mapping[str, Any]) -> None:
    """Container validation is a terminal prerequisite, never optional lore."""
    required = {
        "image_id", "image_tag", "base_image_id", "network_disabled", "pull",
        "container_removed", "command", "stdout_sha256", "prediction_path",
        "prediction_sha256", "target_path", "target_sha256",
    }
    _need(set(evidence) == required, "AOF-S container validation evidence key drift")
    _need(
        evidence["base_image_id"] == plan.LOCAL_DOCKER_BASE_ID
        and evidence["network_disabled"] is True
        and evidence["pull"] is False
        and evidence["container_removed"] is True,
        "AOF-S container validation network/base admission drift",
    )
    _need(
        evidence["command"] == ["--evaluation", "local", "--split", "m2", "--phase", "minival", "--batch-size", "7"],
        "AOF-S container validation command drift",
    )
    for key in ("stdout_sha256", "prediction_sha256", "target_sha256"):
        value = evidence[key]
        _need(isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value),
              f"AOF-S container validation digest drift: {key}")


_TOKEN = object()


@dataclass(frozen=True)
class _Capability:
    token: object
    result_root: Path
    result_parent_device: int
    result_parent_inode: int
    closure: dict[str, str]
    closure_sha256: str
    aofm: dict[str, Any]
    historical_581361: dict[str, Any]


_CONSUMED: set[tuple[int, int, str]] = set()


def _validated_reviewed_closure(
    *,
    repo_root: Path,
    reviewed_closure: Mapping[str, str],
    reviewed_closure_sha256: str,
) -> dict[str, str]:
    """Require a review-supplied closure map rather than self-approval.

    The plan itself is a closure leaf.  Storing an expected hash map inside
    that leaf would create a self-referential hash contract.  The reviewed map
    is therefore an explicit operator authority passed at invocation time and
    compared exactly with the freshly measured explicit closure *before* any
    result-root reservation or model/data/Docker access.
    """
    _need(isinstance(reviewed_closure_sha256, str) and len(reviewed_closure_sha256) == 64,
          "AOF-S reviewed closure digest is required")
    normalized = dict(reviewed_closure)
    _need(
        all(isinstance(name, str) and isinstance(digest, str) and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for name, digest in normalized.items()),
        "AOF-S reviewed closure map encoding drift",
    )
    measured = closure_map(repo_root)
    _need(normalized == measured, "AOF-S externally reviewed closure map mismatch")
    _need(closure_sha256(normalized) == reviewed_closure_sha256,
          "AOF-S externally reviewed closure digest mismatch")
    return measured


def _issue_capability(
    *,
    repo_root: Path,
    result_root: Path,
    reviewed_closure: Mapping[str, str] | None = None,
    reviewed_closure_sha256: str | None = None,
) -> _Capability:
    """Admission before any model/data/Docker access; capability is one-shot."""
    plan.validate_static_authority(repo_root)
    target = result_root
    _need(not os.path.lexists(target), "AOF-S local build root is not fresh")
    artifact_root = repo_root / plan.ARTIFACT_ROOT_RELATIVE
    _need(not os.path.lexists(artifact_root), "AOF-S package artifact root is not fresh")
    _need(target.parent.is_dir() and not target.parent.is_symlink(), "AOF-S result parent invalid")
    parent_info = target.parent.stat(follow_symlinks=False)
    mapping = closure_map(repo_root)
    if reviewed_closure is not None or reviewed_closure_sha256 is not None:
        _need(reviewed_closure is not None and reviewed_closure_sha256 is not None,
              "AOF-S reviewed closure map and digest must be supplied together")
        mapping = _validated_reviewed_closure(
            repo_root=repo_root,
            reviewed_closure=reviewed_closure,
            reviewed_closure_sha256=reviewed_closure_sha256,
        )
    return _Capability(
        token=_TOKEN,
        result_root=target,
        result_parent_device=parent_info.st_dev,
        result_parent_inode=parent_info.st_ino,
        closure=mapping,
        closure_sha256=closure_sha256(mapping),
        aofm=laws.validate_aofm_authority(repo_root),
        historical_581361=laws.validate_581361_lineage(repo_root),
    )


def issue_local_build_capability(
    *,
    repo_root: Path = plan.REPO_ROOT,
    reviewed_closure: Mapping[str, str],
    reviewed_closure_sha256: str,
) -> _Capability:
    """Issue only for the single canonical immutable AOF-S result root."""
    return _issue_capability(
        repo_root=repo_root,
        result_root=repo_root / plan.RESULT_ROOT_RELATIVE,
        reviewed_closure=reviewed_closure,
        reviewed_closure_sha256=reviewed_closure_sha256,
    )


def _issue_synthetic_capability_for_test(*, repo_root: Path, result_root: Path) -> _Capability:
    """Typed test seam; never reachable from the public inert CLI."""
    return _issue_capability(repo_root=repo_root, result_root=result_root)


def _consume(capability: _Capability, repo_root: Path) -> None:
    _need(isinstance(capability, _Capability) and capability.token is _TOKEN, "AOF-S opaque capability required")
    key = (capability.result_parent_device, capability.result_parent_inode, str(capability.result_root))
    _need(key not in _CONSUMED, "AOF-S capability already consumed")
    parent = capability.result_root.parent
    info = parent.stat(follow_symlinks=False)
    _need((info.st_dev, info.st_ino) == (capability.result_parent_device, capability.result_parent_inode), "AOF-S result parent replaced")
    _need(not os.path.lexists(capability.result_root), "AOF-S result root no longer fresh")
    plan.validate_static_authority(repo_root)
    mapping = closure_map(repo_root)
    _need(mapping == capability.closure and closure_sha256(mapping) == capability.closure_sha256, "AOF-S closure drift before attempt")
    _CONSUMED.add(key)


@dataclass(frozen=True)
class LifecycleProfile:
    """Immutable stage receipt profile for additive local-build successors."""

    attempt_schema: str
    attempt_status: str
    terminal_schema: str
    terminal_status: str
    failure_schema: str
    artifact_root_relative: str
    payload_name: str
    attempt_authority: Callable[[Any], Mapping[str, Any]]
    predecessor_authority: Callable[[Any], Mapping[str, Any]]
    input_authority: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    final_revalidate: Callable[[Path, Any], None]
    # Additive successors may validate a sealed predecessor payload directly
    # rather than regenerate/copy it into their fresh artifact root.
    payload_path: Callable[[Path], Path] | None = None


def _v1_profile() -> LifecycleProfile:
    return LifecycleProfile(
        attempt_schema="m2_aof_scalar_static_package_v1_attempt",
        attempt_status="LOCAL_BUILD_RESERVED_NO_NETWORK",
        terminal_schema="m2_aof_scalar_static_package_v1_terminal",
        terminal_status="LOCAL_BUILD_VALIDATED_NOT_SUBMITTED",
        failure_schema="m2_aof_scalar_static_package_v1_failure",
        artifact_root_relative=plan.ARTIFACT_ROOT_RELATIVE,
        payload_name="t4_m2_seed42_dopt4_act30_aofs_identity.pkl",
        attempt_authority=lambda capability: {
            "design_sha256": plan.DESIGN_SHA256, "workorder_sha256": plan.WORKORDER_SHA256,
            "result_sha256": plan.RESULT_SHA256, "closure": capability.closure,
            "closure_sha256": capability.closure_sha256,
        },
        predecessor_authority=lambda capability: {
            "aofm": capability.aofm, "official_581361": capability.historical_581361,
            "exact_scored_and_local_lineages_are_distinct": True,
        },
        input_authority=lambda build: {
            "build_input_authority": build.get("session_records"),
            "local_act30_payload_sha256": plan.LOCAL_ACT30_PAYLOAD_SHA256, "build_network": False,
        },
        final_revalidate=lambda repo_root, capability: (
            plan.validate_static_authority(repo_root),
            _need(closure_map(repo_root) == capability.closure, "AOF-S closure drift before terminal"),
            _need(laws.validate_aofm_authority(repo_root) == capability.aofm, "AOF-M predecessor drift before terminal"),
            _need(laws.validate_581361_lineage(repo_root) == capability.historical_581361, "581361 lineage drift before terminal"),
        ),
    )


def _execute_with_stages(
    capability: Any,
    *,
    repo_root: Path = plan.REPO_ROOT,
    build_payload: Callable[[Path], Mapping[str, Any]],
    validate_payload: Callable[[Path], Mapping[str, Any]],
    docker_preflight: Callable[[], Mapping[str, Any]],
    profile: LifecycleProfile | None = None,
    consume: Callable[[Any, Path], None] | None = None,
) -> dict[str, Any]:
    """Run local stages with immutable prefix-preserving terminal/failure XOR.

    Injectable functions make the lifecycle testable without model/data/Docker.
    Production wiring supplies route-owned functions only; nothing here contains
    a network client or submission helper.
    """
    active_profile = _v1_profile() if profile is None else profile
    (_consume if consume is None else consume)(capability, repo_root)
    root = capability.result_root
    root.mkdir(mode=0o755)
    published: list[str] = []
    try:
        attempt = _write_pair(root, "attempt.json", {
            "schema": active_profile.attempt_schema,
            "status": active_profile.attempt_status,
            **dict(active_profile.attempt_authority(capability)),
            "network_submission": False,
        })
        published.append("attempt.json")
        predecessor = _write_pair(root, "predecessor_authority.json", active_profile.predecessor_authority(capability))
        published.append("predecessor_authority.json")
        payload_path = (
            active_profile.payload_path(repo_root)
            if active_profile.payload_path is not None
            else repo_root / active_profile.artifact_root_relative / active_profile.payload_name
        )
        build = dict(build_payload(payload_path))
        input_authority = _write_pair(root, "input_authority.json", active_profile.input_authority(build))
        published.append("input_authority.json")
        image = dict(docker_preflight())
        _need(isinstance(image.get("container_minival"), Mapping),
              "AOF-S build lacks required offline container minival validation")
        _validate_container_minival(image["container_minival"])
        build_sha = _write_pair(root, "build.json", {
            "payload_sha256": build.get("payload_sha256"),
            "payload_receipt": build,
            "docker": image,
            "network_disabled": True,
        })
        published.append("build.json")
        validation = dict(validate_payload(payload_path))
        validation_sha = _write_pair(root, "validation.json", {
            "validation": validation,
            "payload_sha256": build.get("payload_sha256"),
            "selection_or_refit": False,
            "network": False,
        })
        published.append("validation.json")
        # All potentially failing work has happened before terminal admission.
        active_profile.final_revalidate(repo_root, capability)
        _validate_prefix(root, tuple(published))
        terminal = _write_pair(root, "terminal.json", {
            "schema": active_profile.terminal_schema,
            "status": active_profile.terminal_status,
            "terminal_xor_failure": True,
            "attempt_sha256": attempt,
            "predecessor_authority_sha256": predecessor,
            "input_authority_sha256": input_authority,
            "build_sha256": build_sha,
            "validation_sha256": validation_sha,
            "closure_sha256": capability.closure_sha256,
            "network_submission": False,
        })
        published.append("terminal.json")
        _validate_prefix(root, tuple(published))
        return {"status": active_profile.terminal_status, "terminal_sha256": terminal, "root": str(root)}
    except Exception as error:
        if root.exists() and "terminal.json" not in published:
            failure = _write_pair(root, "failure.json", {
                "schema": active_profile.failure_schema,
                "status": "LOCAL_BUILD_FAILED",
                "published_prefix": list(published),
                "exception_class": type(error).__name__,
                "exception_message": str(error)[:400],
                "network_submission": False,
            })
            _validate_prefix(root, tuple([*published, "failure.json"]))
            raise LifecycleError(f"AOF-S local build failed: {failure}") from error
        raise


def execute_production_local(
    *,
    reviewed_closure: Mapping[str, str],
    reviewed_closure_sha256: str,
    repo_root: Path = plan.REPO_ROOT,
) -> dict[str, Any]:
    """The only route-owned local build path; no caller-provided callbacks.

    This remains strictly local: Docker is queried through its local daemon
    with ``pull=False`` and ``network=none``; no API, registry, or submission
    capability is imported or exposed here.
    """
    from . import docker_local
    from .export_aofs_static_payload import PAYLOAD_NAME, RECEIPT_NAME, build_payload
    from .validate_local import run_production_validation

    capability = issue_local_build_capability(
        repo_root=repo_root,
        reviewed_closure=reviewed_closure,
        reviewed_closure_sha256=reviewed_closure_sha256,
    )
    artifact_root = repo_root / plan.ARTIFACT_ROOT_RELATIVE

    def route_build(path: Path) -> Mapping[str, Any]:
        receipt = dict(build_payload(path))
        payload_descriptor = _seal_artifact(path)
        _need(payload_descriptor["sha256"] == receipt.get("payload_sha256"), "AOF-S payload receipt/body link drift")
        receipt_path = artifact_root / RECEIPT_NAME
        _need(not os.path.lexists(receipt_path), "AOF-S payload receipt already exists")
        receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.chmod(receipt_path, 0o444)
        receipt_descriptor = _seal_artifact(receipt_path)
        return {**receipt, "payload": payload_descriptor, "payload_receipt": receipt_descriptor}

    def route_validation(path: Path) -> Mapping[str, Any]:
        return run_production_validation(path, artifact_root)

    def route_docker() -> Mapping[str, Any]:
        import docker

        client = docker.from_env()
        built = docker_local.build_network_none(
            client,
            repo_root / plan.PACKAGE_RELATIVE,
            docker_local.BASE_IMAGE_ID,
            "spint-m2:aof-scalar-static-local-v1",
        )
        return {
            **built,
            "container_minival": docker_local.run_container_minival(
                client,
                built_image=built,
                repo_root=repo_root,
                artifact_root=artifact_root,
            ),
        }

    return _execute_with_stages(
        capability,
        repo_root=repo_root,
        build_payload=route_build,
        validate_payload=route_validation,
        docker_preflight=route_docker,
    )
