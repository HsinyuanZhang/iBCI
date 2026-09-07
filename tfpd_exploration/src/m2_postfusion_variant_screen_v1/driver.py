"""Root-owned capability boundary for the post-fusion variant screen.

The public CLI intentionally cannot manufacture a capability.  This module is
also deliberately separate from the science code: its only purpose is to bind
the reviewed closure/root before source materialization and to publish an
honest immutable lifecycle.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from . import lifecycle, plan


class DriverError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DriverError(message)


def implementation_closure(repo_root: Path) -> dict[str, object]:
    root = Path(repo_root).absolute()
    files: dict[str, str] = {}
    for relative in plan.STATIC_CLOSURE_RELATIVES:
        path = root / relative
        _require(path.is_file() and not path.is_symlink(), f"postfusion closure path missing/symlink: {relative}")
        files[relative] = plan.sha256_file(path)
    return {"files": files, "closure_sha256": plan.sha256_bytes(plan.canonical_json_bytes(files))}


@dataclass
class _ScreenCapability:
    repo_root: Path
    root_relative: str
    closure_sha256: str
    nonce: str
    consumed: bool = False


_ISSUER_TOKEN = object()


def _mint_capability(repo_root: Path, root_relative: str, *, token: object) -> _ScreenCapability:
    _require(token is _ISSUER_TOKEN, "postfusion capability issuer token mismatch")
    _require(root_relative == plan.SCREEN_ROOT_RELATIVE, "postfusion live root identity drift")
    closure = implementation_closure(Path(repo_root))
    candidate = Path(repo_root).absolute() / root_relative
    _require(not candidate.exists(), "postfusion screen root must be fresh at capability issue")
    nonce = hashlib.sha256(plan.canonical_json_bytes({"root": root_relative, "closure": closure["closure_sha256"]})).hexdigest()
    return _ScreenCapability(Path(repo_root).absolute(), root_relative, str(closure["closure_sha256"]), nonce)


def _mint_test_capability(repo_root: Path) -> _ScreenCapability:
    """Private test seam; no public CLI references this function."""
    return _mint_capability(Path(repo_root), plan.SCREEN_ROOT_RELATIVE, token=_ISSUER_TOKEN)


def _consume(capability: _ScreenCapability, repo_root: Path) -> None:
    _require(isinstance(capability, _ScreenCapability), "postfusion opaque capability type drift")
    _require(not capability.consumed, "postfusion capability was already consumed")
    _require(capability.repo_root == Path(repo_root).absolute(), "postfusion capability repo-root drift")
    _require(capability.root_relative == plan.SCREEN_ROOT_RELATIVE, "postfusion capability root drift")
    closure = implementation_closure(Path(repo_root))
    _require(str(closure["closure_sha256"]) == capability.closure_sha256, "postfusion closure drift after capability issue")
    capability.consumed = True


def execute_screen(*, capability: _ScreenCapability, runner_factory: Callable[..., Any] | None = None,
                   _test_launch_validator: Callable[[], Mapping[str, object]] | None = None) -> tuple[str | None, str | None]:
    """One admitted production route; uncalled by the public CLI.

    Importing the Torch runtime occurs inside the post-attempt body publisher.
    Tests inject a typed fake runner and only use temporary roots.
    """
    repo_root = capability.repo_root
    _consume(capability, repo_root)
    closure = implementation_closure(repo_root)
    progress: dict[str, object] = {"source_materialized": False, "epochs_completed": 0}
    attempt = {
        "schema": f"{plan.SCHEMA}_attempt_v1", "status": "ATTEMPT_RESERVED", "cell": plan.CELL,
        "workorder_sha256": plan.WORKORDER_SHA256, "design_sha256": plan.DESIGN_SHA256,
        "closure_sha256": closure["closure_sha256"], "root_relative": plan.SCREEN_ROOT_RELATIVE,
        "source_or_checkpoint_opened": False, "cuda_initialized": False,
        "target_access": False, "matched_prefusion_control_trained": False,
    }

    def launch() -> Mapping[str, object]:
        # This imports only after attempt publication and validates GPU0 alone;
        # no path enumerates or probes GPU1.
        if _test_launch_validator is not None:
            device = dict(_test_launch_validator())
        else:
            import torch
            from .runner import validate_gpu0_only

            device = validate_gpu0_only(torch)
        return {"schema": f"{plan.SCHEMA}_launch_v1", "device": device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "source_or_checkpoint_opened": False, "target_access": False}

    def bodies(artifact) -> Mapping[str, str]:
        from .runner import PostFusionVariantScreenRunner

        factory = PostFusionVariantScreenRunner if runner_factory is None else runner_factory
        runner = factory(repo_root=repo_root, device="cuda:0")
        authority = runner.prepare()
        progress["source_materialized"] = True
        result = dict(runner.run_fixed_screen())
        progress["epochs_completed"] = plan.SOURCE_EPOCHS_SCREEN
        runtime = result.get("runtime")
        _require(isinstance(runtime, Mapping), "postfusion runner did not report total runtime evidence")
        progress["runtime"] = dict(runtime)
        bodies_by_arm = result.pop("_source_best_checkpoint_bodies")
        checkpoints = result.get("source_best_checkpoints")
        _require(isinstance(bodies_by_arm, Mapping) and set(bodies_by_arm) == set(plan.ARMS),
                 "postfusion source-best checkpoint bodies drift")
        _require(isinstance(checkpoints, Mapping) and set(checkpoints) == set(plan.ARMS),
                 "postfusion source-best checkpoint descriptors drift")
        checkpoint_shas: dict[str, str] = {}
        bound_checkpoints: dict[str, dict[str, object]] = {}
        for arm in plan.ARMS:
            body = bodies_by_arm[arm]
            _require(isinstance(body, bytes), f"{arm}: source-best checkpoint body type drift")
            filename = f"source_best_{arm.lower().replace('-', '_')}.pt"
            digest = artifact.publish_bytes(filename, body)
            descriptor = dict(checkpoints[arm])
            descriptor.update({"filename": filename, "sha256": digest})
            bound_checkpoints[arm] = descriptor
            checkpoint_shas[filename] = digest
        result["source_best_checkpoints"] = bound_checkpoints
        published = {
            "source_authority.json": artifact.publish_json("source_authority.json", dict(authority)),
            "screen.json": artifact.publish_json("screen.json", dict(result)),
        }
        return {**published, **checkpoint_shas}

    def terminal(shas: Mapping[str, str], attempt_sha: str, launch_sha: str) -> Mapping[str, object]:
        del attempt_sha, launch_sha
        current = implementation_closure(repo_root)
        _require(current["closure_sha256"] == closure["closure_sha256"], "postfusion closure drift before terminal")
        return {
            "source_authority_sha256": shas["source_authority.json"], "screen_sha256": shas["screen.json"],
            "source_best_checkpoint_sha256": {
                arm: shas[f"source_best_{arm.lower().replace('-', '_')}.pt"] for arm in plan.ARMS
            },
            "closure_sha256": closure["closure_sha256"], "target_access": False,
            "matched_prefusion_control_trained": False,
            "runtime": dict(progress["runtime"]),
        }

    return lifecycle.execute_stage(
        repo_root=repo_root, root_relative=plan.SCREEN_ROOT_RELATIVE, attempt=attempt,
        launch=launch, bodies=bodies, terminal=terminal, progress=lambda: dict(progress),
    )


__all__ = ("DriverError", "implementation_closure", "execute_screen")
