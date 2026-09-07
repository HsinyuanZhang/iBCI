"""Minimal receipt lifecycle for V2 admission tests; no replay loop is copied."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from . import binding, plan
from tfpd_exploration.src.m2_cpre_a0_chunk_memory_v1 import receipts


class LifecycleError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LifecycleError(message)


def build_attempt(*, surface: str, closure: Mapping[str, Any], v1_failure: Mapping[str, Any],
                  cpre_v2: Mapping[str, Any], environment: Mapping[str, str]) -> dict[str, Any]:
    _require(surface in ("external_post30_local", "within_post30"), "unknown V2 shard surface")
    binding.assert_bound_workorder(closure); binding.assert_science_profile_parity(binding.v1_science_profile())
    return {"schema": "m2_a0_attempt_v2", "status": "ATTEMPT_RESERVED_BEFORE_RUNTIME",
            "route": plan.ROUTE_SCHEMA, "surface": surface, "root_relative": plan.V2_ROOTS[surface],
            "source_closure": dict(closure), "deterministic_environment": dict(environment),
            "v1_external_failure_predecessor": dict(v1_failure),
            "historical_cpre_v2_witness": dict(cpre_v2),
            "science_profile": binding.v1_science_profile().__dict__,
            "parameter_updates": 0, "target_gradients": 0}


def execute_synthetic(*, root: Path, repo_root: Path, capability: object, surface: str,
                      v1_failure: Mapping[str, Any], cpre_v2: Mapping[str, Any],
                      environ: Mapping[str, str], fail_after_launch: bool = False) -> dict[str, str]:
    """No-data V2 lifecycle seam.  The future V2 runner must delegate V1 replay, not this test seam."""
    closure = binding.source_closure(repo_root); binding.assert_bound_workorder(closure)
    held = binding.consume(capability, expected_root=plan.V2_ROOTS[surface], closure=closure, environ=environ)
    _require(held.get("v1_external_failure_predecessor") == dict(v1_failure)
             and held.get("historical_cpre_v2_witness") == dict(cpre_v2), "V2 predecessor binding drift")
    _require(root.name == Path(plan.V2_ROOTS[surface]).name and not root.exists(), "V2 fresh root drift")
    root.mkdir(mode=0o700)
    attempt = receipts.publish_pair(root, "attempt.json", build_attempt(
        surface=surface, closure=closure, v1_failure=v1_failure, cpre_v2=cpre_v2,
        environment=held["deterministic_environment"]))
    stage = "launch"
    try:
        launch = receipts.publish_pair(root, "launch.json", {
            "schema": "m2_a0_launch_v2", "attempt_sha256": attempt.sha256,
            "deterministic_environment": dict(held["deterministic_environment"]),
            "physical_gpu": {"physical_index": 0, "uuid": plan.GPU0_UUID, "gpu1_forbidden": True},
            "parameter_updates": 0, "target_gradients": 0})
        if fail_after_launch:
            raise RuntimeError("synthetic V2 failure")
        _require(binding.source_closure(repo_root) == closure
                 and binding.validate_deterministic_environment(environ) == held["deterministic_environment"],
                 "V2 terminal current closure/deterministic environment drift")
        terminal = receipts.publish_pair(root, "terminal.json", {
            "schema": "m2_a0_terminal_v2", "status": "TERMINAL", "attempt_sha256": attempt.sha256,
            "launch_sha256": launch.sha256, "source_closure": dict(closure),
            "deterministic_environment": dict(held["deterministic_environment"]),
            "v1_external_failure_predecessor": dict(v1_failure), "historical_cpre_v2_witness": dict(cpre_v2),
            "parameter_updates": 0, "target_gradients": 0})
        receipts.verify_topology(root, bodies=("attempt.json", "launch.json", "terminal.json"))
        return {"attempt": attempt.sha256, "launch": launch.sha256, "terminal": terminal.sha256}
    except Exception as exc:
        # Revalidate the current V2 authority before making its immutable
        # failure claim; the historical V1 failure remains an input witness.
        _require(binding.source_closure(repo_root) == closure
                 and binding.validate_deterministic_environment(environ) == held["deterministic_environment"],
                 "V2 failure current closure/deterministic environment drift")
        receipts.publish_failure_preserving_prefix(
            root=root, prefix_bodies=("attempt.json", "launch.json"), attempt_sha256=attempt.sha256,
            schema="m2_a0_failure_v2", stage=stage, error=f"{type(exc).__name__}: {exc}",
            progress={"parameter_updates": 0, "target_gradients": 0, "source_closure": dict(closure),
                      "deterministic_environment": dict(held["deterministic_environment"]),
                      "v1_external_failure_predecessor": dict(v1_failure),
                      "historical_cpre_v2_witness": dict(cpre_v2)})
        raise LifecycleError("synthetic V2 lifecycle failed after immutable attempt") from exc
