"""Held-FD verification for the exact V4 parent-layout failure and V2 payload."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v4 import binding as _v4_binding

from . import plan


class BindingError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise BindingError(message)


def validate_v4_failure_graph(repo_root: Path = plan.REPO_ROOT) -> dict[str, Any]:
    """Bind V4's exact 3-body/6-leaf pre-build failure graph via held FDs."""
    bodies, identity = _v4_binding._graph(
        repo_root / plan.V4_FAILURE_ROOT_RELATIVE, plan.V4_FAILURE_BODIES, "V4 parent-layout failure"
    )
    attempt = json.loads(bodies["attempt.json"])
    predecessor = json.loads(bodies["predecessor_authority.json"])
    failure = json.loads(bodies["failure.json"])
    _need(attempt.get("schema") == "m2_aof_scalar_static_package_v4_attempt"
          and attempt.get("status") == "LOCAL_BUILD_V4_RESERVED_NO_NETWORK"
          and attempt.get("closure_sha256") == plan.V4_FAILURE_CLOSURE_SHA256,
          "V4 attempt schema/status/closure drift")
    _need(set(predecessor) == {"v3_failure", "v2_sealed_artifact"}
          and predecessor["v2_sealed_artifact"].get("bodies") == plan.V2_ARTIFACT_BODIES,
          "V4 predecessor payload lineage drift")
    _need(failure.get("schema") == "m2_aof_scalar_static_package_v4_failure"
          and failure.get("status") == "LOCAL_BUILD_FAILED"
          and failure.get("exception_class") == "FileNotFoundError"
          and failure.get("published_prefix") == ["attempt.json", "predecessor_authority.json"]
          and failure.get("network_submission") is False
          and str(repo_root / plan.ARTIFACT_ROOT_RELATIVE.replace("v5", "v4")) in failure.get("exception_message", ""),
          "V4 exact absent-artifact-parent failure semantics drift")
    return {"root_relative": plan.V4_FAILURE_ROOT_RELATIVE, "root": identity,
            "bodies": dict(plan.V4_FAILURE_BODIES), "closure_sha256": plan.V4_FAILURE_CLOSURE_SHA256,
            "failure_stage": "post_predecessor_pre_input_absent_artifact_parent"}


def validate_v2_sealed_artifact(repo_root: Path = plan.REPO_ROOT) -> dict[str, Any]:
    return _v4_binding.validate_v2_sealed_artifact(repo_root)
