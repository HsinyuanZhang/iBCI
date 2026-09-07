"""Held-FD verification of exact V5 failure and the V2 sealed payload."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v4 import binding as _fd_binding
from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v5 import binding as _v5_binding

from . import plan


class BindingError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise BindingError(message)


def validate_v5_failure_graph(repo_root: Path = plan.REPO_ROOT) -> dict[str, Any]:
    bodies, identity = _fd_binding._graph(repo_root / plan.V5_FAILURE_ROOT_RELATIVE, plan.V5_FAILURE_BODIES, "V5 dependency-layout failure")
    attempt, predecessor = json.loads(bodies["attempt.json"]), json.loads(bodies["predecessor_authority.json"])
    input_authority, failure = json.loads(bodies["input_authority.json"]), json.loads(bodies["failure.json"])
    _need(attempt.get("schema") == "m2_aof_scalar_static_package_v5_attempt"
          and attempt.get("status") == "LOCAL_BUILD_V5_RESERVED_NO_NETWORK"
          and attempt.get("closure_sha256") == plan.V5_FAILURE_CLOSURE_SHA256,
          "V5 attempt schema/status/closure drift")
    _need(set(predecessor) == {"v4_failure", "v2_sealed_artifact"}
          and predecessor["v2_sealed_artifact"].get("bodies") == plan.V2_ARTIFACT_BODIES,
          "V5 predecessor lineage drift")
    _need(input_authority.get("scientific_rebuild") is False and input_authority.get("v2_receipt_sha256") == plan.V2_ARTIFACT_BODIES[plan.RECEIPT_NAME],
          "V5 input payload reuse semantics drift")
    _need(failure.get("schema") == "m2_aof_scalar_static_package_v5_failure"
          and failure.get("exception_class") == "DockerArgvError" and failure.get("network_submission") is False
          and failure.get("published_prefix") == ["attempt.json", "predecessor_authority.json", "input_authority.json"]
          and "/workspace/decode.py" in failure.get("exception_message", "")
          and "from third_party.falcon_chall" in failure.get("exception_message", ""),
          "V5 missing-third_party failure semantics drift")
    return {"root_relative": plan.V5_FAILURE_ROOT_RELATIVE, "root": identity, "bodies": dict(plan.V5_FAILURE_BODIES),
            "closure_sha256": plan.V5_FAILURE_CLOSURE_SHA256, "failure_stage": "post_input_container_missing_third_party"}


def validate_v2_sealed_artifact(repo_root: Path = plan.REPO_ROOT) -> dict[str, Any]:
    return _v5_binding.validate_v2_sealed_artifact(repo_root)
