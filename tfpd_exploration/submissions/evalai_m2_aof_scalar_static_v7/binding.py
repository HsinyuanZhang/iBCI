"""Held-FD verification of V6's exact post-input `src` import failure."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v4 import binding as _fd
from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v6 import binding as _v6
from . import plan
class BindingError(RuntimeError): pass
def _need(v: bool, m: str) -> None:
    if not v: raise BindingError(m)
def validate_v6_failure_graph(repo_root: Path = plan.REPO_ROOT) -> dict[str, Any]:
    bodies, identity = _fd._graph(repo_root / plan.V6_FAILURE_ROOT_RELATIVE, plan.V6_FAILURE_BODIES, "V6 src-import failure")
    attempt, predecessor, input_authority, failure = (json.loads(bodies[n]) for n in ("attempt.json", "predecessor_authority.json", "input_authority.json", "failure.json"))
    _need(attempt.get("schema") == "m2_aof_scalar_static_package_v6_attempt" and attempt.get("closure_sha256") == plan.V6_FAILURE_CLOSURE_SHA256, "V6 attempt drift")
    _need(set(predecessor) == {"v5_failure", "v2_sealed_artifact"} and predecessor["v2_sealed_artifact"].get("bodies") == plan.V2_ARTIFACT_BODIES, "V6 predecessor drift")
    _need(input_authority.get("scientific_rebuild") is False and input_authority.get("v2_receipt_sha256") == plan.V2_ARTIFACT_BODIES[plan.RECEIPT_NAME], "V6 payload drift")
    _need(failure.get("schema") == "m2_aof_scalar_static_package_v6_failure" and failure.get("exception_class") == "DockerArgvError" and failure.get("published_prefix") == ["attempt.json", "predecessor_authority.json", "input_authority.json"] and failure.get("exception_message", "").startswith("AOF-S V6 Docker command failed: Traceback") and "/workspace/decode.py" in failure.get("exception_message", ""), "V6 post-input container failure semantics drift")
    return {"root_relative": plan.V6_FAILURE_ROOT_RELATIVE, "root": identity, "bodies": dict(plan.V6_FAILURE_BODIES), "closure_sha256": plan.V6_FAILURE_CLOSURE_SHA256, "failure_stage": "post_input_container_payload_load_missing_src"}
def validate_v2_sealed_artifact(repo_root: Path = plan.REPO_ROOT) -> dict[str, Any]: return _v6.validate_v2_sealed_artifact(repo_root)
