"""Frozen V3 profile: V2 sealed artifact reuse and Docker-argv repair."""
from __future__ import annotations
from dataclasses import dataclass
from . import plan

@dataclass(frozen=True)
class AofsV3Profile:
    name: str = "aofs_static_v3_local_docker_argv_recovery"
    result_root_relative: str = plan.RESULT_ROOT_RELATIVE
    artifact_root_relative: str = plan.ARTIFACT_ROOT_RELATIVE
    predecessor_kind: str = "exact_aofs_v2_docker_sdk_failure_and_sealed_artifact"

V3_PROFILE = AofsV3Profile()

