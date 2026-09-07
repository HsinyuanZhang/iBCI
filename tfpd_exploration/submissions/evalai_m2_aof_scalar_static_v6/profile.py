"""Frozen V6 profile: V5 container-dependency-layout recovery."""
from __future__ import annotations
from dataclasses import dataclass
from . import plan


@dataclass(frozen=True)
class AofsV6Profile:
    name: str = "aofs_static_v6_container_dependency_layout_only_recovery"
    result_root_relative: str = plan.RESULT_ROOT_RELATIVE
    artifact_root_relative: str = plan.ARTIFACT_ROOT_RELATIVE
    predecessor_kind: str = "exact_aofs_v5_missing_third_party_container_failure_and_v2_sealed_artifact"


V6_PROFILE = AofsV6Profile()
