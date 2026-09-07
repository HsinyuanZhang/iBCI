"""Frozen V5 profile: exact V4 parent-layout failure recovery."""
from __future__ import annotations

from dataclasses import dataclass

from . import plan


@dataclass(frozen=True)
class AofsV5Profile:
    name: str = "aofs_static_v5_parent_layout_only_recovery"
    result_root_relative: str = plan.RESULT_ROOT_RELATIVE
    artifact_root_relative: str = plan.ARTIFACT_ROOT_RELATIVE
    predecessor_kind: str = "exact_aofs_v4_absent_artifact_parent_failure_and_v2_sealed_artifact"


V5_PROFILE = AofsV5Profile()
