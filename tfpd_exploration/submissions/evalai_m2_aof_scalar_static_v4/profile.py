"""Frozen V4 profile: exact V3 layout failure recovery with V2 payload reuse."""
from __future__ import annotations

from dataclasses import dataclass

from . import plan


@dataclass(frozen=True)
class AofsV4Profile:
    name: str = "aofs_static_v4_layout_only_recovery"
    result_root_relative: str = plan.RESULT_ROOT_RELATIVE
    artifact_root_relative: str = plan.ARTIFACT_ROOT_RELATIVE
    predecessor_kind: str = "exact_aofs_v3_container_layout_failure_and_v2_sealed_artifact"


V4_PROFILE = AofsV4Profile()
