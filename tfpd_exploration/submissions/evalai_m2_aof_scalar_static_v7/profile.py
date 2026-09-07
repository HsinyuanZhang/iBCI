"""Frozen V7 profile: literal-root PYTHONPATH-only recovery."""
from __future__ import annotations
from dataclasses import dataclass
from . import plan
@dataclass(frozen=True)
class AofsV7Profile:
    name: str = "aofs_static_v7_literal_root_pythonpath_only_recovery"
    result_root_relative: str = plan.RESULT_ROOT_RELATIVE
    artifact_root_relative: str = plan.ARTIFACT_ROOT_RELATIVE
    predecessor_kind: str = "exact_aofs_v6_post_input_missing_src_container_failure_and_v2_sealed_artifact"
V7_PROFILE = AofsV7Profile()
