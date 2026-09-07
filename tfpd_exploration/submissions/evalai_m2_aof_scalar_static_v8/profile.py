"""Frozen V8 profile: V7 sealed container evidence reused for host validation."""
from __future__ import annotations
from dataclasses import dataclass
from . import plan
@dataclass(frozen=True)
class AofsV8Profile:
 name:str="aofs_static_v8_host_validation_namespace_only_recovery"
 result_root_relative:str=plan.RESULT_ROOT_RELATIVE
 artifact_root_relative:str=plan.ARTIFACT_ROOT_RELATIVE
 predecessor_kind:str="exact_aofs_v7_host_namespace_failure_with_sealed_container_minival"
V8_PROFILE=AofsV8Profile()
