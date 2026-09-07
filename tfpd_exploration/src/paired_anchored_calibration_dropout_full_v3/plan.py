"""Immutable V3 identity; all numerical science is inherited from V2."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from src.paired_anchored_calibration_dropout_full_v2 import plan as v2

CELL = "PACD_MATCHED_FULL_TRAINING_V3"
SCHEMA = "pacd_matched_full_training_v3"
WORK_ORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_PACD_MATCHED_FULL_TRAINING_V3_20260831.md"
ROOT_BASE = "tfpd_exploration/results/paired_anchored_calibration_dropout_full_v3"
ARMS = MappingProxyType({
    "p0": {"short_m": 30, "root": ROOT_BASE + "/p0_fullfull_seed42"},
    "p1": {"short_m": 4, "root": ROOT_BASE + "/p1_m4_seed42"},
    "p2": {"short_m": 10, "root": ROOT_BASE + "/p2_m10_seed42"},
})
V2_FAILURE_RELATIVE = "tfpd_exploration/results/paired_anchored_calibration_dropout_full_v2/p0_fullfull_seed42"
V2_FAILURE_SHAS = MappingProxyType({
    "attempt.json": "f0bc06ee590a68c669819ee4a2893e1f5c3af9408fd4fb2b198dacafaedb4bf6",
    "launch.json": "25de41fbd71fe6fed6f691db7acb018db03a47ba59ed6aa1a6f08995fa73410a",
    "source_authority.json": "452ed16be3c76f5e793ff9e21a58dad33d30e26bdfb1d4ace1cbab3e96e30071",
    "failure.json": "c7f1a893a7c65dbb46b8493c9f8080ec12af5b3c7e97e606cb7d8aed005ea8b0",
})
ZERO_ENCODER_POLICY = "accept_if_paired_unit_mask_empty"
BOUND_PATTERNS = tuple(v2.BOUND_PATTERNS) + (
    "tfpd_exploration/src/paired_anchored_calibration_dropout_full_v3/__init__.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_full_v3/plan.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_full_v3/predecessor.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_full_v3/smoke.py",
    "tfpd_exploration/scripts/run_pacd_full_training_v3.py",
    WORK_ORDER_RELATIVE,
)


@dataclass(frozen=True)
class RuntimePlan:
    """Read-only V3 overrides with the inherited V2 numerical contract."""

    CELL: str = CELL
    SCHEMA: str = SCHEMA
    ARMS: Any = ARMS
    BOUND_PATTERNS: tuple[str, ...] = BOUND_PATTERNS
    ZERO_ENCODER_POLICY: str = ZERO_ENCODER_POLICY
    # Review-only evidence is sealed into the immutable attempt, but remains
    # deliberately outside BOUND_PATTERNS so an audit-comment/test edit cannot
    # invalidate a running job at terminal publication.
    REVIEW_EVIDENCE_PATHS: tuple[str, ...] = tuple(v2.RUNTIME_PLAN.REVIEW_EVIDENCE_PATHS) + (
        "tfpd_exploration/tests/test_pacd_full_training_v3.py",
    )

    def __getattr__(self, name: str) -> Any:
        return getattr(v2.RUNTIME_PLAN, name)


RUNTIME_PLAN = RuntimePlan()
