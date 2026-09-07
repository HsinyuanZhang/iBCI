"""Torch-free immutable identity for the V2 mixed-lineage scorer."""
from __future__ import annotations

from src.paired_anchored_calibration_dropout_score_v1 import plan as v1

CELL = "PACD_MATCHED_SCORE_V2_MIXED_LINEAGE"
SCHEMA = "pacd_matched_score_v2_mixed_lineage"
WORK_ORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_PACD_MATCHED_SCORE_V2_MIXED_LINEAGE_20260831.md"
WORK_ORDER_SHA256 = "3dd978cd8ab27b6f16549020c1d02eeb27717effa6808d4709b24cafc09aa567"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/paired_anchored_calibration_dropout_score_v2_mixed_lineage"
AUTHORITY_ROOT_RELATIVE = "tfpd_exploration/results/paired_anchored_calibration_dropout_score_v2_mixed_lineage_authority"
SYSTEM_ORDER = v1.SYSTEM_ORDER
SURFACE_ORDER = v1.SURFACE_ORDER
BUDGET_ORDER = v1.BUDGET_ORDER
REGIME_ORDER = v1.REGIME_ORDER
ROSTER_COUNTS = v1.ROSTER_COUNTS
EXPECTED_ROW_COUNT = v1.EXPECTED_ROW_COUNT

P0_ROOT = "tfpd_exploration/results/paired_anchored_calibration_dropout_full_v3/p0_fullfull_seed42"
P1_ROOT = "tfpd_exploration/results/paired_anchored_calibration_dropout_full_v4_admission/p1_m4_seed42"
P2_ROOT = "tfpd_exploration/results/paired_anchored_calibration_dropout_full_v4_admission/p2_m10_seed42"
V3_HISTORICAL_CLOSURE = "3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f"
V2_FAILURE_SHA = "c7f1a893a7c65dbb46b8493c9f8080ec12af5b3c7e97e606cb7d8aed005ea8b0"
V4_WORKORDER_SHA256 = "34a67357d66c36816457252e82d5ac7dcecf34340dafff1f372c67571dc09cd3"
V4_PARALLEL_ROUTE_ISOLATION_RELATIVE = "tfpd_exploration/docs/AUDIT_PACD_V4_PARALLEL_ROUTE_ISOLATION_20260831.md"
V4_PARALLEL_ROUTE_ISOLATION_SHA256 = "7c2c27697329e13f53fc2dacd7981cec2f24b88f550ab1af38c266c4cc2dcc52"
V4_SHARED_LIFECYCLE_SEAM_RELATIVE = "tfpd_exploration/docs/AUDIT_PACD_V4_SHARED_LIFECYCLE_SEAM_20260831.md"
V4_SHARED_LIFECYCLE_SEAM_SHA256 = "10bc3e81385e43bb6f9f79ebbdf13db8f5100a37de532d471f5792ecc83c9f8c"
V2_SMOKE_ATTEMPT_SHA = "ef2ebde24864c8105e47b6ef1c925149b128fe9531577dbf58fba227ea11eb3a"
V2_SMOKE_TERMINAL_SHA = "a04be949665a5c57091ba2192793401f43950735142fdb8fac9e2ec6829f9dfe"
V2_SMOKE_CLOSURE_SHA = "1b7ebf98a01e197718588eb9702949304f68b527ba08660ae88dd62857d11339"

# Live graph literals are deliberately unavailable until independently audited
# immutable producer terminals exist.
LIVE_MIXED_PRODUCER_LITERALS = None

BOUND_PATTERNS = tuple(v1.BOUND_PATTERNS) + (
    "tfpd_exploration/src/paired_anchored_calibration_dropout_score_v2_mixed_lineage/__init__.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_score_v2_mixed_lineage/plan.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_score_v2_mixed_lineage/binding.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_score_v2_mixed_lineage/smoke.py",
    # The V3 held-FD predecessor codec is invoked directly by the mixed
    # binding to validate the immutable V2 failure below P0.  Keep the full
    # small import chain explicit: a closure must cover every executed helper,
    # but must never discover artifacts or result directories dynamically.
    "tfpd_exploration/src/paired_anchored_calibration_dropout_full_v3/__init__.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_full_v3/plan.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_full_v3/predecessor.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_full_v2/__init__.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_full_v2/plan.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_full_v1/__init__.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_full_v1/plan.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_v1/__init__.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_v1/plan.py",
    V4_PARALLEL_ROUTE_ISOLATION_RELATIVE,
    # This read-only authority defines the staged V4 device/lifecycle receipt
    # seam whose exact cross-links are verified by this producer codec.
    V4_SHARED_LIFECYCLE_SEAM_RELATIVE,
    "tfpd_exploration/scripts/run_pacd_matched_score_v2_mixed_lineage.py",
    WORK_ORDER_RELATIVE,
)
