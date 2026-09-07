"""Static V2 successor identity; all scientific constants are inherited from V1."""
from __future__ import annotations

from src.paired_anchored_calibration_dropout_v1 import plan as v1

CELL = "PAIRED_ANCHORED_CALIBRATION_DROPOUT_V2"
SCHEMA = "paired_anchored_calibration_dropout_v2"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/paired_anchored_calibration_dropout_v2/smoke_seed42"
WORK_ORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_PACD_V2_20260831.md"
PREDECESSOR_RELATIVE = "tfpd_exploration/results/paired_anchored_calibration_dropout_v1/smoke_seed42"
V1_ATTEMPT_SHA256 = "e1d6cd813b2bc49d89121b3341271a0bf94176abe3811d2f3b6f6ffa4a28b986"
V1_FAILURE_SHA256 = "82ca6850cdd50f0b5ea5073eb89809a0c6ea426ac3f8f7d9e03fc44d1bd7e273"
V1_CLOSURE_SHA256 = "0848f4ca2ce2753757cd557f46a9cba30c4ac5352fb5dbf77fe357c7fd74bdb6"
V1_WORKORDER_SHA256 = "843240ea69b4770f0a8cb4bd25618d4edf94f63488f263f2e74b19c091937dc8"
EXPECTED_SEALED_SHA256 = dict(v1.EXPECTED_SEALED_SHA256)
BOUND_PATTERNS = tuple(v1.BOUND_PATTERNS) + (
    "tfpd_exploration/src/paired_anchored_calibration_dropout_v2/__init__.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_v2/plan.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_v2/predecessor.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_v2/smoke.py",
    "tfpd_exploration/scripts/run_pacd_smoke_v2.py",
    WORK_ORDER_RELATIVE,
)
REVIEW_EVIDENCE_PATHS = (
    "tfpd_exploration/tests/test_paired_anchored_calibration_dropout_v2.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_v1/core.py",
)
