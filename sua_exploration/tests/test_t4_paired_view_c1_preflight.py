from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sua_exploration" / "scripts"))

import audit_t4_paired_view_c1_preflight as audit  # noqa: E402


def test_current_source_semantics_resolve_q30_pool50_and_score_start50() -> None:
    receipt = audit.build_receipt(ROOT)
    assert receipt["bridge_identity"] == "paired-view bridge-Q30/T4-50-source10"
    assert receipt["semantics"]["forward_activity_support"]["trial_list_indices"] == [0, 29]
    assert receipt["semantics"]["t4_label_rate_pool"]["trial_list_indices"] == [0, 49]
    assert receipt["semantics"]["score_trials"]["start_trial_list_index"] == 50
    assert receipt["artifact_audit"]["normalizer_isolation"]["distinct"] is True
    assert receipt["formal_sua_files_opened"] is False
    assert receipt["c1_implementation_may_begin"] is False
    assert receipt["verdict"] == "blocked_fail_closed"


def test_scorer_audit_fails_when_pool_exclusion_is_missing() -> None:
    evaluator = (
        "pool_size=args.pool_size\nselection_mode=FIXED_SELECTION_MODE\n"
        "calibration_n=args.calibration_n\n\"label_feature_calibration_n\": label_feature_pool_size"
    )
    protocol = "indices = select_calibration_trial_indices(rec[\"trials\"], calibration_n, pool_size, mode)"
    with pytest.raises(audit.AuditFailure, match="eval_trials"):
        audit.audit_scorer_semantics(evaluator, protocol)
