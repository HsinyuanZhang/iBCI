"""Regression guard for the selected T4@50 scorer-boundary audit."""
from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_selected_t4_m50_scorer_boundary.py"
SPEC = importlib.util.spec_from_file_location("selected_t4_m50_boundary_audit", SCRIPT)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_m50_partition_excludes_feature_fit_tail_from_scoring() -> None:
    partition = AUDIT.ordered_trial_partition(range(80))

    assert partition["activity_forward_support"]["original_trial_indices"] == list(range(30))
    assert partition["t4_label_rate_fit"]["original_trial_indices"] == list(range(50))
    assert partition["trials_30_49"]["original_trial_indices"] == list(range(30, 50))
    assert partition["trials_30_49"]["scored"] is False
    assert partition["trials_30_49"]["used_to_fit_t4_label_rate_features"] is True
    assert partition["scored"]["original_trial_indices"] == list(range(50, 80))
    assert partition["t4_fit_scored_overlap_original_trial_indices"] == []


def test_partition_preserves_nonconsecutive_original_trial_table_indices() -> None:
    original = [100 + 3 * index for index in range(57)]
    partition = AUDIT.ordered_trial_partition(original)

    assert partition["activity_forward_support"]["original_trial_indices"] == original[:30]
    assert partition["trials_30_49"]["original_trial_indices"] == original[30:50]
    assert partition["scored"]["original_trial_indices"] == original[50:]
