from __future__ import annotations

from pathlib import Path

from sua_exploration.scripts import verify_h1_calibration_future_quadratic_c2f5_metadata_correction as correction


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results/h1_calibration_future_quadratic_c2f5/QC2F5_METADATA_CORRECTION_v1.json"


def test_complete_5d_teacher_row_probe_uses_row_multiset_not_column_marginals() -> None:
    observed = correction.assert_complete_5d_teacher_permutation()
    assert observed["before_row_multiset_sha256"] == observed["after_row_multiset_sha256"]
    assert observed["complete_row_attachment_changed"] is True
    assert observed["fixed_points"] == 0


def test_additive_metadata_correction_binds_immutable_receipt_and_recomputes_lambda_rule() -> None:
    observed = correction.verify(ARTIFACT)
    assert observed["status"] == "PASS"
    assert observed["affected_fields"] == 12
