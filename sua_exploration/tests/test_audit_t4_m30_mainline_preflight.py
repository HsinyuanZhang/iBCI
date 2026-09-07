import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_t4_m30_mainline_preflight import (  # noqa: E402
    EXPECTED_ARTIFACT_SHA256,
    EXPECTED_EPOCHS,
    audit_m30_mainline,
    markdown_receipt,
)


def test_protocol_lock_has_full_three_by_three_matrix():
    assert len(EXPECTED_ARTIFACT_SHA256) == 9
    assert EXPECTED_EPOCHS == [5, 6, 7, 8, 9, 10, 11, 12]


def test_live_cpu_audit_is_fail_closed_without_opening_data():
    receipt = audit_m30_mainline()
    assert receipt["execution"] == {
        "cpu_only": True,
        "nwb_files_opened": [],
        "checkpoint_files_loaded": [],
        "formal_files_opened": False,
        "training_started": False,
        "gpu_used": False,
    }
    assert len(receipt["artifacts"]) == 9
    assert all(row["artifact_sha256_matches_protocol_lock"] for row in receipt["artifacts"])
    assert all(row["run_metadata_sha256_matches_artifact"] for row in receipt["artifacts"])
    assert receipt["eligibility"]["status"] == "fail"
    assert receipt["implementation_scope"]["ph4"]["implemented"] is False
    assert receipt["implementation_scope"]["ph4"]["authorized_by_this_receipt"] is False
    blockers = receipt["eligibility"]["blockers"]
    assert any("TS4@44: label/rate feature scope" in blocker for blocker in blockers)
    assert any("FIXED_POOL_SIZE=50" in blocker for blocker in blockers)
    assert any("historical M30 scorer trace" in blocker for blocker in blockers)


def test_markdown_receipt_never_claims_a_missing_trace_is_actual_indices():
    receipt = audit_m30_mainline()
    markdown = markdown_receipt(receipt)
    assert "Historical actual support/feature/scored indices: **unavailable_in_qualified_artifacts**" in markdown
    assert "descriptor implementation is not authorized" in markdown
    assert "PH4 is not implemented or authorized here" in markdown
