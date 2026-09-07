"""Read-only regression coverage for the permanent r4 pre-GPU incident."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
R4 = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r4"
RESULT_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cells_20260805_r3"


def test_r4_is_permanently_failed_without_execution() -> None:
    incident = json.loads(
        (R4 / "r4_failed_pre_gpu_security_policy_relaxation.json").read_text(encoding="utf-8")
    )
    assert incident["status"] == "FAILED_PRE_GPU_SECURITY_POLICY_RELAXATION"
    assert incident["must_not_use_r4_authorizations"] is True
    assert incident["failure_reason"] == {
        "max_validity_seconds_was_relaxed": True,
        "multiple_trust_anchors_were_accepted": True,
        "required_policy": "single_fixed_anchor_and_MAX_VALIDITY_SECONDS_30h",
    }
    assert all(
        incident[field] is False
        for field in (
            "gpu_used", "score_data_accessed", "endpoint_opened", "cell_opened",
            "authorization_nonce_claimed", "r3_result_root_exists",
        )
    )
    assert len(incident["r4_authorizations"]) == 3
    assert not RESULT_ROOT.exists()
