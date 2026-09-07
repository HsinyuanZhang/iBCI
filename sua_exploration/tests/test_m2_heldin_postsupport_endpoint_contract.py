"""Contract tests for the M2 held-in post-support endpoint audit and protocol."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "SPINT-main/data"
AUDIT_PATH = ROOT / "sua_exploration/results/m2_heldin_postsupport_endpoint_v1/audit.json"
AUDIT_SHA256 = "d6940d156d05cd1cbd43bac95220c1d1db370c50d841de8a51d49a53c9ac70c5"
PROTOCOL_PATH = ROOT / "sua_exploration/results/m2_heldin_postsupport_endpoint_v1/protocol_receipt.json"
PROTOCOL_SHA256 = "7673d360099775e37b199621956d099a9a2a7bafcbe7b074c3146978300f1c49"
SCRIPT = ROOT / "sua_exploration/scripts/audit_m2_heldin_postsupport_endpoint.py"

EXPECTED_HELDIN_POST_SUPPORT = {
    "2020-10-19-Run1": 304,
    "2020-10-19-Run2": 306,
    "2020-10-20-Run1": 215,
    "2020-10-20-Run2": 212,
    "2020-10-27-Run1": 171,
    "2020-10-27-Run2": 218,
    "2020-10-28-Run1": 266,
}
EXPECTED_HELDOUT_POST_SUPPORT = {
    "2020-10-30-Run1": 10,
    "2020-10-30-Run2": 8,
    "2020-11-18-Run1": 8,
    "2020-11-19-Run1": 10,
    "2020-11-24-Run1": 0,
    "2020-11-24-Run2": 0,
}

SEALED_BRANCH_PATTERNS = [
    re.compile(r"B3TStream", re.I),
    re.compile(r"\bK4\b"),
    re.compile(r"\bKS4\b"),
    re.compile(r"SSC-T4", re.I),
    re.compile(r"\bB3TS\b", re.I),
    re.compile(r"T4-CFILM", re.I),
    re.compile(r"clean[-_]?SPINT", re.I),
    re.compile(r"\bTS4\b"),
]

requires_falcon_data = pytest.mark.skipif(
    not (DATA / "000953").is_dir(),
    reason="local FALCON M2 dandiset is not present",
)


def load_audit_module():
    spec = importlib.util.spec_from_file_location("m2_postsupport_audit", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit() -> dict:
    return json.loads(AUDIT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def protocol() -> dict:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


@requires_falcon_data
def test_heldin_per_session_post_support_counts_match_measurement(audit: dict) -> None:
    sessions = audit["heldin_postsupport_endpoint"]["sessions"]
    for session, expected in EXPECTED_HELDIN_POST_SUPPORT.items():
        assert sessions[session]["post_support_trials"] == expected
    assert audit["heldin_postsupport_endpoint"]["totals"]["raw_post_support_trials"] == sum(
        EXPECTED_HELDIN_POST_SUPPORT.values()
    )


@requires_falcon_data
def test_heldout_m33_endpoint_has_36_query_trials_across_four_eligible_sessions(audit: dict) -> None:
    heldout = audit["heldout_m33_endpoint"]
    assert heldout["eligible_session_count"] == 4
    assert heldout["totals"]["raw_query_trials"] == 36
    for session, expected in EXPECTED_HELDOUT_POST_SUPPORT.items():
        assert heldout["sessions"][session]["query_trials_after_support"] == expected


@requires_falcon_data
def test_heldin_endpoint_has_seven_sessions_and_far_more_trials_than_heldout(audit: dict) -> None:
    comparison = audit["endpoint_comparison"]
    assert comparison["heldin_post_support"]["session_count"] == 7
    assert comparison["heldout_m33_eligible"]["session_count"] == 4
    assert comparison["heldin_post_support"]["raw_post_support_trials"] == 1692
    assert comparison["heldout_m33_eligible"]["raw_query_trials"] == 36
    assert (
        comparison["ratios"]["raw_trials_heldin_over_heldout"]
        == 1692 / 36
    )


@requires_falcon_data
def test_all_seven_loso_folds_leave_out_session_absent_from_training(audit: dict) -> None:
    isolation = audit["loso_session_isolation"]
    assert len(isolation) == 7
    for fold in isolation.values():
        assert fold["left_out_session_absent_from_train"] is True
        left_out = fold["left_out_session"]
        assert left_out not in fold["train_session_names"]
        assert left_out in fold["val_heldin_session_names"]


@requires_falcon_data
def test_minival_is_exact_prefix_for_all_seven_heldin_sessions(audit: dict) -> None:
    endpoint = audit["internal_loso_heldin_endpoint"]
    assert endpoint["total_session_count"] == 7
    assert endpoint["contaminated_session_count"] == 7
    for session in endpoint["sessions"].values():
        assert session["minival_is_exact_prefix_of_calib"] is True
        assert session["scored_trials_inside_support_prefix"] is True


def test_protocol_contains_candidate_branch_seal(protocol: dict) -> None:
    seal = protocol["candidate_branch_seal"]
    assert seal["binding"] is True
    assert "B3TStream+T4" in seal["sealed_branches"]
    assert "K4" in seal["sealed_branches"]
    assert protocol["framing"] == seal["framing"]
    assert protocol["evaluation_contract"]["no_pass_fail_gate"] is True
    assert protocol["evaluation_contract"]["no_effective_verdict"] is True


def _text_outside_seal_lists(payload: dict) -> str:
    """Serialize JSON omitting seal branch name lists that must name forbidden branches."""
    scrubbed = json.loads(json.dumps(payload))
    seal = scrubbed.get("candidate_branch_seal")
    if isinstance(seal, dict):
        seal.pop("sealed_branches", None)
    contract = scrubbed.get("evaluation_contract")
    if isinstance(contract, dict):
        contract.pop("forbidden_comparators", None)
    return json.dumps(scrubbed)


def test_emitted_artifacts_contain_no_sealed_candidate_branch_identifiers_outside_seal_lists(
    audit: dict, protocol: dict
) -> None:
    for name, payload in (("audit", audit), ("protocol", protocol)):
        text = _text_outside_seal_lists(payload)
        for pattern in SEALED_BRANCH_PATTERNS:
            assert pattern.search(text) is None, f"{pattern.pattern} found outside seal lists in {name}"


def test_audit_and_protocol_match_recorded_sha256() -> None:
    module = load_audit_module()
    assert module.sha256(AUDIT_PATH) == AUDIT_SHA256
    recorded = (AUDIT_PATH.parent / "audit.sha256").read_text().split()[0]
    assert recorded == AUDIT_SHA256
    assert module.sha256(PROTOCOL_PATH) == PROTOCOL_SHA256
    recorded_protocol = (PROTOCOL_PATH.parent / "protocol_receipt.sha256").read_text().split()[0]
    assert recorded_protocol == PROTOCOL_SHA256


@requires_falcon_data
def test_datamodule_guard_rejects_m2_heldin_query_start_without_widened_guard(audit: dict) -> None:
    probe = audit["datamodule_guard_probe"]
    assert probe["raised"] is True
    assert "heldin_query_start_trial" in probe["error"]


@requires_falcon_data
def test_audit_refuses_to_overwrite_existing_directory(tmp_path: Path) -> None:
    module = load_audit_module()
    out = tmp_path / "audit"
    module.run(out, include_window_audit=False, include_datamodule=False)
    with pytest.raises(FileExistsError):
        module.run(out, include_window_audit=False, include_datamodule=False)
