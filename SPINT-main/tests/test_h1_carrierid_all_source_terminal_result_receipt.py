from __future__ import annotations

import copy
import json
import stat
from pathlib import Path

import pytest

import scripts.h1_carrierid_all_source_terminal_result_receipt as receipt


ROOT = receipt.ROOT
RECEIPT_PATH = ROOT / (
    "pilot_artifacts/h1_carrierid_all_source_official_v1/"
    "H1_CARRIERID_ALL_SOURCE_EVALAI_TERMINAL_RESULT_RECEIPT_v5.json"
)
REGISTERED_STATE = ROOT / (
    "pilot_artifacts/h1_carrierid_all_source_official_v1/"
    "H1_CARRIERID_ALL_SOURCE_EVALAI_PUSH_STATE_v5.json.registered.json"
)
BASE_STATE = ROOT / (
    "pilot_artifacts/h1_carrierid_all_source_official_v1/"
    "H1_CARRIERID_ALL_SOURCE_EVALAI_PUSH_STATE_v5.json"
)


def _receipt_body() -> dict[str, object]:
    return json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))


def _tampered_copy(tmp_path: Path, mutate) -> Path:
    body = _receipt_body()
    mutate(body)
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return path


def test_terminal_receipt_verifies_immutable_bindings_and_deltas():
    result = receipt.verify_terminal_receipt(RECEIPT_PATH)
    assert result["schema"] == receipt.RECEIPT_SCHEMA
    assert result["status"] == receipt.RECEIPT_STATUS
    assert result["submission_id"] == 578689
    assert result["terminal_status"] == "finished"
    assert result["per_session_metrics_available"] is False
    assert result["baselines"]["578473"]["Held Out R2 Mean"] == "0.06502244658739412"
    assert result["baselines"]["578474"]["Normalized Latency"] == "-0.01515589765063262"
    assert stat.S_IMODE(RECEIPT_PATH.stat().st_mode) == 0o444


def test_receipt_records_exact_terminal_flags_urls_and_metrics():
    body = _receipt_body()
    submission = body["submission"]
    assert submission["id"] == 578689
    assert submission["challenge_id"] == 2319
    assert submission["phase_id"] == 4599
    assert submission["team_id"] == 41975
    assert submission["status"] == "finished"
    assert submission["is_public"] is False
    assert submission["is_flagged"] is False
    assert submission["is_verified_by_host"] is False
    assert submission["submitted_at"] == "2026-08-09T22:09:41.512779Z"
    assert submission["started_at"] == "2026-08-09T22:25:51.702890Z"
    assert submission["completed_at"] == "2026-08-09T22:25:51.821669Z"
    assert submission["submission_result_file"] == receipt.EXPECTED_TERMINAL_RESULT_URL
    assert submission["metrics"] == receipt.EXPECTED_METRICS
    assert body["per_session_metrics"]["available"] is False
    assert body["per_session_metrics"]["held_in"]["values"] is None
    assert body["per_session_metrics"]["held_out"]["values"] is None


def test_verifier_rejects_terminal_status_tampering(tmp_path: Path):
    path = _tampered_copy(tmp_path, lambda body: body["submission"].update(status="running"))
    with pytest.raises(receipt.TerminalReceiptError, match="terminal API field status|terminal receipt"):
        receipt.verify_terminal_receipt(path)


def test_verifier_rejects_metric_tampering(tmp_path: Path):
    def mutate(body):
        body["submission"]["metrics"]["test_split_h1"]["Held Out R2 Mean"] += 1e-6

    path = _tampered_copy(tmp_path, mutate)
    with pytest.raises(receipt.TerminalReceiptError, match="metric|canonical"):
        receipt.verify_terminal_receipt(path)


def test_verifier_rejects_new_per_session_values(tmp_path: Path):
    def mutate(body):
        body["per_session_metrics"]["available"] = True
        body["per_session_metrics"]["held_in"]["values"] = {"S0_set_1": 0.4}

    path = _tampered_copy(tmp_path, mutate)
    with pytest.raises(receipt.TerminalReceiptError, match="per-session"):
        receipt.verify_terminal_receipt(path)


def test_verifier_rejects_state_hash_drift(tmp_path: Path):
    path = _tampered_copy(
        tmp_path,
        lambda body: body["state_bindings"]["registered"].update(sha256="0" * 64),
    )
    with pytest.raises(receipt.TerminalReceiptError, match="state hash/mode drift"):
        receipt.verify_terminal_receipt(path)


def test_verifier_rejects_baseline_delta_drift(tmp_path: Path):
    path = _tampered_copy(
        tmp_path,
        lambda body: body["baselines"]["578473"]["delta_candidate_minus_baseline"].update(
            {"Held Out R2 Mean": "0.0"}
        ),
    )
    with pytest.raises(receipt.TerminalReceiptError, match="baseline 578473 delta drift"):
        receipt.verify_terminal_receipt(path)


def test_create_is_append_only_and_refuses_existing_receipt(tmp_path: Path):
    output = tmp_path / "new-receipt.json"
    response = {
        "id": receipt.EXPECTED_SUBMISSION_ID,
        "challenge_phase": receipt.EXPECTED_PHASE_ID,
        "participant_team": receipt.EXPECTED_TEAM_ID,
        "status": "finished",
        "is_public": False,
        "is_flagged": False,
        "is_verified_by_host": False,
        "ignore_submission": False,
        "is_baseline": False,
        "submitted_at": "2026-08-09T22:09:41.512779Z",
        "started_at": "2026-08-09T22:25:51.702890Z",
        "completed_at": "2026-08-09T22:25:51.821669Z",
        "execution_time": 0.118779,
        "rerun_resumed_at": None,
        "submission_result_file": receipt.EXPECTED_TERMINAL_RESULT_URL,
        "stdout_file": receipt.EXPECTED_STDOUT_URL,
        "stderr_file": receipt.EXPECTED_STDERR_URL,
    }
    receipt.create_terminal_receipt(
        registered_state=REGISTERED_STATE,
        base_state=BASE_STATE,
        output=output,
        terminal_response=response,
        metrics=receipt.EXPECTED_METRICS,
    )
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError, match="overwrite|immutable"):
        receipt.create_terminal_receipt(
            registered_state=REGISTERED_STATE,
            base_state=BASE_STATE,
            output=output,
            terminal_response=response,
            metrics=receipt.EXPECTED_METRICS,
        )


def test_verifier_rejects_external_action_claims(tmp_path: Path):
    path = _tampered_copy(
        tmp_path,
        lambda body: body["external_action"].update(resubmission_performed_by_this_tool=True),
    )
    with pytest.raises(receipt.TerminalReceiptError, match="external action"):
        receipt.verify_terminal_receipt(path)
