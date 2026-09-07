"""Contracts for the append-only all-source provenance-gap recovery audit."""
from __future__ import annotations

import json
from pathlib import Path
import stat
from typing import Any

import pytest

from scripts import h1_carrierid_all_source_official_recovery_audit as audit


def _immutable(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o444)
    return path


def _marker_text(log: Path) -> str:
    return "\n".join([
        "start_time=2026-08-10T01:30:29+08:00",
        f"workdir={audit.ROOT}",
        "tmux_session=h1_all_source_gpu1",
        "command=" + "CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 " + " ".join([
            audit.EXPECTED_PYTHON,
            "src/train.py",
            "experiment=h1_carrierid_all_source_official",
            f"official_candidate.asset_manifest_path={audit.ASSET.resolve()}",
            "ckpt_path=null", "test=false",
        ]),
        f"stdout_stderr_log={log}",
    ]) + "\n"


def test_legacy_start_marker_is_explicitly_unbound(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    marker = _immutable(tmp_path / "start.txt", _marker_text(log))
    parsed = audit._parse_start_marker(marker)
    binding = audit._command_binding(parsed["command"])
    assert "nonce" not in parsed
    assert binding["environment"] == {"CUDA_VISIBLE_DEVICES": "1", "PYTHONNOUSERSITE": "1"}
    assert binding["argv"][-1] == "test=false"


def test_marker_nonce_fields_are_rejected_in_gap_recovery(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    marker = _immutable(tmp_path / "start.txt", _marker_text(log) + "nonce=made-up\n")
    with pytest.raises(audit.RecoveryAuditError, match="nonce-bound fields"):
        audit._parse_start_marker(marker)


@pytest.mark.parametrize(
    "entry, match",
    [
        (str(audit.ROOT / "src/train.py"), "executable drift"),
        ("src/other.py", "executable drift"),
        ("src/train.py --unexpected", "argv drift"),
    ],
)
def test_command_binding_rejects_absolute_or_extra_argv_entries(entry: str, match: str) -> None:
    command = "CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 " + " ".join([
        audit.EXPECTED_PYTHON, entry,
        "experiment=h1_carrierid_all_source_official",
        f"official_candidate.asset_manifest_path={audit.ASSET.resolve()}",
        "ckpt_path=null", "test=false",
    ])
    with pytest.raises(audit.RecoveryAuditError, match=match):
        audit._command_binding(command)


def test_process_snapshot_never_invents_post_exit_pid_data() -> None:
    result = audit._process_snapshot(987654321, expected_argv=[])
    assert result["status"] == "unavailable_after_termination"
    assert result["argv"] is None and result["start_time_ticks"] is None
    assert audit._process_snapshot(None, expected_argv=[])["status"] == "not_requested"


def test_append_only_writer_is_immutable_and_non_overwriting(tmp_path: Path) -> None:
    output = tmp_path / "recovery.json"
    digest = audit._write_append_only(output, {"schema": audit.SCHEMA})
    assert len(digest) == 64
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    with pytest.raises(audit.RecoveryAuditError, match="overwrite"):
        audit._write_append_only(output, {"schema": audit.SCHEMA})


def test_recover_publishes_gap_without_fabricating_nonce(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = tmp_path / "train.log"
    log.write_text("source-only training completed\n", encoding="utf-8")
    marker = _immutable(tmp_path / "start.txt", _marker_text(log))
    # Keep the integration test source-only: replace artifact hash/checkpoint
    # readers with deterministic evidence while exercising the append-only
    # transaction and provenance fields.
    monkeypatch.setattr(audit, "EXECUTION", tmp_path / "missing-execution.json")
    monkeypatch.setattr(audit, "_validate_assets", lambda: {"path": "asset", "sha256": "a" * 64})
    monkeypatch.setattr(audit, "_validate_prepared_launch", lambda **_: {"path": "launch", "sha256": "b" * 64, "candidate": {}, "code_sha256": {}})
    monkeypatch.setattr(audit, "_validate_code_hashes", lambda _claimed: {"prepared": {}, "current": {}, "all_match": True})
    monkeypatch.setattr(audit, "_validate_checkpoint", lambda *_args, **_kwargs: {"path": "checkpoint", "sha256": "c" * 64, "global_step": audit.EXPECTED_GLOBAL_STEP})
    output = tmp_path / "audit.json"
    result = audit.recover(checkpoint=tmp_path / "checkpoint.ckpt", output=output, start_marker=marker, train_log=log)
    assert result["status"] == audit.PASS_STATUS
    assert result["provenance_gap"]["execution_receipt_present"] is False
    assert result["provenance_gap"]["nonce_bound_launch_verified"] is False
    assert result["provenance_gap"]["nonce_reconstructed"] is False
    body = json.loads(output.read_text(encoding="utf-8"))
    assert body["scope"]["target_recordings_opened"] == 0
    with pytest.raises(audit.RecoveryAuditError, match="overwrite"):
        audit.recover(checkpoint=tmp_path / "checkpoint.ckpt", output=output, start_marker=marker, train_log=log)


def test_recovery_script_has_no_target_or_evalai_execution_imports() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    for forbidden in ("from src.data.h1_carrierid_date_lodo_target", "NWBHDF5IO", "target_module"):
        assert forbidden not in source
