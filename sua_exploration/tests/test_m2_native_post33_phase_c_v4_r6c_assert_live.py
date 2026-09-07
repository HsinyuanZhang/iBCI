"""Synthetic, score-free tests for the r6c signer liveness watchdog."""
from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from uuid import uuid4
import ast

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_assert_live.py"
WATCH_KWARGS = {
    "matrix_pid": 7654,
    "matrix_starttime": 222,
    "matrix_parent_pid": 5432,
    "matrix_parent_starttime": 333,
    "matrix_pgid": 7654,
}


def _module():
    name = f"_r6c_assert_live_test_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _configure_live_fixture(module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Point every watchdog control boundary at disposable non-workspace files."""
    sandbox = tmp_path / "r6c"
    receipts = sandbox / "receipts"
    cells = sandbox / "cells"
    runtime = sandbox / "runtime"
    ready = runtime / "live_signer_ready.json"
    public = sandbox / "r6c_public.pem"
    signer_source = sandbox / "m2_native_post33_phase_c_v4_r6c_live_signer.py"
    completed = cells / "protocol/phase/stage_a/status.completed.json"
    for directory in (receipts, cells, runtime):
        directory.mkdir(parents=True, exist_ok=True)
    public.write_bytes(b"synthetic public key only\n")
    signer_source.write_text("# synthetic signer source only\n", encoding="utf-8")

    monkeypatch.setattr(module, "R6C_RECEIPTS", receipts)
    monkeypatch.setattr(module, "R6C_CELL_ROOT", cells)
    monkeypatch.setattr(module, "RUNTIME", runtime)
    monkeypatch.setattr(module, "READY", ready)
    monkeypatch.setattr(module, "PUBLIC_KEY", public)
    monkeypatch.setattr(module, "SIGNER_SOURCE", signer_source)
    monkeypatch.setattr(module, "STAGE_A_COMPLETED", completed)
    monkeypatch.setattr(module, "socket", SimpleNamespace(gethostname=lambda: "synthetic-host"))
    monkeypatch.setattr(module, "_proc_starttime", lambda _pid: 111)
    bootstrap_cmdline = module._expected_signer_bootstrap_cmdline()
    command_hash = hashlib.sha256(bootstrap_cmdline).hexdigest()
    monkeypatch.setattr(module, "_proc_cmdline_bytes", lambda _pid: bootstrap_cmdline)
    monkeypatch.setattr(module, "_proc_cmdline_sha256", lambda _pid: command_hash)
    payload: dict[str, object] = {
        "schema": module.READY_SCHEMA,
        "status": "LIVE_PRIVATE_KEY_IN_MEMORY_ONLY",
        "host_id": "synthetic-host",
        "pid": 4321,
        "proc_starttime_ticks": 111,
        "proc_cmdline_sha256": command_hash,
        "controller_source": module._file_metadata(signer_source),
        "public_key": module._file_metadata(public),
        "intended_receipt_root": str(receipts.resolve()),
        "intended_cell_root": str(cells.resolve()),
        "private_key_serialized_or_disk_persisted": False,
        "private_key_retained_live_in_memory": True,
        "private_key_not_supplied_via_argv_environment_file_stdin_or_ipc": True,
    }
    ready.write_text(json.dumps(payload), encoding="utf-8")
    return {"sandbox": sandbox, "payload": payload, "completed": completed}


def test_assert_live_accepts_only_the_exact_synthetic_bound_signer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _configure_live_fixture(module, tmp_path, monkeypatch)
    identity = module.assert_live()
    assert identity.pid == 4321
    assert identity.proc_starttime_ticks == 111
    assert identity.proc_cmdline_sha256 == hashlib.sha256(module._expected_signer_bootstrap_cmdline()).hexdigest()
    assert identity.host_id == "synthetic-host"


def test_assert_live_rejects_a_pid_whose_hash_matches_but_bootstrap_argv_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    fixture = _configure_live_fixture(module, tmp_path, monkeypatch)
    unexpected = b"python\0unexpected.py\0"
    payload = dict(fixture["payload"])
    payload["proc_cmdline_sha256"] = hashlib.sha256(unexpected).hexdigest()
    module.READY.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(module, "_proc_cmdline_bytes", lambda _pid: unexpected)
    with pytest.raises(module.LiveAssertionError, match="fixed r6c bootstrap argv"):
        module.assert_live()


@pytest.mark.parametrize(
    "field,value",
    [
        ("host_id", "different-host"),
        ("status", "NOT_LIVE"),
        ("pid", 0),
        ("proc_starttime_ticks", 112),
        ("proc_cmdline_sha256", "b" * 64),
        ("intended_receipt_root", "/wrong/receipts"),
        ("intended_cell_root", "/wrong/cells"),
        ("controller_source", {"canonical_path": "/wrong.py", "size_bytes": 1, "sha256": "0" * 64}),
        ("public_key", {"canonical_path": "/wrong.pem", "size_bytes": 1, "sha256": "1" * 64}),
        ("private_key_serialized_or_disk_persisted", True),
        ("private_key_retained_live_in_memory", False),
        ("private_key_not_supplied_via_argv_environment_file_stdin_or_ipc", False),
    ],
)
def test_assert_live_fails_closed_for_any_ready_identity_or_root_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    module = _module()
    fixture = _configure_live_fixture(module, tmp_path, monkeypatch)
    payload = dict(fixture["payload"])
    payload[field] = value
    module.READY.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(module.LiveAssertionError):
        module.assert_live()


def test_check_once_is_a_silent_status_interface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _module()
    _configure_live_fixture(module, tmp_path, monkeypatch)
    assert module.main(["--check-once"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""

    module.READY.unlink()
    assert module.main(["--check-once"]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_watch_on_signer_failure_terms_only_the_verified_matrix_group_and_appends_control_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _configure_live_fixture(module, tmp_path, monkeypatch)
    signer = module.SignerIdentity(4321, 111, "a" * 64, "synthetic-host")
    outcomes: list[object] = [signer, module.LiveAssertionError("synthetic signer death")]

    def assert_live():
        value = outcomes.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(module, "assert_live", assert_live)
    monkeypatch.setattr(module, "_assert_matrix_pid_identity", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_stable_stage_a_completed", lambda: False)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(module.os, "getpgid", lambda _pid: 7654)
    monkeypatch.setattr(module.os, "getpgrp", lambda: 1234)
    monkeypatch.setattr(module.os, "killpg", lambda pgid, sig: signals.append((pgid, sig)))

    assert module.watch(**WATCH_KWARGS) == 1
    assert signals == [(7654, module.signal.SIGTERM)]
    receipts = sorted((module.RUNTIME / "assert_live_failures").glob("*.json"))
    assert len(receipts) == 1
    failure = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert failure["schema"] == module.FAILURE_SCHEMA
    assert failure["termination"] == "matrix_process_group_sigterm_sent"
    assert failure["score_or_decision_content_read"] is False
    assert failure["phase_c_result_artifact_modified"] is False
    assert failure["gpu_api_used"] is False
    assert failure["last_verified_signer"]["pid"] == 4321


def test_watch_refuses_to_signal_a_reused_matrix_pid_or_its_unsafe_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _configure_live_fixture(module, tmp_path, monkeypatch)
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(module, "assert_live", lambda: (_ for _ in ()).throw(module.LiveAssertionError("signer dead")))
    monkeypatch.setattr(
        module,
        "_assert_matrix_pid_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(module.MatrixProcessEnded("PID reused")),
    )
    monkeypatch.setattr(module.os, "killpg", lambda pgid, sig: signals.append((pgid, sig)))
    assert module.watch(**WATCH_KWARGS) == 1
    assert signals == []
    failure = json.loads(next((module.RUNTIME / "assert_live_failures").glob("*.json")).read_text(encoding="utf-8"))
    assert failure["termination"] == "matrix_identity_not_live_no_signal"


def test_termination_refuses_a_matrix_process_group_not_dedicated_to_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _configure_live_fixture(module, tmp_path, monkeypatch)
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(module, "_assert_matrix_pid_identity", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module.os, "getpgid", lambda _pid: 8888)
    monkeypatch.setattr(module.os, "getpgrp", lambda: 1234)
    monkeypatch.setattr(module.os, "killpg", lambda pgid, sig: signals.append((pgid, sig)))
    assert module._terminate_matrix_process_group(**WATCH_KWARGS) == (
        "matrix_process_group_not_dedicated_no_signal"
    )
    assert signals == []


def test_matrix_exit_does_not_falsely_accuse_a_live_signer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _configure_live_fixture(module, tmp_path, monkeypatch)
    signer = module.SignerIdentity(4321, 111, "a" * 64, "synthetic-host")
    monkeypatch.setattr(module, "assert_live", lambda: signer)
    monkeypatch.setattr(
        module,
        "_assert_matrix_pid_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(module.MatrixProcessEnded("worker exited")),
    )
    assert module.watch(**WATCH_KWARGS) == 0
    assert not (module.RUNTIME / "assert_live_failures").exists()


def test_matrix_identity_requires_exact_parent_starttime_and_dedicated_setsid_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _configure_live_fixture(module, tmp_path, monkeypatch)
    monkeypatch.setattr(module, "_proc_starttime", lambda pid: {7654: 222, 5432: 333}[pid])
    monkeypatch.setattr(module, "_proc_parent_pid", lambda _pid: 5432)
    monkeypatch.setattr(module.os, "getpgid", lambda _pid: 7654)
    module._assert_matrix_pid_identity(
        7654,
        222,
        parent_pid=5432,
        parent_starttime=333,
        process_group_id=7654,
    )
    monkeypatch.setattr(module.os, "getpgid", lambda _pid: 7777)
    with pytest.raises(module.MatrixLaunchContractError, match="dedicated setsid"):
        module._assert_matrix_pid_identity(
            7654,
            222,
            parent_pid=5432,
            parent_starttime=333,
            process_group_id=7654,
        )


def test_live_but_misgrouped_matrix_is_a_launch_contract_failure_not_a_signer_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _configure_live_fixture(module, tmp_path, monkeypatch)
    signer = module.SignerIdentity(4321, 111, "a" * 64, "synthetic-host")
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(module, "assert_live", lambda: signer)
    monkeypatch.setattr(
        module,
        "_assert_matrix_pid_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(module.MatrixLaunchContractError("not dedicated")),
    )
    monkeypatch.setattr(module.os, "killpg", lambda pgid, sig: signals.append((pgid, sig)))
    assert module.watch(**WATCH_KWARGS) == 2
    assert signals == []
    failure = json.loads(next((module.RUNTIME / "assert_live_failures").glob("*.json")).read_text(encoding="utf-8"))
    assert failure["termination"] == "matrix_launch_contract_invalid_no_signal"


def test_watch_exits_successfully_on_a_stable_score_free_stage_a_completion_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _configure_live_fixture(module, tmp_path, monkeypatch)
    signer = module.SignerIdentity(4321, 111, "a" * 64, "synthetic-host")
    monkeypatch.setattr(module, "assert_live", lambda: signer)
    monkeypatch.setattr(module, "_assert_matrix_pid_identity", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_stable_stage_a_completed", lambda: True)
    assert module.watch(**WATCH_KWARGS) == 0
    assert not (module.RUNTIME / "assert_live_failures").exists()


def test_stable_completion_check_reads_only_metadata_and_requires_two_equal_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    fixture = _configure_live_fixture(module, tmp_path, monkeypatch)
    completed = fixture["completed"]
    assert isinstance(completed, Path)
    completed.parent.mkdir(parents=True)
    completed.write_text('{"score_sealed": true}\n', encoding="utf-8")
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    assert module._stable_stage_a_completed() is True
    source = SCRIPT.read_text(encoding="utf-8")
    start = source.index("def _stable_stage_a_completed")
    end = source.index("\ndef _write_bytes_exclusive", start)
    assert "json.loads" not in source[start:end]


def test_cli_watch_requires_pid_and_starttime_together() -> None:
    module = _module()
    with pytest.raises(SystemExit):
        module.parse_args(["--watch"])
    with pytest.raises(SystemExit):
        module.parse_args(["--check-once", "--pid", "1"])


def test_checker_templates_and_process_identity_snapshot_are_receipt_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    _configure_live_fixture(module, tmp_path, monkeypatch)
    monkeypatch.setattr(module, "_proc_parent_pid", lambda _pid: 5432)
    monkeypatch.setattr(module.os, "getpgid", lambda _pid: 7654)
    snapshot = module.process_identity_snapshot(7654)
    assert snapshot == {
        "schema": module.PROCESS_IDENTITY_SCHEMA,
        "pid": 7654,
        "parent_pid": 5432,
        "proc_starttime_ticks": 111,
        "process_group_id": 7654,
        "proc_cmdline_sha256": hashlib.sha256(module._expected_signer_bootstrap_cmdline()).hexdigest(),
    }
    assert module.check_once_argv_template() == (
        module.WATCHDOG_PYTHON,
        "-u",
        module.WATCHDOG_SCRIPT_RELATIVE,
        "--check-once",
    )
    assert module.watch_argv_template(**WATCH_KWARGS) == (
        module.WATCHDOG_PYTHON,
        "-u",
        module.WATCHDOG_SCRIPT_RELATIVE,
        "--watch",
        "--pid",
        "7654",
        "--pid-starttime",
        "222",
        "--matrix-parent-pid",
        "5432",
        "--matrix-parent-starttime",
        "333",
        "--matrix-pgid",
        "7654",
    )


def test_watchdog_source_has_no_score_data_gpu_or_key_imports() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name.split(".")[0].lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0].lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    forbidden = {"torch", "cuda", "nvidia", "pynvml", "numpy", "cryptography", "sua_exploration"}
    assert not (imported & forbidden)
    assert "os.killpg" in source
    assert "--check-once" in source and "--watch" in source
