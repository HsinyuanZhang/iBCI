"""Adversarial, no-GPU checks for the r6c sealed Stage-A supervisor.

Every behavioral test replaces spawning, waits, signals, and receipt writes
with disposable fakes.  It never calls a real r6c plan, signer, matrix worker,
watcher, CUDA runtime, score payload, or formal endpoint.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
from uuid import uuid4

import pytest

from sua_exploration.mc_maze import m2_native_post33_program_v4 as program


ROOT = Path(__file__).resolve().parents[2]
SUPERVISOR = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_stage_a_supervisor.py"
ROLLOVER = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_rollover.py"


def _load(path: Path, label: str):
    name = f"_r6c_supervisor_adversarial_{label}_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1, f"missing/ambiguous {name}"
    return ast.get_source_segment(source, matches[0]) or ""


def test_supervisor_is_source_sealed_with_the_signer_gate_and_checker() -> None:
    expected = {
        "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_live_signer.py",
        "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_stage_a_continue_gate.py",
        "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_assert_live.py",
        "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_stage_a_supervisor.py",
    }
    roots = set(program.PROGRAM_RUNTIME_ROOTS["open_and_finalize_clis"])
    assert expected <= roots
    assert expected <= set(program.PROGRAM_EXPECTED_CLOSURE)


def test_plan_is_the_last_signer_trigger_after_full_prelaunch_artifacts() -> None:
    """The controller must never see PLAN before launch/proof metadata exists."""
    source = ROLLOVER.read_text(encoding="utf-8")
    launch_write = source.index("write_json_exclusive(launch_path, launch)")
    proof_write = source.index("write_json_exclusive(proof_path, proof)")
    plan_publish = source.index("plan_path = _atomic_publish_json")
    assert launch_write < proof_write < plan_publish


def test_prelaunch_contract_binds_every_runtime_source_and_exact_watch_templates() -> None:
    """A hash for only the checker is not a source-sealed supervision path."""
    source = ROLLOVER.read_text(encoding="utf-8")
    required_bindings = (
        "source_sealed_supervision_contract",
        "file_metadata(CONTROLLER)",
        "file_metadata(GATE)",
        "file_metadata(ASSERT_LIVE)",
        "file_metadata(SUPERVISOR)",
        "check_once_argv_template",
        "watch_argv_template",
        "matrix_must_start_new_session",
        "matrix_process_group_must_equal_matrix_pid",
        "one_independent_watcher_per_gpu_shard",
    )
    missing = [token for token in required_bindings if token not in source]
    assert not missing, f"launch contract misses required sealed bindings: {missing}"


def test_builder_watch_template_uses_the_checkers_exact_placeholder_vocabulary() -> None:
    """A one-token template mismatch makes a receipt look sealed but unusable."""
    checker = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_assert_live.py"
    checker_source = checker.read_text(encoding="utf-8")
    builder_source = ROLLOVER.read_text(encoding="utf-8")
    expected_placeholder = "<MATRIX_PROCESS_GROUP_ID>"
    assert expected_placeholder in checker_source
    assert expected_placeholder in builder_source
    assert "<MATRIX_PGID>" not in builder_source


def test_builder_quarantine_binding_points_to_an_existing_post_remediation_receipt() -> None:
    """The r6c build must not fail after a clean, documented fixture quarantine."""
    source = ROLLOVER.read_text(encoding="utf-8")
    assert "R6C_RUNTIME_QUARANTINE" in source
    expected = ROOT / (
        "sua_exploration/results/m2_native_post33_phase_c_v4_r6c_synthetic_watchdog_incident_20260805/"
        "post_quarantine_verification.json"
    )
    assert expected.is_file()
    assert str(expected.relative_to(ROOT)) in source


def test_cli_accepts_only_gpu_id_and_no_untrusted_command_or_path_overrides() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")
    assert 'add_argument("--gpu-id"' in source
    assert "choices=(0, 1)" in source or "choices=[0, 1]" in source
    assert 'add_argument("--shard"' not in source
    forbidden = (
        "--command",
        "--argv",
        "--authorization",
        "--signature",
        "--launch",
        "--cell-root",
        "--output",
    )
    assert not [argument for argument in forbidden if argument in source]


def test_supervisor_declares_a_dedicated_group_kill_path_and_all_terminal_receipts() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")
    required = (
        "def _kill_matrix_group",
        "os.killpg",
        "expected_starttime",
        "execution_started",
        "execution_completed",
        "execution_failed",
        "stdout",
        "stderr",
        "start_new_session=True",
    )
    missing = [token for token in required if token not in source]
    assert not missing, f"supervisor lacks required crash-safe controls: {missing}"


def test_supervise_interface_is_gpu_id_and_cleanup_helper_refuses_pid_reuse(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public surface is intentionally tiny; cleanup must recheck /proc."""
    module = _load(SUPERVISOR, "interface")
    assert hasattr(module, "_kill_matrix_group")
    # No signal may be emitted if PID starttime changed between watcher failure
    # and supervisor cleanup.
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(module, "_proc_starttime", lambda _pid: 999)
    monkeypatch.setattr(module.os, "getpgid", lambda _pid: 7001)
    monkeypatch.setattr(module.os, "killpg", lambda pgid, sig: signals.append((pgid, sig)))
    assert module._kill_matrix_group(7001, expected_starttime=111) is False
    assert signals == []


def test_source_orders_reservation_and_started_receipt_before_any_matrix_spawn() -> None:
    """Duplicate same-shard launches must lose before creating a GPU worker."""
    source = _function_source(SUPERVISOR, "supervise")
    reservation = min(
        position for token in ("_reserve", "reservation", "execution_started")
        if (position := source.find(token)) >= 0
    )
    spawn = source.find("subprocess.Popen(")
    assert spawn >= 0
    assert reservation < spawn, "an O_EXCL shard reservation must precede matrix spawn"


def test_source_has_fail_closed_cleanup_for_watcher_spawn_receipt_and_early_exit() -> None:
    """Every post-spawn failure path needs PG kill; waiting for training is unsafe."""
    source = _function_source(SUPERVISOR, "supervise")
    # This static gate intentionally demands explicit exception/finally cleanup
    # rather than trusting normal ``matrix.wait(); watcher.wait()`` sequencing.
    assert "try:" in source and "finally:" in source
    assert source.count("_kill_matrix_group(") >= 3, (
        "need independent cleanup paths for watcher-spawn, receipt-write, and watcher-nonzero failures"
    )
    watcher_wait = source.find("watcher.wait()")
    matrix_wait = source.find("matrix.wait()")
    assert watcher_wait >= 0 and matrix_wait >= 0
    assert watcher_wait < matrix_wait, "watcher must be observed before blocking on matrix completion"


class _FakeProcess:
    """Deterministic child-process model used only by the no-GPU failure tests."""

    def __init__(
        self,
        pid: int,
        *,
        polls: tuple[int | None, ...] = (),
        waits: tuple[object, ...] = (),
        default_wait: int | None = None,
    ) -> None:
        self.pid = pid
        self._polls = list(polls)
        self._last_poll: int | None = None
        self._waits = list(waits)
        self._default_wait = default_wait
        self._forced_rc: int | None = None
        self.poll_calls = 0
        self.wait_calls: list[int | None] = []
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self) -> int | None:
        self.poll_calls += 1
        if self._forced_rc is not None:
            return self._forced_rc
        if self._polls:
            self._last_poll = self._polls.pop(0)
        # A non-None poll result means the OS child has already exited.  A
        # subsequent defensive group cleanup may still target descendants, but
        # it cannot rewrite this child's return code.
        if self._last_poll is not None:
            self._forced_rc = self._last_poll
        return self._last_poll

    def wait(self, timeout: int | None = None) -> int | None:
        self.wait_calls.append(timeout)
        if self._waits:
            result = self._waits.pop(0)
            if isinstance(result, BaseException):
                raise result
            assert result is None or isinstance(result, int)
            self._forced_rc = result
            return result
        if self._forced_rc is not None:
            return self._forced_rc
        return self._default_wait

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._forced_rc = -int(signal.SIGTERM)

    def kill(self) -> None:
        self.kill_calls += 1
        self._forced_rc = -int(signal.SIGKILL)

    def force_exit(self, rc: int) -> None:
        self._forced_rc = rc


def _behavioral_harness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    popen_outcomes: list[_FakeProcess | BaseException],
    *,
    reservation_collision: bool = False,
    started_collision: bool = False,
    kill_behavior=None,
):
    """Load a fresh supervisor and replace every external effect with a fake.

    The harness deliberately preserves its important ordering: receipt writes
    are real exclusive files under ``tmp_path``; only child creation, /proc,
    group signals, and signer validation are substituted.  Thus a test catches
    both a process leak and an attempt to publish two terminal receipts.
    """
    module = _load(SUPERVISOR, f"behavior_{uuid4().hex}")
    r6c = tmp_path / "r6c"
    launch_dir = r6c / "launch"
    launch_dir.mkdir(parents=True)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    ready = runtime / "live_signer_ready.json"
    ready.write_text("{}\n", encoding="utf-8")
    launch = launch_dir / "stage_a_commands_r6c.json"
    launch.write_text("{}\n", encoding="utf-8")
    checker = tmp_path / "assert_live.py"
    checker.write_text("# synthetic checker\n", encoding="utf-8")

    monkeypatch.setattr(module, "R6C", r6c)
    monkeypatch.setattr(module, "R6C_ROOT", tmp_path / "cells")
    monkeypatch.setattr(module, "RUNTIME", runtime)
    monkeypatch.setattr(module, "READY", ready)
    monkeypatch.setattr(module, "LAUNCH", launch)
    monkeypatch.setattr(module, "ASSERT_LIVE", checker)
    monkeypatch.setattr(module, "_check_once", lambda: None)
    monkeypatch.setattr(module, "_stage_a_ack_for", lambda _row: tmp_path / "ack.json")
    monkeypatch.setattr(
        module,
        "_launch_row",
        lambda _shard: (
            {"schema": "synthetic_launch"},
            {"matrix_argv": ["/synthetic/matrix"], "cuda_visible_devices": "GPU-synthetic"},
        ),
    )
    monkeypatch.setattr(module, "_utc_now", lambda: "2026-08-05T00:00:00+00:00")
    monkeypatch.setattr(module, "_proc_starttime", lambda pid: 900_000 + pid)
    monkeypatch.setattr(module, "_proc_cmdline_sha256", lambda pid: f"cmdline-{pid}")
    monkeypatch.setattr(
        module,
        "_identity_snapshot",
        lambda pid: {"pid": pid, "proc_starttime_ticks": 900_000 + pid, "process_group_id": pid},
    )
    monkeypatch.setattr(module.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(module.os, "getpgrp", lambda: 777_777)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)

    def fake_metadata(path: Path) -> dict[str, object]:
        file = Path(path)
        return {
            "canonical_path": str(file.resolve()),
            "size_bytes": file.stat().st_size if file.is_file() else 0,
            "sha256": "f" * 64,
        }

    monkeypatch.setattr(module, "file_metadata", fake_metadata)
    writes: list[tuple[Path, dict[str, object]]] = []

    def fake_write(path: Path, payload: dict[str, object]) -> Path:
        file = Path(path)
        if reservation_collision and file.name.endswith("_execution_reserved.json"):
            file.write_text('{"preexisting": true}\n', encoding="utf-8")
            raise FileExistsError(file)
        if started_collision and file.name.endswith("_execution_started.json"):
            file.write_text('{"preexisting": true}\n', encoding="utf-8")
            raise FileExistsError(file)
        if file.exists():
            raise FileExistsError(file)
        file.parent.mkdir(parents=True, exist_ok=True)
        # Persist an inspectable body so ``execution_started.is_file()`` follows
        # the production control flow.  Values are not score/data payloads.
        file.write_text(json.dumps(payload, sort_keys=True, default=str) + "\n", encoding="utf-8")
        writes.append((file, payload))
        return file

    monkeypatch.setattr(module, "write_json_exclusive", fake_write)
    popen_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    processes_by_pid = {
        outcome.pid: outcome for outcome in popen_outcomes if isinstance(outcome, _FakeProcess)
    }

    def fake_popen(*args, **kwargs):
        popen_calls.append((args, kwargs))
        if not popen_outcomes:
            raise AssertionError("supervisor requested an unexpected extra child")
        outcome = popen_outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    kill_events: list[tuple[int, int, signal.Signals]] = []

    def fake_kill(pid: int, *, expected_starttime: int, sig: signal.Signals = signal.SIGTERM) -> bool:
        kill_events.append((pid, expected_starttime, sig))
        process = processes_by_pid.get(pid)
        if kill_behavior is not None:
            return bool(kill_behavior(process, sig))
        # Model ``killpg`` faithfully enough for the relevant safety claim:
        # it may clean up a still-running group, but cannot mutate an already
        # reaped/terminated matrix child's historical return code.
        if process is not None and process.poll() is None:
            process.force_exit(-int(sig))
        return process is not None

    monkeypatch.setattr(module, "_kill_matrix_group", fake_kill)
    return module, writes, popen_calls, kill_events, processes_by_pid


def _terminal_writes(writes: list[tuple[Path, dict[str, object]]]) -> list[tuple[Path, dict[str, object]]]:
    return [
        write
        for write in writes
        if write[0].name.endswith("_execution_completed.json")
        or write[0].name.endswith("_execution_failed.json")
    ]


def test_behavior_watcher_spawn_exception_reaps_matrix_and_publishes_one_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    matrix = _FakeProcess(4101)
    module, writes, calls, kills, _ = _behavioral_harness(
        monkeypatch, tmp_path, [matrix, OSError("synthetic watcher spawn failure")]
    )

    with pytest.raises(OSError, match="watcher spawn failure"):
        module.supervise("gpu0")

    assert len(calls) == 2
    assert kills and kills[0][0] == matrix.pid
    assert matrix.poll() is not None and matrix.wait_calls, "matrix must be killed and reaped before return"
    terminals = _terminal_writes(writes)
    assert [path.name for path, _payload in terminals] == ["stage_a_gpu0_execution_failed.json"]
    assert terminals[0][1]["state"] == "FAILED"


def test_behavior_started_receipt_collision_kills_both_children_and_has_one_terminal_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    matrix = _FakeProcess(4201)
    watcher = _FakeProcess(4202)
    module, writes, calls, kills, _ = _behavioral_harness(
        monkeypatch, tmp_path, [matrix, watcher], started_collision=True
    )

    with pytest.raises(FileExistsError):
        module.supervise("gpu0")

    assert len(calls) == 2
    assert kills and all(pid == matrix.pid for pid, _start, _sig in kills)
    assert matrix.poll() is not None and matrix.wait_calls
    assert watcher.terminate_calls == 1 and watcher.poll() is not None and watcher.wait_calls
    terminals = _terminal_writes(writes)
    assert [path.name for path, _payload in terminals] == ["stage_a_gpu0_execution_failed.json"]
    assert terminals[0][1]["execution_started"] is not None


def test_behavior_watcher_early_nonzero_fails_closed_without_an_orphan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    matrix = _FakeProcess(4301, polls=(None,))
    watcher = _FakeProcess(4302, polls=(17,), waits=(17,))
    module, writes, _calls, kills, _ = _behavioral_harness(monkeypatch, tmp_path, [matrix, watcher])

    with pytest.raises(RuntimeError, match="watchdog failed closed"):
        module.supervise("gpu0")

    assert any(pid == matrix.pid and sig == signal.SIGTERM for pid, _start, sig in kills)
    assert matrix.poll() is not None and matrix.wait_calls
    assert watcher.poll() == 17
    terminals = _terminal_writes(writes)
    assert [path.name for path, _payload in terminals] == ["stage_a_gpu0_execution_failed.json"]


def test_behavior_matrix_early_nonzero_is_reaped_and_never_gets_a_completion_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    matrix = _FakeProcess(4401, polls=(9,), waits=(9,))
    watcher = _FakeProcess(4402, polls=(None,), waits=(0, 0))
    module, writes, _calls, _kills, _ = _behavioral_harness(monkeypatch, tmp_path, [matrix, watcher])

    with pytest.raises(RuntimeError, match="matrix failed"):
        module.supervise("gpu0")

    assert matrix.poll() == 9 and matrix.wait_calls
    assert watcher.wait_calls[:2] == [10, None]
    terminals = _terminal_writes(writes)
    assert [path.name for path, _payload in terminals] == ["stage_a_gpu0_execution_failed.json"]


def test_behavior_duplicate_reservation_prevents_any_child_spawn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module, writes, calls, kills, _ = _behavioral_harness(
        monkeypatch, tmp_path, [], reservation_collision=True
    )

    with pytest.raises(FileExistsError):
        module.supervise("gpu0")

    assert calls == [] and kills == []
    # This is a rejected launch, not an execution: emitting a terminal receipt
    # here would falsely claim ownership of the competing reservation.
    assert _terminal_writes(writes) == []


def test_behavior_healthy_watcher_with_hung_matrix_escalates_term_then_kill_and_fails_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    matrix = _FakeProcess(
        4501,
        polls=(None,),
        waits=(
            subprocess.TimeoutExpired("matrix", 60),
            subprocess.TimeoutExpired("matrix", 30),
            -int(signal.SIGKILL),
        ),
    )
    watcher = _FakeProcess(4502, polls=(0,), waits=(0,))

    def only_kill_ends_matrix(process: _FakeProcess | None, sig: signal.Signals) -> bool:
        assert process is matrix
        if sig == signal.SIGKILL:
            process.force_exit(-int(signal.SIGKILL))
        return True

    module, writes, _calls, kills, _ = _behavioral_harness(
        monkeypatch, tmp_path, [matrix, watcher], kill_behavior=only_kill_ends_matrix
    )

    with pytest.raises(RuntimeError, match="did not exit after healthy watchdog"):
        module.supervise("gpu0")

    signals = [sig for _pid, _start, sig in kills]
    assert signal.SIGTERM in signals and signal.SIGKILL in signals
    assert matrix.poll() == -int(signal.SIGKILL) and matrix.wait_calls[:3] == [60, 30, 10]
    terminals = _terminal_writes(writes)
    assert [path.name for path, _payload in terminals] == ["stage_a_gpu0_execution_failed.json"]
