"""Pure-temp adversarial tests for an early *successful* r6d watcher exit.

These tests exercise only the supervisor's process-control branch using fake
``Popen`` objects.  They do not build r6d, read a dataset or score, start a
subprocess, touch a GPU, or create any production receipt/cell/runtime path.

The safety distinction is important: watcher exit ``0`` is only a completion
signal after the public, stable Stage-A completion marker exists.  Otherwise a
still-live matrix must be stopped just as it would be for a non-zero watcher
failure.  A matrix that exits in the small watcher/matrix polling race remains
the ordinary matrix-return-code path, rather than being killed for a missing
marker.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any
from uuid import uuid4

import pytest


ROOT = Path(__file__).resolve().parents[2]
SUPERVISOR = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_stage_a_supervisor.py"

pytestmark = pytest.mark.skipif(
    not SUPERVISOR.is_file(), reason="r6d supervisor has not landed"
)


def _load(label: str):
    name = f"_r6d_early_watcher_adversarial_{label}_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, SUPERVISOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _SequencedProcess:
    """A no-child Popen surrogate with an explicit poll race script."""

    def __init__(self, pid: int, *, polls: list[int | None], wait_results: list[int] | None = None) -> None:
        self.pid = pid
        self._polls = list(polls)
        self._wait_results = list(wait_results or [])
        self.rc: int | None = None
        self.poll_calls = 0
        self.wait_calls: list[int | None] = []
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self) -> int | None:
        self.poll_calls += 1
        if self._polls:
            result = self._polls.pop(0)
            if result is not None:
                self.rc = result
            return result
        return self.rc

    def wait(self, timeout: int | None = None) -> int:
        self.wait_calls.append(timeout)
        if self._wait_results:
            self.rc = self._wait_results.pop(0)
            return self.rc
        if self.rc is None:
            raise AssertionError(f"unexpected unbounded live-process wait timeout={timeout!r}")
        return self.rc

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.rc = -int(signal.SIGTERM)

    def kill(self) -> None:
        self.kill_calls += 1
        self.rc = -int(signal.SIGKILL)


class _EarlyWatcherCase:
    """A completely temporary supervisor environment and launch recording."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        matrix: _SequencedProcess,
        watcher: _SequencedProcess,
        completion_marker_stable: bool,
    ) -> None:
        self.m = _load("case")
        self.matrix, self.watcher = matrix, watcher
        self.marker_calls = 0
        self.kills: list[tuple[int, int, signal.Signals]] = []
        self.writes: list[tuple[Path, dict[str, Any]]] = []
        m = self.m
        r6d = tmp_path / "r6d"
        ready = tmp_path / "ready.json"
        ready.write_text("{}\n", encoding="utf-8")
        monkeypatch.setattr(m, "R6D", r6d)
        monkeypatch.setattr(m, "READY", ready)
        monkeypatch.setattr(m, "LAUNCH", r6d / "launch/stage_a_commands_r6d.json")
        monkeypatch.setattr(m, "ASSERT_LIVE", tmp_path / "assert_live.py")
        monkeypatch.setattr(m, "_check_once", lambda: None)
        monkeypatch.setattr(
            m, "_verify_contract_and_row",
            lambda _gpu: ({}, {"cuda_visible_devices": "GPU-synthetic"}, {}),
        )
        monkeypatch.setattr(m, "_verify_stage_a_ack", lambda *_args: None)
        monkeypatch.setattr(m, "_matrix_argv", lambda _paths: ["/synthetic/matrix"])
        monkeypatch.setattr(m, "_utc_now", lambda: "2026-08-05T00:00:00+00:00")
        monkeypatch.setattr(m, "_proc_starttime", lambda pid: 900_000 + pid)
        monkeypatch.setattr(m, "_proc_cmdline_sha256", lambda pid: f"cmd-{pid}")
        monkeypatch.setattr(m.os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(m.os, "getpgrp", lambda: 777_777)
        monkeypatch.setattr(m.watchdog, "process_identity_snapshot", lambda pid: {"pid": pid})
        monkeypatch.setattr(
            m,
            "file_metadata",
            lambda path: {
                "canonical_path": str(Path(path).resolve()), "size_bytes": 0, "sha256": "a" * 64,
            },
        )

        # This is intentionally a public checker predicate: the supervisor
        # cannot infer successful completion from a clean watcher exit alone.
        def marker() -> bool:
            self.marker_calls += 1
            return completion_marker_stable

        monkeypatch.setattr(m.watchdog, "stage_a_completion_marker_is_stable", marker)

        def fake_write(path: Path, payload: dict[str, Any]) -> Path:
            path = Path(path)
            if path.exists():
                raise FileExistsError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, sort_keys=True, default=str) + "\n", encoding="utf-8")
            self.writes.append((path, payload))
            return path

        monkeypatch.setattr(m, "write_json_exclusive", fake_write)
        launches = [matrix, watcher]
        monkeypatch.setattr(m.subprocess, "Popen", lambda *_args, **_kw: launches.pop(0))

        def fake_kill(pid: int, *, expected_starttime: int, sig: signal.Signals = signal.SIGTERM) -> bool:
            self.kills.append((pid, expected_starttime, sig))
            assert pid == matrix.pid
            matrix.rc = -int(sig)
            return True

        monkeypatch.setattr(m, "_kill_matrix_group", fake_kill)

    @property
    def terminals(self) -> list[Path]:
        return [
            path for path, _payload in self.writes
            if path.name.endswith("_execution_completed.json") or path.name.endswith("_execution_failed.json")
        ]


def test_r6d_early_clean_watcher_exit_without_stable_marker_kills_reaps_and_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``watcher_rc == 0`` alone cannot orphan a still-live matrix worker."""
    matrix = _SequencedProcess(61_001, polls=[None, None])
    watcher = _SequencedProcess(61_002, polls=[0])
    case = _EarlyWatcherCase(
        monkeypatch, tmp_path, matrix=matrix, watcher=watcher, completion_marker_stable=False
    )

    with pytest.raises(RuntimeError, match="watcher.*completion|completion.*watcher|marker"):
        case.m.supervise(0)

    assert matrix.poll_calls >= 2, "the clean-exit path must close the poll race before deciding"
    assert case.marker_calls == 1
    assert case.kills == [(matrix.pid, 900_000 + matrix.pid, signal.SIGTERM)]
    assert matrix.wait_calls and 60 not in matrix.wait_calls, "missing marker must not receive healthy grace"
    # It has already cleanly exited, so the supervisor may observe it with
    # ``poll`` rather than unnecessarily calling ``wait``/``terminate``.
    assert watcher.terminate_calls == 0 and watcher.poll() == 0
    assert [path.name for path in case.terminals] == ["stage_a_gpu0_execution_failed.json"]


def test_r6d_early_clean_watcher_exit_with_stable_marker_allows_only_bounded_matrix_grace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stable marker authorizes a short completion grace, not an unbounded wait."""
    matrix = _SequencedProcess(62_001, polls=[None, None], wait_results=[0])
    watcher = _SequencedProcess(62_002, polls=[0])
    case = _EarlyWatcherCase(
        monkeypatch, tmp_path, matrix=matrix, watcher=watcher, completion_marker_stable=True
    )

    assert case.m.supervise(0) == 0

    assert matrix.poll_calls >= 2
    assert case.marker_calls == 1
    assert not case.kills
    bounded = [timeout for timeout in matrix.wait_calls if timeout is not None]
    assert bounded and all(0 < timeout <= 60 for timeout in bounded)
    assert [path.name for path in case.terminals] == ["stage_a_gpu0_execution_completed.json"]


@pytest.mark.parametrize("matrix_returncode", (0, 7))
def test_r6d_clean_watcher_exit_then_second_matrix_poll_uses_matrix_returncode_not_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, matrix_returncode: int
) -> None:
    """The matrix may exit between the first pair of polls and the marker check."""
    matrix = _SequencedProcess(63_001, polls=[None, matrix_returncode])
    watcher = _SequencedProcess(63_002, polls=[0])
    case = _EarlyWatcherCase(
        monkeypatch, tmp_path, matrix=matrix, watcher=watcher, completion_marker_stable=False
    )

    if matrix_returncode == 0:
        assert case.m.supervise(0) == 0
        expected = "stage_a_gpu0_execution_completed.json"
    else:
        with pytest.raises(RuntimeError, match="Stage-A matrix failed"):
            case.m.supervise(0)
        expected = "stage_a_gpu0_execution_failed.json"

    assert matrix.poll_calls >= 2
    assert case.marker_calls == 0, "second-poll completion is not a marker-dependent failure"
    assert not case.kills
    assert [path.name for path in case.terminals] == [expected]
