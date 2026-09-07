"""Test-only adversarial contract for the fresh r6d parent-death repair.

This file intentionally contains no r6d builder, bootstrap, signer, GPU, data,
endpoint, score, or subprocess execution.  It becomes active only after the
five fresh r6d production sources exist.  Its key distinction from r6c is
that a dead/drifted *supervisor parent* is evidence to stop the matrix, but is
never grounds to discard the still-valid matrix PID/starttime/setsid process
group identity needed to stop it safely.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import signal
import subprocess
import sys
from uuid import uuid4

import pytest


ROOT = Path(__file__).resolve().parents[2]
R6D = {
    "signer": ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_live_signer.py",
    "gate": ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_stage_a_continue_gate.py",
    "checker": ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_assert_live.py",
    "supervisor": ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_stage_a_supervisor.py",
    "rollover": ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_rollover.py",
}

requires_checker = pytest.mark.skipif(
    not R6D["checker"].is_file(),
    reason="r6d checker has not landed; this is a test-only prospective contract",
)
requires_full_r6d = pytest.mark.skipif(
    not all(path.is_file() for path in R6D.values()),
    reason="the full r6d source closure has not landed; this is a test-only prospective contract",
)
requires_supervisor = pytest.mark.skipif(
    not R6D["supervisor"].is_file(),
    reason="r6d supervisor has not landed; this is a test-only prospective contract",
)


def _load(path: Path, label: str):
    name = f"_r6d_parent_death_adversarial_{label}_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    start = source.find(f"def {name}(")
    assert start >= 0, f"missing required r6d function {name}"
    next_def = source.find("\ndef ", start + 1)
    return source[start:] if next_def < 0 else source[start:next_def]


def _terminator(module):
    function = getattr(module, "_terminate_matrix_process_group", None)
    assert callable(function), (
        "r6d must expose a narrow testable matrix-group terminator; this is "
        "the boundary that distinguishes parent drift from PID/PGID drift"
    )
    return function


def _terminator_kwargs(function, *, matrix_pid: int, matrix_starttime: int, matrix_pgid: int) -> dict[str, int]:
    """Support the r6c-shaped API while allowing clearer r6d parameter names."""
    aliases = {
        "matrix_pid": matrix_pid,
        "pid": matrix_pid,
        "matrix_starttime": matrix_starttime,
        "starttime": matrix_starttime,
        "matrix_pgid": matrix_pgid,
        "process_group_id": matrix_pgid,
        "pgid": matrix_pgid,
        # Deliberately stale parent facts: r6d may retain them for detection,
        # but the group-kill proof must not require them to remain current.
        "matrix_parent_pid": 42005,
        "parent_pid": 42005,
        "matrix_parent_starttime": 810,
        "parent_starttime": 810,
    }
    supplied: dict[str, int] = {}
    for parameter in inspect.signature(function).parameters.values():
        if parameter.name in aliases:
            supplied[parameter.name] = aliases[parameter.name]
        elif parameter.default is inspect.Parameter.empty:
            raise AssertionError(f"unrecognised required r6d terminator parameter: {parameter.name}")
    return supplied


def _patch_core_identity(
    monkeypatch: pytest.MonkeyPatch,
    module,
    *,
    matrix_pid: int,
    matrix_starttime: int,
    observed_pgid: int,
) -> list[tuple[int, signal.Signals]]:
    """Replace only process observation/signalling; no real process is touched."""
    calls: list[tuple[int, signal.Signals]] = []

    def starttime(pid: int) -> int:
        if pid == matrix_pid:
            return matrix_starttime
        # The receipt parent is deliberately not live/current.  A parent-drift
        # aware r6d terminator must not consume this fact as a no-signal veto.
        return 999_999

    monkeypatch.setattr(module, "_proc_starttime", starttime)
    if hasattr(module, "_proc_parent_pid"):
        monkeypatch.setattr(module, "_proc_parent_pid", lambda _pid: 1)
    monkeypatch.setattr(module.os, "getpgid", lambda _pid: observed_pgid)
    monkeypatch.setattr(module.os, "getpgrp", lambda: 771_771)
    monkeypatch.setattr(module.os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    return calls


@requires_checker
def test_r6d_checker_is_fresh_and_declares_the_parent_death_boundary() -> None:
    """No r6c runtime root or silent parent-drift/no-signal fallback may remain."""
    checker = R6D["checker"].read_text(encoding="utf-8")
    assert "_r6d" in checker, "r6d source must use fresh r6d-only roots/identifiers"
    terminate = _function_source(R6D["checker"], "_terminate_matrix_process_group")
    assert "os.killpg" in terminate and "SIGTERM" in terminate
    assert "matrix_pid" in terminate and "matrix_starttime" in terminate
    # The pre-existing r6c failure mode was: parent drift -> no signal.  r6d
    # may record parent drift, but its terminator itself cannot use it as the
    # core identity proof for killpg.
    assert "matrix_identity_not_live_no_signal" not in terminate


@requires_checker
def test_r6d_parent_drift_still_sigterms_the_verified_dedicated_matrix_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The decisive r6c regression: parent drift must not orphan the matrix."""
    module = _load(R6D["checker"], "parent_drift")
    terminate = _terminator(module)
    matrix_pid, matrix_starttime = 31003, 700
    calls = _patch_core_identity(
        monkeypatch,
        module,
        matrix_pid=matrix_pid,
        matrix_starttime=matrix_starttime,
        observed_pgid=matrix_pid,
    )

    terminate(**_terminator_kwargs(
        terminate,
        matrix_pid=matrix_pid,
        matrix_starttime=matrix_starttime,
        matrix_pgid=matrix_pid,
    ))

    assert calls == [(matrix_pid, signal.SIGTERM)], (
        "a stale/reparented supervisor must be recorded as fault evidence, not "
        "discard the still-valid matrix PID/starttime/PGID kill proof"
    )


@requires_checker
def test_r6d_pid_reuse_refuses_to_signal_even_when_the_group_number_looks_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(R6D["checker"], "pid_reuse")
    terminate = _terminator(module)
    matrix_pid, receipt_starttime = 31013, 701
    calls = _patch_core_identity(
        monkeypatch,
        module,
        matrix_pid=matrix_pid,
        matrix_starttime=receipt_starttime + 1,
        observed_pgid=matrix_pid,
    )

    terminate(**_terminator_kwargs(
        terminate,
        matrix_pid=matrix_pid,
        matrix_starttime=receipt_starttime,
        matrix_pgid=matrix_pid,
    ))

    assert calls == [], "PID/starttime drift must fail closed rather than signal a reused PID group"


@requires_checker
def test_r6d_pgid_drift_refuses_to_signal_even_when_pid_starttime_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(R6D["checker"], "pgid_drift")
    terminate = _terminator(module)
    matrix_pid, matrix_starttime = 31023, 702
    calls = _patch_core_identity(
        monkeypatch,
        module,
        matrix_pid=matrix_pid,
        matrix_starttime=matrix_starttime,
        observed_pgid=31024,
    )

    terminate(**_terminator_kwargs(
        terminate,
        matrix_pid=matrix_pid,
        matrix_starttime=matrix_starttime,
        matrix_pgid=matrix_pid,
    ))

    assert calls == [], "PGID drift/shared group must fail closed rather than signal unrelated processes"


@requires_checker
def test_r6d_signer_death_causes_a_verified_matrix_group_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Signer loss must kill the matrix even when no parent probe is possible."""
    module = _load(R6D["checker"], "signer_death")
    matrix_pid, matrix_starttime = 31033, 703
    calls = _patch_core_identity(
        monkeypatch,
        module,
        matrix_pid=matrix_pid,
        matrix_starttime=matrix_starttime,
        observed_pgid=matrix_pid,
    )
    recorded: list[dict[str, object]] = []
    monkeypatch.setattr(
        module,
        "assert_live",
        lambda: (_ for _ in ()).throw(module.LiveAssertionError("synthetic signer death")),
    )
    monkeypatch.setattr(module, "_record_failure", lambda **payload: recorded.append(payload))

    rc = module.watch(
        matrix_pid=matrix_pid,
        matrix_starttime=matrix_starttime,
        matrix_parent_pid=42005,
        matrix_parent_starttime=810,
        matrix_pgid=matrix_pid,
    )

    assert rc == 1
    assert calls == [(matrix_pid, signal.SIGTERM)]
    assert len(recorded) == 1
    assert recorded[0]["termination"] == "matrix_pid_starttime_and_dedicated_pgid_verified_sigterm_sent"


@requires_checker
def test_r6d_watch_parent_drift_kills_the_verified_matrix_instead_of_only_recording_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(R6D["checker"], "watch_parent_drift")
    matrix_pid, matrix_starttime = 31043, 704
    calls = _patch_core_identity(
        monkeypatch,
        module,
        matrix_pid=matrix_pid,
        matrix_starttime=matrix_starttime,
        observed_pgid=matrix_pid,
    )
    recorded: list[dict[str, object]] = []
    signer = module.SignerIdentity(51, 52, "0" * 64, "synthetic")
    monkeypatch.setattr(module, "assert_live", lambda: signer)
    monkeypatch.setattr(module, "_matrix_contract_status", lambda **_kwargs: "parent_drift")
    monkeypatch.setattr(module, "_record_failure", lambda **payload: recorded.append(payload))

    rc = module.watch(
        matrix_pid=matrix_pid,
        matrix_starttime=matrix_starttime,
        matrix_parent_pid=42005,
        matrix_parent_starttime=810,
        matrix_pgid=matrix_pid,
    )

    assert rc == 2
    assert calls == [(matrix_pid, signal.SIGTERM)]
    assert len(recorded) == 1 and recorded[0]["reason"] == "matrix_parent_drift"


@requires_checker
def test_r6d_synthetic_sigkill_of_supervisor_reparents_matrix_but_still_kills_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model the post-SIGKILL view seen by the independent watchdog.

    No process is actually signalled: the former supervisor's PID/starttime is
    made unobservable and the matrix's observed parent is PID 1.  The matrix
    itself retains the recorded starttime and dedicated PGID, so r6d must TERM
    that group instead of copying r6c's ``parent drift -> no signal`` bug.
    """
    module = _load(R6D["checker"], "synthetic_sigkill_supervisor")
    matrix_pid, matrix_starttime = 31053, 705
    expected_supervisor_pid, expected_supervisor_starttime = 42015, 811
    calls = _patch_core_identity(
        monkeypatch,
        module,
        matrix_pid=matrix_pid,
        matrix_starttime=matrix_starttime,
        observed_pgid=matrix_pid,
    )
    recorded: list[dict[str, object]] = []
    signer = module.SignerIdentity(51, 52, "0" * 64, "synthetic")
    monkeypatch.setattr(module, "assert_live", lambda: signer)
    monkeypatch.setattr(module, "_proc_parent_pid", lambda pid: 1 if pid == matrix_pid else 0)

    def starttime_after_sigkill(pid: int) -> int:
        if pid == matrix_pid:
            return matrix_starttime
        if pid == expected_supervisor_pid:
            raise module.LiveAssertionError("synthetic dead supervisor")
        return 999_999

    monkeypatch.setattr(module, "_proc_starttime", starttime_after_sigkill)
    monkeypatch.setattr(module, "_record_failure", lambda **payload: recorded.append(payload))

    rc = module.watch(
        matrix_pid=matrix_pid,
        matrix_starttime=matrix_starttime,
        matrix_parent_pid=expected_supervisor_pid,
        matrix_parent_starttime=expected_supervisor_starttime,
        matrix_pgid=matrix_pid,
    )

    assert rc == 2
    assert calls == [(matrix_pid, signal.SIGTERM)]
    assert len(recorded) == 1 and recorded[0]["reason"] == "matrix_parent_drift"


@requires_supervisor
def test_r6d_pdeathsig_preexec_arms_before_the_post_arm_parent_recheck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No real fork: prove the child hook asks Linux to own the post-SIGKILL stop."""
    module = _load(R6D["supervisor"], "pdeath_arm")
    calls: list[tuple[object, ...]] = []

    class _Libc:
        def prctl(self, *args):
            calls.append(args)
            return 0

    expected_parent = 91_001
    monkeypatch.setattr(module.ctypes, "CDLL", lambda *_args, **_kwargs: _Libc())
    monkeypatch.setattr(module.os, "getppid", lambda: expected_parent)
    hook = module._matrix_pdeathsig_preexec(expected_parent)
    hook()

    assert calls == [
        (module.PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0)
    ], "matrix child must arm PDEATHSIG(SIGTERM) before it can exec GPU code"


@requires_supervisor
def test_r6d_pdeathsig_preexec_exits_if_parent_died_in_fork_to_arm_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The race after fork but before `prctl` is closed by the immediate PPID check."""
    module = _load(R6D["supervisor"], "pdeath_parent_dead")
    calls: list[tuple[object, ...]] = []

    class _Libc:
        def prctl(self, *args):
            calls.append(args)
            return 0

    class _ChildExited(Exception):
        pass

    expected_parent = 91_011
    monkeypatch.setattr(module.ctypes, "CDLL", lambda *_args, **_kwargs: _Libc())
    monkeypatch.setattr(module.os, "getppid", lambda: 1)  # post-SIGKILL reparented child
    monkeypatch.setattr(module.os, "_exit", lambda code: (_ for _ in ()).throw(_ChildExited(code)))

    with pytest.raises(_ChildExited) as excinfo:
        module._matrix_pdeathsig_preexec(expected_parent)()

    assert excinfo.value.args == (127,)
    assert calls == [(module.PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0)]


@requires_supervisor
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-specific")
def test_r6d_real_pdeathsig_synthetic_supervisor_sigkill_terminates_matrix_child() -> None:
    """Exercise the real kernel boundary with disposable Python-only children.

    A fresh synthetic supervisor imports only the r6d *preexec helper*, starts
    a zero-GPU child, waits until the child has armed its SIGTERM handler, then
    SIGKILLs itself.  The child has a short alarm fallback so even a broken
    kernel primitive cannot leak it.  This neither invokes r6d ``main`` nor
    creates a plan, signer, data loader, cell, endpoint, or GPU worker.
    """
    child_code = r'''
import os, signal, sys, time
ready_fd = int(sys.argv[1])
def finish(label, code):
    os.write(1, (label + "\n").encode("ascii"))
    os._exit(code)
signal.signal(signal.SIGTERM, lambda _s, _f: finish("PDEATHSIG", 0))
signal.signal(signal.SIGALRM, lambda _s, _f: finish("FALLBACK_TIMEOUT", 2))
signal.alarm(4)
os.write(ready_fd, b"A")
os.close(ready_fd)
while True:
    time.sleep(0.1)
'''
    supervisor_code = f'''
import os, subprocess, sys
sys.path.insert(0, {str(ROOT)!r})
from sua_exploration.scripts.m2_native_post33_phase_c_v4_r6d_stage_a_supervisor import _matrix_pdeathsig_preexec
ready_r, ready_w = os.pipe()
os.set_inheritable(ready_w, True)
child = subprocess.Popen(
    [sys.executable, "-c", {child_code!r}, str(ready_w)],
    pass_fds=(ready_w,), close_fds=True,
    preexec_fn=_matrix_pdeathsig_preexec(os.getpid()),
)
os.close(ready_w)
if os.read(ready_r, 1) != b"A":
    os._exit(88)
os.close(ready_r)
os.kill(os.getpid(), 9)
'''
    result = subprocess.run(
        [sys.executable, "-c", supervisor_code], cwd=ROOT,
        text=True, capture_output=True, timeout=10, check=False,
    )
    assert result.returncode == -int(signal.SIGKILL)
    assert "PDEATHSIG" in result.stdout
    assert "FALLBACK_TIMEOUT" not in result.stdout


@requires_supervisor
def test_r6d_supervisor_constructs_exact_matrix_argv_and_rejects_raw_row_command_substitution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load(R6D["supervisor"], "argv_and_row")
    data_root = tmp_path / "SPINT-main/data/000953"
    data_root.mkdir(parents=True)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "R6D_ROOT", tmp_path / "cells")
    monkeypatch.setattr(module, "MATRIX", tmp_path / "fixed_matrix.py")
    monkeypatch.setattr(module, "PYTHON", "/synthetic/fixed-python")
    artifacts = {
        name: tmp_path / f"{name}.json"
        for name in ("shard_manifest", "portable", "program", "cost_supplement", "authorization", "signature")
    }
    for path in artifacts.values():
        path.write_text("{}\n", encoding="utf-8")
    argv = module._matrix_argv(artifacts)
    assert argv == [
        "/synthetic/fixed-python", "-u", str(tmp_path / "fixed_matrix.py"), "--execute",
        "--cell-root", str((tmp_path / "cells").resolve()),
        "--workspace-root", str(tmp_path.resolve()),
        "--data-root", str(data_root.resolve()),
        "--shard-manifest", str(artifacts["shard_manifest"].resolve()),
        "--portable-manifest", str(artifacts["portable"].resolve()),
        "--program-receipt", str(artifacts["program"].resolve()),
        "--cost-supplement", str(artifacts["cost_supplement"].resolve()),
        "--authorization", str(artifacts["authorization"].resolve()),
        "--authorization-signature", str(artifacts["signature"].resolve()),
    ]

    # Reach the row-shape guard without opening/signing any real r6d object.
    contract = {
        "controller": {}, "continue_gate": {}, "supervisor": {}, "checker": {}, "ready": {}, "program": {},
        "check_once_argv_template": list(module.watchdog.check_once_argv_template()),
        "watch_argv_template": list(module.watchdog.watch_argv_template()),
        "matrix_must_start_new_session": True,
        "matrix_process_group_must_equal_matrix_pid": True,
        "matrix_pr_set_pdeathsig": "SIGTERM",
        "matrix_pdeathsig_parent_recheck": True,
        "watcher_must_not_use_pdeathsig": True,
        "one_independent_watcher_per_gpu_shard": True,
        "execution_start_receipt_required": True,
    }
    poisoned_row = {
        "gpu_id": 0, "supervisor_argv": [], "authorization": {}, "authorization_signature_path": "",
        "shard_manifest": {}, "program": {}, "portable": {}, "cost_supplement": {},
        "fold_allowlist": [], "cuda_visible_devices": "synthetic", "matrix_argv": ["/attacker/command"],
    }
    launch_payload = {
        "schema": "m2_post33_phase_c_v4_r6d_stage_a_launch_commands_v1",
        "status": "PREPARED_NOT_EXECUTED",
        "cell_root_must_be_absent_until_matrix_execution": str((tmp_path / "cells").resolve()),
        "source_sealed_supervision_contract": contract,
        "commands": {"gpu0": poisoned_row, "gpu1": dict(poisoned_row, gpu_id=1)},
    }
    monkeypatch.setattr(module, "_json", lambda _path: launch_payload)
    monkeypatch.setattr(module, "_metadata", lambda _value, path, _label: path)
    monkeypatch.setattr(module.program_module, "validate_phase_c_program_receipt", lambda _path: {"source_map": []})
    monkeypatch.setattr(module, "_source_is_sealed", lambda *_args: None)
    with pytest.raises(PermissionError, match="row exact-key mismatch"):
        module._verify_contract_and_row(0)


class _FakeProcess:
    """No-child substitute for supervisor receipt-collision cleanup tests."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.rc: int | None = None
        self.wait_calls: list[int | None] = []
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self) -> int | None:
        return self.rc

    def wait(self, timeout: int | None = None) -> int:
        self.wait_calls.append(timeout)
        return -int(signal.SIGTERM) if self.rc is None else self.rc

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.rc = -int(signal.SIGTERM)

    def kill(self) -> None:
        self.kill_calls += 1
        self.rc = -int(signal.SIGKILL)


@requires_supervisor
def test_r6d_started_receipt_collision_reaps_matrix_stops_watcher_and_writes_one_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A post-spawn O_EXCL collision may never leave a GPU worker behind."""
    module = _load(R6D["supervisor"], "started_collision")
    r6d = tmp_path / "r6d"
    (r6d / "launch").mkdir(parents=True)
    ready = tmp_path / "ready.json"
    ready.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(module, "R6D", r6d)
    monkeypatch.setattr(module, "READY", ready)
    monkeypatch.setattr(module, "LAUNCH", r6d / "launch/stage_a_commands_r6d.json")
    monkeypatch.setattr(module, "ASSERT_LIVE", tmp_path / "assert_live.py")
    monkeypatch.setattr(module, "_check_once", lambda: None)
    monkeypatch.setattr(module, "_verify_contract_and_row", lambda _gpu: ({}, {"cuda_visible_devices": "GPU-synthetic"}, {}))
    monkeypatch.setattr(module, "_verify_stage_a_ack", lambda *_args: None)
    monkeypatch.setattr(module, "_matrix_argv", lambda _paths: ["/synthetic/matrix"])
    monkeypatch.setattr(module, "_utc_now", lambda: "2026-08-05T00:00:00+00:00")
    monkeypatch.setattr(module, "_proc_starttime", lambda pid: 800_000 + pid)
    monkeypatch.setattr(module, "_proc_cmdline_sha256", lambda pid: f"cmd-{pid}")
    monkeypatch.setattr(module.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(module.os, "getpgrp", lambda: 777_777)
    monkeypatch.setattr(module.watchdog, "process_identity_snapshot", lambda pid: {"pid": pid})
    monkeypatch.setattr(module, "file_metadata", lambda path: {"canonical_path": str(Path(path).resolve()), "size_bytes": 0, "sha256": "a" * 64})

    writes: list[tuple[Path, dict[str, object]]] = []
    def fake_write(path: Path, payload: dict[str, object]) -> Path:
        path = Path(path)
        if path.name.endswith("_execution_started.json"):
            path.write_text("{\"competing\": true}\n", encoding="utf-8")
            raise FileExistsError(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, default=str) + "\n", encoding="utf-8")
        writes.append((path, payload))
        return path
    monkeypatch.setattr(module, "write_json_exclusive", fake_write)

    matrix, watcher = _FakeProcess(51_001), _FakeProcess(51_002)
    launches: list[dict[str, object]] = []
    outcomes = [matrix, watcher]
    def fake_popen(*_args, **kwargs):
        launches.append(kwargs)
        return outcomes.pop(0)
    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    def fake_kill(pid: int, *, expected_starttime: int, sig: signal.Signals = signal.SIGTERM) -> bool:
        assert pid == matrix.pid and expected_starttime == 800_000 + matrix.pid
        matrix.rc = -int(sig)
        return True
    monkeypatch.setattr(module, "_kill_matrix_group", fake_kill)

    with pytest.raises(FileExistsError):
        module.supervise(0)

    assert len(launches) == 2
    assert callable(launches[0].get("preexec_fn")), "matrix must receive the child PDEATHSIG hook"
    assert "preexec_fn" not in launches[1], "watcher must survive supervisor death without PDEATHSIG"
    assert matrix.poll() is not None and matrix.wait_calls
    assert watcher.terminate_calls == 1 and watcher.poll() is not None and watcher.wait_calls
    terminals = [path.name for path, _payload in writes if "execution_failed" in path.name or "execution_completed" in path.name]
    assert terminals == ["stage_a_gpu0_execution_failed.json"]


@requires_full_r6d
def test_r6d_rollover_and_supervisor_keep_all_execution_paths_fixed_and_score_free() -> None:
    """A parent-death repair must not widen caller-controlled execution authority.

    The fixed native-MUA root is deliberately part of the sealed matrix command.
    What must remain impossible is a caller or launch-row substitution of that
    root (or of the matrix command itself).
    """
    supervisor = R6D["supervisor"].read_text(encoding="utf-8")
    rollover = R6D["rollover"].read_text(encoding="utf-8")
    main = _function_source(R6D["supervisor"], "main")
    row_verifier = _function_source(R6D["supervisor"], "_verify_contract_and_row")
    matrix_argv = _function_source(R6D["supervisor"], "_matrix_argv")
    launch_start = rollover.index("    launch = {")
    launch_body = rollover[launch_start:rollover.index("    launch_meta =", launch_start)]

    # The supervisor command line accepts one selector only.  Matrix flags
    # such as --authorization are internal, fixed arguments and are therefore
    # intentionally checked in _matrix_argv instead of globally forbidden.
    forbidden_caller_cli = ("--command", "--argv", "--authorization", "--cell-root", "--output", "--data-root")
    assert not [token for token in forbidden_caller_cli if token in main]
    assert 'parser.add_argument("--gpu-id"' in main
    assert "start_new_session=True" in supervisor

    # No launch row may carry an arbitrary data root or a raw matrix command.
    assert '"matrix_argv"' not in row_verifier
    assert '"data_root"' not in row_verifier
    assert '"matrix_argv"' not in launch_body
    assert '"data_root"' not in launch_body
    assert '"supervisor_argv"' in launch_body

    # The only data-root construction is the sealed, fixed native-MUA path.
    assert 'ROOT / "SPINT-main/data/000953"' in matrix_argv
    assert 'paths["data_root"]' not in matrix_argv
    for source in (supervisor, rollover):
        assert "pynwb" not in source
        assert "torch" not in source
