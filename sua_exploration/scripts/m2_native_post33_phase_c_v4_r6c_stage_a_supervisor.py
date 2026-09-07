#!/usr/bin/env python3
"""Source-sealed r6c Stage-A matrix supervisor with signer-liveness watchdog.

The caller selects only ``gpu0`` or ``gpu1``.  All executable paths,
capabilities, CUDA UUIDs, and watcher templates come from the immutable r6c
launch receipt; no caller-provided matrix argv or capability path is accepted.
This program has no private key and never opens a score or data payload.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
R6C = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r6c"
R6C_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cells_20260805_r6c"
RUNTIME = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_r6c_live_signer_20260805"
READY = RUNTIME / "live_signer_ready.json"
ASSERT_LIVE = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_assert_live.py"
WATCHDOG_PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"
WATCHDOG_SCRIPT_RELATIVE = "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_assert_live.py"
LAUNCH = R6C / "launch/stage_a_commands_r6c.json"
POLL_SECONDS = 0.25

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (  # noqa: E402
    file_metadata,
    require_canonical_regular_file,
    write_json_exclusive,
)


def _proc_starttime(pid: int) -> int:
    tail = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].strip().split()
    if len(tail) <= 19:
        raise RuntimeError("/proc stat lacks starttime")
    return int(tail[19])


def _proc_cmdline_sha256(pid: int) -> str:
    return hashlib.sha256(Path(f"/proc/{pid}/cmdline").read_bytes()).hexdigest()


def _identity_snapshot(pid: int) -> dict[str, Any]:
    return {
        "schema": "m2_post33_phase_c_v4_r6c_process_identity_v1",
        "pid": pid,
        "parent_pid": os.getppid() if pid == os.getpid() else int(
            Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].strip().split()[1]
        ),
        "proc_starttime_ticks": _proc_starttime(pid),
        "process_group_id": os.getpgid(pid),
        "proc_cmdline_sha256": _proc_cmdline_sha256(pid),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _kill_matrix_group(
    pid: int,
    *,
    expected_starttime: int,
    sig: signal.Signals = signal.SIGTERM,
) -> bool:
    """Signal only the still-live, PID-bound dedicated matrix process group.

    The repeated starttime and PGID checks make cleanup fail closed if Linux
    has reused the matrix PID or the child escaped the sealed ``setsid``
    contract.  A missing/already-finished process needs no signal.
    """
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        return False
    try:
        if _proc_starttime(pid) != expected_starttime:
            return False
        pgid = os.getpgid(pid)
    except (FileNotFoundError, ProcessLookupError, PermissionError, RuntimeError, OSError):
        return False
    if pgid != pid or pgid <= 1 or pgid == os.getpgrp():
        return False
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return False
    return True


def _terminal_payload(
    *,
    state: str,
    shard: str,
    reservation: Path,
    started: Path,
    matrix_pid: int | None,
    matrix_starttime: int | None,
    matrix_rc: int | None,
    watcher_pid: int | None,
    watcher_rc: int | None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": f"m2_post33_phase_c_v4_r6c_stage_a_execution_{state}_v1",
        "state": state.upper(),
        "recorded_at_utc": _utc_now(),
        "shard": shard,
        "reservation": file_metadata(reservation),
        "execution_started": file_metadata(started) if started.is_file() else None,
        "matrix_pid": matrix_pid,
        "matrix_proc_starttime_ticks": matrix_starttime,
        "matrix_returncode": matrix_rc,
        "watcher_pid": watcher_pid,
        "watcher_returncode": watcher_rc,
        "score_or_decision_content_read": False,
        "gpu_api_used_by_supervisor": False,
    }
    if error is not None:
        payload["error"] = {"type": type(error).__name__, "message": str(error)}
    return payload


def _json(path: Path) -> dict[str, Any]:
    return json.loads(require_canonical_regular_file(path).read_text(encoding="utf-8"))


def _check_once() -> None:
    result = subprocess.run(
        [WATCHDOG_PYTHON, "-u", WATCHDOG_SCRIPT_RELATIVE, "--check-once"], cwd=ROOT,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise PermissionError("r6c live signer check-once failed before matrix launch")


def _launch_row(shard: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _json(LAUNCH)
    if payload.get("schema") != "m2_post33_phase_c_v4_r6c_stage_a_launch_commands_v1":
        raise PermissionError("r6c launch receipt schema mismatch")
    if payload.get("status") != "PREPARED_NOT_EXECUTED":
        raise PermissionError("r6c launch receipt is not in pre-execution state")
    if payload.get("cell_root_must_be_absent_until_matrix_execution") != str(R6C_ROOT.resolve()):
        raise PermissionError("r6c launch receipt cell-root substitution")
    commands = payload.get("commands")
    if not isinstance(commands, Mapping) or set(commands) != {"gpu0", "gpu1"} or shard not in commands:
        raise PermissionError("r6c launch shard scope mismatch")
    row = commands[shard]
    if not isinstance(row, Mapping) or set(row) != {
        "matrix_argv", "fold_allowlist", "cuda_visible_devices", "authorization", "authorization_signature_path",
    }:
        raise PermissionError("r6c launch row exact key mismatch")
    argv = row.get("matrix_argv")
    if not isinstance(argv, list) or not all(isinstance(token, str) for token in argv):
        raise PermissionError("r6c matrix argv malformed")
    expected_prefix = [
        "/home/xinyuan/miniconda3/envs/spint/bin/python", "-u",
        str(ROOT / "sua_exploration/scripts/run_m2_native_post33_phase_c_v4_matrix.py"), "--execute",
    ]
    if argv[:4] != expected_prefix:
        raise PermissionError("r6c matrix argv frontdoor substitution")
    _metadata(row.get("authorization"), Path(str(row["authorization"]["canonical_path"])), "authorization")
    signature_path = Path(str(row.get("authorization_signature_path", ""))).resolve()
    expected_signature = Path(str(row["authorization"]["canonical_path"])).with_suffix(".sig").resolve()
    if signature_path != expected_signature or not signature_path.is_file():
        raise PermissionError("r6c matrix authorization signature path mismatch/absent")
    return payload, dict(row)


def _metadata(metadata: Any, expected: Path, label: str) -> Path:
    file = require_canonical_regular_file(expected)
    if metadata != file_metadata(file):
        raise PermissionError(f"r6c supervisor {label} metadata mismatch")
    return file


def _stage_a_ack_for(row: Mapping[str, Any]) -> Path:
    ack = R6C / "signer_acks/stage_a_capabilities_signed.json"
    payload = _json(ack)
    if payload.get("schema") != "m2_post33_phase_c_v4_r6c_stage_a_capabilities_signed_v1":
        raise PermissionError("r6c Stage-A signer ack schema mismatch")
    if payload.get("ready") != file_metadata(READY):
        raise PermissionError("r6c Stage-A signer ack ready substitution")
    signature = Path(str(row["authorization_signature_path"])).resolve(strict=True)
    found = False
    for evidence in payload.get("stage_a", {}).values():
        if isinstance(evidence, Mapping) and evidence.get("signature") == file_metadata(signature):
            found = True
    if not found:
        raise PermissionError("r6c Stage-A signer ack does not bind this signature")
    return ack


def _watch_argv(matrix_pid: int, matrix_starttime: int, parent_pid: int, parent_starttime: int, matrix_pgid: int) -> list[str]:
    return [
        WATCHDOG_PYTHON, "-u", WATCHDOG_SCRIPT_RELATIVE, "--watch", "--pid", str(matrix_pid),
        "--pid-starttime", str(matrix_starttime), "--matrix-parent-pid", str(parent_pid),
        "--matrix-parent-starttime", str(parent_starttime), "--matrix-pgid", str(matrix_pgid),
    ]


def supervise(shard: str) -> int:
    launch, row = _launch_row(shard)
    _check_once()
    _stage_a_ack_for(row)
    matrix_env = {"PATH": os.environ.get("PATH", ""), "CUDA_VISIBLE_DEVICES": str(row["cuda_visible_devices"])}
    if not matrix_env["PATH"]:
        raise RuntimeError("PATH unavailable for r6c matrix supervisor")
    launch_dir = R6C / "launch"
    reservation = launch_dir / f"stage_a_{shard}_execution_reserved.json"
    execution_started = launch_dir / f"stage_a_{shard}_execution_started.json"
    execution_completed = launch_dir / f"stage_a_{shard}_execution_completed.json"
    execution_failed = launch_dir / f"stage_a_{shard}_execution_failed.json"
    stdout = launch_dir / f"stage_a_{shard}_matrix.stdout.log"
    stderr = launch_dir / f"stage_a_{shard}_matrix.stderr.log"

    # This O_EXCL reservation is intentionally before Popen: concurrent or
    # accidental duplicate supervisors cannot create a second GPU worker.
    write_json_exclusive(reservation, {
        "schema": "m2_post33_phase_c_v4_r6c_stage_a_execution_reservation_v1",
        "state": "RESERVED_NOT_STARTED",
        "reserved_at_utc": _utc_now(),
        "shard": shard,
        "launch_receipt": file_metadata(LAUNCH),
        "ready": file_metadata(READY),
        "supervisor_source": file_metadata(Path(__file__).resolve()),
        "supervisor_identity": _identity_snapshot(os.getpid()),
        "stdout": str(stdout.resolve()),
        "stderr": str(stderr.resolve()),
        "score_or_decision_content_read": False,
        "gpu_api_used_by_supervisor": False,
    })

    matrix: subprocess.Popen[Any] | None = None
    watcher: subprocess.Popen[Any] | None = None
    matrix_pid: int | None = None
    matrix_starttime: int | None = None
    matrix_rc: int | None = None
    watcher_rc: int | None = None
    stdout_handle = None
    stderr_handle = None
    try:
        stdout_handle = stdout.open("xb")
        stderr_handle = stderr.open("xb")
        matrix = subprocess.Popen(
            list(row["matrix_argv"]), cwd=ROOT, env=matrix_env,
            stdin=subprocess.DEVNULL, stdout=stdout_handle, stderr=stderr_handle,
            start_new_session=True,
        )
        matrix_pid = matrix.pid
        matrix_starttime = _proc_starttime(matrix_pid)
        matrix_pgid = os.getpgid(matrix_pid)
        if matrix_pgid != matrix_pid:
            _kill_matrix_group(matrix_pid, expected_starttime=matrix_starttime)
            raise PermissionError("r6c matrix did not enter a dedicated session/process group")
        parent_pid = os.getpid()
        parent_starttime = _proc_starttime(parent_pid)
        watcher_argv = _watch_argv(matrix_pid, matrix_starttime, parent_pid, parent_starttime, matrix_pgid)
        watcher = subprocess.Popen(
            watcher_argv, cwd=ROOT,
            env={"PATH": matrix_env["PATH"], "PYTHONPATH": str(ROOT), "PYTHONNOUSERSITE": "1"},
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        watcher_starttime = _proc_starttime(watcher.pid)
        write_json_exclusive(execution_started, {
            "schema": "m2_post33_phase_c_v4_r6c_stage_a_execution_start_v1",
            "state": "STARTED",
            "started_at_utc": _utc_now(),
            "shard": shard,
            "reservation": file_metadata(reservation),
            "launch_receipt": file_metadata(LAUNCH),
            "ready": file_metadata(READY),
            "assert_live_source": file_metadata(ASSERT_LIVE),
            "supervisor_source": file_metadata(Path(__file__).resolve()),
            "stdout": str(stdout.resolve()),
            "stderr": str(stderr.resolve()),
            "matrix": {
                "pid": matrix_pid, "proc_starttime_ticks": matrix_starttime, "pgid": matrix_pgid,
                "parent_pid": parent_pid, "parent_proc_starttime_ticks": parent_starttime,
                "argv": list(row["matrix_argv"]), "cuda_visible_devices": row["cuda_visible_devices"],
            },
            "matrix_identity": _identity_snapshot(matrix_pid),
            "watcher": {
                "pid": watcher.pid, "proc_starttime_ticks": watcher_starttime,
                "proc_cmdline_sha256": _proc_cmdline_sha256(watcher.pid), "argv": watcher_argv,
            },
            "watcher_identity": _identity_snapshot(watcher.pid),
            "initial_assert_live_check_once": "PASS",
            "watcher_required_to_enforce_signer_liveness_until_stage_a_completed": True,
            "score_or_decision_content_read": False,
            "gpu_api_used_by_supervisor": False,
        })

        # Observe both children concurrently.  A failed watcher is a fail-closed
        # event and terminates only the revalidated dedicated matrix group.
        while True:
            watcher_rc = watcher.poll()
            matrix_rc = matrix.poll()
            if watcher_rc is not None:
                if watcher_rc != 0 and matrix_rc is None:
                    _kill_matrix_group(matrix_pid, expected_starttime=matrix_starttime)
                break
            if matrix_rc is not None:
                watcher_rc = watcher.wait(timeout=10)
                break
            time.sleep(POLL_SECONDS)

        # The explicit order is safety-relevant: consume watchdog status before
        # any potentially blocking wait for the training matrix.
        watcher_rc = watcher.wait()
        if watcher_rc != 0:
            _kill_matrix_group(matrix_pid, expected_starttime=matrix_starttime)
            try:
                matrix_rc = matrix.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _kill_matrix_group(
                    matrix_pid, expected_starttime=matrix_starttime, sig=signal.SIGKILL
                )
                matrix_rc = matrix.wait(timeout=10)
            raise RuntimeError(f"r6c signer watchdog failed closed for {shard}: exit={watcher_rc}")
        try:
            # The healthy watcher returns only after the matrix ended or the
            # stable Stage-A completion marker appeared.  A bounded grace
            # period prevents a post-completion worker hang from blocking the
            # supervisor indefinitely.
            matrix_rc = matrix.wait() if matrix.poll() is not None else matrix.wait(timeout=60)
        except subprocess.TimeoutExpired as exc:
            _kill_matrix_group(matrix_pid, expected_starttime=matrix_starttime)
            try:
                matrix_rc = matrix.wait(timeout=30)
            except subprocess.TimeoutExpired:
                _kill_matrix_group(
                    matrix_pid, expected_starttime=matrix_starttime, sig=signal.SIGKILL
                )
                matrix_rc = matrix.wait(timeout=10)
            raise RuntimeError(f"r6c Stage-A matrix did not exit after healthy watchdog for {shard}") from exc
        if matrix_rc != 0:
            raise RuntimeError(f"r6c Stage-A matrix failed for {shard}: exit={matrix_rc}")
        write_json_exclusive(execution_completed, _terminal_payload(
            state="completed", shard=shard, reservation=reservation, started=execution_started,
            matrix_pid=matrix_pid, matrix_starttime=matrix_starttime, matrix_rc=matrix_rc,
            watcher_pid=watcher.pid, watcher_rc=watcher_rc,
        ))
        return matrix_rc
    except BaseException as exc:
        if matrix_pid is not None and matrix_starttime is not None:
            _kill_matrix_group(matrix_pid, expected_starttime=matrix_starttime)
        if matrix is not None:
            try:
                matrix_rc = matrix.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if matrix_pid is not None and matrix_starttime is not None:
                    _kill_matrix_group(matrix_pid, expected_starttime=matrix_starttime, sig=signal.SIGKILL)
                matrix_rc = matrix.wait(timeout=10)
        if watcher is not None:
            watcher_rc = watcher.poll()
            if watcher_rc is None:
                watcher.terminate()
                try:
                    watcher_rc = watcher.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    watcher.kill()
                    watcher_rc = watcher.wait(timeout=10)
        write_json_exclusive(execution_failed, _terminal_payload(
            state="failed", shard=shard, reservation=reservation, started=execution_started,
            matrix_pid=matrix_pid, matrix_starttime=matrix_starttime, matrix_rc=matrix_rc,
            watcher_pid=watcher.pid if watcher is not None else None, watcher_rc=watcher_rc,
            error=exc,
        ))
        raise
    finally:
        # Final idempotent safety net for exceptions in watcher spawn, receipt
        # publication, terminal-receipt construction, or the monitor itself.
        if matrix is not None and matrix.poll() is None and matrix_pid is not None and matrix_starttime is not None:
            _kill_matrix_group(matrix_pid, expected_starttime=matrix_starttime)
        if watcher is not None and watcher.poll() is None:
            watcher.terminate()
        if stdout_handle is not None:
            stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    raise SystemExit(supervise(f"gpu{args.gpu_id}"))


if __name__ == "__main__":
    main()
