#!/usr/bin/env python3
"""Score-free r6d signer liveness assertion and fail-closed matrix watchdog.

This control-plane program reads only the r6d ready receipt, public/controller
source metadata, Linux ``/proc`` identity fields, and the score-free Stage-A
completion marker.  It never imports a data loader, scorer, torch, or CUDA.

The important r6d repair is intentional: parent PID/starttime is still audited,
but it is *not* a prerequisite for killing a matrix whose own PID/starttime and
dedicated ``setsid`` process group remain exact.  Therefore a crashed or
SIGKILLed supervisor cannot reparent a still-valid matrix into an orphaned GPU
worker.  PID reuse or a process-group drift still refuses any signal.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
from typing import Any, Mapping
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
R6D_RECEIPTS = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r6d"
R6D_CELL_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cells_20260805_r6d"
RUNTIME = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_r6d_live_signer_20260805"
READY = RUNTIME / "live_signer_ready.json"
PUBLIC_KEY = ROOT / "sua_exploration/configs/m2_native_post33_phase_c_v4_r6d_root_ed25519_public.pem"
SIGNER_SOURCE = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_live_signer.py"
PROTOCOL_ID = "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1"
PHASE_ID = "PHASE_C_V4"
STAGE_A_COMPLETED = R6D_CELL_ROOT / PROTOCOL_ID / PHASE_ID / "stage_a/status.completed.json"
READY_SCHEMA = "m2_post33_phase_c_v4_r6d_live_signer_ready_v1"
FAILURE_SCHEMA = "m2_post33_phase_c_v4_r6d_assert_live_failure_v1"
PROCESS_IDENTITY_SCHEMA = "m2_post33_phase_c_v4_r6d_process_identity_v1"
POLL_SECONDS = 0.5
WATCHDOG_PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"
WATCHDOG_SCRIPT_RELATIVE = "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_assert_live.py"
EXPECTED_SIGNER_BOOTSTRAP_ARGV = (
    WATCHDOG_PYTHON,
    "-u",
    "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_live_signer.py",
    "--bootstrap",
)


class LiveAssertionError(PermissionError):
    """A non-score control-plane liveness failure."""


class MatrixProcessEnded(LiveAssertionError):
    """The supervised matrix process is no longer the recorded process."""


class MatrixLaunchContractError(LiveAssertionError):
    """A live matrix no longer satisfies its dedicated-process contract."""


@dataclass(frozen=True)
class SignerIdentity:
    pid: int
    proc_starttime_ticks: int
    proc_cmdline_sha256: str
    host_id: str


def _canonical_regular_file(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise LiveAssertionError(f"symlink is forbidden at control boundary: {raw}")
    try:
        canonical = raw.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise LiveAssertionError(f"control file is absent: {raw}") from exc
    if not canonical.is_file() or str(raw) != str(canonical):
        raise LiveAssertionError(f"control file is not canonical regular file: {raw}")
    return canonical


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_metadata(path: str | Path) -> dict[str, Any]:
    canonical = _canonical_regular_file(path)
    return {
        "canonical_path": str(canonical),
        "size_bytes": canonical.stat().st_size,
        "sha256": _sha256_file(canonical),
    }


def _proc_tail(pid: int) -> list[str]:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        raise LiveAssertionError("PID must be an integer greater than one")
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        return text.rsplit(")", 1)[1].strip().split()
    except (FileNotFoundError, OSError, IndexError) as exc:
        raise LiveAssertionError("bound process is no longer observable") from exc


def _proc_starttime(pid: int) -> int:
    try:
        return int(_proc_tail(pid)[19])
    except (IndexError, ValueError) as exc:
        raise LiveAssertionError("/proc stat has no valid process starttime") from exc


def _proc_parent_pid(pid: int) -> int:
    try:
        return int(_proc_tail(pid)[1])
    except (IndexError, ValueError) as exc:
        raise LiveAssertionError("/proc stat has no valid parent PID") from exc


def _proc_cmdline_bytes(pid: int) -> bytes:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, OSError) as exc:
        raise LiveAssertionError("bound process command line is no longer observable") from exc


def _proc_cmdline_sha256(pid: int) -> str:
    return hashlib.sha256(_proc_cmdline_bytes(pid)).hexdigest()


def process_identity_snapshot(pid: int) -> dict[str, Any]:
    """Return a score-free PID/starttime/parent/PGID identity snapshot."""
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError as exc:
        raise LiveAssertionError("bound process group is no longer observable") from exc
    return {
        "schema": PROCESS_IDENTITY_SCHEMA,
        "pid": pid,
        "parent_pid": _proc_parent_pid(pid),
        "proc_starttime_ticks": _proc_starttime(pid),
        "process_group_id": pgid,
        "proc_cmdline_sha256": _proc_cmdline_sha256(pid),
    }


def check_once_argv_template() -> tuple[str, ...]:
    return (WATCHDOG_PYTHON, "-u", WATCHDOG_SCRIPT_RELATIVE, "--check-once")


def watch_argv_template(
    *,
    matrix_pid: int | str = "<MATRIX_PID>",
    matrix_starttime: int | str = "<MATRIX_STARTTIME_TICKS>",
    matrix_parent_pid: int | str = "<MATRIX_PARENT_PID>",
    matrix_parent_starttime: int | str = "<MATRIX_PARENT_STARTTIME_TICKS>",
    matrix_pgid: int | str = "<MATRIX_PROCESS_GROUP_ID>",
) -> tuple[str, ...]:
    return (
        WATCHDOG_PYTHON, "-u", WATCHDOG_SCRIPT_RELATIVE, "--watch",
        "--pid", str(matrix_pid), "--pid-starttime", str(matrix_starttime),
        "--matrix-parent-pid", str(matrix_parent_pid),
        "--matrix-parent-starttime", str(matrix_parent_starttime),
        "--matrix-pgid", str(matrix_pgid),
    )


def _expected_signer_bootstrap_cmdline() -> bytes:
    return b"\0".join(part.encode("utf-8") for part in EXPECTED_SIGNER_BOOTSTRAP_ARGV) + b"\0"


def _parse_ready() -> Mapping[str, Any]:
    try:
        payload = json.loads(_canonical_regular_file(READY).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError) as exc:
        raise LiveAssertionError("ready receipt is incomplete or malformed") from exc
    if not isinstance(payload, Mapping):
        raise LiveAssertionError("ready receipt must be a mapping")
    return payload


def assert_live() -> SignerIdentity:
    """Validate the exact currently-live in-memory r6d signer process."""
    ready = _parse_ready()
    required_true = (
        ready.get("private_key_not_serialized") is True
        and ready.get("private_key_retained_live_in_memory") is True
        and ready.get("private_key_not_supplied_via_argv_environment_file_or_ipc") is True
    )
    if (
        ready.get("schema") != READY_SCHEMA
        or ready.get("status") != "LIVE_PRIVATE_KEY_IN_MEMORY_ONLY"
        or ready.get("host_id") != socket.gethostname()
        or ready.get("intended_receipt_root") != str(R6D_RECEIPTS.resolve())
        or ready.get("intended_cell_root") != str(R6D_CELL_ROOT.resolve())
        or not required_true
        or ready.get("controller_source") != _file_metadata(SIGNER_SOURCE)
        or ready.get("public_key") != _file_metadata(PUBLIC_KEY)
    ):
        raise LiveAssertionError("r6d ready receipt binding mismatch")
    pid, starttime, command = (
        ready.get("pid"), ready.get("proc_starttime_ticks"), ready.get("proc_cmdline_sha256")
    )
    if (
        not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1
        or not isinstance(starttime, int) or isinstance(starttime, bool) or starttime < 0
        or not isinstance(command, str) or len(command) != 64
        or any(character not in "0123456789abcdef" for character in command)
    ):
        raise LiveAssertionError("ready process identity has invalid type/format")
    if _proc_starttime(pid) != starttime:
        raise LiveAssertionError("ready signer PID/starttime identity drift")
    cmdline = _proc_cmdline_bytes(pid)
    if hashlib.sha256(cmdline).hexdigest() != command or cmdline != _expected_signer_bootstrap_cmdline():
        raise LiveAssertionError("ready signer command identity drift")
    return SignerIdentity(pid, starttime, command, str(ready["host_id"]))


def _matrix_identity_for_kill(
    *, matrix_pid: int, matrix_starttime: int, matrix_pgid: int
) -> bool:
    """The minimal, safe predicate for a signal target independent of parent."""
    if (
        not isinstance(matrix_pid, int) or isinstance(matrix_pid, bool) or matrix_pid <= 1
        or not isinstance(matrix_starttime, int) or isinstance(matrix_starttime, bool)
        or not isinstance(matrix_pgid, int) or isinstance(matrix_pgid, bool)
    ):
        return False
    try:
        return (
            _proc_starttime(matrix_pid) == matrix_starttime
            and os.getpgid(matrix_pid) == matrix_pgid == matrix_pid
            and matrix_pgid != os.getpgrp()
        )
    except (LiveAssertionError, ProcessLookupError, OSError):
        return False


def _terminate_matrix_process_group(
    *, matrix_pid: int, matrix_starttime: int, matrix_pgid: int
) -> str:
    """TERM a still-identical dedicated group even if its parent was reaped."""
    if not _matrix_identity_for_kill(
        matrix_pid=matrix_pid, matrix_starttime=matrix_starttime, matrix_pgid=matrix_pgid
    ):
        return "matrix_pid_starttime_or_dedicated_pgid_mismatch_no_signal"
    try:
        os.killpg(matrix_pgid, signal.SIGTERM)
    except ProcessLookupError:
        return "matrix_process_group_missing_no_signal"
    return "matrix_pid_starttime_and_dedicated_pgid_verified_sigterm_sent"


def _matrix_contract_status(
    *,
    matrix_pid: int,
    matrix_starttime: int,
    matrix_parent_pid: int,
    matrix_parent_starttime: int,
    matrix_pgid: int,
) -> str:
    """Return healthy, ended, parent_drift, or group_drift without opening data."""
    if not _matrix_identity_for_kill(
        matrix_pid=matrix_pid, matrix_starttime=matrix_starttime, matrix_pgid=matrix_pgid
    ):
        try:
            _proc_starttime(matrix_pid)
        except LiveAssertionError:
            return "ended"
        return "group_or_starttime_drift"
    try:
        if _proc_parent_pid(matrix_pid) != matrix_parent_pid:
            return "parent_drift"
        if _proc_starttime(matrix_parent_pid) != matrix_parent_starttime:
            return "parent_drift"
    except LiveAssertionError:
        return "parent_drift"
    return "healthy"


def stage_a_completion_marker_is_stable() -> bool:
    """Return whether the score-free Stage-A terminal marker is byte-stable.

    This is deliberately public because the supervisor must independently
    confirm the only legitimate reason a healthy watcher may exit while its
    matrix child is still observable.  It opens no result payload or score.
    """
    try:
        first = _file_metadata(STAGE_A_COMPLETED)
        time.sleep(POLL_SECONDS)
        return first == _file_metadata(STAGE_A_COMPLETED)
    except LiveAssertionError:
        return False


def _write_bytes_exclusive(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(fd, data[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)
    return path


def _record_failure(
    *, reason: str, matrix_pid: int, matrix_starttime: int, matrix_pgid: int,
    signer: SignerIdentity | None, termination: str,
) -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    payload: dict[str, Any] = {
        "schema": FAILURE_SCHEMA,
        "recorded_at_utc": now,
        "failure_reason": reason,
        "matrix_pid": matrix_pid,
        "matrix_proc_starttime_ticks": matrix_starttime,
        "matrix_pgid": matrix_pgid,
        "termination": termination,
        "score_or_decision_content_read": False,
        "phase_c_result_artifact_modified": False,
        "gpu_api_used": False,
        "private_key_read_or_written": False,
    }
    if signer is not None:
        payload["last_verified_signer"] = signer.__dict__
    filename = f"failure_{now.replace(':', '').replace('+00:00', 'Z')}_{os.getpid()}_{uuid4().hex}.json"
    _write_bytes_exclusive(
        RUNTIME / "assert_live_failures" / filename,
        (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )


def check_once() -> int:
    try:
        assert_live()
    except LiveAssertionError:
        return 1
    return 0


def watch(
    *, matrix_pid: int, matrix_starttime: int, matrix_parent_pid: int,
    matrix_parent_starttime: int, matrix_pgid: int,
) -> int:
    """Watch signer and worker; parent drift safely kills a verified worker."""
    last_signer: SignerIdentity | None = None
    while True:
        try:
            last_signer = assert_live()
        except LiveAssertionError as exc:
            termination = _terminate_matrix_process_group(
                matrix_pid=matrix_pid, matrix_starttime=matrix_starttime, matrix_pgid=matrix_pgid
            )
            _record_failure(
                reason=f"{type(exc).__name__}: {exc}", matrix_pid=matrix_pid,
                matrix_starttime=matrix_starttime, matrix_pgid=matrix_pgid,
                signer=last_signer, termination=termination,
            )
            return 1
        matrix_status = _matrix_contract_status(
            matrix_pid=matrix_pid, matrix_starttime=matrix_starttime,
            matrix_parent_pid=matrix_parent_pid,
            matrix_parent_starttime=matrix_parent_starttime, matrix_pgid=matrix_pgid,
        )
        if matrix_status == "ended":
            return 0
        if matrix_status != "healthy":
            termination = _terminate_matrix_process_group(
                matrix_pid=matrix_pid, matrix_starttime=matrix_starttime, matrix_pgid=matrix_pgid
            )
            _record_failure(
                reason=f"matrix_{matrix_status}", matrix_pid=matrix_pid,
                matrix_starttime=matrix_starttime, matrix_pgid=matrix_pgid,
                signer=last_signer, termination=termination,
            )
            return 2
        if stage_a_completion_marker_is_stable():
            return 0
        time.sleep(POLL_SECONDS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-once", action="store_true")
    modes.add_argument("--watch", action="store_true")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--pid-starttime", type=int)
    parser.add_argument("--matrix-parent-pid", type=int)
    parser.add_argument("--matrix-parent-starttime", type=int)
    parser.add_argument("--matrix-pgid", type=int)
    args = parser.parse_args(argv)
    watch_values = (
        args.pid, args.pid_starttime, args.matrix_parent_pid,
        args.matrix_parent_starttime, args.matrix_pgid,
    )
    if args.watch and any(value is None for value in watch_values):
        parser.error("--watch requires all matrix identity fields")
    if args.check_once and any(value is not None for value in watch_values):
        parser.error("--check-once accepts no matrix identity fields")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check_once:
        return check_once()
    return watch(
        matrix_pid=args.pid, matrix_starttime=args.pid_starttime,
        matrix_parent_pid=args.matrix_parent_pid,
        matrix_parent_starttime=args.matrix_parent_starttime, matrix_pgid=args.matrix_pgid,
    )


if __name__ == "__main__":
    raise SystemExit(main())
