#!/usr/bin/env python3
"""Read-only liveness assertion/watchdog for the r6c live signer.

This is deliberately not a Phase-C evaluator, launcher, opener, or signer.
It reads only the r6c signer-ready control receipt, the public/controller
source files named by that receipt, Linux ``/proc`` process identity fields,
and the score-free Stage-A completion marker.  It never imports torch, opens
data or score payloads, initializes CUDA, generates a key, or modifies a
Phase-C receipt/cell root.

``--check-once`` is intentionally silent and returns zero only while the
configured signer is still the exact live process that wrote its ready
receipt.  ``--watch`` additionally checks a matrix worker PID/starttime every
0.5 seconds.  If the signer disappears or any bound identity drifts, it sends
SIGTERM to the *verified, separate* matrix process group and writes one
append-only control-plane failure record beneath the signer's runtime
directory.  A stable score-free Stage-A ``status.completed.json`` ends a
healthy watch successfully.
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
R6C_RECEIPTS = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r6c"
R6C_CELL_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cells_20260805_r6c"
RUNTIME = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_r6c_live_signer_20260805"
READY = RUNTIME / "live_signer_ready.json"
PUBLIC_KEY = ROOT / "sua_exploration/configs/m2_native_post33_phase_c_v4_r6c_root_ed25519_public.pem"
SIGNER_SOURCE = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_live_signer.py"
PROTOCOL_ID = "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1"
PHASE_ID = "PHASE_C_V4"
STAGE_A_COMPLETED = R6C_CELL_ROOT / PROTOCOL_ID / PHASE_ID / "stage_a/status.completed.json"
READY_SCHEMA = "m2_post33_phase_c_v4_r6c_live_signer_ready_v1"
FAILURE_SCHEMA = "m2_post33_phase_c_v4_r6c_assert_live_failure_v1"
POLL_SECONDS = 0.5
# This is intentionally byte-for-byte the holder command recorded before r6c
# plan publication.  A ready-file hash alone would accept a different command
# if an attacker could replace both the process and the receipt before a
# watcher's first observation.
EXPECTED_SIGNER_BOOTSTRAP_ARGV = (
    "/home/xinyuan/miniconda3/envs/spint/bin/python",
    "-u",
    "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_live_signer.py",
    "--bootstrap",
)
WATCHDOG_PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"
WATCHDOG_SCRIPT_RELATIVE = "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6c_assert_live.py"
PROCESS_IDENTITY_SCHEMA = "m2_post33_phase_c_v4_r6c_process_identity_v1"


class LiveAssertionError(PermissionError):
    """A non-score control-plane liveness failure."""


@dataclass(frozen=True)
class SignerIdentity:
    pid: int
    proc_starttime_ticks: int
    proc_cmdline_sha256: str
    host_id: str


class MatrixProcessEnded(LiveAssertionError):
    """The owned matrix process ended; its scheduler owns the exit result."""


class MatrixLaunchContractError(LiveAssertionError):
    """A still-live matrix violates the sealed dedicated-session contract."""


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


def _proc_starttime(pid: int) -> int:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        raise LiveAssertionError("PID must be an integer greater than one")
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, OSError) as exc:
        raise LiveAssertionError("bound process is no longer observable") from exc
    try:
        # Field 2 can contain whitespace enclosed by the final ')'; field 22
        # is therefore offset 19 after the closing parenthesis.
        tail = stat.rsplit(")", 1)[1].strip().split()
        return int(tail[19])
    except (IndexError, ValueError) as exc:
        raise LiveAssertionError("/proc stat has no valid process starttime") from exc


def _proc_parent_pid(pid: int) -> int:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        raise LiveAssertionError("PID must be an integer greater than one")
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        tail = stat.rsplit(")", 1)[1].strip().split()
        return int(tail[1])  # field 4 after field-2 comm() has been removed
    except (FileNotFoundError, OSError) as exc:
        raise LiveAssertionError("bound process parent is no longer observable") from exc
    except (IndexError, ValueError) as exc:
        raise LiveAssertionError("/proc stat has no valid parent PID") from exc


def _proc_cmdline_bytes(pid: int) -> bytes:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, OSError) as exc:
        raise LiveAssertionError("bound process command line is no longer observable") from exc


def _proc_cmdline_sha256(pid: int) -> str:
    return hashlib.sha256(_proc_cmdline_bytes(pid)).hexdigest()


def _expected_signer_bootstrap_cmdline() -> bytes:
    return b"\0".join(part.encode("utf-8") for part in EXPECTED_SIGNER_BOOTSTRAP_ARGV) + b"\0"


def process_identity_snapshot(pid: int) -> dict[str, Any]:
    """Return an exact, score-free Linux process identity for an execution receipt."""
    starttime = _proc_starttime(pid)
    parent = _proc_parent_pid(pid)
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError as exc:
        raise LiveAssertionError("bound process group is no longer observable") from exc
    return {
        "schema": PROCESS_IDENTITY_SCHEMA,
        "pid": pid,
        "parent_pid": parent,
        "proc_starttime_ticks": starttime,
        "process_group_id": pgid,
        "proc_cmdline_sha256": _proc_cmdline_sha256(pid),
    }


def check_once_argv_template() -> tuple[str, ...]:
    """Immutable score-free command template for a prelaunch health check."""
    return (WATCHDOG_PYTHON, "-u", WATCHDOG_SCRIPT_RELATIVE, "--check-once")


def watch_argv_template(
    *,
    matrix_pid: int | str = "<MATRIX_PID>",
    matrix_starttime: int | str = "<MATRIX_STARTTIME_TICKS>",
    matrix_parent_pid: int | str = "<MATRIX_PARENT_PID>",
    matrix_parent_starttime: int | str = "<MATRIX_PARENT_STARTTIME_TICKS>",
    matrix_pgid: int | str = "<MATRIX_PROCESS_GROUP_ID>",
) -> tuple[str, ...]:
    """Immutable watch command template; all actual IDs are receipt-bound."""
    return (
        WATCHDOG_PYTHON,
        "-u",
        WATCHDOG_SCRIPT_RELATIVE,
        "--watch",
        "--pid",
        str(matrix_pid),
        "--pid-starttime",
        str(matrix_starttime),
        "--matrix-parent-pid",
        str(matrix_parent_pid),
        "--matrix-parent-starttime",
        str(matrix_parent_starttime),
        "--matrix-pgid",
        str(matrix_pgid),
    )


def _parse_ready() -> Mapping[str, Any]:
    ready_file = _canonical_regular_file(READY)
    try:
        payload = json.loads(ready_file.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError) as exc:
        raise LiveAssertionError("ready receipt is incomplete or malformed") from exc
    if not isinstance(payload, Mapping):
        raise LiveAssertionError("ready receipt must be a mapping")
    return payload


def assert_live() -> SignerIdentity:
    """Validate the ready receipt and the exact currently live signer process."""
    ready = _parse_ready()
    if ready.get("schema") != READY_SCHEMA:
        raise LiveAssertionError("ready schema mismatch")
    if ready.get("status") != "LIVE_PRIVATE_KEY_IN_MEMORY_ONLY":
        raise LiveAssertionError("ready signer status is not a live in-memory-key state")
    # Accept either an affirmative spelling or its precise Boolean inverse;
    # both forms are auditable without overclaiming anything about OS internals.
    serialized = ready.get("private_key_serialized_or_disk_persisted")
    direct_not_serialized = ready.get("private_key_not_serialized")
    if not (serialized is False or direct_not_serialized is True):
        raise LiveAssertionError("ready lacks a true no-private-key-serialization assertion")
    if ready.get("private_key_retained_live_in_memory") is not True:
        raise LiveAssertionError("ready lacks a live in-memory-key retention assertion")
    no_supply = ready.get("private_key_not_supplied_via_argv_environment_file_or_ipc")
    no_supply_with_stdin = ready.get(
        "private_key_not_supplied_via_argv_environment_file_stdin_or_ipc"
    )
    if not (no_supply is True or no_supply_with_stdin is True):
        raise LiveAssertionError("ready lacks a private-key helper-boundary assertion")
    if ready.get("host_id") != socket.gethostname():
        raise LiveAssertionError("ready host identity differs from observed host")
    if ready.get("intended_receipt_root") != str(R6C_RECEIPTS.resolve()):
        raise LiveAssertionError("ready receipt root is not the exact r6c root")
    if ready.get("intended_cell_root") != str(R6C_CELL_ROOT.resolve()):
        raise LiveAssertionError("ready cell root is not the exact r6c root")

    pid = ready.get("pid")
    starttime = ready.get("proc_starttime_ticks")
    command = ready.get("proc_cmdline_sha256")
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 1
        or not isinstance(starttime, int)
        or isinstance(starttime, bool)
        or starttime < 0
        or not isinstance(command, str)
        or len(command) != 64
        or any(character not in "0123456789abcdef" for character in command)
    ):
        raise LiveAssertionError("ready process identity has invalid type/format")

    if ready.get("controller_source") != _file_metadata(SIGNER_SOURCE):
        raise LiveAssertionError("ready controller-source metadata drift")
    if ready.get("public_key") != _file_metadata(PUBLIC_KEY):
        raise LiveAssertionError("ready public-key metadata drift")
    if _proc_starttime(pid) != starttime:
        raise LiveAssertionError("ready signer PID/starttime identity drift")
    observed_command = _proc_cmdline_bytes(pid)
    if hashlib.sha256(observed_command).hexdigest() != command:
        raise LiveAssertionError("ready signer PID/command identity drift")
    if observed_command != _expected_signer_bootstrap_cmdline():
        raise LiveAssertionError("signer process command differs from fixed r6c bootstrap argv")
    return SignerIdentity(
        pid=pid,
        proc_starttime_ticks=starttime,
        proc_cmdline_sha256=command,
        host_id=str(ready["host_id"]),
    )


def _assert_matrix_pid_identity(
    pid: int,
    starttime: int,
    *,
    parent_pid: int | None = None,
    parent_starttime: int | None = None,
    process_group_id: int | None = None,
) -> None:
    if not isinstance(starttime, int) or isinstance(starttime, bool) or starttime < 0:
        raise LiveAssertionError("matrix starttime must be a non-negative integer")
    try:
        observed_starttime = _proc_starttime(pid)
    except LiveAssertionError as exc:
        raise MatrixProcessEnded("matrix process is no longer observable") from exc
    if observed_starttime != starttime:
        raise MatrixProcessEnded("matrix PID/starttime identity drift")
    if parent_pid is None and parent_starttime is None and process_group_id is None:
        return
    if (
        not isinstance(parent_pid, int)
        or isinstance(parent_pid, bool)
        or parent_pid <= 1
        or not isinstance(parent_starttime, int)
        or isinstance(parent_starttime, bool)
        or parent_starttime < 0
        or not isinstance(process_group_id, int)
        or isinstance(process_group_id, bool)
    ):
        raise MatrixLaunchContractError("matrix parent/process-group identity is invalid")
    try:
        observed_parent = _proc_parent_pid(pid)
        observed_parent_starttime = _proc_starttime(parent_pid)
        observed_pgid = os.getpgid(pid)
    except LiveAssertionError as exc:
        raise MatrixLaunchContractError("matrix parent/group is no longer observable") from exc
    except ProcessLookupError as exc:
        raise MatrixLaunchContractError("matrix process group is no longer observable") from exc
    if observed_parent != parent_pid or observed_parent_starttime != parent_starttime:
        raise MatrixLaunchContractError("matrix parent PID/starttime identity drift")
    if observed_pgid != process_group_id or process_group_id != pid:
        raise MatrixLaunchContractError("matrix must retain its dedicated setsid process group")


def _stable_stage_a_completed() -> bool:
    """Observe only a stable completion marker; do not read score/decision data."""
    try:
        first = _file_metadata(STAGE_A_COMPLETED)
    except LiveAssertionError:
        return False
    time.sleep(POLL_SECONDS)
    try:
        second = _file_metadata(STAGE_A_COMPLETED)
    except LiveAssertionError:
        return False
    return first == second


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
    *,
    reason: str,
    matrix_pid: int,
    matrix_starttime: int,
    signer: SignerIdentity | None,
    termination: str,
) -> Path:
    """Append a control-only fault receipt; never touch Phase-C result paths."""
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    payload: dict[str, Any] = {
        "schema": FAILURE_SCHEMA,
        "recorded_at_utc": now,
        "failure_reason": reason,
        "matrix_pid": matrix_pid,
        "matrix_proc_starttime_ticks": matrix_starttime,
        "termination": termination,
        "score_or_decision_content_read": False,
        "phase_c_result_artifact_modified": False,
        "gpu_api_used": False,
        "private_key_read_or_written": False,
    }
    if signer is not None:
        payload["last_verified_signer"] = {
            "pid": signer.pid,
            "proc_starttime_ticks": signer.proc_starttime_ticks,
            "proc_cmdline_sha256": signer.proc_cmdline_sha256,
            "host_id": signer.host_id,
        }
    filename = f"failure_{now.replace(':', '').replace('+00:00', 'Z')}_{os.getpid()}_{uuid4().hex}.json"
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    # Derive from the current fixed runtime constant at write time.  Besides
    # keeping the production boundary explicit, this prevents a test-only
    # patched runtime from accidentally retaining an import-time real path.
    return _write_bytes_exclusive((RUNTIME / "assert_live_failures") / filename, encoded)


def _terminate_matrix_process_group(
    *,
    matrix_pid: int,
    matrix_starttime: int,
    matrix_parent_pid: int,
    matrix_parent_starttime: int,
    matrix_pgid: int,
) -> str:
    """TERM only a still-identical, separate process group.

    A changed/dead worker must not turn a stale PID into a signal target.  A
    process group shared with the watchdog itself is unsafe and is refused.
    """
    try:
        _assert_matrix_pid_identity(
            matrix_pid,
            matrix_starttime,
            parent_pid=matrix_parent_pid,
            parent_starttime=matrix_parent_starttime,
            process_group_id=matrix_pgid,
        )
    except (MatrixProcessEnded, MatrixLaunchContractError):
        return "matrix_identity_not_live_no_signal"
    try:
        pgid = os.getpgid(matrix_pid)
    except ProcessLookupError:
        return "matrix_process_missing_no_signal"
    # The command receipt must launch the matrix worker under ``setsid``.
    # Refuse a shared shell/tmux/supervisor group even if it differs from the
    # watcher's own group: it may contain processes unrelated to this shard.
    if pgid <= 1 or pgid != matrix_pgid or pgid != matrix_pid or pgid == os.getpgrp():
        return "matrix_process_group_not_dedicated_no_signal"
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return "matrix_process_group_missing_no_signal"
    return "matrix_process_group_sigterm_sent"


def check_once() -> int:
    """Return a zero-output health status: 0 healthy, 1 invalid/dead."""
    try:
        assert_live()
    except LiveAssertionError:
        return 1
    return 0


def watch(
    *,
    matrix_pid: int,
    matrix_starttime: int,
    matrix_parent_pid: int,
    matrix_parent_starttime: int,
    matrix_pgid: int,
) -> int:
    """Watch signer+matrix identity until stable Stage-A completion or fault."""
    last_signer: SignerIdentity | None = None
    while True:
        try:
            last_signer = assert_live()
        except LiveAssertionError as exc:
            termination = _terminate_matrix_process_group(
                matrix_pid=matrix_pid,
                matrix_starttime=matrix_starttime,
                matrix_parent_pid=matrix_parent_pid,
                matrix_parent_starttime=matrix_parent_starttime,
                matrix_pgid=matrix_pgid,
            )
            _record_failure(
                reason=type(exc).__name__ + ": " + str(exc),
                matrix_pid=matrix_pid,
                matrix_starttime=matrix_starttime,
                signer=last_signer,
                termination=termination,
            )
            return 1
        # A matrix worker can legitimately finish before this polling loop
        # sees the Stage-A aggregate marker.  Its scheduler/launcher owns its
        # return code; it is not evidence that the signer has failed.  Do not
        # turn an ordinary worker exit or PID reuse into a spurious fault.
        try:
            _assert_matrix_pid_identity(
                matrix_pid,
                matrix_starttime,
                parent_pid=matrix_parent_pid,
                parent_starttime=matrix_parent_starttime,
                process_group_id=matrix_pgid,
            )
        except MatrixProcessEnded:
            return 0
        except MatrixLaunchContractError as exc:
            _record_failure(
                reason=type(exc).__name__ + ": " + str(exc),
                matrix_pid=matrix_pid,
                matrix_starttime=matrix_starttime,
                signer=last_signer,
                termination="matrix_launch_contract_invalid_no_signal",
            )
            return 2
        if _stable_stage_a_completed():
            return 0
        time.sleep(POLL_SECONDS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check-once", action="store_true")
    modes.add_argument("--watch", action="store_true")
    parser.add_argument("--pid", type=int, help="matrix worker PID (required with --watch)")
    parser.add_argument(
        "--pid-starttime",
        type=int,
        help="Linux /proc starttime ticks for --pid (required with --watch)",
    )
    parser.add_argument("--matrix-parent-pid", type=int, help="receipt-bound matrix parent PID")
    parser.add_argument(
        "--matrix-parent-starttime",
        type=int,
        help="receipt-bound Linux /proc starttime ticks for --matrix-parent-pid",
    )
    parser.add_argument("--matrix-pgid", type=int, help="receipt-bound dedicated matrix process-group ID")
    args = parser.parse_args(argv)
    required_watch = (
        args.pid,
        args.pid_starttime,
        args.matrix_parent_pid,
        args.matrix_parent_starttime,
        args.matrix_pgid,
    )
    if args.watch and any(value is None for value in required_watch):
        parser.error(
            "--watch requires --pid, --pid-starttime, --matrix-parent-pid, "
            "--matrix-parent-starttime, and --matrix-pgid"
        )
    if args.check_once and any(value is not None for value in required_watch):
        parser.error("--check-once does not accept matrix PID arguments")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check_once:
        return check_once()
    return watch(
        matrix_pid=args.pid,
        matrix_starttime=args.pid_starttime,
        matrix_parent_pid=args.matrix_parent_pid,
        matrix_parent_starttime=args.matrix_parent_starttime,
        matrix_pgid=args.matrix_pgid,
    )


if __name__ == "__main__":
    raise SystemExit(main())
