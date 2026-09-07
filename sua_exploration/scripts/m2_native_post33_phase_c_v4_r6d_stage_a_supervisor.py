#!/usr/bin/env python3
"""Sealed r6d Stage-A supervisor; it is the only matrix-launch front door.

The operator may select only ``--gpu-id 0`` or ``--gpu-id 1``.  The launch
receipt contains metadata, not a caller-controlled matrix command; after
checking the signed authorization and source closure, this supervisor builds
the exact matrix argv itself.  Matrix workers get both a dedicated session and
Linux ``PR_SET_PDEATHSIG`` protection before ``exec``.  The watcher deliberately
does *not* receive PDEATHSIG so it can kill a verified matrix after supervisor
parent drift.  A user-systemd service with ``KillMode=control-group`` is an
independent outer containment boundary, not a substitute for these checks.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[2]
R6D = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r6d"
R6D_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cells_20260805_r6d"
RUNTIME = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_r6d_live_signer_20260805"
READY = RUNTIME / "live_signer_ready.json"
PROGRAM = R6D / "program/phase_c_program_r6d.json"
PORTABLE = R6D / "manifest/portable_r6d.json"
COST = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r3/cost/cost_supplement_r3.json"
LAUNCH = R6D / "launch/stage_a_commands_r6d.json"
TRIGGER = R6D / "signer_requests/r6d_live_signer_trigger.json"
CORE = R6D / "signer_requests/r6d_live_signer_plan_core.json"
PROOF = R6D / "r6d_static_pretrigger_evidence.json"
CONTROLLER = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_live_signer.py"
GATE = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_stage_a_continue_gate.py"
ASSERT_LIVE = ROOT / "sua_exploration/scripts/m2_native_post33_phase_c_v4_r6d_assert_live.py"
MATRIX = ROOT / "sua_exploration/scripts/run_m2_native_post33_phase_c_v4_matrix.py"
PYTHON = "/home/xinyuan/miniconda3/envs/spint/bin/python"
POLL_SECONDS = 0.25
POST_COMPLETION_MATRIX_GRACE_SECONDS = 60
PR_SET_PDEATHSIG = 1

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "sua_exploration/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from sua_exploration.mc_maze import m2_native_post33_authorization_v4 as authorization  # noqa: E402
from sua_exploration.mc_maze import m2_native_post33_program_v4 as program_module  # noqa: E402
from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (  # noqa: E402
    file_metadata,
    require_canonical_regular_file,
    sha256_file,
    validate_shard_manifest,
    write_json_exclusive,
)
import m2_native_post33_phase_c_v4_r6d_assert_live as watchdog  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(require_canonical_regular_file(path).read_text(encoding="utf-8"))


def _proc_starttime(pid: int) -> int:
    tail = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].strip().split()
    if len(tail) <= 19:
        raise RuntimeError("/proc stat lacks starttime")
    return int(tail[19])


def _proc_cmdline_sha256(pid: int) -> str:
    return hashlib.sha256(Path(f"/proc/{pid}/cmdline").read_bytes()).hexdigest()


def _metadata(value: Any, expected: Path, label: str) -> Path:
    canonical = require_canonical_regular_file(expected)
    if value != file_metadata(canonical):
        raise PermissionError(f"r6d supervisor {label} metadata mismatch")
    return canonical


def _source_is_sealed(program: Mapping[str, Any], source: Path) -> None:
    relative = source.resolve(strict=True).relative_to(ROOT.resolve(strict=True)).as_posix()
    expected = {
        "relative_path": relative,
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }
    if not isinstance(program.get("source_map"), list) or expected not in program["source_map"]:
        raise PermissionError(f"r6d program does not seal runtime source {relative}")


def _fixed_paths(gpu_id: int) -> dict[str, Path]:
    if gpu_id not in (0, 1):
        raise PermissionError("r6d GPU must be exactly 0 or 1")
    shard = R6D / "manifest" / f"shard_stage_a_gpu{gpu_id}_r6d.json"
    auth = R6D / "auth" / f"stage_a_execution_gpu{gpu_id}_r6d.json"
    return {
        "authorization": auth,
        "signature": auth.with_suffix(".sig"),
        "shard_manifest": shard,
        "program": PROGRAM,
        "portable": PORTABLE,
        "cost_supplement": COST,
    }


def _matrix_argv(paths: Mapping[str, Path]) -> list[str]:
    """Construct, never deserialize, the only permitted matrix command."""
    return [
        PYTHON, "-u", str(MATRIX), "--execute",
        "--cell-root", str(R6D_ROOT.resolve()), "--workspace-root", str(ROOT.resolve()),
        "--data-root", str((ROOT / "SPINT-main/data/000953").resolve(strict=True)),
        "--shard-manifest", str(paths["shard_manifest"].resolve(strict=True)),
        "--portable-manifest", str(paths["portable"].resolve(strict=True)),
        "--program-receipt", str(paths["program"].resolve(strict=True)),
        "--cost-supplement", str(paths["cost_supplement"].resolve(strict=True)),
        "--authorization", str(paths["authorization"].resolve(strict=True)),
        "--authorization-signature", str(paths["signature"].resolve(strict=True)),
    ]


def _matrix_pdeathsig_preexec(expected_parent_pid: int) -> Callable[[], None]:
    """Return a child-only pre-exec arm/recheck for the SIGKILL window.

    The parent may die between fork and ``prctl``.  The immediate PPID check
    closes that race; death after ``prctl`` causes the kernel to SIGTERM the
    unexeced/execed matrix worker.  This function runs only in the Popen child,
    before any GPU code is imported.
    """
    def _arm() -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0) != 0:
            os._exit(126)
        if os.getppid() != expected_parent_pid:
            os._exit(127)
    return _arm


def _kill_matrix_group(pid: int, *, expected_starttime: int, sig: signal.Signals = signal.SIGTERM) -> bool:
    """Signal only a still-identical, dedicated process group."""
    try:
        if _proc_starttime(pid) != expected_starttime:
            return False
        pgid = os.getpgid(pid)
    except (FileNotFoundError, OSError, ProcessLookupError, ValueError):
        return False
    if pgid != pid or pgid <= 1 or pgid == os.getpgrp():
        return False
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return False
    return True


def _reap_matrix(
    matrix: subprocess.Popen[Any] | None, *, pid: int | None, starttime: int | None
) -> int | None:
    if matrix is None:
        return None
    if matrix.poll() is None and pid is not None and starttime is not None:
        _kill_matrix_group(pid, expected_starttime=starttime)
    try:
        return matrix.wait(timeout=30)
    except subprocess.TimeoutExpired:
        if pid is not None and starttime is not None:
            _kill_matrix_group(pid, expected_starttime=starttime, sig=signal.SIGKILL)
        return matrix.wait(timeout=10)


def _stop_watcher(watcher: subprocess.Popen[Any] | None) -> int | None:
    if watcher is None:
        return None
    result = watcher.poll()
    if result is not None:
        return result
    watcher.terminate()
    try:
        return watcher.wait(timeout=10)
    except subprocess.TimeoutExpired:
        watcher.kill()
        return watcher.wait(timeout=10)


def _verify_contract_and_row(gpu_id: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    payload = _json(LAUNCH)
    if payload.get("schema") != "m2_post33_phase_c_v4_r6d_stage_a_launch_commands_v1":
        raise PermissionError("r6d launch schema mismatch")
    if payload.get("status") != "PREPARED_NOT_EXECUTED":
        raise PermissionError("r6d launch state mismatch")
    if payload.get("cell_root_must_be_absent_until_matrix_execution") != str(R6D_ROOT.resolve()):
        raise PermissionError("r6d launch cell-root substitution")
    contract = payload.get("source_sealed_supervision_contract")
    if not isinstance(contract, Mapping) or set(contract) != {
        "controller", "continue_gate", "supervisor", "checker", "ready", "program",
        "check_once_argv_template", "watch_argv_template",
        "matrix_must_start_new_session", "matrix_process_group_must_equal_matrix_pid",
        "matrix_pr_set_pdeathsig", "matrix_pdeathsig_parent_recheck",
        "watcher_must_not_use_pdeathsig", "one_independent_watcher_per_gpu_shard",
        "execution_start_receipt_required",
    }:
        raise PermissionError("r6d supervision contract exact-key mismatch")
    source_values = {
        "controller": CONTROLLER, "continue_gate": GATE, "supervisor": Path(__file__).resolve(),
        "checker": ASSERT_LIVE, "ready": READY, "program": PROGRAM,
    }
    for field, path in source_values.items():
        _metadata(contract.get(field), path, f"contract {field}")
    if (
        tuple(contract.get("check_once_argv_template", ())) != watchdog.check_once_argv_template()
        or tuple(contract.get("watch_argv_template", ())) != watchdog.watch_argv_template()
        or contract.get("matrix_must_start_new_session") is not True
        or contract.get("matrix_process_group_must_equal_matrix_pid") is not True
        or contract.get("matrix_pr_set_pdeathsig") != "SIGTERM"
        or contract.get("matrix_pdeathsig_parent_recheck") is not True
        or contract.get("watcher_must_not_use_pdeathsig") is not True
        or contract.get("one_independent_watcher_per_gpu_shard") is not True
        or contract.get("execution_start_receipt_required") is not True
    ):
        raise PermissionError("r6d supervision contract behavior mismatch")
    program_payload = program_module.validate_phase_c_program_receipt(PROGRAM)
    for source in (CONTROLLER, GATE, ASSERT_LIVE, Path(__file__).resolve()):
        _source_is_sealed(program_payload, source)
    commands = payload.get("commands")
    name = f"gpu{gpu_id}"
    if not isinstance(commands, Mapping) or set(commands) != {"gpu0", "gpu1"}:
        raise PermissionError("r6d command scope mismatch")
    row = commands.get(name)
    expected_keys = {
        "gpu_id", "supervisor_argv", "authorization", "authorization_signature_path",
        "shard_manifest", "program", "portable", "cost_supplement",
        "fold_allowlist", "cuda_visible_devices",
    }
    if not isinstance(row, Mapping) or set(row) != expected_keys:
        raise PermissionError("r6d command row exact-key mismatch")
    paths = _fixed_paths(gpu_id)
    if row.get("gpu_id") != gpu_id:
        raise PermissionError("r6d command GPU index mismatch")
    expected_supervisor_argv = [PYTHON, "-u", str(Path(__file__).resolve()), "--gpu-id", str(gpu_id)]
    if row.get("supervisor_argv") != expected_supervisor_argv:
        raise PermissionError("r6d external supervisor command substitution")
    for field in ("authorization", "shard_manifest", "program", "portable", "cost_supplement"):
        _metadata(row.get(field), paths[field], f"row {field}")
    signature = Path(str(row.get("authorization_signature_path", ""))).resolve()
    if signature != paths["signature"].resolve() or not signature.is_file():
        raise PermissionError("r6d authorization signature missing/substituted")
    shard = validate_shard_manifest(
        paths["shard_manifest"], portable_manifest_path=paths["portable"], cell_root=R6D_ROOT
    )
    if row.get("fold_allowlist") != shard.get("fold_allowlist") or row.get("cuda_visible_devices") != shard.get("gpu_id"):
        raise PermissionError("r6d row/shard scope mismatch")
    authorization.verify_signed_authorization(
        paths["authorization"], paths["signature"], phase_c_program_receipt_path=paths["program"],
        portable_manifest_path=paths["portable"], shard_manifest_path=paths["shard_manifest"],
        cost_supplement_path=paths["cost_supplement"], cell_root=R6D_ROOT,
        now=datetime.now(timezone.utc),
    )
    envelope = _json(paths["authorization"])
    body = envelope.get("authorization")
    if not isinstance(body, Mapping) or (
        body.get("stage") != "stage_a" or body.get("capability_scope") != "cell_execution"
        or body.get("seed_allowlist") != [42] or body.get("gpu_id") != shard.get("gpu_id")
        or body.get("fold_allowlist") != shard.get("fold_allowlist")
    ):
        raise PermissionError("r6d signed Stage-A capability scope mismatch")
    return payload, dict(row), paths


def _verify_stage_a_ack(gpu_id: int, paths: Mapping[str, Path]) -> None:
    ack = _json(R6D / "signer_acks/stage_a_capabilities_signed.json")
    expected_runtime_sources = {
        "controller": file_metadata(CONTROLLER),
        "continue_gate": file_metadata(GATE),
        "supervisor": file_metadata(Path(__file__).resolve()),
        "checker": file_metadata(ASSERT_LIVE),
    }
    if (
        ack.get("schema") != "m2_post33_phase_c_v4_r6d_stage_a_capabilities_signed_v1"
        or ack.get("ready") != file_metadata(READY)
        or ack.get("program") != file_metadata(PROGRAM)
        or ack.get("trigger") != file_metadata(TRIGGER)
        or ack.get("plan_core") != file_metadata(CORE)
        or ack.get("pretrigger_proof") != file_metadata(PROOF)
        or ack.get("launch") != file_metadata(LAUNCH)
        or ack.get("runtime_sources") != expected_runtime_sources
    ):
        raise PermissionError("r6d signer ack control-chain binding mismatch")
    evidence = ack.get("stage_a", {}).get(f"gpu{gpu_id}")
    if not isinstance(evidence, Mapping) or (
        evidence.get("authorization") != file_metadata(paths["authorization"])
        or evidence.get("signature") != file_metadata(paths["signature"])
    ):
        raise PermissionError("r6d signer ack does not bind selected GPU capability")


def _check_once() -> None:
    result = subprocess.run(
        list(watchdog.check_once_argv_template()), cwd=ROOT,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise PermissionError("r6d signer check-once failed before matrix launch")


def _terminal_payload(
    *, state: str, gpu_id: int, reservation: Path, started: Path,
    matrix_pid: int | None, matrix_starttime: int | None, matrix_rc: int | None,
    watcher_pid: int | None, watcher_rc: int | None, error: BaseException | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": f"m2_post33_phase_c_v4_r6d_stage_a_execution_{state}_v1",
        "state": state.upper(), "recorded_at_utc": _utc_now(), "gpu_id": gpu_id,
        "reservation": file_metadata(reservation),
        "execution_started": file_metadata(started) if started.is_file() else None,
        "matrix_pid": matrix_pid, "matrix_proc_starttime_ticks": matrix_starttime,
        "matrix_returncode": matrix_rc, "watcher_pid": watcher_pid,
        "watcher_returncode": watcher_rc, "score_or_decision_content_read": False,
        "gpu_api_used_by_supervisor": False,
    }
    if error is not None:
        payload["error"] = {"type": type(error).__name__, "message": str(error)}
    return payload


def supervise(gpu_id: int) -> int:
    """Run exactly one fixed GPU shard after all sealed control checks pass."""
    if gpu_id not in (0, 1):
        raise ValueError("gpu_id must be 0 or 1")
    _check_once()
    launch, row, paths = _verify_contract_and_row(gpu_id)
    _verify_stage_a_ack(gpu_id, paths)
    launch_dir = R6D / "launch"
    stem = f"stage_a_gpu{gpu_id}"
    reservation = launch_dir / f"{stem}_execution_reserved.json"
    started = launch_dir / f"{stem}_execution_started.json"
    completed = launch_dir / f"{stem}_execution_completed.json"
    failed = launch_dir / f"{stem}_execution_failed.json"
    stdout = launch_dir / f"{stem}_matrix.stdout.log"
    stderr = launch_dir / f"{stem}_matrix.stderr.log"
    write_json_exclusive(reservation, {
        "schema": "m2_post33_phase_c_v4_r6d_stage_a_execution_reservation_v1",
        "state": "RESERVED_NOT_STARTED", "reserved_at_utc": _utc_now(), "gpu_id": gpu_id,
        "launch_receipt": file_metadata(LAUNCH), "ready": file_metadata(READY),
        "supervisor_source": file_metadata(Path(__file__).resolve()),
        "supervisor_identity": watchdog.process_identity_snapshot(os.getpid()),
        "stdout": str(stdout.resolve()), "stderr": str(stderr.resolve()),
        "score_or_decision_content_read": False, "gpu_api_used_by_supervisor": False,
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
        parent_pid = os.getpid()
        parent_starttime = _proc_starttime(parent_pid)
        matrix = subprocess.Popen(
            _matrix_argv(paths), cwd=ROOT,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT),
                 "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": str(row["cuda_visible_devices"])},
            stdin=subprocess.DEVNULL, stdout=stdout_handle, stderr=stderr_handle,
            start_new_session=True, close_fds=True,
            preexec_fn=_matrix_pdeathsig_preexec(parent_pid),
        )
        matrix_pid = matrix.pid
        matrix_starttime = _proc_starttime(matrix_pid)
        matrix_pgid = os.getpgid(matrix_pid)
        if matrix_pgid != matrix_pid:
            _kill_matrix_group(matrix_pid, expected_starttime=matrix_starttime)
            raise PermissionError("r6d matrix failed dedicated-session contract")
        watcher_argv = list(watchdog.watch_argv_template(
            matrix_pid=matrix_pid, matrix_starttime=matrix_starttime,
            matrix_parent_pid=parent_pid, matrix_parent_starttime=parent_starttime,
            matrix_pgid=matrix_pgid,
        ))
        watcher = subprocess.Popen(
            watcher_argv, cwd=ROOT,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT), "PYTHONNOUSERSITE": "1"},
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, close_fds=True,
        )
        watcher_starttime = _proc_starttime(watcher.pid)
        write_json_exclusive(started, {
            "schema": "m2_post33_phase_c_v4_r6d_stage_a_execution_start_v1",
            "state": "STARTED", "started_at_utc": _utc_now(), "gpu_id": gpu_id,
            "reservation": file_metadata(reservation), "launch_receipt": file_metadata(LAUNCH),
            "ready": file_metadata(READY), "assert_live_source": file_metadata(ASSERT_LIVE),
            "supervisor_source": file_metadata(Path(__file__).resolve()),
            "matrix_pdeathsig": "SIGTERM", "matrix_pdeathsig_parent_recheck": True,
            "watcher_must_not_use_pdeathsig": True,
            "stdout": str(stdout.resolve()), "stderr": str(stderr.resolve()),
            "matrix": {"pid": matrix_pid, "proc_starttime_ticks": matrix_starttime,
                       "pgid": matrix_pgid, "parent_pid": parent_pid,
                       "parent_proc_starttime_ticks": parent_starttime,
                       "argv": _matrix_argv(paths), "cuda_visible_devices": row["cuda_visible_devices"]},
            "matrix_identity": watchdog.process_identity_snapshot(matrix_pid),
            "watcher": {"pid": watcher.pid, "proc_starttime_ticks": watcher_starttime,
                        "proc_cmdline_sha256": _proc_cmdline_sha256(watcher.pid), "argv": watcher_argv},
            "watcher_identity": watchdog.process_identity_snapshot(watcher.pid),
            "initial_assert_live_check_once": "PASS",
            "watcher_required_to_enforce_signer_liveness_until_stage_a_completed": True,
            "score_or_decision_content_read": False, "gpu_api_used_by_supervisor": False,
        })
        while True:
            watcher_rc, matrix_rc = watcher.poll(), matrix.poll()
            if watcher_rc is not None:
                # A watcher may report success only after observing a stable
                # score-free Stage-A completion marker.  Re-poll the matrix:
                # it can legitimately be in final teardown, but without that
                # marker a live matrix has just lost its liveness guardian.
                matrix_rc = matrix.poll()
                if watcher_rc != 0 and matrix_rc is None:
                    _kill_matrix_group(matrix_pid, expected_starttime=matrix_starttime)
                if (
                    watcher_rc == 0
                    and matrix_rc is None
                    and not watchdog.stage_a_completion_marker_is_stable()
                ):
                    _kill_matrix_group(matrix_pid, expected_starttime=matrix_starttime)
                    raise RuntimeError(
                        "r6d watchdog exited before a stable Stage-A completion marker"
                    )
                break
            if matrix_rc is not None:
                watcher_rc = watcher.wait(timeout=10)
                break
            time.sleep(POLL_SECONDS)
        watcher_rc = watcher.wait()
        if watcher_rc != 0:
            matrix_rc = _reap_matrix(matrix, pid=matrix_pid, starttime=matrix_starttime)
            raise RuntimeError(f"r6d signer watchdog failed closed: gpu{gpu_id} exit={watcher_rc}")
        try:
            matrix_rc = (
                matrix.wait()
                if matrix.poll() is not None
                else matrix.wait(timeout=POST_COMPLETION_MATRIX_GRACE_SECONDS)
            )
        except subprocess.TimeoutExpired as exc:
            matrix_rc = _reap_matrix(matrix, pid=matrix_pid, starttime=matrix_starttime)
            raise RuntimeError(f"r6d matrix survived healthy watcher grace: gpu{gpu_id}") from exc
        if matrix_rc != 0:
            raise RuntimeError(f"r6d Stage-A matrix failed: gpu{gpu_id} exit={matrix_rc}")
        write_json_exclusive(completed, _terminal_payload(
            state="completed", gpu_id=gpu_id, reservation=reservation, started=started,
            matrix_pid=matrix_pid, matrix_starttime=matrix_starttime, matrix_rc=matrix_rc,
            watcher_pid=watcher.pid, watcher_rc=watcher_rc,
        ))
        return matrix_rc
    except BaseException as exc:
        matrix_rc = _reap_matrix(matrix, pid=matrix_pid, starttime=matrix_starttime)
        watcher_rc = _stop_watcher(watcher)
        try:
            write_json_exclusive(failed, _terminal_payload(
                state="failed", gpu_id=gpu_id, reservation=reservation, started=started,
                matrix_pid=matrix_pid, matrix_starttime=matrix_starttime, matrix_rc=matrix_rc,
                watcher_pid=watcher.pid if watcher is not None else None, watcher_rc=watcher_rc,
                error=exc,
            ))
        finally:
            # The matrix has already been kill/reaped before a failed receipt
            # can itself throw; this finally is an idempotent last resort.
            _reap_matrix(matrix, pid=matrix_pid, starttime=matrix_starttime)
            _stop_watcher(watcher)
        raise
    finally:
        _reap_matrix(matrix, pid=matrix_pid, starttime=matrix_starttime)
        _stop_watcher(watcher)
        if stdout_handle is not None:
            stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int, choices=(0, 1), required=True)
    args = parser.parse_args(argv)
    return supervise(args.gpu_id)


if __name__ == "__main__":
    raise SystemExit(main())
