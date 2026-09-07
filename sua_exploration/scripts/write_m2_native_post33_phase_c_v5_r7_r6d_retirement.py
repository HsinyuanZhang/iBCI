#!/usr/bin/env python3
"""Retire the failed r6d live signer without opening its failure payload.

This append-only control-plane action is intentionally outside all r6d roots.
It verifies the live memory-key holder by PID starttime and argv hash, proves
that no Stage-A decision or Stage-B/full-opening authorization exists using
path topology only, sends one controlled SIGINT, waits for exit, and emits a
new recovery receipt.  It never reads status.failed JSON, stdout/stderr,
selector data, opaque endpoint payloads, scores, or R² values.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
R6D_RECEIPTS = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r6d"
R6D_CELL_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cells_20260805_r6d"
R6D_PHASE = R6D_CELL_ROOT / "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1/PHASE_C_V4"
RUNTIME = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_r6d_live_signer_20260805"
READY = RUNTIME / "live_signer_ready.json"
PUBLIC = ROOT / "sua_exploration/configs/m2_native_post33_phase_c_v4_r6d_root_ed25519_public.pem"
OUT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v5_r7_device_recovery_20260805/r6d_engineering_failure_retirement.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (  # noqa: E402
    file_metadata,
    require_canonical_regular_file,
    write_json_exclusive,
)


def _read_ready() -> dict[str, Any]:
    """READY has no score/decision data and is required to bind the live PID."""
    return json.loads(require_canonical_regular_file(READY, within=RUNTIME).read_text(encoding="utf-8"))


def _proc_starttime(pid: int) -> int:
    tail = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].strip().split()
    if len(tail) <= 19:
        raise RuntimeError("/proc stat lacks process starttime")
    return int(tail[19])


def _proc_cmdline_sha256(pid: int) -> str:
    return hashlib.sha256(Path(f"/proc/{pid}/cmdline").read_bytes()).hexdigest()


def _assert_live_signer(ready: dict[str, Any]) -> int:
    pid = ready.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
        raise PermissionError("r6d READY lacks a valid signer PID")
    if not Path(f"/proc/{pid}").is_dir():
        raise PermissionError("r6d READY signer is no longer live")
    if _proc_starttime(pid) != ready.get("proc_starttime_ticks"):
        raise PermissionError("r6d signer PID starttime mismatch")
    if _proc_cmdline_sha256(pid) != ready.get("proc_cmdline_sha256"):
        raise PermissionError("r6d signer PID argv hash mismatch")
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "strict")
    if "m2_native_post33_phase_c_v4_r6d_live_signer.py" not in cmdline or "--bootstrap" not in cmdline:
        raise PermissionError("r6d signer argv does not identify the live bootstrap controller")
    return pid


def _absent(path: Path, label: str) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise PermissionError(f"r6d retirement requires absent {label}: {path}")
    return {"canonical_path": str(path.resolve()), "present": False}


def _lifecycle_file_topology(path: Path, label: str) -> dict[str, Any]:
    """Record only inode topology; do not open potentially diagnostic content."""
    if path.is_symlink() or not path.is_file():
        raise PermissionError(f"r6d expected lifecycle file missing/non-regular: {label}")
    state = path.stat()
    return {
        "label": label,
        "canonical_path": str(path.resolve(strict=True)),
        "regular_file": True,
        "size_bytes": int(state.st_size),
        "inode": int(state.st_ino),
        "content_opened_or_hashed": False,
    }


def _no_active_r6d_workers() -> list[str]:
    ps = subprocess.run(
        ["ps", "-eo", "pid=,args="], check=True, text=True, capture_output=True
    ).stdout
    worker_tokens = (
        "m2_native_post33_phase_c_v4_r6d_stage_a_supervisor.py",
        "run_m2_native_post33_phase_c_v4_matrix.py",
        "run_m2_native_post33_phase_c_v4_cell_pipeline.py",
        "train_post33_phase_c_v4.py",
        "evaluate_post33_phase_c_v4.py",
    )
    own_pid = os.getpid()
    rows = []
    for line in ps.splitlines():
        fields = line.strip().split(maxsplit=1)
        if not fields or not fields[0].isdigit() or int(fields[0]) == own_pid:
            continue
        argv = fields[1] if len(fields) == 2 else ""
        if str(R6D_CELL_ROOT) in argv and any(token in argv for token in worker_tokens):
            rows.append(line)
    return rows


def _no_gpu_compute_processes() -> bool:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
    )
    return not bool(result.stdout.strip())


def _stage_b_and_decision_absence() -> dict[str, Any]:
    absent = {
        "stage_a_status_completed": _absent(R6D_PHASE / "stage_a/status.completed.json", "Stage-A completion"),
        "stage_a_manifest": _absent(R6D_PHASE / "stage_a/score_sealed_stage_a_manifest.json", "Stage-A manifest"),
        "stage_a_decision": _absent(R6D_PHASE / "stage_a/opened_stage_a_decision.json", "Stage-A decision"),
        "stage_a_decision_signature": _absent(R6D_PHASE / "stage_a/opened_stage_a_decision.sig", "Stage-A decision signature"),
        "conditional_signer_ack": _absent(
            R6D_RECEIPTS / "signer_acks/conditional_stage_b_capabilities_signed.json",
            "conditional Stage-B signer acknowledgement",
        ),
        "validated_stop_ack": _absent(
            R6D_RECEIPTS / "signer_acks/stage_a_futility_stop_no_stage_b_capabilities.json",
            "Stage-A stop acknowledgement",
        ),
    }
    for stem in ("stage_b_execution_gpu0_r6d", "stage_b_execution_gpu1_r6d", "full_opening_r6d"):
        absent[f"{stem}_authorization"] = _absent(
            R6D_RECEIPTS / "auth" / f"{stem}.json", f"{stem} authorization"
        )
        absent[f"{stem}_signature"] = _absent(
            R6D_RECEIPTS / "auth" / f"{stem}.sig", f"{stem} signature"
        )
    return absent


def main() -> None:
    if OUT.exists() or OUT.is_symlink():
        raise FileExistsError(f"append-only r6d retirement already exists: {OUT}")
    if not R6D_RECEIPTS.is_dir() or not R6D_PHASE.is_dir():
        raise FileNotFoundError("r6d receipt/cell roots are required for retirement")
    if not PUBLIC.is_file():
        raise FileNotFoundError("r6d public anchor is missing")
    ready = _read_ready()
    if ready.get("public_key") != file_metadata(PUBLIC):
        raise PermissionError("r6d READY/public-anchor metadata mismatch")
    if ready.get("private_key_retained_live_in_memory") is not True:
        raise PermissionError("r6d READY does not claim a live-only key")
    signer_pid = _assert_live_signer(ready)
    if _no_active_r6d_workers():
        raise PermissionError("r6d worker/supervisor is still live")
    if not _no_gpu_compute_processes():
        raise PermissionError("a GPU compute process is still live; refuse signer retirement")
    topology_before = _stage_b_and_decision_absence()
    failed_cells = [
        _lifecycle_file_topology(
            R6D_PHASE / f"cells/arm-spint/fold-{fold}/seed-42/control/status.failed.json",
            f"spint-fold-{fold}-seed-42 failure status",
        )
        for fold in (0, 1)
    ]

    # Close the PID-reuse race before the only state-changing operation.  SIGINT
    # is chosen so the signer catches KeyboardInterrupt, executes its ``finally``
    # block (clearing the live key reference), and terminates without any path to
    # an absent Stage-A completion/decision or Stage-B authorization.
    if _assert_live_signer(ready) != signer_pid:
        raise PermissionError("r6d signer identity changed before controlled SIGINT")
    os.kill(signer_pid, signal.SIGINT)
    deadline = time.monotonic() + 15.0
    while Path(f"/proc/{signer_pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    if Path(f"/proc/{signer_pid}").exists():
        raise RuntimeError("r6d signer did not exit after controlled SIGINT")
    if _no_active_r6d_workers():
        raise PermissionError("r6d worker/supervisor appeared during signer retirement")
    if not _no_gpu_compute_processes():
        raise PermissionError("GPU compute process appeared during signer retirement")
    topology_after = _stage_b_and_decision_absence()

    payload = {
        "schema": "m2_post33_phase_c_v5_r7_r6d_engineering_failure_retirement_v1",
        "protocol_id": "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1",
        "retired_lineage": "PHASE_C_V4_r6d",
        "disposition": "RETIRED_AFTER_STAGE_A_ENGINEERING_FAILURE_BEFORE_DECISION_OR_STAGE_B",
        "reason": (
            "Post-Trainer.test deployment microbenchmark mixed a Lightning-teardown CPU deployment "
            "network with a plain cached CUDA identity.  v5 repair is separately tested; r6d is not reused."
        ),
        "r6d_receipt_root": str(R6D_RECEIPTS.resolve()),
        "r6d_cell_root": str(R6D_CELL_ROOT.resolve()),
        "ready": file_metadata(READY),
        "public_anchor": file_metadata(PUBLIC),
        "signer_pid": signer_pid,
        "controlled_signal": "SIGINT",
        "signer_process_alive_after_retirement": False,
        "private_key_serialized_or_disk_persisted": False,
        "r6d_worker_or_supervisor_alive_after_retirement": False,
        "gpu_compute_process_alive_after_retirement": False,
        "failed_cell_lifecycle_topology": failed_cells,
        "failure_status_or_log_content_opened": False,
        "endpoint_score_selector_or_r2_opened": False,
        "stage_b_or_full_opening_authorization_materialized": False,
        "stage_a_decision_materialized": False,
        "absence_before_signal": topology_before,
        "absence_after_signal": topology_after,
        "r6d_source_artifact_or_cell_mutation_performed": False,
        "r6d_reuse_forbidden": True,
        "fresh_r7_root_nonce_selector_program_and_authorization_required": True,
        "formal_gpu_rerun_authorized_by_this_receipt": False,
    }
    print(write_json_exclusive(OUT, payload))


if __name__ == "__main__":
    main()
