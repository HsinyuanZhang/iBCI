#!/usr/bin/env python3
"""Retire sealed r6c before GPU/data access and destroy its memory-only key.

This control-plane writer never opens a dataset, endpoint, selector, decision,
or score payload.  It verifies the exact live signer identity, proves that no
r6c cell/execution scope was created, sends SIGINT to that one process, waits
for key-holder exit, and writes one append-only retirement receipt outside the
r6c receipt/runtime roots.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


ROOT = Path(__file__).resolve().parents[2]
R6C = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r6c"
R6C_CELLS = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cells_20260805_r6c"
RUNTIME = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_r6c_live_signer_20260805"
READY = RUNTIME / "live_signer_ready.json"
FAILURE = RUNTIME / "controller_failure.json"
PUBLIC = ROOT / "sua_exploration/configs/m2_native_post33_phase_c_v4_r6c_root_ed25519_public.pem"
OUT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_r6c_retirement_20260805/r6c_pre_gpu_retirement.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (  # noqa: E402
    file_metadata,
    require_canonical_regular_file,
    write_json_exclusive,
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(require_canonical_regular_file(path).read_text(encoding="utf-8"))


def _proc_starttime(pid: int) -> int:
    tail = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].strip().split()
    return int(tail[19])


def _proc_cmdline_sha256(pid: int) -> str:
    return hashlib.sha256(Path(f"/proc/{pid}/cmdline").read_bytes()).hexdigest()


def _assert_current_signer_identity(ready: dict[str, Any]) -> int:
    pid = ready.get("pid")
    if (
        not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 1
        or _proc_starttime(pid) != ready.get("proc_starttime_ticks")
        or _proc_cmdline_sha256(pid) != ready.get("proc_cmdline_sha256")
    ):
        raise PermissionError("r6c live signer PID identity mismatch")
    return pid


def _absent(path: Path, label: str) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise PermissionError(f"r6c retirement requires absent {label}: {path}")
    return {"canonical_path": str(path.resolve()), "present": False}


def _verify_stage_a_signatures() -> list[dict[str, Any]]:
    key = serialization.load_pem_public_key(require_canonical_regular_file(PUBLIC).read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError("r6c public anchor is not Ed25519")
    rows = []
    for ack_key, stem in (
        ("gpu0", "stage_a_execution_gpu0_r6c"),
        ("gpu1", "stage_a_execution_gpu1_r6c"),
        ("opening", "stage_a_opening_r6c"),
    ):
        authorization = require_canonical_regular_file(R6C / "auth" / f"{stem}.json", within=R6C)
        signature_path = require_canonical_regular_file(authorization.with_suffix(".sig"), within=R6C)
        signature_bytes = base64.b64decode(signature_path.read_bytes(), validate=True)
        key.verify(signature_bytes, authorization.read_bytes())
        rows.append({
            "ack_key": ack_key,
            "authorization": file_metadata(authorization),
            "signature": file_metadata(signature_path),
            "decoded_signature_bytes": len(signature_bytes),
            "ed25519_verification": "PASS",
        })
    return rows


def _active_r6c_workers() -> list[str]:
    ps = subprocess.run(
        ["ps", "-eo", "pid=,args="], check=True, text=True, capture_output=True
    ).stdout
    worker_tokens = (
        "m2_native_post33_phase_c_v4_r6c_stage_a_supervisor.py",
        "run_m2_native_post33_phase_c_v4_matrix.py",
        "run_m2_native_post33_phase_c_v4_cell_pipeline.py",
        "train_post33_phase_c_v4.py",
        "evaluate_post33_phase_c_v4.py",
    )
    return [line for line in ps.splitlines() if any(token in line for token in worker_tokens)]


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"append-only r6c retirement already exists: {OUT}")
    if R6C_CELLS.exists() or R6C_CELLS.is_symlink():
        raise PermissionError("r6c cell root exists; this writer is pre-GPU only")
    if _active_r6c_workers():
        raise PermissionError("r6c worker/supervisor process exists; refuse retirement")
    ready = _json(READY)
    pid = _assert_current_signer_identity(ready)
    if ready.get("public_key") != file_metadata(PUBLIC):
        raise PermissionError("r6c ready/public anchor mismatch")
    stage_a = _verify_stage_a_signatures()
    ack = require_canonical_regular_file(R6C / "signer_acks/stage_a_capabilities_signed.json", within=R6C)
    ack_payload = _json(ack)
    if (
        ack_payload.get("schema") != "m2_post33_phase_c_v4_r6c_stage_a_capabilities_signed_v1"
        or ack_payload.get("ready") != file_metadata(READY)
        or not isinstance(ack_payload.get("stage_a"), dict)
        or set(ack_payload["stage_a"]) != {"gpu0", "gpu1", "opening"}
    ):
        raise PermissionError("r6c Stage-A signer ACK schema/READY/scope mismatch")
    for row in stage_a:
        evidence = ack_payload["stage_a"].get(row["ack_key"])
        if not isinstance(evidence, dict) or evidence != {
            "authorization": row["authorization"],
            "signature": row["signature"],
        }:
            raise PermissionError(f"r6c signer ACK binding mismatch for {row['ack_key']}")
    conditional_absence = {
        stem: {
            "authorization": _absent(R6C / "auth" / f"{stem}.json", f"{stem} authorization"),
            "signature": _absent(R6C / "auth" / f"{stem}.sig", f"{stem} signature"),
        }
        for stem in (
            "stage_b_execution_gpu0_r6c",
            "stage_b_execution_gpu1_r6c",
            "full_opening_r6c",
        )
    }
    execution_absence = {
        name: _absent(R6C / "launch" / name, name)
        for name in (
            "stage_a_gpu0_execution_reserved.json",
            "stage_a_gpu1_execution_reserved.json",
            "stage_a_gpu0_execution_started.json",
            "stage_a_gpu1_execution_started.json",
            "stage_a_gpu0_execution_completed.json",
            "stage_a_gpu1_execution_completed.json",
            "stage_a_gpu0_execution_failed.json",
            "stage_a_gpu1_execution_failed.json",
            "stage_a_gpu0_matrix.stdout.log",
            "stage_a_gpu1_matrix.stdout.log",
            "stage_a_gpu0_matrix.stderr.log",
            "stage_a_gpu1_matrix.stderr.log",
        )
    }
    # Close the PID-reuse race: revalidate starttime and cmdline immediately
    # before emitting the only destructive signal.
    if _assert_current_signer_identity(ready) != pid:
        raise PermissionError("r6c signer PID changed before controlled retirement")
    os.kill(pid, signal.SIGINT)
    deadline = time.monotonic() + 15.0
    while Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    if Path(f"/proc/{pid}").exists():
        raise RuntimeError("r6c signer did not exit after controlled SIGINT")
    if _active_r6c_workers():
        raise PermissionError("r6c worker appeared during controlled retirement")
    if R6C_CELLS.exists() or R6C_CELLS.is_symlink():
        raise PermissionError("r6c cell root appeared during controlled retirement")
    for row in execution_absence.values():
        path = Path(str(row["canonical_path"]))
        if path.exists() or path.is_symlink():
            raise PermissionError("r6c execution artifact appeared during controlled retirement")
    failure = require_canonical_regular_file(FAILURE, within=RUNTIME)
    failure_payload = _json(failure)
    if failure_payload.get("failure_class") != "KeyboardInterrupt":
        raise PermissionError("r6c signer terminal receipt is not controlled KeyboardInterrupt")
    payload = {
        "schema": "m2_post33_phase_c_v4_r6c_pre_gpu_retirement_v1",
        "protocol_id": "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1",
        "phase_id": "PHASE_C_V4",
        "disposition": "RETIRED_BEFORE_GPU_OR_DATA_ACCESS",
        "reason": (
            "Independent post-build review found that the sealed r6c supervisor/checker did not "
            "fully enforce the execution contract and could orphan a setsid matrix after an "
            "uncatchable supervisor death.  Formal held-out scope was preserved for fresh r6d."
        ),
        "ready": file_metadata(READY),
        "public_anchor": file_metadata(PUBLIC),
        "signer_ack": file_metadata(ack),
        "verified_stage_a_capabilities": stage_a,
        "conditional_capabilities_absent": conditional_absence,
        "execution_receipts_absent": execution_absence,
        "cell_root": _absent(R6C_CELLS, "r6c cell root"),
        "signer_terminal_failure": file_metadata(failure),
        "signer_pid": pid,
        "signer_process_alive_after_retirement": False,
        "private_key_serialized_or_disk_persisted": False,
        "endpoint_or_score_opened": False,
        "gpu_or_data_worker_started": False,
        "r6c_mutation_or_reuse_forbidden": True,
        "fresh_r6d_root_anchor_and_source_closure_required": True,
    }
    print(write_json_exclusive(OUT, payload))


if __name__ == "__main__":
    main()
