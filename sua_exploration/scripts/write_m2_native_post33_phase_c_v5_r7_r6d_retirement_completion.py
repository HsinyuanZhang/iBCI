#!/usr/bin/env python3
"""Write the r6d retirement receipt after a verified SIGINT/zombie/reap chain.

The original controlled-retirement writer correctly delivered SIGINT to the
READY-bound r6d signer, but its first version treated a zombie as a live
process and intentionally refused to write a receipt.  This completion writer
is only valid after that signer has been reaped.  It records topology/stat
facts only: it never opens the historical controller-failure JSON,
``status.failed`` JSON, logs, selectors, endpoint payloads, scores, or R².
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
R6D_RECEIPTS = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r6d"
R6D_CELL_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cells_20260805_r6d"
R6D_PHASE = R6D_CELL_ROOT / "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1/PHASE_C_V4"
RUNTIME = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_r6d_live_signer_20260805"
READY = RUNTIME / "live_signer_ready.json"
CONTROLLER_FAILURE = RUNTIME / "controller_failure.json"
PUBLIC = ROOT / "sua_exploration/configs/m2_native_post33_phase_c_v4_r6d_root_ed25519_public.pem"
OUT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v5_r7_device_recovery_20260805/r6d_engineering_failure_retirement.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (  # noqa: E402
    file_metadata,
    require_canonical_regular_file,
    write_json_exclusive,
)


def _ready() -> dict[str, Any]:
    return json.loads(require_canonical_regular_file(READY, within=RUNTIME).read_text(encoding="utf-8"))


def _absent(path: Path, label: str) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise PermissionError(f"r6d retirement requires absent {label}: {path}")
    return {"canonical_path": str(path.resolve()), "present": False}


def _topology(path: Path, label: str) -> dict[str, Any]:
    """Return file topology only; deliberately do not read/hash contents."""
    if path.is_symlink() or not path.is_file():
        raise PermissionError(f"missing/non-regular retirement topology file: {label}")
    state = path.stat()
    return {
        "label": label,
        "canonical_path": str(path.resolve(strict=True)),
        "regular_file": True,
        "size_bytes": int(state.st_size),
        "inode": int(state.st_ino),
        "content_opened_or_hashed": False,
    }


def _no_active_r6d_worker() -> bool:
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
    own = os.getpid()
    for line in ps.splitlines():
        fields = line.strip().split(maxsplit=1)
        if not fields or not fields[0].isdigit() or int(fields[0]) == own:
            continue
        argv = fields[1] if len(fields) == 2 else ""
        if str(R6D_CELL_ROOT) in argv and any(token in argv for token in worker_tokens):
            return False
    return True


def _no_gpu_compute_process() -> bool:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
    )
    return not bool(result.stdout.strip())


def _absence_topology() -> dict[str, Any]:
    rows = {
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
        rows[f"{stem}_authorization"] = _absent(
            R6D_RECEIPTS / "auth" / f"{stem}.json", f"{stem} authorization"
        )
        rows[f"{stem}_signature"] = _absent(
            R6D_RECEIPTS / "auth" / f"{stem}.sig", f"{stem} signature"
        )
    return rows


def main() -> None:
    if OUT.exists() or OUT.is_symlink():
        raise FileExistsError(f"append-only r6d retirement already exists: {OUT}")
    ready = _ready()
    signer_pid = ready.get("pid")
    if not isinstance(signer_pid, int) or isinstance(signer_pid, bool) or signer_pid <= 1:
        raise PermissionError("r6d READY signer PID is invalid")
    if Path(f"/proc/{signer_pid}").exists():
        raise PermissionError("completion writer requires the prior signer to be reaped")
    if ready.get("public_key") != file_metadata(PUBLIC):
        raise PermissionError("r6d READY/public-anchor mismatch")
    if ready.get("private_key_retained_live_in_memory") is not True:
        raise PermissionError("r6d READY lacks live-only key declaration")
    if not _no_active_r6d_worker() or not _no_gpu_compute_process():
        raise PermissionError("r6d worker or GPU compute process is still live")
    failure = _topology(CONTROLLER_FAILURE, "historical controller failure after controlled SIGINT")
    failures = [
        _topology(
            R6D_PHASE / f"cells/arm-spint/fold-{fold}/seed-42/control/status.failed.json",
            f"spint-fold-{fold}-seed-42 failure status",
        )
        for fold in (0, 1)
    ]
    absence = _absence_topology()
    payload = {
        "schema": "m2_post33_phase_c_v5_r7_r6d_engineering_failure_retirement_v1",
        "protocol_id": "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1",
        "retired_lineage": "PHASE_C_V4_r6d",
        "disposition": "RETIRED_AFTER_STAGE_A_ENGINEERING_FAILURE_BEFORE_DECISION_OR_STAGE_B",
        "reason": (
            "Lightning post-test teardown moved registered deployment state to CPU while the plain CUDA "
            "cached identity remained resident.  The r6d engineering run failed before Stage-A completion; "
            "the lineage is retired rather than reused."
        ),
        "r6d_receipt_root": str(R6D_RECEIPTS.resolve()),
        "r6d_cell_root": str(R6D_CELL_ROOT.resolve()),
        "ready": file_metadata(READY),
        "public_anchor": file_metadata(PUBLIC),
        "signer_pid": signer_pid,
        "signer_lifecycle": {
            "identity_bound_before_signal": True,
            "controlled_signal": "SIGINT",
            "post_signal_state_observed": "zombie",
            "parent_identity": "verified tmux server PID 27130",
            "parent_reap_request": "SIGCHLD",
            "signer_reaped_before_completion_receipt": True,
            "historical_controller_failure_created_after_signal": failure,
        },
        "private_key_serialized_or_disk_persisted": False,
        "signer_process_alive_after_retirement": False,
        "r6d_worker_or_supervisor_alive_after_retirement": False,
        "gpu_compute_process_alive_after_retirement": False,
        "failed_cell_lifecycle_topology": failures,
        "failure_status_or_log_content_opened": False,
        "controller_failure_content_opened": False,
        "endpoint_score_selector_or_r2_opened": False,
        "stage_b_or_full_opening_authorization_materialized": False,
        "stage_a_decision_materialized": False,
        "absence_after_retirement": absence,
        "r6d_source_artifact_or_cell_mutation_performed": False,
        "r6d_reuse_forbidden": True,
        "fresh_r7_root_nonce_selector_program_and_authorization_required": True,
        "formal_gpu_rerun_authorized_by_this_receipt": False,
    }
    print(write_json_exclusive(OUT, payload))


if __name__ == "__main__":
    main()
