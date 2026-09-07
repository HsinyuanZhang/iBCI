#!/usr/bin/env python3
"""Write the append-only r6b controlled-stop incident without opening scores.

The incident is deliberately outside the r6b receipt and result roots.  It
records only hashes, file topology, status receipts, process-lifecycle facts,
and the static continuation dead-end; no selector, opaque endpoint, or score
payload is read.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
R6B_RECEIPTS = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r6b"
R6B_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_cells_20260805_r6b"
OUT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_r6b_abort_incident_20260805/r6b_controlled_stop_incident.json"
OPENERS = ROOT / "sua_exploration/mc_maze/m2_native_post33_openers_v4.py"
AUTHORIZATION = ROOT / "sua_exploration/mc_maze/m2_native_post33_authorization_v4.py"
PHASE = R6B_ROOT / "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1/PHASE_C_V4"
STOP_AT = "2026-08-05T10:02:03+08:00"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (  # noqa: E402
    file_metadata,
    require_canonical_regular_file,
    write_json_exclusive,
)


def _line_block(path: Path, start: int, end: int) -> dict[str, Any]:
    source = require_canonical_regular_file(path, within=ROOT)
    lines = source.read_text(encoding="utf-8").splitlines()
    if not (1 <= start <= end <= len(lines)):
        raise ValueError("invalid static-code proof line range")
    excerpt = "\n".join(lines[start - 1 : end]) + "\n"
    return {
        "source": file_metadata(source),
        "line_start": start,
        "line_end": end,
        "excerpt_sha256": hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
        "required_tokens": [
            token
            for token in (
                "decision_signature_present",
                "Stage-A decision lacks detached root signature",
                "stage_a_decision_signature",
                "Stage-A continue decision signature",
            )
            if token in excerpt
        ],
    }


def _failed_cell(fold: int) -> dict[str, Any]:
    control = PHASE / f"cells/arm-spint/fold-{fold}/seed-42/control"
    failed = require_canonical_regular_file(control / "status.failed.json", within=R6B_ROOT)
    started = require_canonical_regular_file(control / "status.started.json", within=R6B_ROOT)
    ownership = require_canonical_regular_file(control / "ownership.json", within=R6B_ROOT)
    # These status records contain only lifecycle metadata, not score-bearing
    # data.  Parse them solely to assert the exact externally observed stop.
    status = json.loads(failed.read_text(encoding="utf-8"))
    expected = {
        "schema": "m2_post33_phase_c_cell_status_v4",
        "protocol_id": "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1",
        "phase_id": "PHASE_C_V4",
        "arm": "spint",
        "fold": fold,
        "seed": 42,
        "state": "failed",
        "failure_kind": "signal_2",
        "return_code": 130,
    }
    for field, value in expected.items():
        if status.get(field) != value:
            raise ValueError(f"r6b fold-{fold} stop receipt differs at {field}")
    checkpoints = sorted((control.parent / "run/checkpoints").glob("epoch_*.ckpt"))
    if len(checkpoints) != 3:
        raise ValueError(f"r6b fold-{fold} must retain exactly three checkpoint files")
    return {
        "fold": fold,
        "lifecycle_status": {
            "failure_kind": "signal_2",
            "return_code": 130,
            "state": "failed",
        },
        "ownership": file_metadata(ownership),
        "started": file_metadata(started),
        "failed": file_metadata(failed),
        "checkpoints": [file_metadata(require_canonical_regular_file(p, within=R6B_ROOT)) for p in checkpoints],
    }


def _absent(relative: str) -> dict[str, Any]:
    candidate = PHASE / relative
    if candidate.exists() or candidate.is_symlink():
        raise ValueError(f"r6b controlled stop must not contain {relative}")
    return {"relative_path": relative, "present": False}


def _surviving_r6b_processes(ps_text: str, *, own_pid: int) -> list[str]:
    """Return only live r6b workers, never this audit writer or its shell."""
    active_basenames = {
        "run_m2_native_post33_phase_c_v4_matrix.py",
        "run_m2_native_post33_phase_c_v4_cell_pipeline.py",
        "train_post33_phase_c_v4.py",
        "evaluate_post33_phase_c_v4.py",
    }
    forbidden = []
    for line in ps_text.splitlines():
        fields = line.strip().split(maxsplit=1)
        if not fields or not fields[0].isdigit() or int(fields[0]) == own_pid:
            continue
        argv = fields[1] if len(fields) == 2 else ""
        if str(R6B_ROOT) in argv and any(name in argv for name in active_basenames):
            forbidden.append(line)
    return forbidden


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"append-only incident already exists: {OUT}")
    if not R6B_RECEIPTS.is_dir() or not PHASE.is_dir():
        raise FileNotFoundError("r6b roots required for controlled-stop incident")
    failures = [_failed_cell(fold) for fold in (0, 1)]
    checkpoint_count = sum(len(row["checkpoints"]) for row in failures)
    if checkpoint_count != 6:
        raise ValueError("r6b controlled stop checkpoint count must be exactly six")
    # This is a snapshot of surviving process state after the parent-controlled
    # stop.  It intentionally has no PID inference or private-key inspection.
    ps = subprocess.run(
        ["ps", "-eo", "pid=,args="], check=True, text=True, capture_output=True
    ).stdout
    forbidden = _surviving_r6b_processes(ps, own_pid=os.getpid())
    if forbidden:
        raise RuntimeError("r6b process survived controlled stop")
    decision_signature = _absent("stage_a/opened_stage_a_decision.sig")
    payload = {
        "schema": "m2_post33_phase_c_v4_r6b_controlled_stop_incident_v1",
        "protocol_id": "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1",
        "phase_id": "PHASE_C_V4",
        "r6b_receipt_root": str(R6B_RECEIPTS.resolve()),
        "r6b_cell_root": str(R6B_ROOT.resolve()),
        "stop": {
            "timestamp": STOP_AT,
            "method": "parent_controlled_tmux_process_tree_stop",
            "gpu_launch_continued": False,
            "surviving_r6b_processes_at_incident_write": 0,
        },
        "cell_status": {
            "started": 2,
            "failed": 2,
            "completed": 0,
            "failed_cells": failures,
            "checkpoint_count": checkpoint_count,
        },
        "score_and_stage_topology": {
            "selector_records": [_absent(f"cells/arm-spint/fold-{fold}/seed-42/run/selector_records.json") for fold in (0, 1)],
            "opaque_endpoint_payload": [_absent(f"cells/arm-spint/fold-{fold}/seed-42/run/opaque_endpoint_payload.json") for fold in (0, 1)],
            "score_commitment": [_absent(f"cells/arm-spint/fold-{fold}/seed-42/run/score_commitment.json") for fold in (0, 1)],
            "stage_a_manifest": _absent("stage_a/score_sealed_stage_a_manifest.json"),
            "stage_a_decision": _absent("stage_a/opened_stage_a_decision.json"),
            "stage_a_decision_signature": decision_signature,
            "endpoint_or_score_opened": False,
        },
        "capability_inventory": {
            "issued_stage_a_authorizations": [
                file_metadata(require_canonical_regular_file(p, within=R6B_RECEIPTS))
                for p in sorted((R6B_RECEIPTS / "auth").glob("*.json"))
            ],
            "stage_b_authorization_present": False,
            "full_opening_authorization_present": False,
            "r6b_private_signer_survives": False,
            "private_key_persisted": False,
        },
        "continuation_dead_end_static_proof": {
            "opener_writes_decision_without_signature": _line_block(OPENERS, 124, 165),
            "production_stage_b_requires_detached_decision_signature": _line_block(OPENERS, 223, 240),
            "authorization_requires_exact_decision_and_signature_metadata": _line_block(AUTHORIZATION, 271, 292),
            "conclusion": (
                "With the sole r6b signer gone and no Stage-B/full-opening authorizations issued, "
                "no valid continuation can be created: a new anchor would alter the sealed r6b source map."
            ),
        },
        "disposition": {
            "r6b_root_mutation_forbidden": True,
            "r6b_score_opening_forbidden": True,
            "fresh_r6c_root_and_anchor_required": True,
            "replacement_must_keep_a_live_same_key_signer_through_stage_a_decision_signature_and_conditional_issue": True,
        },
    }
    print(write_json_exclusive(OUT, payload))


if __name__ == "__main__":
    main()
