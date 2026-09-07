#!/usr/bin/env python3
"""Write the immutable, score-free static prelaunch manifest for r10.

The writer is deliberately incapable of creating a cell root, run lock,
selector, checkpoint, evaluator payload, or GPU process.  It records only
source/control hashes and the exact static schedule.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import stat
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (  # noqa: E402
    ARMS, FOLDS, PHASE_ID, PROTOCOL_ID, CellKey, file_metadata, sha256_json,
)
from sua_exploration.mc_maze.m2_native_post33_phase_c_v5_r10_static import (  # noqa: E402
    MANIFEST_SCHEMA, OVERRIDE, PRELAUNCH_DIR, R10_CELL_ROOT, R10_RECEIPT_ROOT,
    STATIC_MANIFEST,
)


R9_RECOVERY = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v5_r10_recovery_20260805"
R9_RETIREMENT = R9_RECOVERY / "r9_continuation_dead_end_retirement.json"
R9_RETIREMENT_SEAL = R9_RECOVERY / "r9_continuation_dead_end_retirement.seal.json"
R9_PROGRAM = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v5_r9_launch_receipts_20260805/program/phase_c_program_v5_r9.json"
COST_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r3/cost"
COST_RECEIPT = COST_ROOT / "base_cost_receipt_r3.json"
COST_SUPPLEMENT = COST_ROOT / "cost_supplement_r3.json"

R10_CONTROL_SOURCE_PATHS = (
    "sua_exploration/mc_maze/m2_native_post33_phase_c_v5_r10_static.py",
    "sua_exploration/scripts/write_m2_native_post33_phase_c_v5_r10_static_prelaunch.py",
    "sua_exploration/scripts/run_m2_native_post33_phase_c_v5_r10_supervisor.py",
    "sua_exploration/scripts/run_m2_native_post33_phase_c_v5_r10_worker_shim.py",
    "sua_exploration/scripts/run_m2_native_post33_phase_c_v5_r10_cell_pipeline.py",
    "sua_exploration/scripts/evaluate_m2_native_post33_phase_c_v5_r10.py",
    "sua_exploration/scripts/open_m2_native_post33_phase_c_v5_r10_static.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _cells(seed_values: tuple[int, ...]) -> list[dict[str, Any]]:
    return [
        {"arm": arm, "fold": fold, "seed": seed}
        for seed in seed_values for fold in FOLDS for arm in ARMS
    ]


def _metadata(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"required source/control file missing: {path}")
    return file_metadata(path)


def _override_payload() -> dict[str, Any]:
    retirement = _metadata(R9_RETIREMENT)
    return {
        "schema": "m2_post33_phase_c_v5_r10_owner_control_plane_simplification_override_v1",
        "status": "APPROVED_STATIC_SINGLE_SUPERVISOR_SUBSTITUTION",
        "binds_r9_retirement": retirement,
        "superseded_recommendation_field": "fresh_r10_root_anchor_nonce_cell_root_and_live_signer_required",
        "reason": "That retired-r9 field was a recovery recommendation, not a scientific invariant. r10 replaces the live signer with one immutable static manifest, a fresh root, an O_EXCL supervisor lock, exact14-before-opening, and no manual branch selection.",
        "invariants_retained": [
            "fresh_r10_cell_root",
            "r9_capabilities_never_relaunched",
            "no_score_open_before_exact14_complete_zero_failed_zero_extra",
            "no_stage_b_before_internal_severe_negative_recomputation",
            "opaque_score_payload_mode_0600",
        ],
        "does_not_authorize": ["r9_relaunch", "pre_stage_a_score_open", "manual_stage_b_selection"],
        "formal_data_opened": False,
        "score_data_opened": False,
    }


def _write_immutable_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(fd, raw[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(path, 0o444)


def _r9_science_closure() -> list[dict[str, Any]]:
    """Reuse the sealed, independently checked 117-entry science source map."""
    program = json.loads(R9_PROGRAM.read_text(encoding="utf-8"))
    rows = program.get("source_map")
    _require(isinstance(rows, list) and len(rows) == 117, "sealed r9 program lacks exact 117-entry science closure")
    observed: list[dict[str, Any]] = []
    for row in rows:
        _require(isinstance(row, dict) and set(row) == {"relative_path", "canonical_path", "size_bytes", "sha256"}, "r9 science source-map row schema")
        source = ROOT / str(row["relative_path"])
        _require(_metadata(source) == {key: row[key] for key in ("canonical_path", "size_bytes", "sha256")}, f"r9 science-source drift: {row['relative_path']}")
        observed.append(dict(row))
    return observed


def build_manifest(*, allow_prelaunch_directory: bool = False) -> dict[str, Any]:
    data_root = ROOT / "SPINT-main/data/000953"
    _require(data_root.is_dir() and not data_root.is_symlink(), "canonical M2 data root unavailable")
    _require(not R10_CELL_ROOT.exists(), "fresh r10 cell root must not pre-exist at prelaunch")
    _require(
        not R10_RECEIPT_ROOT.exists() or (
            allow_prelaunch_directory and PRELAUNCH_DIR.is_dir() and OVERRIDE.is_file() and not STATIC_MANIFEST.exists()
        ),
        "r10 receipt root must be fresh at prelaunch",
    )
    _require((_metadata(R9_RETIREMENT_SEAL)), "r9 retirement seal unavailable")
    rows = _r9_science_closure()
    for relative in R10_CONTROL_SOURCE_PATHS:
        metadata = _metadata(ROOT / relative)
        rows.append({"relative_path": relative, **metadata})
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "PREPARED_NOT_EXECUTED",
        "protocol_id": PROTOCOL_ID,
        "phase_id": PHASE_ID,
        "workspace_root": str(ROOT.resolve()),
        "data_root": str(data_root.resolve(strict=True)),
        "r10_cell_root": str(R10_CELL_ROOT.resolve()),
        "arms_in_order": list(ARMS),
        "stage_a_cells": _cells((42,)),
        "stage_b_cells": _cells((43, 44)),
        "gpu_by_fold": {str(fold): str(fold % 2) for fold in FOLDS},
        "epoch_selection_rule": "max_finite_equal_session_mean_then_earlier_epoch;spint_epochs_0_through_34;t4_epochs_0_through_11;paired_spint_decoder_31_tensors_bit_exact",
        "severe_negative_rule": "(mean42 <= -0.03) OR (pos42 <= 1)",
        "source_map": rows,
        "source_map_sha256": sha256_json(rows),
        "source_map_entry_count": len(rows),
        "cost_receipt": _metadata(COST_RECEIPT),
        "cost_supplement": _metadata(COST_SUPPLEMENT),
        "r9_retirement": _metadata(R9_RETIREMENT),
        "r9_retirement_seal": _metadata(R9_RETIREMENT_SEAL),
        "owner_control_plane_simplification_override": _metadata(OVERRIDE),
        "score_payload_mode": "opaque_payload_0600_no_stdout",
        "formal_data_opened_by_prelaunch": False,
        "score_data_opened_by_prelaunch": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        _require(not R10_RECEIPT_ROOT.exists(), "r10 receipt root already exists")
        _write_immutable_exclusive(OVERRIDE, _override_payload())
        _write_immutable_exclusive(STATIC_MANIFEST, build_manifest(allow_prelaunch_directory=True))
        print(json.dumps({"manifest": str(STATIC_MANIFEST), "status": "PREPARED_NOT_EXECUTED"}, sort_keys=True))
        return
    # Preflight intentionally performs no write and makes no data/GPU import.
    _require(not R10_CELL_ROOT.exists(), "r10 cell root is not fresh")
    missing = [relative for relative in R10_CONTROL_SOURCE_PATHS if not (ROOT / relative).is_file()]
    print(json.dumps({
        "schema": "m2_post33_phase_c_v5_r10_static_prelaunch_dry_run_v1",
        "status": "PASS" if not missing else "BLOCKED_MISSING_SOURCE",
        "missing_source_paths": missing,
        "r10_cell_root_created": False,
        "gpu_used": False,
        "formal_data_opened": False,
        "score_data_opened": False,
    }, sort_keys=True))
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
