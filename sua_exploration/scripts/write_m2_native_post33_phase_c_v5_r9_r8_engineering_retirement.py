#!/usr/bin/env python3
"""Append-only score-free retirement of r8 after its pre-Hydra ABI failure."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (  # noqa: E402
    file_metadata,
    require_canonical_regular_file,
    write_json_exclusive,
)


R8_RECEIPT_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v5_r8_launch_receipts_20260805"
R8_CELL_ROOT = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v5_r8_cells_20260805"
R9_RECOVERY = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v5_r9_device_recovery_20260805"
OUTPUT = R9_RECOVERY / "r8_pre_hydra_abi_engineering_failure_retirement.json"
PHASE = R8_CELL_ROOT / "M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1/PHASE_C_V4"


def _stat_only(path: Path, label: str) -> dict[str, Any]:
    """Record structural evidence without reading or hashing a failed cell file."""
    source = require_canonical_regular_file(path, within=R8_CELL_ROOT)
    info = source.stat()
    return {
        "label": label,
        "canonical_path": str(source),
        "regular_file": True,
        "inode": info.st_ino,
        "size_bytes": info.st_size,
        "mode_octal": f"{info.st_mode & 0o777:04o}",
        "content_opened_or_hashed": False,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("r8 retirement receipt already exists")
    if not R8_CELL_ROOT.is_dir() or R8_CELL_ROOT.is_symlink():
        raise PermissionError("r8 engineering-failure retirement requires its realized cell root")
    auth_claims = sorted((PHASE / "authorization_claims/hw3090").glob("*.json"))
    if len(auth_claims) != 2:
        raise PermissionError("r8 retirement requires exactly two realized authorization nonce claims")
    claims = [file_metadata(require_canonical_regular_file(path, within=R8_CELL_ROOT)) for path in auth_claims]
    expected_failed = [
        PHASE / "cells/arm-spint/fold-0/seed-42/control/status.failed.json",
        PHASE / "cells/arm-spint/fold-1/seed-42/control/status.failed.json",
    ]
    failed = [
        _stat_only(path, f"r8 first SPINT cell fold-{fold} status.failed")
        for fold, path in zip((0, 1), expected_failed, strict=True)
    ]
    prohibited_materialized = {
        "selector_records": list(PHASE.glob("cells/**/selector_records.json")),
        "opaque_payload": list(PHASE.glob("cells/**/opaque_payload.json")),
        "score_commitment": list(PHASE.glob("cells/**/score_commitment.json")),
        "completion_receipt": list(PHASE.glob("cells/**/completion_receipt.json")),
        "deployment_cost_evidence": list(PHASE.glob("cells/**/deployment_cost_evidence.json")),
    }
    if any(paths for paths in prohibited_materialized.values()):
        raise PermissionError("r8 retirement refuses a lineage with selector/score/completion evidence")
    payload = {
        "schema": "m2_post33_phase_c_v5_r9_r8_pre_hydra_abi_engineering_failure_retirement_v1",
        "status": "RETIRED_NEVER_RELAUNCH",
        "retired_lineage": "PHASE_C_V5_R8_DEVICE_REPAIR",
        "reason": "The first Stage-A SPINT wrapper reached the historical v4 pre-Hydra helper, whose hard-coded authorization module/environment ABI rejected the r8 capability environment before Hydra initialization.  This is an engineering-control failure, not an effectiveness result.",
        "r8_receipt_root": str(R8_RECEIPT_ROOT.resolve()),
        "r8_cell_root": str(R8_CELL_ROOT.resolve()),
        "nonce_claims": {
            "count": 2,
            "metadata": claims,
            "contents_opened_for_retirement": False,
            "reuse_forbidden": True,
        },
        "first_failed_cell_statuses": failed,
        "failure_status_or_log_content_opened": False,
        "selector_or_checkpoint_content_opened": False,
        "endpoint_score_or_r2_opened": False,
        "hydra_initialized": False,
        "data_module_constructed": False,
        "cuda_initialized": False,
        "gpu_compute_started": False,
        "formal_data_accessed": False,
        "score_data_accessed": False,
        "prohibited_score_or_completion_artifacts_materialized": False,
        "fresh_r9_root_key_program_auth_nonce_selector_required": True,
        "r8_capabilities_must_never_be_relaunched": True,
    }
    write_json_exclusive(OUTPUT, payload)
    print(json.dumps(file_metadata(OUTPUT), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
