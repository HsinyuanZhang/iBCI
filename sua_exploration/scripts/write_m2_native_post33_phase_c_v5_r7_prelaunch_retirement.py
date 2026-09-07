#!/usr/bin/env python3
"""Append-only retirement of the unlaunched r7 capability after CPU dry-run."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (  # noqa: E402
    file_metadata,
    require_canonical_regular_file,
    write_json_exclusive,
)
from sua_exploration.mc_maze.m2_native_post33_phase_c_v5_r7_program import (  # noqa: E402
    R7_CELL_ROOT,
    R7_RECEIPT_ROOT,
)


PROGRAM = R7_RECEIPT_ROOT / "program/phase_c_program_v5_r7.json"
SUMMARY = R7_RECEIPT_ROOT / "launch/stage_a_ready_not_launched.json"
VERIFIER = R7_RECEIPT_ROOT / "prelaunch/cpu_source_capability_verifier.json"
AUTHS = [
    R7_RECEIPT_ROOT / "auth/stage_a_execution_gpu0_v5_r7.json",
    R7_RECEIPT_ROOT / "auth/stage_a_execution_gpu1_v5_r7.json",
]
OUTPUT = R7_RECEIPT_ROOT / "retirement/r7_prelaunch_matrix_contract_retirement.json"


def main() -> None:
    if R7_CELL_ROOT.exists():
        raise PermissionError("r7 retirement requires proof that no cell/selector root exists")
    if OUTPUT.exists():
        raise FileExistsError("r7 retirement receipt already exists")
    required = [PROGRAM, SUMMARY, VERIFIER, *AUTHS, *(path.with_suffix(".sig") for path in AUTHS)]
    for path in required:
        require_canonical_regular_file(path, within=ROOT)
    payload = {
        "schema": "m2_post33_phase_c_v5_r7_prelaunch_matrix_contract_retirement_v1",
        "status": "RETIRED_NEVER_LAUNCH",
        "reason": "CPU-only matrix dry-run reached no authorization claim, no worker, and no CUDA initialization, then rejected the custom r7 shard field because the dry-run renderer retained the v4 field name.  The r7 program/auth lineage is retired append-only; a fresh later revision must use a fresh root, key, program, manifest, authorization, nonce, and selector.",
        "program": file_metadata(PROGRAM),
        "ready_not_launched_summary": file_metadata(SUMMARY),
        "cpu_verifier": file_metadata(VERIFIER),
        "retired_capabilities": {
            path.stem: {
                "authorization": file_metadata(path),
                "signature": file_metadata(path.with_suffix(".sig")),
            }
            for path in AUTHS
        },
        "r7_cell_root": str(R7_CELL_ROOT.resolve()),
        "cell_root_exists": False,
        "selector_records_materialized": False,
        "authorization_nonce_claim_materialized": False,
        "worker_started": False,
        "cuda_initialized": False,
        "gpu_used": False,
        "formal_data_accessed": False,
        "score_data_accessed": False,
        "r7_capabilities_must_never_be_launched": True,
        "fresh_successor_root_key_program_auth_nonce_selector_required": True,
    }
    write_json_exclusive(OUTPUT, payload)
    os.chmod(OUTPUT, 0o444)
    print(json.dumps(file_metadata(OUTPUT), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
