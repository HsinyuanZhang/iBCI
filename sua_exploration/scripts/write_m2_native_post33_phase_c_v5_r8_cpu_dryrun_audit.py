#!/usr/bin/env python3
"""Append-only receipt for the two non-executing r8 matrix dry-runs."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (  # noqa: E402
    file_metadata,
    require_canonical_regular_file,
    write_json_exclusive,
)
from sua_exploration.mc_maze.m2_native_post33_phase_c_v5_r8_program import (  # noqa: E402
    R8_CELL_ROOT,
    R8_RECEIPT_ROOT,
)


MATRIX = ROOT / "sua_exploration/scripts/run_m2_native_post33_phase_c_v5_r8_matrix.py"
PROGRAM = R8_RECEIPT_ROOT / "program/phase_c_program_v5_r8.json"
PORTABLE = R8_RECEIPT_ROOT / "manifest/portable_v5_r8.json"
COST = ROOT / "sua_exploration/results/m2_native_post33_phase_c_v4_launch_receipts_20260805_r3/cost/cost_supplement_r3.json"
OUTPUT = R8_RECEIPT_ROOT / "prelaunch/matrix_dry_run_cpu_audit.json"


def _run(name: str) -> dict[str, object]:
    shard = R8_RECEIPT_ROOT / "manifest" / f"shard_stage_a_{name}_v5_r8.json"
    command = [
        sys.executable,
        str(MATRIX),
        "--cell-root", str(R8_CELL_ROOT.resolve()),
        "--workspace-root", str(ROOT.resolve()),
        "--data-root", str((ROOT / "SPINT-main/data/000953").resolve(strict=True)),
        "--shard-manifest", str(shard.resolve(strict=True)),
        "--portable-manifest", str(PORTABLE.resolve(strict=True)),
        "--program-receipt", str(PROGRAM.resolve(strict=True)),
        "--cost-supplement", str(COST.resolve(strict=True)),
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"r8 {name} dry-run failed before receipt write: {completed.stderr}")
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict) or payload.get("execution_started") is not False:
        raise PermissionError("r8 dry-run did not remain non-executing")
    return {
        "command": command,
        "shard": file_metadata(shard),
        "result": payload,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("r8 dry-run audit already exists")
    if R8_CELL_ROOT.exists():
        raise PermissionError("r8 dry-run audit requires no cell/selector root before execution")
    for path in (MATRIX, PROGRAM, PORTABLE, COST):
        require_canonical_regular_file(path, within=ROOT)
    payload = {
        "schema": "m2_post33_phase_c_v5_r8_cpu_matrix_dryrun_audit_v1",
        "writer_source": file_metadata(Path(__file__)),
        "matrix_source": file_metadata(MATRIX),
        "program": file_metadata(PROGRAM),
        "portable_manifest": file_metadata(PORTABLE),
        "cost_supplement": file_metadata(COST),
        "cell_root_existed_before": False,
        "shards": {name: _run(name) for name in ("gpu0", "gpu1")},
        "cell_root_exists_after": R8_CELL_ROOT.exists(),
        "selector_records_materialized": False,
        "execution_started": False,
        "gpu_used": False,
        "formal_data_accessed": False,
        "score_data_accessed": False,
        "independent_review_or_execution_authorized": False,
    }
    if payload["cell_root_exists_after"]:
        raise PermissionError("r8 dry-run unexpectedly created a cell/selector root")
    write_json_exclusive(OUTPUT, payload)
    print(json.dumps(file_metadata(OUTPUT), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
