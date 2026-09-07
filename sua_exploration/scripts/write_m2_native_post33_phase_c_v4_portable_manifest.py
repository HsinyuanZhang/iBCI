#!/usr/bin/env python3
"""Write a portable same-absolute-root manifest after Phase-C receipt sealing."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
    PHASE_ID, PROTOCOL_ID, file_metadata, write_json_exclusive,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cell-root", type=Path, required=True)
    parser.add_argument("--phase-c-program-receipt", type=Path, required=True)
    parser.add_argument("--deep-source-audit-receipt", type=Path, required=True)
    parser.add_argument("--phase-a-data-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace_root.resolve(strict=True)
    data = args.data_root.resolve(strict=True)
    program = args.phase_c_program_receipt.resolve(strict=True)
    deep_audit = args.deep_source_audit_receipt.resolve(strict=True)
    audit = args.phase_a_data_audit.resolve(strict=True)
    payload = {
        "schema": "m2_post33_phase_c_portable_transfer_manifest_v4",
        "protocol_id": PROTOCOL_ID,
        "phase_id": PHASE_ID,
        "workspace_root": str(workspace),
        "data_root": str(data),
        "absolute_cell_root": str(args.cell_root.resolve()),
        "same_absolute_paths_required_on_all_hosts": True,
        "hash_closures": [
            {"role": "phase_c_program_receipt", **file_metadata(program)},
            {"role": "deep_source_audit_receipt", **file_metadata(deep_audit)},
            {"role": "phase_a_data_audit", **file_metadata(audit)},
        ],
    }
    print(write_json_exclusive(args.output.resolve(), payload))


if __name__ == "__main__":
    main()
