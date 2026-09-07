#!/usr/bin/env python3
"""Write the exact paired-SPINT completion receipt with O_EXCL.

The script is provided for a future authorized run and is not executed by the
Phase-B score-free plumbing audit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sua_exploration.mc_maze.m2_native_post33_phase_b_v3 import (
    CellKey,
    PROTOCOL_ID,
    build_spint_completion_receipt,
    deterministic_cell_paths,
    write_json_exclusive,
    write_status_exclusive,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(7), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--owner-token", required=True)
    return parser.parse_args()


def finalize(args: argparse.Namespace) -> Path:
    key = CellKey(PROTOCOL_ID, "spint", args.fold, args.seed)
    paths = deterministic_cell_paths(args.cell_root, key)
    owner = json.loads(paths["owner"].read_text(encoding="utf-8"))
    if owner.get("owner_token") != args.owner_token:
        raise PermissionError("cell owner token mismatch")
    selector_payload = json.loads(paths["selector_records"].read_text(encoding="utf-8"))
    if selector_payload.get("schema") != "m2_post33_source_selector_records_v3":
        raise ValueError("unexpected selector record schema")
    receipt = build_spint_completion_receipt(
        key=key,
        selector_records=selector_payload.get("records", []),
        resolved_config_path=paths["resolved_config"],
    )
    written = write_json_exclusive(paths["completion_receipt"], receipt)
    write_status_exclusive(
        args.cell_root,
        key,
        state="completed",
        owner_token=args.owner_token,
        details={
            "completion_receipt": str(written),
            "completion_receipt_schema": receipt["schema"],
        },
    )
    return written


def main() -> None:
    print(finalize(parse_args()))


if __name__ == "__main__":
    main()

