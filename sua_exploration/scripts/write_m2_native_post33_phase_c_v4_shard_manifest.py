#!/usr/bin/env python3
"""Write a deterministic paired SPINT->T4 host/GPU shard manifest."""
from __future__ import annotations

import argparse
from pathlib import Path
import socket
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import (
    PHASE_ID, PROTOCOL_ID, sha256_file, write_json_exclusive,
)


def _csv_ints(value: str) -> list[int]:
    rows = [int(item) for item in value.split(",")]
    if not rows or len(rows) != len(set(rows)):
        raise argparse.ArgumentTypeError("allowlist must be non-empty and unique")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portable-manifest", type=Path, required=True)
    parser.add_argument("--cell-root", type=Path, required=True)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--folds", type=_csv_ints, required=True)
    parser.add_argument("--seeds", type=_csv_ints, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(fold not in range(7) for fold in args.folds):
        raise ValueError("fold allowlist must be within 0..6")
    seed_set = set(args.seeds)
    if seed_set != {42} and not seed_set <= {43, 44} and seed_set != {42, 43, 44}:
        raise ValueError(
            "a shard must be Stage-A seed42-only, Stage-B seeds43/44-only, "
            "or the all-seed full-opening scope"
        )
    portable = args.portable_manifest.resolve(strict=True)
    payload = {
        "schema": "m2_post33_phase_c_shard_manifest_v4",
        "protocol_id": PROTOCOL_ID,
        "phase_id": PHASE_ID,
        "host_id": socket.gethostname(),
        "gpu_id": args.gpu_id,
        "arms_in_order": ["spint", "t4"],
        "paired_same_host_required": True,
        "absolute_cell_root": str(args.cell_root.resolve()),
        "fold_allowlist": args.folds,
        "seed_allowlist": args.seeds,
        "portable_transfer_manifest_sha256": sha256_file(portable),
    }
    print(write_json_exclusive(args.output.resolve(), payload))


if __name__ == "__main__":
    main()
