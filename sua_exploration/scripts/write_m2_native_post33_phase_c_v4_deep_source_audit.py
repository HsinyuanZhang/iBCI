#!/usr/bin/env python3
"""Perform the one-time raw-source hash audit and seal its immutable receipt.

This command is deliberately outside normal training, evaluation, finalizer,
and opener paths.  It is the only Phase-C preflight entrypoint that should
read and SHA-256 every unique raw native-M2 source input.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_cost_v4 import (
    build_deep_source_audit_receipt,
)
from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import write_json_exclusive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-batch-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = args.source_batch_audit.resolve(strict=True)
    payload = build_deep_source_audit_receipt(
        audit, implementation_path=Path(__file__).resolve()
    )
    print(write_json_exclusive(args.output.resolve(), payload))


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    main()
