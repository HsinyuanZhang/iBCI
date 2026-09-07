#!/usr/bin/env python3
"""Seal the exact Phase-C program/source closure after explicit preflight."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_phase_c_v4 import write_json_exclusive
from sua_exploration.mc_maze.m2_native_post33_program_v4 import (
    build_phase_c_program_receipt,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eof-canonicalization-receipt", type=Path, required=True)
    parser.add_argument("--deep-source-audit-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_phase_c_program_receipt(
        eof_canonicalization_receipt_path=args.eof_canonicalization_receipt.resolve(strict=True),
        deep_source_audit_receipt_path=args.deep_source_audit_receipt.resolve(strict=True),
    )
    print(write_json_exclusive(args.output.resolve(), payload))


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    main()
