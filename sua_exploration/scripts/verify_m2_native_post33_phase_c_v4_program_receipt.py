#!/usr/bin/env python3
"""Revalidate the exact Phase-C program/source closure without raw NWB I/O."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_program_v4 import (
    validate_phase_c_program_receipt,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program-receipt", type=Path, required=True)
    args = parser.parse_args()
    validate_phase_c_program_receipt(args.program_receipt.resolve(strict=True))
    print("PASS_PHASE_C_PROGRAM_RECEIPT_V4")


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    main()
