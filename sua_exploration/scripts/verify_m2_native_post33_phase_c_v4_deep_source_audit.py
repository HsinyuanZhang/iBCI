#!/usr/bin/env python3
"""Deliberately re-run the expensive Phase-C raw-source integrity audit.

Normal runtime only validates the sealed receipt shallowly.  Operators use
this explicit command when a new full raw-input integrity check is intended.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze.m2_native_post33_cost_v4 import (
    validate_deep_source_audit_receipt,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-batch-audit", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    audit = args.source_batch_audit.resolve(strict=True)
    receipt = args.receipt.resolve(strict=True)
    validate_deep_source_audit_receipt(
        json.loads(receipt.read_text(encoding="utf-8")),
        source_batch_audit_path=audit,
        deep_verify=True,
    )
    print("PASS_DEEP_SOURCE_INPUT_AUDIT_V4")


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    main()
