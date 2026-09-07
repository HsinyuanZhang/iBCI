#!/usr/bin/env python3
"""Static public entry point for CDM-D Source Execution V1.

This file intentionally imports standard-library modules only.  The reviewed
execution implementation is not imported here because public command-line
flags cannot carry the opaque in-process root capability required to touch
fixed assets, source data, checkpoints, CUDA, or a result root.
"""
from __future__ import annotations

import argparse
import json


WORKORDER_SHA256 = "644ad075ee1f0d7bcd9d97e80ed5b765190bc9b4de278baaef309d0017fe9981"
CELL = "CAUSAL_DUAL_MEMORY_CELL_D_V1"


def _dry_payload() -> dict[str, object]:
    return {
        "cell": CELL,
        "phase": "source_execution_v1_scaffold",
        "workorder_sha256": WORKORDER_SHA256,
        "execution_authorized": False,
        "opens_source": False,
        "loads_checkpoint": False,
        "initializes_cuda": False,
        "creates_root": False,
        "scores": False,
        "launches": False,
        "capability_requirement": "opaque in-process root-reviewed capability",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print only the static no-execution contract")
    parser.add_argument("--source-smoke", action="store_true", help="publicly rejected; requires in-process capability")
    parser.add_argument("--source-gate", action="store_true", help="publicly rejected; requires in-process capability")
    parser.add_argument("--execute", action="store_true", help="publicly rejected; requires in-process capability")
    args = parser.parse_args(argv)
    if args.execute or args.source_smoke or args.source_gate:
        parser.error("execution requires a root-reviewed in-process capability and is unavailable from public CLI")
    print(json.dumps(_dry_payload(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
