#!/usr/bin/env python3
"""Static public entry point for CDM-D independent-activity Source V3.

This module intentionally imports only the standard library.  Public flags
cannot manufacture the opaque in-process capability required to inspect fixed
assets, open source data, load a checkpoint, initialize CUDA, reserve a root,
or run the source smoke/gate.
"""
from __future__ import annotations

import argparse
import json


WORKORDER_SHA256 = "c53511d170069185bae2a7f639466646917f5a0ef84da47fa73f14e102288478"
CELL = "CAUSAL_DUAL_MEMORY_CELL_D_V1"


def _dry_payload() -> dict[str, object]:
    return {
        "cell": CELL,
        "phase": "source_execution_v3_independent_activity_successor",
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
