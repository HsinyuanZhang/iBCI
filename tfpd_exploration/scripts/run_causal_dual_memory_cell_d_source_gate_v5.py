#!/usr/bin/env python3
"""Static public entry point for CDM-D Source Execution V5.

The public command imports only the standard library.  It cannot inspect V4
predecessor roots, resolve source data, load a checkpoint, initialize CUDA,
reserve a V5 root, or manufacture the opaque root-reviewed capability.
"""
from __future__ import annotations

import argparse
import json


WORKORDER_SHA256 = "503fa0276b3bb32fd31b9da51dca961d1214dfef0a0d7c40cd5e8bec3e87cb9b"
CELL = "CAUSAL_DUAL_MEMORY_CELL_D_V1"


def _dry_payload() -> dict[str, object]:
    return {
        "cell": CELL,
        "phase": "source_execution_v5_finalized_row_successor",
        "workorder_sha256": WORKORDER_SHA256,
        "execution_authorized": False,
        "opens_v4_predecessors": False,
        "opens_source": False,
        "loads_checkpoint": False,
        "initializes_cuda": False,
        "creates_root": False,
        "scores": False,
        "launches": False,
        "capability_requirement": "opaque in-process root-reviewed V5 capability",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print only the static no-execution contract")
    parser.add_argument("--source-gate", action="store_true", help="publicly rejected; requires in-process capability")
    parser.add_argument("--execute", action="store_true", help="publicly rejected; requires in-process capability")
    args = parser.parse_args(argv)
    if args.execute or args.source_gate:
        parser.error("execution requires a root-reviewed in-process V5 capability and is unavailable from public CLI")
    print(json.dumps(_dry_payload(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
