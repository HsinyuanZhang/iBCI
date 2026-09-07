#!/usr/bin/env python3
"""Static public entry point for CDM-D Source Execution V2.

This intentionally imports only the standard library.  Public flags cannot
construct the opaque in-process capability required to read immutable assets,
open source data, load a checkpoint, initialize CUDA, reserve a root, or run
the source smoke/gate.
"""
from __future__ import annotations

import argparse
import json


WORKORDER_SHA256 = "b71e8a7796db228f04c73efee841793bf0998413bbb436ecdc36d7c30d6e75ce"
CELL = "CAUSAL_DUAL_MEMORY_CELL_D_V1"


def _dry_payload() -> dict[str, object]:
    return {
        "cell": CELL,
        "phase": "source_execution_v2_theta_raw_bytes_successor",
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
