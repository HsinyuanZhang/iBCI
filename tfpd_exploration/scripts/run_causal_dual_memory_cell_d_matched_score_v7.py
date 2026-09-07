#!/usr/bin/env python3
"""Static public dry entry point for the CDM-D V7 score successor.

The public process can describe the closure-bound V7 plan but cannot create
the opaque in-process root capability required for authority or physical
execution.  It intentionally imports no Torch and performs no path, result,
data, checkpoint, or device operation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap() -> None:
    repository = Path(__file__).resolve().parents[2]
    exploration = repository / "tfpd_exploration"
    for candidate in (exploration, exploration / "src"):
        rendered = str(candidate)
        if rendered not in sys.path:
            sys.path.insert(0, rendered)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CDM-D V7 matched-score static dry plan")
    parser.add_argument("--execute", action="store_true", help="reserved reviewed execution flag")
    parser.add_argument("--root-reviewed", action="store_true", help="reserved reviewed execution flag")
    args = parser.parse_args(argv)
    _bootstrap()
    from src.causal_dual_memory_cell_d_score_v7 import plan

    if args.execute or args.root_reviewed:
        if not (args.execute and args.root_reviewed):
            parser.error("both --execute and --root-reviewed are required; public CLI remains fail-closed")
        parser.error("public CLI has no opaque in-process root capability and cannot execute")
    print(json.dumps(plan.dry_plan(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
