#!/usr/bin/env python3
"""Static public entry point for CDM-D matched-score V1.

This script is intentionally incapable of opening an evaluation asset,
checkpoint, CUDA device, or result root.  A reviewed in-process caller later
uses the package lifecycle and opaque capability; command-line flags alone
are never an execution authority.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap() -> None:
    repository = Path(__file__).resolve().parents[2]
    exploration = repository / "tfpd_exploration"
    # Both roots are literal so metadata-only imports work in a clean parent
    # environment without relying on ambient PYTHONPATH.
    for candidate in (exploration, exploration / "src"):
        rendered = str(candidate)
        if rendered not in sys.path:
            sys.path.insert(0, rendered)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CDM-D matched-score V1 static dry plan")
    parser.add_argument("--execute", action="store_true", help="reserved reviewed execution flag")
    parser.add_argument("--root-reviewed", action="store_true", help="reserved reviewed execution flag")
    args = parser.parse_args(argv)
    _bootstrap()
    from src.causal_dual_memory_cell_d_score_v1 import plan

    if args.execute or args.root_reviewed:
        if not (args.execute and args.root_reviewed):
            parser.error("both --execute and --root-reviewed are required; public CLI remains fail-closed")
        parser.error("public CLI has no opaque in-process root capability and cannot execute")
    print(json.dumps(plan.dry_plan(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    raise SystemExit(main())
