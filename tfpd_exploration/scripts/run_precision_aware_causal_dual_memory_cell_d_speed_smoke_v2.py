#!/usr/bin/env python3
"""Public inert dry CLI for the diagnostic Precision speed-smoke V2 route."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap() -> Path:
    root = Path(__file__).resolve().parents[2]
    exploration = root / "tfpd_exploration"
    source = exploration / "src"
    for candidate in (str(exploration), str(source)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Static V2 diagnostic speed-smoke contract; physical execution is root-only and unavailable publicly",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the literal-only V2 contract")
    args = parser.parse_args(argv)
    if not args.dry_run:
        parser.error("only --dry-run is public; opaque root capability APIs own authority and execution")
    _bootstrap()
    if "torch" in sys.modules:
        raise RuntimeError("speed-smoke V2 dry CLI must not import torch")
    from src.precision_aware_causal_dual_memory_cell_d_speed_smoke_v2 import plan
    print(json.dumps(plan.dry_plan(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
